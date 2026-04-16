from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import websockets.sync.client


@dataclass(frozen=True)
class WebsocketEvalEnvConfig:
    type: str
    task: str | None
    ws_url: str
    observation_height: int
    observation_width: int
    action_dim: int
    max_episode_steps_model: int = 300


def make_env_config(env_type: str, **kwargs) -> WebsocketEvalEnvConfig:
    return WebsocketEvalEnvConfig(type=env_type, **kwargs)


def make_env(
    cfg: WebsocketEvalEnvConfig,
    n_envs: int = 1,
    use_async_envs: bool = False,
) -> dict[str, dict[int, "WebsocketVectorEnv"]]:
    del use_async_envs
    reset_options: dict[str, Any] = {
        "observation_height": int(cfg.observation_height),
        "observation_width": int(cfg.observation_width),
        # Keep server-side fail-limit aligned with client-side model step limit.
        "step_limit": int(cfg.max_episode_steps_model),
    }
    if cfg.task is not None:
        reset_options["task_name"] = cfg.task
    vec = WebsocketVectorEnv(
        ws_url=cfg.ws_url,
        num_envs=n_envs,
        action_dim=int(cfg.action_dim),
        image_shape=(int(cfg.observation_height), int(cfg.observation_width), 3),
        max_episode_steps_model=int(cfg.max_episode_steps_model),
        reset_options=reset_options,
    )
    return {cfg.type: {0: vec}}


def make_env_pre_post_processors(
    env_cfg: WebsocketEvalEnvConfig,
    policy_cfg: Any = None,
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]]]:
    del env_cfg, policy_cfg
    return (lambda x: x), (lambda x: x)


def close_envs(envs: dict[str, dict[int, "WebsocketVectorEnv"]]) -> None:
    visited: set[int] = set()
    for group in envs.values():
        for vec in group.values():
            key = id(vec)
            if key in visited:
                continue
            visited.add(key)
            vec.close()


def preprocess_observation(observations: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "pixels" in observations:
        pixels = observations["pixels"]
        if isinstance(pixels, dict):
            for key, value in pixels.items():
                out[f"observation.images.{key}"] = np.asarray(value)
        else:
            out["observation.images.image"] = np.asarray(pixels)
    if "agent_pos" in observations:
        out["observation.state"] = np.asarray(observations["agent_pos"], dtype=np.float32)
    return out


def add_envs_task(env: "WebsocketVectorEnv", observation: dict[str, Any]) -> dict[str, Any]:
    updated = dict(observation)
    batch_size = 0
    for value in updated.values():
        if isinstance(value, np.ndarray):
            batch_size = int(value.shape[0])
            break
    if batch_size <= 0:
        batch_size = int(getattr(env, "num_envs", 1))

    tasks = list(getattr(env, "latest_task_texts", []) or [])
    if len(tasks) < batch_size:
        tasks.extend([""] * (batch_size - len(tasks)))
    updated["task"] = [str(t) for t in tasks[:batch_size]]
    return updated


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        payload = base64.b64encode(obj.tobytes(order="C")).decode("ascii")
        return {"__ndarray__": True, "dtype": str(obj.dtype), "shape": list(obj.shape), "data": payload}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        if obj.get("__ndarray__"):
            dtype = np.dtype(obj["dtype"])
            shape = tuple(obj["shape"])
            raw = base64.b64decode(obj["data"].encode("ascii"))
            return np.frombuffer(raw, dtype=dtype).reshape(shape)
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


def _stack_observations(obs_list: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not obs_list:
        raise ValueError("obs_list cannot be empty.")

    def _stack(values: Sequence[Any]) -> Any:
        first = values[0]
        if isinstance(first, Mapping):
            return {k: _stack([v[k] for v in values]) for k in first}
        if isinstance(first, np.ndarray):
            return np.stack(values, axis=0)
        if isinstance(first, (str, bytes)):
            return list(values)
        return np.asarray(values)

    return _stack(obs_list)


def _unstack_observations(obs: Mapping[str, Any], batch_size: int) -> list[dict[str, Any]]:
    def _get_item(node: Any, idx: int) -> Any:
        if isinstance(node, Mapping):
            return {k: _get_item(v, idx) for k, v in node.items()}
        if isinstance(node, np.ndarray):
            return node[idx]
        if isinstance(node, list):
            return node[idx]
        return node

    return [_get_item(obs, i) for i in range(batch_size)]


@dataclass
class _WsBatchItem:
    env_id: int
    next_obs: dict[str, Any]
    done: bool
    info: dict[str, Any]


class WebsocketJsonClient:
    def __init__(self, ws_url: str, open_timeout_s: float = 30.0):
        self._conn = websockets.sync.client.connect(
            ws_url,
            compression=None,
            max_size=None,
            open_timeout=open_timeout_s,
            ping_interval=None,
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._conn.send(json.dumps(_to_jsonable(payload)))
        raw = self._conn.recv()
        if not isinstance(raw, str):
            raise RuntimeError("Expected text JSON frame from websocket server.")
        data = json.loads(raw)
        if "error" in data:
            raise RuntimeError(f"Websocket server error: {data['error']}")
        return _from_jsonable(data)

    def batch_reset(
        self,
        env_ids: Sequence[int],
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> list[_WsBatchItem]:
        payload: dict[str, Any] = {"type": "batch_reset", "env_ids": list(env_ids), "seed": seed}
        if options:
            payload["options"] = dict(options)
        response = self._request(payload)
        return self._parse_results(response, expected_type="batch_reset")

    def batch_step(self, env_ids: Sequence[int], action_chunks: Sequence[Any]) -> list[_WsBatchItem]:
        items = [{"env_id": int(eid), "action_chunk": action_chunk} for eid, action_chunk in zip(env_ids, action_chunks, strict=True)]
        response = self._request({"type": "batch_step", "items": items})
        return self._parse_results(response, expected_type="batch_step")

    @staticmethod
    def _parse_results(response: Mapping[str, Any], expected_type: str) -> list[_WsBatchItem]:
        if response.get("type") != expected_type:
            raise RuntimeError(f"Unexpected response type: {response.get('type')} (expected {expected_type}).")
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Response must contain a list field `results`.")
        parsed: list[_WsBatchItem] = []
        for item in results:
            parsed.append(
                _WsBatchItem(
                    env_id=int(item["env_id"]),
                    next_obs=dict(item.get("next_obs", item.get("obs", {}))),
                    done=bool(item.get("done", False)),
                    info=dict(item.get("info", {})),
                )
            )
        return parsed


class WebsocketVectorEnv:
    def __init__(
        self,
        *,
        ws_url: str,
        num_envs: int,
        reset_options: dict[str, Any] | None = None,
        action_dim: int,
        image_shape: tuple[int, int, int] = (480, 640, 3),
        max_episode_steps_model: int = 300,
    ) -> None:
        del action_dim, image_shape
        self.num_envs = int(num_envs)
        self.envs = [SimpleNamespace() for _ in range(self.num_envs)]
        self._client = WebsocketJsonClient(ws_url=ws_url)
        self._env_ids = list(range(self.num_envs))
        self._max_episode_steps_model = int(max_episode_steps_model)
        self._episode_step_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._pending_reset_mask = np.zeros(self.num_envs, dtype=bool)
        self._last_obs_batch: dict[str, Any] | None = None
        self._extra_done_condition: Callable[[int, int, dict[str, Any]], bool] = lambda _eid, _steps, _obs: False
        self._reward_provider: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None
        self._reset_options = reset_options or {}
        self.latest_task_texts: list[str] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        results = self._client.batch_reset(self._env_ids, seed=seed, options=self._reset_options)
        self._episode_step_counts[:] = 0
        self._pending_reset_mask[:] = False
        obs_by_id = [None] * self.num_envs
        info_by_id = [None] * self.num_envs
        for item in results:
            obs_by_id[item.env_id] = item.next_obs
            info_by_id[item.env_id] = item.info
        self.latest_task_texts = [
            str((info_by_id[i] or {}).get("instruction", self._reset_options.get("task_name", "")))
            for i in range(self.num_envs)
        ]
        obs_batch = _stack_observations(obs_by_id)
        info = {"per_env": info_by_id}
        self._last_obs_batch = obs_batch
        return obs_batch, info

    def step(self, action_chunks: np.ndarray):
        if action_chunks.shape[0] != self.num_envs:
            raise ValueError(f"Expected action batch size={self.num_envs}, got {action_chunks.shape[0]}.")

        if self._last_obs_batch is not None:
            obs_by_id: list[dict[str, Any] | None] = _unstack_observations(self._last_obs_batch, self.num_envs)
        else:
            obs_by_id = [None] * self.num_envs
        info_by_id: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        done_env = np.zeros(self.num_envs, dtype=bool)
        done_model = np.zeros(self.num_envs, dtype=bool)
        done_just_now = np.zeros(self.num_envs, dtype=bool)

        active_ids = [i for i in self._env_ids if not self._pending_reset_mask[i]]
        if active_ids:
            active_actions = action_chunks[np.asarray(active_ids, dtype=np.int64)]
            step_results = self._client.batch_step(active_ids, active_actions)
            for item in step_results:
                env_id = item.env_id
                obs_by_id[env_id] = item.next_obs
                info_by_id[env_id] = dict(item.info)
                done_env[env_id] = item.done

            for env_id in active_ids:
                chunk_steps = int(info_by_id[env_id].get("chunk_steps_executed", 1))
                self._episode_step_counts[env_id] += max(1, chunk_steps)
                obs = obs_by_id[env_id]
                step_limit_done = self._episode_step_counts[env_id] >= self._max_episode_steps_model
                extra_done = self._extra_done_condition(env_id, int(self._episode_step_counts[env_id]), obs)
                done_model[env_id] = bool(step_limit_done or extra_done)
                if done_env[env_id] or done_model[env_id]:
                    done_just_now[env_id] = True
                    self._pending_reset_mask[env_id] = True

        if self._pending_reset_mask.all():
            reset_results = self._client.batch_reset(self._env_ids, options=self._reset_options)
            for item in reset_results:
                info_by_id[item.env_id]["final_observation"] = obs_by_id[item.env_id]
                info_by_id[item.env_id]["reset_info"] = item.info
                obs_by_id[item.env_id] = item.next_obs
            self.latest_task_texts = [
                str((info_by_id[i].get("reset_info") or {}).get("instruction", self._reset_options.get("task_name", "")))
                for i in range(self.num_envs)
            ]
            self._episode_step_counts[:] = 0
            self._pending_reset_mask[:] = False

        for env_id, obs in enumerate(obs_by_id):
            if obs is None:
                raise RuntimeError(f"Missing observation for env_id={env_id} after step/reset.")

        next_obs_batch = _stack_observations(obs_by_id)
        truncated = np.zeros(self.num_envs, dtype=bool)
        terminated = done_just_now.astype(bool)
        rewards = np.zeros(self.num_envs, dtype=np.float32)

        if self._reward_provider is not None and self._last_obs_batch is not None:
            prev_obs = _unstack_observations(self._last_obs_batch, self.num_envs)
            next_obs = _unstack_observations(next_obs_batch, self.num_envs)
            rewards = np.asarray(
                self._reward_provider(
                    np.asarray(prev_obs, dtype=object),
                    np.asarray(action_chunks, dtype=np.float32),
                    np.asarray(next_obs, dtype=object),
                    done_just_now.astype(bool),
                ),
                dtype=np.float32,
            )

        self._last_obs_batch = next_obs_batch
        return next_obs_batch, rewards, terminated, truncated, {"per_env": info_by_id}

    def finalize_pending_resets(self) -> bool:
        if not bool(np.any(self._pending_reset_mask)):
            return False
        self._client.batch_reset(self._env_ids, options=self._reset_options)
        self._episode_step_counts[:] = 0
        self._pending_reset_mask[:] = False
        return True

    def close(self) -> None:
        try:
            self.finalize_pending_resets()
        except Exception:
            pass
        self._client.close()
