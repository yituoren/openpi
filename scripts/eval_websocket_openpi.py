#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.policies import policy_config
from openpi.training import config as train_config

from openpi.envs import (  # noqa: E402
    add_envs_task,
    close_envs,
    make_env,
    make_env_config,
    make_env_pre_post_processors,
    preprocess_observation,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _first_vec_env(envs: dict) -> Any:
    for _, group in envs.items():
        for _, vec in group.items():
            return vec
    raise RuntimeError("make_env returned no vector environment")


def _load_rename_map(rename_map: str | None, rename_map_path: Path | None) -> dict[str, str]:
    if rename_map and rename_map_path:
        raise ValueError("Pass at most one of --rename_map and --rename_map_path.")
    if rename_map:
        data = json.loads(rename_map)
    elif rename_map_path:
        with rename_map_path.expanduser().resolve().open(encoding="utf-8") as f:
            data = json.load(f)
    else:
        return {}
    if not isinstance(data, dict):
        raise TypeError("Rename map must be a JSON object mapping str->str.")
    return {str(k): str(v) for k, v in data.items()}


def _to_numpy_tree(x: Any) -> Any:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_tree(v) for v in x]
    return x


def _index_env(x: Any, env_id: int) -> Any:
    if isinstance(x, dict):
        return {k: _index_env(v, env_id) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[env_id]
    if isinstance(x, list):
        return x[env_id]
    return x


def _apply_rename_map(obs: dict[str, Any], rename_map: dict[str, str]) -> dict[str, Any]:
    out = dict(obs)
    for src, dst in rename_map.items():
        if src in out:
            out[dst] = out.pop(src)
    return out


def _infer_env_action_dim(cfg: train_config.TrainConfig) -> int:
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    for t in data_cfg.data_transforms.outputs:
        dim = getattr(t, "action_dim", None)
        if isinstance(dim, int):
            return dim
    return int(cfg.model.action_dim)


def _arr_stats(name: str, arr: np.ndarray) -> str:
    a = np.asarray(arr)
    return (
        f"{name}:shape={tuple(a.shape)} min={float(a.min()):.4f} "
        f"max={float(a.max()):.4f} mean={float(a.mean()):.4f} std={float(a.std()):.4f}"
    )


def parse_args(default_policy_config: str) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    p.add_argument("--policy.config", dest="policy_config", type=str, default=default_policy_config)
    p.add_argument("--env.type", dest="env_type", type=str, default="robotwin", choices=["robotwin", "dreamzero"])
    p.add_argument("--env.task", dest="env_task", type=str, default=None)
    p.add_argument("--env.ws_url", dest="env_ws_url", type=str, default="ws://127.0.0.1:8765")
    p.add_argument("--env.observation_height", dest="env_observation_height", type=int, default=480)
    p.add_argument("--env.observation_width", dest="env_observation_width", type=int, default=640)
    p.add_argument("--env.action_dim", dest="env_action_dim", type=int, default=None)
    p.add_argument("--env.max_episode_steps_model", dest="env_max_episode_steps_model", type=int, default=300)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--rollout.execute_steps", dest="rollout_execute_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rename_map", type=str, default=None)
    p.add_argument("--rename_map_path", type=Path, default=None)
    return p.parse_args()


def run_eval(default_policy_config: str) -> None:
    args = parse_args(default_policy_config)
    rename_map = _load_rename_map(args.rename_map, args.rename_map_path)

    cfg = train_config.get_config(args.policy_config)
    _log(f"[openpi-eval] loading policy config={cfg.name} checkpoint={args.policy_path}")
    policy = policy_config.create_trained_policy(cfg, args.policy_path)

    inferred_action_dim = _infer_env_action_dim(cfg)
    env_action_dim = args.env_action_dim if args.env_action_dim is not None else inferred_action_dim
    _log(f"[openpi-eval] env_action_dim={env_action_dim} (inferred={inferred_action_dim})")

    env_cfg = make_env_config(
        args.env_type,
        task=args.env_task,
        ws_url=args.env_ws_url,
        observation_height=args.env_observation_height,
        observation_width=args.env_observation_width,
        action_dim=env_action_dim,
        max_episode_steps_model=args.env_max_episode_steps_model,
    )
    envs = make_env(env_cfg, n_envs=args.num_envs, use_async_envs=False)
    vec_env = _first_vec_env(envs)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=None)

    total_episodes = 0
    success_episodes = 0
    episode_lengths: list[int] = []
    obs_np, _ = vec_env.reset(seed=args.seed)
    steps_since_reset = np.zeros(vec_env.num_envs, dtype=int)

    while total_episodes < args.episodes:
        obs_batch = preprocess_observation(obs_np)
        obs_batch = add_envs_task(vec_env, obs_batch)
        obs_batch = env_preprocessor(obs_batch)
        obs_batch = _to_numpy_tree(obs_batch)

        action_chunks = []
        for env_id in range(vec_env.num_envs):
            single_obs = _index_env(obs_batch, env_id)
            single_obs = _apply_rename_map(single_obs, rename_map)

            if "prompt" not in single_obs:
                task_text = ""
                if "task" in obs_batch and isinstance(obs_batch["task"], list):
                    task_text = str(obs_batch["task"][env_id])
                single_obs["prompt"] = task_text

            result = policy.infer(single_obs)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise RuntimeError(f"Expected actions shape (H, D), got {actions.shape}")
            action_chunks.append(actions)

        action_np = np.stack(action_chunks, axis=0)
        if args.rollout_execute_steps is not None:
            if args.rollout_execute_steps <= 0:
                raise ValueError("--rollout.execute_steps must be > 0 when set.")
            action_np = action_np[:, : min(args.rollout_execute_steps, action_np.shape[1]), :]

        action_transition = env_postprocessor({"action": action_np})
        action_np = np.asarray(action_transition["action"], dtype=np.float32)

        _log("[openpi-eval] " + _arr_stats("action", action_np))
        obs_np, reward, terminated, truncated, info = vec_env.step(action_np)
        done_chunk = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        steps_since_reset += 1

        for env_id in range(vec_env.num_envs):
            if done_chunk[env_id]:
                total_episodes += 1
                episode_lengths.append(int(steps_since_reset[env_id]))
                steps_since_reset[env_id] = 0
                per_env_info = info.get("per_env", [{}])[env_id] if isinstance(info, dict) else {}
                if per_env_info.get("done_reason", "") == "success":
                    success_episodes += 1
                if total_episodes >= args.episodes:
                    break

    if hasattr(vec_env, "finalize_pending_resets"):
        _ = bool(vec_env.finalize_pending_resets())
    vec_env.close()
    close_envs(envs)

    success_rate = success_episodes / max(1, total_episodes)
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    _log(
        f"[openpi-eval] episodes={total_episodes} success={success_episodes} "
        f"success_rate={success_rate:.3f} avg_len={avg_len:.1f}"
    )
