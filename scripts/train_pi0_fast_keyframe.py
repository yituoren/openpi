#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import random
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi.models import model as _model
from openpi.models import pi0_fast as _pi0_fast
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as train_config
from openpi.training import data_loader as _data_loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    parser.add_argument("--policy.config", dest="policy_config", type=str, default="pi0_fast_robotwin")
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_model(cfg: train_config.TrainConfig, checkpoint_dir: pathlib.Path) -> _pi0_fast.Pi0FAST:
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(jax.tree.map(lambda x: x, params))
    model = nnx.merge(graphdef, state)
    if not isinstance(model, _pi0_fast.Pi0FAST):
        raise TypeError("Keyframe training expects a Pi0FAST model.")
    return model


def _sample_metadata(sample: dict[str, Any], dataset_index: int) -> dict[str, Any]:
    return {
        "dataset_index": int(dataset_index),
        "episode_index": int(sample["episode_index"]) if "episode_index" in sample else None,
        "frame_index": int(sample["frame_index"]) if "frame_index" in sample else dataset_index,
    }


def _build_targets(
    dataset,
    annotations: list[dict[str, Any]],
    radius: int,
) -> np.ndarray:
    centers_by_episode: dict[Any, list[int]] = {}
    for record in annotations:
        if record.get("is_keyframe") in (False, 0, None):
            continue
        episode_index = record.get("episode_index", "__global__")
        frame_index = int(record.get("frame_index", record["dataset_index"]))
        centers_by_episode.setdefault(episode_index, []).append(frame_index)

    targets = np.zeros((len(dataset),), dtype=np.float32)
    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        meta = _sample_metadata(sample, dataset_index)
        episode_index = meta["episode_index"] if meta["episode_index"] is not None else "__global__"
        frame_index = meta["frame_index"]
        centers = centers_by_episode.get(episode_index, [])
        if not centers:
            targets[dataset_index] = 0.0
            continue
        min_dist = min(abs(frame_index - center) for center in centers)
        if min_dist > radius:
            targets[dataset_index] = 0.0
        else:
            targets[dataset_index] = 1.0 - float(min_dist) / float(radius + 1)
    return targets


def _make_batch(dataset, indices: list[int]) -> tuple[_model.Observation, jax.Array]:
    samples = [dataset[i] for i in indices]
    batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs, axis=0)), *samples)
    observation = _model.Observation.from_dict(batch)
    return observation, batch["keyframe_target"]


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    random.seed(args.seed)

    cfg = train_config.get_config(args.policy_config)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_keyframe_head=True))
    checkpoint_dir = pathlib.Path(args.policy_path)
    model = _load_model(cfg, checkpoint_dir)

    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = _data_loader.transform_dataset(
        _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model),
        data_cfg,
    )
    annotations = _load_jsonl(pathlib.Path(args.annotations))
    targets = _build_targets(dataset, annotations, radius=cfg.model.action_horizon)

    class _TargetDataset:
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index: int):
            sample = dict(dataset[index])
            sample["keyframe_target"] = np.asarray(targets[index], dtype=np.float32)
            return sample

    dataset = _TargetDataset()
    graphdef, params = nnx.split(model)
    keyframe_filter = nnx_utils.PathRegex(".*keyframe_head.*")
    tx = cfg.optimizer.create(cfg.lr_schedule.create(), weight_decay_mask=None)
    opt_state = tx.init(params.filter(keyframe_filter))

    @jax.jit
    def train_step(state_params, state_opt, batch_obs, batch_targets):
        model = nnx.merge(graphdef, state_params)

        def loss_fn(model):
            logits = model.predict_keyframe_logits(batch_obs, stop_gradient=True)
            loss = optax.sigmoid_binary_cross_entropy(logits, batch_targets).mean()
            probs = jax.nn.sigmoid(logits)
            mae = jnp.mean(jnp.abs(probs - batch_targets))
            return loss, {"loss": loss, "mae": mae, "avg_prob": jnp.mean(probs)}

        diff_state = nnx.DiffState(0, keyframe_filter)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
        trainable_params = state_params.filter(keyframe_filter)
        updates, new_opt_state = tx.update(grads, state_opt, trainable_params)
        new_params = optax.apply_updates(trainable_params, updates)
        nnx.update(model, new_params)
        metrics = {**metrics, "grad_norm": optax.global_norm(grads), "loss_total": loss}
        return nnx.state(model), new_opt_state, metrics

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        indices = random.sample(range(len(dataset)), k=min(args.batch_size, len(dataset)))
        batch_obs, batch_targets = _make_batch(dataset, indices)
        params, opt_state, metrics = train_step(params, opt_state, batch_obs, batch_targets)

        if step % args.log_interval == 0:
            logging.info("keyframe_train step=%d metrics=%s", step, {k: float(np.asarray(v)) for k, v in metrics.items()})

        if step % args.save_interval == 0 or step == args.steps:
            save_dir = output_dir / f"step_{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            with ocp.PyTreeCheckpointer() as ckptr:
                ckptr.save(
                    save_dir / "params",
                    {"params": params.to_pure_dict()},
                    force=True,
                )


if __name__ == "__main__":
    main()
