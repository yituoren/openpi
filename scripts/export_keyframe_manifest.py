#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
from PIL import Image

from openpi.training import config as train_config
from openpi.training import data_loader as _data_loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--camera-key", type=str, default=None)
    return parser.parse_args()


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def main() -> None:
    args = _parse_args()
    cfg = train_config.get_config(args.config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)

    output_dir = pathlib.Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    count = 0
    with manifest_path.open("w", encoding="utf-8") as f:
        for dataset_index in range(0, len(dataset), max(args.stride, 1)):
            sample = dataset[dataset_index]
            image_dict = sample["image"]
            camera_key = args.camera_key or next(iter(image_dict))
            image = np.asarray(image_dict[camera_key])
            if image.dtype != np.uint8:
                image = np.clip((image + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            image_path = image_dir / f"{dataset_index:08d}_{camera_key}.png"
            Image.fromarray(image).save(image_path)

            record = {
                "dataset_index": dataset_index,
                "image_path": str(image_path.relative_to(output_dir)),
                "camera_key": camera_key,
                "episode_index": _to_python(sample.get("episode_index")),
                "frame_index": _to_python(sample.get("frame_index")),
                "timestamp": _to_python(sample.get("timestamp")),
                "task_index": _to_python(sample.get("task_index")),
                "is_keyframe": None,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
            if count >= args.limit:
                break


if __name__ == "__main__":
    main()
