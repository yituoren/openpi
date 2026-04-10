"""Convert local LIBERO eval rollouts (dumped by `examples/libero/main.py --rollout-save-path`)
into a LeRobot dataset.

The expected on-disk layout (one episode per leaf directory) is:

    {rollout_dir}/
        [optional suite subdirs]/
            taskNN_<task_segment>/
                epNNN_{success,failure}/
                    agentview.mp4
                    wrist.mp4
                    data.npz   # state, actions, task, success, task_suite, ...

Usage:
    uv run examples/libero/convert_libero_rollouts_to_lerobot.py \
        --rollout-dir data/libero/rollouts/pi0_fast_binning_libero \
        --repo-id local/eval_pi0_fast_binning_libero

The schema matches `convert_libero_data_to_lerobot.py` so the resulting dataset
is drop-in compatible with the existing training pipeline.
"""

import shutil
from pathlib import Path

import imageio.v3 as iio
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def main(
    rollout_dir: str,
    repo_id: str,
    *,
    success_only: bool = False,
    push_to_hub: bool = False,
    fps: int = 10,
) -> None:
    rollout_root = Path(rollout_dir)
    if not rollout_root.exists():
        raise FileNotFoundError(f"Rollout directory does not exist: {rollout_root}")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Each episode dir contains a `data.npz` alongside `agentview.mp4` and `wrist.mp4`.
    episode_dirs = sorted({p.parent for p in rollout_root.rglob("data.npz")})
    if not episode_dirs:
        raise RuntimeError(f"No episodes (data.npz) found under {rollout_root}")
    print(f"Found {len(episode_dirs)} episodes under {rollout_root}")

    n_added = 0
    n_skipped = 0
    for ep_dir in episode_dirs:
        data = np.load(ep_dir / "data.npz", allow_pickle=False)
        success = bool(data["success"])
        if success_only and not success:
            n_skipped += 1
            continue

        states = data["state"]
        actions = data["actions"]
        task = str(data["task"])

        agentview = iio.imread(ep_dir / "agentview.mp4")  # (T, H, W, 3) uint8
        wrist = iio.imread(ep_dir / "wrist.mp4")

        T = len(actions)
        if not (len(agentview) == T and len(wrist) == T and len(states) == T):
            print(
                f"Skipping {ep_dir}: length mismatch "
                f"(actions={T}, states={len(states)}, agentview={len(agentview)}, wrist={len(wrist)})"
            )
            n_skipped += 1
            continue

        for t in range(T):
            dataset.add_frame(
                {
                    "image": agentview[t],
                    "wrist_image": wrist[t],
                    "state": states[t].astype(np.float32),
                    "actions": actions[t].astype(np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()
        n_added += 1

    print(f"Saved {n_added} episodes ({n_skipped} skipped) to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "eval"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
