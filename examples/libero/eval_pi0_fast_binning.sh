#!/bin/bash
set -euo pipefail

TASK_SUITES=("libero_10")

COMPOSE_FILE=examples/libero/compose.yml
SERVER_ARGS_VALUE="--env LIBERO policy:checkpoint --policy.config pi0_fast_binning_libero --policy.dir checkpoints/pi0_fast_binning_libero/test/99999"

# Inside the libero container, the host's `data/` is mounted at `/data` (see compose.yml).
# Per-suite rollouts are dumped under this directory and then converted to LeRobot format.
ROLLOUT_DIR_IN_CONTAINER=/data/libero/rollouts/pi0_fast_binning_libero
ROLLOUT_DIR_ON_HOST=data/libero/rollouts/pi0_fast_binning_libero
LEROBOT_REPO_ID=local/eval_pi0_fast_binning_libero

cleanup() {
  docker compose -f "${COMPOSE_FILE}" down --remove-orphans
}

trap cleanup EXIT

for SUITE in "${TASK_SUITES[@]}"; do
  echo "========== Evaluating task suite: $SUITE =========="
  SERVER_ARGS="${SERVER_ARGS_VALUE}" \
  CLIENT_ARGS="--args.task-suite-name $SUITE --args.num-trials-per-task 100 --args.rollout-save-path ${ROLLOUT_DIR_IN_CONTAINER}/${SUITE}" \
    docker compose -f "${COMPOSE_FILE}" up --build --exit-code-from runtime
  cleanup
  echo "========== Finished: $SUITE =========="
done

trap - EXIT

export HF_LEROBOT_HOME="/mnt/data1/data"

echo "========== Converting rollouts to LeRobot format =========="
uv run examples/libero/convert_libero_rollouts_to_lerobot.py \
  --rollout-dir "${ROLLOUT_DIR_ON_HOST}" \
  --repo-id "${LEROBOT_REPO_ID}"
