#!/bin/bash

TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")

for SUITE in "${TASK_SUITES[@]}"; do
  echo "========== Evaluating task suite: $SUITE =========="
  SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_fast_binning_libero --policy.dir checkpoints/pi0_fast_binning_libero/test/29999" \
  CLIENT_ARGS="--args.task-suite-name $SUITE --args.num-trials-per-task 10" \
    docker compose -f examples/libero/compose.yml up --build
  echo "========== Finished: $SUITE =========="
done
