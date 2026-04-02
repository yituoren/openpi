SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_fast_binning_libero --policy.dir checkpoints/pi0_fast_binning_libero/test/29999" \
CLIENT_ARGS="--args.task-suite-name libero_10 --args.num-trials-per-task 10" \
  docker compose -f examples/libero/compose.yml up --build