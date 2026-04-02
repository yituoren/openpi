CUDA_VISIBLE_DEVICES=4 uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi0_fast_binning_libero \
    --policy.dir checkpoints/pi0_fast_binning_libero/test/29999