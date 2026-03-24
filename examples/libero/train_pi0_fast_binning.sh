#!/bin/bash
# Training script for pi0-FAST with binning tokenizer on LIBERO.
#
# This uses a uniform binning tokenizer instead of the learned FAST tokenizer.
# Each continuous action dimension is discretized into 256 uniform bins over [-1, 1].
#
# Prerequisites:
#   - LIBERO data must be converted to LeRobot format first (see convert step below)
#   - Requires a GPU with sufficient memory (set XLA_PYTHON_CLIENT_MEM_FRACTION as needed)

set -euo pipefail

CONFIG_NAME="pi0_fast_binning_libero"

# Step 1: Convert LIBERO data to LeRobot format (skip if already done)
echo "=== Step 1: Converting LIBERO data to LeRobot format ==="
uv run python examples/libero/convert_libero_data_to_lerobot.py

# Step 2: Compute normalization statistics
echo "=== Step 2: Computing normalization statistics ==="
uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

# Step 3: Train
echo "=== Step 3: Training ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "$CONFIG_NAME"
