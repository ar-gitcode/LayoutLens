#!/usr/bin/env bash
# Example training launch for LayoutLens.
#
# Trains the "Layout head, last 12 blocks unfrozen" configuration on the Room
# Envelopes dataset. Edit CONFIG to select a different experiment; see
# docs/configs.md for the readable-name mapping.
#
# Prerequisites:
#   pip install -e ".[train]"
#   export ROOMENV_DATA_DIR=/path/to/datasets/room_envelopes
#   export ROOMENV_WEIGHTS_DIR=/path/to/weights
#
# Usage:
#   bash scripts/train_example.sh            # single GPU
#   NUM_GPUS=4 bash scripts/train_example.sh # multi-GPU via torchrun
set -euo pipefail

CONFIG="${CONFIG:-room_envelopes/e1b_uf12_layout_depth_only_original_regression}"
NUM_GPUS="${NUM_GPUS:-1}"

cd "$(dirname "$0")/.."

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" train.py --config "${CONFIG}"
else
  python train.py --config "${CONFIG}"
fi
