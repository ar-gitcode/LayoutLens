#!/usr/bin/env bash
# Example evaluation launch for LayoutLens.
#
# Runs 3D room-envelope reconstruction for a trained checkpoint using
# ground-truth cameras. Edit CONFIG to match the checkpoint's experiment; see
# docs/configs.md and docs/evaluation.md.
#
# Usage:
#   bash scripts/evaluate_example.sh /path/to/checkpoint.pt
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <checkpoint.pt>" >&2
  exit 1
fi

CHECKPOINT="$1"
CONFIG="${CONFIG:-room_envelopes/e1b_uf12_layout_depth_only_original_regression}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_out/reconstruction}"

cd "$(dirname "$0")/.."

python evaluations/src/3d/eval_room_envelope_reconstruction.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --camera_mode gt
