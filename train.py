#!/usr/bin/env python3
"""Thin wrapper so training can be launched from the repo root.

Delegates to training/launch.py, which uses Hydra to load configs from
training/config/<name>.yaml.

Usage:
  python train.py --config room_envelopes/e1_layout_depth_only_frozen
  torchrun --nproc_per_node=4 train.py --config room_envelopes/e4b_layout_depth_mask_normals_frozen
"""

import os
import sys

_repo_root = os.path.dirname(os.path.abspath(__file__))
_training_dir = os.path.join(_repo_root, "training")
sys.path.insert(0, _repo_root)
sys.path.insert(0, _training_dir)

os.chdir(_training_dir)

from launch import main  # noqa: E402

if __name__ == "__main__":
    main()
