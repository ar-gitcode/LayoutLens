"""Depth-independent ``sys.path`` bootstrap for the eval scripts.

Importing this module makes the repo root, the ``training/`` tree, and every
``evaluations/src/`` subfolder (``common``, ``2d``, ``3d``, ``manifests``, …)
importable by bare module name. Cross-module imports across the eval tree are
therefore flat (``from pointcloud import ...``) rather than package-qualified,
required because folders like ``2d``/``3d`` cannot be Python packages (module
names may not start with a digit).

This module intentionally performs **no** ``os.chdir`` so that importing it
never changes the caller's working directory. Runners that need
``cwd == training/`` do their own ``os.chdir(_training_dir)``.
"""

import os
import sys

_paths_dir = os.path.dirname(os.path.abspath(__file__))      # .../evaluations/src/common
EVAL_ROOT = os.path.dirname(_paths_dir)                      # .../evaluations/src
REPO_ROOT = os.path.dirname(os.path.dirname(EVAL_ROOT))      # .../<repo>
TRAINING_DIR = os.path.join(REPO_ROOT, "training")

_dirs = [TRAINING_DIR, REPO_ROOT, EVAL_ROOT]
for _name in sorted(os.listdir(EVAL_ROOT)):
    _sub = os.path.join(EVAL_ROOT, _name)
    if os.path.isdir(_sub) and not _name.startswith("__"):
        _dirs.append(_sub)

for _p in _dirs:
    if _p not in sys.path:
        sys.path.insert(0, _p)
