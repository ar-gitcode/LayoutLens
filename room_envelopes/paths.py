"""Centralized, environment-overridable filesystem paths.

External data and checkpoints live *outside* the repo and are not redistributed
with it. Their locations default to ``/path/to/...`` placeholders that must be
set for the code to run, by exporting the corresponding ``ROOMENV_*`` environment
variables. Setting the variables relocates every path at once; no code or config
edits are required.

Python scripts import the constants from here; Hydra YAML configs read the same
environment variables via ``${oc.env:ROOMENV_...,<default>}`` interpolation, so
the two stay in sync.

Override examples::

    export ROOMENV_DATA_DIR=/data/room_envelopes
    export ROOMENV_WEIGHTS_DIR=/checkpoints/vggt
"""

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "DATA_WDS_DIR",
    "EXTRINSICS_MANIFEST",
    "WEIGHTS_DIR",
    "EVAL_CACHE_DIR",
    "REPO_EXTRINSICS_MANIFEST",
]


def _env(var: str, default: str) -> str:
    """Return ``$var`` if set (and non-empty), else ``default``."""
    value = os.environ.get(var)
    return value if value else default


# Repo root = parent of the room_envelopes/ package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

# --- External datasets ------------------------------------------------------
DATA_DIR = _env("ROOMENV_DATA_DIR", "/path/to/datasets/room_envelopes")
DATA_WDS_DIR = _env("ROOMENV_DATA_WDS_DIR", f"{DATA_DIR}/data_wds_extracted")
EXTRINSICS_MANIFEST = _env(
    "ROOMENV_EXTRINSICS_MANIFEST", f"{DATA_DIR}/extrinsics_manifest.npz"
)

# --- External checkpoints ---------------------------------------------------
WEIGHTS_DIR = _env("ROOMENV_WEIGHTS_DIR", "/path/to/weights")

# --- In-repo defaults (relative to this repo) -------------------------------
# Seed-pinned eval-manifest JSONs ship inside the repo; default there but allow
# relocation. Earlier configs pointed these at the *previous* repo checkout.
EVAL_CACHE_DIR = _env(
    "ROOMENV_EVAL_CACHE_DIR", str(REPO_ROOT / "training" / "cache" / "room_envelopes")
)
# Validated real-extrinsics manifest tracked in the repo (eval input).
REPO_EXTRINSICS_MANIFEST = str(
    REPO_ROOT / "evaluations" / "room_envelopes" / "extrinsics_manifest.npz"
)
