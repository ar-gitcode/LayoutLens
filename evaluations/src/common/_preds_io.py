"""On-disk prediction artifact I/O for the 2-pass eval.

Pass 1 (``--preds-out`` [+ ``--forward-only``]) runs the model forward and
writes one ``scene_<i>.npz`` prediction shard per scene plus a per-manifest
``index.json`` and a run-level ``run_identity.json``. Pass 2 (``--preds-in``)
loads those shards on a CPU-only box, no model, no GPU, no checkpoint load,
re-fetches GT deterministically from the manifest, and re-scores with the exact
same metric code.

Design notes (see evaluations/README.md "Two-pass eval"):
- Only the prediction keys that any metric path actually consumes are saved:
  ``layout_depth`` (or ``depth`` for E0), ``pose_enc``, ``layout_mask_logits``,
  ``layout_normal``. Everything else VGGT emits (images, pose_enc_list, *_conf,
  world_points, seg_logits, track/vis/conf) is never read by scoring → dropped.
- Depth and pose are precision-critical (chamfer / F-score / delta thresholds /
  pose decode) and are ALWAYS fp32. Mask/normal are robust (sigmoid@0.5 /
  arccos) and may be fp16 under ``--preds-dtype fp16-aux``.
- Identity is two-tier: prediction identity (did the same model produce these?)
  and scoring identity (same GT?). Absolute paths are excluded from the cfg
  hashes so the GPU save-box and CPU score-box may differ on DATA_DIR etc.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np

# Bump when the on-disk layout / keep-set / hashing changes incompatibly.
PREDS_SCHEMA_VERSION = 1

# Aux (non-depth, non-pose) prediction keys, in save order.
_AUX_KEYS = ("layout_mask_logits", "layout_normal")
_POSE_KEY = "pose_enc"
# Keys eligible for fp16 storage under 'fp16-aux' (robust metrics only).
_FP16_OK = {"layout_mask_logits", "layout_normal"}

# Dataset-cfg / cfg keys whose values are filesystem paths that legitimately
# differ between the save box and the score box; excluded from cfg hashes.
_PATH_KEY_HINTS = ("dir", "manifest", "path")


# ---------------------------------------------------------------------------
# numpy / tensor helpers
# ---------------------------------------------------------------------------

def _to_np(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


# ---------------------------------------------------------------------------
# prediction keep-set + dtype policy
# ---------------------------------------------------------------------------

def select_pred_keys(preds_one: dict, use_depth_as_layout: bool) -> list[str]:
    """Return the keep-set of prediction keys present in ``preds_one``.

    Depth source: ``layout_depth`` if present, else ``depth`` (E0 / when
    use_depth_as_layout). Aux + pose keys are included only if present.
    """
    keys: list[str] = []
    if "layout_depth" in preds_one:
        keys.append("layout_depth")
    elif "depth" in preds_one:
        keys.append("depth")
    for k in _AUX_KEYS:
        if k in preds_one:
            keys.append(k)
    if _POSE_KEY in preds_one:
        keys.append(_POSE_KEY)
    return keys


def pred_dtypes_for(keys: list[str], dtype_mode: str) -> dict[str, str]:
    """Map each kept key to its on-disk dtype string for the chosen policy."""
    out = {}
    for k in keys:
        out[k] = "float16" if (dtype_mode == "fp16-aux" and k in _FP16_OK) else "float32"
    return out


def to_shard_arrays(preds_one: dict, keys: list[str], dtype_mode: str) -> dict:
    """Materialize the keep-set to contiguous numpy at the policy dtype.

    ``dtype_mode``: ``'fp32'`` (all fp32, bit-identical metrics) or
    ``'fp16-aux'`` (mask/normal fp16; depth+pose stay fp32).
    """
    dts = pred_dtypes_for(keys, dtype_mode)
    out = {}
    for k in keys:
        a = _to_np(preds_one[k])
        target = np.float16 if dts[k] == "float16" else np.float32
        out[k] = np.ascontiguousarray(a.astype(target))
    return out


# ---------------------------------------------------------------------------
# shard save / load
# ---------------------------------------------------------------------------

def save_scene_shard(path: str, arrays: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Uncompressed: fastest write; mmap-friendly, JSON-free tensor blobs.
    np.savez(path, **arrays)


def load_scene_shard(path: str) -> dict:
    """Load a shard as ``{key: np.ndarray}``.

    fp16 aux keys are upcast to fp32 on load so all downstream math runs in
    fp32 exactly as in a normal (non-cached) run.
    """
    out = {}
    with np.load(path) as z:
        for k in z.files:
            a = z[k]
            if a.dtype == np.float16:
                a = a.astype(np.float32)
            out[k] = a
    return out


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _strip_paths(obj):
    """Recursively replace path-valued entries (by key hint) with a sentinel so
    cfg hashes are stable across boxes with different DATA_DIR / manifest paths.
    """
    if isinstance(obj, dict):
        return {
            k: ("<path>" if any(h in str(k).lower() for h in _PATH_KEY_HINTS)
                else _strip_paths(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_paths(v) for v in obj]
    return obj


def model_cfg_hash(cfg) -> str:
    """sha256 of the resolved ``cfg.model``, the architecture/head set that
    determines the predictions."""
    from omegaconf import OmegaConf
    return _sha256_json(OmegaConf.to_container(cfg.model, resolve=True))


def dataset_cfg_hash(cfg, split: str) -> str:
    """sha256 of the resolved data-split block with absolute paths stripped,
    the GT semantics (mask_source, use_real_extrinsics, load_mask/normal, …)."""
    from omegaconf import OmegaConf
    try:
        # Mirror select_split's test-from-val synthesis so the hash matches what
        # is actually used. Import lazily to avoid a hard dependency cycle.
        from _common import select_split
        block = select_split(cfg, split)
        cont = OmegaConf.to_container(block, resolve=True)
    except Exception:
        cont = {}
    return _sha256_json(_strip_paths(cont))


def manifest_content_hash(manifest: dict) -> str:
    """sha256 over the manifest's GT-determining content: meta (minus paths) +
    the ordered list of (scene_cam, frame ids) per sample."""
    meta = manifest.get("meta", {}) or {}
    samples = manifest.get("samples", []) or []
    payload = {
        "meta": {k: meta.get(k) for k in
                 ("strategy", "split", "num_views", "base_seed", "shuffle")},
        "samples": [[s.get("scene_cam"), list(s.get("ids", []))] for s in samples],
    }
    return _sha256_json(payload)


# ---------------------------------------------------------------------------
# run identity (fail-loud)
# ---------------------------------------------------------------------------

def build_run_identity(*, config_name, model_cfg_sha256, checkpoint_path,
                       checkpoint_sha256, checkpoint_epoch, heads, pred_keys,
                       pred_dtypes, use_depth_as_layout, extrinsics_convention,
                       image_size, preds_dtype_mode, split, seed, max_samples,
                       dataset_cfg_sha256, manifests, git_commit, device,
                       timestamp_utc) -> dict:
    return {
        "schema_version": PREDS_SCHEMA_VERSION,
        "prediction_identity": {
            "config_name": config_name,
            "model_cfg_sha256": model_cfg_sha256,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": checkpoint_epoch,
            "heads": heads,
            "pred_keys": list(pred_keys),
            "pred_dtypes": pred_dtypes,
            "preds_dtype_mode": preds_dtype_mode,
            "use_depth_as_layout": bool(use_depth_as_layout),
            "extrinsics_convention": extrinsics_convention,
            "image_size": [int(image_size[0]), int(image_size[1])],
        },
        "scoring_identity": {
            "split": split,
            "seed": int(seed),
            "max_samples": (int(max_samples) if max_samples is not None else None),
            "dataset_cfg_sha256": dataset_cfg_sha256,
            "manifests": manifests,  # [{label, num_views, n_samples, content_sha256}]
        },
        "informational": {
            "git_commit": git_commit,
            "device": str(device),
            "timestamp_utc": timestamp_utc,
        },
    }


def validate_prediction_identity(saved: dict, *, cfg, split,
                                 checkpoint_sha256: str | None,
                                 extrinsics_convention: str,
                                 requested_camera_modes: list[str]) -> list[str]:
    """Return a list of HARD-FAIL messages (empty == OK) for the prediction +
    scoring-config identity (manifest set is validated separately)."""
    fails: list[str] = []
    pid = saved.get("prediction_identity", {})
    sid = saved.get("scoring_identity", {})

    if saved.get("schema_version") != PREDS_SCHEMA_VERSION:
        fails.append(
            f"schema_version mismatch: artifact={saved.get('schema_version')} "
            f"!= runtime={PREDS_SCHEMA_VERSION}")

    cur_model = model_cfg_hash(cfg)
    if pid.get("model_cfg_sha256") != cur_model:
        fails.append(
            f"model cfg mismatch: artifact={pid.get('model_cfg_sha256')} "
            f"!= --config resolved={cur_model} (different model/heads)")

    cur_ds = dataset_cfg_hash(cfg, split)
    if sid.get("dataset_cfg_sha256") != cur_ds:
        fails.append(
            f"dataset cfg (GT semantics) mismatch: artifact={sid.get('dataset_cfg_sha256')} "
            f"!= --config resolved={cur_ds}")

    if checkpoint_sha256 is not None and pid.get("checkpoint_sha256") != checkpoint_sha256:
        fails.append(
            f"checkpoint mismatch: artifact={pid.get('checkpoint_sha256')} "
            f"!= --checkpoint={checkpoint_sha256}")

    if extrinsics_convention != pid.get("extrinsics_convention"):
        fails.append(
            f"extrinsics_convention mismatch: artifact={pid.get('extrinsics_convention')} "
            f"!= runtime={extrinsics_convention}")

    if "pred" in requested_camera_modes and _POSE_KEY not in pid.get("pred_keys", []):
        fails.append(
            "--camera-mode pred/both requires 'pose_enc' in the saved preds, "
            "but the artifact has none (no camera head at save time)")
    return fails


def validate_manifest_set(saved: dict, discovered: dict) -> list[str]:
    """Validate the manifests pass 2 is about to RUN against the artifact.

    ``discovered`` maps label -> content_sha256 for the manifests pass 2
    discovered (already narrowed by --split / --only-view-counts / etc.). The
    rule is ``discovered ⊆ saved``: every manifest we are about to score must
    have saved predictions in the artifact AND a matching content hash. Saved
    manifests the user chose NOT to run (in the artifact but not discovered,
    e.g. ``--only-view-counts`` filtered them out) are fine and skipped by the
    caller; they are NOT a failure. Returns HARD-FAIL messages only for a
    discovered manifest that has no saved preds, or whose content drifted.
    """
    fails: list[str] = []
    saved_m = {m["label"]: m for m in saved.get("scoring_identity", {}).get("manifests", [])}
    for label, dhash in discovered.items():
        if label not in saved_m:
            fails.append(
                f"manifest '{label}' is being run in pass 2 but has no saved "
                f"predictions in the artifact (--preds-in). Restrict the run to "
                f"saved manifests (e.g. --only-view-counts) or re-run pass 1 for it.")
        elif dhash != saved_m[label].get("content_sha256"):
            fails.append(
                f"manifest '{label}' content hash mismatch: "
                f"artifact={saved_m[label].get('content_sha256')} != discovered={dhash}")
    return fails


# ---------------------------------------------------------------------------
# run-level + per-manifest index files
# ---------------------------------------------------------------------------

def run_identity_path(preds_dir: str) -> str:
    return os.path.join(preds_dir, "run_identity.json")


def write_run_identity(preds_dir: str, identity: dict) -> None:
    os.makedirs(preds_dir, exist_ok=True)
    with open(run_identity_path(preds_dir), "w") as f:
        json.dump(identity, f, indent=2, default=str)


def read_run_identity(preds_dir: str) -> dict:
    with open(run_identity_path(preds_dir)) as f:
        return json.load(f)


def manifest_preds_dir(preds_dir: str, split: str, label: str) -> str:
    return os.path.join(preds_dir, split, label)


def index_path(manifest_dir: str) -> str:
    return os.path.join(manifest_dir, "index.json")


def write_index(manifest_dir: str, index: dict) -> None:
    os.makedirs(manifest_dir, exist_ok=True)
    with open(index_path(manifest_dir), "w") as f:
        json.dump(index, f, indent=2, default=str)


def read_index(manifest_dir: str) -> dict:
    with open(index_path(manifest_dir)) as f:
        return json.load(f)


def scene_shard_path(manifest_dir: str, i: int) -> str:
    return os.path.join(manifest_dir, f"scene_{i:04d}.npz")
