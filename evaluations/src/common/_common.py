"""Shared eval infrastructure used across the 2D and 3D eval scripts.

Single source of truth for:
- Hydra config name → ``(cfg_dir, cfg_name)`` resolution.
- Model + checkpoint loading (with strict=False bookkeeping).
- ``--all_unique_scenes``-style determinism on the inner dataset.
- ``cfg.data.<split>`` selection with a deep-copy fallback for ``test``.
- Per-record aggregation (mean across frames / scenes, NaN-tolerant).

``sys.path`` setup is owned by :mod:`_paths` (imported by the entry-point
runner before this module loads); the per-scene 2D metric code now lives in
``2d/scene_metrics.py`` and the per-concern ``2d/metrics_*.py`` modules.
"""

from __future__ import annotations

import copy
import importlib
import os
from typing import Any

import numpy as np

from _paths import TRAINING_DIR as _training_dir


# ---------------------------------------------------------------------------
# numpy / list utilities
# ---------------------------------------------------------------------------

def to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        try:
            return np.asarray(x)
        except Exception:
            return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def frame(arr, s):
    if isinstance(arr, list):
        return np.asarray(arr[s])
    return np.asarray(arr[s])


# ---------------------------------------------------------------------------
# Hydra config + checkpoint loading
# ---------------------------------------------------------------------------

def resolve_config(name: str) -> tuple[str, str]:
    """Split ``a/b/c`` into ``(<training>/config/a/b, c)`` for Hydra."""
    parts = name.rsplit("/", 1)
    if len(parts) == 2:
        return os.path.join(_training_dir, "config", parts[0]), parts[1]
    return os.path.join(_training_dir, "config"), parts[0]


def load_cfg(config_name: str):
    """Compose the Hydra config **without** building the model or loading a
    checkpoint.

    Used by the CPU-only re-scoring path (``--preds-in``), which needs the cfg
    (dataset target + model head flags) but neither VGGT nor the checkpoint
    weights. Mirrors the chdir + compose that :func:`load_model_and_cfg` does.
    """
    from hydra import initialize_config_dir, compose

    # Hydra config resolution is happier with cwd == training/.
    os.chdir(_training_dir)
    cfg_dir, cfg_name = resolve_config(config_name)
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=cfg_name)
    return cfg


def load_model_and_cfg(config_name: str,
                       checkpoint_path: str | None,
                       device_str: str):
    """Load a Hydra config, build the model, and (optionally) load a checkpoint.

    Returns ``(cfg, model, device, ckpt_meta)`` where ``ckpt_meta`` is a dict
    with keys::

        {
          "checkpoint_path": str | None,
          "checkpoint_epoch": int | None,
          "n_missing_keys": int,
          "n_unexpected_keys": int,
          "missing_keys": list[str],
          "unexpected_keys": list[str],
        }

    Hydra needs ``cwd == training/`` for some config resolutions; this
    function does the chdir itself, so callers don't have to.
    """
    import torch
    from omegaconf import OmegaConf

    cfg = load_cfg(config_name)

    device = torch.device(device_str)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_target = model_cfg.pop("_target_")
    mod_path, cls_name = model_target.rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(mod_path), cls_name)
    model = ModelClass(**model_cfg).to(device)

    ckpt_meta: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": None,
        "n_missing_keys": 0,
        "n_unexpected_keys": 0,
        "missing_keys": [],
        "unexpected_keys": [],
    }

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict):
            # Different training setups use different key names. `prev_epoch`
            # is the trainer.py default (last completed epoch). Fall back to
            # `epoch` for older / external checkpoints.
            for ep_key in ("prev_epoch", "epoch"):
                ep = ckpt.get(ep_key)
                if isinstance(ep, (int, np.integer)):
                    ckpt_meta["checkpoint_epoch"] = int(ep)
                    break
        ckpt_meta["n_missing_keys"] = len(missing)
        ckpt_meta["n_unexpected_keys"] = len(unexpected)
        ckpt_meta["missing_keys"] = list(missing)
        ckpt_meta["unexpected_keys"] = list(unexpected)
        print(f"[checkpoint] loaded {checkpoint_path}")
        if missing:
            print(f"[checkpoint] missing keys (truncated): {list(missing)[:6]}{'…' if len(missing) > 6 else ''}")
        if unexpected:
            print(f"[checkpoint] unexpected keys (truncated): {list(unexpected)[:6]}{'…' if len(unexpected) > 6 else ''}")
    else:
        print("[checkpoint] no checkpoint provided, evaluating randomly initialized / pretrained model")

    model.eval()
    return cfg, model, device, ckpt_meta


# ---------------------------------------------------------------------------
# Dataset construction, split selection + deterministic mode
# ---------------------------------------------------------------------------

def enable_unique_scene_mode(inner_ds) -> int:
    """Switch a ComposedDataset(TupleConcatDataset(child datasets)) into a
    deterministic 'one item per unique sequence' mode.

    Disables both layers of inside_random (TupleConcatDataset and each child
    dataset), overrides each child's reported length to its unique-sequence
    count, and rebuilds ConcatDataset's cumulative_sizes so index dispatch
    maps [0, sum(sequence_list_len)) correctly. Returns the total count.
    """
    from torch.utils.data import ConcatDataset

    tcd = inner_ds.base_dataset
    tcd.inside_random = False
    total_unique = 0
    for child in tcd.datasets:
        child.inside_random = False
        unique = getattr(child, "sequence_list_len", None)
        if unique is None:
            raise RuntimeError(
                f"--all_unique_scenes requires the dataset to expose "
                f"`sequence_list_len`; {type(child).__name__} does not."
            )
        child.len_train = unique
        total_unique += unique
    tcd.cumulative_sizes = ConcatDataset.cumsum(tcd.datasets)
    return total_unique


def select_split(cfg, split: str):
    """Return cfg.data.<split> entry.

    Most Phase 1 configs only define train/val, but inside the val block,
    the dataset's ``split`` kwarg is hardcoded to ``"val"``, so a naive
    fallback silently evaluates on val rooms when the user asked for test.

    To fix this properly, when split=='test' and cfg.data.test is missing,
    we synthesize a real test split by deep-copying cfg.data.val and
    rewriting each dataset_config's ``split: val`` → ``test`` and
    ``len_val`` → ``len_test``. This way ``--split test`` actually iterates
    the dataset's test scenes rather than its validation scenes.
    """
    if split in cfg.data:
        return cfg.data[split]
    if split == "test" and "val" in cfg.data:
        from omegaconf import OmegaConf
        synthesized = copy.deepcopy(cfg.data.val)
        # Hydra-composed configs are in struct mode, which forbids adding
        # keys that weren't in the original (len_test isn't, since the val
        # block only declares len_val). Relax it for our mutation.
        OmegaConf.set_struct(synthesized, False)
        # ComposedDataset wrappers always nest dataset_configs under .dataset.
        for dc in synthesized.dataset.dataset_configs:
            if "split" in dc:
                dc.split = "test"
            if "len_val" in dc:
                dc.len_test = dc.len_val
                del dc.len_val
        print("[data] split='test' synthesized from cfg.data.val "
              "(rewrote dataset_configs[*].split→test, len_val→len_test). "
              "Add an explicit cfg.data.test block to silence this.")
        return synthesized
    for fallback in ("val", "test", "train"):
        if fallback in cfg.data:
            print(f"[data] split '{split}' not in cfg; using '{fallback}'")
            return cfg.data[fallback]
    raise KeyError(f"No data splits found in cfg.data ({list(cfg.data)})")


# ---------------------------------------------------------------------------
# Aggregation (mean across frames / scenes, NaN-tolerant)
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    """Mean every numeric key across records, skipping NaNs and non-numerics.

    Matches the ``np.nanmean`` semantics in
    ``Trainer._finalize_val_metrics`` so 2D eval numbers line up with
    training-time val numbers.
    """
    if not records:
        return {}
    keys = set()
    for r in records:
        keys.update(r.keys())
    out = {}
    for k in keys:
        vals = [r[k] for r in records if k in r and r[k] is not None]
        try:
            arr = np.asarray(vals, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.size == 0:
            continue
        out[k] = float(np.nanmean(arr))
    return out
