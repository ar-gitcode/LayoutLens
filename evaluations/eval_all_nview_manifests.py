#!/usr/bin/env python3
"""Evaluate one config/checkpoint on every N-view test manifest.

One command in -> 2D layout-depth metrics + 3D reconstruction metrics for
each discovered manifest, plus a cross-manifest summary. Reuses the canonical
per-scene metric helpers from :mod:`_common` and
:mod:`eval_room_envelope_reconstruction` so no metric formulas are
duplicated.

Optional geometric post-processing (--enable-postprocess) re-renders each
predicted scene through RANSAC planes / Manhattan-snapped planes / cuboid
geometry and feeds the resulting depth maps back through the same 2D and 3D
metric functions for a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random as _random_mod
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import numpy as np

# --- repo path setup (flat sys.path bootstrap; see common/_paths.py) --------
# Must run *before* any imports of repo-internal modules. The 2d/3d runners
# perform the same _paths bootstrap at their module top; running ours first
# makes subsequent imports idempotent (no-op chdir, harmless duplicate entries).
# This orchestrator lives at the evaluations/ root, one level above the eval
# source tree, so it descends into ``src/`` to reach ``common``.
_d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, os.path.join(_d, "common"))
import _paths  # noqa: E402: adds repo root, training, all eval subdirs to sys.path
from _paths import REPO_ROOT as _repo_root, TRAINING_DIR as _training_dir  # noqa: E402
# Preserve the user's invocation cwd so we can resolve CLI-supplied relative
# paths against it (the chdir below moves us into training/ for Hydra).
_USER_CWD = os.getcwd()
os.chdir(_training_dir)


def _resolve_user_path(path_str: str) -> Path:
    """Resolve a CLI path against the user's original cwd (not training/)."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (Path(_USER_CWD) / p).resolve()


# Default manifest directory: the manifest builder's output dir, anchored to an
# absolute path (TRAINING_DIR) so it is independent of the user's invocation cwd
# and the chdir into training/ above. Holds the seed-pinned 1..5-view manifests.
_DEFAULT_MANIFEST_DIR = os.path.join(_training_dir, "cache", "room_envelopes")

# Light-weight imports first (numpy already loaded above). torch/hydra are
# deferred to keep --dry-run-discovery fast.

from _common import (  # noqa: E402
    to_np,
    frame,
    aggregate,
    load_cfg,
    load_model_and_cfg,
    select_split,
)
import _preds_io as preds_io  # noqa: E402: 2-pass save/load artifact helpers
from scene_metrics import compute_2d_metrics_for_scene  # noqa: E402
from manifest import (  # noqa: E402
    _load_eval_manifest,
    _find_room_envelopes_child,
    _manifest_iter_items,
    _check_manifest_split,
    _resolve_seq_index,
)
from tensor_utils import _to_image_tensor, _strip_batch_dim  # noqa: E402
import _cli  # noqa: E402: shared argparse flag helpers
from _oca_eval_helpers import (  # noqa: E402
    cameras_from_sample,
    detect_heads,
    forward_model,
)
from eval_2d import (  # noqa: E402
    _assert_pred_shapes,
    _check_backbone_keys,
    _git_commit_sha,
    _json_safe,
    _scale_align_preds_per_frame,
)
from eval_room_envelope_reconstruction import (  # noqa: E402
    _compute_3d_metrics_for_scene,
    _resolve_alignment_tracks,
)
from normalization import normalize_sample_vggt_scene  # noqa: E402
from ply_io import write_ply  # noqa: E402
from eval_2d_postprocess import (  # noqa: E402
    _fuse_world_cloud,
    _camera_centres,
)
from eval_metrics import compute_depth_metrics_with_splits  # noqa: E402
from training.geometry.postprocess_planes import (  # noqa: E402
    fit_ransac_envelope,
    snap_to_manhattan,
)
from training.geometry.postprocess_cuboid import fit_cuboid_room  # noqa: E402
from training.geometry.render_planes_to_depth import (  # noqa: E402
    render_planes_to_zdepth_batch,
)


# Methods we know how to evaluate. Order is the default summary order.
ALL_METHODS = ("raw", "ransac", "manhattan", "cuboid")
POSTPROCESS_METHODS = ("ransac", "manhattan", "cuboid")
HOLE_POLICIES_ALL = ("fill", "mask", "zero")


# ---------------------------------------------------------------------------
# Manifest discovery
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    path: Path
    label: str            # "1view", ..., "5view", "mixed", or "single"
    sort_key: tuple       # (0, n) for N-view, (1, 0) for mixed, (2, 0) for single
    num_views: Optional[int]
    strategy: str
    meta_split: str
    n_samples: int


def _load_manifest_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _label_from_meta(filename: str, meta: dict) -> tuple[str, Optional[int]]:
    """Best-effort label inference from filename + meta. Used by --single-manifest.
    Returns (label, num_views_or_None)."""
    m_pv = re.match(r"^eval_manifest_(?:train|val|test)_(\d+)view_seed\d+\.json$",
                    filename)
    if m_pv:
        n = int(m_pv.group(1))
        return f"{n}view", n
    m_mx = re.match(r"^eval_manifest_(?:train|val|test)_mixed_.*_seed\d+\.json$",
                    filename)
    if m_mx:
        return "mixed", None
    if isinstance(meta.get("num_views"), int):
        n = int(meta["num_views"])
        return f"{n}view", n
    if "view_counts" in meta:
        return "mixed", None
    return "single", None


def discover_manifests(manifest_dir: Path,
                       split: str,
                       include_mixed: bool,
                       only_view_counts: Optional[set[int]],
                       manifest_glob_override: Optional[str]) -> list[ManifestEntry]:
    """Discover N-view test manifests under ``manifest_dir``.

    Filename-first: ``eval_manifest_{split}_{N}view_seed\\d+\\.json`` for the
    fixed-N case and ``eval_manifest_{split}_mixed_.*_seed\\d+\\.json`` for the
    mixed case. JSON metadata cross-checks the filename. Ambiguous duplicates
    raise.
    """
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"manifest dir not found: {manifest_dir}")

    pv_re = re.compile(rf"^eval_manifest_{re.escape(split)}_(\d+)view_seed\d+\.json$")
    mx_re = re.compile(rf"^eval_manifest_{re.escape(split)}_mixed_.*_seed\d+\.json$")

    candidates: list[Path] = []
    if manifest_glob_override:
        candidates = sorted(manifest_dir.glob(manifest_glob_override))
    else:
        candidates = sorted(p for p in manifest_dir.iterdir()
                            if p.is_file() and p.suffix == ".json")

    by_label: dict[str, list[ManifestEntry]] = {}
    skipped_reasons: list[str] = []

    for path in candidates:
        name = path.name
        try:
            manifest = _load_manifest_json(path)
        except (OSError, json.JSONDecodeError) as e:
            skipped_reasons.append(f"{name}: cannot parse ({e})")
            continue
        meta = manifest.get("meta", {}) or {}
        meta_split = str(meta.get("split", ""))

        if manifest_glob_override is None:
            # Strict filename-first matching, with metadata cross-check.
            m_pv = pv_re.match(name)
            m_mx = mx_re.match(name)
            if m_pv:
                n = int(m_pv.group(1))
                meta_nv = meta.get("num_views")
                if meta_nv is not None and int(meta_nv) != n:
                    raise RuntimeError(
                        f"manifest {name}: filename N={n} disagrees with "
                        f"meta.num_views={meta_nv}"
                    )
                if meta_split and meta_split != split:
                    skipped_reasons.append(f"{name}: meta.split={meta_split!r} != {split!r}")
                    continue
                entry = ManifestEntry(
                    path=path, label=f"{n}view", sort_key=(0, n),
                    num_views=n, strategy=str(meta.get("strategy", "")),
                    meta_split=meta_split or split,
                    n_samples=len(manifest.get("samples", [])),
                )
                by_label.setdefault(entry.label, []).append(entry)
            elif m_mx:
                if not include_mixed:
                    skipped_reasons.append(f"{name}: mixed manifest (use --include-mixed)")
                    continue
                if meta_split and meta_split != split:
                    skipped_reasons.append(f"{name}: meta.split={meta_split!r} != {split!r}")
                    continue
                if "view_counts" not in meta:
                    skipped_reasons.append(f"{name}: filename says mixed but meta has no view_counts")
                    continue
                entry = ManifestEntry(
                    path=path, label="mixed", sort_key=(1, 0),
                    num_views=None, strategy=str(meta.get("strategy", "")),
                    meta_split=meta_split or split,
                    n_samples=len(manifest.get("samples", [])),
                )
                by_label.setdefault(entry.label, []).append(entry)
            else:
                # Skip everything else silently, pickle files etc.
                continue
        else:
            # Glob override: be lenient on filename, strict on metadata.
            if meta_split and meta_split != split:
                skipped_reasons.append(f"{name}: meta.split={meta_split!r} != {split!r}")
                continue
            label, nv = _label_from_meta(name, meta)
            sort_key = (0, nv) if isinstance(nv, int) else (1, 0)
            entry = ManifestEntry(
                path=path, label=label, sort_key=sort_key,
                num_views=nv, strategy=str(meta.get("strategy", "")),
                meta_split=meta_split or split,
                n_samples=len(manifest.get("samples", [])),
            )
            by_label.setdefault(label, []).append(entry)

    if only_view_counts is not None:
        kept: dict[str, list[ManifestEntry]] = {}
        for lbl, lst in by_label.items():
            if lbl == "mixed" and include_mixed:
                kept[lbl] = lst
                continue
            m = re.match(r"^(\d+)view$", lbl)
            if m and int(m.group(1)) in only_view_counts:
                kept[lbl] = lst
        by_label = kept

    # Ambiguity check.
    dupes = {lbl: lst for lbl, lst in by_label.items() if len(lst) > 1}
    if dupes:
        msg_lines = ["multiple manifests resolved to the same label:"]
        for lbl, lst in dupes.items():
            msg_lines.append(f"  {lbl}:")
            for e in lst:
                msg_lines.append(f"    {e.path}")
        msg_lines.append("pass --manifest-glob or --only-view-counts to disambiguate.")
        raise RuntimeError("\n".join(msg_lines))

    entries = [lst[0] for lst in by_label.values()]
    entries.sort(key=lambda e: e.sort_key)
    return entries


def print_discovery_table(entries: list[ManifestEntry]) -> None:
    if not entries:
        print("Discovered test manifests: (none)")
        return
    print(f"Discovered test manifests: {len(entries)}")
    for e in entries:
        nv = "mixed" if e.num_views is None else f"N={e.num_views}"
        print(f"  {e.label:>7}  {nv:>7}  n_samples={e.n_samples:<6}  "
              f"strategy={e.strategy}  path={e.path}")


def single_manifest_entry(path: Path, split: str) -> ManifestEntry:
    manifest = _load_manifest_json(path)
    meta = manifest.get("meta", {}) or {}
    label, nv = _label_from_meta(path.name, meta)
    sort_key = (2, 0)
    return ManifestEntry(
        path=path, label=label, sort_key=sort_key,
        num_views=nv, strategy=str(meta.get("strategy", "")),
        meta_split=str(meta.get("split", split)),
        n_samples=len(manifest.get("samples", [])),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate one model on every N-view test manifest, with "
                    "optional geometric post-processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Required for non-discovery runs (discovery itself only needs --manifest-dir).
    p.add_argument("--config", default=None,
                   help="Hydra config name, e.g. room_envelopes/e4_layout_depth_mask_normals_frozen")
    p.add_argument("--checkpoint", default=None, help="Path to .pt")
    p.add_argument("--manifest-dir", default=_DEFAULT_MANIFEST_DIR,
                   help="Dir containing manifest JSONs (omit only with "
                        "--single-manifest). Defaults to the manifest builder's "
                        f"output dir ({_DEFAULT_MANIFEST_DIR}).")
    p.add_argument("--output-dir", default=None, help="Root output dir")

    # Common
    _cli.add_split(p, default="test")
    _cli.add_device(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-debug-every", type=int, default=0)
    p.add_argument("--dry-run-discovery", action="store_true",
                   help="List discovered manifests and exit 0 (no model load).")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--allow-split-mismatch", action="store_true")

    # 3D camera / mask / alignment
    _cli.add_camera_mode(p, allow_both=True)
    _ALIGNMENT_CHOICES = (
        "auto", "none", "scale", "sim3", "scale_shift", "scale_shift_cam",
        "all", "all_cam",
    )
    p.add_argument("--alignment", default="auto",
                   choices=_ALIGNMENT_CHOICES,
                   help="Global alignment policy; applied to every eval space "
                        "unless overridden by --metric-alignment / "
                        "--vggt-scene-alignment.")
    # Per-eval-space overrides. ``None`` → fall back to --alignment.
    p.add_argument("--metric-alignment", default=None,
                   choices=_ALIGNMENT_CHOICES,
                   help="Override --alignment for the metric eval space only.")
    p.add_argument("--vggt-scene-alignment", default=None,
                   choices=_ALIGNMENT_CHOICES,
                   help="Override --alignment for the vggt_scene eval space only.")
    p.add_argument("--scale-alignment", default="median_depth",
                   choices=("median_depth", "pointcloud_rms"),
                   help="3D scale alignment track (NOT 2D scale; see --scale-align-2d).")
    _HEADLINE_CHOICES = (
        "raw", "scale", "sim3", "scale_shift", "scale_shift_cam",
    )
    p.add_argument("--headline-alignment", default=None,
                   choices=_HEADLINE_CHOICES,
                   help="Global headline-track override; applied to every eval "
                        "space unless overridden by --metric-headline-alignment "
                        "/ --vggt-scene-headline-alignment.")
    p.add_argument("--metric-headline-alignment", default=None,
                   choices=_HEADLINE_CHOICES,
                   help="Override --headline-alignment for the metric eval space only.")
    p.add_argument("--vggt-scene-headline-alignment", default=None,
                   choices=_HEADLINE_CHOICES,
                   help="Override --headline-alignment for the vggt_scene eval space only.")
    p.add_argument("--eval-space", default="metric",
                   choices=("metric", "vggt_scene", "both"),
                   help="Headline eval space. Default 'metric' (real-world "
                        "metres; alignment is applied in metric space). Pass "
                        "'vggt_scene' for the training-eval scene frame, or "
                        "'both' to emit both passes (reproduces the old default).")
    _cli.add_use_depth_as_layout(p)
    _cli.add_extrinsics_convention(p)
    _cli.add_max_points_per_scene(p)
    p.add_argument(
        "--kdtree-workers", type=int, default=-1,
        help="Workers forwarded to scipy.spatial.cKDTree.query for chamfer / "
             "F-score nearest-neighbour queries. -1 (default) uses all cores; "
             "set to 1 if you wrap this script in a per-scene multiprocessing "
             "pool to avoid CPU oversubscription.",
    )
    p.add_argument("--gt-cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Cache the GT cloud / KD-tree / seen-unseen split trees "
                        "once per (scene, eval-space) and reuse across "
                        "raw/ransac/manhattan/cuboid (GT is method-independent). "
                        "DEFAULT ON; value-preserving. --no-gt-cache rebuilds GT "
                        "per method (legacy behaviour).")

    # 2D scale-align (separate knob from 3D --scale-alignment).
    p.add_argument("--scale-align-2d", default="per_frame",
                   choices=("per_frame", "none"),
                   help="Per-frame median pred/gt depth scaling for metric-space "
                        "2D eval. Default 'per_frame' matches existing eval_2d.py.")

    # Manifest selection
    p.add_argument("--include-mixed", action="store_true")
    p.add_argument("--only-view-counts", default=None,
                   help="Comma list, e.g. '1,2,3'")
    p.add_argument("--manifest-glob", default=None)
    p.add_argument("--single-manifest", default=None,
                   help="PATH; bypass discovery, run on this one manifest only.")

    # 2-pass eval (save predictions / re-score). See README "Two-pass eval".
    p.add_argument("--preds-out", "--preds_out", dest="preds_out", default=None,
                   help="Dir to SAVE per-scene prediction shards + identity "
                        "(pass 1). Combine with --forward-only to skip scoring.")
    p.add_argument("--forward-only", "--forward_only", dest="forward_only",
                   action="store_true",
                   help="Pass 1: run the model forward and save preds only; skip "
                        "all 2D/3D scoring. Requires --preds-out.")
    p.add_argument("--preds-in", "--preds_in", dest="preds_in", default=None,
                   help="Dir to LOAD saved prediction shards from (pass 2). Skips "
                        "model/GPU/checkpoint load; re-fetches GT and re-scores. "
                        "split/seed/max-samples/num-views are adopted from the "
                        "artifact. Mutually exclusive with --preds-out / "
                        "--forward-only.")
    p.add_argument("--preds-dtype", "--preds_dtype", dest="preds_dtype",
                   default="fp32", choices=("fp32", "fp16-aux"),
                   help="Saved-pred precision (pass 1). 'fp32' (default): all "
                        "keys fp32, bit-identical metrics. 'fp16-aux': mask/normal "
                        "fp16 (depth+pose stay fp32), ~40%% smaller, near-identical.")

    # Post-processing
    p.add_argument("--enable-postprocess", action="store_true")
    p.add_argument("--postprocess-methods", default="raw,ransac,manhattan,cuboid",
                   help="Comma list; subset of " + ",".join(ALL_METHODS))
    p.add_argument("--render-holes", default="fill",
                   help="Comma list from fill,mask,zero. Default 'fill'.")
    p.add_argument("--postprocess-fallback", default="raw",
                   choices=("raw", "skip", "error"))
    p.add_argument("--ransac-max-planes", type=int, default=6)
    p.add_argument("--ransac-thresh", type=float, default=0.03)
    p.add_argument("--ransac-min-inliers", type=int, default=500)
    p.add_argument("--ransac-iters", type=int, default=1000)
    p.add_argument("--ransac-vectorized", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Batched/adaptive RANSAC (scores K hypotheses per matmul "
                        "+ Fischler-Bolles early-stop at p=0.99). DEFAULT OFF. "
                        "NOTE: measured ~2.4x SLOWER than the scalar loop on "
                        "multi-view scenes (it materialises an (M,K) distance "
                        "matrix over the full ~200k-point fused cloud), and its "
                        "different RNG draw order changes the plane fit (chamfer "
                        "shifts a few %). Kept as an opt-in for small-cloud cases; "
                        "the scalar path is the faster, canonical default here.")
    p.add_argument("--manhattan-angle-tol-deg", type=float, default=20.0)
    p.add_argument("--manhattan-merge-tol", type=float, default=0.06)
    p.add_argument("--cuboid-method", default="pca_aabb",
                   choices=("pca_aabb", "from_manhattan"))
    p.add_argument("--cuboid-inlier-thresh", type=float, default=0.05)
    p.add_argument("--cuboid-min-box-dim", type=float, default=0.5)
    p.add_argument("--plane-extent-quantile", default="0.01,0.99")
    p.add_argument("--cuboid-quantile", default="0.01,0.99")
    p.add_argument("--render-min-depth", type=float, default=0.05)
    p.add_argument("--render-max-depth", type=float, default=50.0)
    p.add_argument("--max-fuse-points", type=int, default=200_000)

    p.add_argument("--allow-missing-pred-cameras", action="store_true")

    # ----- Profiling -----
    p.add_argument(
        "--profile", action="store_true",
        help="Record per-stage wall-clock timings during the scene loop and "
             "write timing_summary.json under the output root (and per-manifest "
             "subdirs). No effect on metric values.",
    )

    args = p.parse_args(argv)

    # Post-parse normalisation
    if args.only_view_counts is not None:
        try:
            args.only_view_counts = {
                int(x.strip()) for x in args.only_view_counts.split(",") if x.strip()
            }
        except ValueError as e:
            p.error(f"--only-view-counts: {e}")
    args.postprocess_methods_list = [m.strip() for m in args.postprocess_methods.split(",") if m.strip()]
    args.render_holes_list = [h.strip() for h in args.render_holes.split(",") if h.strip()]
    bad_m = [m for m in args.postprocess_methods_list if m not in ALL_METHODS]
    if bad_m:
        p.error(f"unknown --postprocess-methods: {bad_m}; choices={list(ALL_METHODS)}")
    bad_h = [h for h in args.render_holes_list if h not in HOLE_POLICIES_ALL]
    if bad_h:
        p.error(f"unknown --render-holes: {bad_h}; choices={list(HOLE_POLICIES_ALL)}")
    try:
        args.plane_extent_quantile_t = tuple(float(x) for x in args.plane_extent_quantile.split(","))
    except ValueError as e:
        p.error(f"--plane-extent-quantile: {e}")
    try:
        args.cuboid_quantile_t = tuple(float(x) for x in args.cuboid_quantile.split(","))
    except ValueError as e:
        p.error(f"--cuboid-quantile: {e}")
    return args


# ---------------------------------------------------------------------------
# Per-scene helpers
# ---------------------------------------------------------------------------

def _resolve_pred_layout_depth_np(preds_one: dict,
                                  use_depth_as_layout: bool) -> Optional[np.ndarray]:
    """Return (S, H, W) numpy or None. Mirrors the helper in
    eval_room_envelope_reconstruction.py but inlined to avoid the private
    cross-module import (already imported separately for the 3D metric path)."""
    if "layout_depth" in preds_one:
        ld = to_np(preds_one["layout_depth"])
    elif use_depth_as_layout and "depth" in preds_one:
        ld = to_np(preds_one["depth"])
    else:
        return None
    if ld.ndim == 4 and ld.shape[-1] == 1:
        ld = ld[..., 0]
    return ld.astype(np.float32)


def _compute_2d_metrics_with_render_mask(sample: dict,
                                         depth_S: np.ndarray,
                                         render_valid_S: np.ndarray) -> dict:
    """Mirrors eval_2d_postprocess.py:245, depth-only 2D metrics with the
    rendered valid mask AND-ed into the GT valid mask. Used for the 'mask'
    hole policy. Keeps the canonical metric function in
    training/eval_metrics.py untouched."""
    gt_ld = to_np(sample["layout_depths"])
    gt_dm = to_np(sample.get("layout_depth_masks"))
    lm = to_np(sample.get("layout_masks"))
    S = depth_S.shape[0]
    records: list[dict] = []
    for s in range(S):
        gt_s = frame(gt_ld, s)
        gt_valid_s = frame(gt_dm, s).astype(bool) if gt_dm is not None else None
        render_valid_s = render_valid_S[s].astype(bool)
        combined = render_valid_s if gt_valid_s is None else (gt_valid_s & render_valid_s)
        lm_s = frame(lm, s) if lm is not None else None
        records.append(
            compute_depth_metrics_with_splits(depth_S[s], gt_s, combined, lm_s)
        )
    out: dict = {"depth_used": "layout_depth_post_mask"}
    out.update(aggregate(records))
    return out


# ---------------------------------------------------------------------------
# Per-method dispatch (raw + postprocess)
# ---------------------------------------------------------------------------

def _fit_planes_for_method(method: str, fused: np.ndarray,
                           args: argparse.Namespace,
                           cache: dict) -> tuple[list[dict], dict]:
    """Return (planes, diag) for the requested postprocess method. Uses cache
    to avoid re-fitting ransac/manhattan when later methods chain off them."""
    if method == "ransac":
        if "ransac" in cache:
            return cache["ransac"]
        planes = fit_ransac_envelope(
            fused,
            max_planes=args.ransac_max_planes,
            thresh=args.ransac_thresh,
            min_inliers=args.ransac_min_inliers,
            max_iters=args.ransac_iters,
            seed=args.seed,
            extent_quantiles=args.plane_extent_quantile_t,
            vectorized=getattr(args, "ransac_vectorized", False),
        )
        if not planes:
            diag = {"plane_status": "no_planes", "n_planes": 0}
        else:
            diag = {
                "plane_status": "ok",
                "n_planes": len(planes),
                "mean_inlier_ratio": float(np.mean([p["inlier_ratio"] for p in planes])),
                "mean_plane_residual": float(np.mean([p["mean_residual"] for p in planes])),
            }
        cache["ransac"] = (planes, diag)
        return planes, diag

    if method == "manhattan":
        if "manhattan" in cache:
            return cache["manhattan"]
        base_planes, base_diag = _fit_planes_for_method("ransac", fused, args, cache)
        diag_pre = {"n_planes_before": int(base_diag.get("n_planes", 0))}
        if not base_planes:
            diag = {**diag_pre, "manhattan_status": "no_input_planes",
                    "n_planes_after": 0, "n_planes": 0,
                    "manhattan_basis_found": False}
            cache["manhattan"] = ([], diag)
            return [], diag
        snapped, mstatus = snap_to_manhattan(
            base_planes, fused,
            angle_tol_deg=args.manhattan_angle_tol_deg,
            merge_tol=args.manhattan_merge_tol,
            extent_quantiles=args.plane_extent_quantile_t,
        )
        diag = {
            **diag_pre,
            **mstatus,
            "n_planes": len(snapped),
            "n_planes_after": len(snapped),
            "manhattan_basis_found": "basis" in mstatus and mstatus.get("basis") is not None,
        }
        cache["manhattan"] = (snapped, diag)
        return snapped, diag

    if method == "cuboid":
        if "cuboid" in cache:
            return cache["cuboid"]
        manhattan_basis = None
        if args.cuboid_method == "from_manhattan":
            _, m_diag = _fit_planes_for_method("manhattan", fused, args, cache)
            mb = m_diag.get("basis")
            if mb is not None:
                manhattan_basis = np.asarray(mb)
        faces, cstatus = fit_cuboid_room(
            fused,
            method=args.cuboid_method,
            manhattan_basis=manhattan_basis,
            quantile=args.cuboid_quantile_t,
            inlier_thresh=args.cuboid_inlier_thresh,
            min_points=args.ransac_min_inliers,
            min_box_dim=args.cuboid_min_box_dim,
        )
        diag = dict(cstatus)
        diag["cuboid_method"] = args.cuboid_method
        if cstatus.get("cuboid_status") == "ok" and "box_dims" in cstatus:
            dims = list(map(float, cstatus["box_dims"]))
            diag["cuboid_dims"] = dims
            diag["cuboid_volume"] = float(dims[0] * dims[1] * dims[2]) if len(dims) == 3 else float("nan")
            diag["cuboid_degenerate"] = False
        else:
            diag["cuboid_dims"] = [float("nan")] * 3
            diag["cuboid_volume"] = float("nan")
            diag["cuboid_degenerate"] = True
        diag["n_planes"] = len(faces)
        cache["cuboid"] = (faces, diag)
        return faces, diag

    raise ValueError(f"unknown postprocess method: {method!r}")


def _render_method(planes: list[dict], K_S: np.ndarray, E_S: np.ndarray,
                   H: int, W: int, interior_pt: np.ndarray,
                   args: argparse.Namespace) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not planes:
        return None, None
    try:
        d, m = render_planes_to_zdepth_batch(
            planes, K_S, E_S, H, W, interior_pt,
            min_depth=args.render_min_depth, max_depth=args.render_max_depth,
        )
        return d, m
    except Exception as e:
        print(f"  [render] failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

# Mirrors the eval_2d.py:649-665 nesting logic for the 2D metric block.
_2D_DEPTH_KEYS = (
    "absrel_all", "absrel_visible", "absrel_occluded",
    "rmse_all", "rmse_visible", "rmse_occluded",
    "log_rmse_all", "log_rmse_visible", "log_rmse_occluded",
    "delta1_all", "delta1_visible", "delta1_occluded",
    "delta2_all", "delta2_visible", "delta2_occluded",
    "delta3_all", "delta3_visible", "delta3_occluded",
    "silog_all", "silog_visible", "silog_occluded",
    "n_valid_all", "n_valid_visible", "n_valid_occluded",
)
_2D_MASK_SUBKEYS = ("iou", "f1", "precision", "recall", "accuracy", "n_pixels")
_2D_NORMAL_SUBKEYS = (
    "mean_deg", "median_deg",
    "pct_under_11_25", "pct_under_22_5", "pct_under_30",
    "n_valid",
)


def _nest_2d_block(agg: dict, src_prefix: str) -> dict:
    """Build {'depth':..., 'mask':..., 'normals':...} from a flat aggregate.
    src_prefix is e.g. '2d_metric_' or '2d_vggt_scene_'."""
    out = {"depth": {}, "mask": {}, "normals": {}}
    for full_key, v in agg.items():
        if not full_key.startswith(src_prefix):
            continue
        k = full_key[len(src_prefix):]
        if k.startswith("mask_"):
            out["mask"][k[len("mask_"):]] = v
        elif k.startswith("normal_"):
            out["normals"][k[len("normal_"):]] = v
        elif k in _2D_DEPTH_KEYS:
            out["depth"][k] = v
    return {section: items for section, items in out.items() if items}


_3D_TRACK_KEYS = (
    # Back-compat (alias sum); explicit sum/mean variants emitted below.
    "chamfer_l1", "chamfer_l2",
    "chamfer_l1_sum", "chamfer_l1_mean",
    "chamfer_l2_sum", "chamfer_l2_mean",
    "accuracy_mean", "completeness_mean",
    "fscore_0.05", "fscore_0.10", "fscore_0.20",
    "precision_0.05", "precision_0.10", "precision_0.20",
    "recall_0.05", "recall_0.10", "recall_0.20",
    # Physical-equivalent F-scores (populated for vggt_scene track when
    # ``vggt_scene_scale`` is available; NaN/0 in metric track).
    "fscore_physical_0.05m", "fscore_physical_0.10m", "fscore_physical_0.20m",
    "precision_physical_0.05m", "precision_physical_0.10m", "precision_physical_0.20m",
    "recall_physical_0.05m", "recall_physical_0.10m", "recall_physical_0.20m",
)

# Seen/unseen split sub-keys carried inside each 3D track dict (additive;
# computed when layout_masks is available). Mirrors _3D_TRACK_KEYS per split.
_3D_SPLIT_SUBKEYS = tuple(
    f"{split}_{k}" for split in ("seen", "unseen") for k in _3D_TRACK_KEYS
)


def _nest_3d_block(agg: dict, space_prefix: str) -> dict:
    """Build {'raw': {...}, 'scale_aligned': {...}, 'sim3_aligned': {...}, ...}
    from a flat aggregate. space_prefix is e.g. '3d_metric_' or '3d_vggt_scene_'."""
    out = {"raw": {}, "scale_aligned": {}, "sim3_aligned": {},
           "scale_shift_cam_aligned": {}}
    for full_key, v in agg.items():
        if not full_key.startswith(space_prefix):
            continue
        k = full_key[len(space_prefix):]
        # Order matters: scale_shift_cam_aligned_ shares a common prefix with
        # scale_aligned_, so we must check the longer prefix first.
        if k.startswith("raw_"):
            sub = k[len("raw_"):]
            if sub in _3D_TRACK_KEYS or sub in _3D_SPLIT_SUBKEYS:
                out["raw"][sub] = v
        elif k.startswith("scale_shift_cam_aligned_"):
            sub = k[len("scale_shift_cam_aligned_"):]
            if sub in _3D_TRACK_KEYS or sub in _3D_SPLIT_SUBKEYS:
                out["scale_shift_cam_aligned"][sub] = v
        elif k.startswith("scale_aligned_"):
            sub = k[len("scale_aligned_"):]
            if sub in _3D_TRACK_KEYS or sub in _3D_SPLIT_SUBKEYS:
                out["scale_aligned"][sub] = v
        elif k.startswith("sim3_aligned_"):
            sub = k[len("sim3_aligned_"):]
            if sub in _3D_TRACK_KEYS or sub in _3D_SPLIT_SUBKEYS:
                out["sim3_aligned"][sub] = v
    # Diagnostics inside the 3D block.
    # NOTE: ``agg`` values are ``nanmean`` aggregates, not medians. The
    # ``_median`` keys below are misnamed historical aliases kept for
    # backward compatibility; emit a correctly-named ``_mean`` companion
    # alongside each. Prefer the ``_mean`` key in new code.
    if (k := space_prefix + "scale_alignment_factor") in agg:
        out["scale_aligned"]["scale_alignment_factor_median"] = agg[k]
        out["scale_aligned"]["scale_alignment_factor_mean"]   = agg[k]
    if (k := space_prefix + "sim3_scale") in agg:
        out["sim3_aligned"]["sim3_scale_median"] = agg[k]
        out["sim3_aligned"]["sim3_scale_mean"]   = agg[k]
    # scale_shift_cam (LaRI / Room Envelopes camera-frame alignment) diagnostics.
    if (k := space_prefix + "scale_shift_cam_s") in agg:
        out["scale_shift_cam_aligned"]["scale_shift_cam_s_mean"] = agg[k]
    if (k := space_prefix + "scale_shift_cam_tz") in agg:
        out["scale_shift_cam_aligned"]["scale_shift_cam_tz_mean"] = agg[k]
    if (k := space_prefix + "scale_shift_cam_translation_norm") in agg:
        out["scale_shift_cam_aligned"]["scale_shift_cam_translation_norm_mean"] = agg[k]
    if (k := space_prefix + "scale_shift_cam_num_pairs") in agg:
        out["scale_shift_cam_aligned"]["scale_shift_cam_num_pairs_mean"] = agg[k]
    if (k := space_prefix + "scale_shift_cam_failed") in agg:
        out["scale_shift_cam_aligned"]["scale_shift_cam_failed_rate"] = agg[k]
    if (k := space_prefix + "scale_shift_cam_residual_rmse") in agg:
        out["scale_shift_cam_aligned"]["scale_shift_cam_residual_rmse_mean"] = agg[k]
    if (k := space_prefix + "scale_shift_cam_residual_rel") in agg:
        out["scale_shift_cam_aligned"]["scale_shift_cam_residual_rel_mean"] = agg[k]
    if (k := space_prefix + "scale_shift_cam_normal_cond") in agg:
        out["scale_shift_cam_aligned"]["scale_shift_cam_normal_cond_mean"] = agg[k]
    # Seen/unseen point-count diagnostics (means across scenes).
    for _nk in ("n_points_gt_overall", "n_points_gt_seen", "n_points_gt_unseen",
                "n_points_pred_overall", "n_points_pred_seen", "n_points_pred_unseen"):
        if (k := space_prefix + _nk) in agg:
            out[_nk] = agg[k]
    return out


# ---------------------------------------------------------------------------
# Per-manifest evaluator: run all methods × eval spaces × hole policies
# ---------------------------------------------------------------------------

def evaluate_manifest_all_methods(
    *,
    manifest_entry: ManifestEntry,
    manifest: dict,
    items: list[dict],
    mode: str,
    manifest_num_views: Optional[int],
    model,
    device,
    re_child,
    scene_cam_lookup: dict,
    cfg,
    args: argparse.Namespace,
    camera_modes: list[str],
    primary_cm: str,
    args_for_3d_by_cm: dict[str, SimpleNamespace],
    methods: list[str],
    eval_spaces: list[str],
    hole_policies: list[str],
    has_mask_head: bool,
    has_normal_head: bool,
    has_layout_head: bool,
    align_by_cm_space: dict[str, dict],
    use_depth_as_layout: bool,
    output_dir_for_manifest: Path,
    save_debug_every: int,
    run_3d: bool = True,
    preds_mode: Optional[str] = None,            # '2-pass': 'save' | 'load' | None
    preds_dir_for_manifest: Optional[Path] = None,
    forward_only: bool = False,
    preds_dtype: str = "fp32",
    timer=None,
) -> dict:
    """Run the per-scene loop, dispatch to all methods, write per-method
    outputs, and return a dict ``{method: payload}``."""
    import torch

    from _timing import StageTimer
    _timer = timer if timer is not None else StageTimer(enabled=False)

    # Primary camera-mode alignment drives the unprefixed 3D keys, the summary
    # headline, and the payload meta. Secondary (pred) reconstruction in
    # ``--camera-mode both`` is emitted under a ``predcam_`` key prefix.
    align_by_space = align_by_cm_space[primary_cm]

    def _cam_prefix(cm: str) -> str:
        return "predcam_" if (len(camera_modes) > 1 and cm != primary_cm) else ""

    # 2-pass artifact state. 'save' (pass 1): log a per-scene status record and
    # write a shard per non-skipped scene. 'load' (pass 2): read the index and
    # replay pass-1 skips so the scene index space stays identical.
    _preds_dir = str(preds_dir_for_manifest) if preds_dir_for_manifest is not None else None
    index_records: list[dict] = []
    loaded_index: dict[int, dict] = {}
    if preds_mode == "load":
        try:
            _idx = preds_io.read_index(_preds_dir)
        except Exception as e:
            raise RuntimeError(
                f"--preds-in: missing/unreadable index.json for manifest "
                f"'{manifest_entry.label}' under {_preds_dir}: {e}")
        loaded_index = {int(r["i"]): r for r in _idx.get("scenes", [])}
    elif preds_mode == "save":
        os.makedirs(_preds_dir, exist_ok=True)

    method_records: dict[str, list[dict]] = {m: [] for m in methods}
    method_run_counts: dict[str, dict] = {
        m: {"num_failures_2d": 0, "num_failures_3d": 0,
            "num_failures_postprocess": 0, "num_fallbacks_postprocess": 0,
            "num_scenes_skipped": 0, "n_views_total": 0}
        for m in methods
    }
    # Counters shared across methods (forward / dataset failures).
    n_failures_dataset = 0
    n_failures_forward = 0
    sanity_done = False
    n_evaluated = 0
    t_start = time.time()

    output_dir_for_manifest.mkdir(parents=True, exist_ok=True)
    for m in methods:
        (output_dir_for_manifest / m).mkdir(parents=True, exist_ok=True)

    n_total = len(items)
    if args.max_samples is not None:
        n_total = min(n_total, args.max_samples)

    print(f"  iterating {n_total} samples for manifest={manifest_entry.label}")

    for i in range(n_total):
        np.random.seed(args.seed + i)
        _random_mod.seed(args.seed + i)

        item = items[i]
        this_views = (int(manifest_num_views) if mode == "per_view"
                      else int(item["num_views"]))

        # --- Pass 2: replay pass-1 skips (which wrote no shard) before fetching,
        # so the scene index space and counters stay byte-aligned with pass 1. ---
        if preds_mode == "load":
            rec = loaded_index.get(i)
            if rec is None:
                raise RuntimeError(
                    f"--preds-in: no index record for scene {i} of manifest "
                    f"'{manifest_entry.label}'. Artifact incomplete / mismatched.")
            _st = rec.get("status")
            if _st == "skip_dataset":
                n_failures_dataset += 1
                for m in methods:
                    method_run_counts[m]["num_scenes_skipped"] += 1
                continue
            if _st == "skip_forward":
                n_failures_forward += 1
                for m in methods:
                    method_run_counts[m]["num_failures_2d"] += 1
                    method_run_counts[m]["num_failures_3d"] += 1
                    method_run_counts[m]["num_scenes_skipped"] += 1
                continue

        # --- 1) Dataset fetch ---
        try:
            with _timer.time("sample_load"):
                seq_index = _resolve_seq_index(item, scene_cam_lookup)
                sample = re_child.get_data(
                    seq_index=seq_index,
                    img_per_seq=this_views,
                    ids=list(item["ids"]),
                    aspect_ratio=1.0,
                )
        except Exception as e:
            # Pass 2: this scene was 'ok' in pass 1, so a re-fetch failure means
            # the dataset/env diverged, fail loud rather than silently skip.
            if preds_mode == "load":
                raise
            n_failures_dataset += 1
            for m in methods:
                method_run_counts[m]["num_scenes_skipped"] += 1
            if preds_mode == "save":
                index_records.append({"i": i, "status": "skip_dataset",
                                      "ids": list(item.get("ids", [])),
                                      "this_views": this_views})
            if args.strict:
                raise
            print(f"  [item {i}] dataset error: {e}; skipping")
            continue

        # --- 2) Forward (pass 1 / normal) OR load saved preds (pass 2) ---
        try:
            with _timer.time("to_device"):
                imgs_t = _to_image_tensor(sample, device)
                K_t, E_t = (cameras_from_sample(sample, device=device)
                            if preds_mode != "load" else (None, None))
            S = int(imgs_t.shape[1])
            H, W = int(imgs_t.shape[-2]), int(imgs_t.shape[-1])
            if preds_mode == "load":
                # No model: load the cached preds and cross-check the shard
                # belongs to this exact re-fetched scene (fail loud otherwise).
                preds_one = preds_io.load_scene_shard(
                    preds_io.scene_shard_path(_preds_dir, i))
                rec = loaded_index.get(i, {})
                if rec.get("ids") is not None and list(rec["ids"]) != list(item.get("ids", [])):
                    raise RuntimeError(
                        f"--preds-in scene {i}: ids mismatch "
                        f"(shard {rec.get('ids')} != manifest {list(item.get('ids', []))})")
                if rec.get("S") is not None and (
                        int(rec["S"]), int(rec["H"]), int(rec["W"])) != (S, H, W):
                    raise RuntimeError(
                        f"--preds-in scene {i}: image-shape mismatch "
                        f"(shard {(rec.get('S'), rec.get('H'), rec.get('W'))} "
                        f"!= re-fetched {(S, H, W)})")
            else:
                with _timer.time("forward"):
                    with torch.no_grad():
                        preds = forward_model(model, imgs_t, intrinsics=K_t, extrinsics=E_t)
                preds_one = _strip_batch_dim(preds)
        except Exception as e:
            # Pass 2: a load/cross-check failure is an artifact-integrity error,
            # fail loud, never silently skip.
            if preds_mode == "load":
                raise
            n_failures_forward += 1
            for m in methods:
                method_run_counts[m]["num_failures_2d"] += 1
                method_run_counts[m]["num_failures_3d"] += 1
                method_run_counts[m]["num_scenes_skipped"] += 1
            if preds_mode == "save":
                index_records.append({"i": i, "status": "skip_forward",
                                      "ids": list(item.get("ids", [])),
                                      "this_views": this_views})
            if args.strict:
                raise
            print(f"  [item {i}] forward error: {e}; skipping")
            continue

        if not sanity_done:
            _assert_pred_shapes(
                preds_one, S, H, W,
                has_mask_head=has_mask_head,
                has_normal_head=has_normal_head,
                use_depth_as_layout=use_depth_as_layout,
            )
            sanity_done = True

        # --- Pass 1: save this scene's predictions + index record ---
        if preds_mode == "save":
            _keys = preds_io.select_pred_keys(preds_one, use_depth_as_layout)
            preds_io.save_scene_shard(
                preds_io.scene_shard_path(_preds_dir, i),
                preds_io.to_shard_arrays(preds_one, _keys, preds_dtype),
            )
            index_records.append({"i": i, "status": "ok",
                                  "ids": list(item.get("ids", [])),
                                  "this_views": this_views,
                                  "S": S, "H": H, "W": W, "keys": _keys})
            if forward_only:
                n_evaluated += 1
                if n_evaluated % 25 == 0 or (i + 1) == n_total:
                    print(f"  [{n_evaluated}/{n_total}] saved, "
                          f"{time.time() - t_start:.1f}s elapsed")
                continue

        # --- 3) Precompute things shared across methods ---
        # raw 2D scale-aligned preds (used for metric-space 2D headline and
        # for the postprocess fused-cloud basis).
        if args.scale_align_2d == "per_frame":
            try:
                scaled_preds_one, frame_scales = _scale_align_preds_per_frame(
                    preds_one, sample, use_depth_as_layout,
                )
            except Exception as e:
                print(f"  [item {i}] scale-align-2d failed: {e}; falling back to raw")
                scaled_preds_one, frame_scales = preds_one, []
        else:
            scaled_preds_one, frame_scales = preds_one, []

        # vggt_scene normalised sample (only computed once, even if used by
        # both 2D and 3D vggt_scene passes).
        normalized_sample = None
        norm_info = None
        if "vggt_scene" in eval_spaces:
            try:
                normalized_sample, norm_info = normalize_sample_vggt_scene(sample)
            except Exception as e:
                print(f"  [item {i}] vggt_scene normalisation failed: {e}")
                normalized_sample = None

        # Postprocess shared work (fused cloud, interior anchor), only when
        # we need it.
        fused_pred: Optional[np.ndarray] = None
        interior_pt: Optional[np.ndarray] = None
        K_S: Optional[np.ndarray] = None
        E_S: Optional[np.ndarray] = None
        plane_cache: dict = {}
        rendered_cache: dict[str, tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        # Per-scene GT-side cache, shared across all methods/policies/spaces/cms
        # (keyed internally by id(sample_active), so metric vs vggt_scene stay
        # separate). Reset each scene so ids never collide. None => disabled.
        gt_cache: Optional[dict] = {} if getattr(args, "gt_cache", True) else None
        need_post = any(m in POSTPROCESS_METHODS for m in methods)
        if need_post:
            try:
                # pred depth (scale-aligned if requested) as (S, H, W) numpy
                pred_ld_S = _resolve_pred_layout_depth_np(scaled_preds_one, use_depth_as_layout)
                if pred_ld_S is None:
                    raise RuntimeError("no usable layout_depth in predictions")
                K_S = np.stack([np.asarray(k, dtype=np.float32) for k in sample["intrinsics"]], 0)
                E_S = np.stack([np.asarray(e, dtype=np.float32) for e in sample["extrinsics"]], 0)
                with _timer.time("postproc_fused_cloud"):
                    fused_pred = _fuse_world_cloud(
                        pred_ld_S, K_S, E_S,
                        max_points=args.max_fuse_points, seed=args.seed,
                    )
                interior_pt = _camera_centres(E_S).mean(axis=0)
            except Exception as e:
                print(f"  [item {i}] postprocess shared setup failed: {e}")
                fused_pred = None

        # --- 4) Per-method dispatch ---
        for method in methods:
            base_record: dict = {
                "scene_idx": int(i),
                "seq_name": str(sample.get("seq_name", f"scene_{i:04d}")),
                "n_views": int(S),
                "manifest_label": manifest_entry.label,
                "method": method,
            }
            # Postprocess fit (skipped for raw)
            method_preds_one_metric: Optional[dict] = None
            method_preds_one_raw: Optional[dict] = None
            post_diag: dict = {}
            post_success = True
            fallback_used = False
            failure_reason: Optional[str] = None
            rendered_for_method: Optional[np.ndarray] = None
            render_valid_for_method: Optional[np.ndarray] = None

            if method == "raw":
                # 2D metric path uses scale-aligned, 3D uses raw preds_one.
                method_preds_one_metric = scaled_preds_one
                method_preds_one_raw = preds_one
                if frame_scales:
                    fs = [s for s in frame_scales if np.isfinite(s)]
                    base_record["metric_scale_factor_median"] = (
                        float(np.median(fs)) if fs else float("nan")
                    )
            else:
                # POSTPROCESS METHOD: fit, render, build preds dict per
                # hole policy.
                if fused_pred is None:
                    post_success = False
                    failure_reason = "fused_cloud_unavailable"
                else:
                    try:
                        with _timer.time(f"postproc_fit:{method}"):
                            planes, post_diag = _fit_planes_for_method(method, fused_pred, args, plane_cache)
                        if not planes:
                            post_success = False
                            failure_reason = post_diag.get("plane_status") or post_diag.get("cuboid_status") or "no_planes"
                        else:
                            if method not in rendered_cache:
                                with _timer.time(f"postproc_render:{method}"):
                                    d, m_valid = _render_method(planes, K_S, E_S, H, W, interior_pt, args)
                                rendered_cache[method] = (d, m_valid)
                            d, m_valid = rendered_cache[method]
                            if d is None or m_valid is None:
                                post_success = False
                                failure_reason = "render_failed"
                            else:
                                rendered_for_method = d
                                render_valid_for_method = m_valid
                                post_diag["render_coverage"] = float(m_valid.mean())
                    except Exception as e:
                        post_success = False
                        failure_reason = f"fit_error: {e}"

                # Apply fallback rules.
                if not post_success:
                    if args.postprocess_fallback == "error":
                        raise RuntimeError(
                            f"postprocess method={method!r} failed on scene {i}: "
                            f"{failure_reason}"
                        )
                    method_run_counts[method]["num_failures_postprocess"] += 1
                    if args.postprocess_fallback == "skip":
                        method_run_counts[method]["num_scenes_skipped"] += 1
                        # Still write a per-scene record so debug CSVs show it,
                        # but don't include in aggregate (use a sentinel that
                        # `aggregate` will silently drop non-numeric values from).
                        base_record["post_success"] = False
                        base_record["fallback_used"] = False
                        base_record["failure_reason"] = failure_reason
                        base_record["render_coverage"] = float(post_diag.get("render_coverage", 0.0))
                        method_records[method].append(base_record)
                        continue
                    # fallback == "raw": substitute raw preds
                    method_preds_one_metric = scaled_preds_one
                    method_preds_one_raw = preds_one
                    fallback_used = True
                    method_run_counts[method]["num_fallbacks_postprocess"] += 1
                else:
                    # Build a preds dict for the chosen hole policy. We
                    # compute one representative policy per record per
                    # the loop below; for per-scene records (which exist
                    # at method level), we use the *headline* hole policy
                    # (first item in --render-holes). The summary CSV
                    # produces additional rows for the other policies.
                    pass

            # Within this method, iterate eval_spaces × hole_policies. We
            # store per-(method, eval_space, hole_policy) metrics in the
            # same per-scene record with prefixed keys so that aggregation
            # can extract any combination later.
            policies_to_run = hole_policies if method in POSTPROCESS_METHODS else ["fill"]

            for policy in policies_to_run:
                # Determine the predictions to feed downstream for this
                # (method, hole_policy) combination.
                if method == "raw" or fallback_used:
                    preds_for_metric = method_preds_one_metric  # scale-aligned
                    preds_for_3d = method_preds_one_raw         # raw
                elif rendered_for_method is None:
                    # Should not happen, post_success would be False.
                    continue
                else:
                    if policy == "fill":
                        # Use scale-aligned pred to fill holes (same basis as
                        # the fused cloud used to fit the geometry).
                        pred_aligned_S = _resolve_pred_layout_depth_np(
                            scaled_preds_one, use_depth_as_layout,
                        )
                        d_filled = np.where(
                            render_valid_for_method, rendered_for_method, pred_aligned_S,
                        )
                        preds_for_metric = {"layout_depth": d_filled[..., None].astype(np.float32)}
                    elif policy == "mask":
                        # Caller-side: 2D path uses the local mask helper;
                        # 3D path receives the rendered depth (holes=0)
                        # which downstream interprets via depth-validity.
                        preds_for_metric = {"layout_depth": rendered_for_method[..., None].astype(np.float32)}
                    elif policy == "zero":
                        preds_for_metric = {"layout_depth": rendered_for_method[..., None].astype(np.float32)}
                    else:
                        raise ValueError(f"unknown hole policy: {policy!r}")
                    preds_for_3d = preds_for_metric  # same dict for 3D

                # Record key prefix encoding (method, policy) so multiple
                # (policy) results coexist in the same per-scene record.
                policy_tag = f"_{policy}" if method in POSTPROCESS_METHODS and len(hole_policies) > 1 else ""

                # ---- eval-space passes ----
                for es in eval_spaces:
                    sample_active = sample if es == "metric" else normalized_sample
                    if sample_active is None:
                        # vggt_scene normalisation failed; skip this pass.
                        continue
                    es_prefix = "metric_" if es == "metric" else "vggt_scene_"

                    # ---- 2D ----
                    try:
                        _t2d0 = time.perf_counter()
                        if method in POSTPROCESS_METHODS and policy == "mask" and not fallback_used:
                            m2d = _compute_2d_metrics_with_render_mask(
                                sample_active,
                                rendered_for_method,
                                render_valid_for_method,
                            )
                        elif es == "metric":
                            # metric-space uses scale-aligned (or rendered) preds
                            m2d = compute_2d_metrics_for_scene(
                                sample_active, preds_for_metric,
                                use_depth_as_layout=use_depth_as_layout,
                                has_mask_head=has_mask_head,
                                has_normal_head=has_normal_head,
                            )
                        else:
                            # vggt_scene: 2D uses unscaled preds (matches existing 2D eval)
                            # For raw method, preds_for_metric is scaled, switch back to
                            # raw preds_one for vggt_scene.
                            preds_for_2d_vggt = (
                                preds_one if (method == "raw" or fallback_used) else preds_for_metric
                            )
                            m2d = compute_2d_metrics_for_scene(
                                sample_active, preds_for_2d_vggt,
                                use_depth_as_layout=use_depth_as_layout,
                                has_mask_head=has_mask_head,
                                has_normal_head=has_normal_head,
                            )
                        for k, v in m2d.items():
                            if k == "depth_used":
                                continue
                            base_record[f"2d_{es_prefix}{k}{policy_tag}"] = v
                        _timer.record(f"metrics_2d:{method}", time.perf_counter() - _t2d0)
                    except Exception as e:
                        method_run_counts[method]["num_failures_2d"] += 1
                        base_record[f"2d_{es_prefix}error{policy_tag}"] = str(e)
                        if args.strict:
                            raise

                    # ---- 3D ----
                    # Head-aware: skip 3D entirely when there is no depth source
                    # (run_3d=False); 2D / mask / normal / pose still run.
                    # Runs once per reconstruction camera mode (gt and/or pred,
                    # per --camera-mode). The secondary (pred) mode's keys get a
                    # ``predcam_`` prefix so both sets coexist in the record.
                    if run_3d:
                        # For the vggt_scene pass, forward the per-sample
                        # ``vggt_scene_scale`` so the metric helper can emit
                        # physical-equivalent F-score keys
                        # (``fscore_physical_{0.05,0.10,0.20}m``). For the metric
                        # pass leave it None.
                        scene_scale_for_3d = (
                            norm_info.get("vggt_scene_scale")
                            if (es == "vggt_scene" and norm_info is not None)
                            else None
                        )
                        for cm in camera_modes:
                            cam_prefix = _cam_prefix(cm)
                            try:
                                _t3d0 = time.perf_counter()
                                _space_cfg = align_by_cm_space[cm][es]
                                m3d, plys = _compute_3d_metrics_for_scene(
                                    sample=sample_active,
                                    preds_one=preds_for_3d,
                                    args=args_for_3d_by_cm[cm],
                                    image_hw=(H, W),
                                    do_raw=_space_cfg["do_raw"],
                                    do_scale=_space_cfg["do_scale"],
                                    do_sim3=_space_cfg["do_sim3"],
                                    do_scale_shift_cam=_space_cfg["do_scale_shift_cam"],
                                    timer=_timer,
                                    vggt_scene_scale=scene_scale_for_3d,
                                    gt_cache=gt_cache,
                                )
                                _timer.record(f"metrics_3d:{method}", time.perf_counter() - _t3d0)
                                for k, v in m3d.items():
                                    base_record[f"3d_{cam_prefix}{es_prefix}{k}{policy_tag}"] = v
                                # Optional PLY save (only for the headline policy).
                                if (save_debug_every > 0 and (n_evaluated % save_debug_every == 0)
                                        and policy == hole_policies[0]):
                                    ply_dir = output_dir_for_manifest / method / "pointclouds"
                                    ply_dir.mkdir(parents=True, exist_ok=True)
                                    ply_prefix = "vggt_scene_" if es == "vggt_scene" else ""
                                    for nm, pts in plys.items():
                                        try:
                                            write_ply(
                                                str(ply_dir / f"scene_{i:04d}_{cam_prefix}{ply_prefix}{nm}.ply"),
                                                pts,
                                            )
                                        except Exception as e_ply:
                                            print(f"  [ply] {nm} write failed: {e_ply}")
                            except Exception as e:
                                method_run_counts[method]["num_failures_3d"] += 1
                                base_record[f"3d_{cam_prefix}{es_prefix}error{policy_tag}"] = str(e)
                                if args.strict:
                                    raise

            # Postprocess diagnostics
            if method != "raw":
                base_record["post_success"] = bool(post_success)
                base_record["fallback_used"] = bool(fallback_used)
                base_record["failure_reason"] = failure_reason
                base_record["render_coverage"] = float(post_diag.get("render_coverage", 0.0))
                for k, v in post_diag.items():
                    if k == "basis":  # not numeric-friendly
                        continue
                    base_record[f"post_{k}"] = v
            method_records[method].append(base_record)
            method_run_counts[method]["n_views_total"] += S

        n_evaluated += 1
        if n_evaluated % 10 == 0 or (i + 1) == n_total:
            print(f"  [{n_evaluated}/{n_total}] scenes done, "
                  f"{time.time() - t_start:.1f}s elapsed")

    # --- Pass 1: persist the per-scene index (status + identity + skips) ---
    if preds_mode == "save":
        preds_io.write_index(_preds_dir, {
            "manifest_label": manifest_entry.label,
            "manifest_path": str(manifest_entry.path),
            "split": args.split,
            "num_views": manifest_num_views,
            "n_total": n_total,
            "seed": int(args.seed),
            "preds_dtype": preds_dtype,
            "scenes": index_records,
        })
    # Forward-only: shards are written; no scoring → no per-method payloads.
    if forward_only:
        return {}

    # --- Write per-method outputs ---
    payloads: dict[str, dict] = {}
    for method in methods:
        recs = method_records[method]
        method_dir = output_dir_for_manifest / method
        if not recs:
            print(f"  [{manifest_entry.label}/{method}] no records (all scenes skipped)")
            payload = _build_payload(
                recs, manifest_entry, manifest, mode, manifest_num_views,
                method, args, eval_spaces, hole_policies, align_by_space,
                has_mask_head, has_normal_head, has_layout_head,
                use_depth_as_layout,
                n_failures_dataset, n_failures_forward, method_run_counts[method],
            )
        else:
            agg = aggregate(recs)
            payload = _build_payload(
                recs, manifest_entry, manifest, mode, manifest_num_views,
                method, args, eval_spaces, hole_policies, align_by_space,
                has_mask_head, has_normal_head, has_layout_head,
                use_depth_as_layout,
                n_failures_dataset, n_failures_forward, method_run_counts[method],
                agg=agg,
            )

        # Write outputs
        with open(method_dir / "combined_metrics.json", "w") as f:
            json.dump(payload, f, indent=2)
        with open(method_dir / "metrics_2d.json", "w") as f:
            json.dump(
                {"meta": payload["meta"], "metrics_2d": payload["metrics"].get("2d", {})},
                f, indent=2,
            )
        with open(method_dir / "metrics_3d.json", "w") as f:
            json.dump(
                {"meta": payload["meta"], "metrics_3d": payload["metrics"].get("3d", {})},
                f, indent=2,
            )

        payloads[method] = payload

    return payloads


def _build_payload(per_scene_records: list[dict],
                   entry: ManifestEntry,
                   manifest: dict,
                   mode: str,
                   manifest_num_views: Optional[int],
                   method: str,
                   args: argparse.Namespace,
                   eval_spaces: list[str],
                   hole_policies: list[str],
                   align_by_space: dict[str, dict],
                   has_mask_head: bool,
                   has_normal_head: bool,
                   has_layout_head: bool,
                   use_depth_as_layout: bool,
                   n_failures_dataset: int,
                   n_failures_forward: int,
                   method_counts: dict,
                   agg: Optional[dict] = None) -> dict:
    if agg is None:
        agg = {}

    metrics: dict = {"2d": {}, "3d": {}, "diagnostics": {}}
    _3d_track_keys = ("raw", "scale_aligned", "sim3_aligned",
                      "scale_shift_cam_aligned")
    for es in eval_spaces:
        es_prefix = "metric_" if es == "metric" else "vggt_scene_"
        m2d = _nest_2d_block(agg, f"2d_{es_prefix}")
        if m2d:
            metrics["2d"][es] = m2d
        m3d = _nest_3d_block(agg, f"3d_{es_prefix}")
        if any(m3d[k] for k in _3d_track_keys):
            metrics["3d"][es] = m3d
        # Predicted-camera reconstruction (--camera-mode both): the secondary
        # pass is stored under a ``predcam_`` key prefix and surfaced in a
        # parallel ``3d_predcam`` block alongside the gt ``3d`` block.
        m3d_pred = _nest_3d_block(agg, f"3d_predcam_{es_prefix}")
        if any(m3d_pred[k] for k in _3d_track_keys):
            metrics.setdefault("3d_predcam", {})[es] = m3d_pred

    # Diagnostics
    for k in ("metric_scale_factor_median",):
        if k in agg:
            metrics["diagnostics"][k] = agg[k]

    # Postprocess block
    if method in POSTPROCESS_METHODS:
        post_block: dict = {}
        for k in ("post_n_planes", "post_n_planes_after",
                  "post_mean_inlier_ratio", "post_mean_plane_residual",
                  "render_coverage", "post_cuboid_volume"):
            if k in agg:
                post_block[k.replace("post_", "") + "_mean" if k.startswith("post_")
                           else f"{k}_mean"] = agg[k]
        # Rates
        n_recs = len(per_scene_records)
        if n_recs > 0:
            n_success = sum(1 for r in per_scene_records if r.get("post_success") is True)
            n_fallback = sum(1 for r in per_scene_records if r.get("fallback_used") is True)
            n_degen = sum(1 for r in per_scene_records if r.get("post_cuboid_degenerate") is True)
            post_block["post_success_rate"] = float(n_success / n_recs)
            post_block["fallback_rate"] = float(n_fallback / n_recs)
            if method == "cuboid":
                post_block["cuboid_degenerate_rate"] = float(n_degen / n_recs)
        metrics["postprocess"] = post_block

    meta = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "git_commit": _git_commit_sha(),
        "split": args.split,
        "manifest_path": str(entry.path),
        "manifest_label": entry.label,
        "manifest_num_views": manifest_num_views,
        "manifest_meta": manifest.get("meta", {}),
        "method": method,
        "num_scenes_evaluated": len(per_scene_records),
        "num_scenes_skipped": int(method_counts.get("num_scenes_skipped", 0)),
        "num_failures_dataset": int(n_failures_dataset),
        "num_failures_forward": int(n_failures_forward),
        "num_failures_2d": int(method_counts.get("num_failures_2d", 0)),
        "num_failures_3d": int(method_counts.get("num_failures_3d", 0)),
        "num_failures_postprocess": int(method_counts.get("num_failures_postprocess", 0)),
        "num_fallbacks_postprocess": int(method_counts.get("num_fallbacks_postprocess", 0)),
        "num_frames": int(method_counts.get("n_views_total", 0)),
        "num_views_per_scene": (
            int(manifest_num_views) if manifest_num_views is not None
            else "per-item"
        ),
        "seed": int(args.seed),
        "eval_space": args.eval_space,
        "camera_mode": args.camera_mode,
        "alignment": args.alignment,
        "scale_alignment": args.scale_alignment,
        "scale_align_2d": args.scale_align_2d,
        # Per-eval-space alignment / headline (Part A). Each value is the
        # *resolved* alignment policy after applying --metric-alignment /
        # --vggt-scene-alignment overrides; ``headline`` is the headline
        # track used for that space in the summary CSV.
        "alignment_by_space": {
            es: {"alignment": cfg["alignment"], "headline": cfg["headline"]}
            for es, cfg in align_by_space.items()
        },
        # Back-compat: keep the legacy global headline_alignment for any
        # downstream tool that reads it. Equals the metric-space headline
        # when metric is enabled, else the first space's headline.
        "headline_alignment": (
            align_by_space.get("metric", {}).get("headline")
            or next(iter(align_by_space.values()), {}).get("headline")
            or "raw"
        ),
        "render_holes": list(hole_policies),
        "postprocess_fallback": args.postprocess_fallback,
        "heads_enabled": {
            "layout_depth": has_layout_head or use_depth_as_layout,
            "layout_mask": has_mask_head,
            "layout_normal": has_normal_head,
        },
        "use_depth_as_layout": bool(use_depth_as_layout),
        "device": str(args.device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    payload = {
        "meta": _json_safe(meta),
        "metrics": _json_safe(metrics),
        "per_scene": _json_safe(per_scene_records),
    }
    return payload


# ---------------------------------------------------------------------------
# Cross-manifest summary
# ---------------------------------------------------------------------------

_3D_SUMMARY_COLS = [
    "3d_raw_chamfer_l1", "3d_raw_chamfer_l2",
    "3d_raw_fscore_0.05", "3d_raw_fscore_0.10", "3d_raw_fscore_0.20",
    "3d_raw_precision_0.10", "3d_raw_recall_0.10",
    "3d_scale_aligned_chamfer_l1", "3d_scale_aligned_fscore_0.10",
    "3d_sim3_aligned_chamfer_l1", "3d_sim3_aligned_fscore_0.10",
    "3d_scale_shift_cam_aligned_chamfer_l1",
    "3d_scale_shift_cam_aligned_fscore_0.05", "3d_scale_shift_cam_aligned_fscore_0.10",
    # Seen/unseen splits (chamfer_l1 + fscore_0.05/0.10) for the
    # non-sim3 tracks. Added additively; existing columns above are unchanged.
    "3d_raw_seen_chamfer_l1", "3d_raw_seen_fscore_0.05", "3d_raw_seen_fscore_0.10",
    "3d_raw_unseen_chamfer_l1", "3d_raw_unseen_fscore_0.05", "3d_raw_unseen_fscore_0.10",
    "3d_scale_aligned_seen_chamfer_l1", "3d_scale_aligned_seen_fscore_0.05",
    "3d_scale_aligned_seen_fscore_0.10",
    "3d_scale_aligned_unseen_chamfer_l1", "3d_scale_aligned_unseen_fscore_0.05",
    "3d_scale_aligned_unseen_fscore_0.10",
    "3d_scale_shift_cam_aligned_seen_chamfer_l1", "3d_scale_shift_cam_aligned_seen_fscore_0.05",
    "3d_scale_shift_cam_aligned_seen_fscore_0.10",
    "3d_scale_shift_cam_aligned_unseen_chamfer_l1", "3d_scale_shift_cam_aligned_unseen_fscore_0.05",
    "3d_scale_shift_cam_aligned_unseen_fscore_0.10",
    "headline_chamfer_l1", "headline_fscore_0.10",
]


def _extract_summary_row(payload: dict, entry: ManifestEntry, method: str,
                          eval_space: str, hole_policy: str,
                          headline_alignment: str) -> dict:
    """Pull a flat row dict suitable for the summary CSVs from a per-(method,
    manifest) payload."""
    meta = payload["meta"]
    metrics = payload["metrics"]
    # Per-row alignment + headline track + unit. Reading from
    # ``meta.alignment_by_space`` so each row reflects the alignment policy
    # *actually* used for its eval space. ``headline_unit`` disambiguates
    # the ``headline_*`` columns (m for metric, norm for vggt_scene).
    _abs = (meta.get("alignment_by_space") or {}).get(eval_space, {}) or {}
    row_alignment = _abs.get("alignment", meta.get("alignment", ""))
    row_headline  = _abs.get("headline",  headline_alignment)
    row_unit      = "m" if eval_space == "metric" else "norm"
    row: dict = {
        "view_label": entry.label,
        "manifest_name": Path(meta.get("manifest_path", str(entry.path))).name,
        "method": method,
        "eval_space": eval_space,
        "alignment": row_alignment,
        "headline_alignment": row_headline,
        "headline_unit": row_unit,
        "render_holes": hole_policy,
        "num_samples": int(entry.n_samples),
        "num_evaluated": int(meta.get("num_scenes_evaluated", 0)),
        "num_failures_2d": int(meta.get("num_failures_2d", 0)),
        "num_failures_3d": int(meta.get("num_failures_3d", 0)),
        "failure_rate": (
            float(meta.get("num_failures_2d", 0) + meta.get("num_failures_3d", 0))
            / max(1, meta.get("num_scenes_evaluated", 0))
        ),
        "fallback_rate": float(metrics.get("postprocess", {}).get("fallback_rate", 0.0)),
        "render_coverage_mean": float(metrics.get("postprocess", {}).get("render_coverage_mean", float("nan"))),
    }

    block_2d = metrics.get("2d", {}).get(eval_space, {})
    depth = block_2d.get("depth", {}) or {}
    mask = block_2d.get("mask", {}) or {}
    normals = block_2d.get("normals", {}) or {}
    for k in ("absrel_occluded", "absrel_visible", "absrel_all",
              "rmse_occluded", "rmse_visible", "rmse_all",
              "log_rmse_occluded", "log_rmse_visible", "log_rmse_all",
              "delta1_occluded", "delta1_visible", "delta1_all",
              "delta2_all", "delta3_all", "silog_all"):
        row[f"2d_{k}"] = depth.get(k)
    for k in ("iou", "f1", "precision", "recall"):
        row[f"2d_mask_{k}"] = mask.get(k)
    for k in ("mean_deg", "median_deg", "pct_under_22_5"):
        row[f"2d_normal_{k}"] = normals.get(k)

    block_3d = metrics.get("3d", {}).get(eval_space, {}) or {}
    for track in ("raw", "scale_aligned", "sim3_aligned",
                  "scale_shift_cam_aligned"):
        trk = block_3d.get(track, {}) or {}
        for k in _3D_TRACK_KEYS + _3D_SPLIT_SUBKEYS:
            col = f"3d_{track}_{k}"
            if col in _3D_SUMMARY_COLS:
                row[col] = trk.get(k)

    # Prefer the per-space headline encoded into ``meta.alignment_by_space``
    # so each (eval_space) row gets its own headline track. Falls back to
    # the global ``headline_alignment`` argument for callers that still pass
    # it (kept for backward compatibility).
    _row_head = row_headline or headline_alignment
    headline_track = (
        "scale_aligned" if _row_head == "scale"
        else ("sim3_aligned" if _row_head == "sim3"
              else ("scale_shift_cam_aligned" if _row_head == "scale_shift_cam"
                    else "raw"))
    )
    trk_h = block_3d.get(headline_track, {}) or {}
    row["headline_chamfer_l1"] = trk_h.get("chamfer_l1")
    row["headline_fscore_0.10"] = trk_h.get("fscore_0.10")

    post = metrics.get("postprocess", {}) or {}
    row["post_n_planes_mean"] = post.get("n_planes_mean")
    row["post_mean_inlier_ratio"] = post.get("mean_inlier_ratio_mean")
    row["post_render_coverage_mean"] = post.get("render_coverage_mean")
    row["post_success_rate"] = post.get("post_success_rate")
    row["post_fallback_rate"] = post.get("fallback_rate")
    row["post_cuboid_volume_mean"] = post.get("cuboid_volume_mean")
    row["post_cuboid_degenerate_rate"] = post.get("cuboid_degenerate_rate")
    return row


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _resolve_methods(args: argparse.Namespace) -> list[str]:
    if not args.enable_postprocess:
        return ["raw"]
    requested = args.postprocess_methods_list
    if "raw" not in requested:
        requested = ["raw"] + requested
    return [m for m in ALL_METHODS if m in requested]


def _resolve_eval_spaces(args: argparse.Namespace) -> list[str]:
    if args.eval_space == "metric":
        return ["metric"]
    if args.eval_space == "vggt_scene":
        return ["vggt_scene"]
    return ["metric", "vggt_scene"]


def _preflight_pred_camera_check(model, device, re_child, scene_cam_lookup,
                                  entry: ManifestEntry,
                                  allow_missing: bool) -> None:
    """Fetch one sample, run one forward, verify `pose_enc` is present.
    Fails fast unless ``--allow-missing-pred-cameras``."""
    import torch
    manifest = _load_eval_manifest(str(entry.path))
    items, mode, mnv = _manifest_iter_items(manifest)
    if not items:
        raise RuntimeError("pre-flight: first manifest has no samples")
    item = items[0]
    this_views = int(mnv) if mode == "per_view" else int(item.get("num_views", 1))
    seq_index = _resolve_seq_index(item, scene_cam_lookup)
    sample = re_child.get_data(
        seq_index=seq_index, img_per_seq=this_views,
        ids=list(item["ids"]), aspect_ratio=1.0,
    )
    imgs_t = _to_image_tensor(sample, device)
    K_t, E_t = cameras_from_sample(sample, device=device)
    with torch.no_grad():
        preds = forward_model(model, imgs_t, intrinsics=K_t, extrinsics=E_t)
    if "pose_enc" not in preds:
        msg = ("[preflight] --camera-mode=pred but the model output does not "
               "include 'pose_enc'. This config has no pose head; "
               "every 3D pass would raise KeyError.")
        if allow_missing:
            print(msg + " Continuing because --allow-missing-pred-cameras is set.")
        else:
            raise RuntimeError(msg + " Pass --allow-missing-pred-cameras to proceed anyway.")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # --- 2-pass mode resolution + flag validation ---
    is_pass2 = args.preds_in is not None     # load saved preds, re-score (no GPU)
    is_save = args.preds_out is not None      # save preds during forward (pass 1)
    if is_pass2 and (is_save or args.forward_only):
        print("[fatal] --preds-in is mutually exclusive with --preds-out / --forward-only",
              file=sys.stderr)
        return 2
    if args.forward_only and not is_save:
        print("[fatal] --forward-only requires --preds-out", file=sys.stderr)
        return 2

    # Pass 2 adopts the GT-determining params from the artifact BEFORE discovery
    # (discovery filters by split), so the re-fetched GT matches pass 1 exactly.
    run_identity = None
    preds_in_dir = None
    if is_pass2:
        preds_in_dir = str(_resolve_user_path(args.preds_in))
        try:
            run_identity = preds_io.read_run_identity(preds_in_dir)
        except Exception as e:
            print(f"[fatal] could not read run_identity.json under --preds-in: {e}",
                  file=sys.stderr)
            return 2
        _sid = run_identity.get("scoring_identity", {})
        args.split = _sid.get("split", args.split)
        args.seed = int(_sid.get("seed", args.seed))
        args.max_samples = _sid.get("max_samples", args.max_samples)
        print(f"[preds-in] adopted from artifact: split={args.split} seed={args.seed} "
              f"max_samples={args.max_samples}")

    # --- Discovery ---
    if args.single_manifest:
        single_path = _resolve_user_path(args.single_manifest)
        if not single_path.is_file():
            print(f"[fatal] --single-manifest not found: {single_path}", file=sys.stderr)
            return 2
        entries = [single_manifest_entry(single_path, args.split)]
    else:
        if args.manifest_dir is None:
            print("[fatal] --manifest-dir is required (or pass --single-manifest)",
                  file=sys.stderr)
            return 2
        entries = discover_manifests(
            _resolve_user_path(args.manifest_dir),
            split=args.split,
            include_mixed=args.include_mixed,
            only_view_counts=args.only_view_counts,
            manifest_glob_override=args.manifest_glob,
        )

    print_discovery_table(entries)

    if args.dry_run_discovery:
        return 0
    if not entries:
        print("[fatal] no manifests matched", file=sys.stderr)
        return 2

    # Pass 2: every discovered manifest we are about to run must have saved
    # predictions in the artifact with a matching content hash (discovered ⊆
    # saved). Saved manifests the user did NOT discover (e.g. filtered out by
    # --only-view-counts) are simply skipped, not an error.
    if is_pass2:
        discovered_hashes: dict[str, str] = {}
        for e in entries:
            try:
                discovered_hashes[e.label] = preds_io.manifest_content_hash(
                    _load_eval_manifest(str(e.path)))
            except Exception as ex:
                print(f"[fatal] could not hash discovered manifest {e.label}: {ex}",
                      file=sys.stderr)
                return 2
        mfails = preds_io.validate_manifest_set(run_identity, discovered_hashes)
        if mfails:
            for m in mfails:
                print(f"[fatal][preds-in] {m}", file=sys.stderr)
            return 2
        saved_labels = {m["label"] for m in run_identity["scoring_identity"]["manifests"]}
        skipped = sorted(saved_labels - set(discovered_hashes))
        if skipped:
            print(f"[preds-in] scoring {sorted(discovered_hashes)}; "
                  f"saved-but-not-requested (skipped): {skipped}")
        # entries are already the discovered set; every one is validated above.
        entries = [e for e in entries if e.label in saved_labels]

    # --- Sanity ---
    if not args.config:
        print("[fatal] --config is required", file=sys.stderr)
        return 2
    if (not is_pass2) and (not args.checkpoint):
        print("[fatal] --checkpoint is required", file=sys.stderr)
        return 2
    if not args.output_dir:
        print("[fatal] --output-dir is required", file=sys.stderr)
        return 2
    # Resolve checkpoint against the user's original cwd (chdir already moved
    # us into training/). Optional in pass 2 (no model load).
    if args.checkpoint:
        args.checkpoint = str(_resolve_user_path(args.checkpoint))

    import torch
    if args.device is None:
        # Pass 2 is CPU-only (no model); don't grab a GPU.
        args.device = "cuda" if (not is_pass2 and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)

    # --- Load model (pass 1 / normal) or cfg-only (pass 2) ---
    if is_pass2:
        cfg = load_cfg(args.config)
        model = None
        device = torch.device(args.device)
        _pid = run_identity.get("prediction_identity", {})
        ckpt_meta = {
            "checkpoint_path": _pid.get("checkpoint_path"),
            "checkpoint_epoch": _pid.get("checkpoint_epoch"),
            "missing_keys": [],
        }
        # Fail loudly on any prediction/scoring-config mismatch.
        _ckpt_sha = (preds_io.sha256_file(args.checkpoint)
                     if args.checkpoint and os.path.isfile(args.checkpoint) else None)
        _camera_modes_req = (["gt", "pred"] if args.camera_mode == "both"
                             else [args.camera_mode])
        idfails = preds_io.validate_prediction_identity(
            run_identity, cfg=cfg, split=args.split,
            checkpoint_sha256=_ckpt_sha,
            extrinsics_convention=args.extrinsics_convention,
            requested_camera_modes=_camera_modes_req,
        )
        if idfails:
            for m in idfails:
                print(f"[fatal][preds-in] {m}", file=sys.stderr)
            return 2
        print(f"[preds-in] identity OK (config + checkpoint + manifests + schema match)")
    else:
        cfg, model, device, ckpt_meta = load_model_and_cfg(
            args.config, args.checkpoint, args.device,
        )
        _check_backbone_keys(ckpt_meta["missing_keys"])

    # Head detection. Pass 1 / normal: from the live model (DDP-unwrapped attrs,
    # cfg fallback). Pass 2: from the artifact's saved pred-key set, which is the
    # authoritative record of what the model actually emitted. Drives head-aware
    # metric emission (each head's metrics emitted iff its key is present).
    if is_pass2:
        _pred_keys = set(run_identity["prediction_identity"].get("pred_keys", []))
        heads = {
            "depth": "depth" in _pred_keys,
            "layout_depth": "layout_depth" in _pred_keys,
            "mask": "layout_mask_logits" in _pred_keys,
            "normal": "layout_normal" in _pred_keys,
            "camera": "pose_enc" in _pred_keys,
            "point": False, "track": False, "seg": False,
        }
    else:
        heads = detect_heads(model, cfg)
    has_mask_head = heads["mask"]
    has_normal_head = heads["normal"]
    has_layout_head = heads["layout_depth"]
    has_depth_head = heads["depth"]
    has_camera_head = heads["camera"]
    use_depth_as_layout = bool(args.use_depth_as_layout or (not has_layout_head))
    if (not has_layout_head) and (not args.use_depth_as_layout):
        print(
            "[warn] no layout-depth head detected; auto-forcing "
            "--use-depth-as-layout=True (E0 vanilla mode).",
            file=sys.stderr,
        )

    # Head-aware 3D gate: reconstruction needs a usable depth source. With no
    # layout-depth head AND no plain depth head there is nothing to unproject,
    # so skip the 3D pass cleanly instead of raising a KeyError per scene. 2D /
    # mask / normal / pose metrics still run as their own heads allow.
    run_3d = bool(has_layout_head or has_depth_head)
    if not run_3d:
        print(
            "[warn] no layout-depth or depth head detected; skipping 3D "
            "reconstruction metrics (2D / mask / normal / pose still run "
            "per their heads).",
            file=sys.stderr,
        )

    # Reconstruction camera modes to run. ``--camera-mode both`` runs the gt and
    # pred reconstructions in a single pass; ``primary_cm`` (gt when both) drives
    # the unprefixed 3D keys + the summary headline, while the secondary pred run
    # is emitted under a ``predcam_`` key prefix. Each mode resolves its own
    # alignment policy (gt → scale/scale_shift; pred → sim3 / scale+scale_shift).
    camera_modes: list[str] = (["gt", "pred"] if args.camera_mode == "both"
                               else [args.camera_mode])
    primary_cm = camera_modes[0]

    # Public ``scale_shift`` is an alias for the internal camera-frame track.
    def _norm_headline(h):
        return "scale_shift_cam" if h == "scale_shift" else h

    # Resolve per-eval-space alignment. ``--alignment`` is the global default;
    # ``--metric-alignment`` / ``--vggt-scene-alignment`` override it per space.
    # The headline track is resolved independently per space. ``view_count`` (the
    # per-manifest N) makes the ``auto`` policy view-count aware: 1-view manifests
    # additionally get the camera-frame scale_shift track; multi-view do not.
    # ``camera_mode`` selects the gt vs pred ``auto`` policy (never 'both').
    def _resolve_for_space(space: str, view_count=None, camera_mode=None) -> tuple:
        cm = camera_mode or primary_cm
        space_align = (args.metric_alignment if space == "metric"
                       else args.vggt_scene_alignment)
        align = space_align if space_align is not None else args.alignment
        (do_r, do_s, do_si, do_cam,
         default_head) = _resolve_alignment_tracks(
            align, cm, view_count=view_count,
        )
        space_head_override = (args.metric_headline_alignment if space == "metric"
                               else args.vggt_scene_headline_alignment)
        head = _norm_headline(space_head_override or args.headline_alignment or default_head)
        return align, (do_r, do_s, do_si, do_cam), head

    def _build_align_by_space(view_count=None, camera_mode=None) -> dict:
        out: dict[str, dict] = {}
        for _es in _resolve_eval_spaces(args):
            _align_arg, _flags, _head = _resolve_for_space(
                _es, view_count=view_count, camera_mode=camera_mode)
            out[_es] = {
                "alignment": _align_arg,
                "do_raw": _flags[0], "do_scale": _flags[1], "do_sim3": _flags[2],
                "do_scale_shift_cam": _flags[3],
                "headline": _head,
            }
        return out

    # View-count-agnostic resolution for the header print and the back-compat
    # meta (primary camera mode). The *per-manifest*, *per-camera-mode*
    # (view-aware) resolution is rebuilt inside the entry loop.
    align_by_space: dict[str, dict] = _build_align_by_space(
        view_count=None, camera_mode=primary_cm)

    # Back-compat: the legacy global args.alignment / args.headline_alignment
    # used for printing, JSON meta, and any caller that hasn't moved to the
    # per-space view yet. Compute them from the global ``--alignment`` (NOT
    # the per-space overrides) so the meta field stays stable.
    (do_raw, do_scale, do_sim3, do_scale_shift_cam,
     default_headline) = _resolve_alignment_tracks(
        args.alignment, primary_cm,
    )
    headline_alignment = _norm_headline(args.headline_alignment or default_headline)

    # One 3D-args namespace per camera mode (only camera_mode differs).
    args_for_3d_by_cm: dict[str, SimpleNamespace] = {
        cm: SimpleNamespace(
            extrinsics_convention=args.extrinsics_convention,
            use_depth_as_layout=use_depth_as_layout,
            scale_alignment=args.scale_alignment,
            max_points_per_scene=args.max_points_per_scene,
            camera_mode=cm,
            kdtree_workers=args.kdtree_workers,
        )
        for cm in camera_modes
    }

    methods = _resolve_methods(args)
    eval_spaces = _resolve_eval_spaces(args)
    hole_policies = args.render_holes_list

    print("=" * 70)
    print(f"  config             : {args.config}")
    print(f"  checkpoint         : {args.checkpoint} (epoch {ckpt_meta.get('checkpoint_epoch')})")
    print(f"  split              : {args.split}")
    print(f"  device             : {device}")
    print(f"  eval_space         : {args.eval_space}  (passes: {eval_spaces})")
    print(f"  camera_mode        : {args.camera_mode}  (reconstruction modes: {camera_modes}"
          f"{'; pred keys prefixed predcam_' if len(camera_modes) > 1 else ''})")
    print(f"  alignment (global) : {args.alignment}  (do_raw={do_raw}, do_scale={do_scale}, do_sim3={do_sim3}, do_scale_shift_cam={do_scale_shift_cam}; headline={headline_alignment})")
    print(f"                       (auto is view-count aware per manifest; see "
          f"per-manifest 'align[...]' lines below)")
    for _es, _cfg in align_by_space.items():
        print(f"  alignment[{_es:>10s}]: {_cfg['alignment']}  "
              f"(raw={_cfg['do_raw']}, scale={_cfg['do_scale']}, sim3={_cfg['do_sim3']}, "
              f"scale_shift_cam={_cfg['do_scale_shift_cam']}; "
              f"headline={_cfg['headline']})")
    print(f"  scale_alignment    : {args.scale_alignment} (3D)")
    print(f"  scale_align_2d     : {args.scale_align_2d}")
    print(f"  heads_enabled      : layout_depth={has_layout_head or use_depth_as_layout} "
          f"mask={has_mask_head} normal={has_normal_head} camera/pose={has_camera_head}")
    print(f"  run_3d             : {run_3d} "
          f"(pose/intrinsics metrics emitted whenever a camera head exists)")
    print(f"  use_depth_as_layout: {use_depth_as_layout}")
    print(f"  enable_postprocess : {args.enable_postprocess} (methods={methods})")
    print(f"  render_holes       : {hole_policies}")
    print("=" * 70)

    # --- Build dataset once ---
    from hydra.utils import instantiate
    split_cfg = select_split(cfg, args.split)
    inner_ds = instantiate(
        split_cfg.dataset,
        common_config=split_cfg.common_config,
        _recursive_=False,
    )
    re_child = _find_room_envelopes_child(inner_ds)
    re_child.inside_random = False
    scene_cam_lookup = {seq["scene_cam"]: i for i, seq in enumerate(re_child.sequences)}

    # --- Pre-flight pred-camera sanity (needs the live model; skipped in pass 2
    # where pose_enc presence is already validated via the artifact identity, and
    # in pass-1 forward-only where we save all heads and never score). ---
    if ("pred" in camera_modes) and (model is not None) and (not args.forward_only):
        _preflight_pred_camera_check(
            model, device, re_child, scene_cam_lookup,
            entries[0], args.allow_missing_pred_cameras,
        )

    out_root = _resolve_user_path(args.output_dir) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: write the run-identity manifest before the loop (so a mid-run
    # crash still leaves a valid identity for the shards already saved). ---
    preds_out_dir = None
    if is_save:
        preds_out_dir = str(_resolve_user_path(args.preds_out))
        _expected_keys = []
        if has_layout_head:
            _expected_keys.append("layout_depth")
        elif has_depth_head:
            _expected_keys.append("depth")
        if has_mask_head:
            _expected_keys.append("layout_mask_logits")
        if has_normal_head:
            _expected_keys.append("layout_normal")
        if has_camera_head:
            _expected_keys.append("pose_enc")
        try:
            _img = int(cfg.img_size)
        except Exception:
            _img = 224
        ckpt_sha = (preds_io.sha256_file(args.checkpoint)
                    if (args.checkpoint and os.path.isfile(args.checkpoint)) else None)
        identity = preds_io.build_run_identity(
            config_name=args.config,
            model_cfg_sha256=preds_io.model_cfg_hash(cfg),
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=ckpt_sha,
            checkpoint_epoch=ckpt_meta.get("checkpoint_epoch"),
            heads=heads,
            pred_keys=_expected_keys,
            pred_dtypes=preds_io.pred_dtypes_for(_expected_keys, args.preds_dtype),
            use_depth_as_layout=use_depth_as_layout,
            extrinsics_convention=args.extrinsics_convention,
            image_size=(_img, _img),
            preds_dtype_mode=args.preds_dtype,
            split=args.split, seed=args.seed, max_samples=args.max_samples,
            dataset_cfg_sha256=preds_io.dataset_cfg_hash(cfg, args.split),
            manifests=[
                {"label": e.label, "num_views": e.num_views, "n_samples": e.n_samples,
                 "content_sha256": preds_io.manifest_content_hash(
                     _load_eval_manifest(str(e.path)))}
                for e in entries
            ],
            git_commit=_git_commit_sha(),
            device=str(device),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        preds_io.write_run_identity(preds_out_dir, identity)
        print(f"[preds-out] wrote run_identity.json; saving shards ({args.preds_dtype}, "
              f"keys={_expected_keys}) under {preds_out_dir}"
              + ("  [forward-only: scoring skipped]" if args.forward_only else ""))

    from _timing import StageTimer
    timer = StageTimer(enabled=bool(args.profile))

    t_total_start = time.time()
    cross_rows: list[dict] = []

    for entry in entries:
        manifest = _load_eval_manifest(str(entry.path))
        _check_manifest_split(manifest, args.split, args.allow_split_mismatch)
        items, mode, mnv = _manifest_iter_items(manifest)
        # Per-manifest, view-count-aware alignment, per camera mode. ``mnv`` is
        # the manifest's view count (int for per-view manifests, None for mixed →
        # conservative policy: no scale_shift auto-selected). The headline track
        # is selected automatically from the resolved policy.
        align_by_cm_space_entry = {
            cm: _build_align_by_space(view_count=mnv, camera_mode=cm)
            for cm in camera_modes
        }
        print(f"\n=== {entry.label} ({entry.path.name}), {len(items)} samples "
              f"(mode={mode}, num_views={mnv}) ===")
        for cm in camera_modes:
            cm_tag = "predcam" if (len(camera_modes) > 1 and cm != primary_cm) else cm
            for _es, _cfg in align_by_cm_space_entry[cm].items():
                print(f"    align[{cm_tag}/{_es:>10s}]: {_cfg['alignment']} → "
                      f"raw={_cfg['do_raw']} scale={_cfg['do_scale']} "
                      f"sim3={_cfg['do_sim3']} scale_shift_cam={_cfg['do_scale_shift_cam']} "
                      f"headline={_cfg['headline']}")

        # 2-pass: where to save (pass 1) or load (pass 2) this manifest's shards.
        if is_save:
            preds_mode = "save"
            preds_dir_for_manifest = preds_io.manifest_preds_dir(
                preds_out_dir, args.split, entry.label)
        elif is_pass2:
            preds_mode = "load"
            preds_dir_for_manifest = preds_io.manifest_preds_dir(
                preds_in_dir, args.split, entry.label)
        else:
            preds_mode = None
            preds_dir_for_manifest = None

        per_method_payloads = evaluate_manifest_all_methods(
            manifest_entry=entry,
            manifest=manifest,
            items=items,
            mode=mode,
            manifest_num_views=mnv,
            model=model, device=device,
            re_child=re_child, scene_cam_lookup=scene_cam_lookup,
            cfg=cfg, args=args,
            camera_modes=camera_modes, primary_cm=primary_cm,
            args_for_3d_by_cm=args_for_3d_by_cm,
            methods=methods, eval_spaces=eval_spaces, hole_policies=hole_policies,
            has_mask_head=has_mask_head, has_normal_head=has_normal_head,
            has_layout_head=has_layout_head,
            align_by_cm_space=align_by_cm_space_entry,
            use_depth_as_layout=use_depth_as_layout,
            run_3d=run_3d,
            output_dir_for_manifest=out_root / entry.label,
            save_debug_every=args.save_debug_every,
            preds_mode=preds_mode,
            preds_dir_for_manifest=preds_dir_for_manifest,
            forward_only=args.forward_only,
            preds_dtype=args.preds_dtype,
            timer=timer,
        )

        # Pass-1 forward-only saves shards but produces no metrics → no rows.
        if args.forward_only:
            continue
        for method, payload in per_method_payloads.items():
            for es in eval_spaces:
                policies = hole_policies if method in POSTPROCESS_METHODS else ["fill"]
                for hp in policies:
                    cross_rows.append(
                        _extract_summary_row(
                            payload, entry, method, es, hp,
                            align_by_cm_space_entry[primary_cm][es]["headline"])
                    )

    # --- Cross-manifest summary (JSON: one row per manifest/method/space/holes) ---
    # Forward-only (pass 1) produces no metrics, skip the (empty) summary.json.
    if not args.forward_only:
        with open(out_root / "summary.json", "w") as f:
            json.dump(_json_safe(cross_rows), f, indent=2, default=str)

    run_meta = {
        "args": {k: (v if not isinstance(v, set) else sorted(v))
                 for k, v in vars(args).items()},
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt_meta.get("checkpoint_epoch"),
        "git_commit": _git_commit_sha(),
        "manifests": [
            {"label": e.label, "path": str(e.path),
             "num_views": e.num_views, "n_samples": e.n_samples}
            for e in entries
        ],
        "wall_clock_seconds": float(time.time() - t_total_start),
        "device": str(device),
        # 2-pass provenance.
        "mode": ("forward_only" if args.forward_only
                 else ("preds_in" if is_pass2
                       else ("save_and_score" if is_save else "normal"))),
        "preds_out": preds_out_dir,
        "preds_in": preds_in_dir,
        "preds_dtype": args.preds_dtype,
    }
    with open(out_root / "run_meta.json", "w") as f:
        json.dump(_json_safe(run_meta), f, indent=2, default=str)

    if args.forward_only:
        print(f"\n[forward-only] saved prediction shards under {preds_out_dir}")
        print(f"  Pass 2: python evaluations/eval_all_nview_manifests.py "
              f"--config {args.config} --output-dir <out> --preds-in {preds_out_dir}")
    print(f"\nWrote run_meta to {out_root}")
    print(f"Total wall-clock: {time.time() - t_total_start:.1f}s")

    if timer.enabled:
        timing_path = out_root / "timing_summary.json"
        timer.dump_json(timing_path)
        print(f"\nWrote timing summary to {timing_path}")
        print("[profile] Per-stage wall-clock timings (sorted by total):")
        timer.print_table(prefix="  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
