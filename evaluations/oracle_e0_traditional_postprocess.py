#!/usr/bin/env python3
"""Per-scene PARAMETER-SELECTION oracle for traditional post-processing on E0.

What this is
------------
For every scene in the full Room-Envelopes *test* set, this fits the classical
layout post-processing methods (RANSAC planes / Manhattan-snapped planes /
axis-aligned cuboid) on the **predicted E0 (vanilla VGGT) geometry**, sweeping a
grid of method parameters, scores every (method, parameter) trial against the
ground truth with the *exact official* 3D metric + alignment code, and then
selects the single best parameter setting **per scene per method** using GT.

    "Even if classical post-processing may pick the best parameters per scene
     using ground truth, how close does it get to the learned layout head?"

This is an **ORACLE CEILING, not a fair baseline**. GT is used to *select*
parameters, so the numbers are optimistic by construction. The geometry itself
is fit only on E0 predictions (never on the GT cloud). The fair single-default
baseline already lives in ``./eval_out/e0_vanilla_scored``; this script
never touches it.

Design / reuse
--------------
All metric formulas, alignment policy, point-cloud building, KD-tree caching and
the RANSAC/Manhattan/Cuboid fitters + renderer are **imported from the existing
evaluation pipeline** (``eval_all_nview_manifests`` and the modules it wires up)
so the produced numbers are bit-comparable to the official run. Nothing is
duplicated. The only new logic here is: the parameter grid, the per-scene sweep
loop, the GT-based selection, resume/checkpointing and CSV reporting.

The contract that makes the comparison meaningful (all fixed, matching the E0
official run; recorded in ``config.json``, NOT swept):

  config=room_envelopes/e0_vanilla_eval_only   use_depth_as_layout=True
  split=test   camera_mode=gt   eval_space=metric   extrinsics=w2c
  alignment=auto  -> for 2..5-view: raw + scale (headline = scale_aligned)
  scale_alignment=median_depth   render_holes=fill   seed=0
  view counts = same as the E0 official run (2,3,4,5)
  selection metric = scale_aligned_chamfer_l1   (lower is better)

Usage
-----
Smoke (a few scenes per view, tiny grid)::

    python evaluations/oracle_e0_traditional_postprocess.py --smoke

Full run (resumable)::

    python evaluations/oracle_e0_traditional_postprocess.py \
        --grid default --only-view-counts 2,3,4,5 --workers 8 --resume

Outputs land under ``--out-dir`` (default
``./eval_out/e0_per_scene_oracle_traditional_full``):
``config.json``, ``all_trials.csv``, ``best_per_scene.csv``,
``summary_by_method.csv``, ``failures.csv``, ``logs/run.log`` and per-scene
shards under ``shards/<view>/scene_XXXX.json`` (the resume unit).
"""
from __future__ import annotations

import os

# IMPORTANT: pin BLAS / OpenMP thread counts to 1 *before* numpy is imported
# (here or transitively via the orchestrator). Each parameter trial runs SVD
# (RANSAC refine / cuboid PCA) and scipy KD-tree work; under the per-scene
# process pool, an unpinned BLAS spawns ~ncores threads *per worker*, so N
# workers x ncores threads thrash the machine to a near-standstill (observed:
# 0 scenes finished in ~8 min with 4 workers). One BLAS thread per process +
# process-level parallelism is the correct layout for this workload. Forked
# pool workers inherit these. (scipy KD-tree query parallelism is controlled
# separately via --kdtree-workers, forced to 1 when --workers > 1.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

# --- Capture the user's cwd BEFORE importing the orchestrator, which chdir's
#     into training/ at import time for Hydra. CLI paths are resolved against
#     this, and the defaults are absolute anyway. ---
_USER_CWD = os.getcwd()

# The orchestrator lives next to this file. Importing it bootstraps sys.path for
# the whole eval source tree and exposes every helper we reuse.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_all_nview_manifests as E  # noqa: E402  (chdir + sys.path side effects)

# Re-exported helpers from the orchestrator namespace (all already wired).
_fit_planes_for_method = E._fit_planes_for_method
_render_method = E._render_method
_resolve_pred_layout_depth_np = E._resolve_pred_layout_depth_np
_fuse_world_cloud = E._fuse_world_cloud
_camera_centres = E._camera_centres
_scale_align_preds_per_frame = E._scale_align_preds_per_frame
_compute_3d_metrics_for_scene = E._compute_3d_metrics_for_scene
_resolve_alignment_tracks = E._resolve_alignment_tracks
discover_manifests = E.discover_manifests
_load_eval_manifest = E._load_eval_manifest
_manifest_iter_items = E._manifest_iter_items
_resolve_seq_index = E._resolve_seq_index
_find_room_envelopes_child = E._find_room_envelopes_child
load_cfg = E.load_cfg
select_split = E.select_split
preds_io = E.preds_io

import random as _random_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed contract (matches the E0 official run; not swept).
# ---------------------------------------------------------------------------
E0_CONFIG = "room_envelopes/e0_vanilla_eval_only"
USE_DEPTH_AS_LAYOUT = True       # E0 has no layout_depth head; depth is the proxy
CAMERA_MODE = "gt"
EVAL_SPACE = "metric"
EXTRINSICS_CONVENTION = "w2c"
SCALE_ALIGNMENT = "median_depth"
ALIGNMENT = "auto"               # -> (raw, scale) for 2..5-view
HOLE_POLICY = "fill"
SELECTION_METRIC = "scale_aligned_chamfer_l1"   # lower is better

# Metric columns surfaced into the CSVs (the full metric dict is kept in the
# per-scene JSON shards, so no secondary metric is ever lost).
METRIC_COLS = [
    "scale_aligned_chamfer_l1",         # <- selection metric (sum convention)
    "scale_aligned_chamfer_l1_mean",    # paper-style 0.5*(acc+com)
    "scale_aligned_chamfer_l2",
    "scale_aligned_accuracy_mean",
    "scale_aligned_completeness_mean",
    "scale_aligned_fscore_0.05",
    "scale_aligned_fscore_0.10",
    "scale_aligned_fscore_0.20",
    "scale_aligned_precision_0.10",
    "scale_aligned_recall_0.10",
    "scale_aligned_seen_chamfer_l1",
    "scale_aligned_unseen_chamfer_l1",
    "raw_chamfer_l1",
    "raw_fscore_0.10",
]
DIAG_COLS = ["n_planes", "render_coverage", "post_status", "cuboid_volume"]


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
# Each trial spec is {"method", "trial_id", "params": {...}}. ``params`` only
# carries the swept knobs; build_post_args() fills the rest with E0 defaults.
# Manhattan trials carry the RANSAC base they chain off (so the sweep is
# self-describing and the RANSAC-base reuse cache is keyed correctly).
RANSAC_DEFAULT = {"ransac_thresh": 0.03, "ransac_max_planes": 6,
                  "ransac_min_inliers": 500, "ransac_iters": 1000}


def _ransac_label(p: dict) -> str:
    return (f"thresh{p['ransac_thresh']:g}_planes{p['ransac_max_planes']}"
            f"_inl{p['ransac_min_inliers']}_it{p['ransac_iters']}")


def _q_label(q) -> str:
    return f"q{q[0]:g}-{q[1]:g}"


def build_grid(name: str, methods: list[str]) -> list[dict]:
    if name == "smoke":
        ransac_grid = dict(thresh=[0.03, 0.05], planes=[6], inliers=[500])
        manh_bases = [RANSAC_DEFAULT]
        manh_grid = dict(angle=[15.0, 20.0], merge=[0.06])
        cub_grid = dict(method=["pca_aabb"],
                        quantile=[(0.01, 0.99), (0.05, 0.95)],
                        min_box_dim=[0.5])
    elif name == "default":
        ransac_grid = dict(thresh=[0.02, 0.03, 0.05, 0.08], planes=[6, 8],
                           inliers=[300, 500])
        manh_bases = [RANSAC_DEFAULT]
        manh_grid = dict(angle=[10.0, 15.0, 20.0, 30.0], merge=[0.03, 0.06, 0.10])
        cub_grid = dict(method=["pca_aabb"],
                        quantile=[(0.01, 0.99), (0.02, 0.98), (0.05, 0.95)],
                        min_box_dim=[0.3, 0.5])
    elif name == "wide":
        ransac_grid = dict(thresh=[0.02, 0.03, 0.04, 0.05, 0.08],
                           planes=[6, 8, 10], inliers=[200, 500, 1000])
        manh_bases = [RANSAC_DEFAULT,
                      {**RANSAC_DEFAULT, "ransac_thresh": 0.05}]
        manh_grid = dict(angle=[10.0, 15.0, 20.0, 25.0, 30.0],
                         merge=[0.02, 0.04, 0.06, 0.10])
        cub_grid = dict(method=["pca_aabb", "from_manhattan"],
                        quantile=[(0.01, 0.99), (0.02, 0.98), (0.05, 0.95)],
                        min_box_dim=[0.3, 0.5])
    else:
        raise ValueError(f"unknown grid: {name!r}")

    grid: list[dict] = []
    if "raw" in methods:
        grid.append({"method": "raw", "trial_id": "raw", "params": {}})

    if "ransac" in methods:
        for t, pl, inl in itertools.product(
                ransac_grid["thresh"], ransac_grid["planes"], ransac_grid["inliers"]):
            p = {"ransac_thresh": t, "ransac_max_planes": pl,
                 "ransac_min_inliers": inl, "ransac_iters": 1000}
            grid.append({"method": "ransac",
                         "trial_id": f"ransac__{_ransac_label(p)}", "params": p})

    if "manhattan" in methods:
        for base in manh_bases:
            for ang, mrg in itertools.product(manh_grid["angle"], manh_grid["merge"]):
                p = {**base, "manhattan_angle_tol_deg": ang, "manhattan_merge_tol": mrg}
                grid.append({"method": "manhattan",
                             "trial_id": (f"manhattan__{_ransac_label(base)}"
                                          f"_ang{ang:g}_mrg{mrg:g}"),
                             "params": p})

    if "cuboid" in methods:
        for cm, q, mbd in itertools.product(
                cub_grid["method"], cub_grid["quantile"], cub_grid["min_box_dim"]):
            p = {"cuboid_method": cm, "cuboid_quantile": list(q), "cuboid_min_box_dim": mbd}
            grid.append({"method": "cuboid",
                         "trial_id": f"cuboid__{cm}_{_q_label(q)}_mbd{mbd:g}",
                         "params": p})
    return grid


def build_post_args(params: dict, seed: int) -> SimpleNamespace:
    """A namespace holding every field the reused fit/render helpers read."""
    return SimpleNamespace(
        # RANSAC
        ransac_max_planes=int(params.get("ransac_max_planes", 6)),
        ransac_thresh=float(params.get("ransac_thresh", 0.03)),
        ransac_min_inliers=int(params.get("ransac_min_inliers", 500)),
        ransac_iters=int(params.get("ransac_iters", 1000)),
        ransac_vectorized=False,
        seed=int(seed),
        plane_extent_quantile_t=(0.01, 0.99),
        # Manhattan
        manhattan_angle_tol_deg=float(params.get("manhattan_angle_tol_deg", 20.0)),
        manhattan_merge_tol=float(params.get("manhattan_merge_tol", 0.06)),
        # Cuboid
        cuboid_method=str(params.get("cuboid_method", "pca_aabb")),
        cuboid_quantile_t=tuple(params.get("cuboid_quantile", (0.01, 0.99))),
        cuboid_inlier_thresh=float(params.get("cuboid_inlier_thresh", 0.05)),
        cuboid_min_box_dim=float(params.get("cuboid_min_box_dim", 0.5)),
        # Render
        render_min_depth=0.05,
        render_max_depth=50.0,
    )


def _ransac_key(pa: SimpleNamespace) -> tuple:
    return (pa.ransac_thresh, pa.ransac_max_planes, pa.ransac_min_inliers,
            pa.ransac_iters, pa.ransac_vectorized)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _to_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _json_safe(obj):
    """Recursively make a metric dict JSON-serialisable (np scalars/arrays)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(E._repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None  # not a git repo here; code fingerprint below covers reproducibility


def _code_fingerprint() -> dict:
    """sha256 of the post-processing + metric source we depend on (no git here)."""
    root = Path(E._repo_root)
    files = [
        "training/geometry/postprocess_planes.py",
        "training/geometry/postprocess_cuboid.py",
        "training/geometry/render_planes_to_depth.py",
        "training/geometry/room_envelope_geometry.py",
        "evaluations/src/3d/eval_room_envelope_reconstruction.py",
        "evaluations/src/3d/metrics_chamfer.py",
        "evaluations/src/3d/alignment.py",
        "evaluations/src/3d/pointcloud.py",
        "evaluations/eval_all_nview_manifests.py",
        "evaluations/oracle_e0_traditional_postprocess.py",
    ]
    return {f: _sha256_file(root / f) for f in files}


def _resolve_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (Path(_USER_CWD) / q).resolve()


# ---------------------------------------------------------------------------
# Per-worker dataset context (built once per process; fork-safe via pid check).
# ---------------------------------------------------------------------------
_CTX: dict[str, Any] = {"pid": None}
_GLOBALS: dict[str, Any] = {}   # config name, split, etc. (set before pool fork)


def _get_ctx():
    """Lazily build (and memoise per-process) the dataset + scene_cam_lookup."""
    if _CTX.get("pid") == os.getpid() and "re_child" in _CTX:
        return _CTX
    from hydra.utils import instantiate
    cfg = load_cfg(_GLOBALS["config"])
    split_cfg = select_split(cfg, _GLOBALS["split"])
    inner_ds = instantiate(split_cfg.dataset,
                           common_config=split_cfg.common_config,
                           _recursive_=False)
    re_child = _find_room_envelopes_child(inner_ds)
    re_child.inside_random = False
    scene_cam_lookup = {seq["scene_cam"]: i for i, seq in enumerate(re_child.sequences)}
    _CTX.clear()
    _CTX.update(pid=os.getpid(), re_child=re_child, scene_cam_lookup=scene_cam_lookup)
    return _CTX


# ---------------------------------------------------------------------------
# Core per-scene computation
# ---------------------------------------------------------------------------
def _extract_metric_row(m3d: dict) -> dict:
    return {c: _to_float(m3d.get(c)) for c in METRIC_COLS}


def _empty_metric_row() -> dict:
    return {c: float("nan") for c in METRIC_COLS}


def process_scene(view_label: str, num_views: Optional[int], scene_i: int,
                  item: dict, grid: list[dict], opt_metric: str,
                  seed: int, max_points_per_scene: int, max_fuse_points: int,
                  kdtree_workers: int, preds_dir_for_manifest: str,
                  extra_tracks: bool = False) -> dict:
    """Run the full parameter sweep for one scene; return a shard dict."""
    ctx = _get_ctx()
    re_child = ctx["re_child"]
    scene_cam_lookup = ctx["scene_cam_lookup"]

    # Match the official per-scene RNG reset (RANSAC uses its own default_rng,
    # but this keeps any global-RNG-dependent dataset/sampling deterministic).
    np.random.seed(seed + scene_i)
    _random_mod.seed(seed + scene_i)

    this_views = int(num_views) if num_views is not None else int(item["num_views"])

    # ---- GT sample + cached E0 predictions ----
    seq_index = _resolve_seq_index(item, scene_cam_lookup)
    sample = re_child.get_data(seq_index=seq_index, img_per_seq=this_views,
                               ids=list(item["ids"]), aspect_ratio=1.0)
    preds_one = preds_io.load_scene_shard(
        preds_io.scene_shard_path(preds_dir_for_manifest, scene_i))

    seq_name = str(sample.get("seq_name", f"scene_{scene_i:04d}"))

    # ---- per-scene shared setup (identical basis to the official run) ----
    scaled_preds_one, _frame_scales = _scale_align_preds_per_frame(
        preds_one, sample, USE_DEPTH_AS_LAYOUT)
    pred_ld_S = _resolve_pred_layout_depth_np(scaled_preds_one, USE_DEPTH_AS_LAYOUT)
    if pred_ld_S is None:
        raise RuntimeError("no usable depth/layout_depth in E0 predictions")
    S, H, W = pred_ld_S.shape
    K_S = np.stack([np.asarray(k, dtype=np.float32) for k in sample["intrinsics"]], 0)
    E_S = np.stack([np.asarray(e, dtype=np.float32) for e in sample["extrinsics"]], 0)
    fused = _fuse_world_cloud(pred_ld_S, K_S, E_S,
                              max_points=max_fuse_points, seed=seed)
    interior_pt = _camera_centres(E_S).mean(axis=0)

    do_raw, do_scale, do_sim3, do_ssc, _headline = _resolve_alignment_tracks(
        ALIGNMENT, CAMERA_MODE, view_count=this_views)
    # The oracle selects on the `scale` track only. Computing the extra reference
    # tracks (raw / sim3 / scale_shift_cam) roughly doubles the per-trial chamfer
    # cost, the per-scene bottleneck, yet leaves every scale_aligned_* value
    # bit-identical (verified: max |Δ| = 0 across all trials, with the per-scene
    # reseed above). So by default we compute ONLY the scale track. Pass
    # --extra-tracks to restore raw/sim3/scale_shift_cam reference columns
    # (e.g. to reproduce the official combined_metrics layout for 1-view).
    if not extra_tracks:
        do_raw = do_sim3 = do_ssc = False
        do_scale = True

    args_3d = SimpleNamespace(
        extrinsics_convention=EXTRINSICS_CONVENTION,
        use_depth_as_layout=USE_DEPTH_AS_LAYOUT,
        scale_alignment=SCALE_ALIGNMENT,
        max_points_per_scene=int(max_points_per_scene),
        camera_mode=CAMERA_MODE,
        kdtree_workers=int(kdtree_workers),
    )

    gt_cache: dict = {}                 # GT cloud / KD-tree built once, reused
    ransac_cache: dict = {}             # RANSAC plane-set reuse keyed by params
    trials: list[dict] = []

    for spec in grid:
        method = spec["method"]
        row = {
            "view_label": view_label, "scene_idx": int(scene_i),
            "seq_name": seq_name, "n_views": int(S), "method": method,
            "trial_id": spec["trial_id"],
            "params_json": json.dumps(spec["params"], sort_keys=True),
            "status": "ok", "reason": "",
        }
        row.update(_empty_metric_row())
        row.update({c: float("nan") for c in DIAG_COLS})
        row["post_status"] = ""
        diag: dict = {}

        try:
            if method == "raw":
                preds_for_3d = preds_one          # raw depth; 3D scale track aligns it
            else:
                if fused is None or len(fused) == 0:
                    raise RuntimeError("fused_cloud_unavailable")
                pa = build_post_args(spec["params"], seed)
                plane_cache: dict = {}
                rkey = _ransac_key(pa)
                if rkey in ransac_cache:
                    plane_cache["ransac"] = ransac_cache[rkey]
                planes, diag = _fit_planes_for_method(method, fused, pa, plane_cache)
                if "ransac" in plane_cache:
                    ransac_cache.setdefault(rkey, plane_cache["ransac"])
                row["n_planes"] = _to_float(diag.get("n_planes"))
                row["post_status"] = str(diag.get("plane_status")
                                         or diag.get("cuboid_status")
                                         or diag.get("manhattan_status") or "")
                if "cuboid_volume" in diag:
                    row["cuboid_volume"] = _to_float(diag.get("cuboid_volume"))
                if not planes:
                    row["status"] = "no_geometry"
                    row["reason"] = row["post_status"] or "no_planes"
                    trials.append(row)
                    continue
                d, m_valid = _render_method(planes, K_S, E_S, H, W, interior_pt, pa)
                if d is None or m_valid is None:
                    row["status"] = "render_failed"
                    row["reason"] = "render_failed"
                    trials.append(row)
                    continue
                row["render_coverage"] = float(np.asarray(m_valid).mean())
                # hole policy = fill: rendered where valid, scale-aligned pred elsewhere
                d_filled = np.where(m_valid, d, pred_ld_S)
                preds_for_3d = {"layout_depth": d_filled[..., None].astype(np.float32)}

            m3d, _plys = _compute_3d_metrics_for_scene(
                sample=sample, preds_one=preds_for_3d, args=args_3d,
                image_hw=(H, W), do_raw=do_raw, do_scale=do_scale,
                do_sim3=do_sim3, do_scale_shift_cam=do_ssc,
                gt_cache=gt_cache,
            )
            row.update(_extract_metric_row(m3d))
            row["_full_metrics"] = _json_safe(
                {k: v for k, v in m3d.items()
                 if not isinstance(v, (np.ndarray, list))})
        except Exception as e:                       # per-trial soft failure
            row["status"] = "failed"
            row["reason"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()

        trials.append(row)

    best = _select_best(trials, opt_metric)
    return {
        "view_label": view_label, "scene_idx": int(scene_i),
        "seq_name": seq_name, "n_views": int(S),
        "opt_metric": opt_metric, "n_trials": len(trials),
        "trials": trials, "best_per_method": best,
    }


def _select_best(trials: list[dict], opt_metric: str) -> dict:
    """Per method, pick the trial with the lowest opt_metric among valid trials.

    'Valid' = status ok and a finite, positive opt_metric value. Returns
    {method: {best_trial_id, best_value, n_valid, n_total, row}}. raw is
    included too (single trial)."""
    best: dict = {}
    methods = []
    for t in trials:
        if t["method"] not in methods:
            methods.append(t["method"])
    for method in methods:
        cand = [t for t in trials if t["method"] == method]
        valid = [t for t in cand
                 if t["status"] == "ok" and np.isfinite(t.get(opt_metric, np.nan))]
        entry = {"n_total": len(cand), "n_valid": len(valid)}
        if valid:
            win = min(valid, key=lambda t: t[opt_metric])
            # Sanity: the winner's value really is the min over valid trials.
            min_val = min(_to_float(t[opt_metric]) for t in valid)
            entry["sanity_is_min"] = bool(abs(_to_float(win[opt_metric]) - min_val) < 1e-12)
            entry["best_trial_id"] = win["trial_id"]
            entry["best_value"] = _to_float(win[opt_metric])
            entry["row"] = {k: v for k, v in win.items()
                            if k not in ("_full_metrics", "traceback")}
        else:
            entry["sanity_is_min"] = None
            entry["best_trial_id"] = None
            entry["best_value"] = float("nan")
            entry["row"] = None
        best[method] = entry
    return best


# ---------------------------------------------------------------------------
# Shard IO (the resume unit)
# ---------------------------------------------------------------------------
def _shard_path(out_dir: Path, view_label: str, scene_i: int) -> Path:
    return out_dir / "shards" / view_label / f"scene_{scene_i:04d}.json"


def _write_shard(path: Path, shard: dict, grid_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shard = dict(shard)
    shard["grid_hash"] = grid_hash
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(_json_safe(shard), f)
    os.replace(tmp, path)


def _load_shard_if_valid(path: Path, grid_hash: str) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            shard = json.load(f)
        if shard.get("grid_hash") != grid_hash:
            return None
        return shard
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Worker entry (module-level so it is picklable for the process pool)
# ---------------------------------------------------------------------------
def _worker(task: dict) -> dict:
    out_dir = Path(task["out_dir"])
    path = _shard_path(out_dir, task["view_label"], task["scene_idx"])
    cached = _load_shard_if_valid(path, task["grid_hash"])
    if cached is not None:
        return {"view_label": task["view_label"], "scene_idx": task["scene_idx"],
                "status": "cached"}
    try:
        shard = process_scene(
            view_label=task["view_label"], num_views=task["num_views"],
            scene_i=task["scene_idx"], item=task["item"], grid=task["grid"],
            opt_metric=task["opt_metric"], seed=task["seed"],
            max_points_per_scene=task["max_points_per_scene"],
            max_fuse_points=task["max_fuse_points"],
            kdtree_workers=task["kdtree_workers"],
            preds_dir_for_manifest=task["preds_dir_for_manifest"],
            extra_tracks=task.get("extra_tracks", False),
        )
        shard["status"] = "ok"
        _write_shard(path, shard, task["grid_hash"])
        return {"view_label": task["view_label"], "scene_idx": task["scene_idx"],
                "status": "ok"}
    except Exception as e:
        shard = {"view_label": task["view_label"], "scene_idx": task["scene_idx"],
                 "status": "scene_failed",
                 "reason": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc(),
                 "trials": [], "best_per_method": {}}
        _write_shard(path, shard, task["grid_hash"])
        return {"view_label": task["view_label"], "scene_idx": task["scene_idx"],
                "status": "scene_failed", "reason": shard["reason"]}


# ---------------------------------------------------------------------------
# Aggregation -> CSVs
# ---------------------------------------------------------------------------
def _read_all_shards(out_dir: Path, grid_hash: str) -> list[dict]:
    shards = []
    sdir = out_dir / "shards"
    if not sdir.exists():
        return shards
    for p in sorted(sdir.rglob("scene_*.json")):
        try:
            with open(p) as f:
                shard = json.load(f)
            if shard.get("grid_hash") == grid_hash:
                shards.append(shard)
        except (OSError, json.JSONDecodeError):
            continue
    return shards


def _e0_fair_default(e0_scored_dir: Path) -> dict:
    """Read the fair single-default chamfer from the official E0 scored run.
    Returns {(view_label, method): chamfer_l1}."""
    out: dict = {}
    if not e0_scored_dir.exists():
        return out
    test = e0_scored_dir / "test"
    if not test.exists():
        return out
    for view_dir in test.iterdir():
        if not view_dir.is_dir():
            continue
        for method_dir in view_dir.iterdir():
            cm = method_dir / "combined_metrics.json"
            if not cm.exists():
                continue
            try:
                with open(cm) as f:
                    data = json.load(f)
                blk = (data.get("metrics", {}).get("3d", {})
                       .get("metric", {}).get("scale_aligned", {}))
                val = blk.get("chamfer_l1")
                if val is not None:
                    out[(view_dir.name, method_dir.name)] = float(val)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
    return out


def write_csvs(out_dir: Path, grid_hash: str, e0_scored_dir: Path,
               opt_metric: str, log) -> dict:
    shards = _read_all_shards(out_dir, grid_hash)
    base_cols = ["view_label", "scene_idx", "seq_name", "n_views", "method",
                 "trial_id", "params_json", "status", "reason"]
    trial_cols = base_cols + METRIC_COLS + DIAG_COLS

    # ---- all_trials.csv ----
    all_path = out_dir / "all_trials.csv"
    n_trials = 0
    with open(all_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trial_cols, extrasaction="ignore")
        w.writeheader()
        for sh in shards:
            for t in sh.get("trials", []):
                w.writerow({c: t.get(c, "") for c in trial_cols})
                n_trials += 1

    # ---- best_per_scene.csv ----
    best_cols = (["view_label", "scene_idx", "seq_name", "n_views", "method",
                  "selection_metric", "best_trial_id", "best_value",
                  "best_params_json", "n_valid", "n_total", "sanity_is_min"]
                 + METRIC_COLS + DIAG_COLS)
    best_path = out_dir / "best_per_scene.csv"
    best_rows: list[dict] = []
    sanity_violations = 0
    with open(best_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=best_cols, extrasaction="ignore")
        w.writeheader()
        for sh in shards:
            for method, entry in (sh.get("best_per_method") or {}).items():
                rowsrc = entry.get("row") or {}
                if entry.get("sanity_is_min") is False:
                    sanity_violations += 1
                out_row = {
                    "view_label": sh["view_label"], "scene_idx": sh["scene_idx"],
                    "seq_name": sh.get("seq_name", ""), "n_views": sh.get("n_views", ""),
                    "method": method, "selection_metric": opt_metric,
                    "best_trial_id": entry.get("best_trial_id"),
                    "best_value": entry.get("best_value"),
                    "best_params_json": rowsrc.get("params_json", ""),
                    "n_valid": entry.get("n_valid"), "n_total": entry.get("n_total"),
                    "sanity_is_min": entry.get("sanity_is_min"),
                }
                for c in METRIC_COLS + DIAG_COLS:
                    out_row[c] = rowsrc.get(c, "")
                w.writerow(out_row)
                best_rows.append(out_row)

    # ---- summary_by_method.csv ----
    fair = _e0_fair_default(e0_scored_dir)
    summ_cols = ["method", "view_label", "selection_metric", "n_scenes",
                 "n_scenes_with_valid",
                 "oracle_chamfer_l1_mean", "oracle_chamfer_l1_median",
                 "oracle_chamfer_l1_mean_paperstyle",
                 "oracle_fscore_0.05_mean", "oracle_fscore_0.10_mean",
                 "oracle_fscore_0.20_mean",
                 "oracle_accuracy_mean", "oracle_completeness_mean",
                 "fair_default_chamfer_l1", "oracle_minus_fair",
                 "oracle_le_fair_ok"]
    # group best_rows by (method, view_label) and also a pooled "all" view.
    groups: dict[tuple, list[dict]] = {}
    for r in best_rows:
        groups.setdefault((r["method"], r["view_label"]), []).append(r)
        groups.setdefault((r["method"], "all"), []).append(r)

    def _col(rows, c):
        vals = [float(r[c]) for r in rows
                if r.get(c) not in ("", None) and np.isfinite(_to_float(r.get(c)))]
        return vals

    summ_rows = []
    le_violations = 0
    with open(out_dir / "summary_by_method.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ_cols, extrasaction="ignore")
        w.writeheader()
        for (method, view), rows in sorted(groups.items()):
            ch = _col(rows, "scale_aligned_chamfer_l1")
            ch_mean = float(np.mean(ch)) if ch else float("nan")
            fair_val = fair.get((view, method), float("nan"))
            # Aggregate cross-check (only meaningful per-view at full scale; the
            # pooled "all" row has no single official reference). NA when fair
            # is unavailable so a blank never reads as a failed invariant.
            if np.isfinite(ch_mean) and np.isfinite(fair_val):
                le_ok = ch_mean <= fair_val + 1e-6
                if view != "all" and not le_ok:
                    le_violations += 1
            else:
                le_ok = ""
            row = {
                "method": method, "view_label": view,
                "selection_metric": opt_metric, "n_scenes": len(rows),
                "n_scenes_with_valid": len(ch),
                "oracle_chamfer_l1_mean": ch_mean,
                "oracle_chamfer_l1_median": float(np.median(ch)) if ch else float("nan"),
                "oracle_chamfer_l1_mean_paperstyle":
                    float(np.mean(_col(rows, "scale_aligned_chamfer_l1_mean")))
                    if _col(rows, "scale_aligned_chamfer_l1_mean") else float("nan"),
                "oracle_fscore_0.05_mean":
                    float(np.mean(_col(rows, "scale_aligned_fscore_0.05")))
                    if _col(rows, "scale_aligned_fscore_0.05") else float("nan"),
                "oracle_fscore_0.10_mean":
                    float(np.mean(_col(rows, "scale_aligned_fscore_0.10")))
                    if _col(rows, "scale_aligned_fscore_0.10") else float("nan"),
                "oracle_fscore_0.20_mean":
                    float(np.mean(_col(rows, "scale_aligned_fscore_0.20")))
                    if _col(rows, "scale_aligned_fscore_0.20") else float("nan"),
                "oracle_accuracy_mean":
                    float(np.mean(_col(rows, "scale_aligned_accuracy_mean")))
                    if _col(rows, "scale_aligned_accuracy_mean") else float("nan"),
                "oracle_completeness_mean":
                    float(np.mean(_col(rows, "scale_aligned_completeness_mean")))
                    if _col(rows, "scale_aligned_completeness_mean") else float("nan"),
                "fair_default_chamfer_l1": fair_val,
                "oracle_minus_fair": (ch_mean - fair_val)
                    if (np.isfinite(ch_mean) and np.isfinite(fair_val)) else float("nan"),
                "oracle_le_fair_ok": le_ok,
            }
            w.writerow(row)
            summ_rows.append(row)

    # ---- failures.csv ----
    with open(out_dir / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["view_label", "scene_idx", "method", "trial_id", "status", "reason"])
        for sh in shards:
            if sh.get("status") == "scene_failed":
                w.writerow([sh.get("view_label"), sh.get("scene_idx"), "*", "*",
                            "scene_failed", sh.get("reason", "")])
            for t in sh.get("trials", []):
                if t.get("status") not in ("ok",):
                    w.writerow([t.get("view_label"), t.get("scene_idx"),
                                t.get("method"), t.get("trial_id"),
                                t.get("status"), t.get("reason", "")])

    log(f"  aggregation: {len(shards)} shards, {n_trials} trials, "
        f"{len(best_rows)} best-rows")
    if sanity_violations:
        log(f"  WARNING: {sanity_violations} per-scene selection sanity violations "
            f"(best != min over valid trials)")
    if le_violations:
        log(f"  WARNING: {le_violations} (method,view) groups where oracle > "
            f"fair-default (unexpected; default params are in the grid)")
    return {"n_shards": len(shards), "n_trials": n_trials,
            "n_best_rows": len(best_rows),
            "sanity_violations": sanity_violations,
            "oracle_le_fair_violations": le_violations,
            "summary": summ_rows}


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-scene parameter-selection ORACLE for traditional "
                    "post-processing on E0 (vanilla VGGT) geometry. ORACLE "
                    "CEILING, not a fair baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pred-dir", default="./eval_out/e0_vanilla_preds",
                   help="E0 cached prediction dir (must contain run_identity.json).")
    p.add_argument("--out-dir",
                   default="./eval_out/e0_per_scene_oracle_traditional_full")
    p.add_argument("--manifest-dir", default=E._DEFAULT_MANIFEST_DIR)
    p.add_argument("--config", default=E0_CONFIG,
                   help="Hydra config used to build the GT dataset (E0).")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--only-view-counts", default="2,3,4,5",
                   help="Comma list, e.g. '2,3,4,5'. Matches the E0 official run.")
    p.add_argument("--methods", default="raw,ransac,manhattan,cuboid",
                   help="Comma list subset of raw,ransac,manhattan,cuboid.")
    p.add_argument("--grid", default="default", choices=("smoke", "default", "wide"))
    p.add_argument("--opt-metric", default=SELECTION_METRIC,
                   help="Per-scene selection metric (lower is better).")
    p.add_argument("--seed", type=int, default=0,
                   help="RANSAC + per-scene RNG seed (E0 official used 0).")
    p.add_argument("--max-points-per-scene", type=int, default=50000)
    p.add_argument("--max-fuse-points", type=int, default=200000)
    p.add_argument("--kdtree-workers", type=int, default=-1,
                   help="scipy KDTree query workers per scorer call. Forced to 1 "
                        "when --workers > 1 to avoid CPU oversubscription.")
    p.add_argument("--workers", type=int, default=1,
                   help="Per-scene process pool size. 1 = single process.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Global cap on total scenes (after view filtering).")
    p.add_argument("--max-scenes-per-view", type=int, default=None,
                   help="Cap scenes per view manifest (handy for smoke runs).")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip scenes whose shard already exists for this grid.")
    p.add_argument("--e0-scored-dir", default="./eval_out/e0_vanilla_scored",
                   help="Official E0 scored run, for fair-default reference columns.")
    p.add_argument("--extra-tracks", action="store_true",
                   help="Also compute the raw / sim3 / scale_shift_cam reference "
                        "tracks (default: only the `scale` track that selection "
                        "uses). Default-off is ~2-4x faster and leaves every "
                        "scale_aligned_* value bit-identical; turn on only if you "
                        "want those reference columns (e.g. raw_chamfer_l1, or the "
                        "1-view scale_shift_cam track).")
    p.add_argument("--smoke", action="store_true",
                   help="Shortcut: --grid smoke --max-scenes-per-view 3 --workers 1.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip computation; just (re)build CSVs from existing shards.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.grid = "smoke"
        if args.max_scenes_per_view is None:
            args.max_scenes_per_view = 3
        args.workers = 1

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    view_counts = {int(x) for x in args.only_view_counts.split(",") if x.strip()}
    if 1 in view_counts:
        print("NOTE: 1-view requested. Supported (E0 preds + manifest exist; the "
              "E0 official run scored 1-view, so fair-default columns populate). "
              "Caveat: RANSAC/Manhattan/Cuboid are geometrically ill-posed on a "
              "single frustum (one view can't constrain a full room), so expect "
              "more degenerate/fallback trials at full scale; they are recorded "
              "and excluded from oracle selection. For 1-view, alignment=auto also "
              "computes the scale_shift_cam (LaRI/Room-Envelopes) track; oracle "
              "selection still uses scale_aligned_chamfer_l1 for cross-view "
              "consistency. Both tracks are saved per trial. Proceeding.")

    out_dir = _resolve_path(args.out_dir)
    pred_dir = _resolve_path(args.pred_dir)
    manifest_dir = _resolve_path(args.manifest_dir)
    e0_scored_dir = _resolve_path(args.e0_scored_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / "run.log"
    _logf = open(log_path, "a")

    def log(msg: str):
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        _logf.write(line + "\n")
        _logf.flush()

    grid = build_grid(args.grid, methods)
    grid_hash = hashlib.sha256(
        json.dumps([{"method": g["method"], "params": g["params"]} for g in grid],
                   sort_keys=True).encode()).hexdigest()[:16]

    log("=" * 78)
    log("E0 per-scene PARAMETER-SELECTION ORACLE, traditional post-processing")
    log("ORACLE CEILING (GT used only to pick best params per scene); NOT a "
        "fair baseline.")
    log(f"  grid={args.grid} ({len(grid)} trials/scene)  grid_hash={grid_hash}")
    log(f"  methods={methods}  selection_metric={args.opt_metric}")
    log(f"  pred_dir={pred_dir}")
    log(f"  out_dir={out_dir}")
    log(f"  view_counts={sorted(view_counts)}  workers={args.workers}  "
        f"resume={args.resume}")

    # ---- prediction-identity guard ----
    try:
        ident = preds_io.read_run_identity(str(pred_dir))
        pi = ident.get("prediction_identity", {})
        log(f"  pred identity: config={pi.get('config_name')} "
            f"use_depth_as_layout={pi.get('use_depth_as_layout')} "
            f"keys={pi.get('pred_keys')}")
        if pi.get("config_name") not in (E0_CONFIG, None):
            log(f"  WARNING: pred config {pi.get('config_name')!r} != {E0_CONFIG!r}")
    except Exception as e:
        log(f"  WARNING: could not read run_identity.json: {e}")
        ident = {}

    # ---- write config.json (reproducibility) ----
    config_blob = {
        "label": "per_scene_oracle",
        "oracle_kind": "per_scene_parameter_selection_on_predicted_E0_geometry",
        "is_fair_baseline": False,
        "note": ("Geometry fit on E0 predictions; GT used ONLY to select the "
                 "best parameter setting per scene. Optimistic ceiling."),
        "argv": sys.argv,
        "args": vars(args),
        "fixed_contract": {
            "config": args.config, "use_depth_as_layout": USE_DEPTH_AS_LAYOUT,
            "camera_mode": CAMERA_MODE, "eval_space": EVAL_SPACE,
            "extrinsics_convention": EXTRINSICS_CONVENTION,
            "alignment": ALIGNMENT, "scale_alignment": SCALE_ALIGNMENT,
            "hole_policy": HOLE_POLICY, "selection_metric": args.opt_metric,
            "chamfer_convention_note": ("scale_aligned_chamfer_l1 is the SUM "
                "convention (accuracy+completeness); the paper-style 0.5*(a+c) "
                "is saved as scale_aligned_chamfer_l1_mean."),
        },
        "grid_name": args.grid, "grid_hash": grid_hash,
        "grid": grid,
        "n_trials_per_scene": len(grid),
        "split": args.split, "view_counts": sorted(view_counts),
        "manifest_dir": str(manifest_dir), "pred_dir": str(pred_dir),
        "prediction_identity": ident.get("prediction_identity", {}),
        "git_commit": _git_commit(),
        "code_fingerprint_sha256": _code_fingerprint(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(_json_safe(config_blob), f, indent=2)

    if args.aggregate_only:
        log("aggregate-only: rebuilding CSVs from existing shards")
        stats = write_csvs(out_dir, grid_hash, e0_scored_dir, args.opt_metric, log)
        log(f"done: {stats}")
        _logf.close()
        return 0

    # ---- discover manifests + enumerate scenes ----
    entries = discover_manifests(
        manifest_dir, split=args.split, include_mixed=False,
        only_view_counts=view_counts, manifest_glob_override=None)
    entries = [e for e in entries
               if (e.num_views is not None and e.num_views in view_counts)]
    log(f"  discovered {len(entries)} manifests: "
        f"{[(e.label, e.n_samples) for e in entries]}")

    tasks: list[dict] = []
    for ent in entries:
        manifest = _load_eval_manifest(str(ent.path))
        items, mode, manifest_nv = _manifest_iter_items(manifest, ent.num_views)
        preds_dir_for_manifest = str(pred_dir / args.split / ent.label)
        n = len(items)
        if args.max_scenes_per_view is not None:
            n = min(n, args.max_scenes_per_view)
        for i in range(n):
            tasks.append({
                "out_dir": str(out_dir), "view_label": ent.label,
                "num_views": manifest_nv if mode == "per_view" else None,
                "scene_idx": i, "item": items[i], "grid": grid,
                "grid_hash": grid_hash, "opt_metric": args.opt_metric,
                "seed": args.seed,
                "max_points_per_scene": args.max_points_per_scene,
                "max_fuse_points": args.max_fuse_points,
                "kdtree_workers": (1 if args.workers > 1 else args.kdtree_workers),
                "preds_dir_for_manifest": preds_dir_for_manifest,
                "extra_tracks": args.extra_tracks,
            })

    if args.max_samples is not None:
        tasks = tasks[:args.max_samples]

    # Resume: drop tasks whose shard already exists for this grid.
    if args.resume:
        kept = []
        skipped = 0
        for t in tasks:
            if _load_shard_if_valid(_shard_path(out_dir, t["view_label"],
                                                t["scene_idx"]), grid_hash) is not None:
                skipped += 1
            else:
                kept.append(t)
        log(f"  resume: {skipped} scenes already done, {len(kept)} to run")
        tasks = kept

    total = len(tasks)
    log(f"  running {total} scenes x {len(grid)} trials "
        f"(~{total * len(grid)} scorings)")

    # ---- run ----
    t0 = time.time()
    done = 0
    fails = 0
    _GLOBALS["config"] = args.config
    _GLOBALS["split"] = args.split

    if args.workers and args.workers > 1 and total > 0:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            for res in pool.imap_unordered(_worker, tasks, chunksize=1):
                done += 1
                if res.get("status") == "scene_failed":
                    fails += 1
                if done % 20 == 0 or done == total:
                    rate = done / max(1e-6, time.time() - t0)
                    eta = (total - done) / max(1e-6, rate)
                    log(f"  [{done}/{total}] {rate:.2f} scenes/s "
                        f"eta={eta/60:.1f}min fails={fails}")
    else:
        for t in tasks:
            res = _worker(t)
            done += 1
            if res.get("status") == "scene_failed":
                fails += 1
            if done % 10 == 0 or done == total:
                rate = done / max(1e-6, time.time() - t0)
                eta = (total - done) / max(1e-6, rate)
                log(f"  [{done}/{total}] {rate:.2f} scenes/s "
                    f"eta={eta/60:.1f}min fails={fails}")

    log(f"  compute done in {(time.time() - t0)/60:.1f} min "
        f"({done} scenes, {fails} scene-failures)")

    # ---- aggregate ----
    stats = write_csvs(out_dir, grid_hash, e0_scored_dir, args.opt_metric, log)
    log(f"  CSVs written to {out_dir}")
    # Brief headline table to the log.
    for r in stats["summary"]:
        if r["view_label"] == "all":
            log(f"    [{r['method']:>9s}] oracle CD={r['oracle_chamfer_l1_mean']:.4f} "
                f"| fair-default CD={r['fair_default_chamfer_l1']} "
                f"| F@0.10={r['oracle_fscore_0.10_mean']:.4f} "
                f"(n={r['n_scenes']})")
    log("DONE.")
    _logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
