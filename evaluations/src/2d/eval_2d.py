#!/usr/bin/env python3
"""Unified 2D pixel-level evaluation across experiments E0-E8.

Reuses the same metric functions and per-scene 2D implementation as
eval_room_envelope_reconstruction.py (via _common.py) so the numbers line up
with training-time val and with 3D-eval scenes.

Single subcommand:

  run        Evaluate one experiment, write
             <output_dir>/<split>/<experiment>.json.

Results are written as JSON only (vggt_scene + metric-space metrics when
--eval-space both).

Example sweep:

  for E in E0 E1 E2 E3 E4 E5 E6 E7 E8; do
    .venv/bin/python evaluations/src/2d/eval_2d.py run \\
        --experiment $E --split test --output-dir outputs/eval_2d
  done
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# --- repo path setup (flat sys.path bootstrap; see common/_paths.py) --------
_d = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_d) != "src":
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "common"))
import _paths  # noqa: E402: adds repo root, training, all eval subdirs to sys.path
from _paths import REPO_ROOT as _repo_root, TRAINING_DIR as _training_dir  # noqa: E402
# Hydra config_path resolution and downstream imports expect cwd == training/.
os.chdir(_training_dir)

from _common import (  # noqa: E402
    to_np,
    frame,
    load_model_and_cfg,
    enable_unique_scene_mode,
    select_split,
    aggregate,
)
from scene_metrics import compute_2d_metrics_for_scene  # noqa: E402
from _oca_eval_helpers import forward_model, cameras_from_sample  # noqa: E402
from normalization import normalize_sample_vggt_scene  # noqa: E402
from depth_scale import summarize_depth_scale  # noqa: E402
from train_utils import wandb_logger  # noqa: E402
from train_utils import wandb_clean_logger  # noqa: E402


def _scale_align_preds_per_frame(preds_one: dict, sample: dict,
                                 use_depth_as_layout: bool
                                 ) -> tuple[dict, list[float]]:
    """Return a copy of ``preds_one`` whose ``layout_depth`` (or ``depth`` for
    E0) is multiplied per-frame by ``median(gt_depth / pred_depth)``, computed
    over each frame's valid pixel set. Standard monocular-depth eval protocol
    (e.g. MiDaS), needed when pred is in vggt_scene-normalised units but GT
    is metric.

    Also returns the per-frame scale factors (NaN where no overlap exists).
    """
    key = "layout_depth" if "layout_depth" in preds_one else (
        "depth" if use_depth_as_layout and "depth" in preds_one else None
    )
    if key is None:
        return preds_one, []

    ld = to_np(preds_one[key])
    squeezed_last = ld.ndim == 4 and ld.shape[-1] == 1
    if squeezed_last:
        ld_sq = ld[..., 0]
    else:
        ld_sq = ld

    gt_ld = to_np(sample["layout_depths"])
    gt_dm = to_np(sample.get("layout_depth_masks"))

    S = ld_sq.shape[0]
    scales: list[float] = []
    scaled = ld_sq.copy().astype(np.float64)
    for s in range(S):
        gt_s = frame(gt_ld, s)
        valid = (frame(gt_dm, s).astype(bool) if gt_dm is not None else (gt_s > 1e-6))
        diag = summarize_depth_scale(ld_sq[s], gt_s, valid)
        sc = diag.get("median_gt_pred_scale", float("nan"))
        if isinstance(sc, (int, float)) and np.isfinite(sc) and sc > 0:
            scaled[s] = ld_sq[s] * float(sc)
            scales.append(float(sc))
        else:
            scales.append(float("nan"))

    out = dict(preds_one)
    if squeezed_last:
        out[key] = scaled[..., None].astype(ld.dtype)
    else:
        out[key] = scaled.astype(ld.dtype)
    return out, scales


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Maps the short experiment label to (config_name, use_depth_as_layout).
# `use_depth_as_layout=True` is E0's vanilla VGGT (no layout heads at all;
# use the regular depth head as a layout proxy).
EXPERIMENT_REGISTRY: dict[str, tuple[str, bool]] = {
    "E0":  ("room_envelopes/e0_vanilla_eval_only",                        True),
    "E1":  ("room_envelopes/e1_layout_depth_only_frozen",                 False),
    "E2":  ("room_envelopes/e2_layout_depth_mask_frozen",                 False),
    "E3":  ("room_envelopes/e3_layout_depth_normals_frozen",              False),
    "E4":  ("room_envelopes/e4_layout_depth_mask_normals_frozen",         False),
    "E4b": ("room_envelopes/e4b_layout_depth_mask_normals_frozen",        False),
    "E5":  ("room_envelopes/e5_layout_depth_mask_normals_unfreeze_last4", False),
    "E6":  ("room_envelopes/e6_layout_depth_mask_normals_unfreeze_last8", False),
    "E7":  ("room_envelopes/e7_all_heads_unfreeze_last4",                 False),
    "E8":  ("room_envelopes/e8_layout_depth_oca_frozen",                  False),
    "E9":  ("room_envelopes/e9_layout_depth_mask_oca_frozen",             False),
    "E10": ("room_envelopes/e10_layout_depth_mask_oca_epipolar_frozen",   False),
    "E11": ("room_envelopes/e11_layout_depth_mask_oca_unfreeze_last4",    False),
}


def _default_checkpoint_for(experiment: str) -> str | None:
    """Return the conventional best.pt path; None for E0 (no checkpoint)."""
    if experiment == "E0":
        return None
    cfg_name = EXPERIMENT_REGISTRY[experiment][0].split("/", 1)[1]
    return os.path.join(_repo_root, "training", "logs", cfg_name, "ckpts", "best.pt")


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", _repo_root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _check_backbone_keys(missing_keys: list[str]) -> None:
    """Heads-only missing is fine (E0); backbone missing is fatal."""
    head_prefixes = ("layout_depth_head", "layout_mask_head", "layout_normal_head")
    bad = [k for k in missing_keys if not k.startswith(head_prefixes)]
    if bad:
        sample = ", ".join(bad[:5])
        more = f" (+{len(bad)-5} more)" if len(bad) > 5 else ""
        raise RuntimeError(
            f"checkpoint is missing {len(bad)} non-head state_dict key(s): "
            f"{sample}{more}. Refusing to evaluate."
        )


def _assert_pred_shapes(preds_one: dict, S: int, H: int, W: int,
                        has_mask_head: bool, has_normal_head: bool,
                        use_depth_as_layout: bool) -> None:
    """Validate head output shapes (head-aware).

    Predictions arrive in ``preds_one`` (batch-dim stripped). The exact axis
    order depends on the head:
      - layout_depth         : (S, H, W, 1): channel-last
      - layout_mask_logits   : (S, 1, H, W): channel-first
      - layout_normal        : (S, 3, H, W): channel-first
    For E0 (no layout heads), the script falls back to ``preds_one["depth"]``,
    which the regular VGGT depth head emits as (S, H, W, 1).

    Head-aware policy: a head that is *enabled but absent* from the predictions
    (a cfg/checkpoint mismatch, e.g. an enabled-but-untrained head) is reported
    as a one-line warning and skipped, NOT raised. The per-concern metric code
    in ``scene_metrics`` double-gates on actual prediction-key presence, so the
    corresponding metrics are simply not emitted. A head that IS present but has
    an unexpected shape still raises, that signals a real head-layout change,
    not head absence.
    """
    depth_key = "layout_depth"
    if depth_key not in preds_one and use_depth_as_layout and "depth" in preds_one:
        depth_key = "depth"
    if depth_key not in preds_one:
        print(
            "[warn] no usable depth output in preds ('layout_depth', or 'depth' "
            "with --use-depth-as-layout for E0); depth / 3D metrics will be "
            "skipped for this checkpoint.",
            file=sys.stderr,
        )
    else:
        ld_shape = tuple(preds_one[depth_key].shape)
        if ld_shape != (S, H, W, 1):
            raise RuntimeError(
                f"{depth_key} shape {ld_shape} != expected (S={S}, H={H}, W={W}, 1). "
                f"Head output layout has changed, refusing to silently reshape."
            )

    if has_mask_head:
        if "layout_mask_logits" not in preds_one:
            print(
                "[warn] mask head enabled but preds has no 'layout_mask_logits'; "
                "mask metrics will be skipped (head/checkpoint mismatch).",
                file=sys.stderr,
            )
        else:
            ml_shape = tuple(preds_one["layout_mask_logits"].shape)
            if ml_shape != (S, 1, H, W):
                raise RuntimeError(
                    f"layout_mask_logits shape {ml_shape} != expected (S={S}, 1, H={H}, W={W})."
                )

    if has_normal_head:
        if "layout_normal" not in preds_one:
            print(
                "[warn] normal head enabled but preds has no 'layout_normal'; "
                "head normal metrics will be skipped (a depth-derived normal "
                "fallback may still run).",
                file=sys.stderr,
            )
        else:
            pn_shape = tuple(preds_one["layout_normal"].shape)
            if pn_shape != (S, 3, H, W):
                raise RuntimeError(
                    f"layout_normal shape {pn_shape} != expected (S={S}, 3, H={H}, W={W})."
                )


# ---------------------------------------------------------------------------
# JSON-safe conversion (NaN → None, numpy → python)
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Eval-manifest helpers (deterministic val/test sampling)
# ---------------------------------------------------------------------------

from manifest import (  # noqa: E402
    _load_eval_manifest,
    _find_room_envelopes_child,
    _manifest_iter_items,
    _check_manifest_split,
    _resolve_seq_index,
)


# ---------------------------------------------------------------------------
# `run` subcommand
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    import torch
    from hydra.utils import instantiate

    # --- experiment + checkpoint resolution ----------------------------------
    if args.experiment not in EXPERIMENT_REGISTRY:
        print(f"[fatal] unknown --experiment '{args.experiment}'; choices: "
              f"{list(EXPERIMENT_REGISTRY)}", file=sys.stderr)
        return 2

    config_name, use_depth_as_layout = EXPERIMENT_REGISTRY[args.experiment]
    if args.config:
        config_name = args.config  # explicit override
    if args.use_depth_as_layout is not None:
        use_depth_as_layout = args.use_depth_as_layout

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = _default_checkpoint_for(args.experiment)
    if checkpoint_path is not None and not os.path.isfile(checkpoint_path):
        print(f"[fatal] checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2
    if args.experiment != "E0" and checkpoint_path is None:
        print(f"[fatal] {args.experiment} requires --checkpoint", file=sys.stderr)
        return 2

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load model + cfg ----------------------------------------------------
    cfg, model, device, ckpt_meta = load_model_and_cfg(
        config_name, checkpoint_path, args.device,
    )
    _check_backbone_keys(ckpt_meta["missing_keys"])

    has_mask_head = bool(getattr(cfg.model, "enable_layout_mask", False))
    has_normal_head = bool(getattr(cfg.model, "enable_layout_normal", False))
    has_layout_head = bool(getattr(cfg.model, "enable_layout_depth", False))

    if (not has_layout_head) and (not use_depth_as_layout):
        print(
            f"[fatal] cfg.model.enable_layout_depth=False and --use-depth-as-layout "
            f"was not set (and not implied by the experiment registry for "
            f"{args.experiment}). Cannot evaluate depth.",
            file=sys.stderr,
        )
        return 2

    heads_enabled = {
        "layout_depth": has_layout_head or use_depth_as_layout,
        "layout_mask":  has_mask_head,
        "layout_normal": has_normal_head,
    }

    # --- dataset -------------------------------------------------------------
    split_cfg = select_split(cfg, args.split)
    inner_ds = instantiate(
        split_cfg.dataset,
        common_config=split_cfg.common_config,
        _recursive_=False,
    )
    n_unique = enable_unique_scene_mode(inner_ds)
    print(f"[data] all_unique_scenes: {n_unique} unique rooms in '{args.split}'")

    # --- optional eval manifest ---------------------------------------------
    manifest = None
    manifest_items: list[dict] = []
    manifest_mode: str | None = None  # "per_view" or "mixed"
    manifest_num_views: Optional[int] = None
    re_child = None
    scene_cam_lookup: dict[str, int] = {}
    if args.eval_manifest:
        manifest = _load_eval_manifest(args.eval_manifest)
        _check_manifest_split(manifest, args.split, args.allow_split_mismatch)
        # --num-views is optional; only used to cross-check per-view manifests.
        manifest_items, manifest_mode, manifest_num_views = _manifest_iter_items(
            manifest, args.num_views,
        )
        re_child = _find_room_envelopes_child(inner_ds)
        re_child.inside_random = False
        scene_cam_lookup = {
            seq["scene_cam"]: i for i, seq in enumerate(re_child.sequences)
        }
        print(f"[manifest] {args.eval_manifest}")
        print(f"[manifest] mode={manifest_mode} items={len(manifest_items)} "
              f"num_views={manifest_num_views} split={args.split}")

    n_total = len(manifest_items) if manifest else n_unique
    if args.max_scenes is not None:
        n_total = min(n_total, args.max_scenes)

    # --- view-count strategy -------------------------------------------------
    # Default: pinned at args.num_views (4) for cross-experiment comparability.
    # --match-training-sampling: per-scene random pick from img_nums band
    # (reproduces training-time val), for the W&B cross-check only.
    # Manifest mode ignores both: each item carries its own num_views.
    if manifest is not None:
        rng_band = None
        print(f"[views] manifest mode, per-item num_views from manifest")
    elif args.match_training_sampling:
        rng_band = list(split_cfg.common_config.img_nums)
        print(f"[views] match-training-sampling=True: random pick from "
              f"img_nums={rng_band} per scene (NOT pinned)")
    else:
        rng_band = None
        print(f"[views] pinned at --num-views={args.num_views} for every scene")

    # --- header --------------------------------------------------------------
    print("=" * 70)
    print(f"  experiment        : {args.experiment}")
    print(f"  config            : {config_name}")
    print(f"  checkpoint        : {checkpoint_path}")
    print(f"  checkpoint_epoch  : {ckpt_meta['checkpoint_epoch']}")
    print(f"  split             : {args.split}")
    print(f"  eval_space        : {args.eval_space}")
    print(f"  heads_enabled     : {heads_enabled}")
    print(f"  use_depth_as_layout: {use_depth_as_layout}")
    if manifest is not None and manifest_mode == "per_view":
        nv_display: Any = manifest_num_views
    elif manifest is not None and manifest_mode == "mixed":
        nv_display = "per-item (mixed manifest)"
    elif rng_band:
        nv_display = f"random {rng_band}"
    else:
        nv_display = args.num_views if args.num_views is not None else 4
    print(f"  num_views         : {nv_display}")
    print(f"  max_scenes        : {args.max_scenes} (n_unique={n_unique})")
    print(f"  device            : {device}")
    print("=" * 70)

    # --- optional W&B init for clean qualitative logging --------------------
    wandb_enabled = bool(getattr(args, "wandb", False))
    wandb_clean_scenes_budget = int(getattr(args, "wandb_max_scenes", 0) or 0)
    wandb_clean_scenes_logged = 0
    wandb_clouds_dir: Optional[str] = None
    if wandb_enabled:
        from types import SimpleNamespace
        logging_cfg = SimpleNamespace(
            wandb_project=getattr(args, "wandb_project", None) or "vggt-eval",
            wandb_entity=getattr(args, "wandb_entity", None),
            wandb_run_name=(getattr(args, "wandb_run_name", None)
                            or f"{args.experiment}-{args.split}"),
            wandb_mode=getattr(args, "wandb_mode", None) or "online",
        )
        wandb_logger.init_wandb(
            logging_cfg=logging_cfg,
            exp_name=f"{args.experiment}-{args.split}",
            extra_config={
                "experiment": args.experiment,
                "checkpoint": checkpoint_path,
                "checkpoint_epoch": ckpt_meta["checkpoint_epoch"],
                "config": config_name,
                "split": args.split,
                "num_views": args.num_views,
                "max_scenes": args.max_scenes,
                "use_depth_as_layout": use_depth_as_layout,
                "wandb_max_scenes": wandb_clean_scenes_budget,
                "wandb_max_points_preview": getattr(args, "wandb_max_points_preview", 50_000),
            },
        )
        if getattr(args, "wandb_save_full_pointcloud", False):
            wandb_clouds_dir = str(Path(args.output_dir) / args.split
                                   / f"{args.experiment}_clouds")
            os.makedirs(wandb_clouds_dir, exist_ok=True)
            print(f"[wandb] full-resolution PLYs → {wandb_clouds_dir}")
        print(
            f"[wandb] clean qualitative logging enabled, "
            f"project={logging_cfg.wandb_project!r} "
            f"mode={logging_cfg.wandb_mode!r} "
            f"max_scenes={wandb_clean_scenes_budget} "
            f"max_points_preview={getattr(args, 'wandb_max_points_preview', 50_000)}"
        )

    # --- per-scene loop ------------------------------------------------------
    import random as _random_mod

    per_scene_records: list[dict] = []
    skipped: list[int] = []
    sanity_done = False
    n_views_total = 0
    n_eval_scenes = 0
    t_start = time.time()

    for i in range(n_total):
        np.random.seed(args.seed + i)
        _random_mod.seed(args.seed + i)

        if manifest is not None:
            item = manifest_items[i]
            if manifest_mode == "per_view":
                this_views = int(manifest_num_views)
            else:
                this_views = int(item["num_views"])
            try:
                seq_index_resolved = _resolve_seq_index(item, scene_cam_lookup)
                sample = re_child.get_data(
                    seq_index=seq_index_resolved,
                    img_per_seq=this_views,
                    ids=list(item["ids"]),
                    aspect_ratio=1.0,
                )
            except Exception as e:
                skipped.append(i)
                print(f"[manifest item {i} "
                      f"scene_cam={item.get('scene_cam')} "
                      f"seq_index={item.get('seq_index')}] dataset error ({e}); skipping")
                continue
        else:
            if rng_band is not None:
                lo, hi = int(rng_band[0]), int(rng_band[1])
                this_views = int(np.random.randint(lo, hi + 1))
            else:
                this_views = int(args.num_views if args.num_views is not None else 4)

            try:
                sample = inner_ds[(i, this_views, 1.0)]
            except Exception as e:
                skipped.append(i)
                print(f"[scene {i}] dataset error ({e}); skipping")
                continue

        imgs = sample.get("images")
        if imgs is None:
            skipped.append(i)
            print(f"[scene {i}] no images; skipping")
            continue

        if isinstance(imgs, list) or (isinstance(imgs, np.ndarray) and imgs.dtype == np.uint8):
            imgs_np = np.asarray(imgs)
            imgs_t = torch.tensor(imgs_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        else:
            imgs_t = imgs if hasattr(imgs, "to") else torch.as_tensor(imgs)
            if imgs_t.dtype != torch.float32:
                imgs_t = imgs_t.float()
            if imgs_t.max() > 1.5:
                imgs_t = imgs_t / 255.0
            if imgs_t.ndim == 4 and imgs_t.shape[-1] == 3:
                imgs_t = imgs_t.permute(0, 3, 1, 2)
        imgs_t = imgs_t.unsqueeze(0).to(device)  # (1, S, 3, H, W)

        K_t, E_t = cameras_from_sample(sample, device=device)
        with torch.no_grad():
            preds = forward_model(model, imgs_t, intrinsics=K_t, extrinsics=E_t)

        preds_one: dict = {}
        for k, v in preds.items():
            if hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == 1:
                preds_one[k] = v[0]
            else:
                preds_one[k] = v

        S = imgs_t.shape[1]
        H, W = int(imgs_t.shape[-2]), int(imgs_t.shape[-1])

        # --- clean W&B qualitative logging (separate from the legacy ----
        # ---  overlay/error path; never touches metric calculations) ----
        if wandb_enabled and (
            wandb_clean_scenes_budget <= 0
            or wandb_clean_scenes_logged < wandb_clean_scenes_budget
        ):
            try:
                scene_log = dict(sample)
                # Drop the (1,S,3,H,W) batched image tensor onto the scene so
                # the clean logger can sample RGB for the point cloud.
                scene_log["images"] = imgs_t[0].detach().cpu()
                wandb_clean_logger.log_clean_visuals(
                    batch=scene_log,
                    predictions=preds_one,
                    phase="eval",
                    # Scene index is 0,1,2,... within a standalone eval run,
                    # safe to use as the monotonic W&B run step.
                    wandb_step=int(i),
                    epoch=ckpt_meta.get("checkpoint_epoch"),
                    scene_index=int(i),
                    max_samples=1,
                    view_indices=[0],
                    log_2d=bool(getattr(args, "wandb_log_2d", True)),
                    log_3d=bool(getattr(args, "wandb_log_3d", True)),
                    log_gt_3d=bool(getattr(args, "wandb_log_gt_3d", True)),
                    max_points_preview=int(getattr(args, "wandb_max_points_preview", 50_000)),
                    save_full_pointcloud_dir=wandb_clouds_dir,
                    use_depth_as_layout=bool(use_depth_as_layout),
                    tag=str(sample.get("seq_name", f"scene_{i:04d}")),
                )
                wandb_clean_scenes_logged += 1
            except Exception as exc:
                print(f"[wandb] clean log failed for scene {i}: {exc}")

        # One-time sanity checks on the first successful scene.
        if not sanity_done:
            _assert_pred_shapes(preds_one, S, H, W,
                                has_mask_head=has_mask_head,
                                has_normal_head=has_normal_head,
                                use_depth_as_layout=use_depth_as_layout)
            lm = sample.get("layout_masks")
            if lm is not None and not (to_np(lm) > 0.5).any():
                print(f"[warn] first scene has zero visible-structure pixels "
                      f"(layout_masks all <= 0.5). Possible inverted mask convention.")
            sanity_done = True

        # --- compute 2D metrics for each enabled eval space ----------------
        record: dict = {
            "scene_idx": int(i),
            "seq_name": str(sample.get("seq_name", f"scene_{i:04d}")),
            "n_views": int(S),
        }

        if args.eval_space in ("metric", "both"):
            try:
                # Standard monocular-depth eval protocol: per-frame median
                # scaling of pred → match GT scale. AbsRel survives this
                # (it's scale-invariant) and RMSE / LogRMSE become meaningful
                # again. Track scales for diagnostics.
                scaled_preds, frame_scales = _scale_align_preds_per_frame(
                    preds_one, sample, use_depth_as_layout,
                )
                m2d_metric = compute_2d_metrics_for_scene(
                    sample, scaled_preds,
                    use_depth_as_layout=use_depth_as_layout,
                    has_mask_head=has_mask_head,
                    has_normal_head=has_normal_head,
                )
                for k, v in m2d_metric.items():
                    record[f"metric_{k}"] = v
                if frame_scales:
                    valid_scales = [s for s in frame_scales if np.isfinite(s)]
                    record["metric_scale_factor_median"] = (
                        float(np.median(valid_scales)) if valid_scales else float("nan")
                    )
            except KeyError as e:
                print(f"[scene {i}] metric 2D skipped: {e}")

        if args.eval_space in ("vggt_scene", "both"):
            try:
                normalized_sample, norm_info = normalize_sample_vggt_scene(sample)
                record["vggt_scene_scale"] = float(norm_info["vggt_scene_scale"])
                m2d_vggt = compute_2d_metrics_for_scene(
                    normalized_sample, preds_one,
                    use_depth_as_layout=use_depth_as_layout,
                    has_mask_head=has_mask_head,
                    has_normal_head=has_normal_head,
                )
                for k, v in m2d_vggt.items():
                    record[f"vggt_scene_{k}"] = v
            except Exception as e:
                print(f"[scene {i}] vggt_scene normalization/2D failed: {e}")
                record["vggt_scene_error"] = str(e)

        # Per-frame median pred/gt ratio in vggt_scene space, fastest
        # canary for an eval-space wiring bug (should be ~1.0 if correct).
        try:
            pred_for_ratio = preds_one.get("layout_depth")
            if pred_for_ratio is None and use_depth_as_layout:
                pred_for_ratio = preds_one.get("depth")
            if pred_for_ratio is not None:
                ld = to_np(pred_for_ratio)
                if ld.ndim == 4 and ld.shape[-1] == 1:
                    ld = ld[..., 0]
                ns = normalized_sample if args.eval_space in ("vggt_scene", "both") else sample
                gt_ld = to_np(ns["layout_depths"])
                gt_dm = to_np(ns.get("layout_depth_masks"))
                gt_dm_b = gt_dm.astype(bool) if gt_dm is not None else None
                diag = summarize_depth_scale(ld, gt_ld, gt_dm_b)
                record["median_pred_gt_ratio_vggt_scene"] = diag["median_pred_gt_ratio"]
        except Exception:
            pass

        per_scene_records.append(record)
        n_views_total += S
        n_eval_scenes += 1

        if n_eval_scenes % 10 == 0 or i + 1 == n_total:
            print(f"  [{n_eval_scenes}/{n_total}] scenes done, "
                  f"{time.time() - t_start:.1f}s elapsed")

    # --- aggregate -----------------------------------------------------------
    agg = aggregate(per_scene_records)

    def _nest(prefix: str) -> dict:
        nested = {"depth": {}, "mask": {}, "normals": {}}
        for full_key, v in agg.items():
            if not full_key.startswith(prefix):
                continue
            k = full_key[len(prefix):]
            if k.startswith("mask_"):
                nested["mask"][k[len("mask_"):]] = v
            elif k.startswith("normal_"):
                nested["normals"][k[len("normal_"):]] = v
            elif k in ("scene_idx", "n_views", "depth_used",
                       "vggt_scene_scale", "median_pred_gt_ratio_vggt_scene"):
                continue
            elif k.endswith("_all") or k.endswith("_visible") or k.endswith("_occluded") \
                    or k in ("absrel", "rmse", "log_rmse", "delta1", "delta2", "delta3", "silog"):
                nested["depth"][k] = v
        return {section: items for section, items in nested.items() if items}

    metrics_block: dict = {}
    if args.eval_space in ("metric", "both"):
        metrics_block["metric"] = _nest("metric_")
    if args.eval_space in ("vggt_scene", "both"):
        metrics_block["vggt_scene"] = _nest("vggt_scene_")

    diagnostics = {
        "scene_scale_mean":   agg.get("vggt_scene_scale"),
        "median_pred_gt_ratio_vggt_scene": agg.get("median_pred_gt_ratio_vggt_scene"),
        "metric_scale_factor_median":      agg.get("metric_scale_factor_median"),
    }
    metrics_block["diagnostics"] = {k: v for k, v in diagnostics.items() if v is not None}

    meta = {
        "experiment": args.experiment,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": ckpt_meta["checkpoint_epoch"],
        "config": config_name,
        "git_commit": _git_commit_sha(),
        "split": args.split,
        "num_scenes_evaluated": n_eval_scenes,
        "num_scenes_skipped": len(skipped),
        "skipped_scene_ids": skipped[:50],
        "num_frames": n_views_total,
        "num_views_per_scene": (
            f"random {rng_band[0]}-{rng_band[1]}" if rng_band else
            (manifest_num_views if manifest_mode == "per_view"
             else ("per-item" if manifest_mode == "mixed"
                   else int(args.num_views if args.num_views is not None else 4)))
        ),
        "seed": int(args.seed),
        "eval_space": args.eval_space,
        "heads_enabled": heads_enabled,
        "use_depth_as_layout": bool(use_depth_as_layout),
        "missing_state_keys": int(ckpt_meta["n_missing_keys"]),
        "unexpected_state_keys": int(ckpt_meta["n_unexpected_keys"]),
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eval_manifest": args.eval_manifest,
        "eval_manifest_mode": manifest_mode,
        "eval_manifest_meta": (manifest or {}).get("meta") if manifest else None,
    }

    out_payload = {
        "meta": _json_safe(meta),
        "metrics": _json_safe(metrics_block),
        "per_scene": _json_safe(per_scene_records),
    }

    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.experiment}.json"
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"saved: {out_path}")

    # Close the W&B run (if any). Safe no-op when wandb wasn't initialised.
    if wandb_enabled:
        wandb_logger.finish_wandb()
        print(f"[wandb] clean qualitative scenes logged: {wandb_clean_scenes_logged}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="subcommand", required=True)

    # ---- run ----
    pr = sub.add_parser("run", help="Evaluate one experiment.")
    pr.add_argument("--experiment", required=True,
                    choices=list(EXPERIMENT_REGISTRY.keys()))
    pr.add_argument("--checkpoint", default=None,
                    help="Override checkpoint path. Default = "
                         "training/logs/<config_basename>/ckpts/best.pt.")
    pr.add_argument("--config", default=None,
                    help="Override the Hydra config (rare).")
    pr.add_argument("--split", default="test", choices=("train", "val", "test"))
    pr.add_argument("--max-scenes", dest="max_scenes", type=int, default=None,
                    help="Cap on scenes evaluated (else full unique split).")
    pr.add_argument("--num-views", dest="num_views", type=int, default=None,
                    help="Pinned views per scene (default 4 when no manifest is given). "
                         "When --eval-manifest is a per-view manifest, this is "
                         "optional; if provided it must match the manifest's num_views.")
    pr.add_argument("--match-training-sampling", action="store_true",
                    help="For the W&B cross-check: per-scene random pick from "
                         "img_nums band instead of pinned --num-views.")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--eval-space", dest="eval_space",
                    default="both", choices=("metric", "vggt_scene", "both"))
    pr.add_argument("--use-depth-as-layout", dest="use_depth_as_layout",
                    action="store_true", default=None,
                    help="Use 'depth' as a layout-depth proxy (auto-on for E0).")
    pr.add_argument("--output-dir", dest="output_dir", default="outputs/eval_2d")
    pr.add_argument("--device", default=None)
    pr.add_argument("--eval-manifest", dest="eval_manifest", default=None,
                    help="Optional path to a deterministic eval manifest JSON "
                         "(see evaluations/src/manifests/build_room_envelopes_eval_manifest.py).")
    pr.add_argument("--allow-split-mismatch", dest="allow_split_mismatch",
                    action="store_true",
                    help="Allow manifest meta.split to differ from --split.")

    # ---- clean W&B qualitative logging ----
    pr.add_argument("--wandb", action="store_true",
                    help="Enable clean W&B qualitative logging (separate from "
                         "the legacy overlay/error visuals).")
    pr.add_argument("--wandb-project", dest="wandb_project", default="vggt-clean-eval")
    pr.add_argument("--wandb-entity", dest="wandb_entity", default=None)
    pr.add_argument("--wandb-run-name", dest="wandb_run_name", default=None)
    pr.add_argument("--wandb-mode", dest="wandb_mode", default="online",
                    choices=("online", "offline", "disabled"))
    pr.add_argument("--wandb-log-2d", dest="wandb_log_2d", action="store_true",
                    default=True, help="Log clean 2D image panels (default on).")
    pr.add_argument("--no-wandb-log-2d", dest="wandb_log_2d",
                    action="store_false")
    pr.add_argument("--wandb-log-3d", dest="wandb_log_3d", action="store_true",
                    default=True, help="Log W&B Object3D point clouds (default on).")
    pr.add_argument("--no-wandb-log-3d", dest="wandb_log_3d",
                    action="store_false")
    pr.add_argument("--wandb-log-gt-3d", dest="wandb_log_gt_3d",
                    action="store_true", default=True,
                    help="Also log GT 3D reconstruction when GT layout depth "
                         "is available (default on).")
    pr.add_argument("--no-wandb-log-gt-3d", dest="wandb_log_gt_3d",
                    action="store_false")
    pr.add_argument("--wandb-max-scenes", dest="wandb_max_scenes",
                    type=int, default=8,
                    help="Cap on number of scenes for which to log clean "
                         "visuals. 0 = no cap.")
    pr.add_argument("--wandb-max-points-preview",
                    dest="wandb_max_points_preview", type=int, default=50_000,
                    help="Subsample target for W&B Object3D preview.")
    pr.add_argument("--wandb-save-full-pointcloud",
                    dest="wandb_save_full_pointcloud", action="store_true",
                    default=False,
                    help="Also write full-resolution PLYs under "
                         "<output-dir>/<split>/<experiment>_clouds/.")
    pr.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
