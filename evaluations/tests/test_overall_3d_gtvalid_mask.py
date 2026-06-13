#!/usr/bin/env python3
"""Tests for the GT-validity masking of the overall 3D pred→GT metrics.

Validates the change that makes overall 3D Chamfer/F-score consistent with the
2D metrics and seen/unseen splits: predicted points whose *source pixel* has
undefined GT layout depth are dropped from the **pred→GT** side
(accuracy / precision) only, the **gt→pred** side (completeness / recall) keeps
using the full predicted cloud.

Run directly (``python evaluations/tests/test_overall_3d_gtvalid_mask.py``) or
under pytest. No model / dataset / checkpoint required, the integration test
builds a synthetic scene through the *real* cloud-builder + chamfer code path.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

# --- path bootstrap (no installed package; mirror the eval scripts) ----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))  # evaluations/tests -> repo root
for _p in (
    os.path.join(_REPO, "evaluations", "src", "3d"),
    os.path.join(_REPO, "evaluations", "src", "common"),
    os.path.join(_REPO, "training"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics_chamfer import chamfer_and_fscore  # noqa: E402
from pointcloud import (  # noqa: E402
    build_scene_pointcloud_from_batch,
    pred_cloud_gtvalid_mask,
    sample_pointcloud,
    sample_pointcloud_with_companion,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _close(a, b, tol=1e-9) -> bool:
    """Scalar equality that treats NaN == NaN as True."""
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    return abs(fa - fb) <= tol + 1e-9 * max(abs(fa), abs(fb))


def _dicts_close(d_old: dict, d_new: dict) -> list[str]:
    """Return list of keys whose values differ (NaN-aware)."""
    diffs = []
    assert set(d_old) == set(d_new), (set(d_old) ^ set(d_new))
    for k in d_old:
        if not _close(d_old[k], d_new[k]):
            diffs.append(f"{k}: {d_old[k]} != {d_new[k]}")
    return diffs


# ---------------------------------------------------------------------------
# 1. core fix: GT-undefined-sourced pred points excluded from accuracy/precision
# ---------------------------------------------------------------------------

def test_pred_acc_mask_excludes_gtundefined_from_accuracy_precision():
    # GT cloud has 2 points; the predicted cloud has those 2 (near-coincident)
    # PLUS one far "hallucination" sourced from a GT-undefined pixel.
    gt = np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32)
    pred = np.array([[0.0, 0, 0.01],
                     [1.0, 0, 0.01],
                     [10.0, 10, 10]], dtype=np.float32)
    mask = np.array([True, True, False])  # 3rd pred from a GT-undefined pixel
    thr = (0.10,)

    old = chamfer_and_fscore(pred, gt, thresholds=thr)                 # unmasked
    new = chamfer_and_fscore(pred, gt, thresholds=thr, pred_acc_mask=mask)

    # The far point penalises the unmasked accuracy/precision...
    assert old["accuracy_mean"] > new["accuracy_mean"]
    assert old["precision_0.10"] < new["precision_0.10"]
    # ...and is removed by the mask: only the two near points remain.
    assert _close(new["accuracy_mean"], 0.01, tol=1e-4)
    assert _close(new["precision_0.10"], 1.0)
    # Completeness / recall (gt→pred) are computed from the FULL pred cloud in
    # BOTH cases, so they must be identical.
    assert _close(old["completeness_mean"], new["completeness_mean"])
    assert _close(old["recall_0.10"], new["recall_0.10"])
    return old, new


# ---------------------------------------------------------------------------
# 2. identity: all-valid mask (and None) reproduce historical numbers exactly
# ---------------------------------------------------------------------------

def test_identity_when_all_pixels_valid():
    rng = np.random.RandomState(0)
    gt = rng.rand(80, 3).astype(np.float32)
    pred = rng.rand(95, 3).astype(np.float32)

    base = chamfer_and_fscore(pred, gt)
    all_true = chamfer_and_fscore(pred, gt, pred_acc_mask=np.ones(len(pred), bool))
    explicit_none = chamfer_and_fscore(pred, gt, pred_acc_mask=None)

    assert not _dicts_close(base, all_true), _dicts_close(base, all_true)
    assert not _dicts_close(base, explicit_none), _dicts_close(base, explicit_none)


# ---------------------------------------------------------------------------
# 3. co-sampling helper keeps points byte-identical and the companion aligned
# ---------------------------------------------------------------------------

def test_sample_companion_matches_sample_pointcloud():
    pts = np.random.RandomState(2).rand(1000, 3).astype(np.float32)
    comp = np.arange(1000)

    # Sub-sampling branch (N > max_points): identical RNG → identical points.
    a = sample_pointcloud(pts, 100, seed=0xC1)
    b, c = sample_pointcloud_with_companion(pts, comp, 100, seed=0xC1)
    assert np.array_equal(a, b)
    assert len(c) == 100
    # comp = arange, so comp[idx] == idx and pts[comp[idx]] == sampled points.
    assert np.array_equal(pts[c], b)

    # No-op branch (N <= max_points): both returned unchanged.
    a2 = sample_pointcloud(pts, 5000, seed=7)
    b2, c2 = sample_pointcloud_with_companion(pts, comp, 5000, seed=7)
    assert np.array_equal(a2, b2)
    assert np.array_equal(c2, comp)


# ---------------------------------------------------------------------------
# 4. fully-masked query: accuracy NaN / precision 0, completeness untouched
# ---------------------------------------------------------------------------

def test_empty_acc_query_preserves_completeness():
    gt = np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32)
    pred = np.array([[0.0, 0, 0.01], [1.0, 0, 0.01]], dtype=np.float32)
    mask = np.zeros(len(pred), bool)  # no GT-valid-sourced pred points

    out = chamfer_and_fscore(pred, gt, thresholds=(0.10,), pred_acc_mask=mask)
    base = chamfer_and_fscore(pred, gt, thresholds=(0.10,))

    assert math.isnan(out["accuracy_mean"])
    assert _close(out["precision_0.10"], 0.0)
    # gt→pred side is unaffected.
    assert _close(out["completeness_mean"], base["completeness_mean"])
    assert _close(out["recall_0.10"], base["recall_0.10"])


# ---------------------------------------------------------------------------
# 5. length guard
# ---------------------------------------------------------------------------

def test_mask_length_mismatch_raises():
    gt = np.random.RandomState(1).rand(10, 3).astype(np.float32)
    pred = np.random.RandomState(2).rand(12, 3).astype(np.float32)
    try:
        chamfer_and_fscore(pred, gt, pred_acc_mask=np.ones(5, bool))
    except ValueError as e:
        assert "pred_acc_mask length" in str(e)
        return
    raise AssertionError("expected ValueError on mask/pred length mismatch")


# ---------------------------------------------------------------------------
# 6. integration: real cloud builder + mask alignment for the window/door case
# ---------------------------------------------------------------------------

def _synthetic_scene(gt_valid_cols: int, H: int = 8, W: int = 8):
    """1-view scene: pred depth valid everywhere; GT layout depth valid only in
    columns ``[0, gt_valid_cols)`` (the rest are 'undefined', depth 0)."""
    K = np.array([[float(W), 0, W / 2.0],
                  [0, float(H), H / 2.0],
                  [0, 0, 1.0]], dtype=np.float32)
    E = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]], dtype=np.float32)  # identity w2c (3,4)

    pred_depth = np.full((1, H, W), 2.0, dtype=np.float32)
    gt_depth = np.zeros((1, H, W), dtype=np.float32)
    gt_depth[0, :, :gt_valid_cols] = 2.0
    gt_mask = (gt_depth > 1e-6)  # layout_depth_masks == (layout_depth > 1e-6)

    batch = {
        "layout_depths": gt_depth,
        "layout_depth_masks": gt_mask,
        "intrinsics": K[None],
        "extrinsics": E[None],
    }
    preds = {"layout_depth": pred_depth}
    return batch, preds, pred_depth, gt_mask


def test_integration_window_region_not_penalised():
    H = W = 8
    batch, preds, pred_depth, gt_mask = _synthetic_scene(gt_valid_cols=4, H=H, W=W)

    gt_cloud = build_scene_pointcloud_from_batch(batch, preds, mode="gt")
    pred_cloud = build_scene_pointcloud_from_batch(batch, preds, mode="pred")

    # GT cloud = valid columns only; pred cloud = the full image.
    assert len(gt_cloud) == H * 4
    assert len(pred_cloud) == H * W

    # gt_valid_S = (gt_ld > 1e-6) AND layout_depth_masks (here identical).
    gt_valid_S = (np.asarray(batch["layout_depths"]) > 1e-6) & gt_mask
    mask = pred_cloud_gtvalid_mask(pred_depth, gt_valid_S, keep=None)

    # Mask is row-aligned with the predicted cloud and selects the GT-valid half.
    assert mask is not None
    assert len(mask) == len(pred_cloud)
    assert int(mask.sum()) == H * 4

    unmasked = chamfer_and_fscore(pred_cloud, gt_cloud, thresholds=(0.10,))
    masked = chamfer_and_fscore(pred_cloud, gt_cloud, thresholds=(0.10,),
                                pred_acc_mask=mask)

    # The window/door (undefined) half no longer penalises accuracy/precision.
    assert masked["accuracy_mean"] < unmasked["accuracy_mean"]
    assert masked["accuracy_mean"] < 1e-5          # GT-valid preds coincide w/ GT
    assert masked["precision_0.10"] >= unmasked["precision_0.10"]
    # Completeness / recall come from the full pred cloud → unchanged.
    assert _close(masked["completeness_mean"], unmasked["completeness_mean"])
    assert _close(masked["recall_0.10"], unmasked["recall_0.10"])
    return unmasked, masked


def test_integration_all_valid_is_identity():
    H = W = 8
    batch, preds, pred_depth, gt_mask = _synthetic_scene(gt_valid_cols=W, H=H, W=W)

    gt_cloud = build_scene_pointcloud_from_batch(batch, preds, mode="gt")
    pred_cloud = build_scene_pointcloud_from_batch(batch, preds, mode="pred")
    assert len(gt_cloud) == len(pred_cloud) == H * W

    gt_valid_S = (np.asarray(batch["layout_depths"]) > 1e-6) & gt_mask
    mask = pred_cloud_gtvalid_mask(pred_depth, gt_valid_S, keep=None)
    assert int(mask.sum()) == H * W  # everything valid

    unmasked = chamfer_and_fscore(pred_cloud, gt_cloud)
    masked = chamfer_and_fscore(pred_cloud, gt_cloud, pred_acc_mask=mask)
    diffs = _dicts_close(unmasked, masked)
    assert not diffs, diffs


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [
        ("core: exclude GT-undefined from acc/precision",
         test_pred_acc_mask_excludes_gtundefined_from_accuracy_precision),
        ("identity: all-valid mask == unmasked",
         test_identity_when_all_pixels_valid),
        ("co-sampling identity", test_sample_companion_matches_sample_pointcloud),
        ("empty acc query preserves completeness",
         test_empty_acc_query_preserves_completeness),
        ("mask length mismatch raises", test_mask_length_mismatch_raises),
        ("integration: window region not penalised",
         test_integration_window_region_not_penalised),
        ("integration: all-valid is identity",
         test_integration_all_valid_is_identity),
    ]
    failures = 0
    reports = {}
    for name, fn in tests:
        try:
            out = fn()
            print(f"  PASS  {name}")
            if isinstance(out, tuple):
                reports[name] = out
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")

    # Concrete before/after numbers for the report.
    if "core: exclude GT-undefined from acc/precision" in reports:
        old, new = reports["core: exclude GT-undefined from acc/precision"]
        print("\n  [core synthetic] old(unmasked) vs new(masked):")
        for k in ("accuracy_mean", "precision_0.10", "completeness_mean", "recall_0.10",
                  "fscore_0.10"):
            print(f"    {k:18s}  old={old[k]:.6f}  new={new[k]:.6f}")
    if "integration: window region not penalised" in reports:
        u, m = reports["integration: window region not penalised"]
        print("\n  [integration window scene] old(unmasked) vs new(masked):")
        for k in ("accuracy_mean", "precision_0.10", "completeness_mean", "recall_0.10",
                  "fscore_0.10"):
            print(f"    {k:18s}  old={u[k]:.6f}  new={m[k]:.6f}")

    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
