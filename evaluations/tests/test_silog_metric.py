#!/usr/bin/env python3
"""Tests for the SILog (scale-invariant log error) depth metric.

Pins the KITTI-convention formula ``silog = 100 * sqrt(Var(ln pred - ln gt))``,
its non-negativity, and its defining scale-invariance property, and guards
against the previous malformed ``Var(d) - 0.85*mean(d)**2`` form that could go
negative / invert rankings.

Run directly or under pytest. Pure numpy; no model/dataset required.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))  # evaluations/tests -> repo root
for _p in (os.path.join(_REPO, "training"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_metrics import compute_depth_metrics  # noqa: E402


def _silog(pred, gt, mask=None):
    return compute_depth_metrics(np.asarray(pred, float), np.asarray(gt, float),
                                 mask)["silog"]


def _canonical(pred, gt):
    p = np.clip(np.asarray(pred, float), 1e-6, None)
    g = np.asarray(gt, float)
    d = np.log(p) - np.log(g)
    return 100.0 * math.sqrt(max(float(d.var()), 0.0))


# ---------------------------------------------------------------------------
# 1. formula is exactly KITTI 100*sqrt(Var(log-error))
# ---------------------------------------------------------------------------

def test_silog_matches_kitti_formula():
    rng = np.random.RandomState(0)
    for _ in range(20):
        n = int(rng.randint(2, 200))
        gt = rng.uniform(0.5, 10.0, size=n)
        pred = gt * rng.uniform(0.5, 2.0, size=n)  # multiplicative noise
        got = _silog(pred, gt)
        exp = _canonical(pred, gt)
        assert abs(got - exp) < 1e-9, (got, exp)


# ---------------------------------------------------------------------------
# 2. non-negative everywhere, including the pathological case that broke the
#    old formula (large systematic log-offset, tiny residual variance)
# ---------------------------------------------------------------------------

def test_silog_non_negative_random_and_pathological():
    rng = np.random.RandomState(1)
    for _ in range(2000):
        n = int(rng.randint(2, 100))
        gt = rng.uniform(0.3, 12.0, size=n)
        pred = gt * np.exp(rng.normal(0.0, rng.uniform(0.0, 1.0), size=n))
        s = _silog(pred, gt)
        assert s >= 0.0, s

    # near-constant large log offset: old formula returned ~ -0.85 here.
    gt = np.full(20, 2.0)
    pred = gt * math.exp(1.0) + np.linspace(0, 1e-5, 20)  # d ~ 1.0, ~zero variance
    s = _silog(pred, gt)
    assert s >= 0.0
    assert s < 1.0  # variance ~ 0 -> silog ~ 0 (not negative, not large)


# ---------------------------------------------------------------------------
# 3. scale invariance: multiplying ALL predictions by a constant must not
#    change SILog (the property that defines it)
# ---------------------------------------------------------------------------

def test_silog_scale_invariant():
    rng = np.random.RandomState(2)
    gt = rng.uniform(0.5, 8.0, size=500)
    pred = gt * np.exp(rng.normal(0, 0.2, size=500))
    base = _silog(pred, gt)
    for c in (0.1, 0.5, 2.0, 10.0):
        scaled = _silog(pred * c, gt)
        assert abs(scaled - base) < 1e-7, (c, scaled, base)


# ---------------------------------------------------------------------------
# 4. perfect prediction -> 0 ; and a concrete hand value
# ---------------------------------------------------------------------------

def test_silog_known_values():
    gt = np.array([1.0, 2.0, 4.0, 8.0])
    assert _silog(gt.copy(), gt) == 0.0  # exact match -> Var(0) -> 0

    # Two-point case: d = [ln(1.5), ln(1/1.5)] = [+a, -a], mean 0, var a^2.
    a = math.log(1.5)
    pred = np.array([1.5, 1.0]); gt2 = np.array([1.0, 1.5])
    expected = 100.0 * a  # 100*sqrt(var)=100*sqrt(a^2)=100*a
    assert abs(_silog(pred, gt2) - expected) < 1e-9


# ---------------------------------------------------------------------------
# 5. regression guard: must NOT equal the old malformed formula
# ---------------------------------------------------------------------------

def test_silog_not_old_buggy_formula():
    gt = np.full(20, 2.0)
    pred = gt * math.exp(1.0)  # d ~ const 1.0
    p = np.clip(pred, 1e-6, None); d = np.log(p) - np.log(gt)
    old = float(d.var() - 0.85 * (d.mean() ** 2))   # the removed formula
    assert old < 0.0                                 # old was negative here...
    assert _silog(pred, gt) >= 0.0                   # ...new must not be


# ---------------------------------------------------------------------------
# 6. empty valid set still yields NaN (unchanged contract)
# ---------------------------------------------------------------------------

def test_silog_nan_when_no_valid():
    gt = np.zeros(10)        # gt <= EPS -> no valid pixels
    pred = np.ones(10)
    assert math.isnan(_silog(pred, gt))


def _main() -> int:
    tests = [
        ("formula == KITTI 100*sqrt(Var(d))", test_silog_matches_kitti_formula),
        ("non-negative (random + pathological)", test_silog_non_negative_random_and_pathological),
        ("scale invariant", test_silog_scale_invariant),
        ("known values", test_silog_known_values),
        ("regression vs old buggy formula", test_silog_not_old_buggy_formula),
        ("NaN when no valid pixels", test_silog_nan_when_no_valid),
    ]
    fails = 0
    for name, fn in tests:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1; print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(fails)+' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_main())
