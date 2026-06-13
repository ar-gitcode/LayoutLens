"""Depth metric for 2D layout-depth eval.

Thin wrapper around the canonical implementation in
``training/eval_metrics.py`` so the per-scene orchestrator and the
post-processing runner share one source of truth.
"""
from __future__ import annotations

from eval_metrics import compute_depth_metrics_with_splits

__all__ = ["compute_depth_metrics_with_splits"]
