"""Mask metric for 2D layout-mask eval.

Wraps the canonical ``compute_mask_metrics`` and owns the logits→probability
preprocessing used by the per-scene orchestrator.
"""
from __future__ import annotations

import numpy as np

from eval_metrics import compute_mask_metrics

__all__ = ["compute_mask_metrics", "mask_prob_from_logits"]


def mask_prob_from_logits(logits) -> np.ndarray:
    """Sigmoid of layout-mask logits, squeezing a singleton channel axis.

    Accepts ``(S, 1, H, W)`` or ``(S, H, W)`` logits and returns ``(S, H, W)``
    probabilities in ``[0, 1]``.
    """
    ml = np.asarray(logits)
    if ml.ndim == 4 and ml.shape[1] == 1:
        ml = ml[:, 0]
    return 1.0 / (1.0 + np.exp(-ml))
