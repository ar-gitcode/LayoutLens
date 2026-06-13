"""Shared tensor scaffolding for the eval runners: image→model-input
conversion and batch-dim stripping of per-frame predictions.
"""
from __future__ import annotations

import numpy as np


def _to_image_tensor(sample: dict, device) -> "torch.Tensor":
    import torch
    imgs = sample.get("images")
    if imgs is None:
        raise RuntimeError("sample has no 'images'")
    if isinstance(imgs, list) or (isinstance(imgs, np.ndarray) and imgs.dtype == np.uint8):
        imgs_np = np.asarray(imgs)
        t = torch.tensor(imgs_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    else:
        t = imgs if hasattr(imgs, "to") else torch.as_tensor(imgs)
        if t.dtype != torch.float32:
            t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
        if t.ndim == 4 and t.shape[-1] == 3:
            t = t.permute(0, 3, 1, 2)
    return t.unsqueeze(0).to(device)  # (1, S, 3, H, W)


def _strip_batch_dim(preds: dict) -> dict:
    """Drop a leading singleton batch dim from each per-frame prediction tensor."""
    out: dict = {}
    for k, v in preds.items():
        if hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out
