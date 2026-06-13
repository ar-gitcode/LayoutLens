"""Eval-time helpers for OCA-enabled checkpoints.

Eval scripts call ``model(images=...)``. When a checkpoint has OCA enabled
(``model.enable_oca: True``), the layout-depth path consumes
``intrinsics``/``extrinsics`` to compute the epipolar bias. If those kwargs
are not passed, the bias path silently bypasses geometry, making
E10/E11-style evals invalid.

This module provides three small helpers, used by every eval script in
``evaluations/src/``:

    model_has_oca(model)                       -> bool
    cameras_from_sample(sample, device=None)   -> (K_t, E_t) or (None, None)
    forward_model(model, images, intrinsics=..., extrinsics=...) -> dict

``forward_model`` is the call site you should drop into eval scripts in
place of ``model(images=imgs_t)``. For non-OCA checkpoints it preserves the
old behaviour exactly: cameras are not passed even if available.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def model_has_oca(model: torch.nn.Module) -> bool:
    """Return True if the (possibly DDP-wrapped) model has an OCA module.

    Mirrors the trainer's check so behaviour is identical at train and eval.
    """
    module = model.module if hasattr(model, "module") else model
    return getattr(module, "oca", None) is not None


# Canonical head name -> (model attribute, cfg ``enable_*`` flag).
# VGGT sets each attribute to a real nn.Module iff its enable flag was True,
# and to None otherwise (vggt/models/vggt.py). The camera head is the single
# source of BOTH pose and intrinsics (decoded from ``pose_enc``); there is no
# separate intrinsics head.
_HEAD_SPECS = {
    "depth":        ("depth_head",        "enable_depth"),
    "layout_depth": ("layout_depth_head", "enable_layout_depth"),
    "mask":         ("mask_head",         "enable_layout_mask"),
    "normal":       ("normal_head",       "enable_layout_normal"),
    "camera":       ("camera_head",       "enable_camera"),
    "point":        ("point_head",        "enable_point"),
    "track":        ("track_head",        "enable_track"),
    "seg":          ("seg_head",          "enable_seg"),
}


def detect_heads(model: Optional[torch.nn.Module] = None, cfg=None) -> dict:
    """Return ``{head_name: bool}`` for every prediction head the model has.

    Detection prefers the live model object, DDP-unwrapped attribute presence,
    which reflects what was actually built/loaded, and falls back to the Hydra
    ``cfg.model.enable_*`` flags when no model is supplied (or for heads the
    model object does not expose). For VGGT the two are equivalent (each head
    attribute is a real Module iff its enable flag was True), but the attribute
    check is more robust to model/cfg drift, and ``getattr(..., False)`` on the
    cfg is required because base configs omit the layout flags under Hydra's
    struct mode.

    Head names: ``depth``, ``layout_depth``, ``mask``, ``normal``, ``camera``,
    ``point``, ``track``, ``seg``.

    NOTE: this is a *static* check (no forward pass). Callers that also want to
    guard against a cfg/checkpoint mismatch (an enabled-but-untrained head, or a
    head whose prediction key is absent for the current inputs) should AND the
    result with actual prediction-dict key presence at their call site.
    """
    module = None
    if model is not None:
        module = model.module if hasattr(model, "module") else model
    model_cfg = getattr(cfg, "model", None) if cfg is not None else None

    heads: dict = {}
    for name, (attr, flag) in _HEAD_SPECS.items():
        if module is not None and hasattr(module, attr):
            heads[name] = getattr(module, attr, None) is not None
        elif model_cfg is not None:
            heads[name] = bool(getattr(model_cfg, flag, False))
        else:
            heads[name] = False
    return heads


def cameras_from_sample(
    sample: dict,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract intrinsics/extrinsics from a per-sample dataset dict.

    The dataset loader returns ``intrinsics`` as a list of (3,3) np
    arrays and ``extrinsics`` as a list of (3,4) np arrays. This helper
    stacks them into a single batched tensor pair with a leading B=1 dim,
    matching the shape ``VGGT.forward`` expects when OCA is enabled.

    Returns ``(None, None)`` if either field is absent (lets callers stay
    safe on ad-hoc samples that lack cameras).
    """
    K_list = sample.get("intrinsics")
    E_list = sample.get("extrinsics")
    if K_list is None or E_list is None:
        return None, None

    K = np.stack([np.asarray(k, dtype=np.float32) for k in K_list], axis=0)  # (V, 3, 3)
    E = np.stack([np.asarray(e, dtype=np.float32) for e in E_list], axis=0)  # (V, 3, 4)

    K_t = torch.from_numpy(K).unsqueeze(0)
    E_t = torch.from_numpy(E).unsqueeze(0)
    if device is not None:
        K_t = K_t.to(device)
        E_t = E_t.to(device)
    return K_t, E_t


def forward_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    intrinsics: Optional[torch.Tensor] = None,
    extrinsics: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """Forward through ``model``, routing GT cameras through iff OCA is enabled.

    For non-OCA checkpoints the call is exactly ``model(images=images)``,
    bit-for-bit backwards-compatible with all E1-E7 evals.

    Note: this helper does NOT enforce that cameras are present when OCA
    needs them, that is the job of the OCA module itself, which raises a
    clear ``RuntimeError`` if ``use_epipolar_bias=True`` and intrinsics or
    extrinsics are missing. Eval scripts should still call this helper
    *with* cameras whenever the dataset provides them.
    """
    if model_has_oca(model):
        return model(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            **kwargs,
        )
    return model(images=images, **kwargs)
