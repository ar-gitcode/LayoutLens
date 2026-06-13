"""Room Envelopes dataset I/O, the only valid depth decoder lives here.

Depth PNGs are MoGe log-encoded uint16 with ``near`` and ``far`` stored as PNG
``tEXt`` chunks. The decode is:

    t     = (raw - 1) / 65533
    depth = near**(1 - t) * far**t              # metric z-depth, metres

Sentinel raw values:
    raw == 0      → NaN  (unknown / invalid)
    raw == 65535  → +inf (sky / beyond far)

``decode_moge_depth`` collapses both sentinels to ``0.0`` in the returned
float32 array so the rest of the pipeline can treat ``depth > 1e-6`` as the
validity test. ``decode_moge_depth_raw`` preserves NaN/+inf for callers that
want the unmodified MoGe semantics.

Never multiply the decoded depth by ``meters_per_asset_unit``. That value is a
property of recovered Hypersim camera poses and is applied only to pose
translations (see ``room_envelopes.extrinsics``).
"""

from __future__ import annotations

import json
from typing import Tuple

import numpy as np
from PIL import Image


class MissingDepthMetadataError(ValueError):
    """Raised when a depth PNG lacks the required ``near``/``far`` tEXt chunks."""


def _read_near_far(img: Image.Image, path: str) -> Tuple[float, float]:
    info = getattr(img, "info", {}) or {}
    if "near" not in info or "far" not in info:
        raise MissingDepthMetadataError(
            f"Depth PNG {path!r} is missing 'near'/'far' tEXt metadata required by "
            f"the MoGe decoder. Got info keys: {sorted(info)}"
        )
    near = float(info["near"])
    far = float(info["far"])
    if not (np.isfinite(near) and np.isfinite(far)) or near <= 0.0 or far <= near:
        raise MissingDepthMetadataError(
            f"Depth PNG {path!r} has invalid near={near}, far={far}; "
            f"expected 0 < near < far and both finite."
        )
    return near, far


def decode_moge_depth_raw(path: str) -> Tuple[np.ndarray, float, float]:
    """Decode a MoGe log-depth PNG preserving sentinels.

    Returns ``(depth, near, far)`` where ``depth`` is float32 with:
        ``raw == 0``      → ``np.nan`` (unknown / invalid)
        ``raw == 65535``  → ``np.inf`` (sky)
        otherwise         → ``near**(1-t) * far**t`` (metres)
    """
    with Image.open(path) as img:
        img.load()
        near, far = _read_near_far(img, path)
        raw = np.asarray(img)
    if raw.dtype not in (np.uint16, np.int32, np.uint32):
        raise ValueError(
            f"Depth PNG {path!r} unexpected dtype {raw.dtype}; expected uint16-ish."
        )
    mask_nan = raw == 0
    mask_inf = raw == 65535
    t = (raw.astype(np.float32) - 1.0) / 65533.0
    depth = (near ** (1.0 - t)) * (far ** t)
    depth = depth.astype(np.float32, copy=False)
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth, near, far


def decode_moge_depth(path: str) -> Tuple[np.ndarray, float, float]:
    """Decode a MoGe log-depth PNG with sentinels collapsed to ``0.0``.

    Identical to :func:`decode_moge_depth_raw` but NaN (invalid) and +inf (sky)
    are replaced with ``0.0`` so downstream code can use ``depth > 1e-6`` as the
    validity test (matches the pipeline's existing mask convention).
    """
    depth, near, far = decode_moge_depth_raw(path)
    invalid = ~np.isfinite(depth)
    if invalid.any():
        depth = np.where(invalid, np.float32(0.0), depth)
    return depth, near, far


def load_rgb(path: str) -> np.ndarray:
    """RGBA/RGB PNG → ``(H, W, 3)`` uint8."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def load_seen_mask(path: str) -> np.ndarray:
    """uint8 mask PNG → ``(H, W)`` uint8 in ``{0, 1}``, thresholded at 127."""
    with Image.open(path) as img:
        arr = np.array(img, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def load_normal_png(path: str) -> np.ndarray | None:
    """RGB uint8 normal map → ``(H, W, 3)`` float32 unit vectors, or ``None``.

    Z-axis convention fix: Room Envelopes encodes layout normals in a frame
    whose Z axis is flipped relative to what the models were trained against
    (the source normal maps are decoded as ``rgb/127.5 - 1``).
    Empirically, flipping Z brings the mean angular error against the trained
    normal-supervised predictions from ~82° down to ~13°.
    """
    try:
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    normals = arr / 127.5 - 1.0
    normals[..., 2] = -normals[..., 2]
    norms = np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-8)
    return (normals / norms).astype(np.float32)


def load_intrinsics_pixel(json_path: str, H: int, W: int) -> np.ndarray:
    """Read normalized intrinsics from the per-frame JSON and denormalise to pixels.

    The JSON ``intrinsics`` field stores K with ``cx = cy = 0.5`` and ``fx/W``,
    ``fy/H`` so it's resolution-independent. Scale to pixel space here.
    """
    with open(json_path) as f:
        meta = json.load(f)
    K = np.asarray(meta["intrinsics"], dtype=np.float32)
    K[0, :] *= W
    K[1, :] *= H
    return K


def derive_seen_mask(
    visible_depth_m: np.ndarray,
    layout_depth_m: np.ndarray,
    thresh: float,
) -> np.ndarray:
    """Derived visibility mask from relative-depth agreement. Returns uint8 in ``{0, 1}``."""
    both_valid = (layout_depth_m > 1e-6) & (visible_depth_m > 1e-6)
    rel_diff = np.where(
        both_valid,
        np.abs(visible_depth_m - layout_depth_m) / np.maximum(layout_depth_m, 1e-6),
        np.ones_like(layout_depth_m),
    )
    return (both_valid & (rel_diff < thresh)).astype(np.uint8)
