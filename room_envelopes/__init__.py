"""Room Envelopes dataset I/O, depth, normals, masks, intrinsics, extrinsics."""

from .io import (
    MissingDepthMetadataError,
    decode_moge_depth,
    decode_moge_depth_raw,
    load_intrinsics_pixel,
    load_normal_png,
    load_rgb,
    load_seen_mask,
    derive_seen_mask,
)
from .extrinsics import ExtrinsicsManifest, MissingExtrinsicsError

__all__ = [
    "MissingDepthMetadataError",
    "decode_moge_depth",
    "decode_moge_depth_raw",
    "load_intrinsics_pixel",
    "load_normal_png",
    "load_rgb",
    "load_seen_mask",
    "derive_seen_mask",
    "ExtrinsicsManifest",
    "MissingExtrinsicsError",
]
