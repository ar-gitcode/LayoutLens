"""Normalise a sample into the VGGT scene frame (GT depth/cameras)."""
from __future__ import annotations

import numpy as np


def normalize_sample_vggt_scene(sample: dict) -> tuple[dict, dict]:
    """Apply native VGGT-scene normalization to a single eval sample.

    Calls the exact same helper that ``training/trainer.py:_process_batch``
    uses on every train/val batch:
    ``training.train_utils.normalization.normalize_camera_extrinsics_and_points_batch``.
    The helper transforms the scene into the first-camera coordinate frame
    and divides extrinsic translations, depths, world_points, cam_points,
    and layout_depths by a per-sample mean valid-point distance to the
    origin.

    Predictions are **not** touched here; the model is expected to already
    output values in the same scene-normalized space (because the trainer
    normalized GT before the loss). Keeping this helper GT-only mirrors the
    trainer.

    Args:
        sample: per-scene dict from the dataset loader. Required keys:
            ``extrinsics`` (S,3,4), ``world_points`` (S,H,W,3),
            ``cam_points`` (S,H,W,3), ``depths`` (S,H,W),
            ``point_masks`` (S,H,W) bool. ``layout_depths`` (S,H,W) is
            optional but normalized when present. Per-frame entries can be
            lists or stacked arrays.

    Returns:
        ``(normalized_sample, normalization_info)``.

        * ``normalized_sample``: shallow copy of the input with normalized
          ``extrinsics``, ``world_points``, ``cam_points``, ``depths``,
          ``layout_depths`` (all numpy with leading frame dim ``S``).
          ``intrinsics``, ``images``, ``layout_masks``,
          ``layout_depth_masks``, ``layout_normals``,
          ``layout_normal_masks``, ``seg_masks``, ``seq_name``, ``ids``
          are passed through unchanged.
        * ``normalization_info``: dict with ``vggt_scene_scale`` (per-sample
          mean distance, single scalar for B=1), the matching
          ``vggt_scene_scale_{mean,min,max}`` aliases, ``valid_point_count``
          (number of valid pixels going into the scale estimate), and
          ``normalization_source`` (the qualified function name).

    Raises:
        KeyError: if any required field is missing from the sample.
    """
    import torch  # local import: keeps reconstruction_utils torch-free at top.

    # `train_utils.normalization` lives under the `training/` tree; the
    # module-level path setup at the top of this file already prepended
    # the training directory to sys.path, so this import resolves.
    from train_utils.normalization import (  # type: ignore  # noqa: E402
        normalize_camera_extrinsics_and_points_batch,
    )

    def _stack(arr):
        if isinstance(arr, list):
            return np.stack([np.asarray(a) for a in arr], axis=0)
        return np.asarray(arr)

    def _to_t(arr, dtype):
        a = _stack(arr)
        return torch.as_tensor(a, dtype=dtype, device="cpu").unsqueeze(0)

    required = ["extrinsics", "depths", "world_points", "cam_points", "point_masks"]
    missing = [k for k in required if sample.get(k) is None]
    if missing:
        raise KeyError(
            "normalize_sample_vggt_scene: sample missing required field(s) "
            f"{missing}; present keys: {sorted(sample.keys())}"
        )

    extr_t       = _to_t(sample["extrinsics"],   torch.float32)
    cam_pts_t    = _to_t(sample["cam_points"],   torch.float32)
    world_pts_t  = _to_t(sample["world_points"], torch.float32)
    depths_t     = _to_t(sample["depths"],       torch.float32)
    point_mask_t = _to_t(sample["point_masks"],  torch.bool)

    layout_depths_t = None
    if sample.get("layout_depths") is not None:
        layout_depths_t = _to_t(sample["layout_depths"], torch.float32)

    new_extr, new_cam, new_world, new_depths, new_layout = (
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=extr_t,
            cam_points=cam_pts_t,
            world_points=world_pts_t,
            depths=depths_t,
            point_masks=point_mask_t,
            layout_depths=layout_depths_t,
        )
    )

    # Recover the per-sample scalar scale so the caller can log/diagnose it.
    # The helper divides every spatial quantity by the same `avg_scale`, so
    # original / normalized at any valid pixel returns the scale exactly
    # (modulo float precision). We use layout_depths if available, else
    # depths.
    scale = float("nan")
    valid_count = int(point_mask_t.sum().item())
    if layout_depths_t is not None and new_layout is not None:
        keep = (layout_depths_t > 1e-6) & (new_layout.abs() > 1e-12)
        if bool(keep.any()):
            ratio = layout_depths_t[keep].double() / new_layout[keep].double()
            scale = float(ratio.mean().item())
    if not np.isfinite(scale):
        keep = (depths_t > 1e-6) & (new_depths.abs() > 1e-12)
        if bool(keep.any()):
            ratio = depths_t[keep].double() / new_depths[keep].double()
            scale = float(ratio.mean().item())

    def _back(t):
        if t is None:
            return None
        return t.squeeze(0).cpu().numpy()

    normalized_sample = dict(sample)
    normalized_sample["extrinsics"]   = _back(new_extr)
    normalized_sample["cam_points"]   = _back(new_cam)
    normalized_sample["world_points"] = _back(new_world)
    normalized_sample["depths"]       = _back(new_depths)
    if new_layout is not None:
        normalized_sample["layout_depths"] = _back(new_layout)

    info = {
        "vggt_scene_scale":      scale,
        "vggt_scene_scale_mean": scale,
        "vggt_scene_scale_min":  scale,
        "vggt_scene_scale_max":  scale,
        "normalization_source":
            "training.train_utils.normalization.normalize_camera_extrinsics_and_points_batch",
        "valid_point_count":     valid_count,
    }
    return normalized_sample, info

