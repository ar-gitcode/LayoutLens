"""Render bounded planes back to per-view layout-depth maps.

Given a list of plane dicts produced by ``postprocess_planes.py`` or
``postprocess_cuboid.py`` and a set of camera (K, E_w2c) pairs, this module
produces z-depth maps in metres aligned to each camera view.

Algorithm per camera
--------------------
1. Build per-pixel rays in the camera frame using K, then transform them
   into the world frame using the 4×4 camera-to-world matrix.
2. For each plane, intersect each ray with the plane and keep ``t_hit``
   where the hit point ``hit_world`` lies inside the plane's centroid-relative
   2D extent (the ``extent_xy`` field stored on each plane) AND is on the
   room-interior side of every other plane in the set.
3. For each pixel, keep the smallest valid ``t_hit``.
4. Convert ``hit_world`` back to the camera frame and emit ``z = hit_cam[..., 2]``.
5. Return ``(z_depth, valid_mask)`` separately so the caller can choose its
   hole-fill policy (see ``eval_2d_postprocess.py``).

Conventions
-----------
- Extrinsics are 3×4 OpenCV camera-from-world (``X_cam = R · X_world + t``).
  We invert with :func:`vggt.utils.geometry.closed_form_inverse_se3` to get
  the camera centre and ``R_c2w``.
- Output depth is **z-depth in metres** (= ``hit_cam[..., 2]``), matching
  the rest of this repo's layout-depth convention. NOT euclidean distance.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ``training/`` must be on sys.path so the vggt namespace resolves.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_training_dir = os.path.dirname(_this_dir)
_repo_root = os.path.dirname(_training_dir)
for _p in (_training_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_centre_and_R_c2w(extrinsics_3x4: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(camera_centre_world, R_c2w)`` from a 3×4 OpenCV w2c matrix."""
    from vggt.utils.geometry import closed_form_inverse_se3
    E4 = np.eye(4, dtype=np.float32)
    E4[:3, :] = extrinsics_3x4.astype(np.float32)
    c2w = closed_form_inverse_se3(E4[None])[0]
    R_c2w = c2w[:3, :3].astype(np.float32)
    centre = c2w[:3, 3].astype(np.float32)
    return centre, R_c2w


def _pixel_ray_dirs_world(K: np.ndarray, R_c2w: np.ndarray,
                           H: int, W: int) -> np.ndarray:
    """Per-pixel ray directions in world frame, shape (H, W, 3).

    Camera-frame direction has z=1 so the parameter ``t`` along the ray
    equals the camera-frame z of the intersection (= z-depth).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    rx = (uu - cx) / fx
    ry = (vv - cy) / fy
    rz = np.ones_like(rx)
    dirs_cam = np.stack([rx, ry, rz], axis=-1)         # (H, W, 3)
    dirs_world = dirs_cam.reshape(-1, 3) @ R_c2w.T
    return dirs_world.reshape(H, W, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-view renderer
# ---------------------------------------------------------------------------

def render_planes_to_zdepth(planes: list[dict],
                             K: np.ndarray,
                             extrinsics_3x4: np.ndarray,
                             H: int,
                             W: int,
                             room_interior_pt: np.ndarray,
                             *,
                             min_depth: float = 0.05,
                             max_depth: float = 50.0,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Render a single view's z-depth and valid-mask from a plane set.

    Returns ``(depth, valid_mask)`` with shapes ``(H, W)`` float32 and
    ``(H, W)`` bool. Pixels with no valid hit have ``depth = 0`` and
    ``valid_mask = False``; the driver applies the hole-fill policy.
    """
    if not planes:
        return (np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=bool))

    centre_world, R_c2w = _camera_centre_and_R_c2w(extrinsics_3x4)
    dirs_world = _pixel_ray_dirs_world(K, R_c2w, H, W)        # (H, W, 3)

    # Camera centre in camera frame is the origin; we'll convert hits back
    # later. ``t`` here is the camera-frame z (since we parameterised rays
    # with z_cam=1), and it is also the parameter along the world ray
    # ``r(t) = centre_world + t · dirs_world``.
    HW = H * W
    o = centre_world[None, :]                                  # (1, 3)
    d = dirs_world.reshape(HW, 3)                              # (HW, 3)

    # Interior side per plane: sign of (n · room_interior_pt + d_val).
    interior = np.asarray(room_interior_pt, dtype=np.float32)
    plane_normals = np.stack([p["normal"] for p in planes], axis=0).astype(np.float32)  # (P,3)
    plane_d = np.array([p["d"] for p in planes], dtype=np.float32)                       # (P,)
    plane_centroids = np.stack([p["centroid"] for p in planes], axis=0).astype(np.float32)  # (P,3)
    plane_us = np.stack([p["u"] for p in planes], axis=0).astype(np.float32)             # (P,3)
    plane_vs = np.stack([p["v"] for p in planes], axis=0).astype(np.float32)             # (P,3)
    plane_extents = np.array([p["extent_xy"] for p in planes], dtype=np.float32)         # (P,4)

    # ``interior_sign[p] ∈ {-1,+1}``: which side of plane p the interior is on.
    interior_sign = np.sign(plane_normals @ interior + plane_d).astype(np.float32)
    interior_sign[interior_sign == 0] = 1.0  # tie-break

    # Intersect every ray with every plane.
    denom = d @ plane_normals.T                                # (HW, P)
    num   = -(o @ plane_normals.T + plane_d[None, :])           # (1, P) broadcast
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(denom) > 1e-8, num / denom, np.inf) # (HW, P)
    in_front = (t > min_depth) & (t < max_depth)
    t = np.where(in_front, t, np.inf)

    # Bounded-extent test: project hit_world into each plane's (u, v) frame
    # and check the centroid-relative quantile bounds.
    # hit_world[k, p, :] = o + t[k, p] · d[k, :]
    # Compute (hit - centroid) · u and · v for each (ray, plane).
    # This is the heavy bit; we keep it in (HW, P) vectorised form.
    valid_hits = np.isfinite(t)
    P = len(planes)

    extents_in = np.ones_like(t, dtype=bool)
    for p_idx in range(P):
        finite = valid_hits[:, p_idx]
        if not finite.any():
            extents_in[:, p_idx] = False
            continue
        tp = t[finite, p_idx][:, None]                          # (M, 1)
        hit_world = o + tp * d[finite]                          # (M, 3)
        rel = hit_world - plane_centroids[p_idx][None, :]
        u_coord = rel @ plane_us[p_idx]                         # (M,)
        v_coord = rel @ plane_vs[p_idx]                         # (M,)
        xmin, xmax, ymin, ymax = plane_extents[p_idx]
        ok = (u_coord >= xmin) & (u_coord <= xmax) & \
             (v_coord >= ymin) & (v_coord <= ymax)
        full_ok = np.zeros(HW, dtype=bool)
        full_ok[finite] = ok
        extents_in[:, p_idx] = full_ok

    t = np.where(extents_in, t, np.inf)
    valid_hits = np.isfinite(t)
    if not valid_hits.any():
        return (np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=bool))

    # Interior half-space test: for each candidate hit on plane p, the hit
    # point must lie on the interior side of every OTHER plane q. Without
    # this we'd render ceilings through floors.
    # Vectorised: compute hit_world for all (ray, plane_p) pairs, then for
    # each plane q compute sign(hit_world · n_q + d_q) and compare to
    # interior_sign[q]. A hit passes iff this matches for all q ≠ p.
    # Cost: O(HW · P · P). For P ≤ ~10 and HW ≈ 800k this is fine.
    for p_idx in range(P):
        finite = valid_hits[:, p_idx]
        if not finite.any():
            continue
        tp = t[finite, p_idx][:, None]
        hit_world = o + tp * d[finite]                          # (M, 3)
        # Signed distance to every plane: (M, P)
        sd = hit_world @ plane_normals.T + plane_d[None, :]
        # The hit must be on the interior side of every q != p.
        signs = np.sign(sd)
        signs[signs == 0] = 1.0
        # Match where signs[:, q] == interior_sign[q] OR q == p_idx.
        ok_per_q = signs == interior_sign[None, :]
        ok_per_q[:, p_idx] = True  # the hit's own plane is allowed
        full_ok = np.zeros(HW, dtype=bool)
        full_ok[finite] = ok_per_q.all(axis=1)
        t[~full_ok, p_idx] = np.inf

    valid_hits = np.isfinite(t)
    if not valid_hits.any():
        return (np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=bool))

    # First valid hit per ray.
    t_min = np.min(t, axis=1)                                   # (HW,)
    valid = np.isfinite(t_min)
    t_min[~valid] = 0.0

    # Convert chosen world-frame hits back to camera-frame z. Because rays
    # were parameterised with z_cam=1, t along the ray equals the
    # camera-frame z of the hit, so depth = t directly.
    depth = t_min.reshape(H, W).astype(np.float32)
    mask = valid.reshape(H, W)
    return depth, mask


def render_planes_to_zdepth_batch(planes: list[dict],
                                   Ks: np.ndarray,
                                   Es: np.ndarray,
                                   H: int,
                                   W: int,
                                   room_interior_pt: np.ndarray,
                                   **kwargs,
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Render all S views of a scene. ``Ks`` and ``Es`` are (S, 3, 3) and
    (S, 3, 4) arrays of intrinsics and OpenCV w2c extrinsics.

    Returns ``(z_depths, valid_masks)`` with shapes ``(S, H, W)`` float32 and
    ``(S, H, W)`` bool.
    """
    S = Ks.shape[0]
    depth_out = np.zeros((S, H, W), dtype=np.float32)
    mask_out = np.zeros((S, H, W), dtype=bool)
    for s in range(S):
        d, m = render_planes_to_zdepth(planes, Ks[s], Es[s], H, W,
                                        room_interior_pt, **kwargs)
        depth_out[s] = d
        mask_out[s] = m
    return depth_out, mask_out


__all__ = ["render_planes_to_zdepth", "render_planes_to_zdepth_batch"]


# ---------------------------------------------------------------------------
# __main__ smoke checks (K.2 round-trip + K.8 metric-on-holes)
# ---------------------------------------------------------------------------

def _make_synthetic_scene():
    """Build a synthetic 3×3×3 m box room with one camera inside."""
    # Box centred at origin, sides 3 m.
    half = 1.5
    box_points = []
    for axis in range(3):
        for side in (-1, +1):
            others = [i for i in range(3) if i != axis]
            grid = np.linspace(-half, half, 30)
            uu, vv = np.meshgrid(grid, grid)
            face = np.zeros((uu.size, 3), dtype=np.float32)
            face[:, axis] = side * half
            face[:, others[0]] = uu.ravel()
            face[:, others[1]] = vv.ravel()
            box_points.append(face)
    pts = np.concatenate(box_points, axis=0)
    return pts


def _smoke_check_round_trip():
    """K.2: unproject + project should round-trip to ≤ 1e-3 m."""
    from training.geometry.room_envelope_geometry import depth_to_world_points

    H, W = 60, 80
    fx = fy = 60.0
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    # Camera at origin, looking down +Z.
    E = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)

    # A floor 2 m in front of the camera (camera Z=2 plane in world).
    z_const = 2.0
    depth = np.full((H, W), z_const, dtype=np.float32)
    world_pts = depth_to_world_points(depth, K, E)

    # Render a single plane (the floor) using the renderer.
    normal = np.array([0, 0, -1], dtype=np.float32)  # points back toward camera
    d_val = float(z_const)  # normal · x + d = 0 with normal=(0,0,-1), plane Z=2
    # Build a synthetic plane record with very generous extents.
    u, v = np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)
    plane = {
        "normal": normal, "d": d_val,
        "n_inliers": H * W, "inlier_ratio": 1.0, "mean_residual": 0.0,
        "inlier_mask": np.ones(H * W, dtype=bool),
        "u": u, "v": v,
        "centroid": np.array([0, 0, z_const], dtype=np.float32),
        "extent_xy": (-1e3, 1e3, -1e3, 1e3),
    }
    interior = np.array([0, 0, 0], dtype=np.float32)  # camera centre is inside
    rendered, mask = render_planes_to_zdepth([plane], K, E, H, W, interior,
                                              min_depth=0.05, max_depth=100.0)
    err = np.abs(rendered[mask] - z_const).max() if mask.any() else float("inf")
    print(f"[K.2] flat-plane round-trip: max |rendered - z| = {err:.6e} m "
          f"(coverage = {mask.mean():.3f})")
    assert mask.all(), "plane should cover every pixel"
    assert err < 1e-3, "round-trip error too large"

    # Cross-check with unprojection: world_pts of these depth pixels should
    # all lie on the plane normal·x + d = 0.
    res = (world_pts.reshape(-1, 3) @ normal + d_val)
    assert np.abs(res).max() < 1e-3
    print("[K.2] world_pts unproject lies on plane to <1e-3 m  OK")


def _smoke_check_metric_on_holes():
    """K.8: pred=0 holes inflate AbsRel when not filled."""
    from training.eval_metrics import compute_depth_metrics

    gt = np.full((10, 10), 2.0, dtype=np.float32)
    pred_full = np.full_like(gt, 2.0)
    pred_holed = pred_full.copy()
    pred_holed[:5, :] = 0.0  # 50 % holes

    m_full = compute_depth_metrics(pred_full, gt)
    m_holed = compute_depth_metrics(pred_holed, gt)
    print(f"[K.8] AbsRel full  = {m_full['absrel']:.4f}")
    print(f"[K.8] AbsRel holed = {m_holed['absrel']:.4f}")
    assert m_full["absrel"] < 1e-4
    assert m_holed["absrel"] > 0.4, (
        "expected pred=0 holes to inflate AbsRel, confirms the metric "
        "does NOT silently ignore them (justifies --render_holes fill)"
    )


def _smoke_check_bounded_plane_box():
    """The bounded-extent rejection should clip a finite floor inside a box.

    Build a 6-face box and render from a camera inside; the result should be
    a depth map that varies smoothly per pixel (not a constant floor).
    """
    H, W = 30, 30
    fx = fy = 30.0
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    E = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    interior = np.array([0, 0, 0], dtype=np.float32)

    half = 1.0
    planes = []
    for axis in range(3):
        for side in (-1, +1):
            others = [i for i in range(3) if i != axis]
            normal = np.zeros(3, dtype=np.float32)
            normal[axis] = float(side)
            centroid = np.zeros(3, dtype=np.float32)
            centroid[axis] = side * half
            d_val = -float(normal @ centroid)
            u = np.zeros(3, dtype=np.float32); u[others[0]] = 1.0
            v = np.zeros(3, dtype=np.float32); v[others[1]] = 1.0
            planes.append({
                "normal": normal, "d": d_val,
                "n_inliers": 100, "inlier_ratio": 1.0, "mean_residual": 0.0,
                "inlier_mask": np.ones(1, dtype=bool),
                "u": u, "v": v, "centroid": centroid,
                "extent_xy": (-half, half, -half, half),
                "axis_id": axis, "side": side,
            })

    depth, mask = render_planes_to_zdepth(planes, K, E, H, W, interior)
    print(f"[box]  coverage = {mask.mean():.3f}  "
          f"depth range = [{depth[mask].min():.3f}, {depth[mask].max():.3f}] m")
    assert mask.all(), "every pixel should hit some face inside a box"
    # All depths must be ≤ √3 (diagonal of the box) and > 0.
    assert depth[mask].min() > 0
    assert depth[mask].max() < np.sqrt(3) * half + 1e-3


if __name__ == "__main__":
    _smoke_check_round_trip()
    _smoke_check_metric_on_holes()
    _smoke_check_bounded_plane_box()
    print("\n[render_planes_to_depth.py] all smoke checks passed.")
