"""Geometry utilities for room-envelope reconstruction.

Camera convention throughout:
  - Extrinsics are **camera-from-world** (OpenCV, 3×4): transform world points to camera coords.
  - Use `closed_form_inverse_se3` (from vggt.utils.geometry) to get world-from-camera.
  - Intrinsics are standard 3×3 OpenCV matrices: [[fx,0,cx],[0,fy,cy],[0,0,1]].
  - Depth maps are float32 metres; 0 = invalid.

The dataset already provides `world_points` (H,W,3) and `cam_points` (H,W,3) from GT visible
depth. For GT layout depth, recompute with `depth_to_world_points` using the same cameras.
"""

import os
import sys
import numpy as np

# Put training/ on sys.path so vggt imports work when called from scripts/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_training_dir = os.path.dirname(_this_dir)
_repo_root = os.path.dirname(_training_dir)
for _p in [_training_dir, _repo_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Depth → 3D unprojection
# ---------------------------------------------------------------------------

def unproject_depth_to_camera_points(depth: np.ndarray,
                                      intrinsics: np.ndarray) -> np.ndarray:
    """Unproject a depth map to camera-frame 3D points.

    Args:
        depth:      (H, W) float32 metres. 0 = invalid.
        intrinsics: (3, 3) OpenCV intrinsic matrix.

    Returns:
        cam_points: (H, W, 3) float32. Invalid pixels are set to (0, 0, 0).
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    us = np.arange(W, dtype=np.float32)
    vs = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    valid = depth > 1e-6
    x = np.where(valid, (uu - cx) / fx * depth, 0.0)
    y = np.where(valid, (vv - cy) / fy * depth, 0.0)
    z = np.where(valid, depth, 0.0)

    return np.stack([x, y, z], axis=-1).astype(np.float32)


def transform_camera_to_world(points_cam: np.ndarray,
                               extrinsics: np.ndarray) -> np.ndarray:
    """Transform camera-frame points to world frame.

    Args:
        points_cam: (H, W, 3) or (N, 3) float32 camera-frame points.
        extrinsics: (3, 4) camera-from-world matrix.

    Returns:
        world_points: same shape as input.
    """
    from vggt.utils.geometry import closed_form_inverse_se3

    ext_4x4 = np.eye(4, dtype=np.float32)
    ext_4x4[:3, :] = extrinsics
    cam_to_world = closed_form_inverse_se3(ext_4x4[None])[0]  # (4,4)
    R = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]

    orig_shape = points_cam.shape
    pts = points_cam.reshape(-1, 3)
    world = pts @ R.T + t
    return world.reshape(orig_shape).astype(np.float32)


def depth_to_world_points(depth: np.ndarray,
                           intrinsics: np.ndarray,
                           extrinsics: np.ndarray) -> np.ndarray:
    """Unproject depth map to world-frame 3D points.

    Args:
        depth:      (H, W) float32 metres.
        intrinsics: (3, 3) OpenCV intrinsic matrix.
        extrinsics: (3, 4) camera-from-world matrix.

    Returns:
        world_points: (H, W, 3) float32.
    """
    cam_pts = unproject_depth_to_camera_points(depth, intrinsics)
    return transform_camera_to_world(cam_pts, extrinsics)


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def mask_valid_points(points: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Select valid points from a spatial point array.

    Args:
        points:     (H, W, 3) or (N, 3) float32.
        valid_mask: (H, W) or (N,) bool, True = keep.

    Returns:
        (M, 3) float32 array of valid points.
    """
    pts_flat = points.reshape(-1, 3)
    mask_flat = valid_mask.reshape(-1)
    return pts_flat[mask_flat]


def sample_points(points: np.ndarray, max_points: int = 50_000,
                  seed: int = 0) -> np.ndarray:
    """Randomly downsample a point cloud.

    Args:
        points:     (N, 3) float32.
        max_points: Maximum number of output points.
        seed:       Random seed for reproducibility.

    Returns:
        (min(N, max_points), 3) float32.
    """
    N = len(points)
    if N <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, max_points, replace=False)
    return points[idx]


def save_point_cloud_ply(points: np.ndarray, path: str,
                          colors: np.ndarray = None) -> None:
    """Write a point cloud as ASCII PLY (no Open3D required).

    Args:
        points: (N, 3) float32.
        colors: (N, 3) uint8 RGB, optional.
        path:   Output .ply file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    N = len(points)
    has_color = colors is not None and len(colors) == N

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i]
            if has_color:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def chamfer_distance(points_a: np.ndarray,
                     points_b: np.ndarray) -> tuple:
    """Chamfer distance between two point clouds using nearest-neighbour search.

    Args:
        points_a: (N, 3) float32.
        points_b: (M, 3) float32.

    Returns:
        (chamfer_dist, a_to_b_mean, b_to_a_mean) all in metres.
        Returns (inf, inf, inf) if either cloud is empty.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        inf = float("inf")
        return inf, inf, inf

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("scipy is required for chamfer_distance")

    tree_b = cKDTree(points_b)
    tree_a = cKDTree(points_a)

    d_a2b, _ = tree_b.query(points_a, k=1)
    d_b2a, _ = tree_a.query(points_b, k=1)

    a2b = float(d_a2b.mean())
    b2a = float(d_b2a.mean())
    cd  = a2b + b2a
    return cd, a2b, b2a


def coverage_completeness(points_pred: np.ndarray,
                           points_gt: np.ndarray,
                           thresholds: tuple = (0.05, 0.10, 0.20)) -> dict:
    """Fraction of GT points within threshold of any predicted point.

    Args:
        points_pred: (N, 3) predicted layout points.
        points_gt:   (M, 3) GT layout points.
        thresholds:  Tuple of distance thresholds in metres.

    Returns:
        Dict mapping threshold → fraction in [0, 1].
    """
    if len(points_pred) == 0 or len(points_gt) == 0:
        return {t: 0.0 for t in thresholds}

    from scipy.spatial import cKDTree
    tree_pred = cKDTree(points_pred)
    d_gt2pred, _ = tree_pred.query(points_gt, k=1)

    return {t: float((d_gt2pred <= t).mean()) for t in thresholds}


# ---------------------------------------------------------------------------
# RANSAC plane fitting
# ---------------------------------------------------------------------------

def fit_planes_ransac(points: np.ndarray,
                      max_planes: int = 6,
                      thresh: float = 0.05,
                      min_inliers: int = 50,
                      max_iters: int = 200,
                      seed: int = 42) -> list:
    """Iterative RANSAC plane fitting.

    Args:
        points:      (N, 3) float32.
        max_planes:  Maximum number of planes to fit.
        thresh:      Inlier distance threshold in metres.
        min_inliers: Stop if remaining points < min_inliers.
        max_iters:   RANSAC iterations per plane.
        seed:        Random seed.

    Returns:
        List of dicts, one per fitted plane:
          {"normal": [a,b,c], "d": d, "n_inliers": int, "inlier_ratio": float,
           "mean_residual": float}
        The normal is unit-length; plane equation: normal·x + d = 0.
    """
    rng = np.random.default_rng(seed)
    remaining = points.copy()
    planes = []

    for _ in range(max_planes):
        N = len(remaining)
        if N < min_inliers or N < 3:
            break

        best_normal = None
        best_d = None
        best_inliers = None
        best_n_inliers = 0

        for _ in range(max_iters):
            idx = rng.choice(N, 3, replace=False)
            p0, p1, p2 = remaining[idx[0]], remaining[idx[1]], remaining[idx[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm < 1e-8:
                continue
            n = n / norm
            d = -float(n @ p0)

            dist = np.abs(remaining @ n + d)
            inlier_mask = dist < thresh
            n_in = int(inlier_mask.sum())

            if n_in > best_n_inliers:
                best_n_inliers = n_in
                best_normal = n
                best_d = d
                best_inliers = inlier_mask

        if best_normal is None or best_n_inliers < min_inliers:
            break

        # Refit on all inliers with least squares
        inlier_pts = remaining[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid)
        n_refined = Vt[-1]
        if n_refined @ best_normal < 0:
            n_refined = -n_refined
        d_refined = -float(n_refined @ centroid)

        final_dist = np.abs(inlier_pts @ n_refined + d_refined)
        planes.append({
            "normal":        n_refined.tolist(),
            "d":             float(d_refined),
            "n_inliers":     int(best_n_inliers),
            "inlier_ratio":  float(best_n_inliers / N),
            "mean_residual": float(final_dist.mean()),
        })

        # Remove inliers for next iteration
        remaining = remaining[~best_inliers]

    return planes
