"""6-plane axis-aligned cuboid room fit, with bounded faces.

Two construction methods:

  ``pca_aabb``: PCA on the fused world-frame point cloud, then min/max
                       quantile extents along each principal axis. Frame-
                       agnostic; does not require any prior plane fit.
  ``from_manhattan``, Use a pre-discovered Manhattan basis (e1, e2, e3),
                       then min/max quantile extents along each axis on the
                       fused point cloud. Useful as a refinement after the
                       Manhattan baseline.

Output schema (per face)
------------------------
A list of exactly 6 face dicts, two per axis. The schema matches the rest
of the pipeline so the renderer can treat plane fits and cuboid faces
uniformly::

    {
      "normal":        (3,) unit,
      "d":             float (so normal·x + d = 0),
      "n_inliers":     int (= count of inlier points within ``inlier_thresh``),
      "inlier_ratio":  float in [0, 1],
      "mean_residual": float metres (mean |distance| of inliers to face),
      "inlier_mask":   bool (N,) over the input fused cloud,
      "u":             (3,) unit, perpendicular to normal,
      "v":             (3,) unit, = normal × u,
      "centroid":      (3,) face centre in world frame,
      "extent_xy":     (xmin, xmax, ymin, ymax) in centroid-relative (u, v),
                       NATURALLY BOUNDED by the box's other two axes.
      "axis_id":       int in {0, 1, 2},
      "side":          int in {-1, +1},
    }
"""

from __future__ import annotations

import numpy as np


def _principal_axes(points: np.ndarray) -> np.ndarray:
    """Return shape (3,3) axes from PCA, rows ordered by descending variance."""
    centred = points - points.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return Vt  # rows are unit eigenvectors of the cov matrix.


def fit_cuboid_room(points: np.ndarray, *,
                    method: str = "pca_aabb",
                    manhattan_basis: np.ndarray | None = None,
                    quantile: tuple[float, float] = (0.01, 0.99),
                    inlier_thresh: float = 0.05,
                    min_points: int = 1000,
                    min_box_dim: float = 0.5,
                    ) -> tuple[list[dict], dict]:
    """Fit an axis-aligned (in the chosen basis) cuboid to ``points``.

    Args:
        points:         (N, 3) world-frame fused cloud.
        method:         ``pca_aabb`` or ``from_manhattan``.
        manhattan_basis: (3, 3) array with rows (e1, e2, e3), required for
                         ``from_manhattan``.
        quantile:       (q_lo, q_hi) for robust min/max along each axis.
        inlier_thresh:  metres, face inlier band half-thickness.
        min_points:     minimum points required to attempt a fit.
        min_box_dim:    minimum allowed box side length; below this the box
                         is flagged degenerate.

    Returns:
        (faces, status). ``faces`` has exactly 6 entries (see module docstring),
        or 0 entries when the fit is degenerate. ``status`` keys:
          - ``cuboid_status``: ``"ok"`` or ``"degenerate"``.
          - ``method``, ``box_dims`` (axis-aligned extents in metres),
            ``axes`` (3x3 row-stacked basis), ``n_points``.
    """
    N = len(points)
    status: dict = {"method": method, "n_points": N}
    if N < min_points:
        status["cuboid_status"] = "degenerate"
        status["reason"] = "n_points<min_points"
        return [], status

    if method == "from_manhattan":
        if manhattan_basis is None:
            status["cuboid_status"] = "degenerate"
            status["reason"] = "no_manhattan_basis"
            return [], status
        axes = np.asarray(manhattan_basis, dtype=np.float32)
        if axes.shape != (3, 3):
            raise ValueError(f"manhattan_basis must be (3,3), got {axes.shape}")
    elif method == "pca_aabb":
        axes = _principal_axes(points).astype(np.float32)
    else:
        raise ValueError(f"unknown method: {method!r}")

    centre_world = points.mean(axis=0).astype(np.float32)

    # Project points onto each axis (centroid-relative) → 1D coords per axis.
    rel = points - centre_world[None, :]
    proj = rel @ axes.T  # (N, 3), columns = coords along (e1, e2, e3).

    q_lo, q_hi = quantile
    mins = np.quantile(proj, q_lo, axis=0)  # (3,)
    maxs = np.quantile(proj, q_hi, axis=0)  # (3,)
    box_dims = (maxs - mins).astype(np.float32)

    status["box_dims"] = box_dims.tolist()
    status["axes"] = axes.tolist()

    if float(box_dims.min()) < min_box_dim:
        status["cuboid_status"] = "degenerate"
        status["reason"] = f"box_dim<{min_box_dim}"
        return [], status

    faces: list[dict] = []
    for axis_id in range(3):
        e = axes[axis_id]                       # outward axis
        # The two non-axis axes form the in-face frame (u, v).
        others = [i for i in range(3) if i != axis_id]
        u = axes[others[0]]
        v = axes[others[1]]
        u_min = float(mins[others[0]])
        u_max = float(maxs[others[0]])
        v_min = float(mins[others[1]])
        v_max = float(maxs[others[1]])

        for side in (-1, +1):
            offset_along = mins[axis_id] if side == -1 else maxs[axis_id]
            face_centre_world = centre_world + float(offset_along) * e
            # Outward normal points away from the interior. Interior is the
            # box centre, so outward direction is +/- e depending on side.
            normal = (e if side == +1 else -e).astype(np.float32)
            d_val = -float(normal @ face_centre_world)

            # Inlier band along the outward direction.
            face_coord = float(offset_along)
            dist = np.abs(proj[:, axis_id] - face_coord)
            inlier_mask = dist < inlier_thresh
            residuals = dist[inlier_mask]
            n_in = int(inlier_mask.sum())
            mean_res = float(residuals.mean()) if n_in > 0 else 0.0

            # u/v projected to in-face coords (centroid-relative). For the
            # cuboid, the extent IS the box's range along the other two axes.
            faces.append({
                "normal":        normal,
                "d":             d_val,
                "n_inliers":     n_in,
                "inlier_ratio":  float(n_in / max(N, 1)),
                "mean_residual": mean_res,
                "inlier_mask":   inlier_mask,
                "u":             u.astype(np.float32),
                "v":             v.astype(np.float32),
                "centroid":      face_centre_world.astype(np.float32),
                "extent_xy":     (u_min, u_max, v_min, v_max),
                "axis_id":       int(axis_id),
                "side":          int(side),
            })

    status["cuboid_status"] = "ok"
    return faces, status


__all__ = ["fit_cuboid_room"]
