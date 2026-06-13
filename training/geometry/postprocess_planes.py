"""Post-processing in 3D: RANSAC plane fitting (with extents) + Manhattan snap.

Used by ``evaluations/src/2d/eval_2d_postprocess.py`` to turn a fused world-frame
point cloud (per scene) into a set of bounded planes that the renderer in
``render_planes_to_depth.py`` projects back into each camera view.

Conventions
-----------
- Points are world-frame (N, 3) float32.
- Plane equation is ``normal · x + d = 0`` (matches the rest of this repo;
  ``d = -offset`` if you came from the old ``vggt_layout_baselines`` schema).
- Each plane dict carries enough information for the renderer's bounded-hit
  test::

      {
        "normal":        (3,) unit vector,
        "d":             float (signed offset),
        "n_inliers":     int,
        "inlier_ratio":  float in [0, 1],
        "mean_residual": float metres,
        "inlier_mask":   bool (N,) relative to the input point cloud,
        "u":             (3,) unit in-plane axis 1,
        "v":             (3,) unit in-plane axis 2 (= normal × u),
        "centroid":      (3,) world-frame centroid of the inliers,
        "extent_xy":     (xmin, xmax, ymin, ymax) in CENTROID-RELATIVE
                         (u, v) coordinates, robust q01/q99 quantiles.
      }

  The extent is what makes RANSAC planes safe for reprojection, without
  it, a single floor plane would render through every wall.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# RANSAC plane fitting (ported from vggt_layout_baselines/geometry/ransac.py,
# augmented with per-plane finite extent computation).
# ---------------------------------------------------------------------------

def _fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """SVD plane fit. Returns (unit_normal, signed_d) for normal·x + d = 0."""
    centroid = pts.mean(axis=0)
    # full_matrices=False: only need Vt (the (3, 3) right singular vectors);
    # the default would compute U as (N, N) which is O(GB) for N≈100k.
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    d = -float(normal @ centroid)
    return normal, d


def _in_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors (u, v) spanning the plane perpendicular to normal.

    Picks u as the cross product of the normal with the world axis least
    parallel to it, then v = normal × u (right-handed).
    """
    n = normal / (np.linalg.norm(normal) + 1e-12)
    abs_n = np.abs(n)
    pick = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pick[int(np.argmin(abs_n))] = 1.0
    u = np.cross(n, pick)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u.astype(np.float32), v.astype(np.float32)


def _compute_extent(inliers: np.ndarray, centroid: np.ndarray,
                    u: np.ndarray, v: np.ndarray,
                    extent_quantiles: tuple[float, float]
                    ) -> tuple[float, float, float, float]:
    """Robust 2D extent of inliers in (u, v) coordinates, centroid-relative."""
    rel = inliers - centroid[None, :]
    uv = rel @ np.stack([u, v], axis=-1)  # (N, 2)
    q_lo, q_hi = extent_quantiles
    xmin, xmax = np.quantile(uv[:, 0], [q_lo, q_hi])
    ymin, ymax = np.quantile(uv[:, 1], [q_lo, q_hi])
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _make_plane_record(inliers: np.ndarray,
                       global_inlier_mask: np.ndarray,
                       N_total: int,
                       extent_quantiles: tuple[float, float]
                       ) -> dict:
    """Build the full plane dict (normal, d, support, extents) from inliers.

    ``global_inlier_mask`` is shape (N_total,) bool, indexing the original
    fused cloud; ``inliers`` is the (M, 3) subset.
    """
    normal, d = _fit_plane_svd(inliers)
    centroid = inliers.mean(axis=0).astype(np.float32)
    u, v = _in_plane_basis(normal)
    xmin, xmax, ymin, ymax = _compute_extent(inliers, centroid, u, v, extent_quantiles)
    residuals = np.abs(inliers @ normal + d)
    n_in = int(global_inlier_mask.sum())
    return {
        "normal":        normal.astype(np.float32),
        "d":             float(d),
        "n_inliers":     n_in,
        "inlier_ratio":  float(n_in / max(N_total, 1)),
        "mean_residual": float(residuals.mean()),
        "inlier_mask":   global_inlier_mask.astype(bool),
        "u":             u.astype(np.float32),
        "v":             v.astype(np.float32),
        "centroid":      centroid,
        "extent_xy":     (float(xmin), float(xmax), float(ymin), float(ymax)),
    }


def _ransac_single_plane(pts: np.ndarray,
                         threshold: float,
                         n_iters: int,
                         min_inliers: int,
                         rng: np.random.Generator
                         ) -> np.ndarray | None:
    """Run one RANSAC pass on ``pts`` (M, 3). Return the inlier mask (M,)
    relative to ``pts``, or None if no plane passes ``min_inliers``."""
    M = len(pts)
    if M < 3:
        return None

    best_n = 0
    best_normal = None
    best_d = 0.0

    for _ in range(n_iters):
        idx = rng.integers(0, M, 3)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        n = n / norm
        d = -float(n @ p0)
        dist = np.abs(pts @ n + d)
        n_in = int((dist < threshold).sum())
        if n_in > best_n:
            best_n = n_in
            best_normal = n
            best_d = d

    if best_normal is None or best_n < min_inliers:
        return None

    # Refit on inliers with SVD for stability.
    inlier_mask = np.abs(pts @ best_normal + best_d) < threshold
    inlier_pts = pts[inlier_mask]
    if len(inlier_pts) >= 3:
        n_ref, d_ref = _fit_plane_svd(inlier_pts)
        inlier_mask = np.abs(pts @ n_ref + d_ref) < threshold

    if int(inlier_mask.sum()) < min_inliers:
        return None
    return inlier_mask


# Probability that at least one of K samples is all-inlier, used by the
# adaptive stopping rule (Fischler & Bolles 1981; Hartley & Zisserman §4.7.1).
_RANSAC_ADAPTIVE_CONFIDENCE = 0.99
# Per-chunk hypothesis count chosen so the (M, K) distance matrix in
# float32 fits in L2/L3 cache. The bottleneck is the matrix's allocate +
# sum-reduce, both bandwidth-bound, so chunks that overflow cache get
# *slower*, not faster, despite doing fewer numpy calls.
#
# Empirical sweep (box-room synthetic, room_envelopes default 1000 iters):
#   M=33k:  best chunk≈64  (chunk×M ≈ 2.1M, ≈8 MB)
#   M=132k: best chunk≈16  (chunk×M ≈ 2.1M, ≈8 MB)
#   M=198k: best chunk≈16  (chunk×M ≈ 3.2M, ≈12 MB)
# An 8 MB target matches the (chunk × M × 4-byte) sweet spot.
_RANSAC_VECTORIZED_TARGET_BYTES = 8 * 1024 * 1024  # 8 MB
_RANSAC_VECTORIZED_MAX_CHUNK = 64
_RANSAC_VECTORIZED_MIN_CHUNK = 8


def _adaptive_k_target(best_n: int, M: int, k_cap: int,
                       confidence: float = _RANSAC_ADAPTIVE_CONFIDENCE) -> int:
    """Standard RANSAC adaptive iteration count.

    Returns the smallest K such that the probability of having drawn at least
    one all-inlier triple is ``>= confidence``, assuming inlier ratio
    ``w = best_n / M``. Capped at ``k_cap`` (the user's --ransac-iters), and
    never returns less than 1.
    """
    if best_n <= 0 or M <= 0:
        return k_cap
    w = best_n / M
    if w >= 1.0:
        return 1
    w3 = w * w * w
    if w3 <= 0.0:
        return k_cap
    # log(1 - w^3) is strictly negative for w in (0, 1).
    denom = np.log1p(-w3)
    if denom >= 0.0:  # numerical guard; shouldn't happen for w in (0, 1)
        return k_cap
    k_needed = int(np.ceil(np.log(1.0 - confidence) / denom))
    return max(1, min(k_cap, k_needed))


def _ransac_single_plane_vectorized(pts: np.ndarray,
                                    threshold: float,
                                    n_iters: int,
                                    min_inliers: int,
                                    rng: np.random.Generator
                                    ) -> np.ndarray | None:
    """Vectorized + adaptive variant of :func:`_ransac_single_plane`.

    Behavioural equivalence to the scalar path:
      * Same hypothesis model (3-point cross-product), threshold rule, and
        degenerate-normal cutoff (1e-10).
      * Same SVD refit + ``min_inliers`` gate after the loop.
      * Same "best wins on strict inequality" rule (ties keep the earlier
        chunk's hypothesis, matching the scalar `if n_in > best_n` order).

    Speedups:
      1. K hypotheses scored in one ``pts @ N.T`` matmul, chunked to bound
         the (M, K_chunk) float32 distance matrix at ~64 MB.
      2. Adaptive iteration cap: after each chunk, recompute the iteration
         budget from the best inlier ratio; exit early once met. Confidence
         is fixed at 0.99 to match the canonical Fischler-Bolles formula.

    The RNG draw layout differs from the scalar path (we draw ``chunk * 3``
    integers at a time), so byte-identical reproducibility against the scalar
    path is **not** guaranteed, only logical equivalence.
    """
    M = len(pts)
    if M < 3:
        return None
    if n_iters <= 0:
        return None

    # Chunk sized to keep the (M, chunk) float32 distance matrix below the
    # target byte budget. Clamped on both sides.
    bytes_per_col = M * 4  # float32
    chunk_size = max(_RANSAC_VECTORIZED_MIN_CHUNK,
                     min(_RANSAC_VECTORIZED_MAX_CHUNK,
                         _RANSAC_VECTORIZED_TARGET_BYTES // max(bytes_per_col, 1)))

    best_n = 0
    best_normal: np.ndarray | None = None
    best_d = 0.0

    processed = 0
    k_target = n_iters

    while processed < k_target:
        take = int(min(chunk_size, k_target - processed))
        if take <= 0:
            break
        # Sample `take` hypotheses' 3 point indices (with replacement, like
        # the scalar path's `rng.integers(0, M, 3)`).
        idx = rng.integers(0, M, size=(take, 3))
        p0 = pts[idx[:, 0]]
        p1 = pts[idx[:, 1]]
        p2 = pts[idx[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)  # (take, 3)
        norms = np.linalg.norm(normals, axis=1)  # (take,)
        valid = norms > 1e-10
        processed += take  # count degenerate samples toward the budget (matches scalar)
        if not valid.any():
            continue

        n_valid = normals[valid] / norms[valid, None]  # (Kv, 3)
        p0_valid = p0[valid]
        # d such that n · p0 + d = 0 → d = -n · p0
        d_valid = -np.einsum('ij,ij->i', n_valid, p0_valid)  # (Kv,)

        # Distance matrix |pts @ N.T + d| → (M, Kv).
        dist = np.abs(pts @ n_valid.T + d_valid[None, :])
        n_in = (dist < threshold).sum(axis=0)  # (Kv,)

        # Strict-inequality "best wins" matches the scalar loop's ordering:
        # earliest hypothesis with the max wins ties.
        chunk_best = int(np.argmax(n_in))
        chunk_best_n = int(n_in[chunk_best])
        if chunk_best_n > best_n:
            best_n = chunk_best_n
            best_normal = n_valid[chunk_best].astype(pts.dtype, copy=True)
            best_d = float(d_valid[chunk_best])

        # Adaptive cap: shrink k_target as soon as a strong hypothesis appears.
        k_target = min(k_target, _adaptive_k_target(best_n, M, n_iters))

    if best_normal is None or best_n < min_inliers:
        return None

    # SVD refit on inliers, identical to scalar path.
    inlier_mask = np.abs(pts @ best_normal + best_d) < threshold
    inlier_pts = pts[inlier_mask]
    if len(inlier_pts) >= 3:
        n_ref, d_ref = _fit_plane_svd(inlier_pts)
        inlier_mask = np.abs(pts @ n_ref + d_ref) < threshold

    if int(inlier_mask.sum()) < min_inliers:
        return None
    return inlier_mask


def fit_ransac_envelope(points: np.ndarray, *,
                        max_planes: int = 6,
                        thresh: float = 0.03,
                        min_inliers: int = 500,
                        max_iters: int = 1000,
                        seed: int = 42,
                        extent_quantiles: tuple[float, float] = (0.01, 0.99),
                        vectorized: bool = False,
                        ) -> list[dict]:
    """Sequentially fit up to ``max_planes`` planes in ``points`` (N, 3).

    Removes inliers between fits. Returns plane dicts as documented at the
    top of this module. Each plane's ``inlier_mask`` references the original
    ``points`` array (not the remaining-after-removal subset).

    ``vectorized=True`` routes the inner per-iteration loop through
    :func:`_ransac_single_plane_vectorized`, which batches K hypotheses
    through one matmul and applies an adaptive iteration cap (Fischler-Bolles
    p=0.99). Output planes are logically equivalent to the scalar path
    (same SVD refit, same ``min_inliers`` gate) but RNG draw order differs,
    so individual ``inlier_mask`` arrays are not byte-identical.
    """
    rng = np.random.default_rng(seed)
    N = len(points)
    if N < min_inliers:
        return []

    remaining_idx = np.arange(N)
    remaining = points
    planes: list[dict] = []

    _fit_one = (_ransac_single_plane_vectorized
                if vectorized else _ransac_single_plane)

    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        local_mask = _fit_one(
            remaining, thresh, max_iters, min_inliers, rng,
        )
        if local_mask is None:
            break

        global_mask = np.zeros(N, dtype=bool)
        global_mask[remaining_idx[local_mask]] = True
        inliers = points[global_mask]

        planes.append(_make_plane_record(inliers, global_mask, N, extent_quantiles))

        keep = ~local_mask
        remaining = remaining[keep]
        remaining_idx = remaining_idx[keep]

    return planes


# ---------------------------------------------------------------------------
# Manhattan snapping (port + augment of vggt_layout_baselines/geometry/manhattan.py)
# ---------------------------------------------------------------------------

def _cluster_normals_greedy(normals: list[np.ndarray],
                             angle_tol_deg: float = 20.0) -> list[np.ndarray]:
    """Cluster unit normals by angular similarity (n and -n are merged).

    Returns one representative direction per cluster.
    """
    tol_cos = np.cos(np.deg2rad(angle_tol_deg))
    clusters: list[list[np.ndarray]] = []
    for n in normals:
        assigned = False
        for c in clusters:
            rep = np.mean(c, axis=0)
            rep = rep / (np.linalg.norm(rep) + 1e-10)
            if abs(float(n @ rep)) > tol_cos:
                c.append(n if float(n @ rep) > 0 else -n)
                assigned = True
                break
        if not assigned:
            clusters.append([n.copy()])
    out: list[np.ndarray] = []
    for c in clusters:
        rep = np.mean(c, axis=0)
        nrm = np.linalg.norm(rep)
        if nrm > 1e-8:
            out.append((rep / nrm).astype(np.float32))
    return out


def estimate_manhattan_basis(planes: list[dict], *,
                              angle_tol_deg: float = 20.0,
                              ortho_tol_deg: float = 70.0
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Greedy 3-direction basis discovery from input plane normals.

    Algorithm: cluster normals by angular similarity, pick the most-supported
    cluster as ``e1``; then pick the next cluster mostly perpendicular to e1
    as ``e2`` (within ``ortho_tol_deg`` of 90°); then set ``e3 = e1 × e2``
    and re-orthogonalise ``e2 ← e3 × e1``.

    Returns ``(e1, e2, e3)`` unit vectors, or ``None`` if fewer than 2
    perpendicular clusters are found.
    """
    if not planes:
        return None
    # Order by support so larger surfaces dominate the basis.
    sorted_planes = sorted(planes, key=lambda p: -p["n_inliers"])
    normals = [p["normal"] / (np.linalg.norm(p["normal"]) + 1e-12)
               for p in sorted_planes]
    clusters = _cluster_normals_greedy(normals, angle_tol_deg)
    if not clusters:
        return None

    e1 = clusters[0]
    e2 = None
    cos_orth = np.cos(np.deg2rad(ortho_tol_deg))
    for c in clusters[1:]:
        if abs(float(c @ e1)) < cos_orth:
            e2 = c
            break
    if e2 is None:
        return None

    e3 = np.cross(e1, e2)
    e3 = e3 / (np.linalg.norm(e3) + 1e-12)
    e2 = np.cross(e3, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return e1.astype(np.float32), e2.astype(np.float32), e3.astype(np.float32)


def _snap_normal_to_basis(normal: np.ndarray,
                          basis: tuple[np.ndarray, np.ndarray, np.ndarray]
                          ) -> np.ndarray:
    """Snap a normal to the nearest of {±e1, ±e2, ±e3} (largest |dot|)."""
    e1, e2, e3 = basis
    dots = np.array([float(normal @ e1), float(normal @ e2), float(normal @ e3)])
    k = int(np.argmax(np.abs(dots)))
    sign = 1.0 if dots[k] >= 0 else -1.0
    snapped = sign * np.stack([e1, e2, e3])[k]
    return snapped.astype(np.float32)


def snap_to_manhattan(planes: list[dict],
                      points: np.ndarray,
                      *,
                      angle_tol_deg: float = 20.0,
                      merge_tol: float = 0.06,
                      extent_quantiles: tuple[float, float] = (0.01, 0.99)
                      ) -> tuple[list[dict], dict]:
    """Snap each plane's normal to a per-scene Manhattan basis, then merge
    near-duplicates.

    Implements fixes #4 and #5 from the plan:
      - After snapping the normal, *everything* downstream of the normal is
        recomputed from the plane's original inliers under the snapped normal
        (``d``, ``u``, ``v``, ``centroid``, ``extent_xy``).
      - When merging planes that share a snapped normal and have close ``d``
        values, the merged plane is *re-fit from the union of inlier points*,
        not naively unioned in extent space.

    Returns ``(snapped_planes, status)``. Status keys:
      - ``manhattan_status``: ``"ok"`` if the basis was found, else
        ``"fallback_raw_pred"`` (and ``snapped_planes`` is empty).
      - ``n_input``, ``n_after_snap``, ``n_after_merge``.
      - ``basis``: shape-(3,3) array of (e1, e2, e3) or ``None``.
    """
    if not planes:
        return [], {"manhattan_status": "fallback_raw_pred",
                    "n_input": 0, "n_after_snap": 0, "n_after_merge": 0,
                    "basis": None}

    basis = estimate_manhattan_basis(planes, angle_tol_deg=angle_tol_deg)
    if basis is None:
        return [], {"manhattan_status": "fallback_raw_pred",
                    "n_input": len(planes), "n_after_snap": 0,
                    "n_after_merge": 0, "basis": None}

    N_total = len(points)

    # Step 1: snap normals + re-derive everything per plane from its inliers.
    snapped: list[dict] = []
    for p in planes:
        n_snap = _snap_normal_to_basis(p["normal"], basis)
        mask = p["inlier_mask"]
        inliers = points[mask]
        if len(inliers) < 3:
            continue
        snapped.append(_make_plane_record(inliers, mask, N_total, extent_quantiles))
        # _make_plane_record uses SVD on the inliers, which would re-discover
        # a non-snapped normal. We override with the snapped normal here, and
        # re-compute d / u / v / extent under the snapped normal.
        u_snap, v_snap = _in_plane_basis(n_snap)
        centroid = snapped[-1]["centroid"]
        d_snap = -float(n_snap @ centroid)
        xmin, xmax, ymin, ymax = _compute_extent(inliers, centroid,
                                                  u_snap, v_snap, extent_quantiles)
        residuals = np.abs(inliers @ n_snap + d_snap)
        snapped[-1].update({
            "normal":        n_snap,
            "d":             d_snap,
            "u":             u_snap,
            "v":             v_snap,
            "mean_residual": float(residuals.mean()),
            "extent_xy":     (float(xmin), float(xmax), float(ymin), float(ymax)),
        })

    # Step 2: merge planes whose snapped normals are identical and whose d
    # values are within merge_tol. Merging is done by unioning inlier masks
    # and re-running the same plane-record computation on the merged set,
    # so the merged centroid / u / v / extent are coherent (fix #5).
    merged: list[dict] = []
    used = [False] * len(snapped)
    for i in range(len(snapped)):
        if used[i]:
            continue
        group_mask = snapped[i]["inlier_mask"].copy()
        used[i] = True
        for j in range(i + 1, len(snapped)):
            if used[j]:
                continue
            if float(snapped[i]["normal"] @ snapped[j]["normal"]) < 0.999:
                continue  # different snapped axis (or opposite side)
            if abs(snapped[i]["d"] - snapped[j]["d"]) <= merge_tol:
                group_mask = group_mask | snapped[j]["inlier_mask"]
                used[j] = True
        inliers = points[group_mask]
        if len(inliers) < 3:
            continue
        n_snap = snapped[i]["normal"]
        u_snap, v_snap = _in_plane_basis(n_snap)
        centroid = inliers.mean(axis=0).astype(np.float32)
        d_snap = -float(n_snap @ centroid)
        xmin, xmax, ymin, ymax = _compute_extent(inliers, centroid,
                                                  u_snap, v_snap, extent_quantiles)
        residuals = np.abs(inliers @ n_snap + d_snap)
        n_in = int(group_mask.sum())
        merged.append({
            "normal":        n_snap,
            "d":             d_snap,
            "n_inliers":     n_in,
            "inlier_ratio":  float(n_in / max(N_total, 1)),
            "mean_residual": float(residuals.mean()),
            "inlier_mask":   group_mask,
            "u":             u_snap,
            "v":             v_snap,
            "centroid":      centroid,
            "extent_xy":     (float(xmin), float(xmax), float(ymin), float(ymax)),
        })

    return merged, {
        "manhattan_status": "ok",
        "n_input":         len(planes),
        "n_after_snap":    len(snapped),
        "n_after_merge":   len(merged),
        "basis":           np.stack(basis, axis=0).astype(np.float32),
    }


__all__ = [
    "fit_ransac_envelope",
    "snap_to_manhattan",
    "estimate_manhattan_basis",
]
