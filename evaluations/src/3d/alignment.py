"""Pose / point-cloud alignment: Umeyama Sim(3), scale-only, scale-shift."""
from __future__ import annotations

import numpy as np


def _camera_center_from_w2c(extr_3x4: np.ndarray) -> np.ndarray:
    """C = -R^T t for a camera-from-world (w2c) extrinsic."""
    R = extr_3x4[:3, :3]
    t = extr_3x4[:3, 3]
    return -R.T @ t


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Estimate similarity transform (s, R, t) mapping ``src`` → ``dst`` (least-squares).

    Implements the algorithm from Umeyama (1991): "Least-squares estimation of
    transformation parameters between two point patterns".

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points.
        with_scale: if False, restrict to rigid (scale fixed at 1.0).

    Returns:
        (s, R, t) such that  aligned = s * (R @ src.T).T + t  approximates dst.
        Returns identity (s=1, R=I, t=0) if N < 3.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    # Cross-covariance dst x src (note the order: rows = dst, cols = src).
    Sigma = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / max(var_src, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_pred_cameras_to_gt(extr_pred_w2c: np.ndarray,
                             extr_gt_w2c: np.ndarray,
                             with_scale: bool = True):
    """Rigid + scale alignment of predicted cameras to GT cameras.

    Takes per-scene per-frame extrinsics (S, 3, 4) in camera-from-world
    (OpenCV w2c) convention, computes camera centres in world frame for
    both, fits a similarity transform mapping pred-world → gt-world, and
    rewrites each predicted w2c extrinsic so its camera centre lands on
    the corresponding gt camera centre when reprojected through the new
    world frame.

    Convention chosen so the scale factor `s` is exposed separately
    (caller multiplies predicted depth by ``s`` when re-backprojecting):

      Umeyama on camera centres returns ``(s, R, t)`` such that
      ``s · R · C_pred + t ≈ C_gt``  (i.e. similarity pred-world → gt-world).

      Each predicted (R_c, t_c) is replaced with

        R_new = R_c · Rᵀ
        t_new = s · t_c  −  R_c · Rᵀ · t

      so that the new camera centre is exactly ``C_gt = s · R · C_pred + t``
      and the depth value at each pixel under the rebuilt cloud equals
      ``s · d_pred`` (i.e. the caller scales pred depth by `s`).

    Returns ``(aligned_extr (S,3,4), s)``.
    """
    extr_pred_w2c = np.asarray(extr_pred_w2c)
    extr_gt_w2c = np.asarray(extr_gt_w2c)
    S = extr_pred_w2c.shape[0]

    pred_C = np.stack([_camera_center_from_w2c(extr_pred_w2c[s_i]) for s_i in range(S)], axis=0)
    gt_C   = np.stack([_camera_center_from_w2c(extr_gt_w2c[s_i])   for s_i in range(S)], axis=0)
    s, R, t = umeyama_similarity(pred_C, gt_C, with_scale=with_scale)

    aligned = np.zeros_like(extr_pred_w2c, dtype=np.float64)
    RT = R.T  # (3,3)
    for fr in range(S):
        Rc = extr_pred_w2c[fr, :3, :3]
        tc = extr_pred_w2c[fr, :3, 3]
        new_R = Rc @ RT
        new_t = s * tc - Rc @ RT @ t
        aligned[fr, :3, :3] = new_R
        aligned[fr, :3, 3] = new_t
    return aligned.astype(extr_pred_w2c.dtype), float(s)


def align_pointcloud_scale_only(pred_points: np.ndarray,
                                gt_points: np.ndarray,
                                mode: str = "pointcloud_rms"):
    """Scale-only alignment of a fused predicted cloud to a fused GT cloud.

    Geometrically cleaner alignment is to scale predicted depth maps along
    camera rays before fusion. Use this fallback only when per-view depth maps
    are not available.

    Args:
        pred_points: (N, 3) predicted cloud.
        gt_points:   (M, 3) GT cloud.
        mode:        ``"pointcloud_rms"`` (only mode supported). Centres each
            cloud, fits a single uniform scale via ``rms_gt / rms_pred``,
            then translates the rescaled pred cloud to the GT centroid.

    Returns:
        ``(aligned_points, scale)``, ``scale=NaN`` and unscaled points if
        either cloud is too small or degenerate.
    """
    if mode != "pointcloud_rms":
        raise ValueError(f"unknown scale-only mode: {mode!r}")

    p = np.asarray(pred_points, dtype=np.float64)
    g = np.asarray(gt_points, dtype=np.float64)
    if p.ndim != 2 or g.ndim != 2 or p.shape[0] < 3 or g.shape[0] < 3:
        return p.astype(np.float32), float("nan")

    p_c = p.mean(axis=0)
    g_c = g.mean(axis=0)
    rms_p = float(np.sqrt(((p - p_c) ** 2).sum(axis=1).mean()))
    rms_g = float(np.sqrt(((g - g_c) ** 2).sum(axis=1).mean()))
    if rms_p < 1e-9:
        return p.astype(np.float32), float("nan")
    s = rms_g / rms_p
    aligned = (p - p_c) * s + g_c
    return aligned.astype(np.float32), float(s)


def fit_xyz_scale_zshift_lstsq(P: np.ndarray,
                               G: np.ndarray,
                               min_pairs: int = 100):
    """Joint LSQ for the Room-Envelopes / LaRI scale-shift alignment.

    Solves ``min_{s, t_z}  Σ_i || s · P_i  +  (0, 0, t_z) − G_i ||²`` in
    closed form. ``P`` and ``G`` are pixel-corresponded 3-D points
    (``P_i`` and ``G_i`` are the same pixel's predicted and GT world
    points). The transform has a single uniform scale on xyz and a single
    z-translation, this is the alignment described by Bahrami & Campbell
    (Room Envelopes, arXiv:2511.03970) and used by LaRI's
    ``scale_shift_inv_alignment_inverse``.

    Closed form via the 2×2 normal equations on x = [s, t_z]:

        A_i = [[P_xi, 0], [P_yi, 0], [P_zi, 1]]
        b_i = [G_xi, G_yi, G_zi]
        A.T @ A · x = A.T @ b

    which expands to

        [[Σ ||P_i||²,  Σ P_zi];   [s]    [Σ P_i · G_i;]
         [Σ P_zi,     N        ]] [t_z] = [Σ G_zi      ]

    Args:
        P: (N, 3) predicted points.
        G: (N, 3) GT points, corresponded 1:1 with ``P``.
        min_pairs: minimum number of finite pairs required.

    Returns:
        ``(s, t_z, ok, reason, num_pairs)``, ``s, t_z`` floats (NaN on
        failure), ``ok`` bool, ``reason`` short string ("ok" on success),
        ``num_pairs`` pair count actually used after dropping non-finite
        rows. Negative or non-finite ``s`` is treated as a failure.
    """
    P = np.asarray(P, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape != G.shape:
        return (float("nan"), float("nan"), False, "pair_shape_mismatch",
                int(P.shape[0]) if P.ndim == 2 else 0)

    keep = np.isfinite(P).all(axis=1) & np.isfinite(G).all(axis=1)
    P = P[keep]
    G = G[keep]
    n = int(P.shape[0])
    if n < int(min_pairs):
        return float("nan"), float("nan"), False, f"too_few_pairs:{n}<{min_pairs}", n

    Q     = float((P * P).sum())                # Σ ||P_i||²
    Sdot  = float((P * G).sum())                # Σ P_i · G_i
    mu_Pz = float(P[:, 2].sum())
    mu_Gz = float(G[:, 2].sum())

    det = Q * n - mu_Pz * mu_Pz
    if not np.isfinite(det) or abs(det) < 1e-12:
        return float("nan"), float("nan"), False, "degenerate_normal_equations", n

    s   = (Sdot * n - mu_Gz * mu_Pz) / det
    t_z = (Q * mu_Gz - Sdot * mu_Pz) / det
    if not np.isfinite(s) or s <= 0.0:
        return float("nan"), float("nan"), False, f"bad_scale:{s}", n
    if not np.isfinite(t_z):
        return float("nan"), float("nan"), False, f"bad_shift:{t_z}", n
    return float(s), float(t_z), True, "ok", n

