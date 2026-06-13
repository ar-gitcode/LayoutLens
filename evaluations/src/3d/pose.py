"""Predicted-pose decoding (pose_enc) and pose-error metrics."""
from __future__ import annotations

import numpy as np

from alignment import _camera_center_from_w2c, umeyama_similarity


def decode_pred_pose_enc(pose_enc, image_hw):
    """Decode VGGT ``pose_enc`` (B,S,9) → (extr_pred [B,S,3,4], intr_pred [B,S,3,3]).

    The pose encoding format is the standard VGGT ``absT_quaR_FoV``:
      - [..., :3]  absolute translation T
      - [..., 3:7] unit quaternion (xyzw, the convention used by VGGT)
      - [..., 7:]  field of view (fov_h, fov_w), both in radians
    """
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    pose = pose_enc if torch.is_tensor(pose_enc) else torch.as_tensor(pose_enc)
    if pose.dim() == 2:                   # (S, 9) → add batch dim
        pose = pose.unsqueeze(0)
    extr, intr = pose_encoding_to_extri_intri(
        pose.float(), image_hw, pose_encoding_type="absT_quaR_FoV"
    )
    return extr.cpu().numpy(), intr.cpu().numpy()


def compute_pose_error_metrics(
    extr_pred_w2c: np.ndarray,
    extr_gt_w2c: np.ndarray,
    intr_pred: np.ndarray | None = None,
    intr_gt: np.ndarray | None = None,
    image_hw: tuple[int, int] | None = None,
) -> dict:
    """Per-scene pose-vs-GT error metrics for predicted cameras.

    VGGT's camera head is trained against absolute GT pose encoding, so
    per-frame rotations are directly comparable without world-frame alignment.
    Per-frame translations are reported both raw and after a Sim(3) Umeyama
    alignment of camera centres (the standard ATE convention).

    Returns a flat dict; intrinsics-related keys are omitted when not supplied.
    Aligned translation (ATE) requires S >= 3 non-colinear cameras to be
    meaningful; with S < 3 we still report ate_m as the residual after a fit
    on whatever points are available.
    """
    extr_pred = np.asarray(extr_pred_w2c, dtype=np.float64)
    extr_gt   = np.asarray(extr_gt_w2c,   dtype=np.float64)
    if extr_pred.shape[-2:] == (4, 4):
        extr_pred = extr_pred[..., :3, :]
    if extr_gt.shape[-2:] == (4, 4):
        extr_gt = extr_gt[..., :3, :]
    S = int(extr_pred.shape[0])
    out: dict = {"n_frames": S}

    rot_errs = np.empty(S, dtype=np.float64)
    for fr in range(S):
        R_rel = extr_pred[fr, :3, :3] @ extr_gt[fr, :3, :3].T
        cos_th = (np.trace(R_rel) - 1.0) / 2.0
        rot_errs[fr] = np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))
    out["rot_err_deg_mean"]   = float(np.mean(rot_errs))
    out["rot_err_deg_median"] = float(np.median(rot_errs))
    out["rot_err_deg_p95"]    = float(np.percentile(rot_errs, 95))

    pred_C = np.stack([_camera_center_from_w2c(extr_pred[s]) for s in range(S)], axis=0)
    gt_C   = np.stack([_camera_center_from_w2c(extr_gt[s])   for s in range(S)], axis=0)
    raw_trans = np.linalg.norm(pred_C - gt_C, axis=1)
    out["trans_err_raw_m_mean"]   = float(np.mean(raw_trans))
    out["trans_err_raw_m_median"] = float(np.median(raw_trans))
    out["trans_err_raw_m_p95"]    = float(np.percentile(raw_trans, 95))

    s_um, R_um, t_um = umeyama_similarity(pred_C, gt_C, with_scale=True)
    aligned_C = (s_um * (R_um @ pred_C.T)).T + t_um
    residuals = np.linalg.norm(aligned_C - gt_C, axis=1)
    out["ate_m"] = float(np.sqrt((residuals ** 2).mean()))
    out["sim3_scale"] = float(s_um)

    if intr_pred is not None and intr_gt is not None:
        Kp = np.asarray(intr_pred, dtype=np.float64).reshape(S, 3, 3)
        Kg = np.asarray(intr_gt,   dtype=np.float64).reshape(S, 3, 3)
        out["focal_err_px_fx_mean"] = float(np.abs(Kp[:, 0, 0] - Kg[:, 0, 0]).mean())
        out["focal_err_px_fy_mean"] = float(np.abs(Kp[:, 1, 1] - Kg[:, 1, 1]).mean())
        if image_hw is not None:
            H_img, W_img = int(image_hw[0]), int(image_hw[1])
            fov_h_p = 2.0 * np.degrees(np.arctan(W_img / (2.0 * Kp[:, 0, 0])))
            fov_h_g = 2.0 * np.degrees(np.arctan(W_img / (2.0 * Kg[:, 0, 0])))
            fov_v_p = 2.0 * np.degrees(np.arctan(H_img / (2.0 * Kp[:, 1, 1])))
            fov_v_g = 2.0 * np.degrees(np.arctan(H_img / (2.0 * Kg[:, 1, 1])))
            out["fov_h_err_deg_mean"] = float(np.abs(fov_h_p - fov_h_g).mean())
            out["fov_v_err_deg_mean"] = float(np.abs(fov_v_p - fov_v_g).mean())
    return out

