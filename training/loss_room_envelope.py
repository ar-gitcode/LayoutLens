"""Room-envelope multi-task loss for layout-depth + binary mask training.

Enabled losses are controlled by per-task weight arguments.  A weight of 0.0
means the loss is skipped entirely (no forward computation).  If a weight > 0
but the required batch/prediction key is missing, a ValueError is raised early
so misconfigured experiments fail fast rather than silently.

Phase 1 losses:
  layout_l1: masked L1 on layout depth
  layout_silog: scale-invariant log loss on layout depth
  layout_gradient: multi-scale log-depth gradient loss
  mask_bce: BCEWithLogitsLoss on binary layout mask logits
  mask_dice: Dice loss on binary layout mask (optional)

Phase 2 losses:
  normal_consistency: cosine loss between depth-derived normals (pred vs GT)
  normal_head: direct cosine supervision on the predicted normal head
  planar_consistency: proxy local-planarity prior that penalises the spatial
                       variation (total variation) of the surface normals
                       derived from the *predicted* layout depth, over valid
                       layout pixels. Encourages locally planar geometry
                       without any GT planes/normals (orthographic normal
                       model). Disabled by default (weight 0.0).

Phase 2 (E7), wired in this module:
  visible_depth: scale-shift-aligned L1 + multi-scale log-gradient on the
                  original VGGT ``depth_head`` output vs ``batch['depths']``,
                  masked by ``batch['point_masks']``.
  camera: translation + rotation + FoV loss on ``pose_enc_list``
                  (delegates to :func:`training.loss.compute_camera_loss`).

Still placeholder (no-op):
  pointmap

VGGT-original ablation paths (E14 / E15):
  Setting ``camera_loss_type="vggt_original"`` swaps the bespoke ``_camera_loss``
  for :func:`vggt_compute_camera_loss` below, a verbatim copy of the canonical
  ``MultitaskLoss`` camera loss from the upstream VGGT training code.
  Setting ``layout_depth_loss_type="vggt_original"`` disables the custom
  L1+SILog+log-grad path and instead calls :func:`vggt_compute_layout_depth_loss`
  (confidence-weighted L2 + multi-scale ``grad`` loss). Requires the layout-depth
  head to emit ``layout_depth_conf`` (true for ``DPTHead(output_dim=2)``, the
  default in ``vggt.models.vggt.VGGT`` when ``enable_layout_depth=True``).
"""

from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Normal computation helper (module-level, reusable from eval scripts)
# --------------------------------------------------------------------------- #

def compute_normals_from_depth(depth: torch.Tensor,
                                valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Compute surface normals from a depth map using central finite differences.

    Simple gradient-based method (stable default).  Does NOT require intrinsics.
    The normal approximation is (-∂z/∂x, -∂z/∂y, 2), normalized, which gives the
    outward-facing normal under an orthographic approximation.  For room-scale
    depth the approximation is close enough for a consistency loss.

    Args:
        depth:      (B, S, H, W) float32, metres.  0 = invalid.
        valid_mask: (B, S, H, W) bool, optional.  If None, derived from depth > 0.

    Returns:
        normals: (B, S, 3, H, W) float32, unit-length where valid; 0 elsewhere.
    """
    if valid_mask is None:
        valid_mask = depth > 1e-6

    B, S, H, W = depth.shape

    # Central differences; interior pixels only
    # dx[i,j] = depth[i, j+1] - depth[i, j-1]
    dx = torch.zeros_like(depth)
    dy = torch.zeros_like(depth)
    dx[..., 1:-1] = depth[..., 2:] - depth[..., :-2]
    dy[..., 1:-1, :] = depth[..., 2:, :] - depth[..., :-2, :]

    # A pixel's normal is valid only if the pixel itself and its 4 neighbours are valid
    v_right  = valid_mask[..., 2:]
    v_left   = valid_mask[..., :-2]
    v_bot    = valid_mask[..., 2:, :]
    v_top    = valid_mask[..., :-2, :]

    valid_x = torch.zeros_like(valid_mask)
    valid_y = torch.zeros_like(valid_mask)
    valid_x[..., 1:-1] = v_right & v_left & valid_mask[..., 1:-1]
    valid_y[..., 1:-1, :] = v_bot & v_top & valid_mask[..., 1:-1, :]
    valid_n = valid_x & valid_y

    # Normal vector in image space: (-dx, -dy, 2)
    nx = -dx
    ny = -dy
    nz = torch.full_like(depth, 2.0)

    norm = (nx.pow(2) + ny.pow(2) + nz.pow(2)).clamp(min=1e-8).sqrt()
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    # Zero out invalid pixels
    nx = nx * valid_n
    ny = ny * valid_n
    nz = nz * valid_n

    # Stack → (B, S, 3, H, W)
    return torch.stack([nx, ny, nz], dim=2)


class RoomEnvelopeLoss(nn.Module):
    """Multi-task loss for room-envelope reconstruction.

    All weights default to the recommended Phase-1 values; set to 0.0 to disable.
    The trainer reads ``objective`` for backprop and logs every individual key.
    """

    def __init__(
        self,
        layout_l1_weight: float = 1.0,
        layout_silog_weight: float = 0.5,
        gradient_weight: float = 0.1,
        binary_mask_weight: float = 0.05,
        binary_dice_weight: float = 0.0,
        # Phase 2
        normal_consistency_weight: float = 0.0,
        normal_head_weight: float = 0.0,
        visible_depth_weight: float = 0.0,
        pointmap_weight: float = 0.0,
        camera_weight: float = 0.0,
        # Planar-consistency prior (local planarity on the predicted layout
        # depth). PROXY term: penalises the spatial variation of the surface
        # normals derived from the predicted layout depth over valid layout
        # pixels (no GT planes/normals required). 0.0 = disabled, which keeps
        # every existing experiment byte-identical. `edge_aware` down-weights
        # genuine GT layout creases (wall/floor corners, occlusion edges) via
        # exp(-beta * ||Δ gt_normal||), gated at the same stencil as the penalty
        # so the prior flattens within planes. See `_planar_consistency`.
        planar_consistency_weight: float = 0.0,
        planar_consistency_edge_aware: bool = True,
        planar_consistency_edge_beta: float = 4.0,
        # Ablation toggles, swap individual sub-losses for the canonical VGGT
        # implementations copied verbatim at the bottom of this file. Defaults
        # preserve existing E1-E11 behaviour exactly.
        camera_loss_type: str = "room_envelope",
        layout_depth_loss_type: str = "room_envelope",
        # Knobs for the VGGT-original layout-depth path (ignored unless
        # layout_depth_loss_type == "vggt_original"). Defaults mirror the
        # canonical `loss.depth` block from the upstream VGGT training config.
        vggt_layout_depth_weight: float = 1.0,
        vggt_layout_depth_gradient_loss_fn: str = "grad",
        vggt_layout_depth_valid_range: float = 0.98,
        # H4: per-pixel weight applied to the regression and confidence
        # terms of the VGGT-original layout-depth loss. 1.0 = behaviour-
        # neutral (existing runs are byte-identical). Values > 1.0
        # up-weight clutter pixels (layout_masks < 0.5).
        clutter_lambda: float = 1.0,
    ):
        super().__init__()
        self.layout_l1_weight         = layout_l1_weight
        self.layout_silog_weight      = layout_silog_weight
        self.gradient_weight          = gradient_weight
        self.binary_mask_weight       = binary_mask_weight
        self.binary_dice_weight       = binary_dice_weight
        self.normal_consistency_weight = normal_consistency_weight
        self.normal_head_weight       = normal_head_weight
        self.visible_depth_weight     = visible_depth_weight
        self.pointmap_weight          = pointmap_weight
        self.camera_weight            = camera_weight
        self.planar_consistency_weight     = planar_consistency_weight
        self.planar_consistency_edge_aware = bool(planar_consistency_edge_aware)
        self.planar_consistency_edge_beta  = float(planar_consistency_edge_beta)

        if camera_loss_type not in ("room_envelope", "vggt_original"):
            raise ValueError(
                f"camera_loss_type must be 'room_envelope' or 'vggt_original', "
                f"got {camera_loss_type!r}"
            )
        if layout_depth_loss_type not in ("room_envelope", "vggt_original"):
            raise ValueError(
                f"layout_depth_loss_type must be 'room_envelope' or "
                f"'vggt_original', got {layout_depth_loss_type!r}"
            )
        self.camera_loss_type       = camera_loss_type
        self.layout_depth_loss_type = layout_depth_loss_type
        self.vggt_layout_depth_weight              = vggt_layout_depth_weight
        self.vggt_layout_depth_gradient_loss_fn    = vggt_layout_depth_gradient_loss_fn
        self.vggt_layout_depth_valid_range         = vggt_layout_depth_valid_range
        self.clutter_lambda                        = float(clutter_lambda)

    # ------------------------------------------------------------------ #
    # Public forward
    # ------------------------------------------------------------------ #

    def forward(self, predictions: dict, batch: dict) -> dict:
        """Compute all enabled losses and return a flat dict of scalars.

        The key ``objective`` is the weighted sum and is used by the
        trainer for backprop.  All other keys are logged individually.
        """
        losses = {}
        total = torch.tensor(0.0, device=self._device(predictions, batch))

        # ---- layout depth ----
        # In "room_envelope" mode (default): custom L1 + SILog + log-grad on the
        # (B,S,H,W) layout-depth map.
        # In "vggt_original" mode: confidence-weighted L2 + multi-scale gradient
        # loss via the canonical VGGT `compute_layout_depth_loss` (see bottom of
        # this file). We still extract pred/gt/valid below so the normal-
        # consistency term can reuse them, but only when the consistency term
        # is enabled (it depends on raw layout depth, not the VGGT loss output).
        is_vggt_ld = self.layout_depth_loss_type == "vggt_original"
        need_ld_custom = (
            not is_vggt_ld
            and (self.layout_l1_weight > 0 or self.layout_silog_weight > 0
                 or self.gradient_weight > 0)
        )
        need_ld_for_consistency = (
            self.normal_consistency_weight > 0
            and "layout_depth" in predictions
            and "layout_depths" in batch
        )
        if need_ld_custom or need_ld_for_consistency:
            pred_ld, gt_ld, valid_ld = self._get_layout_depth(predictions, batch)

        if is_vggt_ld:
            # Single fused term: conf + reg + grad, weighted by one umbrella.
            ld_dict = self._vggt_layout_depth_loss(predictions, batch)
            losses.update(ld_dict)
            ld_total = (
                ld_dict["loss_conf_layout_depth"]
                + ld_dict["loss_reg_layout_depth"]
                + ld_dict["loss_grad_layout_depth"]
            )
            total = total + self.vggt_layout_depth_weight * ld_total
        else:
            if self.layout_l1_weight > 0:
                v = self._layout_l1(pred_ld, gt_ld, valid_ld)
                losses["loss_layout_l1"] = v
                total = total + self.layout_l1_weight * v

            if self.layout_silog_weight > 0:
                v = self._silog(pred_ld, gt_ld, valid_ld)
                losses["loss_layout_silog"] = v
                total = total + self.layout_silog_weight * v

            if self.gradient_weight > 0:
                v = self._gradient(pred_ld, gt_ld, valid_ld)
                losses["loss_layout_gradient"] = v
                total = total + self.gradient_weight * v

        # ---- binary mask ----
        if self.binary_mask_weight > 0 or self.binary_dice_weight > 0:
            logits, target_mask = self._get_mask(predictions, batch)

        if self.binary_mask_weight > 0:
            v = self._bce_mask(logits, target_mask)
            losses["loss_mask_bce"] = v
            total = total + self.binary_mask_weight * v
        else:
            losses["loss_mask_bce"] = torch.tensor(0.0, device=total.device)

        if self.binary_dice_weight > 0:
            v = self._dice_mask(logits, target_mask)
            losses["loss_mask_dice"] = v
            total = total + self.binary_dice_weight * v
        else:
            losses["loss_mask_dice"] = torch.tensor(0.0, device=total.device)

        # ---- normal consistency (Phase 2) ----
        if self.normal_consistency_weight > 0:
            v = self._normal_consistency(pred_ld, gt_ld, valid_ld)
            losses["loss_normal_consistency"] = v
            total = total + self.normal_consistency_weight * v
        else:
            losses["loss_normal_consistency"] = torch.tensor(0.0, device=total.device)

        # ---- normal head direct supervision (Phase 2) ----
        if self.normal_head_weight > 0:
            pred_n, gt_n, valid_n = self._get_layout_normals(predictions, batch)
            v = self._normal_head_loss(pred_n, gt_n, valid_n)
            losses["loss_normal_head"] = v
            total = total + self.normal_head_weight * v

        # ---- visible (metric) depth, used by E7 ----
        if self.visible_depth_weight > 0:
            v = self._visible_depth_loss(predictions, batch)
            losses["loss_visible_depth"] = v
            total = total + self.visible_depth_weight * v

        # ---- camera, used by E7 ----
        if self.camera_weight > 0:
            if self.camera_loss_type == "vggt_original":
                cam_dict = self._vggt_camera_loss(predictions, batch)
                v = cam_dict["loss_camera"]
                losses.update(cam_dict)
            else:
                v = self._camera_loss(predictions, batch)
                losses["loss_camera"] = v
            total = total + self.camera_weight * v

        # ---- planar consistency, local planarity prior on layout depth ----
        # Self-contained: extracts its own layout-depth pred/gt/valid (does not
        # depend on the conditionally-computed pred_ld above), so it works with
        # both the "room_envelope" and "vggt_original" layout-depth paths.
        if self.planar_consistency_weight > 0:
            v = self._planar_consistency(predictions, batch)
            losses["loss_planar_consistency"] = v
            total = total + self.planar_consistency_weight * v

        # ---- remaining placeholders (always logged as zero so the logger
        #      never KeyErrors). pointmap is still a stub. ----
        for key in ("loss_normal_head", "loss_visible_depth",
                    "loss_pointmap", "loss_camera", "loss_planar_consistency"):
            losses.setdefault(key, torch.tensor(0.0, device=total.device))

        # Fill in any missing Phase 1 loss keys so the logger never KeyErrors.
        # In "vggt_original" layout-depth mode the custom keys are absent and the
        # canonical keys (loss_conf/reg/grad_layout_depth) take their place.
        for key in ("loss_layout_l1", "loss_layout_silog", "loss_layout_gradient",
                    "loss_conf_layout_depth", "loss_reg_layout_depth",
                    "loss_grad_layout_depth"):
            losses.setdefault(key, torch.tensor(0.0, device=total.device))

        losses["objective"] = total
        return losses

    # ------------------------------------------------------------------ #
    # Loss implementations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _layout_l1(pred: torch.Tensor, gt: torch.Tensor,
                   valid: torch.Tensor) -> torch.Tensor:
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (pred - gt).abs()[valid].mean()

    @staticmethod
    def _silog(pred: torch.Tensor, gt: torch.Tensor,
               valid: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        d = torch.log(pred[valid].clamp(min=1e-6)) - torch.log(gt[valid].clamp(min=1e-6))
        return d.pow(2).mean() - lam * d.mean().pow(2)

    @staticmethod
    def _gradient(pred: torch.Tensor, gt: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
        """Multi-scale log-depth gradient L1 loss at scales 1×, 2×, 4×."""
        total_loss = torch.tensor(0.0, device=pred.device)
        n_scales = 0
        for scale in (1, 2, 4):
            p = pred
            g = gt
            v = valid
            if scale > 1:
                p = F.avg_pool2d(pred.float(), scale, stride=scale)
                g = F.avg_pool2d(gt.float(), scale, stride=scale)
                v = F.max_pool2d(valid.float(), scale, stride=scale).bool()
            if v.sum() == 0:
                continue
            log_p = torch.log(p.clamp(min=1e-6))
            log_g = torch.log(g.clamp(min=1e-6))
            # x gradient
            dp_x = log_p[..., :, 1:] - log_p[..., :, :-1]
            dg_x = log_g[..., :, 1:] - log_g[..., :, :-1]
            vx   = v[..., :, 1:] & v[..., :, :-1]
            # y gradient
            dp_y = log_p[..., 1:, :] - log_p[..., :-1, :]
            dg_y = log_g[..., 1:, :] - log_g[..., :-1, :]
            vy   = v[..., 1:, :] & v[..., :-1, :]

            scale_loss = torch.tensor(0.0, device=pred.device)
            if vx.sum() > 0:
                scale_loss = scale_loss + (dp_x - dg_x).abs()[vx].mean()
            if vy.sum() > 0:
                scale_loss = scale_loss + (dp_y - dg_y).abs()[vy].mean()
            total_loss = total_loss + scale_loss
            n_scales += 1

        return total_loss / max(n_scales, 1)

    @staticmethod
    def _bce_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, S, 1, H, W) → squeeze → (B, S, H, W)
        return F.binary_cross_entropy_with_logits(logits.squeeze(2), target)

    @staticmethod
    def _dice_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits.squeeze(2))
        intersection = (p * target).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (p.sum() + target.sum() + 1.0)

    @staticmethod
    def _normal_consistency(pred: torch.Tensor, gt: torch.Tensor,
                             valid: torch.Tensor) -> torch.Tensor:
        """Cosine loss between depth-derived normals (pred vs GT layout depth).

        If no valid pixels exist, returns a graph-safe zero on the same device.
        """
        if valid.sum() == 0:
            return pred.sum() * 0.0

        pred_n = compute_normals_from_depth(pred.clamp(min=1e-6), valid)  # (B,S,3,H,W)
        gt_n   = compute_normals_from_depth(gt.clamp(min=1e-6),   valid)  # (B,S,3,H,W)

        # Valid where both normal vectors are non-zero (i.e. pixel was valid for normals)
        pred_mag = pred_n.norm(dim=2, keepdim=True)
        gt_mag   = gt_n.norm(dim=2, keepdim=True)
        normal_valid = (pred_mag.squeeze(2) > 1e-6) & (gt_mag.squeeze(2) > 1e-6)

        if normal_valid.sum() == 0:
            return pred.sum() * 0.0

        cosine = (pred_n * gt_n).sum(dim=2)  # (B,S,H,W)
        cosine = cosine.clamp(-1.0, 1.0)
        loss = (1.0 - cosine)[normal_valid]
        return loss.mean()

    def _planar_consistency(self, predictions: dict, batch: dict) -> torch.Tensor:
        r"""Proxy planar-consistency prior on the predicted layout depth.

        Encourages the predicted layout-depth geometry to be LOCALLY PLANAR over
        valid layout regions, WITHOUT requiring ground-truth planes or normals.

        Rationale.  A locally planar surface has a locally *constant* surface
        normal, so we penalise the spatial variation (total variation) of the
        surface normals derived from the *predicted* layout depth.  Normals come
        from the existing differentiable :func:`compute_normals_from_depth`
        helper (orthographic finite-difference model
        ``n = normalize(-dz/dx, -dz/dy, 2)``), which is exactly constant across a
        planar region and varies elsewhere.  Because this is the orthographic
        normal approximation (no intrinsics), it is a *proxy* planar term, not a
        perspective-exact one.

        Formula (per frame, averaged over valid pixels and the two directions)::

            n            = compute_normals_from_depth(pred, valid)   # unit normals (B,S,3,H,W)
            d_x ||n||    = || n[..., :, j+1] - n[..., :, j] ||       # horizontal normal change
            d_y ||n||    = || n[..., i+1, :] - n[..., i, :] ||       # vertical normal change
            L_planar     = mean_x( w_x * d_x||n|| ) + mean_y( w_y * d_y||n|| )  (over 2 dirs)

        Differences are taken only between pixels whose normals are BOTH valid,
        so invalid layout pixels, and valid/invalid boundaries, are excluded
        (``it should not apply to invalid layout pixels``).

        Edge-aware gating (``planar_consistency_edge_aware``, default True):
        ``w_{x,y} = exp(-beta * ||Δ gt_normal||)``, the GT layout-depth normal
        total-variation, computed with the SAME stencil as ``dn_x``/``dn_y`` so
        the weight is spatially aligned with the penalty it modulates. It
        down-weights pixels where the GT surface genuinely bends (creases /
        corners / depth steps, e.g. wall-floor or wall-wall edges), so the
        prior flattens *within* planes rather than fighting true GT structure.
        (A naive forward-difference GT-depth gate is mis-aligned: a 1-px GT step
        perturbs the central-difference normals over a 2-px neighbourhood, so the
        gate would miss the penalty it is meant to suppress.) GT layout depth is
        used ONLY as a gate here, never as a regression target, and no GT normal
        maps are required, normals are derived from GT depth on the fly.

        Conservative by default, use a small ``planar_consistency_weight`` so it
        does not dominate the main layout-depth loss.  Returns a graph-safe zero
        when no valid pixels exist.
        """
        pred, gt, valid = self._get_layout_depth(predictions, batch)  # (B,S,H,W)
        if valid.sum() == 0:
            return pred.sum() * 0.0

        # Differentiable per-pixel surface normals from the PREDICTED layout depth.
        n = compute_normals_from_depth(pred.clamp(min=1e-6), valid)   # (B,S,3,H,W)
        nv = n.norm(dim=2) > 1e-6                                      # (B,S,H,W) normal-valid

        # First-order spatial variation of the normal field (normal TV).
        dn_x = (n[..., :, 1:] - n[..., :, :-1]).norm(dim=2)           # (B,S,H,W-1)
        dn_y = (n[..., 1:, :] - n[..., :-1, :]).norm(dim=2)           # (B,S,H-1,W)
        vx = nv[..., :, 1:] & nv[..., :, :-1]
        vy = nv[..., 1:, :] & nv[..., :-1, :]

        # Edge-aware down-weighting: gate by the GT normal total-variation,
        # computed with the SAME stencil as dn_x/dn_y so it is spatially aligned
        # with the penalty (a forward-difference GT-depth gate is offset by ~1px
        # from the central-difference normal disturbance and fails to suppress
        # genuine creases). GT is used only as a gate, never as a target.
        if self.planar_consistency_edge_aware:
            beta = self.planar_consistency_edge_beta
            gt_n = compute_normals_from_depth(gt.clamp(min=1e-6), valid)  # (B,S,3,H,W)
            dgx = (gt_n[..., :, 1:] - gt_n[..., :, :-1]).norm(dim=2)      # aligned w/ dn_x
            dgy = (gt_n[..., 1:, :] - gt_n[..., :-1, :]).norm(dim=2)      # aligned w/ dn_y
            wx = torch.exp(-beta * dgx)
            wy = torch.exp(-beta * dgy)
        else:
            wx = torch.ones_like(dn_x)
            wy = torch.ones_like(dn_y)

        parts = []
        if vx.any():
            parts.append((dn_x * wx)[vx].mean())
        if vy.any():
            parts.append((dn_y * wy)[vy].mean())
        if not parts:
            return pred.sum() * 0.0
        return sum(parts) / len(parts)

    # ------------------------------------------------------------------ #
    # Batch / prediction key extraction
    # ------------------------------------------------------------------ #

    def _get_layout_normals(self, predictions, batch):
        if "layout_normal" not in predictions:
            raise ValueError(
                "normal_head loss enabled but 'layout_normal' not in predictions. "
                "Set enable_layout_normal=True in the model config."
            )
        if "layout_normals" not in batch:
            raise ValueError(
                "normal_head loss enabled but 'layout_normals' not in batch. "
                "Set load_normal=True in the dataset config."
            )
        pred = predictions["layout_normal"]          # (B, S, 3, H, W)
        gt = batch["layout_normals"]
        if isinstance(gt, list):
            gt = torch.stack([torch.as_tensor(x) for x in gt], dim=0)
        gt = gt.to(pred.device)
        # Handle (B, S, H, W, 3) → (B, S, 3, H, W)
        if gt.dim() == 5 and gt.shape[-1] == 3:
            gt = gt.permute(0, 1, 4, 2, 3).contiguous()

        if "layout_normal_masks" in batch and batch["layout_normal_masks"] is not None:
            valid = batch["layout_normal_masks"]
            if isinstance(valid, list):
                valid = torch.stack([torch.as_tensor(x) for x in valid], dim=0)
            valid = valid.to(pred.device).bool()
        else:
            valid = gt.norm(dim=2) > 0.5
        return pred, gt, valid

    @staticmethod
    def _normal_head_loss(pred: torch.Tensor, gt: torch.Tensor,
                          valid: torch.Tensor) -> torch.Tensor:
        """Cosine loss between predicted normals and GT normals."""
        if valid.sum() == 0:
            return pred.sum() * 0.0
        gt = F.normalize(gt, dim=2)
        cosine = (pred * gt).sum(dim=2).clamp(-1.0, 1.0)  # (B, S, H, W)
        return (1.0 - cosine)[valid].mean()

    def _get_layout_depth(self, predictions, batch):
        if "layout_depth" not in predictions:
            raise ValueError(
                "layout_depth loss enabled but 'layout_depth' not in predictions. "
                "Set enable_layout_depth=True in the model config."
            )
        if "layout_depths" not in batch:
            raise ValueError(
                "layout_depth loss enabled but 'layout_depths' not in batch. "
                "Check dataset loader."
            )
        pred = predictions["layout_depth"]          # (B, S, H, W, 1)
        pred = pred.squeeze(-1)                      # → (B, S, H, W)
        gt   = batch["layout_depths"]               # (B, S, H, W)
        if isinstance(gt, list):
            gt = torch.stack([torch.as_tensor(x) for x in gt], dim=0)
        gt = gt.to(pred.device)

        # valid mask: explicit mask if provided, else gt > 0
        if "layout_depth_masks" in batch:
            valid = batch["layout_depth_masks"]
            if isinstance(valid, list):
                valid = torch.stack([torch.as_tensor(x) for x in valid], dim=0)
            valid = valid.to(pred.device).bool()
        else:
            valid = gt > 1e-6

        return pred, gt, valid

    def _get_mask(self, predictions, batch):
        if "layout_mask_logits" not in predictions:
            raise ValueError(
                "Binary mask loss enabled but 'layout_mask_logits' not in predictions. "
                "Set enable_layout_mask=True in the model config."
            )
        if "layout_masks" not in batch:
            raise ValueError(
                "Binary mask loss enabled but 'layout_masks' not in batch. "
                "Set load_mask=True in the dataset config."
            )
        logits = predictions["layout_mask_logits"]  # (B, S, 1, H, W)
        target = batch["layout_masks"]
        if isinstance(target, list):
            target = torch.stack([torch.as_tensor(x) for x in target], dim=0)
        target = target.to(logits.device)
        return logits, target

    # ------------------------------------------------------------------ #
    # Visible (metric) depth supervision:  E7
    # ------------------------------------------------------------------ #

    def _visible_depth_loss(self, predictions, batch):
        """Scale-shift-aligned L1 + multi-scale log-gradient on `depth` head.

        - Aligns predicted scale & shift per-frame in least-squares sense
          before computing L1 (so the head doesn't have to match absolute
          metric depth, only relative geometry, appropriate for monocular
          depth heads).
        - Adds the same multi-scale log-gradient loss used for layout depth
          to encourage edge-aware structure on visible regions.

        Mask: `batch['point_masks']` (bool, valid metric-depth pixels).
        """
        if "depth" not in predictions:
            raise ValueError(
                "visible_depth loss enabled but 'depth' not in predictions. "
                "Set enable_depth=True in the model config."
            )
        if "depths" not in batch:
            raise ValueError(
                "visible_depth loss enabled but 'depths' not in batch."
            )
        pred = predictions["depth"]              # (B,S,H,W,1) or (B,S,H,W)
        if pred.dim() == 5 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        gt = batch["depths"]
        if isinstance(gt, list):
            gt = torch.stack([torch.as_tensor(x) for x in gt], dim=0)
        gt = gt.to(pred.device)

        valid = batch.get("point_masks")
        if valid is None:
            valid = gt > 1e-6
        if isinstance(valid, list):
            valid = torch.stack([torch.as_tensor(x) for x in valid], dim=0)
        valid = valid.to(pred.device).bool()
        valid = valid & (gt > 1e-6)

        if valid.sum() == 0:
            return pred.sum() * 0.0

        # Per-frame least-squares scale+shift alignment of pred to gt
        pred_aligned = self._align_scale_shift_per_frame(pred, gt, valid)

        l1 = (pred_aligned - gt).abs()[valid].mean()
        # Reuse the same multi-scale log-gradient loss as layout depth.
        grad = self._gradient(
            pred_aligned.clamp(min=1e-6),
            gt.clamp(min=1e-6),
            valid,
        )
        # Weighted combination inside the term so a single weight controls it.
        return l1 + 0.1 * grad

    @staticmethod
    def _align_scale_shift_per_frame(pred: torch.Tensor,
                                     gt: torch.Tensor,
                                     valid: torch.Tensor) -> torch.Tensor:
        """Solve s*pred + t = gt per (B,S) frame on valid pixels (least squares).

        Returns the aligned prediction with the same shape as ``pred``.
        Falls back to the original prediction for frames with no valid pixels.
        """
        B, S, H, W = pred.shape
        out = pred.clone()
        p_flat = pred.reshape(B * S, -1)
        g_flat = gt.reshape(B * S, -1)
        v_flat = valid.reshape(B * S, -1)
        for i in range(B * S):
            v_i = v_flat[i]
            n = v_i.sum()
            if n < 8:               # too few pixels to fit; leave as-is
                continue
            p_v = p_flat[i][v_i]
            g_v = g_flat[i][v_i]
            p_mean = p_v.mean()
            g_mean = g_v.mean()
            num = ((p_v - p_mean) * (g_v - g_mean)).sum()
            den = ((p_v - p_mean) ** 2).sum().clamp(min=1e-8)
            s = num / den
            t = g_mean - s * p_mean
            out.reshape(B * S, -1)[i] = s * p_flat[i] + t
        return out

    # ------------------------------------------------------------------ #
    # Camera supervision:  E7
    # ------------------------------------------------------------------ #

    @staticmethod
    def _camera_loss(predictions, batch,
                     gamma: float = 0.6,
                     weight_trans: float = 1.0,
                     weight_rot: float = 1.0,
                     weight_focal: float = 0.5):
        """Camera translation + rotation + FoV loss on ``pose_enc_list``.

        Self-contained re-implementation of the standard VGGT camera loss
        (`training/loss.py:213` :func:`compute_camera_loss`), duplicated
        here only so :class:`RoomEnvelopeLoss` does not pull in the rest of
        ``loss.py`` (and its ``iopath`` dep) when the camera weight is 0.

        Requires ``enable_camera=True`` in the model config so that
        ``predictions["pose_enc_list"]`` is populated.
        """
        if "pose_enc_list" not in predictions:
            raise ValueError(
                "camera loss enabled but 'pose_enc_list' not in predictions. "
                "Set enable_camera=True in the model config."
            )
        from vggt.utils.pose_enc import extri_intri_to_pose_encoding

        pred_pose_encodings = predictions["pose_enc_list"]
        n_stages = len(pred_pose_encodings)

        # GT pose encoding from extrinsics+intrinsics
        gt_extr = batch["extrinsics"]
        gt_intr = batch["intrinsics"]
        if isinstance(gt_extr, list):
            gt_extr = torch.stack([torch.as_tensor(x) for x in gt_extr], dim=0)
        if isinstance(gt_intr, list):
            gt_intr = torch.stack([torch.as_tensor(x) for x in gt_intr], dim=0)
        gt_extr = gt_extr.to(pred_pose_encodings[-1].device)
        gt_intr = gt_intr.to(pred_pose_encodings[-1].device)
        if gt_extr.dim() == 4 and gt_extr.shape[-2:] == (4, 4):
            gt_extr = gt_extr[..., :3, :]                      # (B,S,4,4) → (B,S,3,4)
        image_hw = batch["images"].shape[-2:]
        gt_pose_enc = extri_intri_to_pose_encoding(
            gt_extr, gt_intr, image_hw, pose_encoding_type="absT_quaR_FoV"
        )

        # Optional per-frame validity gate via point_masks (>100 valid pts).
        pm = batch.get("point_masks")
        if pm is not None and not isinstance(pm, list) and pm.dim() == 4:
            valid_frame = pm[:, 0].sum(dim=[-1, -2]) > 100      # (B,), first frame heuristic
        else:
            valid_frame = None

        total_T = total_R = total_FL = 0.0
        for stage_idx in range(n_stages):
            stage_w = gamma ** (n_stages - stage_idx - 1)
            pred = pred_pose_encodings[stage_idx]
            if valid_frame is not None and valid_frame.sum() == 0:
                lT = lR = lFL = (pred * 0).mean()
            else:
                if valid_frame is not None:
                    p = pred[valid_frame]
                    g = gt_pose_enc[valid_frame]
                else:
                    p = pred
                    g = gt_pose_enc
                lT = (p[..., :3] - g[..., :3]).abs().clamp(max=100).mean()
                lR = (p[..., 3:7] - g[..., 3:7]).abs().mean()
                lFL = (p[..., 7:] - g[..., 7:]).abs().mean()
            total_T = total_T + lT * stage_w
            total_R = total_R + lR * stage_w
            total_FL = total_FL + lFL * stage_w

        return (total_T / n_stages) * weight_trans + \
               (total_R / n_stages) * weight_rot + \
               (total_FL / n_stages) * weight_focal

    @staticmethod
    def _device(predictions, batch):
        for v in predictions.values():
            if isinstance(v, torch.Tensor):
                return v.device
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # VGGT-original ablation wrappers
    # ------------------------------------------------------------------ #

    def _vggt_layout_depth_loss(self, predictions, batch):
        """Run the canonical VGGT layout-depth loss on this batch.

        Requires ``predictions['layout_depth']`` (B,S,H,W,1) AND
        ``predictions['layout_depth_conf']`` (B,S,H,W). The default DPT layout-
        depth head (``vggt.models.vggt.VGGT`` with ``enable_layout_depth=True``)
        emits both, see ``vggt/models/vggt.py:152-156``. If the confidence
        tensor is missing we raise rather than fake a constant value, since the
        canonical regression includes a ``-alpha·log(conf)`` regulariser that
        would be undefined.
        """
        if "layout_depth" not in predictions:
            raise ValueError(
                "vggt_original layout_depth loss requires 'layout_depth' in "
                "predictions. Set enable_layout_depth=True in the model config."
            )
        if "layout_depth_conf" not in predictions:
            raise ValueError(
                "vggt_original layout_depth loss requires 'layout_depth_conf' in "
                "predictions but it is missing. The DPT layout-depth head must "
                "be configured with output_dim=2 (depth + confidence). Check "
                "vggt.models.vggt.VGGT, the default already emits "
                "layout_depth_conf when enable_layout_depth=True."
            )
        if "layout_depths" not in batch:
            raise ValueError(
                "vggt_original layout_depth loss requires 'layout_depths' in "
                "the batch. Check dataset loader."
            )
        return vggt_compute_layout_depth_loss(
            predictions,
            batch,
            gamma=1.0,
            alpha=0.2,
            gradient_loss_fn=self.vggt_layout_depth_gradient_loss_fn,
            valid_range=self.vggt_layout_depth_valid_range,
            clutter_lambda=self.clutter_lambda,
        )

    def _vggt_camera_loss(self, predictions, batch):
        """Run the canonical VGGT camera loss on this batch.

        Returns the canonical dict ``{loss_camera, loss_T, loss_R, loss_FL}``.
        The umbrella weight is applied by the caller (``camera_weight``).
        """
        if "pose_enc_list" not in predictions:
            raise ValueError(
                "vggt_original camera loss requires 'pose_enc_list' in "
                "predictions. Set enable_camera=True in the model config."
            )
        return vggt_compute_camera_loss(
            predictions,
            batch,
            loss_type="l1",
            gamma=0.6,
            pose_encoding_type="absT_quaR_FoV",
            weight_trans=1.0,
            weight_rot=1.0,
            weight_focal=0.5,
        )


# =========================================================================== #
# Canonical VGGT loss functions, copied verbatim from
# the upstream VGGT training code
# (the unmodified VGGT ``MultitaskLoss`` helper set). Kept at module level so
# ``RoomEnvelopeLoss`` can opt in via the ``camera_loss_type`` /
# ``layout_depth_loss_type`` constructor args without importing extra code
# (which would also drag in iopath / fvcore deps that this repo does not
# need at training time).
#
# Functions copied (with their original docstrings preserved):
#   - vggt_compute_camera_loss        ← compute_camera_loss
#   - _vggt_camera_loss_single        ← camera_loss_single
#   - vggt_compute_layout_depth_loss  ← compute_layout_depth_loss
#   - _vggt_regression_loss           ← regression_loss
#   - _vggt_gradient_loss_multi_scale ← gradient_loss_multi_scale_wrapper
#   - _vggt_gradient_loss             ← gradient_loss
#   - _vggt_normal_loss               ← normal_loss
#   - _vggt_point_map_to_normal       ← point_map_to_normal
#   - _vggt_filter_by_quantile        ← filter_by_quantile
#   - _vggt_torch_quantile            ← torch_quantile
#
# Names are prefixed ``vggt_`` to avoid accidental shadowing. Formulas, default
# values and ``check_and_fix_inf_nan`` calls are preserved exactly.
# =========================================================================== #

from vggt.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402

# train_utils.general transitively imports iopath, which is not always
# installed in lightweight smoke-test environments. Fall back to a local
# implementation with the exact same semantics if the import fails. The
# canonical version lives at training/train_utils/general.py:29.
try:
    from train_utils.general import check_and_fix_inf_nan  # noqa: E402
except ModuleNotFoundError:                                # pragma: no cover
    import logging as _logging                             # noqa: E402

    def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
        """Fallback copy of ``train_utils.general.check_and_fix_inf_nan``.

        Active only when ``iopath`` is missing (smoke-test env). Numerically
        identical to the canonical helper at
        ``training/train_utils/general.py:29``.
        """
        if input_tensor is None:
            return input_tensor
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            _logging.warning(
                f"Tensor {loss_name} contains inf or nan values. Replacing with zeros."
            )
            input_tensor = torch.where(
                torch.isnan(input_tensor) | torch.isinf(input_tensor),
                torch.zeros_like(input_tensor),
                input_tensor,
            )
        if hard_max is not None:
            input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)
        return input_tensor


def vggt_compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs,
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = _vggt_camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type,
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans
        + avg_loss_R * weight_rot
        + avg_loss_FL * weight_focal
    )

    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL,
    }


def _vggt_camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """Translation, rotation, and focal loss for a batch of pose encodings.

    NOTE: The paper uses smooth l1, but this implementation found L1 more stable.
    """
    if loss_type == "l1":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def vggt_compute_layout_depth_loss(predictions, batch,
                                   gamma=1.0, alpha=0.2,
                                   gradient_loss_fn=None,
                                   valid_range=-1,
                                   clutter_lambda=1.0,
                                   **kwargs):
    """Confidence-weighted regression + multi-scale gradient on layout depth.

    Mirrors ``compute_layout_depth_loss`` in the canonical VGGT loss module.

    ``clutter_lambda``: per-pixel weight on the clutter sub-region
    (``layout_masks < 0.5``). Default 1.0 = behaviour-neutral. Values >
    1.0 give regions OCA's m_clutter gate can update more loss authority.
    Applied to the regression and confidence terms via a weighted mean;
    NOT applied to the gradient term (multi-scale smoothness prior is
    content-agnostic and the per-scale downsampling makes per-pixel
    weighting ill-defined).
    """
    pred_depth = predictions['layout_depth']
    pred_depth_conf = predictions['layout_depth_conf']

    gt_depth = batch['layout_depths']
    if isinstance(gt_depth, list):
        gt_depth = torch.stack([torch.as_tensor(x) for x in gt_depth], dim=0)
    gt_depth = gt_depth.to(pred_depth.device)
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_layout_depth")
    gt_depth = gt_depth[..., None]  # (B, S, H, W, 1)

    if 'layout_depth_masks' in batch and batch['layout_depth_masks'] is not None:
        gt_depth_mask = batch['layout_depth_masks']
        if isinstance(gt_depth_mask, list):
            gt_depth_mask = torch.stack(
                [torch.as_tensor(x) for x in gt_depth_mask], dim=0
            )
        gt_depth_mask = gt_depth_mask.to(pred_depth.device).bool().clone()
    else:
        gt_depth_mask = batch['point_masks'].clone()

    if gt_depth_mask.sum() < 100:
        dummy_loss = (0.0 * pred_depth).mean()
        return {
            "loss_conf_layout_depth": dummy_loss,
            "loss_reg_layout_depth": dummy_loss,
            "loss_grad_layout_depth": dummy_loss,
        }

    pixel_weight = None
    if clutter_lambda is not None and float(clutter_lambda) != 1.0:
        layout_masks = batch.get('layout_masks')
        if layout_masks is not None:
            if isinstance(layout_masks, list):
                layout_masks = torch.stack(
                    [torch.as_tensor(x) for x in layout_masks], dim=0
                )
            lm = layout_masks.to(pred_depth.device).float()
            # 1.0 where pixel is visible-layout (m >= 0.5), clutter_lambda
            # where pixel is clutter (m < 0.5). Match (B, S, H, W, 1) shape.
            w = torch.where(
                lm < 0.5,
                torch.tensor(float(clutter_lambda), dtype=pred_depth.dtype, device=pred_depth.device),
                torch.tensor(1.0, dtype=pred_depth.dtype, device=pred_depth.device),
            )
            if w.ndim == gt_depth.ndim - 1:
                w = w.unsqueeze(-1)
            pixel_weight = w

    loss_conf, loss_grad, loss_reg = _vggt_regression_loss(
        pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
        gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha,
        valid_range=valid_range,
        pixel_weight=pixel_weight,
    )

    return {
        "loss_conf_layout_depth": loss_conf,
        "loss_reg_layout_depth": loss_reg,
        "loss_grad_layout_depth": loss_grad,
    }


def _vggt_regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None,
                          gamma=1.0, alpha=0.2, valid_range=-1,
                          pixel_weight=None):
    """Core confidence-weighted regression with optional multi-scale gradient.

    Copied verbatim from the canonical VGGT ``regression_loss``. ``pixel_weight``
    (optional, same shape as ``gt``) reweights the regression and confidence
    terms via a normalised weighted mean: ``(loss * w[mask]).sum() / w[mask].sum()``.
    Default (None) is byte-identical to the original mean reduction.
    """
    bb, ss, hh, ww, nc = pred.shape

    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")

    if pixel_weight is not None:
        # Reduce the trailing channel axis to align with mask shape.
        w_full = pixel_weight
        if w_full.ndim == mask.ndim + 1 and w_full.shape[-1] == 1:
            w_full = w_full.squeeze(-1)
        w_flat = w_full[mask].to(loss_reg.dtype)
    else:
        w_flat = None

    loss_grad = 0

    if gradient_loss_fn is not None and "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb * ss, hh, ww)
    else:
        to_feed_conf = None

    if gradient_loss_fn is not None and "normal" in gradient_loss_fn:
        loss_grad = _vggt_gradient_loss_multi_scale(
            pred.reshape(bb * ss, hh, ww, nc),
            gt.reshape(bb * ss, hh, ww, nc),
            mask.reshape(bb * ss, hh, ww),
            gradient_loss_fn=_vggt_normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif gradient_loss_fn is not None and "grad" in gradient_loss_fn:
        loss_grad = _vggt_gradient_loss_multi_scale(
            pred.reshape(bb * ss, hh, ww, nc),
            gt.reshape(bb * ss, hh, ww, nc),
            mask.reshape(bb * ss, hh, ww),
            gradient_loss_fn=_vggt_gradient_loss,
            conf=to_feed_conf,
        )

    if loss_conf.numel() > 0:
        if valid_range > 0 and w_flat is None:
            loss_conf = _vggt_filter_by_quantile(loss_conf, valid_range)
        loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf_depth")
        if w_flat is not None and w_flat.sum() > 0:
            # Weighted-mean reduction; quantile outlier filter is bypassed
            # because the filter returns a subset of indices we can't realign
            # with the weight tensor without restructuring the helper. The
            # weighting itself already up-rates the clutter region; outlier
            # rejection becomes secondary in this path.
            loss_conf = (loss_conf * w_flat).sum() / w_flat.sum()
        else:
            loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    if loss_reg.numel() > 0:
        if valid_range > 0 and w_flat is None:
            loss_reg = _vggt_filter_by_quantile(loss_reg, valid_range)
        loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg_depth")
        if w_flat is not None and w_flat.sum() > 0:
            loss_reg = (loss_reg * w_flat).sum() / w_flat.sum()
        else:
            loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def _vggt_gradient_loss_multi_scale(prediction, target, mask, scales=4,
                                    gradient_loss_fn=None, conf=None):
    """Multi-scale wrapper around a per-scale gradient/normal loss."""
    total = 0
    for scale in range(scales):
        step = pow(2, scale)
        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
        )
    total = total / scales
    return total


def _vggt_normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None,
                     gamma=1.0, alpha=0.2):
    """Surface-normal cosine loss derived from point maps."""
    pred_normals, pred_valids = _vggt_point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = _vggt_point_map_to_normal(target,     mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids

    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    dot = torch.sum(pred_normals * gt_normals, dim=-1)
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    loss = 1 - dot

    if loss.numel() < 10:
        return 0

    loss = check_and_fix_inf_nan(loss, "normal_loss")
    if conf is not None:
        conf = conf[None, ...].expand(4, -1, -1, -1)
        conf = conf[all_valid].clone()
        loss = gamma * loss * conf - alpha * torch.log(conf)
        return loss.mean()
    return loss.mean()


def _vggt_gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """L1 gradient loss between prediction and target."""
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)
    if divisor == 0:
        return 0
    return torch.sum(grad_loss) / divisor


def _vggt_point_map_to_normal(point_map, mask, eps=1e-6):
    """Convert a 3D point map into surface normals via cross products."""
    with torch.cuda.amp.autocast(enabled=False):
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1),
                    mode='constant', value=0).permute(0, 2, 3, 1)

        center = pts[:, 1:-1, 1:-1, :]
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2, :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        n1 = torch.cross(up_dir,    left_dir,  dim=-1)
        n2 = torch.cross(left_dir,  down_dir,  dim=-1)
        n3 = torch.cross(down_dir,  right_dir, dim=-1)
        n4 = torch.cross(right_dir, up_dir,    dim=-1)

        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        normals = torch.stack([n1, n2, n3, n4], dim=0)
        valids  = torch.stack([v1, v2, v3, v4], dim=0)
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)
    return normals, valids


def _vggt_filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """Quantile-based outlier filter (keep elements below the requested quantile)."""
    if loss_tensor.numel() <= min_elements:
        return loss_tensor
    if loss_tensor.numel() > 100000000:
        indices = torch.randperm(loss_tensor.numel(),
                                 device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]
    loss_tensor = loss_tensor.clamp(max=hard_max)
    quantile_thresh = _vggt_torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def _vggt_torch_quantile(input, q, dim=None, keepdim: bool = False, *,
                        interpolation: str = "nearest",
                        out: torch.Tensor = None) -> torch.Tensor:
    """Memory-safe scalar-quantile via ``torch.kthvalue`` (matches canonical impl)."""
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    return out.squeeze(dim)
