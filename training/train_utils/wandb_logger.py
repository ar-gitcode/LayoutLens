"""
Weights & Biases logging utilities for VGGT training.

All public functions are safe to call from every DDP rank, they silently
no-op on non-rank-0 processes and when wandb is disabled / unavailable.
"""

import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Segmentation colour palette (6 structural classes)
# ---------------------------------------------------------------------------

# (R, G, B) indexed by class ID 0-5; class 255 (ignore) renders as black.
_SEG_PALETTE = np.array([
    [200, 200, 200],  # 0  wall            light grey
    [140, 100,  50],  # 1  floor           brown
    [100, 180, 230],  # 2  ceiling         sky blue
    [230, 140,  50],  # 3  door            orange
    [ 50, 200, 180],  # 4  window          teal
    [160,  90, 200],  # 5  other-struct    purple
], dtype=np.uint8)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_main_process() -> bool:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _wandb():
    """Return the wandb module, or None if not installed or no active run."""
    try:
        import wandb
        return wandb
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init_wandb(
    logging_cfg: Any,
    exp_name: str = "",
    extra_config: Optional[Dict] = None,
) -> None:
    """
    Initialize a wandb run on the main process only.

    Args:
        logging_cfg: The ``logging`` section of the Hydra config (DictConfig).
        exp_name:    Experiment name used as a fallback run name.
        extra_config: Optional plain dict of extra hyperparameters to store.
    """
    if not _is_main_process():
        return

    wb = _wandb()
    if wb is None:
        log.warning("wandb is not installed, skipping wandb init. Install with: pip install wandb")
        return

    project  = getattr(logging_cfg, "wandb_project",  "vggt")
    entity   = getattr(logging_cfg, "wandb_entity",   None) or None
    run_name = getattr(logging_cfg, "wandb_run_name", None) or exp_name or None
    mode     = getattr(logging_cfg, "wandb_mode",     "online")

    config_to_log: Dict[str, Any] = {}
    if extra_config:
        config_to_log.update(extra_config)

    # Try to serialize the logging config section via OmegaConf
    try:
        from omegaconf import OmegaConf
        config_to_log["logging"] = OmegaConf.to_container(logging_cfg, resolve=True)
    except Exception:
        pass

    try:
        wb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            config=config_to_log,
        )
        log.info(f"wandb run initialized, project={project!r}, name={run_name!r}, mode={mode!r}")
    except Exception as exc:
        log.warning(f"wandb.init failed: {exc}")
        return

    # Register a custom val step axis so val-side metrics are not bound by the
    # global monotonic ``_step`` counter that train logging advances. Without
    # this, val logs (which start at step 0 each epoch) collide with the
    # already-advanced train step and W&B silently drops them.
    try:
        wb.define_metric("val/step")
        wb.define_metric("val/*",          step_metric="val/step")
        wb.define_metric("Loss/val_*",     step_metric="val/step")
        wb.define_metric("Metrics/val/*",  step_metric="val/step")
        wb.define_metric("Visuals/val/*",  step_metric="val/step")
        log.info("wandb: registered 'val/step' as custom step axis for val/*, Loss/val_*, Metrics/val/*, Visuals/val/*")
    except Exception as exc:
        log.debug(f"wandb.define_metric failed: {exc}")


def finish_wandb() -> None:
    """Gracefully finish the active wandb run (main process only)."""
    if not _is_main_process():
        return
    wb = _wandb()
    if wb is None or wb.run is None:
        return
    try:
        wb.finish()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scalar logging
# ---------------------------------------------------------------------------

def log_scalars(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
) -> None:
    """
    Log a dict of scalar metrics to wandb (main process only).

    Args:
        metrics: ``{tag: value}`` mapping.  Values may be plain Python
                 scalars or 0-dim / 1-element torch Tensors.
        step:    Global training step. If ``commit=False`` is passed, this
                 should usually be ``None`` so the data is queued and
                 attached to the next committing ``wandb.log`` call.
        commit:  When ``False``, stages the data without advancing W&B's
                 internal step counter. The next ``log_scalars`` call with
                 ``commit`` unset (or ``True``) will commit everything that
                 has been staged. Use this for ``val`` metrics, whose step
                 counter lags ``train`` and would otherwise be silently
                 dropped by W&B's monotonic-step rule.
    """
    if not _is_main_process():
        return
    wb = _wandb()
    if wb is None or wb.run is None:
        return
    # Coerce any tensors to Python scalars to avoid serialisation issues
    clean: Dict[str, Any] = {}
    for k, v in metrics.items():
        clean[k] = v.item() if torch.is_tensor(v) else v
    try:
        if commit is False:
            # Stage without advancing step; W&B does not accept ``step`` when
            # ``commit=False`` and would otherwise raise / silently drop.
            wb.log(clean, commit=False)
        elif step is None:
            wb.log(clean)
        else:
            wb.log(clean, step=step)
    except Exception as exc:
        log.debug(f"wandb.log (scalars) failed: {exc}")


# ---------------------------------------------------------------------------
# Tensor → numpy visualization helpers
# ---------------------------------------------------------------------------

def prepare_depth_for_vis(
    depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    percentile: float = 98.0,
) -> np.ndarray:
    """
    Robustly normalize a depth map to [0, 1] for visualization.

    Args:
        depth:      (H, W) or (H, W, 1) tensor on any device / dtype.
        mask:       Optional (H, W) bool tensor, True = valid pixel.
                    Only valid pixels are used for percentile estimation.
        percentile: Upper percentile for robust clipping (lower = 100 − p).

    Returns:
        (H, W) float32 numpy array in [0, 1].
    """
    d = depth.detach().cpu().float()
    if d.dim() == 3 and d.shape[-1] == 1:
        d = d.squeeze(-1)
    d_np = d.numpy()

    if mask is not None:
        m_np = mask.detach().cpu().bool().numpy()
        valid = d_np[m_np]
    else:
        flat = d_np.ravel()
        valid = flat[np.isfinite(flat)]

    if len(valid) == 0:
        return np.zeros_like(d_np, dtype=np.float32)

    lo = np.percentile(valid, 100.0 - percentile)
    hi = np.percentile(valid, percentile)

    if hi <= lo:
        return np.zeros_like(d_np, dtype=np.float32)

    return np.clip((d_np - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def prepare_rgb_for_vis(image: torch.Tensor) -> np.ndarray:
    """
    Convert a (3, H, W) image tensor to a (H, W, 3) uint8 numpy array.

    Handles both [0, 1] and [-1, 1] value ranges automatically.
    """
    img = image.detach().cpu().float()
    if img.min() < -0.1:          # likely ImageNet-style [-1, 1]
        img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def prepare_normal_for_vis(normal: torch.Tensor) -> np.ndarray:
    """
    Convert a normal map tensor to a (H, W, 3) uint8 numpy array.

    Args:
        normal: (3, H, W) or (H, W, 3) tensor on any device.
                Values are expected in [-1, 1].

    Returns:
        (H, W, 3) uint8 array with values mapped from [-1, 1] to [0, 255].
    """
    n = normal.detach().cpu().float()
    if n.dim() == 3 and n.shape[0] == 3:
        n = n.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    n = ((n + 1.0) / 2.0).clamp(0.0, 1.0)
    return (n.numpy() * 255).astype(np.uint8)


def prepare_binary_mask_for_vis(mask: torch.Tensor) -> np.ndarray:
    """
    Convert a binary mask tensor to a (H, W) uint8 numpy array.

    Args:
        mask: (H, W), (H, W, 1), or (1, H, W) tensor on any device.

    Returns:
        (H, W) uint8 array with values in [0, 255].
    """
    m = mask.detach().cpu().float()
    if m.dim() == 3 and m.shape[0] == 1:
        m = m.squeeze(0)   # (1, H, W) -> (H, W)
    elif m.dim() == 3 and m.shape[-1] == 1:
        m = m.squeeze(-1)  # (H, W, 1) -> (H, W)
    m = m.clamp(0.0, 1.0)
    return (m.numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Segmentation visualization helpers
# ---------------------------------------------------------------------------

def prepare_seg_mask_for_vis(mask: torch.Tensor, num_classes: int = 6) -> np.ndarray:
    """
    Convert a (H, W) integer segmentation mask to a (H, W, 3) uint8 RGB image.

    Args:
        mask:        (H, W) tensor with class IDs in [0, num_classes) or 255 (ignore).
        num_classes: Number of valid class IDs (matches _SEG_PALETTE length).

    Returns:
        (H, W, 3) uint8 array; ignore pixels (255) render as black.
    """
    m = mask.detach().cpu().long().numpy()
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    palette = _SEG_PALETTE[:num_classes]
    for cls_id in range(num_classes):
        rgb[m == cls_id] = palette[cls_id]
    return rgb


def _blend_seg_on_rgb(
    rgb: np.ndarray,
    seg_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a coloured segmentation map over an RGB image. Both (H,W,3) uint8."""
    out = rgb.astype(np.float32) * (1.0 - alpha) + seg_rgb.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _to_rgb_uint8(x: np.ndarray) -> np.ndarray:
    """Convert any grayscale or float array to (H, W, 3) uint8."""
    if x.dtype != np.uint8:
        lo, hi = float(x.min()), float(x.max())
        if hi > lo:
            x = (x - lo) / (hi - lo)
        else:
            x = np.zeros_like(x, dtype=np.float32)
        x = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    return x


def _side_by_side(left: np.ndarray, right: np.ndarray, sep: int = 4) -> np.ndarray:
    """Return a single image with *left* (GT) and *right* (pred) separated by a white bar."""
    left  = _to_rgb_uint8(left)
    right = _to_rgb_uint8(right)
    h = max(left.shape[0], right.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] < h:
            pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.concatenate([img, pad], axis=0)
        return img

    divider = np.full((h, sep, 3), 255, dtype=np.uint8)
    return np.concatenate([_pad(left), divider, _pad(right)], axis=1)


def _blend_binary_mask_on_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 80, 80),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary mask (uint8 0/255 or float [0,1]) on an RGB image.

    The mask is rendered with the given ``color``; non-mask pixels keep RGB.
    """
    rgb = rgb.astype(np.float32)
    if mask.ndim == 3 and mask.shape[-1] == 3:
        mask = mask.mean(axis=-1)
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)[..., None]
    color_arr = np.asarray(color, dtype=np.float32)[None, None, :]
    blended = rgb * (1.0 - m * alpha) + color_arr * (m * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _depth_error_map(pred: np.ndarray, gt: np.ndarray,
                     mask: np.ndarray = None) -> np.ndarray:
    """Per-pixel |pred-gt| depth error → turbo-coloured RGB."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    err = np.abs(pred - gt)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        err = np.where(m, err, 0.0)
    flat = err[err > 0]
    if flat.size == 0:
        norm = np.zeros_like(err)
    else:
        hi = np.percentile(flat, 95.0)
        norm = np.clip(err / max(hi, 1e-6), 0, 1)
    return _apply_colormap(norm)


def _normal_angular_error_map(pred: np.ndarray, gt: np.ndarray,
                               valid: np.ndarray = None,
                               vmax_deg: float = 30.0) -> np.ndarray:
    """Per-pixel angular error in degrees → red-shaded RGB (0=black, vmax=red)."""
    p = np.asarray(pred, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)
    if p.shape[0] == 3 and p.ndim == 3:
        p = np.transpose(p, (1, 2, 0))
    if g.shape[0] == 3 and g.ndim == 3:
        g = np.transpose(g, (1, 2, 0))
    p = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
    g = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
    dot = np.clip((p * g).sum(axis=-1), -1.0, 1.0)
    err = np.degrees(np.arccos(dot))
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        err = np.where(v, err, 0.0)
    norm = np.clip(err / vmax_deg, 0, 1)
    r = (norm * 255).astype(np.uint8)
    return np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=-1)


def _mask_fp_fn_map(pred_prob: np.ndarray, gt: np.ndarray,
                    threshold: float = 0.5) -> np.ndarray:
    """Visualize mask errors: red=FP (predicted as structure but isn't),
    blue=FN (structure pixel missed), white=TP, black=TN."""
    p = np.asarray(pred_prob, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)
    if p.max() > 1.5:                # logits → probs
        p = 1.0 / (1.0 + np.exp(-p))
    pb = p >= threshold
    gb = g >= 0.5
    rgb = np.zeros((*pb.shape, 3), dtype=np.uint8)
    tp = pb & gb
    fp = pb & ~gb
    fn = ~pb & gb
    rgb[tp] = (255, 255, 255)
    rgb[fp] = (230,  80,  80)
    rgb[fn] = ( 80,  80, 230)
    return rgb


def _apply_colormap(gray: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    """
    Apply a colormap to a (H, W) float32 image in [0, 1].

    Returns:
        (H, W, 3) uint8 RGB image.  Falls back to grayscale if matplotlib
        is unavailable or the colormap lookup fails.
    """
    gray = np.clip(gray, 0.0, 1.0)
    try:
        import matplotlib.cm as mcm
        colored = mcm.get_cmap(cmap)(gray)   # (H, W, 4) RGBA float in [0, 1]
        return (colored[..., :3] * 255).astype(np.uint8)
    except Exception:
        gray_u8 = (gray * 255).astype(np.uint8)
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


# ---------------------------------------------------------------------------
# Visual batch logging
# ---------------------------------------------------------------------------

def log_visual_batch(
    batch: Mapping,
    phase: str,
    step: int,
    epoch: int,
    max_samples: int = 2,
) -> None:
    """
    Log per-task visualizations to separate wandb panels.

    Each visual type gets its own key ``Visuals/{phase}/<type>`` so W&B
    renders them in separate media boxes.  GT and prediction are shown
    side-by-side (left = GT, right = pred) within a single image entry.

    Supported batch keys (shapes after [b, s] indexing):
        images             (B, S, 3, H, W)
        depths             (B, S, H, W)           GT metric depth
        layout_depths      (B, S, H, W)           GT layout depth
        layout_depth_masks (B, S, H, W)           layout depth validity
        depth              (B, S, H, W, 1)        pred metric depth
        layout_depth       (B, S, H, W, 1)        pred layout depth
        seg_masks          (B, S, H, W)           GT seg class IDs
        seg_logits         (B, S, C, H, W)        pred seg logits
        layout_normals / layout_normal_maps / layout_normal  GT normals
        layout_normal_pred / pred_layout_normals /
          layout_normals_pred / layout_normal               pred normals
        layout_masks       (B, S, H, W)           GT layout mask
        layout_mask_logits (B, S, *, H, W)        pred layout mask logits
        seen_masks / seen_mask / point_masks       binary seen mask
    """
    if not _is_main_process():
        return
    wb = _wandb()
    if wb is None or wb.run is None:
        return

    images = batch.get("images")
    if images is None:
        return

    B = images.shape[0]
    n = min(max_samples, B)
    cap_prefix = f"ep={epoch} | step={step}"

    logs: Dict[str, list] = {}

    def add_image(key: str, img: np.ndarray, caption: str) -> None:
        full_key = f"Visuals/{phase}/{key}"
        logs.setdefault(full_key, []).append(wb.Image(img, caption=caption))

    for b in range(n):
        s = 0
        cap = f"{cap_prefix} | b{b}"

        # ── RGB ──────────────────────────────────────────────────────────────
        try:
            add_image("rgb", prepare_rgb_for_vis(images[b, s]),
                      f"{cap} | rgb")
        except Exception:
            pass

        # ── Layout depth  GT | PRED ──────────────────────────────────────────
        gt_ld   = batch.get("layout_depths")
        pred_ld = batch.get("layout_depth")
        ld_mask = batch.get("layout_depth_masks")
        if gt_ld is not None and pred_ld is not None:
            try:
                m        = ld_mask[b, s] if ld_mask is not None else None
                gt_vis   = _apply_colormap(prepare_depth_for_vis(gt_ld[b, s], m))
                pred_vis = _apply_colormap(prepare_depth_for_vis(pred_ld[b, s]))
                add_image("layout_depth", _side_by_side(gt_vis, pred_vis),
                          f"{cap} | layout_depth left=GT right=PRED")
            except Exception:
                pass

        # ── Metric depth  GT | PRED ──────────────────────────────────────────
        gt_d   = batch.get("depths")
        pred_d = batch.get("depth")
        if gt_d is not None and pred_d is not None:
            try:
                gt_vis   = _apply_colormap(prepare_depth_for_vis(gt_d[b, s]))
                pred_vis = _apply_colormap(prepare_depth_for_vis(pred_d[b, s]))
                add_image("depth", _side_by_side(gt_vis, pred_vis),
                          f"{cap} | depth left=GT right=PRED")
            except Exception:
                pass

        # ── Segmentation  GT | PRED ──────────────────────────────────────────
        gt_seg     = batch.get("seg_masks")
        seg_logits = batch.get("seg_logits")
        gt_seg_rgb   = None
        pred_seg_rgb = None

        if gt_seg is not None:
            try:
                gt_seg_rgb = prepare_seg_mask_for_vis(gt_seg[b, s])
            except Exception:
                gt_seg_rgb = None

        if seg_logits is not None:
            try:
                pred_cls     = seg_logits[b, s].detach().cpu().float().argmax(dim=0)
                pred_seg_rgb = prepare_seg_mask_for_vis(pred_cls)
            except Exception:
                pred_seg_rgb = None

        if gt_seg_rgb is not None and pred_seg_rgb is not None:
            try:
                add_image("seg", _side_by_side(gt_seg_rgb, pred_seg_rgb),
                          f"{cap} | seg left=GT right=PRED")
            except Exception:
                pass
            try:
                rgb_np       = prepare_rgb_for_vis(images[b, s])
                gt_overlay   = _blend_seg_on_rgb(rgb_np, gt_seg_rgb)
                pred_overlay = _blend_seg_on_rgb(rgb_np, pred_seg_rgb)
                add_image("seg_overlay", _side_by_side(gt_overlay, pred_overlay),
                          f"{cap} | seg_overlay left=GT right=PRED")
            except Exception:
                pass

        # ── Layout mask  GT | PRED  +  overlay  +  FP/FN error map ──────────
        gt_lm          = batch.get("layout_masks")
        pred_lm_logits = batch.get("layout_mask_logits")
        if gt_lm is not None and pred_lm_logits is not None:
            try:
                gt_vis    = prepare_binary_mask_for_vis(gt_lm[b, s])
                # Sigmoid probability, more informative than a hard threshold
                pred_prob_t = torch.sigmoid(pred_lm_logits[b, s].detach().cpu().float())
                pred_vis  = prepare_binary_mask_for_vis(pred_prob_t)
                add_image("layout_mask", _side_by_side(gt_vis, pred_vis),
                          f"{cap} | layout_mask left=GT right=PRED")

                # Overlay GT and pred on the RGB
                try:
                    rgb_np = prepare_rgb_for_vis(images[b, s])
                    gt_overlay   = _blend_binary_mask_on_rgb(rgb_np, gt_vis,   color=( 80, 200,  80))
                    pred_overlay = _blend_binary_mask_on_rgb(rgb_np, pred_vis, color=(230,  80,  80))
                    add_image("layout_mask_overlay",
                              _side_by_side(gt_overlay, pred_overlay),
                              f"{cap} | layout_mask_overlay left=GT(green) right=PRED(red)")
                except Exception:
                    pass

                # FP (red) / FN (blue) error map
                try:
                    pred_prob_np = pred_prob_t.numpy() if pred_prob_t.ndim == 2 else \
                                   pred_prob_t.squeeze().numpy()
                    gt_np = gt_lm[b, s].detach().cpu().float().numpy() \
                        if torch.is_tensor(gt_lm[b, s]) else np.asarray(gt_lm[b, s])
                    err_map = _mask_fp_fn_map(pred_prob_np, gt_np)
                    add_image("layout_mask_error", err_map,
                              f"{cap} | mask_error red=FP blue=FN white=TP")
                except Exception:
                    pass
            except Exception:
                pass

        # ── Layout depth confidence ─────────────────────────────────────────
        ld_conf = batch.get("layout_depth_conf")
        if ld_conf is not None:
            try:
                conf_t = ld_conf[b, s] if ld_conf.dim() >= 3 else ld_conf
                conf_np = conf_t.detach().cpu().float().numpy()
                if conf_np.ndim == 3 and conf_np.shape[-1] == 1:
                    conf_np = conf_np[..., 0]
                # Robust 98th-pct normalization, then turbo colormap
                flat = conf_np[np.isfinite(conf_np)]
                if flat.size > 0:
                    hi = np.percentile(flat, 98.0)
                    lo = np.percentile(flat,  2.0)
                    rng = max(hi - lo, 1e-6)
                    conf_norm = np.clip((conf_np - lo) / rng, 0, 1)
                    add_image("layout_depth_conf", _apply_colormap(conf_norm),
                              f"{cap} | layout_depth_conf (turbo)")
            except Exception:
                pass

        # ── Layout depth error map (|pred - gt|) ────────────────────────────
        if gt_ld is not None and pred_ld is not None:
            try:
                gt_np = gt_ld[b, s].detach().cpu().float().numpy() \
                    if torch.is_tensor(gt_ld[b, s]) else np.asarray(gt_ld[b, s])
                pld = pred_ld[b, s].detach().cpu().float()
                if pld.dim() == 3 and pld.shape[-1] == 1:
                    pld = pld.squeeze(-1)
                pld_np = pld.numpy()
                if ld_mask is not None:
                    m = ld_mask[b, s].detach().cpu().bool().numpy() \
                        if torch.is_tensor(ld_mask[b, s]) else np.asarray(ld_mask[b, s], dtype=bool)
                else:
                    m = (gt_np > 1e-6)
                err_map = _depth_error_map(pld_np, gt_np, m)
                add_image("layout_depth_error", err_map,
                          f"{cap} | |pred - gt| (turbo, 95th-pct norm)")
            except Exception:
                pass

        # ── Layout normals  GT | PRED (or GT-only) ───────────────────────────
        _GT_NORMAL_KEYS   = ("layout_normals", "layout_normal_maps", "layout_normal")
        _PRED_NORMAL_KEYS = ("layout_normal_pred", "pred_layout_normals",
                             "layout_normals_pred", "layout_normal")

        gt_ln = None
        gt_ln_key = None
        for k in _GT_NORMAL_KEYS:
            if batch.get(k) is not None:
                gt_ln, gt_ln_key = batch[k], k
                break

        pred_ln = None
        for k in _PRED_NORMAL_KEYS:
            if k == gt_ln_key:          # avoid aliasing the same tensor
                continue
            if batch.get(k) is not None:
                pred_ln = batch[k]
                break

        if gt_ln is not None and pred_ln is not None:
            try:
                gt_vis   = prepare_normal_for_vis(gt_ln[b, s])
                pred_vis = prepare_normal_for_vis(pred_ln[b, s])
                add_image("layout_normals", _side_by_side(gt_vis, pred_vis),
                          f"{cap} | layout_normals left=GT right=PRED")
            except Exception:
                pass
            # Angular-error map (red-shaded; black=0°, red=>30°)
            try:
                gt_n_np = gt_ln[b, s].detach().cpu().float().numpy() \
                    if torch.is_tensor(gt_ln[b, s]) else np.asarray(gt_ln[b, s])
                pn_np = pred_ln[b, s].detach().cpu().float().numpy() \
                    if torch.is_tensor(pred_ln[b, s]) else np.asarray(pred_ln[b, s])
                err_map = _normal_angular_error_map(pn_np, gt_n_np)
                add_image("layout_normals_error", err_map,
                          f"{cap} | normals angular error (red=>=30deg)")
            except Exception:
                pass
        elif gt_ln is not None:
            try:
                add_image("layout_normals_gt", prepare_normal_for_vis(gt_ln[b, s]),
                          f"{cap} | layout_normals GT only")
            except Exception:
                pass

        # ── Seen mask (GT only) ──────────────────────────────────────────────
        seen = batch.get("seen_masks")
        if seen is None:
            seen = batch.get("seen_mask")
        if seen is None:
            seen = batch.get("point_masks")
        if seen is not None:
            try:
                add_image("seen_mask", prepare_binary_mask_for_vis(seen[b, s]),
                          f"{cap} | seen_mask")
            except Exception:
                pass

    if logs:
        try:
            if phase == "val":
                # Val visuals live on the ``val/step`` custom axis (registered
                # in init_wandb) so they aren't bound by the global monotonic
                # ``_step``. Attach the val step as a payload field and let
                # wandb commit normally.
                logs["val/step"] = step
                wb.log(logs)
            else:
                wb.log(logs, step=step)
        except Exception as exc:
            log.debug(f"wandb.log (images) failed: {exc}")
