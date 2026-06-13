# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Occlusion-Aware Cross-View Attention (OCA) for amodal layout reconstruction.

OCA augments per-view patch tokens with a residual update derived from cross-
view attention. Clutter pixels in the reference view (where the layout is
occluded) emit queries; visible-layout pixels in source views form the
key/value pool. Attention logits may optionally include an epipolar geometry
bias derived from camera intrinsics/extrinsics. The output projection is
zero-initialised so OCA starts as exact identity.

Equations (schematic):
    m         = sigmoid(mask_logits_at_token)            # (B, V, 1, H_p, W_p)
    m_clutter = 1 - m
    Q         = Wq( F * m_clutter )                      # ref-view queries
    K, V      = Wk/Wv( F * m )                           # src-view keys/values (all views)
    logits    = Q K^T / sqrt(d) + B_geo                  # B_geo shared across heads
    attn      = softmax(logits, dim=last)                # ref attends across all views' tokens
    out       = attn @ V
    update    = m_clutter * Wout(out)                    # zero at init
    F_out     = F + update                                # residual

Self-attention to the same view is suppressed by masking those score entries
to -inf (so attention is effectively over the (V-1) other views' tokens).

Each component is independently config-gated:
    use_epipolar_bias: enable/disable the geometric prior.
    Wout zero-init:    OCA starts at identity regardless of other settings.
    V=1 inputs:        identity passthrough (no cross-view possible).

Tensor shape conventions:
    tokens       (B, V, H_p, W_p, C)                  # patch features
    mask_logits  (B, V, 1, H_p, W_p)                  # at token resolution
    intrinsics   (B, V, 3, 3)                         # full image-pixel space
    extrinsics   (B, V, 3, 4)                         # world-to-cam (OpenCV)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCABlock(nn.Module):
    """Mask-gated cross-view attention with optional epipolar bias."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int = 8,
        qkv_dim: int = 512,
        use_epipolar_bias: bool = False,
        alpha: float = 1.0,
        sigma_pixels: float = 4.0,
        bias_clamp_min: float = -50.0,
        bias_clamp_max: float = 0.0,
        patch_size: int = 14,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if qkv_dim % num_heads != 0:
            raise ValueError(
                f"qkv_dim ({qkv_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.token_dim = token_dim
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.head_dim = qkv_dim // num_heads
        self.use_epipolar_bias = use_epipolar_bias
        self.alpha = alpha
        self.sigma_pixels = sigma_pixels
        self.bias_clamp_min = bias_clamp_min
        self.bias_clamp_max = bias_clamp_max
        self.patch_size = patch_size
        self.eps = eps

        self.q_proj = nn.Linear(token_dim, qkv_dim)
        self.k_proj = nn.Linear(token_dim, qkv_dim)
        self.v_proj = nn.Linear(token_dim, qkv_dim)
        self.out_proj = nn.Linear(qkv_dim, token_dim)

        # Zero-init the residual output: at step 0 the OCA update is identically zero
        # for any input, so loading a pre-OCA checkpoint behaves identically to
        # training without OCA. Only `out_proj.weight/bias` need to be zero,
        # Q/K/V projections receive useful gradients via the loss as soon as the
        # residual leaves zero in the first optimizer step.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        mask_logits: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply OCA residual update.

        Args:
            tokens:       (B, V, H_p, W_p, C)  patch tokens.
            mask_logits:  (B, V, 1, H_p, W_p)  layout-mask logits at token resolution.
                          Sigmoid of these is the layout confidence m.
            intrinsics:   (B, V, 3, 3)         full-image-pixel-space K. Required iff
                          use_epipolar_bias=True.
            extrinsics:   (B, V, 3, 4)         world-to-cam [R|t]. Required iff
                          use_epipolar_bias=True.

        Returns:
            tokens_out:   (B, V, H_p, W_p, C)  same shape as input.
        """
        B, V, H_p, W_p, C = tokens.shape
        N = H_p * W_p

        # Single-view fallback: cross-view attention is undefined.
        if V < 2:
            return tokens

        # m at token resolution, broadcast to (B, V, N, 1) for channel multiply.
        m = torch.sigmoid(mask_logits.reshape(B, V, 1, N).transpose(-1, -2))  # (B, V, N, 1)
        m_clutter = 1.0 - m

        # Flatten spatial axes.
        F_flat = tokens.reshape(B, V, N, C)

        # Mask-gated projections.
        Q = self.q_proj(F_flat * m_clutter)                                # (B, V, N, qkv_dim)
        K = self.k_proj(F_flat * m)                                        # (B, V, N, qkv_dim)
        Vv = self.v_proj(F_flat * m)                                       # (B, V, N, qkv_dim)

        # Multi-head reshape: (B, V, num_heads, N, head_dim).
        Q = Q.reshape(B, V, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        K = K.reshape(B, V, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        Vv = Vv.reshape(B, V, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # Flatten src views into a single KV pool of size V*N.
        # (B, num_heads, V*N, head_dim)
        K_pool = K.permute(0, 2, 1, 3, 4).reshape(B, self.num_heads, V * N, self.head_dim)
        V_pool = Vv.permute(0, 2, 1, 3, 4).reshape(B, self.num_heads, V * N, self.head_dim)

        # Attention scores: (B, V_ref, num_heads, N_ref, V_src*N_src)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(
            Q,                                       # (B, V_ref, H, N, hd)
            K_pool.unsqueeze(1).transpose(-1, -2),   # (B, 1,    H, hd, V*N)
        ) * scale

        # Suppress self-view contributions: for each ref view v_r, mask out the
        # slice of the KV pool whose source view equals v_r.
        device = scores.device
        v_idx = torch.arange(V, device=device)
        self_mask_view = v_idx.view(V, 1) == v_idx.view(1, V)  # (V_ref, V_src) diag-True
        # Expand to (V_ref, V_src*N_src), True at self positions.
        self_mask_pool = self_mask_view.unsqueeze(-1).expand(V, V, N).reshape(V, V * N)
        scores = scores.masked_fill(
            self_mask_pool.view(1, V, 1, 1, V * N), float("-inf")
        )

        # Optional epipolar bias. Hard error if requested but cameras absent,
        # silent bypass would invalidate any E10/E11 evaluation.
        if self.use_epipolar_bias:
            if intrinsics is None or extrinsics is None:
                raise RuntimeError(
                    "OCA epipolar bias is enabled but intrinsics/extrinsics were "
                    "not passed to model.forward. Configure your eval/inference "
                    "code to pass GT cameras (see evaluations/src/common/_oca_eval_helpers.py) "
                    "or disable use_epipolar_bias."
                )
            B_geo = self._compute_epipolar_bias(intrinsics, extrinsics, H_p, W_p)
            # Reshape (B, V_ref, V_src, N_ref, N_src) -> (B, V_ref, 1, N_ref, V_src*N_src)
            # so it broadcasts over the heads dim (shared across heads).
            B_geo = B_geo.permute(0, 1, 3, 2, 4).reshape(B, V, N, V * N).unsqueeze(2)
            # Cast to scores dtype to avoid AMP/dtype-promotion surprises.
            scores = scores + B_geo.to(scores.dtype)

        # Softmax over the KV pool.
        attn = torch.softmax(scores, dim=-1)
        # Defensive: if a row was all -inf (shouldn't happen because V>=2 already
        # leaves V-1 valid src views), softmax produces NaN. Replace with zero.
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)

        # Attention output: (B, V_ref, H, N, V*N) @ (B, 1, H, V*N, hd).
        out = torch.matmul(attn, V_pool.unsqueeze(1))                      # (B, V, H, N, hd)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, V, N, self.qkv_dim)    # (B, V, N, qkv_dim)

        # Output projection (zero-init at start) and clutter-gated residual.
        update = self.out_proj(out)                                        # (B, V, N, C)
        update = update * m_clutter                                        # gate
        update = update.reshape(B, V, H_p, W_p, C)
        return tokens + update

    # ------------------------------------------------------------------ #
    # Epipolar geometry
    # ------------------------------------------------------------------ #

    def _compute_epipolar_bias(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        H_p: int,
        W_p: int,
    ) -> torch.Tensor:
        """Compute the per-pair epipolar attention bias.

        Args:
            intrinsics:   (B, V, 3, 3)
            extrinsics:   (B, V, 3, 4) world-to-cam [R | t]
            H_p, W_p:     token grid size

        Returns:
            bias:         (B, V_ref, V_src, N_ref, N_src) in the original input dtype.
        """
        B, V = intrinsics.shape[:2]
        N = H_p * W_p
        device = intrinsics.device
        orig_dtype = intrinsics.dtype

        # Cast to fp32 for numerical stability under bf16 autocast.
        K = intrinsics.float()
        E = extrinsics.float()
        R = E[..., :3, :3]                                                 # (B, V, 3, 3)
        t = E[..., :3, 3]                                                  # (B, V, 3)

        # Relative pose for every (ref, src) pair.
        R_ref = R.unsqueeze(2)                                             # (B, V_r, 1, 3, 3)
        R_src = R.unsqueeze(1)                                             # (B, 1, V_s, 3, 3)
        t_ref = t.unsqueeze(2)                                             # (B, V_r, 1, 3)
        t_src = t.unsqueeze(1)                                             # (B, 1, V_s, 3)

        R_sr = R_src @ R_ref.transpose(-1, -2)                             # (B, V_r, V_s, 3, 3)
        t_sr = t_src - (R_sr @ t_ref.unsqueeze(-1)).squeeze(-1)            # (B, V_r, V_s, 3)

        # Skew-symmetric matrix of t_sr.
        t_x = self._skew_symmetric(t_sr)                                   # (B, V_r, V_s, 3, 3)

        # K^{-1} with eps on diagonal for numerical safety (K is full-rank in
        # any well-formed batch, so this is belt-and-braces only).
        eye = torch.eye(3, device=device, dtype=K.dtype).view(1, 1, 3, 3)
        K_inv = torch.linalg.inv(K + self.eps * eye)                       # (B, V, 3, 3)
        K_ref_inv = K_inv.unsqueeze(2)                                     # (B, V_r, 1, 3, 3)
        K_src_inv_T = K_inv.transpose(-1, -2).unsqueeze(1)                 # (B, 1, V_s, 3, 3)

        # Fundamental matrix from ref-pixels to src-line.
        F_pairs = K_src_inv_T @ t_x @ R_sr @ K_ref_inv                     # (B, V_r, V_s, 3, 3)

        # Token-pixel grid in homogeneous image-pixel coordinates.
        ps = float(self.patch_size)
        ys = torch.arange(H_p, device=device, dtype=K.dtype) * ps + 0.5 * ps
        xs = torch.arange(W_p, device=device, dtype=K.dtype) * ps + 0.5 * ps
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")                     # (H_p, W_p)
        ones = torch.ones_like(xx)
        x_h = torch.stack([xx, yy, ones], dim=-1).reshape(N, 3)            # (N, 3)

        # Epipolar lines in src view: l = F @ x_ref_h
        lines = torch.einsum("bvsij,nj->bvsni", F_pairs, x_h)              # (B, V_r, V_s, N_r, 3)

        # Distance from src token-pixel to line: |l . x_src| / sqrt(a^2 + b^2)
        numerator = torch.einsum("bvsni,mi->bvsnm", lines, x_h).abs()      # (B, V_r, V_s, N_r, N_s)
        denom = torch.sqrt(lines[..., 0] ** 2 + lines[..., 1] ** 2 + self.eps)  # (B, V_r, V_s, N_r)
        d = numerator / denom.unsqueeze(-1)                                # (B, V_r, V_s, N_r, N_s)

        bias = -self.alpha * (d * d) / (self.sigma_pixels ** 2)
        bias = bias.clamp(min=self.bias_clamp_min, max=self.bias_clamp_max)

        # Degenerate camera pairs: ||t_sr|| ~ 0 implies a degenerate F. Zero
        # the bias for those pairs so attention falls back to image content.
        t_norm = t_sr.norm(dim=-1)                                         # (B, V_r, V_s)
        degenerate = t_norm < self.eps                                      # (B, V_r, V_s)
        bias = bias.masked_fill(
            degenerate.unsqueeze(-1).unsqueeze(-1), 0.0
        )

        return bias.to(orig_dtype)

    @staticmethod
    def _skew_symmetric(t: torch.Tensor) -> torch.Tensor:
        """Build skew-symmetric [t]_x from (..., 3) vectors. Returns (..., 3, 3)."""
        zero = torch.zeros_like(t[..., 0])
        row0 = torch.stack([zero, -t[..., 2], t[..., 1]], dim=-1)
        row1 = torch.stack([t[..., 2], zero, -t[..., 0]], dim=-1)
        row2 = torch.stack([-t[..., 1], t[..., 0], zero], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)


def downsample_mask_logits_to_token_grid(
    mask_logits: torch.Tensor,
    H_p: int,
    W_p: int,
) -> torch.Tensor:
    """Average-pool full-resolution mask logits to the token grid.

    Args:
        mask_logits: (B, V, 1, H, W) raw logits.
        H_p, W_p:    token grid size.

    Returns:
        (B, V, 1, H_p, W_p)
    """
    B, V, _, H, W = mask_logits.shape
    flat = mask_logits.reshape(B * V, 1, H, W)
    pooled = F.adaptive_avg_pool2d(flat, output_size=(H_p, W_p))
    return pooled.reshape(B, V, 1, H_p, W_p)


def patch_tokens_to_spatial(
    layer_tokens: torch.Tensor,
    patch_start_idx: int,
    H_p: int,
    W_p: int,
) -> torch.Tensor:
    """Extract patch tokens from one aggregator layer and reshape spatially.

    Args:
        layer_tokens:    (B, V, P+patch_start_idx, C), one entry of aggregated_tokens_list.
        patch_start_idx: number of leading special tokens (camera + register).
        H_p, W_p:        token grid size.

    Returns:
        (B, V, H_p, W_p, C)
    """
    patch = layer_tokens[:, :, patch_start_idx:]                           # (B, V, P, C)
    B, V, P, C = patch.shape
    if P != H_p * W_p:
        raise ValueError(
            f"Expected {H_p * W_p} patch tokens but got {P}. "
            f"Mismatch between H_p={H_p}, W_p={W_p} and the aggregator output."
        )
    return patch.reshape(B, V, H_p, W_p, C)


def spatial_to_patch_tokens(
    spatial: torch.Tensor,
    leading_special: torch.Tensor,
) -> torch.Tensor:
    """Inverse of `patch_tokens_to_spatial`: rebuild a full layer tensor.

    Args:
        spatial:         (B, V, H_p, W_p, C) updated patch tokens.
        leading_special: (B, V, patch_start_idx, C) the original leading special
                         tokens, kept as-is.

    Returns:
        (B, V, patch_start_idx + H_p*W_p, C), the same shape as one
        entry of aggregated_tokens_list.
    """
    B, V, H_p, W_p, C = spatial.shape
    patch = spatial.reshape(B, V, H_p * W_p, C)
    return torch.cat([leading_special, patch], dim=2)
