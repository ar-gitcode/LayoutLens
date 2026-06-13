# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.seg_head import SegHead
from vggt.heads.track_head import TrackHead
from vggt.heads.mask_head import BinaryMaskHead
from vggt.heads.normal_head import NormalHead
from vggt.heads.oca import (
    OCABlock,
    downsample_mask_logits_to_token_grid,
    patch_tokens_to_spatial,
    spatial_to_patch_tokens,
)


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 enable_layout_depth=False, enable_seg=False, num_seg_classes=4,
                 enable_layout_mask=False, enable_layout_normal=False,
                 enable_oca=False, oca=None):
        super().__init__()

        self.patch_size = patch_size
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.layout_depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_layout_depth else None
        self.seg_head = SegHead(dim_in=2 * embed_dim, num_classes=num_seg_classes) if enable_seg else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        # Room-envelope heads (Phase 1: mask_head; Phase 2: normal_head)
        self.mask_head = BinaryMaskHead(dim_in=2 * embed_dim) if enable_layout_mask else None
        self.normal_head = NormalHead(dim_in=2 * embed_dim) if enable_layout_normal else None

        # Occlusion-Aware Cross-View Attention (OCA), applied to the layout
        # depth head's input only. Built only when both flags are set, so the
        # default (enable_oca=False) preserves existing E1-E7 behaviour exactly.
        if enable_oca:
            if not enable_layout_depth:
                raise ValueError("enable_oca=True requires enable_layout_depth=True.")
            if not enable_layout_mask:
                raise ValueError(
                    "enable_oca=True requires enable_layout_mask=True (OCA needs mask "
                    "logits to compute the clutter gate)."
                )
            oca_kwargs = dict(oca) if oca is not None else {}
            # Default target layers; cannot live in a mutable default arg.
            self._oca_target_layers = list(oca_kwargs.pop("target_layers", [23]))
            self._oca_camera_source = oca_kwargs.pop("camera_source", "gt")
            self._oca_mask_warmup_steps = int(oca_kwargs.pop("mask_warmup_steps", 0))
            self.oca = OCABlock(token_dim=2 * embed_dim, patch_size=patch_size, **oca_kwargs)
        else:
            self.oca = None
            self._oca_target_layers = []
            self._oca_camera_source = "gt"
            self._oca_mask_warmup_steps = 0

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,
                intrinsics: torch.Tensor = None, extrinsics: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            intrinsics (torch.Tensor, optional): GT camera intrinsics, shape [B, S, 3, 3], at full
                image-pixel resolution. Used by OCA when ``enable_oca=True`` and
                ``oca.use_epipolar_bias=True``. Ignored otherwise.
            extrinsics (torch.Tensor, optional): GT camera extrinsics, shape [B, S, 3, 4],
                world-to-cam (OpenCV). Same usage rules as ``intrinsics``.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - layout_depth (torch.Tensor): Predicted layout-depth maps with shape [B, S, H, W, 1] (if enabled)
                - layout_depth_conf (torch.Tensor): Confidence for layout-depth predictions with shape [B, S, H, W] (if enabled)
                - seg_logits (torch.Tensor): Structural segmentation logits [B, S, num_classes, H, W] (if enabled)
                - layout_mask_logits (torch.Tensor): Binary layout mask logits [B, S, 1, H, W] (if enable_layout_mask)
                - layout_normal (torch.Tensor): Layout surface normals [B, S, 3, H, W] (Phase 2, if enable_layout_normal)
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        B, S, _, H, W = images.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            # Run mask_head BEFORE layout_depth_head so OCA can consume mask logits.
            # When OCA is disabled this reorder is a behaviour-neutral change
            # (mask_head and layout_depth_head are independent).
            if self.mask_head is not None:
                predictions["layout_mask_logits"] = self.mask_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

            if self.layout_depth_head is not None:
                tokens_for_ld = self._maybe_apply_oca(
                    aggregated_tokens_list=aggregated_tokens_list,
                    patch_start_idx=patch_start_idx,
                    patch_h=patch_h, patch_w=patch_w,
                    mask_logits=predictions.get("layout_mask_logits"),
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    num_views=S,
                )
                layout_depth, layout_depth_conf = self.layout_depth_head(
                    tokens_for_ld, images=images, patch_start_idx=patch_start_idx
                )
                predictions["layout_depth"] = layout_depth
                predictions["layout_depth_conf"] = layout_depth_conf

            if self.seg_head is not None:
                seg_logits = self.seg_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["seg_logits"] = seg_logits

            if self.normal_head is not None:
                predictions["layout_normal"] = self.normal_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    # ------------------------------------------------------------------ #
    # OCA helper
    # ------------------------------------------------------------------ #
    def _maybe_apply_oca(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
        patch_h: int,
        patch_w: int,
        mask_logits,
        intrinsics,
        extrinsics,
        num_views: int,
    ):
        """Return a token list with OCA applied at ``self._oca_target_layers``.

        When OCA is disabled, returns the input list unchanged. Otherwise
        builds a *new* list, the original is left intact so other heads see
        unchanged features.
        """
        if self.oca is None:
            return aggregated_tokens_list
        if mask_logits is None:
            # OCA was constructed but no mask logits were produced. Fall back
            # to identity rather than raising, protects ad-hoc inference.
            return aggregated_tokens_list
        if num_views < 2:
            # No cross-view attention possible.
            return aggregated_tokens_list

        # Mask logits at token resolution, fp32-stable.
        mask_at_token = downsample_mask_logits_to_token_grid(
            mask_logits.float(), H_p=patch_h, W_p=patch_w
        )

        # Build a NEW list with only target layers replaced. This avoids
        # mutating the shared list other heads rely on.
        new_list = list(aggregated_tokens_list)
        target_layers = self._oca_target_layers or [len(new_list) - 1]
        for layer_idx in target_layers:
            layer_tokens = new_list[layer_idx]
            leading = layer_tokens[:, :, :patch_start_idx]
            spatial = patch_tokens_to_spatial(
                layer_tokens, patch_start_idx=patch_start_idx, H_p=patch_h, W_p=patch_w
            )
            updated_spatial = self.oca(
                spatial,
                mask_at_token,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
            new_list[layer_idx] = spatial_to_patch_tokens(updated_spatial, leading)

        return new_list

