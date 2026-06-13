"""
Lightweight structural segmentation head for VGGT.

Reuses DPTHead (feature_only=True) for dense feature extraction,
then applies a small conv block to produce per-pixel class logits.

Output shape: (B, S, num_classes, H, W), raw logits, no softmax.
"""

import torch
import torch.nn as nn
from vggt.heads.dpt_head import DPTHead

# Feature dimension produced by DPTHead when feature_only=True
_DPT_FEATURE_DIM = 256


class SegHead(nn.Module):
    """
    Structural segmentation head.

    Attaches in the same style as the depth / layout-depth heads:
      - accepts the same aggregated_tokens_list, images, patch_start_idx args
      - internally uses a DPTHead(feature_only=True) for dense feature extraction
      - projects to num_classes logits via a small conv block

    Default: 6 structural classes
      wall | floor | ceiling | door | window | other-structure
    Non-structural pixels are assigned ignore_index (255) in the dataset and
    are excluded from the cross-entropy loss.
    """

    def __init__(self, dim_in: int, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = DPTHead(dim_in=dim_in, feature_only=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(_DPT_FEATURE_DIM, _DPT_FEATURE_DIM // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(_DPT_FEATURE_DIM // 2, num_classes, kernel_size=1),
        )

    def forward(self, aggregated_tokens_list, images, patch_start_idx):
        """
        Args:
            aggregated_tokens_list: list of tensors from Aggregator layers
            images: (B, S, 3, H, W)
            patch_start_idx: int

        Returns:
            seg_logits: (B, S, num_classes, H, W)
        """
        # features: (B, S, _DPT_FEATURE_DIM, H, W)
        features = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)
        B, S, C, H, W = features.shape
        logits = self.classifier(features.reshape(B * S, C, H, W))
        return logits.reshape(B, S, self.num_classes, H, W)
