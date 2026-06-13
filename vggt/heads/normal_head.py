"""Layout surface normal head for room-envelope reconstruction.

Outputs L2-normalized normals (B, S, 3, H, W).
Mirrors BinaryMaskHead: DPTHead(feature_only=True) → small Conv classifier → F.normalize.
"""

import torch.nn as nn
import torch.nn.functional as F

from vggt.heads.dpt_head import DPTHead


class NormalHead(nn.Module):
    """Predicts unit-length room-envelope surface normals.

    Architecture mirrors BinaryMaskHead: shared DPT feature extractor → lightweight
    3-channel classifier → L2 normalize per pixel.
    Output is L2-normalized; apply cosine loss externally.
    """

    def __init__(self, dim_in: int):
        super().__init__()
        self.feature_extractor = DPTHead(dim_in=dim_in, feature_only=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

    def forward(self, aggregated_tokens_list, images, patch_start_idx):
        """
        Args:
            aggregated_tokens_list: list of token tensors from the VGGT aggregator
            images: (B, S, 3, H, W) input images
            patch_start_idx: int, patch offset for positional embedding alignment

        Returns:
            normals: (B, S, 3, H, W) L2-normalized surface normals
        """
        # features: (B, S, 256, H, W)
        features = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)
        B, S, C, H, W = features.shape
        raw = self.classifier(features.reshape(B * S, C, H, W))  # (B*S, 3, H, W)
        normals = F.normalize(raw, dim=1)                         # unit-length per pixel
        return normals.reshape(B, S, 3, H, W)
