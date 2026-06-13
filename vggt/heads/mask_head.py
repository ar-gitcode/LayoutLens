"""Binary layout mask head for room-envelope reconstruction.

Outputs raw logits (B, S, 1, H, W). Apply BCEWithLogitsLoss externally.
Mirrors SegHead pattern: DPTHead(feature_only=True) → small Conv classifier.
"""

import torch.nn as nn

from vggt.heads.dpt_head import DPTHead


class BinaryMaskHead(nn.Module):
    """Predicts a binary layout mask: 1=structural surface visible, 0=occluded/non-structural.

    Architecture mirrors SegHead: shared DPT feature extractor → lightweight 1-channel classifier.
    Output is raw logits; apply sigmoid + threshold (0.5) at inference, BCEWithLogitsLoss at train.
    """

    def __init__(self, dim_in: int):
        super().__init__()
        self.feature_extractor = DPTHead(dim_in=dim_in, feature_only=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, aggregated_tokens_list, images, patch_start_idx):
        """
        Args:
            aggregated_tokens_list: list of token tensors from the VGGT aggregator
            images: (B, S, 3, H, W) input images
            patch_start_idx: int, patch offset for positional embedding alignment

        Returns:
            logits: (B, S, 1, H, W) raw logits (no sigmoid applied)
        """
        # features: (B, S, 256, H, W)
        features = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)
        B, S, C, H, W = features.shape
        logits = self.classifier(features.reshape(B * S, C, H, W))  # (B*S, 1, H, W)
        return logits.reshape(B, S, 1, H, W)
