import torch
import torch.nn as nn


class PrototypeLayer(nn.Module):
    def __init__(self, num_prototypes, channels):
        super().__init__()

        # each prototype is a learnable vector
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, channels)
        )

    def forward(self, features):
        """
        features: [B, C, H, W]
        returns: similarity [B, num_prototypes]
        """

        B, C, H, W = features.shape

        # flatten spatial patches
        patches = features.flatten(2).permute(0, 2, 1)
        # shape → [B, HW, C]

        # compute L2 distance
        distances = torch.cdist(patches, self.prototypes)
        # shape → [B, HW, P]

        # take closest patch for each prototype
        min_distances = distances.min(dim=1).values
        # shape → [B, P]

        similarity = -min_distances

        return similarity
