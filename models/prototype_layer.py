# models/prototype_layer.py
import torch
import torch.nn as nn
import torch

class ClassSpecificPrototypeLayer(nn.Module):
    def __init__(self, num_classes, prototypes_per_class, channels):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.num_prototypes = num_classes * prototypes_per_class
        self.channels = channels
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, channels))
        # mapping from prototype index to class
        self.prototype_class_map = torch.arange(self.num_prototypes) // prototypes_per_class

    def forward(self, features):
        B, C, H, W = features.shape
        patches = features.flatten(2).permute(0,2,1)  # [B, HW, C]
        distances = torch.cdist(patches, self.prototypes)  # [B, HW, P]
        min_distances = distances.min(dim=1).values  # [B, P]
        similarity = -min_distances
        return similarity, distances  # distances [B, HW, P] kept for locating patch coords