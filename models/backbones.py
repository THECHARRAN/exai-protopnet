# models/backbones.py
import timm
import torch.nn as nn
import torch

class TimmFeatureBackbone(nn.Module):
    def __init__(self, name="tf_efficientnet_b3_ns", pretrained=True, out_ch=512, img_size=320):
        super().__init__()
        self.name = name
        self.img_size = img_size
        m = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(len(timm.create_model(name, pretrained=pretrained).feature_info)-1,))
        self.body = m
        dummy = torch.zeros(1,3,img_size,img_size)
        with torch.no_grad():
            feats = self.body(dummy)[0]
        self.out_ch = feats.shape[1]

    def forward(self, x):
        out = self.body(x)[0]
        return out  # [B, C, H, W]