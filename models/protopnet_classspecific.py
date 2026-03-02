# models/protopnet_classspecific.py
import torch.nn as nn
from models.backbones import TimmFeatureBackbone
from models.prototype_layer import ClassSpecificPrototypeLayer
import torch

class ProtoPNetCS(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b3_ns", pretrained=True, num_classes=4, prototypes_per_class=15, img_size=320):
        super().__init__()
        self.backbone = TimmFeatureBackbone(name=backbone_name, pretrained=pretrained, img_size=img_size)
        channels = self.backbone.out_ch
        self.feature_dim = self.backbone.out_ch
        self.prototype_layer = ClassSpecificPrototypeLayer(num_classes=num_classes, prototypes_per_class=prototypes_per_class, channels=channels)
        self.classifier = nn.Linear(self.prototype_layer.num_prototypes, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        sim, distances = self.prototype_layer(feats)
        logits = self.classifier(sim)
        return logits, sim, feats, distances