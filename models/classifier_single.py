import timm
import torch.nn as nn

class TimmClassifier(nn.Module):
    def __init__(self, name="tf_efficientnet_b4", pretrained=True, num_classes=4):
        super().__init__()

        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,      # remove classifier
            global_pool="avg"
        )

        self.feature_dim = self.backbone.num_features
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)   # [B, feature_dim]
        logits = self.head(feats)
        return logits, feats