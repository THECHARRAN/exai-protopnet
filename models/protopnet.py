import torch.nn as nn
from .backbone import Backbone
from .prototype_layer import PrototypeLayer


class ProtoPNet(nn.Module):
    def __init__(self, num_classes=10, num_prototypes=30):
        super().__init__()

        self.backbone = Backbone()

        self.prototype_layer = PrototypeLayer(
            num_prototypes=num_prototypes,
            channels=512
        )

        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):

        features = self.backbone(x)

        similarity = self.prototype_layer(features)

        logits = self.classifier(similarity)

        return logits, similarity
