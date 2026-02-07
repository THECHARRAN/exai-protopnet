import torch.nn as nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights="DEFAULT")

        # remove avgpool + fc
        self.features = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        return self.features(x)
