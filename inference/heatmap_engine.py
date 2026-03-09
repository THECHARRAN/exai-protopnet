import torch
import torch.nn.functional as F
import numpy as np

def generate_heatmap(proto_model, image_tensor, device):

    proto_model.eval()

    with torch.no_grad():

        feats = proto_model.backbone(image_tensor)

        B,C,H,W = feats.shape

        patches = feats.flatten(2).permute(0,2,1)

        proto = proto_model.prototype_layer.prototypes

        dists = torch.cdist(patches,proto)

        min_d,_ = dists.min(dim=2)

        activation = 1/(1+min_d)

        activation = activation.view(H,W)

        activation = activation.cpu()

        activation = (activation - activation.min()) / (
            activation.max() - activation.min() + 1e-6
        )

        heatmap = F.interpolate(
            activation.unsqueeze(0).unsqueeze(0),
            size=(384,384),
            mode="bicubic",
            align_corners=False
        )[0,0].numpy()

    return heatmap