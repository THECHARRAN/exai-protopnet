import torch
import torch.nn.functional as F
import numpy as np
import cv2


def compute_prototype_activation(model, x):

    feats = model.backbone(x)

    B,C,H,W = feats.shape

    patches = feats.flatten(2).permute(0,2,1)

    proto = model.prototype_layer.prototypes

    dists = torch.cdist(patches, proto)

    similarity = 1/(1+dists)

    similarity = similarity.view(B,H,W,-1)

    return similarity, feats


def generate_heatmap(model, x):

    sim,_ = compute_prototype_activation(model,x)

    heat = sim.max(dim=3)[0]

    heat = heat[0]

    heat = (heat-heat.min())/(heat.max()-heat.min()+1e-6)

    heat = F.interpolate(
        heat.unsqueeze(0).unsqueeze(0),
        size=(384,384),
        mode="bicubic",
        align_corners=False
    )[0,0].cpu().detach().numpy()

    return heat


def get_top_prototypes(model,x,topk=3):

    sim,_ = compute_prototype_activation(model,x)

    sim = sim.max(dim=(1,2))[0]

    sim = sim[0]

    values,idx = torch.topk(sim,topk)

    return idx.cpu().numpy(),values.cpu().numpy()


def detect_tumor_bbox(heatmap,thr=0.6):

    mask = heatmap>thr

    coords = np.column_stack(np.where(mask))

    if len(coords)==0:

        return None

    y0,x0 = coords.min(axis=0)

    y1,x1 = coords.max(axis=0)

    return (x0,y0,x1,y1)


def overlay_heatmap(image, heatmap, bbox=None):

    import numpy as np
    import cv2

    # Convert PIL → numpy
    image = np.array(image)

    # If grayscale convert to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize heatmap to image size
    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    # Blend heatmap with image
    overlay = cv2.addWeighted(
        image,
        0.65,
        heatmap_color,
        0.35,
        0
    )

    # Draw bounding box
    if bbox:

        x0,y0,x1,y1 = bbox

        cv2.rectangle(
            overlay,
            (x0,y0),
            (x1,y1),
            (0,255,0),
            2
        )

    return overlay


def compute_metrics(heatmap):

    mask = heatmap>0.6

    tumor_area = mask.sum()/heatmap.size

    coords = np.argwhere(mask)

    centroid = coords.mean(axis=0) if len(coords)>0 else (0,0)

    intensity = heatmap[mask].mean() if mask.any() else 0

    irregularity = heatmap[mask].std() if mask.any() else 0

    return {

        "tumor_area_%":float(tumor_area*100),

        "centroid":centroid.tolist(),

        "activation_intensity":float(intensity),

        "edge_irregularity":float(irregularity)
    }