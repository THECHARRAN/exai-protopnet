# utils/explain.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm

def get_topk_patches_for_prototypes(model, feature_bank, meta, dataset, topk=5, device='cpu'):
    """
    For each prototype, find topk nearest patches in the feature_bank.
    Returns a list of length P where each element is:
      [(dataset_idx, row, col, distance), ...] (topk items)
    Note: feature_bank: [N, C] (cpu tensor); model.prototype_layer.prototypes on device
    """
    # prototypes on CPU for distance calc
    prototypes = model.prototype_layer.prototypes.detach().cpu()  # [P, C]

    # compute distances: use cdist to get [N, P], then for each prototype pick smallest distances
    # (we will compute per-prototype neighbors)
    with torch.no_grad():
        # distances shape [N, P]
        distances = torch.cdist(feature_bank, prototypes)  # CPU tensors

    P = prototypes.shape[0]
    results = []

    for p in range(P):
        dists_p = distances[:, p]  # [N]
        topk_vals, topk_idxs = torch.topk(-dists_p, k=topk)  # negative to get smallest distances
        items = []
        for v, idx in zip(topk_vals, topk_idxs):
            patch_idx = idx.item()
            d = -v.item()
            dataset_idx, row, col = meta[patch_idx]
            items.append((dataset_idx, row, col, d))
        results.append(items)
    return results


def crop_patch_from_image(img_tensor, row, col, H_feat, W_feat, image_size=128):
    """
    img_tensor: [C, H, W] normalized (the same transform used in dataset)
    row, col: patch location in feature grid (0..H_feat-1, 0..W_feat-1)
    H_feat, W_feat: feature map spatial dims (e.g. 4,4)
    Returns: patch image tensor [C, patch_size, patch_size] in CPU (unnormalized to [0,1])
    """
    C, H, W = img_tensor.shape
    patch_h = image_size // H_feat
    patch_w = image_size // W_feat
    top = row * patch_h
    left = col * patch_w
    patch = img_tensor[:, top: top+patch_h, left: left+patch_w].cpu().clone()
    # de-normalize assuming mean=0.5 std=0.5 used earlier
    patch = patch * 0.5 + 0.5
    patch = torch.clamp(patch, 0, 1)
    return patch


def visualize_prototype_neighbors(model, dataset, neighbors, H_feat=4, W_feat=4, topk=5, cols=5, save_path=None):
    """
    model: ProtoPNet (for prototype index label)
    dataset: dataset object (dataset[i] returns (img_tensor, label))
    neighbors: output of get_topk_patches_for_prototypes
    Visualizes a grid where each row = prototype, columns = topk example patches
    """
    P = len(neighbors)
    rows = P
    # For big P, limit to a few prototypes to show
    max_display_protos = min(8, P)  # show first 8 prototypes by default
    fig, axs = plt.subplots(max_display_protos, topk, figsize=(topk*2.2, max_display_protos*2.2))

    for i in range(max_display_protos):
        for j in range(topk):
            ds_idx, row, col, dist = neighbors[i][j]
            img_tensor, _ = dataset[ds_idx]    # returns transformed image tensor [C,H,W]
            patch = crop_patch_from_image(img_tensor, row, col, H_feat, W_feat)
            ax = axs[i][j] if max_display_protos>1 else axs[j]
            ax.imshow(patch.permute(1,2,0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if j==0:
                ax.set_ylabel(f"Proto {i}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def visualize_heatmap_for_image(model, image, device, resize_to=128):
    """
    image: single image tensor [C,H,W] (normalized)
    Produces a heatmap showing how close each patch is to ANY prototype.
    Higher value == patch is closer to some prototype.
    """
    model.eval()
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.backbone(image)  # [1, C, H, W]
        B, C, Hf, Wf = feats.shape
        patches = feats.flatten(2).permute(0,2,1)  # [1, HW, C]
        # distances: [1, HW, P]
        proto = model.prototype_layer.prototypes.to(device)  # [P, C]
        dists = torch.cdist(patches, proto)  # [1, HW, P]
        min_dists, _ = dists.min(dim=2)  # [1, HW]
        activation = 1.0 / (1.0 + min_dists)  # inverse distance to show closeness
        activation = activation.view(Hf, Wf).cpu().numpy()
    # upsample to image size
    activation_map = np.array(activation)
    activation_map_resized = F.interpolate(torch.tensor(activation_map).unsqueeze(0).unsqueeze(0),
                                          size=(resize_to, resize_to), mode='bilinear', align_corners=False)[0,0].numpy()
    # show
    img = (image[0].cpu() * 0.5 + 0.5).permute(1,2,0).numpy()
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.imshow(activation_map_resized, cmap='jet', alpha=0.45)
    plt.title("Prototype activation heatmap (higher = more prototype-like)")
    plt.axis('off')
    plt.show()
def evaluate_accuracy(model, loader, device):
    """
    Simple accuracy evaluation
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):

            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)

            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total

    print(f"Accuracy: {acc:.2f}%")

    return acc