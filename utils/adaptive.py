# utils/adaptive.py
import torch
from sklearn.cluster import MiniBatchKMeans

def build_feature_bank_with_meta(model, loader, device):
    """
    Builds a feature bank AND metadata for each patch.

    Returns:
        feature_bank: torch.FloatTensor of shape [N_patches, C]
        meta: list of tuples [(dataset_idx, patch_row, patch_col), ...] length N_patches
    NOTE: loader must be created with shuffle=False for correct dataset_idx mapping.
    """
    model.eval()
    bank = []
    meta = []

    with torch.no_grad():
        dataset_base_idx = 0
        for images, _ in loader:
            B = images.size(0)
            images = images.to(device)

            features = model.backbone(images)  # [B, C, H, W]
            B, C, H, W = features.shape

            # patches: [B, HW, C]
            patches = features.flatten(2).permute(0, 2, 1)

            # convert to [B*HW, C] and collect meta
            patches_reshaped = patches.reshape(-1, patches.size(-1)).cpu()  # CPU to reduce GPU mem

            bank.append(patches_reshaped)

            # record meta: for each batch item and each spatial patch index
            for b in range(B):
                dataset_idx = dataset_base_idx + b
                for p in range(H * W):
                    row = p // W
                    col = p % W
                    meta.append((dataset_idx, row, col))
            dataset_base_idx += B

    feature_bank = torch.cat(bank, dim=0)  # [N, C]
    return feature_bank, meta


def refine_prototypes(feature_bank, num_prototypes, random_state=0):
    """
    Run MiniBatchKMeans and return centers as torch tensor [num_prototypes, C]
    """
    kmeans = MiniBatchKMeans(n_clusters=num_prototypes, batch_size=2048, random_state=random_state)
    kmeans.fit(feature_bank.numpy())
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    return centers


def update_prototypes(model, centers, device):
    """
    Replace model prototypes with cluster centers.
    Assumes model.prototype_layer.prototypes shape [P, C]
    """
    centers = centers.to(device)
    model.prototype_layer.prototypes.data.copy_(centers)
