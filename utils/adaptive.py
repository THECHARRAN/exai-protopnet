import torch


def build_feature_bank(model, loader, device):
    """
    Collect ALL patch embeddings from dataset.

    Returns:
        feature_bank → [N, C]
    """

    model.eval()

    bank = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            # only backbone features
            features = model.backbone(images)
            # [B, C, H, W]

            patches = features.flatten(2).permute(0, 2, 1)
            # [B, HW, C]

            patches = patches.reshape(-1, patches.size(-1))
            # [B*HW, C]

            bank.append(patches.cpu())

    feature_bank = torch.cat(bank, dim=0)

    return feature_bank
from sklearn.cluster import MiniBatchKMeans


def refine_prototypes(feature_bank, num_prototypes):
    """
    Run k-means clustering on feature bank.

    Returns:
        centers → [num_prototypes, C]
    """

    kmeans = MiniBatchKMeans(
        n_clusters=num_prototypes,
        batch_size=2048,
        random_state=0
    )

    kmeans.fit(feature_bank.numpy())

    centers = torch.tensor(kmeans.cluster_centers_)

    return centers
def update_prototypes(model, centers, device):
    """
    Replace current prototypes with cluster centers
    """

    centers = centers.to(device)

    model.prototype_layer.prototypes.data.copy_(centers)
