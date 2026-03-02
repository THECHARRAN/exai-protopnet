# utils/adaptive_classspecific.py
import torch
from sklearn.cluster import MiniBatchKMeans
import numpy as np

def build_feature_bank_with_meta_and_label(model, loader, device):
    model.eval()
    bank = []
    meta = []
    labels_meta = []
    with torch.no_grad():
        dataset_base_idx = 0
        for images, labels in loader:
            B = images.size(0)
            images = images.to(device)
            feats = model.backbone(images)
            B, C, H, W = feats.shape
            patches = feats.flatten(2).permute(0,2,1)
            patches_reshaped = patches.reshape(-1, patches.size(-1)).cpu()
            bank.append(patches_reshaped)
            for b in range(B):
                lab = labels[b].item()
                for p in range(H*W):
                    meta.append((dataset_base_idx + b, p // W, p % W))
                    labels_meta.append(lab)
            dataset_base_idx += B
    feature_bank = torch.cat(bank, dim=0)
    labels_meta = np.array(labels_meta, dtype=np.int32)
    return feature_bank, meta, labels_meta

def refine_prototypes_classwise(feature_bank, labels_meta, num_classes, prototypes_per_class, random_state=0):
    centers_all = []
    for cls in range(num_classes):
        cls_idxs = np.where(labels_meta == cls)[0]
        if len(cls_idxs) == 0:
            centers = np.random.randn(prototypes_per_class, feature_bank.shape[1]).astype(np.float32)
        else:
            kmeans = MiniBatchKMeans(n_clusters=prototypes_per_class, batch_size=2048, random_state=random_state)
            kmeans.fit(feature_bank[cls_idxs].numpy())
            centers = kmeans.cluster_centers_.astype(np.float32)
        centers_all.append(centers)
    centers_all = np.vstack(centers_all)
    return torch.tensor(centers_all, dtype=torch.float32)