# demo_visualize.py
import torch
from dataset import get_loaders
from models.protopnet import ProtoPNet
from utils.adaptive import build_feature_bank_with_meta, refine_prototypes
from utils.explain import (
    get_topk_patches_for_prototypes,
    visualize_prototype_neighbors,
    visualize_heatmap_for_image
)
from utils.explain import visualize_prototype_neighbors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset but ensure shuffle=False when creating loader for feature bank
train_loader, test_loader = get_loaders(batch_size=64)
# NOTE: get_loaders default created shuffle=True for train loader.
# For building feature bank, construct a loader with shuffle=False:
from torch.utils.data import DataLoader
from dataset import get_loaders as _get_loaders_raw
# If your dataset.get_loaders cannot set shuffle flag, create manually:
from torchvision import transforms
train_dataset = train_loader.dataset
# Create non-shuffled loader only for feature bank:
train_loader_noshuffle = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# load model
model = ProtoPNet(num_classes=10, num_prototypes=30).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# 1) Build feature bank + meta (this can be saved to disk for faster demos)
print("Building feature bank (this may take a little while)...")
feature_bank, meta = build_feature_bank_with_meta(model, train_loader_noshuffle, device)
print("Feature bank shape:", feature_bank.shape)

# 2) Optionally re-run clustering to show prototypes aligned with feature_bank (not necessary if already done)
# centers = refine_prototypes(feature_bank, num_prototypes=30)
# model.prototype_layer.prototypes.data.copy_(centers.to(device))

# 3) find top-k neighbors for each prototype
print("Finding prototype neighbors...")
neighbors = get_topk_patches_for_prototypes(model, feature_bank, meta, train_dataset, topk=5, device=device)

# 4) visualize prototype neighbor patches (shows a small set of prototypes)
visualize_prototype_neighbors(model, train_dataset, neighbors, H_feat=4, W_feat=4, topk=5, save_path="proto_neighbors.png")

# 5) show heatmaps for a few test images
print("Showing heatmaps for 3 test images...")
test_iter = iter(test_loader)
imgs, labels = next(test_iter)
for i in range(3):
    print("Label:", labels[i].item())
    visualize_heatmap_for_image(model, imgs[i], device, resize_to=128)

print("Demo finished. Saved proto_neighbors.png")
