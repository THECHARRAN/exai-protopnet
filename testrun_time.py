# testrun_time.py
import time, torch
from dataset import get_loaders
from models.classifier_single import TimmClassifier
from torch.utils.data import Subset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_loaders(image_size=320, batch_size=8)
# use tiny subset
subset = Subset(train_loader.dataset, list(range(min(200, len(train_loader.dataset)))))
from torch.utils.data import DataLoader
loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=0)
model = TimmClassifier(name="tf_efficientnet_b4", pretrained=True, num_classes=4).to(device)
model.train()
t0 = time.time()
for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)
t1 = time.time()
print("One epoch (200 samples) forward time:", t1 - t0, "s")