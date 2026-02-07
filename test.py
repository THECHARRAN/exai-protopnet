import torch
from dataset import get_loaders
from models.protopnet import ProtoPNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _ = get_loaders()

model = ProtoPNet().to(device)

images, _ = next(iter(train_loader))
images = images.to(device)

logits, similarity = model(images)

print("logits:", logits.shape)
print("similarity:", similarity.shape)
