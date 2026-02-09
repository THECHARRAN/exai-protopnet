import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import get_loaders
from models.protopnet import ProtoPNet
from utils.adaptive import (
    build_feature_bank_with_meta,
    refine_prototypes,
    update_prototypes
)
from utils.explain import evaluate_accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, test_loader = get_loaders(batch_size=64)
train_loader_noshuffle = DataLoader(
    train_loader.dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
model = ProtoPNet(num_classes=10, num_prototypes=30).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)


def train_one_epoch(epoch):

    model.train()
    running_loss = 0

    loop = tqdm(train_loader, leave=False)

    for images, labels in loop:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

    return running_loss / len(train_loader)


def main():

    epochs = 15           
    best_acc = 0

    for epoch in range(epochs):

        print(f"\n========== Epoch {epoch} ==========")

        loss = train_one_epoch(epoch)

        print(f"Train Loss: {loss:.4f}")
        acc = evaluate_accuracy(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model.pth")
            print(" Best model saved")
        if epoch % 5 == 0 and epoch > 0:

            print("\n Refining prototypes with clustering...")

            bank, _ = build_feature_bank_with_meta(
                model,
                train_loader_noshuffle,
                device
            )
            centers = refine_prototypes(
                bank,
                num_prototypes=30
            )
            update_prototypes(model, centers, device)
            print(" Prototypes updated")

    print("\nTraining complete.")
    print("Best Accuracy:", best_acc)
if __name__ == "__main__":
    main()
