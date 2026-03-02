# dataset.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_loaders(data_root="./dataset", image_size=224, batch_size=32, val_frac=0.15, test_frac=0.15, seed=42):
    # Medical-safe augmentations for train
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=15),            # small rotations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(200),                       # optional to remove margins
        transforms.Resize((image_size, image_size)),
        transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(200),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    full_train = datasets.ImageFolder(root=f"{data_root}/train", transform=train_transform)

    # deterministic split into train/val/test if original dataset not already split
    total = len(full_train)
    val_len = int(total * val_frac)
    test_len = int(total * test_frac)
    train_len = total - val_len - test_len

    torch.manual_seed(seed)
    train_ds, val_ds, extra_ds = random_split(full_train, [train_len, val_len, test_len])
    # If you have separate test folder, you can instead use ImageFolder(data_root/test, transform=val_transform)

    # override transforms for val/test (random_split keeps transform from original dataset object)
    val_ds.dataset.transform = val_transform
    extra_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(extra_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Classes:", full_train.classes)
    return train_loader, val_loader, test_loader