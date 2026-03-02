# train_ensemble.py

import time
import torch
from tqdm import tqdm

from dataset import get_loaders
from models.classifier_single import TimmClassifier
from models.protopnet_classspecific import ProtoPNetCS
from losses import FocalLoss, CenterLoss
from utils.adaptive_classspecific import (
    build_feature_bank_with_meta_and_label,
    refine_prototypes_classwise,
)
from utils.explain import evaluate_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= CONFIG =================
BATCH = 16
IMG_SIZE = 384
EPOCHS = 15
NUM_CLASSES = 4
PROTOS_PER_CLASS = 20
# ==========================================


# -------------------------------------------------
# TRAIN FUNCTION
# -------------------------------------------------
def train_model(
    model,
    name,
    train_loader,
    val_loader,
    epochs,
    lr=2e-4,
    use_prototypes=False,
    prototype_refine_every=2,
):

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    focal = FocalLoss(gamma=2.0)

    feat_dim = model.feature_dim

    center_loss = CenterLoss(
        num_classes=NUM_CLASSES,
        feat_dim=feat_dim,
        device=device,
        lambda_c=0.001,
    )

    scaler = torch.amp.GradScaler("cuda")

    best_val = 0.0
    no_imp = 0

    for epoch in range(epochs):

        t0 = time.time()
        model.train()
        running = 0.0

        loop = tqdm(train_loader, desc=f"{name}-ep{epoch}")

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                if use_prototypes:
                    logits, sim, feats, _ = model(images)
                    feats_pool = feats.mean(dim=[2,3])
                else:
                    logits, feats_pool = model(images)

                loss_cls = focal(logits, labels)
                loss_center = center_loss(feats_pool, labels)

                loss = loss_cls + loss_center

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        epoch_time = time.time() - t0
        print(
            f"{name} epoch {epoch} "
            f"loss {running/len(train_loader):.4f} "
            f"time {epoch_time:.1f}s"
        )

        # ---------- VALIDATION ----------
        val_acc = evaluate_accuracy(model, val_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), f"{name}_best.pth")
            no_imp = 0
        else:
            no_imp += 1

        # ---------- PROTOTYPE REFINEMENT ----------
        if use_prototypes and epoch > 0 and epoch % prototype_refine_every == 0:

            print("Refining prototypes classwise...")

            from torch.utils.data import DataLoader

            loader_noshuffle = DataLoader(
                train_loader.dataset,
                batch_size=BATCH,
                shuffle=False,
                num_workers=0,
            )

            bank, meta, labels_meta = build_feature_bank_with_meta_and_label(
                model, loader_noshuffle, device
            )

            centers = refine_prototypes_classwise(
                bank,
                labels_meta,
                NUM_CLASSES,
                PROTOS_PER_CLASS,
            )

            model.prototype_layer.prototypes.data.copy_(
                centers.to(device)
            )

            print("Updated prototypes.")

        if no_imp >= 6:
            print("Early stopping triggered")
            break

    return model, best_val


# -------------------------------------------------
# MAIN ENTRYPOINT (WINDOWS SAFE)
# -------------------------------------------------
def main():

    train_loader, val_loader, test_loader = get_loaders(
        image_size=IMG_SIZE,
        batch_size=BATCH,
    )

    # ---------- EfficientNet ----------
    eff_model = TimmClassifier(
        name="tf_efficientnet_b4",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    eff_model, _ = train_model(
        eff_model,
        "effnet_b4",
        train_loader,
        val_loader,
        epochs=EPOCHS,
    )

    # ---------- ConvNeXt ----------
    conv_model = TimmClassifier(
        name="convnext_base",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    conv_model, _ = train_model(
        conv_model,
        "convnext_base",
        train_loader,
        val_loader,
        epochs=EPOCHS,
    )

    # ---------- ProtoPNet ----------
    proto_model = ProtoPNetCS(
        backbone_name="tf_efficientnet_b3_ns",
        pretrained=True,
        num_classes=NUM_CLASSES,
        prototypes_per_class=PROTOS_PER_CLASS,
        img_size=IMG_SIZE,
    )

    proto_model, _ = train_model(
        proto_model,
        "protopnet_cs",
        train_loader,
        val_loader,
        epochs=EPOCHS,
        use_prototypes=True,
    )

    # ---------- BUILD FEATURE BANK ----------
    from torch.utils.data import DataLoader

    loader_noshuffle = DataLoader(
        train_loader.dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
    )

    bank, meta, labels_meta = build_feature_bank_with_meta_and_label(
        proto_model,
        loader_noshuffle,
        device,
    )

    torch.save(bank, "feature_bank.pt")
    torch.save(meta, "meta.pt")
    torch.save(labels_meta, "labels_meta.pt")

    print("Feature bank saved.")


# -------------------------------------------------
# WINDOWS SAFE START
# -------------------------------------------------
if __name__ == "__main__":
    main()