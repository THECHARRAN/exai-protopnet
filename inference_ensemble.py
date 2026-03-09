import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.classifier_single import TimmClassifier
from models.protopnet_classspecific import ProtoPNetCS
from utils.explain import visualize_heatmap_for_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 384

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(200),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

def load_models():

    eff = TimmClassifier(
        name="tf_efficientnet_b4",
        pretrained=False,
        num_classes=4
    ).to(device)

    eff.load_state_dict(
        torch.load("effnet_b4_best.pth", map_location=device)
    )

    conv = TimmClassifier(
        name="convnext_base",
        pretrained=False,
        num_classes=4
    ).to(device)

    conv.load_state_dict(
        torch.load("convnext_base_best.pth", map_location=device)
    )

    proto = ProtoPNetCS(
        backbone_name="tf_efficientnet_b3_ns",
        pretrained=False,
        num_classes=4,
        prototypes_per_class=20,
        img_size=IMG_SIZE
    ).to(device)

    proto.load_state_dict(
        torch.load("protopnet_cs_best.pth", map_location=device)
    )

    eff.eval()
    conv.eval()
    proto.eval()

    return eff, conv, proto


# ------------------------------------------------
# ENSEMBLE PREDICTION
# ------------------------------------------------

def predict(image_path, eff, conv, proto):

    img = Image.open(image_path).convert("L")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():

        logits1, _ = eff(x)
        logits2, _ = conv(x)
        logits3, _, _, _ = proto(x)

        probs1 = torch.softmax(logits1, dim=1)
        probs2 = torch.softmax(logits2, dim=1)
        probs3 = torch.softmax(logits3, dim=1)

        probs = (probs1 + probs2 + probs3) / 3

        probs = probs.cpu().numpy()[0]

    return probs, img, x


# ------------------------------------------------
# MAIN DEMO
# ------------------------------------------------

def main():

    print("\nLoading models...")
    eff, conv, proto = load_models()

    img_path = input("\nEnter MRI image path: ")

    probs, img, x = predict(img_path, eff, conv, proto)

    top3 = np.argsort(probs)[::-1][:3]

    print("\nTop Predictions:\n")

    for i in top3:
        print(f"{CLASS_NAMES[i]} : {probs[i]*100:.2f}%")

    print("\nGenerating heatmap explanation...\n")

    visualize_heatmap_for_image(proto, x[0].cpu(), device)


if __name__ == "__main__":
    main()