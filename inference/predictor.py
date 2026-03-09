import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.classifier_single import TimmClassifier
from models.protopnet_classspecific import ProtoPNetCS

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
    transforms.Normalize([0.5]*3,[0.5]*3)
])


class MRIEnsemblePredictor:

    def __init__(self):

        self.eff = TimmClassifier(
            name="tf_efficientnet_b4",
            pretrained=False,
            num_classes=4
        ).to(device)

        self.eff.load_state_dict(
            torch.load("effnet_b4_best.pth", map_location=device)
        )

        self.conv = TimmClassifier(
            name="convnext_base",
            pretrained=False,
            num_classes=4
        ).to(device)

        self.conv.load_state_dict(
            torch.load("convnext_base_best.pth", map_location=device)
        )

        self.proto = ProtoPNetCS(
            backbone_name="tf_efficientnet_b3_ns",
            pretrained=False,
            num_classes=4,
            prototypes_per_class=20,   # ← match checkpoint
            img_size=IMG_SIZE
        ).to(device)

        self.proto.load_state_dict(
            torch.load("protopnet_cs_best.pth", map_location=device)
        )

        self.eff.eval()
        self.conv.eval()
        self.proto.eval()

    def preprocess(self, image):
    # If uploaded from Streamlit it may already be a PIL image
        if isinstance(image, Image.Image):
            img = image.convert("L")
        else:
            img = Image.open(image).convert("L")

        tensor = transform(img).unsqueeze(0).to(device)

        return img, tensor

    def predict(self, image):

        img, x = self.preprocess(image)

        with torch.no_grad():

            logits1,_ = self.eff(x)
            logits2,_ = self.conv(x)
            logits3,_,_,_ = self.proto(x)

            p1 = torch.softmax(logits1,dim=1)
            p2 = torch.softmax(logits2,dim=1)
            p3 = torch.softmax(logits3,dim=1)

            probs = (p1+p2+p3)/3

        probs = probs.cpu().numpy()[0]

        top3 = np.argsort(probs)[::-1][:3]

        result = []

        for i in top3:

            result.append({
                "class":CLASS_NAMES[i],
                "confidence":float(probs[i])
            })

        return result, img, x, self.proto