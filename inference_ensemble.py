# inference_ensemble.py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.explain import visualize_heatmap_for_image, get_topk_patches_for_prototypes, visualize_prototype_neighbors
from utils.adaptive_classspecific import build_feature_bank_with_meta_and_label
from models.protopnet_classspecific import ProtoPNetCS
from models.classifier_single import TimmClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 320
pre = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(200),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

tta_transforms = [
    pre,
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)]),
]

def load_models():
    eff = TimmClassifier(name="tf_efficientnet_b4", pretrained=False, num_classes=4).to(device)
    eff.load_state_dict(torch.load("effnet_b4_best.pth", map_location=device))
    swin = TimmClassifier(name="swin_base_patch4_window7_224", pretrained=False, num_classes=4).to(device)
    swin.load_state_dict(torch.load("swin_base_best.pth", map_location=device))
    proto = ProtoPNetCS(backbone_name="tf_efficientnet_b3_ns", pretrained=False, num_classes=4, prototypes_per_class=15, img_size=IMG_SIZE).to(device)
    proto.load_state_dict(torch.load("protopnet_cs_best.pth", map_location=device))
    return [eff, swin, proto]

def tta_predict(models, img_pil, tta=tta_transforms):
    preds = []
    sims = None
    for tfm in tta:
        x = tfm(img_pil).unsqueeze(0).to(device)
        ensemble_logits = []
        proto_sims = None
        for m in models:
            m.eval()
            with torch.no_grad():
                out = m(x)
                if isinstance(out, tuple):
                    logits = out[0]
                    if len(out) > 1:
                        proto_sims = out[1] if proto_sims is None else proto_sims + out[1]
                else:
                    logits = out
            ensemble_logits.append(torch.softmax(logits, dim=1).cpu().numpy())
        ensemble_logits = np.mean(np.concatenate(ensemble_logits, axis=0), axis=0)
        preds.append(ensemble_logits)
        if proto_sims is not None:
            sims = proto_sims.cpu().numpy()
    final = np.mean(np.stack(preds, axis=0), axis=0)
    return final, sims

def infer_image(image_path):
    img = Image.open(image_path).convert("L")
    models = load_models()
    probs, sims = tta_predict(models, img)
    top3 = probs.argsort()[::-1][:3]
    print("Top3 classes:", top3, "scores:", probs[top3])
    # heatmap from proto model
    proto_model = models[-1]
    x = pre(img).unsqueeze(0).to(device)
    visualize_heatmap_for_image(proto_model, x[0].cpu(), device, resize_to=IMG_SIZE)
    # if you have saved feature_bank and meta you can show prototypes neighbors
    return top3, probs[top3]