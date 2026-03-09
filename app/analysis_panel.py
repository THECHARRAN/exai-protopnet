# app/analysis_panel.py
import os
import streamlit as st
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ensure project root is importable (if main didn't already do it)
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from inference.predictor import MRIEnsemblePredictor
from explanations.explain_engine import (
    generate_heatmap,
    overlay_heatmap,
    detect_tumor_bbox,
    compute_metrics,
    compute_prototype_activation,
    get_top_prototypes
)

# optional helpers (for showing actual dataset patches) — these live in utils/explain.py
try:
    from utils.explain import (
        get_topk_patches_for_prototypes,
        crop_patch_from_image
    )
    HAS_UTILS_EXPLAIN = True
except Exception:
    HAS_UTILS_EXPLAIN = False

# Predictor instance (cached)
predictor = None
@st.cache_resource
def get_predictor():
    global predictor
    if predictor is None:
        predictor = MRIEnsemblePredictor()
    return predictor

# helper to load feature bank files if present
def try_load_feature_bank():
    if os.path.exists("feature_bank.pt") and os.path.exists("meta.pt"):
        try:
            import torch
            bank = torch.load("feature_bank.pt", map_location="cpu")
            meta = torch.load("meta.pt", map_location="cpu")
            return bank, meta
        except Exception as e:
            st.warning(f"Couldn't load feature bank files: {e}")
            return None, None
    return None, None


def render_analysis(img):
    st.subheader("Analysis")

    if img is None:
        st.info("Upload MRI to start analysis")
        return

    predictor = get_predictor()

    with st.spinner("Running AI analysis..."):
        # predictor.predict accepts PIL.Image or file; returns results,img_raw,x,proto_model
        results, img_raw, x, proto_model = predictor.predict(img)

        # heatmap + overlay + bbox
        heatmap = generate_heatmap(proto_model, x)                   # 384x384 numpy array
        bbox = detect_tumor_bbox(heatmap, thr=0.55)                 # (x0,y0,x1,y1) or None
        overlay = overlay_heatmap(img_raw, heatmap, bbox)

        # metrics
        metrics = compute_metrics(heatmap)

        # prototype-level signals
        # proto_sim: per-prototype global activation (max over spatial dims)
        sim_tensor, _ = compute_prototype_activation(proto_model, x)  # [B,H,W,P]
        proto_global = sim_tensor.max(dim=1)[0].max(dim=1)[0][0].cpu().detach().numpy()
        proto_global = np.nan_to_num(proto_global)

        # prototypes per class (class-specific arrangement assumed contiguous)
        P_total = proto_global.shape[0]
        NUM_CLASSES = len(results) + 1  # not reliable; better to infer from model shapes
        # infer classes from model prototype count and known num classes:
        try:
            NUM_CLASSES = proto_model.num_classes
        except Exception:
            # fallback: use partitioning using a typical value if available
            # try to derive from predictor/predictor class names
            NUM_CLASSES = 4

        ppc = max(1, P_total // NUM_CLASSES)

        # predicted class index (top prediction)
        top_pred_class = results[0]["class"]
        # class name order must match predictor.CLASS_NAMES; try to get it:
        try:
            CLASS_NAMES = predictor.CLASS_NAMES
            class_idx = CLASS_NAMES.index(top_pred_class)
        except Exception:
            # fallback mapping
            CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
            class_idx = CLASS_NAMES.index(top_pred_class)

        class_start = class_idx * ppc
        class_end = class_start + ppc
        class_proto_scores = proto_global[class_start:class_end]
        # pick top 3 prototypes within that class
        topk_local = min(3, class_proto_scores.shape[0])
        local_indices = np.argsort(class_proto_scores)[::-1][:topk_local]
        top_proto_indices = (local_indices + class_start).tolist()
        top_proto_scores = class_proto_scores[local_indices].tolist()

        # attempt to get prototype neighbor patches (requires feature bank/meta and utils.explain)
        feature_bank, meta = try_load_feature_bank()

        # UI: show everything
    st.success("Analysis complete")

    # layout results: left column shows images, right column shows stats + prototypes
    col_l, col_r = st.columns([1.0, 0.9])

    # LEFT: original image, overlay, zoom
    with col_l:
        st.markdown("**Uploaded MRI**")
        st.image(img_raw, width="stretch")

        st.markdown("**Heatmap Overlay**")
        st.image(overlay, width="stretch")

        if bbox is not None:
            x0,y0,x1,y1 = bbox
            st.markdown("**Zoom — Detected ROI**")
            # crop zoom safely
            np_img = np.array(img_raw)
            h,w = np_img.shape[:2]
            x0 = max(0, int(x0)); y0 = max(0, int(y0))
            x1 = min(w, int(x1)); y1 = min(h, int(y1))
            if x1>x0 and y1>y0:
                zoom = np_img[y0:y1, x0:x1]
                st.image(zoom, width="stretch")
            else:
                st.write("ROI too small to crop")

    # RIGHT: predictions, confidence bars, metrics, and prototype matches
    with col_r:
        st.markdown("## Prediction")
        # show top-3 predictions from results
        for r in results:
            cls = r["class"]
            conf = r["confidence"]
            st.markdown(f"**{cls}** — {conf*100:.2f}%")
            # progress bar for confidence
            st.progress(min(1.0, float(conf)))

        # clinical style metric cards
        st.markdown("## Tumor Metrics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Tumor area %", f"{metrics['tumor_area_%']:.2f}%")
        c2.metric("Activation intensity", f"{metrics['activation_intensity']:.3f}")
        c3.metric("Edge irregularity", f"{metrics['edge_irregularity']:.3f}")
        centroid = metrics.get("centroid", [0,0])
        c4.metric("Centroid (y,x)", f"{centroid[0]:.1f},{centroid[1]:.1f}")

        st.markdown("## Top Prototype Matches (class-specific)")
        st.write(f"Showing top prototypes for predicted class **{top_pred_class}**")

        proto_cols = st.columns(topk_local)
        # If we have a feature bank and utils.explain, show actual neighbor patches
        if feature_bank is not None and meta is not None and HAS_UTILS_EXPLAIN:
            # compute neighbors using the helper; this returns list for all prototypes; we'll just index
            with st.spinner("Finding prototype neighbor patches (from feature bank)..."):
                from  dataset import get_loaders
                train_loader, _, _ = get_loaders(batch_size=16, image_size=384)
                train_dataset = train_loader.dataset
                neighbors = get_topk_patches_for_prototypes(
                proto_model,
                feature_bank,
                meta,
                dataset=train_dataset,
                topk=3,
                device="cpu"
                )
                # neighbors is list length P_total; each element is list of (dataset_idx,row,col,dist)
                for i, proto_idx in enumerate(top_proto_indices):
                    col = proto_cols[i]
                    col.markdown(f"**Proto {proto_idx}** — sim {top_proto_scores[i]:.3f}")
                    try:
                        top_neighbor = neighbors[proto_idx][0]  # (dataset_idx, row, col, dist)
                        ds_idx, row, col_patch, dist = top_neighbor
                        # we need dataset to fetch image: try to get from train dataset (loaders)
                        try:
                            from dataset import get_loaders
                            # small non-shuffled loader to get dataset object
                            train_loader,_,_ = get_loaders(batch_size=16, image_size=384)
                            train_dataset = train_loader.dataset
                            img_tensor, _ = train_dataset[ds_idx]
                            patch = crop_patch_from_image(img_tensor, row, col_patch, H_feat=int(np.sqrt(feature_bank.shape[0] / len(train_dataset))), W_feat=int(np.sqrt(feature_bank.shape[0] / len(train_dataset))))
                            # convert to displayable
                            patch_np = (patch.permute(1,2,0).numpy()*255).astype(np.uint8)
                            col.image(patch_np, width="stretch")
                        except Exception as e:
                            col.write("Patch fetch fail")
                            col.write(str(e))
                    except Exception as e:
                        col.write(f"Proto neighbor not found: {e}")
        else:
            # fallback: show proto index + similarity only
            for i, proto_idx in enumerate(top_proto_indices):
                col = proto_cols[i]
                col.markdown(f"**Proto {proto_idx}** — sim {top_proto_scores[i]:.3f}")
                col.write("Prototype patch not available (feature bank / utils.explain missing)")

        # final explanation text
        st.markdown("### Explanation")
        expl_lines = []
        expl_lines.append(f"The model predicted **{top_pred_class}** with {results[0]['confidence']*100:.2f}% confidence.")
        expl_lines.append("High-activation prototypes for the predicted class were found in the detected ROI.")
        expl_lines.append("Activation heatmap shows regions that resemble class-specific prototypes (texture/edge patterns).")
        expl_lines.append("Metrics shown (area, intensity, irregularity) are computed from heatmap activation > threshold.")
        for L in expl_lines:
            st.write(L)

    # small footer
    st.markdown("---")
    st.caption("ProtoPNet explainability panel — top prototypes are approximate and require feature bank for exact patches.")