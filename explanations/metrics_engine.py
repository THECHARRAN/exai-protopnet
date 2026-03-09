import numpy as np

def compute_metrics(heatmap):

    mask = heatmap > 0.6

    tumor_area = mask.sum() / heatmap.size

    centroid = np.mean(np.argwhere(mask),axis=0)

    intensity = heatmap[mask].mean()

    edge_irregularity = np.std(heatmap[mask])

    return {
        "tumor_area_%": float(tumor_area*100),
        "centroid": centroid.tolist(),
        "activation_intensity": float(intensity),
        "edge_irregularity": float(edge_irregularity)
    }