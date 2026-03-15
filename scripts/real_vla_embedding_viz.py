#!/usr/bin/env python3
"""Experiment 157: Embedding space visualization.

Projects ID and OOD embeddings to 2D via PCA to visualize cluster structure
at L3 and L32. Also computes inter-cluster distances and angular separation.
"""

import json, os, sys, datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent
EXPERIMENTS_DIR = REPO_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = str(EXPERIMENTS_DIR)

SIZE = (256, 256)
rng = np.random.RandomState(42)

def create_highway(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]; img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_urban(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]; img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]; img[SIZE[0]//2:] = [60, 60, 60]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_rural(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [100, 180, 255]; img[SIZE[0]//3:SIZE[0]*2//3] = [34, 139, 34]; img[SIZE[0]*2//3:] = [90, 90, 90]
    return np.clip(img.astype(np.int16) + rng.randint(-8, 9, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def apply_fog(a, alpha):
    return np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8)
def apply_night(a): return np.clip(a*0.15, 0, 255).astype(np.uint8)
def apply_blur(a, r=8): return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=r)))
def apply_noise(a, s=50): return np.clip(a.astype(np.float32)+np.random.normal(0,s,a.shape), 0, 255).astype(np.uint8)
def apply_occlusion(a):
    o=a.copy(); h,w=o.shape[:2]; o[h//4:3*h//4, w//4:3*w//4]=128; return o

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 157: Embedding Space Visualization")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [3, 32]

    creators = [create_highway, create_urban, create_rural]
    n_id = 12

    # ID embeddings
    print("\n--- Extracting ID embeddings ---", flush=True)
    id_embeddings = {l: [] for l in layers}
    id_labels = []
    scene_names = ["highway", "urban", "rural"]
    for i in range(n_id):
        arr = creators[i % 3](i)
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_embeddings[l].append(h[l])
        id_labels.append(scene_names[i % 3])
        print(f"  ID image {i} ({scene_names[i%3]})", flush=True)

    # OOD embeddings
    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
    }
    n_ood_per = 4

    ood_embeddings = {l: [] for l in layers}
    ood_labels = []
    print("\n--- Extracting OOD embeddings ---", flush=True)
    for cat, tfn in ood_transforms.items():
        for j in range(n_ood_per):
            arr = tfn(creators[j % 3](j))
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_embeddings[l].append(h[l])
            ood_labels.append(cat)
            print(f"  OOD {cat} image {j}", flush=True)

    # PCA projection for each layer
    results = {}
    for l in layers:
        id_embs = np.array(id_embeddings[l])
        ood_embs = np.array(ood_embeddings[l])
        all_embs = np.vstack([id_embs, ood_embs])

        # Center on ID mean
        center = id_embs.mean(axis=0)
        centered = all_embs - center

        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_2d = centered @ Vt[:2].T  # project onto first 2 PCs
        var_explained = (S[:2]**2) / (S**2).sum()

        n_id_total = len(id_embs)
        n_ood_total = len(ood_embs)

        id_2d = pca_2d[:n_id_total]
        ood_2d = pca_2d[n_id_total:]

        # Inter-cluster distances in 2D
        id_centroid_2d = id_2d.mean(axis=0)
        id_radius_2d = float(np.max(np.linalg.norm(id_2d - id_centroid_2d, axis=1)))

        # Per-category OOD centroids in 2D
        ood_cat_centroids = {}
        idx = 0
        for cat in ood_transforms:
            cat_pts = ood_2d[idx:idx+n_ood_per]
            ood_cat_centroids[cat] = {
                "centroid": cat_pts.mean(axis=0).tolist(),
                "distance_from_id": float(np.linalg.norm(cat_pts.mean(axis=0) - id_centroid_2d)),
                "spread": float(np.std(np.linalg.norm(cat_pts - cat_pts.mean(axis=0), axis=1))),
            }
            idx += n_ood_per

        results[f"L{l}"] = {
            "var_explained_pc1": float(var_explained[0]),
            "var_explained_pc2": float(var_explained[1]),
            "var_explained_total": float(var_explained.sum()),
            "id_centroid_2d": id_centroid_2d.tolist(),
            "id_radius_2d": id_radius_2d,
            "id_points_2d": id_2d.tolist(),
            "id_labels": id_labels,
            "ood_points_2d": ood_2d.tolist(),
            "ood_labels": ood_labels,
            "ood_category_centroids": ood_cat_centroids,
            "singular_values_top10": S[:10].tolist(),
        }
        print(f"\n  L{l}: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, "
              f"total={var_explained.sum():.3f}", flush=True)
        print(f"  ID radius (2D): {id_radius_2d:.4f}", flush=True)
        for cat, info in ood_cat_centroids.items():
            print(f"  {cat}: dist_from_id={info['distance_from_id']:.4f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "embedding_visualization",
        "experiment_number": 157,
        "timestamp": ts,
        "n_id": n_id, "n_ood_per_cat": n_ood_per,
        "ood_categories": list(ood_transforms.keys()),
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"embedding_viz_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
