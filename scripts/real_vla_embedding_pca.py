#!/usr/bin/env python3
"""Experiment 200: Embedding PCA visualization — project ID and OOD embeddings
into 2D/3D via PCA to visualize cluster structure.
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

def main():
    print("=" * 60)
    print("Experiment 200: Embedding PCA Visualization")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [1, 3, 32]

    creators = [create_highway, create_urban, create_rural]
    n_id = 12

    def extract_all(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    # ID embeddings
    print("\n--- Extracting ID embeddings ---", flush=True)
    id_arrs = [creators[i%3](i) for i in range(n_id)]
    id_embs = {l: [] for l in layers}
    id_labels = []
    for i, arr in enumerate(id_arrs):
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            id_embs[l].append(h[l])
        id_labels.append(["highway", "urban", "rural"][i%3])

    # OOD embeddings
    print("--- Extracting OOD embeddings ---", flush=True)
    ood_transforms = {
        "fog": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_embs = {l: [] for l in layers}
    ood_labels = []
    for cat, tfn in ood_transforms.items():
        for i, arr in enumerate(id_arrs[:6]):  # use first 6 for OOD
            h = extract_all(Image.fromarray(tfn(arr)))
            for l in layers:
                ood_embs[l].append(h[l])
            ood_labels.append(cat)

    # PCA
    print("\n--- Computing PCA ---", flush=True)
    results = {}
    for l in layers:
        all_embs = np.array(id_embs[l] + ood_embs[l])
        mean = all_embs.mean(axis=0)
        centered = all_embs - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Project to 3D
        pca_3d = centered @ Vt[:3].T
        
        # Explained variance
        explained = S**2 / np.sum(S**2)
        
        n_id_total = len(id_embs[l])
        n_ood_total = len(ood_embs[l])
        
        results[f"L{l}"] = {
            "explained_variance": explained[:10].tolist(),
            "cumulative_variance": np.cumsum(explained[:10]).tolist(),
            "id_pca": pca_3d[:n_id_total].tolist(),
            "ood_pca": pca_3d[n_id_total:].tolist(),
            "id_labels": id_labels,
            "ood_labels": ood_labels,
            "singular_values": S[:10].tolist(),
        }
        
        print(f"  L{l}: top-3 explained={explained[:3].sum():.4f} "
              f"top-1={explained[0]:.4f} top-2={explained[1]:.4f} top-3={explained[2]:.4f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "embedding_pca",
        "experiment_number": 200,
        "timestamp": ts,
        "n_id": n_id, "n_ood_per_cat": 6,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"embedding_pca_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
