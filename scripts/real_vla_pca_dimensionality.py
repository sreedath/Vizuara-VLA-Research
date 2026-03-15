#!/usr/bin/env python3
"""Experiment 147: PCA dimensionality analysis of ID vs OOD embeddings.

Analyzes the intrinsic dimensionality of embedding space at L3 and L32.
Questions: How many PCA components capture the OOD signal? Is the ID manifold
lower-dimensional than OOD? What does the variance spectrum look like?
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
def apply_snow(a):
    o=a.astype(np.float32)*0.7+76.5; o[np.random.random(a.shape[:2])>0.97]=255
    return np.clip(o,0,255).astype(np.uint8)

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def main():
    print("=" * 60)
    print("Experiment 147: PCA Dimensionality Analysis")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    creators = [create_highway, create_urban, create_rural]
    layers = [3, 32]

    # Generate images: 20 ID + various OOD
    n_id = 20
    id_arrs = [creators[i%3](i) for i in range(n_id)]

    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
        "snow": apply_snow,
    }
    n_ood_per = 5

    # Collect all embeddings
    print(f"\n--- Collecting ID embeddings (n={n_id}) ---", flush=True)
    id_embs = {l: [] for l in layers}
    for i, arr in enumerate(id_arrs):
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_embs[l].append(h[l])
        if (i+1) % 5 == 0:
            print(f"  ID {i+1}/{n_id}", flush=True)

    print(f"\n--- Collecting OOD embeddings ---", flush=True)
    ood_embs = {l: {c: [] for c in ood_transforms} for l in layers}
    for cat, tfn in ood_transforms.items():
        for j in range(n_ood_per):
            arr = tfn(id_arrs[j % n_id])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_embs[l][cat].append(h[l])
        print(f"  {cat}: done", flush=True)

    # PCA Analysis
    print("\n--- PCA Analysis ---", flush=True)
    results = {}
    for l in layers:
        ln = f"L{l}"
        id_mat = np.array(id_embs[l])  # (n_id, 4096)
        centroid = id_mat.mean(axis=0)

        # SVD of centered ID data
        centered = id_mat - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        total_var = np.sum(S**2)
        explained_ratio = np.cumsum(S**2) / total_var

        # How many components to explain 90%, 95%, 99%?
        dims_90 = int(np.searchsorted(explained_ratio, 0.90) + 1)
        dims_95 = int(np.searchsorted(explained_ratio, 0.95) + 1)
        dims_99 = int(np.searchsorted(explained_ratio, 0.99) + 1)

        # Project ID and OOD onto top-k PCA components
        k_values = [1, 2, 3, 5, 10, min(n_id-1, 15)]
        projection_results = {}
        for k in k_values:
            Vk = Vt[:k]  # (k, 4096)
            # Project ID
            id_proj = (id_mat - centroid) @ Vk.T  # (n_id, k)
            id_recon_err = np.mean(np.sum((centered - (centered @ Vk.T @ Vk))**2, axis=1))

            # Project each OOD category
            ood_proj_results = {}
            for cat in ood_transforms:
                ood_mat = np.array(ood_embs[l][cat])
                ood_centered = ood_mat - centroid
                ood_proj = ood_centered @ Vk.T
                ood_recon_err = np.mean(np.sum((ood_centered - (ood_centered @ Vk.T @ Vk))**2, axis=1))

                # Separation in PCA space
                id_norms = np.linalg.norm(id_proj, axis=1)
                ood_norms = np.linalg.norm(ood_proj, axis=1)
                pooled = np.sqrt((np.var(id_norms) + np.var(ood_norms)) / 2 + 1e-10)
                d_prime = float((np.mean(ood_norms) - np.mean(id_norms)) / pooled)

                ood_proj_results[cat] = {
                    "recon_error": float(ood_recon_err),
                    "d_prime_pca_norm": d_prime,
                    "mean_proj_norm": float(np.mean(ood_norms)),
                }

            projection_results[f"k={k}"] = {
                "k": k,
                "explained_var": float(explained_ratio[k-1]) if k <= len(explained_ratio) else 1.0,
                "id_recon_error": float(id_recon_err),
                "id_mean_proj_norm": float(np.mean(np.linalg.norm(id_proj, axis=1))),
                "per_category": ood_proj_results,
            }

        # Singular value spectrum
        sv_spectrum = S[:min(20, len(S))].tolist()

        results[ln] = {
            "n_id": n_id,
            "embedding_dim": int(id_mat.shape[1]),
            "dims_90pct": dims_90,
            "dims_95pct": dims_95,
            "dims_99pct": dims_99,
            "singular_values_top20": [float(s) for s in sv_spectrum],
            "explained_ratio_cumulative": [float(e) for e in explained_ratio[:min(20, len(explained_ratio))]],
            "projection_results": projection_results,
        }

        print(f"\n{ln}: dims for 90%={dims_90}, 95%={dims_95}, 99%={dims_99}")
        print(f"  Top SV: {sv_spectrum[:5]}")
        for k in k_values:
            pr = projection_results[f"k={k}"]
            print(f"  k={k}: explained={pr['explained_var']:.3f}, id_recon={pr['id_recon_error']:.4f}")

    # Save
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {"experiment": "pca_dimensionality", "experiment_number": 147, "timestamp": ts,
              "n_id": n_id, "n_ood_per_cat": n_ood_per,
              "ood_categories": list(ood_transforms.keys()),
              "layers": layers, "results": results}
    path = os.path.join(RESULTS_DIR, f"pca_dimensionality_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
