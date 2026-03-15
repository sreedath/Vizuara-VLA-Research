#!/usr/bin/env python3
"""Experiment 180: OOD type clustering — do different corruptions form
distinct clusters in embedding space?

Computes inter- and intra-cluster distances to determine if the detector
could identify the TYPE of OOD corruption, not just detect it.
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
def apply_snow(a):
    snow = np.random.randint(180, 256, a.shape, dtype=np.uint8)
    mask = np.random.random(a.shape[:2]) > 0.85
    out = a.copy()
    out[mask] = snow[mask]
    return out

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 180: OOD Type Clustering Analysis")
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
    n_per = 8

    # Base images
    base_arrs = [creators[i%3](i) for i in range(n_per)]

    ood_types = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "snow": apply_snow,
    }

    # Extract ID embeddings
    print("\n--- Extracting embeddings ---", flush=True)
    id_embs = {l: [] for l in layers}
    for arr in base_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_embs[l].append(h[l])

    # Extract OOD embeddings per type
    ood_embs = {cat: {l: [] for l in layers} for cat in ood_types}
    for cat, tfn in ood_types.items():
        for arr in base_arrs:
            h = extract_hidden(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for l in layers:
                ood_embs[cat][l].append(h[l])
        print(f"  {cat}: {n_per} embeddings", flush=True)

    # Compute pairwise distances between cluster centroids
    results = {}
    for l in layers:
        layer_results = {}

        # Centroids
        id_centroid = np.array(id_embs[l]).mean(axis=0)
        ood_centroids = {cat: np.array(ood_embs[cat][l]).mean(axis=0) for cat in ood_types}

        # Intra-cluster compactness (mean pairwise cosine distance within each cluster)
        compactness = {"ID": float(np.mean([cosine_distance(id_embs[l][i], id_embs[l][j])
                                           for i in range(n_per) for j in range(i+1, n_per)]))}
        for cat in ood_types:
            dists = [cosine_distance(ood_embs[cat][l][i], ood_embs[cat][l][j])
                    for i in range(n_per) for j in range(i+1, n_per)]
            compactness[cat] = float(np.mean(dists))

        # Inter-cluster distances (centroid-to-centroid)
        all_cats = ["ID"] + list(ood_types.keys())
        all_centroids = [id_centroid] + [ood_centroids[c] for c in ood_types]
        inter_dists = {}
        for i, c1 in enumerate(all_cats):
            for j, c2 in enumerate(all_cats):
                if i < j:
                    d = cosine_distance(all_centroids[i], all_centroids[j])
                    inter_dists[f"{c1}_vs_{c2}"] = d

        # Silhouette-like score for each OOD type
        # (inter-cluster dist to ID centroid) / (intra-cluster compactness)
        silhouettes = {}
        for cat in ood_types:
            inter = cosine_distance(id_centroid, ood_centroids[cat])
            intra = compactness[cat]
            silhouettes[cat] = float(inter / (intra + 1e-10))

        # Can we classify OOD type? Nearest centroid classification accuracy
        # For each OOD sample, find nearest OOD centroid
        n_correct = 0
        n_total = 0
        for true_cat in ood_types:
            for e in ood_embs[true_cat][l]:
                min_dist = float('inf')
                pred_cat = None
                for cand_cat in ood_types:
                    d = cosine_distance(e, ood_centroids[cand_cat])
                    if d < min_dist:
                        min_dist = d
                        pred_cat = cand_cat
                if pred_cat == true_cat:
                    n_correct += 1
                n_total += 1
        classification_acc = n_correct / n_total

        layer_results["compactness"] = compactness
        layer_results["inter_cluster"] = inter_dists
        layer_results["silhouettes"] = silhouettes
        layer_results["ood_type_classification_acc"] = classification_acc

        print(f"\n  L{l} compactness:", flush=True)
        for c, v in compactness.items():
            print(f"    {c}: {v:.6f}", flush=True)
        print(f"  L{l} OOD type classification acc: {classification_acc:.4f}", flush=True)
        print(f"  L{l} silhouettes:", flush=True)
        for c, v in silhouettes.items():
            print(f"    {c}: {v:.2f}", flush=True)

        results[f"L{l}"] = layer_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "ood_clustering",
        "experiment_number": 180,
        "timestamp": ts,
        "n_per_type": n_per,
        "ood_types": list(ood_types.keys()),
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"ood_clustering_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
