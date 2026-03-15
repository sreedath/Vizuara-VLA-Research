#!/usr/bin/env python3
"""Experiment 194: d-prime sensitivity analysis — compute d' (discriminability index)
across layers 1-7 and compare with multi-layer combination strategies.

Establishes the statistical power of each early layer for OOD detection.
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

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def compute_dprime(id_scores, ood_scores):
    """Compute d' (discriminability index)."""
    id_mean = np.mean(id_scores)
    ood_mean = np.mean(ood_scores)
    id_std = np.std(id_scores)
    ood_std = np.std(ood_scores)
    pooled_std = np.sqrt((id_std**2 + ood_std**2) / 2)
    if pooled_std < 1e-10:
        return float('inf') if ood_mean > id_mean else 0.0
    return float((ood_mean - id_mean) / pooled_std)

def main():
    print("=" * 60)
    print("Experiment 194: d-prime Analysis Across Early Layers")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = list(range(8))  # 0-7

    creators = [create_highway, create_urban, create_rural]
    n_cal = 10
    n_test = 8

    # Extract all embeddings
    print("\n--- Extracting embeddings ---", flush=True)
    def extract_all(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            cal_embs[l].append(h[l])

    centroids = {l: np.array(cal_embs[l]).mean(axis=0) for l in layers}

    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_embs = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            id_embs[l].append(h[l])

    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_embs_all = {l: [] for l in layers}
    ood_embs_per = {cat: {l: [] for l in layers} for cat in ood_transforms}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h = extract_all(Image.fromarray(tfn(arr)))
            for l in layers:
                ood_embs_all[l].append(h[l])
                ood_embs_per[cat][l].append(h[l])

    # Compute per-layer metrics
    print("\n--- Per-layer analysis ---", flush=True)
    results = {}
    for l in layers:
        id_dists = [cosine_distance(e, centroids[l]) for e in id_embs[l]]
        ood_dists = [cosine_distance(e, centroids[l]) for e in ood_embs_all[l]]

        auroc = compute_auroc(id_dists, ood_dists)
        dprime = compute_dprime(id_dists, ood_dists)
        sep = float(np.mean(ood_dists) / (np.mean(id_dists) + 1e-10))

        # Per-category d'
        per_cat_dprime = {}
        for cat in ood_transforms:
            cat_dists = [cosine_distance(e, centroids[l]) for e in ood_embs_per[cat][l]]
            per_cat_dprime[cat] = compute_dprime(id_dists, cat_dists)

        results[str(l)] = {
            "layer": l,
            "auroc": auroc,
            "dprime": dprime,
            "separation_ratio": sep,
            "id_mean": float(np.mean(id_dists)),
            "id_std": float(np.std(id_dists)),
            "ood_mean": float(np.mean(ood_dists)),
            "ood_std": float(np.std(ood_dists)),
            "per_cat_dprime": per_cat_dprime,
        }
        print(f"  L{l}: AUROC={auroc:.4f} d'={dprime:.2f} sep={sep:.2f}", flush=True)
        for cat, dp in per_cat_dprime.items():
            print(f"    {cat}: d'={dp:.2f}", flush=True)

    # Multi-layer combination: OR-gate
    print("\n--- Multi-layer OR-gate ---", flush=True)
    for combo_layers in [[1, 3], [1, 7], [1, 3, 7]]:
        combo_name = "+".join(f"L{l}" for l in combo_layers)
        # OR-gate: max distance across layers
        id_max = [max(cosine_distance(id_embs[l][i], centroids[l]) for l in combo_layers) for i in range(n_test)]
        ood_max = [max(cosine_distance(ood_embs_all[l][i], centroids[l]) for l in combo_layers) for i in range(len(ood_embs_all[layers[0]]))]
        auroc_combo = compute_auroc(id_max, ood_max)
        dprime_combo = compute_dprime(id_max, ood_max)
        results[f"combo_{combo_name}"] = {"auroc": auroc_combo, "dprime": dprime_combo}
        print(f"  {combo_name}: AUROC={auroc_combo:.4f} d'={dprime_combo:.2f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "dprime_analysis",
        "experiment_number": 194,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"dprime_analysis_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
