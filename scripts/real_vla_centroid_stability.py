#!/usr/bin/env python3
"""Experiment 202: Centroid stability — how much does the centroid change
across different calibration subsets? Measures the robustness of threshold
selection by bootstrapping calibration sets.
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

def main():
    print("=" * 60)
    print("Experiment 202: Centroid Stability Analysis")
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
    n_total = 20  # total calibration pool
    n_test = 8

    def extract_all(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    # Extract all calibration embeddings
    print("\n--- Extracting calibration pool ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_total)]
    all_cal_embs = {l: [] for l in layers}
    for i, arr in enumerate(cal_arrs):
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            all_cal_embs[l].append(h[l])
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{n_total}", flush=True)

    # Extract test embeddings (ID and OOD)
    print("--- Extracting test ---", flush=True)
    test_arrs = [creators[(i+n_total)%3](i+n_total) for i in range(n_test)]
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
    ood_embs = {l: [] for l in layers}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h = extract_all(Image.fromarray(tfn(arr)))
            for l in layers:
                ood_embs[l].append(h[l])

    # Bootstrap centroids
    print("\n--- Bootstrap analysis ---", flush=True)
    n_bootstrap = 50
    cal_sizes = [3, 5, 10, 15]
    bootstrap_rng = np.random.RandomState(123)

    results = {}
    for n_cal in cal_sizes:
        cal_results = {l: {"aurocs": [], "centroid_dists": []} for l in layers}
        
        # Full centroid (from all 20)
        full_centroids = {l: np.array(all_cal_embs[l]).mean(axis=0) for l in layers}
        
        for b in range(n_bootstrap):
            indices = bootstrap_rng.choice(n_total, n_cal, replace=False)
            
            for l in layers:
                subset = [all_cal_embs[l][i] for i in indices]
                centroid = np.array(subset).mean(axis=0)
                
                # Centroid distance from full centroid
                cdist = cosine_distance(centroid, full_centroids[l])
                cal_results[l]["centroid_dists"].append(cdist)
                
                # AUROC with this centroid
                id_dists = [cosine_distance(e, centroid) for e in id_embs[l]]
                ood_dists = [cosine_distance(e, centroid) for e in ood_embs[l]]
                auroc = compute_auroc(id_dists, ood_dists)
                cal_results[l]["aurocs"].append(auroc)
        
        for l in layers:
            aurocs = cal_results[l]["aurocs"]
            cdists = cal_results[l]["centroid_dists"]
            print(f"  n={n_cal} L{l}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f} "
                  f"min={np.min(aurocs):.4f} centroid_drift={np.mean(cdists):.8f}±{np.std(cdists):.8f}", flush=True)
        
        results[f"n{n_cal}"] = {
            f"L{l}": {
                "mean_auroc": float(np.mean(cal_results[l]["aurocs"])),
                "std_auroc": float(np.std(cal_results[l]["aurocs"])),
                "min_auroc": float(np.min(cal_results[l]["aurocs"])),
                "max_auroc": float(np.max(cal_results[l]["aurocs"])),
                "mean_centroid_dist": float(np.mean(cal_results[l]["centroid_dists"])),
                "std_centroid_dist": float(np.std(cal_results[l]["centroid_dists"])),
                "aurocs": cal_results[l]["aurocs"],
            } for l in layers
        }

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "centroid_stability",
        "experiment_number": 202,
        "timestamp": ts,
        "n_total": n_total, "n_test": n_test,
        "n_bootstrap": n_bootstrap,
        "cal_sizes": cal_sizes,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"centroid_stability_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
