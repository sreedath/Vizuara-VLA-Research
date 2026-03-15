#!/usr/bin/env python3
"""Experiment 170: Outlier-robust calibration — comparing centroid estimators.

Tests whether robust centroid estimation (geometric median, trimmed mean,
medoid) improves OOD detection when calibration data contains outliers.
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

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def geometric_median(points, tol=1e-7, max_iter=200):
    """Weiszfeld's algorithm for geometric median."""
    y = np.mean(points, axis=0)
    for _ in range(max_iter):
        dists = np.linalg.norm(points - y, axis=1)
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        y_new = np.average(points, axis=0, weights=weights)
        if np.linalg.norm(y_new - y) < tol:
            break
        y = y_new
    return y

def trimmed_mean(points, trim_frac=0.2):
    """Trimmed mean: remove trim_frac of points farthest from initial mean."""
    mean = np.mean(points, axis=0)
    dists = np.linalg.norm(points - mean, axis=1)
    n_keep = max(2, int(len(points) * (1 - trim_frac)))
    keep_idx = np.argsort(dists)[:n_keep]
    return np.mean(points[keep_idx], axis=0)

def medoid(points):
    """Point in dataset closest to all others."""
    dists = np.array([[np.linalg.norm(a - b) for b in points] for a in points])
    total_dists = dists.sum(axis=1)
    return points[np.argmin(total_dists)]

def main():
    print("=" * 60)
    print("Experiment 170: Outlier-Robust Calibration")
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
    n_clean_cal = 8
    n_test = 8

    # Generate clean calibration embeddings
    print("\n--- Extracting clean calibration embeddings ---", flush=True)
    clean_cal_arrs = [creators[i%3](i) for i in range(n_clean_cal)]
    clean_cal_embs = {l: [] for l in layers}
    for arr in clean_cal_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            clean_cal_embs[l].append(h[l])

    # Generate "contaminated" calibration: add 2 OOD samples to mimic mislabeled data
    print("--- Extracting contaminated calibration embeddings ---", flush=True)
    contam_arrs = [apply_fog(clean_cal_arrs[0], 0.5), apply_night(clean_cal_arrs[1])]
    contam_embs = {l: [] for l in layers}
    for arr in contam_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            contam_embs[l].append(h[l])

    # Test embeddings
    print("--- Extracting test embeddings ---", flush=True)
    test_arrs = [creators[(i+n_clean_cal)%3](i+n_clean_cal) for i in range(n_test)]
    id_embs = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
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
        for j in range(n_test):
            arr = tfn(test_arrs[j % n_test])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_embs[l].append(h[l])
    print("  All embeddings extracted.", flush=True)

    # Contamination levels: 0, 1, 2 outliers added
    contamination_levels = {
        "clean": {l: np.array(clean_cal_embs[l]) for l in layers},
        "1_outlier": {l: np.array(clean_cal_embs[l] + [contam_embs[l][0]]) for l in layers},
        "2_outliers": {l: np.array(clean_cal_embs[l] + contam_embs[l]) for l in layers},
    }

    estimators = {
        "mean": lambda pts: np.mean(pts, axis=0),
        "geometric_median": geometric_median,
        "trimmed_mean_20": lambda pts: trimmed_mean(pts, 0.2),
        "trimmed_mean_30": lambda pts: trimmed_mean(pts, 0.3),
        "medoid": medoid,
    }

    results = {}
    for contam_name, cal_data in contamination_levels.items():
        print(f"\n=== Contamination: {contam_name} ===", flush=True)
        contam_results = {}
        for est_name, est_fn in estimators.items():
            est_results = {}
            for l in layers:
                centroid = est_fn(cal_data[l])
                id_dists = [cosine_distance(e, centroid) for e in id_embs[l]]
                ood_dists = [cosine_distance(e, centroid) for e in ood_embs[l]]
                auroc = compute_auroc(id_dists, ood_dists)
                est_results[f"L{l}"] = {
                    "auroc": auroc,
                    "id_mean": float(np.mean(id_dists)),
                    "id_std": float(np.std(id_dists)),
                    "ood_mean": float(np.mean(ood_dists)),
                    "ood_std": float(np.std(ood_dists)),
                }
                print(f"  {est_name:>20s} L{l}: AUROC={auroc:.4f}", flush=True)
            contam_results[est_name] = est_results
        results[contam_name] = contam_results

    # Summary
    print("\n" + "=" * 80)
    print("AUROC COMPARISON TABLE")
    print(f"{'Estimator':>22s}", end="")
    for contam in contamination_levels:
        for l in layers:
            print(f" {contam[:5]}_L{l:>2d}", end="")
    print()
    for est_name in estimators:
        print(f"{est_name:>22s}", end="")
        for contam in contamination_levels:
            for l in layers:
                auroc = results[contam][est_name][f"L{l}"]["auroc"]
                print(f"   {auroc:.4f}", end="")
        print()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "robust_calibration",
        "experiment_number": 170,
        "timestamp": ts,
        "n_clean_cal": n_clean_cal,
        "n_contam": 2,
        "n_test_id": n_test,
        "n_ood_total": len(ood_embs[layers[0]]),
        "ood_categories": list(ood_transforms.keys()),
        "layers": layers,
        "contamination_levels": list(contamination_levels.keys()),
        "estimators": list(estimators.keys()),
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"robust_calibration_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
