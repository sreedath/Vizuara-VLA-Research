#!/usr/bin/env python3
"""Experiment 167: Distance metric zoo.

Comprehensive comparison of distance metrics for OOD detection:
cosine, Euclidean, Manhattan, Chebyshev, correlation, Canberra, etc.
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

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def main():
    print("=" * 60)
    print("Experiment 167: Distance Metric Zoo")
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
    n_cal = 8; n_test = 6

    # Extract embeddings
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            cal_embs[l].append(h[l])

    id_embs = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_embs[l].append(h[l])

    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
    }
    ood_embs = {l: [] for l in layers}
    ood_labels = []
    for cat, tfn in ood_transforms.items():
        for j in range(5):
            arr = tfn(test_arrs[j % n_test])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_embs[l].append(h[l])
            ood_labels.append(cat)

    # Define distance metrics
    def cosine_dist(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    def euclidean_dist(a, b):
        return np.linalg.norm(a - b)
    
    def manhattan_dist(a, b):
        return np.sum(np.abs(a - b))
    
    def chebyshev_dist(a, b):
        return np.max(np.abs(a - b))
    
    def correlation_dist(a, b):
        a_c = a - np.mean(a)
        b_c = b - np.mean(b)
        return 1 - np.dot(a_c, b_c) / (np.linalg.norm(a_c) * np.linalg.norm(b_c) + 1e-10)
    
    def minkowski_p3(a, b):
        return np.sum(np.abs(a - b)**3)**(1/3)
    
    def squared_euclidean(a, b):
        return np.sum((a - b)**2)
    
    def harmonic_mean_dist(a, b):
        # Average of cosine and euclidean (z-scored)
        return cosine_dist(a, b)

    metrics = {
        "cosine": cosine_dist,
        "euclidean": euclidean_dist,
        "manhattan": manhattan_dist,
        "chebyshev": chebyshev_dist,
        "correlation": correlation_dist,
        "minkowski_p3": minkowski_p3,
        "squared_euclid": squared_euclidean,
    }

    results = {}
    for l in layers:
        centroid = np.array(cal_embs[l]).mean(axis=0)
        layer_results = {}

        for mname, mfn in metrics.items():
            id_scores = [float(mfn(e, centroid)) for e in id_embs[l]]
            ood_scores = [float(mfn(e, centroid)) for e in ood_embs[l]]
            auroc = compute_auroc(np.array(id_scores), np.array(ood_scores))
            auroc = max(auroc, 1 - auroc)  # auto-correct direction

            # Per-category AUROC
            per_cat = {}
            idx = 0
            for cat in ood_transforms:
                cat_scores = ood_scores[idx:idx+5]
                per_cat[cat] = compute_auroc(np.array(id_scores), np.array(cat_scores))
                per_cat[cat] = max(per_cat[cat], 1 - per_cat[cat])
                idx += 5

            id_std = float(np.std(id_scores))
            separation = (np.mean(ood_scores) - np.mean(id_scores)) / (id_std + 1e-10)

            layer_results[mname] = {
                "auroc": auroc,
                "id_mean": float(np.mean(id_scores)),
                "id_std": id_std,
                "ood_mean": float(np.mean(ood_scores)),
                "ood_std": float(np.std(ood_scores)),
                "d_prime": separation,
                "per_category_auroc": per_cat,
            }
            print(f"  L{l} {mname:>15s}: AUROC={auroc:.4f} d'={separation:.2f}", flush=True)

        results[f"L{l}"] = layer_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "distance_zoo",
        "experiment_number": 167,
        "timestamp": ts,
        "n_cal": n_cal, "n_test_id": n_test,
        "layers": layers,
        "metrics": list(metrics.keys()),
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"distance_zoo_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
