#!/usr/bin/env python3
"""Experiment 162: Sample efficiency curve with confidence intervals.

Systematic study of how detection AUROC scales with calibration set size,
from n=2 to n=20, with bootstrapped confidence intervals.
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

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def main():
    print("=" * 60)
    print("Experiment 162: Sample Efficiency Curve with CIs")
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

    # Generate a large pool of images
    n_pool = 20  # max calibration size
    n_test = 8
    n_ood_per = 4

    pool_arrs = [creators[i%3](i) for i in range(n_pool)]
    test_arrs = [creators[(i+n_pool)%3](i+n_pool) for i in range(n_test)]

    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
    }

    # Pre-compute all embeddings
    print("\n--- Pre-computing all embeddings ---", flush=True)
    pool_embs = {l: [] for l in layers}
    for i, arr in enumerate(pool_arrs):
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            pool_embs[l].append(h[l])
        if i % 5 == 0: print(f"  Pool {i}/{n_pool}", flush=True)

    test_embs = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            test_embs[l].append(h[l])

    ood_embs = {l: [] for l in layers}
    for cat, tfn in ood_transforms.items():
        for j in range(n_ood_per):
            arr = tfn(test_arrs[j % n_test])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_embs[l].append(h[l])
    print("  All embeddings pre-computed.", flush=True)

    # Sample efficiency: sweep n_cal from 2 to 20
    cal_sizes = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20]
    n_bootstrap = 20
    np_rng = np.random.RandomState(123)

    results = {}
    for n_cal in cal_sizes:
        print(f"\n--- n_cal = {n_cal} ---", flush=True)
        layer_results = {}
        for l in layers:
            bootstrap_aurocs = []
            for b in range(n_bootstrap):
                # Sample n_cal from pool (with replacement for bootstrap)
                indices = np_rng.choice(n_pool, size=n_cal, replace=True)
                cal_sub = [pool_embs[l][i] for i in indices]
                centroid = np.array(cal_sub).mean(axis=0)

                # Compute distances
                id_dists = [cosine_distance(e, centroid) for e in test_embs[l]]
                ood_dists = [cosine_distance(e, centroid) for e in ood_embs[l]]
                auroc = compute_auroc(id_dists, ood_dists)
                bootstrap_aurocs.append(auroc)

            aurocs = np.array(bootstrap_aurocs)
            layer_results[f"L{l}"] = {
                "mean_auroc": float(np.mean(aurocs)),
                "std_auroc": float(np.std(aurocs)),
                "ci_lower": float(np.percentile(aurocs, 2.5)),
                "ci_upper": float(np.percentile(aurocs, 97.5)),
                "min_auroc": float(np.min(aurocs)),
                "max_auroc": float(np.max(aurocs)),
            }
            print(f"  L{l}: AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f} "
                  f"[{np.percentile(aurocs, 2.5):.4f}, {np.percentile(aurocs, 97.5):.4f}]", flush=True)

        results[str(n_cal)] = layer_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "sample_efficiency_curve",
        "experiment_number": 162,
        "timestamp": ts,
        "n_pool": n_pool, "n_test": n_test, "n_ood_per_cat": n_ood_per,
        "n_bootstrap": n_bootstrap,
        "cal_sizes": cal_sizes,
        "layers": layers,
        "ood_categories": list(ood_transforms.keys()),
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"sample_eff_curve_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
