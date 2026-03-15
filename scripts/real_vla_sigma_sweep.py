#!/usr/bin/env python3
"""Experiment 179: Sigma threshold sweep — FPR/recall trade-off.

Sweeps sigma from 1 to 6 to characterize the precision-recall trade-off
of the threshold-based detector.
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
    print("Experiment 179: Sigma Threshold Sweep")
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
    n_cal = 10
    n_test = 10

    # Calibrate
    print("--- Calibrating ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    cal_stats = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        c = np.array(cal_embs).mean(axis=0)
        dists = [cosine_distance(e, c) for e in cal_embs]
        centroids[l] = c
        cal_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}
    print(f"  L3: mean={cal_stats[3]['mean']:.6f} std={cal_stats[3]['std']:.6f}", flush=True)
    print(f"  L32: mean={cal_stats[32]['mean']:.6f} std={cal_stats[32]['std']:.6f}", flush=True)

    # ID test
    print("--- ID test ---", flush=True)
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_dists = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_dists[l].append(cosine_distance(h[l], centroids[l]))

    # OOD test
    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
    }
    ood_dists = {l: {c: [] for c in ood_transforms} for l in layers}
    for cat, tfn in ood_transforms.items():
        for i in range(n_test):
            arr = tfn(test_arrs[i % n_test])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_dists[l][cat].append(cosine_distance(h[l], centroids[l]))

    # Sigma sweep
    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    results = {}

    for sigma in sigmas:
        thresh = {l: cal_stats[l]["mean"] + sigma * cal_stats[l]["std"] for l in layers}

        # FPR (OR-gate)
        id_flagged = sum(1 for i in range(n_test)
                        if id_dists[3][i] > thresh[3] or id_dists[32][i] > thresh[32])
        fpr = id_flagged / n_test

        # Per-category recall
        per_cat = {}
        total_flag = 0
        total_ood = 0
        for cat in ood_transforms:
            n_cat = len(ood_dists[3][cat])
            flagged = sum(1 for j in range(n_cat)
                         if ood_dists[3][cat][j] > thresh[3] or ood_dists[32][cat][j] > thresh[32])
            per_cat[cat] = flagged / n_cat
            total_flag += flagged
            total_ood += n_cat

        recall = total_flag / total_ood
        prec = total_flag / (total_flag + id_flagged) if (total_flag + id_flagged) > 0 else 1.0
        f1 = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else 0.0

        results[str(sigma)] = {
            "sigma": sigma,
            "fpr": float(fpr),
            "recall": float(recall),
            "precision": float(prec),
            "f1": float(f1),
            "per_category_recall": {c: float(v) for c, v in per_cat.items()},
            "thresholds": {f"L{l}": thresh[l] for l in layers},
        }
        print(f"  σ={sigma:.1f}: FPR={fpr:.3f} recall={recall:.3f} prec={prec:.3f} F1={f1:.3f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "sigma_sweep",
        "experiment_number": 179,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "ood_categories": list(ood_transforms.keys()),
        "layers": layers,
        "cal_stats": cal_stats,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"sigma_sweep_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
