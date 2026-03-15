#!/usr/bin/env python3
"""Experiment 187: Calibration set size — how many images needed for reliable detection?

Sweeps calibration set size from 1 to 20 to find the minimum number
of clean images needed for stable OOD detection.
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

def main():
    print("=" * 60)
    print("Experiment 187: Calibration Set Size Sweep")
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
    max_cal = 20
    n_test = 6

    # Pre-generate and extract all embeddings
    print("\n--- Extracting all embeddings ---", flush=True)
    all_cal_arrs = [creators[i%3](i) for i in range(max_cal)]
    all_cal_embs = {l: [] for l in layers}
    for arr in all_cal_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            all_cal_embs[l].append(h[l])
    print(f"  Cal: {max_cal} embeddings", flush=True)

    test_arrs = [creators[(i+max_cal)%3](i+max_cal) for i in range(n_test)]
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
        for arr in test_arrs:
            h = extract_hidden(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for l in layers:
                ood_embs[l].append(h[l])
    print(f"  OOD: {len(ood_embs[3])} embeddings", flush=True)

    # Sweep calibration size
    cal_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    print("\n--- Calibration size sweep ---", flush=True)

    results = {}
    for n_cal in cal_sizes:
        cal_result = {}
        for l in layers:
            centroid = np.array(all_cal_embs[l][:n_cal]).mean(axis=0)
            id_dists = [cosine_distance(e, centroid) for e in id_embs[l]]
            ood_dists = [cosine_distance(e, centroid) for e in ood_embs[l]]
            auroc = compute_auroc(id_dists, ood_dists)
            sep = float(np.mean(ood_dists) / (np.mean(id_dists) + 1e-10))

            # Centroid stability: distance from full centroid
            full_centroid = np.array(all_cal_embs[l]).mean(axis=0)
            centroid_shift = cosine_distance(centroid, full_centroid)

            cal_result[f"L{l}"] = {
                "auroc": auroc,
                "separation_ratio": sep,
                "centroid_shift": centroid_shift,
                "id_mean": float(np.mean(id_dists)),
                "ood_mean": float(np.mean(ood_dists)),
            }

        results[str(n_cal)] = cal_result
        print(f"  n={n_cal:2d}: L3 AUROC={cal_result['L3']['auroc']:.4f} shift={cal_result['L3']['centroid_shift']:.6f} | L32 AUROC={cal_result['L32']['auroc']:.4f} shift={cal_result['L32']['centroid_shift']:.6f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "calibration_size",
        "experiment_number": 187,
        "timestamp": ts,
        "max_cal": max_cal, "n_test": n_test,
        "cal_sizes": cal_sizes,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"calibration_size_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
