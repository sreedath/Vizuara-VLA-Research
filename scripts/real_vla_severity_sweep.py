#!/usr/bin/env python3
"""Experiment 196: Corruption severity sweep — at what severity does detection
become perfect? Maps the sensitivity curve of the cosine detector.

Sweeps fog alpha (0.05-0.8), night brightness (0.05-0.9), blur radius (1-16),
noise std (5-80).
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
    print("Experiment 196: Corruption Severity Sweep")
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
    n_cal = 10
    n_test = 6

    def extract_all(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    # Calibrate
    print("\n--- Calibration ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            cal_embs[l].append(h[l])
    centroids = {l: np.array(cal_embs[l]).mean(axis=0) for l in layers}

    # ID baseline
    print("--- ID baseline ---", flush=True)
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_dists = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            id_dists[l].append(cosine_distance(h[l], centroids[l]))

    # Severity sweeps
    severity_configs = {
        "fog": {
            "alphas": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
            "transform": lambda a, alpha: np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8),
        },
        "night": {
            "alphas": [0.90, 0.70, 0.50, 0.30, 0.15, 0.10, 0.05],
            "transform": lambda a, brightness: np.clip(a*brightness, 0, 255).astype(np.uint8),
        },
        "blur": {
            "alphas": [1, 2, 3, 4, 6, 8, 12, 16],
            "transform": lambda a, radius: np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=radius))),
        },
        "noise": {
            "alphas": [5, 10, 15, 20, 30, 40, 50, 60, 80],
            "transform": lambda a, std: np.clip(a.astype(np.float32)+np.random.normal(0,std,a.shape), 0, 255).astype(np.uint8),
        },
    }

    results = {}
    for corruption, config in severity_configs.items():
        print(f"\n--- {corruption} severity sweep ---", flush=True)
        corruption_results = []
        for severity in config["alphas"]:
            ood_dists = {l: [] for l in layers}
            for arr in test_arrs:
                corrupted = config["transform"](arr, severity)
                h = extract_all(Image.fromarray(corrupted))
                for l in layers:
                    ood_dists[l].append(cosine_distance(h[l], centroids[l]))

            layer_metrics = {}
            for l in layers:
                auroc = compute_auroc(id_dists[l], ood_dists[l])
                sep = float(np.mean(ood_dists[l]) / (np.mean(id_dists[l]) + 1e-10))
                layer_metrics[f"L{l}"] = {
                    "auroc": auroc,
                    "separation": sep,
                    "mean_dist": float(np.mean(ood_dists[l])),
                }
            corruption_results.append({
                "severity": severity,
                "layers": layer_metrics,
            })
            print(f"  {corruption}={severity}: L1={layer_metrics['L1']['auroc']:.3f} L3={layer_metrics['L3']['auroc']:.3f} L32={layer_metrics['L32']['auroc']:.3f}", flush=True)

        results[corruption] = corruption_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "severity_sweep",
        "experiment_number": 196,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "id_means": {f"L{l}": float(np.mean(id_dists[l])) for l in layers},
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"severity_sweep_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
