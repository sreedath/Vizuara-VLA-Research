#!/usr/bin/env python3
"""Experiment 175: Fine-grained corruption severity vs. AUROC curves.

Traces AUROC as a function of corruption intensity for each type,
identifying detection thresholds and sensitivity profiles.
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
def apply_night(a, b=0.15):
    return np.clip(a*b, 0, 255).astype(np.uint8)
def apply_blur(a, r=8):
    return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=r)))
def apply_noise(a, s=50):
    return np.clip(a.astype(np.float32)+np.random.normal(0,s,a.shape), 0, 255).astype(np.uint8)

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
    print("Experiment 175: Corruption Severity vs. AUROC Curves")
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
    n_cal = 8
    n_test = 6

    # Calibrate
    print("--- Calibrating ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        centroids[l] = np.array(cal_embs).mean(axis=0)

    # ID test embeddings
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_dists = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_dists[l].append(cosine_distance(h[l], centroids[l]))

    # Severity sweeps
    severity_configs = {
        "fog": {
            "levels": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80],
            "fn": lambda arr, s: apply_fog(arr, s),
            "label": "alpha",
        },
        "night": {
            "levels": [0.80, 0.60, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02],
            "fn": lambda arr, s: apply_night(arr, s),
            "label": "brightness",
        },
        "blur": {
            "levels": [1, 2, 3, 4, 6, 8, 10, 15, 20],
            "fn": lambda arr, s: apply_blur(arr, s),
            "label": "radius",
        },
        "noise": {
            "levels": [5, 10, 15, 20, 30, 40, 50, 70, 100],
            "fn": lambda arr, s: apply_noise(arr, s),
            "label": "sigma",
        },
    }

    results = {}
    for corr_name, config in severity_configs.items():
        print(f"\n=== {corr_name} ===", flush=True)
        corr_results = []
        for severity in config["levels"]:
            ood_dists = {l: [] for l in layers}
            for arr in test_arrs:
                corrupted = config["fn"](arr, severity)
                h = extract_hidden(model, processor, Image.fromarray(corrupted), prompt, layers)
                for l in layers:
                    ood_dists[l].append(cosine_distance(h[l], centroids[l]))

            level_result = {"severity": severity}
            for l in layers:
                auroc = compute_auroc(id_dists[l], ood_dists[l])
                mean_dist = float(np.mean(ood_dists[l]))
                level_result[f"L{l}_auroc"] = auroc
                level_result[f"L{l}_mean_dist"] = mean_dist
            corr_results.append(level_result)
            print(f"  {config['label']}={severity}: L3={level_result['L3_auroc']:.4f} "
                  f"L32={level_result['L32_auroc']:.4f}", flush=True)
        results[corr_name] = corr_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "severity_curves",
        "experiment_number": 175,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"severity_curves_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
