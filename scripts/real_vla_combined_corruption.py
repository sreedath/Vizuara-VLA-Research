#!/usr/bin/env python3
"""Experiment 188: Combined corruptions — how does the detector handle
multiple simultaneous corruptions?

Tests if detection improves, degrades, or saturates when two corruptions
are applied simultaneously (e.g., fog+noise, night+blur).
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
    print("Experiment 188: Combined Corruption Detection")
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
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    cal_dists = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        c = np.array(cal_embs).mean(axis=0)
        centroids[l] = c
        cal_dists[l] = [cosine_distance(e, c) for e in cal_embs]

    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    # ID distances
    id_dists = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_dists[l].append(cosine_distance(h[l], centroids[l]))

    # Single corruptions
    single = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }

    # Combined corruptions
    combined = {
        "fog+noise": lambda a: apply_noise(apply_fog(a, 0.6)),
        "fog+blur": lambda a: apply_blur(apply_fog(a, 0.6)),
        "night+noise": lambda a: apply_noise(apply_night(a)),
        "night+blur": lambda a: apply_blur(apply_night(a)),
        "blur+noise": lambda a: apply_noise(apply_blur(a)),
        "fog+night": lambda a: apply_night(apply_fog(a, 0.6)),
        "fog+blur+noise": lambda a: apply_noise(apply_blur(apply_fog(a, 0.6))),
        "night+blur+noise": lambda a: apply_noise(apply_blur(apply_night(a))),
    }

    all_corruptions = {**single, **combined}
    results = {}

    for cat, tfn in all_corruptions.items():
        print(f"\n  Processing: {cat}", flush=True)
        ood_embs = {l: [] for l in layers}
        for arr in test_arrs:
            h = extract_hidden(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for l in layers:
                ood_embs[l].append(h[l])

        cat_results = {}
        for l in layers:
            ood_d = [cosine_distance(e, centroids[l]) for e in ood_embs[l]]
            auroc = compute_auroc(id_dists[l], ood_d)
            cat_results[f"L{l}"] = {
                "auroc": auroc,
                "mean_dist": float(np.mean(ood_d)),
                "max_dist": float(np.max(ood_d)),
            }
            print(f"    L{l}: AUROC={auroc:.4f} mean_dist={np.mean(ood_d):.6f}", flush=True)
        results[cat] = cat_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "combined_corruption",
        "experiment_number": 188,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "corruption_types": list(all_corruptions.keys()),
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"combined_corruption_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
