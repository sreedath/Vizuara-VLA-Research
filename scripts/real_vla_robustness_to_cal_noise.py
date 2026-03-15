#!/usr/bin/env python3
"""Experiment 168: Robustness to calibration noise.

Tests how noisy calibration images (slightly corrupted) affect detection
quality. Real deployments may have imperfect calibration data.
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
    print("Experiment 168: Robustness to Calibration Noise")
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
    n_cal = 9  # 3 per type
    n_test = 6

    # Generate base images
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    # Define calibration noise levels
    cal_noise_levels = {
        "clean": lambda a: a,
        "noise_5": lambda a: apply_noise(a, 5),
        "noise_10": lambda a: apply_noise(a, 10),
        "noise_20": lambda a: apply_noise(a, 20),
        "fog_5": lambda a: apply_fog(a, 0.05),
        "fog_10": lambda a: apply_fog(a, 0.10),
        "fog_15": lambda a: apply_fog(a, 0.15),
        "blur_1": lambda a: apply_blur(a, 1),
        "blur_2": lambda a: apply_blur(a, 2),
        "mixed_light": lambda a: apply_noise(apply_fog(a, 0.05), 5),
        "mixed_medium": lambda a: apply_noise(apply_fog(a, 0.10), 10),
    }

    # OOD test conditions (fixed)
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur_8": apply_blur,
        "noise_50": lambda a: apply_noise(a, 50),
    }
    n_ood_per = 5

    # Pre-extract test embeddings (same for all calibration conditions)
    print("\n--- Pre-extracting test embeddings ---", flush=True)
    id_test_embs = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_test_embs[l].append(h[l])

    ood_test_embs = {l: [] for l in layers}
    for cat, tfn in ood_transforms.items():
        for j in range(n_ood_per):
            arr = tfn(test_arrs[j % n_test])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_test_embs[l].append(h[l])

    results = {}
    for noise_name, noise_fn in cal_noise_levels.items():
        print(f"\n--- Cal noise: {noise_name} ---", flush=True)
        # Apply noise to calibration images
        noisy_cal = [noise_fn(arr) for arr in cal_arrs]
        
        # Extract calibration embeddings
        cal_embs = {l: [] for l in layers}
        for arr in noisy_cal:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                cal_embs[l].append(h[l])
        
        # Compute centroid and AUROC
        entry = {}
        for l in layers:
            centroid = np.array(cal_embs[l]).mean(axis=0)
            id_dists = [cosine_distance(e, centroid) for e in id_test_embs[l]]
            ood_dists = [cosine_distance(e, centroid) for e in ood_test_embs[l]]
            auroc = compute_auroc(np.array(id_dists), np.array(ood_dists))
            
            entry[f"L{l}_auroc"] = auroc
            entry[f"L{l}_id_mean"] = float(np.mean(id_dists))
            entry[f"L{l}_ood_mean"] = float(np.mean(ood_dists))
            print(f"  L{l}: AUROC={auroc:.4f} ID={np.mean(id_dists):.6f} OOD={np.mean(ood_dists):.4f}", flush=True)
        
        results[noise_name] = entry

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "calibration_noise_robustness",
        "experiment_number": 168,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test, "n_ood_per": n_ood_per,
        "cal_noise_levels": list(cal_noise_levels.keys()),
        "ood_conditions": list(ood_transforms.keys()),
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"cal_noise_robustness_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
