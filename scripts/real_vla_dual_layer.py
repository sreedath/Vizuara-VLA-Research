#!/usr/bin/env python3
"""Experiment 205: Dual-layer ensemble (L3 + L32) — combine early and late layers
to cover both color/intensity and structural OOD types. Tests OR-gate and
average combination strategies.
"""

import json, os, sys, datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO

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

def apply_rain(img, intensity=0.3):
    result = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    n_drops = int(h * w * intensity * 0.001)
    for _ in range(n_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h-10)
        length = np.random.randint(5, 15)
        result[y:min(y+length, h), x] = result[y:min(y+length, h), x] * 0.5 + 127
    result *= 0.7
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_occlusion(img, frac=0.25):
    result = img.copy()
    h, w = img.shape[:2]
    bh, bw = int(h * frac), int(w * frac)
    y = np.random.randint(0, h - bh)
    x = np.random.randint(0, w - bw)
    result[y:y+bh, x:x+bw] = 0
    return result

def apply_jpeg_artifact(img, quality=5):
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))

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
    print("Experiment 205: Dual-Layer Ensemble Detection")
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
    n_test = 8

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

    # Calibration statistics (for normalization)
    cal_dists = {l: [cosine_distance(e, centroids[l]) for e in cal_embs[l]] for l in layers}
    cal_means = {l: np.mean(cal_dists[l]) for l in layers}
    cal_stds = {l: np.std(cal_dists[l]) + 1e-10 for l in layers}

    # ID test
    print("--- ID test ---", flush=True)
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_dists = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            id_dists[l].append(cosine_distance(h[l], centroids[l]))

    # All corruption types
    all_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "rain": lambda a: apply_rain(a, 0.3),
        "occlusion": lambda a: apply_occlusion(a, 0.25),
        "jpeg_q5": lambda a: apply_jpeg_artifact(a, 5),
    }

    print("--- OOD test ---", flush=True)
    ood_dists = {cat: {l: [] for l in layers} for cat in all_transforms}
    for cat, tfn in all_transforms.items():
        for arr in test_arrs:
            np.random.seed(42)
            h = extract_all(Image.fromarray(tfn(arr)))
            for l in layers:
                ood_dists[cat][l].append(cosine_distance(h[l], centroids[l]))

    # Evaluate strategies
    print("\n--- Strategy comparison ---", flush=True)
    results = {}
    
    strategies = {
        "L3_only": lambda id_d, ood_d: (id_d[3], ood_d[3]),
        "L32_only": lambda id_d, ood_d: (id_d[32], ood_d[32]),
        "L1_only": lambda id_d, ood_d: (id_d[1], ood_d[1]),
        "max_L3_L32": lambda id_d, ood_d: (
            [max((id_d[3][i] - cal_means[3]) / cal_stds[3], 
                 (id_d[32][i] - cal_means[32]) / cal_stds[32]) for i in range(len(id_d[3]))],
            [max((ood_d[3][i] - cal_means[3]) / cal_stds[3],
                 (ood_d[32][i] - cal_means[32]) / cal_stds[32]) for i in range(len(ood_d[3]))]
        ),
        "avg_L3_L32": lambda id_d, ood_d: (
            [((id_d[3][i] - cal_means[3]) / cal_stds[3] + 
              (id_d[32][i] - cal_means[32]) / cal_stds[32]) / 2 for i in range(len(id_d[3]))],
            [((ood_d[3][i] - cal_means[3]) / cal_stds[3] +
              (ood_d[32][i] - cal_means[32]) / cal_stds[32]) / 2 for i in range(len(ood_d[3]))]
        ),
    }

    for cat in all_transforms:
        cat_results = {}
        for strat_name, strat_fn in strategies.items():
            id_scores, ood_scores = strat_fn(
                {l: id_dists[l] for l in layers},
                {l: ood_dists[cat][l] for l in layers}
            )
            auroc = compute_auroc(id_scores, ood_scores)
            cat_results[strat_name] = auroc
        
        results[cat] = cat_results
        print(f"  {cat}: " + " ".join(f"{s}={v:.3f}" for s, v in cat_results.items()), flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "dual_layer_ensemble",
        "experiment_number": 205,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"dual_layer_ensemble_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
