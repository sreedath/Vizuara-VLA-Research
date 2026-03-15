#!/usr/bin/env python3
"""Experiment 204: Novel OOD types — can the detector catch corruption types
never seen during calibration? Tests rain, snow, occlusion, color shift,
and contrast reduction.
"""

import json, os, sys, datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

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

# Novel corruption types
def apply_rain(img, intensity=0.3):
    """Simulate rain streaks."""
    result = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    n_drops = int(h * w * intensity * 0.001)
    for _ in range(n_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h-10)
        length = np.random.randint(5, 15)
        result[y:min(y+length, h), x] = result[y:min(y+length, h), x] * 0.5 + 127
    # Add overall darkening
    result *= 0.7
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_snow(img, intensity=0.4):
    """Simulate snow (white specks + brightness increase)."""
    result = img.copy().astype(np.float32)
    result = result * 0.8 + 40  # overall brightening
    h, w = img.shape[:2]
    n_flakes = int(h * w * intensity * 0.002)
    for _ in range(n_flakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        size = np.random.randint(1, 4)
        result[max(0,y-size):min(h,y+size), max(0,x-size):min(w,x+size)] = 255
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_occlusion(img, frac=0.25):
    """Simulate partial occlusion (random black rectangle)."""
    result = img.copy()
    h, w = img.shape[:2]
    bh, bw = int(h * frac), int(w * frac)
    y = np.random.randint(0, h - bh)
    x = np.random.randint(0, w - bw)
    result[y:y+bh, x:x+bw] = 0
    return result

def apply_color_shift(img, shift=50):
    """Simulate color temperature shift."""
    result = img.copy().astype(np.float32)
    result[:,:,0] += shift  # increase red
    result[:,:,2] -= shift  # decrease blue
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_contrast_reduction(img, factor=0.3):
    """Reduce contrast."""
    mean = np.mean(img, axis=(0,1), keepdims=True)
    result = mean + (img.astype(np.float32) - mean) * factor
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_jpeg_artifact(img, quality=5):
    """Simulate heavy JPEG compression artifacts."""
    from io import BytesIO
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))

def apply_pixelation(img, block=16):
    """Simulate pixelation / downscale-upscale."""
    pil = Image.fromarray(img)
    small = pil.resize((img.shape[1]//block, img.shape[0]//block), Image.NEAREST)
    return np.array(small.resize(pil.size, Image.NEAREST))

def main():
    print("=" * 60)
    print("Experiment 204: Novel OOD Type Detection")
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

    # Calibrate (clean only)
    print("\n--- Calibration (clean) ---", flush=True)
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

    # Novel OOD types
    novel_transforms = {
        "rain": lambda a: apply_rain(a, 0.3),
        "snow": lambda a: apply_snow(a, 0.4),
        "occlusion_25": lambda a: apply_occlusion(a, 0.25),
        "color_shift": lambda a: apply_color_shift(a, 50),
        "low_contrast": lambda a: apply_contrast_reduction(a, 0.3),
        "jpeg_q5": lambda a: apply_jpeg_artifact(a, 5),
        "pixelation": lambda a: apply_pixelation(a, 16),
    }

    print("\n--- Novel OOD detection ---", flush=True)
    results = {}
    for cat, tfn in novel_transforms.items():
        ood_dists = {l: [] for l in layers}
        for arr in test_arrs:
            np.random.seed(42)  # consistent randomness for rain/snow
            h = extract_all(Image.fromarray(tfn(arr)))
            for l in layers:
                ood_dists[l].append(cosine_distance(h[l], centroids[l]))

        cat_results = {}
        for l in layers:
            auroc = compute_auroc(id_dists[l], ood_dists[l])
            sep = float(np.mean(ood_dists[l]) / (np.mean(id_dists[l]) + 1e-10))
            cat_results[f"L{l}"] = {
                "auroc": auroc,
                "separation": sep,
                "mean_dist": float(np.mean(ood_dists[l])),
            }
        results[cat] = cat_results
        print(f"  {cat}: L1={cat_results['L1']['auroc']:.3f} "
              f"L3={cat_results['L3']['auroc']:.3f} "
              f"L32={cat_results['L32']['auroc']:.3f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "novel_ood",
        "experiment_number": 204,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "id_means": {f"L{l}": float(np.mean(id_dists[l])) for l in layers},
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"novel_ood_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
