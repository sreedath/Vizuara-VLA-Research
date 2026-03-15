#!/usr/bin/env python3
"""Experiment 183: Cross-corruption transfer — does calibrating on one
corruption type detect others?

Tests the generalizability of OOD detection by calibrating with only one
type of corruption and testing whether it detects unseen corruption types.
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
def apply_snow(a):
    snow = np.random.randint(180, 256, a.shape, dtype=np.uint8)
    mask = np.random.random(a.shape[:2]) > 0.85
    out = a.copy(); out[mask] = snow[mask]; return out
def apply_rain(a):
    out = a.copy()
    for _ in range(200):
        x = np.random.randint(0, SIZE[1])
        y = np.random.randint(0, SIZE[0]-15)
        out[y:y+15, x:min(x+1,SIZE[1]-1)] = [180, 180, 220]
    return np.clip(out.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)

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
    print("Experiment 183: Cross-Corruption Transfer")
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
    n_base = 8

    # Generate base images
    base_arrs = [creators[i%3](i) for i in range(n_base)]

    corruption_types = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "snow": apply_snow,
        "rain": apply_rain,
    }

    # Extract ID embeddings (clean images as calibration)
    print("\n--- Extracting ID embeddings ---", flush=True)
    id_embs = {l: [] for l in layers}
    for arr in base_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_embs[l].append(h[l])

    id_centroid = {l: np.array(id_embs[l]).mean(axis=0) for l in layers}
    id_dists = {l: [cosine_distance(e, id_centroid[l]) for e in id_embs[l]] for l in layers}

    # Extract OOD embeddings per corruption type
    print("--- Extracting OOD embeddings ---", flush=True)
    ood_embs = {cat: {l: [] for l in layers} for cat in corruption_types}
    for cat, tfn in corruption_types.items():
        for arr in base_arrs:
            h = extract_hidden(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for l in layers:
                ood_embs[cat][l].append(h[l])
        print(f"  {cat}: done", flush=True)

    # Cross-corruption transfer matrix:
    # For each corruption type C_train, compute threshold from C_train distances
    # Then test detection on all other corruption types C_test
    print("\n--- Cross-corruption transfer ---", flush=True)
    results = {}
    for l in layers:
        layer_results = {}

        # Baseline: calibrate with clean, test each corruption
        baseline = {}
        for cat in corruption_types:
            ood_dists = [cosine_distance(e, id_centroid[l]) for e in ood_embs[cat][l]]
            baseline[cat] = compute_auroc(id_dists[l], ood_dists)
        layer_results["baseline_auroc"] = baseline
        print(f"\n  L{l} Baseline AUROCs: {baseline}", flush=True)

        # Cross-corruption: calibrate threshold from one corruption, test on others
        # For each train corruption, compute distances and use its mean+3*std as threshold
        transfer_matrix = {}
        for train_cat in corruption_types:
            train_dists = [cosine_distance(e, id_centroid[l]) for e in ood_embs[train_cat][l]]
            # The "threshold" knowledge: we know the OOD distances from train corruption
            # We test: does the same centroid-based detector catch OTHER corruptions?
            # This is simply AUROC of clean vs each test corruption
            # But more interesting: if we set threshold at 3σ from clean, does it catch all?
            cal_mean = float(np.mean(id_dists[l]))
            cal_std = float(np.std(id_dists[l]))
            thresh = cal_mean + 3.0 * cal_std

            row = {}
            for test_cat in corruption_types:
                test_dists = [cosine_distance(e, id_centroid[l]) for e in ood_embs[test_cat][l]]
                # Detection rate at this threshold
                detected = sum(1 for d in test_dists if d > thresh)
                row[test_cat] = detected / len(test_dists)
            transfer_matrix[train_cat] = row

        # Actually the more interesting thing: cross-centroid detection
        # Calibrate centroid from CORRUPTED images of one type, test distance from that centroid
        cross_centroid = {}
        for train_cat in corruption_types:
            train_centroid = np.array(ood_embs[train_cat][l]).mean(axis=0)
            row = {}
            for test_cat in corruption_types:
                # Distance of test_cat samples from train_cat centroid
                test_dists = [cosine_distance(e, train_centroid) for e in ood_embs[test_cat][l]]
                # Distance of clean from train_cat centroid
                clean_dists = [cosine_distance(e, train_centroid) for e in id_embs[l]]
                # If same corruption: should be close (small distance)
                # If different: could be either
                row[test_cat] = {
                    "mean_dist": float(np.mean(test_dists)),
                    "clean_mean_dist": float(np.mean(clean_dists)),
                }
            cross_centroid[train_cat] = row

        layer_results["transfer_detection_at_3sigma"] = transfer_matrix
        layer_results["cross_centroid_distances"] = cross_centroid

        # Summary: for each test corruption, what fraction of train corruptions detect it?
        detectability = {}
        for test_cat in corruption_types:
            detected_by = sum(1 for train_cat in corruption_types
                            if transfer_matrix[train_cat][test_cat] >= 0.5)
            detectability[test_cat] = detected_by / len(corruption_types)
        layer_results["detectability_fraction"] = detectability
        print(f"  L{l} detectability: {detectability}", flush=True)

        results[f"L{l}"] = layer_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "cross_corruption_transfer",
        "experiment_number": 183,
        "timestamp": ts,
        "n_base": n_base,
        "corruption_types": list(corruption_types.keys()),
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"cross_corruption_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
