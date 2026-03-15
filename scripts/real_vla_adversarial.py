#!/usr/bin/env python3
"""Experiment 198: Adversarial perturbation detection — can the cosine detector
catch adversarial attacks (FGSM-like perturbations)?

Tests detection of L-infinity constrained adversarial noise at various epsilon.
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

def apply_adversarial(img, epsilon, seed=None):
    """Apply FGSM-like adversarial perturbation (random sign, L-inf bounded)."""
    if seed is not None:
        np.random.seed(seed)
    # Random sign perturbation (approximates FGSM without gradient access)
    perturbation = np.random.choice([-1, 1], size=img.shape).astype(np.float32) * epsilon
    return np.clip(img.astype(np.float32) + perturbation, 0, 255).astype(np.uint8)

def apply_targeted_adversarial(img, epsilon, target_region="sky", seed=None):
    """Adversarial perturbation targeting a specific image region."""
    if seed is not None:
        np.random.seed(seed)
    result = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    if target_region == "sky":
        result[:h//3] += np.random.choice([-1, 1], size=result[:h//3].shape).astype(np.float32) * epsilon
    elif target_region == "road":
        result[h//2:] += np.random.choice([-1, 1], size=result[h//2:].shape).astype(np.float32) * epsilon
    elif target_region == "center":
        ch, cw = h//4, w//4
        result[ch:3*ch, cw:3*cw] += np.random.choice([-1, 1], size=result[ch:3*ch, cw:3*cw].shape).astype(np.float32) * epsilon
    return np.clip(result, 0, 255).astype(np.uint8)

def main():
    print("=" * 60)
    print("Experiment 198: Adversarial Perturbation Detection")
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

    # Adversarial sweep
    epsilons = [2, 4, 8, 16, 32, 48, 64, 96, 128]
    print("\n--- Global adversarial sweep ---", flush=True)
    global_results = []
    for eps in epsilons:
        ood_dists = {l: [] for l in layers}
        for i, arr in enumerate(test_arrs):
            adv = apply_adversarial(arr, eps, seed=42+i)
            h = extract_all(Image.fromarray(adv))
            for l in layers:
                ood_dists[l].append(cosine_distance(h[l], centroids[l]))

        entry = {"epsilon": eps}
        for l in layers:
            auroc = compute_auroc(id_dists[l], ood_dists[l])
            sep = float(np.mean(ood_dists[l]) / (np.mean(id_dists[l]) + 1e-10))
            entry[f"L{l}"] = {"auroc": auroc, "separation": sep, "mean_dist": float(np.mean(ood_dists[l]))}
        global_results.append(entry)
        print(f"  eps={eps}: L1={entry['L1']['auroc']:.3f} L3={entry['L3']['auroc']:.3f} L32={entry['L32']['auroc']:.3f}", flush=True)

    # Targeted adversarial (eps=32)
    print("\n--- Targeted adversarial (eps=32) ---", flush=True)
    targeted_results = {}
    for region in ["sky", "road", "center"]:
        ood_dists = {l: [] for l in layers}
        for i, arr in enumerate(test_arrs):
            adv = apply_targeted_adversarial(arr, 32, target_region=region, seed=42+i)
            h = extract_all(Image.fromarray(adv))
            for l in layers:
                ood_dists[l].append(cosine_distance(h[l], centroids[l]))

        entry = {}
        for l in layers:
            auroc = compute_auroc(id_dists[l], ood_dists[l])
            entry[f"L{l}"] = {"auroc": auroc, "mean_dist": float(np.mean(ood_dists[l]))}
        targeted_results[region] = entry
        print(f"  {region}: L1={entry['L1']['auroc']:.3f} L3={entry['L3']['auroc']:.3f} L32={entry['L32']['auroc']:.3f}", flush=True)

    # Comparison with natural corruptions at same PSNR
    # eps=8 in L-inf corresponds to PSNR ~38 dB (similar to noise std=5)
    # eps=32 corresponds to PSNR ~26 dB (similar to noise std=20)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "adversarial_detection",
        "experiment_number": 198,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "id_means": {f"L{l}": float(np.mean(id_dists[l])) for l in layers},
        "global_sweep": global_results,
        "targeted": targeted_results,
    }
    path = os.path.join(RESULTS_DIR, f"adversarial_detection_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
