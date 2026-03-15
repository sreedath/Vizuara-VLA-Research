#!/usr/bin/env python3
"""Experiment 163: Token position analysis for OOD detection.

Compares OOD detection using embeddings from different token positions:
last token, first token, mean pooling, max pooling. Determines whether
position choice matters for detection quality.
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

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def extract_multi_position(model, processor, image, prompt, layers):
    """Extract embeddings from multiple token positions."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    
    results = {}
    for l in layers:
        hs = fwd.hidden_states[l][0]  # (seq_len, dim)
        seq_len = hs.shape[0]
        results[l] = {
            "last": hs[-1].float().cpu().numpy(),
            "first": hs[0].float().cpu().numpy(),
            "mean": hs.mean(dim=0).float().cpu().numpy(),
            "max": hs.max(dim=0).values.float().cpu().numpy(),
            "mid": hs[seq_len//2].float().cpu().numpy(),
            "seq_len": seq_len,
        }
    return results

def main():
    print("=" * 60)
    print("Experiment 163: Token Position Analysis")
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
    positions = ["last", "first", "mean", "max", "mid"]

    creators = [create_highway, create_urban, create_rural]
    n_cal = 8
    n_test = 6

    # Calibration
    print("\n--- Calibrating ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    cal_embs = {l: {p: [] for p in positions} for l in layers}
    for arr in cal_arrs:
        mp = extract_multi_position(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            for p in positions:
                cal_embs[l][p].append(mp[l][p])

    centroids = {l: {p: np.array(cal_embs[l][p]).mean(axis=0) for p in positions} for l in layers}

    # Test
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    n_ood_per = 5

    # ID test
    id_dists = {l: {p: [] for p in positions} for l in layers}
    for arr in test_arrs:
        mp = extract_multi_position(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            for p in positions:
                id_dists[l][p].append(cosine_distance(mp[l][p], centroids[l][p]))

    # OOD test
    ood_dists = {l: {p: [] for p in positions} for l in layers}
    for cat, tfn in ood_transforms.items():
        print(f"  OOD: {cat}", flush=True)
        for j in range(n_ood_per):
            arr = tfn(test_arrs[j % n_test])
            mp = extract_multi_position(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                for p in positions:
                    ood_dists[l][p].append(cosine_distance(mp[l][p], centroids[l][p]))

    # Compute AUROC for each position
    results = {}
    for l in layers:
        layer_results = {}
        for p in positions:
            auroc = compute_auroc(np.array(id_dists[l][p]), np.array(ood_dists[l][p]))
            id_mean = float(np.mean(id_dists[l][p]))
            ood_mean = float(np.mean(ood_dists[l][p]))
            layer_results[p] = {
                "auroc": auroc,
                "id_mean": id_mean,
                "ood_mean": ood_mean,
                "separation": ood_mean / (id_mean + 1e-10),
            }
            print(f"  L{l} {p:>5s}: AUROC={auroc:.4f} ID={id_mean:.6f} OOD={ood_mean:.6f}", flush=True)
        results[f"L{l}"] = layer_results

    # Sequence length info
    test_mp = extract_multi_position(model, processor, Image.fromarray(test_arrs[0]), prompt, layers)
    seq_len = test_mp[layers[0]]["seq_len"]
    print(f"\n  Sequence length: {seq_len}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "token_position",
        "experiment_number": 163,
        "timestamp": ts,
        "n_cal": n_cal, "n_test_id": n_test,
        "n_ood_total": n_ood_per * len(ood_transforms),
        "positions": positions,
        "layers": layers,
        "seq_len": seq_len,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"token_position_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
