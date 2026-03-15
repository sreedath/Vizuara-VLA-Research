#!/usr/bin/env python3
"""Experiment 185: Token position analysis — which token positions carry
the strongest OOD signal?

Instead of only using the last token, compare OOD detection using embeddings
from different token positions (first, middle, image tokens, last).
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

def extract_multi_position(model, processor, image, prompt, layers):
    """Extract hidden states at multiple token positions."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    seq_len = fwd.hidden_states[0].shape[1]
    positions = {
        "first": 0,
        "quarter": seq_len // 4,
        "middle": seq_len // 2,
        "three_quarter": 3 * seq_len // 4,
        "last": -1,
        "second_last": -2,
    }

    result = {}
    for pos_name, pos_idx in positions.items():
        result[pos_name] = {l: fwd.hidden_states[l][0, pos_idx, :].float().cpu().numpy() for l in layers}

    result["_seq_len"] = seq_len
    return result

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
    print("Experiment 185: Token Position OOD Signal Analysis")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [3, 16, 32]

    creators = [create_highway, create_urban, create_rural]
    n_cal = 6
    n_test = 4

    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }

    # Extract calibration embeddings at all positions
    print("\n--- Extracting embeddings ---", flush=True)
    positions = ["first", "quarter", "middle", "three_quarter", "second_last", "last"]
    cal_embs = {pos: {l: [] for l in layers} for pos in positions}
    seq_len = None

    for arr in cal_arrs:
        h = extract_multi_position(model, processor, Image.fromarray(arr), prompt, layers)
        if seq_len is None:
            seq_len = h["_seq_len"]
            print(f"  Sequence length: {seq_len}", flush=True)
        for pos in positions:
            for l in layers:
                cal_embs[pos][l].append(h[pos][l])

    # Centroids per position
    centroids = {pos: {l: np.array(cal_embs[pos][l]).mean(axis=0) for l in layers} for pos in positions}

    # ID test
    id_embs = {pos: {l: [] for l in layers} for pos in positions}
    for arr in test_arrs:
        h = extract_multi_position(model, processor, Image.fromarray(arr), prompt, layers)
        for pos in positions:
            for l in layers:
                id_embs[pos][l].append(h[pos][l])

    # OOD test
    ood_embs = {pos: {l: [] for l in layers} for pos in positions}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h = extract_multi_position(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for pos in positions:
                for l in layers:
                    ood_embs[pos][l].append(h[pos][l])

    # Compute AUROC per position and layer
    print("\n--- AUROC by position and layer ---", flush=True)
    results = {"seq_len": seq_len}
    for pos in positions:
        pos_results = {}
        for l in layers:
            id_dists = [cosine_distance(e, centroids[pos][l]) for e in id_embs[pos][l]]
            ood_dists = [cosine_distance(e, centroids[pos][l]) for e in ood_embs[pos][l]]
            auroc = compute_auroc(id_dists, ood_dists)
            sep = float(np.mean(ood_dists) / (np.mean(id_dists) + 1e-10))

            pos_results[f"L{l}"] = {
                "auroc": auroc,
                "id_mean": float(np.mean(id_dists)),
                "ood_mean": float(np.mean(ood_dists)),
                "separation_ratio": sep,
            }
            print(f"  {pos:15s} L{l:2d}: AUROC={auroc:.4f} sep={sep:.2f}", flush=True)
        results[pos] = pos_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "token_position",
        "experiment_number": 185,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "positions": positions,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"token_position_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
