#!/usr/bin/env python3
"""Experiment 189: Mean pooling vs last token — does aggregating across
multiple token positions improve OOD detection?

Compares last-token embedding, mean pooling over all tokens, and mean pooling
over the second half of tokens (which carry most OOD signal per Exp 185).
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

def extract_pooled(model, processor, image, prompt, layers):
    """Extract multiple pooling strategies."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    seq_len = fwd.hidden_states[0].shape[1]
    half = seq_len // 2

    result = {}
    for l in layers:
        hs = fwd.hidden_states[l][0].float().cpu().numpy()  # [seq_len, hidden_dim]
        result[l] = {
            "last": hs[-1],
            "mean_all": hs.mean(axis=0),
            "mean_second_half": hs[half:].mean(axis=0),
            "mean_last_quarter": hs[3*seq_len//4:].mean(axis=0),
            "max_pool": hs.max(axis=0),
        }
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
    print("Experiment 189: Pooling Strategy Comparison")
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
    pool_methods = ["last", "mean_all", "mean_second_half", "mean_last_quarter", "max_pool"]

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

    # Extract all embeddings with all pooling methods
    print("\n--- Extracting embeddings ---", flush=True)
    cal_embs = {pm: {l: [] for l in layers} for pm in pool_methods}
    for arr in cal_arrs:
        h = extract_pooled(model, processor, Image.fromarray(arr), prompt, layers)
        for pm in pool_methods:
            for l in layers:
                cal_embs[pm][l].append(h[l][pm])

    id_embs = {pm: {l: [] for l in layers} for pm in pool_methods}
    for arr in test_arrs:
        h = extract_pooled(model, processor, Image.fromarray(arr), prompt, layers)
        for pm in pool_methods:
            for l in layers:
                id_embs[pm][l].append(h[l][pm])

    ood_embs = {pm: {l: [] for l in layers} for pm in pool_methods}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h = extract_pooled(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for pm in pool_methods:
                for l in layers:
                    ood_embs[pm][l].append(h[l][pm])

    # Compute AUROC per pooling method
    print("\n--- AUROC by pooling method ---", flush=True)
    results = {}
    for pm in pool_methods:
        pm_results = {}
        for l in layers:
            centroid = np.array(cal_embs[pm][l]).mean(axis=0)
            id_dists = [cosine_distance(e, centroid) for e in id_embs[pm][l]]
            ood_dists = [cosine_distance(e, centroid) for e in ood_embs[pm][l]]
            auroc = compute_auroc(id_dists, ood_dists)
            sep = float(np.mean(ood_dists) / (np.mean(id_dists) + 1e-10))

            pm_results[f"L{l}"] = {
                "auroc": auroc,
                "separation_ratio": sep,
                "id_mean": float(np.mean(id_dists)),
                "ood_mean": float(np.mean(ood_dists)),
            }
            print(f"  {pm:20s} L{l}: AUROC={auroc:.4f} sep={sep:.2f}", flush=True)
        results[pm] = pm_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "pooling_comparison",
        "experiment_number": 189,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "pool_methods": pool_methods,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"pooling_comparison_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
