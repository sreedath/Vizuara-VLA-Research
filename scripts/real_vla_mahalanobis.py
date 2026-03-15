#!/usr/bin/env python3
"""Experiment 145: Mahalanobis distance vs cosine distance at L3 and L32.

Compares covariance-aware Mahalanobis distance with cosine distance.
Tests whether accounting for covariance structure improves fog detection.
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
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.randint(-5, 6, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.randint(-5, 6, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_rural(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [100, 180, 255]
    img[SIZE[0]//3:SIZE[0]*2//3] = [34, 139, 34]
    img[SIZE[0]*2//3:] = [90, 90, 90]
    noise = rng.randint(-8, 9, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def apply_fog(arr, alpha):
    fog = np.full_like(arr, [200, 200, 210])
    return np.clip(arr * (1 - alpha) + fog * alpha, 0, 255).astype(np.uint8)

def apply_night(arr):
    return np.clip(arr * 0.15, 0, 255).astype(np.uint8)

def apply_blur(arr, radius=8):
    return np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius)))

def apply_noise_corruption(arr, std=50):
    return np.clip(arr.astype(np.float32) + np.random.normal(0, std, arr.shape), 0, 255).astype(np.uint8)

def apply_occlusion(arr, frac=0.3):
    out = arr.copy(); h, w = out.shape[:2]
    bh, bw = int(h*frac), int(w*frac)
    out[h//2-bh//2:h//2+bh//2, w//2-bw//2:w//2+bw//2] = 128
    return out

def apply_snow(arr):
    out = arr.astype(np.float32) * 0.7 + 76.5
    out[np.random.random(arr.shape[:2]) > 0.97] = 255
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_overexpose(arr):
    return np.clip(arr.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_metrics(id_dists, ood_dists):
    id_arr, ood_arr = np.array(id_dists), np.array(ood_dists)
    labels = np.array([0]*len(id_dists) + [1]*len(ood_dists))
    scores = np.concatenate([id_arr, ood_arr])
    n_pos, n_neg = int(np.sum(labels==1)), int(np.sum(labels==0))
    correct = sum(1 for i in range(len(scores)) for j in range(len(scores))
                  if labels[i]==1 and labels[j]==0 and scores[i]>scores[j])
    correct += 0.5 * sum(1 for i in range(len(scores)) for j in range(len(scores))
                         if labels[i]==1 and labels[j]==0 and scores[i]==scores[j])
    auroc = correct / (n_pos * n_neg) if n_pos*n_neg > 0 else 0.5
    pooled = np.sqrt((np.var(id_arr) + np.var(ood_arr)) / 2 + 1e-10)
    d = float((np.mean(ood_arr) - np.mean(id_arr)) / pooled)
    return {"auroc": float(auroc), "d_prime": d,
            "id_mean": float(np.mean(id_arr)), "id_std": float(np.std(id_arr)),
            "ood_mean": float(np.mean(ood_arr)), "ood_std": float(np.std(ood_arr))}

def main():
    print("=" * 60)
    print("Experiment 145: Mahalanobis vs Cosine Distance")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    creators = [create_highway, create_urban, create_rural]
    n_cal, n_test_id = 12, 10
    layers = [3, 32]

    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test_id)]

    ood_categories = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night, "blur": apply_blur,
        "noise": apply_noise_corruption, "occlusion": apply_occlusion,
        "snow": apply_snow, "overexpose": apply_overexpose,
    }

    # Calibration
    print(f"\n--- Calibration (n={n_cal}) ---", flush=True)
    cal_embs = {l: [] for l in layers}
    for i, arr in enumerate(cal_arrs):
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            cal_embs[l].append(h[l])
        print(f"  Cal {i+1}/{n_cal}", flush=True)

    # Stats
    stats = {}
    for l in layers:
        embs = np.array(cal_embs[l])
        centroid = embs.mean(axis=0)
        centered = embs - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(n_cal - 1, len(S))
        stats[l] = {"centroid": centroid, "S_k": S[:k], "Vt_k": Vt[:k], "n_cal": n_cal}

    def mahal(x, l):
        diff = x - stats[l]["centroid"]
        proj = stats[l]["Vt_k"] @ diff
        scaled = proj / (stats[l]["S_k"] / np.sqrt(stats[l]["n_cal"]-1) + 1e-6)
        return float(np.sqrt(np.sum(scaled**2)))

    # Test ID
    print(f"\n--- Test ID (n={n_test_id}) ---", flush=True)
    id_res = {l: {"cosine": [], "mahalanobis": []} for l in layers}
    for i, arr in enumerate(test_arrs):
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_res[l]["cosine"].append(cosine_distance(h[l], stats[l]["centroid"]))
            id_res[l]["mahalanobis"].append(mahal(h[l], l))
        print(f"  ID {i+1}/{n_test_id}", flush=True)

    # OOD
    print(f"\n--- OOD ({len(ood_categories)} cats) ---", flush=True)
    n_ood = 6
    ood_res = {l: {c: {"cosine": [], "mahalanobis": []} for c in ood_categories} for l in layers}
    for cat, tfn in ood_categories.items():
        for j in range(n_ood):
            arr = tfn(test_arrs[j % n_test_id])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_res[l][cat]["cosine"].append(cosine_distance(h[l], stats[l]["centroid"]))
                ood_res[l][cat]["mahalanobis"].append(mahal(h[l], l))
        print(f"  {cat}: done", flush=True)

    # Metrics
    cats = list(ood_categories.keys())
    results = {}
    for l in layers:
        ln = f"L{l}"
        results[ln] = {}
        for m in ["cosine", "mahalanobis"]:
            all_ood = []
            per_cat = {}
            for c in cats:
                cd = ood_res[l][c][m]
                all_ood.extend(cd)
                per_cat[c] = compute_metrics(id_res[l][m], cd)
            results[ln][m] = {"overall": compute_metrics(id_res[l][m], all_ood), "per_category": per_cat}

    # Print
    print("\n" + "=" * 80)
    for ln in ["L3", "L32"]:
        print(f"\n--- {ln} ---")
        for m in ["cosine", "mahalanobis"]:
            o = results[ln][m]["overall"]
            print(f"  {m:15s}: AUROC={o['auroc']:.4f}  d'={o['d_prime']:.1f}")
            for c in cats:
                cr = results[ln][m]["per_category"][c]
                print(f"    {c:15s}: AUROC={cr['auroc']:.4f}  d'={cr['d_prime']:.1f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"experiment": "mahalanobis_vs_cosine", "experiment_number": 145, "timestamp": ts,
           "n_cal": n_cal, "n_test_id": n_test_id, "n_ood_per_cat": n_ood,
           "ood_categories": cats, "layers": layers, "results": results}
    path = os.path.join(RESULTS_DIR, f"mahalanobis_vs_cosine_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
