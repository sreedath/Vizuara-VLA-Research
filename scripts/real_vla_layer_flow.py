#!/usr/bin/env python3
"""Experiment 159: Layer-wise OOD information flow.

Tracks how OOD signal (cosine distance from ID centroid) propagates through
ALL 33 layers of the model (layers 0-32). Identifies where OOD information
is amplified, suppressed, or transformed.
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

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 159: Layer-wise OOD Information Flow")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    all_layers = list(range(33))  # layers 0-32

    creators = [create_highway, create_urban, create_rural]
    n_cal = 8

    # Extract ALL layer embeddings for calibration
    print("\n--- Calibrating (all 33 layers) ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    cal_embs = {l: [] for l in all_layers}

    for i, arr in enumerate(cal_arrs):
        inputs = processor(prompt, Image.fromarray(arr)).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        for l in all_layers:
            cal_embs[l].append(fwd.hidden_states[l][0, -1, :].float().cpu().numpy())
        if i % 2 == 0:
            print(f"  Cal image {i}/{n_cal}", flush=True)

    centroids = {}
    cal_stats = {}
    for l in all_layers:
        embs = np.array(cal_embs[l])
        centroid = embs.mean(axis=0)
        dists = [cosine_distance(e, centroid) for e in embs]
        centroids[l] = centroid
        cal_stats[l] = {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "max": float(np.max(dists)),
            "norm": float(np.linalg.norm(centroid)),
        }

    # OOD corruptions
    ood_types = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
    }
    n_test = 4

    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    # ID test distances
    print("\n--- ID test (all layers) ---", flush=True)
    id_layer_dists = {l: [] for l in all_layers}
    for arr in test_arrs:
        inputs = processor(prompt, Image.fromarray(arr)).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        for l in all_layers:
            emb = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()
            id_layer_dists[l].append(cosine_distance(emb, centroids[l]))

    # OOD distances per layer per category
    ood_layer_dists = {cat: {l: [] for l in all_layers} for cat in ood_types}
    for cat, tfn in ood_types.items():
        print(f"\n--- OOD: {cat} ---", flush=True)
        for j in range(n_test):
            arr = tfn(test_arrs[j % n_test])
            inputs = processor(prompt, Image.fromarray(arr)).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            for l in all_layers:
                emb = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()
                ood_layer_dists[cat][l].append(cosine_distance(emb, centroids[l]))

    # Compute per-layer d' and AUROC
    results = {
        "cal_stats_per_layer": {str(l): cal_stats[l] for l in all_layers},
        "id_mean_per_layer": {},
        "ood_per_category": {},
    }

    for l in all_layers:
        results["id_mean_per_layer"][str(l)] = float(np.mean(id_layer_dists[l]))

    for cat in ood_types:
        cat_results = {}
        for l in all_layers:
            id_m = np.mean(id_layer_dists[l])
            id_s = np.std(id_layer_dists[l])
            ood_m = np.mean(ood_layer_dists[cat][l])
            ood_s = np.std(ood_layer_dists[cat][l])
            pooled_std = np.sqrt((id_s**2 + ood_s**2) / 2 + 1e-15)
            d_prime = (ood_m - id_m) / pooled_std if pooled_std > 1e-10 else 0.0
            separation = (ood_m - id_m) / (id_m + 1e-10)
            cat_results[str(l)] = {
                "ood_mean": float(ood_m),
                "d_prime": float(d_prime),
                "separation_ratio": float(separation),
            }
        results["ood_per_category"][cat] = cat_results

    # Print summary
    print("\n" + "=" * 80)
    print("LAYER-WISE OOD SIGNAL (d' for each corruption)")
    header = f"{'Layer':>6}"
    for cat in ood_types:
        header += f" {cat[:8]:>10}"
    print(header)
    for l in all_layers:
        row = f"{l:>6}"
        for cat in ood_types:
            dp = results["ood_per_category"][cat][str(l)]["d_prime"]
            row += f" {dp:10.2f}"
        print(row)

    # Find peak layers per category
    print("\nPEAK LAYERS:")
    for cat in ood_types:
        dprimes = [(l, results["ood_per_category"][cat][str(l)]["d_prime"]) for l in all_layers]
        peak_l, peak_dp = max(dprimes, key=lambda x: x[1])
        print(f"  {cat}: peak at L{peak_l} (d'={peak_dp:.2f})")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "layer_flow",
        "experiment_number": 159,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "n_layers": 33,
        "ood_categories": list(ood_types.keys()),
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"layer_flow_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
