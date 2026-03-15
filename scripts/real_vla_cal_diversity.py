#!/usr/bin/env python3
"""Experiment 152: Calibration set diversity vs quantity.

Tests whether diverse calibration images (highway + urban + rural) perform
better than homogeneous ones (highway only) at the same sample size.
Also tests interpolation: how much diversity is needed?
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
def apply_snow(a):
    o=a.astype(np.float32)*0.7+76.5; o[np.random.random(a.shape[:2])>0.97]=255
    return np.clip(o,0,255).astype(np.uint8)

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_d, ood_d):
    correct = sum(1 for o in ood_d for i in id_d if o > i)
    correct += 0.5 * sum(1 for o in ood_d for i in id_d if o == i)
    return correct / (len(id_d) * len(ood_d)) if len(id_d) * len(ood_d) > 0 else 0.5

def main():
    print("=" * 60)
    print("Experiment 152: Calibration Diversity vs Quantity")
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

    # Pre-generate pool of images
    n_per_type = 12
    highway_pool = [create_highway(i) for i in range(n_per_type)]
    urban_pool = [create_urban(i + 100) for i in range(n_per_type)]
    rural_pool = [create_rural(i + 200) for i in range(n_per_type)]

    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
        "snow": apply_snow,
    }

    # Pre-compute ALL embeddings
    print("\n--- Computing all embeddings ---", flush=True)
    all_pools = {"highway": highway_pool, "urban": urban_pool, "rural": rural_pool}
    embeddings = {}
    for pool_name, pool in all_pools.items():
        embeddings[pool_name] = []
        for i, arr in enumerate(pool):
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            embeddings[pool_name].append(h)
        print(f"  {pool_name}: {len(pool)} images", flush=True)

    # Test set: diverse mix (not used in calibration)
    n_test = 8
    test_embs = []
    for i in range(n_test):
        pool_name = ["highway", "urban", "rural"][i % 3]
        idx = n_per_type // 2 + i // 3  # Use second half of pool
        test_embs.append(embeddings[pool_name][idx])
    print(f"  Test ID: {n_test} images (diverse)", flush=True)

    # OOD embeddings
    n_ood_per = 5
    ood_embs = {c: [] for c in ood_transforms}
    for cat, tfn in ood_transforms.items():
        for j in range(n_ood_per):
            pool_name = ["highway", "urban", "rural"][j % 3]
            idx = n_per_type // 2 + j // 3
            arr = tfn(all_pools[pool_name][idx])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            ood_embs[cat].append(h)
        print(f"  OOD {cat}: {n_ood_per}", flush=True)

    # Calibration strategies
    n_cal = 9  # Total calibration images
    cal_configs = {
        "highway_only": [(embeddings["highway"][i], "highway") for i in range(n_cal)],
        "urban_only": [(embeddings["urban"][i], "urban") for i in range(n_cal)],
        "rural_only": [(embeddings["rural"][i], "rural") for i in range(n_cal)],
        "diverse_3each": ([(embeddings["highway"][i], "highway") for i in range(3)] +
                          [(embeddings["urban"][i], "urban") for i in range(3)] +
                          [(embeddings["rural"][i], "rural") for i in range(3)]),
        "diverse_6hw_3ur": ([(embeddings["highway"][i], "highway") for i in range(6)] +
                            [(embeddings["urban"][i], "urban") for i in range(3)]),
    }

    # Evaluate each config
    print("\n--- Evaluating configs ---", flush=True)
    results = {}
    cats = list(ood_transforms.keys())

    for config_name, cal_set in cal_configs.items():
        config_results = {}
        for l in layers:
            cal_vecs = [item[0][l] for item in cal_set]
            centroid = np.mean(cal_vecs, axis=0)
            cal_dists = [cosine_distance(v, centroid) for v in cal_vecs]

            # Test ID
            id_dists = [cosine_distance(te[l], centroid) for te in test_embs]

            # OOD
            all_ood = []
            per_cat = {}
            for cat in cats:
                ood_dists = [cosine_distance(oe[l], centroid) for oe in ood_embs[cat]]
                all_ood.extend(ood_dists)
                per_cat[cat] = compute_auroc(id_dists, ood_dists)

            overall_auroc = compute_auroc(id_dists, all_ood)
            id_arr, ood_arr = np.array(id_dists), np.array(all_ood)
            pooled = np.sqrt((np.var(id_arr) + np.var(ood_arr)) / 2 + 1e-10)
            d_prime = float((np.mean(ood_arr) - np.mean(id_arr)) / pooled)

            config_results[f"L{l}"] = {
                "overall_auroc": float(overall_auroc),
                "d_prime": d_prime,
                "cal_radius": float(np.max(cal_dists)),
                "id_max": float(np.max(id_dists)),
                "per_category": {c: float(v) for c, v in per_cat.items()},
            }

        results[config_name] = config_results
        print(f"  {config_name:20s}: L3 AUROC={config_results['L3']['overall_auroc']:.4f}  L32 AUROC={config_results['L32']['overall_auroc']:.4f}", flush=True)

    # Summary
    print("\n" + "=" * 80)
    for cfg in cal_configs:
        r3 = results[cfg]["L3"]
        r32 = results[cfg]["L32"]
        print(f"  {cfg:20s}: L3 AUROC={r3['overall_auroc']:.4f} d'={r3['d_prime']:.1f} | L32 AUROC={r32['overall_auroc']:.4f} d'={r32['d_prime']:.1f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {"experiment": "cal_diversity", "experiment_number": 152, "timestamp": ts,
              "n_cal": n_cal, "n_test_id": n_test, "n_ood_per_cat": n_ood_per,
              "ood_categories": cats, "layers": layers,
              "cal_configs": list(cal_configs.keys()),
              "results": results}
    path = os.path.join(RESULTS_DIR, f"cal_diversity_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
