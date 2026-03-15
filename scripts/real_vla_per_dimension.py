#!/usr/bin/env python3
"""Experiment 197: Per-dimension OOD signal analysis — which embedding dimensions
carry the OOD signal? Computes per-dimension effect sizes and identifies
the most informative dimensions.
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

def main():
    print("=" * 60)
    print("Experiment 197: Per-Dimension OOD Signal Analysis")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [1, 3]

    creators = [create_highway, create_urban, create_rural]
    n_cal = 10
    n_test = 8

    def extract_all(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    # Calibration
    print("\n--- Calibration ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            cal_embs[l].append(h[l])

    # ID test
    print("--- ID test ---", flush=True)
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_embs = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            id_embs[l].append(h[l])

    # OOD test
    print("--- OOD test ---", flush=True)
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_embs = {l: [] for l in layers}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h = extract_all(Image.fromarray(tfn(arr)))
            for l in layers:
                ood_embs[l].append(h[l])

    # Per-dimension analysis
    print("\n--- Per-dimension analysis ---", flush=True)
    results = {}
    for l in layers:
        id_arr = np.array(id_embs[l])   # [n_test, 4096]
        ood_arr = np.array(ood_embs[l])  # [n_ood, 4096]
        cal_arr = np.array(cal_embs[l])  # [n_cal, 4096]

        n_dims = id_arr.shape[1]

        # Per-dimension Cohen's d
        id_mean = id_arr.mean(axis=0)
        ood_mean = ood_arr.mean(axis=0)
        id_std = id_arr.std(axis=0) + 1e-10
        ood_std = ood_arr.std(axis=0) + 1e-10
        pooled_std = np.sqrt((id_std**2 + ood_std**2) / 2)
        cohens_d = np.abs(ood_mean - id_mean) / pooled_std

        # Per-dimension AUROC (for top dimensions)
        top_dims = np.argsort(cohens_d)[-50:][::-1]  # Top 50 by Cohen's d
        dim_aurocs = []
        for d in top_dims:
            auroc = compute_auroc(np.abs(id_arr[:, d] - cal_arr.mean(axis=0)[d]).tolist(),
                                  np.abs(ood_arr[:, d] - cal_arr.mean(axis=0)[d]).tolist())
            dim_aurocs.append({"dim": int(d), "cohens_d": float(cohens_d[d]), "auroc": auroc})

        # Signal concentration
        sorted_d = np.sort(cohens_d)[::-1]
        cumulative = np.cumsum(sorted_d) / np.sum(sorted_d)
        # How many dims for 50%, 80%, 90% of total effect
        n50 = int(np.searchsorted(cumulative, 0.5)) + 1
        n80 = int(np.searchsorted(cumulative, 0.8)) + 1
        n90 = int(np.searchsorted(cumulative, 0.9)) + 1

        # Random subsets: AUROC with k random dimensions vs k top dimensions
        subset_results = []
        centroid = cal_arr.mean(axis=0)
        for k in [10, 50, 100, 500, 1000, 2048]:
            # Top-k dimensions
            top_k = np.argsort(cohens_d)[-k:]
            id_top = [cosine_distance(e[top_k], centroid[top_k]) for e in id_embs[l]]
            ood_top = [cosine_distance(e[top_k], centroid[top_k]) for e in ood_embs[l]]
            auroc_top = compute_auroc(id_top, ood_top)

            # Random-k dimensions
            rng_local = np.random.RandomState(42)
            rand_k = rng_local.choice(n_dims, k, replace=False)
            id_rand = [cosine_distance(e[rand_k], centroid[rand_k]) for e in id_embs[l]]
            ood_rand = [cosine_distance(e[rand_k], centroid[rand_k]) for e in ood_embs[l]]
            auroc_rand = compute_auroc(id_rand, ood_rand)

            # Bottom-k (worst) dimensions
            bot_k = np.argsort(cohens_d)[:k]
            id_bot = [cosine_distance(e[bot_k], centroid[bot_k]) for e in id_embs[l]]
            ood_bot = [cosine_distance(e[bot_k], centroid[bot_k]) for e in ood_embs[l]]
            auroc_bot = compute_auroc(id_bot, ood_bot)

            subset_results.append({
                "k": k, "auroc_top": auroc_top, "auroc_random": auroc_rand, "auroc_bottom": auroc_bot
            })
            print(f"  L{l} k={k}: top={auroc_top:.3f} rand={auroc_rand:.3f} bot={auroc_bot:.3f}", flush=True)

        layer_results = {
            "mean_cohens_d": float(np.mean(cohens_d)),
            "max_cohens_d": float(np.max(cohens_d)),
            "median_cohens_d": float(np.median(cohens_d)),
            "n_significant": int(np.sum(cohens_d > 0.8)),  # convention: d>0.8 is large
            "n50_pct": n50,
            "n80_pct": n80,
            "n90_pct": n90,
            "top_dimensions": dim_aurocs[:20],
            "subset_aurocs": subset_results,
        }
        results[f"L{l}"] = layer_results
        print(f"\n  L{l}: mean_d={np.mean(cohens_d):.4f} max_d={np.max(cohens_d):.4f} n_sig={np.sum(cohens_d>0.8)}", flush=True)
        print(f"        50%={n50} dims, 80%={n80} dims, 90%={n90} dims (of {n_dims})", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "per_dimension_analysis",
        "experiment_number": 197,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "n_dims": 4096,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"per_dimension_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
