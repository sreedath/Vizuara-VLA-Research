#!/usr/bin/env python3
"""Experiment 171: Scene diversity stress test.

Tests OOD detection when calibration and test use maximally diverse scenes
(6 scene types instead of 3) to validate that the detector generalizes
beyond the original highway/urban/rural set.
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

# New scene types for diversity
def create_parking(idx):
    """Parking lot: gray ground with white line markings."""
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//4] = [160, 180, 200]  # overcast sky
    img[SIZE[0]//4:] = [100, 100, 100]  # concrete
    # Parking lines
    for x in range(0, SIZE[1], 40):
        img[SIZE[0]//2:, x:x+2] = [255, 255, 255]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_tunnel(idx):
    """Tunnel: dark walls with overhead lighting."""
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [40, 40, 45]  # dark concrete
    img[SIZE[0]//2:] = [70, 70, 70]  # road
    # Overhead lights
    for x in range(SIZE[1]//6, SIZE[1], SIZE[1]//3):
        img[10:20, x-5:x+5] = [255, 240, 200]
    return np.clip(img.astype(np.int16) + rng.randint(-3, 4, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_desert(idx):
    """Desert road: sandy terrain, blue sky."""
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [80, 140, 220]  # blue sky
    img[SIZE[0]//3:SIZE[0]*2//3] = [210, 180, 140]  # sand
    img[SIZE[0]*2//3:] = [90, 85, 75]  # dark road
    return np.clip(img.astype(np.int16) + rng.randint(-8, 9, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def apply_fog(a, alpha):
    return np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8)
def apply_night(a): return np.clip(a*0.15, 0, 255).astype(np.uint8)
def apply_blur(a, r=8): return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=r)))
def apply_noise(a, s=50): return np.clip(a.astype(np.float32)+np.random.normal(0,s,a.shape), 0, 255).astype(np.uint8)

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
    print("Experiment 171: Scene Diversity Stress Test")
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

    # Scene sets to compare
    scene_sets = {
        "3_scenes": [create_highway, create_urban, create_rural],
        "6_scenes": [create_highway, create_urban, create_rural, create_parking, create_tunnel, create_desert],
    }

    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }

    n_per_scene = 3  # images per scene type
    n_test_per = 2

    results = {}
    for set_name, creators in scene_sets.items():
        print(f"\n=== Scene set: {set_name} ({len(creators)} types) ===", flush=True)
        n_scenes = len(creators)

        # Calibrate
        print("  Calibrating...", flush=True)
        cal_embs = {l: [] for l in layers}
        for i in range(n_per_scene * n_scenes):
            arr = creators[i % n_scenes](i)
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                cal_embs[l].append(h[l])

        centroids = {}
        cal_stats = {}
        for l in layers:
            c = np.array(cal_embs[l]).mean(axis=0)
            dists = [cosine_distance(e, c) for e in cal_embs[l]]
            centroids[l] = c
            cal_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists)),
                           "max": float(np.max(dists)), "min": float(np.min(dists))}
        print(f"  Cal stats: L3 mean={cal_stats[3]['mean']:.6f} std={cal_stats[3]['std']:.6f}", flush=True)
        print(f"             L32 mean={cal_stats[32]['mean']:.6f} std={cal_stats[32]['std']:.6f}", flush=True)

        # Test ID
        id_embs = {l: [] for l in layers}
        for i in range(n_test_per * n_scenes):
            arr = creators[i % n_scenes](100 + i)
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                id_embs[l].append(h[l])

        # Test OOD
        ood_embs = {l: {c: [] for c in ood_transforms} for l in layers}
        for cat, tfn in ood_transforms.items():
            for i in range(n_test_per * n_scenes):
                arr = tfn(creators[i % n_scenes](200 + i))
                h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
                for l in layers:
                    ood_embs[l][cat].append(h[l])

        # Compute metrics
        set_results = {"cal_stats": cal_stats, "n_cal": n_per_scene * n_scenes,
                       "n_test_id": n_test_per * n_scenes, "n_scenes": n_scenes}
        for l in layers:
            id_dists = [cosine_distance(e, centroids[l]) for e in id_embs[l]]
            all_ood_dists = []
            per_cat = {}
            for cat in ood_transforms:
                cat_dists = [cosine_distance(e, centroids[l]) for e in ood_embs[l][cat]]
                auroc = compute_auroc(id_dists, cat_dists)
                per_cat[cat] = {"auroc": auroc, "mean": float(np.mean(cat_dists))}
                all_ood_dists.extend(cat_dists)
                print(f"  L{l} {cat}: AUROC={auroc:.4f}", flush=True)

            overall_auroc = compute_auroc(id_dists, all_ood_dists)
            set_results[f"L{l}"] = {
                "overall_auroc": overall_auroc,
                "id_mean": float(np.mean(id_dists)),
                "id_std": float(np.std(id_dists)),
                "per_category": per_cat,
            }
            print(f"  L{l} overall: AUROC={overall_auroc:.4f}", flush=True)

        # OR-gate
        sigma = 3.0
        thresh = {l: cal_stats[l]["mean"] + sigma * cal_stats[l]["std"] for l in layers}
        n_id_total = n_test_per * n_scenes
        id_flagged = sum(1 for i in range(n_id_total)
                        if cosine_distance(id_embs[3][i], centroids[3]) > thresh[3]
                        or cosine_distance(id_embs[32][i], centroids[32]) > thresh[32])

        total_ood_flagged = 0
        total_ood = 0
        or_per_cat = {}
        for cat in ood_transforms:
            n_cat = n_test_per * n_scenes
            flagged = sum(1 for i in range(n_cat)
                         if cosine_distance(ood_embs[3][cat][i], centroids[3]) > thresh[3]
                         or cosine_distance(ood_embs[32][cat][i], centroids[32]) > thresh[32])
            or_per_cat[cat] = flagged / n_cat
            total_ood_flagged += flagged
            total_ood += n_cat

        recall = total_ood_flagged / total_ood
        fpr = id_flagged / n_id_total
        prec = total_ood_flagged / (total_ood_flagged + id_flagged) if (total_ood_flagged + id_flagged) > 0 else 1.0
        f1 = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else 0.0

        set_results["or_gate"] = {
            "fpr": float(fpr), "recall": float(recall), "precision": float(prec), "f1": float(f1),
            "per_category_recall": {c: float(v) for c, v in or_per_cat.items()},
        }
        print(f"  OR-gate: recall={recall:.3f} FPR={fpr:.3f} F1={f1:.3f}", flush=True)

        results[set_name] = set_results

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON: 3-scene vs 6-scene calibration")
    for metric in ["overall_auroc"]:
        for l in layers:
            v3 = results["3_scenes"][f"L{l}"][metric]
            v6 = results["6_scenes"][f"L{l}"][metric]
            print(f"  L{l} {metric}: 3-scene={v3:.4f}, 6-scene={v6:.4f}, delta={v6-v3:+.4f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "scene_diversity",
        "experiment_number": 171,
        "timestamp": ts,
        "n_per_scene_cal": n_per_scene,
        "n_per_scene_test": n_test_per,
        "ood_categories": list(ood_transforms.keys()),
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"scene_diversity_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
