#!/usr/bin/env python3
"""Experiment 172: Multi-centroid scene-aware OOD detection.

After experiment 171 showed that diverse scenes dilute a single centroid,
this tests nearest-centroid routing: one centroid per scene type,
distance computed to the closest centroid.
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

def create_parking(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//4] = [160, 180, 200]
    img[SIZE[0]//4:] = [100, 100, 100]
    for x in range(0, SIZE[1], 40):
        img[SIZE[0]//2:, x:x+2] = [255, 255, 255]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_tunnel(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [40, 40, 45]
    img[SIZE[0]//2:] = [70, 70, 70]
    for x in range(SIZE[1]//6, SIZE[1], SIZE[1]//3):
        img[10:20, x-5:x+5] = [255, 240, 200]
    return np.clip(img.astype(np.int16) + rng.randint(-3, 4, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_desert(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [80, 140, 220]
    img[SIZE[0]//3:SIZE[0]*2//3] = [210, 180, 140]
    img[SIZE[0]*2//3:] = [90, 85, 75]
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
    print("Experiment 172: Multi-Centroid Scene-Aware Detection")
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

    all_creators = {
        "highway": create_highway,
        "urban": create_urban,
        "rural": create_rural,
        "parking": create_parking,
        "tunnel": create_tunnel,
        "desert": create_desert,
    }

    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }

    n_cal_per = 3
    n_test_per = 2

    # Build per-scene centroids
    print("\n--- Building per-scene centroids ---", flush=True)
    per_scene_embs = {scene: {l: [] for l in layers} for scene in all_creators}
    for scene_name, creator in all_creators.items():
        for i in range(n_cal_per):
            arr = creator(i)
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                per_scene_embs[scene_name][l].append(h[l])
        for l in layers:
            print(f"  {scene_name} L{l}: {len(per_scene_embs[scene_name][l])} cal embeddings", flush=True)

    per_scene_centroids = {}
    per_scene_stats = {}
    for scene_name in all_creators:
        per_scene_centroids[scene_name] = {}
        per_scene_stats[scene_name] = {}
        for l in layers:
            embs = np.array(per_scene_embs[scene_name][l])
            c = embs.mean(axis=0)
            dists = [cosine_distance(e, c) for e in embs]
            per_scene_centroids[scene_name][l] = c
            per_scene_stats[scene_name][l] = {
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
            }

    # Global centroid (for comparison)
    global_centroids = {}
    global_stats = {}
    for l in layers:
        all_embs = []
        for scene_name in all_creators:
            all_embs.extend(per_scene_embs[scene_name][l])
        all_embs = np.array(all_embs)
        c = all_embs.mean(axis=0)
        dists = [cosine_distance(e, c) for e in all_embs]
        global_centroids[l] = c
        global_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}

    # Test
    print("\n--- Testing ---", flush=True)
    id_test_embs = {l: [] for l in layers}
    id_test_scenes = []
    for scene_name, creator in all_creators.items():
        for i in range(n_test_per):
            arr = creator(100 + i)
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                id_test_embs[l].append(h[l])
            id_test_scenes.append(scene_name)

    ood_test_embs = {l: [] for l in layers}
    ood_test_cats = []
    for cat, tfn in ood_transforms.items():
        for scene_name, creator in all_creators.items():
            for i in range(n_test_per):
                arr = tfn(creator(200 + i))
                h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
                for l in layers:
                    ood_test_embs[l].append(h[l])
                ood_test_cats.append(cat)

    # Three detection strategies
    strategies = {}

    # Strategy 1: Global single centroid
    for l in layers:
        id_dists = [cosine_distance(e, global_centroids[l]) for e in id_test_embs[l]]
        ood_dists = [cosine_distance(e, global_centroids[l]) for e in ood_test_embs[l]]
        auroc = compute_auroc(id_dists, ood_dists)
        strategies.setdefault("global_centroid", {})[f"L{l}"] = {
            "auroc": auroc, "id_mean": float(np.mean(id_dists)), "ood_mean": float(np.mean(ood_dists)),
        }
        print(f"  Global centroid L{l}: AUROC={auroc:.4f}", flush=True)

    # Strategy 2: Nearest centroid (min distance to any scene centroid)
    for l in layers:
        id_dists_nc = []
        for e in id_test_embs[l]:
            min_dist = min(cosine_distance(e, per_scene_centroids[s][l]) for s in all_creators)
            id_dists_nc.append(min_dist)
        ood_dists_nc = []
        for e in ood_test_embs[l]:
            min_dist = min(cosine_distance(e, per_scene_centroids[s][l]) for s in all_creators)
            ood_dists_nc.append(min_dist)
        auroc = compute_auroc(id_dists_nc, ood_dists_nc)
        strategies.setdefault("nearest_centroid", {})[f"L{l}"] = {
            "auroc": auroc, "id_mean": float(np.mean(id_dists_nc)), "ood_mean": float(np.mean(ood_dists_nc)),
        }
        print(f"  Nearest centroid L{l}: AUROC={auroc:.4f}", flush=True)

    # Strategy 3: Oracle centroid (correct scene label known)
    for l in layers:
        id_dists_oracle = []
        for idx, e in enumerate(id_test_embs[l]):
            scene = id_test_scenes[idx]
            d = cosine_distance(e, per_scene_centroids[scene][l])
            id_dists_oracle.append(d)
        # For OOD, use the scene the OOD was derived from
        ood_dists_oracle = []
        ood_idx = 0
        for cat in ood_transforms:
            for scene_name in all_creators:
                for i in range(n_test_per):
                    e = ood_test_embs[l][ood_idx]
                    d = cosine_distance(e, per_scene_centroids[scene_name][l])
                    ood_dists_oracle.append(d)
                    ood_idx += 1
        auroc = compute_auroc(id_dists_oracle, ood_dists_oracle)
        strategies.setdefault("oracle_centroid", {})[f"L{l}"] = {
            "auroc": auroc, "id_mean": float(np.mean(id_dists_oracle)), "ood_mean": float(np.mean(ood_dists_oracle)),
        }
        print(f"  Oracle centroid L{l}: AUROC={auroc:.4f}", flush=True)

    # OR-gate for each strategy
    sigma = 3.0
    for strat_name in ["global_centroid", "nearest_centroid"]:
        if strat_name == "global_centroid":
            def get_dist(emb, l):
                return cosine_distance(emb, global_centroids[l])
            stats = global_stats
        else:
            def get_dist(emb, l):
                return min(cosine_distance(emb, per_scene_centroids[s][l]) for s in all_creators)
            # Use per-scene average stats
            stats = {}
            for l in layers:
                all_means = [per_scene_stats[s][l]["mean"] for s in all_creators]
                all_stds = [per_scene_stats[s][l]["std"] for s in all_creators]
                stats[l] = {"mean": float(np.mean(all_means)), "std": float(np.mean(all_stds))}

        thresh = {l: stats[l]["mean"] + sigma * stats[l]["std"] for l in layers}
        n_id = len(id_test_embs[layers[0]])
        n_ood = len(ood_test_embs[layers[0]])

        id_flagged = sum(1 for i in range(n_id)
                        if get_dist(id_test_embs[3][i], 3) > thresh[3]
                        or get_dist(id_test_embs[32][i], 32) > thresh[32])
        ood_flagged = sum(1 for i in range(n_ood)
                         if get_dist(ood_test_embs[3][i], 3) > thresh[3]
                         or get_dist(ood_test_embs[32][i], 32) > thresh[32])

        recall = ood_flagged / n_ood
        fpr = id_flagged / n_id
        prec = ood_flagged / (ood_flagged + id_flagged) if (ood_flagged + id_flagged) > 0 else 1.0
        f1 = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else 0.0
        strategies[strat_name]["or_gate"] = {
            "recall": float(recall), "fpr": float(fpr), "precision": float(prec), "f1": float(f1),
        }
        print(f"  {strat_name} OR-gate: recall={recall:.3f} FPR={fpr:.3f} F1={f1:.3f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "multi_centroid_scenes",
        "experiment_number": 172,
        "timestamp": ts,
        "n_cal_per_scene": n_cal_per,
        "n_test_per_scene": n_test_per,
        "n_scenes": len(all_creators),
        "scene_names": list(all_creators.keys()),
        "ood_categories": list(ood_transforms.keys()),
        "layers": layers,
        "strategies": strategies,
    }
    path = os.path.join(RESULTS_DIR, f"multi_centroid_scenes_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
