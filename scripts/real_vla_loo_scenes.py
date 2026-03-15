#!/usr/bin/env python3
"""Experiment 177: Leave-one-scene-out cross-validation.

Calibrates on N-1 scene types, tests on held-out scene type.
Measures whether novel scenes are incorrectly flagged as OOD.
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

def apply_fog(a, alpha):
    return np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8)
def apply_night(a): return np.clip(a*0.15, 0, 255).astype(np.uint8)

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
    print("Experiment 177: Leave-One-Scene-Out Cross-Validation")
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

    all_scenes = {
        "highway": create_highway,
        "urban": create_urban,
        "rural": create_rural,
        "parking": create_parking,
        "tunnel": create_tunnel,
    }

    n_cal_per = 3
    n_test_per = 3

    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
    }

    # Pre-extract all embeddings
    print("\n--- Pre-extracting all embeddings ---", flush=True)
    all_embs = {}
    for scene_name, creator in all_scenes.items():
        scene_embs = {l: [] for l in layers}
        for i in range(n_cal_per + n_test_per):
            arr = creator(i)
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                scene_embs[l].append(h[l])
        all_embs[scene_name] = scene_embs
        print(f"  {scene_name}: {n_cal_per + n_test_per} embeddings", flush=True)

    # OOD embeddings
    ood_embs = {}
    for scene_name, creator in all_scenes.items():
        ood_embs[scene_name] = {cat: {l: [] for l in layers} for cat in ood_transforms}
        for cat, tfn in ood_transforms.items():
            for i in range(n_test_per):
                arr = tfn(creator(n_cal_per + i))
                h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
                for l in layers:
                    ood_embs[scene_name][cat][l].append(h[l])

    results = {}
    scene_names = list(all_scenes.keys())

    for held_out in scene_names:
        print(f"\n=== Holding out: {held_out} ===", flush=True)
        train_scenes = [s for s in scene_names if s != held_out]

        centroids = {}
        cal_stats = {}
        for l in layers:
            cal_embs = []
            for s in train_scenes:
                cal_embs.extend(all_embs[s][l][:n_cal_per])
            cal_embs = np.array(cal_embs)
            c = cal_embs.mean(axis=0)
            dists = [cosine_distance(e, c) for e in cal_embs]
            centroids[l] = c
            cal_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}

        novel_dists = {l: [] for l in layers}
        for l in layers:
            for e in all_embs[held_out][l][n_cal_per:]:
                novel_dists[l].append(cosine_distance(e, centroids[l]))

        same_dists = {l: [] for l in layers}
        for l in layers:
            for s in train_scenes:
                for e in all_embs[s][l][n_cal_per:]:
                    same_dists[l].append(cosine_distance(e, centroids[l]))

        all_ood_dists = {l: [] for l in layers}
        for cat in ood_transforms:
            for s in train_scenes:
                for l in layers:
                    all_ood_dists[l].extend([cosine_distance(e, centroids[l]) for e in ood_embs[s][cat][l]])

        fold_results = {}
        for l in layers:
            auroc_novel_vs_ood = compute_auroc(novel_dists[l], all_ood_dists[l])
            auroc_novel_vs_same = compute_auroc(same_dists[l], novel_dists[l])
            fold_results[f"L{l}"] = {
                "novel_mean_dist": float(np.mean(novel_dists[l])),
                "same_mean_dist": float(np.mean(same_dists[l])),
                "ood_mean_dist": float(np.mean(all_ood_dists[l])),
                "auroc_novel_as_id_vs_ood": auroc_novel_vs_ood,
                "auroc_novel_vs_same": auroc_novel_vs_same,
            }
            print(f"  L{l}: novel_dist={np.mean(novel_dists[l]):.6f} same_dist={np.mean(same_dists[l]):.6f} "
                  f"ood_dist={np.mean(all_ood_dists[l]):.6f}", flush=True)
            print(f"       novel_vs_ood={auroc_novel_vs_ood:.4f} novel_vs_same={auroc_novel_vs_same:.4f}", flush=True)

        sigma = 3.0
        thresh = {l: cal_stats[l]["mean"] + sigma * cal_stats[l]["std"] for l in layers}
        novel_flagged = sum(1 for i in range(n_test_per)
                          if novel_dists[3][i] > thresh[3] or novel_dists[32][i] > thresh[32])
        fold_results["fpr_on_novel"] = novel_flagged / n_test_per
        print(f"  OR-gate FPR on novel scene: {novel_flagged}/{n_test_per} = {novel_flagged/n_test_per:.3f}", flush=True)

        results[held_out] = fold_results

    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT SUMMARY")
    for scene in scene_names:
        r = results[scene]
        print(f"  {scene:>10s}: FPR={r['fpr_on_novel']:.3f} "
              f"L3_novel_vs_ood={r['L3']['auroc_novel_as_id_vs_ood']:.4f} "
              f"L32_novel_vs_ood={r['L32']['auroc_novel_as_id_vs_ood']:.4f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "loo_scenes",
        "experiment_number": 177,
        "timestamp": ts,
        "n_cal_per": n_cal_per,
        "n_test_per": n_test_per,
        "scenes": scene_names,
        "ood_categories": list(ood_transforms.keys()),
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"loo_scenes_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
