#!/usr/bin/env python3
"""Experiment 355: Calibration Set Size Analysis

How many calibration images are needed?
1. Detection with 1, 2, 5, 10, 20 calibration images
2. Centroid stability vs calibration set size
3. Random vs diverse calibration strategies
4. One-shot vs multi-shot calibration comparison
5. Calibration efficiency: performance per image
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate 30 scenes for calibration pool and 10 for testing
    print("Generating scenes...")
    cal_seeds = list(range(0, 3000, 100))[:30]
    test_seeds = list(range(5000, 6000, 100))[:10]

    cal_images = {}
    cal_embs = {}
    for seed in cal_seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        cal_images[seed] = Image.fromarray(px)
        cal_embs[seed] = extract_hidden(model, processor, cal_images[seed], prompt)

    test_images = {}
    test_embs = {}
    for seed in test_seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        test_images[seed] = Image.fromarray(px)
        test_embs[seed] = extract_hidden(model, processor, test_images[seed], prompt)

    # ========== 1. Per-scene calibration (one-shot) ==========
    print("\n=== One-Shot Per-Scene Calibration ===")

    oneshot_results = {}
    for test_seed in test_seeds:
        cal = test_embs[test_seed]

        emb = extract_hidden(model, processor, test_images[test_seed], prompt)
        id_dist = float(cosine_dist(cal, emb))

        per_type = {}
        for ct in ctypes:
            img = apply_corruption(test_images[test_seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            ood_dist = float(cosine_dist(cal, emb))
            per_type[ct] = {'distance': ood_dist, 'detected': ood_dist > id_dist}

        oneshot_results[str(test_seed)] = {
            'id_dist': id_dist,
            'per_type': per_type,
        }

    all_detected_oneshot = all(
        oneshot_results[str(s)]['per_type'][ct]['detected']
        for s in test_seeds for ct in ctypes
    )
    results['oneshot'] = {
        'per_scene': oneshot_results,
        'all_detected': all_detected_oneshot,
    }
    print(f"  One-shot all detected: {all_detected_oneshot}")

    # ========== 2. Centroid calibration with varying set sizes ==========
    print("\n=== Centroid Calibration vs Set Size ===")

    cal_size_results = {}
    rng = np.random.RandomState(42)

    for n_cal in [1, 2, 3, 5, 10, 15, 20, 30]:
        n_trials = 10 if n_cal < 30 else 1
        trial_aurocs = {ct: [] for ct in ctypes}

        for trial in range(n_trials):
            if n_cal >= len(cal_seeds):
                subset = cal_seeds
            else:
                subset = rng.choice(cal_seeds, n_cal, replace=False).tolist()

            centroid = np.mean([cal_embs[s] for s in subset], axis=0)

            for ct in ctypes:
                id_dists = []
                ood_dists = []
                for test_seed in test_seeds:
                    id_dists.append(float(cosine_dist(centroid, test_embs[test_seed])))
                    img = apply_corruption(test_images[test_seed], ct, 0.5)
                    emb = extract_hidden(model, processor, img, prompt)
                    ood_dists.append(float(cosine_dist(centroid, emb)))

                auroc = compute_auroc(id_dists, ood_dists)
                trial_aurocs[ct].append(float(auroc))

        per_type = {}
        for ct in ctypes:
            per_type[ct] = {
                'mean_auroc': float(np.mean(trial_aurocs[ct])),
                'std_auroc': float(np.std(trial_aurocs[ct])),
                'min_auroc': float(min(trial_aurocs[ct])),
                'all_perfect': all(a == 1.0 for a in trial_aurocs[ct]),
            }

        cal_size_results[str(n_cal)] = {
            'n_trials': n_trials,
            'per_type': per_type,
        }
        auroc_str = ', '.join(ct + '=' + format(per_type[ct]['mean_auroc'], '.3f') for ct in ctypes)
        print(f"  n_cal={n_cal}: {auroc_str}")

    results['calibration_size'] = cal_size_results

    # ========== 3. Centroid stability ==========
    print("\n=== Centroid Stability ===")

    stability = {}
    for n_cal in [1, 2, 5, 10, 20]:
        centroids = []
        for trial in range(20):
            if n_cal >= len(cal_seeds):
                subset = cal_seeds
            else:
                subset = rng.choice(cal_seeds, n_cal, replace=False).tolist()
            centroid = np.mean([cal_embs[s] for s in subset], axis=0)
            centroids.append(centroid)

        pairwise = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                pairwise.append(float(cosine_dist(centroids[i], centroids[j])))

        stability[str(n_cal)] = {
            'mean_centroid_dist': float(np.mean(pairwise)),
            'max_centroid_dist': float(max(pairwise)),
            'std_centroid_dist': float(np.std(pairwise)),
        }
        print(f"  n_cal={n_cal}: centroid var={np.mean(pairwise):.6f} (max={max(pairwise):.6f})")

    results['stability'] = stability

    # ========== 4. Nearest-centroid vs single-centroid ==========
    print("\n=== Nearest vs Single Centroid ===")

    global_centroid = np.mean([cal_embs[s] for s in cal_seeds], axis=0)

    comparison = {}
    for test_seed in test_seeds:
        nearest_seed = min(cal_seeds, key=lambda s: cosine_dist(cal_embs[s], test_embs[test_seed]))
        nearest_cal = cal_embs[nearest_seed]

        for ct in ctypes:
            img = apply_corruption(test_images[test_seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)

            d_nearest = float(cosine_dist(nearest_cal, emb))
            d_global = float(cosine_dist(global_centroid, emb))
            d_self = float(cosine_dist(test_embs[test_seed], emb))

            key = str(test_seed) + '_' + ct
            comparison[key] = {
                'nearest': d_nearest,
                'global_centroid': d_global,
                'self_cal': d_self,
                'best_method': 'self' if d_self >= d_nearest and d_self >= d_global
                    else ('nearest' if d_nearest >= d_global else 'global'),
            }

    best_counts = {}
    for v in comparison.values():
        m = v['best_method']
        best_counts[m] = best_counts.get(m, 0) + 1

    results['nearest_vs_centroid'] = {
        'comparisons': comparison,
        'best_counts': best_counts,
    }
    print(f"  Best method counts: {best_counts}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/calibration_size_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
