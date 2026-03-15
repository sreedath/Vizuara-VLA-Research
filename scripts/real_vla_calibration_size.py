#!/usr/bin/env python3
"""Experiment 376: Calibration Set Size Sensitivity

How many clean calibration images are needed for reliable detection?
1. AUROC as function of N (1, 2, 3, 5, 10, 15, 20 calibration images)
2. Threshold stability: how much does threshold vary with N?
3. Leave-one-out cross-validation at each N
4. Bootstrap confidence intervals for threshold
5. Minimum calibration set for guaranteed AUROC
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

    # Generate images (large pool for calibration experiments)
    print("Generating images...")
    all_seeds = list(range(0, 3000, 100))[:30]
    images = {}
    all_embs = {}
    for seed in all_seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        all_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    # Corrupt embeddings (test set: last 10 seeds)
    test_seeds = all_seeds[20:]  # 10 test scenes
    cal_pool = all_seeds[:20]    # 20 calibration pool
    corrupt_embs = {ct: {} for ct in ctypes}
    for ct in ctypes:
        for seed in test_seeds:
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_embs[ct][seed] = extract_hidden(model, processor, corrupt_img, prompt)

    print(f"  {len(cal_pool)} calibration pool, {len(test_seeds)} test scenes")

    # ========== 1. AUROC vs Calibration Set Size ==========
    print("\n=== AUROC vs Calibration Size ===")

    cal_sizes = [1, 2, 3, 5, 7, 10, 15, 20]
    auroc_vs_n = {}

    for n in cal_sizes:
        n_trials = min(20, max(5, 30 // n))
        trial_aurocs = {ct: [] for ct in ctypes}
        trial_thresholds = []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial * 100 + n)
            cal_seeds = list(rng.choice(cal_pool, size=min(n, len(cal_pool)), replace=False))

            centroid = np.mean([all_embs[s] for s in cal_seeds], axis=0)
            cal_dists = [cosine_dist(centroid, all_embs[s]) for s in cal_seeds]
            threshold = max(cal_dists)
            trial_thresholds.append(threshold)

            for ct in ctypes:
                test_clean_dists = [cosine_dist(centroid, all_embs[s]) for s in test_seeds]
                test_corrupt_dists = [cosine_dist(centroid, corrupt_embs[ct][s]) for s in test_seeds]
                auroc = compute_auroc(test_clean_dists, test_corrupt_dists)
                trial_aurocs[ct].append(auroc)

        auroc_vs_n[str(n)] = {
            'n_trials': n_trials,
            'threshold_mean': float(np.mean(trial_thresholds)),
            'threshold_std': float(np.std(trial_thresholds)),
            'threshold_max': float(max(trial_thresholds)),
        }
        for ct in ctypes:
            auroc_vs_n[str(n)][f'{ct}_auroc_mean'] = float(np.mean(trial_aurocs[ct]))
            auroc_vs_n[str(n)][f'{ct}_auroc_min'] = float(min(trial_aurocs[ct]))
            auroc_vs_n[str(n)][f'{ct}_auroc_std'] = float(np.std(trial_aurocs[ct]))

        print(f"  N={n}: thresh={np.mean(trial_thresholds):.6f}+/-{np.std(trial_thresholds):.6f}, "
              f"fog={np.mean(trial_aurocs['fog']):.4f}, noise={np.mean(trial_aurocs['noise']):.4f}")

    results['auroc_vs_n'] = auroc_vs_n

    # ========== 2. Leave-One-Out Cross-Validation ==========
    print("\n=== Leave-One-Out (N=20) ===")

    loo_results = {}
    full_centroid = np.mean([all_embs[s] for s in cal_pool], axis=0)
    full_dists = [cosine_dist(full_centroid, all_embs[s]) for s in cal_pool]
    full_threshold = max(full_dists)

    for ct in ctypes:
        loo_aurocs = []
        loo_thresholds = []
        for leave_out_seed in cal_pool:
            remaining = [s for s in cal_pool if s != leave_out_seed]
            loo_centroid = np.mean([all_embs[s] for s in remaining], axis=0)
            loo_cal_dists = [cosine_dist(loo_centroid, all_embs[s]) for s in remaining]
            loo_threshold = max(loo_cal_dists)
            loo_thresholds.append(loo_threshold)

            test_clean = [cosine_dist(loo_centroid, all_embs[s]) for s in test_seeds]
            test_corrupt = [cosine_dist(loo_centroid, corrupt_embs[ct][s]) for s in test_seeds]
            loo_aurocs.append(compute_auroc(test_clean, test_corrupt))

        loo_results[ct] = {
            'mean_auroc': float(np.mean(loo_aurocs)),
            'min_auroc': float(min(loo_aurocs)),
            'std_auroc': float(np.std(loo_aurocs)),
            'threshold_mean': float(np.mean(loo_thresholds)),
            'threshold_std': float(np.std(loo_thresholds)),
        }
        print(f"  {ct}: LOO AUROC={np.mean(loo_aurocs):.4f}+/-{np.std(loo_aurocs):.4f}")

    results['leave_one_out'] = loo_results

    # ========== 3. Bootstrap Confidence Intervals ==========
    print("\n=== Bootstrap CI (N=10, 1000 resamples) ===")

    bootstrap = {}
    n_boot = 1000
    cal_10 = cal_pool[:10]

    for ct in ctypes:
        boot_aurocs = []
        boot_thresholds = []

        for b in range(n_boot):
            rng = np.random.RandomState(b)
            boot_idx = rng.choice(len(cal_10), size=len(cal_10), replace=True)
            boot_seeds = [cal_10[i] for i in boot_idx]

            boot_centroid = np.mean([all_embs[s] for s in boot_seeds], axis=0)
            boot_cal_dists = [cosine_dist(boot_centroid, all_embs[s]) for s in boot_seeds]
            boot_threshold = max(boot_cal_dists)
            boot_thresholds.append(boot_threshold)

            test_clean = [cosine_dist(boot_centroid, all_embs[s]) for s in test_seeds]
            test_corrupt = [cosine_dist(boot_centroid, corrupt_embs[ct][s]) for s in test_seeds]
            boot_aurocs.append(compute_auroc(test_clean, test_corrupt))

        bootstrap[ct] = {
            'auroc_mean': float(np.mean(boot_aurocs)),
            'auroc_ci_lower': float(np.percentile(boot_aurocs, 2.5)),
            'auroc_ci_upper': float(np.percentile(boot_aurocs, 97.5)),
            'threshold_ci_lower': float(np.percentile(boot_thresholds, 2.5)),
            'threshold_ci_upper': float(np.percentile(boot_thresholds, 97.5)),
        }
        print(f"  {ct}: AUROC={np.mean(boot_aurocs):.4f} "
              f"[{np.percentile(boot_aurocs, 2.5):.4f}, {np.percentile(boot_aurocs, 97.5):.4f}]")

    results['bootstrap'] = bootstrap

    # ========== 4. Centroid Convergence ==========
    print("\n=== Centroid Convergence ===")

    convergence = {}
    full_centroid_20 = np.mean([all_embs[s] for s in cal_pool], axis=0)

    for n in cal_sizes:
        dists_to_full = []
        for trial in range(10):
            rng = np.random.RandomState(trial * 200 + n)
            subset = list(rng.choice(cal_pool, size=min(n, len(cal_pool)), replace=False))
            sub_centroid = np.mean([all_embs[s] for s in subset], axis=0)
            dists_to_full.append(cosine_dist(sub_centroid, full_centroid_20))

        convergence[str(n)] = {
            'mean_dist_to_full': float(np.mean(dists_to_full)),
            'max_dist_to_full': float(max(dists_to_full)),
            'std_dist_to_full': float(np.std(dists_to_full)),
        }
        print(f"  N={n}: dist_to_full={np.mean(dists_to_full):.8f}")

    results['centroid_convergence'] = convergence

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
