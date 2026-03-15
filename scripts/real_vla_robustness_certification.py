#!/usr/bin/env python3
"""Experiment 360: Formal Robustness Certification

Can we certify that detection will work for unseen scenes?
1. Leave-one-out cross-validation AUROC
2. Bootstrap confidence intervals on detection gap
3. Hoeffding bound on detection probability
4. Worst-case scene analysis
5. Certified detection radius
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

    # Generate embeddings for 30 scenes
    print("Generating embeddings for 30 scenes...")
    seeds = list(range(0, 3000, 100))[:30]
    images = {}
    clean_embs = {}
    corrupt_embs = {ct: {} for ct in ctypes}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

        for ct in ctypes:
            corrupted = apply_corruption(images[seed], ct, 0.5)
            corrupt_embs[ct][seed] = extract_hidden(model, processor, corrupted, prompt)

    # ========== 1. Leave-One-Out Cross-Validation ==========
    print("\n=== Leave-One-Out CV ===")

    loo_results = {}
    for ct in ctypes:
        aurocs = []
        gaps = []
        for i, test_seed in enumerate(seeds):
            # Train on all except test_seed
            train_seeds = [s for s in seeds if s != test_seed]
            centroid = np.mean([clean_embs[s] for s in train_seeds], axis=0)

            # ID distances (train scenes)
            id_dists = [float(cosine_dist(centroid, clean_embs[s])) for s in train_seeds]
            # Also include test scene as ID
            test_id_dist = float(cosine_dist(centroid, clean_embs[test_seed]))

            # OOD distance for test scene
            test_ood_dist = float(cosine_dist(centroid, corrupt_embs[ct][test_seed]))

            # AUROC with test scene
            auroc = compute_auroc(id_dists + [test_id_dist], [test_ood_dist])
            aurocs.append(auroc)
            gaps.append(test_ood_dist - max(id_dists + [test_id_dist]))

        loo_results[ct] = {
            'mean_auroc': float(np.mean(aurocs)),
            'min_auroc': float(min(aurocs)),
            'std_auroc': float(np.std(aurocs)),
            'all_perfect': all(a >= 1.0 - 1e-10 for a in aurocs),
            'mean_gap': float(np.mean(gaps)),
            'min_gap': float(min(gaps)),
        }
        print(f"  {ct}: LOO mean_AUROC={np.mean(aurocs):.4f}, min={min(aurocs):.4f}, "
              f"gap_min={min(gaps):.6f}")

    results['loo_cv'] = loo_results

    # ========== 2. Bootstrap Confidence Intervals ==========
    print("\n=== Bootstrap CI (1000 resamples) ===")

    centroid = np.mean([clean_embs[s] for s in seeds], axis=0)
    bootstrap_results = {}

    for ct in ctypes:
        rng = np.random.RandomState(42)
        boot_aurocs = []
        boot_gaps = []

        for b in range(1000):
            boot_idx = rng.choice(len(seeds), len(seeds), replace=True)
            boot_seeds = [seeds[i] for i in boot_idx]

            id_dists = [float(cosine_dist(centroid, clean_embs[s])) for s in boot_seeds]
            ood_dists = [float(cosine_dist(centroid, corrupt_embs[ct][s])) for s in boot_seeds]

            auroc = compute_auroc(id_dists, ood_dists)
            gap = min(ood_dists) - max(id_dists)
            boot_aurocs.append(float(auroc))
            boot_gaps.append(float(gap))

        bootstrap_results[ct] = {
            'auroc_mean': float(np.mean(boot_aurocs)),
            'auroc_ci_lower': float(np.percentile(boot_aurocs, 2.5)),
            'auroc_ci_upper': float(np.percentile(boot_aurocs, 97.5)),
            'gap_mean': float(np.mean(boot_gaps)),
            'gap_ci_lower': float(np.percentile(boot_gaps, 2.5)),
            'gap_ci_upper': float(np.percentile(boot_gaps, 97.5)),
            'gap_positive_fraction': float(np.mean(np.array(boot_gaps) > 0)),
        }
        ci_l = np.percentile(boot_aurocs, 2.5)
        ci_u = np.percentile(boot_aurocs, 97.5)
        print(f"  {ct}: AUROC CI=[{ci_l:.4f}, {ci_u:.4f}], "
              f"gap positive {np.mean(np.array(boot_gaps) > 0)*100:.1f}%")

    results['bootstrap'] = bootstrap_results

    # ========== 3. Hoeffding Bound ==========
    print("\n=== Hoeffding Bound ===")

    hoeffding_results = {}
    for ct in ctypes:
        id_dists = [float(cosine_dist(centroid, clean_embs[s])) for s in seeds]
        ood_dists = [float(cosine_dist(centroid, corrupt_embs[ct][s])) for s in seeds]

        # Empirical detection rate
        thresh = max(id_dists)
        detected = sum(1 for d in ood_dists if d > thresh)
        p_hat = detected / len(ood_dists)

        # Hoeffding: P(|p_hat - p| >= eps) <= 2*exp(-2*n*eps^2)
        n = len(ood_dists)
        for eps in [0.01, 0.05, 0.1]:
            bound = 2 * np.exp(-2 * n * eps**2)
            conf = 1 - bound
            lower_bound = max(0, p_hat - eps)
            hoeffding_results[ct + '_eps_' + str(eps)] = {
                'p_hat': float(p_hat),
                'n': n,
                'epsilon': eps,
                'bound': float(bound),
                'confidence': float(conf),
                'lower_bound_on_p': float(lower_bound),
            }

        print(f"  {ct}: p_hat={p_hat:.3f}, n={n}, "
              f"P(p>={p_hat-0.1:.2f})>={1-2*np.exp(-2*n*0.1**2):.6f}")

    results['hoeffding'] = hoeffding_results

    # ========== 4. Worst-Case Scene Analysis ==========
    print("\n=== Worst-Case Scene Analysis ===")

    worst_case = {}
    for ct in ctypes:
        scene_gaps = {}
        for seed in seeds:
            id_dist = float(cosine_dist(centroid, clean_embs[seed]))
            ood_dist = float(cosine_dist(centroid, corrupt_embs[ct][seed]))
            scene_gaps[str(seed)] = {
                'id_dist': id_dist,
                'ood_dist': ood_dist,
                'gap': ood_dist - max(float(cosine_dist(centroid, clean_embs[s])) for s in seeds),
                'margin': ood_dist / (max(float(cosine_dist(centroid, clean_embs[s])) for s in seeds) + 1e-10),
            }

        # Find worst scene
        worst_seed = min(scene_gaps.keys(), key=lambda s: scene_gaps[s]['gap'])
        best_seed = max(scene_gaps.keys(), key=lambda s: scene_gaps[s]['gap'])

        worst_case[ct] = {
            'worst_scene': worst_seed,
            'worst_gap': scene_gaps[worst_seed]['gap'],
            'worst_margin': scene_gaps[worst_seed]['margin'],
            'best_scene': best_seed,
            'best_gap': scene_gaps[best_seed]['gap'],
            'best_margin': scene_gaps[best_seed]['margin'],
            'median_gap': float(np.median([v['gap'] for v in scene_gaps.values()])),
        }
        print(f"  {ct}: worst_gap={scene_gaps[worst_seed]['gap']:.6f}, "
              f"margin={scene_gaps[worst_seed]['margin']:.2f}x, "
              f"median_gap={worst_case[ct]['median_gap']:.6f}")

    results['worst_case'] = worst_case

    # ========== 5. Certified Detection Radius ==========
    print("\n=== Certified Detection Radius ===")

    # For each corruption, what is the minimum severity we can certify detection for?
    radius_results = {}
    for ct in ctypes:
        certified_sevs = []
        for sev in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            all_detected = True
            for seed in seeds:
                corrupted = apply_corruption(images[seed], ct, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = float(cosine_dist(centroid, emb))
                clean_max = max(float(cosine_dist(centroid, clean_embs[s])) for s in seeds)
                if d <= clean_max:
                    all_detected = False
                    break

            certified_sevs.append({
                'severity': float(sev),
                'all_detected': all_detected,
            })

        # Find minimum certified severity
        min_certified = None
        for sv in certified_sevs:
            if sv['all_detected'] and min_certified is None:
                min_certified = sv['severity']

        radius_results[ct] = {
            'per_severity': certified_sevs,
            'min_certified_severity': min_certified,
        }
        print(f"  {ct}: min_certified_severity={min_certified}")

    results['certified_radius'] = radius_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/robustness_certification_{ts}.json"
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
