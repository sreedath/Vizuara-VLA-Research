#!/usr/bin/env python3
"""Experiment 345: Statistical Power and Sample Complexity

Rigorous statistical analysis of detection power:
1. Bootstrap confidence intervals for AUROC
2. Effect of calibration sample size on detection power
3. Minimum corruption severity for statistical significance
4. Multiple testing correction across corruption types
5. Bayesian detection framework with prior/posterior
"""

import json, time, os, sys, math
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

    # Create scenes
    seeds = list(range(0, 1500, 100))[:15]  # 15 scenes
    scenes = {}
    cal_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scenes[seed] = Image.fromarray(px)
        cal_embs[seed] = extract_hidden(model, processor, scenes[seed], prompt)
        print(f"  Scene {seed} calibrated")

    ctypes = ['fog', 'night', 'noise', 'blur']

    # ========== 1. Bootstrap AUROC confidence intervals ==========
    print("\n=== Bootstrap AUROC CIs ===")

    # Collect all distances
    all_id = []
    all_ood = {}
    for ct in ctypes:
        all_ood[ct] = []

    for seed in seeds:
        cal = cal_embs[seed]
        # ID: clean re-embedding
        emb = extract_hidden(model, processor, scenes[seed], prompt)
        all_id.append(float(cosine_dist(cal, emb)))

        # OOD: each corruption at 0.5
        for ct in ctypes:
            img = apply_corruption(scenes[seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            all_ood[ct].append(float(cosine_dist(cal, emb)))

    # Bootstrap
    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    bootstrap_results = {}

    for ct in ctypes:
        aurocs = []
        for _ in range(n_bootstrap):
            id_sample = rng.choice(all_id, len(all_id), replace=True)
            ood_sample = rng.choice(all_ood[ct], len(all_ood[ct]), replace=True)
            aurocs.append(compute_auroc(id_sample, ood_sample))

        aurocs = np.array(aurocs)
        bootstrap_results[ct] = {
            'mean_auroc': float(np.mean(aurocs)),
            'std_auroc': float(np.std(aurocs)),
            'ci_lower': float(np.percentile(aurocs, 2.5)),
            'ci_upper': float(np.percentile(aurocs, 97.5)),
            'pct_perfect': float(np.mean(aurocs == 1.0) * 100),
        }
        print(f"  {ct}: AUROC={np.mean(aurocs):.4f} [{np.percentile(aurocs, 2.5):.4f}, "
              f"{np.percentile(aurocs, 97.5):.4f}], {np.mean(aurocs==1.0)*100:.1f}% perfect")

    results['bootstrap'] = bootstrap_results

    # ========== 2. Detection power vs severity ==========
    print("\n=== Detection Power vs Severity ===")

    severity_power = {}
    for sev in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        for ct in ctypes:
            ood_dists = []
            for seed in seeds[:10]:
                img = apply_corruption(scenes[seed], ct, sev)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(cal_embs[seed], emb)
                ood_dists.append(float(d))

            auroc = compute_auroc([0.0] * 10, ood_dists)
            min_d = min(ood_dists)
            mean_d = np.mean(ood_dists)

            key = f"{ct}_sev{sev}"
            severity_power[key] = {
                'auroc': float(auroc),
                'min_distance': float(min_d),
                'mean_distance': float(mean_d),
                'all_detected': bool(min_d > 0),
            }

        detected_all = sum(1 for ct in ctypes if severity_power[f"{ct}_sev{sev}"]['all_detected'])
        print(f"  Sev={sev}: {detected_all}/4 types fully detected")

    results['severity_power'] = severity_power

    # ========== 3. Effect size analysis ==========
    print("\n=== Effect Size (Cohen's d) ===")

    effect_sizes = {}
    for ct in ctypes:
        ood = np.array(all_ood[ct])
        id_arr = np.array(all_id)

        # Cohen's d
        pooled_std = np.sqrt((np.var(ood) + np.var(id_arr)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(ood) - np.mean(id_arr)) / pooled_std
        else:
            cohens_d = float('inf')  # Zero ID variance

        # Glass's delta (using ID std)
        if np.std(id_arr) > 0:
            glass_delta = (np.mean(ood) - np.mean(id_arr)) / np.std(id_arr)
        else:
            glass_delta = float('inf')

        effect_sizes[ct] = {
            'id_mean': float(np.mean(id_arr)),
            'id_std': float(np.std(id_arr)),
            'ood_mean': float(np.mean(ood)),
            'ood_std': float(np.std(ood)),
            'cohens_d': float(cohens_d) if not math.isinf(cohens_d) else 'infinite',
            'glass_delta': float(glass_delta) if not math.isinf(glass_delta) else 'infinite',
            'separation': 'perfect' if np.min(ood) > np.max(id_arr) else 'partial',
        }
        print(f"  {ct}: Cohen's d={'infinite' if math.isinf(cohens_d) else f'{cohens_d:.2f}'}, "
              f"separation={effect_sizes[ct]['separation']}")

    results['effect_sizes'] = effect_sizes

    # ========== 4. Multiple testing correction ==========
    print("\n=== Multiple Testing ===")

    # Bonferroni correction for testing K corruption types
    K = len(ctypes)
    alpha = 0.05
    bonferroni_alpha = alpha / K

    # For each type, compute p-value using permutation test
    n_perms = 500
    pvalues = {}
    for ct in ctypes:
        ood = all_ood[ct]
        combined = all_id + ood
        observed_diff = np.mean(ood) - np.mean(all_id)

        count_extreme = 0
        for _ in range(n_perms):
            perm = rng.permutation(combined)
            perm_id = perm[:len(all_id)]
            perm_ood = perm[len(all_id):]
            perm_diff = np.mean(perm_ood) - np.mean(perm_id)
            if perm_diff >= observed_diff:
                count_extreme += 1

        p = (count_extreme + 1) / (n_perms + 1)
        pvalues[ct] = {
            'p_value': float(p),
            'significant_uncorrected': bool(p < alpha),
            'significant_bonferroni': bool(p < bonferroni_alpha),
        }
        print(f"  {ct}: p={p:.4f}, Bonferroni sig={p < bonferroni_alpha}")

    results['multiple_testing'] = {
        'alpha': alpha,
        'bonferroni_alpha': bonferroni_alpha,
        'K': K,
        'per_type': pvalues,
    }

    # ========== 5. Bayesian detection ==========
    print("\n=== Bayesian Framework ===")

    bayesian = {}
    # Prior: P(corrupt) = 0.1 (10% base rate)
    prior_corrupt = 0.1

    for ct in ctypes:
        ood = np.array(all_ood[ct])
        # Likelihood: P(d > t | corrupt) and P(d > t | clean)
        threshold = 0
        p_detect_given_corrupt = np.mean(ood > threshold)
        p_detect_given_clean = np.mean(np.array(all_id) > threshold)

        # Posterior: P(corrupt | d > t)
        if p_detect_given_corrupt * prior_corrupt + p_detect_given_clean * (1 - prior_corrupt) > 0:
            posterior = (p_detect_given_corrupt * prior_corrupt) / \
                       (p_detect_given_corrupt * prior_corrupt + p_detect_given_clean * (1 - prior_corrupt))
        else:
            posterior = 0

        # Log Bayes factor
        if p_detect_given_clean > 0 and p_detect_given_corrupt > 0:
            bayes_factor = p_detect_given_corrupt / p_detect_given_clean
            log_bf = np.log10(bayes_factor)
        else:
            bayes_factor = float('inf')
            log_bf = float('inf')

        bayesian[ct] = {
            'prior_corrupt': prior_corrupt,
            'p_detect_given_corrupt': float(p_detect_given_corrupt),
            'p_detect_given_clean': float(p_detect_given_clean),
            'posterior_corrupt': float(posterior),
            'bayes_factor': float(bayes_factor) if not math.isinf(bayes_factor) else 'infinite',
            'log10_bayes_factor': float(log_bf) if not math.isinf(log_bf) else 'infinite',
        }
        bf_str = 'infinite' if math.isinf(bayes_factor) else f'{bayes_factor:.1f}'
        print(f"  {ct}: P(corrupt|detect)={posterior:.4f}, BF={bf_str}")

    results['bayesian'] = bayesian

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/statistical_power_{ts}.json"
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
