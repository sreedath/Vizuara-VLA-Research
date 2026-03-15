#!/usr/bin/env python3
"""Experiment 382: Severity Detection Threshold Mapping

For each corruption type, what is the minimum detectable severity?
1. Fine-grained severity sweep (0.01 to 1.0 in small steps)
2. AUROC vs severity curve
3. Detection threshold crossing point
4. Distance vs severity functional form (linear? exponential?)
5. Cross-corruption threshold comparison
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
    id_s, ood_s = np.asarray(id_scores), np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
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

    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    centroid = np.mean(list(clean_embs.values()), axis=0)
    clean_dists = [cosine_dist(centroid, clean_embs[s]) for s in seeds]
    threshold = max(clean_dists)
    print(f"  Threshold: {threshold:.6f}")

    # ========== 1. Fine-Grained Severity Sweep ==========
    print("\n=== Fine-Grained Severity Sweep ===")

    severities = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3,
                  0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    sweep = {}
    for ct in ctypes:
        ct_data = {}
        for sev in severities:
            ood_dists = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                d = cosine_dist(emb, centroid)
                ood_dists.append(d)

            auroc = compute_auroc(clean_dists, ood_dists)
            detection_rate = sum(1 for d in ood_dists if d > threshold) / len(ood_dists)

            ct_data[str(sev)] = {
                'mean_dist': float(np.mean(ood_dists)),
                'max_dist': float(max(ood_dists)),
                'min_dist': float(min(ood_dists)),
                'auroc': float(auroc),
                'detection_rate': float(detection_rate),
            }

        sweep[ct] = ct_data

        # Find threshold crossing
        first_detected = None
        first_auroc_1 = None
        for sev in severities:
            if ct_data[str(sev)]['detection_rate'] > 0 and first_detected is None:
                first_detected = sev
            if ct_data[str(sev)]['auroc'] >= 1.0 and first_auroc_1 is None:
                first_auroc_1 = sev

        print(f"  {ct}: first_detect={first_detected}, first_AUROC1={first_auroc_1}")

    results['severity_sweep'] = sweep

    # ========== 2. Functional Form Analysis ==========
    print("\n=== Functional Form Analysis ===")

    form_analysis = {}
    for ct in ctypes:
        sevs = np.array(severities)
        dists = np.array([sweep[ct][str(s)]['mean_dist'] for s in severities])

        # Test linear fit: dist = a * sev + b
        if len(sevs) > 1:
            coeffs_linear = np.polyfit(sevs, dists, 1)
            pred_linear = np.polyval(coeffs_linear, sevs)
            ss_res_linear = np.sum((dists - pred_linear)**2)
            ss_tot = np.sum((dists - np.mean(dists))**2)
            r2_linear = 1 - ss_res_linear / max(ss_tot, 1e-20)

            # Test quadratic: dist = a * sev^2 + b * sev + c
            coeffs_quad = np.polyfit(sevs, dists, 2)
            pred_quad = np.polyval(coeffs_quad, sevs)
            ss_res_quad = np.sum((dists - pred_quad)**2)
            r2_quad = 1 - ss_res_quad / max(ss_tot, 1e-20)

            # Test log: dist = a * log(sev) + b
            log_sevs = np.log(sevs + 1e-10)
            coeffs_log = np.polyfit(log_sevs, dists, 1)
            pred_log = np.polyval(coeffs_log, log_sevs)
            ss_res_log = np.sum((dists - pred_log)**2)
            r2_log = 1 - ss_res_log / max(ss_tot, 1e-20)

            best_form = 'linear' if r2_linear >= max(r2_quad, r2_log) else \
                        'quadratic' if r2_quad >= r2_log else 'logarithmic'

            form_analysis[ct] = {
                'r2_linear': float(r2_linear),
                'r2_quadratic': float(r2_quad),
                'r2_logarithmic': float(r2_log),
                'best_form': best_form,
                'linear_slope': float(coeffs_linear[0]),
                'linear_intercept': float(coeffs_linear[1]),
            }
            print(f"  {ct}: linear R²={r2_linear:.4f}, quad R²={r2_quad:.4f}, "
                  f"log R²={r2_log:.4f} → {best_form}")

    results['functional_form'] = form_analysis

    # ========== 3. Cross-Corruption Threshold Comparison ==========
    print("\n=== Cross-Corruption Threshold Comparison ===")

    threshold_map = {}
    for ct in ctypes:
        # Find minimum severity for 100% detection rate
        min_100 = None
        for sev in severities:
            if sweep[ct][str(sev)]['detection_rate'] >= 1.0:
                min_100 = sev
                break

        # Find minimum severity for AUROC=1.0
        min_auroc_1 = None
        for sev in severities:
            if sweep[ct][str(sev)]['auroc'] >= 1.0:
                min_auroc_1 = sev
                break

        # Find minimum severity for >50% detection
        min_50 = None
        for sev in severities:
            if sweep[ct][str(sev)]['detection_rate'] >= 0.5:
                min_50 = sev
                break

        threshold_map[ct] = {
            'min_severity_100pct': min_100,
            'min_severity_auroc1': min_auroc_1,
            'min_severity_50pct': min_50,
            'dist_at_01': float(sweep[ct]['0.1']['mean_dist']),
            'dist_at_05': float(sweep[ct]['0.5']['mean_dist']),
            'dist_at_10': float(sweep[ct]['1.0']['mean_dist']),
        }
        print(f"  {ct}: 50%={min_50}, 100%={min_100}, AUROC1={min_auroc_1}")

    results['threshold_comparison'] = threshold_map
    results['clean_threshold'] = float(threshold)

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/severity_threshold_{ts}.json"
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
