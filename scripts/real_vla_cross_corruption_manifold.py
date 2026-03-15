#!/usr/bin/env python3
"""Experiment 341: Cross-Corruption Manifold Structure

Analyze the geometric relationships between different corruption types:
1. Pairwise angles between corruption directions across severity levels
2. Interpolation paths between corruption types in embedding space
3. Superposition effects: do combined corruptions add linearly?
4. Manifold curvature along corruption trajectories
5. Identify corruption subspaces and their dimensionality
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
    elif ctype == 'rain':
        rng = np.random.RandomState(42)
        for _ in range(200):
            x, y = rng.randint(0, 224), rng.randint(0, 224)
            length = rng.randint(5, 20)
            for k in range(length):
                if y + k < 224:
                    arr[y+k, x, :] = np.clip(arr[y+k, x, :] + 0.3 * severity, 0, 1)
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    elif ctype == 'frost':
        rng = np.random.RandomState(42)
        frost = rng.random((224, 224, 3)) * 0.4 * severity
        arr = np.clip(arr + frost, 0, 1) * (1 - 0.3 * severity) + 0.6 * severity * 0.3
        arr = np.clip(arr, 0, 1)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def apply_combined(image, ct1, sev1, ct2, sev2):
    """Apply two corruptions sequentially."""
    img1 = apply_corruption(image, ct1, sev1)
    return apply_corruption(img1, ct2, sev2)

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

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
    seeds = [0, 100, 200, 300, 400]
    scenes = {}
    cal_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scenes[seed] = Image.fromarray(px)
        cal_embs[seed] = extract_hidden(model, processor, scenes[seed], prompt)
        print(f"  Scene {seed} calibrated")

    ctypes = ['fog', 'night', 'noise', 'blur', 'rain', 'frost']
    severities = [0.1, 0.3, 0.5, 0.7, 1.0]

    # ========== 1. Corruption direction vectors ==========
    print("\n=== Corruption Direction Vectors ===")

    # For each scene, compute shift vectors for each corruption at each severity
    shift_vectors = {}  # (seed, ctype, sev) -> vector
    for seed in seeds:
        cal = cal_embs[seed]
        for ct in ctypes:
            for sev in severities:
                img = apply_corruption(scenes[seed], ct, sev)
                emb = extract_hidden(model, processor, img, prompt)
                shift = emb - cal
                shift_vectors[(seed, ct, sev)] = shift

    # ========== 2. Pairwise angles between corruption types ==========
    print("\n=== Pairwise Corruption Angles ===")

    pairwise_angles = {}
    for i, ct1 in enumerate(ctypes):
        for j, ct2 in enumerate(ctypes):
            if j <= i:
                continue
            angles_across = []
            for seed in seeds:
                for sev in [0.3, 0.5, 0.7]:
                    v1 = shift_vectors[(seed, ct1, sev)]
                    v2 = shift_vectors[(seed, ct2, sev)]
                    sim = cosine_sim(v1, v2)
                    angle = np.degrees(np.arccos(np.clip(sim, -1, 1)))
                    angles_across.append(angle)

            key = f"{ct1}_vs_{ct2}"
            pairwise_angles[key] = {
                'mean_angle': float(np.mean(angles_across)),
                'std_angle': float(np.std(angles_across)),
                'min_angle': float(np.min(angles_across)),
                'max_angle': float(np.max(angles_across)),
                'near_orthogonal': bool(abs(np.mean(angles_across) - 90) < 20),
                'aligned': bool(np.mean(angles_across) < 30),
                'anti_aligned': bool(np.mean(angles_across) > 150),
            }
            print(f"  {key}: {np.mean(angles_across):.1f}° ± {np.std(angles_across):.1f}°")

    results['pairwise_angles'] = pairwise_angles

    # ========== 3. Severity direction consistency ==========
    print("\n=== Severity Direction Consistency ===")

    sev_consistency = {}
    for ct in ctypes:
        angles_list = []
        for seed in seeds:
            # Compare direction at different severities
            for si in range(len(severities)):
                for sj in range(si+1, len(severities)):
                    v1 = shift_vectors[(seed, ct, severities[si])]
                    v2 = shift_vectors[(seed, ct, severities[sj])]
                    sim = cosine_sim(v1, v2)
                    angles_list.append(np.degrees(np.arccos(np.clip(sim, -1, 1))))

        sev_consistency[ct] = {
            'mean_angle': float(np.mean(angles_list)),
            'max_angle': float(np.max(angles_list)),
            'direction_stable': bool(np.mean(angles_list) < 15),
        }
        print(f"  {ct}: mean_angle={np.mean(angles_list):.1f}°, "
              f"stable={sev_consistency[ct]['direction_stable']}")

    results['severity_consistency'] = sev_consistency

    # ========== 4. Linearity of combined corruptions ==========
    print("\n=== Superposition Linearity ===")

    linearity = {}
    for seed in seeds[:3]:  # 3 scenes for speed
        cal = cal_embs[seed]
        for i, ct1 in enumerate(ctypes[:4]):  # 4 types for speed
            for j, ct2 in enumerate(ctypes[:4]):
                if j <= i:
                    continue
                sev = 0.5

                # Individual shifts
                v1 = shift_vectors[(seed, ct1, sev)]
                v2 = shift_vectors[(seed, ct2, sev)]

                # Combined (apply ct1 then ct2)
                combined_img = apply_combined(scenes[seed], ct1, sev, ct2, sev)
                combined_emb = extract_hidden(model, processor, combined_img, prompt)
                actual_shift = combined_emb - cal

                # Predicted (linear superposition)
                predicted_shift = v1 + v2

                # How well does linear prediction match?
                sim = cosine_sim(actual_shift, predicted_shift)
                actual_norm = np.linalg.norm(actual_shift)
                predicted_norm = np.linalg.norm(predicted_shift)
                norm_ratio = actual_norm / predicted_norm if predicted_norm > 1e-10 else 0

                key = f"s{seed}_{ct1}+{ct2}"
                linearity[key] = {
                    'direction_sim': float(sim),
                    'actual_norm': float(actual_norm),
                    'predicted_norm': float(predicted_norm),
                    'norm_ratio': float(norm_ratio),
                    'is_linear': bool(sim > 0.9 and 0.5 < norm_ratio < 1.5),
                }

    # Aggregate
    sims = [v['direction_sim'] for v in linearity.values()]
    ratios = [v['norm_ratio'] for v in linearity.values()]
    results['linearity'] = {
        'per_combination': linearity,
        'mean_direction_sim': float(np.mean(sims)),
        'mean_norm_ratio': float(np.mean(ratios)),
        'pct_linear': float(sum(1 for v in linearity.values() if v['is_linear']) / len(linearity)),
    }
    print(f"  Mean direction similarity: {np.mean(sims):.3f}")
    print(f"  Mean norm ratio: {np.mean(ratios):.3f}")
    print(f"  % linear: {results['linearity']['pct_linear']*100:.1f}%")

    # ========== 5. Corruption manifold dimensionality ==========
    print("\n=== Manifold Dimensionality ===")

    # Collect all shift vectors for PCA
    all_shifts = []
    labels = []
    for seed in seeds:
        for ct in ctypes:
            for sev in severities:
                all_shifts.append(shift_vectors[(seed, ct, sev)])
                labels.append(f"{ct}_{sev}")

    shift_matrix = np.array(all_shifts)
    # Center
    mean_shift = shift_matrix.mean(axis=0)
    centered = shift_matrix - mean_shift

    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    total_var = np.sum(S**2)
    cum_var = np.cumsum(S**2) / total_var

    # Find dimensionality at various thresholds
    dim_results = {}
    for thresh in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        dim = int(np.searchsorted(cum_var, thresh) + 1)
        dim_results[str(thresh)] = dim

    # Per-corruption subspace dimensionality
    per_type_dim = {}
    for ct in ctypes:
        ct_shifts = [shift_vectors[(seed, ct, sev)] for seed in seeds for sev in severities]
        ct_matrix = np.array(ct_shifts)
        ct_centered = ct_matrix - ct_matrix.mean(axis=0)
        _, ct_S, _ = np.linalg.svd(ct_centered, full_matrices=False)
        ct_total = np.sum(ct_S**2)
        ct_cum = np.cumsum(ct_S**2) / ct_total if ct_total > 0 else np.zeros_like(ct_S)
        dim90 = int(np.searchsorted(ct_cum, 0.9) + 1) if ct_total > 0 else 0
        per_type_dim[ct] = {
            'dim_90pct': dim90,
            'top3_var': float(ct_cum[2]) if len(ct_cum) > 2 else 0,
            'top1_var': float(ct_cum[0]) if len(ct_cum) > 0 else 0,
        }
        print(f"  {ct}: 90% in {dim90}D, PC1={per_type_dim[ct]['top1_var']*100:.1f}%")

    results['manifold'] = {
        'total_samples': len(all_shifts),
        'embedding_dim': shift_matrix.shape[1],
        'dim_at_threshold': dim_results,
        'top10_singular_values': S[:10].tolist(),
        'per_type_dimensionality': per_type_dim,
    }

    print(f"\n  Overall: 90% variance in {dim_results['0.9']}D, 99% in {dim_results['0.99']}D")

    # ========== 6. Commutativity of corruption order ==========
    print("\n=== Commutativity Test ===")

    commutativity = {}
    for seed in seeds[:3]:
        cal = cal_embs[seed]
        for i, ct1 in enumerate(ctypes[:4]):
            for j, ct2 in enumerate(ctypes[:4]):
                if j <= i:
                    continue
                sev = 0.5
                # Order 1: ct1 then ct2
                img12 = apply_combined(scenes[seed], ct1, sev, ct2, sev)
                emb12 = extract_hidden(model, processor, img12, prompt)

                # Order 2: ct2 then ct1
                img21 = apply_combined(scenes[seed], ct2, sev, ct1, sev)
                emb21 = extract_hidden(model, processor, img21, prompt)

                d = cosine_dist(emb12, emb21)
                sim = cosine_sim(emb12 - cal, emb21 - cal)

                key = f"s{seed}_{ct1}_{ct2}"
                commutativity[key] = {
                    'distance': float(d),
                    'shift_similarity': float(sim),
                    'commutative': bool(d < 1e-4),
                }

    comm_dists = [v['distance'] for v in commutativity.values()]
    comm_sims = [v['shift_similarity'] for v in commutativity.values()]
    results['commutativity'] = {
        'per_pair': commutativity,
        'mean_order_distance': float(np.mean(comm_dists)),
        'max_order_distance': float(np.max(comm_dists)),
        'mean_shift_similarity': float(np.mean(comm_sims)),
        'pct_commutative': float(sum(1 for v in commutativity.values() if v['commutative']) / len(commutativity)),
    }
    print(f"  Mean order distance: {np.mean(comm_dists):.6f}")
    print(f"  % commutative: {results['commutativity']['pct_commutative']*100:.1f}%")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/cross_corruption_manifold_{ts}.json"
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
