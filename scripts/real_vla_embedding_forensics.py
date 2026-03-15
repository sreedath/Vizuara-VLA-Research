#!/usr/bin/env python3
"""Experiment 337: Embedding Forensics (Real OpenVLA-7B)

Deep forensic analysis of embedding structure:
1. Per-dimension corruption fingerprints
2. Sign pattern analysis across corruptions
3. Magnitude distribution of shift vectors
4. Dimensional contribution to detection
5. Sparse vs dense signal characterization
6. Reconstruction from top-k dimensions
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

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    ctypes = ['fog', 'night', 'noise', 'blur']
    shift_vectors = {}

    for ct in ctypes:
        img = apply_corruption(base_img, ct, 1.0)
        emb = extract_hidden(model, processor, img, prompt)
        shift_vectors[ct] = emb - clean_emb

    # ========== 1. Shift magnitude distribution ==========
    print("\n=== Shift Magnitude Distribution ===")
    magnitude_results = {}

    for ct in ctypes:
        shift = shift_vectors[ct]
        abs_shift = np.abs(shift)

        # Statistics
        magnitude_results[ct] = {
            'l2_norm': float(np.linalg.norm(shift)),
            'l1_norm': float(np.sum(abs_shift)),
            'linf_norm': float(np.max(abs_shift)),
            'mean_abs': float(np.mean(abs_shift)),
            'std_abs': float(np.std(abs_shift)),
            'median_abs': float(np.median(abs_shift)),
            'sparsity_ratio': float(np.sum(abs_shift < 1e-6) / len(shift)),
            'top1_pct': float(np.max(abs_shift) / np.sum(abs_shift) * 100),
            'top10_pct': float(np.sort(abs_shift)[-10:].sum() / np.sum(abs_shift) * 100),
            'top100_pct': float(np.sort(abs_shift)[-100:].sum() / np.sum(abs_shift) * 100),
            'gini': float(np.sum(np.abs(np.subtract.outer(abs_shift, abs_shift))) / (2 * len(shift) * np.sum(abs_shift))),
        }
        print(f"  {ct}: L2={magnitude_results[ct]['l2_norm']:.4f}, sparsity={magnitude_results[ct]['sparsity_ratio']:.3f}, gini={magnitude_results[ct]['gini']:.3f}")

    results['magnitudes'] = magnitude_results

    # ========== 2. Sign pattern analysis ==========
    print("\n=== Sign Pattern Analysis ===")
    sign_results = {}

    for ct in ctypes:
        shift = shift_vectors[ct]
        signs = np.sign(shift)

        n_positive = int(np.sum(signs > 0))
        n_negative = int(np.sum(signs < 0))
        n_zero = int(np.sum(signs == 0))

        sign_results[ct] = {
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_zero': n_zero,
            'pos_ratio': float(n_positive / len(shift)),
        }
        print(f"  {ct}: +{n_positive} / -{n_negative} / 0:{n_zero}")

    # Cross-corruption sign agreement
    sign_agreement = {}
    for i, ct1 in enumerate(ctypes):
        for ct2 in ctypes[i+1:]:
            s1 = np.sign(shift_vectors[ct1])
            s2 = np.sign(shift_vectors[ct2])
            agree = float(np.sum(s1 == s2) / len(s1))
            sign_agreement[f"{ct1}_vs_{ct2}"] = agree
            print(f"  Sign agree {ct1}-{ct2}: {agree:.3f}")

    results['signs'] = sign_results
    results['sign_agreement'] = sign_agreement

    # ========== 3. Top-k reconstruction ==========
    print("\n=== Top-k Reconstruction ===")
    topk_results = {}

    for ct in ctypes:
        shift = shift_vectors[ct]
        full_d = cosine_dist(clean_emb, clean_emb + shift)

        recon_data = {}
        for k in [1, 5, 10, 50, 100, 500, 1000, 2000]:
            # Keep only top-k dimensions
            abs_shift = np.abs(shift)
            threshold = np.sort(abs_shift)[-k] if k < len(shift) else 0
            mask = abs_shift >= threshold
            sparse_shift = shift * mask

            sparse_emb = clean_emb + sparse_shift
            recon_d = cosine_dist(clean_emb, sparse_emb)

            # Also check if direction is preserved
            dir_sim = cosine_sim(shift, sparse_shift) if np.linalg.norm(sparse_shift) > 0 else 0

            recon_data[str(k)] = {
                'distance': float(recon_d),
                'ratio_to_full': float(recon_d / full_d) if full_d > 0 else 0,
                'direction_preserved': float(dir_sim),
            }

        topk_results[ct] = recon_data
        print(f"  {ct}: top-10={recon_data['10']['ratio_to_full']:.3f}, top-100={recon_data['100']['ratio_to_full']:.3f}")

    results['topk_reconstruction'] = topk_results

    # ========== 4. Per-dimension detection power ==========
    print("\n=== Per-Dimension Detection Power ===")
    dim_power = {}

    # For each dimension, how much does it contribute to detection?
    for ct in ctypes:
        shift = shift_vectors[ct]
        # Contribution = |shift[d]| * |clean_emb[d]| (product determines cosine distance contribution)
        contributions = np.abs(shift) * np.abs(clean_emb)
        sorted_contrib = np.argsort(contributions)[::-1]

        dim_power[ct] = {
            'top5_dims': sorted_contrib[:5].tolist(),
            'top5_contrib': contributions[sorted_contrib[:5]].tolist(),
            'top5_pct': float(contributions[sorted_contrib[:5]].sum() / contributions.sum() * 100),
            'top50_pct': float(contributions[sorted_contrib[:50]].sum() / contributions.sum() * 100),
            'effective_dims': int(np.sum(contributions > contributions.mean())),
        }
        print(f"  {ct}: top5_pct={dim_power[ct]['top5_pct']:.1f}%, effective={dim_power[ct]['effective_dims']}")

    results['dim_power'] = dim_power

    # ========== 5. Cross-scene fingerprint stability ==========
    print("\n=== Cross-Scene Fingerprint Stability ===")
    fingerprint_results = {}

    for ct in ctypes:
        shifts = []
        for seed in [42, 99, 123, 777, 2000]:
            rng = np.random.RandomState(seed)
            px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            scene_img = Image.fromarray(px)
            scene_clean = extract_hidden(model, processor, scene_img, prompt)
            scene_corrupt = extract_hidden(model, processor, apply_corruption(scene_img, ct, 1.0), prompt)
            shifts.append(scene_corrupt - scene_clean)

        shifts = np.array(shifts)  # 5 × 4096
        mean_shift = shifts.mean(axis=0)

        # Stability per dimension
        dim_cv = np.std(shifts, axis=0) / (np.abs(np.mean(shifts, axis=0)) + 1e-10)
        stable_dims = int(np.sum(dim_cv < 0.5))  # CV < 50%

        # Sign consistency
        sign_consistency = np.mean(np.abs(np.mean(np.sign(shifts), axis=0)))

        fingerprint_results[ct] = {
            'mean_cv': float(np.mean(dim_cv)),
            'median_cv': float(np.median(dim_cv)),
            'stable_dims': stable_dims,
            'sign_consistency': float(sign_consistency),
        }
        print(f"  {ct}: stable_dims={stable_dims}/4096, sign_consistency={sign_consistency:.3f}")

    results['fingerprints'] = fingerprint_results

    # ========== 6. Embedding norm decomposition ==========
    print("\n=== Norm Decomposition ===")
    norm_results = {}

    for ct in ctypes:
        shift = shift_vectors[ct]
        clean_norm = np.linalg.norm(clean_emb)
        corrupt_emb = clean_emb + shift
        corrupt_norm = np.linalg.norm(corrupt_emb)

        # Decompose shift into parallel and perpendicular to clean
        clean_dir = clean_emb / clean_norm
        parallel_component = np.dot(shift, clean_dir) * clean_dir
        perp_component = shift - parallel_component

        norm_results[ct] = {
            'clean_norm': float(clean_norm),
            'corrupt_norm': float(corrupt_norm),
            'norm_change_pct': float((corrupt_norm - clean_norm) / clean_norm * 100),
            'parallel_magnitude': float(np.linalg.norm(parallel_component)),
            'perpendicular_magnitude': float(np.linalg.norm(perp_component)),
            'parallel_pct': float(np.linalg.norm(parallel_component) / np.linalg.norm(shift) * 100),
        }
        print(f"  {ct}: parallel={norm_results[ct]['parallel_pct']:.1f}%, perp={100-norm_results[ct]['parallel_pct']:.1f}%")

    results['norm_decomposition'] = norm_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/embed_forensics_{ts}.json"
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
