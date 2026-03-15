#!/usr/bin/env python3
"""Experiment 342: Gradient-Free Adversarial Analysis

Can an adversary evade the detector without gradient access?
1. Minimum perturbation for maximum embedding shift
2. Targeted perturbation: can we maximize shift in a specific direction?
3. Gradient-free evasion: can corrupted input be modified to reduce distance?
4. Steganographic corruption: can corruption be hidden in specific channels?
5. Defense-aware attacks: what if adversary knows the detection method?
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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    # Base scene
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    cal_emb = extract_hidden(model, processor, base_img, prompt)

    # ========== 1. Random search for maximum shift ==========
    print("\n=== Random Search: Max Shift per Pixel Budget ===")

    budgets = [1, 5, 10, 20, 50, 100]  # max pixel change per channel
    max_shift_per_budget = {}
    rng = np.random.RandomState(42)

    for budget in budgets:
        best_dist = 0
        best_perturbation = None

        # Try 50 random perturbations
        for trial in range(50):
            perturbed = pixels.copy().astype(np.int16)
            # Random sparse perturbation
            n_pixels = rng.randint(1, 100)
            for _ in range(n_pixels):
                x, y = rng.randint(0, 224), rng.randint(0, 224)
                c = rng.randint(0, 3)
                delta = rng.randint(-budget, budget + 1)
                perturbed[y, x, c] = np.clip(perturbed[y, x, c] + delta, 0, 255)

            img = Image.fromarray(perturbed.astype(np.uint8))
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_emb, emb)

            if d > best_dist:
                best_dist = d
                best_perturbation = {
                    'n_pixels': n_pixels,
                    'budget': budget,
                    'distance': float(d),
                }

        max_shift_per_budget[str(budget)] = {
            'best_distance': float(best_dist),
            'n_trials': 50,
        }
        print(f"  Budget ±{budget}: best_dist={best_dist:.6f}")

    results['random_search'] = max_shift_per_budget

    # ========== 2. Structured perturbations ==========
    print("\n=== Structured Perturbations ===")

    struct_results = {}

    # Single-channel perturbation
    for channel, cname in enumerate(['red', 'green', 'blue']):
        perturbed = pixels.copy()
        perturbed[:, :, channel] = np.clip(perturbed[:, :, channel].astype(np.int16) + 20, 0, 255).astype(np.uint8)
        img = Image.fromarray(perturbed)
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(cal_emb, emb)
        struct_results[f'channel_{cname}_+20'] = float(d)
        print(f"  {cname} channel +20: d={d:.6f}")

    # Frequency-targeted
    arr_f = np.array(base_img).astype(np.float32)
    # Low-freq: uniform shift
    perturbed_low = np.clip(arr_f + 10, 0, 255).astype(np.uint8)
    img = Image.fromarray(perturbed_low)
    emb = extract_hidden(model, processor, img, prompt)
    struct_results['uniform_+10'] = float(cosine_dist(cal_emb, emb))

    # High-freq: checkerboard
    checker = np.zeros((224, 224, 3))
    checker[::2, ::2, :] = 20
    checker[1::2, 1::2, :] = 20
    perturbed_hi = np.clip(arr_f + checker, 0, 255).astype(np.uint8)
    img = Image.fromarray(perturbed_hi)
    emb = extract_hidden(model, processor, img, prompt)
    struct_results['checker_+20'] = float(cosine_dist(cal_emb, emb))

    # Gradient perturbation
    grad = np.linspace(-20, 20, 224).reshape(1, 224, 1) * np.ones((224, 1, 3))
    perturbed_grad = np.clip(arr_f + grad, 0, 255).astype(np.uint8)
    img = Image.fromarray(perturbed_grad)
    emb = extract_hidden(model, processor, img, prompt)
    struct_results['gradient_±20'] = float(cosine_dist(cal_emb, emb))

    print(f"  Uniform +10: d={struct_results['uniform_+10']:.6f}")
    print(f"  Checker +20: d={struct_results['checker_+20']:.6f}")
    print(f"  Gradient ±20: d={struct_results['gradient_±20']:.6f}")

    results['structured'] = struct_results

    # ========== 3. Evasion: can corrupted input be steered back? ==========
    print("\n=== Evasion Attempts ===")

    evasion_results = {}
    for ct in ['fog', 'night', 'blur']:
        corrupt_img = apply_corruption(base_img, ct, 0.5)
        corrupt_emb = extract_hidden(model, processor, corrupt_img, prompt)
        original_dist = cosine_dist(cal_emb, corrupt_emb)

        corrupt_arr = np.array(corrupt_img).astype(np.float32)
        clean_arr = np.array(base_img).astype(np.float32)

        best_evasion_dist = original_dist

        # Strategy 1: Add the negative of the pixel-space corruption
        reverse = np.clip(2 * clean_arr - corrupt_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(reverse)
        emb = extract_hidden(model, processor, img, prompt)
        d_reverse = cosine_dist(cal_emb, emb)

        # Strategy 2: Blend corrupted back toward clean
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            blended = np.clip(corrupt_arr * (1 - alpha) + clean_arr * alpha, 0, 255).astype(np.uint8)
            img = Image.fromarray(blended)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_emb, emb)
            if d < best_evasion_dist:
                best_evasion_dist = d

        # Strategy 3: Random perturbation of corrupted image
        for trial in range(30):
            perturbed = corrupt_arr + rng.randn(*corrupt_arr.shape) * 5
            perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
            img = Image.fromarray(perturbed)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_emb, emb)
            if d < best_evasion_dist:
                best_evasion_dist = d

        evasion_results[ct] = {
            'original_dist': float(original_dist),
            'reverse_dist': float(d_reverse),
            'best_evasion_dist': float(best_evasion_dist),
            'evasion_ratio': float(best_evasion_dist / original_dist) if original_dist > 0 else 0,
            'evasion_successful': bool(best_evasion_dist == 0),
        }
        print(f"  {ct}: original={original_dist:.6f}, best_evasion={best_evasion_dist:.6f}, "
              f"ratio={best_evasion_dist/original_dist:.3f}")

    results['evasion'] = evasion_results

    # ========== 4. Defense-aware attack: targeted search ==========
    print("\n=== Defense-Aware Attack ===")

    defense_results = {}
    # An adversary who knows the detection method tries to minimize distance
    # while maximizing visual corruption
    for ct in ['fog', 'night', 'blur']:
        # Apply corruption then systematically search for additive correction
        corrupt_img = apply_corruption(base_img, ct, 0.5)
        corrupt_emb = extract_hidden(model, processor, corrupt_img, prompt)
        original_dist = cosine_dist(cal_emb, corrupt_emb)

        corrupt_arr = np.array(corrupt_img).astype(np.float32)

        best_dist = original_dist
        best_correction = None
        n_evals = 0

        # Hill climbing: try random corrections, keep if distance decreases
        current_arr = corrupt_arr.copy()
        current_dist = original_dist

        for step in range(100):
            # Random correction patch
            correction = rng.randn(*corrupt_arr.shape) * 3
            candidate = np.clip(current_arr + correction, 0, 255).astype(np.uint8)
            img = Image.fromarray(candidate)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_emb, emb)
            n_evals += 1

            if d < current_dist:
                current_arr = candidate.astype(np.float32)
                current_dist = d
                if d < best_dist:
                    best_dist = d

        # How much visual corruption remains?
        visual_diff = np.mean(np.abs(current_arr - np.array(base_img).astype(np.float32))) / 255.0

        defense_results[ct] = {
            'original_dist': float(original_dist),
            'best_attacked_dist': float(best_dist),
            'reduction_ratio': float(best_dist / original_dist) if original_dist > 0 else 0,
            'n_evaluations': n_evals,
            'remaining_visual_diff': float(visual_diff),
            'evasion_successful': bool(best_dist == 0),
        }
        print(f"  {ct}: {original_dist:.6f} -> {best_dist:.6f} "
              f"({best_dist/original_dist*100:.1f}% of original), visual_diff={visual_diff:.4f}")

    results['defense_aware'] = defense_results

    # ========== 5. Imperceptible corruption test ==========
    print("\n=== Imperceptible Corruption ===")

    imperceptible = {}
    # Can we create corruption that's invisible to humans but detected?
    for n_pixels in [1, 5, 10, 50, 100, 500]:
        perturbed = pixels.copy()
        positions = rng.choice(224*224, n_pixels, replace=False)
        for pos in positions:
            y, x = pos // 224, pos % 224
            perturbed[y, x, :] = 255 - perturbed[y, x, :]  # Flip selected pixels

        img = Image.fromarray(perturbed)
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(cal_emb, emb)

        imperceptible[str(n_pixels)] = {
            'distance': float(d),
            'detected': bool(d > 0),
            'pct_pixels': float(n_pixels / (224*224) * 100),
        }
        print(f"  {n_pixels} flipped pixels ({n_pixels/(224*224)*100:.3f}%): d={d:.6f}")

    results['imperceptible'] = imperceptible

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/adversarial_analysis_{ts}.json"
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
