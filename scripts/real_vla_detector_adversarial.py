#!/usr/bin/env python3
"""Experiment 372: Adversarial Robustness of the OOD Detector

Can carefully crafted perturbations fool the corruption detector?
1. Clean image + random noise at various L-inf budgets
2. Gradient-free adversarial: find perturbation that minimizes detector score
3. Patch attack: adversarial patch that masks corruption
4. Corruption + counter-perturbation: corrupt then "fix" detector score
5. Detector evasion difficulty quantification
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
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 500, 100))[:5]
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
    print(f"  Detection threshold: {threshold:.6f}")

    # ========== 1. Random Noise at Various L-inf Budgets ==========
    print("\n=== Random Noise L-inf Attack ===")

    linf_results = {}
    for eps in [1, 2, 4, 8, 16, 32, 64]:
        false_positives = 0
        n_total = 0
        dists = []
        for seed in seeds:
            arr = np.array(images[seed]).copy()
            rng = np.random.RandomState(seed + 7777)
            perturbed = np.clip(arr.astype(np.int16) + rng.randint(-eps, eps+1, arr.shape), 0, 255).astype(np.uint8)
            pert_img = Image.fromarray(perturbed)
            emb = extract_hidden(model, processor, pert_img, prompt)
            d = cosine_dist(emb, centroid)
            dists.append(d)
            n_total += 1
            if d > threshold:
                false_positives += 1

        linf_results[str(eps)] = {
            'mean_dist': float(np.mean(dists)),
            'max_dist': float(max(dists)),
            'false_positive_rate': false_positives / n_total,
            'exceeds_threshold': any(d > threshold for d in dists),
        }
        print(f"  eps={eps}: mean_dist={np.mean(dists):.6f}, "
              f"FPR={false_positives}/{n_total}")

    results['linf_noise'] = linf_results

    # ========== 2. Gradient-Free Adversarial Search ==========
    print("\n=== Gradient-Free Adversarial (minimizing detector score on corrupted) ===")

    adversarial = {}
    for ct in ctypes:
        for seed in seeds[:2]:
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_emb = extract_hidden(model, processor, corrupt_img, prompt)
            base_dist = cosine_dist(corrupt_emb, centroid)

            # Random search: try to find perturbation that minimizes cosine dist
            best_dist = base_dist
            best_delta = None
            n_trials = 50

            for trial in range(n_trials):
                rng = np.random.RandomState(seed * 1000 + trial)
                arr = np.array(corrupt_img).copy()

                # Random patch perturbation (32x32)
                py, px = rng.randint(0, 192), rng.randint(0, 192)
                delta = rng.randint(-20, 21, (32, 32, 3))
                arr[py:py+32, px:px+32] = np.clip(
                    arr[py:py+32, px:px+32].astype(np.int16) + delta, 0, 255
                ).astype(np.uint8)

                pert_img = Image.fromarray(arr)
                emb = extract_hidden(model, processor, pert_img, prompt)
                d = cosine_dist(emb, centroid)

                if d < best_dist:
                    best_dist = d
                    best_delta = (py, px)

            key = f"{ct}_seed{seed}"
            adversarial[key] = {
                'base_dist': float(base_dist),
                'best_dist': float(best_dist),
                'reduction': float(base_dist - best_dist),
                'reduction_pct': float((base_dist - best_dist) / base_dist * 100),
                'still_detected': best_dist > threshold,
                'n_trials': n_trials,
            }
            print(f"  {ct}_s{seed}: base={base_dist:.6f}, best={best_dist:.6f}, "
                  f"reduction={((base_dist-best_dist)/base_dist*100):.1f}%, "
                  f"{'DETECTED' if best_dist > threshold else 'EVADED'}")

    results['adversarial'] = adversarial

    # ========== 3. Counter-Perturbation: Corrupt + "Fix" ==========
    print("\n=== Counter-Perturbation ===")

    counter = {}
    for ct in ctypes:
        seed = seeds[0]
        corrupt_img = apply_corruption(images[seed], ct, 0.5)
        corrupt_arr = np.array(corrupt_img).astype(np.float32)
        clean_arr = np.array(images[seed]).astype(np.float32)

        # Try mixing corrupt with clean at various ratios
        for mix in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            mixed = ((1 - mix) * corrupt_arr + mix * clean_arr)
            mixed_img = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))
            emb = extract_hidden(model, processor, mixed_img, prompt)
            d = cosine_dist(emb, centroid)

            key = f"{ct}_mix{mix}"
            counter[key] = {
                'mix_ratio': mix,
                'dist': float(d),
                'detected': d > threshold,
            }

        # Find minimum mix to evade
        evade_mix = None
        for mix in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            if counter[f"{ct}_mix{mix}"]['dist'] <= threshold:
                evade_mix = mix
                break

        print(f"  {ct}: evade_mix={evade_mix} "
              f"(dist@0.5mix={counter[f'{ct}_mix0.5']['dist']:.6f})")

    results['counter_perturbation'] = counter

    # ========== 4. Patch Attack on Corrupted Images ==========
    print("\n=== Adversarial Patch Attack ===")

    patch_attack = {}
    for ct in ctypes:
        seed = seeds[0]
        corrupt_img = apply_corruption(images[seed], ct, 0.5)
        corrupt_arr = np.array(corrupt_img).copy()
        base_dist = cosine_dist(extract_hidden(model, processor, corrupt_img, prompt), centroid)

        best_dist = base_dist
        for patch_size in [16, 32, 64, 112]:
            # Place clean patch on corrupted image
            clean_arr = np.array(images[seed])
            patched = corrupt_arr.copy()
            cy, cx = 112 - patch_size//2, 112 - patch_size//2
            patched[cy:cy+patch_size, cx:cx+patch_size] = \
                clean_arr[cy:cy+patch_size, cx:cx+patch_size]

            patched_img = Image.fromarray(patched)
            emb = extract_hidden(model, processor, patched_img, prompt)
            d = cosine_dist(emb, centroid)

            patch_attack[f"{ct}_patch{patch_size}"] = {
                'patch_size': patch_size,
                'patch_area_pct': float((patch_size * patch_size) / (224 * 224) * 100),
                'dist': float(d),
                'base_dist': float(base_dist),
                'reduction_pct': float((base_dist - d) / base_dist * 100),
                'evaded': d <= threshold,
            }
            if d < best_dist:
                best_dist = d

        evade_patch = None
        for ps in [16, 32, 64, 112]:
            if patch_attack[f"{ct}_patch{ps}"]['evaded']:
                evade_patch = ps
                break

        print(f"  {ct}: base={base_dist:.6f}, evade_patch_size={evade_patch}")

    results['patch_attack'] = patch_attack

    # ========== 5. Evasion Difficulty Quantification ==========
    print("\n=== Evasion Difficulty ===")

    difficulty = {}
    for ct in ctypes:
        # How much of the image needs to be "fixed" to evade detection?
        seed = seeds[0]
        corrupt_img = apply_corruption(images[seed], ct, 0.5)
        corrupt_arr = np.array(corrupt_img).copy()
        clean_arr = np.array(images[seed])
        base_dist = cosine_dist(extract_hidden(model, processor, corrupt_img, prompt), centroid)

        for frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
            n_pixels = int(224 * 224 * frac)
            rng = np.random.RandomState(42)
            fixed = corrupt_arr.copy()
            for _ in range(n_pixels):
                y = rng.randint(0, 224)
                x_pos = rng.randint(0, 224)
                fixed[y, x_pos] = clean_arr[y, x_pos]

            fixed_img = Image.fromarray(fixed)
            emb = extract_hidden(model, processor, fixed_img, prompt)
            d = cosine_dist(emb, centroid)

            difficulty[f"{ct}_frac{frac}"] = {
                'fraction_fixed': frac,
                'dist': float(d),
                'base_dist': float(base_dist),
                'evaded': d <= threshold,
            }

        # Find minimum fraction to evade
        evade_frac = None
        for frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
            if difficulty[f"{ct}_frac{frac}"]['evaded']:
                evade_frac = frac
                break

        print(f"  {ct}: min_fix_fraction={evade_frac}")

    results['evasion_difficulty'] = difficulty

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/detector_adversarial_{ts}.json"
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
