#!/usr/bin/env python3
"""Experiment 380: Embedding Stability Under Benign Variation

Does the detector trigger on natural, non-corrupted variations?
1. Random seed diversity: how much do different scenes vary?
2. Small pixel perturbations: ±1, ±2, ±5 pixel values
3. Slight rotation/flip: minor geometric transforms
4. Color jitter: small random color shifts
5. Additive Gaussian noise at very low sigma
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

    # Generate images
    print("Generating images...")
    seeds = list(range(0, 2000, 100))[:20]
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
    print(f"  {len(seeds)} scenes, threshold={threshold:.6f}")

    # ========== 1. Scene Diversity ==========
    print("\n=== Scene Diversity ===")

    pairwise = []
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            pairwise.append(cosine_dist(clean_embs[seeds[i]], clean_embs[seeds[j]]))

    results['scene_diversity'] = {
        'mean_pairwise': float(np.mean(pairwise)),
        'std_pairwise': float(np.std(pairwise)),
        'max_pairwise': float(max(pairwise)),
        'min_pairwise': float(min(pairwise)),
        'n_pairs': len(pairwise),
        'threshold': float(threshold),
    }
    print(f"  Pairwise: mean={np.mean(pairwise):.6f}, max={max(pairwise):.6f}")

    # ========== 2. Small Pixel Perturbations ==========
    print("\n=== Small Pixel Perturbations ===")

    pixel_pert = {}
    for delta in [1, 2, 3, 5, 10, 20]:
        pert_dists = []
        for seed in seeds[:10]:
            arr = np.array(images[seed]).copy()
            rng = np.random.RandomState(seed + 999)
            noise = rng.randint(-delta, delta+1, arr.shape)
            perturbed = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            emb = extract_hidden(model, processor, Image.fromarray(perturbed), prompt)
            d = cosine_dist(emb, centroid)
            pert_dists.append(d)

        fpr = sum(1 for d in pert_dists if d > threshold) / len(pert_dists)
        pixel_pert[str(delta)] = {
            'mean_dist': float(np.mean(pert_dists)),
            'max_dist': float(max(pert_dists)),
            'false_positive_rate': float(fpr),
        }
        print(f"  delta=±{delta}: mean_dist={np.mean(pert_dists):.6f}, FPR={fpr:.2f}")

    results['pixel_perturbation'] = pixel_pert

    # ========== 3. Geometric Transforms ==========
    print("\n=== Geometric Transforms ===")

    geometric = {}
    for seed in seeds[:5]:
        # Horizontal flip
        flipped = images[seed].transpose(Image.FLIP_LEFT_RIGHT)
        emb = extract_hidden(model, processor, flipped, prompt)
        geometric[f"hflip_s{seed}"] = {
            'dist': float(cosine_dist(emb, centroid)),
            'detected': cosine_dist(emb, centroid) > threshold,
        }

        # Small rotation
        for angle in [1, 2, 5, 10]:
            rotated = images[seed].rotate(angle, fillcolor=(128, 128, 128))
            emb = extract_hidden(model, processor, rotated, prompt)
            geometric[f"rot{angle}_s{seed}"] = {
                'dist': float(cosine_dist(emb, centroid)),
                'detected': cosine_dist(emb, centroid) > threshold,
            }

    # Summarize
    hflip_dists = [geometric[k]['dist'] for k in geometric if 'hflip' in k]
    rot1_dists = [geometric[k]['dist'] for k in geometric if 'rot1_' in k]
    rot5_dists = [geometric[k]['dist'] for k in geometric if 'rot5_' in k]
    rot10_dists = [geometric[k]['dist'] for k in geometric if 'rot10_' in k]

    results['geometric'] = {
        'hflip_mean_dist': float(np.mean(hflip_dists)),
        'hflip_fpr': float(sum(1 for d in hflip_dists if d > threshold) / len(hflip_dists)),
        'rot1_mean_dist': float(np.mean(rot1_dists)),
        'rot1_fpr': float(sum(1 for d in rot1_dists if d > threshold) / len(rot1_dists)),
        'rot5_mean_dist': float(np.mean(rot5_dists)),
        'rot5_fpr': float(sum(1 for d in rot5_dists if d > threshold) / len(rot5_dists)),
        'rot10_mean_dist': float(np.mean(rot10_dists)),
        'rot10_fpr': float(sum(1 for d in rot10_dists if d > threshold) / len(rot10_dists)),
        'detail': geometric,
    }
    print(f"  hflip: FPR={results['geometric']['hflip_fpr']:.2f}")
    print(f"  rot1: FPR={results['geometric']['rot1_fpr']:.2f}")
    print(f"  rot5: FPR={results['geometric']['rot5_fpr']:.2f}")
    print(f"  rot10: FPR={results['geometric']['rot10_fpr']:.2f}")

    # ========== 4. Low-Sigma Gaussian Noise ==========
    print("\n=== Low-Sigma Gaussian Noise ===")

    gauss_noise = {}
    for sigma in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        noise_dists = []
        for seed in seeds[:10]:
            arr = np.array(images[seed]).astype(np.float32) / 255.0
            rng = np.random.RandomState(seed + 5555)
            noisy = arr + rng.randn(*arr.shape) * sigma
            noisy = np.clip(noisy, 0, 1)
            noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
            emb = extract_hidden(model, processor, noisy_img, prompt)
            d = cosine_dist(emb, centroid)
            noise_dists.append(d)

        fpr = sum(1 for d in noise_dists if d > threshold) / len(noise_dists)
        gauss_noise[str(sigma)] = {
            'mean_dist': float(np.mean(noise_dists)),
            'max_dist': float(max(noise_dists)),
            'false_positive_rate': float(fpr),
        }
        print(f"  sigma={sigma}: mean_dist={np.mean(noise_dists):.6f}, FPR={fpr:.2f}")

    results['gaussian_noise'] = gauss_noise

    # ========== 5. Color Jitter ==========
    print("\n=== Color Jitter ===")

    color_jitter = {}
    for jitter_range in [0.01, 0.02, 0.05, 0.1, 0.2]:
        jitter_dists = []
        for seed in seeds[:10]:
            arr = np.array(images[seed]).astype(np.float32) / 255.0
            rng = np.random.RandomState(seed + 8888)
            # Per-channel multiplicative jitter
            jitter = 1.0 + rng.uniform(-jitter_range, jitter_range, 3)
            jittered = np.clip(arr * jitter[np.newaxis, np.newaxis, :], 0, 1)
            jittered_img = Image.fromarray((jittered * 255).astype(np.uint8))
            emb = extract_hidden(model, processor, jittered_img, prompt)
            d = cosine_dist(emb, centroid)
            jitter_dists.append(d)

        fpr = sum(1 for d in jitter_dists if d > threshold) / len(jitter_dists)
        color_jitter[str(jitter_range)] = {
            'mean_dist': float(np.mean(jitter_dists)),
            'max_dist': float(max(jitter_dists)),
            'false_positive_rate': float(fpr),
        }
        print(f"  jitter=±{jitter_range}: mean_dist={np.mean(jitter_dists):.6f}, FPR={fpr:.2f}")

    results['color_jitter'] = color_jitter

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/benign_variation_{ts}.json"
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
