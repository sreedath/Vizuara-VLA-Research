#!/usr/bin/env python3
"""Experiment 419: Embedding Space Interpolation

Studies the geometry between clean and corrupted embeddings by interpolating
in image space and checking whether intermediate points correspond to
meaningful intermediate corruption states. Tests linearity of the clean→corrupt
trajectory and whether the OOD detection boundary is sharp or gradual.

Tests:
1. Image-space interpolation between clean and corrupt (21 alpha levels)
2. Detection boundary sharpness: at what interpolation fraction does AUROC drop?
3. Embedding-space linearity: does image interpolation produce linear embedding paths?
4. Cross-corruption interpolation: what's "between" fog and night?
5. Convexity: are all clean-corrupt mixtures detected?
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
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
    corruptions = ['fog', 'night', 'noise', 'blur']

    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Extract embeddings
    print("Extracting embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    corrupt_embs = {}
    for c in corruptions:
        corrupt_embs[c] = [extract_hidden(model, processor, apply_corruption(s, c), prompt) for s in scenes]

    results = {}

    # === Test 1: Image-space interpolation ===
    print("\n=== Image-Space Interpolation ===")
    alphas = np.linspace(0, 1, 21)
    interp_results = {}

    for c in corruptions:
        print(f"  {c}:")
        curve = {}
        for alpha in alphas:
            dists = []
            for i, s in enumerate(scenes):
                clean_arr = np.array(s).astype(np.float32) / 255.0
                corrupt_img = apply_corruption(s, c)
                corrupt_arr = np.array(corrupt_img).astype(np.float32) / 255.0
                interp_arr = (1 - alpha) * clean_arr + alpha * corrupt_arr
                interp_img = Image.fromarray((np.clip(interp_arr, 0, 1) * 255).astype(np.uint8))
                emb = extract_hidden(model, processor, interp_img, prompt)
                dists.append(cosine_dist(emb, centroid))

            auroc = float(compute_auroc(clean_dists, dists)) if alpha > 0 else 0.5
            curve[f"{alpha:.2f}"] = {
                "mean_dist": float(np.mean(dists)),
                "std_dist": float(np.std(dists)),
                "auroc": auroc,
            }
            print(f"    α={alpha:.2f}: dist={np.mean(dists):.6f}, AUROC={auroc:.4f}")
        interp_results[c] = curve
    results["image_interpolation"] = interp_results

    # === Test 2: Detection boundary ===
    print("\n=== Detection Boundary ===")
    boundary = {}
    for c in corruptions:
        first_high = None
        first_perfect = None
        for alpha in alphas:
            auroc = interp_results[c][f"{alpha:.2f}"]["auroc"]
            if auroc >= 0.8 and first_high is None:
                first_high = float(alpha)
            if auroc >= 1.0 and first_perfect is None:
                first_perfect = float(alpha)
        boundary[c] = {
            "alpha_auroc_0.8": first_high,
            "alpha_auroc_1.0": first_perfect,
            "boundary_width": (first_perfect - first_high) if (first_high and first_perfect) else None,
        }
        print(f"  {c}: AUROC>0.8 at α={first_high}, AUROC=1.0 at α={first_perfect}")
    results["detection_boundary"] = boundary

    # === Test 3: Linearity test ===
    print("\n=== Embedding Linearity ===")
    linearity = {}
    for c in corruptions:
        errors = []
        for i in range(len(scenes)):
            clean_e = clean_embs[i]
            corrupt_e = corrupt_embs[c][i]
            for alpha in [0.25, 0.5, 0.75]:
                clean_arr = np.array(scenes[i]).astype(np.float32) / 255.0
                corrupt_arr = np.array(apply_corruption(scenes[i], c)).astype(np.float32) / 255.0
                interp_arr = (1 - alpha) * clean_arr + alpha * corrupt_arr
                interp_img = Image.fromarray((np.clip(interp_arr, 0, 1) * 255).astype(np.uint8))
                actual_emb = extract_hidden(model, processor, interp_img, prompt)
                linear_emb = (1 - alpha) * clean_e + alpha * corrupt_e
                error = cosine_dist(actual_emb, linear_emb)
                errors.append(error)

        linearity[c] = {
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "std_error": float(np.std(errors)),
        }
        print(f"  {c}: mean_error={np.mean(errors):.6f}, max_error={np.max(errors):.6f}")
    results["linearity"] = linearity

    # === Test 4: Cross-corruption interpolation ===
    print("\n=== Cross-Corruption Interpolation ===")
    cross_interp = {}
    for c1 in corruptions:
        for c2 in corruptions:
            if c1 >= c2:
                continue
            mid_dists = []
            for i, s in enumerate(scenes):
                arr1 = np.array(apply_corruption(s, c1)).astype(np.float32) / 255.0
                arr2 = np.array(apply_corruption(s, c2)).astype(np.float32) / 255.0
                mid_arr = 0.5 * arr1 + 0.5 * arr2
                mid_img = Image.fromarray((np.clip(mid_arr, 0, 1) * 255).astype(np.uint8))
                emb = extract_hidden(model, processor, mid_img, prompt)
                mid_dists.append(cosine_dist(emb, centroid))

            key = f"{c1}_{c2}"
            cross_interp[key] = {
                "midpoint_mean_dist": float(np.mean(mid_dists)),
                "c1_mean_dist": float(np.mean([cosine_dist(corrupt_embs[c1][i], centroid) for i in range(len(scenes))])),
                "c2_mean_dist": float(np.mean([cosine_dist(corrupt_embs[c2][i], centroid) for i in range(len(scenes))])),
                "auroc": float(compute_auroc(clean_dists, mid_dists)),
            }
            print(f"  {key}: mid_dist={np.mean(mid_dists):.6f}, AUROC={cross_interp[key]['auroc']:.4f}")
    results["cross_interpolation"] = cross_interp

    # === Test 5: Convexity ===
    print("\n=== Convexity Test ===")
    convexity = {}
    for c in corruptions:
        all_detected = True
        min_auroc = 1.0
        for alpha in [0.3, 0.5, 0.7]:
            dists = []
            for i, s in enumerate(scenes):
                clean_arr = np.array(s).astype(np.float32) / 255.0
                corrupt_arr = np.array(apply_corruption(s, c)).astype(np.float32) / 255.0
                mix_arr = (1 - alpha) * clean_arr + alpha * corrupt_arr
                mix_img = Image.fromarray((np.clip(mix_arr, 0, 1) * 255).astype(np.uint8))
                emb = extract_hidden(model, processor, mix_img, prompt)
                dists.append(cosine_dist(emb, centroid))
            auroc = compute_auroc(clean_dists, dists)
            if auroc < 1.0:
                all_detected = False
            min_auroc = min(min_auroc, auroc)
        convexity[c] = {
            "all_mixtures_detected": all_detected,
            "min_mixture_auroc": float(min_auroc),
        }
        print(f"  {c}: all_detected={all_detected}, min_auroc={min_auroc:.4f}")
    results["convexity"] = convexity

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/embedding_interpolation_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
