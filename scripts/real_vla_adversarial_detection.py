#!/usr/bin/env python3
"""Experiment 403: Adversarial Perturbation Detection

Can the detector identify adversarial perturbations (small, targeted pixel changes)?
Tests detection on gradient-free adversarial-like perturbations.

Tests:
1. L-infinity bounded uniform noise at various epsilon
2. Targeted single-pixel attacks (one pixel changed drastically)
3. Patch-based perturbations (small patches of random noise)
4. Spatial frequency attacks (high-freq only vs low-freq only)
5. Imperceptible perturbations vs perceptible corruptions
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
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

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

    scenes = []
    for seed in [42, 123, 456, 789, 999]:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    # Clean centroid
    print("Clean baseline...")
    clean_embs = []
    for scene in scenes:
        emb = extract_hidden(model, processor, scene, prompt)
        clean_embs.append(emb)
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]
    threshold = max(clean_dists) * 1.5
    print(f"  Threshold: {threshold:.6f}")

    results = {"clean_dists": [float(d) for d in clean_dists], "threshold": float(threshold)}

    # === Test 1: L-inf uniform noise at various epsilon ===
    print("\n=== L-inf Uniform Noise ===")
    linf_results = {}
    for eps in [1, 2, 4, 8, 16, 32, 64, 128]:
        eps_dists = []
        for si, scene in enumerate(scenes):
            arr = np.array(scene).astype(np.float32)
            rng = np.random.RandomState(42 + si)
            noise = rng.uniform(-eps, eps, arr.shape)
            perturbed = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(perturbed)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, centroid)
            eps_dists.append(d)

        auroc = compute_auroc(clean_dists, eps_dists)
        detected = sum(1 for d in eps_dists if d > threshold)
        linf_results[str(eps)] = {
            "mean_dist": float(np.mean(eps_dists)),
            "auroc": float(auroc),
            "detected": f"{detected}/5",
            "l2_norm_approx": float(eps * np.sqrt(224*224*3) / 255.0)
        }
        print(f"  eps={eps}/255: mean={np.mean(eps_dists):.6f}, auroc={auroc:.3f}, det={detected}/5")

    results["linf_noise"] = linf_results

    # === Test 2: Single pixel attack ===
    print("\n=== Single Pixel Attack ===")
    pixel_results = {}
    for n_pixels in [1, 5, 10, 50, 100, 500]:
        px_dists = []
        for si, scene in enumerate(scenes):
            arr = np.array(scene).copy()
            rng = np.random.RandomState(42 + si)
            h, w = arr.shape[:2]
            for _ in range(n_pixels):
                y, x = rng.randint(0, h), rng.randint(0, w)
                arr[y, x] = rng.choice([0, 255], 3)
            img = Image.fromarray(arr)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, centroid)
            px_dists.append(d)

        auroc = compute_auroc(clean_dists, px_dists)
        detected = sum(1 for d in px_dists if d > threshold)
        pixel_results[str(n_pixels)] = {
            "mean_dist": float(np.mean(px_dists)),
            "auroc": float(auroc),
            "detected": f"{detected}/5",
            "pct_pixels": float(n_pixels / (224*224) * 100)
        }
        print(f"  {n_pixels} pixels ({n_pixels/(224*224)*100:.3f}%): "
              f"mean={np.mean(px_dists):.6f}, auroc={auroc:.3f}, det={detected}/5")

    results["pixel_attack"] = pixel_results

    # === Test 3: Patch perturbation ===
    print("\n=== Patch Perturbation ===")
    patch_results = {}
    for patch_size in [4, 8, 16, 32, 64, 112]:
        patch_dists = []
        for si, scene in enumerate(scenes):
            arr = np.array(scene).copy()
            rng = np.random.RandomState(42 + si)
            h, w = arr.shape[:2]
            # Place random noise patch in center
            y = (h - patch_size) // 2
            x = (w - patch_size) // 2
            arr[y:y+patch_size, x:x+patch_size] = rng.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, centroid)
            patch_dists.append(d)

        auroc = compute_auroc(clean_dists, patch_dists)
        detected = sum(1 for d in patch_dists if d > threshold)
        pct = (patch_size / 224) ** 2 * 100
        patch_results[str(patch_size)] = {
            "mean_dist": float(np.mean(patch_dists)),
            "auroc": float(auroc),
            "detected": f"{detected}/5",
            "pct_area": float(pct)
        }
        print(f"  {patch_size}x{patch_size} ({pct:.1f}%): "
              f"mean={np.mean(patch_dists):.6f}, auroc={auroc:.3f}, det={detected}/5")

    results["patch_perturbation"] = patch_results

    # === Test 4: Frequency-domain attacks ===
    print("\n=== Frequency Domain Attacks ===")
    freq_results = {}
    for freq_type in ['high', 'low']:
        freq_dists = []
        for si, scene in enumerate(scenes):
            arr = np.array(scene).astype(np.float32) / 255.0
            rng = np.random.RandomState(42 + si)

            if freq_type == 'high':
                # High-frequency noise (checkerboard pattern)
                checker = np.zeros_like(arr)
                for c in range(3):
                    for y in range(224):
                        for x in range(224):
                            if (y + x) % 2 == 0:
                                checker[y, x, c] = 0.2
                            else:
                                checker[y, x, c] = -0.2
                arr = np.clip(arr + checker, 0, 1)
            else:
                # Low-frequency perturbation (smooth gradient)
                grad = np.zeros_like(arr)
                for c in range(3):
                    shift = rng.uniform(-0.3, 0.3)
                    for y in range(224):
                        grad[y, :, c] = shift * (y / 224)
                arr = np.clip(arr + grad, 0, 1)

            img = Image.fromarray((arr * 255).astype(np.uint8))
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, centroid)
            freq_dists.append(d)

        auroc = compute_auroc(clean_dists, freq_dists)
        detected = sum(1 for d in freq_dists if d > threshold)
        freq_results[freq_type] = {
            "mean_dist": float(np.mean(freq_dists)),
            "auroc": float(auroc),
            "detected": f"{detected}/5"
        }
        print(f"  {freq_type}-freq: mean={np.mean(freq_dists):.6f}, auroc={auroc:.3f}, det={detected}/5")

    results["frequency_attacks"] = freq_results

    # === Test 5: Comparison with perceptible corruptions ===
    print("\n=== Imperceptible vs Perceptible ===")
    from PIL import ImageFilter as IF

    def apply_corruption(image, ctype, severity=1.0):
        arr = np.array(image).astype(np.float32) / 255.0
        if ctype == 'fog':
            arr = arr * (1 - 0.6 * severity) + 0.6 * severity
        elif ctype == 'night':
            arr = arr * max(0.01, 1.0 - 0.95 * severity)
        elif ctype == 'blur':
            return image.filter(IF.GaussianBlur(radius=10 * severity))
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    comparison = {}
    for c in ['fog', 'night', 'blur']:
        c_dists = []
        for scene in scenes:
            corrupted = apply_corruption(scene, c, 1.0)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = cosine_dist(emb, centroid)
            c_dists.append(d)
        comparison[c] = float(np.mean(c_dists))

    # Find the epsilon threshold for detection
    detection_boundary = None
    for eps in [1, 2, 4, 8, 16, 32, 64, 128]:
        if linf_results[str(eps)]["auroc"] >= 0.95:
            detection_boundary = eps
            break

    results["comparison"] = comparison
    results["detection_boundary_eps"] = detection_boundary

    print(f"  Detection boundary: eps={detection_boundary}/255")
    print(f"  Fog dist: {comparison['fog']:.6f}")
    print(f"  Night dist: {comparison['night']:.6f}")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/adversarial_detection_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
