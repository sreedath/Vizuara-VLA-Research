#!/usr/bin/env python3
"""Experiment 399: Novel Corruption Types Detection

Tests detection on corruption types NOT in the original 4 (fog/night/noise/blur):
1. Rain simulation (diagonal streaks + darkening)
2. Snow simulation (white dots + brightness)
3. Occlusion (random rectangular patches)
4. Color shift (hue/saturation manipulation)
5. Compression artifacts (JPEG-like quantization)
6. Sensor dead pixels (random stuck pixels)

Tests whether the calibrated detector generalizes to unseen corruption types.
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

def apply_rain(image, severity=1.0, seed=42):
    """Simulate rain with diagonal streaks and darkening."""
    arr = np.array(image).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed)
    h, w = arr.shape[:2]
    # Darken
    arr = arr * (1.0 - 0.3 * severity)
    # Add diagonal rain streaks
    n_drops = int(500 * severity)
    for _ in range(n_drops):
        x, y = rng.randint(0, w), rng.randint(0, h)
        length = rng.randint(5, 20)
        for k in range(length):
            ny, nx = y + k, x + k // 2
            if 0 <= ny < h and 0 <= nx < w:
                arr[ny, nx] = np.clip(arr[ny, nx] + 0.3, 0, 1)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def apply_snow(image, severity=1.0, seed=42):
    """Simulate snow with white dots and brightness increase."""
    arr = np.array(image).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed)
    # Brighten
    arr = arr * (1.0 - 0.2 * severity) + 0.2 * severity
    # Add snow particles
    n_flakes = int(1000 * severity)
    h, w = arr.shape[:2]
    for _ in range(n_flakes):
        x, y = rng.randint(0, w), rng.randint(0, h)
        size = rng.randint(1, 3)
        for dy in range(-size, size+1):
            for dx in range(-size, size+1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    arr[ny, nx] = 1.0
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def apply_occlusion(image, severity=1.0, seed=42):
    """Add random rectangular occlusions (black patches)."""
    arr = np.array(image).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed)
    h, w = arr.shape[:2]
    n_patches = int(3 + 5 * severity)
    for _ in range(n_patches):
        pw = rng.randint(10, int(40 * severity) + 10)
        ph = rng.randint(10, int(40 * severity) + 10)
        x, y = rng.randint(0, w - pw), rng.randint(0, h - ph)
        arr[y:y+ph, x:x+pw] = 0.0
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def apply_color_shift(image, severity=1.0, seed=42):
    """Shift color channels independently."""
    arr = np.array(image).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed)
    shifts = rng.uniform(-0.3 * severity, 0.3 * severity, 3)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] + shifts[c]
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def apply_jpeg_artifact(image, severity=1.0, seed=42):
    """Simulate JPEG compression artifacts via block quantization."""
    arr = np.array(image).astype(np.float32) / 255.0
    block_size = max(2, int(16 * severity))
    h, w = arr.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = arr[y:y+block_size, x:x+block_size]
            arr[y:y+block_size, x:x+block_size] = block.mean(axis=(0, 1))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def apply_dead_pixels(image, severity=1.0, seed=42):
    """Simulate sensor dead pixels (stuck at 0 or 255)."""
    arr = np.array(image).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed)
    h, w = arr.shape[:2]
    n_dead = int(500 * severity)
    for _ in range(n_dead):
        x, y = rng.randint(0, w), rng.randint(0, h)
        if rng.random() > 0.5:
            arr[y, x] = 1.0  # stuck white
        else:
            arr[y, x] = 0.0  # stuck black
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

NOVEL_CORRUPTIONS = {
    'rain': apply_rain,
    'snow': apply_snow,
    'occlusion': apply_occlusion,
    'color_shift': apply_color_shift,
    'jpeg': apply_jpeg_artifact,
    'dead_pixels': apply_dead_pixels,
}

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

    results = {}

    # Extract clean embeddings and compute centroid
    print("Clean embeddings...")
    clean_embs = []
    for scene in scenes:
        emb = extract_hidden(model, processor, scene, prompt)
        clean_embs.append(emb)
    clean_centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, clean_centroid) for e in clean_embs]
    threshold = max(clean_dists) * 1.5  # Simple threshold
    print(f"  Clean dists: {[f'{d:.6f}' for d in clean_dists]}")
    print(f"  Threshold: {threshold:.6f}")
    results["clean_dists"] = [float(d) for d in clean_dists]
    results["threshold"] = float(threshold)

    # Test each novel corruption
    for cname, cfunc in NOVEL_CORRUPTIONS.items():
        print(f"\n=== {cname} ===")
        corrupt_data = {"by_severity": {}, "by_scene": {}}

        all_corrupt_dists = []
        for sev in [0.3, 0.5, 0.7, 1.0]:
            sev_dists = []
            for si, scene in enumerate(scenes):
                corrupted = cfunc(scene, severity=sev, seed=42+si)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = cosine_dist(emb, clean_centroid)
                sev_dists.append(d)
                all_corrupt_dists.append(d)

            auroc = compute_auroc(clean_dists, sev_dists)
            detected = sum(1 for d in sev_dists if d > threshold)

            corrupt_data["by_severity"][str(sev)] = {
                "dists": [float(d) for d in sev_dists],
                "mean_dist": float(np.mean(sev_dists)),
                "auroc": float(auroc),
                "detection_rate": detected / len(sev_dists)
            }
            print(f"  sev={sev}: mean={np.mean(sev_dists):.6f}, auroc={auroc:.3f}, det={detected}/5")

        # Overall AUROC
        overall_auroc = compute_auroc(clean_dists, all_corrupt_dists)
        corrupt_data["overall_auroc"] = float(overall_auroc)
        corrupt_data["mean_dist"] = float(np.mean(all_corrupt_dists))
        corrupt_data["all_detected"] = all(d > threshold for d in all_corrupt_dists)
        print(f"  Overall AUROC: {overall_auroc:.3f}")

        results[cname] = corrupt_data

    # Cross-comparison: novel vs original corruptions
    print("\n=== Comparison with Original Corruptions ===")
    from PIL import ImageFilter
    def apply_original(image, ctype, severity=1.0):
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

    original_corruptions = ['fog', 'night', 'noise', 'blur']
    comparison = {}
    for oc in original_corruptions:
        oc_dists = []
        for scene in scenes:
            corrupted = apply_original(scene, oc, 1.0)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = cosine_dist(emb, clean_centroid)
            oc_dists.append(d)
        auroc = compute_auroc(clean_dists, oc_dists)
        comparison[oc] = {
            "mean_dist": float(np.mean(oc_dists)),
            "auroc": float(auroc)
        }
        print(f"  {oc}: mean={np.mean(oc_dists):.6f}, auroc={auroc:.3f}")

    results["original_comparison"] = comparison

    # Ranking all corruptions by mean distance
    all_corruptions = {}
    for c in NOVEL_CORRUPTIONS:
        all_corruptions[c] = results[c]["mean_dist"]
    for c in original_corruptions:
        all_corruptions[c] = comparison[c]["mean_dist"]

    ranked = sorted(all_corruptions.items(), key=lambda x: x[1], reverse=True)
    print("\n=== Corruption Severity Ranking ===")
    for i, (name, dist) in enumerate(ranked):
        tag = "NOVEL" if name in NOVEL_CORRUPTIONS else "ORIGINAL"
        print(f"  {i+1}. {name} ({tag}): {dist:.6f}")
    results["severity_ranking"] = [{"name": n, "dist": float(d), "novel": n in NOVEL_CORRUPTIONS}
                                    for n, d in ranked]

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/novel_corruptions_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
