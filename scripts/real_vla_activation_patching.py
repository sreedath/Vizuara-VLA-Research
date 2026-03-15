#!/usr/bin/env python3
"""Experiment 436: Activation Patching — Spatial Importance for Detection

Tests which spatial regions of the input image contribute most to
corruption detection by systematically occluding image patches and
measuring the change in cosine distance.

Tests:
1. Grid-based occlusion (divide image into patches, occlude each)
2. Center vs periphery importance
3. Progressive masking (how much can we mask and still detect?)
4. Corruption-specific spatial importance
5. Random occlusion robustness
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

def occlude_patch(image, row, col, grid_size, fill='gray'):
    """Occlude a grid patch with gray."""
    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    ph, pw = h // grid_size, w // grid_size
    y0, y1 = row * ph, (row + 1) * ph
    x0, x1 = col * pw, (col + 1) * pw
    if fill == 'gray':
        arr[y0:y1, x0:x1] = 128
    elif fill == 'black':
        arr[y0:y1, x0:x1] = 0
    elif fill == 'random':
        arr[y0:y1, x0:x1] = np.random.RandomState(42).randint(0, 255, (y1-y0, x1-x0, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Extract clean embeddings
    print("Extracting clean embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    results = {"n_scenes": len(scenes)}

    # === Test 1: Grid-based occlusion importance ===
    print("\n=== Grid-Based Occlusion Importance (4x4) ===")
    grid_size = 4
    grid_importance = {}
    # Use scene 0 with fog as test case
    test_scene = scenes[0]
    fog_scene = apply_corruption(test_scene, 'fog')
    fog_emb = extract_hidden(model, processor, fog_scene, prompt)
    fog_dist_full = cosine_dist(fog_emb, centroid)

    for row in range(grid_size):
        for col in range(grid_size):
            # Occlude this patch in the fog image
            occluded = occlude_patch(fog_scene, row, col, grid_size)
            emb = extract_hidden(model, processor, occluded, prompt)
            dist = cosine_dist(emb, centroid)
            key = f"{row}_{col}"
            grid_importance[key] = {
                "distance": float(dist),
                "distance_change": float(dist - fog_dist_full),
                "pct_change": float((dist - fog_dist_full) / fog_dist_full * 100) if fog_dist_full > 0 else 0,
            }
            print(f"  [{row},{col}]: dist={dist:.6f} (Δ={dist - fog_dist_full:.6f})")
    results["grid_importance"] = grid_importance
    results["fog_dist_full"] = float(fog_dist_full)

    # === Test 2: Center vs periphery ===
    print("\n=== Center vs Periphery ===")
    center_vs_periph = {}
    for c in ['fog', 'night', 'blur']:
        for s_idx, s in enumerate(scenes[:3]):
            corr_img = apply_corruption(s, c)
            # Center: inner 50%
            arr_center = np.array(corr_img).copy()
            arr_center[:56, :, :] = 128
            arr_center[168:, :, :] = 128
            arr_center[:, :56, :] = 128
            arr_center[:, 168:, :] = 128
            center_emb = extract_hidden(model, processor, Image.fromarray(arr_center), prompt)
            center_dist = cosine_dist(center_emb, centroid)

            # Periphery: outer ring
            arr_periph = np.array(corr_img).copy()
            arr_periph[56:168, 56:168, :] = 128
            periph_emb = extract_hidden(model, processor, Image.fromarray(arr_periph), prompt)
            periph_dist = cosine_dist(periph_emb, centroid)

            full_emb = extract_hidden(model, processor, corr_img, prompt)
            full_dist = cosine_dist(full_emb, centroid)

            key = f"{c}_scene{s_idx}"
            center_vs_periph[key] = {
                "center_dist": float(center_dist),
                "periphery_dist": float(periph_dist),
                "full_dist": float(full_dist),
            }

    # Aggregate
    for c in ['fog', 'night', 'blur']:
        center_mean = np.mean([center_vs_periph[f"{c}_scene{i}"]["center_dist"] for i in range(3)])
        periph_mean = np.mean([center_vs_periph[f"{c}_scene{i}"]["periphery_dist"] for i in range(3)])
        full_mean = np.mean([center_vs_periph[f"{c}_scene{i}"]["full_dist"] for i in range(3)])
        print(f"  {c}: center={center_mean:.6f}, periphery={periph_mean:.6f}, full={full_mean:.6f}")
    results["center_vs_periphery"] = center_vs_periph

    # === Test 3: Progressive masking ===
    print("\n=== Progressive Masking ===")
    mask_results = {}
    rng = np.random.RandomState(42)
    mask_fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]

    for c in ['fog', 'night']:
        per_frac = {}
        for frac in mask_fractions:
            auroc_vals = []
            for s in scenes:
                corr_img = apply_corruption(s, c)
                arr = np.array(corr_img).copy()
                # Randomly mask fraction of pixels
                if frac > 0:
                    mask = rng.random(arr.shape[:2]) < frac
                    arr[mask] = 128
                emb = extract_hidden(model, processor, Image.fromarray(arr), prompt)
                auroc_vals.append(float(cosine_dist(emb, centroid)))

            auroc = float(compute_auroc(clean_dists, auroc_vals))
            per_frac[str(frac)] = {"auroc": auroc}
        mask_results[c] = per_frac
        print(f"  {c}: 0% mask AUROC={per_frac['0.0']['auroc']:.4f}, "
              f"50% mask AUROC={per_frac['0.5']['auroc']:.4f}, "
              f"90% mask AUROC={per_frac['0.9']['auroc']:.4f}")
    results["progressive_masking"] = mask_results

    # === Test 4: Corruption-specific spatial maps (7x7 grid) ===
    print("\n=== Corruption-Specific Spatial Maps (7x7) ===")
    spatial_maps = {}
    test_scene = scenes[0]
    grid7 = 7
    for c in ['fog', 'night']:
        corr_img = apply_corruption(test_scene, c)
        full_emb = extract_hidden(model, processor, corr_img, prompt)
        full_dist = float(cosine_dist(full_emb, centroid))

        patch_map = {}
        for row in range(grid7):
            for col in range(grid7):
                occluded = occlude_patch(corr_img, row, col, grid7)
                emb = extract_hidden(model, processor, occluded, prompt)
                dist = float(cosine_dist(emb, centroid))
                patch_map[f"{row}_{col}"] = float(dist - full_dist)

        spatial_maps[c] = {"full_dist": full_dist, "patch_changes": patch_map}
        # Find most important patch
        max_key = max(patch_map, key=lambda k: abs(patch_map[k]))
        print(f"  {c}: most important patch={max_key}, Δ={patch_map[max_key]:.6f}")
    results["spatial_maps"] = spatial_maps

    out_path = "/workspace/Vizuara-VLA-Research/experiments/activation_patching_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
