#!/usr/bin/env python3
"""Experiment 404: Input Resolution Sensitivity

How does detection performance change with non-standard input resolutions?
Tests whether the detector works when images are pre-processed at different sizes.

Tests:
1. Various input resolutions (112, 160, 224, 320, 448)
2. Aspect ratio variations (16:9, 4:3, 1:1)
3. Resolution-dependent corruption sensitivity
4. Crop regions (center vs corner)
5. Resolution and detection margin correlation
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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']
    results = {}

    # === Test 1: Various input resolutions (square) ===
    print("=== Resolution Sensitivity ===")
    resolutions = [56, 112, 160, 224, 320, 448]
    res_results = {}

    for res in resolutions:
        print(f"\n  Resolution: {res}x{res}")
        scenes = []
        for seed in [42, 123, 456, 789, 999]:
            scenes.append(Image.fromarray(
                np.random.RandomState(seed).randint(0, 255, (res, res, 3), dtype=np.uint8)))

        # Clean embeddings
        clean_embs = []
        for scene in scenes:
            emb = extract_hidden(model, processor, scene, prompt)
            clean_embs.append(emb)
        centroid = np.mean(clean_embs, axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

        res_data = {"clean_mean_dist": float(np.mean(clean_dists))}
        for c in corruptions:
            corrupt_dists = []
            for scene in scenes:
                corrupted = apply_corruption(scene, c, 1.0)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = cosine_dist(emb, centroid)
                corrupt_dists.append(d)
            auroc = compute_auroc(clean_dists, corrupt_dists)
            res_data[c] = {
                "auroc": float(auroc),
                "mean_dist": float(np.mean(corrupt_dists))
            }
            print(f"    {c}: auroc={auroc:.3f}, dist={np.mean(corrupt_dists):.6f}")

        res_results[str(res)] = res_data

    results["resolution"] = res_results

    # === Test 2: Aspect ratios ===
    print("\n=== Aspect Ratio Sensitivity ===")
    aspect_results = {}
    aspect_ratios = {
        "1:1": (224, 224),
        "4:3": (224, 168),
        "16:9": (224, 126),
        "3:4": (168, 224),
        "9:16": (126, 224),
        "wide": (448, 112),
        "tall": (112, 448),
    }

    for name, (w, h) in aspect_ratios.items():
        print(f"\n  Aspect: {name} ({w}x{h})")
        scenes = []
        for seed in [42, 123, 456]:
            scenes.append(Image.fromarray(
                np.random.RandomState(seed).randint(0, 255, (h, w, 3), dtype=np.uint8)))

        clean_embs = []
        for scene in scenes:
            emb = extract_hidden(model, processor, scene, prompt)
            clean_embs.append(emb)
        centroid = np.mean(clean_embs, axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

        asp_data = {}
        for c in ['fog', 'night']:
            corrupt_dists = []
            for scene in scenes:
                corrupted = apply_corruption(scene, c, 1.0)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = cosine_dist(emb, centroid)
                corrupt_dists.append(d)
            auroc = compute_auroc(clean_dists, corrupt_dists)
            asp_data[c] = {"auroc": float(auroc), "mean_dist": float(np.mean(corrupt_dists))}
            print(f"    {c}: auroc={auroc:.3f}")

        aspect_results[name] = asp_data

    results["aspect_ratio"] = aspect_results

    # === Test 3: Crop regions ===
    print("\n=== Crop Region Sensitivity ===")
    crop_results = {}
    # Start with a larger image, crop different regions
    base_size = 448
    crop_size = 224

    for seed in [42]:
        base_img = Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (base_size, base_size, 3), dtype=np.uint8))

        crops = {
            "center": (112, 112, 336, 336),
            "top_left": (0, 0, 224, 224),
            "top_right": (224, 0, 448, 224),
            "bottom_left": (0, 224, 224, 448),
            "bottom_right": (224, 224, 448, 448),
        }

        for crop_name, box in crops.items():
            cropped = base_img.crop(box)

            # Get clean and fog embeddings
            clean_emb = extract_hidden(model, processor, cropped, prompt)
            fog_img = apply_corruption(cropped, 'fog', 1.0)
            fog_emb = extract_hidden(model, processor, fog_img, prompt)

            d = cosine_dist(clean_emb, fog_emb)
            crop_results[crop_name] = {"fog_dist": float(d)}
            print(f"  {crop_name}: fog_dist={d:.6f}")

    results["crop_regions"] = crop_results

    # === Test 4: Embedding stability across resolutions ===
    print("\n=== Cross-Resolution Embedding Stability ===")
    ref_scene_224 = Image.fromarray(
        np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))
    ref_emb = extract_hidden(model, processor, ref_scene_224, prompt)

    stability = {}
    for res in [56, 112, 160, 320, 448]:
        # Resize the same conceptual image
        resized = ref_scene_224.resize((res, res), Image.BILINEAR)
        emb = extract_hidden(model, processor, resized, prompt)
        d = cosine_dist(ref_emb, emb)
        stability[str(res)] = float(d)
        print(f"  224->{res}: cosine_dist={d:.6f}")

    results["cross_resolution_stability"] = stability

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/resolution_sensitivity_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
