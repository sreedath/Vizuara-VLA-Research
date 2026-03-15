#!/usr/bin/env python3
"""Experiment 295: Semantic Content Analysis
Tests detection across semantically different image types:
1. Random noise images (current approach)
2. Solid color images
3. Gradient images
4. Structured patterns (checkerboard, stripes)
5. Natural-looking scenes (generated programmatically)
6. Cross-semantic detection: calibrate on one type, test on another
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def create_test_images():
    """Create semantically diverse test images."""
    images = {}

    # Random noise
    rng = np.random.RandomState(42)
    images["random_noise"] = Image.fromarray(rng.randint(50, 200, (224, 224, 3), dtype=np.uint8))

    # Solid colors
    images["solid_gray"] = Image.new('RGB', (224, 224), (128, 128, 128))
    images["solid_red"] = Image.new('RGB', (224, 224), (200, 50, 50))
    images["solid_green"] = Image.new('RGB', (224, 224), (50, 180, 50))

    # Gradient
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        arr[i, :, :] = int(i / 224 * 255)
    images["gradient_v"] = Image.fromarray(arr)

    # Horizontal gradient
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for j in range(224):
        arr[:, j, :] = int(j / 224 * 255)
    images["gradient_h"] = Image.fromarray(arr)

    # Checkerboard
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            if (i // 28 + j // 28) % 2 == 0:
                arr[i, j] = [200, 200, 200]
            else:
                arr[i, j] = [50, 50, 50]
    images["checkerboard"] = Image.fromarray(arr)

    # Stripes
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for j in range(224):
        if (j // 16) % 2 == 0:
            arr[:, j] = [180, 100, 50]
        else:
            arr[:, j] = [50, 100, 180]
    images["stripes"] = Image.fromarray(arr)

    # Simple scene: sky + ground
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:112, :] = [135, 206, 235]  # sky blue
    arr[112:, :] = [34, 139, 34]     # forest green
    images["sky_ground"] = Image.fromarray(arr)

    # Road-like scene
    arr = np.ones((224, 224, 3), dtype=np.uint8) * 100  # gray background
    arr[140:, :] = [64, 64, 64]  # dark road
    arr[140:, 90:134] = [255, 255, 255]  # white lane marking
    images["road"] = Image.fromarray(arr)

    # Indoor scene with objects
    img = Image.new('RGB', (224, 224), (200, 180, 160))  # beige wall
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 100, 170, 200], fill=(139, 90, 43))  # table
    draw.rectangle([80, 80, 110, 100], fill=(255, 0, 0))  # red object
    draw.rectangle([130, 70, 160, 100], fill=(0, 0, 255))  # blue object
    images["indoor"] = img

    return images

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    results = {
        "experiment": "semantic_content",
        "experiment_number": 295,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    images = create_test_images()
    corruptions = ['fog', 'night', 'blur', 'noise']

    # Part 1: Per-image corruption distances
    print("\n=== Part 1: Per-Image Corruption Distances ===")
    per_image_results = {}

    for img_name, img in images.items():
        clean_emb = extract_hidden(model, processor, img, prompt)

        dists = {}
        for c in corruptions:
            corrupted = apply_corruption(img, c, 1.0)
            emb = extract_hidden(model, processor, corrupted, prompt)
            dists[c] = float(cosine(clean_emb, emb))

        per_image_results[img_name] = dists
        print(f"  {img_name}: " + ", ".join([f"{c}={dists[c]:.6f}" for c in corruptions]))

    results["per_image_distances"] = per_image_results

    # Part 2: Cross-image clean distances
    print("\n=== Part 2: Cross-Image Clean Distances ===")
    clean_embs = {}
    for img_name, img in images.items():
        clean_embs[img_name] = extract_hidden(model, processor, img, prompt)

    cross_distances = {}
    img_names = list(images.keys())
    for i, n1 in enumerate(img_names):
        for n2 in img_names[i+1:]:
            d = float(cosine(clean_embs[n1], clean_embs[n2]))
            cross_distances[f"{n1}_vs_{n2}"] = d

    results["cross_image_distances"] = cross_distances

    # Part 3: Per-image AUROC (calibrate and test on same image)
    print("\n=== Part 3: Per-Image AUROC ===")
    per_image_auroc = {}
    for img_name in images:
        id_dists = [0.0]  # same-image clean distance is 0
        ood_dists = list(per_image_results[img_name].values())
        auroc = compute_auroc(id_dists, ood_dists)
        per_image_auroc[img_name] = auroc
        print(f"  {img_name}: AUROC={auroc:.3f}")
    results["per_image_auroc"] = per_image_auroc

    # Part 4: Cross-semantic detection (calibrate on one type, test on another)
    print("\n=== Part 4: Cross-Semantic Detection ===")
    cross_semantic = {}
    for cal_name in ['random_noise', 'solid_gray', 'road', 'indoor']:
        cal_emb = clean_embs[cal_name]
        cross_semantic[cal_name] = {}
        for test_name in images:
            if test_name == cal_name:
                continue
            id_dist = float(cosine(cal_emb, clean_embs[test_name]))
            ood_dists = []
            for c in corruptions:
                corrupted = apply_corruption(images[test_name], c, 1.0)
                emb = extract_hidden(model, processor, corrupted, prompt)
                ood_dists.append(float(cosine(cal_emb, emb)))
            auroc = compute_auroc([id_dist], ood_dists)
            cross_semantic[cal_name][test_name] = {
                "auroc": auroc,
                "clean_distance": id_dist,
                "corruption_distances": dict(zip(corruptions, ood_dists))
            }
        avg_auroc = np.mean([v["auroc"] for v in cross_semantic[cal_name].values()])
        print(f"  Cal={cal_name}: avg cross-semantic AUROC={avg_auroc:.3f}")
    results["cross_semantic"] = cross_semantic

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/semantic_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
