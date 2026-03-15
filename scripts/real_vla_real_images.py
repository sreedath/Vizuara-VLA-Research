#!/usr/bin/env python3
"""Experiment 425: Structured Scene Analysis

Tests OOD detection on structured (non-random) images: gradients, patterns,
solid colors, and natural-like scenes. Random images may not represent
realistic robot camera inputs — do structured scenes behave differently?

Tests:
1. Gradient images (horizontal/vertical/diagonal)
2. Solid color images
3. Checkerboard and stripe patterns
4. Natural-like structured scenes (Gaussian blobs, edges)
5. Cross-domain centroid: does a centroid from random images detect
   corruption in structured images?
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
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

def create_structured_images():
    """Create diverse structured images."""
    images = {}

    # Horizontal gradient
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for x in range(224):
        arr[:, x, :] = int(x / 223 * 255)
    images["h_gradient"] = Image.fromarray(arr)

    # Vertical gradient
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        arr[y, :, :] = int(y / 223 * 255)
    images["v_gradient"] = Image.fromarray(arr)

    # Diagonal gradient (RGB channels)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            arr[y, x, 0] = int(x / 223 * 255)
            arr[y, x, 1] = int(y / 223 * 255)
            arr[y, x, 2] = int((x + y) / 446 * 255)
    images["rgb_gradient"] = Image.fromarray(arr)

    # Solid colors
    for color, rgb in [("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255)), ("gray", (128, 128, 128))]:
        arr = np.full((224, 224, 3), rgb, dtype=np.uint8)
        images[f"solid_{color}"] = Image.fromarray(arr)

    # Checkerboard
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    block = 32
    for y in range(224):
        for x in range(224):
            if ((x // block) + (y // block)) % 2 == 0:
                arr[y, x] = 255
    images["checkerboard"] = Image.fromarray(arr)

    # Horizontal stripes
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        if (y // 16) % 2 == 0:
            arr[y, :] = 255
    images["h_stripes"] = Image.fromarray(arr)

    # Gaussian blobs (natural-like)
    rng = np.random.RandomState(42)
    arr = np.zeros((224, 224, 3), dtype=np.float32)
    for _ in range(5):
        cx, cy = rng.randint(30, 194, 2)
        sigma = rng.uniform(20, 50)
        color = rng.uniform(0.3, 1.0, 3)
        yy, xx = np.mgrid[:224, :224]
        blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        arr += blob[:, :, None] * color[None, None, :]
    arr = np.clip(arr / arr.max() * 255, 0, 255).astype(np.uint8)
    images["gaussian_blobs"] = Image.fromarray(arr)

    # Edge image (simulating object boundaries)
    img = Image.new('RGB', (224, 224), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 100, 100], fill=(100, 50, 50))
    draw.ellipse([120, 80, 200, 180], fill=(50, 100, 50))
    draw.polygon([(50, 150), (100, 200), (150, 180)], fill=(50, 50, 150))
    images["shapes"] = img

    return images

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']

    # Create structured images
    structured = create_structured_images()
    print(f"Created {len(structured)} structured images: {list(structured.keys())}")

    # Also create random images for comparison
    random_seeds = [42, 123, 456, 789, 999]
    random_scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in random_seeds]

    # Extract embeddings
    print("\nExtracting structured image embeddings...")
    struct_embs = {}
    for name, img in structured.items():
        struct_embs[name] = extract_hidden(model, processor, img, prompt)
    print("  Done")

    print("Extracting random image embeddings...")
    random_embs = [extract_hidden(model, processor, s, prompt) for s in random_scenes]
    random_centroid = np.mean(random_embs, axis=0)
    print("  Done")

    results = {"n_structured": len(structured), "n_random": len(random_scenes)}

    # === Test 1: Within-category centroids ===
    print("\n=== Structured Image OOD Detection ===")
    struct_list = list(structured.keys())
    struct_emb_list = [struct_embs[name] for name in struct_list]
    struct_centroid = np.mean(struct_emb_list, axis=0)
    struct_clean_dists = [cosine_dist(e, struct_centroid) for e in struct_emb_list]

    # Per-corruption detection using structured centroid
    struct_detection = {}
    for c in corruptions:
        ood_dists = []
        for name, img in structured.items():
            emb = extract_hidden(model, processor, apply_corruption(img, c), prompt)
            ood_dists.append(float(cosine_dist(emb, struct_centroid)))
        auroc = float(compute_auroc(struct_clean_dists, ood_dists))
        struct_detection[c] = {"auroc": auroc, "mean_ood_dist": float(np.mean(ood_dists))}
        print(f"  Structured centroid, {c}: AUROC={auroc:.4f}")
    struct_detection["clean_mean_dist"] = float(np.mean(struct_clean_dists))
    results["structured_detection"] = struct_detection

    # === Test 2: Cross-domain centroid ===
    print("\n=== Cross-Domain Centroid (Random → Structured) ===")
    cross_detection = {}
    # Use random centroid to detect corruption in structured images
    struct_dists_from_random = [cosine_dist(e, random_centroid) for e in struct_emb_list]
    for c in corruptions:
        ood_dists = []
        for name, img in structured.items():
            emb = extract_hidden(model, processor, apply_corruption(img, c), prompt)
            ood_dists.append(float(cosine_dist(emb, random_centroid)))
        auroc = float(compute_auroc(struct_dists_from_random, ood_dists))
        cross_detection[c] = {"auroc": auroc}
        print(f"  Random centroid → structured {c}: AUROC={auroc:.4f}")
    results["cross_domain_detection"] = cross_detection

    # === Test 3: Per-image-type embedding similarity ===
    print("\n=== Per-Image-Type Embedding Analysis ===")
    per_type = {}
    for name in struct_list:
        dist_to_random = cosine_dist(struct_embs[name], random_centroid)
        dist_to_struct = cosine_dist(struct_embs[name], struct_centroid)
        per_type[name] = {
            "dist_to_random_centroid": float(dist_to_random),
            "dist_to_struct_centroid": float(dist_to_struct),
        }
        print(f"  {name}: to_random={dist_to_random:.6f}, to_struct={dist_to_struct:.6f}")
    results["per_type_distances"] = per_type

    # === Test 4: Structured vs random embedding distance ===
    print("\n=== Structured vs Random Embedding Distance ===")
    random_clean_dists = [cosine_dist(e, random_centroid) for e in random_embs]
    domain_gap = {
        "struct_to_random_centroid_mean": float(np.mean(struct_dists_from_random)),
        "random_to_random_centroid_mean": float(np.mean(random_clean_dists)),
        "domain_gap_ratio": float(np.mean(struct_dists_from_random) / max(np.mean(random_clean_dists), 1e-10)),
    }
    # Can we distinguish structured from random?
    auroc_domain = float(compute_auroc(random_clean_dists, struct_dists_from_random))
    domain_gap["domain_auroc"] = auroc_domain
    print(f"  Structured vs Random AUROC: {auroc_domain:.4f}")
    print(f"  Domain gap ratio: {domain_gap['domain_gap_ratio']:.2f}")
    results["domain_gap"] = domain_gap

    # === Test 5: Per-structured-type corruption sensitivity ===
    print("\n=== Per-Type Corruption Sensitivity ===")
    type_sensitivity = {}
    for name, img in structured.items():
        clean_emb = struct_embs[name]
        sens = {}
        for c in corruptions:
            corrupt_emb = extract_hidden(model, processor, apply_corruption(img, c), prompt)
            d = cosine_dist(corrupt_emb, clean_emb)
            sens[c] = float(d)
        type_sensitivity[name] = sens
        print(f"  {name}: fog={sens['fog']:.6f}, night={sens['night']:.6f}")
    results["type_sensitivity"] = type_sensitivity

    out_path = "/workspace/Vizuara-VLA-Research/experiments/structured_scenes_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
