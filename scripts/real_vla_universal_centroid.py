#!/usr/bin/env python3
"""Experiment 426: Universal Centroid Investigation

Following the critical domain specificity finding from Exp 425, this experiment
investigates whether a mixed-domain centroid (combining random + structured images)
can achieve robust cross-domain OOD detection.

Tests:
1. Mixed centroid (random + structured) detection performance
2. Leave-one-domain-out cross-validation
3. Per-corruption direction consistency across domains
4. Relative distance detection (within-domain change detection)
5. Corruption direction alignment between domains
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

def create_image_domains():
    """Create multiple image domains."""
    domains = {}

    # Domain 1: Random images
    random_imgs = []
    for s in [42, 123, 456, 789, 999]:
        random_imgs.append(Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)))
    domains["random"] = random_imgs

    # Domain 2: Gradient images
    gradient_imgs = []
    for angle in [0, 45, 90, 135, 180]:
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        rad = np.radians(angle)
        for y in range(224):
            for x in range(224):
                val = int(((x * np.cos(rad) + y * np.sin(rad)) / (224 * (abs(np.cos(rad)) + abs(np.sin(rad))))) * 255)
                arr[y, x] = np.clip(val, 0, 255)
        gradient_imgs.append(Image.fromarray(arr))
    domains["gradient"] = gradient_imgs

    # Domain 3: Solid colors
    solid_imgs = []
    for rgb in [(255,0,0), (0,255,0), (0,0,255), (128,128,128), (255,255,0)]:
        solid_imgs.append(Image.fromarray(np.full((224, 224, 3), rgb, dtype=np.uint8)))
    domains["solid"] = solid_imgs

    # Domain 4: Patterned images
    pattern_imgs = []
    for block_size in [8, 16, 32, 48, 64]:
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        for y in range(224):
            for x in range(224):
                if ((x // block_size) + (y // block_size)) % 2 == 0:
                    arr[y, x] = 255
        pattern_imgs.append(Image.fromarray(arr))
    domains["pattern"] = pattern_imgs

    # Domain 5: Gaussian blob scenes
    blob_imgs = []
    for seed in [10, 20, 30, 40, 50]:
        rng = np.random.RandomState(seed)
        arr = np.zeros((224, 224, 3), dtype=np.float32)
        for _ in range(5):
            cx, cy = rng.randint(30, 194, 2)
            sigma = rng.uniform(20, 50)
            color = rng.uniform(0.3, 1.0, 3)
            yy, xx = np.mgrid[:224, :224]
            blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            arr += blob[:, :, None] * color[None, None, :]
        arr = np.clip(arr / max(arr.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
        blob_imgs.append(Image.fromarray(arr))
    domains["blobs"] = blob_imgs

    return domains

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']

    domains = create_image_domains()
    print(f"Created {len(domains)} domains: {list(domains.keys())}")

    # Extract all embeddings
    print("\nExtracting embeddings...")
    domain_embs = {}
    domain_corrupt_embs = {}
    for dname, imgs in domains.items():
        domain_embs[dname] = [extract_hidden(model, processor, img, prompt) for img in imgs]
        domain_corrupt_embs[dname] = {}
        for c in corruptions:
            domain_corrupt_embs[dname][c] = [extract_hidden(model, processor, apply_corruption(img, c), prompt) for img in imgs]
        print(f"  {dname}: {len(imgs)} images extracted")

    results = {"domains": list(domains.keys()), "n_per_domain": {d: len(imgs) for d, imgs in domains.items()}}

    # === Test 1: Mixed centroid ===
    print("\n=== Mixed Centroid Detection ===")
    all_clean = []
    for dname in domains:
        all_clean.extend(domain_embs[dname])
    mixed_centroid = np.mean(all_clean, axis=0)

    mixed_detection = {}
    for dname in domains:
        clean_dists = [cosine_dist(e, mixed_centroid) for e in domain_embs[dname]]
        per_corr = {}
        all_ood = []
        for c in corruptions:
            ood_dists = [cosine_dist(e, mixed_centroid) for e in domain_corrupt_embs[dname][c]]
            all_ood.extend(ood_dists)
            per_corr[c] = float(compute_auroc(clean_dists, ood_dists))
        overall = float(compute_auroc(clean_dists, all_ood))
        mixed_detection[dname] = {"overall": overall, "per_corruption": per_corr}
        print(f"  {dname}: AUROC={overall:.4f}")
    results["mixed_centroid"] = mixed_detection

    # === Test 2: Domain-specific centroids ===
    print("\n=== Domain-Specific Centroids ===")
    specific_detection = {}
    for dname in domains:
        centroid = np.mean(domain_embs[dname], axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in domain_embs[dname]]
        all_ood = []
        for c in corruptions:
            all_ood.extend([cosine_dist(e, centroid) for e in domain_corrupt_embs[dname][c]])
        auroc = float(compute_auroc(clean_dists, all_ood))
        specific_detection[dname] = auroc
        print(f"  {dname}: AUROC={auroc:.4f}")
    results["specific_centroids"] = specific_detection

    # === Test 3: Leave-one-domain-out ===
    print("\n=== Leave-One-Domain-Out ===")
    loo_results = {}
    for held_out in domains:
        # Build centroid from all OTHER domains
        train_embs = []
        for dname in domains:
            if dname != held_out:
                train_embs.extend(domain_embs[dname])
        loo_centroid = np.mean(train_embs, axis=0)

        # Test on held-out domain
        clean_dists = [cosine_dist(e, loo_centroid) for e in domain_embs[held_out]]
        all_ood = []
        for c in corruptions:
            all_ood.extend([cosine_dist(e, loo_centroid) for e in domain_corrupt_embs[held_out][c]])
        auroc = float(compute_auroc(clean_dists, all_ood))
        loo_results[held_out] = auroc
        print(f"  Held-out {held_out}: AUROC={auroc:.4f}")
    results["leave_one_out"] = loo_results

    # === Test 4: Relative distance detection ===
    print("\n=== Relative Distance Detection ===")
    # Instead of absolute distance to centroid, use distance change from clean to corrupt
    relative_detection = {}
    for dname in domains:
        centroid = np.mean(domain_embs[dname], axis=0)
        # Clean baseline distances
        clean_base = [cosine_dist(e, centroid) for e in domain_embs[dname]]
        mean_clean = np.mean(clean_base)

        # For each corruption, measure relative increase
        id_scores = [abs(d - mean_clean) for d in clean_base]
        all_ood = []
        for c in corruptions:
            for e in domain_corrupt_embs[dname][c]:
                d = cosine_dist(e, centroid)
                all_ood.append(abs(d - mean_clean))
        auroc = float(compute_auroc(id_scores, all_ood))
        relative_detection[dname] = auroc
        print(f"  {dname}: relative AUROC={auroc:.4f}")
    results["relative_detection"] = relative_detection

    # === Test 5: Corruption direction alignment ===
    print("\n=== Corruption Direction Alignment ===")
    direction_alignment = {}
    for c in corruptions:
        # For each domain, compute mean corruption direction
        directions = {}
        for dname in domains:
            clean_mean = np.mean(domain_embs[dname], axis=0)
            corrupt_mean = np.mean(domain_corrupt_embs[dname][c], axis=0)
            direction = corrupt_mean - clean_mean
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                directions[dname] = direction / norm
            else:
                directions[dname] = direction

        # Pairwise alignment between domains
        alignments = {}
        domain_names = list(domains.keys())
        for i, d1 in enumerate(domain_names):
            for j, d2 in enumerate(domain_names):
                if i >= j:
                    continue
                cos_sim = float(np.dot(directions[d1], directions[d2]))
                alignments[f"{d1}_vs_{d2}"] = cos_sim

        mean_alignment = float(np.mean(list(alignments.values())))
        direction_alignment[c] = {
            "pairwise": alignments,
            "mean_alignment": mean_alignment,
        }
        print(f"  {c}: mean alignment={mean_alignment:.4f}")
    results["direction_alignment"] = direction_alignment

    out_path = "/workspace/Vizuara-VLA-Research/experiments/universal_centroid_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
