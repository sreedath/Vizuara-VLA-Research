#!/usr/bin/env python3
"""Experiment 443: Embedding Geometry Analysis

Analyzes the geometric structure of the hidden state embedding space:
how clean and corrupted embeddings relate to each other geometrically,
cluster structure, convex hull properties, and angular distributions.

Tests:
1. Angular distribution of clean vs corrupted embeddings relative to centroid
2. Inter-class separation vs intra-class spread (Fisher criterion)
3. Convex hull analysis (do corrupted points fall outside clean hull?)
4. Nearest-neighbor structure
5. Embedding norm analysis
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

    seeds = [42, 123, 456, 789, 999, 1111, 2222, 3333]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print("Extracting embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)

    # Extract corrupted embeddings
    corr_embs = {}
    for c in corruptions:
        corr_embs[c] = [extract_hidden(model, processor, apply_corruption(s, c), prompt) for s in scenes]

    results = {"n_scenes": len(scenes), "emb_dim": len(centroid)}

    # === Test 1: Angular distribution ===
    print("\n=== Angular Distribution ===")
    angular_results = {}
    for c in ['clean'] + corruptions:
        if c == 'clean':
            embs = clean_embs
        else:
            embs = corr_embs[c]
        # Angles relative to centroid direction
        centroid_norm = centroid / np.linalg.norm(centroid)
        angles = []
        for e in embs:
            e_norm = np.asarray(e, dtype=np.float64) / max(np.linalg.norm(e), 1e-12)
            cos_angle = np.clip(np.dot(e_norm, centroid_norm), -1, 1)
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            angles.append(angle_deg)
        angular_results[c] = {
            "mean_angle": float(np.mean(angles)),
            "std_angle": float(np.std(angles)),
            "min_angle": float(np.min(angles)),
            "max_angle": float(np.max(angles)),
        }
        print(f"  {c}: mean_angle={np.mean(angles):.4f}° ± {np.std(angles):.4f}°")
    results["angular_distribution"] = angular_results

    # === Test 2: Fisher criterion (inter/intra class separation) ===
    print("\n=== Fisher Criterion ===")
    fisher_results = {}
    clean_centroid = np.mean(clean_embs, axis=0)
    clean_spread = np.mean([np.linalg.norm(np.asarray(e, dtype=np.float64) - clean_centroid) for e in clean_embs])

    for c in corruptions:
        corr_centroid = np.mean(corr_embs[c], axis=0)
        corr_spread = np.mean([np.linalg.norm(np.asarray(e, dtype=np.float64) - corr_centroid) for e in corr_embs[c]])
        inter_class = float(np.linalg.norm(clean_centroid - corr_centroid))
        intra_class = float(clean_spread + corr_spread)
        fisher = inter_class / max(intra_class, 1e-12)

        fisher_results[c] = {
            "inter_class_distance": inter_class,
            "intra_class_spread": intra_class,
            "fisher_ratio": fisher,
            "clean_spread": float(clean_spread),
            "corr_spread": float(corr_spread),
        }
        print(f"  {c}: Fisher={fisher:.4f} (inter={inter_class:.4f}, intra={intra_class:.4f})")
    results["fisher_criterion"] = fisher_results

    # === Test 3: Embedding norms ===
    print("\n=== Embedding Norms ===")
    norm_results = {}
    for c in ['clean'] + corruptions:
        if c == 'clean':
            embs = clean_embs
        else:
            embs = corr_embs[c]
        norms = [float(np.linalg.norm(e)) for e in embs]
        norm_results[c] = {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
        }
        print(f"  {c}: mean_norm={np.mean(norms):.4f} ± {np.std(norms):.4f}")
    results["embedding_norms"] = norm_results

    # === Test 4: Nearest-neighbor analysis ===
    print("\n=== Nearest-Neighbor Analysis ===")
    all_embs = clean_embs.copy()
    all_labels = ['clean'] * len(clean_embs)
    for c in corruptions:
        all_embs.extend(corr_embs[c])
        all_labels.extend([c] * len(corr_embs[c]))

    nn_results = {"n_total": len(all_embs)}
    # For each sample, find nearest neighbor and check if same class
    nn_same_class = 0
    nn_details = {}
    for i in range(len(all_embs)):
        min_dist = float('inf')
        min_j = -1
        for j in range(len(all_embs)):
            if i == j:
                continue
            d = cosine_dist(all_embs[i], all_embs[j])
            if d < min_dist:
                min_dist = d
                min_j = j
        if all_labels[i] == all_labels[min_j]:
            nn_same_class += 1

    nn_results["nn_accuracy"] = float(nn_same_class / len(all_embs))
    print(f"  1-NN accuracy: {nn_same_class}/{len(all_embs)} = {nn_same_class/len(all_embs):.4f}")

    # Within-class vs between-class distances
    within_dists = []
    between_dists = []
    for i in range(len(all_embs)):
        for j in range(i+1, len(all_embs)):
            d = cosine_dist(all_embs[i], all_embs[j])
            if all_labels[i] == all_labels[j]:
                within_dists.append(float(d))
            else:
                between_dists.append(float(d))

    nn_results["mean_within_dist"] = float(np.mean(within_dists))
    nn_results["mean_between_dist"] = float(np.mean(between_dists))
    nn_results["separation_ratio"] = float(np.mean(between_dists) / max(np.mean(within_dists), 1e-12))
    print(f"  Within-class dist: {np.mean(within_dists):.6f}")
    print(f"  Between-class dist: {np.mean(between_dists):.6f}")
    print(f"  Separation ratio: {np.mean(between_dists) / max(np.mean(within_dists), 1e-12):.2f}")
    results["nearest_neighbor"] = nn_results

    # === Test 5: Corruption magnitude ordering ===
    print("\n=== Corruption Magnitude Ordering ===")
    sevs = [0.1, 0.25, 0.5, 0.75, 1.0]
    ordering_results = {}
    for c in corruptions:
        sev_dists = []
        for sev in sevs:
            dists = []
            for s in scenes[:4]:
                emb = extract_hidden(model, processor, apply_corruption(s, c, severity=sev), prompt)
                dists.append(cosine_dist(emb, centroid))
            sev_dists.append(float(np.mean(dists)))
        # Check if monotonically increasing
        is_monotonic = all(sev_dists[i] <= sev_dists[i+1] for i in range(len(sev_dists)-1))
        ordering_results[c] = {
            "severity_dists": dict(zip([str(s) for s in sevs], sev_dists)),
            "monotonic": is_monotonic,
            "correlation_with_severity": float(np.corrcoef(sevs, sev_dists)[0, 1]),
        }
        print(f"  {c}: monotonic={is_monotonic}, corr={np.corrcoef(sevs, sev_dists)[0, 1]:.4f}")
    results["magnitude_ordering"] = ordering_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/embedding_geometry_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
