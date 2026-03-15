#!/usr/bin/env python3
"""Experiment 409: Mixed/Simultaneous Corruption Analysis

Tests detector behavior when multiple corruption types are applied simultaneously.
Real-world conditions often involve multiple degradations (e.g., fog + noise,
night + blur). This is critical for understanding if the detector handles
compound corruptions that weren't in the calibration set.

Tests:
1. Pairwise corruption combinations at equal severity
2. Dominant + minor corruption (one strong, one weak)
3. Triple corruption combinations
4. Detection accuracy for compound corruptions
5. Embedding displacement for compounds vs individuals
6. Superposition hypothesis: is compound displacement ≈ sum of individual displacements?
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

def apply_multi_corruption(image, corruptions_with_severity):
    """Apply multiple corruptions sequentially."""
    result = image
    for ctype, severity in corruptions_with_severity:
        result = apply_corruption(result, ctype, severity)
    return result

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

    # Generate scenes
    scenes = []
    seeds = [42, 123, 456, 789, 999]
    for seed in seeds:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    # Get clean embeddings and centroids
    print("Clean embeddings...")
    clean_embeddings = []
    for si, scene in enumerate(scenes):
        emb = extract_hidden(model, processor, scene, prompt)
        clean_embeddings.append(emb)
    centroid = np.mean(clean_embeddings, axis=0)
    id_scores = [cosine_dist(e, centroid) for e in clean_embeddings]

    # Get individual corruption embeddings (for superposition test)
    print("Individual corruption embeddings...")
    individual_embeddings = {}
    individual_displacements = {}
    for c in corruptions:
        individual_embeddings[c] = []
        individual_displacements[c] = []
        for si, scene in enumerate(scenes):
            corrupted = apply_corruption(scene, c)
            emb = extract_hidden(model, processor, corrupted, prompt)
            individual_embeddings[c].append(emb)
            individual_displacements[c].append(emb - clean_embeddings[si])

    results = {}

    # === Test 1: Pairwise combinations at equal severity ===
    print("\n=== Pairwise Combinations (sev=1.0) ===")
    pairwise_results = {}
    for i in range(len(corruptions)):
        for j in range(i+1, len(corruptions)):
            c1, c2 = corruptions[i], corruptions[j]
            key = f"{c1}+{c2}"
            print(f"  {key}...")

            ood_scores = []
            displacements = []
            for si, scene in enumerate(scenes):
                corrupted = apply_multi_corruption(scene, [(c1, 1.0), (c2, 1.0)])
                emb = extract_hidden(model, processor, corrupted, prompt)
                dist = cosine_dist(emb, centroid)
                ood_scores.append(dist)
                displacements.append(emb - clean_embeddings[si])

            auroc = compute_auroc(id_scores, ood_scores)

            # Individual distances for comparison
            ind_dists_1 = [cosine_dist(individual_embeddings[c1][si], centroid) for si in range(len(scenes))]
            ind_dists_2 = [cosine_dist(individual_embeddings[c2][si], centroid) for si in range(len(scenes))]

            # Superposition test: is compound ≈ sum of individual displacements?
            superposition_errors = []
            for si in range(len(scenes)):
                predicted = individual_displacements[c1][si] + individual_displacements[c2][si]
                actual = displacements[si]
                error = np.linalg.norm(actual - predicted) / max(np.linalg.norm(actual), 1e-12)
                superposition_errors.append(error)

            pairwise_results[key] = {
                "auroc": float(auroc),
                "mean_dist": float(np.mean(ood_scores)),
                "individual_dists": {
                    c1: float(np.mean(ind_dists_1)),
                    c2: float(np.mean(ind_dists_2))
                },
                "ratio_to_max_individual": float(np.mean(ood_scores) / max(np.mean(ind_dists_1), np.mean(ind_dists_2))),
                "superposition_error": float(np.mean(superposition_errors)),
                "is_superadditive": bool(np.mean(ood_scores) > np.mean(ind_dists_1) + np.mean(ind_dists_2))
            }
            print(f"    AUROC={auroc:.4f}, dist={np.mean(ood_scores):.6f}, superpos_err={np.mean(superposition_errors):.4f}")

    results["pairwise_equal"] = pairwise_results

    # === Test 2: Dominant + minor corruption ===
    print("\n=== Dominant + Minor Corruption ===")
    dominant_minor = {}
    for c1 in corruptions:
        for c2 in corruptions:
            if c1 == c2:
                continue
            key = f"{c1}(1.0)+{c2}(0.3)"

            ood_scores = []
            for si, scene in enumerate(scenes):
                corrupted = apply_multi_corruption(scene, [(c1, 1.0), (c2, 0.3)])
                emb = extract_hidden(model, processor, corrupted, prompt)
                ood_scores.append(cosine_dist(emb, centroid))

            auroc = compute_auroc(id_scores, ood_scores)
            ind_dists = [cosine_dist(individual_embeddings[c1][si], centroid) for si in range(len(scenes))]
            ratio = float(np.mean(ood_scores) / np.mean(ind_dists))

            dominant_minor[key] = {
                "auroc": float(auroc),
                "mean_dist": float(np.mean(ood_scores)),
                "dominant_only_dist": float(np.mean(ind_dists)),
                "amplification_ratio": ratio
            }
            print(f"  {key}: AUROC={auroc:.4f}, dist={np.mean(ood_scores):.6f}, ratio={ratio:.3f}")

    results["dominant_minor"] = dominant_minor

    # === Test 3: Triple combinations ===
    print("\n=== Triple Combinations ===")
    triple_results = {}
    from itertools import combinations
    for combo in combinations(corruptions, 3):
        key = "+".join(combo)
        print(f"  {key}...")

        ood_scores = []
        for si, scene in enumerate(scenes):
            corrupted = apply_multi_corruption(scene, [(c, 1.0) for c in combo])
            emb = extract_hidden(model, processor, corrupted, prompt)
            ood_scores.append(cosine_dist(emb, centroid))

        auroc = compute_auroc(id_scores, ood_scores)

        # Compare to max individual
        max_ind = max(
            np.mean([cosine_dist(individual_embeddings[c][si], centroid) for si in range(len(scenes))])
            for c in combo
        )

        triple_results[key] = {
            "auroc": float(auroc),
            "mean_dist": float(np.mean(ood_scores)),
            "max_individual_dist": float(max_ind),
            "amplification": float(np.mean(ood_scores) / max_ind)
        }
        print(f"    AUROC={auroc:.4f}, dist={np.mean(ood_scores):.6f}")

    results["triple_combinations"] = triple_results

    # === Test 4: Quadruple (all corruptions at once) ===
    print("\n=== All Four Corruptions ===")
    ood_scores = []
    for si, scene in enumerate(scenes):
        corrupted = apply_multi_corruption(scene, [(c, 1.0) for c in corruptions])
        emb = extract_hidden(model, processor, corrupted, prompt)
        ood_scores.append(cosine_dist(emb, centroid))

    auroc = compute_auroc(id_scores, ood_scores)
    results["all_four"] = {
        "auroc": float(auroc),
        "mean_dist": float(np.mean(ood_scores)),
        "std_dist": float(np.std(ood_scores))
    }
    print(f"  All four: AUROC={auroc:.4f}, dist={np.mean(ood_scores):.6f}")

    # === Test 5: Order sensitivity ===
    print("\n=== Application Order Sensitivity ===")
    order_results = {}
    # Test fog+night vs night+fog
    for c1, c2 in [('fog', 'night'), ('fog', 'blur'), ('noise', 'night')]:
        dists_order1 = []
        dists_order2 = []
        for si, scene in enumerate(scenes):
            img1 = apply_multi_corruption(scene, [(c1, 1.0), (c2, 1.0)])
            img2 = apply_multi_corruption(scene, [(c2, 1.0), (c1, 1.0)])
            emb1 = extract_hidden(model, processor, img1, prompt)
            emb2 = extract_hidden(model, processor, img2, prompt)
            dists_order1.append(cosine_dist(emb1, centroid))
            dists_order2.append(cosine_dist(emb2, centroid))

        order_diff = float(abs(np.mean(dists_order1) - np.mean(dists_order2)))
        emb_diff = float(np.mean([cosine_dist(
            extract_hidden(model, processor, apply_multi_corruption(scenes[0], [(c1, 1.0), (c2, 1.0)]), prompt),
            extract_hidden(model, processor, apply_multi_corruption(scenes[0], [(c2, 1.0), (c1, 1.0)]), prompt)
        )]))

        order_results[f"{c1}_{c2}"] = {
            "order1_dist": float(np.mean(dists_order1)),
            "order2_dist": float(np.mean(dists_order2)),
            "order_difference": order_diff,
            "embedding_distance_between_orders": emb_diff
        }
        print(f"  {c1}→{c2}: {np.mean(dists_order1):.6f}, {c2}→{c1}: {np.mean(dists_order2):.6f}, diff={order_diff:.6f}")

    results["order_sensitivity"] = order_results

    # Save
    results["n_scenes"] = len(scenes)
    results["n_corruptions"] = len(corruptions)
    out_path = "/workspace/Vizuara-VLA-Research/experiments/mixed_corruptions_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
