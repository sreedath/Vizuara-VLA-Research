#!/usr/bin/env python3
"""Experiment 427: Multi-Corruption Combination Analysis

Studies what happens when multiple corruptions are applied simultaneously.
Real-world scenarios often involve compound degradation (e.g., fog + night,
blur + noise). Does the detector handle compound corruptions?

Tests:
1. Pairwise corruption combinations
2. Triple corruption stacking
3. All-four corruption stacking
4. Compound corruption geometry
5. Application order effects
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from itertools import combinations

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

def apply_multi_corruption(image, ctypes, severity=1.0):
    result = image
    for c in ctypes:
        result = apply_corruption(result, c, severity)
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

    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print("Extracting clean embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    print("Extracting single corruption embeddings...")
    single_embs = {}
    single_dists = {}
    for c in corruptions:
        embs = [extract_hidden(model, processor, apply_corruption(s, c), prompt) for s in scenes]
        single_embs[c] = embs
        single_dists[c] = [cosine_dist(e, centroid) for e in embs]

    results = {"n_scenes": len(scenes)}

    # === Test 1: Pairwise combinations ===
    print("\n=== Pairwise Corruption Combinations ===")
    pairs = list(combinations(corruptions, 2))
    pair_results = {}
    for c1, c2 in pairs:
        ood_dists = []
        for s in scenes:
            img = apply_multi_corruption(s, [c1, c2])
            emb = extract_hidden(model, processor, img, prompt)
            ood_dists.append(float(cosine_dist(emb, centroid)))

        auroc = float(compute_auroc(clean_dists, ood_dists))
        single_max = max(np.mean(single_dists[c1]), np.mean(single_dists[c2]))
        pair_mean = float(np.mean(ood_dists))

        key = f"{c1}+{c2}"
        pair_results[key] = {
            "auroc": auroc,
            "mean_dist": pair_mean,
            "c1_mean_dist": float(np.mean(single_dists[c1])),
            "c2_mean_dist": float(np.mean(single_dists[c2])),
            "amplification": float(pair_mean / single_max) if single_max > 1e-10 else 0.0,
        }
        print(f"  {key}: AUROC={auroc:.4f}, dist={pair_mean:.6f}, amp={pair_results[key]['amplification']:.2f}x")
    results["pairwise"] = pair_results

    # === Test 2: Triple combinations ===
    print("\n=== Triple Corruption Combinations ===")
    triples = list(combinations(corruptions, 3))
    triple_results = {}
    for combo in triples:
        ood_dists = []
        for s in scenes:
            img = apply_multi_corruption(s, list(combo))
            emb = extract_hidden(model, processor, img, prompt)
            ood_dists.append(float(cosine_dist(emb, centroid)))

        auroc = float(compute_auroc(clean_dists, ood_dists))
        key = "+".join(combo)
        triple_results[key] = {
            "auroc": auroc,
            "mean_dist": float(np.mean(ood_dists)),
        }
        print(f"  {key}: AUROC={auroc:.4f}, dist={np.mean(ood_dists):.6f}")
    results["triples"] = triple_results

    # === Test 3: All-four stacking ===
    print("\n=== All-Four Corruption Stacking ===")
    all_four_dists = []
    for s in scenes:
        img = apply_multi_corruption(s, corruptions)
        emb = extract_hidden(model, processor, img, prompt)
        all_four_dists.append(float(cosine_dist(emb, centroid)))

    all_four_auroc = float(compute_auroc(clean_dists, all_four_dists))
    results["all_four"] = {
        "auroc": all_four_auroc,
        "mean_dist": float(np.mean(all_four_dists)),
    }
    print(f"  All four: AUROC={all_four_auroc:.4f}, dist={np.mean(all_four_dists):.6f}")

    # === Test 4: Compound corruption geometry ===
    print("\n=== Compound Corruption Geometry ===")
    geometry = {}
    for c1, c2 in pairs:
        d_c1_list = []
        d_c2_list = []
        d_clean_list = []
        for i in range(len(scenes)):
            compound_emb = extract_hidden(model, processor, apply_multi_corruption(scenes[i], [c1, c2]), prompt)
            d_c1_list.append(cosine_dist(compound_emb, single_embs[c1][i]))
            d_c2_list.append(cosine_dist(compound_emb, single_embs[c2][i]))
            d_clean_list.append(cosine_dist(compound_emb, clean_embs[i]))

        key = f"{c1}+{c2}"
        mean_dc1 = float(np.mean(d_c1_list))
        mean_dc2 = float(np.mean(d_c2_list))
        geometry[key] = {
            "dist_to_c1": mean_dc1,
            "dist_to_c2": mean_dc2,
            "dist_to_clean": float(np.mean(d_clean_list)),
            "closer_to": c2 if mean_dc2 < mean_dc1 else c1,
        }
        print(f"  {key}: closer to {geometry[key]['closer_to']} (d_c1={mean_dc1:.6f}, d_c2={mean_dc2:.6f})")
    results["compound_geometry"] = geometry

    # === Test 5: Application order effects ===
    print("\n=== Application Order Effects ===")
    order_effects = {}
    for c1, c2 in pairs:
        dists_12 = []
        dists_21 = []
        for s in scenes:
            emb_12 = extract_hidden(model, processor, apply_multi_corruption(s, [c1, c2]), prompt)
            emb_21 = extract_hidden(model, processor, apply_multi_corruption(s, [c2, c1]), prompt)
            dists_12.append(float(cosine_dist(emb_12, centroid)))
            dists_21.append(float(cosine_dist(emb_21, centroid)))

        order_diff = float(np.mean(np.abs(np.array(dists_12) - np.array(dists_21))))
        key = f"{c1}+{c2}"
        order_effects[key] = {
            "forward_mean": float(np.mean(dists_12)),
            "reverse_mean": float(np.mean(dists_21)),
            "order_difference": order_diff,
            "commutative": order_diff < 0.0001,
        }
        print(f"  {key}: fwd={np.mean(dists_12):.6f}, rev={np.mean(dists_21):.6f}, diff={order_diff:.6f}")
    results["order_effects"] = order_effects

    out_path = "/workspace/Vizuara-VLA-Research/experiments/multi_corruption_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
