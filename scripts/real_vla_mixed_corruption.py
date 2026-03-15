#!/usr/bin/env python3
"""Experiment 445: Mixed/Compound Corruption Analysis

Tests detection under simultaneous multiple corruptions — real-world
conditions rarely involve just one corruption type. A foggy night with
sensor noise is more realistic than fog alone.

Tests:
1. Pairwise corruption combinations (fog+night, fog+noise, etc.)
2. Triple corruption combinations
3. Does mixed corruption distance exceed individual corruptions?
4. Classification of mixed vs single corruptions
5. Severity interaction effects
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

def apply_multi_corruption(image, corruptions_with_sev):
    """Apply multiple corruptions sequentially."""
    result = image
    for ctype, sev in corruptions_with_sev:
        result = apply_corruption(result, ctype, sev)
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

def cosine_sim(a, b):
    return 1.0 - cosine_dist(a, b)

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

    print("Extracting clean embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    # Get single corruption distances for comparison
    print("Getting single corruption baselines...")
    single_dists = {}
    single_embs = {}
    for c in corruptions:
        embs = [extract_hidden(model, processor, apply_corruption(s, c), prompt) for s in scenes]
        dists = [cosine_dist(e, centroid) for e in embs]
        single_dists[c] = float(np.mean(dists))
        single_embs[c] = embs

    results = {"n_scenes": len(scenes)}
    results["single_corruption_dists"] = single_dists

    # === Test 1: Pairwise combinations ===
    print("\n=== Pairwise Corruption Combinations ===")
    pair_results = {}
    for i, c1 in enumerate(corruptions):
        for j, c2 in enumerate(corruptions):
            if j <= i:
                continue
            combo_dists = []
            for s in scenes:
                mixed = apply_multi_corruption(s, [(c1, 1.0), (c2, 1.0)])
                emb = extract_hidden(model, processor, mixed, prompt)
                combo_dists.append(cosine_dist(emb, centroid))

            auroc = float(compute_auroc(clean_dists, combo_dists))
            mean_dist = float(np.mean(combo_dists))
            # Is combined > max of individuals?
            max_single = max(single_dists[c1], single_dists[c2])
            sum_single = single_dists[c1] + single_dists[c2]

            pair_results[f"{c1}+{c2}"] = {
                "mean_dist": mean_dist,
                "auroc": auroc,
                "max_single_dist": max_single,
                "sum_single_dist": sum_single,
                "ratio_to_max": mean_dist / max_single if max_single > 0 else 0,
                "superadditive": mean_dist > sum_single,
            }
            print(f"  {c1}+{c2}: dist={mean_dist:.6f}, auroc={auroc:.4f}, ratio_to_max={mean_dist/max_single:.2f}x")
    results["pairwise"] = pair_results

    # === Test 2: Triple combinations ===
    print("\n=== Triple Corruption Combinations ===")
    triple_results = {}
    from itertools import combinations
    for combo in combinations(corruptions, 3):
        mixed_dists = []
        for s in scenes:
            mixed = apply_multi_corruption(s, [(c, 1.0) for c in combo])
            emb = extract_hidden(model, processor, mixed, prompt)
            mixed_dists.append(cosine_dist(emb, centroid))

        auroc = float(compute_auroc(clean_dists, mixed_dists))
        key = "+".join(combo)
        triple_results[key] = {
            "mean_dist": float(np.mean(mixed_dists)),
            "auroc": auroc,
        }
        print(f"  {key}: dist={np.mean(mixed_dists):.6f}, auroc={auroc:.4f}")
    results["triple"] = triple_results

    # === Test 3: All four corruptions ===
    print("\n=== All Four Corruptions ===")
    all4_dists = []
    for s in scenes:
        mixed = apply_multi_corruption(s, [(c, 1.0) for c in corruptions])
        emb = extract_hidden(model, processor, mixed, prompt)
        all4_dists.append(cosine_dist(emb, centroid))
    results["all_four"] = {
        "mean_dist": float(np.mean(all4_dists)),
        "auroc": float(compute_auroc(clean_dists, all4_dists)),
    }
    print(f"  All four: dist={np.mean(all4_dists):.6f}, auroc={compute_auroc(clean_dists, all4_dists):.4f}")

    # === Test 4: Severity interaction ===
    print("\n=== Severity Interaction (fog+night) ===")
    sev_interact = {}
    for fog_sev in [0.25, 0.5, 0.75, 1.0]:
        for night_sev in [0.25, 0.5, 0.75, 1.0]:
            dists = []
            for s in scenes[:4]:
                mixed = apply_multi_corruption(s, [('fog', fog_sev), ('night', night_sev)])
                emb = extract_hidden(model, processor, mixed, prompt)
                dists.append(cosine_dist(emb, centroid))
            auroc = float(compute_auroc(clean_dists[:4], dists))
            key = f"fog{fog_sev}_night{night_sev}"
            sev_interact[key] = {
                "mean_dist": float(np.mean(dists)),
                "auroc": auroc,
            }
    results["severity_interaction"] = sev_interact
    print(f"  fog0.25+night0.25: auroc={sev_interact['fog0.25_night0.25']['auroc']:.4f}")
    print(f"  fog1.0+night1.0: auroc={sev_interact['fog1.0_night1.0']['auroc']:.4f}")

    # === Test 5: Mixed vs single classification ===
    print("\n=== Mixed vs Single Direction Similarity ===")
    # Do mixed corruptions create new directions or are they near single corruption centroids?
    mixed_dir_results = {}
    for pair_name, pair_data in pair_results.items():
        c1, c2 = pair_name.split("+")
        mixed_embs = []
        for s in scenes:
            mixed = apply_multi_corruption(s, [(c1, 1.0), (c2, 1.0)])
            emb = extract_hidden(model, processor, mixed, prompt)
            mixed_embs.append(emb)
        mixed_centroid = np.mean(mixed_embs, axis=0)
        mixed_dir = mixed_centroid - centroid

        # Similarity to each single corruption direction
        sims = {}
        for c in corruptions:
            single_centroid = np.mean(single_embs[c], axis=0)
            single_dir = single_centroid - centroid
            sims[c] = float(cosine_sim(mixed_dir, single_dir))

        mixed_dir_results[pair_name] = sims
        best_match = max(sims, key=sims.get)
        print(f"  {pair_name}: best match={best_match} (sim={sims[best_match]:.4f})")
    results["mixed_direction_similarity"] = mixed_dir_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/mixed_corruption_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
