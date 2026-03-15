#!/usr/bin/env python3
"""Experiment 395: Compositional Corruption Analysis

Tests detection under simultaneous multiple corruptions applied to the same
image. Real-world scenarios often involve compound degradation (e.g., fog
at night, blurry noise).

Tests:
1. All pairwise corruption combinations at severity 0.5
2. Triple corruption combinations
3. Quadruple (all corruptions simultaneously)
4. Severity balancing in pairs (0.2/0.8 vs 0.5/0.5 vs 0.8/0.2)
5. Detection distance vs sum-of-individual distances
6. AUROC for compound corruptions
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from itertools import combinations

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

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

def apply_multi_corruption(image, corruptions_with_sevs):
    """Apply multiple corruptions sequentially."""
    result = image
    for ctype, sev in corruptions_with_sevs:
        result = apply_corruption(result, ctype, severity=sev)
    return result

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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    corruptions = ['fog', 'night', 'noise', 'blur']

    # Get clean centroid
    print("Computing clean centroid...")
    clean_embs = []
    for i in range(5):
        arr = np.array(img).astype(np.float32)
        arr += np.random.RandomState(100 + i).randn(*arr.shape) * 0.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        emb = extract_hidden(model, processor, Image.fromarray(arr), prompt)
        clean_embs.append(emb)
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    results = {}

    # 1. Individual corruption baselines
    print("\n=== Individual Corruption Baselines ===")
    individual = {}
    for c in corruptions:
        corrupted = apply_corruption(img, c, severity=0.5)
        emb = extract_hidden(model, processor, corrupted, prompt)
        d = cosine_dist(emb, centroid)
        individual[c] = float(d)
        print(f"  {c}: dist={d:.6f}")
    results["individual_baselines"] = individual

    # 2. Pairwise combinations at severity 0.5 each
    print("\n=== Pairwise Combinations ===")
    pairwise = {}
    for c1, c2 in combinations(corruptions, 2):
        # Apply c1 then c2
        corrupted = apply_multi_corruption(img, [(c1, 0.5), (c2, 0.5)])
        emb = extract_hidden(model, processor, corrupted, prompt)
        d = cosine_dist(emb, centroid)

        # Also test reverse order
        corrupted_rev = apply_multi_corruption(img, [(c2, 0.5), (c1, 0.5)])
        emb_rev = extract_hidden(model, processor, corrupted_rev, prompt)
        d_rev = cosine_dist(emb_rev, centroid)

        sum_individual = individual[c1] + individual[c2]
        sub_additivity = float(d) / sum_individual if sum_individual > 0 else 0

        pairwise[f"{c1}+{c2}"] = {
            "dist": float(d),
            "dist_reversed": float(d_rev),
            "order_effect": float(abs(d - d_rev)),
            "sum_individual": sum_individual,
            "sub_additivity": sub_additivity,
            "is_sub_additive": float(d) < sum_individual
        }
        print(f"  {c1}+{c2}: dist={d:.6f}, reversed={d_rev:.6f}, sum={sum_individual:.6f}, ratio={sub_additivity:.4f}")
    results["pairwise"] = pairwise

    # 3. Triple combinations
    print("\n=== Triple Combinations ===")
    triple = {}
    for combo in combinations(corruptions, 3):
        corrupted = apply_multi_corruption(img, [(c, 0.5) for c in combo])
        emb = extract_hidden(model, processor, corrupted, prompt)
        d = cosine_dist(emb, centroid)
        sum_individual = sum(individual[c] for c in combo)
        triple["+".join(combo)] = {
            "dist": float(d),
            "sum_individual": sum_individual,
            "ratio": float(d) / sum_individual if sum_individual > 0 else 0
        }
        print(f"  {'+'.join(combo)}: dist={d:.6f}, ratio={float(d)/sum_individual:.4f}")
    results["triple"] = triple

    # 4. Quadruple (all at once)
    print("\n=== All Four Corruptions ===")
    corrupted_all = apply_multi_corruption(img, [(c, 0.5) for c in corruptions])
    emb_all = extract_hidden(model, processor, corrupted_all, prompt)
    d_all = cosine_dist(emb_all, centroid)
    sum_all = sum(individual.values())
    results["quadruple"] = {
        "dist": float(d_all),
        "sum_individual": sum_all,
        "ratio": float(d_all) / sum_all if sum_all > 0 else 0
    }
    print(f"  All four: dist={d_all:.6f}, sum={sum_all:.6f}, ratio={d_all/sum_all:.4f}")

    # 5. Severity balancing in pairs
    print("\n=== Severity Balancing ===")
    balance_results = {}
    for c1, c2 in [('fog', 'night'), ('fog', 'noise'), ('noise', 'blur')]:
        balances = []
        for s1 in [0.2, 0.4, 0.5, 0.6, 0.8]:
            s2 = 1.0 - s1
            corrupted = apply_multi_corruption(img, [(c1, s1), (c2, s2)])
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = cosine_dist(emb, centroid)
            balances.append({"s1": s1, "s2": s2, "dist": float(d)})
        balance_results[f"{c1}+{c2}"] = balances
        summary = [(b['s1'], round(b['dist'], 6)) for b in balances]
        print(f"  {c1}+{c2}: {summary}")
    results["severity_balancing"] = balance_results

    # 6. AUROC for compound corruptions
    print("\n=== AUROC for Compounds ===")
    compound_aurocs = {}
    for combo_name in list(pairwise.keys()) + list(triple.keys()) + ["all"]:
        if combo_name == "all":
            combo_list = corruptions
        elif "+" in combo_name:
            combo_list = combo_name.split("+")
        else:
            continue

        ood_dists = []
        for i in range(5):
            arr = np.array(img).astype(np.float32)
            arr += np.random.RandomState(300 + i).randn(*arr.shape) * 0.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            perturbed = Image.fromarray(arr)
            corrupted = apply_multi_corruption(perturbed, [(c, 0.5) for c in combo_list])
            emb = extract_hidden(model, processor, corrupted, prompt)
            ood_dists.append(cosine_dist(emb, centroid))

        auroc = compute_auroc(clean_dists, ood_dists)
        compound_aurocs[combo_name] = float(auroc)
        print(f"  {combo_name}: AUROC={auroc:.4f}")
    results["compound_aurocs"] = compound_aurocs

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/compositional_corruption_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
