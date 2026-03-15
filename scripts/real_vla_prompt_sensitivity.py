#!/usr/bin/env python3
"""Experiment 441: Prompt Sensitivity Analysis

Tests how different action prompts affect OOD detection quality.
VLA models condition on language — does the choice of prompt matter
for corruption detection? Can we ensemble multiple prompts?

Tests:
1. Detection AUROC across different prompts
2. Embedding similarity across prompts (same image, different prompts)
3. Prompt ensemble detection (average distances from multiple prompts)
4. Prompt-specific failure modes
5. Prompt length vs detection quality
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

    corruptions = ['fog', 'night', 'noise', 'blur']

    prompts = {
        "pick": "In: What action should the robot take to pick up the object?\nOut:",
        "place": "In: What action should the robot take to place the object?\nOut:",
        "move": "In: What action should the robot take to move forward?\nOut:",
        "push": "In: What action should the robot take to push the object?\nOut:",
        "stack": "In: What action should the robot take to stack the blocks?\nOut:",
        "open": "In: What action should the robot take to open the drawer?\nOut:",
        "close": "In: What action should the robot take to close the drawer?\nOut:",
        "minimal": "In: Act.\nOut:",
    }

    seeds = [42, 123, 456, 789, 999, 1111, 2222, 3333]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    results = {"n_scenes": len(scenes), "n_prompts": len(prompts)}

    # === Test 1: Per-prompt detection AUROC ===
    print("\n=== Per-Prompt Detection AUROC ===")
    prompt_results = {}
    prompt_centroids = {}
    prompt_clean_embs = {}

    for pname, prompt in prompts.items():
        print(f"  Processing prompt: {pname}")
        clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
        centroid = np.mean(clean_embs, axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

        prompt_centroids[pname] = centroid
        prompt_clean_embs[pname] = clean_embs

        per_corr = {}
        for c in corruptions:
            ood_dists = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                ood_dists.append(cosine_dist(emb, centroid))
            auroc = float(compute_auroc(clean_dists, ood_dists))
            per_corr[c] = auroc

        mean_auroc = float(np.mean(list(per_corr.values())))
        prompt_results[pname] = {
            "auroc_per_corruption": per_corr,
            "mean_auroc": mean_auroc,
            "mean_clean_dist": float(np.mean(clean_dists)),
            "std_clean_dist": float(np.std(clean_dists)),
            "prompt_length": len(prompt),
        }
        print(f"    mean AUROC={mean_auroc:.4f}, per-corr: {per_corr}")
    results["per_prompt_detection"] = prompt_results

    # === Test 2: Cross-prompt embedding similarity ===
    print("\n=== Cross-Prompt Embedding Similarity ===")
    cross_prompt_sim = {}
    scene0 = scenes[0]
    pnames = list(prompts.keys())
    embs_scene0 = {}
    for pname, prompt in prompts.items():
        embs_scene0[pname] = extract_hidden(model, processor, scene0, prompt)

    for i, p1 in enumerate(pnames):
        for j, p2 in enumerate(pnames):
            if j <= i:
                continue
            sim = cosine_sim(embs_scene0[p1], embs_scene0[p2])
            cross_prompt_sim[f"{p1}_vs_{p2}"] = float(sim)

    results["cross_prompt_similarity"] = cross_prompt_sim
    sims = list(cross_prompt_sim.values())
    print(f"  Mean cross-prompt similarity: {np.mean(sims):.6f}")
    print(f"  Min: {np.min(sims):.6f}, Max: {np.max(sims):.6f}")

    # === Test 3: Prompt ensemble detection ===
    print("\n=== Prompt Ensemble Detection ===")
    ensemble_configs = {
        "single_pick": ["pick"],
        "pair_pick_place": ["pick", "place"],
        "triple_pick_place_move": ["pick", "place", "move"],
        "quad_pick_place_move_push": ["pick", "place", "move", "push"],
        "all_8": list(prompts.keys()),
        "diverse_3": ["pick", "minimal", "stack"],
    }

    ensemble_results = {}
    for ename, prompt_subset in ensemble_configs.items():
        clean_avg_dists = []
        for s_idx, s in enumerate(scenes):
            dists = []
            for pname in prompt_subset:
                emb = prompt_clean_embs[pname][s_idx]
                dists.append(cosine_dist(emb, prompt_centroids[pname]))
            clean_avg_dists.append(float(np.mean(dists)))

        per_corr_ens = {}
        for c in corruptions:
            ood_avg_dists = []
            for s in scenes:
                dists = []
                for pname in prompt_subset:
                    emb = extract_hidden(model, processor, apply_corruption(s, c), prompts[pname])
                    dists.append(cosine_dist(emb, prompt_centroids[pname]))
                ood_avg_dists.append(float(np.mean(dists)))
            auroc = float(compute_auroc(clean_avg_dists, ood_avg_dists))
            per_corr_ens[c] = auroc

        mean_ens = float(np.mean(list(per_corr_ens.values())))
        ensemble_results[ename] = {
            "n_prompts": len(prompt_subset),
            "auroc_per_corruption": per_corr_ens,
            "mean_auroc": mean_ens,
        }
        print(f"  {ename} ({len(prompt_subset)} prompts): mean={mean_ens:.4f}")
    results["ensemble_detection"] = ensemble_results

    # === Test 4: Centroid distance across prompts ===
    print("\n=== Centroid Geometry Across Prompts ===")
    centroid_geometry = {}
    for i, p1 in enumerate(pnames):
        for j, p2 in enumerate(pnames):
            if j <= i:
                continue
            dist = cosine_dist(prompt_centroids[p1], prompt_centroids[p2])
            centroid_geometry[f"{p1}_vs_{p2}"] = float(dist)

    results["centroid_distances"] = centroid_geometry
    cdists = list(centroid_geometry.values())
    print(f"  Mean centroid distance: {np.mean(cdists):.6f}")
    print(f"  Min: {np.min(cdists):.6f}, Max: {np.max(cdists):.6f}")

    # === Test 5: Prompt length vs detection quality ===
    print("\n=== Prompt Length vs Detection Quality ===")
    length_vs_quality = {}
    for pname, pdata in prompt_results.items():
        length_vs_quality[pname] = {
            "prompt_length": pdata["prompt_length"],
            "mean_auroc": pdata["mean_auroc"],
        }
    results["length_vs_quality"] = length_vs_quality

    lengths = [v["prompt_length"] for v in length_vs_quality.values()]
    aurocs = [v["mean_auroc"] for v in length_vs_quality.values()]
    if len(set(lengths)) > 1:
        corr = float(np.corrcoef(lengths, aurocs)[0, 1])
    else:
        corr = 0.0
    results["length_auroc_correlation"] = corr
    print(f"  Length-AUROC correlation: {corr:.4f}")

    out_path = "/workspace/Vizuara-VLA-Research/experiments/prompt_sensitivity_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
