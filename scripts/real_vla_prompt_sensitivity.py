#!/usr/bin/env python3
"""Experiment 423: Prompt Sensitivity Analysis

Studies how different text prompts affect the model's OOD detection capability.
The standard prompt is fixed, but does the specific wording matter? Are some
prompts more sensitive to visual corruption than others?

Tests:
1. Multiple prompt variants with the same image
2. Per-prompt OOD detection AUROC
3. Prompt-agnostic vs prompt-specific centroids
4. Cross-prompt embedding similarity
5. Prompt length effects on detection
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

    # Multiple prompt variants
    prompts = {
        "standard": "In: What action should the robot take to pick up the object?\nOut:",
        "short": "In: Pick up the object.\nOut:",
        "verbose": "In: What is the next action the robot arm should execute in order to successfully grasp and pick up the target object from the table?\nOut:",
        "different_task": "In: What action should the robot take to push the block?\nOut:",
        "minimal": "In: Act.\nOut:",
        "place": "In: What action should the robot take to place the object?\nOut:",
    }

    corruptions = ['fog', 'night', 'noise', 'blur']

    seeds = [42, 123, 456]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    results = {"n_prompts": len(prompts), "n_scenes": len(scenes)}

    # === Test 1: Per-prompt embeddings ===
    print("Extracting embeddings for all prompts...")
    prompt_embs = {}  # {prompt_name: {scene_idx: embedding}}
    for pname, prompt in prompts.items():
        embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
        prompt_embs[pname] = embs
        print(f"  {pname}: done")

    # === Test 2: Cross-prompt similarity ===
    print("\n=== Cross-Prompt Similarity ===")
    cross_sim = {}
    for p1 in prompts:
        for p2 in prompts:
            if p1 >= p2:
                continue
            dists = []
            for s in range(len(scenes)):
                d = cosine_dist(prompt_embs[p1][s], prompt_embs[p2][s])
                dists.append(float(d))
            key = f"{p1}_vs_{p2}"
            cross_sim[key] = {
                "mean_dist": float(np.mean(dists)),
                "max_dist": float(np.max(dists)),
            }
            print(f"  {key}: mean_dist={np.mean(dists):.6f}")
    results["cross_prompt_similarity"] = cross_sim

    # === Test 3: Per-prompt OOD detection ===
    print("\n=== Per-Prompt OOD Detection ===")
    detection = {}
    for pname, prompt in prompts.items():
        centroid = np.mean(prompt_embs[pname], axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in prompt_embs[pname]]

        per_corr = {}
        all_ood = []
        for c in corruptions:
            ood_dists = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                d = cosine_dist(emb, centroid)
                ood_dists.append(float(d))
                all_ood.append(float(d))
            per_corr[c] = {
                "auroc": float(compute_auroc(clean_dists, ood_dists)),
                "mean_dist": float(np.mean(ood_dists)),
            }

        overall = float(compute_auroc(clean_dists, all_ood))
        detection[pname] = {
            "overall_auroc": overall,
            "clean_mean_dist": float(np.mean(clean_dists)),
            "per_corruption": per_corr,
        }
        print(f"  {pname}: AUROC={overall:.4f}")
    results["per_prompt_detection"] = detection

    # === Test 4: Prompt-agnostic centroid ===
    print("\n=== Prompt-Agnostic Centroid ===")
    # Build centroid from ALL prompts' clean embeddings
    all_clean = []
    for pname in prompts:
        all_clean.extend(prompt_embs[pname])
    agnostic_centroid = np.mean(all_clean, axis=0)

    agnostic_detection = {}
    for pname, prompt in prompts.items():
        clean_dists = [cosine_dist(e, agnostic_centroid) for e in prompt_embs[pname]]
        all_ood = []
        for c in corruptions:
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                all_ood.append(float(cosine_dist(emb, agnostic_centroid)))
        auroc = float(compute_auroc(clean_dists, all_ood))
        agnostic_detection[pname] = {"auroc": auroc}
        print(f"  {pname}: agnostic AUROC={auroc:.4f}")
    results["agnostic_detection"] = agnostic_detection

    # === Test 5: Prompt length effect ===
    print("\n=== Prompt Length Effects ===")
    length_data = {}
    for pname, prompt in prompts.items():
        # Tokenize to get length
        inputs = processor(prompt, scenes[0])
        seq_len = inputs['input_ids'].shape[1]
        length_data[pname] = {
            "token_length": int(seq_len),
            "char_length": len(prompt),
            "auroc": detection[pname]["overall_auroc"],
        }
        print(f"  {pname}: {seq_len} tokens, {len(prompt)} chars, AUROC={detection[pname]['overall_auroc']:.4f}")
    results["prompt_length"] = length_data

    out_path = "/workspace/Vizuara-VLA-Research/experiments/prompt_sensitivity_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
