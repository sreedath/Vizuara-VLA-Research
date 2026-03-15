#!/usr/bin/env python3
"""Experiment 370: Prompt Diversity and Detection Robustness

Test OOD detection across a wide variety of prompts:
1. 20 diverse prompts (actions, questions, descriptions)
2. Per-prompt detection AUROC
3. Cross-prompt centroid stability
4. Prompt length sensitivity
5. Adversarial prompt testing
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

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

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

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

    ctypes = ['fog', 'night', 'noise', 'blur']

    # 20 diverse prompts
    prompts = [
        "In: What action should the robot take to pick up the object?\nOut:",
        "In: What action should the robot take to move forward?\nOut:",
        "In: What action should the robot take to turn left?\nOut:",
        "In: What action should the robot take to grasp the cup?\nOut:",
        "In: What action should the robot take to place the block?\nOut:",
        "In: What action should the robot take to push the button?\nOut:",
        "In: What action should the robot take to open the drawer?\nOut:",
        "In: What action should the robot take to close the gripper?\nOut:",
        "In: What action should the robot take to lift the box?\nOut:",
        "In: What action should the robot take to stack the plates?\nOut:",
        "In: What action should the robot take?\nOut:",
        "In: Robot action?\nOut:",
        "In: Next action for the manipulator?\nOut:",
        "In: What should the robot do with the red object?\nOut:",
        "In: How should the arm move to reach the target?\nOut:",
        "In: Determine the next robot action.\nOut:",
        "In: Predict the action to complete the task.\nOut:",
        "In: What manipulation action is needed?\nOut:",
        "In: Output the 7-DOF action vector.\nOut:",
        "In: What action should the robot take to navigate to the goal?\nOut:",
    ]

    results = {}

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # ========== 1. Per-Prompt Detection AUROC ==========
    print("\n=== Per-Prompt Detection ===")

    per_prompt = {}
    for pi, prompt in enumerate(prompts):
        # Get clean embeddings for this prompt
        clean_embs = {}
        for seed in seeds[:5]:
            clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

        centroid = np.mean(list(clean_embs.values()), axis=0)
        id_dists = [cosine_dist(centroid, clean_embs[s]) for s in seeds[:5]]

        prompt_aurocs = {}
        for ct in ctypes:
            ood_dists = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                corrupt_emb = extract_hidden(model, processor, corrupt_img, prompt)
                ood_dists.append(cosine_dist(centroid, corrupt_emb))
            auroc = compute_auroc(id_dists, ood_dists)
            prompt_aurocs[ct] = float(auroc)

        mean_auroc = float(np.mean(list(prompt_aurocs.values())))
        per_prompt[f"prompt_{pi}"] = {
            'prompt': prompt[:60] + '...' if len(prompt) > 60 else prompt,
            'prompt_length': len(prompt),
            'per_corruption': prompt_aurocs,
            'mean_auroc': mean_auroc,
        }
        auroc_str = ', '.join(f'{ct}={prompt_aurocs[ct]:.3f}' for ct in ctypes)
        print(f"  P{pi}: mean={mean_auroc:.3f} [{auroc_str}]")

    results['per_prompt'] = per_prompt

    # ========== 2. Cross-Prompt Centroid Stability ==========
    print("\n=== Cross-Prompt Centroid Stability ===")

    centroids = {}
    for pi, prompt in enumerate(prompts):
        embs = [extract_hidden(model, processor, images[seeds[0]], prompt)]
        centroids[pi] = embs[0]

    # Pairwise distances between centroids
    cross_dists = []
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            d = cosine_dist(centroids[i], centroids[j])
            cross_dists.append(d)

    centroid_stability = {
        'mean_cross_dist': float(np.mean(cross_dists)),
        'min_cross_dist': float(min(cross_dists)),
        'max_cross_dist': float(max(cross_dists)),
        'std_cross_dist': float(np.std(cross_dists)),
    }
    print(f"  Cross-prompt centroid dist: mean={np.mean(cross_dists):.6f}, "
          f"range=[{min(cross_dists):.6f}, {max(cross_dists):.6f}]")

    results['centroid_stability'] = centroid_stability

    # ========== 3. Prompt Length vs Detection ==========
    print("\n=== Prompt Length Analysis ===")

    length_analysis = {}
    for pi, prompt in enumerate(prompts):
        length_analysis[f"prompt_{pi}"] = {
            'length': len(prompt),
            'mean_auroc': per_prompt[f"prompt_{pi}"]['mean_auroc'],
        }

    lengths = [v['length'] for v in length_analysis.values()]
    aurocs = [v['mean_auroc'] for v in length_analysis.values()]
    if np.std(lengths) > 0:
        corr = float(np.corrcoef(lengths, aurocs)[0, 1])
    else:
        corr = 0.0

    results['length_correlation'] = {
        'correlation': corr,
        'min_length': min(lengths),
        'max_length': max(lengths),
        'per_prompt': length_analysis,
    }
    print(f"  Length-AUROC correlation: {corr:.4f}")
    print(f"  Length range: {min(lengths)}-{max(lengths)} chars")

    # ========== 4. Best and Worst Prompts ==========
    print("\n=== Best and Worst Prompts ===")

    sorted_prompts = sorted(per_prompt.items(), key=lambda x: x[1]['mean_auroc'])
    worst = sorted_prompts[0]
    best = sorted_prompts[-1]

    best_worst = {
        'best_prompt': best[0],
        'best_auroc': best[1]['mean_auroc'],
        'best_text': best[1]['prompt'],
        'worst_prompt': worst[0],
        'worst_auroc': worst[1]['mean_auroc'],
        'worst_text': worst[1]['prompt'],
        'range': best[1]['mean_auroc'] - worst[1]['mean_auroc'],
    }
    results['best_worst'] = best_worst
    print(f"  Best: {best[0]} (AUROC={best[1]['mean_auroc']:.4f})")
    print(f"  Worst: {worst[0]} (AUROC={worst[1]['mean_auroc']:.4f})")
    print(f"  Range: {best_worst['range']:.4f}")

    # ========== 5. Universal Prompt Detection ==========
    print("\n=== Universal Prompt Detection ===")

    # Train on one prompt, test on all others
    train_prompt = prompts[0]  # standard prompt
    train_embs = {}
    for seed in seeds[:5]:
        train_embs[seed] = extract_hidden(model, processor, images[seed], train_prompt)
    train_centroid = np.mean(list(train_embs.values()), axis=0)

    universal = {}
    for pi, test_prompt in enumerate(prompts):
        id_dists = []
        for seed in seeds[:5]:
            emb = extract_hidden(model, processor, images[seed], test_prompt)
            id_dists.append(cosine_dist(train_centroid, emb))

        per_ct = {}
        for ct in ctypes:
            ood_dists = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                corrupt_emb = extract_hidden(model, processor, corrupt_img, test_prompt)
                ood_dists.append(cosine_dist(train_centroid, corrupt_emb))
            per_ct[ct] = float(compute_auroc(id_dists, ood_dists))

        universal[f"prompt_{pi}"] = {
            'per_corruption': per_ct,
            'mean_auroc': float(np.mean(list(per_ct.values()))),
        }

    # Summary
    all_aurocs = [v['mean_auroc'] for v in universal.values()]
    results['universal_prompt'] = {
        'mean_auroc': float(np.mean(all_aurocs)),
        'min_auroc': float(min(all_aurocs)),
        'all_perfect': all(a >= 1.0 - 1e-10 for a in all_aurocs),
        'per_prompt': universal,
    }
    print(f"  Universal (train on P0): mean={np.mean(all_aurocs):.4f}, "
          f"min={min(all_aurocs):.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/prompt_diversity_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
