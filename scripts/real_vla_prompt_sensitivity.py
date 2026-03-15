#!/usr/bin/env python3
"""Experiment 350: Prompt Sensitivity Deep Dive

Extended prompt analysis with 15 diverse prompts:
1. Detection AUROC across all prompts
2. Embedding distance magnitude variation
3. Cross-prompt embedding similarity (do prompts share detection space?)
4. Adversarial prompt construction (minimize detection signal)
5. Prompt length vs detection sensitivity
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

    # 15 diverse prompts
    prompts = {
        'standard': "In: What action should the robot take to pick up the object?\nOut:",
        'drive': "In: What action should the robot take to drive forward safely?\nOut:",
        'place': "In: What action should the robot take to place the cup on the table?\nOut:",
        'push': "In: What action should the robot take to push the block left?\nOut:",
        'open': "In: What action should the robot take to open the drawer?\nOut:",
        'pour': "In: What action should the robot take to pour water into the glass?\nOut:",
        'stack': "In: What action should the robot take to stack the blocks?\nOut:",
        'minimal': "In: Act.\nOut:",
        'verbose': "In: Given the current visual observation, what is the optimal 7-dimensional action vector the robot should execute to accomplish the grasping task?\nOut:",
        'no_context': "In: What should happen next?\nOut:",
        'imperative': "In: Pick up the red block now.\nOut:",
        'question': "In: How should the arm move?\nOut:",
        'third_person': "In: Describe the robot's next action to grasp the object.\nOut:",
        'navigate': "In: What action should the robot take to navigate to the door?\nOut:",
        'avoid': "In: What action should the robot take to avoid the obstacle?\nOut:",
    }

    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Base scenes
    seeds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    scenes = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scenes[seed] = Image.fromarray(px)

    # ========== 1. Per-prompt detection performance ==========
    print("\n=== Per-Prompt Detection Performance ===")

    prompt_results = {}
    prompt_embeddings = {}  # store for cross-prompt analysis

    for pname, prompt in prompts.items():
        print(f"  Testing prompt: {pname}...")

        cal_embs = {}
        for seed in seeds:
            cal_embs[seed] = extract_hidden(model, processor, scenes[seed], prompt)

        # ID distances
        id_dists = []
        for seed in seeds:
            emb = extract_hidden(model, processor, scenes[seed], prompt)
            id_dists.append(float(cosine_dist(cal_embs[seed], emb)))

        # OOD distances per type
        per_type = {}
        for ct in ctypes:
            ood_dists = []
            for seed in seeds:
                img = apply_corruption(scenes[seed], ct, 0.5)
                emb = extract_hidden(model, processor, img, prompt)
                ood_dists.append(float(cosine_dist(cal_embs[seed], emb)))

            auroc = compute_auroc(id_dists, ood_dists)
            per_type[ct] = {
                'auroc': float(auroc),
                'mean_dist': float(np.mean(ood_dists)),
                'min_dist': float(min(ood_dists)),
                'max_dist': float(max(ood_dists)),
            }

        # Store one scene's embeddings for cross-prompt analysis
        seed0 = seeds[0]
        prompt_embeddings[pname] = {
            'clean': cal_embs[seed0],
            'fog': extract_hidden(model, processor, apply_corruption(scenes[seed0], 'fog', 0.5), prompt),
        }

        all_aurocs = [per_type[ct]['auroc'] for ct in ctypes]
        prompt_results[pname] = {
            'prompt_length': len(prompt),
            'id_mean': float(np.mean(id_dists)),
            'id_max': float(max(id_dists)),
            'per_type': per_type,
            'mean_auroc': float(np.mean(all_aurocs)),
            'min_auroc': float(min(all_aurocs)),
            'all_perfect': all(a == 1.0 for a in all_aurocs),
        }
        auroc_strs = ', '.join(str(ct) + '=' + format(per_type[ct]['auroc'], '.3f') for ct in ctypes)
        print(f"    AUROCs: {auroc_strs}")

    results['per_prompt'] = prompt_results

    # ========== 2. Cross-prompt embedding similarity ==========
    print("\n=== Cross-Prompt Embedding Similarity ===")

    cross_prompt = {}
    pnames = list(prompts.keys())

    for i, p1 in enumerate(pnames):
        for j, p2 in enumerate(pnames):
            if i >= j:
                continue
            # Clean embedding similarity
            clean_sim = cosine_dist(prompt_embeddings[p1]['clean'], prompt_embeddings[p2]['clean'])
            # Fog embedding similarity
            fog_sim = cosine_dist(prompt_embeddings[p1]['fog'], prompt_embeddings[p2]['fog'])

            key = f"{p1}_vs_{p2}"
            cross_prompt[key] = {
                'clean_distance': float(clean_sim),
                'fog_distance': float(fog_sim),
            }

    # Summary stats
    clean_dists_cross = [v['clean_distance'] for v in cross_prompt.values()]
    fog_dists_cross = [v['fog_distance'] for v in cross_prompt.values()]
    cross_summary = {
        'mean_clean_dist': float(np.mean(clean_dists_cross)),
        'max_clean_dist': float(max(clean_dists_cross)),
        'mean_fog_dist': float(np.mean(fog_dists_cross)),
        'max_fog_dist': float(max(fog_dists_cross)),
        'n_pairs': len(cross_prompt),
    }
    print(f"  Mean cross-prompt clean dist: {np.mean(clean_dists_cross):.6f}")
    print(f"  Max cross-prompt clean dist: {max(clean_dists_cross):.6f}")

    results['cross_prompt'] = {'pairs': cross_prompt, 'summary': cross_summary}

    # ========== 3. Prompt length correlation ==========
    print("\n=== Prompt Length vs Detection ===")

    length_corr = {}
    for pname in pnames:
        pr = prompt_results[pname]
        length_corr[pname] = {
            'length': pr['prompt_length'],
            'mean_auroc': pr['mean_auroc'],
            'fog_dist': pr['per_type']['fog']['mean_dist'],
            'night_dist': pr['per_type']['night']['mean_dist'],
        }

    # Compute correlation
    lengths = [length_corr[p]['length'] for p in pnames]
    fog_dists_len = [length_corr[p]['fog_dist'] for p in pnames]
    # Pearson correlation
    if np.std(lengths) > 0 and np.std(fog_dists_len) > 0:
        corr = np.corrcoef(lengths, fog_dists_len)[0, 1]
    else:
        corr = 0.0

    results['length_correlation'] = {
        'per_prompt': length_corr,
        'pearson_r_fog': float(corr),
    }
    print(f"  Pearson r (length vs fog dist): {corr:.4f}")

    # ========== 4. Distance magnitude ranking ==========
    print("\n=== Distance Magnitude Ranking ===")

    ranking = {}
    for ct in ctypes:
        sorted_prompts = sorted(pnames, key=lambda p: prompt_results[p]['per_type'][ct]['mean_dist'], reverse=True)
        ranking[ct] = {
            'most_sensitive': sorted_prompts[0],
            'least_sensitive': sorted_prompts[-1],
            'max_dist': prompt_results[sorted_prompts[0]]['per_type'][ct]['mean_dist'],
            'min_dist': prompt_results[sorted_prompts[-1]]['per_type'][ct]['mean_dist'],
            'ratio': prompt_results[sorted_prompts[0]]['per_type'][ct]['mean_dist'] /
                     max(prompt_results[sorted_prompts[-1]]['per_type'][ct]['mean_dist'], 1e-10),
        }
        print(f"  {ct}: most={sorted_prompts[0]} ({ranking[ct]['max_dist']:.6f}), "
              f"least={sorted_prompts[-1]} ({ranking[ct]['min_dist']:.6f}), "
              f"ratio={ranking[ct]['ratio']:.2f}x")

    results['ranking'] = ranking

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/prompt_sensitivity_{ts}.json"
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
