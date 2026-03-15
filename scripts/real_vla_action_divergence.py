#!/usr/bin/env python3
"""Experiment 375: Action Space Divergence Under Corruption

Detailed analysis of how corruption affects the action output space:
1. Per-dimension action divergence: which action dims most affected
2. Action magnitude vs direction: does corruption scale or rotate actions?
3. Action entropy: does corruption increase action uncertainty?
4. Cross-scene action consistency: do different scenes corrupt similarly?
5. Action token distribution shift: which tokens change most
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

def get_action_tokens(model, processor, image, prompt, n_tokens=7):
    """Get the predicted action token IDs and logits."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    token_ids = generated.sequences[0, -n_tokens:].cpu().numpy()
    scores = [s[0].float().cpu() for s in generated.scores[:n_tokens]]
    return token_ids, scores

def tokens_to_actions(token_ids, n_bins=256, min_val=-1.0, max_val=1.0):
    """Convert OpenVLA action tokens to continuous values."""
    bin_width = (max_val - min_val) / n_bins
    actions = []
    for tid in token_ids:
        bin_idx = tid - 31744  # OpenVLA action token offset
        if 0 <= bin_idx < n_bins:
            val = min_val + (bin_idx + 0.5) * bin_width
        else:
            val = 0.0
        actions.append(val)
    return np.array(actions)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate images
    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # ========== 1. Collect Clean and Corrupt Actions ==========
    print("\n=== Collecting Actions ===")

    clean_tokens = {}
    clean_actions = {}
    clean_scores = {}
    for seed in seeds:
        tids, scores = get_action_tokens(model, processor, images[seed], prompt)
        clean_tokens[seed] = tids
        clean_actions[seed] = tokens_to_actions(tids)
        clean_scores[seed] = scores
        print(f"  clean_s{seed}: tokens={tids.tolist()}, actions={clean_actions[seed].round(3).tolist()}")

    corrupt_tokens = {ct: {} for ct in ctypes}
    corrupt_actions = {ct: {} for ct in ctypes}
    corrupt_scores = {ct: {} for ct in ctypes}
    for ct in ctypes:
        for seed in seeds:
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            tids, scores = get_action_tokens(model, processor, corrupt_img, prompt)
            corrupt_tokens[ct][seed] = tids
            corrupt_actions[ct][seed] = tokens_to_actions(tids)
            corrupt_scores[ct][seed] = scores
        print(f"  {ct}: collected {len(seeds)} corrupt actions")

    # ========== 2. Per-Dimension Action Divergence ==========
    print("\n=== Per-Dimension Divergence ===")

    dim_divergence = {}
    for ct in ctypes:
        per_dim_diff = []
        for d in range(7):
            diffs = [abs(clean_actions[s][d] - corrupt_actions[ct][s][d]) for s in seeds]
            per_dim_diff.append({
                'mean_diff': float(np.mean(diffs)),
                'max_diff': float(max(diffs)),
                'std_diff': float(np.std(diffs)),
                'token_changed_rate': float(sum(1 for s in seeds
                    if clean_tokens[s][d] != corrupt_tokens[ct][s][d]) / len(seeds)),
            })
        dim_divergence[ct] = per_dim_diff
        most_affected = max(range(7), key=lambda d: per_dim_diff[d]['mean_diff'])
        least_affected = min(range(7), key=lambda d: per_dim_diff[d]['mean_diff'])
        print(f"  {ct}: most_affected=dim{most_affected} ({per_dim_diff[most_affected]['mean_diff']:.4f}), "
              f"least=dim{least_affected} ({per_dim_diff[least_affected]['mean_diff']:.4f})")

    results['per_dim_divergence'] = dim_divergence

    # ========== 3. Action Vector Analysis ==========
    print("\n=== Action Vector Analysis ===")

    vector_analysis = {}
    for ct in ctypes:
        magnitudes_clean = [float(np.linalg.norm(clean_actions[s])) for s in seeds]
        magnitudes_corrupt = [float(np.linalg.norm(corrupt_actions[ct][s])) for s in seeds]

        cos_sims = []
        for s in seeds:
            a, b = clean_actions[s], corrupt_actions[ct][s]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-10 and nb > 1e-10:
                cos_sims.append(float(np.dot(a, b) / (na * nb)))
            else:
                cos_sims.append(0.0)

        l2_dists = [float(np.linalg.norm(clean_actions[s] - corrupt_actions[ct][s])) for s in seeds]

        vector_analysis[ct] = {
            'clean_magnitude_mean': float(np.mean(magnitudes_clean)),
            'corrupt_magnitude_mean': float(np.mean(magnitudes_corrupt)),
            'magnitude_ratio': float(np.mean(magnitudes_corrupt) / max(np.mean(magnitudes_clean), 1e-10)),
            'cosine_similarity_mean': float(np.mean(cos_sims)),
            'cosine_similarity_min': float(min(cos_sims)),
            'l2_distance_mean': float(np.mean(l2_dists)),
            'l2_distance_max': float(max(l2_dists)),
            'direction_change': float(1.0 - np.mean(cos_sims)),
        }
        print(f"  {ct}: cos_sim={np.mean(cos_sims):.4f}, "
              f"mag_ratio={vector_analysis[ct]['magnitude_ratio']:.4f}, "
              f"L2={np.mean(l2_dists):.4f}")

    results['vector_analysis'] = vector_analysis

    # ========== 4. Token-Level Analysis ==========
    print("\n=== Token-Level Analysis ===")

    token_analysis = {}
    for ct in ctypes:
        total_tokens = 0
        changed_tokens = 0
        token_shifts = []

        for s in seeds:
            for d in range(7):
                total_tokens += 1
                c_tok = int(clean_tokens[s][d])
                x_tok = int(corrupt_tokens[ct][s][d])
                if c_tok != x_tok:
                    changed_tokens += 1
                    token_shifts.append(abs(c_tok - x_tok))

        # Top-1 confidence for clean vs corrupt
        clean_conf = []
        corrupt_conf = []
        for s in seeds:
            for d in range(min(7, len(clean_scores[s]))):
                probs = torch.softmax(clean_scores[s][d], dim=-1)
                clean_conf.append(float(probs.max()))
            for d in range(min(7, len(corrupt_scores[ct][s]))):
                probs = torch.softmax(corrupt_scores[ct][s][d], dim=-1)
                corrupt_conf.append(float(probs.max()))

        token_analysis[ct] = {
            'token_change_rate': float(changed_tokens / total_tokens),
            'mean_token_shift': float(np.mean(token_shifts)) if token_shifts else 0,
            'max_token_shift': int(max(token_shifts)) if token_shifts else 0,
            'clean_top1_conf_mean': float(np.mean(clean_conf)),
            'corrupt_top1_conf_mean': float(np.mean(corrupt_conf)),
            'confidence_ratio': float(np.mean(corrupt_conf) / max(np.mean(clean_conf), 1e-10)),
        }
        print(f"  {ct}: change_rate={token_analysis[ct]['token_change_rate']:.3f}, "
              f"mean_shift={token_analysis[ct]['mean_token_shift']:.1f}, "
              f"conf_ratio={token_analysis[ct]['confidence_ratio']:.4f}")

    results['token_analysis'] = token_analysis

    # ========== 5. Cross-Scene Action Consistency ==========
    print("\n=== Cross-Scene Consistency ===")

    consistency = {}
    for ct in ctypes:
        corrupt_act_arr = np.array([corrupt_actions[ct][s] for s in seeds])
        clean_act_arr = np.array([clean_actions[s] for s in seeds])

        corrupt_cos_sims = []
        clean_cos_sims = []
        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):
                a, b = corrupt_act_arr[i], corrupt_act_arr[j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-10 and nb > 1e-10:
                    corrupt_cos_sims.append(float(np.dot(a, b) / (na * nb)))
                a, b = clean_act_arr[i], clean_act_arr[j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-10 and nb > 1e-10:
                    clean_cos_sims.append(float(np.dot(a, b) / (na * nb)))

        clean_var = [float(np.var(clean_act_arr[:, d])) for d in range(7)]
        corrupt_var = [float(np.var(corrupt_act_arr[:, d])) for d in range(7)]

        consistency[ct] = {
            'clean_pairwise_cos_mean': float(np.mean(clean_cos_sims)) if clean_cos_sims else 0,
            'corrupt_pairwise_cos_mean': float(np.mean(corrupt_cos_sims)) if corrupt_cos_sims else 0,
            'clean_per_dim_var': clean_var,
            'corrupt_per_dim_var': corrupt_var,
            'variance_ratio': float(np.mean(corrupt_var) / max(np.mean(clean_var), 1e-10)),
        }
        print(f"  {ct}: clean_cos={consistency[ct]['clean_pairwise_cos_mean']:.4f}, "
              f"corrupt_cos={consistency[ct]['corrupt_pairwise_cos_mean']:.4f}, "
              f"var_ratio={consistency[ct]['variance_ratio']:.4f}")

    results['cross_scene_consistency'] = consistency

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/action_divergence_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, torch.Tensor): return obj.tolist()
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
