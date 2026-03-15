#!/usr/bin/env python3
"""Experiment 384: Action Token Distribution Under Corruption

How does corruption affect the distribution of predicted action tokens?
1. Token entropy per action dimension
2. Token probability concentration (top-k coverage)
3. Token ID shift patterns (which tokens change?)
4. Cross-dimension correlation of token changes
5. Token diversity: how many unique tokens per dimension across scenes
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
    """Generate action tokens and return token IDs + logits."""
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
    
    # Get logit distributions for action tokens (31744-31999)
    action_logits = []
    for i, scores in enumerate(generated.scores[:n_tokens]):
        logits = scores[0, 31744:32000].float().cpu().numpy()
        action_logits.append(logits)
    
    return token_ids, action_logits

def entropy(probs):
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log2(p))

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

    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # Get clean tokens
    print("Getting clean action tokens...")
    clean_tokens = {}
    clean_logits = {}
    for seed in seeds:
        toks, logits = get_action_tokens(model, processor, images[seed], prompt)
        clean_tokens[seed] = toks
        clean_logits[seed] = logits

    # ========== 1. Token Entropy ==========
    print("\n=== Token Entropy ===")

    token_entropy = {'clean': {}}
    # Clean entropy
    for dim in range(7):
        dim_entropies = []
        for seed in seeds:
            logits = clean_logits[seed][dim]
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            dim_entropies.append(entropy(probs))
        token_entropy['clean'][f'dim{dim}'] = {
            'mean_entropy': float(np.mean(dim_entropies)),
            'std_entropy': float(np.std(dim_entropies)),
        }

    for ct in ctypes:
        ct_entropy = {}
        for dim in range(7):
            dim_entropies = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                _, logits = get_action_tokens(model, processor, corrupt_img, prompt)
                probs = np.exp(logits[dim] - np.max(logits[dim]))
                probs = probs / np.sum(probs)
                dim_entropies.append(entropy(probs))
            ct_entropy[f'dim{dim}'] = {
                'mean_entropy': float(np.mean(dim_entropies)),
                'std_entropy': float(np.std(dim_entropies)),
            }
        token_entropy[ct] = ct_entropy
        print(f"  {ct}: entropies=" +
              ", ".join(f"d{d}={ct_entropy[f'dim{d}']['mean_entropy']:.2f}" for d in range(7)))

    results['token_entropy'] = token_entropy

    # ========== 2. Top-K Coverage ==========
    print("\n=== Top-K Coverage ===")

    topk_coverage = {}
    for ct in ['clean'] + ctypes:
        coverage = {f'top{k}': [] for k in [1, 3, 5, 10]}
        for seed in seeds[:5]:
            if ct == 'clean':
                logits_list = clean_logits[seed]
            else:
                _, logits_list = get_action_tokens(model, processor,
                    apply_corruption(images[seed], ct, 0.5), prompt)
            for dim in range(7):
                probs = np.exp(logits_list[dim] - np.max(logits_list[dim]))
                probs = probs / np.sum(probs)
                sorted_probs = np.sort(probs)[::-1]
                for k in [1, 3, 5, 10]:
                    coverage[f'top{k}'].append(float(np.sum(sorted_probs[:k])))

        topk_coverage[ct] = {k: float(np.mean(v)) for k, v in coverage.items()}
        print(f"  {ct}: top1={topk_coverage[ct]['top1']:.4f}, "
              f"top5={topk_coverage[ct]['top5']:.4f}")

    results['topk_coverage'] = topk_coverage

    # ========== 3. Token ID Shift ==========
    print("\n=== Token ID Shift ===")

    token_shifts = {}
    for ct in ctypes:
        shifts = []
        changes = []
        for seed in seeds[:5]:
            corrupt_toks, _ = get_action_tokens(model, processor,
                apply_corruption(images[seed], ct, 0.5), prompt)
            shift = corrupt_toks.astype(np.int32) - clean_tokens[seed].astype(np.int32)
            shifts.append(shift)
            changes.append(np.sum(shift != 0))

        shifts = np.array(shifts)
        token_shifts[ct] = {
            'mean_shift_per_dim': [float(np.mean(shifts[:, d])) for d in range(7)],
            'std_shift_per_dim': [float(np.std(shifts[:, d])) for d in range(7)],
            'mean_abs_shift': [float(np.mean(np.abs(shifts[:, d]))) for d in range(7)],
            'dims_changed': float(np.mean(changes)),
            'any_change_rate': float(np.mean(np.any(shifts != 0, axis=1))),
        }
        print(f"  {ct}: dims_changed={np.mean(changes):.1f}/7, "
              f"abs_shifts={[f'{np.mean(np.abs(shifts[:, d])):.1f}' for d in range(7)]}")

    results['token_shifts'] = token_shifts

    # ========== 4. Token Diversity ==========
    print("\n=== Token Diversity ===")

    diversity = {'clean': {}}
    for dim in range(7):
        unique_tokens = set(clean_tokens[s][dim] for s in seeds)
        diversity['clean'][f'dim{dim}'] = len(unique_tokens)

    for ct in ctypes:
        ct_div = {}
        for dim in range(7):
            corrupt_tokens_set = set()
            for seed in seeds[:5]:
                corrupt_toks, _ = get_action_tokens(model, processor,
                    apply_corruption(images[seed], ct, 0.5), prompt)
                corrupt_tokens_set.add(int(corrupt_toks[dim]))
            ct_div[f'dim{dim}'] = len(corrupt_tokens_set)
        diversity[ct] = ct_div

    results['token_diversity'] = diversity
    for name in ['clean'] + ctypes:
        print(f"  {name}: " + ", ".join(f"d{d}={diversity[name][f'dim{d}']}" for d in range(7)))

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/token_distribution_{ts}.json"
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
