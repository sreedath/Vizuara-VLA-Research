#!/usr/bin/env python3
"""Experiment 362: Action Degradation Under Corruption

How do corruptions affect the actual action outputs?
1. Action token distribution shift per corruption
2. Action magnitude changes
3. Action direction changes (cosine similarity of action vectors)
4. Severity-dependent action degradation curves
5. Safety-critical action dimension analysis
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

def get_action_tokens(model, processor, image, prompt, n_actions=7):
    """Get the predicted action tokens."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=n_actions + 5,
            do_sample=False,
        )
    input_len = inputs['input_ids'].shape[1]
    gen_tokens = output[0, input_len:].cpu().tolist()
    return gen_tokens

def tokens_to_actions(tokens, n_bins=256, action_range=(-1, 1)):
    """Convert action tokens to continuous values."""
    actions = []
    for t in tokens:
        bin_idx = t - 31744
        if 0 <= bin_idx < n_bins:
            val = action_range[0] + (bin_idx + 0.5) * (action_range[1] - action_range[0]) / n_bins
            actions.append(float(val))
    return actions

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

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 1500, 100))[:15]
    images = {}
    clean_tokens = {}
    clean_actions = {}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        tokens = get_action_tokens(model, processor, images[seed], prompt)
        clean_tokens[seed] = tokens
        clean_actions[seed] = tokens_to_actions(tokens)

    print(f"  Generated {len(seeds)} scenes, typical action len: {len(clean_actions[seeds[0]])}")

    # ========== 1. Token Shift Per Corruption ==========
    print("\n=== Action Token Shift ===")

    token_shift = {}
    for ct in ctypes:
        token_diffs = []
        action_diffs = []
        n_changed = 0
        n_total = 0

        for seed in seeds:
            img = apply_corruption(images[seed], ct, 0.5)
            corrupt_tok = get_action_tokens(model, processor, img, prompt)
            corrupt_act = tokens_to_actions(corrupt_tok)

            clean = clean_tokens[seed]
            min_len = min(len(clean), len(corrupt_tok))
            for i in range(min_len):
                n_total += 1
                if clean[i] != corrupt_tok[i]:
                    n_changed += 1
                    token_diffs.append(abs(clean[i] - corrupt_tok[i]))

            clean_a = clean_actions[seed]
            min_a = min(len(clean_a), len(corrupt_act))
            if min_a > 0:
                for i in range(min_a):
                    action_diffs.append(abs(clean_a[i] - corrupt_act[i]))

        token_shift[ct] = {
            'fraction_changed': n_changed / n_total if n_total > 0 else 0,
            'mean_token_diff': float(np.mean(token_diffs)) if token_diffs else 0,
            'max_token_diff': float(max(token_diffs)) if token_diffs else 0,
            'mean_action_diff': float(np.mean(action_diffs)) if action_diffs else 0,
            'max_action_diff': float(max(action_diffs)) if action_diffs else 0,
            'n_total': n_total,
            'n_changed': n_changed,
        }
        print(f"  {ct}: {n_changed}/{n_total} tokens changed ({n_changed/n_total*100:.1f}%), "
              + (f"mean_action_diff={np.mean(action_diffs):.4f}" if action_diffs else "no action diffs"))

    results['token_shift'] = token_shift

    # ========== 2. Severity-Dependent Action Change ==========
    print("\n=== Severity vs Action Change ===")

    sev_action = {}
    for ct in ctypes:
        per_sev = {}
        for sev in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            n_changed = 0
            n_total = 0
            action_diffs = []

            for seed in seeds[:5]:
                img = apply_corruption(images[seed], ct, sev)
                corrupt_tok = get_action_tokens(model, processor, img, prompt)
                corrupt_act = tokens_to_actions(corrupt_tok)

                clean = clean_tokens[seed]
                min_len = min(len(clean), len(corrupt_tok))
                for i in range(min_len):
                    n_total += 1
                    if clean[i] != corrupt_tok[i]:
                        n_changed += 1

                clean_a = clean_actions[seed]
                min_a = min(len(clean_a), len(corrupt_act))
                for i in range(min_a):
                    action_diffs.append(abs(clean_a[i] - corrupt_act[i]))

            per_sev[str(sev)] = {
                'fraction_changed': n_changed / n_total if n_total > 0 else 0,
                'mean_action_diff': float(np.mean(action_diffs)) if action_diffs else 0,
            }

        sev_action[ct] = per_sev
        changes = [per_sev[str(s)]['fraction_changed'] for s in [0.05, 0.1, 0.5, 1.0]]
        print(f"  {ct}: change@[0.05,0.1,0.5,1.0] = [{', '.join(f'{c:.3f}' for c in changes)}]")

    results['severity_action'] = sev_action

    # ========== 3. Per-Dimension Analysis ==========
    print("\n=== Per-Dimension Action Analysis ===")

    dim_analysis = {}
    for ct in ctypes:
        dim_diffs = {}
        for seed in seeds:
            img = apply_corruption(images[seed], ct, 0.5)
            corrupt_act = tokens_to_actions(get_action_tokens(model, processor, img, prompt))
            clean_a = clean_actions[seed]
            min_d = min(len(clean_a), len(corrupt_act))
            for d in range(min_d):
                if d not in dim_diffs:
                    dim_diffs[d] = []
                dim_diffs[d].append(abs(clean_a[d] - corrupt_act[d]))

        per_dim = {}
        for d in sorted(dim_diffs.keys()):
            per_dim[str(d)] = {
                'mean_diff': float(np.mean(dim_diffs[d])),
                'max_diff': float(max(dim_diffs[d])),
                'any_changed': any(v > 0 for v in dim_diffs[d]),
            }

        dim_analysis[ct] = per_dim
        dims_changed = sum(1 for v in per_dim.values() if v['any_changed'])
        print(f"  {ct}: {dims_changed}/{len(per_dim)} dims changed")

    results['per_dimension'] = dim_analysis

    # ========== 4. Action Vector Cosine Similarity ==========
    print("\n=== Action Vector Similarity ===")

    vec_sim = {}
    for ct in ctypes:
        sims = []
        for seed in seeds:
            clean_a = np.array(clean_actions[seed])
            img = apply_corruption(images[seed], ct, 0.5)
            corrupt_act = np.array(tokens_to_actions(get_action_tokens(model, processor, img, prompt)))
            min_d = min(len(clean_a), len(corrupt_act))
            if min_d > 0:
                a = clean_a[:min_d]
                b = corrupt_act[:min_d]
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a > 1e-10 and norm_b > 1e-10:
                    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
                    sims.append(cos_sim)

        vec_sim[ct] = {
            'mean_cos_sim': float(np.mean(sims)) if sims else 0,
            'min_cos_sim': float(min(sims)) if sims else 0,
            'std_cos_sim': float(np.std(sims)) if sims else 0,
        }
        if sims:
            print(f"  {ct}: cos_sim={np.mean(sims):.4f} (min={min(sims):.4f})")
        else:
            print(f"  {ct}: no data")

    results['action_similarity'] = vec_sim

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/action_degradation_{ts}.json"
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
