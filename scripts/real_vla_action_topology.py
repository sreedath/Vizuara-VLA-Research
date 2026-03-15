#!/usr/bin/env python3
"""Experiment 349: Action Space Topology Under Corruption

Analyzes the structure of OpenVLA's 7D action token outputs:
1. Action token distribution under different corruptions
2. Hamming distance between clean and corrupted action vectors
3. Per-dimension sensitivity (which action dims change first?)
4. Action trajectory coherence under gradual corruption
5. Cross-scene action diversity vs corruption uniformity
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def get_action_tokens(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    tokens = out[0, inputs['input_ids'].shape[1]:].cpu().tolist()
    return tokens[:7]

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

def token_to_bin(token_id):
    """Convert VLA action token to bin index (0-255)"""
    return token_id - 31744

def hamming_distance(tokens1, tokens2):
    """Count number of differing positions"""
    return sum(1 for a, b in zip(tokens1, tokens2) if a != b)

def action_l1_distance(tokens1, tokens2):
    """L1 distance in bin space"""
    return sum(abs(token_to_bin(a) - token_to_bin(b)) for a, b in zip(tokens1, tokens2))

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

    # ========== 1. Multi-scene action distribution ==========
    print("\n=== Multi-Scene Action Distribution ===")

    seeds = list(range(0, 2000, 100))[:20]
    scene_data = {}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(px)

        clean_tokens = get_action_tokens(model, processor, img, prompt)
        clean_bins = [token_to_bin(t) for t in clean_tokens]

        corrupt_tokens = {}
        for ct in ctypes:
            corrupted = apply_corruption(img, ct, 0.5)
            tokens = get_action_tokens(model, processor, corrupted, prompt)
            bins = [token_to_bin(t) for t in tokens]
            corrupt_tokens[ct] = {
                'tokens': tokens,
                'bins': bins,
                'hamming': hamming_distance(clean_tokens, tokens),
                'l1': action_l1_distance(clean_tokens, tokens),
                'changed_dims': [i for i in range(7) if clean_tokens[i] != tokens[i]],
            }

        scene_data[str(seed)] = {
            'clean_tokens': clean_tokens,
            'clean_bins': clean_bins,
            'corrupt': corrupt_tokens,
        }
        print(f"  Scene {seed}: clean={clean_bins}, "
              f"fog_hamming={corrupt_tokens['fog']['hamming']}, "
              f"night_hamming={corrupt_tokens['night']['hamming']}")

    results['scene_actions'] = scene_data

    # ========== 2. Per-dimension sensitivity ==========
    print("\n=== Per-Dimension Sensitivity ===")

    dim_sensitivity = {ct: [0] * 7 for ct in ctypes}
    dim_magnitude = {ct: [[] for _ in range(7)] for ct in ctypes}

    for seed_str, sd in scene_data.items():
        for ct in ctypes:
            for dim in sd['corrupt'][ct]['changed_dims']:
                dim_sensitivity[ct][dim] += 1
                clean_bin = sd['clean_bins'][dim]
                corrupt_bin = sd['corrupt'][ct]['bins'][dim]
                dim_magnitude[ct][dim].append(abs(corrupt_bin - clean_bin))

    sensitivity_summary = {}
    for ct in ctypes:
        per_dim = []
        for dim in range(7):
            per_dim.append({
                'change_frequency': dim_sensitivity[ct][dim] / len(seeds),
                'mean_magnitude': float(np.mean(dim_magnitude[ct][dim])) if dim_magnitude[ct][dim] else 0,
                'max_magnitude': int(max(dim_magnitude[ct][dim])) if dim_magnitude[ct][dim] else 0,
            })
        sensitivity_summary[ct] = per_dim
        freqs = [p['change_frequency'] for p in per_dim]
        print(f"  {ct} change freq by dim: {[f'{f:.2f}' for f in freqs]}")

    results['dim_sensitivity'] = sensitivity_summary

    # ========== 3. Severity-action trajectory ==========
    print("\n=== Severity-Action Trajectory ===")

    # Use seed=42 base scene
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    clean_tokens = get_action_tokens(model, processor, base_img, prompt)

    trajectories = {}
    for ct in ctypes:
        sevs = np.linspace(0, 1, 21)  # 0.0, 0.05, ..., 1.0
        traj = []

        for sev in sevs:
            if sev == 0:
                img = base_img
            else:
                img = apply_corruption(base_img, ct, sev)
            tokens = get_action_tokens(model, processor, img, prompt)
            bins = [token_to_bin(t) for t in tokens]
            hamming = hamming_distance(clean_tokens, tokens)
            l1 = action_l1_distance(clean_tokens, tokens)
            traj.append({
                'severity': float(sev),
                'tokens': tokens,
                'bins': bins,
                'hamming': hamming,
                'l1': l1,
            })

        # Find severity thresholds for 1st, 3rd, 5th, 7th dim change
        change_thresholds = {}
        for n_dims in [1, 3, 5, 7]:
            for t in traj:
                if t['hamming'] >= n_dims:
                    change_thresholds[str(n_dims)] = t['severity']
                    break
            else:
                change_thresholds[str(n_dims)] = None

        # Action trajectory smoothness: how often does direction reverse?
        reversals = 0
        for dim in range(7):
            bin_vals = [t['bins'][dim] for t in traj]
            for i in range(2, len(bin_vals)):
                if (bin_vals[i] - bin_vals[i-1]) * (bin_vals[i-1] - bin_vals[i-2]) < 0:
                    reversals += 1

        trajectories[ct] = {
            'trajectory': traj,
            'change_thresholds': change_thresholds,
            'total_reversals': reversals,
            'final_hamming': traj[-1]['hamming'],
            'final_l1': traj[-1]['l1'],
        }
        print(f"  {ct}: 1st change at sev={change_thresholds.get('1')}, "
              f"all 7 at sev={change_thresholds.get('7')}, "
              f"reversals={reversals}, final L1={traj[-1]['l1']}")

    results['trajectories'] = trajectories

    # ========== 4. Action convergence across corruptions ==========
    print("\n=== Action Convergence ===")

    # Do different corruptions converge to the same action at high severity?
    convergence = {}
    for sev in [0.3, 0.5, 0.7, 1.0]:
        action_sets = {}
        for ct in ctypes:
            img = apply_corruption(base_img, ct, sev)
            tokens = get_action_tokens(model, processor, img, prompt)
            action_sets[ct] = tokens

        # Pairwise Hamming between corruption types
        pairwise = {}
        for i, ct1 in enumerate(ctypes):
            for j, ct2 in enumerate(ctypes):
                if i >= j:
                    continue
                h = hamming_distance(action_sets[ct1], action_sets[ct2])
                l1 = action_l1_distance(action_sets[ct1], action_sets[ct2])
                pairwise[f"{ct1}_vs_{ct2}"] = {'hamming': h, 'l1': l1}

        mean_hamming = np.mean([p['hamming'] for p in pairwise.values()])
        convergence[str(sev)] = {
            'actions': {ct: [token_to_bin(t) for t in tokens] for ct, tokens in action_sets.items()},
            'pairwise': pairwise,
            'mean_hamming': float(mean_hamming),
        }
        print(f"  sev={sev}: mean inter-corruption hamming={mean_hamming:.1f}")

    results['convergence'] = convergence

    # ========== 5. Action entropy across scenes ==========
    print("\n=== Action Entropy Across Scenes ===")

    action_entropy = {}
    for ct in ['clean'] + ctypes:
        dim_entropies = []
        for dim in range(7):
            if ct == 'clean':
                bins = [scene_data[str(s)]['clean_bins'][dim] for s in seeds]
            else:
                bins = [scene_data[str(s)]['corrupt'][ct]['bins'][dim] for s in seeds]

            # Entropy of bin distribution
            unique, counts = np.unique(bins, return_counts=True)
            probs = counts / counts.sum()
            h = -sum(p * np.log2(p) for p in probs if p > 0)
            dim_entropies.append(float(h))

        action_entropy[ct] = {
            'per_dim_entropy': dim_entropies,
            'mean_entropy': float(np.mean(dim_entropies)),
            'max_entropy': float(max(dim_entropies)),
        }
        print(f"  {ct}: mean H={np.mean(dim_entropies):.3f} bits, "
              f"max H={max(dim_entropies):.3f} bits")

    results['action_entropy'] = action_entropy

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/action_topology_{ts}.json"
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
