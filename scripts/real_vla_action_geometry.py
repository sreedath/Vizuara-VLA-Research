#!/usr/bin/env python3
"""Experiment 334: Action Space Geometry (Real OpenVLA-7B)

Deep analysis of how corruptions affect the action output space:
1. Per-dimension action sensitivity to each corruption
2. Action token probability landscapes
3. Action space trajectories under increasing severity
4. Cross-corruption action overlap
5. Action reversibility analysis
6. Confidence calibration per action dimension
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

def get_action_logits(model, processor, image, prompt, n_steps=7):
    """Generate action tokens and return logits at each step."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    input_ids = inputs['input_ids']

    all_logits = []
    all_tokens = []

    for step in range(n_steps):
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :].float().cpu()
        # Action tokens: 31744-31999 (256 bins)
        action_logits = logits[31744:32000].numpy()

        token_id = logits.argmax().item()
        all_logits.append(action_logits)
        all_tokens.append(token_id)

        # Append token for next step
        new_token = torch.tensor([[token_id]], device=model.device)
        input_ids = torch.cat([input_ids, new_token], dim=1)
        if 'attention_mask' in inputs:
            inputs['attention_mask'] = torch.cat([
                inputs['attention_mask'],
                torch.ones(1, 1, device=model.device, dtype=inputs['attention_mask'].dtype)
            ], dim=1)
        inputs['input_ids'] = input_ids

    return all_tokens, all_logits

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    # Base image
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    # ========== 1. Clean action baseline ==========
    print("\n=== Clean Action Baseline ===")
    clean_tokens, clean_logits = get_action_logits(model, processor, base_img, prompt)
    clean_bins = [t - 31744 if 31744 <= t <= 31999 else t for t in clean_tokens]
    print(f"  Clean tokens: {clean_tokens}")
    print(f"  Clean bins: {clean_bins}")
    results['clean_baseline'] = {
        'tokens': clean_tokens,
        'bins': clean_bins,
    }

    # ========== 2. Per-corruption action change ==========
    print("\n=== Per-Corruption Action Changes ===")
    ctypes = ['fog', 'night', 'noise', 'blur']
    sevs = [0.1, 0.25, 0.5, 0.75, 1.0]

    action_changes = {}
    for ct in ctypes:
        ct_results = {}
        for sev in sevs:
            img = apply_corruption(base_img, ct, sev)
            tokens, logits = get_action_logits(model, processor, img, prompt)
            bins = [t - 31744 if 31744 <= t <= 31999 else t for t in tokens]

            # Per-dimension changes
            dim_changes = []
            for d in range(7):
                if d < len(bins) and d < len(clean_bins):
                    dim_changes.append(int(bins[d]) - int(clean_bins[d]))
                else:
                    dim_changes.append(0)

            n_changed = sum(1 for dc in dim_changes if dc != 0)
            total_deviation = sum(abs(dc) for dc in dim_changes)

            # Top-1 probability at each step
            top1_probs = []
            for step_logits in logits:
                probs = np.exp(step_logits - np.max(step_logits))
                probs = probs / probs.sum()
                top1_probs.append(float(np.max(probs)))

            ct_results[str(sev)] = {
                'tokens': tokens,
                'bins': bins,
                'dim_changes': dim_changes,
                'n_changed': n_changed,
                'total_deviation': total_deviation,
                'top1_probs': top1_probs,
            }
            print(f"  {ct}@{sev}: {n_changed}/7 changed, deviation={total_deviation}, bins={bins}")

        action_changes[ct] = ct_results

    results['action_changes'] = action_changes

    # ========== 3. Cross-corruption action overlap ==========
    print("\n=== Cross-Corruption Action Overlap ===")
    overlap_results = {}

    # Get actions at full severity for each corruption
    full_actions = {}
    for ct in ctypes:
        img = apply_corruption(base_img, ct, 1.0)
        tokens, _ = get_action_logits(model, processor, img, prompt)
        bins = [t - 31744 if 31744 <= t <= 31999 else t for t in tokens]
        full_actions[ct] = bins

    # Pairwise action similarity
    for i, ct1 in enumerate(ctypes):
        for ct2 in ctypes[i+1:]:
            same_dims = sum(1 for d in range(7) if full_actions[ct1][d] == full_actions[ct2][d])
            total_diff = sum(abs(full_actions[ct1][d] - full_actions[ct2][d]) for d in range(7))
            overlap_results[f"{ct1}_vs_{ct2}"] = {
                'same_dims': same_dims,
                'total_diff': total_diff,
                'actions1': full_actions[ct1],
                'actions2': full_actions[ct2],
            }
            print(f"  {ct1} vs {ct2}: {same_dims}/7 same, total_diff={total_diff}")

    results['cross_corruption_overlap'] = overlap_results

    # ========== 4. Per-dimension sensitivity ==========
    print("\n=== Per-Dimension Sensitivity ===")
    dim_sensitivity = {}

    for dim in range(7):
        dim_data = {}
        for ct in ctypes:
            changes = []
            for sev in sevs:
                if str(sev) in action_changes[ct]:
                    dc = action_changes[ct][str(sev)]['dim_changes']
                    if dim < len(dc):
                        changes.append(abs(dc[dim]))
                    else:
                        changes.append(0)

            dim_data[ct] = {
                'changes': changes,
                'max_change': max(changes) if changes else 0,
                'first_change_sev': next((sevs[i] for i, c in enumerate(changes) if c > 0), None),
            }

        dim_sensitivity[str(dim)] = dim_data
        print(f"  Dim {dim}: fog_max={dim_data['fog']['max_change']}, night_max={dim_data['night']['max_change']}, blur_max={dim_data['blur']['max_change']}")

    results['dim_sensitivity'] = dim_sensitivity

    # ========== 5. Logit distribution analysis ==========
    print("\n=== Logit Distribution Analysis ===")
    logit_analysis = {}

    for ct in ['clean'] + ctypes:
        if ct == 'clean':
            _, logits = get_action_logits(model, processor, base_img, prompt)
        else:
            img = apply_corruption(base_img, ct, 1.0)
            _, logits = get_action_logits(model, processor, img, prompt)

        per_step = []
        for step, step_logits in enumerate(logits):
            probs = np.exp(step_logits - np.max(step_logits))
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            top1 = float(np.max(probs))
            top5 = float(np.sort(probs)[-5:].sum())

            per_step.append({
                'entropy': float(entropy),
                'top1_prob': top1,
                'top5_prob': top5,
                'max_logit': float(np.max(step_logits)),
                'mean_logit': float(np.mean(step_logits)),
            })

        logit_analysis[ct] = per_step
        print(f"  {ct}: mean_entropy={np.mean([s['entropy'] for s in per_step]):.3f}, mean_top1={np.mean([s['top1_prob'] for s in per_step]):.3f}")

    results['logit_analysis'] = logit_analysis

    # ========== 6. Multi-scene action consistency ==========
    print("\n=== Multi-Scene Action Consistency ===")
    scene_consistency = {}

    for seed in [42, 99, 123, 777]:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scene_img = Image.fromarray(px)

        scene_clean_tokens, _ = get_action_logits(model, processor, scene_img, prompt)
        scene_clean_bins = [t - 31744 if 31744 <= t <= 31999 else t for t in scene_clean_tokens]

        per_ct = {}
        for ct in ctypes:
            img = apply_corruption(scene_img, ct, 0.5)
            tokens, _ = get_action_logits(model, processor, img, prompt)
            bins = [t - 31744 if 31744 <= t <= 31999 else t for t in tokens]
            n_changed = sum(1 for d in range(7) if d < len(bins) and d < len(scene_clean_bins) and bins[d] != scene_clean_bins[d])
            per_ct[ct] = {'n_changed': n_changed, 'bins': bins}

        scene_consistency[str(seed)] = {
            'clean_bins': scene_clean_bins,
            'corrupted': per_ct,
        }
        print(f"  Scene {seed}: clean={scene_clean_bins}, changed: {', '.join(f'{ct}={v[\"n_changed\"]}' for ct, v in per_ct.items())}")

    results['scene_consistency'] = scene_consistency

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/action_geometry_{ts}.json"
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
