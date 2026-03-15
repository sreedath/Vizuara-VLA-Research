#!/usr/bin/env python3
"""Experiment 444: Corruption Impact on Action Predictions

Directly measures how corruptions change the robot's predicted actions.
Beyond detection — what HAPPENS to the robot's behavior under corruption?
This connects OOD detection to actual safety consequences.

Tests:
1. Action token shift under each corruption
2. Action dimension-wise analysis (which dims change most)
3. Action confidence vs corruption severity
4. Predicted trajectory deviation
5. Action entropy analysis
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

def get_action_logits(model, processor, image, prompt, n_action_tokens=7):
    """Get logits for each action token position."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=n_action_tokens,
            do_sample=False, output_scores=True, return_dict_in_generate=True
        )
    scores = [s.float().cpu() for s in output.scores]  # list of (1, vocab_size)
    tokens = output.sequences[0, -n_action_tokens:].cpu().tolist()
    return tokens, scores

def token_to_action_value(token_id, bin_start=31744, n_bins=256):
    """Convert action token to continuous value in [-1, 1]."""
    bin_idx = token_id - bin_start
    if bin_idx < 0 or bin_idx >= n_bins:
        return 0.0
    return -1.0 + (2.0 * bin_idx + 1.0) / n_bins

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']
    action_dims = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

    seeds = [42, 123, 456, 789, 999, 1111, 2222, 3333]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    results = {"n_scenes": len(scenes)}

    # === Test 1: Clean action predictions ===
    print("\n=== Clean Action Predictions ===")
    clean_actions = {}
    clean_tokens_all = {}
    for s_idx, s in enumerate(scenes):
        tokens, scores = get_action_logits(model, processor, s, prompt)
        values = [token_to_action_value(t) for t in tokens]
        clean_actions[s_idx] = values
        clean_tokens_all[s_idx] = tokens

        # Top-1 probabilities
        probs = [torch.softmax(sc[0], dim=0).max().item() for sc in scores]
        if s_idx < 3:
            print(f"  Scene {s_idx}: tokens={tokens}, values=[{', '.join(f'{v:.3f}' for v in values)}], top1_probs=[{', '.join(f'{p:.3f}' for p in probs)}]")
    results["clean_actions"] = {str(k): v for k, v in clean_actions.items()}

    # === Test 2: Action shift under corruption ===
    print("\n=== Action Shift Under Corruption ===")
    shift_results = {}
    for c in corruptions:
        per_scene = {}
        all_shifts = []
        for s_idx, s in enumerate(scenes):
            corr_img = apply_corruption(s, c)
            tokens, scores = get_action_logits(model, processor, corr_img, prompt)
            values = [token_to_action_value(t) for t in tokens]
            clean_vals = clean_actions[s_idx]
            shifts = [v - cv for v, cv in zip(values, clean_vals)]
            all_shifts.append(shifts)
            per_scene[str(s_idx)] = {
                "tokens": tokens,
                "values": values,
                "shifts": shifts,
                "l2_shift": float(np.linalg.norm(shifts)),
                "max_shift": float(max(abs(s) for s in shifts)),
            }
            # Top-1 probs
            probs = [torch.softmax(sc[0], dim=0).max().item() for sc in scores]
            per_scene[str(s_idx)]["top1_probs"] = probs

        all_shifts_arr = np.array(all_shifts)
        shift_results[c] = {
            "per_scene": per_scene,
            "mean_l2_shift": float(np.mean([v["l2_shift"] for v in per_scene.values()])),
            "mean_per_dim_shift": [float(np.mean(np.abs(all_shifts_arr[:, d]))) for d in range(7)],
            "std_per_dim_shift": [float(np.std(all_shifts_arr[:, d])) for d in range(7)],
            "tokens_changed_pct": float(np.mean([sum(1 for t, ct in zip(per_scene[str(i)]["tokens"], clean_tokens_all[i]) if t != ct) / 7 for i in range(len(scenes))]) * 100),
        }
        print(f"  {c}: mean_l2_shift={shift_results[c]['mean_l2_shift']:.4f}, tokens_changed={shift_results[c]['tokens_changed_pct']:.0f}%")
    results["action_shifts"] = shift_results

    # === Test 3: Confidence vs severity ===
    print("\n=== Confidence vs Severity ===")
    conf_results = {}
    for c in corruptions:
        per_sev = {}
        for sev in [0.1, 0.25, 0.5, 0.75, 1.0]:
            all_probs = []
            for s in scenes[:4]:
                corr_img = apply_corruption(s, c, severity=sev)
                _, scores = get_action_logits(model, processor, corr_img, prompt)
                probs = [torch.softmax(sc[0], dim=0).max().item() for sc in scores]
                all_probs.extend(probs)
            per_sev[str(sev)] = {
                "mean_top1_prob": float(np.mean(all_probs)),
                "min_top1_prob": float(np.min(all_probs)),
            }
        conf_results[c] = per_sev
        print(f"  {c}: sev=0.1 conf={per_sev['0.1']['mean_top1_prob']:.3f}, sev=1.0 conf={per_sev['1.0']['mean_top1_prob']:.3f}")
    results["confidence_vs_severity"] = conf_results

    # === Test 4: Action entropy analysis ===
    print("\n=== Action Entropy Analysis ===")
    entropy_results = {}
    bin_range = range(31744, 32000)  # 256 action bins

    for condition in ['clean', 'fog', 'night']:
        entropies = []
        for s_idx, s in enumerate(scenes[:4]):
            if condition == 'clean':
                img = s
            else:
                img = apply_corruption(s, condition)
            _, scores = get_action_logits(model, processor, img, prompt)
            for sc in scores:
                probs = torch.softmax(sc[0], dim=0)
                action_probs = probs[31744:32000].numpy()
                action_probs = action_probs / (action_probs.sum() + 1e-12)
                ent = -float(np.sum(action_probs * np.log(action_probs + 1e-12)))
                entropies.append(ent)

        entropy_results[condition] = {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
        }
        print(f"  {condition}: mean_entropy={np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
    results["action_entropy"] = entropy_results

    # === Test 5: Cross-scene action agreement ===
    print("\n=== Cross-Scene Action Agreement ===")
    agreement = {}
    for condition in ['clean'] + corruptions:
        tokens_per_scene = []
        for s_idx, s in enumerate(scenes):
            if condition == 'clean':
                img = s
            else:
                img = apply_corruption(s, condition)
            tokens, _ = get_action_logits(model, processor, img, prompt)
            tokens_per_scene.append(tokens)

        # For each action dimension, count unique tokens across scenes
        unique_per_dim = []
        for d in range(7):
            dim_tokens = [t[d] for t in tokens_per_scene]
            unique_per_dim.append(len(set(dim_tokens)))

        agreement[condition] = {
            "unique_tokens_per_dim": unique_per_dim,
            "mean_unique": float(np.mean(unique_per_dim)),
            "max_unique": int(np.max(unique_per_dim)),
        }
        print(f"  {condition}: unique_per_dim={unique_per_dim}, mean={np.mean(unique_per_dim):.1f}")
    results["action_agreement"] = agreement

    out_path = "/workspace/Vizuara-VLA-Research/experiments/action_impact_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
