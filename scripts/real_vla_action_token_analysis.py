#!/usr/bin/env python3
"""Experiment 431: Action Token Semantic Analysis

Analyzes how the actual decoded action tokens change under corruption.
Goes beyond hidden states to examine the model's final output — the
action commands it would send to the robot.

Tests:
1. Clean action distribution across scenes
2. Corrupted action deviation from clean
3. Action dimension sensitivity (which of 7 dims change most)
4. Action collapse under severe corruption
5. Token probability entropy (confidence degradation)
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
    """Generate action tokens and return token IDs + logits entropy."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    # Extract generated token IDs (action tokens)
    gen_ids = out.sequences[0, inputs['input_ids'].shape[1]:]
    token_ids = gen_ids.cpu().tolist()

    # Compute entropy of logits for each generated token
    entropies = []
    for score in out.scores:
        probs = torch.softmax(score[0].float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        entropies.append(entropy)

    return token_ids, entropies

def token_to_action(token_ids, n_bins=256, token_offset=31744):
    """Convert action token IDs to continuous action values."""
    actions = []
    for tid in token_ids:
        bin_idx = tid - token_offset
        if 0 <= bin_idx < n_bins:
            action_val = (bin_idx / (n_bins - 1)) * 2 - 1  # Map to [-1, 1]
            actions.append(action_val)
        else:
            actions.append(float('nan'))
    return actions

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']

    seeds = [42, 123, 456, 789, 999, 1234, 5678, 9999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    results = {"n_scenes": len(scenes)}

    # === Test 1: Clean action distribution ===
    print("\n=== Clean Action Distribution ===")
    clean_tokens = []
    clean_actions = []
    clean_entropies = []
    for i, s in enumerate(scenes):
        tids, ents = get_action_tokens(model, processor, s, prompt)
        actions = token_to_action(tids)
        clean_tokens.append(tids)
        clean_actions.append(actions)
        clean_entropies.append(ents)
        print(f"  Scene {i}: tokens={tids}, actions=[{', '.join(f'{a:.3f}' for a in actions)}]")

    clean_actions = np.array(clean_actions)
    clean_entropies = np.array(clean_entropies)
    results["clean_actions"] = {
        "mean_per_dim": clean_actions.mean(axis=0).tolist(),
        "std_per_dim": clean_actions.std(axis=0).tolist(),
        "mean_entropy": float(clean_entropies.mean()),
        "std_entropy": float(clean_entropies.std()),
        "entropy_per_dim": clean_entropies.mean(axis=0).tolist(),
        "unique_token_sets": len(set(tuple(t) for t in clean_tokens)),
        "all_tokens": clean_tokens,
    }
    print(f"  Unique action sets: {results['clean_actions']['unique_token_sets']}/{len(scenes)}")
    print(f"  Mean entropy: {clean_entropies.mean():.4f}")

    # === Test 2: Corrupted action deviation ===
    print("\n=== Corrupted Action Deviation ===")
    corruption_results = {}
    for c in corruptions:
        corr_tokens = []
        corr_actions = []
        corr_entropies = []
        for s in scenes:
            tids, ents = get_action_tokens(model, processor, apply_corruption(s, c), prompt)
            actions = token_to_action(tids)
            corr_tokens.append(tids)
            corr_actions.append(actions)
            corr_entropies.append(ents)

        corr_actions_arr = np.array(corr_actions)
        corr_entropies_arr = np.array(corr_entropies)

        # How many scenes have identical actions under corruption?
        identical_count = sum(1 for i in range(len(scenes))
                           if clean_tokens[i] == corr_tokens[i])

        # Mean L1 deviation per action dimension
        l1_per_dim = np.mean(np.abs(corr_actions_arr - clean_actions), axis=0).tolist()

        # Overall deviation
        mean_l1 = float(np.mean(np.abs(corr_actions_arr - clean_actions)))

        # Action collapse: are all corrupted scenes producing same action?
        unique_corrupt = len(set(tuple(t) for t in corr_tokens))

        corruption_results[c] = {
            "identical_to_clean": identical_count,
            "pct_identical": float(identical_count / len(scenes) * 100),
            "mean_l1_deviation": mean_l1,
            "l1_per_dim": l1_per_dim,
            "unique_actions": unique_corrupt,
            "action_collapse": unique_corrupt == 1,
            "mean_entropy": float(corr_entropies_arr.mean()),
            "entropy_change": float(corr_entropies_arr.mean() - clean_entropies.mean()),
            "all_tokens": corr_tokens,
        }
        print(f"  {c}: {identical_count}/{len(scenes)} identical, L1={mean_l1:.4f}, "
              f"unique={unique_corrupt}, entropy_Δ={corr_entropies_arr.mean() - clean_entropies.mean():.4f}")
    results["corruption_deviation"] = corruption_results

    # === Test 3: Per-dimension sensitivity ===
    print("\n=== Per-Dimension Action Sensitivity ===")
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    dim_sensitivity = {}
    for d in range(7):
        per_corr = {}
        for c in corruptions:
            corr_actions_arr = np.array([token_to_action(t) for t in corruption_results[c]["all_tokens"]])
            per_corr[c] = float(np.mean(np.abs(corr_actions_arr[:, d] - clean_actions[:, d])))
        dim_sensitivity[dim_names[d]] = per_corr
        most_affected = max(per_corr, key=per_corr.get)
        print(f"  {dim_names[d]}: most affected by {most_affected} (L1={per_corr[most_affected]:.4f})")
    results["dim_sensitivity"] = dim_sensitivity

    # === Test 4: Severity-dependent action change ===
    print("\n=== Severity-Dependent Action Change ===")
    severity_actions = {}
    for c in ['fog', 'night']:
        sev_results = {}
        for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
            sev_tokens = []
            sev_actions = []
            for s in scenes:
                tids, _ = get_action_tokens(model, processor, apply_corruption(s, c, sev), prompt)
                sev_tokens.append(tids)
                sev_actions.append(token_to_action(tids))

            sev_actions_arr = np.array(sev_actions)
            mean_l1 = float(np.mean(np.abs(sev_actions_arr - clean_actions)))
            unique = len(set(tuple(t) for t in sev_tokens))
            sev_results[str(sev)] = {
                "mean_l1": mean_l1,
                "unique_actions": unique,
                "collapse": unique == 1,
            }
        severity_actions[c] = sev_results
        print(f"  {c}: sev=0.1 L1={sev_results['0.1']['mean_l1']:.4f}, "
              f"sev=1.0 L1={sev_results['1.0']['mean_l1']:.4f}")
    results["severity_actions"] = severity_actions

    # === Test 5: Token probability analysis ===
    print("\n=== Token Probability Concentration ===")
    # For a subset, look at top-k probability mass
    prob_results = {}
    for condition in ['clean', 'fog', 'night']:
        top1_probs = []
        top5_probs = []
        for i, s in enumerate(scenes[:3]):  # Just 3 scenes to save time
            if condition == 'clean':
                img = s
            else:
                img = apply_corruption(s, condition)
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            for score in out.scores:
                probs = torch.softmax(score[0].float(), dim=-1)
                sorted_probs = torch.sort(probs, descending=True).values
                top1_probs.append(float(sorted_probs[0]))
                top5_probs.append(float(sorted_probs[:5].sum()))

        prob_results[condition] = {
            "mean_top1_prob": float(np.mean(top1_probs)),
            "mean_top5_prob": float(np.mean(top5_probs)),
            "min_top1_prob": float(np.min(top1_probs)),
        }
        print(f"  {condition}: top1={np.mean(top1_probs):.4f}, top5={np.mean(top5_probs):.4f}")
    results["token_probabilities"] = prob_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/action_token_analysis_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
