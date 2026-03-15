#!/usr/bin/env python3
"""Experiment 405: Token-Level Action Analysis

Examines how corruption affects individual action token predictions
across the 7 action dimensions (x, y, z, roll, pitch, yaw, gripper).

Tests:
1. Per-dimension action token changes under corruption
2. Which dimensions are most/least sensitive to corruption
3. Action token distribution shape (clean vs corrupt)
4. Gripper dimension behavior (binary vs continuous)
5. Correlation between action change magnitude and detection distance
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

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

ACTION_TOKEN_START = 31744
ACTION_TOKEN_END = 31999
N_BINS = 256
ACTION_DIMS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']

    scenes = []
    for seed in [42, 123, 456, 789, 999]:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    def generate_actions(image, n_tokens=7):
        """Generate action tokens autoregressively."""
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
            )
        # Extract generated tokens (excluding input)
        input_len = inputs['input_ids'].shape[1]
        action_tokens = generated[0, input_len:].cpu().tolist()
        return action_tokens

    def get_action_logits(image):
        """Get logits for the first action token (one-step)."""
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        logits = out.logits[0, -1, :].float().cpu().numpy()
        hidden = out.hidden_states[3][0, -1, :].float().cpu().numpy()
        # Action token probabilities
        action_logits = logits[ACTION_TOKEN_START:ACTION_TOKEN_END+1]
        action_probs = np.exp(action_logits - action_logits.max())
        action_probs = action_probs / action_probs.sum()
        return action_probs, hidden

    results = {}

    # === Test 1: Full action generation ===
    print("=== Full Action Generation ===")
    gen_results = {}
    for condition in ['clean'] + corruptions:
        print(f"\n  {condition}:")
        condition_data = []
        for si, scene in enumerate(scenes):
            if condition == 'clean':
                img = scene
            else:
                img = apply_corruption(scene, condition, 1.0)
            tokens = generate_actions(img)
            # Convert to bin indices
            bin_indices = [t - ACTION_TOKEN_START if ACTION_TOKEN_START <= t <= ACTION_TOKEN_END
                          else -1 for t in tokens]
            condition_data.append({
                "tokens": tokens,
                "bins": bin_indices
            })
            bins_str = ", ".join(str(b) for b in bin_indices[:7])
            print(f"    scene {si}: bins=[{bins_str}]")

        gen_results[condition] = condition_data

    results["generated_actions"] = gen_results

    # === Test 2: Per-dimension analysis ===
    print("\n=== Per-Dimension Token Analysis ===")
    dim_analysis = {}
    for di, dim_name in enumerate(ACTION_DIMS):
        clean_bins = [gen_results['clean'][si]['bins'][di]
                      for si in range(len(scenes))
                      if di < len(gen_results['clean'][si]['bins'])]

        dim_data = {"clean_bins": clean_bins}
        for c in corruptions:
            corrupt_bins = [gen_results[c][si]['bins'][di]
                           for si in range(len(scenes))
                           if di < len(gen_results[c][si]['bins'])]

            if clean_bins and corrupt_bins:
                # Mean absolute bin change
                changes = [abs(cl - co) for cl, co in zip(clean_bins, corrupt_bins)]
                mean_change = np.mean(changes)
                max_change = max(changes)
                n_changed = sum(1 for ch in changes if ch > 0)
            else:
                mean_change = 0
                max_change = 0
                n_changed = 0

            dim_data[c] = {
                "corrupt_bins": corrupt_bins,
                "mean_bin_change": float(mean_change),
                "max_bin_change": int(max_change),
                "n_changed": n_changed,
                "pct_changed": float(n_changed / max(len(clean_bins), 1) * 100)
            }

        dim_analysis[dim_name] = dim_data
        print(f"  {dim_name}:")
        for c in corruptions:
            d = dim_data[c]
            print(f"    {c}: mean_change={d['mean_bin_change']:.1f}, "
                  f"max={d['max_bin_change']}, changed={d['n_changed']}/5")

    results["dim_analysis"] = dim_analysis

    # === Test 3: First-token probability analysis ===
    print("\n=== First Token Probability Analysis ===")
    prob_results = {}

    # Clean centroid for distance
    clean_hiddens = []
    for scene in scenes:
        _, h = get_action_logits(scene)
        clean_hiddens.append(h)
    centroid = np.mean(clean_hiddens, axis=0)

    for condition in ['clean'] + corruptions:
        print(f"\n  {condition}:")
        all_probs = []
        all_dists = []
        for si, scene in enumerate(scenes):
            if condition == 'clean':
                img = scene
            else:
                img = apply_corruption(scene, condition, 1.0)

            probs, hidden = get_action_logits(img)
            d = cosine_dist(hidden, centroid)
            all_probs.append(probs)
            all_dists.append(d)

            top_bin = np.argmax(probs)
            top_prob = probs[top_bin]
            # Entropy of action distribution
            ent = -np.sum(probs * np.log(probs + 1e-10))
            print(f"    scene {si}: top_bin={top_bin}, prob={top_prob:.4f}, ent={ent:.4f}")

        mean_probs = np.mean(all_probs, axis=0)
        prob_results[condition] = {
            "mean_top_bin": int(np.argmax(mean_probs)),
            "mean_confidence": float(np.max(mean_probs)),
            "mean_entropy": float(-np.sum(mean_probs * np.log(mean_probs + 1e-10))),
            "mean_dist": float(np.mean(all_dists))
        }

    results["probability_analysis"] = prob_results

    # === Test 4: Severity-action curve ===
    print("\n=== Severity-Action Curve ===")
    sev_action = {}
    ref_scene = scenes[0]
    clean_tokens = generate_actions(ref_scene)
    clean_bins = [t - ACTION_TOKEN_START if ACTION_TOKEN_START <= t <= ACTION_TOKEN_END else -1
                  for t in clean_tokens]

    for c in corruptions:
        sev_data = {}
        for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
            corrupted = apply_corruption(ref_scene, c, sev)
            tokens = generate_actions(corrupted)
            bins = [t - ACTION_TOKEN_START if ACTION_TOKEN_START <= t <= ACTION_TOKEN_END else -1
                    for t in tokens]

            total_change = sum(abs(cl - co) for cl, co in zip(clean_bins, bins) if cl >= 0 and co >= 0)
            n_dims_changed = sum(1 for cl, co in zip(clean_bins, bins) if cl != co and cl >= 0 and co >= 0)

            sev_data[str(sev)] = {
                "bins": bins,
                "total_change": int(total_change),
                "n_dims_changed": n_dims_changed
            }

        sev_action[c] = sev_data
        print(f"  {c}:", end="")
        for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
            d = sev_data[str(sev)]
            print(f" sev={sev}:Δ{d['total_change']}({d['n_dims_changed']}d)", end="")
        print()

    results["severity_action"] = sev_action

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/token_action_analysis_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
