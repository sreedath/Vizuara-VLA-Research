#!/usr/bin/env python3
"""Experiment 319: Multi-Step Action Trajectory Under Corruption
Measures how corruption affects action sequences over multiple generation steps:
1. Action token sequence (7+ steps) under each corruption
2. Token-by-token deviation from clean trajectory
3. Cumulative action error over steps
4. Action diversity (unique tokens per dimension)
5. Auto-regressive error propagation
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def get_extended_actions(model, processor, image, prompt, n_tokens=21):
    """Generate more action tokens than the standard 7."""
    ACTION_TOKEN_START = 31744
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    input_len = inputs['input_ids'].shape[1]
    gen_tokens = generated[0, input_len:].cpu().numpy()
    return [int(t) for t in gen_tokens]

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    ACTION_TOKEN_START = 31744

    results = {
        "experiment": "action_trajectory",
        "experiment_number": 319,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    # Part 1: Extended Action Generation (21 tokens = 3 action steps)
    print("=== Part 1: Extended Action Generation ===")
    clean_tokens = get_extended_actions(model, processor, base_img, prompt, 21)
    clean_actions = [t - ACTION_TOKEN_START for t in clean_tokens if ACTION_TOKEN_START <= t < ACTION_TOKEN_START + 256]
    print(f"  Clean tokens: {clean_tokens[:7]} (raw)")
    print(f"  Clean actions: {clean_actions[:7]} (action-space)")

    extended_results = {}
    for c in corruptions:
        for sev in [0.1, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            tokens = get_extended_actions(model, processor, corrupted, prompt, 21)
            actions = [t - ACTION_TOKEN_START for t in tokens if ACTION_TOKEN_START <= t < ACTION_TOKEN_START + 256]

            # Compare with clean
            n_common = min(len(clean_actions), len(actions))
            diffs = [abs(a - b) for a, b in zip(clean_actions[:n_common], actions[:n_common])]
            changed = [1 if a != b else 0 for a, b in zip(clean_actions[:n_common], actions[:n_common])]

            # Cumulative error
            cumulative = np.cumsum(diffs).tolist()

            # Per-step (groups of 7)
            step_errors = []
            for step in range(min(3, n_common // 7)):
                step_diffs = diffs[step*7:(step+1)*7]
                step_errors.append({
                    "step": step,
                    "total_deviation": sum(step_diffs),
                    "n_changed": sum(changed[step*7:(step+1)*7]),
                    "max_deviation": max(step_diffs) if step_diffs else 0,
                })

            key = f"{c}_{sev}"
            extended_results[key] = {
                "raw_tokens": tokens[:21],
                "actions": actions[:21],
                "n_action_tokens": len(actions),
                "per_token_diff": diffs,
                "cumulative_error": cumulative,
                "step_errors": step_errors,
                "total_deviation": sum(diffs),
                "n_changed": sum(changed),
            }
            print(f"  {key}: {len(actions)} action tokens, {sum(changed)}/{n_common} changed, "
                  f"total_dev={sum(diffs)}")

    results["extended_actions"] = extended_results

    # Part 2: Action Determinism Check
    print("\n=== Part 2: Action Determinism ===")
    determinism = []
    for i in range(5):
        tokens = get_extended_actions(model, processor, base_img, prompt, 21)
        determinism.append(tokens[:21])

    all_identical = all(d == determinism[0] for d in determinism)
    results["determinism"] = {
        "n_passes": 5,
        "all_identical": all_identical,
        "tokens": determinism,
    }
    print(f"  5 passes: all identical = {all_identical}")

    # Part 3: Per-Severity Action Divergence Curve
    print("\n=== Part 3: Severity → Action Divergence ===")
    divergence_curves = {}

    for c in corruptions:
        print(f"  {c}...")
        curve = []
        for sev in np.linspace(0.05, 1.0, 10):
            corrupted = apply_corruption(base_img, c, float(sev))
            tokens = get_extended_actions(model, processor, corrupted, prompt, 7)
            actions = [t - ACTION_TOKEN_START for t in tokens if ACTION_TOKEN_START <= t < ACTION_TOKEN_START + 256]
            n_common = min(len(clean_actions), len(actions))
            diffs = [abs(a - b) for a, b in zip(clean_actions[:n_common], actions[:n_common])]
            changed = sum(1 for d in diffs if d > 0)

            # Also get embedding distance
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(clean_emb, emb))

            curve.append({
                "severity": float(sev),
                "cosine_distance": d,
                "total_deviation": sum(diffs),
                "n_changed": changed,
                "n_total": n_common,
            })

        divergence_curves[c] = curve

    results["divergence_curves"] = divergence_curves

    # Part 4: Token Position Sensitivity (which of the 7 dims is most affected)
    print("\n=== Part 4: Per-Dimension Sensitivity ===")
    dim_sensitivity = {}

    for c in corruptions:
        dim_deviations = [[] for _ in range(7)]
        for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            tokens = get_extended_actions(model, processor, corrupted, prompt, 7)
            actions = [t - ACTION_TOKEN_START for t in tokens if ACTION_TOKEN_START <= t < ACTION_TOKEN_START + 256]
            for dim in range(min(7, len(actions), len(clean_actions))):
                dim_deviations[dim].append(abs(actions[dim] - clean_actions[dim]))

        dim_sensitivity[c] = {
            f"dim{d}": {
                "mean_deviation": float(np.mean(devs)),
                "max_deviation": int(max(devs)),
                "always_changed": all(d > 0 for d in devs),
            } for d, devs in enumerate(dim_deviations)
        }

    results["dim_sensitivity"] = dim_sensitivity

    # Summary
    print("\n=== Summary ===")
    for c in corruptions:
        ext = extended_results.get(f"{c}_1.0", {})
        print(f"  {c} (sev=1.0): {ext.get('n_changed', 0)}/{ext.get('n_action_tokens', 0)} tokens changed, "
              f"total_dev={ext.get('total_deviation', 0)}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/trajectory_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
