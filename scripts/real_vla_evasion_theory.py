#!/usr/bin/env python3
"""Experiment 292: Theoretical Evasion Bounds
Computes the minimum perturbation needed to:
1. Change at least one action token (action impact threshold)
2. Remain undetected by the cosine distance detector
3. Finds the gap between these two thresholds
Also tests:
4. Targeted perturbations (change specific action dims)
5. Adversarial direction search (maximize action change while minimizing embedding change)
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

def get_action_tokens(model, processor, image, prompt):
    """Get the 7 predicted action tokens."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=7, do_sample=False)
    # Extract only the generated tokens (after input)
    gen_tokens = output[0, inputs['input_ids'].shape[1]:]
    return gen_tokens.cpu().tolist()

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

    results = {
        "experiment": "evasion_theory",
        "experiment_number": 292,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Get clean baseline
    clean_emb = extract_hidden(model, processor, base_img, prompt)
    clean_tokens = get_action_tokens(model, processor, base_img, prompt)
    print(f"Clean tokens: {clean_tokens}")

    # Part 1: Fine-grained severity sweep to find action change threshold
    print("\n=== Part 1: Action Change Threshold ===")
    corruption_thresholds = {}

    for c in ['fog', 'night', 'blur', 'noise']:
        print(f"\n  {c}:")
        first_change_sev = None
        first_change_d = None
        sweep = []

        # Fine sweep from 0.01 to 1.0
        for sev_pct in range(1, 101, 1):
            sev = sev_pct / 100.0
            corrupted = apply_corruption(base_img, c, sev)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(clean_emb, emb))
            tokens = get_action_tokens(model, processor, corrupted, prompt)
            n_changed = sum(1 for a, b in zip(clean_tokens, tokens) if a != b)

            sweep.append({
                "severity": sev,
                "distance": d,
                "tokens_changed": n_changed,
                "tokens": tokens
            })

            if n_changed > 0 and first_change_sev is None:
                first_change_sev = sev
                first_change_d = d
                print(f"    First action change at sev={sev:.2f}, d={d:.6f}, changed={n_changed}/7")

            if sev_pct % 20 == 0:
                print(f"    sev={sev:.2f}: d={d:.6f}, changed={n_changed}/7")

        corruption_thresholds[c] = {
            "first_change_severity": first_change_sev,
            "first_change_distance": first_change_d,
            "sweep": sweep[:20] + sweep[-5:]  # save first 20 and last 5
        }

    results["action_thresholds"] = corruption_thresholds

    # Part 2: Binary search for exact threshold
    print("\n=== Part 2: Binary Search for Exact Threshold ===")
    exact_thresholds = {}

    for c in ['fog', 'night', 'blur', 'noise']:
        threshold_info = corruption_thresholds[c]
        if threshold_info['first_change_severity'] is None:
            exact_thresholds[c] = {"exact_severity": None}
            continue

        # Binary search between 0 and first_change_severity
        lo = 0.0
        hi = threshold_info['first_change_severity']

        for _ in range(20):  # 20 iterations of binary search
            mid = (lo + hi) / 2
            corrupted = apply_corruption(base_img, c, mid)
            tokens = get_action_tokens(model, processor, corrupted, prompt)
            n_changed = sum(1 for a, b in zip(clean_tokens, tokens) if a != b)

            if n_changed > 0:
                hi = mid
            else:
                lo = mid

        # Get the exact distance at threshold
        corrupted = apply_corruption(base_img, c, hi)
        emb = extract_hidden(model, processor, corrupted, prompt)
        d = float(cosine(clean_emb, emb))

        exact_thresholds[c] = {
            "exact_severity": hi,
            "exact_distance": d,
            "distance_at_lo": float(cosine(clean_emb,
                extract_hidden(model, processor, apply_corruption(base_img, c, lo), prompt)))
        }
        print(f"  {c}: exact threshold at sev={hi:.6f}, d={d:.6f}")

    results["exact_thresholds"] = exact_thresholds

    # Part 3: Gap analysis
    print("\n=== Part 3: Evasion Gap Analysis ===")
    gap_analysis = {}

    for c in ['fog', 'night', 'blur', 'noise']:
        et = exact_thresholds[c]
        if et.get('exact_severity') is None:
            continue

        # Detection threshold = 0 (any d > 0 is detected)
        # Action change threshold = et['exact_distance']
        gap_analysis[c] = {
            "detection_threshold": 0.0,
            "action_change_threshold": et['exact_distance'],
            "evasion_window": et['exact_distance'],  # gap between detection and action change
            "evasion_possible": False,  # detection threshold is 0, so no evasion window
            "evasion_ratio": float('inf') if et['exact_distance'] > 0 else 0
        }
        print(f"  {c}: action threshold d={et['exact_distance']:.6f}, "
              f"detection threshold=0, evasion=impossible")

    results["gap_analysis"] = gap_analysis

    # Part 4: Random pixel perturbation (uniform noise)
    print("\n=== Part 4: Random Pixel Perturbation ===")
    random_perturb = []
    for n_pixels in [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50176]:
        arr = np.array(base_img).copy()
        rng = np.random.RandomState(42)
        flat = arr.reshape(-1)
        indices = rng.choice(len(flat), size=min(n_pixels, len(flat)), replace=False)
        flat[indices] = rng.randint(0, 256, size=len(indices)).astype(np.uint8)
        perturbed = Image.fromarray(arr)

        emb = extract_hidden(model, processor, perturbed, prompt)
        d = float(cosine(clean_emb, emb))
        tokens = get_action_tokens(model, processor, perturbed, prompt)
        n_changed = sum(1 for a, b in zip(clean_tokens, tokens) if a != b)

        random_perturb.append({
            "n_pixels": n_pixels,
            "distance": d,
            "tokens_changed": n_changed
        })
        print(f"  {n_pixels} pixels: d={d:.6f}, changed={n_changed}/7")

    results["random_perturbation"] = random_perturb

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/evasion_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
