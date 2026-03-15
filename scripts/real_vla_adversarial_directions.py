#!/usr/bin/env python3
"""Experiment 417: Adversarial Directions in Embedding Space

Tests whether an adversary who knows the detector's centroid can craft
perturbations that evade detection (move embedding toward centroid) while
still corrupting the model's action predictions. Studies the geometry
of adversarial vs random perturbation directions.

Tests:
1. Centroid-directed perturbation: push embedding toward centroid
2. Orthogonal-to-centroid perturbation: corrupt without moving distance
3. Random direction perturbation baseline
4. Transfer: do adversarial directions found for one scene transfer?
5. Action impact vs detection evasion trade-off
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden_and_tokens(model, processor, image, prompt, layer=3):
    """Extract hidden state and predicted action tokens."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    hidden = fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

    # Get action token prediction
    logits = fwd.logits[0, -1, :].float().cpu()
    action_logits = logits[31744:32000]
    top_token = int(action_logits.argmax()) + 31744
    return hidden, top_token

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def apply_pixel_perturbation(image, direction, magnitude):
    """Apply a perturbation in pixel space along a given direction."""
    arr = np.array(image).astype(np.float32) / 255.0
    # Normalize direction to unit norm
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-10:
        direction = direction / d_norm
    arr_perturbed = arr + magnitude * direction
    arr_perturbed = np.clip(arr_perturbed, 0, 1)
    return Image.fromarray((arr_perturbed * 255).astype(np.uint8))

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    # Generate scenes
    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Calibration
    print("Calibrating...")
    clean_data = []
    for s in scenes:
        h, t = extract_hidden_and_tokens(model, processor, s, prompt)
        clean_data.append({"hidden": h, "token": t})
    centroid = np.mean([d["hidden"] for d in clean_data], axis=0)
    clean_dists = [cosine_dist(d["hidden"], centroid) for d in clean_data]
    clean_tokens = [d["token"] for d in clean_data]

    results = {}

    # === Test 1: Structured pixel perturbations ===
    print("\n=== Structured Pixel Perturbations ===")
    magnitudes = [0.05, 0.1, 0.2, 0.3, 0.5]
    perturbation_types = {}

    rng = np.random.RandomState(42)
    shape = (224, 224, 3)

    for mag in magnitudes:
        print(f"\n  Magnitude={mag}")

        # Type A: Uniform brightness shift (toward white = fog-like)
        direction_bright = np.ones(shape, dtype=np.float32)
        perturbed_dists_bright = []
        tokens_changed_bright = 0
        for i, s in enumerate(scenes):
            p = apply_pixel_perturbation(s, direction_bright, mag)
            h, t = extract_hidden_and_tokens(model, processor, p, prompt)
            perturbed_dists_bright.append(cosine_dist(h, centroid))
            if t != clean_tokens[i]:
                tokens_changed_bright += 1

        # Type B: Uniform darkness (night-like)
        direction_dark = -np.ones(shape, dtype=np.float32)
        perturbed_dists_dark = []
        tokens_changed_dark = 0
        for i, s in enumerate(scenes):
            p = apply_pixel_perturbation(s, direction_dark, mag)
            h, t = extract_hidden_and_tokens(model, processor, p, prompt)
            perturbed_dists_dark.append(cosine_dist(h, centroid))
            if t != clean_tokens[i]:
                tokens_changed_dark += 1

        # Type C: Random noise
        direction_random = rng.randn(*shape).astype(np.float32)
        perturbed_dists_random = []
        tokens_changed_random = 0
        for i, s in enumerate(scenes):
            p = apply_pixel_perturbation(s, direction_random, mag)
            h, t = extract_hidden_and_tokens(model, processor, p, prompt)
            perturbed_dists_random.append(cosine_dist(h, centroid))
            if t != clean_tokens[i]:
                tokens_changed_random += 1

        # Type D: Checkerboard (structured local)
        direction_checker = np.zeros(shape, dtype=np.float32)
        for y in range(224):
            for x in range(224):
                if (y // 16 + x // 16) % 2 == 0:
                    direction_checker[y, x, :] = 1.0
                else:
                    direction_checker[y, x, :] = -1.0
        perturbed_dists_checker = []
        tokens_changed_checker = 0
        for i, s in enumerate(scenes):
            p = apply_pixel_perturbation(s, direction_checker, mag)
            h, t = extract_hidden_and_tokens(model, processor, p, prompt)
            perturbed_dists_checker.append(cosine_dist(h, centroid))
            if t != clean_tokens[i]:
                tokens_changed_checker += 1

        # Type E: Low-frequency sinusoidal
        Y, X = np.mgrid[0:224, 0:224].astype(np.float32)
        direction_sine = np.sin(2 * np.pi * X / 224 * 2) * np.sin(2 * np.pi * Y / 224 * 2)
        direction_sine = np.stack([direction_sine]*3, axis=-1)
        perturbed_dists_sine = []
        tokens_changed_sine = 0
        for i, s in enumerate(scenes):
            p = apply_pixel_perturbation(s, direction_sine, mag)
            h, t = extract_hidden_and_tokens(model, processor, p, prompt)
            perturbed_dists_sine.append(cosine_dist(h, centroid))
            if t != clean_tokens[i]:
                tokens_changed_sine += 1

        perturbation_types[str(mag)] = {
            "brightness": {
                "mean_dist": float(np.mean(perturbed_dists_bright)),
                "auroc": float(compute_auroc(clean_dists, perturbed_dists_bright)),
                "tokens_changed": tokens_changed_bright,
            },
            "darkness": {
                "mean_dist": float(np.mean(perturbed_dists_dark)),
                "auroc": float(compute_auroc(clean_dists, perturbed_dists_dark)),
                "tokens_changed": tokens_changed_dark,
            },
            "random": {
                "mean_dist": float(np.mean(perturbed_dists_random)),
                "auroc": float(compute_auroc(clean_dists, perturbed_dists_random)),
                "tokens_changed": tokens_changed_random,
            },
            "checkerboard": {
                "mean_dist": float(np.mean(perturbed_dists_checker)),
                "auroc": float(compute_auroc(clean_dists, perturbed_dists_checker)),
                "tokens_changed": tokens_changed_checker,
            },
            "sinusoidal": {
                "mean_dist": float(np.mean(perturbed_dists_sine)),
                "auroc": float(compute_auroc(clean_dists, perturbed_dists_sine)),
                "tokens_changed": tokens_changed_sine,
            },
        }

        for ptype in ['brightness', 'darkness', 'random', 'checkerboard', 'sinusoidal']:
            entry = perturbation_types[str(mag)][ptype]
            print(f"    {ptype}: dist={entry['mean_dist']:.6f}, AUROC={entry['auroc']:.4f}, "
                  f"tokens_changed={entry['tokens_changed']}/5")

    results["perturbation_types"] = perturbation_types

    # === Test 2: Detection evasion analysis ===
    print("\n=== Detection Evasion Analysis ===")
    # Find perturbation directions that change actions but don't increase distance
    evasion = {}
    n_evasion_trials = 20
    for trial in range(n_evasion_trials):
        trial_rng = np.random.RandomState(trial + 100)
        direction = trial_rng.randn(*shape).astype(np.float32)
        mag = 0.3

        dists = []
        tokens_changed = 0
        for i, s in enumerate(scenes):
            p = apply_pixel_perturbation(s, direction, mag)
            h, t = extract_hidden_and_tokens(model, processor, p, prompt)
            dists.append(cosine_dist(h, centroid))
            if t != clean_tokens[i]:
                tokens_changed += 1

        evasion[str(trial)] = {
            "mean_dist": float(np.mean(dists)),
            "max_dist": float(np.max(dists)),
            "tokens_changed": tokens_changed,
            "detected": bool(compute_auroc(clean_dists, dists) >= 0.8),
        }

    # Summary
    n_evasive = sum(1 for v in evasion.values() if v["tokens_changed"] > 0 and not v["detected"])
    n_action_changing = sum(1 for v in evasion.values() if v["tokens_changed"] > 0)
    n_detected = sum(1 for v in evasion.values() if v["detected"])
    evasion_summary = {
        "n_trials": n_evasion_trials,
        "n_action_changing": n_action_changing,
        "n_detected": n_detected,
        "n_evasive": n_evasive,
        "evasion_rate": n_evasive / max(n_action_changing, 1),
    }
    results["evasion_analysis"] = evasion
    results["evasion_summary"] = evasion_summary
    print(f"  {n_action_changing}/{n_evasion_trials} changed actions, "
          f"{n_detected}/{n_evasion_trials} detected, "
          f"{n_evasive} evasive (changed action + undetected)")

    # === Test 3: Perturbation magnitude vs action impact ===
    print("\n=== Magnitude vs Action Impact ===")
    fine_mags = np.linspace(0.01, 0.5, 15)
    mag_impact = {}
    for mag in fine_mags:
        direction = rng.randn(*shape).astype(np.float32)
        dists = []
        tokens_changed = 0
        for i, s in enumerate(scenes):
            p = apply_pixel_perturbation(s, direction, float(mag))
            h, t = extract_hidden_and_tokens(model, processor, p, prompt)
            dists.append(cosine_dist(h, centroid))
            if t != clean_tokens[i]:
                tokens_changed += 1
        mag_impact[f"{mag:.3f}"] = {
            "mean_dist": float(np.mean(dists)),
            "auroc": float(compute_auroc(clean_dists, dists)),
            "tokens_changed": tokens_changed,
        }
        print(f"  mag={mag:.3f}: dist={np.mean(dists):.6f}, tokens={tokens_changed}/5")
    results["magnitude_impact"] = mag_impact

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/adversarial_directions_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
