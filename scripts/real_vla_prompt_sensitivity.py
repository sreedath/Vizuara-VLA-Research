#!/usr/bin/env python3
"""Experiment 400: Prompt Sensitivity Analysis

How do different prompts affect OOD detection performance?
Tests whether the detection system depends on the specific prompt used.

Tests:
1. Multiple prompt formulations for the same task
2. Completely different task descriptions
3. Minimal vs verbose prompts
4. Prompt with/without instruction formatting
5. Cross-prompt detection transfer
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

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    # Different prompt formulations
    prompts = {
        "standard": "In: What action should the robot take to pick up the object?\nOut:",
        "short": "In: Pick up the object.\nOut:",
        "verbose": "In: Given the current visual observation, what is the optimal action the robot should take to successfully grasp and pick up the target object?\nOut:",
        "navigate": "In: What action should the robot take to navigate to the goal?\nOut:",
        "place": "In: What action should the robot take to place the object on the table?\nOut:",
        "minimal": "In: Action?\nOut:",
        "no_format": "Pick up the object",
        "question_only": "What should the robot do?",
    }

    corruptions = ['fog', 'night', 'noise', 'blur']

    # Use multiple scenes
    scenes = []
    for seed in [42, 123, 456]:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    results = {}

    # === Test 1: Per-prompt detection performance ===
    print("=== Per-Prompt Detection ===")
    for pname, prompt in prompts.items():
        print(f"\n  Prompt: '{pname}'")

        # Clean embeddings
        clean_embs = []
        for scene in scenes:
            emb = extract_hidden(model, processor, scene, prompt)
            clean_embs.append(emb)
        centroid = np.mean(clean_embs, axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

        prompt_results = {"clean_dists": [float(d) for d in clean_dists]}

        for c in corruptions:
            corrupt_dists = []
            for scene in scenes:
                corrupted = apply_corruption(scene, c, 1.0)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = cosine_dist(emb, centroid)
                corrupt_dists.append(d)

            auroc = compute_auroc(clean_dists, corrupt_dists)
            prompt_results[c] = {
                "corrupt_dists": [float(d) for d in corrupt_dists],
                "mean_dist": float(np.mean(corrupt_dists)),
                "auroc": float(auroc)
            }
            print(f"    {c}: mean={np.mean(corrupt_dists):.6f}, auroc={auroc:.3f}")

        results[pname] = prompt_results

    # === Test 2: Cross-prompt embedding similarity ===
    print("\n=== Cross-Prompt Embedding Similarity ===")
    ref_scene = scenes[0]
    prompt_embs = {}
    for pname, prompt in prompts.items():
        prompt_embs[pname] = extract_hidden(model, processor, ref_scene, prompt)

    cross_prompt = {}
    for p1 in prompts:
        for p2 in prompts:
            if p1 >= p2:
                continue
            d = cosine_dist(prompt_embs[p1], prompt_embs[p2])
            key = f"{p1}_vs_{p2}"
            cross_prompt[key] = float(d)

    # How much do prompts change embeddings vs corruptions?
    std_emb = prompt_embs["standard"]
    prompt_dists = {p: float(cosine_dist(prompt_embs[p], std_emb)) for p in prompts if p != "standard"}

    fog_emb = extract_hidden(model, processor,
                             apply_corruption(ref_scene, 'fog', 1.0),
                             prompts["standard"])
    fog_dist = float(cosine_dist(fog_emb, std_emb))

    results["cross_prompt"] = {
        "pairwise": cross_prompt,
        "vs_standard": prompt_dists,
        "fog_dist_for_comparison": fog_dist
    }
    print(f"  Prompt variation range: {min(prompt_dists.values()):.6f} - {max(prompt_dists.values()):.6f}")
    print(f"  Fog corruption dist: {fog_dist:.6f}")

    # === Test 3: Cross-prompt detection transfer ===
    print("\n=== Cross-Prompt Detection Transfer ===")
    # Calibrate on one prompt, test on another
    transfer = {}
    cal_prompt = "standard"
    cal_clean_embs = []
    for scene in scenes:
        emb = extract_hidden(model, processor, scene, prompts[cal_prompt])
        cal_clean_embs.append(emb)
    cal_centroid = np.mean(cal_clean_embs, axis=0)
    cal_clean_dists = [cosine_dist(e, cal_centroid) for e in cal_clean_embs]

    for test_prompt_name, test_prompt in prompts.items():
        if test_prompt_name == cal_prompt:
            continue

        # Test with different prompt
        test_clean_dists = []
        for scene in scenes:
            emb = extract_hidden(model, processor, scene, test_prompt)
            d = cosine_dist(emb, cal_centroid)
            test_clean_dists.append(d)

        test_corrupt_dists = []
        for scene in scenes:
            corrupted = apply_corruption(scene, 'fog', 1.0)
            emb = extract_hidden(model, processor, corrupted, test_prompt)
            d = cosine_dist(emb, cal_centroid)
            test_corrupt_dists.append(d)

        auroc = compute_auroc(test_clean_dists, test_corrupt_dists)
        # False positive rate: clean samples exceeding calibration threshold
        cal_thresh = max(cal_clean_dists) * 1.5
        fpr = sum(1 for d in test_clean_dists if d > cal_thresh) / len(test_clean_dists)

        transfer[test_prompt_name] = {
            "auroc": float(auroc),
            "fpr": float(fpr),
            "mean_clean_dist": float(np.mean(test_clean_dists)),
            "mean_corrupt_dist": float(np.mean(test_corrupt_dists))
        }
        print(f"  {cal_prompt}->{test_prompt_name}: auroc={auroc:.3f}, fpr={fpr:.1%}")

    results["cross_prompt_transfer"] = transfer

    # === Test 4: Prompt length vs detection ===
    print("\n=== Prompt Length Analysis ===")
    length_analysis = {}
    for pname, prompt in prompts.items():
        n_chars = len(prompt)
        # Get mean AUROC across corruptions
        aurocs = [results[pname][c]["auroc"] for c in corruptions]
        mean_auroc = np.mean(aurocs)
        length_analysis[pname] = {
            "n_chars": n_chars,
            "mean_auroc": float(mean_auroc)
        }

    results["length_analysis"] = length_analysis

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/prompt_sensitivity_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
