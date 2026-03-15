#!/usr/bin/env python3
"""Experiment 322: Robustness to Inference Variations
Tests detection under practical deployment variations:
1. Different prompts (10 diverse prompts)
2. Different image sizes (resized before processor)
3. Repeated inference after GPU warm-up variations
4. Sequential vs fresh model state
5. Combined prompt + scene variations
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "inference_robustness",
        "experiment_number": 322,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']

    # Part 1: 10 Different Prompts
    print("=== Part 1: Prompt Robustness (10 prompts) ===")
    prompts = [
        "In: What action should the robot take to pick up the object?\nOut:",
        "In: What action should the robot take to move forward?\nOut:",
        "In: What action should the robot take to grasp the cup?\nOut:",
        "In: What action should the robot take to push the block?\nOut:",
        "In: What action should the robot take to lift the bottle?\nOut:",
        "In: What action should the robot take to reach the target?\nOut:",
        "In: What action should the robot take to open the drawer?\nOut:",
        "In: What action should the robot take to close the gripper?\nOut:",
        "In: What action should the robot take to place the object?\nOut:",
        "In: What action should the robot take to stack the blocks?\nOut:",
    ]

    prompt_results = {}
    for i, p in enumerate(prompts):
        clean_emb = extract_hidden(model, processor, base_img, p)

        # Check clean stability
        clean_emb2 = extract_hidden(model, processor, base_img, p)
        clean_d = float(cosine(clean_emb, clean_emb2))

        # Check OOD
        ood_dists = []
        for c in corruptions:
            corrupted = apply_corruption(base_img, c, 0.5)
            emb = extract_hidden(model, processor, corrupted, p)
            d = float(cosine(clean_emb, emb))
            ood_dists.append(d)

        auroc = compute_auroc([clean_d], ood_dists)
        prompt_results[f"prompt_{i}"] = {
            "prompt": p[:50] + "...",
            "clean_distance": clean_d,
            "ood_distances": dict(zip(corruptions, ood_dists)),
            "min_ood": float(min(ood_dists)),
            "auroc": auroc,
            "perfect": clean_d == 0 and min(ood_dists) > 0,
        }
        print(f"  P{i}: clean_d={clean_d:.10f}, min_ood={min(ood_dists):.6f}, AUROC={auroc:.4f}")

    results["prompts"] = prompt_results

    # Part 2: Image Size Variations
    print("\n=== Part 2: Image Size Robustness ===")
    default_prompt = prompts[0]

    size_results = {}
    for size in [64, 128, 224, 256, 384, 512]:
        # Resize image to different sizes (processor handles final resize)
        resized = base_img.resize((size, size), Image.BILINEAR)
        clean_emb = extract_hidden(model, processor, resized, default_prompt)
        clean_emb2 = extract_hidden(model, processor, resized, default_prompt)
        clean_d = float(cosine(clean_emb, clean_emb2))

        ood_dists = []
        for c in corruptions:
            # Apply corruption to resized image
            corrupted = apply_corruption(resized, c, 0.5)
            emb = extract_hidden(model, processor, corrupted, default_prompt)
            d = float(cosine(clean_emb, emb))
            ood_dists.append(d)

        auroc = compute_auroc([clean_d], ood_dists)
        size_results[size] = {
            "clean_distance": clean_d,
            "min_ood": float(min(ood_dists)),
            "auroc": auroc,
            "perfect": clean_d == 0 and min(ood_dists) > 0,
        }
        print(f"  {size}×{size}: clean_d={clean_d:.10f}, min_ood={min(ood_dists):.6f}, AUROC={auroc:.4f}")

    results["image_sizes"] = size_results

    # Part 3: Sequential Inference Stability
    print("\n=== Part 3: Sequential Stability (20 passes) ===")
    clean_emb_ref = extract_hidden(model, processor, base_img, default_prompt)

    sequential_dists = []
    for i in range(20):
        emb = extract_hidden(model, processor, base_img, default_prompt)
        d = float(cosine(clean_emb_ref, emb))
        sequential_dists.append(d)

    results["sequential"] = {
        "n_passes": 20,
        "all_zero": all(d == 0 for d in sequential_dists),
        "max_distance": float(max(sequential_dists)),
        "distances": sequential_dists,
    }
    print(f"  All zero: {results['sequential']['all_zero']}, max: {max(sequential_dists):.10f}")

    # Part 4: Cross-Prompt Detection (calibrate with one prompt, detect with another)
    print("\n=== Part 4: Cross-Prompt Detection ===")
    cross_prompt = {}

    # Calibrate with prompt 0
    cal_emb = extract_hidden(model, processor, base_img, prompts[0])

    for i in [1, 3, 5, 9]:
        # Clean with different prompt
        emb = extract_hidden(model, processor, base_img, prompts[i])
        clean_d = float(cosine(cal_emb, emb))

        # OOD with different prompt
        ood_dists = []
        for c in corruptions:
            corrupted = apply_corruption(base_img, c, 0.5)
            emb_c = extract_hidden(model, processor, corrupted, prompts[i])
            d = float(cosine(cal_emb, emb_c))
            ood_dists.append(d)

        cross_prompt[f"cal0_test{i}"] = {
            "clean_distance": clean_d,
            "ood_distances": dict(zip(corruptions, ood_dists)),
            "min_ood": float(min(ood_dists)),
            "separable": clean_d < min(ood_dists),
        }
        print(f"  Cal P0 → Test P{i}: clean_d={clean_d:.6f}, min_ood={min(ood_dists):.6f}, "
              f"sep={clean_d < min(ood_dists)}")

    results["cross_prompt"] = cross_prompt

    # Part 5: Combined Variation (different prompt + different scene)
    print("\n=== Part 5: Combined Variations ===")
    combined = {}

    for seed in [0, 99, 777]:
        np.random.seed(seed)
        scene = Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8))

        for p_idx in [0, 3, 7]:
            p = prompts[p_idx]
            emb = extract_hidden(model, processor, scene, p)
            emb2 = extract_hidden(model, processor, scene, p)
            clean_d = float(cosine(emb, emb2))

            ood_dists = []
            for c in corruptions:
                corrupted = apply_corruption(scene, c, 0.5)
                emb_c = extract_hidden(model, processor, corrupted, p)
                d = float(cosine(emb, emb_c))
                ood_dists.append(d)

            key = f"s{seed}_p{p_idx}"
            combined[key] = {
                "clean_d": clean_d,
                "min_ood": float(min(ood_dists)),
                "perfect": clean_d == 0 and min(ood_dists) > 0,
            }

    n_perfect = sum(1 for v in combined.values() if v['perfect'])
    results["combined"] = {
        "tests": combined,
        "n_perfect": n_perfect,
        "n_total": len(combined),
    }
    print(f"  {n_perfect}/{len(combined)} combinations perfect")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/inference_robust_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
