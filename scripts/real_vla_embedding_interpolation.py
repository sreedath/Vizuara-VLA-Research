#!/usr/bin/env python3
"""Experiment 346: Embedding Space Interpolation Analysis

Detailed study of the paths between clean and corrupted embeddings:
1. Linear vs actual interpolation paths
2. Detection boundary characterization (where does d first exceed threshold?)
3. Embedding smoothness (Lipschitz constant estimation)
4. Cross-corruption interpolation (paths between different corruption types)
5. Severity-action mapping (which severity causes first action change?)
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

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    # Base scene
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    cal_emb = extract_hidden(model, processor, base_img, prompt)
    clean_tokens = get_action_tokens(model, processor, base_img, prompt)

    ctypes = ['fog', 'night', 'noise', 'blur']

    # ========== 1. Fine-grained severity interpolation ==========
    print("\n=== Fine-Grained Severity Interpolation ===")

    interp_results = {}
    for ct in ctypes:
        severities = np.linspace(0, 1, 21)  # 0, 0.05, 0.10, ..., 1.0
        distances = []
        embeddings = []

        for sev in severities:
            if sev == 0:
                img = base_img
            else:
                img = apply_corruption(base_img, ct, sev)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_emb, emb)
            distances.append(float(d))
            embeddings.append(emb)

        # Compute path curvature
        # Compare actual path length vs straight-line distance
        path_length = 0
        for i in range(1, len(embeddings)):
            path_length += np.linalg.norm(embeddings[i] - embeddings[i-1])
        straight_dist = np.linalg.norm(embeddings[-1] - embeddings[0])
        curvature_ratio = path_length / straight_dist if straight_dist > 0 else 1.0

        # Monotonicity
        is_monotone = all(distances[i] <= distances[i+1] for i in range(len(distances)-1))

        # Smoothness (max step-to-step distance change)
        max_step = max(abs(distances[i+1] - distances[i]) for i in range(len(distances)-1))

        interp_results[ct] = {
            'severities': severities.tolist(),
            'distances': distances,
            'path_length': float(path_length),
            'straight_distance': float(straight_dist),
            'curvature_ratio': float(curvature_ratio),
            'is_monotone': bool(is_monotone),
            'max_step': float(max_step),
        }
        print(f"  {ct}: curvature={curvature_ratio:.3f}, monotone={is_monotone}, max_step={max_step:.6f}")

    results['interpolation'] = interp_results

    # ========== 2. Action change boundary ==========
    print("\n=== Action Change Boundary ===")

    action_boundary = {}
    for ct in ctypes:
        first_change_sev = None
        n_changed = 0

        for sev in np.linspace(0, 1, 21):
            if sev == 0:
                continue
            img = apply_corruption(base_img, ct, sev)
            tokens = get_action_tokens(model, processor, img, prompt)
            changed = sum(1 for a, b in zip(clean_tokens, tokens) if a != b)

            if changed > 0 and first_change_sev is None:
                first_change_sev = float(sev)

            if sev in [0.05, 0.1, 0.3, 0.5, 1.0]:
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(cal_emb, emb)
                print(f"  {ct}@{sev}: {changed}/7 dims changed, d={d:.6f}")

        action_boundary[ct] = {
            'first_change_severity': first_change_sev,
            'clean_tokens': clean_tokens,
        }

    results['action_boundary'] = action_boundary

    # ========== 3. Lipschitz constant estimation ==========
    print("\n=== Lipschitz Constant ===")

    lipschitz = {}
    for ct in ctypes:
        lip_values = []
        prev_emb = cal_emb
        prev_arr = np.array(base_img).astype(np.float32) / 255.0

        for sev in np.linspace(0.05, 1.0, 20):
            img = apply_corruption(base_img, ct, sev)
            emb = extract_hidden(model, processor, img, prompt)
            curr_arr = np.array(img).astype(np.float32) / 255.0

            # Input distance (L2 in pixel space)
            input_dist = np.linalg.norm(curr_arr - prev_arr)
            # Output distance (L2 in embedding space)
            output_dist = np.linalg.norm(emb - prev_emb)

            if input_dist > 0:
                lip_values.append(output_dist / input_dist)

            prev_emb = emb
            prev_arr = curr_arr

        lipschitz[ct] = {
            'mean_lipschitz': float(np.mean(lip_values)),
            'max_lipschitz': float(np.max(lip_values)),
            'std_lipschitz': float(np.std(lip_values)),
        }
        print(f"  {ct}: L={np.mean(lip_values):.4f} (max={np.max(lip_values):.4f})")

    results['lipschitz'] = lipschitz

    # ========== 4. Cross-corruption paths ==========
    print("\n=== Cross-Corruption Paths ===")

    cross_paths = {}
    for ct1 in ctypes:
        for ct2 in ctypes:
            if ct1 >= ct2:
                continue

            # Embed endpoints
            img1 = apply_corruption(base_img, ct1, 0.5)
            img2 = apply_corruption(base_img, ct2, 0.5)
            emb1 = extract_hidden(model, processor, img1, prompt)
            emb2 = extract_hidden(model, processor, img2, prompt)

            # Interpolate in pixel space
            arr1 = np.array(img1).astype(np.float32)
            arr2 = np.array(img2).astype(np.float32)

            interp_dists = []
            for alpha in np.linspace(0, 1, 11):
                blended = np.clip(arr1 * (1 - alpha) + arr2 * alpha, 0, 255).astype(np.uint8)
                img = Image.fromarray(blended)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(cal_emb, emb)
                interp_dists.append(float(d))

            # Is the path convex (does distance stay above min endpoints)?
            min_endpoint = min(interp_dists[0], interp_dists[-1])
            is_convex = all(d >= min_endpoint * 0.9 for d in interp_dists)

            key = f"{ct1}_to_{ct2}"
            cross_paths[key] = {
                'distances': interp_dists,
                'min_along_path': float(min(interp_dists)),
                'max_along_path': float(max(interp_dists)),
                'is_convex': bool(is_convex),
            }
            print(f"  {key}: min={min(interp_dists):.6f}, max={max(interp_dists):.6f}, "
                  f"convex={is_convex}")

    results['cross_paths'] = cross_paths

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/embedding_interpolation_{ts}.json"
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
