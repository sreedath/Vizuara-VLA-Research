#!/usr/bin/env python3
"""Experiment 286: Corruption Transition Manifold
Analyzes embedding paths between corruption types:
1. Interpolate between two corruption types (e.g., fog→night)
2. Check if transition path passes through clean-like regions
3. Compute geodesic distances between all corruption pairs
4. Analyze embedding curvature along transitions
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

def apply_corruption(image, corruption_type, severity):
    """Apply a specific corruption at given severity."""
    if severity == 0:
        return image.copy()
    img = image.copy()
    if corruption_type == "fog":
        fog = Image.new('RGB', img.size, (200, 200, 200))
        return Image.blend(img, fog, severity * 0.8)
    elif corruption_type == "night":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.0 - severity * 0.9)
    elif corruption_type == "blur":
        radius = severity * 20
        return img.filter(ImageFilter.GaussianBlur(radius=max(0.1, radius)))
    elif corruption_type == "noise":
        arr = np.array(img).astype(np.float32)
        noise = np.random.RandomState(42).normal(0, severity * 100, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    elif corruption_type == "snow":
        arr = np.array(img).astype(np.float32)
        rng = np.random.RandomState(42)
        mask = rng.random(arr.shape[:2]) < severity * 0.3
        arr[mask] = 255
        return Image.fromarray(arr.astype(np.uint8))
    elif corruption_type == "rain":
        arr = np.array(img).astype(np.float32)
        rng = np.random.RandomState(42)
        for _ in range(int(severity * 200)):
            x = rng.randint(0, arr.shape[1])
            y_start = rng.randint(0, arr.shape[0])
            length = rng.randint(5, 20)
            y_end = min(y_start + length, arr.shape[0])
            arr[y_start:y_end, x, :] = arr[y_start:y_end, x, :] * 0.5 + 128
        return Image.fromarray(arr.astype(np.uint8))
    return img

def extract_hidden(model, processor, image, prompt, layer=3):
    """Extract hidden state at specified layer."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\n"
    image = Image.new('RGB', (224, 224))
    rng = np.random.RandomState(42)
    image = Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Get clean centroid
    print("Getting clean centroid...")
    clean_emb = extract_hidden(model, processor, image, prompt)

    # Get corruption embeddings at full severity
    corruptions = ["fog", "night", "blur", "noise", "snow", "rain"]
    full_corruption_embs = {}
    for c in corruptions:
        print(f"  Getting {c} embedding...")
        corrupted = apply_corruption(image, c, 1.0)
        full_corruption_embs[c] = extract_hidden(model, processor, corrupted, prompt)

    results = {
        "experiment": "corruption_transition_manifold",
        "experiment_number": 286,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Part 1: Pairwise transition paths
    # Interpolate between two corruption types at severity=1.0
    # by blending their corrupted images
    print("\n=== Part 1: Pairwise Transition Paths ===")
    transition_paths = {}
    n_steps = 11  # 0, 0.1, ..., 1.0
    pair_distances = {}

    for i, c1 in enumerate(corruptions):
        for c2 in corruptions[i+1:]:
            pair_key = f"{c1}_to_{c2}"
            print(f"  Transition: {pair_key}")
            img1 = apply_corruption(image, c1, 1.0)
            img2 = apply_corruption(image, c2, 1.0)

            path = []
            for step in range(n_steps):
                alpha = step / (n_steps - 1)
                # Pixel-level interpolation
                blended = Image.blend(img1, img2, alpha)
                emb = extract_hidden(model, processor, blended, prompt)
                d_clean = float(cosine(clean_emb, emb))
                d_c1 = float(cosine(full_corruption_embs[c1], emb))
                d_c2 = float(cosine(full_corruption_embs[c2], emb))
                path.append({
                    "alpha": round(alpha, 2),
                    "d_clean": d_clean,
                    "d_c1": d_c1,
                    "d_c2": d_c2
                })

            transition_paths[pair_key] = path
            # Direct distance between corruption embeddings
            pair_distances[pair_key] = float(cosine(
                full_corruption_embs[c1], full_corruption_embs[c2]))

    results["transition_paths"] = transition_paths
    results["pair_distances"] = pair_distances

    # Part 2: Minimum distance to clean along each transition
    print("\n=== Part 2: Minimum Clean Distance Along Transitions ===")
    min_clean_distances = {}
    for pair_key, path in transition_paths.items():
        min_d = min(p["d_clean"] for p in path)
        min_alpha = [p["alpha"] for p in path if p["d_clean"] == min_d][0]
        min_clean_distances[pair_key] = {
            "min_d_clean": min_d,
            "at_alpha": min_alpha,
            "still_detectable": min_d > 0
        }
    results["min_clean_distances"] = min_clean_distances

    # Part 3: Geodesic distance matrix
    print("\n=== Part 3: Geodesic Distance Matrix ===")
    geodesic_matrix = {}
    for c1 in corruptions:
        geodesic_matrix[c1] = {}
        for c2 in corruptions:
            if c1 == c2:
                geodesic_matrix[c1][c2] = 0.0
            else:
                geodesic_matrix[c1][c2] = float(cosine(
                    full_corruption_embs[c1], full_corruption_embs[c2]))
    results["geodesic_matrix"] = geodesic_matrix

    # Part 4: Path curvature analysis
    print("\n=== Part 4: Path Curvature ===")
    curvature = {}
    for pair_key, path in transition_paths.items():
        # Straight-line distance between endpoints
        c1, c2 = pair_key.split("_to_")
        straight_line = float(cosine(full_corruption_embs[c1], full_corruption_embs[c2]))
        # Arc length (sum of consecutive distances)
        arc_segments = []
        prev_emb = full_corruption_embs[c1]
        for step in range(1, n_steps):
            alpha = step / (n_steps - 1)
            img1 = apply_corruption(image, c1, 1.0)
            img2 = apply_corruption(image, c2, 1.0)
            blended = Image.blend(img1, img2, alpha)
            curr_emb = extract_hidden(model, processor, blended, prompt)
            arc_segments.append(float(cosine(prev_emb, curr_emb)))
            prev_emb = curr_emb
        arc_length = sum(arc_segments)
        curvature[pair_key] = {
            "straight_line": straight_line,
            "arc_length": arc_length,
            "curvature_ratio": arc_length / straight_line if straight_line > 0 else 0,
            "is_geodesic": arc_length / straight_line < 1.1 if straight_line > 0 else True
        }
    results["curvature"] = curvature

    # Part 5: Severity cross-fade
    # Decrease c1 severity while increasing c2 severity
    print("\n=== Part 5: Severity Cross-Fade ===")
    crossfade_pairs = [("fog", "night"), ("blur", "noise"), ("night", "blur")]
    crossfade_results = {}
    for c1, c2 in crossfade_pairs:
        key = f"{c1}_to_{c2}_crossfade"
        print(f"  Cross-fade: {key}")
        path = []
        for step in range(n_steps):
            alpha = step / (n_steps - 1)
            sev1 = 1.0 - alpha  # decreasing
            sev2 = alpha  # increasing
            # Apply both corruptions sequentially
            img_step = apply_corruption(image, c1, sev1)
            img_step = apply_corruption(img_step, c2, sev2)
            emb = extract_hidden(model, processor, img_step, prompt)
            d_clean = float(cosine(clean_emb, emb))
            path.append({
                "alpha": round(alpha, 2),
                f"sev_{c1}": round(sev1, 2),
                f"sev_{c2}": round(sev2, 2),
                "d_clean": d_clean
            })
        crossfade_results[key] = path
    results["crossfade"] = crossfade_results

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/transition_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Transition paths: {len(transition_paths)}")
    for pk, v in min_clean_distances.items():
        print(f"  {pk}: min_d_clean={v['min_d_clean']:.6f} at alpha={v['at_alpha']}, detectable={v['still_detectable']}")
    print(f"\nCurvature ratios:")
    for pk, v in curvature.items():
        print(f"  {pk}: ratio={v['curvature_ratio']:.3f}, geodesic={v['is_geodesic']}")

if __name__ == "__main__":
    main()
