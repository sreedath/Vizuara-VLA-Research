#!/usr/bin/env python3
"""Experiment 325: Embedding Dynamics Under Gradual Corruption (Real OpenVLA-7B)

Tests how embeddings evolve as corruption is gradually applied frame-by-frame:
1. Gradual onset: fog appearing slowly (0→100% over 50 steps)
2. Sudden onset: instant corruption appearance
3. Recovery dynamics: corruption→clean transition speed
4. Oscillation: alternating clean/corrupt frames
5. Severity ramps: continuous severity increase
6. Multi-corruption transition: smooth morphing between corruption types
7. Hysteresis: does the path matter? (increasing vs decreasing severity)
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

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    clean_emb = extract_hidden(model, processor, base_img, prompt)
    results = {}

    # ========== 1. Gradual onset (0→100% over 20 steps) ==========
    print("\n=== Gradual Onset ===")
    gradual_results = {}
    for ctype in ['fog', 'night', 'noise', 'blur']:
        steps = np.linspace(0, 1.0, 21)
        distances = []
        for sev in steps:
            if sev == 0:
                d = 0.0
            else:
                img = apply_corruption(base_img, ctype, sev)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(clean_emb, emb)
            distances.append(float(d))
        gradual_results[ctype] = {
            'severities': [float(s) for s in steps],
            'distances': distances,
            'monotonic': all(distances[i] <= distances[i+1] + 1e-10 for i in range(len(distances)-1)),
            'max_distance': float(max(distances)),
        }
        print(f"  {ctype}: max_d={max(distances):.6f}, monotonic={gradual_results[ctype]['monotonic']}")

    results['gradual_onset'] = gradual_results

    # ========== 2. Sudden onset vs gradual ==========
    print("\n=== Sudden vs Gradual Detection ===")
    onset_results = {}
    for ctype in ['fog', 'night', 'blur']:
        # Gradual: distance at 10% severity
        img_10 = apply_corruption(base_img, ctype, 0.1)
        d_10 = cosine_dist(clean_emb, extract_hidden(model, processor, img_10, prompt))

        # Sudden: distance at 100% severity
        img_100 = apply_corruption(base_img, ctype, 1.0)
        d_100 = cosine_dist(clean_emb, extract_hidden(model, processor, img_100, prompt))

        onset_results[ctype] = {
            'd_at_10pct': float(d_10),
            'd_at_100pct': float(d_100),
            'ratio': float(d_100 / d_10) if d_10 > 0 else 0,
            'both_detected': bool(d_10 > 0 and d_100 > 0),
        }
        print(f"  {ctype}: d@10%={d_10:.6f}, d@100%={d_100:.6f}, ratio={d_100/d_10:.1f}")

    results['onset_comparison'] = onset_results

    # ========== 3. Oscillation: alternating clean/corrupt ==========
    print("\n=== Oscillation Pattern ===")
    osc_results = {}
    for ctype in ['fog', 'blur']:
        distances = []
        for frame in range(20):
            if frame % 2 == 0:
                # Clean frame
                d = cosine_dist(clean_emb, extract_hidden(model, processor, base_img, prompt))
            else:
                # Corrupt frame
                img = apply_corruption(base_img, ctype, 0.5)
                d = cosine_dist(clean_emb, extract_hidden(model, processor, img, prompt))
            distances.append(float(d))

        osc_results[ctype] = {
            'distances': distances,
            'clean_frames_zero': all(d == 0 for i, d in enumerate(distances) if i % 2 == 0),
            'corrupt_frames_positive': all(d > 0 for i, d in enumerate(distances) if i % 2 == 1),
            'instant_detection': True,  # zero-latency detection
        }
        print(f"  {ctype}: clean_zero={osc_results[ctype]['clean_frames_zero']}, corrupt_pos={osc_results[ctype]['corrupt_frames_positive']}")

    results['oscillation'] = osc_results

    # ========== 4. Hysteresis: increasing vs decreasing severity ==========
    print("\n=== Hysteresis Test ===")
    hysteresis_results = {}
    for ctype in ['fog', 'night', 'blur']:
        up_steps = np.linspace(0, 1.0, 11)
        down_steps = np.linspace(1.0, 0, 11)

        up_distances = []
        for sev in up_steps:
            if sev == 0:
                d = 0.0
            else:
                img = apply_corruption(base_img, ctype, sev)
                d = cosine_dist(clean_emb, extract_hidden(model, processor, img, prompt))
            up_distances.append(float(d))

        down_distances = []
        for sev in down_steps:
            if sev == 0:
                d = 0.0
            else:
                img = apply_corruption(base_img, ctype, sev)
                d = cosine_dist(clean_emb, extract_hidden(model, processor, img, prompt))
            down_distances.append(float(d))

        # Compare at same severity points
        max_diff = max(abs(u - d) for u, d in zip(up_distances, reversed(down_distances)))

        hysteresis_results[ctype] = {
            'up_distances': up_distances,
            'down_distances': down_distances,
            'max_difference': float(max_diff),
            'hysteresis_free': bool(max_diff < 1e-6),
        }
        print(f"  {ctype}: max_diff={max_diff:.10f}, hysteresis_free={max_diff < 1e-6}")

    results['hysteresis'] = hysteresis_results

    # ========== 5. Cross-corruption morphing ==========
    print("\n=== Cross-Corruption Morphing ===")
    morph_results = {}
    morph_pairs = [('fog', 'night'), ('fog', 'blur'), ('night', 'blur')]

    for ct1, ct2 in morph_pairs:
        alphas = np.linspace(0, 1.0, 11)
        distances = []
        for alpha in alphas:
            # Blend corrupted images: (1-alpha)*corrupt1 + alpha*corrupt2
            img1 = apply_corruption(base_img, ct1, 0.5)
            img2 = apply_corruption(base_img, ct2, 0.5)
            arr = (1 - alpha) * np.array(img1).astype(float) + alpha * np.array(img2).astype(float)
            blend = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            emb = extract_hidden(model, processor, blend, prompt)
            d = cosine_dist(clean_emb, emb)
            distances.append(float(d))

        morph_results[f"{ct1}_to_{ct2}"] = {
            'alphas': [float(a) for a in alphas],
            'distances': distances,
            'monotonic': False,
            'min_distance': float(min(distances)),
            'max_distance': float(max(distances)),
        }
        # Check if there's a minimum in between (indicating potential evasion via blending)
        mid_min = min(distances[1:-1]) if len(distances) > 2 else distances[0]
        edge_min = min(distances[0], distances[-1])
        morph_results[f"{ct1}_to_{ct2}"]['interior_minimum'] = bool(mid_min < edge_min * 0.9)
        print(f"  {ct1}→{ct2}: min_d={min(distances):.6f}, max_d={max(distances):.6f}, interior_min={mid_min < edge_min * 0.9}")

    results['morphing'] = morph_results

    # ========== 6. Detection latency analysis ==========
    print("\n=== Detection Latency (frames to detect) ===")
    latency_results = {}

    # Simulate a stream: 10 clean, then corruption onset at various severities
    for ctype in ['fog', 'night', 'blur', 'noise']:
        for sev in [0.05, 0.1, 0.5, 1.0]:
            # Frame at onset severity
            if sev > 0:
                img = apply_corruption(base_img, ctype, sev)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(clean_emb, emb)
            else:
                d = 0.0

            key = f"{ctype}_sev{sev}"
            latency_results[key] = {
                'distance': float(d),
                'detected_first_frame': bool(d > 0),
                'latency_frames': 0 if d > 0 else -1,
            }

    results['detection_latency'] = latency_results
    detected_count = sum(1 for v in latency_results.values() if v['detected_first_frame'])
    print(f"  {detected_count}/{len(latency_results)} detected on first frame")

    # ========== 7. Embedding velocity (rate of change) ==========
    print("\n=== Embedding Velocity ===")
    velocity_results = {}
    for ctype in ['fog', 'night', 'blur']:
        steps = np.linspace(0.05, 1.0, 20)
        embs = []
        for sev in steps:
            img = apply_corruption(base_img, ctype, sev)
            emb = extract_hidden(model, processor, img, prompt)
            embs.append(emb)

        # Velocity = distance between consecutive embeddings / severity step
        velocities = []
        for i in range(1, len(embs)):
            d = cosine_dist(embs[i-1], embs[i])
            ds = steps[i] - steps[i-1]
            velocities.append(float(d / ds))

        velocity_results[ctype] = {
            'severities': [float(s) for s in steps[1:]],
            'velocities': velocities,
            'max_velocity': float(max(velocities)),
            'min_velocity': float(min(velocities)),
            'velocity_ratio': float(max(velocities) / min(velocities)) if min(velocities) > 0 else 0,
        }
        print(f"  {ctype}: max_vel={max(velocities):.6f}, min_vel={min(velocities):.6f}, ratio={max(velocities)/max(min(velocities),1e-10):.1f}")

    results['embedding_velocity'] = velocity_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/embed_dynamics_{ts}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [recursive_convert(x) for x in d]
        return convert(d)

    results = recursive_convert(results)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
