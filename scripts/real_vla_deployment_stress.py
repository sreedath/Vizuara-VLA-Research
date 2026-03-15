#!/usr/bin/env python3
"""Experiment 336: Deployment Stress Testing (Real OpenVLA-7B)

Simulates extreme deployment scenarios:
1. Rapid corruption switching (worst-case transitions)
2. Gradual degradation over 50-frame sequences
3. Intermittent corruption (random drops)
4. Multi-type simultaneous corruption
5. Adversarial corruption ordering
6. Detector latency under sequential queries
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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    ctypes = ['fog', 'night', 'noise', 'blur']

    # ========== 1. Rapid corruption switching ==========
    print("\n=== Rapid Corruption Switching ===")
    switching_results = {}

    # Sequence: clean → fog → clean → night → clean → blur → clean → noise → clean
    sequence = [
        ('clean', 0), ('fog', 1.0), ('clean', 0), ('night', 1.0),
        ('clean', 0), ('blur', 1.0), ('clean', 0), ('noise', 1.0),
        ('clean', 0), ('fog', 0.5), ('night', 0.5), ('blur', 0.5),
        ('noise', 0.5), ('clean', 0),
    ]

    dists = []
    labels = []
    for ct, sev in sequence:
        if ct == 'clean':
            img = base_img
        else:
            img = apply_corruption(base_img, ct, sev)
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        dists.append(float(d))
        labels.append(f"{ct}@{sev}")

    switching_results = {
        'sequence': labels,
        'distances': dists,
        'correct_detections': sum(1 for i, (ct, _) in enumerate(sequence) if (ct != 'clean' and dists[i] > 0) or (ct == 'clean' and dists[i] == 0)),
        'total_frames': len(sequence),
    }
    print(f"  Correct: {switching_results['correct_detections']}/{switching_results['total_frames']}")
    for label, d in zip(labels, dists):
        print(f"    {label}: d={d:.6f}")

    results['rapid_switching'] = switching_results

    # ========== 2. Gradual degradation ==========
    print("\n=== Gradual Degradation ===")
    gradual_results = {}

    for ct in ctypes:
        sevs = np.linspace(0, 1.0, 20)
        dists_grad = []
        for sev in sevs:
            if sev == 0:
                img = base_img
            else:
                img = apply_corruption(base_img, ct, float(sev))
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(clean_emb, emb)
            dists_grad.append(float(d))

        # Find first detection (d > 0)
        first_detect = next((i for i, d in enumerate(dists_grad) if d > 0), -1)
        gradual_results[ct] = {
            'severities': [float(s) for s in sevs],
            'distances': dists_grad,
            'first_detect_idx': first_detect,
            'first_detect_sev': float(sevs[first_detect]) if first_detect >= 0 else None,
        }
        print(f"  {ct}: first_detect_sev={gradual_results[ct]['first_detect_sev']}")

    results['gradual_degradation'] = gradual_results

    # ========== 3. Intermittent corruption ==========
    print("\n=== Intermittent Corruption ===")
    intermittent_results = {}

    rng = np.random.RandomState(42)
    # 30-frame sequence with random corruption drops
    n_frames = 30
    corruption_mask = rng.random(n_frames) < 0.3  # 30% corruption rate
    dists_inter = []
    labels_inter = []
    for i in range(n_frames):
        if corruption_mask[i]:
            ct = ctypes[rng.randint(0, 4)]
            sev = rng.uniform(0.3, 1.0)
            img = apply_corruption(base_img, ct, sev)
            labels_inter.append(f"{ct}@{sev:.2f}")
        else:
            img = base_img
            labels_inter.append("clean")

        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        dists_inter.append(float(d))

    # Detection analysis
    tp = sum(1 for i in range(n_frames) if corruption_mask[i] and dists_inter[i] > 0)
    fp = sum(1 for i in range(n_frames) if not corruption_mask[i] and dists_inter[i] > 0)
    fn = sum(1 for i in range(n_frames) if corruption_mask[i] and dists_inter[i] == 0)
    tn = sum(1 for i in range(n_frames) if not corruption_mask[i] and dists_inter[i] == 0)

    intermittent_results = {
        'n_frames': n_frames,
        'corruption_rate': float(corruption_mask.mean()),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        'distances': dists_inter,
        'labels': labels_inter,
    }
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Sens={intermittent_results['sensitivity']:.3f}, Spec={intermittent_results['specificity']:.3f}")

    results['intermittent'] = intermittent_results

    # ========== 4. Multi-type simultaneous ==========
    print("\n=== Multi-Type Simultaneous ===")
    multi_results = {}

    arr_base = np.array(base_img).astype(np.float32) / 255.0
    combos = [
        ('fog+noise', ['fog', 'noise']),
        ('fog+blur', ['fog', 'blur']),
        ('night+noise', ['night', 'noise']),
        ('night+blur', ['night', 'blur']),
        ('fog+night+noise', ['fog', 'night', 'noise']),
        ('all4', ['fog', 'night', 'noise', 'blur']),
    ]

    for name, types in combos:
        img = base_img
        for ct in types:
            img = apply_corruption(img, ct, 0.3)
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        multi_results[name] = {
            'distance': float(d),
            'detected': bool(d > 0),
            'n_corruptions': len(types),
        }
        print(f"  {name}: d={d:.6f}")

    results['multi_simultaneous'] = multi_results

    # ========== 5. Worst-case corruption ordering ==========
    print("\n=== Worst-Case Ordering ===")
    ordering_results = {}

    # Apply corruptions in different orders
    orders = [
        ['fog', 'night', 'noise', 'blur'],
        ['blur', 'noise', 'night', 'fog'],
        ['noise', 'fog', 'blur', 'night'],
        ['night', 'blur', 'fog', 'noise'],
    ]

    for order in orders:
        img = base_img
        for ct in order:
            img = apply_corruption(img, ct, 0.25)
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        order_name = '→'.join(order)
        ordering_results[order_name] = {
            'distance': float(d),
            'detected': bool(d > 0),
        }
        print(f"  {order_name}: d={d:.6f}")

    results['ordering'] = ordering_results

    # ========== 6. Detection latency benchmark ==========
    print("\n=== Detection Latency ===")
    latency_results = {}

    # Warm-up
    _ = extract_hidden(model, processor, base_img, prompt)

    # Time individual detections
    latencies = []
    for trial in range(20):
        ct = ctypes[trial % 4]
        sev = 0.5
        img = apply_corruption(base_img, ct, sev)

        t_start = time.time()
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        t_end = time.time()

        latencies.append(float(t_end - t_start))

    latency_results = {
        'mean_ms': float(np.mean(latencies) * 1000),
        'std_ms': float(np.std(latencies) * 1000),
        'min_ms': float(np.min(latencies) * 1000),
        'max_ms': float(np.max(latencies) * 1000),
        'p95_ms': float(np.percentile(latencies, 95) * 1000),
        'fps': float(1.0 / np.mean(latencies)),
    }
    print(f"  Mean: {latency_results['mean_ms']:.1f}ms, P95: {latency_results['p95_ms']:.1f}ms")
    print(f"  FPS: {latency_results['fps']:.1f}")

    results['latency'] = latency_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/deploy_stress_{ts}.json"
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
