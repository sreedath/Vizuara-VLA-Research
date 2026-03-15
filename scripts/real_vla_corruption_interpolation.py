#!/usr/bin/env python3
"""Experiment 388: Corruption Interpolation in Embedding Space

How do embeddings transition between corruption types?
1. Linear interpolation between two corruption types (fog→night, etc)
2. Path straightness: is the embedding path linear or curved?
3. Detection along interpolation paths
4. Midpoint corruption identity: what does 50/50 look like?
5. Interpolation in pixel space vs embedding space
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
    if na < 1e-10 or nb < 1e-10: return 0.0
    return 1.0 - dot / (na * nb)

def cosine_sim(a, b):
    return 1.0 - cosine_dist(a, b)

def compute_auroc(id_scores, ood_scores):
    id_s, ood_s = np.asarray(id_scores), np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    centroid = np.mean(list(clean_embs.values()), axis=0)
    clean_dists = [cosine_dist(centroid, clean_embs[s]) for s in seeds]

    # Get pure corruption embeddings
    corrupt_embs = {}
    for ct in ctypes:
        corrupt_embs[ct] = extract_hidden(model, processor,
            apply_corruption(images[seeds[0]], ct, 0.5), prompt)

    # ========== 1. Pixel-Space Interpolation ==========
    print("\n=== Pixel-Space Interpolation ===")

    pairs = [('fog', 'night'), ('fog', 'blur'), ('fog', 'noise'),
             ('night', 'blur'), ('night', 'noise'), ('noise', 'blur')]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    interpolation = {}
    for ct1, ct2 in pairs:
        path_data = {'dists': [], 'detection_dists': []}

        for alpha in alphas:
            # Interpolate in pixel space
            arr1 = np.array(apply_corruption(images[seeds[0]], ct1, 0.5)).astype(np.float32)
            arr2 = np.array(apply_corruption(images[seeds[0]], ct2, 0.5)).astype(np.float32)
            interp_arr = (1 - alpha) * arr1 + alpha * arr2
            interp_img = Image.fromarray(interp_arr.astype(np.uint8))

            emb = extract_hidden(model, processor, interp_img, prompt)
            dist = cosine_dist(emb, centroid)
            path_data['dists'].append(float(dist))
            path_data['detection_dists'].append(float(dist))

        # Path straightness
        emb_start = extract_hidden(model, processor,
            apply_corruption(images[seeds[0]], ct1, 0.5), prompt)
        emb_end = extract_hidden(model, processor,
            apply_corruption(images[seeds[0]], ct2, 0.5), prompt)

        # Measure deviation from straight line in embedding space
        deviations = []
        for i, alpha in enumerate(alphas[1:-1], 1):
            expected = (1 - alpha) * emb_start + alpha * emb_end
            arr1 = np.array(apply_corruption(images[seeds[0]], ct1, 0.5)).astype(np.float32)
            arr2 = np.array(apply_corruption(images[seeds[0]], ct2, 0.5)).astype(np.float32)
            interp_arr = (1 - alpha) * arr1 + alpha * arr2
            actual = extract_hidden(model, processor,
                Image.fromarray(interp_arr.astype(np.uint8)), prompt)
            dev = cosine_dist(expected, actual)
            deviations.append(float(dev))

        path_data['straightness'] = 1.0 - float(np.mean(deviations))
        path_data['max_deviation'] = float(max(deviations)) if deviations else 0
        path_data['endpoint_dist'] = float(cosine_dist(emb_start, emb_end))

        interpolation[f'{ct1}_to_{ct2}'] = path_data
        print(f"  {ct1}→{ct2}: straightness={path_data['straightness']:.4f}, "
              f"endpoint_dist={path_data['endpoint_dist']:.6f}")

    results['interpolation'] = interpolation

    # ========== 2. Midpoint Analysis ==========
    print("\n=== Midpoint Analysis ===")

    midpoints = {}
    for ct1, ct2 in pairs:
        arr1 = np.array(apply_corruption(images[seeds[0]], ct1, 0.5)).astype(np.float32)
        arr2 = np.array(apply_corruption(images[seeds[0]], ct2, 0.5)).astype(np.float32)
        mid_arr = 0.5 * arr1 + 0.5 * arr2
        mid_emb = extract_hidden(model, processor,
            Image.fromarray(mid_arr.astype(np.uint8)), prompt)

        # Which corruption is the midpoint closer to?
        sims = {ct: float(cosine_sim(mid_emb, corrupt_embs[ct])) for ct in ctypes}
        closest = max(sims, key=sims.get)

        midpoints[f'{ct1}_to_{ct2}'] = {
            'closest_corruption': closest,
            'similarities': sims,
            'dist_to_centroid': float(cosine_dist(mid_emb, centroid)),
        }
        print(f"  {ct1}+{ct2} midpoint closest to: {closest} "
              f"(sim={sims[closest]:.4f})")

    results['midpoints'] = midpoints

    # ========== 3. Detection Along Paths ==========
    print("\n=== Detection Along Paths ===")

    # Are all interpolation points detected?
    for key, data_path in interpolation.items():
        all_detected = all(d > max(clean_dists) for d in data_path['dists'])
        min_dist = min(data_path['dists'])
        results[f'path_{key}_all_detected'] = all_detected
        results[f'path_{key}_min_dist'] = float(min_dist)
        results[f'path_{key}_min_vs_threshold'] = float(min_dist / max(max(clean_dists), 1e-10))

    # ========== 4. Severity Interpolation ==========
    print("\n=== Severity Interpolation ===")

    severity_interp = {}
    for ct in ctypes:
        path_dists = []
        for sev in np.linspace(0, 1, 11):
            if sev == 0:
                emb = clean_embs[seeds[0]]
            else:
                emb = extract_hidden(model, processor,
                    apply_corruption(images[seeds[0]], ct, sev), prompt)
            path_dists.append(float(cosine_dist(emb, centroid)))

        # Check monotonicity
        diffs = [path_dists[i+1] - path_dists[i] for i in range(len(path_dists)-1)]
        monotonic = all(d >= -1e-8 for d in diffs)

        severity_interp[ct] = {
            'dists': path_dists,
            'monotonic': monotonic,
            'n_decreases': sum(1 for d in diffs if d < -1e-8),
        }
        print(f"  {ct}: monotonic={monotonic}, min_dist={min(path_dists):.6f}, "
              f"max_dist={max(path_dists):.6f}")

    results['severity_interpolation'] = severity_interp

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/corruption_interpolation_{ts}.json"
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
