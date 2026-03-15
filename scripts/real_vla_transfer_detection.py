#!/usr/bin/env python3
"""Experiment 361: Transfer Detection

Can a detector calibrated on one scene type detect corruptions in another?
1. Train on random pixels, test on structured scenes
2. Train on structured, test on random
3. Cross-scene-type transfer matrix
4. Universal calibration: mixed training set
5. Domain gap quantification
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def create_random(seed):
    rng = np.random.RandomState(seed)
    px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(px)

def create_gradient(seed):
    rng = np.random.RandomState(seed)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    horizon = 80 + rng.randint(-20, 20)
    for y in range(horizon):
        t = y / horizon
        img[y] = [int(135 + 50*t), int(206 - 50*t), int(235 - 30*t)]
    ground_color = rng.randint(40, 120, 3)
    img[horizon:] = ground_color
    for _ in range(3):
        x, y = rng.randint(0, 200), rng.randint(horizon, 200)
        c = rng.randint(50, 255, 3)
        img[y:y+20, x:x+20] = c
    return Image.fromarray(img)

def create_checkerboard(seed):
    rng = np.random.RandomState(seed)
    size = rng.choice([8, 16, 32])
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    c1 = rng.randint(20, 120, 3)
    c2 = rng.randint(130, 255, 3)
    for y in range(224):
        for x in range(224):
            if ((y // size) + (x // size)) % 2:
                img[y, x] = c1
            else:
                img[y, x] = c2
    return Image.fromarray(img)

def create_natural_texture(seed):
    rng = np.random.RandomState(seed)
    coarse = rng.randint(50, 200, (28, 28, 3), dtype=np.uint8)
    img = Image.fromarray(coarse).resize((224, 224), Image.BILINEAR)
    return img

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

    scene_types = {
        'random': create_random,
        'gradient': create_gradient,
        'checkerboard': create_checkerboard,
        'natural': create_natural_texture,
    }

    # Generate embeddings for each scene type
    print("Generating embeddings...")
    n_per_type = 10
    embeddings = {}  # {scene_type: {seed: {'clean': emb, 'fog': emb, ...}}}
    images_cache = {}

    for stype, create_fn in scene_types.items():
        embeddings[stype] = {}
        images_cache[stype] = {}
        for i in range(n_per_type):
            seed = 42 + i * 100
            img = create_fn(seed)
            images_cache[stype][seed] = img
            clean_emb = extract_hidden(model, processor, img, prompt)
            embeddings[stype][seed] = {'clean': clean_emb}
            for ct in ctypes:
                corrupted = apply_corruption(img, ct, 0.5)
                embeddings[stype][seed][ct] = extract_hidden(model, processor, corrupted, prompt)
        print(f"  {stype}: done")

    # ========== 1. Cross-Scene Transfer Matrix ==========
    print("\n=== Cross-Scene Transfer Matrix ===")

    transfer_matrix = {}
    for train_type in scene_types:
        train_seeds = list(embeddings[train_type].keys())
        centroid = np.mean([embeddings[train_type][s]['clean'] for s in train_seeds], axis=0)

        for test_type in scene_types:
            test_seeds = list(embeddings[test_type].keys())

            id_dists = [float(cosine_dist(centroid, embeddings[test_type][s]['clean'])) for s in test_seeds]

            per_ct = {}
            for ct in ctypes:
                ood_dists = [float(cosine_dist(centroid, embeddings[test_type][s][ct])) for s in test_seeds]
                auroc = float(compute_auroc(id_dists, ood_dists))
                per_ct[ct] = auroc

            key = train_type + '_to_' + test_type
            transfer_matrix[key] = {
                'per_type': per_ct,
                'mean_auroc': float(np.mean(list(per_ct.values()))),
                'is_same_domain': train_type == test_type,
            }

    # Print as matrix
    for train_type in scene_types:
        row = []
        for test_type in scene_types:
            key = train_type + '_to_' + test_type
            row.append(f"{transfer_matrix[key]['mean_auroc']:.3f}")
        print(f"  Train={train_type}: " + ', '.join(f'{tt}={r}' for tt, r in zip(scene_types.keys(), row)))

    results['transfer_matrix'] = transfer_matrix

    # ========== 2. Domain Gap Quantification ==========
    print("\n=== Domain Gap ===")

    domain_gap = {}
    for t1 in scene_types:
        for t2 in scene_types:
            if t1 >= t2:
                continue
            c1 = np.mean([embeddings[t1][s]['clean'] for s in embeddings[t1]], axis=0)
            c2 = np.mean([embeddings[t2][s]['clean'] for s in embeddings[t2]], axis=0)
            gap = float(cosine_dist(c1, c2))
            domain_gap[t1 + '_vs_' + t2] = gap

    results['domain_gap'] = domain_gap
    closest = min(domain_gap.items(), key=lambda x: x[1])
    farthest = max(domain_gap.items(), key=lambda x: x[1])
    print(f"  Closest domains: {closest[0]} = {closest[1]:.6f}")
    print(f"  Farthest domains: {farthest[0]} = {farthest[1]:.6f}")

    # ========== 3. Universal Calibration ==========
    print("\n=== Universal Calibration ===")

    # Centroid from mixed training
    all_clean = []
    for stype in scene_types:
        for seed in embeddings[stype]:
            all_clean.append(embeddings[stype][seed]['clean'])
    universal_centroid = np.mean(all_clean, axis=0)

    universal_results = {}
    for test_type in scene_types:
        test_seeds = list(embeddings[test_type].keys())
        id_dists = [float(cosine_dist(universal_centroid, embeddings[test_type][s]['clean'])) for s in test_seeds]

        per_ct = {}
        for ct in ctypes:
            ood_dists = [float(cosine_dist(universal_centroid, embeddings[test_type][s][ct])) for s in test_seeds]
            auroc = float(compute_auroc(id_dists, ood_dists))
            per_ct[ct] = auroc

        universal_results[test_type] = {
            'per_type': per_ct,
            'mean_auroc': float(np.mean(list(per_ct.values()))),
        }
        auroc_str = ', '.join(ct + '=' + format(per_ct[ct], '.3f') for ct in ctypes)
        print(f"  {test_type}: {auroc_str}")

    results['universal'] = universal_results

    # ========== 4. Transfer Degradation Analysis ==========
    print("\n=== Transfer Degradation ===")

    degradation = {}
    for train_type in scene_types:
        for test_type in scene_types:
            if train_type == test_type:
                continue
            same_key = train_type + '_to_' + train_type
            cross_key = train_type + '_to_' + test_type
            same_auroc = transfer_matrix[same_key]['mean_auroc']
            cross_auroc = transfer_matrix[cross_key]['mean_auroc']
            deg = same_auroc - cross_auroc
            degradation[cross_key] = {
                'same_domain': same_auroc,
                'cross_domain': cross_auroc,
                'degradation': float(deg),
                'relative_degradation': float(deg / same_auroc) if same_auroc > 0 else 0,
            }

    # Find worst transfer
    worst = max(degradation.items(), key=lambda x: x[1]['degradation'])
    best = min(degradation.items(), key=lambda x: x[1]['degradation'])

    results['degradation'] = {
        'per_transfer': degradation,
        'worst_transfer': {'key': worst[0], 'degradation': worst[1]['degradation']},
        'best_transfer': {'key': best[0], 'degradation': best[1]['degradation']},
    }
    print(f"  Worst transfer: {worst[0]} (deg={worst[1]['degradation']:.4f})")
    print(f"  Best transfer: {best[0]} (deg={best[1]['degradation']:.4f})")

    # ========== 5. Per-Corruption Transfer ==========
    print("\n=== Per-Corruption Transfer Robustness ===")

    ct_transfer = {}
    for ct in ctypes:
        cross_aurocs = []
        for train_type in scene_types:
            for test_type in scene_types:
                if train_type == test_type:
                    continue
                key = train_type + '_to_' + test_type
                cross_aurocs.append(transfer_matrix[key]['per_type'][ct])

        ct_transfer[ct] = {
            'mean_cross_auroc': float(np.mean(cross_aurocs)),
            'min_cross_auroc': float(min(cross_aurocs)),
            'std_cross_auroc': float(np.std(cross_aurocs)),
            'all_perfect': all(a >= 1.0 - 1e-10 for a in cross_aurocs),
        }
        print(f"  {ct}: cross_mean={np.mean(cross_aurocs):.4f}, min={min(cross_aurocs):.4f}")

    results['per_corruption_transfer'] = ct_transfer

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/transfer_detection_{ts}.json"
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
