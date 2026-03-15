#!/usr/bin/env python3
"""Experiment 357: Multi-Corruption Composition

What happens when multiple corruptions are applied simultaneously?
1. Pairwise corruption composition: fog+noise, night+blur, etc.
2. Triple corruption stacking
3. Composition order effects (commutative?)
4. Severity interaction: additive vs multiplicative
5. Detection under composed corruptions
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
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

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    clean_embs = {}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    centroid = np.mean([clean_embs[s] for s in seeds], axis=0)
    clean_dists = [float(cosine_dist(centroid, clean_embs[s])) for s in seeds]
    clean_max = max(clean_dists)

    # Single corruption baselines
    print("\nSingle corruption baselines...")
    single_dists = {}
    for ct in ctypes:
        dists = []
        for seed in seeds:
            img = apply_corruption(images[seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            dists.append(float(cosine_dist(centroid, emb)))
        single_dists[ct] = dists
        print(f"  {ct}: mean={np.mean(dists):.6f}")

    results['single_baselines'] = {ct: {'mean': float(np.mean(d)), 'std': float(np.std(d))} for ct, d in single_dists.items()}

    # ========== 1. Pairwise Composition ==========
    print("\n=== Pairwise Corruption Composition ===")

    pairwise = {}
    for i, ct1 in enumerate(ctypes):
        for j, ct2 in enumerate(ctypes):
            if i >= j:
                continue
            key = ct1 + '+' + ct2
            dists_12 = []
            dists_21 = []

            for seed in seeds:
                # Apply ct1 then ct2
                img12 = apply_corruption(images[seed], ct1, 0.5)
                img12 = apply_corruption(img12, ct2, 0.5)
                emb12 = extract_hidden(model, processor, img12, prompt)
                dists_12.append(float(cosine_dist(centroid, emb12)))

                # Apply ct2 then ct1
                img21 = apply_corruption(images[seed], ct2, 0.5)
                img21 = apply_corruption(img21, ct1, 0.5)
                emb21 = extract_hidden(model, processor, img21, prompt)
                dists_21.append(float(cosine_dist(centroid, emb21)))

            mean_12 = float(np.mean(dists_12))
            mean_21 = float(np.mean(dists_21))
            expected_sum = float(np.mean(single_dists[ct1])) + float(np.mean(single_dists[ct2]))
            expected_max = max(float(np.mean(single_dists[ct1])), float(np.mean(single_dists[ct2])))

            # Are compositions commutative?
            order_diffs = [abs(d12 - d21) for d12, d21 in zip(dists_12, dists_21)]

            pairwise[key] = {
                'ct1_then_ct2_mean': mean_12,
                'ct2_then_ct1_mean': mean_21,
                'order_diff_mean': float(np.mean(order_diffs)),
                'order_diff_max': float(max(order_diffs)),
                'commutative': float(max(order_diffs)) < 0.001,
                'sum_of_singles': expected_sum,
                'max_of_singles': expected_max,
                'superadditivity': mean_12 / expected_sum if expected_sum > 0 else 0,
                'vs_max': mean_12 / expected_max if expected_max > 0 else 0,
                'auroc_12': float(compute_auroc(clean_dists, dists_12)),
                'auroc_21': float(compute_auroc(clean_dists, dists_21)),
            }
            print(f"  {key}: d={mean_12:.6f} (order_diff={np.mean(order_diffs):.6f}), "
                  f"super={mean_12/expected_sum:.2f}x sum, {mean_12/expected_max:.2f}x max")

    results['pairwise'] = pairwise

    # ========== 2. Triple Composition ==========
    print("\n=== Triple Corruption Composition ===")

    triple = {}
    triple_combos = [
        ('fog', 'noise', 'blur'),
        ('night', 'noise', 'blur'),
        ('fog', 'night', 'noise'),
        ('fog', 'night', 'blur'),
    ]

    for combo in triple_combos:
        key = '+'.join(combo)
        dists = []

        for seed in seeds:
            img = images[seed]
            for ct in combo:
                img = apply_corruption(img, ct, 0.3)  # lower severity for triple
            emb = extract_hidden(model, processor, img, prompt)
            dists.append(float(cosine_dist(centroid, emb)))

        sum_singles = sum(float(np.mean(single_dists[ct])) for ct in combo)
        max_single = max(float(np.mean(single_dists[ct])) for ct in combo)

        triple[key] = {
            'mean_dist': float(np.mean(dists)),
            'std_dist': float(np.std(dists)),
            'sum_of_singles': sum_singles,
            'max_of_singles': max_single,
            'superadditivity': float(np.mean(dists)) / sum_singles if sum_singles > 0 else 0,
            'auroc': float(compute_auroc(clean_dists, dists)),
        }
        print(f"  {key}: d={np.mean(dists):.6f}, super={np.mean(dists)/sum_singles:.2f}x, "
              f"AUROC={compute_auroc(clean_dists, dists):.4f}")

    results['triple'] = triple

    # ========== 3. Severity Interaction ==========
    print("\n=== Severity Interaction ===")

    # For fog+blur: vary severity of each independently
    sev_interaction = {}
    ct1, ct2 = 'fog', 'blur'
    seed = seeds[0]

    for s1 in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for s2 in [0.1, 0.3, 0.5, 0.7, 1.0]:
            img = apply_corruption(images[seed], ct1, s1)
            img = apply_corruption(img, ct2, s2)
            emb = extract_hidden(model, processor, img, prompt)
            d = float(cosine_dist(centroid, emb))
            sev_interaction[f"fog={s1}_blur={s2}"] = d

    results['severity_interaction'] = {
        'corruption_pair': f"{ct1}+{ct2}",
        'grid': sev_interaction,
    }

    # Check if interaction is additive: d(s1,s2) ≈ d(s1,0) + d(0,s2)?
    print(f"  Severity interaction grid ({ct1}+{ct2}):")
    for s1 in [0.1, 0.5, 1.0]:
        row = []
        for s2 in [0.1, 0.5, 1.0]:
            d = sev_interaction[f"fog={s1}_blur={s2}"]
            row.append(f"{d:.5f}")
        print(f"    fog={s1}: blur=[{', '.join(row)}]")

    # ========== 4. Cancellation Detection ==========
    print("\n=== Cancellation Analysis ===")
    # Can one corruption cancel another? (e.g., night darkens, fog brightens)

    cancel = {}
    for seed in seeds[:5]:
        # fog + night at varying severities
        for sf in [0.3, 0.5, 0.7]:
            for sn in [0.3, 0.5, 0.7]:
                img = apply_corruption(images[seed], 'fog', sf)
                img = apply_corruption(img, 'night', sn)
                emb = extract_hidden(model, processor, img, prompt)
                d = float(cosine_dist(centroid, emb))
                key = f"seed={seed}_fog={sf}_night={sn}"
                cancel[key] = {
                    'distance': d,
                    'detected': d > clean_max,
                }

    # Find minimum distance (closest to clean)
    min_cancel = min(cancel.items(), key=lambda x: x[1]['distance'])
    max_cancel = max(cancel.items(), key=lambda x: x[1]['distance'])

    n_detected = sum(1 for v in cancel.values() if v['detected'])

    results['cancellation'] = {
        'pairs_tested': len(cancel),
        'min_dist': {'key': min_cancel[0], 'distance': min_cancel[1]['distance']},
        'max_dist': {'key': max_cancel[0], 'distance': max_cancel[1]['distance']},
        'n_detected': n_detected,
        'detection_rate': n_detected / len(cancel),
        'any_cancelled': min_cancel[1]['distance'] <= clean_max,
    }
    print(f"  fog+night cancellation: min_d={min_cancel[1]['distance']:.6f} "
          f"(clean_max={clean_max:.6f}), detection={n_detected}/{len(cancel)}")

    # ========== 5. All-Four Composition ==========
    print("\n=== All-Four Corruption Composition ===")

    all4_dists = []
    for seed in seeds:
        img = images[seed]
        for ct in ctypes:
            img = apply_corruption(img, ct, 0.25)  # mild each
        emb = extract_hidden(model, processor, img, prompt)
        all4_dists.append(float(cosine_dist(centroid, emb)))

    results['all_four'] = {
        'mean_dist': float(np.mean(all4_dists)),
        'std_dist': float(np.std(all4_dists)),
        'auroc': float(compute_auroc(clean_dists, all4_dists)),
        'all_detected': all(d > clean_max for d in all4_dists),
        'vs_sum_singles': float(np.mean(all4_dists)) / sum(float(np.mean(single_dists[ct])) for ct in ctypes),
    }
    print(f"  All four at 0.25: mean_d={np.mean(all4_dists):.6f}, "
          f"AUROC={compute_auroc(clean_dists, all4_dists):.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/multi_corruption_{ts}.json"
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
