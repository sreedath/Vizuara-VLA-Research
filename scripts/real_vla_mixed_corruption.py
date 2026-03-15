#!/usr/bin/env python3
"""Experiment 377: Mixed Corruption Detection

What happens when multiple corruptions are applied simultaneously?
1. Pairwise corruption combinations
2. All-four combined corruption
3. Detection performance on mixed vs single
4. Embedding direction of mixed corruptions
5. Corruption dominance hierarchy
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

def apply_mixed(image, types_sevs):
    """Apply multiple corruptions sequentially."""
    result = image.copy()
    for ctype, sev in types_sevs:
        result = apply_corruption(result, ctype, sev)
    return result

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
    return float(np.dot(a, b) / (na * nb))

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

    # Generate images
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
    threshold = max(clean_dists)

    # Single corruption embeddings for comparison
    single_embs = {ct: {} for ct in ctypes}
    for ct in ctypes:
        for seed in seeds:
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            single_embs[ct][seed] = extract_hidden(model, processor, corrupt_img, prompt)

    print(f"  Threshold: {threshold:.6f}")

    # ========== 1. Pairwise Corruption Combinations ==========
    print("\n=== Pairwise Combinations ===")

    pairwise = {}
    for i, ct1 in enumerate(ctypes):
        for ct2 in ctypes[i+1:]:
            pair_dists = []
            pair_embs = {}
            for seed in seeds:
                mixed_img = apply_mixed(images[seed], [(ct1, 0.5), (ct2, 0.5)])
                emb = extract_hidden(model, processor, mixed_img, prompt)
                pair_embs[seed] = emb
                d = cosine_dist(emb, centroid)
                pair_dists.append(d)

            # How close is mixed to each single corruption?
            sim_to_ct1 = [cosine_sim(pair_embs[s] - centroid, single_embs[ct1][s] - centroid) for s in seeds]
            sim_to_ct2 = [cosine_sim(pair_embs[s] - centroid, single_embs[ct2][s] - centroid) for s in seeds]

            key = f"{ct1}+{ct2}"
            pairwise[key] = {
                'mean_dist': float(np.mean(pair_dists)),
                'min_dist': float(min(pair_dists)),
                'all_detected': all(d > threshold for d in pair_dists),
                'detection_rate': float(sum(1 for d in pair_dists if d > threshold) / len(pair_dists)),
                'sim_to_first': float(np.mean(sim_to_ct1)),
                'sim_to_second': float(np.mean(sim_to_ct2)),
                'dominant': ct1 if np.mean(sim_to_ct1) > np.mean(sim_to_ct2) else ct2,
            }
            # Individual single dists for comparison
            d1_mean = np.mean([cosine_dist(centroid, single_embs[ct1][s]) for s in seeds])
            d2_mean = np.mean([cosine_dist(centroid, single_embs[ct2][s]) for s in seeds])
            pairwise[key]['single_dist_first'] = float(d1_mean)
            pairwise[key]['single_dist_second'] = float(d2_mean)
            pairwise[key]['superadditivity'] = float(np.mean(pair_dists) / (d1_mean + d2_mean))

            print(f"  {key}: dist={np.mean(pair_dists):.6f}, "
                  f"det={pairwise[key]['detection_rate']:.2f}, "
                  f"dominant={pairwise[key]['dominant']}")

    results['pairwise'] = pairwise

    # ========== 2. All-Four Combined ==========
    print("\n=== All-Four Combined ===")

    all_four = {}
    for seed in seeds:
        mixed_img = apply_mixed(images[seed], [(ct, 0.25) for ct in ctypes])
        emb = extract_hidden(model, processor, mixed_img, prompt)
        d = cosine_dist(emb, centroid)

        # Similarity to each single corruption
        sims = {}
        for ct in ctypes:
            shift_mixed = emb - centroid
            shift_single = single_embs[ct][seed] - centroid
            sims[ct] = cosine_sim(shift_mixed, shift_single)

        all_four[str(seed)] = {
            'dist': float(d),
            'detected': d > threshold,
            'similarity_to_singles': {ct: float(sims[ct]) for ct in ctypes},
        }

    all_four_dists = [all_four[str(s)]['dist'] for s in seeds]
    results['all_four'] = {
        'per_scene': all_four,
        'mean_dist': float(np.mean(all_four_dists)),
        'detection_rate': float(sum(1 for d in all_four_dists if d > threshold) / len(all_four_dists)),
        'mean_sim_to_singles': {ct: float(np.mean([all_four[str(s)]['similarity_to_singles'][ct] for s in seeds])) for ct in ctypes},
    }
    print(f"  All-four: mean_dist={np.mean(all_four_dists):.6f}, "
          f"det_rate={results['all_four']['detection_rate']:.2f}")

    # ========== 3. Corruption Dominance Hierarchy ==========
    print("\n=== Corruption Dominance ===")

    dominance = {}
    for ct in ctypes:
        wins = 0
        total = 0
        for key, val in pairwise.items():
            parts = key.split('+')
            if ct in parts:
                total += 1
                if val['dominant'] == ct:
                    wins += 1
        dominance[ct] = {
            'wins': wins,
            'total': total,
            'win_rate': float(wins / max(total, 1)),
        }
        print(f"  {ct}: {wins}/{total} dominant")

    results['dominance'] = dominance

    # ========== 4. Order Sensitivity ==========
    print("\n=== Order Sensitivity ===")

    order_sens = {}
    for i, ct1 in enumerate(ctypes):
        for ct2 in ctypes[i+1:]:
            fwd_dists = []
            rev_dists = []
            for seed in seeds[:5]:
                fwd_img = apply_mixed(images[seed], [(ct1, 0.5), (ct2, 0.5)])
                rev_img = apply_mixed(images[seed], [(ct2, 0.5), (ct1, 0.5)])
                fwd_emb = extract_hidden(model, processor, fwd_img, prompt)
                rev_emb = extract_hidden(model, processor, rev_img, prompt)
                fwd_dists.append(cosine_dist(fwd_emb, centroid))
                rev_dists.append(cosine_dist(rev_emb, centroid))

            key = f"{ct1}+{ct2}"
            order_sens[key] = {
                'forward_mean_dist': float(np.mean(fwd_dists)),
                'reverse_mean_dist': float(np.mean(rev_dists)),
                'order_matters': abs(np.mean(fwd_dists) - np.mean(rev_dists)) > threshold,
                'relative_diff': float(abs(np.mean(fwd_dists) - np.mean(rev_dists)) / max(np.mean(fwd_dists), 1e-10)),
            }
            print(f"  {key}: fwd={np.mean(fwd_dists):.6f}, rev={np.mean(rev_dists):.6f}, "
                  f"diff={order_sens[key]['relative_diff']:.4f}")

    results['order_sensitivity'] = order_sens

    # ========== 5. AUROC for Mixed Corruptions ==========
    print("\n=== Mixed Corruption AUROC ===")

    mixed_auroc = {}
    for key in pairwise:
        ct1, ct2 = key.split('+')
        mixed_dists = []
        for seed in seeds:
            mixed_img = apply_mixed(images[seed], [(ct1, 0.5), (ct2, 0.5)])
            emb = extract_hidden(model, processor, mixed_img, prompt)
            mixed_dists.append(cosine_dist(emb, centroid))

        auroc = compute_auroc(clean_dists, mixed_dists)
        mixed_auroc[key] = float(auroc)
        print(f"  {key}: AUROC={auroc:.4f}")

    results['mixed_auroc'] = mixed_auroc

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/mixed_corruption_{ts}.json"
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
