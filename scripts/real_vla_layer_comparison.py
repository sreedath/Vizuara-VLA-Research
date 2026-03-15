#!/usr/bin/env python3
"""Experiment 333: Comprehensive Layer Profiling (Real OpenVLA-7B)

Deep analysis across ALL layers to characterize:
1. Per-layer detection AUROC for each corruption type
2. Per-layer sensitivity ranking
3. Layer-to-layer embedding similarity
4. Signal amplification profile
5. Optimal layer selection strategy
6. Layer ensemble analysis
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden_all_layers(model, processor, image, prompt, layers=None):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if layers is None:
        layers = list(range(len(fwd.hidden_states)))
    result = {}
    for l in layers:
        if l < len(fwd.hidden_states):
            result[l] = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()
    return result

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

    # Sample layers across the full range (33 layers: 0-32)
    test_layers = [0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28, 31, 32]

    # Create images
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    # Also test with a second scene for cross-scene analysis
    rng2 = np.random.RandomState(99)
    pixels2 = rng2.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    scene2_img = Image.fromarray(pixels2)

    ctypes = ['fog', 'night', 'noise', 'blur']
    sevs = [0.25, 0.5, 1.0]

    # ========== 1. Collect all-layer embeddings ==========
    print("\n=== Collecting All-Layer Embeddings ===")

    clean1_all = extract_hidden_all_layers(model, processor, base_img, prompt, test_layers)
    clean2_all = extract_hidden_all_layers(model, processor, scene2_img, prompt, test_layers)

    corrupt_all = {}
    for ct in ctypes:
        for sev in sevs:
            img = apply_corruption(base_img, ct, sev)
            embs = extract_hidden_all_layers(model, processor, img, prompt, test_layers)
            corrupt_all[f"{ct}_{sev}"] = embs
            print(f"  Collected {ct}@{sev}")

    # Also for scene 2
    corrupt2_all = {}
    for ct in ctypes:
        img = apply_corruption(scene2_img, ct, 0.5)
        embs = extract_hidden_all_layers(model, processor, img, prompt, test_layers)
        corrupt2_all[ct] = embs

    # ========== 2. Per-layer AUROC ==========
    print("\n=== Per-Layer AUROC ===")
    layer_auroc = {}

    for layer in test_layers:
        clean_d = cosine_dist(clean1_all[layer], clean2_all[layer])
        per_ct_auroc = {}
        for ct in ctypes:
            ood_dists = []
            for sev in sevs:
                d = cosine_dist(clean1_all[layer], corrupt_all[f"{ct}_{sev}"][layer])
                ood_dists.append(float(d))
            auroc = compute_auroc([clean_d], ood_dists)
            per_ct_auroc[ct] = float(auroc)

        # Overall AUROC
        all_ood = []
        for ct in ctypes:
            for sev in sevs:
                all_ood.append(float(cosine_dist(clean1_all[layer], corrupt_all[f"{ct}_{sev}"][layer])))
        overall_auroc = compute_auroc([clean_d], all_ood)

        layer_auroc[str(layer)] = {
            'per_corruption': per_ct_auroc,
            'overall': float(overall_auroc),
            'clean_dist': float(clean_d),
        }
        print(f"  L{layer}: AUROC={overall_auroc:.3f}, clean_d={clean_d:.6f}, per_ct={per_ct_auroc}")

    results['layer_auroc'] = layer_auroc

    # ========== 3. Signal amplification ==========
    print("\n=== Signal Amplification Profile ===")
    amplification = {}

    for ct in ctypes:
        dists_by_layer = {}
        for layer in test_layers:
            d = cosine_dist(clean1_all[layer], corrupt_all[f"{ct}_1.0"][layer])
            dists_by_layer[str(layer)] = float(d)

        # Amplification relative to L0
        if dists_by_layer.get('0', 0) > 0:
            amp = {k: v / dists_by_layer['0'] for k, v in dists_by_layer.items()}
        else:
            amp = {k: 0.0 for k in dists_by_layer}

        amplification[ct] = {
            'distances': dists_by_layer,
            'amplification_vs_L0': amp,
        }
        print(f"  {ct}: L0={dists_by_layer.get('0', 0):.6f} → L32={dists_by_layer.get('32', 0):.6f}")

    results['amplification'] = amplification

    # ========== 4. Layer-to-layer similarity ==========
    print("\n=== Layer Similarity (for fog@1.0 shift) ===")
    layer_sim = {}

    # Compute shift vectors per layer
    shift_vectors = {}
    for layer in test_layers:
        shift_vectors[layer] = corrupt_all["fog_1.0"][layer] - clean1_all[layer]

    for i, l1 in enumerate(test_layers):
        for l2 in test_layers[i+1:]:
            sim = cosine_sim(shift_vectors[l1], shift_vectors[l2])
            layer_sim[f"L{l1}_L{l2}"] = float(sim)

    results['layer_shift_similarity'] = layer_sim

    # ========== 5. Cross-scene per-layer ==========
    print("\n=== Cross-Scene Per-Layer ===")
    cross_scene = {}

    for layer in test_layers:
        # Direction consistency: is shift vector same for both scenes?
        shift1 = corrupt_all["fog_1.0"][layer] - clean1_all[layer]
        shift2 = corrupt2_all["fog"][layer] - clean2_all[layer]
        sim = cosine_sim(shift1, shift2)
        cross_scene[str(layer)] = {
            'direction_sim': float(sim),
            'scene1_dist': float(cosine_dist(clean1_all[layer], corrupt_all["fog_1.0"][layer])),
            'scene2_dist': float(cosine_dist(clean2_all[layer], corrupt2_all["fog"][layer])),
        }
        print(f"  L{layer}: dir_sim={sim:.4f}")

    results['cross_scene_layers'] = cross_scene

    # ========== 6. Embedding norm profile ==========
    print("\n=== Embedding Norm Profile ===")
    norm_profile = {}

    for layer in test_layers:
        clean_norm = float(np.linalg.norm(clean1_all[layer]))
        norms = {'clean': clean_norm}
        for ct in ctypes:
            corrupt_norm = float(np.linalg.norm(corrupt_all[f"{ct}_1.0"][layer]))
            norms[ct] = corrupt_norm
            norms[f"{ct}_change_pct"] = float((corrupt_norm - clean_norm) / clean_norm * 100)

        norm_profile[str(layer)] = norms
        print(f"  L{layer}: clean_norm={clean_norm:.2f}, fog_change={norms.get('fog_change_pct', 0):.2f}%")

    results['norm_profile'] = norm_profile

    # ========== 7. Optimal layer strategy ==========
    print("\n=== Optimal Layer Strategy ===")
    # Compute margin (min OOD - max clean) per layer
    strategy_results = {}
    for layer in test_layers:
        clean_d = cosine_dist(clean1_all[layer], clean2_all[layer])
        min_ood = float('inf')
        max_ood = 0
        for ct in ctypes:
            for sev in sevs:
                d = cosine_dist(clean1_all[layer], corrupt_all[f"{ct}_{sev}"][layer])
                min_ood = min(min_ood, d)
                max_ood = max(max_ood, d)

        margin = min_ood - clean_d
        strategy_results[str(layer)] = {
            'clean_dist': float(clean_d),
            'min_ood': float(min_ood),
            'max_ood': float(max_ood),
            'margin': float(margin),
            'margin_ratio': float(min_ood / clean_d) if clean_d > 0 else float('inf'),
        }
        print(f"  L{layer}: margin={margin:.6f}, ratio={strategy_results[str(layer)]['margin_ratio']:.1f}x")

    results['layer_strategy'] = strategy_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/layer_comparison_{ts}.json"
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
