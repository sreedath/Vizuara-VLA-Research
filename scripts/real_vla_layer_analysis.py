"""
Layer-wise Hidden State Analysis for OOD Detection.

Investigates which transformer layers provide the best OOD detection signal
by extracting hidden states from multiple layers (early, middle, late) and
comparing their cosine distance AUROC.

Hypothesis: Later layers may encode more task-specific (action-relevant)
information while early layers encode more visual features. The optimal
layer for OOD detection may differ from the last layer.

Measures:
1. Per-layer cosine distance AUROC
2. Layer combination (early + late) performance
3. Inter-layer agreement on OOD ranking
4. Representational geometry (spread, isotropy) per layer

Experiment 50 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_multilayer(model, processor, image, prompt, layer_indices):
    """Extract hidden states from multiple specified layers."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    result = {}
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        # outputs.hidden_states is a tuple of (n_generated_tokens,)
        # Each element is a tuple of (n_layers+1,) tensors
        # We use the last generated token's hidden states
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            n_layers = len(last_step) - 1  # -1 for embedding layer
            for layer_idx in layer_indices:
                if layer_idx < len(last_step):
                    h = last_step[layer_idx][0, -1, :].float().cpu().numpy()
                    result[layer_idx] = h
            result['n_layers'] = n_layers
        else:
            result[0] = last_step[0, -1, :].float().cpu().numpy()
            result['n_layers'] = 1
    else:
        for layer_idx in layer_indices:
            result[layer_idx] = np.zeros(4096)
        result['n_layers'] = 0

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("LAYER-WISE HIDDEN STATE ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # First, determine the number of layers from a single forward pass
    print("\nProbing model architecture...", flush=True)
    test_img = Image.fromarray(create_highway(0))
    probe = extract_multilayer(model, processor, test_img, prompt, list(range(40)))
    n_layers = probe['n_layers']
    print(f"  Model has {n_layers} transformer layers (+1 embedding)", flush=True)

    # Select layers to analyze: embedding (0), early, quarter, middle, three-quarter, late, last
    if n_layers >= 32:
        layer_indices = [0, 1, 4, 8, 12, 16, 20, 24, 28, n_layers-2, n_layers-1, n_layers]
    else:
        step = max(1, n_layers // 8)
        layer_indices = list(range(0, n_layers + 1, step))
        if n_layers not in layer_indices:
            layer_indices.append(n_layers)
    layer_indices = sorted(set(layer_indices))
    print(f"  Analyzing layers: {layer_indices}", flush=True)

    # Calibration
    print("\nCalibration (15 per scene × 2 scenes)...", flush=True)
    cal_data = {}
    for layer_idx in layer_indices:
        cal_data[layer_idx] = []

    for fn in [create_highway, create_urban]:
        for i in range(15):
            data = extract_multilayer(model, processor,
                                      Image.fromarray(fn(i + 9000)), prompt,
                                      layer_indices)
            for layer_idx in layer_indices:
                if layer_idx in data:
                    cal_data[layer_idx].append(data[layer_idx])

    # Compute per-layer centroids
    centroids = {}
    for layer_idx in layer_indices:
        if cal_data[layer_idx]:
            centroids[layer_idx] = np.mean(cal_data[layer_idx], axis=0)
            print(f"  Layer {layer_idx}: centroid norm = {np.linalg.norm(centroids[layer_idx]):.2f}, "
                  f"dim = {centroids[layer_idx].shape[0]}", flush=True)

    # Test set
    print("\nTesting...", flush=True)
    test_fns = {
        'highway': (create_highway, False, 12),
        'urban': (create_urban, False, 12),
        'noise': (create_noise, True, 8),
        'indoor': (create_indoor, True, 8),
        'inverted': (create_inverted, True, 8),
        'blackout': (create_blackout, True, 8),
    }

    test_results = []
    total = sum(v[2] for v in test_fns.values())
    cnt = 0
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            data = extract_multilayer(model, processor,
                                      Image.fromarray(fn(i + 200)), prompt,
                                      layer_indices)
            entry = {
                'scenario': scene,
                'is_ood': is_ood,
            }
            for layer_idx in layer_indices:
                if layer_idx in data and layer_idx in centroids:
                    cos = cosine_dist(data[layer_idx], centroids[layer_idx])
                    norm = float(np.linalg.norm(data[layer_idx]))
                    entry[f'cos_L{layer_idx}'] = cos
                    entry[f'norm_L{layer_idx}'] = norm

            test_results.append(entry)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}_{i}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    easy = [r for r in test_results if not r['is_ood']]
    ood = [r for r in test_results if r['is_ood']]
    labels = [0]*len(easy) + [1]*len(ood)
    all_r = easy + ood

    # 1. Per-layer AUROC
    print("\n1. Per-Layer AUROC for OOD Detection", flush=True)
    print("-" * 60, flush=True)

    layer_aurocs = {}
    print(f"\n  {'Layer':>8} {'AUROC':>8} {'ID cos':>10} {'OOD cos':>10} {'Sep':>8}", flush=True)
    print("  " + "-" * 50, flush=True)

    for layer_idx in layer_indices:
        key = f'cos_L{layer_idx}'
        if key in all_r[0]:
            scores = [r[key] for r in all_r]
            try:
                auroc = roc_auc_score(labels, scores)
            except ValueError:
                auroc = 0.5
            id_mean = np.mean([r[key] for r in easy])
            ood_mean = np.mean([r[key] for r in ood])
            sep = ood_mean - id_mean
            layer_aurocs[layer_idx] = auroc
            print(f"  L{layer_idx:>6} {auroc:>8.3f} {id_mean:>10.4f} {ood_mean:>10.4f} "
                  f"{sep:>+8.4f}", flush=True)

    best_layer = max(layer_aurocs, key=layer_aurocs.get)
    print(f"\n  Best layer: L{best_layer} (AUROC = {layer_aurocs[best_layer]:.3f})", flush=True)

    # 2. Layer combination
    print("\n2. Layer Combinations", flush=True)
    print("-" * 60, flush=True)

    combos = {}
    # Try pairs: early + last, mid + last, best + last
    for other in layer_indices:
        if other == layer_indices[-1]:
            continue
        key_other = f'cos_L{other}'
        key_last = f'cos_L{layer_indices[-1]}'
        if key_other in all_r[0] and key_last in all_r[0]:
            for w in [0.3, 0.5, 0.7]:
                combined = [w * r[key_other] + (1-w) * r[key_last] for r in all_r]
                try:
                    auroc = roc_auc_score(labels, combined)
                except ValueError:
                    auroc = 0.5
                combo_name = f"L{other}({w:.1f})+L{layer_indices[-1]}({1-w:.1f})"
                combos[combo_name] = auroc

    # Sort by AUROC
    top_combos = sorted(combos.items(), key=lambda x: -x[1])[:10]
    print(f"\n  {'Combination':<40} {'AUROC':>8}", flush=True)
    print("  " + "-" * 50, flush=True)
    for name, auroc in top_combos:
        print(f"  {name:<40} {auroc:>8.3f}", flush=True)

    # 3. Representational geometry per layer
    print("\n3. Representational Geometry", flush=True)
    print("-" * 60, flush=True)

    print(f"\n  {'Layer':>8} {'ID norm':>10} {'OOD norm':>10} {'ID spread':>10} {'OOD spread':>10}",
          flush=True)
    print("  " + "-" * 55, flush=True)

    geom_data = {}
    for layer_idx in layer_indices:
        norm_key = f'norm_L{layer_idx}'
        if norm_key in all_r[0]:
            id_norms = [r[norm_key] for r in easy]
            ood_norms = [r[norm_key] for r in ood]
            # Spread = std of norms within group
            id_spread = np.std(id_norms)
            ood_spread = np.std(ood_norms)
            geom_data[layer_idx] = {
                'id_norm': float(np.mean(id_norms)),
                'ood_norm': float(np.mean(ood_norms)),
                'id_spread': float(id_spread),
                'ood_spread': float(ood_spread),
            }
            print(f"  L{layer_idx:>6} {np.mean(id_norms):>10.1f} {np.mean(ood_norms):>10.1f} "
                  f"{id_spread:>10.1f} {ood_spread:>10.1f}", flush=True)

    # 4. Inter-layer agreement
    print("\n4. Inter-Layer Agreement", flush=True)
    print("-" * 60, flush=True)

    # Rank correlation between layers
    if len(layer_aurocs) > 2:
        layer_list = sorted(layer_aurocs.keys())
        print(f"\n  Rank correlation of OOD scores between layers:", flush=True)
        for i, la in enumerate(layer_list):
            for lb in layer_list[i+1:]:
                ka = f'cos_L{la}'
                kb = f'cos_L{lb}'
                if ka in all_r[0] and kb in all_r[0]:
                    scores_a = [r[ka] for r in all_r]
                    scores_b = [r[kb] for r in all_r]
                    corr = np.corrcoef(scores_a, scores_b)[0, 1]
                    if abs(la - lb) <= 4 or la == layer_list[0] or la == layer_list[-1]:
                        print(f"    L{la} vs L{lb}: r = {corr:.3f}", flush=True)

    # 5. Per-scenario per-layer AUROC
    print("\n5. Per-Scenario Per-Layer AUROC", flush=True)
    print("-" * 60, flush=True)

    ood_types = ['noise', 'indoor', 'inverted', 'blackout']
    header = f"  {'Layer':>8}"
    for ood_type in ood_types:
        header += f" {ood_type:>10}"
    print(header, flush=True)
    print("  " + "-" * (10 + 11 * len(ood_types)), flush=True)

    for layer_idx in layer_indices:
        key = f'cos_L{layer_idx}'
        if key not in all_r[0]:
            continue
        row = f"  L{layer_idx:>6}"
        for ood_type in ood_types:
            type_ood = [r for r in ood if r['scenario'] == ood_type]
            type_labels = [0]*len(easy) + [1]*len(type_ood)
            type_all = easy + type_ood
            type_scores = [r[key] for r in type_all]
            try:
                auroc = roc_auc_score(type_labels, type_scores)
            except ValueError:
                auroc = 0.5
            row += f" {auroc:>10.3f}"
        print(row, flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  Total layers: {n_layers}", flush=True)
    print(f"  Layers analyzed: {len(layer_indices)}", flush=True)
    print(f"  Best single layer: L{best_layer} (AUROC = {layer_aurocs[best_layer]:.3f})", flush=True)
    print(f"  Last layer: L{layer_indices[-1]} (AUROC = {layer_aurocs.get(layer_indices[-1], 0):.3f})", flush=True)
    if top_combos:
        print(f"  Best combination: {top_combos[0][0]} (AUROC = {top_combos[0][1]:.3f})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'layer_analysis',
        'experiment_number': 50,
        'timestamp': timestamp,
        'n_layers': n_layers,
        'layers_analyzed': layer_indices,
        'n_cal': 30,
        'n_test': len(test_results),
        'layer_aurocs': {str(k): v for k, v in layer_aurocs.items()},
        'best_layer': best_layer,
        'top_combinations': [{
            'name': name, 'auroc': auroc,
        } for name, auroc in top_combos[:10]],
        'geometry': {str(k): v for k, v in geom_data.items()},
        'results': [{k: v for k, v in r.items()} for r in test_results],
    }
    output_path = os.path.join(RESULTS_DIR, f"layer_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
