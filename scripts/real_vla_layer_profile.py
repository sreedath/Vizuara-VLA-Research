#!/usr/bin/env python3
"""Experiment 351: Layer-Depth Sensitivity Profile

Comprehensive layer analysis across ALL hidden layers:
1. Distance at each layer (0-32) for all corruption types
2. AUROC at each layer (multi-scene)
3. Layer-corruption interaction (which layers best separate which types?)
4. Attention token position: last token vs mean vs max pooling
5. Layer gradient: where does detection signal appear/peak/decay?
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden_all_layers(model, processor, image, prompt):
    """Extract hidden states from ALL layers."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    # Returns tuple of (n_layers+1,) tensors
    all_hidden = []
    for h in fwd.hidden_states:
        all_hidden.append(h[0, -1, :].float().cpu().numpy())  # last token
    return all_hidden

def extract_hidden_pooling(model, processor, image, prompt, layer=3):
    """Extract with different pooling strategies."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    h = fwd.hidden_states[layer][0].float().cpu().numpy()  # (seq_len, hidden_dim)
    return {
        'last': h[-1],
        'mean': h.mean(axis=0),
        'max': h.max(axis=0),
        'first': h[0],
    }

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

    # ========== 1. Full layer profile (single scene) ==========
    print("\n=== Full Layer Profile ===")

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    clean_layers = extract_hidden_all_layers(model, processor, base_img, prompt)
    n_layers = len(clean_layers)
    print(f"  Total layers: {n_layers}")

    layer_dists = {}
    for ct in ctypes:
        img = apply_corruption(base_img, ct, 0.5)
        corrupt_layers = extract_hidden_all_layers(model, processor, img, prompt)

        dists = []
        for layer_idx in range(n_layers):
            d = cosine_dist(clean_layers[layer_idx], corrupt_layers[layer_idx])
            dists.append(float(d))

        layer_dists[ct] = dists
        peak_layer = np.argmax(dists)
        print(f"  {ct}: peak at layer {peak_layer} (d={max(dists):.6f}), "
              f"first nonzero at L{next((i for i, d in enumerate(dists) if d > 0), 'none')}")

    results['layer_profile'] = {
        'n_layers': n_layers,
        'distances': layer_dists,
    }

    # ========== 2. Multi-scene AUROC per layer ==========
    print("\n=== Multi-Scene AUROC Per Layer ===")

    seeds = list(range(0, 1000, 100))[:10]
    # Select key layers to test
    test_layers = [0, 1, 2, 3, 4, 8, 16, 24, n_layers-2, n_layers-1]
    test_layers = [l for l in test_layers if l < n_layers]

    layer_auroc = {}
    for layer_idx in test_layers:
        # Collect calibration embeddings
        cal_embs = {}
        for seed in seeds:
            rng = np.random.RandomState(seed)
            px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(px)
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            cal_embs[seed] = fwd.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()

        # ID distances
        id_dists = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(px)
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            emb = fwd.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
            id_dists.append(float(cosine_dist(cal_embs[seed], emb)))

        per_type = {}
        for ct in ctypes:
            ood_dists = []
            for seed in seeds:
                rng = np.random.RandomState(seed)
                px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(px)
                corrupted = apply_corruption(img, ct, 0.5)
                inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    fwd = model(**inputs, output_hidden_states=True)
                emb = fwd.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
                ood_dists.append(float(cosine_dist(cal_embs[seed], emb)))

            auroc = compute_auroc(id_dists, ood_dists)
            per_type[ct] = {
                'auroc': float(auroc),
                'mean_dist': float(np.mean(ood_dists)),
            }

        layer_auroc[str(layer_idx)] = per_type
        auroc_str = ', '.join(f'{ct}={per_type[ct]["auroc"]:.3f}' for ct in ctypes)
        print(f"  Layer {layer_idx}: {auroc_str}")

    results['layer_auroc'] = layer_auroc

    # ========== 3. Pooling strategy comparison ==========
    print("\n=== Pooling Strategy Comparison ===")

    pooling_results = {}
    # Use layer 3 (default)
    clean_pool = extract_hidden_pooling(model, processor, base_img, prompt, layer=3)

    for ct in ctypes:
        img = apply_corruption(base_img, ct, 0.5)
        corrupt_pool = extract_hidden_pooling(model, processor, img, prompt, layer=3)

        pool_dists = {}
        for strategy in ['last', 'mean', 'max', 'first']:
            d = cosine_dist(clean_pool[strategy], corrupt_pool[strategy])
            pool_dists[strategy] = float(d)

        pooling_results[ct] = pool_dists
        print(f"  {ct}: last={pool_dists['last']:.6f}, mean={pool_dists['mean']:.6f}, "
              f"max={pool_dists['max']:.6f}, first={pool_dists['first']:.6f}")

    results['pooling'] = pooling_results

    # ========== 4. Layer gradient (rate of change) ==========
    print("\n=== Layer Gradient ===")

    gradient_results = {}
    for ct in ctypes:
        dists = layer_dists[ct]
        # Compute gradient: d[i+1] - d[i]
        grad = [dists[i+1] - dists[i] for i in range(len(dists)-1)]

        # Find layers with steepest increase
        peak_grad_idx = np.argmax(grad)
        max_grad = max(grad)

        # Find where signal first appears (d > 1e-6)
        first_nonzero = next((i for i, d in enumerate(dists) if d > 1e-6), n_layers)

        # Find where signal peaks
        peak_idx = np.argmax(dists)

        gradient_results[ct] = {
            'gradient': [float(g) for g in grad],
            'peak_gradient_layer': int(peak_grad_idx),
            'max_gradient': float(max_grad),
            'first_signal_layer': first_nonzero,
            'peak_signal_layer': int(peak_idx),
        }
        print(f"  {ct}: signal starts L{first_nonzero}, peaks L{peak_idx}, "
              f"steepest growth L{peak_grad_idx}")

    results['layer_gradient'] = gradient_results

    # ========== 5. Severity x Layer interaction ==========
    print("\n=== Severity x Layer Interaction ===")

    sev_layer = {}
    for sev in [0.1, 0.3, 0.5, 1.0]:
        for ct in ['fog', 'blur']:
            img = apply_corruption(base_img, ct, sev)
            corrupt_layers = extract_hidden_all_layers(model, processor, img, prompt)

            dists = []
            for layer_idx in range(n_layers):
                d = cosine_dist(clean_layers[layer_idx], corrupt_layers[layer_idx])
                dists.append(float(d))

            peak = int(np.argmax(dists))
            key = f"{ct}_sev{sev}"
            sev_layer[key] = {
                'distances': dists,
                'peak_layer': peak,
                'peak_dist': float(max(dists)),
            }
            print(f"  {key}: peak at L{peak} (d={max(dists):.6f})")

    results['severity_layer'] = sev_layer

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/layer_profile_{ts}.json"
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
