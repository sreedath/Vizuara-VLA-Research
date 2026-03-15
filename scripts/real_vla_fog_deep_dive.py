"""
Fog OOD Detection Deep Dive.

Fog was the only category to show overlap with ID in Exp 138.
This experiment performs a deep analysis:
1. Fog at varying opacity levels (10%-90%)
2. Per-layer analysis for fog specifically
3. Larger sample sizes for precise AUROC
4. Compare fog-specific centroid vs generic centroid

Experiment 139 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 32001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 32002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog(idx, opacity=0.5):
    """Highway with fog overlay at specified opacity."""
    rng = np.random.default_rng(idx * 32020 + int(opacity * 100))
    base = create_highway(idx + 60000)
    fog = np.full_like(base, 200)
    result = (base.astype(float) * (1-opacity) + fog.astype(float) * opacity)
    noise = rng.integers(-3, 4, result.shape, dtype=np.int16)
    return np.clip(result + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt, layer=-1):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()


def extract_multi_layer(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}


def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("FOG OOD DETECTION DEEP DIVE", flush=True)
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
    layers_to_check = [3, 8, 16, 24, 32]

    # Collect ID embeddings (larger set)
    print("\n--- Collecting ID embeddings ---", flush=True)
    id_embeds = {l: [] for l in layers_to_check}
    for idx in range(20):
        img = Image.fromarray(create_highway(idx + 3700))
        hs = extract_multi_layer(model, processor, img, prompt, layers_to_check)
        if hs:
            for l in layers_to_check:
                id_embeds[l].append(hs[l])
    for idx in range(20):
        img = Image.fromarray(create_urban(idx + 3700))
        hs = extract_multi_layer(model, processor, img, prompt, layers_to_check)
        if hs:
            for l in layers_to_check:
                id_embeds[l].append(hs[l])

    centroids = {l: np.mean(id_embeds[l], axis=0) for l in layers_to_check}
    print(f"  ID: {len(id_embeds[layers_to_check[0]])} samples", flush=True)

    # Fog at varying opacity levels
    opacity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n--- Fog at varying opacity ---", flush=True)
    fog_results = {}

    for opacity in opacity_levels:
        fog_embeds = {l: [] for l in layers_to_check}
        for idx in range(15):
            img = Image.fromarray(create_fog(idx + 3700, opacity))
            hs = extract_multi_layer(model, processor, img, prompt, layers_to_check)
            if hs:
                for l in layers_to_check:
                    fog_embeds[l].append(hs[l])

        layer_results = {}
        for l in layers_to_check:
            id_arr = np.array(id_embeds[l])
            fog_arr = np.array(fog_embeds[l])
            id_dists = np.array([cosine_distance(e, centroids[l]) for e in id_arr])
            fog_dists = np.array([cosine_distance(e, centroids[l]) for e in fog_arr])

            labels = np.array([0]*len(id_dists) + [1]*len(fog_dists))
            scores = np.concatenate([id_dists, fog_dists])
            auroc = float(roc_auc_score(labels, scores))
            d = float((np.mean(fog_dists) - np.mean(id_dists)) / (np.std(id_dists) + 1e-10))
            gap = float(np.min(fog_dists) - np.max(id_dists))

            layer_results[str(l)] = {
                'auroc': auroc,
                'd_prime': d,
                'fog_mean_dist': float(np.mean(fog_dists)),
                'fog_min_dist': float(np.min(fog_dists)),
                'gap': gap,
            }

        # Use last layer for summary
        summary = layer_results['32']
        print(f"  opacity={opacity:.1f}: AUROC={summary['auroc']:.4f}, d={summary['d_prime']:.2f}, "
              f"gap={summary['gap']:.4f}", flush=True)

        fog_results[str(opacity)] = {
            'opacity': opacity,
            'n_fog': len(fog_embeds[layers_to_check[0]]),
            'per_layer': layer_results,
        }

    # Find critical opacity (where AUROC drops below 1.0)
    print("\n--- Critical Opacity Analysis ---", flush=True)
    for opacity in opacity_levels:
        res = fog_results[str(opacity)]['per_layer']
        best_layer = max(layers_to_check, key=lambda l: res[str(l)]['d_prime'])
        best_auroc = res[str(best_layer)]['auroc']
        best_d = res[str(best_layer)]['d_prime']
        print(f"  opacity={opacity:.1f}: best=L{best_layer} AUROC={best_auroc:.4f} d={best_d:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    id_dists_final = np.array([cosine_distance(e, centroids[32]) for e in id_embeds[32]])
    output = {
        'experiment': 'fog_deep_dive',
        'experiment_number': 139,
        'timestamp': timestamp,
        'n_id': len(id_embeds[layers_to_check[0]]),
        'id_max_dist': float(np.max(id_dists_final)),
        'id_mean_dist': float(np.mean(id_dists_final)),
        'layers': layers_to_check,
        'fog_results': fog_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"fog_deep_dive_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
