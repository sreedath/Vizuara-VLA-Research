"""
Layer-wise Hidden State Analysis.

Extracts hidden states from every transformer layer (0-32) and measures
OOD detection performance at each layer to determine where the OOD
signal emerges, peaks, and how it evolves through the network.

Experiment 114 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 8001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 8002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 8003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 8004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 8010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 8014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


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

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    # Extract ALL layer hidden states for each image
    print("\n--- Collecting all-layer hidden states ---", flush=True)
    all_layer_data = {}  # cat -> list of {layer_idx: hidden_vec}

    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        cat_samples = []
        for i in range(10):
            img = Image.fromarray(fn(i + 1500))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)

            if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
                continue

            # Extract last-token hidden from each layer
            layer_hiddens = {}
            for layer_idx, hs in enumerate(fwd.hidden_states):
                layer_hiddens[layer_idx] = hs[0, -1, :].float().cpu().numpy()

            cat_samples.append(layer_hiddens)

        all_layer_data[cat_name] = {'samples': cat_samples, 'group': group}
        print(f"    {len(cat_samples)} samples, {len(cat_samples[0])} layers", flush=True)

    n_layers = len(all_layer_data['highway']['samples'][0])
    print(f"\nTotal layers: {n_layers}", flush=True)

    # Per-layer evaluation
    print("\n--- Per-layer OOD detection ---", flush=True)
    layer_results = {}

    for layer_idx in range(n_layers):
        # Build calibration and test sets
        cal_embeds = []
        test_embeds = []
        test_labels = []

        for cat_name, data in all_layer_data.items():
            for s_idx, sample in enumerate(data['samples']):
                vec = sample[layer_idx]
                if data['group'] == 'ID':
                    if s_idx < 5:
                        cal_embeds.append(vec)
                    else:
                        test_embeds.append(vec)
                        test_labels.append(0)
                else:
                    test_embeds.append(vec)
                    test_labels.append(1)

        cal_embeds = np.array(cal_embeds)
        test_embeds = np.array(test_embeds)
        test_labels = np.array(test_labels)

        centroid = np.mean(cal_embeds, axis=0)
        scores = np.array([cosine_dist(e, centroid) for e in test_embeds])

        id_scores = scores[test_labels == 0]
        ood_scores = scores[test_labels == 1]

        auroc = float(roc_auc_score(test_labels, scores))
        d = float((np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10))

        # Layer-specific stats
        norm_mean = float(np.mean([np.linalg.norm(sample[layer_idx])
                                    for data in all_layer_data.values()
                                    for sample in data['samples']]))

        layer_results[str(layer_idx)] = {
            'layer': layer_idx,
            'auroc': auroc,
            'd': d,
            'id_score_mean': float(np.mean(id_scores)),
            'id_score_std': float(np.std(id_scores)),
            'ood_score_mean': float(np.mean(ood_scores)),
            'ood_score_std': float(np.std(ood_scores)),
            'mean_norm': norm_mean,
            'dim': int(cal_embeds.shape[1]),
        }

        if layer_idx % 4 == 0 or layer_idx == n_layers - 1:
            print(f"  Layer {layer_idx:2d}: AUROC={auroc:.4f}, d={d:.2f}, "
                  f"norm={norm_mean:.1f}, dim={cal_embeds.shape[1]}", flush=True)

    # Per-category per-layer analysis for hardest OOD category
    print("\n--- Per-category at selected layers ---", flush=True)
    per_cat_layers = {}
    for layer_idx in [0, 8, 16, 24, n_layers - 1]:
        cal_embeds = []
        for cat_name, data in all_layer_data.items():
            if data['group'] == 'ID':
                for s in data['samples'][:5]:
                    cal_embeds.append(s[layer_idx])
        centroid = np.mean(cal_embeds, axis=0)

        cat_scores = {}
        for cat_name, data in all_layer_data.items():
            scores = [cosine_dist(s[layer_idx], centroid) for s in data['samples']]
            cat_scores[cat_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'group': data['group'],
            }
        per_cat_layers[str(layer_idx)] = cat_scores
        print(f"  Layer {layer_idx}: " +
              ", ".join(f"{c}={v['mean']:.3f}" for c, v in cat_scores.items()), flush=True)

    # Multi-layer fusion
    print("\n--- Multi-layer fusion ---", flush=True)
    fusion_results = {}

    # Concatenate selected layers
    for layer_combo_name, layer_combo in [
        ('last_only', [n_layers - 1]),
        ('last_two', [n_layers - 2, n_layers - 1]),
        ('last_four', list(range(n_layers - 4, n_layers))),
        ('every_4th', list(range(0, n_layers, 4))),
        ('all_layers', list(range(n_layers))),
    ]:
        cal_embeds = []
        test_embeds = []
        test_labels = []

        for cat_name, data in all_layer_data.items():
            for s_idx, sample in enumerate(data['samples']):
                vec = np.concatenate([sample[l] for l in layer_combo])
                if data['group'] == 'ID':
                    if s_idx < 5:
                        cal_embeds.append(vec)
                    else:
                        test_embeds.append(vec)
                        test_labels.append(0)
                else:
                    test_embeds.append(vec)
                    test_labels.append(1)

        cal_embeds = np.array(cal_embeds)
        test_embeds = np.array(test_embeds)
        test_labels = np.array(test_labels)

        centroid = np.mean(cal_embeds, axis=0)
        scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
        id_scores = scores[test_labels == 0]
        ood_scores = scores[test_labels == 1]

        auroc = float(roc_auc_score(test_labels, scores))
        d = float((np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10))

        fusion_results[layer_combo_name] = {
            'layers': layer_combo,
            'n_layers': len(layer_combo),
            'total_dim': int(cal_embeds.shape[1]),
            'auroc': auroc,
            'd': d,
        }
        print(f"  {layer_combo_name} ({len(layer_combo)} layers, {cal_embeds.shape[1]}d): "
              f"AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'layer_analysis',
        'experiment_number': 114,
        'timestamp': timestamp,
        'n_layers': n_layers,
        'layer_results': layer_results,
        'per_category_layers': per_cat_layers,
        'fusion_results': fusion_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"layer_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
