"""
Dual-Layer OOD Detector.

Tests a dual-layer detection architecture that monitors both Layer 3
(photometric features) and Layer 32 (semantic features) simultaneously.
The detector flags OOD if EITHER layer exceeds its threshold.

Key question: Does the dual-layer detector resolve the fog vulnerability
while maintaining perfect detection on all other categories?

Experiment 140 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 33001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 33002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 33003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 33004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight(idx):
    rng = np.random.default_rng(idx * 33010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 33014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog(idx, opacity=0.5):
    rng = np.random.default_rng(idx * 33020 + int(opacity * 100))
    base = create_highway(idx + 70000)
    fog = np.full_like(base, 200)
    result = (base.astype(float) * (1-opacity) + fog.astype(float) * opacity)
    noise = rng.integers(-3, 4, result.shape, dtype=np.int16)
    return np.clip(result + noise, 0, 255).astype(np.uint8)

def create_rain(idx):
    rng = np.random.default_rng(idx * 33021)
    base = create_highway(idx + 80000)
    result = (base.astype(float) * 0.7).astype(np.uint8)
    for _ in range(50):
        x = rng.integers(0, SIZE[1])
        y0 = rng.integers(0, SIZE[0]//2)
        length = rng.integers(10, 40)
        y1 = min(y0 + length, SIZE[0])
        result[y0:y1, max(0,x-1):min(SIZE[1],x+1)] = np.clip(
            result[y0:y1, max(0,x-1):min(SIZE[1],x+1)].astype(int) + 80, 0, 255).astype(np.uint8)
    return result

def create_desert(idx):
    rng = np.random.default_rng(idx * 33024)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 180, 120]
    img[SIZE[0]//2:] = [180, 160, 100]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [220, 200, 140]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


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
    print("DUAL-LAYER OOD DETECTOR", flush=True)
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
    monitor_layers = [3, 32]

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight, 'OOD'),
        'snow': (create_snow, 'OOD'),
        'fog_30': (lambda idx: create_fog(idx, 0.3), 'OOD'),
        'fog_50': (lambda idx: create_fog(idx, 0.5), 'OOD'),
        'rain': (create_rain, 'OOD'),
        'desert': (create_desert, 'OOD'),
    }

    print("\n--- Collecting embeddings ---", flush=True)
    embeds_by_cat = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds_by_cat[cat_name] = {'group': group, 'embeds': {l: [] for l in monitor_layers}}
        for i in range(15):
            img = Image.fromarray(fn(i + 3800))
            hs = extract_multi_layer(model, processor, img, prompt, monitor_layers)
            if hs:
                for l in monitor_layers:
                    embeds_by_cat[cat_name]['embeds'][l].append(hs[l])

    # Compute centroids from ID
    id_embeds = {l: [] for l in monitor_layers}
    for cat_name, data in embeds_by_cat.items():
        if data['group'] == 'ID':
            for l in monitor_layers:
                id_embeds[l].extend(data['embeds'][l])

    centroids = {l: np.mean(id_embeds[l], axis=0) for l in monitor_layers}
    print(f"\nID embeddings: {len(id_embeds[monitor_layers[0]])}", flush=True)

    # Compute scores for each category and layer
    print("\n--- Per-Category Scores ---", flush=True)
    results = {}
    for cat_name, data in embeds_by_cat.items():
        layer_scores = {}
        for l in monitor_layers:
            dists = [cosine_distance(e, centroids[l]) for e in data['embeds'][l]]
            layer_scores[str(l)] = {
                'mean': float(np.mean(dists)),
                'std': float(np.std(dists)),
                'min': float(np.min(dists)),
                'max': float(np.max(dists)),
                'scores': dists,
            }
        results[cat_name] = {
            'group': data['group'],
            'n': len(data['embeds'][monitor_layers[0]]),
            'per_layer': layer_scores,
        }
        print(f"  {cat_name:12s} ({data['group']}): L3={layer_scores['3']['mean']:.4f}, L32={layer_scores['32']['mean']:.4f}", flush=True)

    # Evaluate detection methods
    print("\n--- Detection Comparison ---", flush=True)
    methods = {
        'L3_only': lambda scores: scores[3],
        'L32_only': lambda scores: scores[32],
        'max': lambda scores: max(scores[3], scores[32]),
        'mean': lambda scores: (scores[3] + scores[32]) / 2,
    }

    # Compute ID score statistics for thresholding
    id_scores = {l: [] for l in monitor_layers}
    for cat_name, data in results.items():
        if data['group'] == 'ID':
            for l in monitor_layers:
                id_scores[l].extend(data['per_layer'][str(l)]['scores'])

    detection_results = {}
    for method_name, score_fn in methods.items():
        all_scores = []
        all_labels = []
        for cat_name, data in results.items():
            for i in range(data['n']):
                s = {l: data['per_layer'][str(l)]['scores'][i] for l in monitor_layers}
                combined = score_fn(s)
                all_scores.append(combined)
                all_labels.append(0 if data['group'] == 'ID' else 1)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        auroc = float(roc_auc_score(all_labels, all_scores))
        id_s = all_scores[all_labels == 0]
        ood_s = all_scores[all_labels == 1]
        d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))

        detection_results[method_name] = {
            'auroc': auroc,
            'd_prime': d,
        }
        print(f"  {method_name:10s}: AUROC={auroc:.4f}, d={d:.1f}", flush=True)

    # Per-category AUROC for the dual (max) detector
    print("\n--- Per-Category AUROC (max detector) ---", flush=True)
    per_cat_auroc = {}
    for cat_name, data in results.items():
        if data['group'] == 'OOD':
            id_combined = []
            for i in range(len(id_scores[monitor_layers[0]])):
                s = {l: id_scores[l][i] for l in monitor_layers}
                id_combined.append(max(s[3], s[32]))

            ood_combined = []
            for i in range(data['n']):
                s = {l: data['per_layer'][str(l)]['scores'][i] for l in monitor_layers}
                ood_combined.append(max(s[3], s[32]))

            labels = np.array([0]*len(id_combined) + [1]*len(ood_combined))
            scores = np.array(id_combined + ood_combined)
            auroc = float(roc_auc_score(labels, scores))
            d = float((np.mean(ood_combined) - np.mean(id_combined)) / (np.std(id_combined) + 1e-10))

            per_cat_auroc[cat_name] = {
                'auroc': auroc,
                'd_prime': d,
            }
            print(f"  {cat_name:12s}: AUROC={auroc:.4f}, d={d:.1f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'dual_layer_detector',
        'experiment_number': 140,
        'timestamp': timestamp,
        'monitor_layers': monitor_layers,
        'detection_results': detection_results,
        'per_category_auroc': per_cat_auroc,
        'per_category_scores': {
            cat_name: {
                'group': data['group'],
                'L3_mean': data['per_layer']['3']['mean'],
                'L32_mean': data['per_layer']['32']['mean'],
            }
            for cat_name, data in results.items()
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"dual_layer_detector_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
