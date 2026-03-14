"""
Multi-Layer Embedding Fusion for OOD Detection.

Tests whether combining embeddings from multiple layers improves
detection over single last-layer. Strategies:
1. Single layer baselines
2. Concatenation + cosine distance
3. Average embeddings across layers
4. Max score across layers

Experiment 128 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 21001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 21002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 21003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 21004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 21010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 21014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_multi_layer(model, processor, image, prompt, layer_indices):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    result = {}
    for li in layer_indices:
        if li < len(fwd.hidden_states):
            result[li] = fwd.hidden_states[li][0, -1, :].float().cpu().numpy()
    return result


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def eval_detector(test_embeds, test_labels, cal_embeds):
    centroid = np.mean(cal_embeds, axis=0)
    scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
    auroc = float(roc_auc_score(test_labels, scores))
    id_s = scores[test_labels == 0]
    ood_s = scores[test_labels == 1]
    d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
    return auroc, d


def main():
    print("=" * 70, flush=True)
    print("MULTI-LAYER EMBEDDING FUSION", flush=True)
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
    layer_indices = [3, 8, 16, 24, 28, 32]

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    N_CAL = 8
    N_TEST = 7

    print("\n--- Collecting multi-layer embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        cat_data = []
        for i in range(N_CAL + N_TEST):
            ml = extract_multi_layer(model, processor, Image.fromarray(fn(i + 2800)), prompt, layer_indices)
            if ml:
                cat_data.append(ml)
        embeddings[cat_name] = {'data': cat_data, 'group': group}

    cal_per_layer = {li: [] for li in layer_indices}
    test_per_layer = {li: [] for li in layer_indices}
    test_labels = []

    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            for i, ml in enumerate(data['data']):
                if i < N_CAL:
                    for li in layer_indices:
                        cal_per_layer[li].append(ml[li])
                else:
                    for li in layer_indices:
                        test_per_layer[li].append(ml[li])
                    test_labels.append(0)
        else:
            for ml in data['data']:
                for li in layer_indices:
                    test_per_layer[li].append(ml[li])
                test_labels.append(1)

    for li in layer_indices:
        cal_per_layer[li] = np.array(cal_per_layer[li])
        test_per_layer[li] = np.array(test_per_layer[li])
    test_labels = np.array(test_labels)

    print(f"\nCal: {len(cal_per_layer[layer_indices[0]])}, Test: {len(test_labels)}", flush=True)

    # Strategy 1: Single layer baselines
    print("\n--- Single Layer ---", flush=True)
    single_results = {}
    for li in layer_indices:
        auroc, d = eval_detector(test_per_layer[li], test_labels, cal_per_layer[li])
        single_results[li] = {'auroc': auroc, 'd': d}
        print(f"  Layer {li:2d}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Strategy 2: Concatenation
    print("\n--- Concatenation ---", flush=True)
    concat_results = {}
    for layers in [[28, 32], [16, 28, 32], [3, 16, 28, 32], [3, 8, 16, 24, 28, 32]]:
        label = "+".join(str(l) for l in layers)
        cal_concat = np.concatenate([cal_per_layer[li] for li in layers], axis=1)
        test_concat = np.concatenate([test_per_layer[li] for li in layers], axis=1)
        auroc, d = eval_detector(test_concat, test_labels, cal_concat)
        concat_results[label] = {'auroc': auroc, 'd': d, 'dims': int(cal_concat.shape[1])}
        print(f"  [{label}] ({cal_concat.shape[1]}d): AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Strategy 3: Average across layers
    print("\n--- Layer Average ---", flush=True)
    avg_results = {}
    for layers in [[28, 32], [16, 28, 32], [3, 16, 28, 32], [3, 8, 16, 24, 28, 32]]:
        label = "avg(" + "+".join(str(l) for l in layers) + ")"
        cal_avg = np.mean([cal_per_layer[li] for li in layers], axis=0)
        test_avg = np.mean([test_per_layer[li] for li in layers], axis=0)
        auroc, d = eval_detector(test_avg, test_labels, cal_avg)
        avg_results[label] = {'auroc': auroc, 'd': d}
        print(f"  {label}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Strategy 4: Max score across layers
    print("\n--- Max Score ---", flush=True)
    max_results = {}
    for layers in [[28, 32], [16, 28, 32], [3, 16, 28, 32]]:
        label = "max(" + "+".join(str(l) for l in layers) + ")"
        centroids = {li: np.mean(cal_per_layer[li], axis=0) for li in layers}
        scores = []
        for idx in range(len(test_labels)):
            layer_scores = [cosine_dist(test_per_layer[li][idx], centroids[li]) for li in layers]
            scores.append(max(layer_scores))
        scores = np.array(scores)
        auroc = float(roc_auc_score(test_labels, scores))
        id_s = scores[test_labels == 0]
        ood_s = scores[test_labels == 1]
        d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
        max_results[label] = {'auroc': auroc, 'd': d}
        print(f"  {label}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Ranking
    print("\n--- Ranked by D-prime ---", flush=True)
    all_methods = {}
    for li, res in single_results.items():
        all_methods[f"layer_{li}"] = res
    all_methods.update(concat_results)
    all_methods.update(avg_results)
    all_methods.update(max_results)

    sorted_methods = sorted(all_methods.items(), key=lambda x: x[1]['d'], reverse=True)
    for name, res in sorted_methods[:10]:
        print(f"  {name:30s}: AUROC={res['auroc']:.4f}, d={res['d']:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'multi_layer_fusion',
        'experiment_number': 128,
        'timestamp': timestamp,
        'layer_indices': layer_indices,
        'single_layer': {str(k): v for k, v in single_results.items()},
        'concatenation': concat_results,
        'average': avg_results,
        'max_score': max_results,
        'ranking': [(name, res) for name, res in sorted_methods[:10]],
    }
    output_path = os.path.join(RESULTS_DIR, f"multi_layer_fusion_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
