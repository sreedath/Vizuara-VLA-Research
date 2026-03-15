"""
Distance Metric Comparison.

Comprehensive comparison of distance metrics for OOD detection:
1. Cosine distance (baseline)
2. L2 (Euclidean) distance
3. L1 (Manhattan) distance
4. Chebyshev (L-infinity) distance
5. Angular distance (arccos of cosine similarity)
6. Correlation distance
7. Bray-Curtis distance

Experiment 132 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 25001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 25002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 25003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 25004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 25010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 25014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def evaluate_metric(test_embeds, test_labels, centroid, metric_fn):
    scores = np.array([metric_fn(e, centroid) for e in test_embeds])
    auroc = float(roc_auc_score(test_labels, scores))
    id_s = scores[test_labels == 0]
    ood_s = scores[test_labels == 1]
    d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
    return auroc, d


def main():
    print("=" * 70, flush=True)
    print("DISTANCE METRIC COMPARISON", flush=True)
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

    print("\n--- Collecting embeddings ---", flush=True)
    cal_embeds = []
    test_embeds = []
    test_labels = []

    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(15):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 3000)), prompt)
            if h is not None:
                embeds.append(h)
        if group == 'ID':
            cal_embeds.extend(embeds[:8])
            for e in embeds[8:]:
                test_embeds.append(e)
                test_labels.append(0)
        else:
            for e in embeds:
                test_embeds.append(e)
                test_labels.append(1)

    cal_embeds = np.array(cal_embeds)
    test_embeds = np.array(test_embeds)
    test_labels = np.array(test_labels)
    centroid = np.mean(cal_embeds, axis=0)

    print(f"\nCal: {len(cal_embeds)}, Test: {len(test_labels)}", flush=True)

    # Define metrics
    metrics = {
        'cosine': lambda a, b: float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)),
        'euclidean': lambda a, b: float(np.linalg.norm(a - b)),
        'manhattan': lambda a, b: float(np.sum(np.abs(a - b))),
        'chebyshev': lambda a, b: float(np.max(np.abs(a - b))),
        'angular': lambda a, b: float(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10), -1, 1))),
        'correlation': lambda a, b: float(1 - np.corrcoef(a, b)[0, 1]),
        'bray_curtis': lambda a, b: float(np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + np.sum(np.abs(b)) + 1e-10)),
    }

    print("\n--- Distance Metric Comparison ---", flush=True)
    results = {}
    for name, fn in metrics.items():
        auroc, d = evaluate_metric(test_embeds, test_labels, centroid, fn)
        results[name] = {'auroc': auroc, 'd': d}
        print(f"  {name:15s}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Ranking
    print("\n--- Ranked by D-prime ---", flush=True)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['d'], reverse=True)
    for name, res in sorted_results:
        print(f"  {name:15s}: AUROC={res['auroc']:.4f}, d={res['d']:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'distance_metrics',
        'experiment_number': 132,
        'timestamp': timestamp,
        'n_cal': len(cal_embeds),
        'n_test': len(test_labels),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"distance_metrics_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
