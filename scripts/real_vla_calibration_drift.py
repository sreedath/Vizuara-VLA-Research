"""
Calibration Set Diversity Analysis.

Tests whether diversity within the calibration set matters: compare
centroids from (a) highway-only, (b) urban-only, (c) mixed highway+urban.
Also tests whether adding more diverse ID scenarios improves detection.

Experiment 91 in the CalibDrive series.
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
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 5014)
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
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("CALIBRATION SET DIVERSITY ANALYSIS", flush=True)
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

    # Build calibration pools
    print("\nBuilding calibration pools...", flush=True)
    highway_cal = []
    for i in range(20):
        h = extract_hidden(model, processor,
                           Image.fromarray(create_highway(i + 9000)), prompt)
        if h is not None:
            highway_cal.append(h)

    urban_cal = []
    for i in range(20):
        h = extract_hidden(model, processor,
                           Image.fromarray(create_urban(i + 9000)), prompt)
        if h is not None:
            urban_cal.append(h)

    print(f"  Highway pool: {len(highway_cal)}", flush=True)
    print(f"  Urban pool: {len(urban_cal)}", flush=True)

    # Test data
    print("\nCollecting test data...", flush=True)
    id_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                id_hidden.append(h)

    ood_hidden = []
    ood_labels = []
    for fn, name in [(create_noise, 'noise'), (create_indoor, 'indoor'),
                     (create_twilight_highway, 'twilight'), (create_snow, 'snow')]:
        for i in range(6):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                ood_hidden.append(h)
                ood_labels.append(name)

    print(f"  ID: {len(id_hidden)}, OOD: {len(ood_hidden)}", flush=True)

    # Test different calibration compositions
    cal_configs = {
        'highway_only': highway_cal[:15],
        'urban_only': urban_cal[:15],
        'mixed_15': highway_cal[:8] + urban_cal[:7],
        'mixed_30': highway_cal[:15] + urban_cal[:15],
        'highway_5': highway_cal[:5],
        'urban_5': urban_cal[:5],
        'mixed_5': highway_cal[:3] + urban_cal[:2],
    }

    results = {}
    for name, cal_set in cal_configs.items():
        centroid = np.mean(cal_set, axis=0)

        id_scores = [cosine_dist(h, centroid) for h in id_hidden]
        ood_scores = [cosine_dist(h, centroid) for h in ood_hidden]

        labels = [0]*len(id_scores) + [1]*len(ood_scores)
        scores = id_scores + ood_scores
        auroc = roc_auc_score(labels, scores)

        id_arr = np.array(id_scores)
        ood_arr = np.array(ood_scores)
        pooled = np.sqrt((id_arr.var() + ood_arr.var()) / 2)
        d = (ood_arr.mean() - id_arr.mean()) / (pooled + 1e-10)

        # Per-category AUROC
        cat_aurocs = {}
        for cat in set(ood_labels):
            cat_scores = [cosine_dist(h, centroid)
                          for h, l in zip(ood_hidden, ood_labels) if l == cat]
            cat_labels = [0]*len(id_scores) + [1]*len(cat_scores)
            cat_all = id_scores + cat_scores
            cat_aurocs[cat] = float(roc_auc_score(cat_labels, cat_all))

        results[name] = {
            'n_cal': len(cal_set),
            'auroc': float(auroc),
            'cohens_d': float(d),
            'id_mean': float(id_arr.mean()),
            'ood_mean': float(ood_arr.mean()),
            'per_category': cat_aurocs,
        }
        print(f"  {name:<15}: AUROC={auroc:.3f}, d={d:.2f}", flush=True)

    # Centroid similarity between configs
    centroids = {name: np.mean(cal_set, axis=0) for name, cal_set in cal_configs.items()}
    centroid_dists = {}
    for n1 in ['highway_only', 'urban_only', 'mixed_15']:
        for n2 in ['highway_only', 'urban_only', 'mixed_15']:
            if n1 < n2:
                d = cosine_dist(centroids[n1], centroids[n2])
                centroid_dists[f"{n1}_vs_{n2}"] = float(d)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'calibration_diversity',
        'experiment_number': 91,
        'timestamp': timestamp,
        'n_id': len(id_hidden),
        'n_ood': len(ood_hidden),
        'results': results,
        'centroid_distances': centroid_dists,
    }
    output_path = os.path.join(RESULTS_DIR, f"calibration_diversity_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
