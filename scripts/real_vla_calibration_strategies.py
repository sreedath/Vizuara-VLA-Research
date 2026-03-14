"""
Multi-Image Calibration Strategies.

Compares different ways to use multiple calibration images:
1. Centroid (current approach) — cosine distance to mean embedding
2. Nearest-neighbor — cosine distance to closest calibration point
3. Farthest-neighbor — cosine distance to farthest calibration point
4. Average-to-all — mean cosine distance to all calibration points
5. Per-class centroids — separate highway/urban centroids, take min distance
6. K-sigma envelope — flag if distance > mean + k*std of calibration distances

Experiment 125 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 18001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 18002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 18003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 18004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 18010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 18014)
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


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("MULTI-IMAGE CALIBRATION STRATEGIES", flush=True)
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

    N_CAL = 10
    N_TEST = 10

    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(N_CAL + N_TEST):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 2500)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}

    # Build cal/test sets
    cal_highway = embeddings['highway']['embeds'][:N_CAL]
    cal_urban = embeddings['urban']['embeds'][:N_CAL]
    cal_all = np.concatenate([cal_highway, cal_urban])

    test_embeds = []
    test_labels = []
    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            for e in data['embeds'][N_CAL:]:
                test_embeds.append(e)
                test_labels.append(0)
        else:
            for e in data['embeds']:
                test_embeds.append(e)
                test_labels.append(1)

    test_embeds = np.array(test_embeds)
    test_labels = np.array(test_labels)
    print(f"\nCal: {len(cal_all)}, Test: {len(test_labels)}", flush=True)

    # Strategy 1: Centroid
    print("\n--- Strategy 1: Centroid ---", flush=True)
    centroid = np.mean(cal_all, axis=0)
    s1_scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
    s1_auroc = float(roc_auc_score(test_labels, s1_scores))
    id1 = s1_scores[test_labels == 0]
    ood1 = s1_scores[test_labels == 1]
    s1_d = float((np.mean(ood1) - np.mean(id1)) / (np.std(id1) + 1e-10))
    print(f"  Centroid: AUROC={s1_auroc:.4f}, d={s1_d:.2f}", flush=True)

    # Strategy 2: Nearest neighbor
    print("\n--- Strategy 2: Nearest Neighbor ---", flush=True)
    s2_scores = np.array([
        min(cosine_dist(e, c) for c in cal_all)
        for e in test_embeds
    ])
    s2_auroc = float(roc_auc_score(test_labels, s2_scores))
    id2 = s2_scores[test_labels == 0]
    ood2 = s2_scores[test_labels == 1]
    s2_d = float((np.mean(ood2) - np.mean(id2)) / (np.std(id2) + 1e-10))
    print(f"  Nearest-Neighbor: AUROC={s2_auroc:.4f}, d={s2_d:.2f}", flush=True)

    # Strategy 3: Farthest neighbor
    print("\n--- Strategy 3: Farthest Neighbor ---", flush=True)
    s3_scores = np.array([
        max(cosine_dist(e, c) for c in cal_all)
        for e in test_embeds
    ])
    s3_auroc = float(roc_auc_score(test_labels, s3_scores))
    id3 = s3_scores[test_labels == 0]
    ood3 = s3_scores[test_labels == 1]
    s3_d = float((np.mean(ood3) - np.mean(id3)) / (np.std(id3) + 1e-10))
    print(f"  Farthest-Neighbor: AUROC={s3_auroc:.4f}, d={s3_d:.2f}", flush=True)

    # Strategy 4: Average to all
    print("\n--- Strategy 4: Average-to-All ---", flush=True)
    s4_scores = np.array([
        np.mean([cosine_dist(e, c) for c in cal_all])
        for e in test_embeds
    ])
    s4_auroc = float(roc_auc_score(test_labels, s4_scores))
    id4 = s4_scores[test_labels == 0]
    ood4 = s4_scores[test_labels == 1]
    s4_d = float((np.mean(ood4) - np.mean(id4)) / (np.std(id4) + 1e-10))
    print(f"  Average-to-All: AUROC={s4_auroc:.4f}, d={s4_d:.2f}", flush=True)

    # Strategy 5: Per-class centroids (min distance)
    print("\n--- Strategy 5: Per-Class Centroids ---", flush=True)
    centroid_hw = np.mean(cal_highway, axis=0)
    centroid_ur = np.mean(cal_urban, axis=0)
    s5_scores = np.array([
        min(cosine_dist(e, centroid_hw), cosine_dist(e, centroid_ur))
        for e in test_embeds
    ])
    s5_auroc = float(roc_auc_score(test_labels, s5_scores))
    id5 = s5_scores[test_labels == 0]
    ood5 = s5_scores[test_labels == 1]
    s5_d = float((np.mean(ood5) - np.mean(id5)) / (np.std(id5) + 1e-10))
    print(f"  Per-Class Centroids: AUROC={s5_auroc:.4f}, d={s5_d:.2f}", flush=True)

    # Strategy 6: K-nearest neighbors (k=3)
    print("\n--- Strategy 6: 3-NN Distance ---", flush=True)
    s6_scores = np.array([
        np.mean(sorted([cosine_dist(e, c) for c in cal_all])[:3])
        for e in test_embeds
    ])
    s6_auroc = float(roc_auc_score(test_labels, s6_scores))
    id6 = s6_scores[test_labels == 0]
    ood6 = s6_scores[test_labels == 1]
    s6_d = float((np.mean(ood6) - np.mean(id6)) / (np.std(id6) + 1e-10))
    print(f"  3-NN Distance: AUROC={s6_auroc:.4f}, d={s6_d:.2f}", flush=True)

    # Summary comparison
    print("\n--- Summary ---", flush=True)
    strategies = {
        'centroid': {'auroc': s1_auroc, 'd': s1_d},
        'nearest_neighbor': {'auroc': s2_auroc, 'd': s2_d},
        'farthest_neighbor': {'auroc': s3_auroc, 'd': s3_d},
        'average_to_all': {'auroc': s4_auroc, 'd': s4_d},
        'per_class_centroid': {'auroc': s5_auroc, 'd': s5_d},
        '3nn_distance': {'auroc': s6_auroc, 'd': s6_d},
    }
    for name, res in strategies.items():
        print(f"  {name:25s}: AUROC={res['auroc']:.4f}, d={res['d']:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'calibration_strategies',
        'experiment_number': 125,
        'timestamp': timestamp,
        'n_cal': len(cal_all),
        'n_test': len(test_labels),
        'strategies': strategies,
    }
    output_path = os.path.join(RESULTS_DIR, f"calibration_strategies_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
