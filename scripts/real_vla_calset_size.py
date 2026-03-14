"""
Calibration Set Size Sensitivity.

How few calibration samples does cosine distance need to be effective?
Tests: 1, 2, 3, 5, 8, 10, 15, 20, 30 calibration samples.

For each size, we repeat 5 times with different random subsets and
report mean/std AUROC. This establishes the minimum viable calibration
set for practical deployment.

Experiment 70 in the CalibDrive series.
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

# OOD
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

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


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
    print("CALIBRATION SET SIZE SENSITIVITY", flush=True)
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

    # Collect a large calibration pool (40 samples — 20 per scene type)
    print("\nCollecting calibration pool...", flush=True)
    cal_pool = []
    for fn in [create_highway, create_urban]:
        for i in range(20):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 8000)), prompt)
            if h is not None:
                cal_pool.append(h)
    print(f"  Pool size: {len(cal_pool)}", flush=True)

    # Collect test data
    print("\nCollecting test data...", flush=True)
    test_fns = {
        'highway': (create_highway, False, 10),
        'urban': (create_urban, False, 10),
        'noise': (create_noise, True, 8),
        'indoor': (create_indoor, True, 8),
        'twilight': (create_twilight_highway, True, 8),
        'blackout': (create_blackout, True, 6),
    }

    test_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                test_data.append({'hidden': h, 'is_ood': is_ood, 'scenario': scene})
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    print(f"  Test samples: {len(test_data)}", flush=True)

    # Test different calibration sizes
    cal_sizes = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    n_repeats = 5

    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    results = {}
    rng = np.random.default_rng(42)

    for cal_n in cal_sizes:
        if cal_n > len(cal_pool):
            continue

        repeat_aurocs = []
        repeat_aurocs_far = []
        repeat_aurocs_near = []

        for rep in range(n_repeats):
            # Random subset of calibration pool
            indices = rng.choice(len(cal_pool), size=cal_n, replace=False)
            cal_subset = [cal_pool[i] for i in indices]
            centroid = np.mean(cal_subset, axis=0)

            # Compute cosine distances
            labels = []
            scores = []
            labels_far = []
            scores_far = []
            labels_near = []
            scores_near = []

            for d in test_data:
                cos = cosine_dist(d['hidden'], centroid)
                labels.append(1 if d['is_ood'] else 0)
                scores.append(cos)

                if d['scenario'] in ['noise', 'blackout']:
                    labels_far.append(1)
                    scores_far.append(cos)
                elif d['scenario'] == 'twilight':
                    labels_near.append(1)
                    scores_near.append(cos)
                elif not d['is_ood']:
                    labels_far.append(0)
                    scores_far.append(cos)
                    labels_near.append(0)
                    scores_near.append(cos)

            auroc = roc_auc_score(labels, scores)
            repeat_aurocs.append(auroc)

            if len(set(labels_far)) > 1:
                repeat_aurocs_far.append(roc_auc_score(labels_far, scores_far))
            if len(set(labels_near)) > 1:
                repeat_aurocs_near.append(roc_auc_score(labels_near, scores_near))

        results[cal_n] = {
            'all_mean': float(np.mean(repeat_aurocs)),
            'all_std': float(np.std(repeat_aurocs)),
            'all_min': float(np.min(repeat_aurocs)),
            'all_max': float(np.max(repeat_aurocs)),
            'far_mean': float(np.mean(repeat_aurocs_far)) if repeat_aurocs_far else None,
            'far_std': float(np.std(repeat_aurocs_far)) if repeat_aurocs_far else None,
            'near_mean': float(np.mean(repeat_aurocs_near)) if repeat_aurocs_near else None,
            'near_std': float(np.std(repeat_aurocs_near)) if repeat_aurocs_near else None,
        }
        print(f"  N={cal_n:>3}: all={np.mean(repeat_aurocs):.3f}±{np.std(repeat_aurocs):.3f} "
              f"[{np.min(repeat_aurocs):.3f}, {np.max(repeat_aurocs):.3f}]  "
              f"far={np.mean(repeat_aurocs_far):.3f}  "
              f"near={np.mean(repeat_aurocs_near):.3f}",
              flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'calset_size',
        'experiment_number': 70,
        'timestamp': timestamp,
        'n_cal_pool': len(cal_pool),
        'n_test': len(test_data),
        'n_repeats': n_repeats,
        'cal_sizes': cal_sizes,
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"calset_size_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
