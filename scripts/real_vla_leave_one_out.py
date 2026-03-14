"""
Leave-One-Out OOD Generalization.

Tests whether calibration generalizes to completely unseen OOD types.
For each OOD category, train on all other categories and test on
the held-out category. If detection works, the method generalizes
to novel OOD types not seen during calibration.

Experiment 79 in the CalibDrive series.
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

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)

def create_inverted(idx):
    img = create_highway(idx + 3000)
    return (255 - img).astype(np.uint8)


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
    print("LEAVE-ONE-OUT OOD GENERALIZATION", flush=True)
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

    # Calibrate
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            if h is not None:
                cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  {len(cal_hidden)} samples", flush=True)

    # Collect all test data
    ood_fns = {
        'noise': (create_noise, 8),
        'indoor': (create_indoor, 8),
        'twilight': (create_twilight_highway, 8),
        'snow': (create_snow, 8),
        'blackout': (create_blackout, 6),
        'inverted': (create_inverted, 8),
    }

    # ID test data
    id_test = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                id_test.append({'hidden': h, 'scenario': 'id'})

    # OOD test data by category
    ood_by_cat = {}
    cnt = 0
    total = sum(v[1] for v in ood_fns.values())
    for cat, (fn, n) in ood_fns.items():
        ood_by_cat[cat] = []
        for i in range(n):
            cnt += 1
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                ood_by_cat[cat].append({'hidden': h, 'scenario': cat})
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {cat}", flush=True)

    print(f"\n  ID: {len(id_test)} samples", flush=True)
    for cat, data in ood_by_cat.items():
        print(f"  {cat}: {len(data)} samples", flush=True)

    # Leave-one-out: for each OOD category, test detection
    print("\n" + "=" * 70, flush=True)
    print("LEAVE-ONE-OUT RESULTS", flush=True)
    print("=" * 70, flush=True)

    results = {}
    for held_out in ood_by_cat:
        # Test: ID vs held-out OOD category
        labels = [0]*len(id_test) + [1]*len(ood_by_cat[held_out])
        scores = ([cosine_dist(d['hidden'], centroid) for d in id_test] +
                  [cosine_dist(d['hidden'], centroid) for d in ood_by_cat[held_out]])

        auroc = roc_auc_score(labels, scores)
        results[held_out] = {
            'auroc': float(auroc),
            'n_ood': len(ood_by_cat[held_out]),
        }
        print(f"  {held_out:<12}: AUROC={auroc:.3f} (n={len(ood_by_cat[held_out])})", flush=True)

    # All OOD combined
    all_ood = []
    for data in ood_by_cat.values():
        all_ood.extend(data)
    labels_all = [0]*len(id_test) + [1]*len(all_ood)
    scores_all = ([cosine_dist(d['hidden'], centroid) for d in id_test] +
                  [cosine_dist(d['hidden'], centroid) for d in all_ood])
    auroc_all = roc_auc_score(labels_all, scores_all)
    results['all_combined'] = {'auroc': float(auroc_all), 'n_ood': len(all_ood)}
    print(f"  {'ALL':<12}: AUROC={auroc_all:.3f} (n={len(all_ood)})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'leave_one_out',
        'experiment_number': 79,
        'timestamp': timestamp,
        'n_id': len(id_test),
        'n_ood_categories': len(ood_by_cat),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"leave_one_out_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
