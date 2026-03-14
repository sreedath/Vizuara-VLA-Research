"""
Mixed ID/OOD Batch Detection.

Tests whether the detector correctly identifies individual OOD
images when they appear within sequences of mostly-ID images,
simulating real deployment where OOD inputs are rare events.

Also measures detection at different OOD contamination rates
(1%, 5%, 10%, 25%, 50%) to map the phase transition behavior.

Experiment 86 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

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
    print("MIXED BATCH DETECTION", flush=True)
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
        for i in range(15):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            if h is not None:
                cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    cal_dists = [cosine_dist(h, centroid) for h in cal_hidden]
    threshold = np.mean(cal_dists) + 3 * np.std(cal_dists)
    print(f"  {len(cal_hidden)} calibration samples", flush=True)
    print(f"  Threshold (μ+3σ): {threshold:.4f}", flush=True)

    # Pre-extract a pool of ID and OOD hidden states
    print("\nBuilding sample pool...", flush=True)
    id_pool = []
    for fn in [create_highway, create_urban]:
        for i in range(30):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 2000)), prompt)
            if h is not None:
                id_pool.append(h)

    ood_pool = []
    ood_fns = [create_noise, create_indoor, create_snow]
    for fn in ood_fns:
        for i in range(10):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 2000)), prompt)
            if h is not None:
                ood_pool.append(h)

    print(f"  ID pool: {len(id_pool)}, OOD pool: {len(ood_pool)}", flush=True)

    # Test at different contamination rates
    contamination_rates = [0.01, 0.05, 0.10, 0.25, 0.50]
    batch_size = 50  # Simulated batch size
    n_trials = 5

    rng = np.random.default_rng(42)
    results = {}

    print("\nTesting contamination rates...", flush=True)
    for rate in contamination_rates:
        n_ood = max(1, int(batch_size * rate))
        n_id = batch_size - n_ood

        trial_aurocs = []
        trial_aps = []
        trial_precisions_at_100 = []

        for trial in range(n_trials):
            # Sample batch
            id_idx = rng.choice(len(id_pool), n_id, replace=True)
            ood_idx = rng.choice(len(ood_pool), n_ood, replace=True)

            batch_hidden = [id_pool[i] for i in id_idx] + [ood_pool[i] for i in ood_idx]
            batch_labels = [0]*n_id + [1]*n_ood

            # Shuffle
            order = rng.permutation(len(batch_hidden))
            batch_hidden = [batch_hidden[i] for i in order]
            batch_labels = [batch_labels[i] for i in order]

            # Score
            batch_scores = [cosine_dist(h, centroid) for h in batch_hidden]

            # Metrics
            auroc = roc_auc_score(batch_labels, batch_scores)
            ap = average_precision_score(batch_labels, batch_scores)

            # Precision at 100% recall (how many false positives when catching all OOD)
            predictions = [1 if s > threshold else 0 for s in batch_scores]
            tp = sum(1 for p, l in zip(predictions, batch_labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, batch_labels) if p == 1 and l == 0)
            recall = tp / n_ood if n_ood > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            trial_aurocs.append(auroc)
            trial_aps.append(ap)
            trial_precisions_at_100.append(precision)

        results[f"{rate:.2f}"] = {
            'contamination_rate': rate,
            'n_id': n_id,
            'n_ood': n_ood,
            'mean_auroc': float(np.mean(trial_aurocs)),
            'std_auroc': float(np.std(trial_aurocs)),
            'mean_ap': float(np.mean(trial_aps)),
            'std_ap': float(np.std(trial_aps)),
            'mean_precision': float(np.mean(trial_precisions_at_100)),
        }
        print(f"  rate={rate:.2f}: AUROC={np.mean(trial_aurocs):.3f}±{np.std(trial_aurocs):.3f}, "
              f"AP={np.mean(trial_aps):.3f}, precision={np.mean(trial_precisions_at_100):.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'mixed_batch',
        'experiment_number': 86,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'id_pool_size': len(id_pool),
        'ood_pool_size': len(ood_pool),
        'batch_size': batch_size,
        'n_trials': n_trials,
        'threshold': float(threshold),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"mixed_batch_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
