"""
Calibration Sample Efficiency for OOD Detection.

Tests the minimum number of calibration samples needed for
effective OOD detection, from 1 to 30 samples.

This is critical for practical deployment: if 1-5 samples suffice,
setup cost is minimal. If 20+ are needed, deployment overhead increases.

Experiment 56 in the CalibDrive series.
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
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)
    return hidden


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("CALIBRATION SAMPLE EFFICIENCY", flush=True)
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

    # First, collect a large calibration pool
    print("\nCollecting calibration pool (30 samples)...", flush=True)
    cal_pool = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            cal_pool.append(h)
    print(f"  Calibration pool: {len(cal_pool)} samples", flush=True)

    # Test set
    print("\nCollecting test set...", flush=True)
    test_fns = {
        'highway': (create_highway, False, 10),
        'urban': (create_urban, False, 10),
        'noise': (create_noise, True, 8),
        'indoor': (create_indoor, True, 8),
        'inverted': (create_inverted, True, 8),
        'blackout': (create_blackout, True, 8),
    }

    test_hidden = []
    test_labels = []
    test_scenarios = []
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 200)), prompt)
            test_hidden.append(h)
            test_labels.append(1 if is_ood else 0)
            test_scenarios.append(scene)

    test_hidden = np.array(test_hidden)
    test_labels = np.array(test_labels)
    print(f"  Test set: {len(test_hidden)} samples "
          f"({sum(test_labels==0)} ID, {sum(test_labels==1)} OOD)", flush=True)

    # Test different calibration sizes
    print("\n" + "=" * 70, flush=True)
    print("EFFICIENCY ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    cal_sizes = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    n_trials = 5  # Multiple trials for small sizes (random subset selection)

    print(f"\n  {'N_cal':>6} {'Mean AUROC':>12} {'Std':>8} {'Min':>8} {'Max':>8}", flush=True)
    print("  " + "-" * 44, flush=True)

    results = {}
    for n_cal in cal_sizes:
        trial_aurocs = []
        rng = np.random.default_rng(42)

        for trial in range(n_trials):
            if n_cal >= len(cal_pool):
                subset = list(range(len(cal_pool)))
            else:
                subset = rng.choice(len(cal_pool), size=n_cal, replace=False).tolist()

            centroid = np.mean([cal_pool[j] for j in subset], axis=0)
            scores = [cosine_dist(h, centroid) for h in test_hidden]
            try:
                auroc = roc_auc_score(test_labels, scores)
            except ValueError:
                auroc = 0.5
            trial_aurocs.append(auroc)

        mean_auroc = np.mean(trial_aurocs)
        std_auroc = np.std(trial_aurocs)
        min_auroc = min(trial_aurocs)
        max_auroc = max(trial_aurocs)

        results[n_cal] = {
            'mean': float(mean_auroc),
            'std': float(std_auroc),
            'min': float(min_auroc),
            'max': float(max_auroc),
            'trials': trial_aurocs,
        }

        print(f"  {n_cal:>6} {mean_auroc:>12.3f} {std_auroc:>8.3f} "
              f"{min_auroc:>8.3f} {max_auroc:>8.3f}", flush=True)

    # Find minimum N for 95% of max AUROC
    max_auroc = results[30]['mean']
    threshold_95 = 0.95 * max_auroc
    min_n_95 = None
    for n_cal in cal_sizes:
        if results[n_cal]['mean'] >= threshold_95:
            min_n_95 = n_cal
            break

    print(f"\n  Max AUROC (N=30): {max_auroc:.3f}", flush=True)
    print(f"  95% of max ({threshold_95:.3f}): achieved at N={min_n_95}", flush=True)

    # Per-OOD type at different calibration sizes
    print("\n  Per-OOD type analysis:", flush=True)
    for n_cal in [1, 5, 10, 30]:
        if n_cal >= len(cal_pool):
            subset = list(range(len(cal_pool)))
        else:
            subset = list(range(n_cal))
        centroid = np.mean([cal_pool[j] for j in subset], axis=0)

        print(f"\n    N_cal = {n_cal}:", flush=True)
        ood_types = ['noise', 'indoor', 'inverted', 'blackout']
        easy = test_hidden[test_labels == 0]
        for ood_type in ood_types:
            mask = [s == ood_type for s in test_scenarios]
            type_ood = test_hidden[mask]
            type_labels = [0]*len(easy) + [1]*len(type_ood)
            type_hidden = np.vstack([easy, type_ood])
            type_scores = [cosine_dist(h, centroid) for h in type_hidden]
            auroc = roc_auc_score(type_labels, type_scores)
            print(f"      {ood_type}: {auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'cal_efficiency',
        'experiment_number': 56,
        'timestamp': timestamp,
        'n_cal_pool': len(cal_pool),
        'n_test': len(test_hidden),
        'n_trials': n_trials,
        'results': {str(k): v for k, v in results.items()},
        'min_n_95': min_n_95,
        'max_auroc': float(max_auroc),
    }
    output_path = os.path.join(RESULTS_DIR, f"cal_efficiency_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
