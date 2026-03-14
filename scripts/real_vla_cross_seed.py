"""
Cross-Seed Robustness.

Tests whether OOD detection results are robust across different
random seeds for synthetic image generation. Runs the full
detection pipeline 5 times with different seed offsets.

Experiment 87 in the CalibDrive series.
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


def create_highway(idx, seed_offset=0):
    rng = np.random.default_rng((idx + seed_offset) * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx, seed_offset=0):
    rng = np.random.default_rng((idx + seed_offset) * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx, seed_offset=0):
    rng = np.random.default_rng((idx + seed_offset) * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx, seed_offset=0):
    rng = np.random.default_rng((idx + seed_offset) * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx, seed_offset=0):
    rng = np.random.default_rng((idx + seed_offset) * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx, seed_offset=0):
    rng = np.random.default_rng((idx + seed_offset) * 5014)
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


def run_trial(model, processor, prompt, seed_offset):
    """Run one complete detection trial with given seed offset."""
    # Calibrate
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000, seed_offset)), prompt)
            if h is not None:
                cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)

    # ID test
    id_scores = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500, seed_offset)), prompt)
            if h is not None:
                id_scores.append(cosine_dist(h, centroid))

    # OOD test
    ood_scores = []
    for fn in [create_noise, create_indoor, create_twilight_highway, create_snow]:
        for i in range(6):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500, seed_offset)), prompt)
            if h is not None:
                ood_scores.append(cosine_dist(h, centroid))

    labels = [0]*len(id_scores) + [1]*len(ood_scores)
    scores = id_scores + ood_scores
    auroc = roc_auc_score(labels, scores)

    id_arr = np.array(id_scores)
    ood_arr = np.array(ood_scores)
    pooled = np.sqrt((id_arr.var() + ood_arr.var()) / 2)
    d = (ood_arr.mean() - id_arr.mean()) / (pooled + 1e-10)

    return {
        'auroc': float(auroc),
        'cohens_d': float(d),
        'n_cal': len(cal_hidden),
        'n_id': len(id_scores),
        'n_ood': len(ood_scores),
        'id_mean': float(id_arr.mean()),
        'ood_mean': float(ood_arr.mean()),
    }


def main():
    print("=" * 70, flush=True)
    print("CROSS-SEED ROBUSTNESS", flush=True)
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

    seed_offsets = [0, 10000, 20000, 30000, 40000]
    results = {}

    for i, offset in enumerate(seed_offsets):
        print(f"\nTrial {i+1}/{len(seed_offsets)} (seed_offset={offset})...", flush=True)
        trial = run_trial(model, processor, prompt, offset)
        results[str(offset)] = trial
        print(f"  AUROC={trial['auroc']:.3f}, d={trial['cohens_d']:.2f}, "
              f"ID={trial['id_mean']:.4f}, OOD={trial['ood_mean']:.4f}", flush=True)

    # Summary statistics
    aurocs = [r['auroc'] for r in results.values()]
    ds = [r['cohens_d'] for r in results.values()]

    print("\n" + "=" * 70, flush=True)
    print("CROSS-SEED SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.4f}", flush=True)
    print(f"  Cohen's d: {np.mean(ds):.2f} ± {np.std(ds):.2f}", flush=True)
    print(f"  All perfect: {all(a == 1.0 for a in aurocs)}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'cross_seed',
        'experiment_number': 87,
        'timestamp': timestamp,
        'n_seeds': len(seed_offsets),
        'seed_offsets': seed_offsets,
        'results': results,
        'summary': {
            'mean_auroc': float(np.mean(aurocs)),
            'std_auroc': float(np.std(aurocs)),
            'mean_cohens_d': float(np.mean(ds)),
            'std_cohens_d': float(np.std(ds)),
            'all_perfect': all(a == 1.0 for a in aurocs),
        }
    }
    output_path = os.path.join(RESULTS_DIR, f"cross_seed_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
