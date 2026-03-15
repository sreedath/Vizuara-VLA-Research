"""
Layer 3 Sample Efficiency.

Exp 133 showed L32 needs n>=8 for perfect detection.
Does L3 need more or fewer samples given its different geometry?

Experiment 143 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 36001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 36002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 36003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 36004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight(idx):
    rng = np.random.default_rng(idx * 36010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 36014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog(idx):
    rng = np.random.default_rng(idx * 36020)
    base = create_highway(idx + 120000)
    fog = np.full_like(base, 200)
    result = (base.astype(float) * 0.5 + fog.astype(float) * 0.5)
    noise = rng.integers(-3, 4, result.shape, dtype=np.int16)
    return np.clip(result + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()


def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("LAYER 3 SAMPLE EFFICIENCY", flush=True)
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
        'twilight': (create_twilight, 'OOD'),
        'snow': (create_snow, 'OOD'),
        'fog': (create_fog, 'OOD'),
    }

    print("\n--- Collecting L3 embeddings ---", flush=True)
    id_embeds = []
    ood_embeds = []

    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        for i in range(20):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 4100)), prompt, layer=3)
            if h is not None:
                if group == 'ID':
                    id_embeds.append(h)
                else:
                    ood_embeds.append(h)

    id_embeds = np.array(id_embeds)
    ood_embeds = np.array(ood_embeds)
    print(f"\nID: {len(id_embeds)}, OOD: {len(ood_embeds)}", flush=True)

    test_id = id_embeds[-10:]
    cal_pool = id_embeds[:-10]

    test_embeds = np.vstack([test_id, ood_embeds])
    test_labels = np.array([0]*len(test_id) + [1]*len(ood_embeds))

    cal_sizes = [1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30]
    cal_sizes = [s for s in cal_sizes if s <= len(cal_pool)]

    print("\n--- Sample Efficiency (L3) ---", flush=True)
    results = {}
    for n_cal in cal_sizes:
        n_trials = 20 if n_cal <= 5 else 10 if n_cal <= 10 else 5
        trial_aurocs = []
        trial_ds = []

        for trial in range(n_trials):
            rng = np.random.default_rng(trial * 999 + n_cal)
            indices = rng.choice(len(cal_pool), size=min(n_cal, len(cal_pool)), replace=False)
            centroid = np.mean(cal_pool[indices], axis=0)

            scores = np.array([cosine_distance(e, centroid) for e in test_embeds])
            auroc = float(roc_auc_score(test_labels, scores))
            id_s = scores[test_labels == 0]
            ood_s = scores[test_labels == 1]
            d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
            trial_aurocs.append(auroc)
            trial_ds.append(d)

        results[str(n_cal)] = {
            'n_cal': n_cal,
            'n_trials': n_trials,
            'mean_auroc': float(np.mean(trial_aurocs)),
            'std_auroc': float(np.std(trial_aurocs)),
            'min_auroc': float(np.min(trial_aurocs)),
            'mean_d': float(np.mean(trial_ds)),
            'std_d': float(np.std(trial_ds)),
        }
        print(f"  n={n_cal:3d}: AUROC={np.mean(trial_aurocs):.4f}±{np.std(trial_aurocs):.4f}, "
              f"d={np.mean(trial_ds):.1f}±{np.std(trial_ds):.1f} "
              f"(min_AUROC={np.min(trial_aurocs):.4f})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'l3_sample_efficiency',
        'experiment_number': 143,
        'timestamp': timestamp,
        'layer': 3,
        'n_id_total': len(id_embeds),
        'n_ood_total': len(ood_embeds),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"l3_sample_efficiency_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
