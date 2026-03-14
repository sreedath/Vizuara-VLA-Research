"""
Embedding Dimensionality Analysis.

Tests how PCA-reduced hidden states affect cosine OOD detection.
OpenVLA has 4096-dimensional hidden states. Can we reduce to
much lower dimensions while maintaining detection?

Tests: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 dims.

Experiment 78 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

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

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("EMBEDDING DIMENSIONALITY ANALYSIS", flush=True)
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

    # Collect hidden states
    print("\nCollecting hidden states...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            img = Image.fromarray(fn(i + 9000))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                h = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
                cal_hidden.append(h)
    print(f"  Calibration: {len(cal_hidden)} samples", flush=True)

    test_fns = {
        'highway': (create_highway, False, 10),
        'urban': (create_urban, False, 10),
        'noise': (create_noise, True, 8),
        'indoor': (create_indoor, True, 8),
        'twilight': (create_twilight_highway, True, 6),
        'blackout': (create_blackout, True, 4),
    }

    test_hidden = []
    test_labels = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            img = Image.fromarray(fn(i + 500))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                h = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
                test_hidden.append(h)
            test_labels.append(1 if is_ood else 0)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    cal_arr = np.array(cal_hidden)
    test_arr = np.array(test_hidden)
    full_dim = cal_arr.shape[1]
    print(f"  Full dimensionality: {full_dim}", flush=True)

    # Test dimensionality reduction
    dims_to_test = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    dims_to_test = [d for d in dims_to_test if d < min(len(cal_hidden), full_dim)]
    dims_to_test.append(full_dim)  # Also test full dim

    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    results = {}
    for n_dim in dims_to_test:
        if n_dim == full_dim:
            # Full dimensionality (no PCA)
            centroid = np.mean(cal_arr, axis=0)
            scores = [cosine_dist(h, centroid) for h in test_arr]
        else:
            # PCA reduction
            all_data = np.vstack([cal_arr, test_arr])
            pca = PCA(n_components=n_dim, random_state=42)
            all_reduced = pca.fit_transform(all_data)
            cal_reduced = all_reduced[:len(cal_arr)]
            test_reduced = all_reduced[len(cal_arr):]

            centroid = np.mean(cal_reduced, axis=0)
            scores = [cosine_dist(h, centroid) for h in test_reduced]

        auroc = roc_auc_score(test_labels, scores)

        id_scores = [s for s, l in zip(scores, test_labels) if l == 0]
        ood_scores = [s for s, l in zip(scores, test_labels) if l == 1]
        pooled_std = np.sqrt((np.std(id_scores)**2 + np.std(ood_scores)**2) / 2)
        cohens_d = abs(np.mean(id_scores) - np.mean(ood_scores)) / (pooled_std + 1e-10)

        results[n_dim] = {
            'auroc': float(auroc),
            'cohens_d': float(cohens_d),
            'id_mean': float(np.mean(id_scores)),
            'ood_mean': float(np.mean(ood_scores)),
        }
        dim_label = f"{n_dim}" if n_dim != full_dim else f"{n_dim} (full)"
        print(f"  Dim={dim_label:<12}: AUROC={auroc:.3f}  d={cohens_d:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'embedding_dim',
        'experiment_number': 78,
        'timestamp': timestamp,
        'full_dim': full_dim,
        'n_cal': len(cal_hidden),
        'n_test': len(test_hidden),
        'results': {str(k): v for k, v in results.items()},
    }
    output_path = os.path.join(RESULTS_DIR, f"embedding_dim_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
