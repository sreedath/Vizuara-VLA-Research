"""
Hidden Layer Sweep for OOD Detection.

Tests cosine distance OOD detection at every 4th hidden layer of
OpenVLA-7B (32 layers total). Which layer provides the best signal?

Hypothesis: later layers capture more task-specific features and
provide better OOD separation, but earlier layers may capture
more general visual features useful for cross-domain detection.

Experiment 72 in the CalibDrive series.
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

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("HIDDEN LAYER SWEEP", flush=True)
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

    # Layers to test (every 4th + first/last)
    layers_to_test = [0, 4, 8, 12, 16, 20, 24, 28, 31, 32]  # 0=embedding, 32=final

    # Collect all hidden states for all images in one pass
    print("\nCollecting hidden states for calibration...", flush=True)
    cal_hidden_by_layer = {l: [] for l in layers_to_test}

    for fn in [create_highway, create_urban]:
        for i in range(10):
            img = Image.fromarray(fn(i + 9000))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)

            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                for l in layers_to_test:
                    if l < len(fwd.hidden_states):
                        h = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()
                        cal_hidden_by_layer[l].append(h)

    centroids = {}
    for l in layers_to_test:
        if cal_hidden_by_layer[l]:
            centroids[l] = np.mean(cal_hidden_by_layer[l], axis=0)
    print(f"  Calibrated layers: {list(centroids.keys())}", flush=True)

    # Test
    print("\nCollecting test hidden states...", flush=True)
    test_fns = {
        'highway': (create_highway, False, 8),
        'urban': (create_urban, False, 8),
        'noise': (create_noise, True, 6),
        'indoor': (create_indoor, True, 6),
        'twilight': (create_twilight_highway, True, 6),
        'blackout': (create_blackout, True, 4),
    }

    test_hidden_by_layer = {l: [] for l in layers_to_test}
    test_labels = []
    test_scenarios = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())

    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            img = Image.fromarray(fn(i + 700))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)

            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                for l in layers_to_test:
                    if l < len(fwd.hidden_states):
                        h = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()
                        test_hidden_by_layer[l].append(h)

            test_labels.append(1 if is_ood else 0)
            test_scenarios.append(scene)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    # Compute AUROC per layer
    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    results = {}
    for l in layers_to_test:
        if l not in centroids or not test_hidden_by_layer[l]:
            continue

        scores = [cosine_dist(h, centroids[l]) for h in test_hidden_by_layer[l]]
        auroc = roc_auc_score(test_labels, scores)

        # Per-scenario
        per_scene = {}
        for scene in set(test_scenarios):
            mask = [s == scene for s in test_scenarios]
            scene_scores = [s for s, m in zip(scores, mask) if m]
            per_scene[scene] = float(np.mean(scene_scores))

        # Effect size
        id_scores = [s for s, lab in zip(scores, test_labels) if lab == 0]
        ood_scores = [s for s, lab in zip(scores, test_labels) if lab == 1]
        pooled_std = np.sqrt((np.std(id_scores)**2 + np.std(ood_scores)**2) / 2)
        cohens_d = abs(np.mean(id_scores) - np.mean(ood_scores)) / (pooled_std + 1e-10)

        results[l] = {
            'auroc': float(auroc),
            'cohens_d': float(cohens_d),
            'id_mean': float(np.mean(id_scores)),
            'ood_mean': float(np.mean(ood_scores)),
            'per_scene': per_scene,
        }
        print(f"  Layer {l:>2}: AUROC={auroc:.3f}  d={cohens_d:.2f}  "
              f"ID={np.mean(id_scores):.4f}  OOD={np.mean(ood_scores):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'layer_sweep',
        'experiment_number': 72,
        'timestamp': timestamp,
        'layers_tested': layers_to_test,
        'n_cal': len(cal_hidden_by_layer[layers_to_test[0]]),
        'n_test': len(test_labels),
        'results': {str(k): v for k, v in results.items()},
    }
    output_path = os.path.join(RESULTS_DIR, f"layer_sweep_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
