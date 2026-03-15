"""
Fine-Grained Early Layer Sweep.

Since Exp 140 showed L3 vastly outperforms L32 (d=128 vs d=26),
we do a fine-grained sweep of layers 1-10 to find the true optimal layer.

Experiment 141 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 34001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 34002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 34003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 34004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight(idx):
    rng = np.random.default_rng(idx * 34010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 34014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog(idx):
    rng = np.random.default_rng(idx * 34020)
    base = create_highway(idx + 90000)
    fog = np.full_like(base, 200)
    result = (base.astype(float) * 0.5 + fog.astype(float) * 0.5)
    noise = rng.integers(-3, 4, result.shape, dtype=np.int16)
    return np.clip(result + noise, 0, 255).astype(np.uint8)


def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("FINE-GRAINED EARLY LAYER SWEEP", flush=True)
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
    # Sweep layers 1-10 + last layer for comparison
    layers = list(range(1, 11)) + [16, 24, 32]

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight, 'OOD'),
        'snow': (create_snow, 'OOD'),
        'fog': (create_fog, 'OOD'),
    }

    print("\n--- Collecting embeddings ---", flush=True)
    embeds = {l: {'id': [], 'ood': []} for l in layers}

    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        for i in range(12):
            img = Image.fromarray(fn(i + 3900))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                for l in layers:
                    if l < len(fwd.hidden_states):
                        h = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()
                        if group == 'ID':
                            embeds[l]['id'].append(h)
                        else:
                            embeds[l]['ood'].append(h)

    # Evaluate each layer
    print("\n--- Layer Sweep Results ---", flush=True)
    results = {}
    for l in layers:
        id_arr = np.array(embeds[l]['id'])
        ood_arr = np.array(embeds[l]['ood'])
        centroid = np.mean(id_arr, axis=0)

        id_dists = np.array([cosine_distance(e, centroid) for e in id_arr])
        ood_dists = np.array([cosine_distance(e, centroid) for e in ood_arr])

        labels = np.array([0]*len(id_dists) + [1]*len(ood_dists))
        scores = np.concatenate([id_dists, ood_dists])
        auroc = float(roc_auc_score(labels, scores))
        d = float((np.mean(ood_dists) - np.mean(id_dists)) / (np.std(id_dists) + 1e-10))
        gap = float(np.min(ood_dists) - np.max(id_dists))

        results[str(l)] = {
            'layer': l,
            'auroc': auroc,
            'd_prime': d,
            'gap': gap,
            'id_mean': float(np.mean(id_dists)),
            'id_max': float(np.max(id_dists)),
            'ood_mean': float(np.mean(ood_dists)),
            'ood_min': float(np.min(ood_dists)),
        }
        print(f"  Layer {l:2d}: AUROC={auroc:.4f}, d={d:.1f}, gap={gap:.4f}", flush=True)

    # Best layer
    best = max(results.items(), key=lambda x: x[1]['d_prime'])
    print(f"\n  Best: Layer {best[0]} with d={best[1]['d_prime']:.1f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'early_layer_sweep',
        'experiment_number': 141,
        'timestamp': timestamp,
        'layers': layers,
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"early_layer_sweep_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
