"""
Layer 3 Comprehensive OOD Detection.

Comprehensive evaluation of the Layer 3 detector across ALL OOD
categories we've tested, with larger sample sizes for precise
AUROC and d-prime estimates.

Experiment 142 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 35001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 35002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 35003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 35004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight(idx):
    rng = np.random.default_rng(idx * 35010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 35014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog(idx, opacity=0.5):
    rng = np.random.default_rng(idx * 35020 + int(opacity * 100))
    base = create_highway(idx + 100000)
    fog = np.full_like(base, 200)
    result = (base.astype(float) * (1-opacity) + fog.astype(float) * opacity)
    noise = rng.integers(-3, 4, result.shape, dtype=np.int16)
    return np.clip(result + noise, 0, 255).astype(np.uint8)

def create_rain(idx):
    rng = np.random.default_rng(idx * 35021)
    base = create_highway(idx + 110000)
    result = (base.astype(float) * 0.7).astype(np.uint8)
    for _ in range(50):
        x = rng.integers(0, SIZE[1])
        y0 = rng.integers(0, SIZE[0]//2)
        length = rng.integers(10, 40)
        y1 = min(y0 + length, SIZE[0])
        result[y0:y1, max(0,x-1):min(SIZE[1],x+1)] = np.clip(
            result[y0:y1, max(0,x-1):min(SIZE[1],x+1)].astype(int) + 80, 0, 255).astype(np.uint8)
    return result

def create_construction(idx):
    rng = np.random.default_rng(idx * 35022)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    for i in range(3):
        x = SIZE[1]//4 + i * SIZE[1]//4
        img[SIZE[0]//2-20:SIZE[0]//2+5, max(0,x-10):min(SIZE[1],x+10)] = [255, 140, 0]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_underwater(idx):
    rng = np.random.default_rng(idx * 35023)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    for y in range(SIZE[0]):
        blue = int(150 - y * 0.3)
        img[y, :] = [0, max(0, blue-30), max(0, blue)]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_desert(idx):
    rng = np.random.default_rng(idx * 35024)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 180, 120]
    img[SIZE[0]//2:] = [180, 160, 100]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [220, 200, 140]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_night(idx):
    rng = np.random.default_rng(idx * 35025)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [10, 10, 30]
    img[SIZE[0]//2:] = [30, 30, 30]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [100, 100, 50]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_forest(idx):
    rng = np.random.default_rng(idx * 35026)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [100, 140, 100]
    img[SIZE[0]//2:] = [50, 80, 30]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


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
    print("LAYER 3 COMPREHENSIVE OOD DETECTION", flush=True)
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
        'fog_30': (lambda idx: create_fog(idx, 0.3), 'OOD'),
        'fog_50': (lambda idx: create_fog(idx, 0.5), 'OOD'),
        'fog_70': (lambda idx: create_fog(idx, 0.7), 'OOD'),
        'rain': (create_rain, 'OOD'),
        'construction': (create_construction, 'OOD'),
        'underwater': (create_underwater, 'OOD'),
        'desert': (create_desert, 'OOD'),
        'night': (create_night, 'OOD'),
        'forest': (create_forest, 'OOD'),
    }

    print("\n--- Collecting L3 embeddings ---", flush=True)
    id_embeds = []
    ood_by_cat = {}

    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(15):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 4000)), prompt, layer=3)
            if h is not None:
                embeds.append(h)
        if group == 'ID':
            id_embeds.extend(embeds)
        else:
            ood_by_cat[cat_name] = embeds

    id_arr = np.array(id_embeds)
    centroid = np.mean(id_arr, axis=0)
    id_dists = np.array([cosine_distance(e, centroid) for e in id_arr])

    print(f"\nID: {len(id_embeds)}, max_dist={np.max(id_dists):.6f}", flush=True)

    # Evaluate each category
    print("\n--- Per-Category Results (L3) ---", flush=True)
    results = {}
    for cat_name, embeds in ood_by_cat.items():
        ood_arr = np.array(embeds)
        ood_dists = np.array([cosine_distance(e, centroid) for e in ood_arr])

        labels = np.array([0]*len(id_dists) + [1]*len(ood_dists))
        scores = np.concatenate([id_dists, ood_dists])
        auroc = float(roc_auc_score(labels, scores))
        d = float((np.mean(ood_dists) - np.mean(id_dists)) / (np.std(id_dists) + 1e-10))
        gap = float(np.min(ood_dists) - np.max(id_dists))

        results[cat_name] = {
            'n': len(embeds),
            'auroc': auroc,
            'd_prime': d,
            'gap': gap,
            'mean_dist': float(np.mean(ood_dists)),
            'min_dist': float(np.min(ood_dists)),
        }
        print(f"  {cat_name:15s}: AUROC={auroc:.4f}, d={d:.1f}, gap={gap:.6f}", flush=True)

    # Overall
    all_ood = []
    for embeds in ood_by_cat.values():
        all_ood.extend(embeds)
    all_ood_dists = np.array([cosine_distance(e, centroid) for e in all_ood])
    all_labels = np.array([0]*len(id_dists) + [1]*len(all_ood_dists))
    all_scores = np.concatenate([id_dists, all_ood_dists])
    overall_auroc = float(roc_auc_score(all_labels, all_scores))
    overall_d = float((np.mean(all_ood_dists) - np.mean(id_dists)) / (np.std(id_dists) + 1e-10))
    print(f"\n  OVERALL: AUROC={overall_auroc:.4f}, d={overall_d:.1f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'l3_comprehensive',
        'experiment_number': 142,
        'timestamp': timestamp,
        'layer': 3,
        'n_id': len(id_embeds),
        'id_max_dist': float(np.max(id_dists)),
        'id_mean_dist': float(np.mean(id_dists)),
        'overall': {
            'auroc': overall_auroc,
            'd_prime': overall_d,
        },
        'per_category': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"l3_comprehensive_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
