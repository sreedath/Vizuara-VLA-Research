"""
OR-Gate Dual-Layer Detector.

Combines L3 and L32 with an OR-gate: flag OOD if EITHER layer's
cosine distance exceeds its own threshold. L3 catches all non-fog
categories, L32 catches semantic domain shifts including fog.

Experiment 144 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 37001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 37002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 37003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 37004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight(idx):
    rng = np.random.default_rng(idx * 37010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 37014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog(idx, opacity=0.5):
    rng = np.random.default_rng(idx * 37020 + int(opacity * 100))
    base = create_highway(idx + 130000)
    fog = np.full_like(base, 200)
    result = (base.astype(float) * (1-opacity) + fog.astype(float) * opacity)
    noise = rng.integers(-3, 4, result.shape, dtype=np.int16)
    return np.clip(result + noise, 0, 255).astype(np.uint8)

def create_desert(idx):
    rng = np.random.default_rng(idx * 37024)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 180, 120]
    img[SIZE[0]//2:] = [180, 160, 100]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [220, 200, 140]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_night(idx):
    rng = np.random.default_rng(idx * 37025)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [10, 10, 30]
    img[SIZE[0]//2:] = [30, 30, 30]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [100, 100, 50]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_multi_layer(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}


def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("OR-GATE DUAL-LAYER DETECTOR", flush=True)
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
    layers = [3, 32]

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight, 'OOD'),
        'snow': (create_snow, 'OOD'),
        'fog_30': (lambda idx: create_fog(idx, 0.3), 'OOD'),
        'fog_50': (lambda idx: create_fog(idx, 0.5), 'OOD'),
        'desert': (create_desert, 'OOD'),
        'night': (create_night, 'OOD'),
    }

    print("\n--- Collecting embeddings ---", flush=True)
    cat_data = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = {l: [] for l in layers}
        for i in range(20):
            img = Image.fromarray(fn(i + 4200))
            hs = extract_multi_layer(model, processor, img, prompt, layers)
            if hs:
                for l in layers:
                    embeds[l].append(hs[l])
        cat_data[cat_name] = {'group': group, 'embeds': embeds}

    # Build ID calibration (first 15 per ID category)
    id_embeds = {l: [] for l in layers}
    id_test = {l: [] for l in layers}
    for cat_name, data in cat_data.items():
        if data['group'] == 'ID':
            for l in layers:
                id_embeds[l].extend(data['embeds'][l][:15])
                id_test[l].extend(data['embeds'][l][15:])

    centroids = {l: np.mean(id_embeds[l], axis=0) for l in layers}

    # Compute thresholds from ID (3-sigma)
    id_scores = {}
    for l in layers:
        dists = [cosine_distance(e, centroids[l]) for e in id_embeds[l]]
        id_scores[l] = dists
        print(f"\n  L{l} ID scores: mean={np.mean(dists):.6f}, max={np.max(dists):.6f}, 3σ={np.mean(dists)+3*np.std(dists):.6f}", flush=True)

    # Set thresholds
    thresholds = {l: float(np.mean(id_scores[l]) + 3*np.std(id_scores[l])) for l in layers}
    print(f"\n  Thresholds: L3={thresholds[3]:.6f}, L32={thresholds[32]:.6f}", flush=True)

    # Evaluate each OOD category with different detection strategies
    print("\n--- Detection Comparison ---", flush=True)
    strategies = {
        'L3_only': lambda s3, s32: s3 > thresholds[3],
        'L32_only': lambda s3, s32: s32 > thresholds[32],
        'OR_gate': lambda s3, s32: (s3 > thresholds[3]) or (s32 > thresholds[32]),
        'AND_gate': lambda s3, s32: (s3 > thresholds[3]) and (s32 > thresholds[32]),
    }

    all_results = {}
    for strat_name, strat_fn in strategies.items():
        tp, fp, fn_count, tn = 0, 0, 0, 0
        per_cat = {}

        for cat_name, data in cat_data.items():
            cat_tp, cat_fn = 0, 0
            cat_fp, cat_tn = 0, 0
            for i in range(len(data['embeds'][layers[0]])):
                s3 = cosine_distance(data['embeds'][3][i], centroids[3])
                s32 = cosine_distance(data['embeds'][32][i], centroids[32])
                detected = strat_fn(s3, s32)
                if data['group'] == 'OOD':
                    if detected:
                        tp += 1; cat_tp += 1
                    else:
                        fn_count += 1; cat_fn += 1
                else:
                    if detected:
                        fp += 1; cat_fp += 1
                    else:
                        tn += 1; cat_tn += 1

            if data['group'] == 'OOD':
                recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
                per_cat[cat_name] = {'tp': cat_tp, 'fn': cat_fn, 'recall': recall}

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        all_results[strat_name] = {
            'tp': tp, 'fp': fp, 'fn': fn_count, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr,
            'per_category': per_cat,
        }
        print(f"\n  {strat_name:12s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f}", flush=True)
        for cat_name, cat_res in per_cat.items():
            print(f"    {cat_name:15s}: {cat_res['tp']}/{cat_res['tp']+cat_res['fn']} detected (recall={cat_res['recall']:.3f})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'or_gate_detector',
        'experiment_number': 144,
        'timestamp': timestamp,
        'layers': layers,
        'thresholds': {str(l): thresholds[l] for l in layers},
        'results': all_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"or_gate_detector_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
