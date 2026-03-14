"""
Temporal Consistency under Sequential Inputs.

Simulates a driving sequence with gradual OOD transition — starting
from clear highway, smoothly interpolating to an OOD scene, and
measuring how the detector responds frame-by-frame. Tests whether
the detector produces smooth, predictable score trajectories.

Experiment 121 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 15001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 15003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 15004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("TEMPORAL SEQUENCE ANALYSIS", flush=True)
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

    # Build calibration centroid
    print("\n--- Calibrating ---", flush=True)
    cal_embeds = []
    for i in range(10):
        h = extract_hidden(model, processor, Image.fromarray(create_highway(i + 2200)), prompt)
        if h is not None:
            cal_embeds.append(h)
    centroid = np.mean(cal_embeds, axis=0)
    print(f"  Calibration: {len(cal_embeds)} samples", flush=True)

    # Sequence 1: Highway → Noise (smooth interpolation)
    print("\n--- Sequence: Highway → Noise ---", flush=True)
    base_highway = create_highway(9999)
    base_noise = create_noise(9999)
    alphas = np.linspace(0, 1, 21)  # 0%, 5%, ..., 100% noise
    seq1_results = []
    for alpha in alphas:
        blended = np.clip(
            (1 - alpha) * base_highway.astype(float) + alpha * base_noise.astype(float),
            0, 255
        ).astype(np.uint8)
        h = extract_hidden(model, processor, Image.fromarray(blended), prompt)
        if h is not None:
            score = cosine_dist(h, centroid)
            seq1_results.append({'alpha': float(alpha), 'score': score})
            print(f"  alpha={alpha:.2f}: score={score:.4f}", flush=True)

    # Sequence 2: Highway → Indoor (smooth interpolation)
    print("\n--- Sequence: Highway → Indoor ---", flush=True)
    base_indoor = create_indoor(9999)
    seq2_results = []
    for alpha in alphas:
        blended = np.clip(
            (1 - alpha) * base_highway.astype(float) + alpha * base_indoor.astype(float),
            0, 255
        ).astype(np.uint8)
        h = extract_hidden(model, processor, Image.fromarray(blended), prompt)
        if h is not None:
            score = cosine_dist(h, centroid)
            seq2_results.append({'alpha': float(alpha), 'score': score})
            print(f"  alpha={alpha:.2f}: score={score:.4f}", flush=True)

    # Sequence 3: Sudden transition (highway, highway, ..., noise, noise, ...)
    print("\n--- Sequence: Sudden transition ---", flush=True)
    seq3_results = []
    for i in range(20):
        if i < 10:
            img = create_highway(i + 3000)
            label = 'highway'
        else:
            img = create_noise(i + 3000)
            label = 'noise'
        h = extract_hidden(model, processor, Image.fromarray(img), prompt)
        if h is not None:
            score = cosine_dist(h, centroid)
            seq3_results.append({'frame': i, 'label': label, 'score': score})
            print(f"  frame={i} ({label}): score={score:.4f}", flush=True)

    # Sequence 4: Oscillating (highway, noise, highway, noise, ...)
    print("\n--- Sequence: Oscillating ---", flush=True)
    seq4_results = []
    for i in range(20):
        if i % 2 == 0:
            img = create_highway(i + 4000)
            label = 'highway'
        else:
            img = create_noise(i + 4000)
            label = 'noise'
        h = extract_hidden(model, processor, Image.fromarray(img), prompt)
        if h is not None:
            score = cosine_dist(h, centroid)
            seq4_results.append({'frame': i, 'label': label, 'score': score})
            print(f"  frame={i} ({label}): score={score:.4f}", flush=True)

    # Analyze score smoothness
    print("\n--- Score Smoothness ---", flush=True)
    for seq_name, seq_data in [('highway_to_noise', seq1_results),
                                ('highway_to_indoor', seq2_results)]:
        scores = [d['score'] for d in seq_data]
        diffs = np.diff(scores)
        monotonicity = np.sum(diffs > 0) / len(diffs)
        max_jump = float(np.max(np.abs(diffs)))
        mean_step = float(np.mean(np.abs(diffs)))
        print(f"  {seq_name}: monotonicity={monotonicity:.2f}, max_jump={max_jump:.4f}, "
              f"mean_step={mean_step:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'temporal_sequence',
        'experiment_number': 121,
        'timestamp': timestamp,
        'sequences': {
            'highway_to_noise': seq1_results,
            'highway_to_indoor': seq2_results,
            'sudden_transition': seq3_results,
            'oscillating': seq4_results,
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"temporal_sequence_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
