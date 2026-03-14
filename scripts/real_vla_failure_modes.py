"""
Failure Mode Analysis.

Systematically tests when and how the OOD detector fails by creating
adversarial-like scenarios that push the boundaries of detection:
1. ID-mimicking OOD (constructed to look like driving in hidden space)
2. Near-boundary interpolations at fine granularity
3. Color-only OOD (same structure, wrong colors)
4. Texture-only OOD (same layout, random textures)

Experiment 101 in the CalibDrive series.
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

def create_inverted_highway(idx):
    """Highway structure with inverted colors — same layout, wrong palette."""
    rng = np.random.default_rng(idx * 5020)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [120, 49, 20]  # Inverted sky
    img[SIZE[0]//2:] = [175, 175, 175]  # Inverted road
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [0, 0, 0]  # Inverted lane
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_green_highway(idx):
    """Highway with green sky (alien world) — subtle semantic shift."""
    rng = np.random.default_rng(idx * 5021)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [50, 235, 50]  # Green sky
    img[SIZE[0]//2:] = [80, 80, 80]  # Normal road
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_red_highway(idx):
    """Highway with red sky (sunset-like) — near-ID semantic."""
    rng = np.random.default_rng(idx * 5022)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [235, 120, 80]  # Red sky
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_textured_road(idx):
    """Road with random texture overlay — structure preserved, texture changed."""
    rng = np.random.default_rng(idx * 5023)
    base = create_highway(idx)
    texture = rng.integers(0, 60, base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + texture, 0, 255).astype(np.uint8)

def create_shifted_highway(idx):
    """Highway with shifted horizon — same colors, wrong geometry."""
    rng = np.random.default_rng(idx * 5024)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]*3//4] = [135, 206, 235]  # Sky takes 3/4
    img[SIZE[0]*3//4:] = [80, 80, 80]  # Road only 1/4
    img[SIZE[0]*3//4:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_highway_rotated(idx):
    """Highway rotated 90 degrees — driving content, wrong orientation."""
    img = create_highway(idx)
    return np.rot90(img).copy()


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
    print("FAILURE MODE ANALYSIS", flush=True)
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

    # Build calibration
    print("\nBuilding calibration...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            h = extract_hidden(model, processor,
                              Image.fromarray(fn(i + 9000)), prompt)
            if h is not None:
                cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    cal_scores = [cosine_dist(h, centroid) for h in cal_hidden]
    threshold = np.mean(cal_scores) + 3 * np.std(cal_scores)
    print(f"  Centroid from {len(cal_hidden)} samples, threshold={threshold:.4f}", flush=True)

    # Test challenging scenarios
    scenarios = {
        'inverted_highway': (create_inverted_highway, 'Same layout, inverted colors'),
        'green_highway': (create_green_highway, 'Normal road, green sky'),
        'red_highway': (create_red_highway, 'Normal road, red/sunset sky'),
        'textured_road': (create_textured_road, 'Highway + random texture overlay'),
        'shifted_horizon': (create_shifted_highway, 'Highway with shifted horizon'),
        'rotated_highway': (create_highway_rotated, 'Highway rotated 90°'),
    }

    results = {}
    for name, (fn, desc) in scenarios.items():
        print(f"\n--- {name}: {desc} ---", flush=True)
        scores = []
        for i in range(10):
            h = extract_hidden(model, processor,
                              Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                score = cosine_dist(h, centroid)
                detected = score > threshold
                scores.append(score)
                status = "OOD" if detected else "ID"
                print(f"  Sample {i}: score={score:.4f} [{status}]", flush=True)

        arr = np.array(scores)
        detection_rate = float(np.mean([s > threshold for s in scores]))
        results[name] = {
            'description': desc,
            'n': len(scores),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'detection_rate': detection_rate,
            'scores': [float(s) for s in scores],
        }
        print(f"  Mean={arr.mean():.4f}±{arr.std():.4f}, Detection rate={detection_rate:.0%}", flush=True)

    # Compare with known ID baseline
    print("\n--- ID baseline ---", flush=True)
    id_scores = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            h = extract_hidden(model, processor,
                              Image.fromarray(fn(i + 700)), prompt)
            if h is not None:
                score = cosine_dist(h, centroid)
                id_scores.append(score)
    id_arr = np.array(id_scores)
    results['id_baseline'] = {
        'mean': float(id_arr.mean()),
        'std': float(id_arr.std()),
        'max': float(id_arr.max()),
        'detection_rate': float(np.mean([s > threshold for s in id_scores])),
    }
    print(f"  ID: mean={id_arr.mean():.4f}±{id_arr.std():.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'failure_modes',
        'experiment_number': 101,
        'timestamp': timestamp,
        'threshold': float(threshold),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"failure_modes_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
