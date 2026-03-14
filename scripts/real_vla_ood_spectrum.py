"""
OOD Hardness Spectrum.

Tests detection on a wider range of OOD categories spanning the
difficulty continuum — from trivially detectable (random noise) to
near-ID (rain highway, night highway, fog highway).

Experiment 120 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 14001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 14002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Existing OOD categories
def create_noise(idx):
    rng = np.random.default_rng(idx * 14003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 14004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 14010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 14014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# NEW near-ID OOD categories
def create_rain_highway(idx):
    """Highway with rain effect — streaks and darker sky."""
    rng = np.random.default_rng(idx * 14020)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [100, 110, 120]  # overcast sky
    img[SIZE[0]//2:] = [50, 50, 55]     # wet road (darker)
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 200]  # dimmer lane lines
    # Rain streaks
    for _ in range(30):
        x = rng.integers(0, SIZE[1])
        y = rng.integers(0, SIZE[0])
        length = rng.integers(5, 20)
        for dy in range(length):
            if y+dy < SIZE[0]:
                img[y+dy, min(x, SIZE[1]-1)] = [180, 180, 190]
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_night_highway(idx):
    """Highway at night — dark with headlights."""
    rng = np.random.default_rng(idx * 14021)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [10, 10, 20]    # dark sky
    img[SIZE[0]//2:] = [30, 30, 35]    # dark road
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [150, 150, 100]  # dim lane lines
    # Headlight glow
    cx, cy = SIZE[1]//2, SIZE[0]//2 + 40
    for dx in range(-30, 31):
        for dy in range(-15, 16):
            dist = (dx**2 + dy**2)**0.5
            if dist < 30 and 0 <= cy+dy < SIZE[0] and 0 <= cx+dx < SIZE[1]:
                brightness = int(max(0, 80 - dist * 2.5))
                img[cy+dy, cx+dx] = np.clip(img[cy+dy, cx+dx].astype(int) + brightness, 0, 255)
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog_highway(idx):
    """Highway in fog — washed out, low contrast."""
    rng = np.random.default_rng(idx * 14022)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    # Fog blends everything toward gray
    img[:SIZE[0]//2] = [180, 185, 190]  # foggy sky
    img[SIZE[0]//2:] = [140, 140, 145]  # foggy road
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [160, 160, 165]  # barely visible lines
    noise = rng.integers(-15, 16, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_construction(idx):
    """Construction zone — orange cones, barriers."""
    rng = np.random.default_rng(idx * 14023)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]  # normal sky
    img[SIZE[0]//2:] = [80, 80, 80]     # road
    # Orange barriers
    for cx in range(50, SIZE[1]-50, 40):
        img[SIZE[0]//2-10:SIZE[0]//2+10, cx-5:cx+5] = [255, 140, 0]
    # Construction stripes
    img[SIZE[0]//2:SIZE[0]//2+5, :] = [255, 200, 0]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_solid_color(idx):
    """Solid color — camera failure mode."""
    rng = np.random.default_rng(idx * 14024)
    color = rng.integers(0, 256, 3)
    img = np.full((*SIZE, 3), color, dtype=np.uint8)
    return img


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
    print("OOD HARDNESS SPECTRUM", flush=True)
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
        # ID
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        # Near-ID OOD (hardest)
        'rain_highway': (create_rain_highway, 'OOD'),
        'fog_highway': (create_fog_highway, 'OOD'),
        'night_highway': (create_night_highway, 'OOD'),
        'construction': (create_construction, 'OOD'),
        # Standard OOD
        'snow': (create_snow, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        # Easy OOD
        'noise': (create_noise, 'OOD'),
        'solid_color': (create_solid_color, 'OOD'),
    }

    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(10):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 2100)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}

    # Calibrate on ID
    cal_embeds = []
    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            cal_embeds.extend(data['embeds'][:5])
    cal_embeds = np.array(cal_embeds)
    centroid = np.mean(cal_embeds, axis=0)

    # Test: remaining ID + all OOD
    test_embeds = []
    test_labels = []
    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            for e in data['embeds'][5:]:
                test_embeds.append(e)
                test_labels.append(0)
        else:
            for e in data['embeds']:
                test_embeds.append(e)
                test_labels.append(1)

    test_embeds = np.array(test_embeds)
    test_labels = np.array(test_labels)

    # Overall AUROC
    scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
    overall_auroc = float(roc_auc_score(test_labels, scores))
    id_scores = scores[test_labels == 0]
    ood_scores = scores[test_labels == 1]
    overall_d = float((np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10))
    print(f"\nOverall: AUROC={overall_auroc:.4f}, d={overall_d:.2f}", flush=True)

    # Per-category analysis
    print("\n--- Per-Category Results ---", flush=True)
    per_cat = {}
    for cat_name, data in embeddings.items():
        cat_scores = [cosine_dist(e, centroid) for e in data['embeds']]
        per_cat[cat_name] = {
            'group': data['group'],
            'mean_score': float(np.mean(cat_scores)),
            'std_score': float(np.std(cat_scores)),
            'min_score': float(np.min(cat_scores)),
            'max_score': float(np.max(cat_scores)),
        }

        # Per-category AUROC (this category as OOD vs all ID)
        if data['group'] == 'OOD':
            cat_test = list(id_scores) + cat_scores
            cat_labels = [0] * len(id_scores) + [1] * len(cat_scores)
            cat_auroc = float(roc_auc_score(cat_labels, cat_test))
            per_cat[cat_name]['auroc'] = cat_auroc
        else:
            per_cat[cat_name]['auroc'] = None

        auroc_val = per_cat[cat_name]['auroc']
        auroc_str = 'N/A' if auroc_val is None else f"{auroc_val:.4f}"
        print(f"  {cat_name:20s}: score={np.mean(cat_scores):.4f}+/-{np.std(cat_scores):.4f}, auroc={auroc_str}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'ood_spectrum',
        'experiment_number': 120,
        'timestamp': timestamp,
        'overall_auroc': overall_auroc,
        'overall_d': overall_d,
        'per_category': per_cat,
    }
    output_path = os.path.join(RESULTS_DIR, f"ood_spectrum_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
