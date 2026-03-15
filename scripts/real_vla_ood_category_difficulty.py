"""
OOD Category Difficulty Ranking.

Computes a comprehensive difficulty ranking for different OOD categories
by analyzing their cosine distance distributions, overlap with ID, and
how well each is detected. Includes both original categories and new
fine-grained subtypes.

Key question: Which OOD categories are hardest to detect, and why?

Experiment 138 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 31001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 31002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 31003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 31004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight(idx):
    rng = np.random.default_rng(idx * 31010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 31014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog(idx):
    """Highway with fog overlay."""
    rng = np.random.default_rng(idx * 31020)
    base = create_highway(idx + 50000)
    fog = np.full_like(base, 200)
    alpha = 0.5
    result = (base.astype(float) * (1-alpha) + fog.astype(float) * alpha)
    noise = rng.integers(-3, 4, result.shape, dtype=np.int16)
    return np.clip(result + noise, 0, 255).astype(np.uint8)

def create_rain(idx):
    """Highway with rain streaks."""
    rng = np.random.default_rng(idx * 31021)
    base = create_highway(idx + 60000)
    # Darken and add vertical streaks
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
    """Road with construction zone (orange barriers)."""
    rng = np.random.default_rng(idx * 31022)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    # Orange barriers
    for i in range(3):
        x = SIZE[1]//4 + i * SIZE[1]//4
        img[SIZE[0]//2-20:SIZE[0]//2+5, max(0,x-10):min(SIZE[1],x+10)] = [255, 140, 0]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_underwater(idx):
    """Underwater scene (extreme OOD)."""
    rng = np.random.default_rng(idx * 31023)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    # Blue gradient
    for y in range(SIZE[0]):
        blue = int(150 - y * 0.3)
        img[y, :] = [0, max(0, blue-30), max(0, blue)]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_desert(idx):
    """Desert road."""
    rng = np.random.default_rng(idx * 31024)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 180, 120]  # sandy sky
    img[SIZE[0]//2:] = [180, 160, 100]  # sandy road
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [220, 200, 140]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("OOD CATEGORY DIFFICULTY RANKING", flush=True)
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
        'rain': (create_rain, 'OOD'),
        'construction': (create_construction, 'OOD'),
        'underwater': (create_underwater, 'OOD'),
        'desert': (create_desert, 'OOD'),
    }

    # Collect ID calibration
    print("\n--- Collecting embeddings ---", flush=True)
    id_embeds = []
    ood_by_cat = {}

    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(12):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 3600)), prompt)
            if h is not None:
                embeds.append(h)
        if group == 'ID':
            id_embeds.extend(embeds)
        else:
            ood_by_cat[cat_name] = embeds

    id_embeds = np.array(id_embeds)
    centroid = np.mean(id_embeds, axis=0)
    id_dists = np.array([cosine_distance(e, centroid) for e in id_embeds])
    id_max = float(np.max(id_dists))
    id_mean = float(np.mean(id_dists))

    print(f"\nID: {len(id_embeds)} embeds, max dist={id_max:.4f}", flush=True)

    # Evaluate each OOD category
    print("\n--- Category Difficulty Ranking ---", flush=True)
    results = {}
    for cat_name, embeds in ood_by_cat.items():
        ood_arr = np.array(embeds)
        ood_dists = np.array([cosine_distance(e, centroid) for e in ood_arr])

        # AUROC
        labels = np.array([0]*len(id_dists) + [1]*len(ood_dists))
        scores = np.concatenate([id_dists, ood_dists])
        auroc = float(roc_auc_score(labels, scores))
        d = float((np.mean(ood_dists) - id_mean) / (np.std(id_dists) + 1e-10))

        gap = float(np.min(ood_dists) - id_max)
        overlap = gap < 0

        results[cat_name] = {
            'n': len(embeds),
            'mean_dist': float(np.mean(ood_dists)),
            'std_dist': float(np.std(ood_dists)),
            'min_dist': float(np.min(ood_dists)),
            'max_dist': float(np.max(ood_dists)),
            'gap': gap,
            'overlap': overlap,
            'auroc': auroc,
            'd_prime': d,
        }
        status = "OVERLAP!" if overlap else f"gap={gap:.4f}"
        print(f"  {cat_name:15s}: dist={np.mean(ood_dists):.4f}±{np.std(ood_dists):.4f}, "
              f"min={np.min(ood_dists):.4f}, {status}, AUROC={auroc:.4f}, d={d:.1f}", flush=True)

    # Rank by difficulty (smallest gap = hardest)
    print("\n--- Ranked by Difficulty (hardest first) ---", flush=True)
    sorted_cats = sorted(results.items(), key=lambda x: x[1]['gap'])
    for cat_name, res in sorted_cats:
        print(f"  {cat_name:15s}: gap={res['gap']:.4f}, d={res['d_prime']:.1f}, AUROC={res['auroc']:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'ood_category_difficulty',
        'experiment_number': 138,
        'timestamp': timestamp,
        'id_stats': {
            'n': len(id_embeds),
            'mean_dist': id_mean,
            'max_dist': id_max,
            'std_dist': float(np.std(id_dists)),
        },
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"ood_category_difficulty_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
