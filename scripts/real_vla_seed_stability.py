"""
Seed and Replication Stability.

Tests whether repeated inference on the SAME image produces the
exact same embedding and detection score. This verifies:
1. Deterministic inference (no stochastic elements)
2. Numerical stability across repeated calls
3. Reproducibility of detection scores

Experiment 129 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 22001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 22003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 22004)
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
    print("SEED AND REPLICATION STABILITY", flush=True)
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

    # Test 1: Repeated inference on same image
    print("\n--- Test 1: Repeated Inference ---", flush=True)
    test_images = {
        'highway_0': Image.fromarray(create_highway(9001)),
        'noise_0': Image.fromarray(create_noise(9001)),
        'indoor_0': Image.fromarray(create_indoor(9001)),
    }

    N_REPEATS = 10
    repeat_results = {}

    for img_name, img in test_images.items():
        print(f"\n  {img_name} ({N_REPEATS} repeats):", flush=True)
        embeddings = []
        for r in range(N_REPEATS):
            h = extract_hidden(model, processor, img, prompt)
            if h is not None:
                embeddings.append(h)

        embeddings = np.array(embeddings)
        # Pairwise cosine distances
        pair_dists = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                pair_dists.append(cosine_dist(embeddings[i], embeddings[j]))

        # L2 distances
        l2_dists = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                l2_dists.append(float(np.linalg.norm(embeddings[i] - embeddings[j])))

        # Bit-exact check
        all_identical = all(np.array_equal(embeddings[0], embeddings[i]) for i in range(1, len(embeddings)))

        repeat_results[img_name] = {
            'n_repeats': len(embeddings),
            'all_identical': bool(all_identical),
            'max_cosine_dist': float(max(pair_dists)) if pair_dists else 0.0,
            'mean_cosine_dist': float(np.mean(pair_dists)) if pair_dists else 0.0,
            'max_l2_dist': float(max(l2_dists)) if l2_dists else 0.0,
            'mean_l2_dist': float(np.mean(l2_dists)) if l2_dists else 0.0,
        }

        print(f"    Identical: {all_identical}", flush=True)
        print(f"    Max cosine dist: {repeat_results[img_name]['max_cosine_dist']:.8f}", flush=True)
        print(f"    Max L2 dist: {repeat_results[img_name]['max_l2_dist']:.6f}", flush=True)

    # Test 2: Score stability — compute detection scores repeatedly
    print("\n--- Test 2: Score Stability ---", flush=True)
    # Build a simple centroid from 5 highway images
    cal_embeds = []
    for i in range(5):
        h = extract_hidden(model, processor, Image.fromarray(create_highway(i + 9100)), prompt)
        if h is not None:
            cal_embeds.append(h)
    centroid = np.mean(cal_embeds, axis=0)

    score_stability = {}
    for img_name, img in test_images.items():
        scores = []
        for r in range(N_REPEATS):
            h = extract_hidden(model, processor, img, prompt)
            if h is not None:
                scores.append(cosine_dist(h, centroid))

        score_stability[img_name] = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'range_score': float(np.max(scores) - np.min(scores)),
        }
        print(f"  {img_name}: score={np.mean(scores):.6f} +/- {np.std(scores):.8f}, "
              f"range={np.max(scores)-np.min(scores):.8f}", flush=True)

    # Test 3: Different PRNG seeds for same synthetic scene type
    print("\n--- Test 3: Cross-Seed Variation ---", flush=True)
    seed_results = {}
    for create_fn, name in [(create_highway, 'highway'), (create_noise, 'noise')]:
        scores = []
        for seed in range(20):
            h = extract_hidden(model, processor, Image.fromarray(create_fn(seed + 9200)), prompt)
            if h is not None:
                scores.append(cosine_dist(h, centroid))

        seed_results[name] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'cv': float(np.std(scores) / (np.mean(scores) + 1e-10)),
        }
        print(f"  {name}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}, "
              f"CV={np.std(scores)/np.mean(scores):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'seed_stability',
        'experiment_number': 129,
        'timestamp': timestamp,
        'repeat_results': repeat_results,
        'score_stability': score_stability,
        'seed_variation': seed_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"seed_stability_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
