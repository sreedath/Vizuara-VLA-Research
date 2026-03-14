"""
Embedding Stability Under Repeated Inference.

Tests whether the same image produces identical hidden states across
multiple forward passes. Any stochastic variation would affect
detection reliability.

Experiment 107 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image

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

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)


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
    print("EMBEDDING STABILITY UNDER REPEATED INFERENCE", flush=True)
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

    # Test 1: Same image, 20 repeated passes
    print("\n--- Test 1: Repeated passes on same highway image ---", flush=True)
    hw_img = Image.fromarray(create_highway(42))
    hw_embeds = []
    for i in range(20):
        h = extract_hidden(model, processor, hw_img, prompt)
        if h is not None:
            hw_embeds.append(h)
    hw_embeds = np.array(hw_embeds)

    # Pairwise cosine distances
    hw_pairwise = []
    for i in range(len(hw_embeds)):
        for j in range(i+1, len(hw_embeds)):
            hw_pairwise.append(cosine_dist(hw_embeds[i], hw_embeds[j]))

    print(f"  Pairwise cosine distances: mean={np.mean(hw_pairwise):.6f}, "
          f"max={np.max(hw_pairwise):.6f}, min={np.min(hw_pairwise):.6f}", flush=True)
    print(f"  L2 distance: mean={np.mean([np.linalg.norm(hw_embeds[0]-hw_embeds[i]) for i in range(1, len(hw_embeds))]):.6f}", flush=True)

    # Test 2: Same noise image, 20 repeated passes
    print("\n--- Test 2: Repeated passes on same noise image ---", flush=True)
    noise_img = Image.fromarray(create_noise(42))
    noise_embeds = []
    for i in range(20):
        h = extract_hidden(model, processor, noise_img, prompt)
        if h is not None:
            noise_embeds.append(h)
    noise_embeds = np.array(noise_embeds)

    noise_pairwise = []
    for i in range(len(noise_embeds)):
        for j in range(i+1, len(noise_embeds)):
            noise_pairwise.append(cosine_dist(noise_embeds[i], noise_embeds[j]))

    print(f"  Pairwise cosine distances: mean={np.mean(noise_pairwise):.6f}, "
          f"max={np.max(noise_pairwise):.6f}, min={np.min(noise_pairwise):.6f}", flush=True)

    # Test 3: Different highway images
    print("\n--- Test 3: Different highway images (for comparison) ---", flush=True)
    diff_hw_embeds = []
    for i in range(20):
        h = extract_hidden(model, processor, Image.fromarray(create_highway(i + 1000)), prompt)
        if h is not None:
            diff_hw_embeds.append(h)
    diff_hw_embeds = np.array(diff_hw_embeds)

    diff_hw_pairwise = []
    for i in range(len(diff_hw_embeds)):
        for j in range(i+1, len(diff_hw_embeds)):
            diff_hw_pairwise.append(cosine_dist(diff_hw_embeds[i], diff_hw_embeds[j]))

    print(f"  Pairwise cosine distances: mean={np.mean(diff_hw_pairwise):.6f}, "
          f"max={np.max(diff_hw_pairwise):.6f}, min={np.min(diff_hw_pairwise):.6f}", flush=True)

    # Test 4: Cosine score stability
    print("\n--- Test 4: Score stability ---", flush=True)
    cal_embeds = diff_hw_embeds[:10]
    centroid = np.mean(cal_embeds, axis=0)

    hw_scores = [cosine_dist(e, centroid) for e in hw_embeds]
    noise_scores = [cosine_dist(e, centroid) for e in noise_embeds]

    print(f"  Highway repeated scores: mean={np.mean(hw_scores):.6f}, "
          f"std={np.std(hw_scores):.6f}, range=[{np.min(hw_scores):.6f}, {np.max(hw_scores):.6f}]", flush=True)
    print(f"  Noise repeated scores: mean={np.mean(noise_scores):.6f}, "
          f"std={np.std(noise_scores):.6f}, range=[{np.min(noise_scores):.6f}, {np.max(noise_scores):.6f}]", flush=True)

    # Bit-exact check
    identical_count = 0
    for i in range(1, len(hw_embeds)):
        if np.array_equal(hw_embeds[0], hw_embeds[i]):
            identical_count += 1

    print(f"\n  Bit-exact matches (highway): {identical_count}/{len(hw_embeds)-1}", flush=True)

    identical_noise = 0
    for i in range(1, len(noise_embeds)):
        if np.array_equal(noise_embeds[0], noise_embeds[i]):
            identical_noise += 1

    print(f"  Bit-exact matches (noise): {identical_noise}/{len(noise_embeds)-1}", flush=True)

    # Per-dimension variance
    hw_dim_var = np.var(hw_embeds, axis=0)
    noise_dim_var = np.var(noise_embeds, axis=0)
    diff_dim_var = np.var(diff_hw_embeds, axis=0)

    print(f"\n  Per-dim variance (same hw): mean={np.mean(hw_dim_var):.8f}, max={np.max(hw_dim_var):.8f}", flush=True)
    print(f"  Per-dim variance (same noise): mean={np.mean(noise_dim_var):.8f}, max={np.max(noise_dim_var):.8f}", flush=True)
    print(f"  Per-dim variance (diff hw): mean={np.mean(diff_dim_var):.8f}, max={np.max(diff_dim_var):.8f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'inference_stability',
        'experiment_number': 107,
        'timestamp': timestamp,
        'same_highway': {
            'n_passes': len(hw_embeds),
            'pairwise_cosine': {
                'mean': float(np.mean(hw_pairwise)),
                'max': float(np.max(hw_pairwise)),
                'min': float(np.min(hw_pairwise)),
                'std': float(np.std(hw_pairwise)),
            },
            'bit_exact_matches': identical_count,
            'per_dim_variance_mean': float(np.mean(hw_dim_var)),
            'per_dim_variance_max': float(np.max(hw_dim_var)),
            'score_mean': float(np.mean(hw_scores)),
            'score_std': float(np.std(hw_scores)),
            'score_range': [float(np.min(hw_scores)), float(np.max(hw_scores))],
        },
        'same_noise': {
            'n_passes': len(noise_embeds),
            'pairwise_cosine': {
                'mean': float(np.mean(noise_pairwise)),
                'max': float(np.max(noise_pairwise)),
                'min': float(np.min(noise_pairwise)),
                'std': float(np.std(noise_pairwise)),
            },
            'bit_exact_matches': identical_noise,
            'per_dim_variance_mean': float(np.mean(noise_dim_var)),
            'per_dim_variance_max': float(np.max(noise_dim_var)),
            'score_mean': float(np.mean(noise_scores)),
            'score_std': float(np.std(noise_scores)),
            'score_range': [float(np.min(noise_scores)), float(np.max(noise_scores))],
        },
        'different_highway': {
            'n_images': len(diff_hw_embeds),
            'pairwise_cosine': {
                'mean': float(np.mean(diff_hw_pairwise)),
                'max': float(np.max(diff_hw_pairwise)),
                'min': float(np.min(diff_hw_pairwise)),
            },
            'per_dim_variance_mean': float(np.mean(diff_dim_var)),
            'per_dim_variance_max': float(np.max(diff_dim_var)),
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"inference_stability_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
