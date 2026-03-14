"""
Action Token Vocabulary Analysis.

Examines the full 256-bin action token distribution for ID vs OOD:
which tokens are used, how concentrated the distribution is, whether
certain tokens are ID-specific or OOD-specific, and the entropy
structure across action dimensions.

Experiment 111 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from scipy import stats as scipy_stats

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

def create_snow(idx):
    rng = np.random.default_rng(idx * 5014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def get_action_tokens(model, processor, image, prompt, n_tokens=7):
    """Generate action tokens and return their IDs."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=n_tokens, do_sample=False,
        )
    # Get just the generated tokens
    gen_tokens = out[0, inputs['input_ids'].shape[1]:]
    return gen_tokens.cpu().numpy()


def main():
    print("=" * 70, flush=True)
    print("ACTION TOKEN VOCABULARY ANALYSIS", flush=True)
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
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    print("\n--- Generating action tokens ---", flush=True)
    all_tokens = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        tokens_list = []
        for i in range(15):
            tokens = get_action_tokens(model, processor, Image.fromarray(fn(i + 1300)), prompt)
            tokens_list.append(tokens.tolist())
            if i == 0:
                print(f"    First tokens: {tokens}", flush=True)
        all_tokens[cat_name] = {'tokens': tokens_list, 'group': group}

    # Analyze token distributions
    print("\n--- Token Distribution Analysis ---", flush=True)
    id_tokens = []
    ood_tokens = []
    for cat_name, data in all_tokens.items():
        if data['group'] == 'ID':
            id_tokens.extend(data['tokens'])
        else:
            ood_tokens.extend(data['tokens'])

    id_tokens = np.array(id_tokens)
    ood_tokens = np.array(ood_tokens)

    # Per-dimension analysis
    n_dims = min(id_tokens.shape[1], ood_tokens.shape[1])
    dim_results = {}
    for d in range(n_dims):
        id_vals = id_tokens[:, d]
        ood_vals = ood_tokens[:, d]

        # Unique tokens
        id_unique = set(id_vals.tolist())
        ood_unique = set(ood_vals.tolist())

        # Token overlap
        overlap = id_unique & ood_unique
        id_only = id_unique - ood_unique
        ood_only = ood_unique - id_unique

        # Entropy
        id_counts = np.bincount(id_vals, minlength=32768)
        id_probs = id_counts[id_counts > 0] / id_counts.sum()
        id_entropy = float(scipy_stats.entropy(id_probs, base=2))

        ood_counts = np.bincount(ood_vals, minlength=32768)
        ood_probs = ood_counts[ood_counts > 0] / ood_counts.sum()
        ood_entropy = float(scipy_stats.entropy(ood_probs, base=2))

        dim_results[f'dim_{d}'] = {
            'id_unique': len(id_unique),
            'ood_unique': len(ood_unique),
            'overlap': len(overlap),
            'id_only': len(id_only),
            'ood_only': len(ood_only),
            'id_entropy': id_entropy,
            'ood_entropy': ood_entropy,
            'id_mode': int(scipy_stats.mode(id_vals, keepdims=False).mode),
            'ood_mode': int(scipy_stats.mode(ood_vals, keepdims=False).mode),
            'id_mean': float(np.mean(id_vals)),
            'ood_mean': float(np.mean(ood_vals)),
        }
        print(f"  Dim {d}: ID unique={len(id_unique)}, OOD unique={len(ood_unique)}, "
              f"overlap={len(overlap)}, ID_entropy={id_entropy:.2f}, OOD_entropy={ood_entropy:.2f}", flush=True)

    # Per-category token consistency
    print("\n--- Per-Category Token Consistency ---", flush=True)
    category_consistency = {}
    for cat_name, data in all_tokens.items():
        tokens_arr = np.array(data['tokens'])
        consistency = []
        for d in range(n_dims):
            vals = tokens_arr[:, d]
            mode_count = np.max(np.bincount(vals))
            consistency.append(float(mode_count / len(vals)))
        category_consistency[cat_name] = {
            'mean_consistency': float(np.mean(consistency)),
            'per_dim_consistency': consistency,
            'group': data['group'],
        }
        print(f"  {cat_name} ({data['group']}): mean consistency = {np.mean(consistency):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_vocabulary',
        'experiment_number': 111,
        'timestamp': timestamp,
        'n_dims': n_dims,
        'n_id': len(id_tokens),
        'n_ood': len(ood_tokens),
        'dim_results': dim_results,
        'category_consistency': category_consistency,
        'sample_tokens': {k: v['tokens'][:3] for k, v in all_tokens.items()},
    }
    output_path = os.path.join(RESULTS_DIR, f"action_vocabulary_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
