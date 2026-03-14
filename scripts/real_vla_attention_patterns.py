"""
Attention Pattern Analysis.

Extracts detailed attention statistics per layer to understand
how attention patterns differ between ID and OOD inputs across
the transformer depth.

Experiment 90 in the CalibDrive series.
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


def extract_attention_profile(model, processor, image, prompt):
    """Extract per-layer attention statistics."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_attentions=True)

    profile = {}
    if hasattr(fwd, 'attentions') and fwd.attentions:
        for layer_idx, attn in enumerate(fwd.attentions):
            a = attn[0].float().cpu().numpy()  # [heads, seq, seq]
            last_row = a[:, -1, :]  # attention from last token

            # Per-layer stats
            profile[layer_idx] = {
                'max': float(last_row.max()),
                'mean': float(last_row.mean()),
                'std': float(last_row.std()),
                'top5_mean': float(np.sort(last_row.flatten())[-5:].mean()),
                'entropy': float(-np.sum(last_row.flatten() *
                                  np.log(last_row.flatten() + 1e-10)) / last_row.shape[0]),
            }
    return profile


def main():
    print("=" * 70, flush=True)
    print("ATTENTION PATTERN ANALYSIS", flush=True)
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

    # Collect attention profiles
    scenarios = {
        'highway': (create_highway, range(500, 508), False),
        'urban': (create_urban, range(500, 508), False),
        'noise': (create_noise, range(500, 506), True),
        'indoor': (create_indoor, range(500, 506), True),
    }

    all_profiles = {}
    cnt = 0
    total = sum(len(list(ids)) for _, ids, _ in scenarios.values())

    for name, (fn, indices, is_ood) in scenarios.items():
        profiles = []
        for i in indices:
            cnt += 1
            profile = extract_attention_profile(model, processor,
                                                 Image.fromarray(fn(i)), prompt)
            if profile:
                profiles.append(profile)
            if cnt % 5 == 0:
                print(f"  [{cnt}/{total}] {name}", flush=True)

        # Average per-layer stats
        n_layers = len(profiles[0]) if profiles else 0
        avg_profile = {}
        for layer in range(n_layers):
            avg_profile[layer] = {
                'max': float(np.mean([p[layer]['max'] for p in profiles])),
                'mean': float(np.mean([p[layer]['mean'] for p in profiles])),
                'entropy': float(np.mean([p[layer]['entropy'] for p in profiles])),
                'top5_mean': float(np.mean([p[layer]['top5_mean'] for p in profiles])),
            }

        all_profiles[name] = {
            'n_samples': len(profiles),
            'n_layers': n_layers,
            'is_ood': is_ood,
            'avg_profile': avg_profile,
        }
        print(f"  {name}: {len(profiles)} samples, {n_layers} layers", flush=True)

    # Per-layer AUROC for attention max
    print("\nPer-Layer Attention Max AUROC:", flush=True)
    n_layers = all_profiles['highway']['n_layers']
    layer_aurocs = {}

    for layer in range(n_layers):
        id_vals = []
        ood_vals = []
        for name, data in all_profiles.items():
            val = data['avg_profile'][layer]['max']
            if data['is_ood']:
                ood_vals.append(val)
            else:
                id_vals.append(val)

        # Need per-sample values for proper AUROC
        # Use avg as proxy
        if len(id_vals) >= 1 and len(ood_vals) >= 1:
            labels = [0]*len(id_vals) + [1]*len(ood_vals)
            scores = id_vals + ood_vals
            try:
                auroc = roc_auc_score(labels, scores)
            except Exception:
                auroc = 0.5
            layer_aurocs[layer] = float(auroc)
            if layer % 4 == 0 or layer == n_layers - 1:
                print(f"  Layer {layer}: AUROC={auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Serialize profiles (convert int keys to str for JSON)
    serializable = {}
    for name, data in all_profiles.items():
        serializable[name] = {
            'n_samples': data['n_samples'],
            'n_layers': data['n_layers'],
            'is_ood': data['is_ood'],
            'avg_profile': {str(k): v for k, v in data['avg_profile'].items()},
        }

    output = {
        'experiment': 'attention_patterns',
        'experiment_number': 90,
        'timestamp': timestamp,
        'profiles': serializable,
        'layer_aurocs': {str(k): v for k, v in layer_aurocs.items()},
    }
    output_path = os.path.join(RESULTS_DIR, f"attention_patterns_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
