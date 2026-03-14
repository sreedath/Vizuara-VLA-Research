"""
Token Position Analysis.

Tests which token position in the hidden state sequence carries the
most OOD-discriminative information: first token, last token, mean
pooling, or max pooling across all positions.

Experiment 109 in the CalibDrive series.
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


def extract_all_positions(model, processor, image, prompt):
    """Extract hidden states at all token positions from last layer."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    # Full sequence from last layer: [1, seq_len, hidden_dim]
    full = fwd.hidden_states[-1][0].float().cpu().numpy()
    return full  # shape: (seq_len, hidden_dim)


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("TOKEN POSITION ANALYSIS", flush=True)
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

    # Collect full sequence hidden states
    print("\n--- Collecting full-sequence hidden states ---", flush=True)
    all_sequences = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        seqs = []
        for i in range(10):
            full = extract_all_positions(model, processor, Image.fromarray(fn(i + 1100)), prompt)
            if full is not None:
                seqs.append(full)
                if i == 0:
                    print(f"    Sequence length: {full.shape[0]}, dim: {full.shape[1]}", flush=True)
        all_sequences[cat_name] = {'seqs': seqs, 'group': group}

    # Define pooling strategies
    def pool_last(seq):
        return seq[-1]

    def pool_first(seq):
        return seq[0]

    def pool_mean(seq):
        return np.mean(seq, axis=0)

    def pool_max(seq):
        return np.max(seq, axis=0)

    def pool_second_last(seq):
        return seq[-2] if len(seq) > 1 else seq[-1]

    def pool_quarter(seq):
        """Mean of last quarter of positions."""
        n = max(1, len(seq) // 4)
        return np.mean(seq[-n:], axis=0)

    def pool_mid(seq):
        mid = len(seq) // 2
        return seq[mid]

    pooling_methods = {
        'last': pool_last,
        'first': pool_first,
        'mean': pool_mean,
        'max': pool_max,
        'second_last': pool_second_last,
        'last_quarter_mean': pool_quarter,
        'middle': pool_mid,
    }

    # Evaluate each pooling strategy
    print("\n--- Evaluating Pooling Strategies ---", flush=True)
    results = {}
    for pool_name, pool_fn in pooling_methods.items():
        # Build cal/test
        cal_embeds = []
        test_embeds = []
        test_labels = []

        for cat_name, data in all_sequences.items():
            pooled = [pool_fn(s) for s in data['seqs']]
            if data['group'] == 'ID':
                cal_embeds.extend(pooled[:5])
                for e in pooled[5:]:
                    test_embeds.append(e)
                    test_labels.append(0)
            else:
                for e in pooled:
                    test_embeds.append(e)
                    test_labels.append(1)

        cal_embeds = np.array(cal_embeds)
        test_embeds_arr = np.array(test_embeds)
        test_labels_arr = np.array(test_labels)

        centroid = np.mean(cal_embeds, axis=0)
        scores = [cosine_dist(e, centroid) for e in test_embeds_arr]
        auroc = roc_auc_score(test_labels_arr, scores)

        id_scores = [s for s, l in zip(scores, test_labels_arr) if l == 0]
        ood_scores = [s for s, l in zip(scores, test_labels_arr) if l == 1]
        d = (np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10)

        results[pool_name] = {
            'auroc': float(auroc),
            'd': float(d),
            'id_mean': float(np.mean(id_scores)),
            'ood_mean': float(np.mean(ood_scores)),
            'id_std': float(np.std(id_scores)),
            'ood_std': float(np.std(ood_scores)),
        }
        print(f"  {pool_name:20}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Position-by-position analysis on a subset
    print("\n--- Position-by-Position Analysis ---", flush=True)
    seq_len = all_sequences['highway']['seqs'][0].shape[0]
    print(f"  Sequence length: {seq_len}", flush=True)

    # Sample every 10th position
    position_results = {}
    positions = list(range(0, seq_len, max(1, seq_len // 20)))
    if seq_len - 1 not in positions:
        positions.append(seq_len - 1)

    for pos in positions:
        cal_embeds = []
        test_embeds = []
        test_labels = []

        for cat_name, data in all_sequences.items():
            pos_embeds = [s[pos] for s in data['seqs']]
            if data['group'] == 'ID':
                cal_embeds.extend(pos_embeds[:5])
                for e in pos_embeds[5:]:
                    test_embeds.append(e)
                    test_labels.append(0)
            else:
                for e in pos_embeds:
                    test_embeds.append(e)
                    test_labels.append(1)

        cal_embeds = np.array(cal_embeds)
        test_embeds_arr = np.array(test_embeds)
        centroid = np.mean(cal_embeds, axis=0)
        scores = [cosine_dist(e, centroid) for e in test_embeds_arr]
        auroc = roc_auc_score(test_labels, scores)

        id_scores = [s for s, l in zip(scores, test_labels) if l == 0]
        ood_scores = [s for s, l in zip(scores, test_labels) if l == 1]
        d = (np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10)

        position_results[pos] = {
            'auroc': float(auroc),
            'd': float(d),
        }
        print(f"  pos={pos:4d}/{seq_len}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'token_position_analysis',
        'experiment_number': 109,
        'timestamp': timestamp,
        'sequence_length': seq_len,
        'pooling_results': results,
        'position_results': {str(k): v for k, v in position_results.items()},
    }
    output_path = os.path.join(RESULTS_DIR, f"token_position_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
