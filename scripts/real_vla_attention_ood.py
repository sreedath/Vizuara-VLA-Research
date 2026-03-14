"""
Attention Pattern OOD Analysis.

Extracts attention weights from the last layer and analyzes how
attention distributions differ between ID and OOD inputs. Tests
whether attention entropy, attention to image tokens, and attention
concentration can serve as OOD signals.

Experiment 117 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 11001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 11002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 11003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 11004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 11010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 11014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_attention(model, processor, image, prompt):
    """Extract attention from last layer and hidden state."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True, output_attentions=True)

    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None

    hidden = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Attention from last layer
    if hasattr(fwd, 'attentions') and fwd.attentions:
        # Shape: (1, n_heads, seq_len, seq_len)
        last_attn = fwd.attentions[-1][0].float().cpu().numpy()
        # Get last token's attention pattern
        last_token_attn = last_attn[:, -1, :]  # (n_heads, seq_len)
    else:
        last_token_attn = None

    return {
        'hidden': hidden,
        'attention': last_token_attn,
        'seq_len': last_token_attn.shape[-1] if last_token_attn is not None else 0,
    }


def attention_entropy(attn_row):
    """Shannon entropy of attention distribution."""
    attn_row = attn_row + 1e-10
    return float(-np.sum(attn_row * np.log(attn_row)))


def main():
    print("=" * 70, flush=True)
    print("ATTENTION PATTERN OOD ANALYSIS", flush=True)
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

    print("\n--- Collecting attention patterns ---", flush=True)
    all_data = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        samples = []
        for i in range(10):
            result = extract_attention(model, processor, Image.fromarray(fn(i + 1800)), prompt)
            if result is not None and result['attention'] is not None:
                samples.append(result)
        all_data[cat_name] = {'samples': samples, 'group': group}
        print(f"    {len(samples)} samples, seq_len={samples[0]['seq_len']}, "
              f"n_heads={samples[0]['attention'].shape[0]}", flush=True)

    n_heads = all_data['highway']['samples'][0]['attention'].shape[0]
    seq_len = all_data['highway']['samples'][0]['seq_len']
    print(f"\nn_heads={n_heads}, seq_len={seq_len}", flush=True)

    # Compute attention-based features for each sample
    print("\n--- Computing attention features ---", flush=True)
    features = {}
    for cat_name, data in all_data.items():
        cat_features = []
        for sample in data['samples']:
            attn = sample['attention']  # (n_heads, seq_len)

            # Per-head entropy
            head_entropies = [attention_entropy(attn[h]) for h in range(n_heads)]

            # Mean attention entropy across heads
            mean_entropy = np.mean(head_entropies)

            # Max attention weight (concentration)
            max_attn = float(np.max(attn))
            mean_max_attn = float(np.mean([np.max(attn[h]) for h in range(n_heads)]))

            # Attention to first vs last quarter of sequence
            q1 = seq_len // 4
            first_q_attn = float(np.mean([np.sum(attn[h, :q1]) for h in range(n_heads)]))
            last_q_attn = float(np.mean([np.sum(attn[h, -q1:]) for h in range(n_heads)]))

            # Top-5 attention concentration
            top5_attn = float(np.mean([np.sort(attn[h])[-5:].sum() for h in range(n_heads)]))

            cat_features.append({
                'mean_entropy': mean_entropy,
                'head_entropies': head_entropies,
                'max_attn': max_attn,
                'mean_max_attn': mean_max_attn,
                'first_q_attn': first_q_attn,
                'last_q_attn': last_q_attn,
                'top5_concentration': top5_attn,
                'hidden': sample['hidden'],
            })
        features[cat_name] = {'features': cat_features, 'group': data['group']}

    # Evaluate attention-based OOD detectors
    print("\n--- Attention-based OOD detection ---", flush=True)
    cal_feats = []
    test_feats = []
    test_labels = []

    for cat_name, data in features.items():
        if data['group'] == 'ID':
            cal_feats.extend(data['features'][:5])
            for f in data['features'][5:]:
                test_feats.append(f)
                test_labels.append(0)
        else:
            for f in data['features']:
                test_feats.append(f)
                test_labels.append(1)

    test_labels = np.array(test_labels)

    # Cosine distance baseline
    from functools import reduce
    cal_centroid = np.mean([f['hidden'] for f in cal_feats], axis=0)
    cos_scores = np.array([float(1 - np.dot(f['hidden'], cal_centroid) /
                                 (np.linalg.norm(f['hidden']) * np.linalg.norm(cal_centroid) + 1e-10))
                           for f in test_feats])
    cos_auroc = float(roc_auc_score(test_labels, cos_scores))

    # Attention entropy detector
    ent_scores = np.array([f['mean_entropy'] for f in test_feats])
    ent_auroc_pos = float(roc_auc_score(test_labels, ent_scores))
    ent_auroc_neg = float(roc_auc_score(test_labels, -ent_scores))
    ent_auroc = max(ent_auroc_pos, ent_auroc_neg)

    # Max attention detector
    max_scores = np.array([f['mean_max_attn'] for f in test_feats])
    max_auroc_pos = float(roc_auc_score(test_labels, max_scores))
    max_auroc_neg = float(roc_auc_score(test_labels, -max_scores))
    max_auroc = max(max_auroc_pos, max_auroc_neg)

    # Top5 concentration
    top5_scores = np.array([f['top5_concentration'] for f in test_feats])
    top5_auroc_pos = float(roc_auc_score(test_labels, top5_scores))
    top5_auroc_neg = float(roc_auc_score(test_labels, -top5_scores))
    top5_auroc = max(top5_auroc_pos, top5_auroc_neg)

    # First-quarter attention
    fq_scores = np.array([f['first_q_attn'] for f in test_feats])
    fq_auroc_pos = float(roc_auc_score(test_labels, fq_scores))
    fq_auroc_neg = float(roc_auc_score(test_labels, -fq_scores))
    fq_auroc = max(fq_auroc_pos, fq_auroc_neg)

    detector_results = {
        'cosine_baseline': {'auroc': cos_auroc},
        'attention_entropy': {'auroc': ent_auroc},
        'max_attention': {'auroc': max_auroc},
        'top5_concentration': {'auroc': top5_auroc},
        'first_quarter_attn': {'auroc': fq_auroc},
    }
    for name, res in detector_results.items():
        print(f"  {name}: AUROC={res['auroc']:.4f}", flush=True)

    # Per-category attention stats
    print("\n--- Per-Category Attention Stats ---", flush=True)
    per_cat_stats = {}
    for cat_name, data in features.items():
        entropies = [f['mean_entropy'] for f in data['features']]
        max_attns = [f['mean_max_attn'] for f in data['features']]
        top5s = [f['top5_concentration'] for f in data['features']]
        first_qs = [f['first_q_attn'] for f in data['features']]
        per_cat_stats[cat_name] = {
            'group': data['group'],
            'entropy_mean': float(np.mean(entropies)),
            'entropy_std': float(np.std(entropies)),
            'max_attn_mean': float(np.mean(max_attns)),
            'top5_mean': float(np.mean(top5s)),
            'first_q_mean': float(np.mean(first_qs)),
        }
        print(f"  {cat_name}: entropy={np.mean(entropies):.3f}, "
              f"max_attn={np.mean(max_attns):.4f}, "
              f"top5={np.mean(top5s):.4f}, "
              f"first_q={np.mean(first_qs):.4f}", flush=True)

    # Per-head AUROC
    print("\n--- Per-Head AUROC (top/bottom 5) ---", flush=True)
    head_aurocs = []
    for h in range(n_heads):
        h_scores = np.array([f['head_entropies'][h] for f in test_feats])
        auroc_p = float(roc_auc_score(test_labels, h_scores))
        auroc_n = float(roc_auc_score(test_labels, -h_scores))
        head_aurocs.append(max(auroc_p, auroc_n))

    sorted_heads = np.argsort(head_aurocs)[::-1]
    for h in sorted_heads[:5]:
        print(f"  Head {h}: AUROC={head_aurocs[h]:.4f}", flush=True)
    print("  ...")
    for h in sorted_heads[-5:]:
        print(f"  Head {h}: AUROC={head_aurocs[h]:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'attention_ood',
        'experiment_number': 117,
        'timestamp': timestamp,
        'n_heads': n_heads,
        'seq_len': seq_len,
        'detector_results': detector_results,
        'per_category': per_cat_stats,
        'head_aurocs': head_aurocs,
    }
    output_path = os.path.join(RESULTS_DIR, f"attention_ood_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
