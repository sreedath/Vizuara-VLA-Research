"""
Feature Importance via Dimension Ablation.

Tests which groups of hidden state dimensions are most important for
OOD detection by masking (zeroing) dimension groups and measuring the
impact on separation. Also tests random subsets to establish baselines.

Experiment 105 in the CalibDrive series.
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


def extract_hidden(model, processor, image, prompt):
    """Extract last hidden state."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    """Cosine distance between two vectors."""
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def evaluate_dims(cal_embeds, test_embeds, test_labels, mask):
    """Evaluate detection using only dimensions specified by mask."""
    cal_masked = cal_embeds[:, mask]
    test_masked = test_embeds[:, mask]
    centroid = np.mean(cal_masked, axis=0)
    scores = [cosine_dist(e, centroid) for e in test_masked]
    auroc = roc_auc_score(test_labels, scores)
    id_scores = [s for s, l in zip(scores, test_labels) if l == 0]
    ood_scores = [s for s, l in zip(scores, test_labels) if l == 1]
    d = (np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10)
    return float(auroc), float(d), float(np.mean(id_scores)), float(np.mean(ood_scores))


def main():
    print("=" * 70, flush=True)
    print("FEATURE IMPORTANCE VIA DIMENSION ABLATION", flush=True)
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

    # Collect embeddings
    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(15):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 800)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}
        print(f"    Collected {len(embeds)} embeddings", flush=True)

    # Build cal/test sets
    cal_embeds = []
    test_embeds = []
    test_labels = []
    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            cal_embeds.append(data['embeds'][:10])
            for e in data['embeds'][10:]:
                test_embeds.append(e)
                test_labels.append(0)
        else:
            for e in data['embeds']:
                test_embeds.append(e)
                test_labels.append(1)

    cal_embeds = np.concatenate(cal_embeds, axis=0)
    test_embeds = np.array(test_embeds)
    test_labels = np.array(test_labels)
    dim = cal_embeds.shape[1]

    print(f"\nDim: {dim}, Cal: {cal_embeds.shape[0]}, Test: {len(test_labels)}", flush=True)

    # Baseline: all dimensions
    full_mask = np.ones(dim, dtype=bool)
    base_auroc, base_d, base_id, base_ood = evaluate_dims(cal_embeds, test_embeds, test_labels, full_mask)
    print(f"\nBaseline (all {dim} dims): AUROC={base_auroc:.4f}, d={base_d:.2f}", flush=True)

    # Test 1: Block ablation — divide into 16 blocks of 256 dims
    print("\n--- Block Ablation (16 blocks × 256 dims) ---", flush=True)
    n_blocks = 16
    block_size = dim // n_blocks
    block_results = {}
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        # Remove this block
        mask = np.ones(dim, dtype=bool)
        mask[start:end] = False
        auroc, d, id_m, ood_m = evaluate_dims(cal_embeds, test_embeds, test_labels, mask)
        # Also test with only this block
        only_mask = np.zeros(dim, dtype=bool)
        only_mask[start:end] = True
        auroc_only, d_only, id_only, ood_only = evaluate_dims(cal_embeds, test_embeds, test_labels, only_mask)

        block_results[f"block_{b}"] = {
            'start': start, 'end': end,
            'without_auroc': auroc, 'without_d': d,
            'only_auroc': auroc_only, 'only_d': d_only,
            'importance': base_d - d,  # Drop in d when removed
        }
        print(f"  Block {b} [{start}:{end}]: without={auroc:.4f}/d={d:.2f}, only={auroc_only:.4f}/d={d_only:.2f}, importance={base_d-d:.2f}", flush=True)

    # Test 2: Top-k dimensions by variance difference
    print("\n--- Variance-Based Feature Selection ---", flush=True)
    cal_var = np.var(cal_embeds, axis=0)
    all_embeds = np.concatenate([cal_embeds, test_embeds[test_labels == 1]], axis=0)
    ood_embeds = test_embeds[test_labels == 1]
    ood_var = np.var(ood_embeds, axis=0)
    var_diff = np.abs(cal_var - ood_var)
    sorted_dims = np.argsort(var_diff)[::-1]

    variance_results = {}
    for n_dims in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        if n_dims > dim:
            continue
        mask = np.zeros(dim, dtype=bool)
        mask[sorted_dims[:n_dims]] = True
        auroc, d, id_m, ood_m = evaluate_dims(cal_embeds, test_embeds, test_labels, mask)
        variance_results[f"top_{n_dims}"] = {
            'n_dims': n_dims, 'auroc': auroc, 'd': d,
        }
        print(f"  Top-{n_dims} variance dims: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Test 3: Random subsets (average over 5 seeds)
    print("\n--- Random Dimension Subsets ---", flush=True)
    random_results = {}
    for n_dims in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        if n_dims > dim:
            continue
        aurocs = []
        ds = []
        for seed in range(5):
            rng = np.random.default_rng(seed * 42 + n_dims)
            selected = rng.choice(dim, n_dims, replace=False)
            mask = np.zeros(dim, dtype=bool)
            mask[selected] = True
            auroc, d, _, _ = evaluate_dims(cal_embeds, test_embeds, test_labels, mask)
            aurocs.append(auroc)
            ds.append(d)
        random_results[f"random_{n_dims}"] = {
            'n_dims': n_dims,
            'auroc_mean': float(np.mean(aurocs)),
            'auroc_std': float(np.std(aurocs)),
            'd_mean': float(np.mean(ds)),
            'd_std': float(np.std(ds)),
        }
        print(f"  Random {n_dims} dims: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}, d={np.mean(ds):.2f}±{np.std(ds):.2f}", flush=True)

    # Test 4: Mean-difference based selection
    print("\n--- Mean-Difference Feature Selection ---", flush=True)
    id_mean = np.mean(cal_embeds, axis=0)
    ood_mean = np.mean(ood_embeds, axis=0)
    mean_diff = np.abs(id_mean - ood_mean)
    sorted_mean_dims = np.argsort(mean_diff)[::-1]

    mean_diff_results = {}
    for n_dims in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        if n_dims > dim:
            continue
        mask = np.zeros(dim, dtype=bool)
        mask[sorted_mean_dims[:n_dims]] = True
        auroc, d, id_m, ood_m = evaluate_dims(cal_embeds, test_embeds, test_labels, mask)
        mean_diff_results[f"meandiff_{n_dims}"] = {
            'n_dims': n_dims, 'auroc': auroc, 'd': d,
        }
        print(f"  Top-{n_dims} mean-diff dims: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'feature_importance',
        'experiment_number': 105,
        'timestamp': timestamp,
        'dim': dim,
        'baseline': {'auroc': base_auroc, 'd': base_d, 'id_mean': base_id, 'ood_mean': base_ood},
        'block_ablation': block_results,
        'variance_selection': variance_results,
        'random_subsets': random_results,
        'mean_diff_selection': mean_diff_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"feature_importance_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
