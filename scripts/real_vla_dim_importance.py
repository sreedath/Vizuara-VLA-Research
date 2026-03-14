"""
Embedding Dimension Importance for OOD Detection.

Identifies which dimensions of the 4096-d hidden state carry the
strongest OOD signal.  Tests individual-dimension AUROC, cumulative
ablation (masking top-k most important dims), and random subspace
detection.

Experiment 123 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 16001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 16002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 16003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 16004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 16010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 16014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
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


def cosine_dist_masked(a, b, mask):
    """Cosine distance using only dimensions where mask=True."""
    am = a[mask]
    bm = b[mask]
    return float(1 - np.dot(am, bm) / (np.linalg.norm(am) * np.linalg.norm(bm) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("EMBEDDING DIMENSION IMPORTANCE", flush=True)
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

    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(15):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 2300)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}

    # Split into cal/test
    cal_embeds = []
    test_embeds = []
    test_labels = []

    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            cal_embeds.extend(data['embeds'][:8])
            for e in data['embeds'][8:]:
                test_embeds.append(e)
                test_labels.append(0)
        else:
            for e in data['embeds']:
                test_embeds.append(e)
                test_labels.append(1)

    cal_embeds = np.array(cal_embeds)
    test_embeds = np.array(test_embeds)
    test_labels = np.array(test_labels)
    dim = cal_embeds.shape[1]

    centroid = np.mean(cal_embeds, axis=0)
    print(f"\nCal: {len(cal_embeds)}, Test: {len(test_labels)}, Dim: {dim}", flush=True)

    # Baseline
    base_scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
    base_auroc = float(roc_auc_score(test_labels, base_scores))
    id_base = base_scores[test_labels == 0]
    ood_base = base_scores[test_labels == 1]
    base_d = float((np.mean(ood_base) - np.mean(id_base)) / (np.std(id_base) + 1e-10))
    print(f"Baseline: AUROC={base_auroc:.4f}, d={base_d:.2f}", flush=True)

    # 1. Per-dimension discriminability
    print("\n--- Per-Dimension Analysis ---", flush=True)
    dim_scores = np.zeros(dim)
    for d_idx in range(dim):
        id_vals = test_embeds[test_labels == 0, d_idx]
        ood_vals = test_embeds[test_labels == 1, d_idx]
        id_std = np.std(id_vals)
        if id_std > 1e-10:
            dim_scores[d_idx] = abs(np.mean(ood_vals) - np.mean(id_vals)) / id_std
        else:
            dim_scores[d_idx] = 0.0

    ranked_dims = np.argsort(dim_scores)[::-1]
    print(f"Top-10 dimensions (by |d|):", flush=True)
    top_dims_info = []
    for rank, d_idx in enumerate(ranked_dims[:10]):
        print(f"  Rank {rank}: dim {d_idx}, |d|={dim_scores[d_idx]:.2f}", flush=True)
        top_dims_info.append({'rank': rank, 'dim': int(d_idx), 'abs_d': float(dim_scores[d_idx])})

    # Distribution of per-dim discriminability
    pct_above_1 = float(np.mean(dim_scores > 1.0) * 100)
    pct_above_3 = float(np.mean(dim_scores > 3.0) * 100)
    pct_above_5 = float(np.mean(dim_scores > 5.0) * 100)
    print(f"\nDims with |d|>1: {pct_above_1:.1f}%", flush=True)
    print(f"Dims with |d|>3: {pct_above_3:.1f}%", flush=True)
    print(f"Dims with |d|>5: {pct_above_5:.1f}%", flush=True)

    # 2. Subspace detection: use only top-k dimensions
    print("\n--- Top-K Subspace Detection ---", flush=True)
    subspace_results = {}
    for k in [10, 50, 100, 256, 512, 1024, 2048]:
        if k > dim:
            continue
        mask = np.zeros(dim, dtype=bool)
        mask[ranked_dims[:k]] = True
        sub_scores = np.array([cosine_dist_masked(e, centroid, mask) for e in test_embeds])
        sub_auroc = float(roc_auc_score(test_labels, sub_scores))
        id_sub = sub_scores[test_labels == 0]
        ood_sub = sub_scores[test_labels == 1]
        sub_d = float((np.mean(ood_sub) - np.mean(id_sub)) / (np.std(id_sub) + 1e-10))
        print(f"  Top-{k:4d}: AUROC={sub_auroc:.4f}, d={sub_d:.2f}", flush=True)
        subspace_results[k] = {'auroc': sub_auroc, 'd': sub_d}

    # 3. Bottom-K dimensions (least discriminative)
    print("\n--- Bottom-K Subspace Detection ---", flush=True)
    bottom_results = {}
    for k in [10, 50, 100, 256, 512, 1024, 2048]:
        if k > dim:
            continue
        mask = np.zeros(dim, dtype=bool)
        mask[ranked_dims[-k:]] = True
        bot_scores = np.array([cosine_dist_masked(e, centroid, mask) for e in test_embeds])
        bot_auroc = float(roc_auc_score(test_labels, bot_scores))
        id_bot = bot_scores[test_labels == 0]
        ood_bot = bot_scores[test_labels == 1]
        bot_d = float((np.mean(ood_bot) - np.mean(id_bot)) / (np.std(id_bot) + 1e-10))
        print(f"  Bottom-{k:4d}: AUROC={bot_auroc:.4f}, d={bot_d:.2f}", flush=True)
        bottom_results[k] = {'auroc': bot_auroc, 'd': bot_d}

    # 4. Random subspace detection (5 trials each)
    print("\n--- Random Subspace Detection ---", flush=True)
    rng = np.random.default_rng(42)
    random_results = {}
    for k in [10, 50, 100, 256, 512, 1024, 2048]:
        if k > dim:
            continue
        trial_aurocs = []
        trial_ds = []
        for trial in range(5):
            rand_dims = rng.choice(dim, k, replace=False)
            mask = np.zeros(dim, dtype=bool)
            mask[rand_dims] = True
            rand_scores = np.array([cosine_dist_masked(e, centroid, mask) for e in test_embeds])
            rand_auroc = float(roc_auc_score(test_labels, rand_scores))
            id_rand = rand_scores[test_labels == 0]
            ood_rand = rand_scores[test_labels == 1]
            rand_d = float((np.mean(ood_rand) - np.mean(id_rand)) / (np.std(id_rand) + 1e-10))
            trial_aurocs.append(rand_auroc)
            trial_ds.append(rand_d)
        mean_auroc = float(np.mean(trial_aurocs))
        mean_d = float(np.mean(trial_ds))
        print(f"  Random-{k:4d}: AUROC={mean_auroc:.4f}+/-{np.std(trial_aurocs):.4f}, "
              f"d={mean_d:.2f}+/-{np.std(trial_ds):.2f}", flush=True)
        random_results[k] = {
            'auroc_mean': mean_auroc,
            'auroc_std': float(np.std(trial_aurocs)),
            'd_mean': mean_d,
            'd_std': float(np.std(trial_ds)),
        }

    # 5. Ablation: remove top-k dims and measure remaining detection
    print("\n--- Ablation: Remove Top-K Dims ---", flush=True)
    ablation_results = {}
    for k in [10, 50, 100, 256, 512, 1024, 2048]:
        if k >= dim:
            continue
        mask = np.ones(dim, dtype=bool)
        mask[ranked_dims[:k]] = False
        abl_scores = np.array([cosine_dist_masked(e, centroid, mask) for e in test_embeds])
        abl_auroc = float(roc_auc_score(test_labels, abl_scores))
        id_abl = abl_scores[test_labels == 0]
        ood_abl = abl_scores[test_labels == 1]
        abl_d = float((np.mean(ood_abl) - np.mean(id_abl)) / (np.std(id_abl) + 1e-10))
        print(f"  Remove top-{k:4d}: AUROC={abl_auroc:.4f}, d={abl_d:.2f}", flush=True)
        ablation_results[k] = {'auroc': abl_auroc, 'd': abl_d}

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'dim_importance',
        'experiment_number': 123,
        'timestamp': timestamp,
        'dim': dim,
        'n_cal': len(cal_embeds),
        'n_test': len(test_labels),
        'baseline': {'auroc': base_auroc, 'd': base_d},
        'top_dims': top_dims_info,
        'dim_discriminability': {
            'pct_above_1': pct_above_1,
            'pct_above_3': pct_above_3,
            'pct_above_5': pct_above_5,
            'max_d': float(dim_scores.max()),
            'mean_d': float(dim_scores.mean()),
            'median_d': float(np.median(dim_scores)),
        },
        'top_k_subspace': {str(k): v for k, v in subspace_results.items()},
        'bottom_k_subspace': {str(k): v for k, v in bottom_results.items()},
        'random_subspace': {str(k): v for k, v in random_results.items()},
        'ablation_remove_top_k': {str(k): v for k, v in ablation_results.items()},
    }
    output_path = os.path.join(RESULTS_DIR, f"dim_importance_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
