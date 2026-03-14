"""
Projection Head Analysis.

Tests whether linear/nonlinear projections of the hidden state can
improve OOD detection: PCA projection, random projection, LDA
projection, and variance-maximizing projection.

Experiment 110 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def evaluate(cal_embeds, test_embeds, test_labels):
    centroid = np.mean(cal_embeds, axis=0)
    scores = [cosine_dist(e, centroid) for e in test_embeds]
    auroc = roc_auc_score(test_labels, scores)
    id_scores = [s for s, l in zip(scores, test_labels) if l == 0]
    ood_scores = [s for s, l in zip(scores, test_labels) if l == 1]
    d = (np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10)
    return float(auroc), float(d)


def main():
    print("=" * 70, flush=True)
    print("PROJECTION HEAD ANALYSIS", flush=True)
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
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 1200)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}

    # Split: train (for projection fitting), test
    train_embeds = []
    train_labels = []
    cal_embeds = []
    test_embeds = []
    test_labels = []

    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            cal_embeds.append(data['embeds'][:5])
            for e in data['embeds'][:10]:
                train_embeds.append(e)
                train_labels.append(0)
            for e in data['embeds'][10:]:
                test_embeds.append(e)
                test_labels.append(0)
        else:
            for e in data['embeds'][:10]:
                train_embeds.append(e)
                train_labels.append(1)
            for e in data['embeds'][10:]:
                test_embeds.append(e)
                test_labels.append(1)

    cal_embeds = np.concatenate(cal_embeds, axis=0)
    train_embeds = np.array(train_embeds)
    train_labels = np.array(train_labels)
    test_embeds = np.array(test_embeds)
    test_labels = np.array(test_labels)

    dim = cal_embeds.shape[1]
    print(f"\nDim: {dim}, Cal: {cal_embeds.shape[0]}, Train: {len(train_labels)}, Test: {len(test_labels)}", flush=True)

    # Baseline: full dim
    base_auroc, base_d = evaluate(cal_embeds, test_embeds, test_labels)
    print(f"\nBaseline (full {dim}): AUROC={base_auroc:.4f}, d={base_d:.2f}", flush=True)

    results = {'baseline': {'auroc': base_auroc, 'd': base_d, 'dims': dim}}

    # PCA projections
    print("\n--- PCA Projection ---", flush=True)
    max_comp = min(train_embeds.shape[0], train_embeds.shape[1]) - 1
    for n_comp in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if n_comp > max_comp:
            continue
        pca = PCA(n_components=n_comp)
        pca.fit(train_embeds)
        cal_proj = pca.transform(cal_embeds)
        test_proj = pca.transform(test_embeds)
        auroc, d = evaluate(cal_proj, test_proj, test_labels)
        results[f'pca_{n_comp}'] = {'auroc': auroc, 'd': d, 'dims': n_comp,
                                     'explained_var': float(sum(pca.explained_variance_ratio_))}
        print(f"  PCA-{n_comp}: AUROC={auroc:.4f}, d={d:.2f}, "
              f"var_explained={sum(pca.explained_variance_ratio_):.4f}", flush=True)

    # Random projection
    print("\n--- Random Projection ---", flush=True)
    for n_comp in [8, 16, 32, 64, 128, 256]:
        aurocs = []
        ds = []
        for seed in range(5):
            rng = np.random.default_rng(seed * 100 + n_comp)
            proj = rng.standard_normal((dim, n_comp)) / np.sqrt(n_comp)
            cal_proj = cal_embeds @ proj
            test_proj = test_embeds @ proj
            auroc, d = evaluate(cal_proj, test_proj, test_labels)
            aurocs.append(auroc)
            ds.append(d)
        results[f'random_{n_comp}'] = {
            'auroc_mean': float(np.mean(aurocs)), 'auroc_std': float(np.std(aurocs)),
            'd_mean': float(np.mean(ds)), 'd_std': float(np.std(ds)),
            'dims': n_comp,
        }
        print(f"  Random-{n_comp}: AUROC={np.mean(aurocs):.4f}+/-{np.std(aurocs):.4f}, "
              f"d={np.mean(ds):.2f}+/-{np.std(ds):.2f}", flush=True)

    # LDA projection (supervised, 1D since binary)
    print("\n--- LDA Projection ---", flush=True)
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(train_embeds, train_labels)
    cal_lda = lda.transform(cal_embeds)
    test_lda = lda.transform(test_embeds)
    # For 1D, use direct threshold instead of cosine
    lda_scores = test_lda.ravel()
    # Try both directions
    auroc_pos = roc_auc_score(test_labels, lda_scores)
    auroc_neg = roc_auc_score(test_labels, -lda_scores)
    lda_auroc = max(auroc_pos, auroc_neg)
    lda_direction = 'positive' if auroc_pos >= auroc_neg else 'negative'
    id_lda = [s for s, l in zip(lda_scores, test_labels) if l == 0]
    ood_lda = [s for s, l in zip(lda_scores, test_labels) if l == 1]
    lda_d = abs(np.mean(ood_lda) - np.mean(id_lda)) / (np.std(id_lda) + 1e-10)
    results['lda'] = {
        'auroc': float(lda_auroc), 'd': float(lda_d), 'dims': 1,
        'direction': lda_direction,
    }
    print(f"  LDA (1D): AUROC={lda_auroc:.4f}, d={lda_d:.2f}", flush=True)

    # Whitened projection (PCA + unit variance)
    print("\n--- Whitened Projection ---", flush=True)
    for n_comp in [8, 16, 32]:
        pca = PCA(n_components=n_comp, whiten=True)
        pca.fit(train_embeds)
        cal_proj = pca.transform(cal_embeds)
        test_proj = pca.transform(test_embeds)
        auroc, d = evaluate(cal_proj, test_proj, test_labels)
        results[f'whitened_{n_comp}'] = {'auroc': auroc, 'd': d, 'dims': n_comp}
        print(f"  Whitened PCA-{n_comp}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'projection_head',
        'experiment_number': 110,
        'timestamp': timestamp,
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"projection_head_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
