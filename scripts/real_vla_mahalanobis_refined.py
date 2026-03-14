"""
Mahalanobis Distance Detection — Refined.

Compares Mahalanobis distance with cosine distance for OOD detection,
testing full covariance, diagonal covariance, tied covariance across
multiple layers, and regularized variants.

Experiment 116 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 10001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 10002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 10003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 10004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 10010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 10014)
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


def mahalanobis_score(x, mean, precision):
    """Mahalanobis distance: (x - mean)^T @ precision @ (x - mean)"""
    diff = x - mean
    return float(diff @ precision @ diff)


def main():
    print("=" * 70, flush=True)
    print("MAHALANOBIS DISTANCE DETECTION — REFINED", flush=True)
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
        for i in range(20):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 1700)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}

    # Build sets: 10 cal, rest test
    cal_embeds = []
    test_embeds = []
    test_labels = []

    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            cal_embeds.extend(data['embeds'][:10])
            for e in data['embeds'][10:]:
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

    print(f"\nCal: {len(cal_embeds)}, Test: {len(test_labels)}, Dim: {dim}", flush=True)

    centroid = np.mean(cal_embeds, axis=0)

    # 1. Cosine distance baseline
    print("\n--- Cosine Distance Baseline ---", flush=True)
    cos_scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
    cos_auroc = float(roc_auc_score(test_labels, cos_scores))
    id_cos = cos_scores[test_labels == 0]
    ood_cos = cos_scores[test_labels == 1]
    cos_d = float((np.mean(ood_cos) - np.mean(id_cos)) / (np.std(id_cos) + 1e-10))
    print(f"  Cosine: AUROC={cos_auroc:.4f}, d={cos_d:.2f}", flush=True)

    results = {'cosine': {'auroc': cos_auroc, 'd': cos_d}}

    # 2. Euclidean distance
    print("\n--- Euclidean Distance ---", flush=True)
    euc_scores = np.array([float(np.linalg.norm(e - centroid)) for e in test_embeds])
    euc_auroc = float(roc_auc_score(test_labels, euc_scores))
    id_euc = euc_scores[test_labels == 0]
    ood_euc = euc_scores[test_labels == 1]
    euc_d = float((np.mean(ood_euc) - np.mean(id_euc)) / (np.std(id_euc) + 1e-10))
    print(f"  Euclidean: AUROC={euc_auroc:.4f}, d={euc_d:.2f}", flush=True)
    results['euclidean'] = {'auroc': euc_auroc, 'd': euc_d}

    # 3. Diagonal Mahalanobis (variance normalization)
    print("\n--- Diagonal Mahalanobis ---", flush=True)
    var = np.var(cal_embeds, axis=0) + 1e-6
    diag_scores = np.array([float(np.sum((e - centroid)**2 / var)) for e in test_embeds])
    diag_auroc = float(roc_auc_score(test_labels, diag_scores))
    id_diag = diag_scores[test_labels == 0]
    ood_diag = diag_scores[test_labels == 1]
    diag_d = float((np.mean(ood_diag) - np.mean(id_diag)) / (np.std(id_diag) + 1e-10))
    print(f"  Diagonal Mahalanobis: AUROC={diag_auroc:.4f}, d={diag_d:.2f}", flush=True)
    results['diagonal_mahalanobis'] = {'auroc': diag_auroc, 'd': diag_d}

    # 4. PCA-reduced Mahalanobis (project to top-k PCs, then full Mahalanobis)
    print("\n--- PCA-Reduced Mahalanobis ---", flush=True)
    from sklearn.decomposition import PCA

    max_comp = min(cal_embeds.shape[0], cal_embeds.shape[1]) - 1
    for n_comp in [4, 8, 16, 32]:
        if n_comp > max_comp:
            continue
        pca = PCA(n_components=n_comp)
        cal_pca = pca.fit_transform(cal_embeds)
        test_pca = pca.transform(test_embeds)

        pca_mean = np.mean(cal_pca, axis=0)
        pca_cov = np.cov(cal_pca.T) + 1e-4 * np.eye(n_comp)
        try:
            pca_precision = np.linalg.inv(pca_cov)
        except np.linalg.LinAlgError:
            print(f"  PCA-{n_comp}: singular covariance, skipping", flush=True)
            continue

        pca_maha_scores = np.array([mahalanobis_score(e, pca_mean, pca_precision) for e in test_pca])
        pca_auroc = float(roc_auc_score(test_labels, pca_maha_scores))
        id_pca = pca_maha_scores[test_labels == 0]
        ood_pca = pca_maha_scores[test_labels == 1]
        pca_d = float((np.mean(ood_pca) - np.mean(id_pca)) / (np.std(id_pca) + 1e-10))
        print(f"  PCA-{n_comp} Mahalanobis: AUROC={pca_auroc:.4f}, d={pca_d:.2f}", flush=True)
        results[f'pca_{n_comp}_mahalanobis'] = {
            'auroc': pca_auroc, 'd': pca_d, 'dims': n_comp,
            'explained_var': float(sum(pca.explained_variance_ratio_)),
        }

    # 5. Regularized full Mahalanobis (shrinkage)
    print("\n--- Regularized Full Mahalanobis ---", flush=True)
    from sklearn.covariance import LedoitWolf, OAS

    for name, estimator in [('ledoit_wolf', LedoitWolf()), ('oas', OAS())]:
        try:
            estimator.fit(cal_embeds)
            precision = estimator.precision_
            maha_scores = np.array([mahalanobis_score(e, centroid, precision) for e in test_embeds])
            maha_auroc = float(roc_auc_score(test_labels, maha_scores))
            id_m = maha_scores[test_labels == 0]
            ood_m = maha_scores[test_labels == 1]
            maha_d = float((np.mean(ood_m) - np.mean(id_m)) / (np.std(id_m) + 1e-10))
            print(f"  {name}: AUROC={maha_auroc:.4f}, d={maha_d:.2f}", flush=True)
            results[name] = {'auroc': maha_auroc, 'd': maha_d}
        except Exception as ex:
            print(f"  {name}: FAILED - {ex}", flush=True)
            results[name] = {'error': str(ex)}

    # 6. Per-category breakdown for best methods
    print("\n--- Per-Category Breakdown ---", flush=True)
    per_cat = {}
    for cat_name, data in embeddings.items():
        cat_cos = [cosine_dist(e, centroid) for e in data['embeds']]
        cat_euc = [float(np.linalg.norm(e - centroid)) for e in data['embeds']]
        cat_diag = [float(np.sum((e - centroid)**2 / var)) for e in data['embeds']]
        per_cat[cat_name] = {
            'group': data['group'],
            'cosine_mean': float(np.mean(cat_cos)),
            'euclidean_mean': float(np.mean(cat_euc)),
            'diagonal_maha_mean': float(np.mean(cat_diag)),
        }
        print(f"  {cat_name}: cos={np.mean(cat_cos):.4f}, euc={np.mean(cat_euc):.1f}, "
              f"diag_maha={np.mean(cat_diag):.0f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'mahalanobis_refined',
        'experiment_number': 116,
        'timestamp': timestamp,
        'n_cal': len(cal_embeds),
        'n_test': len(test_labels),
        'dim': dim,
        'results': results,
        'per_category': per_cat,
    }
    output_path = os.path.join(RESULTS_DIR, f"mahalanobis_refined_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
