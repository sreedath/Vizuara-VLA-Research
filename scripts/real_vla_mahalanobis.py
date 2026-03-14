"""
Mahalanobis Distance OOD Detection.

Compares Mahalanobis distance (accounts for covariance structure)
against cosine distance (treats all dimensions equally) for OOD
detection. Tests whether modeling feature correlations improves
detection in the PCA-reduced space.

Experiment 89 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

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
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def mahalanobis_dist(x, mean, cov_inv):
    diff = x - mean
    return float(np.sqrt(diff @ cov_inv @ diff))


def main():
    print("=" * 70, flush=True)
    print("MAHALANOBIS DISTANCE OOD DETECTION", flush=True)
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

    # Calibrate
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            if h is not None:
                cal_hidden.append(h)
    cal_matrix = np.array(cal_hidden)
    centroid = cal_matrix.mean(axis=0)
    print(f"  {len(cal_hidden)} calibration samples", flush=True)

    # Test data
    print("\nCollecting test data...", flush=True)
    id_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                id_hidden.append(h)

    ood_hidden = []
    ood_fns = [create_noise, create_indoor, create_twilight_highway, create_snow]
    for fn in ood_fns:
        for i in range(8):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                ood_hidden.append(h)

    print(f"  ID: {len(id_hidden)}, OOD: {len(ood_hidden)}", flush=True)

    # Test at different PCA dimensions
    pca_dims = [4, 8, 16, 32, 64, 128]
    results = {}

    for d in pca_dims:
        print(f"\nPCA-{d}...", flush=True)
        pca = PCA(n_components=d)
        pca.fit(cal_matrix)

        cal_proj = pca.transform(cal_matrix)
        id_proj = pca.transform(np.array(id_hidden))
        ood_proj = pca.transform(np.array(ood_hidden))

        cal_mean = cal_proj.mean(axis=0)
        cal_cov = np.cov(cal_proj.T) + 1e-6 * np.eye(d)
        cal_cov_inv = np.linalg.inv(cal_cov)

        id_cos = [cosine_dist(h, cal_mean) for h in id_proj]
        ood_cos = [cosine_dist(h, cal_mean) for h in ood_proj]
        labels = [0]*len(id_cos) + [1]*len(ood_cos)
        cos_auroc = roc_auc_score(labels, id_cos + ood_cos)

        id_mah = [mahalanobis_dist(h, cal_mean, cal_cov_inv) for h in id_proj]
        ood_mah = [mahalanobis_dist(h, cal_mean, cal_cov_inv) for h in ood_proj]
        mah_auroc = roc_auc_score(labels, id_mah + ood_mah)

        id_euc = [float(np.linalg.norm(h - cal_mean)) for h in id_proj]
        ood_euc = [float(np.linalg.norm(h - cal_mean)) for h in ood_proj]
        euc_auroc = roc_auc_score(labels, id_euc + ood_euc)

        results[str(d)] = {
            'cosine_auroc': float(cos_auroc),
            'mahalanobis_auroc': float(mah_auroc),
            'euclidean_auroc': float(euc_auroc),
            'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
        }
        print(f"  Cosine: {cos_auroc:.3f}, Mahalanobis: {mah_auroc:.3f}, "
              f"Euclidean: {euc_auroc:.3f} (var: {pca.explained_variance_ratio_.sum():.3f})", flush=True)

    # Full dimension
    print("\nFull 4096-dim...", flush=True)
    id_cos_full = [cosine_dist(h, centroid) for h in id_hidden]
    ood_cos_full = [cosine_dist(h, centroid) for h in ood_hidden]
    labels = [0]*len(id_cos_full) + [1]*len(ood_cos_full)
    cos_auroc_full = roc_auc_score(labels, id_cos_full + ood_cos_full)

    id_euc_full = [float(np.linalg.norm(h - centroid)) for h in id_hidden]
    ood_euc_full = [float(np.linalg.norm(h - centroid)) for h in ood_hidden]
    euc_auroc_full = roc_auc_score(labels, id_euc_full + ood_euc_full)

    results['4096'] = {
        'cosine_auroc': float(cos_auroc_full),
        'euclidean_auroc': float(euc_auroc_full),
        'mahalanobis_auroc': None,
        'pca_variance_explained': 1.0,
    }
    print(f"  Cosine: {cos_auroc_full:.3f}, Euclidean: {euc_auroc_full:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'mahalanobis',
        'experiment_number': 89,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_id': len(id_hidden),
        'n_ood': len(ood_hidden),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"mahalanobis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
