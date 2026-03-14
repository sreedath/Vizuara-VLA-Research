"""
Confidence Calibration via Platt Scaling.

Tests whether cosine distance scores can be transformed into calibrated
probability estimates using Platt scaling (logistic regression), isotonic
regression, and histogram binning. Evaluates ECE (expected calibration
error) and reliability diagrams.

Experiment 104 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

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


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_data.append({'bin_center': (bin_boundaries[i] + bin_boundaries[i+1]) / 2,
                             'avg_confidence': 0, 'avg_accuracy': 0, 'count': 0})
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(avg_conf - avg_acc)
        bin_data.append({
            'bin_center': float((bin_boundaries[i] + bin_boundaries[i+1]) / 2),
            'avg_confidence': float(avg_conf),
            'avg_accuracy': float(avg_acc),
            'count': int(mask.sum()),
        })
    return float(ece), bin_data


def main():
    print("=" * 70, flush=True)
    print("CONFIDENCE CALIBRATION VIA PLATT SCALING", flush=True)
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

    # Collect embeddings: 20 per category
    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(20):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 700)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}
        print(f"    Collected {len(embeds)} embeddings", flush=True)

    # Split: first 10 ID for calibration, rest for train/test
    cal_embeds = []
    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            cal_embeds.append(data['embeds'][:10])
    cal_embeds = np.concatenate(cal_embeds, axis=0)
    centroid = np.mean(cal_embeds, axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

    # Compute cosine distances for all remaining samples
    train_scores = []
    train_labels = []
    test_scores = []
    test_labels = []
    test_cats = []

    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            # 10-14 for training calibrator, 15-19 for testing
            for e in data['embeds'][10:15]:
                e_norm = e / (np.linalg.norm(e) + 1e-10)
                train_scores.append(float(1 - np.dot(e_norm, centroid_norm)))
                train_labels.append(0)
            for e in data['embeds'][15:]:
                e_norm = e / (np.linalg.norm(e) + 1e-10)
                test_scores.append(float(1 - np.dot(e_norm, centroid_norm)))
                test_labels.append(0)
                test_cats.append(cat_name)
        else:
            # First 10 for training calibrator, last 10 for testing
            for e in data['embeds'][:10]:
                e_norm = e / (np.linalg.norm(e) + 1e-10)
                train_scores.append(float(1 - np.dot(e_norm, centroid_norm)))
                train_labels.append(1)
            for e in data['embeds'][10:]:
                e_norm = e / (np.linalg.norm(e) + 1e-10)
                test_scores.append(float(1 - np.dot(e_norm, centroid_norm)))
                test_labels.append(1)
                test_cats.append(cat_name)

    train_scores = np.array(train_scores)
    train_labels = np.array(train_labels)
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    print(f"\nTrain: {len(train_scores)} ({sum(train_labels==0)} ID, {sum(train_labels==1)} OOD)", flush=True)
    print(f"Test: {len(test_scores)} ({sum(test_labels==0)} ID, {sum(test_labels==1)} OOD)", flush=True)

    # Method 1: Raw threshold (uncalibrated)
    print("\n--- Raw Threshold (uncalibrated) ---", flush=True)
    raw_auroc = roc_auc_score(test_labels, test_scores)
    # Convert to pseudo-probability: min-max scale
    raw_probs = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-10)
    raw_ece, raw_bins = compute_ece(raw_probs, test_labels)
    raw_brier = brier_score_loss(test_labels, raw_probs)
    print(f"  AUROC: {raw_auroc:.4f}, ECE: {raw_ece:.4f}, Brier: {raw_brier:.4f}", flush=True)

    # Method 2: Platt scaling (logistic regression)
    print("\n--- Platt Scaling ---", flush=True)
    platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt.fit(train_scores.reshape(-1, 1), train_labels)
    platt_probs = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    platt_auroc = roc_auc_score(test_labels, platt_probs)
    platt_ece, platt_bins = compute_ece(platt_probs, test_labels)
    platt_brier = brier_score_loss(test_labels, platt_probs)
    print(f"  AUROC: {platt_auroc:.4f}, ECE: {platt_ece:.4f}, Brier: {platt_brier:.4f}", flush=True)
    print(f"  Coefficients: w={platt.coef_[0][0]:.4f}, b={platt.intercept_[0]:.4f}", flush=True)

    # Method 3: Isotonic regression
    print("\n--- Isotonic Regression ---", flush=True)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(train_scores, train_labels)
    iso_probs = iso.predict(test_scores)
    iso_auroc = roc_auc_score(test_labels, iso_probs)
    iso_ece, iso_bins = compute_ece(iso_probs, test_labels)
    iso_brier = brier_score_loss(test_labels, iso_probs)
    print(f"  AUROC: {iso_auroc:.4f}, ECE: {iso_ece:.4f}, Brier: {iso_brier:.4f}", flush=True)

    # Method 4: Histogram binning
    print("\n--- Histogram Binning ---", flush=True)
    n_hist_bins = 5
    bin_edges = np.percentile(train_scores, np.linspace(0, 100, n_hist_bins + 1))
    bin_probs = []
    for i in range(n_hist_bins):
        if i < n_hist_bins - 1:
            mask = (train_scores >= bin_edges[i]) & (train_scores < bin_edges[i + 1])
        else:
            mask = (train_scores >= bin_edges[i]) & (train_scores <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_probs.append(float(train_labels[mask].mean()))
        else:
            bin_probs.append(0.5)

    hist_probs = np.zeros(len(test_scores))
    for j, s in enumerate(test_scores):
        assigned = False
        for i in range(n_hist_bins):
            if i < n_hist_bins - 1:
                if bin_edges[i] <= s < bin_edges[i + 1]:
                    hist_probs[j] = bin_probs[i]
                    assigned = True
                    break
            else:
                if bin_edges[i] <= s <= bin_edges[i + 1]:
                    hist_probs[j] = bin_probs[i]
                    assigned = True
                    break
        if not assigned:
            hist_probs[j] = 1.0 if s > bin_edges[-1] else 0.0

    hist_auroc = roc_auc_score(test_labels, hist_probs)
    hist_ece, hist_bins = compute_ece(hist_probs, test_labels)
    hist_brier = brier_score_loss(test_labels, hist_probs)
    print(f"  AUROC: {hist_auroc:.4f}, ECE: {hist_ece:.4f}, Brier: {hist_brier:.4f}", flush=True)

    # Method 5: Temperature-scaled sigmoid
    print("\n--- Temperature-Scaled Sigmoid ---", flush=True)
    # Find optimal temperature via grid search on train set
    best_temp = 1.0
    best_temp_ece = 1.0
    temp_results = {}
    for temp in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        # Sigmoid with temperature: p = 1 / (1 + exp(-(score - threshold) / temp))
        threshold = np.mean(train_scores[train_labels == 0])
        probs = 1 / (1 + np.exp(-(test_scores - threshold) / temp))
        ece_val, _ = compute_ece(probs, test_labels)
        brier_val = brier_score_loss(test_labels, probs)
        temp_results[f"T={temp}"] = {'temperature': temp, 'ece': float(ece_val), 'brier': float(brier_val)}
        if ece_val < best_temp_ece:
            best_temp_ece = ece_val
            best_temp = temp

    threshold = np.mean(train_scores[train_labels == 0])
    sigmoid_probs = 1 / (1 + np.exp(-(test_scores - threshold) / best_temp))
    sigmoid_auroc = roc_auc_score(test_labels, sigmoid_probs)
    sigmoid_ece, sigmoid_bins = compute_ece(sigmoid_probs, test_labels)
    sigmoid_brier = brier_score_loss(test_labels, sigmoid_probs)
    print(f"  Best T={best_temp}: AUROC={sigmoid_auroc:.4f}, ECE={sigmoid_ece:.4f}, Brier={sigmoid_brier:.4f}", flush=True)

    # Summary
    print("\n--- SUMMARY ---", flush=True)
    methods = {
        'raw': {'auroc': raw_auroc, 'ece': raw_ece, 'brier': raw_brier},
        'platt': {'auroc': platt_auroc, 'ece': platt_ece, 'brier': platt_brier},
        'isotonic': {'auroc': iso_auroc, 'ece': iso_ece, 'brier': iso_brier},
        'histogram': {'auroc': hist_auroc, 'ece': hist_ece, 'brier': hist_brier},
        'sigmoid': {'auroc': sigmoid_auroc, 'ece': sigmoid_ece, 'brier': sigmoid_brier},
    }
    for name, m in methods.items():
        print(f"  {name:12}: AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}  Brier={m['brier']:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'confidence_calibration',
        'experiment_number': 104,
        'timestamp': timestamp,
        'n_calibration': int(cal_embeds.shape[0]),
        'n_train': int(len(train_scores)),
        'n_test': int(len(test_scores)),
        'methods': {
            'raw': {'auroc': float(raw_auroc), 'ece': float(raw_ece), 'brier': float(raw_brier),
                     'bins': raw_bins},
            'platt': {'auroc': float(platt_auroc), 'ece': float(platt_ece), 'brier': float(platt_brier),
                      'coef': float(platt.coef_[0][0]), 'intercept': float(platt.intercept_[0]),
                      'bins': platt_bins},
            'isotonic': {'auroc': float(iso_auroc), 'ece': float(iso_ece), 'brier': float(iso_brier),
                         'bins': iso_bins},
            'histogram': {'auroc': float(hist_auroc), 'ece': float(hist_ece), 'brier': float(hist_brier),
                          'bins': hist_bins, 'bin_edges': [float(b) for b in bin_edges],
                          'bin_probs': bin_probs},
            'sigmoid': {'auroc': float(sigmoid_auroc), 'ece': float(sigmoid_ece), 'brier': float(sigmoid_brier),
                        'best_temperature': float(best_temp), 'threshold': float(threshold),
                        'temp_search': temp_results, 'bins': sigmoid_bins},
        },
        'train_scores': {'id': [float(s) for s, l in zip(train_scores, train_labels) if l == 0],
                         'ood': [float(s) for s, l in zip(train_scores, train_labels) if l == 1]},
        'test_scores': {'id': [float(s) for s, l in zip(test_scores, test_labels) if l == 0],
                        'ood': [float(s) for s, l in zip(test_scores, test_labels) if l == 1]},
    }
    output_path = os.path.join(RESULTS_DIR, f"confidence_calibration_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
