"""
Norm-Based OOD Detection.

Tests whether embedding norm alone can detect OOD inputs, building
on the Experiment 94 finding that OOD norms (80-92) exceed ID norms
(75-76). Compares L2 norm, L1 norm, Linf norm, and combinations
with cosine distance.

Experiment 96 in the CalibDrive series.
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
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("NORM-BASED OOD DETECTION", flush=True)
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

    # Collect embeddings
    print("\nCollecting embeddings...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            h = extract_hidden(model, processor,
                              Image.fromarray(fn(i + 9000)), prompt)
            if h is not None:
                cal_hidden.append(h)

    id_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            h = extract_hidden(model, processor,
                              Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                id_hidden.append(h)

    ood_hidden = []
    ood_labels = []
    for fn, name in [(create_noise, 'noise'), (create_indoor, 'indoor'),
                     (create_twilight_highway, 'twilight'), (create_snow, 'snow')]:
        for i in range(8):
            h = extract_hidden(model, processor,
                              Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                ood_hidden.append(h)
                ood_labels.append(name)

    print(f"  Cal: {len(cal_hidden)}, ID: {len(id_hidden)}, OOD: {len(ood_hidden)}", flush=True)

    centroid = np.mean(cal_hidden, axis=0)
    cal_norms = np.array([np.linalg.norm(h) for h in cal_hidden])
    cal_mean_norm = cal_norms.mean()

    results = {}

    # 1. L2 norm (higher = OOD)
    id_l2 = [np.linalg.norm(h) for h in id_hidden]
    ood_l2 = [np.linalg.norm(h) for h in ood_hidden]
    labels = [0]*len(id_l2) + [1]*len(ood_l2)
    results['l2_norm'] = {
        'auroc': float(roc_auc_score(labels, id_l2 + ood_l2)),
        'id_mean': float(np.mean(id_l2)), 'id_std': float(np.std(id_l2)),
        'ood_mean': float(np.mean(ood_l2)), 'ood_std': float(np.std(ood_l2)),
    }
    print(f"  L2 norm: AUROC={results['l2_norm']['auroc']:.3f}", flush=True)

    # 2. L2 norm deviation from calibration mean
    id_l2dev = [abs(np.linalg.norm(h) - cal_mean_norm) for h in id_hidden]
    ood_l2dev = [abs(np.linalg.norm(h) - cal_mean_norm) for h in ood_hidden]
    results['l2_deviation'] = {
        'auroc': float(roc_auc_score(labels, id_l2dev + ood_l2dev)),
        'id_mean': float(np.mean(id_l2dev)), 'ood_mean': float(np.mean(ood_l2dev)),
    }
    print(f"  L2 deviation: AUROC={results['l2_deviation']['auroc']:.3f}", flush=True)

    # 3. L1 norm
    id_l1 = [np.linalg.norm(h, ord=1) for h in id_hidden]
    ood_l1 = [np.linalg.norm(h, ord=1) for h in ood_hidden]
    results['l1_norm'] = {
        'auroc': float(roc_auc_score(labels, id_l1 + ood_l1)),
        'id_mean': float(np.mean(id_l1)), 'ood_mean': float(np.mean(ood_l1)),
    }
    print(f"  L1 norm: AUROC={results['l1_norm']['auroc']:.3f}", flush=True)

    # 4. Linf norm (max absolute value)
    id_linf = [np.linalg.norm(h, ord=np.inf) for h in id_hidden]
    ood_linf = [np.linalg.norm(h, ord=np.inf) for h in ood_hidden]
    results['linf_norm'] = {
        'auroc': float(roc_auc_score(labels, id_linf + ood_linf)),
        'id_mean': float(np.mean(id_linf)), 'ood_mean': float(np.mean(ood_linf)),
    }
    print(f"  Linf norm: AUROC={results['linf_norm']['auroc']:.3f}", flush=True)

    # 5. Cosine distance baseline
    id_cos = [cosine_dist(h, centroid) for h in id_hidden]
    ood_cos = [cosine_dist(h, centroid) for h in ood_hidden]
    results['cosine'] = {
        'auroc': float(roc_auc_score(labels, id_cos + ood_cos)),
        'id_mean': float(np.mean(id_cos)), 'ood_mean': float(np.mean(ood_cos)),
    }
    print(f"  Cosine: AUROC={results['cosine']['auroc']:.3f}", flush=True)

    # 6. Euclidean distance
    id_euc = [np.linalg.norm(h - centroid) for h in id_hidden]
    ood_euc = [np.linalg.norm(h - centroid) for h in ood_hidden]
    results['euclidean'] = {
        'auroc': float(roc_auc_score(labels, id_euc + ood_euc)),
        'id_mean': float(np.mean(id_euc)), 'ood_mean': float(np.mean(ood_euc)),
    }
    print(f"  Euclidean: AUROC={results['euclidean']['auroc']:.3f}", flush=True)

    # 7. Combined: cosine + norm deviation (sum of z-scores)
    id_cos_arr = np.array(id_cos)
    ood_cos_arr = np.array(ood_cos)
    all_cos = np.concatenate([id_cos_arr, ood_cos_arr])
    cos_z_id = (id_cos_arr - all_cos.mean()) / (all_cos.std() + 1e-10)
    cos_z_ood = (ood_cos_arr - all_cos.mean()) / (all_cos.std() + 1e-10)

    id_l2dev_arr = np.array(id_l2dev)
    ood_l2dev_arr = np.array(ood_l2dev)
    all_l2dev = np.concatenate([id_l2dev_arr, ood_l2dev_arr])
    l2_z_id = (id_l2dev_arr - all_l2dev.mean()) / (all_l2dev.std() + 1e-10)
    l2_z_ood = (ood_l2dev_arr - all_l2dev.mean()) / (all_l2dev.std() + 1e-10)

    id_combined = (cos_z_id + l2_z_id).tolist()
    ood_combined = (cos_z_ood + l2_z_ood).tolist()
    results['cosine_plus_norm'] = {
        'auroc': float(roc_auc_score(labels, id_combined + ood_combined)),
    }
    print(f"  Cosine+Norm: AUROC={results['cosine_plus_norm']['auroc']:.3f}", flush=True)

    # 8. Product: cosine × norm deviation
    id_prod = (cos_z_id * l2_z_id).tolist()
    ood_prod = (cos_z_ood * l2_z_ood).tolist()
    results['cosine_times_norm'] = {
        'auroc': float(roc_auc_score(labels, id_prod + ood_prod)),
    }
    print(f"  Cosine×Norm: AUROC={results['cosine_times_norm']['auroc']:.3f}", flush=True)

    # Per-category norm analysis
    print("\n--- Per-category norms ---", flush=True)
    per_cat_norms = {}
    for cat in set(ood_labels):
        cat_norms = [np.linalg.norm(h) for h, l in zip(ood_hidden, ood_labels) if l == cat]
        per_cat_norms[cat] = {
            'mean': float(np.mean(cat_norms)),
            'std': float(np.std(cat_norms)),
        }
        # Per-category AUROC for norm
        cat_scores = cat_norms
        cat_labels = [0]*len(id_l2) + [1]*len(cat_scores)
        cat_all = id_l2 + cat_scores
        per_cat_norms[cat]['auroc'] = float(roc_auc_score(cat_labels, cat_all))
        print(f"  {cat}: norm={np.mean(cat_norms):.1f}±{np.std(cat_norms):.1f}, AUROC={per_cat_norms[cat]['auroc']:.3f}", flush=True)

    results['per_category_norms'] = per_cat_norms
    results['cal_mean_norm'] = float(cal_mean_norm)
    results['id_mean_norm'] = float(np.mean(id_l2))

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'norm_detection',
        'experiment_number': 96,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_id': len(id_hidden),
        'n_ood': len(ood_hidden),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"norm_detection_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
