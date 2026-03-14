"""
Mahalanobis Distance OOD Detection.

Compares Mahalanobis distance (accounts for covariance structure)
against cosine distance for OOD detection. Mahalanobis is a classical
approach from Lee et al. (2018) "A Simple Unified Framework for
Detecting Out-of-Distribution Samples and Adversarial Attacks."

Also tests feature-norm based detection and combines signals.

Experiment 58 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf

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
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)
    return hidden


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


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

    # Calibration set (30 samples)
    print("\nCollecting calibration embeddings...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            cal_hidden.append(h)
            if len(cal_hidden) % 10 == 0:
                print(f"  Cal: {len(cal_hidden)}/30", flush=True)
    cal_hidden = np.array(cal_hidden)
    print(f"  Calibration: {len(cal_hidden)} samples, dim={cal_hidden.shape[1]}", flush=True)

    # Compute statistics
    print("\nComputing Mahalanobis statistics...", flush=True)
    centroid = np.mean(cal_hidden, axis=0)
    feature_norms_cal = np.linalg.norm(cal_hidden, axis=1)
    print(f"  ID feature norm: {np.mean(feature_norms_cal):.2f} ± {np.std(feature_norms_cal):.2f}",
          flush=True)

    # PCA reduction for Mahalanobis (full 4096-d covariance is ill-conditioned)
    # n_components must be <= min(n_samples, n_features) = 30
    from sklearn.decomposition import PCA
    max_pca = min(len(cal_hidden) - 1, 256)  # Leave 1 DOF
    for n_pca in [8, 12, 16, 20, 25, 29]:
        if n_pca > max_pca:
            break
        pca = PCA(n_components=n_pca)
        cal_pca = pca.fit_transform(cal_hidden)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA-{n_pca}: {var_explained:.3f} variance explained", flush=True)

    # Use PCA-20 for Mahalanobis (leaves 10 DOF for covariance estimation)
    pca = PCA(n_components=20)
    cal_pca = pca.fit_transform(cal_hidden)
    centroid_pca = np.mean(cal_pca, axis=0)

    # Ledoit-Wolf shrinkage estimator (robust for high-d)
    lw = LedoitWolf()
    lw.fit(cal_pca)
    precision = lw.precision_

    def mahalanobis_dist(x_pca):
        diff = x_pca - centroid_pca
        return float(np.sqrt(diff @ precision @ diff))

    # Test set
    print("\nCollecting test set...", flush=True)
    test_fns = {
        'highway': (create_highway, False, 10),
        'urban': (create_urban, False, 10),
        'noise': (create_noise, True, 8),
        'indoor': (create_indoor, True, 8),
        'inverted': (create_inverted, True, 8),
        'blackout': (create_blackout, True, 8),
    }

    test_hidden = []
    test_labels = []
    test_scenarios = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 200)), prompt)
            test_hidden.append(h)
            test_labels.append(1 if is_ood else 0)
            test_scenarios.append(scene)
            if cnt % 10 == 0:
                print(f"  Test: [{cnt}/{total}] {scene}_{i}", flush=True)

    test_hidden = np.array(test_hidden)
    test_labels = np.array(test_labels)
    test_pca = pca.transform(test_hidden)

    print(f"  Test: {len(test_hidden)} ({sum(test_labels==0)} ID, {sum(test_labels==1)} OOD)",
          flush=True)

    # Compute all scores
    print("\n" + "=" * 70, flush=True)
    print("DETECTION RESULTS", flush=True)
    print("=" * 70, flush=True)

    cosine_scores = [cosine_dist(h, centroid) for h in test_hidden]
    mahal_scores = [mahalanobis_dist(x) for x in test_pca]
    norm_scores = [float(np.linalg.norm(h)) for h in test_hidden]
    norm_diff_scores = [abs(float(np.linalg.norm(h)) - np.mean(feature_norms_cal))
                        for h in test_hidden]

    methods = {
        'cosine': cosine_scores,
        'mahalanobis': mahal_scores,
        'feature_norm_diff': norm_diff_scores,
    }

    print(f"\n  {'Method':<25} {'AUROC':>8} {'ID Mean':>10} {'OOD Mean':>10} {'Ratio':>8}",
          flush=True)
    print("  " + "-" * 63, flush=True)

    aurocs = {}
    for name, scores in methods.items():
        id_vals = [s for s, l in zip(scores, test_labels) if l == 0]
        ood_vals = [s for s, l in zip(scores, test_labels) if l == 1]
        auroc = roc_auc_score(test_labels, scores)
        aurocs[name] = auroc
        ratio = np.mean(ood_vals) / (np.mean(id_vals) + 1e-10)
        print(f"  {name:<25} {auroc:>8.3f} {np.mean(id_vals):>10.4f} "
              f"{np.mean(ood_vals):>10.4f} {ratio:>8.2f}x", flush=True)

    # Combinations
    print("\n  Combinations:", flush=True)
    combos = {
        'cosine+mahal': (0.5, 'cosine', 0.5, 'mahalanobis'),
        '0.7cos+0.3mahal': (0.7, 'cosine', 0.3, 'mahalanobis'),
        '0.3cos+0.7mahal': (0.3, 'cosine', 0.7, 'mahalanobis'),
        'cos+norm_diff': (0.5, 'cosine', 0.5, 'feature_norm_diff'),
    }

    for combo_name, (w1, m1, w2, m2) in combos.items():
        # Normalize each score to [0, 1]
        s1 = np.array(methods[m1])
        s2 = np.array(methods[m2])
        s1_norm = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-10)
        s2_norm = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-10)
        combined = w1 * s1_norm + w2 * s2_norm
        auroc = roc_auc_score(test_labels, combined)
        aurocs[combo_name] = auroc
        print(f"  {combo_name:<25} {auroc:>8.3f}", flush=True)

    # Per-scenario breakdown
    print("\n  Per-scenario AUROC:", flush=True)
    ood_types = ['noise', 'indoor', 'inverted', 'blackout']
    id_hidden = test_hidden[test_labels == 0]
    id_pca = test_pca[test_labels == 0]

    per_scenario = {}
    for ood_type in ood_types:
        mask = np.array([s == ood_type for s in test_scenarios])
        type_hidden = np.vstack([id_hidden, test_hidden[mask]])
        type_pca = np.vstack([id_pca, test_pca[mask]])
        type_labels = [0]*len(id_hidden) + [1]*int(mask.sum())

        cos_s = [cosine_dist(h, centroid) for h in type_hidden]
        mah_s = [mahalanobis_dist(x) for x in type_pca]

        cos_auroc = roc_auc_score(type_labels, cos_s)
        mah_auroc = roc_auc_score(type_labels, mah_s)
        per_scenario[ood_type] = {
            'cosine': float(cos_auroc),
            'mahalanobis': float(mah_auroc),
        }
        print(f"    {ood_type:<12} cosine={cos_auroc:.3f}  mahal={mah_auroc:.3f}  "
              f"delta={mah_auroc-cos_auroc:+.3f}", flush=True)

    # PCA dimension sweep for Mahalanobis
    print("\n  PCA dimension sweep for Mahalanobis:", flush=True)
    pca_sweep = {}
    for n_pca in [4, 8, 12, 16, 20, 25, 29]:
        pca_tmp = PCA(n_components=n_pca)
        cal_tmp = pca_tmp.fit_transform(cal_hidden)
        test_tmp = pca_tmp.transform(test_hidden)
        cent_tmp = np.mean(cal_tmp, axis=0)
        lw_tmp = LedoitWolf()
        lw_tmp.fit(cal_tmp)
        prec_tmp = lw_tmp.precision_
        scores_tmp = [float(np.sqrt((x - cent_tmp) @ prec_tmp @ (x - cent_tmp)))
                      for x in test_tmp]
        auroc = roc_auc_score(test_labels, scores_tmp)
        pca_sweep[n_pca] = float(auroc)
        print(f"    PCA-{n_pca:>3}: AUROC={auroc:.3f}", flush=True)

    # Feature norm statistics
    print("\n  Feature norm statistics:", flush=True)
    for name in sorted(set(test_scenarios)):
        mask = np.array([s == name for s in test_scenarios])
        norms = np.linalg.norm(test_hidden[mask], axis=1)
        print(f"    {name:<12}: norm={np.mean(norms):.2f} ± {np.std(norms):.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'mahalanobis',
        'experiment_number': 58,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_test': len(test_hidden),
        'aurocs': {k: float(v) for k, v in aurocs.items()},
        'per_scenario': per_scenario,
        'pca_sweep': pca_sweep,
        'cal_norm_mean': float(np.mean(feature_norms_cal)),
        'cal_norm_std': float(np.std(feature_norms_cal)),
    }
    output_path = os.path.join(RESULTS_DIR, f"mahalanobis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
