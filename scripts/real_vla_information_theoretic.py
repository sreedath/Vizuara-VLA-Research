#!/usr/bin/env python3
"""Experiment 392: Information-Theoretic Analysis of Detection Channel

Formalizes the detection system as an information channel and measures
capacity, mutual information, and error bounds using information theory.

Tests:
1. Mutual information between corruption label and detection distance
2. Channel capacity under different noise models
3. Rate-distortion analysis
4. Fisher information in detection distance
5. KL divergence between clean and corrupt distance distributions
6. Entropy of detection decisions across severity levels
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def kl_divergence_empirical(p_samples, q_samples, n_bins=50):
    """Estimate KL(P||Q) from samples using histogram approximation."""
    all_samples = np.concatenate([p_samples, q_samples])
    bins = np.linspace(min(all_samples) - 1e-10, max(all_samples) + 1e-10, n_bins + 1)
    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)
    # Add smoothing
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    bin_width = bins[1] - bins[0]
    kl = np.sum(p_hist * np.log(p_hist / q_hist)) * bin_width
    return float(kl)

def mutual_information_discrete(labels, predictions, n_classes=5):
    """Compute MI between discrete labels and continuous predictions."""
    # Discretize predictions
    bins = np.linspace(min(predictions), max(predictions) + 1e-10, n_classes + 1)
    pred_discrete = np.digitize(predictions, bins) - 1
    pred_discrete = np.clip(pred_discrete, 0, n_classes - 1)

    # Joint distribution
    n = len(labels)
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # P(X, Y)
    joint = np.zeros((n_labels, n_classes))
    for i, l in enumerate(labels):
        li = np.where(unique_labels == l)[0][0]
        joint[li, pred_discrete[i]] += 1
    joint /= n

    # Marginals
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)

    # MI = sum P(x,y) log(P(x,y) / (P(x)P(y)))
    mi = 0.0
    for i in range(n_labels):
        for j in range(n_classes):
            if joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_x[i] * p_y[j]))
    return float(mi)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    corruptions = ['fog', 'night', 'noise', 'blur']
    n_samples = 15

    # Collect clean and corrupted embeddings with variation
    print("Collecting embeddings...")
    clean_embs = []
    clean_dists = []

    for i in range(n_samples):
        arr = np.array(img).astype(np.float32)
        arr += np.random.RandomState(100 + i).randn(*arr.shape) * 0.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        emb = extract_hidden(model, processor, Image.fromarray(arr), prompt)
        clean_embs.append(emb)

    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    corrupt_data = {}
    for c in corruptions:
        corrupt_data[c] = {"embs": [], "dists": []}
        for i in range(n_samples):
            arr = np.array(img).astype(np.float32)
            arr += np.random.RandomState(200 + i).randn(*arr.shape) * 0.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            corrupted = apply_corruption(Image.fromarray(arr), c)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = cosine_dist(emb, centroid)
            corrupt_data[c]["embs"].append(emb)
            corrupt_data[c]["dists"].append(d)
        print(f"  {c}: mean_dist={np.mean(corrupt_data[c]['dists']):.6f}")

    results = {}

    # 1. KL Divergence between clean and corrupt distributions
    print("\n=== KL Divergence ===")
    kl_results = {}
    for c in corruptions:
        kl_fwd = kl_divergence_empirical(np.array(clean_dists), np.array(corrupt_data[c]["dists"]))
        kl_rev = kl_divergence_empirical(np.array(corrupt_data[c]["dists"]), np.array(clean_dists))
        js_div = (kl_fwd + kl_rev) / 2
        kl_results[c] = {
            "kl_clean_to_corrupt": kl_fwd,
            "kl_corrupt_to_clean": kl_rev,
            "js_divergence": js_div
        }
        print(f"  {c}: KL(C||OOD)={kl_fwd:.4f}, KL(OOD||C)={kl_rev:.4f}, JS={js_div:.4f}")
    results["kl_divergence"] = kl_results

    # 2. Mutual Information (binary: clean vs corrupt)
    print("\n=== Mutual Information (Binary) ===")
    mi_binary = {}
    for c in corruptions:
        labels = [0] * len(clean_dists) + [1] * len(corrupt_data[c]["dists"])
        dists = clean_dists + corrupt_data[c]["dists"]
        mi = mutual_information_discrete(labels, dists, n_classes=10)
        mi_binary[c] = float(mi)
        print(f"  {c}: MI = {mi:.4f} bits")
    results["mi_binary"] = mi_binary

    # 3. Mutual Information (multi-class: clean + 4 corruptions)
    print("\n=== Mutual Information (5-class) ===")
    labels_5 = [0] * len(clean_dists)
    dists_5 = list(clean_dists)
    for i, c in enumerate(corruptions):
        labels_5.extend([i + 1] * len(corrupt_data[c]["dists"]))
        dists_5.extend(corrupt_data[c]["dists"])
    mi_5class = mutual_information_discrete(labels_5, dists_5, n_classes=20)
    results["mi_5class"] = float(mi_5class)
    max_entropy = np.log2(5)
    results["mi_5class_normalized"] = float(mi_5class / max_entropy)
    print(f"  MI = {mi_5class:.4f} bits (max={max_entropy:.4f}, normalized={mi_5class/max_entropy:.4f})")

    # 4. Fisher Information in detection distance
    print("\n=== Fisher Information (Severity Parameter) ===")
    fisher_results = {}
    for c in corruptions:
        sevs = [0.2, 0.4, 0.6, 0.8, 1.0]
        mean_dists = []
        for sev in sevs:
            sev_dists = []
            for i in range(5):
                arr = np.array(img).astype(np.float32)
                arr += np.random.RandomState(300 + i).randn(*arr.shape) * 0.5
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                corrupted = apply_corruption(Image.fromarray(arr), c, severity=sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                sev_dists.append(cosine_dist(emb, centroid))
            mean_dists.append(np.mean(sev_dists))

        # Fisher info = (d mu/d theta)^2 / var
        # Approximate derivative numerically
        gradients = np.gradient(mean_dists, sevs)
        # Use variance from samples
        var_est = np.var(clean_dists) + 1e-20  # Lower bound
        fisher = [(g ** 2) / var_est for g in gradients]

        fisher_results[c] = {
            "severities": sevs,
            "mean_dists": [float(d) for d in mean_dists],
            "gradients": [float(g) for g in gradients],
            "fisher_info": [float(f) for f in fisher],
            "mean_fisher": float(np.mean(fisher))
        }
        print(f"  {c}: mean Fisher={np.mean(fisher):.2f}")
    results["fisher_info"] = fisher_results

    # 5. Channel capacity estimation
    print("\n=== Channel Capacity ===")
    # Binary symmetric channel model
    for c in corruptions:
        # Optimal threshold
        all_scores = clean_dists + corrupt_data[c]["dists"]
        all_labels = [0] * len(clean_dists) + [1] * len(corrupt_data[c]["dists"])

        best_acc = 0
        best_thresh = 0
        for thresh in np.linspace(min(all_scores), max(all_scores), 100):
            preds = [1 if s > thresh else 0 for s in all_scores]
            acc = np.mean([p == l for p, l in zip(preds, all_labels)])
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        # Error probability
        p_error = 1.0 - best_acc
        # BSC capacity = 1 - H(p)
        if p_error > 0 and p_error < 1:
            h_p = -p_error * np.log2(p_error) - (1 - p_error) * np.log2(1 - p_error)
        else:
            h_p = 0.0
        capacity = 1.0 - h_p

        results.setdefault("channel_capacity", {})[c] = {
            "best_accuracy": float(best_acc),
            "error_probability": float(p_error),
            "bsc_capacity": float(capacity),
            "optimal_threshold": float(best_thresh)
        }
        print(f"  {c}: acc={best_acc:.4f}, p_err={p_error:.4f}, capacity={capacity:.4f} bits")

    # 6. Entropy of detection decision across severities
    print("\n=== Decision Entropy vs Severity ===")
    threshold = float(np.max(clean_dists) * 1.1)
    entropy_results = {}
    for c in corruptions:
        sev_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        entropies = []
        for sev in sev_list:
            decisions = []
            for i in range(10):
                arr = np.array(img).astype(np.float32)
                arr += np.random.RandomState(400 + i).randn(*arr.shape) * 0.5
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                corrupted = apply_corruption(Image.fromarray(arr), c, severity=sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = cosine_dist(emb, centroid)
                decisions.append(1 if d > threshold else 0)

            p_detect = np.mean(decisions)
            if 0 < p_detect < 1:
                entropy = -p_detect * np.log2(p_detect) - (1 - p_detect) * np.log2(1 - p_detect)
            else:
                entropy = 0.0
            entropies.append({"severity": sev, "p_detect": float(p_detect), "entropy": float(entropy)})

        entropy_results[c] = entropies
        print(f"  {c}: entropies={[e['entropy'] for e in entropies]}")
    results["decision_entropy"] = entropy_results

    # 7. Distribution separation metrics
    print("\n=== Distribution Separation ===")
    sep_results = {}
    for c in corruptions:
        clean_arr = np.array(clean_dists)
        ood_arr = np.array(corrupt_data[c]["dists"])

        # Bhattacharyya distance
        mu1, mu2 = np.mean(clean_arr), np.mean(ood_arr)
        var1, var2 = np.var(clean_arr) + 1e-20, np.var(ood_arr) + 1e-20
        bhat = 0.25 * np.log(0.25 * (var1/var2 + var2/var1 + 2)) + \
               0.25 * ((mu1 - mu2)**2 / (var1 + var2))

        # Hellinger distance
        hellinger = np.sqrt(1 - np.exp(-bhat))

        # Cohen's d
        pooled_std = np.sqrt((var1 + var2) / 2)
        cohens_d = abs(mu1 - mu2) / pooled_std if pooled_std > 0 else float('inf')

        sep_results[c] = {
            "bhattacharyya": float(bhat),
            "hellinger": float(hellinger),
            "cohens_d": float(cohens_d),
            "clean_mean": float(mu1),
            "ood_mean": float(mu2),
            "clean_std": float(np.sqrt(var1)),
            "ood_std": float(np.sqrt(var2))
        }
        print(f"  {c}: Bhat={bhat:.4f}, Hellinger={hellinger:.4f}, Cohen's d={cohens_d:.2f}")
    results["separation_metrics"] = sep_results

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/information_theoretic_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
