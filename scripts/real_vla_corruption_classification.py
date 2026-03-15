#!/usr/bin/env python3
"""Experiment 439: Corruption Classification Analysis

Tests whether the TYPE of corruption (fog, night, noise, blur) can be
identified from hidden state embeddings, not just the presence of corruption.
For autonomous driving, knowing *what* is wrong enables targeted responses
(e.g., activate fog lights vs. switch to IR camera).

Tests:
  1. Nearest-centroid corruption classifier
  2. Direction-based classification via displacement vectors
  3. Confusion matrix across corruption types
  4. Multi-severity classification robustness
  5. Leave-one-scene-out classification accuracy
"""
import time
import json
import torch
import numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from collections import defaultdict
from itertools import combinations

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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


def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)


def cosine_sim(a, b):
    return 1.0 - cosine_dist(a, b)


def classify_nearest_centroid(embedding, centroids, metric="cosine"):
    """Return the corruption type whose centroid is closest."""
    best_label, best_dist = None, float('inf')
    for label, centroid in centroids.items():
        if metric == "cosine":
            d = cosine_dist(embedding, centroid)
        else:
            d = float(np.linalg.norm(np.asarray(embedding) - np.asarray(centroid)))
        if d < best_dist:
            best_dist = d
            best_label = label
    return best_label, best_dist


def classify_by_direction(displacement, direction_vectors, metric="cosine"):
    """Classify by finding the corruption direction most aligned with displacement."""
    best_label, best_sim = None, -float('inf')
    disp = np.asarray(displacement, dtype=np.float64)
    disp_norm = np.linalg.norm(disp)
    if disp_norm < 1e-12:
        return None, 0.0
    disp_unit = disp / disp_norm
    for label, direction in direction_vectors.items():
        sim = cosine_sim(disp_unit, direction)
        if sim > best_sim:
            best_sim = sim
            best_label = label
    return best_label, best_sim


def build_confusion_matrix(labels, predictions, class_names):
    """Build a confusion matrix dict: true_label -> predicted_label -> count."""
    matrix = {t: {p: 0 for p in class_names} for t in class_names}
    for true, pred in zip(labels, predictions):
        matrix[true][pred] += 1
    return matrix


def confusion_matrix_accuracy(matrix, class_names):
    """Compute overall and per-class accuracy from a confusion matrix dict."""
    total_correct = 0
    total = 0
    per_class = {}
    for true_label in class_names:
        row_total = sum(matrix[true_label].values())
        correct = matrix[true_label].get(true_label, 0)
        total_correct += correct
        total += row_total
        per_class[true_label] = correct / max(row_total, 1)
    overall = total_correct / max(total, 1)
    return overall, per_class


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

print("=" * 70)
print("Experiment 439: Corruption Classification Analysis")
print("=" * 70)

# --- Model loading ---
print("\nLoading OpenVLA-7B...")
prompt = "In: What action should the robot take to pick up the object?\nOut:"
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()
print("Model loaded.")

# --- Configuration ---
SEEDS = [42, 123, 456, 789, 999, 1111, 2222, 3333]
CORRUPTIONS = ['fog', 'night', 'noise', 'blur']
SEVERITIES = [0.25, 0.5, 0.75, 1.0]
N_SCENES = len(SEEDS)
LAYER = 3

# --- Generate scenes ---
print(f"\nGenerating {N_SCENES} random 224x224 scenes...")
scenes = []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    pixels = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    scenes.append(Image.fromarray(pixels))

# --- Extract all embeddings ---
print("\nExtracting clean embeddings...")
clean_embeddings = []
for si, scene in enumerate(scenes):
    h = extract_hidden(model, processor, scene, prompt, layer=LAYER)
    clean_embeddings.append(h)
    print(f"  Scene {si} (seed={SEEDS[si]}): dim={h.shape[0]}, norm={np.linalg.norm(h):.4f}")

clean_centroid = np.mean(clean_embeddings, axis=0)

print("\nExtracting corrupted embeddings (severity=1.0)...")
# corruption_embeddings[ctype][scene_idx] = embedding
corruption_embeddings = {c: [] for c in CORRUPTIONS}
for ctype in CORRUPTIONS:
    for si, scene in enumerate(scenes):
        corrupted = apply_corruption(scene, ctype, severity=1.0)
        h = extract_hidden(model, processor, corrupted, prompt, layer=LAYER)
        corruption_embeddings[ctype].append(h)
    print(f"  {ctype}: {len(corruption_embeddings[ctype])} embeddings extracted")

print("\nExtracting multi-severity embeddings...")
# severity_embeddings[ctype][sev_idx][scene_idx] = embedding
severity_embeddings = {c: {s: [] for s in SEVERITIES} for c in CORRUPTIONS}
for ctype in CORRUPTIONS:
    for sev in SEVERITIES:
        for si, scene in enumerate(scenes):
            if sev == 1.0:
                # Reuse already-extracted embeddings
                severity_embeddings[ctype][sev].append(corruption_embeddings[ctype][si])
            else:
                corrupted = apply_corruption(scene, ctype, severity=sev)
                h = extract_hidden(model, processor, corrupted, prompt, layer=LAYER)
                severity_embeddings[ctype][sev].append(h)
        print(f"  {ctype} sev={sev}: done")

results = {}

# =========================================================================
# TEST 1: Nearest-Centroid Corruption Classifier
# =========================================================================
print("\n" + "=" * 70)
print("TEST 1: Nearest-Centroid Corruption Classifier")
print("=" * 70)

# Compute per-corruption centroids from all scenes at severity=1.0
centroids = {}
for ctype in CORRUPTIONS:
    centroids[ctype] = np.mean(corruption_embeddings[ctype], axis=0)

# Classify each corrupted embedding by nearest centroid
true_labels = []
pred_labels = []
distances = []
for ctype in CORRUPTIONS:
    for si in range(N_SCENES):
        emb = corruption_embeddings[ctype][si]
        pred, dist = classify_nearest_centroid(emb, centroids, metric="cosine")
        true_labels.append(ctype)
        pred_labels.append(pred)
        distances.append(dist)
        status = "OK" if pred == ctype else f"WRONG (pred={pred})"
        print(f"  Scene {si} [{ctype}] -> {status}  (cosine_dist={dist:.6f})")

accuracy_centroid = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)
print(f"\n  Overall centroid classifier accuracy: {accuracy_centroid:.4f} ({accuracy_centroid*100:.1f}%)")

# Also try euclidean
true_labels_euc = []
pred_labels_euc = []
for ctype in CORRUPTIONS:
    for si in range(N_SCENES):
        emb = corruption_embeddings[ctype][si]
        pred, dist = classify_nearest_centroid(emb, centroids, metric="euclidean")
        true_labels_euc.append(ctype)
        pred_labels_euc.append(pred)

accuracy_centroid_euc = sum(1 for t, p in zip(true_labels_euc, pred_labels_euc) if t == p) / len(true_labels_euc)
print(f"  Euclidean metric accuracy: {accuracy_centroid_euc:.4f} ({accuracy_centroid_euc*100:.1f}%)")

# Inter-centroid distances
centroid_distances = {}
for c1, c2 in combinations(CORRUPTIONS, 2):
    d_cos = cosine_dist(centroids[c1], centroids[c2])
    d_euc = float(np.linalg.norm(centroids[c1] - centroids[c2]))
    centroid_distances[f"{c1}_vs_{c2}"] = {
        "cosine_dist": d_cos,
        "euclidean_dist": d_euc
    }
    print(f"  Centroid distance {c1} vs {c2}: cosine={d_cos:.6f}, euclidean={d_euc:.4f}")

results["test1_nearest_centroid"] = {
    "accuracy_cosine": accuracy_centroid,
    "accuracy_euclidean": accuracy_centroid_euc,
    "n_samples": len(true_labels),
    "centroid_distances": centroid_distances,
    "mean_classification_cosine_dist": float(np.mean(distances))
}

# =========================================================================
# TEST 2: Direction-Based Classification
# =========================================================================
print("\n" + "=" * 70)
print("TEST 2: Direction-Based Classification (Displacement Vectors)")
print("=" * 70)

# Compute mean displacement direction per corruption (from clean centroid)
direction_vectors = {}
for ctype in CORRUPTIONS:
    displacements = []
    for si in range(N_SCENES):
        disp = corruption_embeddings[ctype][si] - clean_embeddings[si]
        norm = np.linalg.norm(disp)
        if norm > 1e-12:
            displacements.append(disp / norm)
    mean_dir = np.mean(displacements, axis=0)
    mean_dir_norm = np.linalg.norm(mean_dir)
    if mean_dir_norm > 1e-12:
        mean_dir = mean_dir / mean_dir_norm
    direction_vectors[ctype] = mean_dir

# Within-corruption direction consistency
print("\n  Within-corruption direction consistency:")
direction_consistency = {}
for ctype in CORRUPTIONS:
    sims = []
    for si in range(N_SCENES):
        disp = corruption_embeddings[ctype][si] - clean_embeddings[si]
        disp_norm = np.linalg.norm(disp)
        if disp_norm > 1e-12:
            disp_unit = disp / disp_norm
            sim = cosine_sim(disp_unit, direction_vectors[ctype])
            sims.append(sim)
    mean_c = float(np.mean(sims))
    std_c = float(np.std(sims))
    direction_consistency[ctype] = {"mean_sim": mean_c, "std_sim": std_c}
    print(f"    {ctype}: alignment with mean direction = {mean_c:.4f} +/- {std_c:.4f}")

# Cross-corruption direction similarity
print("\n  Cross-corruption direction separability:")
direction_cross_sims = {}
for c1, c2 in combinations(CORRUPTIONS, 2):
    sim = cosine_sim(direction_vectors[c1], direction_vectors[c2])
    direction_cross_sims[f"{c1}_vs_{c2}"] = sim
    print(f"    {c1} vs {c2}: direction similarity = {sim:.4f}")

# Classify by direction
true_labels_dir = []
pred_labels_dir = []
alignment_scores = []
for ctype in CORRUPTIONS:
    for si in range(N_SCENES):
        disp = corruption_embeddings[ctype][si] - clean_embeddings[si]
        pred, score = classify_by_direction(disp, direction_vectors)
        true_labels_dir.append(ctype)
        pred_labels_dir.append(pred)
        alignment_scores.append(score)
        status = "OK" if pred == ctype else f"WRONG (pred={pred})"
        print(f"  Scene {si} [{ctype}] -> {status}  (alignment={score:.4f})")

accuracy_direction = sum(
    1 for t, p in zip(true_labels_dir, pred_labels_dir) if t == p
) / len(true_labels_dir)
print(f"\n  Direction classifier accuracy: {accuracy_direction:.4f} ({accuracy_direction*100:.1f}%)")

results["test2_direction_classification"] = {
    "accuracy": accuracy_direction,
    "direction_consistency": direction_consistency,
    "cross_corruption_direction_similarity": direction_cross_sims,
    "mean_alignment_score": float(np.mean(alignment_scores)),
    "n_samples": len(true_labels_dir)
}

# =========================================================================
# TEST 3: Confusion Matrix Across Corruption Types
# =========================================================================
print("\n" + "=" * 70)
print("TEST 3: Confusion Matrix Across Corruption Types")
print("=" * 70)

# Centroid-based confusion matrix
cm_centroid = build_confusion_matrix(true_labels, pred_labels, CORRUPTIONS)
overall_acc_cm, per_class_acc_cm = confusion_matrix_accuracy(cm_centroid, CORRUPTIONS)

print("\n  Centroid Classifier Confusion Matrix:")
header = "  {:>8s}".format("true\\pred") + "".join(f"  {c:>7s}" for c in CORRUPTIONS)
print(header)
for true_c in CORRUPTIONS:
    row = f"  {true_c:>8s}"
    for pred_c in CORRUPTIONS:
        row += f"  {cm_centroid[true_c][pred_c]:>7d}"
    print(row)

print(f"\n  Per-class accuracy (centroid):")
for c in CORRUPTIONS:
    print(f"    {c}: {per_class_acc_cm[c]:.4f}")

# Direction-based confusion matrix
cm_direction = build_confusion_matrix(true_labels_dir, pred_labels_dir, CORRUPTIONS)
overall_acc_dir_cm, per_class_acc_dir_cm = confusion_matrix_accuracy(cm_direction, CORRUPTIONS)

print("\n  Direction Classifier Confusion Matrix:")
print(header)
for true_c in CORRUPTIONS:
    row = f"  {true_c:>8s}"
    for pred_c in CORRUPTIONS:
        row += f"  {cm_direction[true_c][pred_c]:>7d}"
    print(row)

print(f"\n  Per-class accuracy (direction):")
for c in CORRUPTIONS:
    print(f"    {c}: {per_class_acc_dir_cm[c]:.4f}")

# Most confused pairs
most_confused_centroid = []
for c1 in CORRUPTIONS:
    for c2 in CORRUPTIONS:
        if c1 != c2 and cm_centroid[c1][c2] > 0:
            most_confused_centroid.append({
                "true": c1, "predicted": c2, "count": cm_centroid[c1][c2]
            })
most_confused_centroid.sort(key=lambda x: x["count"], reverse=True)

most_confused_direction = []
for c1 in CORRUPTIONS:
    for c2 in CORRUPTIONS:
        if c1 != c2 and cm_direction[c1][c2] > 0:
            most_confused_direction.append({
                "true": c1, "predicted": c2, "count": cm_direction[c1][c2]
            })
most_confused_direction.sort(key=lambda x: x["count"], reverse=True)

if most_confused_centroid:
    print(f"\n  Most confused pairs (centroid):")
    for mc in most_confused_centroid[:5]:
        print(f"    {mc['true']} misclassified as {mc['predicted']}: {mc['count']} times")

results["test3_confusion_matrix"] = {
    "centroid": {
        "confusion_matrix": {t: {p: cm_centroid[t][p] for p in CORRUPTIONS} for t in CORRUPTIONS},
        "overall_accuracy": overall_acc_cm,
        "per_class_accuracy": per_class_acc_cm,
        "most_confused_pairs": most_confused_centroid
    },
    "direction": {
        "confusion_matrix": {t: {p: cm_direction[t][p] for p in CORRUPTIONS} for t in CORRUPTIONS},
        "overall_accuracy": overall_acc_dir_cm,
        "per_class_accuracy": per_class_acc_dir_cm,
        "most_confused_pairs": most_confused_direction
    }
}

# =========================================================================
# TEST 4: Multi-Severity Classification Robustness
# =========================================================================
print("\n" + "=" * 70)
print("TEST 4: Multi-Severity Classification Robustness")
print("=" * 70)

# Use centroids computed from severity=1.0 to classify all severities
severity_results = {}
for sev in SEVERITIES:
    true_sev = []
    pred_centroid_sev = []
    pred_direction_sev = []
    for ctype in CORRUPTIONS:
        for si in range(N_SCENES):
            emb = severity_embeddings[ctype][sev][si]

            # Centroid classification
            pred_c, _ = classify_nearest_centroid(emb, centroids, metric="cosine")
            true_sev.append(ctype)
            pred_centroid_sev.append(pred_c)

            # Direction classification
            disp = emb - clean_embeddings[si]
            pred_d, _ = classify_by_direction(disp, direction_vectors)
            pred_direction_sev.append(pred_d)

    acc_c = sum(1 for t, p in zip(true_sev, pred_centroid_sev) if t == p) / len(true_sev)
    acc_d = sum(1 for t, p in zip(true_sev, pred_direction_sev) if t == p) / len(true_sev)

    # Per-corruption accuracy at this severity
    per_corruption_centroid = {}
    per_corruption_direction = {}
    idx = 0
    for ctype in CORRUPTIONS:
        correct_c = 0
        correct_d = 0
        for si in range(N_SCENES):
            if pred_centroid_sev[idx] == ctype:
                correct_c += 1
            if pred_direction_sev[idx] == ctype:
                correct_d += 1
            idx += 1
        per_corruption_centroid[ctype] = correct_c / N_SCENES
        per_corruption_direction[ctype] = correct_d / N_SCENES

    severity_results[str(sev)] = {
        "centroid_accuracy": acc_c,
        "direction_accuracy": acc_d,
        "per_corruption_centroid": per_corruption_centroid,
        "per_corruption_direction": per_corruption_direction,
        "n_samples": len(true_sev)
    }
    print(f"\n  Severity {sev}:")
    print(f"    Centroid accuracy:  {acc_c:.4f} ({acc_c*100:.1f}%)")
    print(f"    Direction accuracy: {acc_d:.4f} ({acc_d*100:.1f}%)")
    for ctype in CORRUPTIONS:
        print(f"      {ctype}: centroid={per_corruption_centroid[ctype]:.2f}, "
              f"direction={per_corruption_direction[ctype]:.2f}")

# Severity degradation analysis
print("\n  Severity degradation summary:")
for method in ["centroid", "direction"]:
    accs = [severity_results[str(s)][f"{method}_accuracy"] for s in SEVERITIES]
    print(f"    {method}: {' -> '.join(f'{a:.2f}' for a in accs)}")
    if len(accs) >= 2:
        print(f"      Drop from max to min severity: {max(accs) - min(accs):.4f}")

results["test4_multi_severity"] = {
    "severity_results": severity_results,
    "severities_tested": SEVERITIES,
    "training_severity": 1.0,
    "note": "Centroids trained at severity=1.0, tested at all severities"
}

# =========================================================================
# TEST 5: Leave-One-Scene-Out Classification Accuracy
# =========================================================================
print("\n" + "=" * 70)
print("TEST 5: Leave-One-Scene-Out Classification Accuracy")
print("=" * 70)

loso_true = []
loso_pred_centroid = []
loso_pred_direction = []
loso_per_scene = {}

for held_out in range(N_SCENES):
    train_indices = [i for i in range(N_SCENES) if i != held_out]

    # Build centroids from training scenes only
    loso_centroids = {}
    for ctype in CORRUPTIONS:
        train_embeds = [corruption_embeddings[ctype][i] for i in train_indices]
        loso_centroids[ctype] = np.mean(train_embeds, axis=0)

    # Build direction vectors from training scenes only
    loso_directions = {}
    for ctype in CORRUPTIONS:
        disps = []
        for i in train_indices:
            disp = corruption_embeddings[ctype][i] - clean_embeddings[i]
            norm = np.linalg.norm(disp)
            if norm > 1e-12:
                disps.append(disp / norm)
        mean_dir = np.mean(disps, axis=0)
        mean_dir_norm = np.linalg.norm(mean_dir)
        if mean_dir_norm > 1e-12:
            mean_dir = mean_dir / mean_dir_norm
        loso_directions[ctype] = mean_dir

    # Classify held-out scene
    scene_correct_centroid = 0
    scene_correct_direction = 0
    scene_total = 0
    for ctype in CORRUPTIONS:
        emb = corruption_embeddings[ctype][held_out]

        # Centroid
        pred_c, _ = classify_nearest_centroid(emb, loso_centroids, metric="cosine")
        loso_true.append(ctype)
        loso_pred_centroid.append(pred_c)
        if pred_c == ctype:
            scene_correct_centroid += 1

        # Direction
        disp = emb - clean_embeddings[held_out]
        pred_d, _ = classify_by_direction(disp, loso_directions)
        loso_pred_direction.append(pred_d)
        if pred_d == ctype:
            scene_correct_direction += 1

        scene_total += 1

    scene_acc_c = scene_correct_centroid / scene_total
    scene_acc_d = scene_correct_direction / scene_total
    loso_per_scene[f"scene_{held_out}_seed_{SEEDS[held_out]}"] = {
        "centroid_accuracy": scene_acc_c,
        "direction_accuracy": scene_acc_d,
        "n_corruptions_tested": scene_total
    }
    print(f"  Scene {held_out} (seed={SEEDS[held_out]}): "
          f"centroid={scene_acc_c:.2f}, direction={scene_acc_d:.2f}")

loso_acc_centroid = sum(
    1 for t, p in zip(loso_true, loso_pred_centroid) if t == p
) / len(loso_true)
loso_acc_direction = sum(
    1 for t, p in zip(loso_true, loso_pred_direction) if t == p
) / len(loso_true)

# LOSO confusion matrices
loso_cm_centroid = build_confusion_matrix(loso_true, loso_pred_centroid, CORRUPTIONS)
loso_cm_direction = build_confusion_matrix(loso_true, loso_pred_direction, CORRUPTIONS)
_, loso_per_class_c = confusion_matrix_accuracy(loso_cm_centroid, CORRUPTIONS)
_, loso_per_class_d = confusion_matrix_accuracy(loso_cm_direction, CORRUPTIONS)

print(f"\n  LOSO Overall Accuracy:")
print(f"    Centroid:  {loso_acc_centroid:.4f} ({loso_acc_centroid*100:.1f}%)")
print(f"    Direction: {loso_acc_direction:.4f} ({loso_acc_direction*100:.1f}%)")

print(f"\n  LOSO Per-class Accuracy:")
for c in CORRUPTIONS:
    print(f"    {c}: centroid={loso_per_class_c[c]:.4f}, direction={loso_per_class_d[c]:.4f}")

print("\n  LOSO Centroid Confusion Matrix:")
print("  {:>8s}".format("true\\pred") + "".join(f"  {c:>7s}" for c in CORRUPTIONS))
for true_c in CORRUPTIONS:
    row = f"  {true_c:>8s}"
    for pred_c in CORRUPTIONS:
        row += f"  {loso_cm_centroid[true_c][pred_c]:>7d}"
    print(row)

results["test5_leave_one_scene_out"] = {
    "centroid_accuracy": loso_acc_centroid,
    "direction_accuracy": loso_acc_direction,
    "per_scene": loso_per_scene,
    "per_class_centroid": loso_per_class_c,
    "per_class_direction": loso_per_class_d,
    "confusion_matrix_centroid": {
        t: {p: loso_cm_centroid[t][p] for p in CORRUPTIONS} for t in CORRUPTIONS
    },
    "confusion_matrix_direction": {
        t: {p: loso_cm_direction[t][p] for p in CORRUPTIONS} for t in CORRUPTIONS
    },
    "n_folds": N_SCENES,
    "n_total_predictions": len(loso_true)
}

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary = {
    "test1_centroid_accuracy_cosine": results["test1_nearest_centroid"]["accuracy_cosine"],
    "test1_centroid_accuracy_euclidean": results["test1_nearest_centroid"]["accuracy_euclidean"],
    "test2_direction_accuracy": results["test2_direction_classification"]["accuracy"],
    "test3_centroid_overall_accuracy": results["test3_confusion_matrix"]["centroid"]["overall_accuracy"],
    "test3_direction_overall_accuracy": results["test3_confusion_matrix"]["direction"]["overall_accuracy"],
    "test4_severity_centroid_accuracies": {
        str(s): severity_results[str(s)]["centroid_accuracy"] for s in SEVERITIES
    },
    "test4_severity_direction_accuracies": {
        str(s): severity_results[str(s)]["direction_accuracy"] for s in SEVERITIES
    },
    "test5_loso_centroid_accuracy": loso_acc_centroid,
    "test5_loso_direction_accuracy": loso_acc_direction
}

for key, val in summary.items():
    if isinstance(val, dict):
        print(f"  {key}:")
        for k2, v2 in val.items():
            print(f"    {k2}: {v2:.4f}")
    else:
        print(f"  {key}: {val:.4f}")

# Determine if corruption types are classifiable
classifiable = (loso_acc_centroid > 0.5 or loso_acc_direction > 0.5)
print(f"\n  Corruption types classifiable from embeddings: {classifiable}")
print(f"  Best method: {'centroid' if loso_acc_centroid >= loso_acc_direction else 'direction'}")

results["summary"] = summary
results["classifiable"] = classifiable
results["best_method"] = "centroid" if loso_acc_centroid >= loso_acc_direction else "direction"

# =========================================================================
# Save results
# =========================================================================
ts = time.strftime("%Y%m%d_%H%M%S")
output = {
    "experiment": "corruption_classification",
    "experiment_number": 439,
    "timestamp": ts,
    "config": {
        "n_scenes": N_SCENES,
        "seeds": SEEDS,
        "corruptions": CORRUPTIONS,
        "severities": SEVERITIES,
        "layer": LAYER,
        "image_size": 224,
        "model": "openvla/openvla-7b"
    },
    "results": results
}

out_path = "/workspace/Vizuara-VLA-Research/experiments/corruption_classification_" + ts + ".json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {out_path}")
print("Experiment 439 complete.")
