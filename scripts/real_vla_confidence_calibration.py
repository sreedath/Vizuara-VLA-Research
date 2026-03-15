#!/usr/bin/env python3
"""Experiment 414: Confidence Calibration — OOD Distance vs Token Entropy

Studies whether the model's own token-level uncertainty (logit entropy)
correlates with our external OOD detector's cosine distance. Tests whether
entropy can serve as a complementary or standalone confidence signal, and
whether the model "knows" it's receiving corrupted input.

Tests:
1. Distance vs action-token entropy correlation across severity levels
2. Distance vs full-vocabulary entropy correlation
3. Top-1 action probability as confidence measure
4. Per-corruption entropy profiles
5. Joint distance-entropy detection (can combining them improve over either alone?)
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

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

def extract_hidden_and_entropy(model, processor, image, prompt, layer=3):
    """Extract hidden state AND entropy metrics in a single forward pass."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    hidden = fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

    # Logits for next-token prediction
    logits = fwd.logits[0, -1, :].float().cpu()

    # Full vocabulary entropy
    probs = torch.softmax(logits, dim=0)
    full_entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))

    # Action token entropy (token IDs 31744-31999)
    action_logits = logits[31744:32000]
    action_probs = torch.softmax(action_logits, dim=0)
    action_entropy = float(-torch.sum(action_probs * torch.log(action_probs + 1e-10)))

    # Top-1 action probability
    top1_prob = float(action_probs.max())
    top1_token = int(action_probs.argmax()) + 31744

    # Top-5 action tokens
    top5_vals, top5_idx = torch.topk(action_probs, 5)
    top5 = [(int(idx) + 31744, float(val)) for idx, val in zip(top5_idx, top5_vals)]

    return hidden, {
        "full_entropy": full_entropy,
        "action_entropy": action_entropy,
        "top1_prob": top1_prob,
        "top1_token": top1_token,
        "top5": top5
    }

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']
    severities = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Generate scenes
    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # === Calibration: extract clean embeddings and entropies ===
    print("Extracting clean embeddings and entropies...")
    clean_data = []
    for s in scenes:
        h, ent = extract_hidden_and_entropy(model, processor, s, prompt)
        clean_data.append({"hidden": h, "entropy": ent})

    centroid = np.mean([d["hidden"] for d in clean_data], axis=0)
    clean_dists = [cosine_dist(d["hidden"], centroid) for d in clean_data]
    clean_action_ents = [d["entropy"]["action_entropy"] for d in clean_data]
    clean_full_ents = [d["entropy"]["full_entropy"] for d in clean_data]
    clean_top1_probs = [d["entropy"]["top1_prob"] for d in clean_data]

    results = {}

    # === Test 1: Per-condition distance and entropy ===
    print("\n=== Per-Condition Profiles ===")
    all_points = []
    condition_results = {}

    # Clean condition
    condition_results["clean"] = {
        "dist_mean": float(np.mean(clean_dists)),
        "dist_std": float(np.std(clean_dists)),
        "action_entropy_mean": float(np.mean(clean_action_ents)),
        "action_entropy_std": float(np.std(clean_action_ents)),
        "full_entropy_mean": float(np.mean(clean_full_ents)),
        "full_entropy_std": float(np.std(clean_full_ents)),
        "top1_prob_mean": float(np.mean(clean_top1_probs)),
        "top1_prob_std": float(np.std(clean_top1_probs)),
    }
    for i, d in enumerate(clean_data):
        all_points.append({
            "condition": "clean", "severity": 0.0, "scene": i,
            "dist": clean_dists[i],
            "action_entropy": d["entropy"]["action_entropy"],
            "full_entropy": d["entropy"]["full_entropy"],
            "top1_prob": d["entropy"]["top1_prob"],
        })
    print(f"  clean: dist={np.mean(clean_dists):.6f}, act_ent={np.mean(clean_action_ents):.3f}, top1={np.mean(clean_top1_probs):.4f}")

    for c in corruptions:
        for sev in severities:
            cname = f"{c}_{sev}"
            dists, action_ents, full_ents, top1s = [], [], [], []
            for i, s in enumerate(scenes):
                corrupted = apply_corruption(s, c, sev)
                h, ent = extract_hidden_and_entropy(model, processor, corrupted, prompt)
                d = cosine_dist(h, centroid)
                dists.append(d)
                action_ents.append(ent["action_entropy"])
                full_ents.append(ent["full_entropy"])
                top1s.append(ent["top1_prob"])
                all_points.append({
                    "condition": cname, "severity": sev, "scene": i,
                    "dist": d,
                    "action_entropy": ent["action_entropy"],
                    "full_entropy": ent["full_entropy"],
                    "top1_prob": ent["top1_prob"],
                })

            condition_results[cname] = {
                "dist_mean": float(np.mean(dists)),
                "dist_std": float(np.std(dists)),
                "action_entropy_mean": float(np.mean(action_ents)),
                "action_entropy_std": float(np.std(action_ents)),
                "full_entropy_mean": float(np.mean(full_ents)),
                "full_entropy_std": float(np.std(full_ents)),
                "top1_prob_mean": float(np.mean(top1s)),
                "top1_prob_std": float(np.std(top1s)),
            }
            print(f"  {cname}: dist={np.mean(dists):.6f}, act_ent={np.mean(action_ents):.3f}, top1={np.mean(top1s):.4f}")

    results["conditions"] = condition_results

    # === Test 2: Correlations ===
    print("\n=== Correlations ===")
    dist_arr = np.array([p["dist"] for p in all_points])
    ae_arr = np.array([p["action_entropy"] for p in all_points])
    fe_arr = np.array([p["full_entropy"] for p in all_points])
    t1_arr = np.array([p["top1_prob"] for p in all_points])

    correlations = {
        "dist_vs_action_entropy": pearson(dist_arr, ae_arr),
        "dist_vs_full_entropy": pearson(dist_arr, fe_arr),
        "dist_vs_top1_prob": pearson(dist_arr, t1_arr),
        "action_entropy_vs_full_entropy": pearson(ae_arr, fe_arr),
    }
    for k, v in correlations.items():
        print(f"  {k}: r = {v:.4f}")
    results["correlations"] = correlations

    # === Test 3: Per-corruption correlations ===
    print("\n=== Per-Corruption Correlations ===")
    per_corr_correlations = {}
    for c in corruptions:
        c_points = [p for p in all_points if p["condition"].startswith(c)]
        if len(c_points) < 3:
            continue
        c_dist = np.array([p["dist"] for p in c_points])
        c_ae = np.array([p["action_entropy"] for p in c_points])
        c_t1 = np.array([p["top1_prob"] for p in c_points])
        per_corr_correlations[c] = {
            "dist_vs_action_entropy": pearson(c_dist, c_ae),
            "dist_vs_top1_prob": pearson(c_dist, c_t1),
            "n_points": len(c_points),
        }
        print(f"  {c}: dist_vs_act_ent={per_corr_correlations[c]['dist_vs_action_entropy']:.4f}, "
              f"dist_vs_top1={per_corr_correlations[c]['dist_vs_top1_prob']:.4f}")
    results["per_corruption_correlations"] = per_corr_correlations

    # === Test 4: Entropy-only AUROC ===
    print("\n=== Entropy-Only AUROC ===")
    entropy_auroc = {}
    for c in corruptions:
        for sev in severities:
            cname = f"{c}_{sev}"
            c_ents = [p["action_entropy"] for p in all_points if p["condition"] == cname]
            auroc_ent = compute_auroc(clean_action_ents, c_ents)
            auroc_dist = compute_auroc(clean_dists,
                                       [p["dist"] for p in all_points if p["condition"] == cname])
            entropy_auroc[cname] = {
                "auroc_entropy": float(auroc_ent),
                "auroc_distance": float(auroc_dist),
            }
            print(f"  {cname}: AUROC(entropy)={auroc_ent:.4f}, AUROC(distance)={auroc_dist:.4f}")
    results["entropy_auroc"] = entropy_auroc

    # === Test 5: Joint detection (distance + entropy) ===
    print("\n=== Joint Detection ===")
    # Normalize dist and entropy to [0,1] range
    if np.std(dist_arr) > 1e-10:
        dist_norm = (dist_arr - np.min(dist_arr)) / (np.max(dist_arr) - np.min(dist_arr) + 1e-10)
    else:
        dist_norm = np.zeros_like(dist_arr)
    if np.std(ae_arr) > 1e-10:
        ae_norm = (ae_arr - np.min(ae_arr)) / (np.max(ae_arr) - np.min(ae_arr) + 1e-10)
    else:
        ae_norm = np.zeros_like(ae_arr)

    joint_scores = dist_norm + ae_norm
    clean_joint = [joint_scores[i] for i in range(len(all_points)) if all_points[i]["condition"] == "clean"]

    joint_results = {}
    for c in corruptions:
        c_joint = [joint_scores[i] for i in range(len(all_points))
                   if all_points[i]["condition"] == f"{c}_1.0"]
        if len(c_joint) > 0:
            joint_auroc = compute_auroc(clean_joint, c_joint)
            joint_results[c] = float(joint_auroc)
            print(f"  {c}: joint AUROC={joint_auroc:.4f}")
    results["joint_auroc"] = joint_results

    # === Test 6: Does the model "know"? (entropy direction) ===
    print("\n=== Model Self-Awareness ===")
    awareness = {}
    for c in corruptions:
        clean_mean_ent = float(np.mean(clean_action_ents))
        corrupt_ents = [p["action_entropy"] for p in all_points if p["condition"] == f"{c}_1.0"]
        corrupt_mean_ent = float(np.mean(corrupt_ents))
        direction = "increases" if corrupt_mean_ent > clean_mean_ent else "decreases"
        awareness[c] = {
            "clean_entropy": clean_mean_ent,
            "corrupt_entropy": corrupt_mean_ent,
            "direction": direction,
            "ratio": corrupt_mean_ent / (clean_mean_ent + 1e-10),
        }
        print(f"  {c}: clean_ent={clean_mean_ent:.3f}, corrupt_ent={corrupt_mean_ent:.3f} ({direction})")
    results["model_awareness"] = awareness

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/confidence_calibration_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
