#!/usr/bin/env python3
"""Experiment 317: Deep Information-Theoretic Analysis
Formalizes detection through information theory:
1. Mutual information between corruption and embedding distance
2. Channel capacity of the corruption-to-distance channel
3. Rate-distortion tradeoff under dimension reduction
4. Fisher information for severity estimation
5. Entropy of action token distribution under corruption
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine
from scipy import stats

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

def get_action_tokens(model, processor, image, prompt):
    ACTION_TOKEN_START = 31744
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=7, do_sample=False)
    input_len = inputs['input_ids'].shape[1]
    gen_tokens = generated[0, input_len:].cpu().numpy()
    return [int(t - ACTION_TOKEN_START) for t in gen_tokens]

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "info_theory_deep",
        "experiment_number": 317,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    clean_emb = extract_hidden(model, processor, base_img, prompt)
    clean_actions = get_action_tokens(model, processor, base_img, prompt)

    # Part 1: Distance-Severity Information Content
    print("=== Part 1: Distance-Severity Mutual Information ===")
    mi_results = {}

    severities = np.linspace(0.05, 1.0, 20)

    for c in corruptions:
        print(f"  {c}...")
        dists = []
        for sev in severities:
            corrupted = apply_corruption(base_img, c, float(sev))
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(clean_emb, emb))
            dists.append(d)

        dists = np.array(dists)

        # Spearman correlation (rank-based MI proxy)
        rho, p_val = stats.spearmanr(severities, dists)

        # Mutual information via binning
        n_bins = 5
        sev_bins = np.digitize(severities, np.linspace(0, 1, n_bins + 1))
        dist_bins = np.digitize(dists, np.linspace(0, max(dists) * 1.01, n_bins + 1))

        # Joint histogram
        joint = np.zeros((n_bins, n_bins))
        for sb, db in zip(sev_bins, dist_bins):
            sb_idx = min(sb - 1, n_bins - 1)
            db_idx = min(db - 1, n_bins - 1)
            joint[sb_idx, db_idx] += 1
        joint /= joint.sum()

        # Marginals
        p_sev = joint.sum(axis=1)
        p_dist = joint.sum(axis=0)

        # MI
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and p_sev[i] > 0 and p_dist[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (p_sev[i] * p_dist[j]))

        # Max MI (entropy of severity, uniform)
        h_sev = -np.sum(p_sev[p_sev > 0] * np.log2(p_sev[p_sev > 0]))

        mi_results[c] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "mutual_info_bits": float(mi),
            "severity_entropy_bits": float(h_sev),
            "normalized_mi": float(mi / h_sev) if h_sev > 0 else 0,
            "distances": dists.tolist(),
        }
        print(f"    MI={mi:.3f} bits, H(sev)={h_sev:.3f} bits, NMI={mi/h_sev:.3f}, rho={rho:.4f}")

    results["mutual_information"] = mi_results

    # Part 2: Action Token Entropy Under Corruption
    print("\n=== Part 2: Action Token Entropy ===")
    action_entropy = {}

    for c in corruptions:
        print(f"  {c}...")
        action_sets = []
        for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            actions = get_action_tokens(model, processor, corrupted, prompt)
            action_sets.append({"severity": sev, "actions": actions})

        # Compute per-dimension entropy across severities
        dim_entropies = []
        for dim in range(7):
            vals = [a["actions"][dim] for a in action_sets]
            unique, counts = np.unique(vals, return_counts=True)
            probs = counts / counts.sum()
            h = -np.sum(probs * np.log2(probs))
            dim_entropies.append(float(h))

        action_entropy[c] = {
            "action_sets": action_sets,
            "per_dim_entropy": dim_entropies,
            "mean_entropy": float(np.mean(dim_entropies)),
            "max_entropy": float(max(dim_entropies)),
        }
        print(f"    Mean entropy: {np.mean(dim_entropies):.3f} bits, max: {max(dim_entropies):.3f} bits")

    results["action_entropy"] = action_entropy

    # Part 3: Dimension-wise Fisher Information
    print("\n=== Part 3: Fisher Information for Severity ===")
    fisher = {}

    for c in corruptions:
        print(f"  {c}...")
        # Collect embeddings at fine severity grid
        sevs = np.linspace(0.05, 0.95, 10)
        embs = []
        for sev in sevs:
            corrupted = apply_corruption(base_img, c, float(sev))
            emb = extract_hidden(model, processor, corrupted, prompt)
            embs.append(emb)

        embs = np.array(embs)  # [10, 4096]

        # Numerical gradient: d(embedding)/d(severity)
        gradients = np.diff(embs, axis=0) / np.diff(sevs)[:, None]  # [9, 4096]

        # Fisher info ~ sum of squared gradients (assuming unit noise)
        fisher_per_dim = np.mean(gradients ** 2, axis=0)  # [4096]
        total_fisher = float(np.sum(fisher_per_dim))

        # Top dimensions by Fisher info
        top_dims = np.argsort(fisher_per_dim)[-10:][::-1]

        fisher[c] = {
            "total_fisher_info": total_fisher,
            "mean_fisher_per_dim": float(np.mean(fisher_per_dim)),
            "top_10_dims": top_dims.tolist(),
            "top_10_fisher": fisher_per_dim[top_dims].tolist(),
            "n_informative_dims": int(np.sum(fisher_per_dim > np.mean(fisher_per_dim))),
            "gini_coefficient": float(1 - 2 * np.sum(np.sort(fisher_per_dim).cumsum() / fisher_per_dim.sum()) / len(fisher_per_dim)),
        }
        print(f"    Total Fisher: {total_fisher:.2f}, informative dims: {fisher[c]['n_informative_dims']}/4096, "
              f"Gini: {fisher[c]['gini_coefficient']:.3f}")

    results["fisher"] = fisher

    # Part 4: Rate-Distortion Analysis
    print("\n=== Part 4: Rate-Distortion (Dimension Truncation) ===")
    rd = {}

    for c in corruptions:
        corrupted = apply_corruption(base_img, c, 0.5)
        emb = extract_hidden(model, processor, corrupted, prompt)
        full_dist = float(cosine(clean_emb, emb))

        trunc_results = []
        for n_dims in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            trunc_dist = float(cosine(clean_emb[:n_dims], emb[:n_dims]))
            distortion = abs(trunc_dist - full_dist) / full_dist if full_dist > 0 else 0
            trunc_results.append({
                "dims": n_dims,
                "rate_bits": float(np.log2(n_dims)),
                "distance": trunc_dist,
                "distortion": float(distortion),
                "detected": trunc_dist > 0,
            })

        rd[c] = trunc_results

    results["rate_distortion"] = rd
    for c in corruptions:
        print(f"  {c}: ", end="")
        for r in rd[c]:
            if r['dims'] in [1, 4, 32, 256, 4096]:
                print(f"{r['dims']}D={r['distance']:.6f} ", end="")
        print()

    # Part 5: Cross-corruption information
    print("\n=== Part 5: Cross-Corruption Type Distinguishability ===")
    # Measure how much information distance carries about corruption TYPE
    type_dists = {}
    for c in corruptions:
        corrupted = apply_corruption(base_img, c, 0.5)
        emb = extract_hidden(model, processor, corrupted, prompt)
        type_dists[c] = float(cosine(clean_emb, emb))

    # Can we distinguish types from distance alone?
    # Also check embedding direction
    type_dirs = {}
    for c in corruptions:
        corrupted = apply_corruption(base_img, c, 0.5)
        emb = extract_hidden(model, processor, corrupted, prompt)
        diff = emb - clean_emb
        type_dirs[c] = diff / (np.linalg.norm(diff) + 1e-30)

    # Pairwise direction similarity
    type_confusion = {}
    for c1 in corruptions:
        for c2 in corruptions:
            if c1 < c2:
                sim = float(np.dot(type_dirs[c1], type_dirs[c2]))
                type_confusion[f"{c1}_vs_{c2}"] = sim
                print(f"  {c1} vs {c2}: cos_sim={sim:.4f}")

    results["type_distinguishability"] = {
        "distances": type_dists,
        "direction_similarities": type_confusion,
    }

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/info_deep_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
