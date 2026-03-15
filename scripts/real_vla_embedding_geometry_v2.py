#!/usr/bin/env python3
"""Experiment 173: Embedding space geometry — isotropy and curvature analysis.

Analyzes geometric properties of VLA embeddings: isotropy, angular
concentration, intrinsic dimensionality, and whether ID/OOD occupy
different geometric substructures.
"""

import json, os, sys, datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent
EXPERIMENTS_DIR = REPO_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = str(EXPERIMENTS_DIR)

SIZE = (256, 256)
rng = np.random.RandomState(42)

def create_highway(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]; img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_urban(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]; img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]; img[SIZE[0]//2:] = [60, 60, 60]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_rural(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [100, 180, 255]; img[SIZE[0]//3:SIZE[0]*2//3] = [34, 139, 34]; img[SIZE[0]*2//3:] = [90, 90, 90]
    return np.clip(img.astype(np.int16) + rng.randint(-8, 9, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def apply_fog(a, alpha):
    return np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8)
def apply_night(a): return np.clip(a*0.15, 0, 255).astype(np.uint8)
def apply_blur(a, r=8): return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=r)))
def apply_noise(a, s=50): return np.clip(a.astype(np.float32)+np.random.normal(0,s,a.shape), 0, 255).astype(np.uint8)

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def main():
    print("=" * 60)
    print("Experiment 173: Embedding Space Geometry Analysis")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [3, 32]

    creators = [create_highway, create_urban, create_rural]
    n_id = 10

    # Extract ID embeddings
    print("\n--- Extracting ID embeddings ---", flush=True)
    id_arrs = [creators[i%3](i) for i in range(n_id)]
    id_embs = {l: [] for l in layers}
    for arr in id_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_embs[l].append(h[l])

    # Extract OOD embeddings
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_embs = {l: [] for l in layers}
    ood_labels = []
    for cat, tfn in ood_transforms.items():
        for j in range(6):
            arr = tfn(id_arrs[j % n_id])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_embs[l].append(h[l])
            ood_labels.append(cat)
    print("  All embeddings extracted.", flush=True)

    results = {}
    for l in layers:
        layer_results = {}
        id_mat = np.array(id_embs[l])
        ood_mat = np.array(ood_embs[l])

        # 1. Isotropy analysis
        centered = id_mat - id_mat.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        isotropy_ratio = float(eigvals.min() / eigvals.max()) if len(eigvals) > 0 else 0
        p = eigvals / eigvals.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-20))))
        layer_results["isotropy_ratio"] = isotropy_ratio
        layer_results["effective_rank"] = eff_rank
        layer_results["n_positive_eigvals"] = int(len(eigvals))
        print(f"\n  L{l} isotropy ratio: {isotropy_ratio:.6f}", flush=True)
        print(f"  L{l} effective rank: {eff_rank:.2f} / {len(eigvals)}", flush=True)

        # 2. Angular concentration
        id_pairwise_cos = []
        for i in range(len(id_mat)):
            for j in range(i+1, len(id_mat)):
                id_pairwise_cos.append(cosine_similarity(id_mat[i], id_mat[j]))
        layer_results["id_cosine_sim"] = {
            "mean": float(np.mean(id_pairwise_cos)),
            "std": float(np.std(id_pairwise_cos)),
            "min": float(np.min(id_pairwise_cos)),
            "max": float(np.max(id_pairwise_cos)),
        }
        print(f"  L{l} ID pairwise cos sim: mean={np.mean(id_pairwise_cos):.6f} std={np.std(id_pairwise_cos):.6f}", flush=True)

        # 3. ID-OOD angular separation
        cross_cos = []
        for ie in id_mat:
            for oe in ood_mat:
                cross_cos.append(cosine_similarity(ie, oe))
        layer_results["id_ood_cosine_sim"] = {
            "mean": float(np.mean(cross_cos)),
            "std": float(np.std(cross_cos)),
        }
        print(f"  L{l} ID-OOD cos sim: mean={np.mean(cross_cos):.6f} std={np.std(cross_cos):.6f}", flush=True)

        # 4. OOD intra-category similarity
        ood_intra = {}
        for cat in ood_transforms:
            cat_idx = [i for i, lb in enumerate(ood_labels) if lb == cat]
            cat_embs = ood_mat[cat_idx]
            pairwise = []
            for i in range(len(cat_embs)):
                for j in range(i+1, len(cat_embs)):
                    pairwise.append(cosine_similarity(cat_embs[i], cat_embs[j]))
            ood_intra[cat] = {"mean": float(np.mean(pairwise)) if pairwise else 0}
        layer_results["ood_intra_sim"] = ood_intra

        # 5. Intrinsic dimensionality (MLE)
        for name, mat in [("id", id_mat), ("ood", ood_mat)]:
            n = len(mat)
            dists_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dists_matrix[i, j] = np.linalg.norm(mat[i] - mat[j])
            k = min(n - 1, 8)
            int_dims = []
            for i in range(n):
                sorted_dists = np.sort(dists_matrix[i])[1:k+1]
                if sorted_dists[-1] > 0:
                    log_ratios = np.log(sorted_dists[-1] / (sorted_dists[:-1] + 1e-20))
                    if np.sum(log_ratios) > 0:
                        int_dims.append(float((k - 1) / np.sum(log_ratios)))
            layer_results[f"{name}_intrinsic_dim"] = {
                "mean": float(np.mean(int_dims)) if int_dims else 0,
                "std": float(np.std(int_dims)) if int_dims else 0,
            }
            print(f"  L{l} {name} intrinsic dim: {np.mean(int_dims):.2f} ± {np.std(int_dims):.2f}" if int_dims else f"  L{l} {name} intrinsic dim: N/A", flush=True)

        # 6. Norm statistics
        id_norms = np.linalg.norm(id_mat, axis=1)
        ood_norms = np.linalg.norm(ood_mat, axis=1)
        layer_results["id_norms"] = {"mean": float(np.mean(id_norms)), "std": float(np.std(id_norms))}
        layer_results["ood_norms"] = {"mean": float(np.mean(ood_norms)), "std": float(np.std(ood_norms))}
        norm_auroc = compute_auroc(id_norms.tolist(), ood_norms.tolist())
        layer_results["norm_auroc"] = norm_auroc
        print(f"  L{l} norm AUROC: {norm_auroc:.4f}", flush=True)

        # 7. Anisotropy
        mean_dir = id_mat.mean(axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
        id_aniso = [cosine_similarity(e, mean_dir) for e in id_mat]
        ood_aniso = [cosine_similarity(e, mean_dir) for e in ood_mat]
        layer_results["anisotropy"] = {
            "id_to_mean": {"mean": float(np.mean(id_aniso)), "std": float(np.std(id_aniso))},
            "ood_to_mean": {"mean": float(np.mean(ood_aniso)), "std": float(np.std(ood_aniso))},
        }
        print(f"  L{l} anisotropy ID→mean: {np.mean(id_aniso):.6f}, OOD→mean: {np.mean(ood_aniso):.6f}", flush=True)

        results[f"L{l}"] = layer_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "embedding_geometry_v2",
        "experiment_number": 173,
        "timestamp": ts,
        "n_id": n_id,
        "n_ood": len(ood_embs[layers[0]]),
        "ood_categories": list(ood_transforms.keys()),
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"embedding_geometry_v2_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
