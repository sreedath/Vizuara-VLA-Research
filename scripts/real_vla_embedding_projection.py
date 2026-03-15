#!/usr/bin/env python3
"""Experiment 181: Embedding projection analysis — do ID and OOD embeddings
project differently onto the action head's weight space?

Tests whether the OOD signal is visible in the action-relevant subspace
of the embedding, connecting detection to action divergence.
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

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def main():
    print("=" * 60)
    print("Experiment 181: Embedding Projection Analysis")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    # Extract LM head weights for action tokens (31744-31999)
    print("Extracting LM head action subspace...", flush=True)
    lm_head = model.language_model.lm_head.weight.data.float().cpu().numpy()  # [vocab, hidden]
    action_weights = lm_head[31744:32000]  # [256, hidden_dim]
    print(f"  Action head shape: {action_weights.shape}", flush=True)

    # SVD of action weight subspace
    U, S, Vt = np.linalg.svd(action_weights, full_matrices=False)
    action_subspace = Vt[:10]  # Top 10 directions of action token space
    print(f"  Action subspace singular values: {S[:10]}", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [3, 32]

    creators = [create_highway, create_urban, create_rural]
    n_cal = 8
    n_test = 6

    # Calibrate
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        centroids[l] = np.array(cal_embs).mean(axis=0)

    # Test
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_embs = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_embs[l].append(h[l])

    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_embs = {l: [] for l in layers}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h = extract_hidden(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for l in layers:
                ood_embs[l].append(h[l])

    # Project embeddings onto action subspace
    results = {}
    for l in layers:
        layer_results = {}

        # Full-space AUROC
        id_dists = [cosine_distance(e, centroids[l]) for e in id_embs[l]]
        ood_dists = [cosine_distance(e, centroids[l]) for e in ood_embs[l]]
        full_auroc = compute_auroc(id_dists, ood_dists)

        # Action-subspace projection AUROC (project onto top-k action directions)
        for k in [2, 5, 10]:
            proj = action_subspace[:k]
            id_proj = [e @ proj.T for e in id_embs[l]]
            ood_proj = [e @ proj.T for e in ood_embs[l]]
            cent_proj = centroids[l] @ proj.T

            id_proj_dists = [float(np.linalg.norm(p - cent_proj)) for p in id_proj]
            ood_proj_dists = [float(np.linalg.norm(p - cent_proj)) for p in ood_proj]
            proj_auroc = compute_auroc(id_proj_dists, ood_proj_dists)
            layer_results[f"action_proj_{k}_auroc"] = proj_auroc
            print(f"  L{l} action_proj_{k}: AUROC={proj_auroc:.4f}", flush=True)

        # Complementary subspace (orthogonal to action space)
        # Project onto null space of action subspace
        proj_matrix = action_subspace[:10].T @ action_subspace[:10]  # projection
        for e_type, embs_list, name in [("id", id_embs[l], "id"), ("ood", ood_embs[l], "ood")]:
            null_projs = []
            action_projs = []
            for e in embs_list:
                action_proj = proj_matrix @ e
                null_proj = e - action_proj
                null_projs.append(float(np.linalg.norm(null_proj)))
                action_projs.append(float(np.linalg.norm(action_proj)))
            layer_results[f"{name}_action_norm"] = {"mean": float(np.mean(action_projs)), "std": float(np.std(action_projs))}
            layer_results[f"{name}_null_norm"] = {"mean": float(np.mean(null_projs)), "std": float(np.std(null_projs))}

        # AUROC using only action subspace vs only null subspace
        id_action = [float(np.linalg.norm(proj_matrix @ e - proj_matrix @ centroids[l])) for e in id_embs[l]]
        ood_action = [float(np.linalg.norm(proj_matrix @ e - proj_matrix @ centroids[l])) for e in ood_embs[l]]
        auroc_action = compute_auroc(id_action, ood_action)

        id_null = [float(np.linalg.norm((e - proj_matrix @ e) - (centroids[l] - proj_matrix @ centroids[l]))) for e in id_embs[l]]
        ood_null = [float(np.linalg.norm((e - proj_matrix @ e) - (centroids[l] - proj_matrix @ centroids[l]))) for e in ood_embs[l]]
        auroc_null = compute_auroc(id_null, ood_null)

        layer_results["full_auroc"] = full_auroc
        layer_results["action_subspace_auroc"] = auroc_action
        layer_results["null_subspace_auroc"] = auroc_null
        print(f"  L{l}: full={full_auroc:.4f} action={auroc_action:.4f} null={auroc_null:.4f}", flush=True)

        results[f"L{l}"] = layer_results

    results["action_singular_values"] = S[:20].tolist()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "embedding_projection",
        "experiment_number": 181,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"embedding_projection_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
