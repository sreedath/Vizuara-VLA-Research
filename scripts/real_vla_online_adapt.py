#!/usr/bin/env python3
"""Experiment 208: Online centroid adaptation — can the centroid be updated
incrementally as new clean frames arrive? Tests exponential moving average
(EMA) update of the centroid.
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
def apply_noise(a, s=50): return np.clip(a.astype(np.float32)+np.random.normal(0,s,a.shape), 0, 255).astype(np.uint8)

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
    print("Experiment 208: Online Centroid Adaptation")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [1, 3]

    creators = [create_highway, create_urban, create_rural]

    def extract_all(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    # Extract all embeddings first (30 total: 3 for initial, 20 for online, 7 for test)
    print("\n--- Extracting all embeddings ---", flush=True)
    all_arrs = [creators[i%3](i) for i in range(30)]
    all_embs = []
    for i, arr in enumerate(all_arrs):
        h = extract_all(Image.fromarray(arr))
        all_embs.append(h)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/30", flush=True)

    # OOD test embeddings
    ood_transforms = {
        "fog": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "noise": lambda a: apply_noise(a, 50),
    }
    test_ids = list(range(23, 30))  # last 7 for testing
    ood_embs = {cat: [] for cat in ood_transforms}
    for cat, tfn in ood_transforms.items():
        for i in test_ids:
            h = extract_all(Image.fromarray(tfn(all_arrs[i])))
            ood_embs[cat].append(h)

    # Simulate online learning
    print("\n--- Online adaptation ---", flush=True)
    ema_alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    results = {}
    for alpha in ema_alphas:
        alpha_results = {}
        
        for l in layers:
            # Start with 3 initial samples
            initial_embs = [all_embs[i][l] for i in range(3)]
            centroid = np.mean(initial_embs, axis=0)
            
            # Track AUROC over time
            auroc_trace = []
            
            # Online updates with frames 3-22
            for t in range(3, 23):
                new_emb = all_embs[t][l]
                centroid = (1 - alpha) * centroid + alpha * new_emb
                
                # Evaluate AUROC at this point
                id_dists = [cosine_distance(all_embs[i][l], centroid) for i in test_ids]
                all_ood_dists = []
                for cat in ood_transforms:
                    for emb in ood_embs[cat]:
                        all_ood_dists.append(cosine_distance(emb[l], centroid))
                
                auroc = compute_auroc(id_dists, all_ood_dists)
                auroc_trace.append({"t": t, "auroc": auroc})
            
            # Also evaluate static centroid (n=3 only, no adaptation)
            static_centroid = np.mean(initial_embs, axis=0)
            id_dists_static = [cosine_distance(all_embs[i][l], static_centroid) for i in test_ids]
            all_ood_static = []
            for cat in ood_transforms:
                for emb in ood_embs[cat]:
                    all_ood_static.append(cosine_distance(emb[l], static_centroid))
            auroc_static = compute_auroc(id_dists_static, all_ood_static)
            
            alpha_results[f"L{l}"] = {
                "auroc_trace": auroc_trace,
                "final_auroc": auroc_trace[-1]["auroc"],
                "static_auroc_n3": auroc_static,
            }
            
            print(f"  alpha={alpha} L{l}: final={auroc_trace[-1]['auroc']:.4f} "
                  f"static_n3={auroc_static:.4f}", flush=True)
        
        results[f"alpha_{alpha}"] = alpha_results

    # Also: batch centroid with all 23 samples
    batch_results = {}
    for l in layers:
        batch_centroid = np.mean([all_embs[i][l] for i in range(23)], axis=0)
        id_dists = [cosine_distance(all_embs[i][l], batch_centroid) for i in test_ids]
        all_ood_dists = []
        for cat in ood_transforms:
            for emb in ood_embs[cat]:
                all_ood_dists.append(cosine_distance(emb[l], batch_centroid))
        batch_results[f"L{l}"] = compute_auroc(id_dists, all_ood_dists)
        print(f"  batch_n23 L{l}: {batch_results[f'L{l}']:.4f}", flush=True)
    results["batch_n23"] = batch_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "online_adaptation",
        "experiment_number": 208,
        "timestamp": ts,
        "ema_alphas": ema_alphas,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"online_adaptation_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
