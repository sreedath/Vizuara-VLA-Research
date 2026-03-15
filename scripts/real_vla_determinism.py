#!/usr/bin/env python3
"""Experiment 178: Inference determinism — repeated forward passes on same image.

Tests whether the same image produces identical embeddings across
multiple forward passes (determinism check) and measures the variance.
"""

import json, os, sys, datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image

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

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 178: Inference Determinism")
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

    images = [Image.fromarray(create_highway(i)) for i in range(3)]
    n_repeats = 10

    results = {}
    for img_idx, img in enumerate(images):
        print(f"\n--- Image {img_idx} ({n_repeats} repeats) ---", flush=True)
        embs = {l: [] for l in layers}
        for rep in range(n_repeats):
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                embs[l].append(h[l])

        img_results = {}
        for l in layers:
            emb_arr = np.array(embs[l])
            diffs = np.abs(emb_arr - emb_arr[0])
            max_diff = float(diffs.max())
            mean_diff = float(diffs.mean())

            pairwise_cos = []
            for i in range(n_repeats):
                for j in range(i+1, n_repeats):
                    pairwise_cos.append(cosine_distance(emb_arr[i], emb_arr[j]))

            l2_dists = [float(np.linalg.norm(emb_arr[i] - emb_arr[i+1]))
                       for i in range(n_repeats - 1)]

            img_results[f"L{l}"] = {
                "max_element_diff": max_diff,
                "mean_element_diff": mean_diff,
                "all_identical": bool(max_diff == 0),
                "pairwise_cosine_dist": {
                    "mean": float(np.mean(pairwise_cos)) if pairwise_cos else 0,
                    "max": float(np.max(pairwise_cos)) if pairwise_cos else 0,
                },
                "l2_between_passes": {
                    "mean": float(np.mean(l2_dists)),
                    "max": float(np.max(l2_dists)),
                },
                "embedding_norm": float(np.linalg.norm(emb_arr[0])),
            }
            status = "IDENTICAL" if max_diff == 0 else f"max_diff={max_diff:.2e}"
            print(f"  L{l}: {status}, cos_dist_mean={np.mean(pairwise_cos):.2e}", flush=True)

        results[f"image_{img_idx}"] = img_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "determinism",
        "experiment_number": 178,
        "timestamp": ts,
        "n_images": len(images),
        "n_repeats": n_repeats,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"determinism_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
