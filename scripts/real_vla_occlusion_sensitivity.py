#!/usr/bin/env python3
"""Experiment 174: Occlusion sensitivity mapping for OOD detection.

Systematically occludes different regions of the input image to identify
which spatial regions contribute most to the OOD detection signal.
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

def apply_fog(a, alpha):
    return np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8)
def apply_night(a): return np.clip(a*0.15, 0, 255).astype(np.uint8)

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 174: Occlusion Sensitivity Mapping")
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

    # Use highway as base scene
    base_img = create_highway(0)

    # Get base embedding (no occlusion)
    print("\n--- Base embedding ---", flush=True)
    base_h = extract_hidden(model, processor, Image.fromarray(base_img), prompt, layers)

    # Also get OOD reference embeddings
    fog_img = apply_fog(base_img, 0.6)
    night_img = apply_night(base_img)
    fog_h = extract_hidden(model, processor, Image.fromarray(fog_img), prompt, layers)
    night_h = extract_hidden(model, processor, Image.fromarray(night_img), prompt, layers)

    # Grid-based occlusion
    grid_sizes = [2, 4, 8]  # 2x2, 4x4, 8x8 grids

    results = {}
    for gs in grid_sizes:
        print(f"\n--- Grid {gs}x{gs} ---", flush=True)
        cell_h = SIZE[0] // gs
        cell_w = SIZE[1] // gs

        grid_results = {}
        for r in range(gs):
            for c in range(gs):
                # Occlude this cell with gray
                occluded = base_img.copy()
                occluded[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = 128

                h = extract_hidden(model, processor, Image.fromarray(occluded), prompt, layers)

                cell_result = {}
                for l in layers:
                    dist_to_base = cosine_distance(h[l], base_h[l])
                    dist_to_fog = cosine_distance(h[l], fog_h[l])
                    dist_to_night = cosine_distance(h[l], night_h[l])
                    cell_result[f"L{l}"] = {
                        "dist_to_base": dist_to_base,
                        "dist_to_fog": dist_to_fog,
                        "dist_to_night": dist_to_night,
                    }
                grid_results[f"{r}_{c}"] = cell_result
                print(f"  Cell ({r},{c}): L3_base={cell_result['L3']['dist_to_base']:.6f} "
                      f"L32_base={cell_result['L32']['dist_to_base']:.6f}", flush=True)

        results[f"grid_{gs}"] = grid_results

    # Also test: which regions matter most for OOD detection specifically
    # Occlude regions of the OOD image and see if occlusion restores ID-like embedding
    print("\n--- OOD region sensitivity (fog) ---", flush=True)
    gs = 4
    cell_h = SIZE[0] // gs
    cell_w = SIZE[1] // gs

    fog_sensitivity = {}
    base_fog_dist = {l: cosine_distance(fog_h[l], base_h[l]) for l in layers}
    print(f"  Full fog distance: L3={base_fog_dist[3]:.6f} L32={base_fog_dist[32]:.6f}", flush=True)

    for r in range(gs):
        for c in range(gs):
            # Take fog image but restore this cell to clean
            restored = fog_img.copy()
            restored[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = base_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]

            h = extract_hidden(model, processor, Image.fromarray(restored), prompt, layers)
            cell_result = {}
            for l in layers:
                dist_to_base = cosine_distance(h[l], base_h[l])
                reduction = base_fog_dist[l] - dist_to_base
                cell_result[f"L{l}"] = {
                    "dist_to_base": dist_to_base,
                    "reduction": reduction,
                    "pct_reduction": reduction / base_fog_dist[l] * 100 if base_fog_dist[l] > 0 else 0,
                }
            fog_sensitivity[f"{r}_{c}"] = cell_result
            print(f"  Restore ({r},{c}): L3_reduction={cell_result['L3']['pct_reduction']:.1f}% "
                  f"L32_reduction={cell_result['L32']['pct_reduction']:.1f}%", flush=True)

    results["fog_sensitivity_4x4"] = fog_sensitivity
    results["base_fog_dist"] = {f"L{l}": base_fog_dist[l] for l in layers}

    # Night sensitivity
    print("\n--- OOD region sensitivity (night) ---", flush=True)
    base_night_dist = {l: cosine_distance(night_h[l], base_h[l]) for l in layers}
    print(f"  Full night distance: L3={base_night_dist[3]:.6f} L32={base_night_dist[32]:.6f}", flush=True)

    night_sensitivity = {}
    for r in range(gs):
        for c in range(gs):
            restored = night_img.copy()
            restored[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = base_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]

            h = extract_hidden(model, processor, Image.fromarray(restored), prompt, layers)
            cell_result = {}
            for l in layers:
                dist_to_base = cosine_distance(h[l], base_h[l])
                reduction = base_night_dist[l] - dist_to_base
                cell_result[f"L{l}"] = {
                    "dist_to_base": dist_to_base,
                    "reduction": reduction,
                    "pct_reduction": reduction / base_night_dist[l] * 100 if base_night_dist[l] > 0 else 0,
                }
            night_sensitivity[f"{r}_{c}"] = cell_result
            print(f"  Restore ({r},{c}): L3_reduction={cell_result['L3']['pct_reduction']:.1f}% "
                  f"L32_reduction={cell_result['L32']['pct_reduction']:.1f}%", flush=True)

    results["night_sensitivity_4x4"] = night_sensitivity
    results["base_night_dist"] = {f"L{l}": base_night_dist[l] for l in layers}

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "occlusion_sensitivity",
        "experiment_number": 174,
        "timestamp": ts,
        "grid_sizes": grid_sizes,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"occlusion_sensitivity_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
