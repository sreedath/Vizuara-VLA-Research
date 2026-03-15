#!/usr/bin/env python3
"""Experiment 203: Temporal consistency — does detection remain stable
across sequential frames from the same driving scene? Simulates frame-to-frame
variation and measures detection consistency.
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

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def create_temporal_sequence(base_creator, n_frames, scene_noise=3):
    """Create a sequence of frames with small temporal variation."""
    base_rng = np.random.RandomState(42)
    base = base_creator(0)
    frames = []
    for t in range(n_frames):
        # Small random jitter per frame (simulating camera noise/motion)
        noise = np.random.RandomState(100+t).randint(-scene_noise, scene_noise+1, base.shape).astype(np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames

def main():
    print("=" * 60)
    print("Experiment 203: Temporal Consistency Analysis")
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

    # Calibrate
    print("\n--- Calibration ---", flush=True)
    cal_embs = {l: [] for l in layers}
    for i in range(10):
        arr = creators[i%3](i)
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            cal_embs[l].append(h[l])
    centroids = {l: np.array(cal_embs[l]).mean(axis=0) for l in layers}

    # Temporal sequences
    n_frames = 20
    print(f"\n--- Temporal sequences ({n_frames} frames each) ---", flush=True)
    
    results = {}
    for scene_name, creator in [("highway", create_highway), ("urban", create_urban), ("rural", create_rural)]:
        print(f"\n  Scene: {scene_name}", flush=True)
        
        # Clean sequence
        clean_frames = create_temporal_sequence(creator, n_frames, scene_noise=3)
        clean_dists = {l: [] for l in layers}
        for f in clean_frames:
            h = extract_all(Image.fromarray(f))
            for l in layers:
                clean_dists[l].append(cosine_distance(h[l], centroids[l]))
        
        # Fog sequence (gradual onset)
        fog_dists = {l: [] for l in layers}
        fog_alphas = np.linspace(0, 0.6, n_frames)
        for t, frame in enumerate(clean_frames):
            fogged = apply_fog(frame, fog_alphas[t])
            h = extract_all(Image.fromarray(fogged))
            for l in layers:
                fog_dists[l].append(cosine_distance(h[l], centroids[l]))
        
        # Night sequence (sudden onset at frame 10)
        night_dists = {l: [] for l in layers}
        for t, frame in enumerate(clean_frames):
            if t >= 10:
                processed = apply_night(frame)
            else:
                processed = frame
            h = extract_all(Image.fromarray(processed))
            for l in layers:
                night_dists[l].append(cosine_distance(h[l], centroids[l]))
        
        scene_results = {}
        for l in layers:
            scene_results[f"L{l}"] = {
                "clean": {
                    "dists": clean_dists[l],
                    "mean": float(np.mean(clean_dists[l])),
                    "std": float(np.std(clean_dists[l])),
                    "cv": float(np.std(clean_dists[l]) / (np.mean(clean_dists[l]) + 1e-10)),
                },
                "fog_gradual": {
                    "dists": fog_dists[l],
                    "detection_frame": next((t for t, d in enumerate(fog_dists[l]) 
                                           if d > np.max(clean_dists[l])), -1),
                },
                "night_sudden": {
                    "dists": night_dists[l],
                    "detection_frame": next((t for t, d in enumerate(night_dists[l])
                                           if d > np.max(clean_dists[l])), -1),
                },
            }
            print(f"    L{l} clean: {np.mean(clean_dists[l]):.6f}±{np.std(clean_dists[l]):.6f} "
                  f"CV={np.std(clean_dists[l])/(np.mean(clean_dists[l])+1e-10):.4f}", flush=True)
            print(f"    L{l} fog onset frame: {scene_results[f'L{l}']['fog_gradual']['detection_frame']}", flush=True)
            print(f"    L{l} night onset frame: {scene_results[f'L{l}']['night_sudden']['detection_frame']}", flush=True)
        
        results[scene_name] = scene_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "temporal_consistency",
        "experiment_number": 203,
        "timestamp": ts,
        "n_frames": n_frames,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"temporal_consistency_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
