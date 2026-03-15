#!/usr/bin/env python3
"""Experiment 422: Temporal Consistency Analysis

Studies how corruption affects the temporal consistency of predicted actions
and embeddings across sequential frames. In a real robot deployment, the
model receives a stream of frames — does corruption cause sudden jumps
in the embedding/action space, or gradual drift?

Tests:
1. Frame-to-frame embedding stability (clean sequence vs corrupted)
2. Action prediction stability across sequential frames
3. Sudden vs gradual corruption onset detection
4. Temporal smoothing effects on OOD detection
5. Embedding velocity (rate of change) as OOD signal
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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def get_action_tokens(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=7, do_sample=False)
    new_tokens = gen[0, inputs['input_ids'].shape[1]:]
    return new_tokens.cpu().numpy()

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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']

    # Generate a "video sequence" by creating slightly different frames
    base_seed = 42
    rng = np.random.RandomState(base_seed)
    base_arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype(np.float32)

    # Create 10 sequential frames with small perturbations
    n_frames = 10
    frames = []
    for i in range(n_frames):
        noise = rng.randn(224, 224, 3) * 3.0
        frame_arr = np.clip(base_arr + noise * i, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame_arr))

    print(f"Generated {n_frames} sequential frames")

    print("Extracting clean frame embeddings...")
    clean_embs = [extract_hidden(model, processor, f, prompt) for f in frames]
    centroid = np.mean(clean_embs, axis=0)

    results = {"n_frames": n_frames}

    # === Test 1: Frame-to-frame embedding stability ===
    print("\n=== Frame-to-Frame Embedding Stability ===")
    stability = {}

    clean_consecutive = []
    for i in range(n_frames - 1):
        d = cosine_dist(clean_embs[i], clean_embs[i+1])
        clean_consecutive.append(float(d))
    stability["clean"] = {
        "consecutive_dists": clean_consecutive,
        "mean": float(np.mean(clean_consecutive)),
        "max": float(np.max(clean_consecutive)),
    }
    print(f"  Clean: mean consecutive dist={np.mean(clean_consecutive):.6f}")

    for c in corruptions:
        corrupt_embs = [extract_hidden(model, processor, apply_corruption(f, c), prompt) for f in frames]
        consec = []
        for i in range(n_frames - 1):
            d = cosine_dist(corrupt_embs[i], corrupt_embs[i+1])
            consec.append(float(d))
        centroid_dists = [float(cosine_dist(e, centroid)) for e in corrupt_embs]
        stability[c] = {
            "consecutive_dists": consec,
            "mean": float(np.mean(consec)),
            "max": float(np.max(consec)),
            "centroid_dists": centroid_dists,
            "mean_centroid_dist": float(np.mean(centroid_dists)),
        }
        print(f"  {c}: mean consecutive={np.mean(consec):.6f}, mean centroid={np.mean(centroid_dists):.6f}")
    results["frame_stability"] = stability

    # === Test 2: Action prediction stability ===
    print("\n=== Action Prediction Stability ===")
    action_stability = {}

    clean_actions = [get_action_tokens(model, processor, f, prompt) for f in frames]
    clean_action_changes = sum(1 for i in range(n_frames-1) if not np.array_equal(clean_actions[i], clean_actions[i+1]))
    action_stability["clean"] = {
        "n_action_changes": int(clean_action_changes),
        "unique_actions": int(len(set(tuple(a.tolist()) for a in clean_actions))),
    }
    print(f"  Clean: {clean_action_changes} action changes across {n_frames} frames, {action_stability['clean']['unique_actions']} unique")

    for c in corruptions:
        corrupt_actions = [get_action_tokens(model, processor, apply_corruption(f, c), prompt) for f in frames]
        changes = sum(1 for i in range(n_frames-1) if not np.array_equal(corrupt_actions[i], corrupt_actions[i+1]))
        unique = len(set(tuple(a.tolist()) for a in corrupt_actions))
        matches_clean = sum(1 for i in range(n_frames) if np.array_equal(corrupt_actions[i], clean_actions[i]))
        action_stability[c] = {
            "n_action_changes": int(changes),
            "unique_actions": int(unique),
            "matches_clean": int(matches_clean),
        }
        print(f"  {c}: {changes} changes, {unique} unique, {matches_clean}/{n_frames} match clean")
    results["action_stability"] = action_stability

    # === Test 3: Sudden vs gradual corruption onset ===
    print("\n=== Corruption Onset Detection ===")
    onset_results = {}
    for c in corruptions:
        mixed_embs = []
        mixed_dists = []
        for i in range(n_frames):
            if i < 5:
                emb = clean_embs[i]
            else:
                emb = extract_hidden(model, processor, apply_corruption(frames[i], c), prompt)
            mixed_embs.append(emb)
            mixed_dists.append(float(cosine_dist(emb, centroid)))

        velocities = []
        for i in range(n_frames - 1):
            velocities.append(float(cosine_dist(mixed_embs[i], mixed_embs[i+1])))

        if velocities:
            max_vel_idx = int(np.argmax(velocities))
            onset_velocity = velocities[max_vel_idx]
        else:
            max_vel_idx = -1
            onset_velocity = 0.0

        onset_results[c] = {
            "centroid_dists": mixed_dists,
            "velocities": velocities,
            "onset_frame": max_vel_idx,
            "onset_velocity": float(onset_velocity),
            "pre_onset_mean_vel": float(np.mean(velocities[:4])) if len(velocities) >= 4 else 0.0,
            "post_onset_mean_vel": float(np.mean(velocities[5:])) if len(velocities) > 5 else 0.0,
        }
        print(f"  {c}: onset detected at frame {max_vel_idx} (velocity={onset_velocity:.6f})")
    results["corruption_onset"] = onset_results

    # === Test 4: Temporal smoothing ===
    print("\n=== Temporal Smoothing for Detection ===")
    smoothing = {}
    for c in corruptions:
        single_frame_ood = [float(cosine_dist(
            extract_hidden(model, processor, apply_corruption(f, c), prompt), centroid))
            for f in frames]

        if len(single_frame_ood) >= 3:
            smoothed_3 = [float(np.mean(single_frame_ood[max(0,i-1):i+2])) for i in range(len(single_frame_ood))]
        else:
            smoothed_3 = single_frame_ood

        if len(single_frame_ood) >= 5:
            smoothed_5 = [float(np.mean(single_frame_ood[max(0,i-2):i+3])) for i in range(len(single_frame_ood))]
        else:
            smoothed_5 = single_frame_ood

        clean_dists = [float(cosine_dist(e, centroid)) for e in clean_embs]

        auroc_single = float(compute_auroc(clean_dists, single_frame_ood))
        auroc_3 = float(compute_auroc(clean_dists, smoothed_3))
        auroc_5 = float(compute_auroc(clean_dists, smoothed_5))

        smoothing[c] = {
            "single_auroc": auroc_single,
            "smooth_3_auroc": auroc_3,
            "smooth_5_auroc": auroc_5,
            "single_mean": float(np.mean(single_frame_ood)),
            "single_std": float(np.std(single_frame_ood)),
        }
        print(f"  {c}: single={auroc_single:.4f}, 3-frame={auroc_3:.4f}, 5-frame={auroc_5:.4f}")
    results["temporal_smoothing"] = smoothing

    # === Test 5: Embedding velocity as OOD signal ===
    print("\n=== Embedding Velocity as OOD Signal ===")
    velocity_detection = {}

    clean_vels = [float(cosine_dist(clean_embs[i], clean_embs[i+1])) for i in range(n_frames-1)]

    for c in corruptions:
        mixed_embs = []
        for i in range(n_frames):
            if i < 5:
                mixed_embs.append(clean_embs[i])
            else:
                mixed_embs.append(extract_hidden(model, processor, apply_corruption(frames[i], c), prompt))
        mixed_vels = [float(cosine_dist(mixed_embs[i], mixed_embs[i+1])) for i in range(n_frames-1)]

        onset_vel = mixed_vels[4] if len(mixed_vels) > 4 else 0.0
        max_clean_vel = max(clean_vels) if clean_vels else 0.0

        velocity_detection[c] = {
            "clean_velocities": clean_vels,
            "mixed_velocities": mixed_vels,
            "onset_velocity": float(onset_vel),
            "max_clean_velocity": float(max_clean_vel),
            "velocity_ratio": float(onset_vel / max_clean_vel) if max_clean_vel > 1e-12 else float('inf'),
        }
        print(f"  {c}: onset_vel={onset_vel:.6f}, max_clean_vel={max_clean_vel:.6f}, ratio={velocity_detection[c]['velocity_ratio']:.2f}")
    results["velocity_detection"] = velocity_detection

    out_path = "/workspace/Vizuara-VLA-Research/experiments/temporal_consistency_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
