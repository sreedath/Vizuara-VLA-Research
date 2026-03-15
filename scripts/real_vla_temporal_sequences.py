#!/usr/bin/env python3
"""Experiment 394: Embedding Dynamics Under Temporal Corruption Sequences

Tests how embeddings evolve when corruption appears/disappears over time,
simulating real driving scenarios with transient weather changes.

Tests:
1. Clean → corrupt → clean transition dynamics
2. Gradual onset (severity ramp from 0 to 1)
3. Gradual offset (severity ramp from 1 to 0)
4. Oscillating corruption (severity cycles)
5. Multi-corruption sequences (fog → night → blur)
6. Recovery time after corruption removal
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

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

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    corruptions = ['fog', 'night', 'noise', 'blur']

    # Get clean centroid
    print("Computing clean centroid...")
    clean_embs = []
    for i in range(5):
        arr = np.array(img).astype(np.float32)
        arr += np.random.RandomState(100 + i).randn(*arr.shape) * 0.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        emb = extract_hidden(model, processor, Image.fromarray(arr), prompt)
        clean_embs.append(emb)
    centroid = np.mean(clean_embs, axis=0)

    results = {}

    # 1. Clean → Corrupt → Clean step transition
    print("\n=== Step Transition ===")
    step_results = {}
    for c in corruptions:
        sequence = []
        # 5 clean
        for i in range(5):
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, centroid)
            sequence.append({"frame": len(sequence), "severity": 0.0, "dist": float(d), "phase": "clean_before"})
        # 5 corrupted
        for i in range(5):
            corrupted = apply_corruption(img, c)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = cosine_dist(emb, centroid)
            sequence.append({"frame": len(sequence), "severity": 1.0, "dist": float(d), "phase": "corrupted"})
        # 5 clean after
        for i in range(5):
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, centroid)
            sequence.append({"frame": len(sequence), "severity": 0.0, "dist": float(d), "phase": "clean_after"})

        step_results[c] = sequence
        clean_before = max(s["dist"] for s in sequence if s["phase"] == "clean_before")
        corrupt_min = min(s["dist"] for s in sequence if s["phase"] == "corrupted")
        clean_after = max(s["dist"] for s in sequence if s["phase"] == "clean_after")
        print(f"  {c}: clean_max={clean_before:.6f}, corrupt_min={corrupt_min:.6f}, recovery={clean_after:.6f}")
    results["step_transition"] = step_results

    # 2. Gradual onset
    print("\n=== Gradual Onset ===")
    onset_results = {}
    for c in corruptions:
        sevs = np.linspace(0, 1, 20)
        sequence = []
        for sev in sevs:
            test_img = apply_corruption(img, c, severity=float(sev)) if sev > 0.001 else img
            emb = extract_hidden(model, processor, test_img, prompt)
            d = cosine_dist(emb, centroid)
            sequence.append({"severity": float(sev), "dist": float(d)})
        onset_results[c] = sequence
        print(f"  {c}: onset profile computed")
    results["gradual_onset"] = onset_results

    # 3. Gradual offset
    print("\n=== Gradual Offset ===")
    offset_results = {}
    for c in corruptions:
        sevs = np.linspace(1, 0, 20)
        sequence = []
        for sev in sevs:
            test_img = apply_corruption(img, c, severity=float(sev)) if sev > 0.001 else img
            emb = extract_hidden(model, processor, test_img, prompt)
            d = cosine_dist(emb, centroid)
            sequence.append({"severity": float(sev), "dist": float(d)})
        offset_results[c] = sequence
    results["gradual_offset"] = offset_results

    # 4. Oscillating corruption
    print("\n=== Oscillating ===")
    osc_results = {}
    for c in ['fog', 'noise']:
        n_frames = 30
        sequence = []
        for i in range(n_frames):
            sev = 0.5 * (1 + np.sin(2 * np.pi * i / 10))
            test_img = apply_corruption(img, c, severity=float(sev)) if sev > 0.001 else img
            emb = extract_hidden(model, processor, test_img, prompt)
            d = cosine_dist(emb, centroid)
            sequence.append({"frame": i, "severity": float(sev), "dist": float(d)})
        osc_results[c] = sequence
        sevs_arr = [s["severity"] for s in sequence]
        dists_arr = [s["dist"] for s in sequence]
        corr = float(np.corrcoef(sevs_arr, dists_arr)[0, 1])
        print(f"  {c}: severity-dist corr r={corr:.4f}")
    results["oscillating"] = osc_results

    # 5. Multi-corruption sequence
    print("\n=== Multi-Corruption Sequence ===")
    multi = []
    for c in ['fog', 'night', 'noise', 'blur']:
        for i in range(3):
            corrupted = apply_corruption(img, c)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = cosine_dist(emb, centroid)
            multi.append({"frame": len(multi), "corruption": c, "dist": float(d)})
    for i in range(3):
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(emb, centroid)
        multi.append({"frame": len(multi), "corruption": "clean", "dist": float(d)})
    results["multi_corruption_sequence"] = multi

    # 6. Onset/offset symmetry
    print("\n=== Symmetry ===")
    symmetry = {}
    for c in corruptions:
        onset_d = [s["dist"] for s in onset_results[c]]
        offset_d = [s["dist"] for s in offset_results[c]][::-1]
        diffs = [abs(a - b) for a, b in zip(onset_d, offset_d)]
        symmetry[c] = {
            "mean_asymmetry": float(np.mean(diffs)),
            "max_asymmetry": float(np.max(diffs)),
            "is_symmetric": float(np.max(diffs)) < 0.0001
        }
        print(f"  {c}: max_asym={np.max(diffs):.8f}, symmetric={np.max(diffs) < 0.0001}")
    results["symmetry"] = symmetry

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/temporal_sequences_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
