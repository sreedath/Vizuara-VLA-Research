#!/usr/bin/env python3
"""Experiment 301: Temporal Corruption Sequence Analysis
Simulates realistic driving scenarios with dynamic corruption transitions:
1. Clean -> fog -> clean transition (entering/exiting fog bank)
2. Gradual night onset (sunset simulation)
3. Intermittent noise (sensor glitches)
4. Multi-corruption journey
5. Detection response time
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

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
        "experiment": "temporal_sequence",
        "experiment_number": 301,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    clean_emb = extract_hidden(model, processor, base_img, prompt)

    # Scenario 1: Fog bank
    print("=== Scenario 1: Fog Bank ===")
    fog_sequence = []
    severities = [0]*5 + [i/10 for i in range(1, 11)] + [1 - i/10 for i in range(1, 11)] + [0]*5
    for t, sev in enumerate(severities):
        img = apply_corruption(base_img, 'fog', sev) if sev > 0 else base_img
        emb = extract_hidden(model, processor, img, prompt)
        d = float(cosine(clean_emb, emb))
        fog_sequence.append({"time": t, "severity": sev, "distance": d, "detected": d > 0})
    tp = sum(1 for f in fog_sequence if f["severity"] > 0 and f["detected"])
    fn = sum(1 for f in fog_sequence if f["severity"] > 0 and not f["detected"])
    fp = sum(1 for f in fog_sequence if f["severity"] == 0 and f["detected"])
    tn = sum(1 for f in fog_sequence if f["severity"] == 0 and not f["detected"])
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    results["fog_bank"] = {"sequence": fog_sequence, "tp": tp, "fn": fn, "fp": fp, "tn": tn}

    # Scenario 2: Sunset
    print("=== Scenario 2: Sunset ===")
    sunset_sequence = []
    for t in range(31):
        sev = t / 30.0
        img = apply_corruption(base_img, 'night', sev) if sev > 0 else base_img
        emb = extract_hidden(model, processor, img, prompt)
        d = float(cosine(clean_emb, emb))
        sunset_sequence.append({"time": t, "severity": sev, "distance": d, "detected": d > 0})
    first_detect = next((t for t, s in enumerate(sunset_sequence) if s["detected"]), -1)
    first_sev = sunset_sequence[first_detect]["severity"] if first_detect >= 0 else -1
    print(f"  First detection at t={first_detect}, severity={first_sev:.4f}")
    results["sunset"] = {"sequence": sunset_sequence, "first_detect_time": first_detect, "first_detect_severity": first_sev}

    # Scenario 3: Sensor glitches
    print("=== Scenario 3: Sensor Glitches ===")
    glitch_sequence = []
    pattern = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]
    for t, is_glitch in enumerate(pattern):
        img = apply_corruption(base_img, 'noise', 0.5) if is_glitch else base_img
        emb = extract_hidden(model, processor, img, prompt)
        d = float(cosine(clean_emb, emb))
        glitch_sequence.append({"time": t, "is_glitch": bool(is_glitch), "distance": d, "detected": d > 0})
    tp = sum(1 for g in glitch_sequence if g["is_glitch"] and g["detected"])
    fn = sum(1 for g in glitch_sequence if g["is_glitch"] and not g["detected"])
    fp = sum(1 for g in glitch_sequence if not g["is_glitch"] and g["detected"])
    tn = sum(1 for g in glitch_sequence if not g["is_glitch"] and not g["detected"])
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    results["glitches"] = {"sequence": glitch_sequence, "tp": tp, "fn": fn, "fp": fp, "tn": tn}

    # Scenario 4: Multi-corruption journey (48 frames)
    print("=== Scenario 4: Multi-Corruption Journey ===")
    journey = []
    segments = [
        ("clean", 0, 5), ("fog", 0.5, 5), ("fog", 1.0, 5), ("clean", 0, 3),
        ("night", 0.3, 5), ("night", 0.8, 5), ("blur", 0.5, 5),
        ("clean", 0, 5), ("noise", 1.0, 5), ("clean", 0, 5)
    ]
    t = 0
    for ctype, sev, n_frames in segments:
        for _ in range(n_frames):
            img = apply_corruption(base_img, ctype, sev) if sev > 0 else base_img
            emb = extract_hidden(model, processor, img, prompt)
            d = float(cosine(clean_emb, emb))
            journey.append({"time": t, "corruption": ctype, "severity": sev, "distance": d, "detected": d > 0})
            t += 1
    tp = sum(1 for j in journey if j["severity"] > 0 and j["detected"])
    fn = sum(1 for j in journey if j["severity"] > 0 and not j["detected"])
    fp = sum(1 for j in journey if j["severity"] == 0 and j["detected"])
    tn = sum(1 for j in journey if j["severity"] == 0 and not j["detected"])
    sens = tp / (tp + fn + 1e-30)
    spec = tn / (tn + fp + 1e-30)
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}, Sens={sens:.3f}, Spec={spec:.3f}")
    results["journey"] = {"sequence": journey, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                          "sensitivity": sens, "specificity": spec}

    # Scenario 5: Response time
    print("=== Scenario 5: Response Time ===")
    response_times = {}
    for c in ['fog', 'night', 'blur', 'noise']:
        for sev in [0.05, 0.1, 0.3, 0.5, 1.0]:
            img = apply_corruption(base_img, c, sev)
            emb = extract_hidden(model, processor, img, prompt)
            d = float(cosine(clean_emb, emb))
            key = f"{c}_sev{sev}"
            response_times[key] = {"corruption": c, "severity": sev, "distance": d,
                                    "detected_first_frame": d > 0}
            print(f"  {key}: d={d:.6f}, instant={d > 0}")
    results["response_times"] = response_times

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/temporal_seq_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
