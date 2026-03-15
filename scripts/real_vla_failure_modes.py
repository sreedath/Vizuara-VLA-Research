#!/usr/bin/env python3
"""Experiment 314: Failure Mode Characterization
Systematically probes detector limitations and failure conditions:
1. Geometric transforms (rotation, flip, crop)
2. Camera settings (brightness, contrast, saturation)
3. JPEG compression artifacts
4. Pixel-level perturbations
5. Borderline ultra-low severity corruptions
6. Action-preserving corruptions (detected but harmless)
"""

import torch
import numpy as np
import json
import io
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance
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

def get_action_tokens(model, processor, image, prompt):
    ACTION_TOKEN_START = 31744
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=7, do_sample=False)
    input_len = inputs['input_ids'].shape[1]
    gen_tokens = generated[0, input_len:].cpu().numpy()
    return [int(t - ACTION_TOKEN_START) for t in gen_tokens]

def jpeg_compress(image, quality):
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

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
        "experiment": "failure_modes",
        "experiment_number": 314,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    clean_emb = extract_hidden(model, processor, base_img, prompt)
    clean_actions = get_action_tokens(model, processor, base_img, prompt)

    # Part 1: Geometric Transforms
    print("=== Part 1: Geometric Transforms ===")
    geometric = {}

    for angle in [1, 5, 10, 30, 45, 90, 180]:
        rotated = base_img.rotate(angle, fillcolor=(128, 128, 128))
        emb = extract_hidden(model, processor, rotated, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, rotated, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        geometric[f"rotate_{angle}"] = {"distance": d, "actions_changed": n_changed}
        print(f"  Rotate {angle}: d={d:.8f}, actions_changed={n_changed}")

    for flip_type in ['horizontal', 'vertical']:
        if flip_type == 'horizontal':
            flipped = base_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            flipped = base_img.transpose(Image.FLIP_TOP_BOTTOM)
        emb = extract_hidden(model, processor, flipped, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, flipped, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        geometric[f"flip_{flip_type}"] = {"distance": d, "actions_changed": n_changed}
        print(f"  Flip {flip_type}: d={d:.8f}, actions_changed={n_changed}")

    for crop_pct in [0.95, 0.90, 0.80, 0.70]:
        sz = int(224 * crop_pct)
        offset = (224 - sz) // 2
        cropped = base_img.crop((offset, offset, offset + sz, offset + sz))
        cropped = cropped.resize((224, 224), Image.BILINEAR)
        emb = extract_hidden(model, processor, cropped, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, cropped, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        geometric[f"crop_{crop_pct}"] = {"distance": d, "actions_changed": n_changed}
        print(f"  Crop {crop_pct*100:.0f}%: d={d:.8f}, actions_changed={n_changed}")

    results["geometric"] = geometric

    # Part 2: Camera Settings
    print("\n=== Part 2: Camera Settings ===")
    camera = {}

    for factor in [0.5, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.5, 2.0]:
        enhanced = ImageEnhance.Brightness(base_img).enhance(factor)
        emb = extract_hidden(model, processor, enhanced, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, enhanced, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        camera[f"brightness_{factor}"] = {"distance": d, "actions_changed": n_changed}
        print(f"  Brightness {factor}: d={d:.8f}, actions_changed={n_changed}")

    for factor in [0.5, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
        enhanced = ImageEnhance.Contrast(base_img).enhance(factor)
        emb = extract_hidden(model, processor, enhanced, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, enhanced, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        camera[f"contrast_{factor}"] = {"distance": d, "actions_changed": n_changed}
        print(f"  Contrast {factor}: d={d:.8f}, actions_changed={n_changed}")

    for factor in [0.0, 0.5, 0.8, 1.2, 1.5, 2.0]:
        enhanced = ImageEnhance.Color(base_img).enhance(factor)
        emb = extract_hidden(model, processor, enhanced, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, enhanced, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        camera[f"saturation_{factor}"] = {"distance": d, "actions_changed": n_changed}
        print(f"  Saturation {factor}: d={d:.8f}, actions_changed={n_changed}")

    results["camera"] = camera

    # Part 3: JPEG Compression
    print("\n=== Part 3: JPEG Compression ===")
    jpeg_results = {}

    for quality in [1, 5, 10, 20, 30, 50, 70, 90, 95]:
        compressed = jpeg_compress(base_img, quality)
        emb = extract_hidden(model, processor, compressed, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, compressed, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        jpeg_results[f"q{quality}"] = {"distance": d, "actions_changed": n_changed}
        print(f"  JPEG Q={quality}: d={d:.8f}, actions_changed={n_changed}")

    results["jpeg"] = jpeg_results

    # Part 4: Pixel-Level Perturbations
    print("\n=== Part 4: Pixel-Level Perturbations ===")
    pixel_shifts = {}

    for n_pixels in [1, 10, 100, 1000, 5000, 10000]:
        arr = np.array(base_img).copy()
        rng = np.random.RandomState(42)
        indices = rng.choice(224 * 224, size=n_pixels, replace=False)
        for idx in indices:
            r, c_idx = divmod(idx, 224)
            arr[r, c_idx] = rng.randint(0, 256, size=3)
        modified = Image.fromarray(arr)
        emb = extract_hidden(model, processor, modified, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, modified, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        pixel_shifts[f"random_{n_pixels}px"] = {
            "distance": d,
            "actions_changed": n_changed,
            "pct_pixels": n_pixels / (224 * 224) * 100,
        }
        print(f"  {n_pixels} random pixels ({n_pixels/(224*224)*100:.2f}%): d={d:.8f}, actions_changed={n_changed}")

    for offset_val in [1, 5, 10, 20, 50]:
        arr = np.array(base_img).astype(np.int16) + offset_val
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        modified = Image.fromarray(arr)
        emb = extract_hidden(model, processor, modified, prompt)
        d = float(cosine(clean_emb, emb))
        actions = get_action_tokens(model, processor, modified, prompt)
        n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
        pixel_shifts[f"uniform_+{offset_val}"] = {"distance": d, "actions_changed": n_changed}
        print(f"  Uniform +{offset_val}: d={d:.8f}, actions_changed={n_changed}")

    results["pixel_shifts"] = pixel_shifts

    # Part 5: Borderline Cases
    print("\n=== Part 5: Borderline Cases ===")
    borderline = {}

    for c in ['fog', 'noise', 'night', 'blur']:
        for sev in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]:
            corrupted = apply_corruption(base_img, c, sev)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(clean_emb, emb))
            borderline[f"{c}_{sev}"] = {"distance": d, "detected": d > 0}
            print(f"  {c} sev={sev}: d={d:.10f}, detected={d > 0}")

    results["borderline"] = borderline

    # Part 6: Action-Preserving Corruptions
    print("\n=== Part 6: Action-Preserving Corruptions ===")
    action_preserving = {}

    for c in ['fog', 'night', 'blur', 'noise']:
        for sev_val in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
            corrupted = apply_corruption(base_img, c, sev_val)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(clean_emb, emb))
            actions = get_action_tokens(model, processor, corrupted, prompt)
            n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
            action_preserving[f"{c}_{sev_val}"] = {
                "distance": d,
                "detected": d > 0,
                "actions_changed": n_changed,
                "false_alarm": d > 0 and n_changed == 0,
            }
            if n_changed == 0 and d > 0:
                print(f"  * {c} sev={sev_val}: DETECTED but actions UNCHANGED (d={d:.6f})")
            elif n_changed > 0:
                print(f"    {c} sev={sev_val}: detected={d>0}, actions_changed={n_changed}")

    n_false_alarms = sum(1 for v in action_preserving.values() if v.get("false_alarm", False))
    n_total_tests = len(action_preserving)
    results["action_preserving"] = {
        "tests": action_preserving,
        "n_false_alarms": n_false_alarms,
        "n_total": n_total_tests,
        "false_alarm_rate": n_false_alarms / n_total_tests if n_total_tests > 0 else 0,
    }
    print(f"\n  Action-preserving false alarms: {n_false_alarms}/{n_total_tests}")

    # Summary
    print("\n=== Summary ===")
    all_tests = {}
    for category in ['geometric', 'camera', 'jpeg', 'pixel_shifts', 'borderline']:
        data = results[category]
        for k, v in data.items():
            if isinstance(v, dict) and 'distance' in v:
                all_tests[f"{category}/{k}"] = v['distance']

    detected = {k: d for k, d in all_tests.items() if d > 0}
    undetected = {k: d for k, d in all_tests.items() if d == 0}

    print(f"  Total tests: {len(all_tests)}")
    print(f"  Detected (d>0): {len(detected)}")
    print(f"  Undetected (d=0): {len(undetected)}")
    if undetected:
        print(f"  Undetected: {list(undetected.keys())}")

    results["summary"] = {
        "total_tests": len(all_tests),
        "detected": len(detected),
        "undetected": len(undetected),
        "undetected_conditions": list(undetected.keys()),
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
    out_path = f"experiments/failure_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
