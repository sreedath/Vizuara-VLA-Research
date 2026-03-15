#!/usr/bin/env python3
"""Experiment 291: Robustness to Image Preprocessing Variations
Tests whether common preprocessing transforms affect detection:
1. Resize to different resolutions then back to 224x224
2. JPEG compression at various quality levels
3. Small rotations (1-10 degrees)
4. Horizontal/vertical flips
5. Color space conversions (RGB->BGR->RGB, grayscale->RGB)
6. Small translations (1-10 pixels)
7. Brightness/contrast jitter
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine
import io

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
        "experiment": "preprocess_robustness",
        "experiment_number": 291,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Get clean centroid
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    # Get corruption distances for reference
    ref_distances = {}
    for c in ['fog', 'night', 'blur', 'noise']:
        cimg = apply_corruption(base_img, c, 1.0)
        emb = extract_hidden(model, processor, cimg, prompt)
        ref_distances[c] = float(cosine(clean_emb, emb))
    results["reference_corruption_distances"] = ref_distances
    print(f"Reference: {ref_distances}")

    # Part 1: Resize transforms
    print("\n=== Part 1: Resize Robustness ===")
    resize_results = []
    for res in [64, 112, 160, 224, 320, 448, 640, 1024]:
        resized = base_img.resize((res, res), Image.BILINEAR)
        resized_back = resized.resize((224, 224), Image.BILINEAR)
        emb = extract_hidden(model, processor, resized_back, prompt)
        d = float(cosine(clean_emb, emb))
        resize_results.append({"resolution": res, "distance": d})
        print(f"  {res}x{res}: d={d:.6f}")
    results["resize"] = resize_results

    # Part 2: JPEG compression
    print("\n=== Part 2: JPEG Compression ===")
    jpeg_results = []
    for q in [100, 95, 90, 80, 70, 50, 30, 10, 5, 1]:
        buf = io.BytesIO()
        base_img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        jpeg_img = Image.open(buf).convert('RGB')
        emb = extract_hidden(model, processor, jpeg_img, prompt)
        d = float(cosine(clean_emb, emb))
        jpeg_results.append({"quality": q, "distance": d})
        print(f"  Q={q}: d={d:.6f}")
    results["jpeg"] = jpeg_results

    # Part 3: Rotation
    print("\n=== Part 3: Rotation ===")
    rotation_results = []
    for angle in [0.1, 0.5, 1, 2, 5, 10, 15, 30, 45, 90, 180]:
        rotated = base_img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        emb = extract_hidden(model, processor, rotated, prompt)
        d = float(cosine(clean_emb, emb))
        rotation_results.append({"angle": angle, "distance": d})
        print(f"  {angle}°: d={d:.6f}")
    results["rotation"] = rotation_results

    # Part 4: Flips
    print("\n=== Part 4: Flips ===")
    flip_results = {}
    for flip_type, flip_fn in [("horizontal", ImageOps.mirror), ("vertical", ImageOps.flip),
                                 ("both", lambda img: ImageOps.flip(ImageOps.mirror(img)))]:
        flipped = flip_fn(base_img)
        emb = extract_hidden(model, processor, flipped, prompt)
        d = float(cosine(clean_emb, emb))
        flip_results[flip_type] = d
        print(f"  {flip_type}: d={d:.6f}")
    results["flips"] = flip_results

    # Part 5: Color space transforms
    print("\n=== Part 5: Color Space Transforms ===")
    color_results = {}

    # BGR->RGB (swap channels)
    arr = np.array(base_img)
    bgr = Image.fromarray(arr[:, :, ::-1])
    emb = extract_hidden(model, processor, bgr, prompt)
    color_results["bgr"] = float(cosine(clean_emb, emb))

    # Grayscale -> RGB
    gray = ImageOps.grayscale(base_img)
    gray_rgb = gray.convert('RGB')
    emb = extract_hidden(model, processor, gray_rgb, prompt)
    color_results["grayscale"] = float(cosine(clean_emb, emb))

    # Inverted
    inverted = ImageOps.invert(base_img)
    emb = extract_hidden(model, processor, inverted, prompt)
    color_results["inverted"] = float(cosine(clean_emb, emb))

    print(f"  BGR: d={color_results['bgr']:.6f}")
    print(f"  Grayscale: d={color_results['grayscale']:.6f}")
    print(f"  Inverted: d={color_results['inverted']:.6f}")
    results["color_space"] = color_results

    # Part 6: Translations
    print("\n=== Part 6: Translations ===")
    translation_results = []
    for shift in [1, 2, 5, 10, 20, 50]:
        shifted = base_img.transform(base_img.size, Image.AFFINE, (1, 0, shift, 0, 1, 0),
                                      fillcolor=(128, 128, 128))
        emb = extract_hidden(model, processor, shifted, prompt)
        d = float(cosine(clean_emb, emb))
        translation_results.append({"shift_pixels": shift, "distance": d})
        print(f"  {shift}px: d={d:.6f}")
    results["translation"] = translation_results

    # Part 7: Brightness/Contrast jitter
    print("\n=== Part 7: Brightness/Contrast Jitter ===")
    jitter_results = []
    for factor in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]:
        bright = ImageEnhance.Brightness(base_img).enhance(factor)
        emb = extract_hidden(model, processor, bright, prompt)
        d = float(cosine(clean_emb, emb))
        jitter_results.append({"brightness_factor": factor, "distance": d})
        print(f"  brightness={factor}: d={d:.6f}")
    results["brightness"] = jitter_results

    # Part 8: Combined: preprocess + corruption detection
    print("\n=== Part 8: Preprocess + Corruption Detection ===")
    combined = {}
    preprocs = [
        ("jpeg_50", lambda img: Image.open(io.BytesIO(b''.join([img.save(buf := io.BytesIO(), format='JPEG', quality=50), None][0:0] or [buf.seek(0), buf.read()]))).convert('RGB') if False else (lambda: (img.save(buf := io.BytesIO(), format='JPEG', quality=50), buf.seek(0), Image.open(buf).convert('RGB')))()[-1]),
        ("rotate_5", lambda img: img.rotate(5, resample=Image.BILINEAR, fillcolor=(128, 128, 128))),
        ("resize_112", lambda img: img.resize((112, 112)).resize((224, 224))),
    ]

    # Simpler approach for JPEG
    def jpeg_compress(img, q=50):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    preprocs_simple = [
        ("none", lambda img: img),
        ("jpeg_50", lambda img: jpeg_compress(img, 50)),
        ("rotate_5", lambda img: img.rotate(5, resample=Image.BILINEAR, fillcolor=(128, 128, 128))),
        ("resize_112", lambda img: img.resize((112, 112)).resize((224, 224))),
    ]

    for prep_name, prep_fn in preprocs_simple:
        combined[prep_name] = {}
        for c in ['fog', 'night', 'blur', 'noise']:
            corrupted = apply_corruption(base_img, c, 1.0)
            preprocessed = prep_fn(corrupted)

            # Centroid also preprocessed (calibration with same preprocessing)
            centroid_img = prep_fn(base_img)
            centroid_emb = extract_hidden(model, processor, centroid_img, prompt)

            emb = extract_hidden(model, processor, preprocessed, prompt)
            d = float(cosine(centroid_emb, emb))
            combined[prep_name][c] = d
            print(f"  {prep_name} + {c}: d={d:.6f}")

    results["combined_detection"] = combined

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/preprocess_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
