#!/usr/bin/env python3
"""Experiment 289: Gradient-Based Sensitivity Analysis
Computes input-pixel gradients of the cosine distance to understand
which image regions drive the OOD signal:
1. Gradient of cosine distance w.r.t. input pixels
2. Saliency maps for different corruption types
3. Gradient magnitude distribution across spatial regions
4. Comparison of gradient-based vs occlusion-based attribution
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance
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

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "gradient_sensitivity",
        "experiment_number": 289,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Get clean embedding as reference
    print("Getting clean reference embedding...")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    clean_emb = fwd.hidden_states[3][0, -1, :].float()  # keep on GPU
    clean_norm = clean_emb / clean_emb.norm()
    clean_norm = clean_norm.detach()

    # Part 1: Gradient of cosine distance w.r.t. pixel_values for each corruption
    print("\n=== Part 1: Pixel Gradient Saliency Maps ===")
    corruptions = ['fog', 'night', 'blur', 'noise']
    gradient_stats = {}

    for c in corruptions:
        print(f"  Computing gradient for {c}...")
        corrupted = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)

        # Enable gradient computation on pixel values
        pixel_values = inputs['pixel_values'].clone().detach().requires_grad_(True)
        inputs_grad = {k: v for k, v in inputs.items()}
        inputs_grad['pixel_values'] = pixel_values

        # Forward pass with gradients
        model.zero_grad()
        fwd = model(**inputs_grad, output_hidden_states=True)
        emb = fwd.hidden_states[3][0, -1, :].float()

        # Compute cosine distance
        emb_norm = emb / emb.norm()
        cos_sim = torch.dot(clean_norm, emb_norm)
        cos_dist = 1.0 - cos_sim

        # Backward pass
        cos_dist.backward()

        if pixel_values.grad is not None:
            grad = pixel_values.grad.float().cpu().numpy()[0]  # [C, H, W]
            grad_magnitude = np.sqrt(np.sum(grad**2, axis=0))  # [H, W]

            # Spatial statistics
            h, w = grad_magnitude.shape
            quadrants = {
                "top_left": grad_magnitude[:h//2, :w//2].mean(),
                "top_right": grad_magnitude[:h//2, w//2:].mean(),
                "bottom_left": grad_magnitude[h//2:, :w//2].mean(),
                "bottom_right": grad_magnitude[h//2:, w//2:].mean(),
                "center": grad_magnitude[h//4:3*h//4, w//4:3*w//4].mean(),
                "border": np.concatenate([
                    grad_magnitude[:h//8, :].flatten(),
                    grad_magnitude[-h//8:, :].flatten(),
                    grad_magnitude[:, :w//8].flatten(),
                    grad_magnitude[:, -w//8:].flatten()
                ]).mean()
            }

            # Percentile distribution
            percentiles = {
                f"p{p}": float(np.percentile(grad_magnitude, p))
                for p in [10, 25, 50, 75, 90, 95, 99]
            }

            # Channel-wise analysis
            channel_means = {
                f"ch{i}": float(np.abs(grad[i]).mean())
                for i in range(min(3, grad.shape[0]))
            }

            gradient_stats[c] = {
                "cosine_distance": float(cos_dist.item()),
                "grad_mean": float(grad_magnitude.mean()),
                "grad_std": float(grad_magnitude.std()),
                "grad_max": float(grad_magnitude.max()),
                "grad_min": float(grad_magnitude.min()),
                "quadrants": {k: float(v) for k, v in quadrants.items()},
                "percentiles": percentiles,
                "channel_means": channel_means,
                "sparsity_90pct": float(np.mean(grad_magnitude > np.percentile(grad_magnitude, 90))),
                "center_vs_border_ratio": float(quadrants["center"] / quadrants["border"]) if quadrants["border"] > 0 else 0,
                "max_quadrant": max(quadrants, key=quadrants.get),
                "min_quadrant": min(quadrants, key=quadrants.get)
            }
            print(f"    d={cos_dist.item():.6f}, grad_mean={grad_magnitude.mean():.6e}, "
                  f"center/border={quadrants['center']/quadrants['border']:.2f}")
        else:
            gradient_stats[c] = {"error": "No gradient computed"}
            print(f"    No gradient available for {c}")

    results["gradient_stats"] = gradient_stats

    # Part 2: Gradient similarity between corruption types
    print("\n=== Part 2: Gradient Similarity Between Corruptions ===")
    grad_vecs = {}
    for c in corruptions:
        if "error" not in gradient_stats[c]:
            corrupted = apply_corruption(base_img, c, 1.0)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            pixel_values = inputs['pixel_values'].clone().detach().requires_grad_(True)
            inputs_grad = {k: v for k, v in inputs.items()}
            inputs_grad['pixel_values'] = pixel_values
            model.zero_grad()
            fwd = model(**inputs_grad, output_hidden_states=True)
            emb = fwd.hidden_states[3][0, -1, :].float()
            emb_norm = emb / emb.norm()
            cos_dist = 1.0 - torch.dot(clean_norm, emb_norm)
            cos_dist.backward()
            if pixel_values.grad is not None:
                grad_vecs[c] = pixel_values.grad.float().cpu().numpy().flatten()

    grad_similarities = {}
    for c1 in corruptions:
        for c2 in corruptions:
            if c1 < c2 and c1 in grad_vecs and c2 in grad_vecs:
                g1, g2 = grad_vecs[c1], grad_vecs[c2]
                sim = float(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-30))
                grad_similarities[f"{c1}_vs_{c2}"] = sim
                print(f"  {c1} vs {c2}: cos_sim = {sim:.4f}")
    results["gradient_similarities"] = grad_similarities

    # Part 3: Gradient at different severities
    print("\n=== Part 3: Gradient vs Severity ===")
    severity_gradients = {}
    for c in ['fog', 'night']:
        severity_gradients[c] = []
        for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            pixel_values = inputs['pixel_values'].clone().detach().requires_grad_(True)
            inputs_grad = {k: v for k, v in inputs.items()}
            inputs_grad['pixel_values'] = pixel_values
            model.zero_grad()
            fwd = model(**inputs_grad, output_hidden_states=True)
            emb = fwd.hidden_states[3][0, -1, :].float()
            emb_norm = emb / emb.norm()
            cos_dist = 1.0 - torch.dot(clean_norm, emb_norm)
            cos_dist.backward()
            if pixel_values.grad is not None:
                grad = pixel_values.grad.float().cpu().numpy()[0]
                grad_mag = np.sqrt(np.sum(grad**2, axis=0))
                severity_gradients[c].append({
                    "severity": sev,
                    "distance": float(cos_dist.item()),
                    "grad_mean": float(grad_mag.mean()),
                    "grad_max": float(grad_mag.max()),
                    "grad_std": float(grad_mag.std())
                })
                print(f"  {c} sev={sev}: d={cos_dist.item():.6f}, grad_mean={grad_mag.mean():.6e}")
    results["severity_gradients"] = severity_gradients

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/gradient_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
