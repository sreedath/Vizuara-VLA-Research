#!/usr/bin/env python3
"""Experiment 293: Weight Perturbation Sensitivity
Tests how sensitive the detector is to model weight perturbations:
1. Add Gaussian noise to model weights at various scales
2. Measure how detection distances change
3. Test if detection remains valid after quantization-like perturbations
4. Measure embedding stability under weight noise
"""

import torch
import numpy as np
import json
import copy
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

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
        "experiment": "weight_perturbation",
        "experiment_number": 293,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Get baseline embeddings
    print("Getting baseline embeddings...")
    clean_emb = extract_hidden(model, processor, base_img, prompt)
    corruption_embs = {}
    for c in ['fog', 'night', 'blur', 'noise']:
        cimg = apply_corruption(base_img, c, 1.0)
        corruption_embs[c] = extract_hidden(model, processor, cimg, prompt)

    baseline_distances = {c: float(cosine(clean_emb, e)) for c, e in corruption_embs.items()}
    results["baseline_distances"] = baseline_distances
    print(f"Baseline: {baseline_distances}")

    # Part 1: Perturb a specific layer's weights and measure effect
    print("\n=== Part 1: Layer-Specific Weight Perturbation ===")
    layer_perturb_results = {}

    # Target layers: attention weights at layers 0, 3, 15, 31
    target_layers = [0, 3, 15, 31]
    noise_scales = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    for target_layer in target_layers:
        layer_name = f"model.layers.{target_layer}.self_attn.q_proj.weight"
        print(f"\n  Perturbing {layer_name}...")

        # Find the parameter
        param = None
        for name, p in model.named_parameters():
            if layer_name in name:
                param = p
                break

        if param is None:
            print(f"    Parameter not found!")
            continue

        original_data = param.data.clone()
        param_norm = float(param.data.float().norm())

        layer_results = []
        for scale in noise_scales:
            # Add noise
            noise = torch.randn_like(param.data).to(param.data.dtype) * scale
            param.data = original_data + noise

            # Re-extract embeddings
            new_clean = extract_hidden(model, processor, base_img, prompt)
            clean_shift = float(cosine(clean_emb, new_clean))

            new_corruption_dists = {}
            aurocs = {}
            for c in ['fog', 'night', 'blur', 'noise']:
                cimg = apply_corruption(base_img, c, 1.0)
                new_emb = extract_hidden(model, processor, cimg, prompt)
                new_corruption_dists[c] = float(cosine(new_clean, new_emb))
                # AUROC with just 1 clean and 1 corrupted
                aurocs[c] = 1.0 if new_corruption_dists[c] > clean_shift else 0.0

            layer_results.append({
                "noise_scale": scale,
                "relative_noise": scale / param_norm,
                "clean_shift": clean_shift,
                "corruption_distances": new_corruption_dists,
                "aurocs": aurocs,
                "detection_preserved": all(new_corruption_dists[c] > clean_shift for c in ['fog', 'night', 'blur', 'noise'])
            })
            print(f"    scale={scale:.1e}: clean_shift={clean_shift:.6f}, "
                  f"preserved={layer_results[-1]['detection_preserved']}")

            # Restore weights
            param.data = original_data.clone()

        layer_perturb_results[f"layer_{target_layer}"] = {
            "param_norm": param_norm,
            "results": layer_results
        }

    results["layer_perturbation"] = layer_perturb_results

    # Part 2: Global weight noise (all parameters)
    print("\n=== Part 2: Global Weight Noise ===")
    global_results = []

    for scale in [1e-7, 1e-6, 1e-5, 1e-4]:
        print(f"  Global noise scale={scale:.1e}...")
        # Save all original weights
        original_state = {}
        for name, param in model.named_parameters():
            original_state[name] = param.data.clone()

        # Add noise to ALL parameters
        for name, param in model.named_parameters():
            noise = torch.randn_like(param.data).to(param.data.dtype) * scale
            param.data = param.data + noise

        # Re-extract
        new_clean = extract_hidden(model, processor, base_img, prompt)
        clean_shift = float(cosine(clean_emb, new_clean))

        corruption_results = {}
        id_dists = [clean_shift]
        ood_dists = []

        for c in ['fog', 'night', 'blur', 'noise']:
            cimg = apply_corruption(base_img, c, 1.0)
            new_emb = extract_hidden(model, processor, cimg, prompt)
            d = float(cosine(new_clean, new_emb))
            corruption_results[c] = d
            ood_dists.append(d)

        auroc = compute_auroc(id_dists, ood_dists)

        global_results.append({
            "noise_scale": scale,
            "clean_shift": clean_shift,
            "corruption_distances": corruption_results,
            "auroc": auroc,
            "all_detected": all(corruption_results[c] > clean_shift for c in corruption_results)
        })
        print(f"    clean_shift={clean_shift:.6f}, AUROC={auroc:.3f}, all_detected={global_results[-1]['all_detected']}")

        # Restore ALL weights
        for name, param in model.named_parameters():
            param.data = original_state[name]

    results["global_perturbation"] = global_results

    # Part 3: Quantization simulation (round weights to fewer bits)
    print("\n=== Part 3: Quantization Simulation ===")
    quant_results = []

    for bits in [16, 12, 8, 6, 4]:
        print(f"  Simulating {bits}-bit quantization...")
        original_state = {}
        for name, param in model.named_parameters():
            original_state[name] = param.data.clone()

        # Simulate quantization by rounding
        for name, param in model.named_parameters():
            p_float = param.data.float()
            p_max = p_float.abs().max()
            if p_max > 0:
                # Scale to [-1, 1], quantize, scale back
                n_levels = 2**bits
                scaled = p_float / p_max
                quantized = torch.round(scaled * (n_levels / 2)) / (n_levels / 2)
                param.data = (quantized * p_max).to(param.data.dtype)

        # Re-extract
        new_clean = extract_hidden(model, processor, base_img, prompt)
        clean_shift = float(cosine(clean_emb, new_clean))

        corruption_results = {}
        for c in ['fog', 'night', 'blur', 'noise']:
            cimg = apply_corruption(base_img, c, 1.0)
            new_emb = extract_hidden(model, processor, cimg, prompt)
            corruption_results[c] = float(cosine(new_clean, new_emb))

        quant_results.append({
            "bits": bits,
            "clean_shift": clean_shift,
            "corruption_distances": corruption_results,
            "all_detected": all(corruption_results[c] > clean_shift for c in corruption_results)
        })
        print(f"    clean_shift={clean_shift:.6f}, all_detected={quant_results[-1]['all_detected']}")

        # Restore
        for name, param in model.named_parameters():
            param.data = original_state[name]

    results["quantization"] = quant_results

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/weight_perturb_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
