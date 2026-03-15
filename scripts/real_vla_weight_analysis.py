#!/usr/bin/env python3
"""Experiment 437: Weight Space & Gradient Sensitivity Analysis

Examines the relationship between model weights and OOD detection
by computing input gradients and analyzing which weight matrices
contribute most to the corruption-sensitive hidden state dimensions.

Tests:
1. Input gradient magnitude under clean vs corrupted
2. Gradient direction alignment across corruptions
3. Weight matrix statistics per layer
4. Fisher information approximation
5. Gradient-based corruption attribution
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

def cosine_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

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

    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    results = {"n_scenes": len(scenes)}

    # === Test 1: Input gradient analysis ===
    print("\n=== Input Gradient Analysis ===")
    # Compute gradient of hidden state norm w.r.t. pixel inputs
    grad_results = {}
    for condition in ['clean', 'fog', 'night']:
        grad_norms = []
        for s_idx, s in enumerate(scenes[:3]):
            img = s if condition == 'clean' else apply_corruption(s, condition)
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)

            # Enable gradient on pixel values
            pixel_values = inputs.get('pixel_values', None)
            if pixel_values is not None:
                pixel_values = pixel_values.clone().detach().requires_grad_(True)
                inputs['pixel_values'] = pixel_values

            fwd = model(**inputs, output_hidden_states=True)
            hidden = fwd.hidden_states[3][0, -1, :]
            hidden_norm = hidden.float().norm()
            hidden_norm.backward()

            if pixel_values is not None and pixel_values.grad is not None:
                grad = pixel_values.grad.float().cpu().numpy()
                grad_norms.append(float(np.linalg.norm(grad)))
                print(f"  {condition} scene {s_idx}: grad_norm={np.linalg.norm(grad):.6f}")
            else:
                grad_norms.append(0.0)
                print(f"  {condition} scene {s_idx}: no pixel gradient available")
            model.zero_grad()

        grad_results[condition] = {
            "mean_grad_norm": float(np.mean(grad_norms)),
            "std_grad_norm": float(np.std(grad_norms)),
        }
    results["input_gradients"] = grad_results

    # === Test 2: Weight matrix statistics ===
    print("\n=== Weight Matrix Statistics ===")
    weight_stats = {}
    for name, param in model.named_parameters():
        if 'layers.2.' in name and ('weight' in name or 'bias' in name):
            w = param.float().cpu().detach().numpy()
            weight_stats[name] = {
                "shape": list(w.shape),
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "sparsity": float(np.mean(np.abs(w) < 0.001)),
                "frobenius_norm": float(np.linalg.norm(w)),
            }
    # Print summary
    print(f"  Found {len(weight_stats)} weight matrices in layer 2")
    for name, stats in list(weight_stats.items())[:5]:
        print(f"    {name}: shape={stats['shape']}, std={stats['std']:.6f}")
    results["weight_stats"] = weight_stats

    # === Test 3: Hidden state gradient w.r.t. specific dimension ===
    print("\n=== Discriminative Dimension Gradients ===")
    # Dimension 1512 was found to be most discriminative
    disc_dim = 1512
    disc_grad_results = {}
    for condition in ['clean', 'fog']:
        dim_grads = []
        for s_idx, s in enumerate(scenes[:3]):
            img = s if condition == 'clean' else apply_corruption(s, condition)
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)

            pixel_values = inputs.get('pixel_values', None)
            if pixel_values is not None:
                pixel_values = pixel_values.clone().detach().requires_grad_(True)
                inputs['pixel_values'] = pixel_values

            fwd = model(**inputs, output_hidden_states=True)
            target_val = fwd.hidden_states[3][0, -1, disc_dim].float()
            target_val.backward()

            if pixel_values is not None and pixel_values.grad is not None:
                grad = pixel_values.grad.float().cpu().numpy()
                dim_grads.append(float(np.linalg.norm(grad)))
            else:
                dim_grads.append(0.0)
            model.zero_grad()

        disc_grad_results[condition] = {
            "mean_grad_norm": float(np.mean(dim_grads)),
            "dimension": disc_dim,
        }
        print(f"  dim {disc_dim} ({condition}): mean_grad_norm={np.mean(dim_grads):.6f}")
    results["disc_dim_gradients"] = disc_grad_results

    # === Test 4: Model parameter count ===
    print("\n=== Model Architecture Summary ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count parameters per component
    component_params = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0]
        if component not in component_params:
            component_params[component] = 0
        component_params[component] += param.numel()

    results["model_architecture"] = {
        "total_params": total_params,
        "total_params_billions": float(total_params / 1e9),
        "component_params": {k: int(v) for k, v in component_params.items()},
    }
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    for comp, count in sorted(component_params.items(), key=lambda x: -x[1])[:5]:
        print(f"    {comp}: {count:,} ({count/total_params*100:.1f}%)")

    # === Test 5: Embedding similarity between clean/corrupt per layer ===
    print("\n=== Cross-Condition Embedding Similarity ===")
    sim_results = {}
    s = scenes[0]
    fog_img = apply_corruption(s, 'fog')
    night_img = apply_corruption(s, 'night')

    inputs_clean = processor(prompt, s).to(model.device, dtype=torch.bfloat16)
    inputs_fog = processor(prompt, fog_img).to(model.device, dtype=torch.bfloat16)
    inputs_night = processor(prompt, night_img).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        h_clean = model(**inputs_clean, output_hidden_states=True).hidden_states
        h_fog = model(**inputs_fog, output_hidden_states=True).hidden_states
        h_night = model(**inputs_night, output_hidden_states=True).hidden_states

    for layer in [0, 1, 3, 8, 16, 31, len(h_clean)-1]:
        if layer >= len(h_clean):
            continue
        clean_emb = h_clean[layer][0, -1, :].float().cpu().numpy()
        fog_emb = h_fog[layer][0, -1, :].float().cpu().numpy()
        night_emb = h_night[layer][0, -1, :].float().cpu().numpy()

        sim_cf = cosine_sim(clean_emb, fog_emb)
        sim_cn = cosine_sim(clean_emb, night_emb)
        sim_fn = cosine_sim(fog_emb, night_emb)

        sim_results[str(layer)] = {
            "clean_fog": sim_cf,
            "clean_night": sim_cn,
            "fog_night": sim_fn,
        }
        print(f"  Layer {layer}: clean↔fog={sim_cf:.6f}, clean↔night={sim_cn:.6f}, fog↔night={sim_fn:.6f}")
    results["cross_condition_similarity"] = sim_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/weight_analysis_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
