#!/usr/bin/env python3
"""Experiment 402: Uncertainty Decomposition Analysis

Decompose model uncertainty into aleatoric (data) and epistemic (model) components.
Uses MC dropout approximation and token probability analysis.

Tests:
1. Token logit entropy under clean vs corrupt (aleatoric proxy)
2. MC dropout variance (epistemic proxy via dropout at inference)
3. Top-k action token probability spread
4. Calibration curve for action confidence
5. Entropy-distance correlation
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
    corruptions = ['fog', 'night', 'noise', 'blur']
    ACTION_TOKEN_START = 31744
    ACTION_TOKEN_END = 31999

    scenes = []
    for seed in [42, 123, 456, 789, 999]:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    def get_logits_and_hidden(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        logits = out.logits[0, -1, :].float().cpu().numpy()  # (vocab_size,)
        hidden = out.hidden_states[3][0, -1, :].float().cpu().numpy()
        return logits, hidden

    results = {}

    # === Test 1: Full logit analysis ===
    print("=== Logit Analysis ===")
    for condition in ['clean'] + corruptions:
        print(f"\n  {condition}:")
        condition_data = {"per_scene": []}

        for si, scene in enumerate(scenes):
            if condition == 'clean':
                img = scene
            else:
                img = apply_corruption(scene, condition, 1.0)

            logits, hidden = get_logits_and_hidden(img)

            # Softmax probabilities
            logits_shifted = logits - logits.max()
            probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()

            # Full entropy
            eps = 1e-10
            full_entropy = -np.sum(probs * np.log(probs + eps))

            # Action token analysis
            action_logits = logits[ACTION_TOKEN_START:ACTION_TOKEN_END+1]
            action_probs_raw = np.exp(action_logits - action_logits.max())
            action_probs = action_probs_raw / action_probs_raw.sum()
            action_entropy = -np.sum(action_probs * np.log(action_probs + eps))

            # Total probability mass on action tokens
            action_mass = float(probs[ACTION_TOKEN_START:ACTION_TOKEN_END+1].sum())

            # Top-k action tokens
            top_k = 5
            top_indices = np.argsort(action_probs)[-top_k:][::-1]
            top_probs = action_probs[top_indices]

            # Confidence = max action probability
            confidence = float(np.max(action_probs))

            # Token ID spread
            top_token_ids = [int(ACTION_TOKEN_START + idx) for idx in top_indices]
            token_spread = int(max(top_indices) - min(top_indices))

            scene_data = {
                "full_entropy": float(full_entropy),
                "action_entropy": float(action_entropy),
                "action_mass": action_mass,
                "confidence": confidence,
                "top_k_probs": [float(p) for p in top_probs],
                "top_k_tokens": top_token_ids,
                "token_spread": token_spread,
            }
            condition_data["per_scene"].append(scene_data)

        # Aggregate
        all_ent = [s["full_entropy"] for s in condition_data["per_scene"]]
        all_act_ent = [s["action_entropy"] for s in condition_data["per_scene"]]
        all_conf = [s["confidence"] for s in condition_data["per_scene"]]
        all_mass = [s["action_mass"] for s in condition_data["per_scene"]]

        condition_data["mean_full_entropy"] = float(np.mean(all_ent))
        condition_data["mean_action_entropy"] = float(np.mean(all_act_ent))
        condition_data["mean_confidence"] = float(np.mean(all_conf))
        condition_data["mean_action_mass"] = float(np.mean(all_mass))

        print(f"    full_ent={np.mean(all_ent):.4f}, act_ent={np.mean(all_act_ent):.4f}, "
              f"conf={np.mean(all_conf):.4f}, mass={np.mean(all_mass):.4f}")

        results[condition] = condition_data

    # === Test 2: MC Dropout approximation ===
    print("\n=== MC Dropout Analysis ===")
    mc_results = {}

    def enable_dropout(model):
        """Enable dropout layers for MC inference."""
        count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
                count += 1
        return count

    n_dropout = enable_dropout(model)
    print(f"  Enabled {n_dropout} dropout layers")

    n_mc_samples = 10
    for condition in ['clean', 'fog', 'night']:
        print(f"\n  {condition} ({n_mc_samples} MC samples):")
        mc_data = {"per_scene": []}

        for si, scene in enumerate(scenes[:3]):  # Use 3 scenes for speed
            if condition == 'clean':
                img = scene
            else:
                img = apply_corruption(scene, condition, 1.0)

            mc_hiddens = []
            mc_logits = []
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)

            for _ in range(n_mc_samples):
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                h = out.hidden_states[3][0, -1, :].float().cpu().numpy()
                l = out.logits[0, -1, :].float().cpu().numpy()
                mc_hiddens.append(h)
                mc_logits.append(l)

            mc_hiddens = np.array(mc_hiddens)
            mc_logits = np.array(mc_logits)

            # Epistemic uncertainty = variance across MC samples
            hidden_var = float(np.mean(np.var(mc_hiddens, axis=0)))

            # Pairwise cosine distances between MC samples
            mc_cos_dists = []
            for i in range(n_mc_samples):
                for j in range(i+1, n_mc_samples):
                    mc_cos_dists.append(cosine_dist(mc_hiddens[i], mc_hiddens[j]))
            mean_mc_dist = float(np.mean(mc_cos_dists))

            # Logit variance
            logit_var = float(np.mean(np.var(mc_logits, axis=0)))

            mc_data["per_scene"].append({
                "hidden_var": hidden_var,
                "mean_mc_cosine_dist": mean_mc_dist,
                "logit_var": logit_var,
            })
            print(f"    scene {si}: hidden_var={hidden_var:.8f}, mc_cos={mean_mc_dist:.8f}, logit_var={logit_var:.4f}")

        mc_data["mean_hidden_var"] = float(np.mean([s["hidden_var"] for s in mc_data["per_scene"]]))
        mc_data["mean_mc_cosine"] = float(np.mean([s["mean_mc_cosine_dist"] for s in mc_data["per_scene"]]))
        mc_results[condition] = mc_data

    model.eval()  # Restore eval mode
    results["mc_dropout"] = mc_results

    # === Test 3: Entropy-distance correlation ===
    print("\n=== Entropy-Distance Correlation ===")
    all_dists = []
    all_entropies = []

    # Get clean centroid
    clean_hiddens = []
    for scene in scenes:
        inputs = processor(prompt, scene).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        clean_hiddens.append(fwd.hidden_states[3][0, -1, :].float().cpu().numpy())
    centroid = np.mean(clean_hiddens, axis=0)

    for condition in ['clean'] + corruptions:
        for si, scene in enumerate(scenes):
            if condition == 'clean':
                img = scene
            else:
                img = apply_corruption(scene, condition, 1.0)

            logits, hidden = get_logits_and_hidden(img)
            d = cosine_dist(hidden, centroid)
            logits_shifted = logits - logits.max()
            probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
            ent = -np.sum(probs * np.log(probs + 1e-10))

            all_dists.append(d)
            all_entropies.append(ent)

    # Correlation
    correlation = float(np.corrcoef(all_dists, all_entropies)[0, 1])
    print(f"  Distance-entropy correlation: {correlation:.4f}")
    results["entropy_distance_correlation"] = correlation

    # === Test 4: Severity-entropy curve ===
    print("\n=== Severity-Entropy Curve ===")
    sev_entropy = {}
    for c in corruptions:
        sev_data = {}
        for sev in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            ents = []
            for scene in scenes[:3]:
                if sev == 0.0:
                    img = scene
                else:
                    img = apply_corruption(scene, c, sev)
                logits, _ = get_logits_and_hidden(img)
                logits_shifted = logits - logits.max()
                probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
                ent = -np.sum(probs * np.log(probs + 1e-10))
                ents.append(ent)
            sev_data[str(sev)] = float(np.mean(ents))
        sev_entropy[c] = sev_data
        vals = [sev_data[str(s)] for s in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]]
        print(f"  {c}: {' -> '.join(f'{v:.3f}' for v in vals)}")

    results["severity_entropy"] = sev_entropy

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/uncertainty_decomposition_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
