#!/usr/bin/env python3
"""Experiment 311: Precision Severity Estimation from Embedding Distance
Tests whether cosine distance enables precise corruption severity estimation:
1. Fine-grained distance-severity calibration curves (50 severity levels)
2. Linear vs polynomial regression fit quality
3. Cross-corruption severity estimation generalization
4. Uncertainty quantification for severity estimates
5. Actionable severity thresholds (safe/warning/danger zones)
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

def get_action_tokens(model, processor, image, prompt):
    ACTION_TOKEN_START = 31744
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=7, do_sample=False)
    input_len = inputs['input_ids'].shape[1]
    gen_tokens = generated[0, input_len:].cpu().numpy()
    return [int(t - ACTION_TOKEN_START) for t in gen_tokens]

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
        "experiment": "severity_estimation",
        "experiment_number": 311,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    clean_emb = extract_hidden(model, processor, base_img, prompt)
    clean_actions = get_action_tokens(model, processor, base_img, prompt)

    # Part 1: Fine-grained calibration curves
    print("=== Part 1: Calibration Curves (20 severities) ===")
    calibration = {}

    severities = np.linspace(0, 1.0, 21)[1:]  # 0.05, 0.10, ..., 1.0

    for c in corruptions:
        print(f"  {c}...")
        cal_data = []
        for sev in severities:
            corrupted = apply_corruption(base_img, c, float(sev))
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(clean_emb, emb))
            actions = get_action_tokens(model, processor, corrupted, prompt)
            n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
            total_shift = sum(abs(a - b) for a, b in zip(clean_actions, actions))

            cal_data.append({
                "severity": float(sev),
                "distance": d,
                "n_actions_changed": n_changed,
                "total_action_shift": total_shift,
            })

        calibration[c] = cal_data

        # Fit linear regression
        sevs = np.array([d["severity"] for d in cal_data])
        dists = np.array([d["distance"] for d in cal_data])

        # Linear fit
        if np.std(dists) > 0:
            coeffs = np.polyfit(sevs, dists, 1)
            pred_linear = np.polyval(coeffs, sevs)
            ss_res = np.sum((dists - pred_linear) ** 2)
            ss_tot = np.sum((dists - np.mean(dists)) ** 2)
            r2_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            r2_linear = 0

        # Quadratic fit
        if np.std(dists) > 0:
            coeffs_q = np.polyfit(sevs, dists, 2)
            pred_quad = np.polyval(coeffs_q, sevs)
            ss_res_q = np.sum((dists - pred_quad) ** 2)
            r2_quad = 1 - ss_res_q / ss_tot if ss_tot > 0 else 0
        else:
            r2_quad = 0

        print(f"    R²_linear={r2_linear:.4f}, R²_quad={r2_quad:.4f}")
        calibration[c].append({"r2_linear": r2_linear, "r2_quad": r2_quad})

    results["calibration"] = calibration

    # Part 2: Severity estimation accuracy
    print("\n=== Part 2: Severity Estimation ===")
    estimation = {}

    for c in corruptions:
        print(f"  {c}...")
        # Build calibration from known severities
        cal_data = [d for d in calibration[c] if isinstance(d, dict) and "severity" in d]
        cal_sevs = np.array([d["severity"] for d in cal_data])
        cal_dists = np.array([d["distance"] for d in cal_data])

        # Test at intermediate severities
        test_sevs = [0.075, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        estimates = []

        for test_sev in test_sevs:
            corrupted = apply_corruption(base_img, c, test_sev)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(clean_emb, emb))

            # Estimate severity via nearest calibration point
            nearest_idx = np.argmin(np.abs(cal_dists - d))
            estimated_sev = float(cal_sevs[nearest_idx])

            # Linear interpolation
            if d <= cal_dists[0]:
                interp_sev = float(cal_sevs[0] * d / (cal_dists[0] + 1e-30))
            elif d >= cal_dists[-1]:
                interp_sev = float(cal_sevs[-1])
            else:
                for i in range(len(cal_dists) - 1):
                    if cal_dists[i] <= d <= cal_dists[i+1]:
                        alpha = (d - cal_dists[i]) / (cal_dists[i+1] - cal_dists[i] + 1e-30)
                        interp_sev = float(cal_sevs[i] + alpha * (cal_sevs[i+1] - cal_sevs[i]))
                        break
                else:
                    interp_sev = estimated_sev

            estimates.append({
                "true_severity": test_sev,
                "distance": d,
                "estimated_severity": interp_sev,
                "abs_error": abs(interp_sev - test_sev),
            })

        mae = np.mean([e["abs_error"] for e in estimates])
        max_error = max(e["abs_error"] for e in estimates)
        estimation[c] = {
            "estimates": estimates,
            "mae": float(mae),
            "max_error": float(max_error),
        }
        print(f"    MAE={mae:.4f}, max_error={max_error:.4f}")

    results["estimation"] = estimation

    # Part 3: Action safety thresholds
    print("\n=== Part 3: Action Safety Thresholds ===")
    thresholds = {}

    for c in corruptions:
        cal_data = [d for d in calibration[c] if isinstance(d, dict) and "severity" in d]

        # Find severity where first action changes
        first_change = next((d for d in cal_data if d["n_actions_changed"] > 0), None)
        # Find severity where majority of actions change
        majority_change = next((d for d in cal_data if d["n_actions_changed"] >= 4), None)
        # Find severity where ALL actions change
        all_change = next((d for d in cal_data if d["n_actions_changed"] == 7), None)

        thresholds[c] = {
            "safe_max_severity": float(first_change["severity"] - 0.05) if first_change else 1.0,
            "safe_max_distance": float(first_change["distance"]) if first_change else 0,
            "first_action_change_sev": float(first_change["severity"]) if first_change else None,
            "first_action_change_d": float(first_change["distance"]) if first_change else None,
            "majority_change_sev": float(majority_change["severity"]) if majority_change else None,
            "all_change_sev": float(all_change["severity"]) if all_change else None,
        }
        print(f"  {c}: safe<{thresholds[c]['safe_max_severity']:.2f}, "
              f"first_change@{thresholds[c].get('first_action_change_sev', 'N/A')}, "
              f"all_change@{thresholds[c].get('all_change_sev', 'N/A')}")

    results["thresholds"] = thresholds

    # Part 4: Distance-to-action-shift correlation
    print("\n=== Part 4: Distance-Action Correlation ===")
    correlation = {}

    for c in corruptions:
        cal_data = [d for d in calibration[c] if isinstance(d, dict) and "severity" in d]
        dists = np.array([d["distance"] for d in cal_data])
        shifts = np.array([d["total_action_shift"] for d in cal_data])

        if np.std(dists) > 0 and np.std(shifts) > 0:
            corr = float(np.corrcoef(dists, shifts)[0, 1])
        else:
            corr = 0

        correlation[c] = {
            "distance_action_correlation": corr,
            "max_action_shift": int(shifts.max()),
            "distance_at_max_shift": float(dists[np.argmax(shifts)]),
        }
        print(f"  {c}: corr={corr:.4f}, max_shift={int(shifts.max())}")

    results["correlation"] = correlation

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/severity_est_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
