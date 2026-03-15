#!/usr/bin/env python3
"""Experiment 391: Action Safety Analysis Under Corruption

Deep analysis of how corruptions affect action predictions and whether
detection can prevent dangerous actions. Correlates detection distance
with action divergence magnitude across severity levels.

Tests:
1. Per-dimension action token shifts under each corruption at multiple severities
2. Action vector direction/magnitude changes
3. Detection distance vs action divergence correlation
4. Action stability across varied corrupted frames
5. Detection-before-danger timing analysis
6. Action confidence vs actual divergence
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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def extract_actions(model, processor, image, prompt):
    """Extract action tokens and logits."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_ids = out.sequences[0, input_len:].cpu().tolist()

    # Get confidence scores
    confidences = []
    if hasattr(out, 'scores') and out.scores:
        for step_scores in out.scores:
            probs = torch.softmax(step_scores[0], dim=-1)
            top_prob = probs.max().item()
            confidences.append(top_prob)

    # Convert tokens to normalized actions
    actions = []
    for tid in generated_ids[:7]:
        if 31744 <= tid <= 31999:
            actions.append((tid - 31744) / 255.0)
        else:
            actions.append(tid / 32000.0)

    return np.array(actions), np.array(generated_ids[:7]), confidences

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
    severities = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Clean baseline
    print("Clean baseline...")
    clean_actions, clean_tokens, clean_conf = extract_actions(model, processor, img, prompt)
    clean_hidden = extract_hidden(model, processor, img, prompt)
    print(f"  Actions: {clean_actions}")
    print(f"  Tokens: {clean_tokens}")
    print(f"  Confidence: {clean_conf}")

    results = {
        "clean_actions": clean_actions.tolist(),
        "clean_tokens": clean_tokens.tolist(),
        "clean_confidence": clean_conf,
        "severity_analysis": {},
        "per_dim_vulnerability": {},
        "detection_before_danger": {},
        "confidence_vs_divergence": []
    }

    # Severity sweep per corruption
    print("\n=== Severity Analysis ===")
    for c in corruptions:
        results["severity_analysis"][c] = []
        for sev in severities:
            corrupted = apply_corruption(img, c, severity=sev)
            actions, tokens, conf = extract_actions(model, processor, corrupted, prompt)
            hidden = extract_hidden(model, processor, corrupted, prompt)

            # Action metrics
            diff = actions - clean_actions
            l2_dist = float(np.linalg.norm(diff))
            max_dim_shift = float(np.max(np.abs(diff)))
            n_changed = int(np.sum(tokens != clean_tokens))

            # Direction change
            na, nc = np.linalg.norm(clean_actions), np.linalg.norm(actions)
            if na > 1e-12 and nc > 1e-12:
                cos_sim = float(np.dot(clean_actions, actions) / (na * nc))
            else:
                cos_sim = 0.0

            # Detection distance
            det_dist = float(cosine_dist(hidden, clean_hidden))

            # Mean confidence
            mean_conf = float(np.mean(conf)) if conf else 0.0

            entry = {
                "severity": sev,
                "actions": actions.tolist(),
                "tokens": tokens.tolist(),
                "l2_dist": l2_dist,
                "max_dim_shift": max_dim_shift,
                "n_tokens_changed": n_changed,
                "cos_sim": cos_sim,
                "detection_dist": det_dist,
                "mean_confidence": mean_conf,
                "per_dim_shift": diff.tolist()
            }
            results["severity_analysis"][c].append(entry)
            results["confidence_vs_divergence"].append({
                "corruption": c, "severity": sev,
                "confidence": mean_conf, "l2_dist": l2_dist,
                "detection_dist": det_dist
            })
            print(f"  {c} sev={sev:.2f}: l2={l2_dist:.4f}, changed={n_changed}/7, "
                  f"det={det_dist:.6f}, conf={mean_conf:.4f}")

    # Per-dimension vulnerability
    print("\n=== Per-Dimension Vulnerability ===")
    for c in corruptions:
        full_sev = results["severity_analysis"][c][-1]  # severity=1.0
        dim_shifts = np.abs(full_sev["per_dim_shift"])
        most_vulnerable = int(np.argmax(dim_shifts))
        least_vulnerable = int(np.argmin(dim_shifts))
        results["per_dim_vulnerability"][c] = {
            "abs_shifts": dim_shifts.tolist(),
            "most_vulnerable_dim": most_vulnerable,
            "least_vulnerable_dim": least_vulnerable,
            "most_vulnerable_shift": float(dim_shifts[most_vulnerable]),
            "least_vulnerable_shift": float(dim_shifts[least_vulnerable])
        }
        print(f"  {c}: most_vuln=dim{most_vulnerable} ({dim_shifts[most_vulnerable]:.4f}), "
              f"least_vuln=dim{least_vulnerable} ({dim_shifts[least_vulnerable]:.4f})")

    # Detection-before-danger: at what severity does detection trigger vs action change?
    print("\n=== Detection Before Danger ===")
    detection_threshold = 7.8e-5  # From experiment 387 (10-sample calibration)
    for c in corruptions:
        det_sev = None
        action_sev = None
        for entry in results["severity_analysis"][c]:
            if det_sev is None and entry["detection_dist"] > detection_threshold:
                det_sev = entry["severity"]
            if action_sev is None and entry["n_tokens_changed"] > 0:
                action_sev = entry["severity"]

        results["detection_before_danger"][c] = {
            "detection_trigger_severity": det_sev,
            "action_change_severity": action_sev,
            "detection_leads": det_sev is not None and (action_sev is None or det_sev <= action_sev)
        }
        print(f"  {c}: detection triggers at sev={det_sev}, action changes at sev={action_sev}, "
              f"detection_leads={det_sev is not None and (action_sev is None or det_sev <= action_sev)}")

    # Correlation analysis
    print("\n=== Correlation: Detection Distance vs Action L2 ===")
    all_det = [e["detection_dist"] for e in results["confidence_vs_divergence"]]
    all_l2 = [e["l2_dist"] for e in results["confidence_vs_divergence"]]
    all_conf = [e["confidence"] for e in results["confidence_vs_divergence"]]

    if len(all_det) > 2:
        det_l2_corr = float(np.corrcoef(all_det, all_l2)[0, 1])
        conf_l2_corr = float(np.corrcoef(all_conf, all_l2)[0, 1])
        det_conf_corr = float(np.corrcoef(all_det, all_conf)[0, 1])
        results["correlations"] = {
            "detection_vs_action_l2": det_l2_corr,
            "confidence_vs_action_l2": conf_l2_corr,
            "detection_vs_confidence": det_conf_corr
        }
        print(f"  det vs l2: r={det_l2_corr:.4f}")
        print(f"  conf vs l2: r={conf_l2_corr:.4f}")
        print(f"  det vs conf: r={det_conf_corr:.4f}")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/action_safety_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
