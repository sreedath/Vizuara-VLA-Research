#!/usr/bin/env python3
"""Experiment 328: Token-Level Probability Analysis (Real OpenVLA-7B)

Analyzes how corruption affects the model's action token probabilities:
1. Top-k token probabilities under each corruption
2. Entropy of next-token distribution
3. Confidence (max probability) as corruption metric
4. Token rank changes under corruption
5. KL divergence between clean and corrupt distributions
6. Agreement between different corruption types
7. Calibration of softmax probabilities
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

def get_logits_and_hidden(model, processor, image, prompt, layer=3):
    """Get both logits and hidden states."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    logits = out.logits[0, -1, :].float().cpu()
    hidden = out.hidden_states[layer][0, -1, :].float().cpu().numpy()
    return logits, hidden

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    results = {}

    # Action token range: 31744-31999
    ACTION_START = 31744
    ACTION_END = 31999

    # ========== 1. Clean baseline ==========
    print("\n=== Clean Baseline ===")
    clean_logits, clean_hidden = get_logits_and_hidden(model, processor, base_img, prompt)
    clean_probs = torch.softmax(clean_logits, dim=0).numpy()

    # Action token probabilities
    action_probs = clean_probs[ACTION_START:ACTION_END+1]
    clean_action_sum = float(np.sum(action_probs))
    clean_top_token = int(np.argmax(clean_probs))
    clean_top_prob = float(clean_probs[clean_top_token])
    clean_entropy = float(-np.sum(clean_probs * np.log(clean_probs + 1e-20)))
    clean_action_entropy = float(-np.sum(action_probs * np.log(action_probs + 1e-20) / max(np.sum(action_probs), 1e-20)))

    # Top-5 tokens
    top5_idx = np.argsort(clean_probs)[-5:][::-1]
    clean_top5 = [(int(idx), float(clean_probs[idx])) for idx in top5_idx]

    results['clean'] = {
        'top_token': clean_top_token,
        'top_prob': clean_top_prob,
        'entropy': clean_entropy,
        'action_prob_sum': clean_action_sum,
        'top5': clean_top5,
        'is_action_token': bool(ACTION_START <= clean_top_token <= ACTION_END),
    }
    print(f"  Top token: {clean_top_token} (p={clean_top_prob:.4f})")
    print(f"  Entropy: {clean_entropy:.4f}")
    print(f"  Action token total prob: {clean_action_sum:.4f}")

    # ========== 2. Corruption effects on logits ==========
    print("\n=== Corruption Effects on Logits ===")
    ctypes = ['fog', 'night', 'noise', 'blur']
    corruption_results = {}

    for ct in ctypes:
        img = apply_corruption(base_img, ct, 1.0)
        logits, hidden = get_logits_and_hidden(model, processor, img, prompt)
        probs = torch.softmax(logits, dim=0).numpy()

        action_probs_c = probs[ACTION_START:ACTION_END+1]
        top_token = int(np.argmax(probs))
        top_prob = float(probs[top_token])
        entropy = float(-np.sum(probs * np.log(probs + 1e-20)))

        # KL divergence from clean
        kl = float(np.sum(clean_probs * np.log((clean_probs + 1e-20) / (probs + 1e-20))))

        # Top-5
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5 = [(int(idx), float(probs[idx])) for idx in top5_idx]

        # Rank of clean top token under corruption
        sorted_idx = np.argsort(probs)[::-1]
        clean_token_rank = int(np.where(sorted_idx == clean_top_token)[0][0]) + 1

        # Token changed?
        token_changed = bool(top_token != clean_top_token)

        # Cosine distance in logit space
        logit_cos_dist = float(cosine_dist(clean_logits.numpy(), logits.numpy()))

        # Hidden state distance
        hidden_dist = float(cosine_dist(clean_hidden, hidden))

        corruption_results[ct] = {
            'top_token': top_token,
            'top_prob': top_prob,
            'entropy': entropy,
            'entropy_change': float(entropy - clean_entropy),
            'kl_divergence': kl,
            'action_prob_sum': float(np.sum(action_probs_c)),
            'top5': top5,
            'clean_token_rank': clean_token_rank,
            'token_changed': token_changed,
            'logit_cosine_dist': logit_cos_dist,
            'hidden_cosine_dist': hidden_dist,
        }
        print(f"  {ct}: token={top_token} ({'CHANGED' if token_changed else 'same'}), p={top_prob:.4f}, entropy={entropy:.4f}, KL={kl:.4f}")

    results['corruptions'] = corruption_results

    # ========== 3. Severity sweep of entropy and confidence ==========
    print("\n=== Severity Sweep ===")
    severity_results = {}
    for ct in ctypes:
        entries = []
        for sev in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            img = apply_corruption(base_img, ct, sev)
            logits, hidden = get_logits_and_hidden(model, processor, img, prompt)
            probs = torch.softmax(logits, dim=0).numpy()

            top_token = int(np.argmax(probs))
            top_prob = float(probs[top_token])
            entropy = float(-np.sum(probs * np.log(probs + 1e-20)))
            kl = float(np.sum(clean_probs * np.log((clean_probs + 1e-20) / (probs + 1e-20))))
            hidden_d = float(cosine_dist(clean_hidden, hidden))

            entries.append({
                'severity': float(sev),
                'top_token': top_token,
                'top_prob': top_prob,
                'entropy': entropy,
                'kl': kl,
                'hidden_dist': hidden_d,
                'token_changed': bool(top_token != clean_top_token),
            })

        severity_results[ct] = entries
        changes = [e for e in entries if e['token_changed']]
        print(f"  {ct}: token changes at {[e['severity'] for e in changes]}")

    results['severity_sweep'] = severity_results

    # ========== 4. Temperature scaling ==========
    print("\n=== Temperature Scaling ===")
    temp_results = {}
    for temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        clean_p = torch.softmax(clean_logits / temp, dim=0).numpy()
        fog_logits, _ = get_logits_and_hidden(model, processor,
                                               apply_corruption(base_img, 'fog', 1.0), prompt)
        fog_p = torch.softmax(fog_logits / temp, dim=0).numpy()

        kl = float(np.sum(clean_p * np.log((clean_p + 1e-20) / (fog_p + 1e-20))))
        clean_ent = float(-np.sum(clean_p * np.log(clean_p + 1e-20)))
        fog_ent = float(-np.sum(fog_p * np.log(fog_p + 1e-20)))

        temp_results[str(temp)] = {
            'kl': kl,
            'clean_entropy': clean_ent,
            'fog_entropy': fog_ent,
            'clean_max_prob': float(np.max(clean_p)),
            'fog_max_prob': float(np.max(fog_p)),
        }
        print(f"  T={temp}: KL={kl:.4f}, clean_ent={clean_ent:.4f}, fog_ent={fog_ent:.4f}")

    results['temperature'] = temp_results

    # ========== 5. Logit vs hidden correlation ==========
    print("\n=== Logit-Hidden Correlation ===")
    logit_hidden = []
    for ct in ctypes:
        for sev in [0.1, 0.25, 0.5, 0.75, 1.0]:
            img = apply_corruption(base_img, ct, sev)
            logits, hidden = get_logits_and_hidden(model, processor, img, prompt)
            probs = torch.softmax(logits, dim=0).numpy()

            hidden_d = float(cosine_dist(clean_hidden, hidden))
            logit_d = float(cosine_dist(clean_logits.numpy(), logits.numpy()))
            kl = float(np.sum(clean_probs * np.log((clean_probs + 1e-20) / (probs + 1e-20))))

            logit_hidden.append({
                'type': ct,
                'severity': float(sev),
                'hidden_dist': hidden_d,
                'logit_dist': logit_d,
                'kl': kl,
            })

    # Correlation
    h_dists = [x['hidden_dist'] for x in logit_hidden]
    l_dists = [x['logit_dist'] for x in logit_hidden]
    kl_vals = [x['kl'] for x in logit_hidden]

    corr_hl = float(np.corrcoef(h_dists, l_dists)[0, 1]) if len(h_dists) > 1 else 0
    corr_hk = float(np.corrcoef(h_dists, kl_vals)[0, 1]) if len(h_dists) > 1 else 0

    results['logit_hidden_correlation'] = {
        'hidden_logit_corr': corr_hl,
        'hidden_kl_corr': corr_hk,
        'data': logit_hidden,
    }
    print(f"  Hidden-Logit correlation: {corr_hl:.4f}")
    print(f"  Hidden-KL correlation: {corr_hk:.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/token_analysis_{ts}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return obj

    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [recursive_convert(x) for x in d]
        return convert(d)

    results = recursive_convert(results)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
