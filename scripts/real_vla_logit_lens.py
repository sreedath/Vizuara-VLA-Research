#!/usr/bin/env python3
"""Experiment 366: Logit Lens Analysis Under Corruption

Apply the "logit lens" technique: project hidden states from each layer
through the unembedding matrix to see how token predictions form across layers.
1. Action token prediction emergence across layers
2. Layer at which corruption first changes the predicted token
3. Prediction entropy profile across layers
4. Corruption vs clean prediction divergence per layer
5. "Decision boundary" layer identification
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

def get_all_hidden_states(model, processor, image, prompt):
    """Get hidden states from all layers."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    return [h[0, -1, :].float().cpu() for h in fwd.hidden_states], inputs['input_ids'].shape[1]

def project_to_logits(hidden, lm_head):
    """Project hidden state through the language model head to get logits."""
    with torch.no_grad():
        logits = lm_head(hidden.unsqueeze(0).to(lm_head.weight.device, dtype=lm_head.weight.dtype))
    return logits[0, 0].float().cpu()

def entropy(logits):
    """Compute entropy of softmax distribution."""
    probs = torch.softmax(logits, dim=-1).numpy()
    return -float(np.sum(probs * np.log2(probs + 1e-12)))

def kl_divergence(logits_p, logits_q):
    """KL(P||Q) where P and Q are softmax distributions."""
    p = torch.softmax(logits_p, dim=-1).numpy() + 1e-12
    q = torch.softmax(logits_q, dim=-1).numpy() + 1e-12
    return float(np.sum(p * np.log2(p / q)))

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    # Get the LM head
    lm_head = model.language_model.lm_head

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 800, 100))[:8]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # Get clean hidden states
    print("Computing clean hidden states...")
    clean_hidden = {}
    for seed in seeds:
        hidden_states, seq_len = get_all_hidden_states(model, processor, images[seed], prompt)
        clean_hidden[seed] = hidden_states
    n_layers = len(clean_hidden[seeds[0]])
    print(f"  {n_layers} hidden states (including embedding layer)")

    # ========== 1. Token Prediction Across Layers ==========
    print("\n=== Token Prediction Across Layers ===")

    token_emergence = {}
    for seed in seeds[:4]:
        per_layer = {}
        final_logits = project_to_logits(clean_hidden[seed][-1], lm_head)
        final_token = int(torch.argmax(final_logits))

        for l in range(n_layers):
            logits = project_to_logits(clean_hidden[seed][l], lm_head)
            pred_token = int(torch.argmax(logits))
            sorted_indices = torch.sort(logits, descending=True).indices
            matches = (sorted_indices == final_token).nonzero(as_tuple=True)[0]
            rank_of_final = int(matches[0]) if len(matches) > 0 else -1

            per_layer[str(l)] = {
                'predicted_token': pred_token,
                'matches_final': pred_token == final_token,
                'rank_of_final_token': rank_of_final,
                'entropy': entropy(logits),
                'top1_prob': float(torch.softmax(logits, dim=-1).max()),
            }

        # Find first layer where prediction matches final
        first_match = None
        for l in range(n_layers):
            if per_layer[str(l)]['matches_final']:
                first_match = l
                break

        token_emergence[str(seed)] = {
            'final_token': final_token,
            'first_match_layer': first_match,
            'per_layer': per_layer,
        }
        print(f"  seed={seed}: final_token={final_token}, first_match@layer={first_match}")

    results['token_emergence'] = token_emergence

    # ========== 2. Corruption Impact on Logit Lens ==========
    print("\n=== Corruption Impact on Logit Lens ===")

    corruption_impact = {}
    for ct in ctypes:
        per_seed = {}
        for seed in seeds[:4]:
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_hidden, _ = get_all_hidden_states(model, processor, corrupt_img, prompt)

            divergence_layers = []
            first_diverge = None
            for l in range(n_layers):
                clean_logits = project_to_logits(clean_hidden[seed][l], lm_head)
                corrupt_logits = project_to_logits(corrupt_hidden[l], lm_head)

                clean_pred = int(torch.argmax(clean_logits))
                corrupt_pred = int(torch.argmax(corrupt_logits))

                kl = kl_divergence(clean_logits, corrupt_logits)

                divergence_layers.append({
                    'layer': l,
                    'tokens_differ': clean_pred != corrupt_pred,
                    'kl_divergence': float(kl),
                    'clean_entropy': entropy(clean_logits),
                    'corrupt_entropy': entropy(corrupt_logits),
                })

                if clean_pred != corrupt_pred and first_diverge is None:
                    first_diverge = l

            per_seed[str(seed)] = {
                'first_diverge_layer': first_diverge,
                'per_layer': divergence_layers,
            }

        # Aggregate
        first_diverges = [per_seed[str(s)]['first_diverge_layer'] for s in seeds[:4] if per_seed[str(s)]['first_diverge_layer'] is not None]
        mean_kl_per_layer = {}
        for l in range(n_layers):
            kls = [per_seed[str(s)]['per_layer'][l]['kl_divergence'] for s in seeds[:4]]
            mean_kl_per_layer[str(l)] = float(np.mean(kls))

        corruption_impact[ct] = {
            'mean_first_diverge': float(np.mean(first_diverges)) if first_diverges else None,
            'min_first_diverge': int(min(first_diverges)) if first_diverges else None,
            'mean_kl_per_layer': mean_kl_per_layer,
            'per_seed': per_seed,
        }

        if first_diverges:
            print(f"  {ct}: first_diverge mean={np.mean(first_diverges):.1f}, "
                  f"min={min(first_diverges)}")
        else:
            print(f"  {ct}: tokens never diverge")

    results['corruption_impact'] = corruption_impact

    # ========== 3. Entropy Profile Across Layers ==========
    print("\n=== Entropy Profile ===")

    entropy_profile = {}
    for seed in seeds[:4]:
        clean_ents = []
        for l in range(n_layers):
            logits = project_to_logits(clean_hidden[seed][l], lm_head)
            clean_ents.append(entropy(logits))

        entropy_profile[str(seed)] = {
            'clean_entropy': clean_ents,
            'max_entropy_layer': int(np.argmax(clean_ents)),
            'min_entropy_layer': int(np.argmin(clean_ents)),
            'final_entropy': clean_ents[-1],
        }
        print(f"  seed={seed}: max_ent@L{np.argmax(clean_ents)} ({max(clean_ents):.2f}), "
              f"final_ent={clean_ents[-1]:.2f}")

    results['entropy_profile'] = entropy_profile

    # ========== 4. KL Divergence Peak Layer ==========
    print("\n=== KL Divergence Peak Layer ===")

    kl_peaks = {}
    for ct in ctypes:
        mean_kl = corruption_impact[ct]['mean_kl_per_layer']
        layers_kl = [(int(l), v) for l, v in mean_kl.items()]
        peak_layer = max(layers_kl, key=lambda x: x[1])
        kl_peaks[ct] = {
            'peak_layer': peak_layer[0],
            'peak_kl': peak_layer[1],
            'early_mean_kl': float(np.mean([v for l, v in layers_kl if l < n_layers//2])),
            'late_mean_kl': float(np.mean([v for l, v in layers_kl if l >= n_layers//2])),
        }
        print(f"  {ct}: peak@L{peak_layer[0]} (KL={peak_layer[1]:.4f}), "
              f"early/late={kl_peaks[ct]['early_mean_kl']:.4f}/{kl_peaks[ct]['late_mean_kl']:.4f}")

    results['kl_peaks'] = kl_peaks

    # ========== 5. Decision Boundary Layer ==========
    print("\n=== Decision Boundary Layer ===")

    # For each corruption, find the layer where >50% of seeds have diverged
    boundary = {}
    for ct in ctypes:
        for l in range(n_layers):
            n_diverged = 0
            for seed in seeds[:4]:
                fl = corruption_impact[ct]['per_seed'][str(seed)]['first_diverge_layer']
                if fl is not None and fl <= l:
                    n_diverged += 1
            if n_diverged >= 2:  # majority of 4
                boundary[ct] = {
                    'boundary_layer': l,
                    'fraction_diverged': n_diverged / 4,
                }
                print(f"  {ct}: decision boundary at layer {l} ({n_diverged}/4 diverged)")
                break
        if ct not in boundary:
            boundary[ct] = {'boundary_layer': None, 'fraction_diverged': 0}
            print(f"  {ct}: no clear boundary")

    results['decision_boundary'] = boundary

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/logit_lens_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, torch.Tensor): return obj.tolist()
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
