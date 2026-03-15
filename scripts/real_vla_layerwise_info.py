#!/usr/bin/env python3
"""Experiment 298: Layer-wise Information Flow
Traces how OOD detection information flows through transformer layers:
1. Per-layer cosine distance profiles for each corruption
2. Layer-wise classification accuracy (which layer first enables type ID)
3. Per-layer Fisher information (sensitivity to severity)
4. Information gain between consecutive layers
5. Layer-wise AUROC with different token positions
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
        "experiment": "layerwise_info_flow",
        "experiment_number": 298,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    # Sample layers: every 2 layers for fine-grained profile
    target_layers = list(range(0, 33))  # L0 to L32

    # Part 1: Per-layer cosine distance profile
    print("=== Part 1: Per-Layer Distance Profile ===")

    # Get all clean hidden states
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    clean_hiddens = {}
    for li in target_layers:
        if li < len(fwd.hidden_states):
            clean_hiddens[li] = fwd.hidden_states[li][0, -1, :].float().cpu().numpy()

    n_layers_available = len(fwd.hidden_states)
    print(f"  Model has {n_layers_available} hidden states")

    distance_profiles = {}
    for c in corruptions:
        print(f"  Processing {c}...")
        distance_profiles[c] = {}
        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd_c = model(**inputs, output_hidden_states=True)

            dists = []
            for li in target_layers:
                if li < n_layers_available:
                    emb = fwd_c.hidden_states[li][0, -1, :].float().cpu().numpy()
                    d = float(cosine(clean_hiddens[li], emb))
                    dists.append({"layer": li, "distance": d})
            distance_profiles[c][f"sev_{sev}"] = dists

            d_vals = [x["distance"] for x in dists]
            print(f"    sev={sev}: min_d={min(d_vals):.6f} (L{d_vals.index(min(d_vals))}), "
                  f"max_d={max(d_vals):.6f} (L{d_vals.index(max(d_vals))})")

    results["distance_profiles"] = distance_profiles

    # Part 2: Per-layer AUROC
    print("\n=== Part 2: Per-Layer AUROC ===")
    layer_aurocs = {}

    # Clean ID distances (3 identical passes)
    id_dists = {li: [] for li in target_layers if li < n_layers_available}
    for _ in range(3):
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_clean = model(**inputs, output_hidden_states=True)
        for li in target_layers:
            if li < n_layers_available:
                emb = fwd_clean.hidden_states[li][0, -1, :].float().cpu().numpy()
                d = float(cosine(clean_hiddens[li], emb))
                id_dists[li].append(d)

    for c in corruptions:
        layer_aurocs[c] = []
        ood_dists = {li: [] for li in target_layers if li < n_layers_available}
        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd_c = model(**inputs, output_hidden_states=True)
            for li in target_layers:
                if li < n_layers_available:
                    emb = fwd_c.hidden_states[li][0, -1, :].float().cpu().numpy()
                    d = float(cosine(clean_hiddens[li], emb))
                    ood_dists[li].append(d)

        for li in target_layers:
            if li < n_layers_available:
                auroc = compute_auroc(id_dists[li], ood_dists[li])
                layer_aurocs[c].append({"layer": li, "auroc": auroc})

        auroc_vals = [x["auroc"] for x in layer_aurocs[c]]
        first_perfect = next((i for i, a in enumerate(auroc_vals) if a >= 1.0), -1)
        print(f"  {c}: first AUROC=1.0 at L{first_perfect if first_perfect >= 0 else 'none'}, "
              f"min={min(auroc_vals):.3f}")

    results["layer_aurocs"] = layer_aurocs

    # Part 3: Per-layer classification accuracy
    print("\n=== Part 3: Per-Layer Type Classification ===")
    # Compute direction vectors per layer
    classification_by_layer = []

    for li in target_layers:
        if li < n_layers_available:
            directions = {}
            for c in corruptions:
                corrupted = apply_corruption(base_img, c, 1.0)
                inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    fwd_c = model(**inputs, output_hidden_states=True)
                emb = fwd_c.hidden_states[li][0, -1, :].float().cpu().numpy()
                diff = emb - clean_hiddens[li]
                norm = np.linalg.norm(diff)
                directions[c] = diff / norm if norm > 0 else diff

            # Test classification at severity 0.5
            correct = 0
            total = 0
            for c_true in corruptions:
                corrupted = apply_corruption(base_img, c_true, 0.5)
                inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    fwd_c = model(**inputs, output_hidden_states=True)
                emb = fwd_c.hidden_states[li][0, -1, :].float().cpu().numpy()
                diff = emb - clean_hiddens[li]
                best = max(corruptions, key=lambda c: np.dot(diff, directions[c]))
                if best == c_true:
                    correct += 1
                total += 1

            acc = correct / total
            classification_by_layer.append({"layer": li, "accuracy": acc, "correct": correct, "total": total})
            if li % 8 == 0 or li == n_layers_available - 1:
                print(f"  L{li}: {correct}/{total} = {acc:.3f}")

    results["classification_by_layer"] = classification_by_layer

    # Part 4: Information gain between layers
    print("\n=== Part 4: Layer-to-Layer Information Gain ===")
    info_gain = {}
    for c in corruptions:
        gains = []
        prev_d = None
        for entry in distance_profiles[c]["sev_1.0"]:
            d = entry["distance"]
            if prev_d is not None:
                gain = d - prev_d
                gains.append({"from_layer": entry["layer"]-1, "to_layer": entry["layer"], "gain": gain})
            prev_d = d
        info_gain[c] = gains
        # Find max gain layer
        max_gain = max(gains, key=lambda x: x["gain"])
        min_gain = min(gains, key=lambda x: x["gain"])
        print(f"  {c}: max gain L{max_gain['from_layer']}→L{max_gain['to_layer']} ({max_gain['gain']:+.6f}), "
              f"max loss L{min_gain['from_layer']}→L{min_gain['to_layer']} ({min_gain['gain']:+.6f})")

    results["info_gain"] = info_gain

    # Part 5: Token position comparison at key layers
    print("\n=== Part 5: Token Position Analysis ===")
    token_positions = {}
    for tok_name, tok_idx in [("first", 0), ("middle", -1), ("last", -1)]:
        # Get sequence length
        seq_len = fwd.hidden_states[0].shape[1]
        if tok_name == "first":
            tidx = 0
        elif tok_name == "middle":
            tidx = seq_len // 2
        else:
            tidx = -1

        # Clean reference
        clean_ref = {}
        for li in [0, 3, 7, 15, 31]:
            if li < n_layers_available:
                clean_ref[li] = fwd.hidden_states[li][0, tidx, :].float().cpu().numpy()

        # Corrupted distances
        tok_dists = {}
        for c in corruptions:
            corrupted = apply_corruption(base_img, c, 1.0)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd_c = model(**inputs, output_hidden_states=True)
            tok_dists[c] = {}
            for li in [0, 3, 7, 15, 31]:
                if li < n_layers_available:
                    emb = fwd_c.hidden_states[li][0, tidx, :].float().cpu().numpy()
                    d = float(cosine(clean_ref[li], emb))
                    tok_dists[c][f"L{li}"] = d

        token_positions[tok_name] = {"token_index": tidx, "distances": tok_dists}
        print(f"  {tok_name} (idx={tidx}): fog_L3={tok_dists['fog'].get('L3', 'N/A'):.6f}, "
              f"night_L3={tok_dists['night'].get('L3', 'N/A'):.6f}")

    results["token_positions"] = token_positions

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/layerwise_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
