#!/usr/bin/env python3
"""Experiment 454: Edge Case and Robustness Analysis.

Tests edge cases that could break the detector in deployment:
adversarial-style images, extreme values, unusual inputs, and
boundary conditions.

Analyses:
  1. Extreme input images (all-black, all-white, solid colours, gradients, etc.)
  2. Image size / resolution variants
  3. Partial corruption (half, quadrant, small patch, border)
  4. Sequential corruption application order (permutations of 2-corruption combos)
  5. Adversarial-inspired perturbations (pixel changes, channel swap, flip, rotation)
  6. Determinism verification (5 identical forward passes)

Calibration: 8 clean scenes (seeds 42, 123, 456, 789, 1000, 2000, 3000, 4000).
"""

import torch, json, os, sys, numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime
from itertools import permutations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent
EXPERIMENTS_DIR = REPO_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = str(EXPERIMENTS_DIR)

# ---------------------------------------------------------------------------
# Standard image factory
# ---------------------------------------------------------------------------
def make_image(seed=42):
    """Create a calibration scene image from a seed (224x224 RGB)."""
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# ---------------------------------------------------------------------------
# Standard corruption helpers (from experiment prompt spec)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Embedding extraction (layer 3, last token)
# ---------------------------------------------------------------------------
def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

# ---------------------------------------------------------------------------
# Cosine distance
# ---------------------------------------------------------------------------
def cosine_dist(a, b):
    na = np.linalg.norm(a) + 1e-10
    nb = np.linalg.norm(b) + 1e-10
    return float(1.0 - np.dot(a / na, b / nb))

# ---------------------------------------------------------------------------
# Hand-written AUROC (Wilcoxon–Mann–Whitney) with trapezoid fallback
# ---------------------------------------------------------------------------
def compute_auroc(id_scores, ood_scores):
    """Compute AUROC treating higher score = more OOD.

    Uses the WMW statistic. Falls back to trapezoidal rule when all scores
    are identical (degenerate case).
    """
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
    n_id = len(id_s)
    n_ood = len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5

    # Check for degenerate (all identical) case -> trapezoid fallback
    all_scores = np.concatenate([id_s, ood_s])
    if np.all(all_scores == all_scores[0]):
        # Perfectly indistinguishable — AUROC = 0.5 by trapezoid
        return 0.5

    # WMW statistic
    count = 0.0
    for o in ood_s:
        count += float(np.sum(o > id_s)) + 0.5 * float(np.sum(o == id_s))
    return count / (n_id * n_ood)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)
    return model, processor

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CAL_SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
PROMPT = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

def build_centroid(model, processor):
    print(f"\n[Calibration] {len(CAL_SEEDS)} scenes...", flush=True)
    embeddings = []
    for i, seed in enumerate(CAL_SEEDS):
        img = make_image(seed)
        emb = extract_hidden(model, processor, img, PROMPT)
        embeddings.append(emb)
        print(f"  Cal {i+1}/{len(CAL_SEEDS)} (seed={seed})", flush=True)
    emb_array = np.array(embeddings)
    centroid = emb_array.mean(axis=0)
    # Compute calibration distances for threshold estimation
    cal_dists = [cosine_dist(e, centroid) for e in embeddings]
    cal_mean = float(np.mean(cal_dists))
    cal_std = float(np.std(cal_dists))
    threshold = cal_mean + 3.0 * cal_std  # 3-sigma threshold
    print(f"  Centroid built. cal_mean={cal_mean:.6f}, cal_std={cal_std:.6f}, "
          f"threshold(3σ)={threshold:.6f}", flush=True)
    return centroid, cal_mean, cal_std, threshold, cal_dists

# ---------------------------------------------------------------------------
# Analysis 1: Extreme input images
# ---------------------------------------------------------------------------
def analysis_extreme_inputs(model, processor, centroid, threshold):
    """Test edge-case synthetic images for OOD detection."""
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 1: Extreme Input Images", flush=True)
    print("=" * 70, flush=True)

    H, W = 224, 224

    def checkerboard(block=8):
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for r in range(0, H, block):
            for c in range(0, W, block):
                val = 255 if ((r // block + c // block) % 2 == 0) else 0
                arr[r:r+block, c:c+block] = val
        return arr

    def gradient_bw():
        col = np.linspace(0, 255, W, dtype=np.uint8)
        arr = np.tile(col, (H, 1))
        return np.stack([arr, arr, arr], axis=2)

    def high_freq_noise():
        rng = np.random.RandomState(9999)  # different seed from cal images
        return rng.randint(0, 256, (H, W, 3), dtype=np.uint8)

    extreme_images = {
        "all_black":     np.zeros((H, W, 3), dtype=np.uint8),
        "all_white":     np.full((H, W, 3), 255, dtype=np.uint8),
        "pure_red":      np.array([[[255, 0, 0]]] * 1, dtype=np.uint8).repeat(H, 0)
                         .repeat(W, 1).reshape(H, W, 3),
        "pure_green":    np.array([[[0, 255, 0]]] * 1, dtype=np.uint8).repeat(H, 0)
                         .repeat(W, 1).reshape(H, W, 3),
        "pure_blue":     np.array([[[0, 0, 255]]] * 1, dtype=np.uint8).repeat(H, 0)
                         .repeat(W, 1).reshape(H, W, 3),
        "gradient_bw":   gradient_bw(),
        "checkerboard":  checkerboard(block=8),
        "hf_noise":      high_freq_noise(),
    }

    results = {}
    total = len(extreme_images)
    for idx, (name, arr) in enumerate(extreme_images.items(), 1):
        img = Image.fromarray(arr)
        emb = extract_hidden(model, processor, img, PROMPT)
        dist = cosine_dist(emb, centroid)
        flagged = bool(dist > threshold)
        results[name] = {
            "cosine_dist": float(dist),
            "flagged_as_ood": flagged,
            "threshold": float(threshold),
        }
        print(f"  [{idx}/{total}] {name:<20}: dist={dist:.6f}, flagged={flagged}", flush=True)

    n_flagged = sum(1 for r in results.values() if r["flagged_as_ood"])
    print(f"\n  {n_flagged}/{total} extreme images flagged as OOD.", flush=True)
    return results

# ---------------------------------------------------------------------------
# Analysis 2: Image size / resolution variants
# ---------------------------------------------------------------------------
def analysis_resolution(model, processor, centroid, threshold):
    """Test various input resolutions (processor handles resize internally)."""
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 2: Image Size / Resolution", flush=True)
    print("=" * 70, flush=True)

    base_seed = 42
    base_arr = np.random.RandomState(base_seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)

    size_cases = {
        "tiny_32x32":       (32, 32),
        "small_64x64":      (64, 64),
        "native_224x224":   (224, 224),
        "large_512x512":    (512, 512),
        "large_1024x1024":  (1024, 1024),
        "nonsquare_224x112": (224, 112),
        "nonsquare_112x224": (112, 224),
    }

    # Get baseline from native size
    base_img = Image.fromarray(base_arr)
    base_emb = extract_hidden(model, processor, base_img, PROMPT)
    baseline_dist = cosine_dist(base_emb, centroid)

    results = {}
    total = len(size_cases)
    for idx, (name, (h, w)) in enumerate(size_cases.items(), 1):
        resized = base_img.resize((w, h), Image.LANCZOS)
        emb = extract_hidden(model, processor, resized, PROMPT)
        dist = cosine_dist(emb, centroid)
        flagged = bool(dist > threshold)
        dist_from_native = cosine_dist(emb, base_emb)
        results[name] = {
            "size": [h, w],
            "cosine_dist_from_centroid": float(dist),
            "cosine_dist_from_native": float(dist_from_native),
            "flagged_as_ood": flagged,
            "threshold": float(threshold),
        }
        print(f"  [{idx}/{total}] {name:<26}: dist_centroid={dist:.6f}, "
              f"dist_native={dist_from_native:.6f}, flagged={flagged}", flush=True)

    print(f"\n  Baseline (native 224x224): dist={baseline_dist:.6f}", flush=True)
    results["baseline_native_224x224"] = {
        "cosine_dist_from_centroid": float(baseline_dist),
        "flagged_as_ood": bool(baseline_dist > threshold),
    }
    return results

# ---------------------------------------------------------------------------
# Analysis 3: Partial corruption
# ---------------------------------------------------------------------------
def analysis_partial_corruption(model, processor, centroid, threshold):
    """Apply night corruption to only a portion of the image."""
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 3: Partial Corruption", flush=True)
    print("=" * 70, flush=True)

    base_img = make_image(seed=42)
    base_arr = np.array(base_img).astype(np.float32)
    H, W = base_arr.shape[:2]

    def night_array(arr):
        """Apply night to a float32 [0,255] array, return uint8."""
        return np.clip(arr * max(0.01, 1.0 - 0.95), 0, 255).astype(np.uint8)

    def make_partial(base_float, mask):
        """mask is bool array (H,W); True = apply night, False = keep clean."""
        out = base_float.copy()
        dark = night_array(base_float)
        out[mask] = dark[mask].astype(np.float32)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    # Build masks
    top_mask = np.zeros((H, W), dtype=bool)
    top_mask[:H//2, :] = True

    bot_mask = np.zeros((H, W), dtype=bool)
    bot_mask[H//2:, :] = True

    left_mask = np.zeros((H, W), dtype=bool)
    left_mask[:, :W//2] = True

    right_mask = np.zeros((H, W), dtype=bool)
    right_mask[:, W//2:] = True

    patch_mask = np.zeros((H, W), dtype=bool)
    pr, pc = 50, 50  # 50x50 patch at centre
    patch_mask[H//2-pr//2:H//2+pr//2, W//2-pc//2:W//2+pc//2] = True

    border_mask = np.zeros((H, W), dtype=bool)
    b = 20
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True

    full_mask = np.ones((H, W), dtype=bool)

    partial_cases = {
        "clean":               np.zeros((H, W), dtype=bool),
        "top_half":            top_mask,
        "bottom_half":         bot_mask,
        "left_half":           left_mask,
        "right_half":          right_mask,
        "center_patch_50x50":  patch_mask,
        "border_20px":         border_mask,
        "fully_corrupted":     full_mask,
    }

    base_float = base_arr.copy()
    results = {}
    total = len(partial_cases)
    for idx, (name, mask) in enumerate(partial_cases.items(), 1):
        img = make_partial(base_float, mask)
        emb = extract_hidden(model, processor, img, PROMPT)
        dist = cosine_dist(emb, centroid)
        flagged = bool(dist > threshold)
        frac = float(mask.sum()) / (H * W)
        results[name] = {
            "corrupted_fraction": round(frac, 4),
            "cosine_dist": float(dist),
            "flagged_as_ood": flagged,
            "threshold": float(threshold),
        }
        print(f"  [{idx}/{total}] {name:<26}: frac={frac:.3f}, "
              f"dist={dist:.6f}, flagged={flagged}", flush=True)

    return results

# ---------------------------------------------------------------------------
# Analysis 4: Sequential corruption application order
# ---------------------------------------------------------------------------
def analysis_corruption_order(model, processor, centroid, threshold):
    """Test whether the order of applying two corruptions affects detection."""
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 4: Sequential Corruption Application Order", flush=True)
    print("=" * 70, flush=True)

    corruption_types = ["fog", "night", "noise", "blur"]

    # Generate all 2-element permutations of distinct corruptions
    pairs = list(permutations(corruption_types, 2))

    base_img = make_image(seed=42)
    results = {}

    for a, b in pairs:
        # Order A->B
        img_ab = apply_corruption(apply_corruption(base_img, a, 1.0), b, 1.0)
        # Order B->A
        img_ba = apply_corruption(apply_corruption(base_img, b, 1.0), a, 1.0)

        emb_ab = extract_hidden(model, processor, img_ab, PROMPT)
        emb_ba = extract_hidden(model, processor, img_ba, PROMPT)

        dist_ab = cosine_dist(emb_ab, centroid)
        dist_ba = cosine_dist(emb_ba, centroid)
        order_diff = abs(dist_ab - dist_ba)
        # Embedding-space difference between the two orders
        emb_diff = cosine_dist(emb_ab, emb_ba)

        key = f"{a}_then_{b}"
        key_rev = f"{b}_then_{a}"
        results[key] = {
            "order": f"{a}->{b}",
            "cosine_dist": float(dist_ab),
            "flagged": bool(dist_ab > threshold),
        }
        results[key_rev] = {
            "order": f"{b}->{a}",
            "cosine_dist": float(dist_ba),
            "flagged": bool(dist_ba > threshold),
        }
        # Store pairwise comparison under a combined key
        combo = f"{a}+{b}"
        if combo not in results:
            results[combo] = {}
        results[combo]["order_distance_diff"] = float(order_diff)
        results[combo]["embedding_diff_between_orders"] = float(emb_diff)
        results[combo]["order_matters"] = bool(order_diff > 1e-6)

        print(f"  {a}->{b}: dist={dist_ab:.6f}  |  "
              f"{b}->{a}: dist={dist_ba:.6f}  |  "
              f"diff={order_diff:.2e}  emb_diff={emb_diff:.2e}", flush=True)

    return results

# ---------------------------------------------------------------------------
# Analysis 5: Adversarial-inspired perturbations
# ---------------------------------------------------------------------------
def analysis_adversarial_perturbations(model, processor, centroid, threshold):
    """Test pixel-level changes, channel swaps, flips, and rotations."""
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 5: Adversarial-Inspired Perturbations", flush=True)
    print("=" * 70, flush=True)

    base_img = make_image(seed=42)
    base_arr = np.array(base_img).copy()
    H, W = base_arr.shape[:2]

    # -- Baseline --
    base_emb = extract_hidden(model, processor, base_img, PROMPT)
    base_dist = cosine_dist(base_emb, centroid)

    rng = np.random.RandomState(777)

    def pixel_change(arr, n_pixels):
        """Change exactly n_pixels pixels to random values."""
        out = arr.copy()
        coords = rng.choice(H * W, size=n_pixels, replace=False)
        rows, cols = coords // W, coords % W
        out[rows, cols] = rng.randint(0, 256, (n_pixels, 3), dtype=np.uint8)
        return out

    perturbation_cases = {}

    # Pixel-level perturbations
    for n in [1, 10, 100, 1000]:
        perturbation_cases[f"pixel_change_{n}"] = Image.fromarray(pixel_change(base_arr, n))

    # Channel swaps
    gbr_arr = base_arr[:, :, [1, 2, 0]]  # R->G, G->B, B->R (GBR)
    brg_arr = base_arr[:, :, [2, 0, 1]]  # R->B, G->R, B->G (BRG)
    perturbation_cases["channel_swap_GBR"] = Image.fromarray(gbr_arr)
    perturbation_cases["channel_swap_BRG"] = Image.fromarray(brg_arr)

    # Geometric transforms
    perturbation_cases["horizontal_flip"] = base_img.transpose(Image.FLIP_LEFT_RIGHT)
    perturbation_cases["rotate_90"] = base_img.rotate(90, expand=False)

    results = {"baseline": {
        "cosine_dist": float(base_dist),
        "flagged": bool(base_dist > threshold),
    }}

    total = len(perturbation_cases)
    for idx, (name, img) in enumerate(perturbation_cases.items(), 1):
        emb = extract_hidden(model, processor, img, PROMPT)
        dist = cosine_dist(emb, centroid)
        dist_from_base = cosine_dist(emb, base_emb)
        flagged = bool(dist > threshold)
        results[name] = {
            "cosine_dist_from_centroid": float(dist),
            "cosine_dist_from_base": float(dist_from_base),
            "flagged_as_ood": flagged,
            "threshold": float(threshold),
        }
        print(f"  [{idx}/{total}] {name:<25}: dist_centroid={dist:.6f}, "
              f"dist_base={dist_from_base:.6f}, flagged={flagged}", flush=True)

    n_flagged = sum(1 for k, r in results.items() if k != "baseline" and r.get("flagged_as_ood"))
    print(f"\n  {n_flagged}/{total} perturbations flagged as OOD.", flush=True)
    return results

# ---------------------------------------------------------------------------
# Analysis 6: Determinism verification
# ---------------------------------------------------------------------------
def analysis_determinism(model, processor):
    """Run the same image through the model 5 times and check bit-identity."""
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 6: Determinism Verification", flush=True)
    print("=" * 70, flush=True)

    N_RUNS = 5
    img = make_image(seed=42)
    embeddings = []
    for i in range(N_RUNS):
        emb = extract_hidden(model, processor, img, PROMPT)
        embeddings.append(emb)
        print(f"  Run {i+1}/{N_RUNS} done.", flush=True)

    reference = embeddings[0]
    deviations = []
    bit_identical_all = True
    for i in range(1, N_RUNS):
        diff = embeddings[i] - reference
        max_abs = float(np.max(np.abs(diff)))
        l2 = float(np.linalg.norm(diff))
        cos_d = cosine_dist(embeddings[i], reference)
        bit_identical = bool(np.all(diff == 0.0))
        if not bit_identical:
            bit_identical_all = False
        deviations.append({
            "run": i + 1,
            "max_abs_diff": max_abs,
            "l2_diff": l2,
            "cosine_dist_from_run1": cos_d,
            "bit_identical_to_run1": bit_identical,
        })
        print(f"  Run {i+1} vs run 1: max_abs={max_abs:.2e}, l2={l2:.2e}, "
              f"cos_dist={cos_d:.2e}, bit_identical={bit_identical}", flush=True)

    result = {
        "n_runs": N_RUNS,
        "all_bit_identical": bit_identical_all,
        "deviations": deviations,
    }
    if bit_identical_all:
        print("\n  DETERMINISTIC: all 5 runs produce identical embeddings.", flush=True)
    else:
        max_seen = max(d["max_abs_diff"] for d in deviations)
        print(f"\n  NON-DETERMINISTIC: max deviation across runs = {max_seen:.2e}", flush=True)
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70, flush=True)
    print("EXPERIMENT 454: Edge Case and Robustness Analysis", flush=True)
    print(f"Timestamp: {timestamp}", flush=True)
    print("=" * 70, flush=True)

    model, processor = load_model()

    # Build calibration centroid and 3-sigma threshold
    centroid, cal_mean, cal_std, threshold, cal_dists = build_centroid(model, processor)

    # Run all analyses
    r1 = analysis_extreme_inputs(model, processor, centroid, threshold)
    r2 = analysis_resolution(model, processor, centroid, threshold)
    r3 = analysis_partial_corruption(model, processor, centroid, threshold)
    r4 = analysis_corruption_order(model, processor, centroid, threshold)
    r5 = analysis_adversarial_perturbations(model, processor, centroid, threshold)
    r6 = analysis_determinism(model, processor)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\nCalibration: mean={cal_mean:.6f}, std={cal_std:.6f}, "
          f"threshold(3σ)={threshold:.6f}", flush=True)

    print("\n[Analysis 1 — Extreme Images]", flush=True)
    for name, r in r1.items():
        print(f"  {name:<20}: dist={r['cosine_dist']:.6f}  flagged={r['flagged_as_ood']}", flush=True)

    print("\n[Analysis 2 — Resolutions]", flush=True)
    for name, r in r2.items():
        if "cosine_dist_from_centroid" in r:
            print(f"  {name:<30}: dist={r['cosine_dist_from_centroid']:.6f}  "
                  f"flagged={r['flagged_as_ood']}", flush=True)

    print("\n[Analysis 3 — Partial Corruption]", flush=True)
    for name, r in r3.items():
        print(f"  {name:<28}: frac={r['corrupted_fraction']:.3f}  "
              f"dist={r['cosine_dist']:.6f}  flagged={r['flagged_as_ood']}", flush=True)

    print("\n[Analysis 4 — Corruption Order]", flush=True)
    for key, r in r4.items():
        if "order_distance_diff" in r:
            print(f"  {key:<12}: diff={r['order_distance_diff']:.2e}  "
                  f"emb_diff={r['embedding_diff_between_orders']:.2e}  "
                  f"order_matters={r['order_matters']}", flush=True)

    print("\n[Analysis 5 — Adversarial Perturbations]", flush=True)
    for name, r in r5.items():
        if "cosine_dist_from_centroid" in r:
            print(f"  {name:<25}: dist={r['cosine_dist_from_centroid']:.6f}  "
                  f"flagged={r['flagged_as_ood']}", flush=True)

    print("\n[Analysis 6 — Determinism]", flush=True)
    print(f"  all_bit_identical={r6['all_bit_identical']}", flush=True)
    if not r6["all_bit_identical"]:
        diffs = [d["max_abs_diff"] for d in r6["deviations"]]
        print(f"  max deviation across runs: {max(diffs):.2e}", flush=True)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "experiment": "robustness_edge_cases",
        "experiment_number": 454,
        "timestamp": timestamp,
        "calibration": {
            "seeds": CAL_SEEDS,
            "n_cal": len(CAL_SEEDS),
            "cal_mean": float(cal_mean),
            "cal_std": float(cal_std),
            "threshold_3sigma": float(threshold),
            "cal_dists": [float(d) for d in cal_dists],
        },
        "analysis_1_extreme_inputs": r1,
        "analysis_2_resolution": r2,
        "analysis_3_partial_corruption": r3,
        "analysis_4_corruption_order": r4,
        "analysis_5_adversarial_perturbations": r5,
        "analysis_6_determinism": r6,
    }

    out_path = os.path.join(RESULTS_DIR, f"robustness_edge_cases_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
