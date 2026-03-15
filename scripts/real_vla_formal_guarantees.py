#!/usr/bin/env python3
"""Experiment 313: Formal Safety Guarantees & Statistical Testing
Provides rigorous statistical backing for deployment safety claims:
1. Hoeffding bounds on detection error probability
2. Clopper-Pearson exact confidence intervals
3. Sequential probability ratio test (SPRT) for online monitoring
4. Finite-sample PAC bounds on false positive/negative rates
5. Worst-case analysis across corruption types and severities
"""

import torch
import numpy as np
import json
import time
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine
from scipy import stats

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
        "experiment": "formal_guarantees",
        "experiment_number": 313,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    # Part 1: Repeated clean embedding stability (N=30 passes)
    print("=== Part 1: Clean Embedding Stability (30 passes) ===")
    clean_distances = []
    clean_embeddings = []

    for i in range(30):
        emb = extract_hidden(model, processor, base_img, prompt)
        d = float(cosine(clean_emb, emb))
        clean_distances.append(d)
        clean_embeddings.append(emb)

    # Check bit-identical
    identical_count = sum(1 for e in clean_embeddings if np.array_equal(e, clean_emb))

    results["clean_stability"] = {
        "n_passes": 30,
        "all_distances_zero": all(d == 0.0 for d in clean_distances),
        "bit_identical_count": identical_count,
        "max_distance": float(max(clean_distances)),
        "mean_distance": float(np.mean(clean_distances)),
    }
    print(f"  All zero: {results['clean_stability']['all_distances_zero']}")
    print(f"  Bit-identical: {identical_count}/30")

    # Part 2: Corruption distance distribution (20 severities × 4 corruptions)
    print("\n=== Part 2: Corruption Distance Distribution ===")
    ood_distances = {}
    all_ood_dists = []

    severities = np.linspace(0.05, 1.0, 20)

    for c in corruptions:
        print(f"  {c}...")
        dists = []
        for sev in severities:
            corrupted = apply_corruption(base_img, c, float(sev))
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(clean_emb, emb))
            dists.append(d)
            all_ood_dists.append(d)
        ood_distances[c] = dists

    results["ood_distribution"] = {}
    for c in corruptions:
        dists = ood_distances[c]
        results["ood_distribution"][c] = {
            "min": float(min(dists)),
            "max": float(max(dists)),
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "all_positive": all(d > 0 for d in dists),
        }
        print(f"    min={min(dists):.8f}, max={max(dists):.8f}, all>0: {all(d > 0 for d in dists)}")

    # Part 3: Hoeffding Bound on detection error
    print("\n=== Part 3: Hoeffding Bounds ===")
    # For binary classification: P(error) <= 2*exp(-2*n*epsilon^2)
    # We have n=30 clean (all d=0) and n=80 OOD (all d>0)
    # Empirical error = 0/110 = 0

    n_clean = 30
    n_ood = len(all_ood_dists)  # 80
    n_total = n_clean + n_ood

    # Number correct
    n_correct_clean = sum(1 for d in clean_distances if d == 0.0)
    n_correct_ood = sum(1 for d in all_ood_dists if d > 0)
    n_correct = n_correct_clean + n_correct_ood
    empirical_error = 1.0 - n_correct / n_total

    hoeffding_bounds = {}
    for confidence in [0.90, 0.95, 0.99, 0.999]:
        # Hoeffding: P(|p_hat - p| > eps) <= 2*exp(-2*n*eps^2)
        # Solving: eps = sqrt(ln(2/delta) / (2*n))
        delta = 1 - confidence
        eps = np.sqrt(np.log(2 / delta) / (2 * n_total))
        # Upper bound on true error
        error_upper = empirical_error + eps
        hoeffding_bounds[f"{confidence:.3f}"] = {
            "delta": delta,
            "epsilon": float(eps),
            "error_upper_bound": float(error_upper),
            "accuracy_lower_bound": float(1 - error_upper),
        }
        print(f"  {confidence*100:.1f}% confidence: error < {error_upper:.4f}, accuracy > {1-error_upper:.4f}")

    results["hoeffding"] = hoeffding_bounds

    # Part 4: Clopper-Pearson Exact Confidence Intervals
    print("\n=== Part 4: Clopper-Pearson Exact CIs ===")
    clopper_pearson = {}

    # For sensitivity (TP / (TP + FN))
    tp = n_correct_ood
    fn = n_ood - n_correct_ood
    for confidence in [0.90, 0.95, 0.99]:
        alpha = 1 - confidence
        if tp == n_ood:  # Perfect detection
            # One-sided lower bound for p=1.0
            lower = 1 - (alpha / 2) ** (1 / n_ood)
        else:
            lower = stats.beta.ppf(alpha / 2, tp, fn + 1)
        upper = 1.0 if tp == n_ood else stats.beta.ppf(1 - alpha / 2, tp + 1, fn)

        clopper_pearson[f"sensitivity_{confidence:.2f}"] = {
            "n": n_ood,
            "k": tp,
            "lower": float(lower),
            "upper": float(upper),
            "point_estimate": float(tp / n_ood),
        }
        print(f"  Sensitivity {confidence*100:.0f}%: [{lower:.6f}, {upper:.6f}]")

    # For specificity (TN / (TN + FP))
    tn = n_correct_clean
    fp = n_clean - n_correct_clean
    for confidence in [0.90, 0.95, 0.99]:
        alpha = 1 - confidence
        if tn == n_clean:
            lower = 1 - (alpha / 2) ** (1 / n_clean)
        else:
            lower = stats.beta.ppf(alpha / 2, tn, fp + 1)
        upper = 1.0 if tn == n_clean else stats.beta.ppf(1 - alpha / 2, tn + 1, fp)

        clopper_pearson[f"specificity_{confidence:.2f}"] = {
            "n": n_clean,
            "k": tn,
            "lower": float(lower),
            "upper": float(upper),
            "point_estimate": float(tn / n_clean),
        }
        print(f"  Specificity {confidence*100:.0f}%: [{lower:.6f}, {upper:.6f}]")

    results["clopper_pearson"] = clopper_pearson

    # Part 5: Sequential Probability Ratio Test (SPRT)
    print("\n=== Part 5: Sequential Probability Ratio Test ===")
    # Test H0: frame is clean vs H1: frame is corrupted
    # Using likelihood ratio on cosine distance
    # Under H0: d = 0 always (degenerate distribution)
    # Under H1: d > 0

    # Simulate SPRT on the 100-frame stream from exp 312
    # p0 = P(d=0 | clean) = 1.0
    # p1 = P(d>0 | corrupted) = 1.0
    # So SPRT decides instantly (1 frame)

    # More interesting: test with a probabilistic threshold
    # Use threshold tau and model P(d > tau) under each hypothesis
    min_ood_dist = min(all_ood_dists)
    thresholds = [0, min_ood_dist / 2, min_ood_dist, min_ood_dist * 2]

    sprt_results = {}
    for tau in thresholds:
        # For each threshold, how many frames needed to decide?
        # SPRT with alpha=beta=0.01 (type I and II error rates)
        alpha_sprt = 0.01
        beta_sprt = 0.01
        A = np.log((1 - beta_sprt) / alpha_sprt)  # Upper boundary
        B = np.log(beta_sprt / (1 - alpha_sprt))  # Lower boundary

        # Test on clean frames
        clean_decisions = []
        for d in clean_distances:
            # Binary: d > tau?
            if d > tau:
                llr = np.log(0.99 / 0.01)  # Strong evidence for H1
            else:
                llr = np.log(0.01 / 0.99)  # Strong evidence for H0
            clean_decisions.append({
                "distance": d,
                "exceeds_threshold": d > tau,
                "llr": float(llr),
                "decision": "H1_corrupted" if llr >= A else ("H0_clean" if llr <= B else "continue"),
            })

        # Test on OOD frames
        ood_decisions = []
        for d in all_ood_dists:
            if d > tau:
                llr = np.log(0.99 / 0.01)
            else:
                llr = np.log(0.01 / 0.99)
            ood_decisions.append({
                "distance": d,
                "exceeds_threshold": d > tau,
                "llr": float(llr),
                "decision": "H1_corrupted" if llr >= A else ("H0_clean" if llr <= B else "continue"),
            })

        n_clean_correct = sum(1 for d in clean_decisions if d["decision"] == "H0_clean")
        n_ood_correct = sum(1 for d in ood_decisions if d["decision"] == "H1_corrupted")

        sprt_results[f"tau={tau:.8f}"] = {
            "threshold": float(tau),
            "clean_correct": n_clean_correct,
            "clean_total": len(clean_decisions),
            "ood_correct": n_ood_correct,
            "ood_total": len(ood_decisions),
            "frames_to_decide": 1,  # SPRT decides in 1 frame (single observation)
        }
        print(f"  tau={tau:.2e}: clean {n_clean_correct}/{len(clean_decisions)}, "
              f"ood {n_ood_correct}/{len(ood_decisions)}")

    results["sprt"] = sprt_results

    # Part 6: PAC bounds
    print("\n=== Part 6: PAC Bounds ===")
    # PAC learning bound: P(error > eps) <= delta
    # For empirical risk minimizer with 0 errors on n samples:
    # eps <= ln(1/delta) / n

    pac_bounds = {}
    for delta in [0.10, 0.05, 0.01, 0.001]:
        eps_clean = np.log(1 / delta) / n_clean
        eps_ood = np.log(1 / delta) / n_ood
        eps_total = np.log(1 / delta) / n_total

        pac_bounds[f"delta={delta}"] = {
            "delta": delta,
            "confidence": 1 - delta,
            "fp_rate_upper": float(eps_clean),
            "fn_rate_upper": float(eps_ood),
            "total_error_upper": float(eps_total),
        }
        print(f"  delta={delta}: FPR<{eps_clean:.4f}, FNR<{eps_ood:.4f}, total<{eps_total:.4f}")

    results["pac_bounds"] = pac_bounds

    # Part 7: Separation margin statistics
    print("\n=== Part 7: Separation Margin Analysis ===")
    # For each corruption and severity, compute margin over threshold=0
    margin_analysis = {}

    for c in corruptions:
        dists = ood_distances[c]
        margin_analysis[c] = {
            "min_distance": float(min(dists)),
            "min_severity_tested": float(severities[0]),
            "margin_over_clean_max": float(min(dists) - 0.0),  # clean max is 0
            "log10_min_distance": float(np.log10(min(dists))) if min(dists) > 0 else float('-inf'),
            "distance_at_5pct": float(dists[0]),
            "distance_at_10pct": float(dists[1]),
            "distance_at_50pct": float(dists[9]),
            "distance_at_100pct": float(dists[-1]),
        }
        print(f"  {c}: min_d={min(dists):.2e}, d@5%={dists[0]:.2e}, d@100%={dists[-1]:.2e}")

    results["margins"] = margin_analysis

    # Part 8: Multi-seed robustness (5 different random images)
    print("\n=== Part 8: Multi-Seed Robustness ===")
    seed_results = []

    for seed in [0, 13, 42, 99, 777]:
        np.random.seed(seed)
        px = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(px)

        ref_emb = extract_hidden(model, processor, img, prompt)

        # Check clean stability
        ref_emb2 = extract_hidden(model, processor, img, prompt)
        clean_d = float(cosine(ref_emb, ref_emb2))

        # Check minimum OOD distance across corruptions
        min_ood = float('inf')
        for c in corruptions:
            corrupted = apply_corruption(img, c, 0.05)
            emb = extract_hidden(model, processor, corrupted, prompt)
            d = float(cosine(ref_emb, emb))
            min_ood = min(min_ood, d)

        seed_results.append({
            "seed": seed,
            "clean_distance": clean_d,
            "min_ood_distance_5pct": float(min_ood),
            "separable": clean_d == 0.0 and min_ood > 0,
        })
        print(f"  seed={seed}: clean_d={clean_d:.10f}, min_ood@5%={min_ood:.2e}, "
              f"separable={clean_d == 0.0 and min_ood > 0}")

    results["multi_seed"] = seed_results

    # Part 9: Compute combined formal guarantee
    print("\n=== Part 9: Combined Safety Certificate ===")
    all_separable = all(s["separable"] for s in seed_results)
    perfect_clean = results["clean_stability"]["all_distances_zero"]
    all_ood_detected = all(d > 0 for d in all_ood_dists)

    # Worst-case bounds across all tests
    worst_case = {
        "n_clean_tests": n_clean + 5,  # 30 clean passes + 5 seed tests
        "n_ood_tests": n_ood,  # 80 OOD tests
        "all_clean_zero": perfect_clean and all_separable,
        "all_ood_positive": all_ood_detected,
        "min_ood_distance_any": float(min(all_ood_dists)),
        "max_clean_distance_any": float(max(clean_distances)),
        "perfect_separation": perfect_clean and all_ood_detected,
        # With n=110 tests and 0 errors:
        # At 99% confidence, true error < 4.18%
        # At 95% confidence, true error < 2.72%
        "pac_99pct_error_bound": float(np.log(1/0.01) / n_total),
        "pac_95pct_error_bound": float(np.log(1/0.05) / n_total),
    }

    results["safety_certificate"] = worst_case
    print(f"  Perfect separation: {worst_case['perfect_separation']}")
    print(f"  Min OOD distance: {worst_case['min_ood_distance_any']:.2e}")
    print(f"  PAC 99% error bound: {worst_case['pac_99pct_error_bound']:.4f}")
    print(f"  PAC 95% error bound: {worst_case['pac_95pct_error_bound']:.4f}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/formal_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
