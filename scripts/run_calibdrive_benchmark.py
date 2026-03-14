"""
Run Full CalibDrive Benchmark.

Generates comprehensive results across all scenarios and UQ methods
using the realistic VLA simulator. Produces paper-ready analysis.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.calibration.realistic_simulator import (
    RealisticVLASimulator,
    SCENARIO_CONFIGS,
)
from src.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    failure_detection_metrics,
    sparsification_error,
    trajectory_l2_error,
)
from src.calibration.methods import SelectivePredictor


def evaluate_calibration(confidences, errors, error_threshold=2.0, num_bins=15):
    """Compute all calibration metrics."""
    accuracies = (errors < error_threshold).astype(float)
    cal = expected_calibration_error(confidences, accuracies, num_bins)
    bs = brier_score(confidences, accuracies)
    failure = failure_detection_metrics(
        1 - confidences, errors, error_threshold
    )
    sparse = sparsification_error(1 - confidences, errors)
    sp = SelectivePredictor()
    sp_results = sp.evaluate(confidences, errors, error_threshold)

    return {
        "ece": cal["ece"],
        "mce": cal["mce"],
        "brier": bs,
        "auroc": failure["auroc"],
        "auprc": failure["auprc"],
        "ause": sparse["ause"],
        "failure_rate": failure["failure_rate"],
        "reliability_bins": cal["bins"],
        "coverage_curve": sp_results["coverage_curve"],
    }


def run_benchmark():
    """Run the full CalibDrive benchmark."""
    print("=" * 70)
    print("CalibDrive Benchmark: Uncertainty-Aware VLA Evaluation")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    sim = RealisticVLASimulator(seed=42, model_quality="medium")
    benchmark_data = sim.generate_full_benchmark(samples_per_scenario=500)

    combined = benchmark_data["combined"]
    per_scenario = benchmark_data["per_scenario"]

    all_results = {}

    # ==========================================
    # 1. Baseline Calibration (no UQ)
    # ==========================================
    print("\n--- Phase 1: Baseline Calibration Assessment ---")

    # Overall baseline
    baseline_metrics = evaluate_calibration(
        combined["confidences"], combined["errors"]
    )
    all_results["baseline_overall"] = baseline_metrics
    print(f"\nOverall Baseline:")
    print(f"  ECE:    {baseline_metrics['ece']:.4f}")
    print(f"  Brier:  {baseline_metrics['brier']:.4f}")
    print(f"  AUROC:  {baseline_metrics['auroc']:.4f}")
    print(f"  Failure Rate: {baseline_metrics['failure_rate']:.4f}")

    # Per-scenario baseline
    print(f"\nPer-Scenario Baseline:")
    print(f"{'Scenario':<25} {'ECE':>8} {'Brier':>8} {'AUROC':>8} {'FailRate':>10}")
    print("-" * 65)

    scenario_results = {}
    for scenario_name, data in per_scenario.items():
        metrics = evaluate_calibration(data["confidences"], data["errors"])
        scenario_results[scenario_name] = metrics
        config = SCENARIO_CONFIGS[scenario_name]
        print(
            f"{scenario_name:<25} "
            f"{metrics['ece']:>8.4f} "
            f"{metrics['brier']:>8.4f} "
            f"{metrics['auroc']:>8.4f} "
            f"{metrics['failure_rate']:>10.4f}"
        )
    all_results["baseline_per_scenario"] = scenario_results

    # Per-difficulty baseline
    print(f"\nPer-Difficulty Baseline:")
    for difficulty in ["easy", "medium", "hard"]:
        mask = combined["difficulties"] == difficulty
        if mask.sum() == 0:
            continue
        metrics = evaluate_calibration(
            combined["confidences"][mask], combined["errors"][mask]
        )
        all_results[f"baseline_{difficulty}"] = metrics
        print(
            f"  {difficulty:<10}: ECE={metrics['ece']:.4f}, "
            f"Brier={metrics['brier']:.4f}, "
            f"AUROC={metrics['auroc']:.4f}"
        )

    # ==========================================
    # 2. UQ Methods Comparison
    # ==========================================
    print("\n--- Phase 2: UQ Methods Comparison ---")

    methods_results = {}

    # MC Dropout
    print("\nApplying MC Dropout (N=20)...")
    mc_result = sim.apply_mc_dropout(
        combined["predictions"], num_samples=20
    )
    mc_metrics = evaluate_calibration(
        mc_result["confidences"], combined["errors"]
    )
    methods_results["mc_dropout"] = mc_metrics
    print(f"  ECE: {mc_metrics['ece']:.4f} (baseline: {baseline_metrics['ece']:.4f})")

    # Deep Ensemble
    print("\nApplying Deep Ensemble (M=5)...")
    ensemble_result = sim.apply_ensemble(
        combined["predictions"], combined["ground_truth"], num_models=5
    )
    ensemble_metrics = evaluate_calibration(
        ensemble_result["confidences"], combined["errors"]
    )
    methods_results["deep_ensemble"] = ensemble_metrics
    print(f"  ECE: {ensemble_metrics['ece']:.4f} (baseline: {baseline_metrics['ece']:.4f})")

    # Temperature Scaling
    print("\nApplying Temperature Scaling...")
    ts_result = sim.apply_temperature_scaling(
        combined["confidences"], combined["errors"]
    )
    ts_metrics = evaluate_calibration(
        ts_result["calibrated_confidences"], combined["errors"]
    )
    methods_results["temperature_scaling"] = ts_metrics
    print(
        f"  ECE: {ts_metrics['ece']:.4f} (baseline: {baseline_metrics['ece']:.4f}), "
        f"T={ts_result['temperature']:.3f}"
    )

    # Conformal Prediction
    print("\nApplying Conformal Prediction (alpha=0.1)...")
    cp_result = sim.apply_conformal_prediction(
        combined["predictions"], combined["ground_truth"], alpha=0.1
    )
    cp_metrics = evaluate_calibration(
        cp_result["confidences"], combined["errors"]
    )
    methods_results["conformal"] = cp_metrics
    print(
        f"  ECE: {cp_metrics['ece']:.4f} (baseline: {baseline_metrics['ece']:.4f}), "
        f"Coverage: {cp_result['empirical_coverage']:.3f}"
    )

    all_results["methods_comparison"] = methods_results

    # ==========================================
    # 3. Method Comparison Table
    # ==========================================
    print("\n--- Phase 3: Method Comparison Summary ---")
    print(f"\n{'Method':<25} {'ECE':>8} {'Brier':>8} {'AUROC':>8} {'AUSE':>8} {'ECE Δ':>8}")
    print("=" * 70)

    baseline_ece = baseline_metrics["ece"]
    for method_name, metrics in [
        ("Baseline (no UQ)", baseline_metrics),
        ("MC Dropout (N=20)", mc_metrics),
        ("Deep Ensemble (M=5)", ensemble_metrics),
        ("Temperature Scaling", ts_metrics),
        ("Conformal (α=0.1)", cp_metrics),
    ]:
        ece_delta = metrics["ece"] - baseline_ece
        print(
            f"{method_name:<25} "
            f"{metrics['ece']:>8.4f} "
            f"{metrics['brier']:>8.4f} "
            f"{metrics['auroc']:>8.4f} "
            f"{metrics['ause']:>8.4f} "
            f"{ece_delta:>+8.4f}"
        )

    # ==========================================
    # 4. Selective Prediction Analysis
    # ==========================================
    print("\n--- Phase 4: Selective Prediction Analysis ---")

    # Compare selective prediction across methods
    method_confidences = {
        "Baseline": combined["confidences"],
        "MC Dropout": mc_result["confidences"],
        "Deep Ensemble": ensemble_result["confidences"],
        "Temp Scaling": ts_result["calibrated_confidences"],
        "Conformal": cp_result["confidences"],
    }

    selective_results = {}
    for method_name, confs in method_confidences.items():
        sp = SelectivePredictor()
        sp_res = sp.evaluate(confs, combined["errors"], 2.0)
        selective_results[method_name] = sp_res

        # Print key points on the coverage curve
        print(f"\n  {method_name} — AUROC: {sp_res['auroc_failure_detection']:.4f}")

        # Find failure rate at coverage 90%, 80%, 70%, 50%
        target_coverages = [0.9, 0.8, 0.7, 0.5]
        for target in target_coverages:
            best_point = min(
                sp_res["coverage_curve"],
                key=lambda p: abs(p["coverage"] - target)
            )
            if best_point["coverage"] > 0:
                print(
                    f"    Coverage={best_point['coverage']:.2f}: "
                    f"FailRate={best_point['selective_failure_rate']:.4f}, "
                    f"SelError={best_point['selective_error']:.3f}"
                )

    # Summary: collision rate reduction at 80% coverage
    print(f"\n  Collision Rate Reduction at ~80% Coverage:")
    baseline_full_failure = combined["errors"][combined["errors"] > 2.0].shape[0] / len(combined["errors"])
    print(f"    Full coverage failure rate: {baseline_full_failure:.4f}")
    for method_name, sp_res in selective_results.items():
        point_80 = min(
            sp_res["coverage_curve"],
            key=lambda p: abs(p["coverage"] - 0.8)
        )
        if point_80["coverage"] > 0:
            reduction = (1 - point_80["selective_failure_rate"] / max(baseline_full_failure, 1e-8)) * 100
            print(
                f"    {method_name:<20}: FailRate={point_80['selective_failure_rate']:.4f} "
                f"(Reduction: {reduction:+.1f}%)"
            )

    all_results["selective_prediction"] = {
        method: {
            "auroc": res["auroc_failure_detection"],
            "coverage_curve": res["coverage_curve"],
        }
        for method, res in selective_results.items()
    }

    # ==========================================
    # 5. Save Results
    # ==========================================
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"calibdrive_benchmark_{timestamp}.json"

    # Deep convert
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        return convert_numpy(obj)

    with open(results_path, "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Full results saved to {results_path}")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    run_benchmark()
