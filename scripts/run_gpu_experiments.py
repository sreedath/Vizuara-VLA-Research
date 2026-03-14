"""
GPU Experiment Runner for CalibDrive.

Runs the full CalibDrive experiment suite on GPU with real VLA models.
Designed for RunPod or similar GPU instances.

Usage:
    PYTHONPATH=. python3 scripts/run_gpu_experiments.py \
        --model openvla/openvla-7b \
        --dataset synthetic \
        --methods all
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEM = torch.cuda.get_device_properties(0).total_mem / 1e9
    else:
        GPU_NAME = "None"
        GPU_MEM = 0
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "None"
    GPU_MEM = 0


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def log_to_tsv(results_path, commit, ece, brier, auroc, collision_rate, status, description):
    with open(results_path, "a") as f:
        f.write(
            f"{commit}\t{ece:.6f}\t{brier:.6f}\t{auroc:.6f}\t"
            f"{collision_rate:.6f}\t{status}\t{description}\n"
        )


def run_with_synthetic_data(methods, num_samples=4000):
    """Run experiments with realistic synthetic data."""
    from src.calibration.realistic_simulator import RealisticVLASimulator
    from src.evaluation.metrics import (
        expected_calibration_error,
        brier_score,
        failure_detection_metrics,
        sparsification_error,
    )
    from src.calibration.methods import SelectivePredictor

    print(f"\nUsing realistic synthetic data (N={num_samples})")

    sim = RealisticVLASimulator(seed=42, model_quality="medium")
    benchmark = sim.generate_full_benchmark(samples_per_scenario=num_samples // 8)
    combined = benchmark["combined"]

    results = {}
    commit = get_git_hash()
    tsv_path = "experiments/results.tsv"

    # Method runners
    method_runners = {
        "baseline": lambda: {
            "confidences": combined["confidences"],
            "description": "Baseline (no UQ)",
        },
        "mc_dropout": lambda: {
            **sim.apply_mc_dropout(combined["predictions"], num_samples=20),
            "description": "MC Dropout (N=20)",
        },
        "mc_dropout_50": lambda: {
            **sim.apply_mc_dropout(combined["predictions"], num_samples=50),
            "description": "MC Dropout (N=50)",
        },
        "ensemble_3": lambda: {
            **sim.apply_ensemble(
                combined["predictions"], combined["ground_truth"], num_models=3
            ),
            "description": "Deep Ensemble (M=3)",
        },
        "ensemble_5": lambda: {
            **sim.apply_ensemble(
                combined["predictions"], combined["ground_truth"], num_models=5
            ),
            "description": "Deep Ensemble (M=5)",
        },
        "ensemble_7": lambda: {
            **sim.apply_ensemble(
                combined["predictions"], combined["ground_truth"], num_models=7
            ),
            "description": "Deep Ensemble (M=7)",
        },
        "temp_scaling": lambda: {
            **sim.apply_temperature_scaling(combined["confidences"], combined["errors"]),
            "description": "Temperature Scaling",
        },
        "conformal_05": lambda: {
            **sim.apply_conformal_prediction(
                combined["predictions"], combined["ground_truth"], alpha=0.05
            ),
            "description": "Conformal (alpha=0.05)",
        },
        "conformal_10": lambda: {
            **sim.apply_conformal_prediction(
                combined["predictions"], combined["ground_truth"], alpha=0.10
            ),
            "description": "Conformal (alpha=0.10)",
        },
        "conformal_20": lambda: {
            **sim.apply_conformal_prediction(
                combined["predictions"], combined["ground_truth"], alpha=0.20
            ),
            "description": "Conformal (alpha=0.20)",
        },
    }

    # Filter to requested methods
    if methods != "all":
        requested = methods.split(",")
        method_runners = {
            k: v for k, v in method_runners.items() if k in requested
        }

    for method_name, runner in method_runners.items():
        print(f"\n{'=' * 50}")
        print(f"Running: {method_name}")
        print(f"{'=' * 50}")

        start = time.time()
        result = runner()
        elapsed = time.time() - start

        # Get confidences
        confs = result.get(
            "calibrated_confidences",
            result.get("confidences", combined["confidences"])
        )

        # Evaluate
        accuracies = (combined["errors"] < 2.0).astype(float)
        cal = expected_calibration_error(confs, accuracies)
        bs = brier_score(confs, accuracies)
        failure = failure_detection_metrics(1 - confs, combined["errors"], 2.0)
        sparse = sparsification_error(1 - confs, combined["errors"])

        # Selective prediction at 80% coverage
        sp = SelectivePredictor()
        sp_res = sp.evaluate(confs, combined["errors"], 2.0)
        point_80 = min(
            sp_res["coverage_curve"],
            key=lambda p: abs(p["coverage"] - 0.8)
        )

        metrics = {
            "method": method_name,
            "description": result["description"],
            "ece": cal["ece"],
            "mce": cal["mce"],
            "brier": bs,
            "auroc": failure["auroc"],
            "auprc": failure["auprc"],
            "ause": sparse["ause"],
            "failure_rate": failure["failure_rate"],
            "selective_failure_80": point_80["selective_failure_rate"],
            "elapsed": elapsed,
        }
        results[method_name] = metrics

        print(f"  ECE:    {metrics['ece']:.4f}")
        print(f"  Brier:  {metrics['brier']:.4f}")
        print(f"  AUROC:  {metrics['auroc']:.4f}")
        print(f"  AUSE:   {metrics['ause']:.4f}")
        print(f"  SelFail@80%: {metrics['selective_failure_80']:.4f}")
        print(f"  Time:   {elapsed:.1f}s")

        # Log to TSV
        status = "keep"
        log_to_tsv(
            tsv_path, commit, metrics["ece"], metrics["brier"],
            metrics["auroc"], metrics["selective_failure_80"],
            status, result["description"],
        )

    return results


def run_with_openvla(model_name, num_samples=100):
    """Run experiments with real OpenVLA model on GPU."""
    from src.models.openvla_driver import OpenVLADriver

    print(f"\nLoading {model_name}...")
    driver = OpenVLADriver(model_name=model_name, device="cuda")

    # Generate synthetic driving images for now
    # TODO: Replace with NAVSIM data loader
    rng = np.random.RandomState(42)

    predictions = []
    confidences = []
    mc_results = []

    for i in range(num_samples):
        # Synthetic image
        image = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Standard prediction
        pred = driver.predict(image, return_logits=True)
        predictions.append(pred.trajectory)
        confidences.append(pred.confidence)

        # MC Dropout (every 10th sample for speed)
        if i % 10 == 0:
            mc = driver.predict_with_mc_dropout(image, num_samples=10)
            mc_results.append(mc)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples")

    return {
        "predictions": np.array(predictions),
        "confidences": np.array(confidences),
        "mc_results": mc_results,
    }


def main():
    parser = argparse.ArgumentParser(description="CalibDrive GPU Experiments")
    parser.add_argument(
        "--model", type=str, default="openvla/openvla-7b",
        help="Model name",
    )
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "navsim"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--methods", type=str, default="all",
        help="Methods to run (comma-separated or 'all')",
    )
    parser.add_argument(
        "--num-samples", type=int, default=4000,
        help="Number of samples",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CalibDrive GPU Experiment Runner")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPU: {GPU_NAME} ({GPU_MEM:.1f} GB)" if GPU_AVAILABLE else "GPU: Not available")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Methods: {args.methods}")
    print("=" * 60)

    if args.dataset == "synthetic":
        results = run_with_synthetic_data(args.methods, args.num_samples)
    elif GPU_AVAILABLE:
        results = run_with_openvla(args.model, args.num_samples)
    else:
        print("No GPU available. Falling back to synthetic data.")
        results = run_with_synthetic_data(args.methods, args.num_samples)

    # Save results
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"gpu_experiment_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(
            {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in results.items()},
            f, indent=2,
        )

    print(f"\nResults saved to {output_path}")

    # Print final comparison
    print(f"\n{'=' * 80}")
    print(f"{'Method':<25} {'ECE':>8} {'Brier':>8} {'AUROC':>8} {'SelFail@80':>12}")
    print(f"{'=' * 80}")
    for name, m in results.items():
        print(
            f"{m['description']:<25} "
            f"{m['ece']:>8.4f} "
            f"{m['brier']:>8.4f} "
            f"{m['auroc']:>8.4f} "
            f"{m['selective_failure_80']:>12.4f}"
        )
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
