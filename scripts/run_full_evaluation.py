"""
Run Full CalibDrive Evaluation.

Runs all UQ methods and generates comparison results.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.calibration.pipeline import CalibDrivePipeline, PipelineConfig


METHODS = [
    ("none", "Baseline (no UQ)"),
    ("mc_dropout", "MC Dropout (N=20)"),
    ("temperature_scaling", "Temperature Scaling"),
    ("conformal", "Conformal Prediction (alpha=0.1)"),
]


def run_all_methods():
    """Run evaluation for all UQ methods and compare."""
    results = []

    for method, description in METHODS:
        print(f"\n{'=' * 60}")
        print(f"Running: {description}")
        print(f"{'=' * 60}")

        config = PipelineConfig(
            model_name="openvla/openvla-7b",
            uq_method=method,
            dataset_name="synthetic",
            num_samples=2000,
            prediction_horizon=10,
            seed=42,
        )

        pipeline = CalibDrivePipeline(config)
        start = time.time()
        result = pipeline.run()
        elapsed = time.time() - start

        print(f"  ECE:    {result.ece:.4f}")
        print(f"  Brier:  {result.brier:.4f}")
        print(f"  AUROC:  {result.auroc:.4f}")
        print(f"  ADE:    {result.ade:.4f}")
        print(f"  FDE:    {result.fde:.4f}")
        print(f"  AUSE:   {result.ause:.4f}")
        print(f"  Time:   {elapsed:.1f}s")

        results.append({
            "method": method,
            "description": description,
            "ece": result.ece,
            "mce": result.mce,
            "brier": result.brier,
            "nll": result.nll,
            "auroc": result.auroc,
            "auprc": result.auprc,
            "ause": result.ause,
            "ade": result.ade,
            "fde": result.fde,
            "collision_rate": result.collision_rate_value,
            "elapsed_seconds": elapsed,
        })

    # Save comparison
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    comparison_path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print(f"\n{'=' * 80}")
    print(f"{'Method':<30} {'ECE':>8} {'Brier':>8} {'AUROC':>8} {'ADE':>8} {'AUSE':>8}")
    print(f"{'=' * 80}")
    for r in results:
        print(
            f"{r['description']:<30} "
            f"{r['ece']:>8.4f} "
            f"{r['brier']:>8.4f} "
            f"{r['auroc']:>8.4f} "
            f"{r['ade']:>8.4f} "
            f"{r['ause']:>8.4f}"
        )
    print(f"{'=' * 80}")
    print(f"\nResults saved to {comparison_path}")


if __name__ == "__main__":
    run_all_methods()
