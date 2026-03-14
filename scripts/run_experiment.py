"""
Experiment Runner for CalibDrive.

Orchestrates the experiment loop:
1. Load VLA model and dataset
2. Run inference with uncertainty quantification
3. Evaluate calibration and driving metrics
4. Log results to experiments/results.tsv
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_git_hash() -> str:
    """Get current short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def log_result(
    results_path: str,
    commit: str,
    ece: float,
    brier: float,
    auroc: float,
    collision_rate: float,
    status: str,
    description: str,
):
    """Append experiment result to results.tsv."""
    with open(results_path, "a") as f:
        f.write(
            f"{commit}\t{ece:.6f}\t{brier:.6f}\t{auroc:.6f}\t"
            f"{collision_rate:.6f}\t{status}\t{description}\n"
        )


def run_calibration_experiment(config: dict) -> dict:
    """Run a single calibration experiment.

    Args:
        config: Experiment configuration dict.

    Returns:
        Dict with all metrics.
    """
    from src.evaluation.metrics import (
        expected_calibration_error,
        brier_score,
        failure_detection_metrics,
        trajectory_l2_error,
        sparsification_error,
    )

    # Seed for reproducibility
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_name = config["model"]["name"]
    uq_method = config["uncertainty"]["method"]
    dataset_name = config["dataset"]["name"]

    print(f"[CalibDrive] Model: {model_name}")
    print(f"[CalibDrive] UQ Method: {uq_method}")
    print(f"[CalibDrive] Dataset: {dataset_name}")
    print(f"[CalibDrive] Running experiment...")

    # -- Placeholder for actual model loading and inference --
    # In a full implementation, this would:
    # 1. Load the VLA model
    # 2. Load the driving dataset
    # 3. Run inference with the specified UQ method
    # 4. Compute all metrics

    # For now, generate synthetic data to validate the pipeline
    N = config.get("num_samples", 1000)
    T = config.get("prediction_horizon", 10)

    # Simulate predictions and ground truth
    gt_trajectories = np.random.randn(N, T, 2) * 5
    noise_scale = config.get("noise_scale", 1.0)
    pred_trajectories = gt_trajectories + np.random.randn(N, T, 2) * noise_scale

    # Simulate uncertainties (correlated with error for a good estimator)
    errors = np.linalg.norm(
        pred_trajectories - gt_trajectories, axis=(1, 2)
    )  # (N,)
    uncertainties = errors + np.random.randn(N) * 0.5  # noisy but correlated
    uncertainties = np.clip(uncertainties, 0, None)

    # Simulate confidences (inverse of uncertainty, normalized)
    confidences = 1.0 / (1.0 + uncertainties)

    # Binary accuracy (within threshold)
    ade_per_sample = np.linalg.norm(
        pred_trajectories - gt_trajectories, axis=-1
    ).mean(axis=1)
    accuracy_threshold = config.get("accuracy_threshold", 2.0)
    accuracies = (ade_per_sample < accuracy_threshold).astype(float)

    # Compute metrics
    cal_result = expected_calibration_error(confidences, accuracies)
    bs = brier_score(confidences, accuracies)
    traj_metrics = trajectory_l2_error(pred_trajectories, gt_trajectories)
    failure_metrics = failure_detection_metrics(
        uncertainties, ade_per_sample, error_threshold=accuracy_threshold
    )
    sparse_metrics = sparsification_error(uncertainties, ade_per_sample)

    results = {
        "ece": cal_result["ece"],
        "mce": cal_result["mce"],
        "brier_score": bs,
        "ade": traj_metrics["ade"],
        "fde": traj_metrics["fde"],
        "auroc": failure_metrics["auroc"],
        "auprc": failure_metrics["auprc"],
        "ause": sparse_metrics["ause"],
        "failure_rate": failure_metrics["failure_rate"],
        "reliability_bins": cal_result["bins"],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="CalibDrive Experiment Runner")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--description", type=str, default="",
        help="Short description of this experiment",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    commit = get_git_hash()

    print(f"{'=' * 60}")
    print(f"CalibDrive Experiment")
    print(f"Commit: {commit}")
    print(f"Config: {args.config}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'=' * 60}")

    start = time.time()
    results = run_calibration_experiment(config)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  ECE:            {results['ece']:.6f}")
    print(f"  Brier Score:    {results['brier_score']:.6f}")
    print(f"  AUROC:          {results['auroc']:.6f}")
    print(f"  ADE:            {results['ade']:.4f}")
    print(f"  FDE:            {results['fde']:.4f}")
    print(f"  AUSE:           {results['ause']:.6f}")
    print(f"  Failure Rate:   {results['failure_rate']:.4f}")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Save detailed results
    results_dir = Path("experiments")
    results_dir.mkdir(exist_ok=True)

    detail_path = results_dir / f"detail_{commit}_{int(time.time())}.json"
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log to results.tsv
    description = args.description or config.get("description", "experiment")
    log_result(
        str(results_dir / "results.tsv"),
        commit=commit,
        ece=results["ece"],
        brier=results["brier_score"],
        auroc=results["auroc"],
        collision_rate=results.get("collision_rate", 0.0),
        status="keep",
        description=description,
    )

    print(f"\nResults saved to {detail_path}")
    print(f"Logged to experiments/results.tsv")


if __name__ == "__main__":
    main()
