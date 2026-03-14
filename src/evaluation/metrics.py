"""
Calibration and Driving Performance Metrics.

Provides comprehensive evaluation for uncertainty-aware VLA models:
- Calibration metrics (ECE, MCE, Brier Score, NLL)
- Driving performance metrics (L2 error, collision rate, PDMS)
- Uncertainty quality metrics (AUROC, AUPRC, sparsification)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
) -> dict:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidences: Model confidence scores (N,).
        accuracies: Binary correctness indicators (N,).
        num_bins: Number of bins for calibration.

    Returns:
        Dict with ECE, MCE, per-bin statistics, and reliability diagram data.
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ece = 0.0
    bin_mce = 0.0
    bins_data = []

    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        bin_size = mask.sum()

        if bin_size == 0:
            bins_data.append({
                "bin_lo": lo, "bin_hi": hi,
                "count": 0, "avg_conf": 0.0, "avg_acc": 0.0, "gap": 0.0,
            })
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        gap = abs(avg_acc - avg_conf)

        bin_ece += (bin_size / len(confidences)) * gap
        bin_mce = max(bin_mce, gap)

        bins_data.append({
            "bin_lo": lo, "bin_hi": hi,
            "count": int(bin_size), "avg_conf": float(avg_conf),
            "avg_acc": float(avg_acc), "gap": float(gap),
        })

    return {
        "ece": float(bin_ece),
        "mce": float(bin_mce),
        "num_bins": num_bins,
        "bins": bins_data,
    }


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier Score (mean squared error of probability estimates).

    Args:
        probabilities: Predicted probabilities (N,) or (N, C).
        labels: True labels (N,) as indices or one-hot.

    Returns:
        Brier score (lower is better).
    """
    if probabilities.ndim == 1:
        return float(np.mean((probabilities - labels) ** 2))

    # Multi-class: one-hot encode labels if needed
    if labels.ndim == 1:
        n_classes = probabilities.shape[1]
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(labels)), labels] = 1
        labels = one_hot

    return float(np.mean(np.sum((probabilities - labels) ** 2, axis=1)))


def negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Negative Log-Likelihood.

    Args:
        probabilities: Predicted probabilities (N,) or (N, C).
        labels: True labels (N,).

    Returns:
        Mean NLL (lower is better).
    """
    eps = 1e-10
    if probabilities.ndim == 1:
        clipped = np.clip(probabilities, eps, 1 - eps)
        nll = -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        return float(nll.mean())

    clipped = np.clip(probabilities, eps, 1.0)
    nll = -np.log(clipped[np.arange(len(labels)), labels])
    return float(nll.mean())


def trajectory_l2_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Compute L2 displacement error for trajectory predictions.

    Args:
        predicted: Predicted trajectories (N, T, 2).
        ground_truth: Ground truth trajectories (N, T, 2).

    Returns:
        Dict with ADE (average displacement error) and FDE (final displacement error).
    """
    # Per-timestep displacement
    displacement = np.linalg.norm(predicted - ground_truth, axis=-1)  # (N, T)

    ade = float(displacement.mean())
    fde = float(displacement[:, -1].mean())

    return {
        "ade": ade,
        "fde": fde,
        "per_timestep_error": displacement.mean(axis=0).tolist(),
    }


def collision_rate(
    predicted_trajectories: np.ndarray,
    obstacle_positions: np.ndarray,
    collision_radius: float = 2.0,
) -> dict:
    """Compute collision rate between predicted trajectories and obstacles.

    Args:
        predicted_trajectories: Predicted trajectories (N, T, 2).
        obstacle_positions: Obstacle positions (N, M, 2) where M is max obstacles.
        collision_radius: Distance threshold for collision detection.

    Returns:
        Dict with collision rate and per-scenario breakdown.
    """
    N, T, _ = predicted_trajectories.shape
    collisions = np.zeros(N, dtype=bool)

    for i in range(N):
        for t in range(T):
            pred_pos = predicted_trajectories[i, t]
            distances = np.linalg.norm(obstacle_positions[i] - pred_pos, axis=-1)
            if np.any(distances < collision_radius):
                collisions[i] = True
                break

    return {
        "collision_rate": float(collisions.mean()),
        "num_collisions": int(collisions.sum()),
        "total_scenarios": N,
    }


def failure_detection_metrics(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    error_threshold: float = 2.0,
) -> dict:
    """Evaluate how well uncertainty predicts failures.

    Args:
        uncertainties: Uncertainty estimates (N,).
        errors: Prediction errors (N,).
        error_threshold: Error threshold defining "failure".

    Returns:
        Dict with AUROC, AUPRC, and optimal threshold.
    """
    is_failure = (errors > error_threshold).astype(int)

    if is_failure.sum() == 0 or is_failure.sum() == len(is_failure):
        return {
            "auroc": 0.5,
            "auprc": float(is_failure.mean()),
            "failure_rate": float(is_failure.mean()),
        }

    auroc = roc_auc_score(is_failure, uncertainties)
    auprc = average_precision_score(is_failure, uncertainties)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "failure_rate": float(is_failure.mean()),
    }


def sparsification_error(
    uncertainties: np.ndarray,
    errors: np.ndarray,
) -> dict:
    """Compute sparsification error (oracle vs predicted ordering).

    Measures how well uncertainty ranking matches error ranking.
    A perfect uncertainty estimator would remove high-error samples first.

    Args:
        uncertainties: Uncertainty estimates (N,).
        errors: Prediction errors (N,).

    Returns:
        Dict with sparsification curve and AUSE (area under sparsification error).
    """
    n = len(uncertainties)
    fractions = np.linspace(0, 1, 20)

    # Oracle: remove by decreasing error
    oracle_order = np.argsort(-errors)
    # Predicted: remove by decreasing uncertainty
    pred_order = np.argsort(-uncertainties)

    oracle_curve = []
    pred_curve = []

    for frac in fractions:
        keep = max(1, int(n * (1 - frac)))

        oracle_remaining = oracle_order[:keep]
        oracle_err = errors[oracle_remaining].mean()
        oracle_curve.append(oracle_err)

        pred_remaining = pred_order[:keep]
        pred_err = errors[pred_remaining].mean()
        pred_curve.append(pred_err)

    oracle_curve = np.array(oracle_curve)
    pred_curve = np.array(pred_curve)
    sparsification_err = pred_curve - oracle_curve

    ause = float(np.trapz(sparsification_err, fractions))

    return {
        "ause": ause,
        "fractions": fractions.tolist(),
        "oracle_curve": oracle_curve.tolist(),
        "predicted_curve": pred_curve.tolist(),
    }
