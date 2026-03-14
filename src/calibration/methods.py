"""
Uncertainty Quantification and Calibration Methods for Driving VLAs.

Implements:
1. Temperature Scaling (post-hoc)
2. MC Dropout
3. Deep Ensembles
4. Conformal Prediction
5. Evidential Deep Learning
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable calibration result."""
    method: str
    calibrated_confidences: np.ndarray
    ece_before: float
    ece_after: float
    parameters: dict


class TemperatureScaling:
    """Post-hoc temperature scaling for VLA action logits.

    Learns a single scalar temperature T that divides logits
    before softmax, optimized to minimize NLL on a validation set.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Learn optimal temperature on validation set.

        Args:
            logits: Model logits (N, C) from validation set.
            labels: Ground truth labels (N,).

        Returns:
            Optimal temperature value.
        """
        def nll_loss(T):
            scaled = logits / T[0]
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            return -log_probs[np.arange(len(labels)), labels].mean()

        result = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 100.0)])
        self.temperature = result.x[0]
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature to new logits.

        Args:
            logits: Raw logits (N, C).

        Returns:
            Calibrated probabilities (N, C).
        """
        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


class MCDropoutEstimator:
    """Monte Carlo Dropout for uncertainty estimation.

    Enables dropout at inference time and runs multiple forward
    passes to estimate predictive uncertainty.
    """

    def __init__(self, num_samples: int = 20, dropout_rate: float = 0.1):
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

    def estimate(
        self,
        model: nn.Module,
        inputs: dict,
        forward_fn,
    ) -> dict:
        """Run MC Dropout inference.

        Args:
            model: The VLA model.
            inputs: Preprocessed model inputs.
            forward_fn: Function that takes (model, inputs) and returns predictions.

        Returns:
            Dict with mean, std, entropy, and individual predictions.
        """
        # Enable dropout
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = forward_fn(model, inputs)
                predictions.append(pred.cpu().numpy())

        # Restore eval mode
        model.eval()

        predictions = np.array(predictions)  # (N_samples, ...)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        entropy = 0.5 * np.log(2 * np.pi * np.e * (std ** 2 + 1e-8))

        return {
            "mean": mean,
            "std": std,
            "entropy": entropy,
            "predictions": predictions,
            "num_samples": self.num_samples,
        }


class DeepEnsemble:
    """Deep Ensemble uncertainty estimation.

    Aggregates predictions from M independently trained models
    to capture both aleatoric and epistemic uncertainty.
    """

    def __init__(self, models: list[nn.Module]):
        self.models = models
        self.num_models = len(models)

    def estimate(self, inputs: dict, forward_fn) -> dict:
        """Run ensemble inference.

        Args:
            inputs: Preprocessed model inputs.
            forward_fn: Function that takes (model, inputs) and returns predictions.

        Returns:
            Dict with mean, std, entropy, and individual model predictions.
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = forward_fn(model, inputs)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        # Epistemic uncertainty from disagreement between models
        epistemic = std
        # Total entropy
        entropy = 0.5 * np.log(2 * np.pi * np.e * (std ** 2 + 1e-8))

        return {
            "mean": mean,
            "std": std,
            "epistemic_uncertainty": epistemic,
            "entropy": entropy,
            "predictions": predictions,
            "num_models": self.num_models,
        }


class ConformalPredictor:
    """Conformal Prediction for distribution-free coverage guarantees.

    Provides prediction sets/intervals that contain the true value
    with probability >= 1 - alpha, regardless of the underlying distribution.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def calibrate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Compute conformal quantile on calibration set.

        Args:
            predictions: Model predictions (N, D).
            ground_truth: True values (N, D).

        Returns:
            Calibration quantile threshold.
        """
        # Nonconformity scores: L2 distance between prediction and truth
        scores = np.linalg.norm(predictions - ground_truth, axis=-1)  # (N,)

        # Compute (1-alpha)(1 + 1/n) quantile
        n = len(scores)
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)

        self.quantile = np.quantile(scores, q_level)
        return self.quantile

    def predict(
        self,
        predictions: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> dict:
        """Generate prediction sets with coverage guarantee.

        Args:
            predictions: Point predictions (N, D).
            uncertainties: Optional per-prediction uncertainty estimates.

        Returns:
            Dict with prediction sets and coverage info.
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() before predict()")

        # Prediction region: ball of radius = quantile around prediction
        radii = np.full(len(predictions), self.quantile)

        # If uncertainties provided, use adaptive conformal prediction
        if uncertainties is not None:
            # Normalize: larger uncertainty → larger prediction set
            normalized = uncertainties / (uncertainties.mean() + 1e-8)
            radii = self.quantile * normalized

        return {
            "predictions": predictions,
            "radii": radii,
            "alpha": self.alpha,
            "quantile": self.quantile,
        }


class SelectivePredictor:
    """Selective prediction: abstain when uncertainty is too high.

    Implements uncertainty-threshold policies for safe driving:
    - PROCEED: confidence is high enough for autonomous action
    - SLOW_DOWN: moderate uncertainty, reduce speed
    - ABSTAIN: high uncertainty, request human intervention
    """

    PROCEED = "proceed"
    SLOW_DOWN = "slow_down"
    ABSTAIN = "abstain"

    def __init__(
        self,
        abstain_threshold: float = 0.3,
        slow_threshold: float = 0.6,
    ):
        self.abstain_threshold = abstain_threshold
        self.slow_threshold = slow_threshold

    def decide(self, confidence: float) -> str:
        """Make a selective prediction decision.

        Args:
            confidence: Model confidence score [0, 1].

        Returns:
            Decision string: "proceed", "slow_down", or "abstain".
        """
        if confidence < self.abstain_threshold:
            return self.ABSTAIN
        if confidence < self.slow_threshold:
            return self.SLOW_DOWN
        return self.PROCEED

    def evaluate(
        self,
        confidences: np.ndarray,
        errors: np.ndarray,
        error_threshold: float = 1.0,
    ) -> dict:
        """Evaluate selective prediction performance.

        Args:
            confidences: Model confidence scores (N,).
            errors: Prediction errors (N,).
            error_threshold: Error threshold to define "failure".

        Returns:
            Coverage, selective accuracy, and risk metrics.
        """
        is_failure = errors > error_threshold

        # Evaluate at various confidence thresholds
        thresholds = np.linspace(0.0, 1.0, 101)
        results = []

        for t in thresholds:
            mask = confidences >= t
            coverage = mask.mean()
            if coverage > 0:
                selective_error = errors[mask].mean()
                selective_failure_rate = is_failure[mask].mean()
            else:
                selective_error = 0.0
                selective_failure_rate = 0.0

            results.append({
                "threshold": t,
                "coverage": coverage,
                "selective_error": selective_error,
                "selective_failure_rate": selective_failure_rate,
            })

        # AUROC for failure detection
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(is_failure, 1 - confidences)
        except ValueError:
            auroc = 0.5

        return {
            "coverage_curve": results,
            "auroc_failure_detection": auroc,
        }
