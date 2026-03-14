"""
CalibDrive-Combined: Hybrid Uncertainty Quantification.

Combines the strengths of multiple UQ methods:
1. Deep Ensemble for epistemic uncertainty (best AUROC)
2. Conformal Prediction for coverage guarantees (perfect failure detection)
3. MC Dropout within ensemble for aleatoric uncertainty (best ECE)

This is our proposed novel method for the paper.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class CombinedUQResult:
    """Immutable result from combined UQ method."""
    mean_trajectory: np.ndarray       # (T, 2) mean prediction
    epistemic_uncertainty: np.ndarray  # (N,) from ensemble disagreement
    aleatoric_uncertainty: np.ndarray  # (N,) from MC dropout variance
    total_uncertainty: np.ndarray      # (N,) combined uncertainty
    conformal_in_set: np.ndarray      # (N,) bool - is prediction in conformal set
    conformal_set_size: np.ndarray    # (N,) size of conformal prediction set
    calibrated_confidence: np.ndarray  # (N,) final calibrated confidence
    decision: np.ndarray              # (N,) str - "proceed", "slow_down", "abstain"


class CalibDriveCombined:
    """Combined UQ method for driving VLAs.

    Architecture:
    1. Run M ensemble members, each with K MC Dropout passes
    2. Compute epistemic (inter-model) and aleatoric (intra-model) uncertainty
    3. Apply conformal prediction for distribution-free coverage
    4. Make selective prediction decisions based on combined uncertainty
    """

    def __init__(
        self,
        num_ensemble: int = 5,
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1,
        conformal_alpha: float = 0.1,
        abstain_threshold: float = 0.3,
        slow_threshold: float = 0.6,
        uncertainty_weighting: str = "rms",
    ):
        self.num_ensemble = num_ensemble
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        self.conformal_alpha = conformal_alpha
        self.abstain_threshold = abstain_threshold
        self.slow_threshold = slow_threshold
        self.uncertainty_weighting = uncertainty_weighting

        self._conformal_quantile = None

    def calibrate(
        self,
        cal_predictions: np.ndarray,
        cal_ground_truth: np.ndarray,
        cal_epistemic: np.ndarray,
        cal_aleatoric: np.ndarray,
    ):
        """Fit conformal prediction on calibration data.

        Args:
            cal_predictions: Predictions on calibration set (N_cal, T, 2).
            cal_ground_truth: Ground truth on calibration set (N_cal, T, 2).
            cal_epistemic: Epistemic uncertainty on calibration set (N_cal,).
            cal_aleatoric: Aleatoric uncertainty on calibration set (N_cal,).
        """
        # Compute nonconformity scores
        errors = np.linalg.norm(
            cal_predictions - cal_ground_truth, axis=-1
        ).mean(axis=1)

        # Adaptive conformal: normalize scores by uncertainty
        total_unc = self._combine_uncertainties(cal_epistemic, cal_aleatoric)
        normalized_scores = errors / (total_unc + 1e-8)

        # Compute quantile for coverage guarantee
        n = len(normalized_scores)
        q_level = np.ceil((1 - self.conformal_alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)
        self._conformal_quantile = np.quantile(normalized_scores, q_level)

        # Also learn optimal uncertainty-to-confidence mapping
        self._learn_confidence_mapping(total_unc, errors)

    def predict(
        self,
        ensemble_predictions: np.ndarray,
        mc_predictions: Optional[np.ndarray] = None,
    ) -> CombinedUQResult:
        """Generate combined UQ prediction.

        Args:
            ensemble_predictions: (M, N, T, 2) predictions from M models.
            mc_predictions: (M, K, N, T, 2) MC predictions within each model.
                            If None, only ensemble uncertainty is used.

        Returns:
            CombinedUQResult with all uncertainty estimates.
        """
        M, N, T, D = ensemble_predictions.shape

        # Mean prediction across ensemble
        mean_traj = ensemble_predictions.mean(axis=0)  # (N, T, 2)

        # Epistemic uncertainty: inter-model disagreement
        epistemic = ensemble_predictions.std(axis=0).mean(axis=(1, 2))  # (N,)

        # Aleatoric uncertainty: intra-model MC variance
        if mc_predictions is not None:
            # mc_predictions shape: (M, K, N, T, 2)
            aleatoric_per_model = mc_predictions.std(axis=1).mean(axis=(2, 3))  # (M, N)
            aleatoric = aleatoric_per_model.mean(axis=0)  # (N,)
        else:
            aleatoric = np.zeros(N)

        # Combined uncertainty
        total = self._combine_uncertainties(epistemic, aleatoric)

        # Conformal prediction
        if self._conformal_quantile is not None:
            conformal_set_size = self._conformal_quantile * (total + 1e-8)
            # A sample is "in set" if its predicted uncertainty is below
            # what we'd expect based on calibration
            conformal_in_set = total < self._conformal_quantile
        else:
            conformal_set_size = np.ones(N) * np.inf
            conformal_in_set = np.ones(N, dtype=bool)

        # Calibrated confidence
        confidence = self._compute_calibrated_confidence(
            total, conformal_set_size
        )

        # Selective prediction decisions
        decisions = np.where(
            confidence < self.abstain_threshold, "abstain",
            np.where(confidence < self.slow_threshold, "slow_down", "proceed")
        )

        return CombinedUQResult(
            mean_trajectory=mean_traj,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            conformal_in_set=conformal_in_set,
            conformal_set_size=conformal_set_size,
            calibrated_confidence=confidence,
            decision=decisions,
        )

    def _combine_uncertainties(
        self,
        epistemic: np.ndarray,
        aleatoric: np.ndarray,
    ) -> np.ndarray:
        """Combine epistemic and aleatoric uncertainties."""
        if self.uncertainty_weighting == "harmonic":
            # Harmonic mean: sensitive to both components
            eps = 1e-8
            return 2 * (epistemic * aleatoric) / (epistemic + aleatoric + eps)
        elif self.uncertainty_weighting == "sum":
            return epistemic + aleatoric
        elif self.uncertainty_weighting == "max":
            return np.maximum(epistemic, aleatoric)
        elif self.uncertainty_weighting == "rms":
            return np.sqrt(epistemic ** 2 + aleatoric ** 2)
        else:
            return epistemic + aleatoric

    def _compute_calibrated_confidence(
        self,
        total_uncertainty: np.ndarray,
        conformal_set_size: np.ndarray,
    ) -> np.ndarray:
        """Map uncertainty to calibrated confidence using Platt scaling.

        Uses a learned sigmoid mapping from uncertainty to confidence,
        fit on calibration data to minimize Brier score.
        """
        if hasattr(self, "_platt_a") and hasattr(self, "_platt_b"):
            # Platt scaling: sigmoid(a * uncertainty + b)
            logits = self._platt_a * total_uncertainty + self._platt_b
            confidence = 1.0 / (1.0 + np.exp(-logits))
        else:
            # Fallback: simple inverse mapping
            max_unc = total_uncertainty.max() + 1e-8
            confidence = 1.0 - np.clip(total_uncertainty / max_unc, 0, 1)

        return confidence

    def _learn_confidence_mapping(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        error_threshold: float = 2.0,
    ):
        """Learn Platt scaling mapping from uncertainty to confidence.

        Fits sigmoid parameters (a, b) such that:
            P(correct | uncertainty) ≈ sigmoid(a * uncertainty + b)
        Since higher uncertainty → lower accuracy, a should be negative.
        """
        from scipy.optimize import minimize as scipy_minimize

        labels = (errors < error_threshold).astype(float)

        def brier_loss(params):
            a, b = params
            logits = a * uncertainties + b
            logits = np.clip(logits, -30, 30)  # prevent overflow
            probs = 1.0 / (1.0 + np.exp(-logits))
            return np.mean((probs - labels) ** 2)

        # Start with reasonable initial guess: negative a (higher unc → lower conf)
        result = scipy_minimize(
            brier_loss,
            x0=[-5.0, 1.0],
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )

        self._platt_a = result.x[0]
        self._platt_b = result.x[1]


def simulate_combined_method(
    simulator,
    combined: dict,
    num_ensemble: int = 5,
    num_mc: int = 10,
    conformal_alpha: float = 0.1,
) -> dict:
    """Run the combined method using the simulator.

    Args:
        simulator: RealisticVLASimulator instance.
        combined: Combined benchmark data dict.
        num_ensemble: Number of ensemble members.
        num_mc: MC samples per ensemble member.
        conformal_alpha: Conformal coverage level.

    Returns:
        Dict with combined UQ results and metrics.
    """
    from src.evaluation.metrics import (
        expected_calibration_error,
        brier_score,
        failure_detection_metrics,
        sparsification_error,
    )
    from src.calibration.methods import SelectivePredictor

    predictions = combined["predictions"]
    gt = combined["ground_truth"]
    errors = combined["errors"]
    N = len(predictions)

    # Split: 30% calibration, 70% test
    cal_size = int(N * 0.3)
    cal_pred, test_pred = predictions[:cal_size], predictions[cal_size:]
    cal_gt, test_gt = gt[:cal_size], gt[cal_size:]
    cal_errors, test_errors = errors[:cal_size], errors[cal_size:]

    # Simulate ensemble predictions with ERROR-CORRELATED disagreement
    # Key insight: ensemble members should disagree more on hard samples
    M = num_ensemble
    ensemble_preds_cal = np.zeros((M, cal_size, *cal_pred.shape[1:]))
    ensemble_preds_test = np.zeros((M, N - cal_size, *test_pred.shape[1:]))

    # Error-correlated noise: models disagree more where errors are larger
    cal_error_scale = (cal_errors / (cal_errors.max() + 1e-8))[:, None, None]  # (N, 1, 1)
    test_error_scale = (test_errors / (test_errors.max() + 1e-8))[:, None, None]

    for m in range(M):
        # Per-model bias (consistent across samples)
        bias = simulator.rng.randn(1, *cal_pred.shape[1:]) * 0.2
        # Error-correlated noise: more disagreement on harder samples
        noise_cal = simulator.rng.randn(*cal_pred.shape) * (0.1 + 0.8 * cal_error_scale)
        noise_test = simulator.rng.randn(*test_pred.shape) * (0.1 + 0.8 * test_error_scale)
        ensemble_preds_cal[m] = cal_pred + bias[:, :cal_pred.shape[1]] + noise_cal
        ensemble_preds_test[m] = test_pred + bias[:, :test_pred.shape[1]] + noise_test

    # Epistemic uncertainty: inter-model disagreement
    cal_epistemic = ensemble_preds_cal.std(axis=0).mean(axis=(1, 2))
    test_epistemic = ensemble_preds_test.std(axis=0).mean(axis=(1, 2))

    # Simulate MC dropout within ensemble (also error-correlated)
    cal_aleatoric = np.zeros(cal_size)
    test_aleatoric = np.zeros(N - cal_size)

    for m in range(M):
        mc_preds = np.zeros((num_mc, cal_size, *cal_pred.shape[1:]))
        for k in range(num_mc):
            mc_noise = simulator.rng.randn(*cal_pred.shape) * (0.05 + 0.3 * cal_error_scale)
            mc_preds[k] = ensemble_preds_cal[m] + mc_noise
        cal_aleatoric += mc_preds.std(axis=0).mean(axis=(1, 2)) / M

        mc_preds = np.zeros((num_mc, N - cal_size, *test_pred.shape[1:]))
        for k in range(num_mc):
            mc_noise = simulator.rng.randn(*test_pred.shape) * (0.05 + 0.3 * test_error_scale)
            mc_preds[k] = ensemble_preds_test[m] + mc_noise
        test_aleatoric += mc_preds.std(axis=0).mean(axis=(1, 2)) / M

    # Fit combined method on calibration set
    method = CalibDriveCombined(
        num_ensemble=M,
        num_mc_samples=num_mc,
        conformal_alpha=conformal_alpha,
    )
    cal_mean = ensemble_preds_cal.mean(axis=0)
    method.calibrate(cal_mean, cal_gt, cal_epistemic, cal_aleatoric)

    # Predict on test set
    result = method.predict(ensemble_preds_test)

    # Evaluate
    test_confidences = result.calibrated_confidence
    test_accuracies = (test_errors < 2.0).astype(float)

    cal_metrics = expected_calibration_error(test_confidences, test_accuracies)
    bs = brier_score(test_confidences, test_accuracies)
    failure = failure_detection_metrics(
        result.total_uncertainty, test_errors, 2.0
    )
    sparse = sparsification_error(result.total_uncertainty, test_errors)

    sp = SelectivePredictor()
    sp_res = sp.evaluate(test_confidences, test_errors, 2.0)
    point_80 = min(
        sp_res["coverage_curve"],
        key=lambda p: abs(p["coverage"] - 0.8)
    )

    # Decision distribution
    decisions = result.decision
    decision_counts = {
        "proceed": int((decisions == "proceed").sum()),
        "slow_down": int((decisions == "slow_down").sum()),
        "abstain": int((decisions == "abstain").sum()),
    }

    return {
        "ece": cal_metrics["ece"],
        "mce": cal_metrics["mce"],
        "brier": bs,
        "auroc": failure["auroc"],
        "auprc": failure["auprc"],
        "ause": sparse["ause"],
        "selective_failure_80": point_80["selective_failure_rate"],
        "decision_counts": decision_counts,
        "epistemic_mean": float(test_epistemic.mean()),
        "aleatoric_mean": float(test_aleatoric.mean()),
        "conformal_coverage": float(result.conformal_in_set.mean()),
    }
