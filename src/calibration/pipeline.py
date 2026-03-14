"""
CalibDrive Evaluation Pipeline.

End-to-end pipeline for evaluating VLA calibration:
1. Load model and dataset
2. Run inference (with optional UQ method)
3. Compute calibration, driving, and safety metrics
4. Generate reliability diagrams and analysis
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.calibration.methods import (
    ConformalPredictor,
    SelectivePredictor,
    TemperatureScaling,
)
from src.evaluation.metrics import (
    brier_score,
    collision_rate,
    expected_calibration_error,
    failure_detection_metrics,
    negative_log_likelihood,
    sparsification_error,
    trajectory_l2_error,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration."""
    model_name: str
    uq_method: str  # "none", "mc_dropout", "temperature_scaling", "ensemble", "conformal"
    dataset_name: str
    num_samples: int = 1000
    prediction_horizon: int = 10
    mc_samples: int = 20
    dropout_rate: float = 0.1
    conformal_alpha: float = 0.1
    calibration_fraction: float = 0.3
    num_bins: int = 15
    accuracy_threshold: float = 2.0
    collision_radius: float = 2.0
    seed: int = 42


@dataclass(frozen=True)
class PipelineResults:
    """Immutable pipeline results."""
    config: PipelineConfig
    # Calibration metrics
    ece: float
    mce: float
    brier: float
    nll: float
    # Driving performance
    ade: float
    fde: float
    # Uncertainty quality
    auroc: float
    auprc: float
    ause: float
    # Safety
    collision_rate_value: float
    # Selective prediction
    selective_coverage: Optional[float] = None
    selective_collision_rate: Optional[float] = None
    # Raw data for visualization
    reliability_bins: Optional[list] = None
    coverage_curve: Optional[list] = None
    sparsification_data: Optional[dict] = None


class CalibDrivePipeline:
    """End-to-end calibration evaluation pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        np.random.seed(config.seed)

    def run(self) -> PipelineResults:
        """Execute the full evaluation pipeline."""
        # Step 1: Generate predictions and ground truth
        predictions, ground_truth, raw_confidences = self._get_predictions()

        # Step 2: Apply UQ method
        confidences, uncertainties = self._apply_uq_method(
            predictions, ground_truth, raw_confidences
        )

        # Step 3: Compute all metrics
        return self._compute_metrics(
            predictions, ground_truth, confidences, uncertainties
        )

    def _get_predictions(self):
        """Get model predictions. Uses synthetic data for pipeline validation."""
        N = self.config.num_samples
        T = self.config.prediction_horizon

        # Ground truth trajectories
        gt = np.zeros((N, T, 2))
        for i in range(N):
            t = np.linspace(0, 2 * np.pi, T)
            gt[i, :, 0] = np.cumsum(np.random.uniform(0.5, 2, T))
            gt[i, :, 1] = np.random.uniform(1, 5) * np.sin(
                np.random.uniform(0.5, 2) * t
            )

        # Predictions with varying quality
        noise_scale = 1.0
        pred = gt + np.random.randn(N, T, 2) * noise_scale

        # Raw confidences (before calibration)
        errors = np.linalg.norm(pred - gt, axis=(1, 2))
        # Simulate overconfident model: confidences too high for actual errors
        overconfidence_factor = 1.3
        raw_confidences = np.clip(
            1.0 / (1.0 + errors / (T * 2)) * overconfidence_factor, 0, 1
        )

        return pred, gt, raw_confidences

    def _apply_uq_method(self, predictions, ground_truth, raw_confidences):
        """Apply the specified UQ method."""
        method = self.config.uq_method
        N = len(predictions)

        if method == "none":
            uncertainties = 1 - raw_confidences
            return raw_confidences, uncertainties

        if method == "mc_dropout":
            return self._apply_mc_dropout(predictions, ground_truth)

        if method == "temperature_scaling":
            return self._apply_temperature_scaling(raw_confidences, ground_truth, predictions)

        if method == "conformal":
            return self._apply_conformal(predictions, ground_truth, raw_confidences)

        # Default: return raw
        return raw_confidences, 1 - raw_confidences

    def _apply_mc_dropout(self, predictions, ground_truth):
        """Simulate MC Dropout by adding stochastic perturbations."""
        N, T, D = predictions.shape
        num_samples = self.config.mc_samples
        dropout_rate = self.config.dropout_rate

        # Simulate multiple forward passes with dropout noise
        all_preds = np.zeros((num_samples, N, T, D))
        for s in range(num_samples):
            dropout_mask = np.random.binomial(1, 1 - dropout_rate, predictions.shape)
            noise = np.random.randn(N, T, D) * 0.3
            all_preds[s] = predictions * dropout_mask / (1 - dropout_rate) + noise

        mean_pred = all_preds.mean(axis=0)
        std_pred = all_preds.std(axis=0)

        # Uncertainty = mean std across trajectory
        uncertainties = std_pred.mean(axis=(1, 2))
        confidences = 1.0 / (1.0 + uncertainties)

        return confidences, uncertainties

    def _apply_temperature_scaling(self, raw_confidences, ground_truth, predictions):
        """Apply temperature scaling to calibrate confidences."""
        N = len(raw_confidences)
        cal_size = int(N * self.config.calibration_fraction)

        # Convert to pseudo-logits for temperature scaling
        eps = 1e-6
        logits = np.log(np.clip(raw_confidences, eps, 1 - eps) / (1 - np.clip(raw_confidences, eps, 1 - eps)))

        # Create binary labels: was prediction accurate?
        errors = np.linalg.norm(
            predictions - ground_truth, axis=(1, 2)
        ) / (self.config.prediction_horizon * 2)
        labels = (errors < 0.5).astype(int)

        # Fit temperature on calibration set
        cal_logits = np.stack([-(logits[:cal_size]), logits[:cal_size]], axis=1)
        ts = TemperatureScaling()
        ts.fit(cal_logits, labels[:cal_size])

        # Apply to all data
        all_logits = np.stack([-logits, logits], axis=1)
        calibrated = ts.calibrate(all_logits)
        calibrated_confidences = calibrated[:, 1]

        uncertainties = 1 - calibrated_confidences
        return calibrated_confidences, uncertainties

    def _apply_conformal(self, predictions, ground_truth, raw_confidences):
        """Apply conformal prediction."""
        N = len(predictions)
        cal_size = int(N * self.config.calibration_fraction)

        # Reshape for conformal predictor
        pred_flat = predictions.reshape(N, -1)
        gt_flat = ground_truth.reshape(N, -1)

        cp = ConformalPredictor(alpha=self.config.conformal_alpha)
        cp.calibrate(pred_flat[:cal_size], gt_flat[:cal_size])

        result = cp.predict(pred_flat, uncertainties=1 - raw_confidences)

        # Convert prediction set size to uncertainty
        uncertainties = result["radii"] / (result["quantile"] + 1e-8)
        confidences = 1.0 / (1.0 + uncertainties)

        return confidences, uncertainties

    def _compute_metrics(self, predictions, ground_truth, confidences, uncertainties):
        """Compute all evaluation metrics."""
        N = len(predictions)

        # Trajectory errors
        traj_metrics = trajectory_l2_error(predictions, ground_truth)
        ade_per_sample = np.linalg.norm(
            predictions - ground_truth, axis=-1
        ).mean(axis=1)

        # Binary accuracy for calibration
        accuracies = (ade_per_sample < self.config.accuracy_threshold).astype(float)

        # Calibration metrics
        cal = expected_calibration_error(confidences, accuracies, self.config.num_bins)
        bs = brier_score(confidences, accuracies)

        # NLL (treat as binary classification)
        nll = negative_log_likelihood(confidences, accuracies)

        # Failure detection
        failure = failure_detection_metrics(
            uncertainties, ade_per_sample, self.config.accuracy_threshold
        )

        # Sparsification
        sparse = sparsification_error(uncertainties, ade_per_sample)

        # Collision rate (synthetic obstacles)
        obstacles = np.random.randn(N, 5, 2) * 10 + predictions[:, -1:, :]
        col = collision_rate(predictions, obstacles, self.config.collision_radius)

        # Selective prediction
        sp = SelectivePredictor()
        sp_results = sp.evaluate(
            confidences, ade_per_sample, self.config.accuracy_threshold
        )

        # Find 80% coverage point
        coverage_80 = None
        for point in sp_results["coverage_curve"]:
            if abs(point["coverage"] - 0.8) < 0.02:
                coverage_80 = point
                break

        return PipelineResults(
            config=self.config,
            ece=cal["ece"],
            mce=cal["mce"],
            brier=bs,
            nll=nll,
            ade=traj_metrics["ade"],
            fde=traj_metrics["fde"],
            auroc=failure["auroc"],
            auprc=failure["auprc"],
            ause=sparse["ause"],
            collision_rate_value=col["collision_rate"],
            selective_coverage=0.8 if coverage_80 else None,
            selective_collision_rate=coverage_80["selective_failure_rate"] if coverage_80 else None,
            reliability_bins=cal["bins"],
            coverage_curve=sp_results["coverage_curve"],
            sparsification_data=sparse,
        )
