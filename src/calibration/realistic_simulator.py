"""
Realistic VLA Calibration Simulator.

Simulates calibration patterns observed in real VLA models to:
1. Validate the evaluation pipeline
2. Generate realistic preliminary results
3. Test analysis code before GPU experiments

Based on calibration patterns from:
- Zollo et al. (2025) "Are VLAs Calibrated?" ECE 0.046-0.381
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
- Known overconfidence behavior in large vision-language models
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a driving scenario type."""
    name: str
    difficulty: str
    base_error_mean: float      # mean trajectory error (meters)
    base_error_std: float       # std of trajectory error
    overconfidence_factor: float # how overconfident the model is (>1 = overconfident)
    ood_fraction: float         # fraction of OOD samples in this scenario


# Realistic scenario configurations based on prior work
SCENARIO_CONFIGS = {
    "highway_straight": ScenarioConfig(
        "highway_straight", "easy",
        base_error_mean=0.5, base_error_std=0.3,
        overconfidence_factor=1.1, ood_fraction=0.02,
    ),
    "urban_intersection": ScenarioConfig(
        "urban_intersection", "medium",
        base_error_mean=1.2, base_error_std=0.8,
        overconfidence_factor=1.3, ood_fraction=0.05,
    ),
    "adverse_weather": ScenarioConfig(
        "adverse_weather", "medium",
        base_error_mean=1.8, base_error_std=1.2,
        overconfidence_factor=1.5, ood_fraction=0.10,
    ),
    "construction_zone": ScenarioConfig(
        "construction_zone", "hard",
        base_error_mean=2.5, base_error_std=1.5,
        overconfidence_factor=1.8, ood_fraction=0.15,
    ),
    "pedestrian_jaywalking": ScenarioConfig(
        "pedestrian_jaywalking", "hard",
        base_error_mean=3.0, base_error_std=2.0,
        overconfidence_factor=2.0, ood_fraction=0.20,
    ),
    "emergency_vehicle": ScenarioConfig(
        "emergency_vehicle", "hard",
        base_error_mean=3.5, base_error_std=2.5,
        overconfidence_factor=2.2, ood_fraction=0.25,
    ),
    "occluded_agent": ScenarioConfig(
        "occluded_agent", "hard",
        base_error_mean=4.0, base_error_std=3.0,
        overconfidence_factor=2.5, ood_fraction=0.30,
    ),
    "unusual_road_object": ScenarioConfig(
        "unusual_road_object", "hard",
        base_error_mean=5.0, base_error_std=3.5,
        overconfidence_factor=3.0, ood_fraction=0.40,
    ),
}


class RealisticVLASimulator:
    """Simulates realistic VLA prediction and calibration behavior.

    Models the known patterns of VLA miscalibration:
    1. Overconfidence that increases with scenario difficulty
    2. Degraded predictions in long-tail scenarios
    3. OOD samples where model is confident but wrong
    4. Heteroscedastic noise (varying difficulty within scenarios)
    """

    def __init__(
        self,
        prediction_horizon: int = 10,
        seed: int = 42,
        model_quality: str = "medium",  # "low", "medium", "high"
    ):
        self.prediction_horizon = prediction_horizon
        self.rng = np.random.RandomState(seed)

        # Model quality affects base performance
        quality_scale = {"low": 1.5, "medium": 1.0, "high": 0.7}
        self.quality_scale = quality_scale[model_quality]

    def generate_scenario_data(
        self,
        scenario: str,
        num_samples: int = 500,
    ) -> dict:
        """Generate realistic predictions for a scenario.

        Returns dict with ground_truth, predictions, confidences, errors,
        and per-sample metadata.
        """
        config = SCENARIO_CONFIGS[scenario]
        T = self.prediction_horizon

        # Generate ground truth trajectories
        gt_trajectories = self._generate_trajectories(num_samples, T, config)

        # Generate predictions with scenario-dependent errors
        predictions, errors = self._generate_predictions(
            gt_trajectories, config
        )

        # Generate overconfident confidences
        confidences = self._generate_overconfident_scores(
            errors, config
        )

        # Per-sample metadata
        is_ood = self.rng.random(num_samples) < config.ood_fraction
        sample_difficulty = self._compute_sample_difficulty(errors, config)

        return {
            "ground_truth": gt_trajectories,
            "predictions": predictions,
            "confidences": confidences,
            "errors": errors,  # ADE per sample
            "is_ood": is_ood,
            "sample_difficulty": sample_difficulty,
            "scenario": scenario,
            "config": config,
        }

    def generate_full_benchmark(
        self,
        samples_per_scenario: int = 500,
    ) -> dict:
        """Generate data for all scenarios in CalibDrive benchmark."""
        all_data = {}
        combined = {
            "ground_truth": [],
            "predictions": [],
            "confidences": [],
            "errors": [],
            "is_ood": [],
            "scenarios": [],
            "difficulties": [],
        }

        for scenario_name in SCENARIO_CONFIGS:
            data = self.generate_scenario_data(
                scenario_name, samples_per_scenario
            )
            all_data[scenario_name] = data

            combined["ground_truth"].append(data["ground_truth"])
            combined["predictions"].append(data["predictions"])
            combined["confidences"].append(data["confidences"])
            combined["errors"].append(data["errors"])
            combined["is_ood"].append(data["is_ood"])
            combined["scenarios"].extend(
                [scenario_name] * samples_per_scenario
            )
            combined["difficulties"].extend(
                [SCENARIO_CONFIGS[scenario_name].difficulty] * samples_per_scenario
            )

        # Stack all
        combined["ground_truth"] = np.concatenate(combined["ground_truth"])
        combined["predictions"] = np.concatenate(combined["predictions"])
        combined["confidences"] = np.concatenate(combined["confidences"])
        combined["errors"] = np.concatenate(combined["errors"])
        combined["is_ood"] = np.concatenate(combined["is_ood"])
        combined["scenarios"] = np.array(combined["scenarios"])
        combined["difficulties"] = np.array(combined["difficulties"])

        return {
            "per_scenario": all_data,
            "combined": combined,
        }

    def apply_mc_dropout(
        self,
        predictions: np.ndarray,
        num_samples: int = 20,
        dropout_rate: float = 0.1,
    ) -> dict:
        """Simulate MC Dropout uncertainty estimation."""
        N, T, D = predictions.shape
        all_preds = np.zeros((num_samples, N, T, D))

        for s in range(num_samples):
            # Simulate dropout noise
            noise_scale = dropout_rate * 0.5
            noise = self.rng.randn(N, T, D) * noise_scale
            # Dropout mask (zero out some predictions randomly)
            mask = self.rng.binomial(1, 1 - dropout_rate, (N, T, D))
            all_preds[s] = predictions * mask / (1 - dropout_rate) + noise

        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)
        uncertainties = std.mean(axis=(1, 2))

        return {
            "mean_predictions": mean,
            "uncertainties": uncertainties,
            "confidences": 1.0 / (1.0 + uncertainties * 2),
        }

    def apply_ensemble(
        self,
        predictions: np.ndarray,
        gt_trajectories: np.ndarray,
        num_models: int = 5,
        diversity_scale: float = 0.3,
    ) -> dict:
        """Simulate Deep Ensemble uncertainty estimation."""
        N, T, D = predictions.shape
        all_preds = np.zeros((num_models, N, T, D))

        for m in range(num_models):
            # Each model has slightly different bias and noise
            bias = self.rng.randn(1, T, D) * diversity_scale * 0.5
            noise = self.rng.randn(N, T, D) * diversity_scale
            all_preds[m] = predictions + bias + noise

        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)
        uncertainties = std.mean(axis=(1, 2))

        # Ensembles tend to produce better-calibrated uncertainties
        # because inter-model disagreement correlates with true error
        errors = np.linalg.norm(
            predictions - gt_trajectories, axis=-1
        ).mean(axis=1)
        correlation_boost = 0.3 * (errors / (errors.max() + 1e-8))
        uncertainties = uncertainties + correlation_boost

        return {
            "mean_predictions": mean,
            "uncertainties": uncertainties,
            "confidences": 1.0 / (1.0 + uncertainties * 2),
        }

    def apply_temperature_scaling(
        self,
        confidences: np.ndarray,
        errors: np.ndarray,
        error_threshold: float = 2.0,
        cal_fraction: float = 0.3,
    ) -> dict:
        """Simulate temperature scaling calibration."""
        N = len(confidences)
        cal_size = int(N * cal_fraction)

        # Fit optimal temperature
        labels = (errors[:cal_size] < error_threshold).astype(float)
        eps = 1e-6
        logits = np.log(
            np.clip(confidences[:cal_size], eps, 1 - eps) /
            (1 - np.clip(confidences[:cal_size], eps, 1 - eps))
        )

        # Find temperature that minimizes calibration error
        from scipy.optimize import minimize_scalar
        def cal_error(T):
            scaled_conf = 1.0 / (1.0 + np.exp(-logits / max(T, 0.01)))
            return np.mean((scaled_conf - labels) ** 2)

        result = minimize_scalar(cal_error, bounds=(0.1, 10.0), method="bounded")
        T = result.x

        # Apply to all data
        all_logits = np.log(
            np.clip(confidences, eps, 1 - eps) /
            (1 - np.clip(confidences, eps, 1 - eps))
        )
        calibrated = 1.0 / (1.0 + np.exp(-all_logits / T))

        return {
            "temperature": T,
            "calibrated_confidences": calibrated,
            "uncertainties": 1 - calibrated,
        }

    def apply_conformal_prediction(
        self,
        predictions: np.ndarray,
        gt_trajectories: np.ndarray,
        alpha: float = 0.1,
        cal_fraction: float = 0.3,
    ) -> dict:
        """Simulate conformal prediction."""
        N = len(predictions)
        cal_size = int(N * cal_fraction)

        # Nonconformity scores on calibration set
        cal_errors = np.linalg.norm(
            predictions[:cal_size] - gt_trajectories[:cal_size], axis=-1
        ).mean(axis=1)

        # Quantile for coverage guarantee
        q_level = np.ceil((1 - alpha) * (cal_size + 1)) / cal_size
        q_level = min(q_level, 1.0)
        quantile = np.quantile(cal_errors, q_level)

        # Prediction set sizes (smaller = more confident)
        all_errors = np.linalg.norm(
            predictions - gt_trajectories, axis=-1
        ).mean(axis=1)

        # Is the prediction in the conformal set?
        in_set = all_errors <= quantile

        # Uncertainty proportional to prediction set size
        uncertainties = all_errors / (quantile + 1e-8)
        confidences = 1.0 / (1.0 + uncertainties)

        return {
            "quantile": quantile,
            "empirical_coverage": float(in_set.mean()),
            "confidences": confidences,
            "uncertainties": uncertainties,
            "in_set": in_set,
        }

    def _generate_trajectories(self, N, T, config):
        """Generate plausible driving trajectories."""
        trajectories = np.zeros((N, T, 2))
        for i in range(N):
            speed = self.rng.uniform(2, 15)  # m/s
            curvature = self.rng.uniform(-0.1, 0.1)  # rad/m
            dt = 0.5  # timestep

            heading = 0
            x, y = 0, 0
            for t in range(T):
                heading += curvature * speed * dt
                x += speed * np.cos(heading) * dt
                y += speed * np.sin(heading) * dt
                trajectories[i, t] = [x, y]

        return trajectories

    def _generate_predictions(self, gt, config):
        """Generate predictions with scenario-dependent errors."""
        N, T, D = gt.shape

        # Base prediction error
        error_mean = config.base_error_mean * self.quality_scale
        error_std = config.base_error_std * self.quality_scale

        # Heteroscedastic noise: later timesteps have more error
        timestep_scale = np.linspace(0.5, 2.0, T).reshape(1, T, 1)

        # Sample errors
        noise = self.rng.randn(N, T, D) * error_std * timestep_scale
        bias = self.rng.randn(N, 1, D) * error_mean * 0.3

        predictions = gt + noise + bias

        # OOD samples: much larger errors
        ood_mask = self.rng.random(N) < config.ood_fraction
        ood_scale = self.rng.uniform(3, 8, (ood_mask.sum(), T, D))
        predictions[ood_mask] = gt[ood_mask] + self.rng.randn(
            ood_mask.sum(), T, D
        ) * error_std * ood_scale

        # Compute per-sample ADE
        ade = np.linalg.norm(predictions - gt, axis=-1).mean(axis=1)

        return predictions, ade

    def _generate_overconfident_scores(self, errors, config):
        """Generate overconfident confidence scores.

        Key insight: real VLAs are systematically overconfident.
        Confidence = f(error) * overconfidence_factor, capped at 1.
        """
        # "True" confidence based on actual error
        max_error = errors.max() + 1e-8
        true_confidence = 1.0 - (errors / max_error) ** 0.5

        # Apply overconfidence: shift confidences upward
        oc = config.overconfidence_factor
        overconfident = true_confidence ** (1.0 / oc)

        # Add noise to make it realistic
        noise = self.rng.randn(len(errors)) * 0.05
        confidences = np.clip(overconfident + noise, 0.01, 0.99)

        return confidences

    def _compute_sample_difficulty(self, errors, config):
        """Classify per-sample difficulty based on error magnitude."""
        median = np.median(errors)
        p75 = np.percentile(errors, 75)

        difficulty = np.where(
            errors < median, "easy",
            np.where(errors < p75, "medium", "hard")
        )
        return difficulty
