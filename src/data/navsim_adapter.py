"""
NAVSIM Dataset Adapter for CalibDrive.

Loads NAVSIM data and converts it to the CalibDrive format for
VLA calibration evaluation. Supports both NAVSIMv1 and v2.

Setup:
    git clone https://github.com/autonomousvision/navsim
    cd navsim && pip install -e .
    # Download data via huggingface:
    # huggingface-cli download --repo-type dataset autonomousvision/navsim_logs
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from navsim.common.dataclasses import Scene, AgentInput
    from navsim.evaluate.pdms import PDMSMetric
    HAS_NAVSIM = True
except ImportError:
    HAS_NAVSIM = False


@dataclass(frozen=True)
class NAVSIMSample:
    """Processed NAVSIM sample for CalibDrive evaluation."""
    scene_token: str
    front_image: np.ndarray         # (H, W, 3)
    ego_trajectory_gt: np.ndarray   # (T, 3) [x, y, heading]
    ego_speed: float
    scenario_description: str
    difficulty: str
    metadata: dict


class NAVSIMAdapter:
    """Adapter for loading and processing NAVSIM data.

    Converts NAVSIM scenes into CalibDrive evaluation format.
    """

    def __init__(
        self,
        data_root: str = "~/navsim_data",
        split: str = "val",
        max_samples: Optional[int] = None,
    ):
        if not HAS_NAVSIM:
            raise ImportError(
                "NAVSIM not installed. Run: "
                "git clone https://github.com/autonomousvision/navsim && "
                "cd navsim && pip install -e ."
            )

        self.data_root = Path(data_root).expanduser()
        self.split = split
        self.max_samples = max_samples
        self.samples = []

    def load(self) -> list[NAVSIMSample]:
        """Load and process NAVSIM scenes."""
        # NAVSIM data loading via its API
        from navsim.common.dataloader import SceneLoader

        loader = SceneLoader(
            data_root=str(self.data_root),
            split=self.split,
        )

        scenes = loader.get_scenes()
        if self.max_samples:
            scenes = scenes[:self.max_samples]

        self.samples = [self._process_scene(scene) for scene in scenes]
        return self.samples

    def _process_scene(self, scene) -> NAVSIMSample:
        """Convert a NAVSIM scene to CalibDrive format."""
        # Extract front camera image
        front_image = scene.get_camera_image("CAM_FRONT")

        # Extract ego trajectory (future T timesteps)
        ego_future = scene.get_ego_future_trajectory()
        ego_trajectory = np.array([
            [pose.x, pose.y, pose.heading]
            for pose in ego_future
        ])

        # Classify scenario difficulty based on scene attributes
        difficulty = self._classify_difficulty(scene)
        description = self._describe_scenario(scene)

        return NAVSIMSample(
            scene_token=scene.token,
            front_image=front_image,
            ego_trajectory_gt=ego_trajectory,
            ego_speed=scene.get_ego_speed(),
            scenario_description=description,
            difficulty=difficulty,
            metadata={
                "location": getattr(scene, "location", "unknown"),
                "weather": getattr(scene, "weather", "unknown"),
                "num_agents": len(scene.get_agents()) if hasattr(scene, "get_agents") else 0,
            },
        )

    def _classify_difficulty(self, scene) -> str:
        """Classify scene difficulty for CalibDrive benchmark."""
        # Heuristic difficulty classification based on scene properties
        num_agents = len(scene.get_agents()) if hasattr(scene, "get_agents") else 0
        ego_speed = scene.get_ego_speed() if hasattr(scene, "get_ego_speed") else 0

        # Simple heuristic: more agents + higher speed = harder
        complexity = num_agents * 0.3 + ego_speed * 0.1

        if complexity < 3:
            return "easy"
        if complexity < 7:
            return "medium"
        return "hard"

    def _describe_scenario(self, scene) -> str:
        """Generate natural language description of scenario."""
        parts = []
        if hasattr(scene, "location"):
            parts.append(f"Location: {scene.location}")
        if hasattr(scene, "weather"):
            parts.append(f"Weather: {scene.weather}")

        ego_speed = scene.get_ego_speed() if hasattr(scene, "get_ego_speed") else 0
        parts.append(f"Ego speed: {ego_speed:.1f} m/s")

        return ". ".join(parts) if parts else "Standard driving scenario"

    def generate_vla_prompt(self, sample: NAVSIMSample) -> str:
        """Generate a VLA prompt for the given sample.

        Returns a driving-specific prompt that can be fed to VLA models.
        """
        return (
            f"You are driving a vehicle. {sample.scenario_description}. "
            f"Current speed: {sample.ego_speed:.1f} m/s. "
            "Predict the future trajectory of the ego vehicle for the next "
            "4 seconds as a sequence of (x, y) waypoints in the ego frame."
        )

    def compute_pdms(
        self,
        predicted_trajectory: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Compute PDMS (Predictive Driver Model Score).

        NAVSIM's primary metric combining multiple driving quality aspects.
        """
        if not HAS_NAVSIM:
            # Fallback: simple trajectory similarity
            error = np.linalg.norm(
                predicted_trajectory[:, :2] - ground_truth[:, :2], axis=-1
            )
            # Simple PDMS proxy: 1 - normalized_error
            return float(max(0, 1 - error.mean() / 10.0))

        metric = PDMSMetric()
        return metric.compute(predicted_trajectory, ground_truth)


class NAVSIMCalibDriveWrapper:
    """Wrapper that runs CalibDrive evaluation on NAVSIM data.

    Connects the NAVSIM adapter with CalibDrive calibration pipeline.
    """

    def __init__(
        self,
        navsim_adapter: NAVSIMAdapter,
        vla_model,
        uq_method: str = "none",
    ):
        self.adapter = navsim_adapter
        self.model = vla_model
        self.uq_method = uq_method

    def run_evaluation(self) -> dict:
        """Run full CalibDrive evaluation on NAVSIM data."""
        samples = self.adapter.load()

        predictions = []
        ground_truths = []
        confidences = []

        for sample in samples:
            prompt = self.adapter.generate_vla_prompt(sample)

            # Get model prediction with uncertainty
            result = self._predict_with_uncertainty(
                sample.front_image, prompt
            )

            predictions.append(result["trajectory"])
            ground_truths.append(sample.ego_trajectory_gt[:, :2])
            confidences.append(result["confidence"])

        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        confidences = np.array(confidences)

        # Compute errors
        errors = np.linalg.norm(
            predictions - ground_truths, axis=-1
        ).mean(axis=1)

        return {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "confidences": confidences,
            "errors": errors,
            "samples": samples,
        }

    def _predict_with_uncertainty(self, image, prompt):
        """Get VLA prediction with uncertainty estimate."""
        if self.uq_method == "mc_dropout":
            return self.model.forward_with_dropout(image, prompt)
        return self.model.forward(image, prompt)
