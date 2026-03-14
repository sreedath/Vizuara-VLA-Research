"""
NAVSIM VLA Agent with Calibration Support.

Wraps a VLA model (OpenVLA or OpenDriveVLA) as a NAVSIM AbstractAgent
for standardized evaluation with CalibDrive uncertainty metrics.

Usage:
    agent = CalibDriveNavsimAgent(model_name="openvla/openvla-7b")
    # Register with NAVSIM evaluation pipeline
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.common.dataclasses import (
        AgentInput,
        SensorConfig,
        Trajectory,
        TrajectorySampling,
    )
    HAS_NAVSIM = True
except ImportError:
    HAS_NAVSIM = False


@dataclass(frozen=True)
class CalibDrivePrediction:
    """Immutable prediction with calibration signals."""
    trajectory: np.ndarray         # (40, 3) [x, y, heading]
    confidence: float              # aggregated confidence [0, 1]
    per_dim_entropy: list[float]   # per-action-dim entropy
    per_dim_max_prob: list[float]  # per-action-dim max probability
    top_k_mass: float              # top-5 bin mass
    raw_logits: Optional[np.ndarray] = None  # (D, 256) raw logits


if HAS_NAVSIM and HAS_TORCH:
    class CalibDriveNavsimAgent(AbstractAgent):
        """NAVSIM agent wrapping a VLA with uncertainty extraction."""

        def __init__(
            self,
            model_name: str = "openvla/openvla-7b",
            device: str = "cuda",
            num_mc_samples: int = 0,
            store_logits: bool = True,
        ):
            super().__init__(
                trajectory_sampling=TrajectorySampling(
                    num_poses=40,
                    interval_length=0.1,
                ),
            )
            self.model_name = model_name
            self.device = device
            self.num_mc_samples = num_mc_samples
            self.store_logits = store_logits

            # Stored results for calibration analysis
            self.predictions_log: list[CalibDrivePrediction] = []
            self._model = None
            self._processor = None

        @property
        def name(self) -> str:
            return "calibdrive_vla"

        def initialize(self):
            """Load the VLA model."""
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True,
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()

        def get_sensor_config(self) -> SensorConfig:
            """Only request front camera for current frame."""
            config = SensorConfig.build_no_sensors()
            config.cam_f0 = [3]  # latest frame
            return config

        def compute_trajectory(
            self,
            agent_input: AgentInput,
        ) -> Trajectory:
            """Compute trajectory with uncertainty signals."""
            from PIL import Image

            # Extract front camera
            front_cam = agent_input.cameras[3].cam_f0
            pil_image = Image.fromarray(front_cam.image.astype(np.uint8))

            # Ego speed for prompt
            ego_v = np.array(agent_input.ego_statuses[3].ego_velocity)
            speed = float(np.linalg.norm(ego_v))

            prompt = (
                "In: What action should the robot take to drive forward "
                f"at {speed:.1f} m/s safely?\nOut:"
            )

            # Run inference
            if self.num_mc_samples > 0:
                result = self._predict_mc_dropout(pil_image, prompt)
            else:
                result = self._predict_single(pil_image, prompt)

            # Store for calibration analysis
            self.predictions_log.append(result)

            return Trajectory(
                poses=result.trajectory,
                trajectory_sampling=self._trajectory_sampling,
            )

        def _predict_single(
            self,
            image,
            prompt: str,
        ) -> CalibDrivePrediction:
            """Single forward pass with logit extraction."""
            inputs = self._processor(
                prompt, image,
            ).to(self.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=7,  # 7 action dims
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Extract action logits (last 256 vocab entries = action bins)
            action_logits = []
            for step_scores in outputs.scores:
                bin_logits = step_scores[0, -256:].cpu().float().numpy()
                action_logits.append(bin_logits)

            action_logits = np.array(action_logits)  # (7, 256)

            # Compute calibration signals
            from src.data.navsim_adapter import extract_calibration_signals_from_logits
            cal_signals = extract_calibration_signals_from_logits(action_logits)

            # Decode action tokens to trajectory
            trajectory = self._decode_to_trajectory(outputs, inputs, speed=5.0)

            return CalibDrivePrediction(
                trajectory=trajectory,
                confidence=cal_signals["confidence_geomean"],
                per_dim_entropy=cal_signals["per_dim_entropy"],
                per_dim_max_prob=cal_signals["per_dim_max_prob"],
                top_k_mass=cal_signals["top_k_mass"],
                raw_logits=action_logits if self.store_logits else None,
            )

        def _predict_mc_dropout(
            self,
            image,
            prompt: str,
        ) -> CalibDrivePrediction:
            """MC Dropout: multiple passes with dropout enabled."""
            # Enable dropout
            for m in self._model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()

            trajectories = []
            all_confidences = []

            for _ in range(self.num_mc_samples):
                result = self._predict_single(image, prompt)
                trajectories.append(result.trajectory)
                all_confidences.append(result.confidence)

            # Disable dropout
            self._model.eval()

            trajectories = np.array(trajectories)
            mean_traj = trajectories.mean(axis=0)
            std_traj = trajectories.std(axis=0)

            # MC uncertainty: mean trajectory std
            mc_uncertainty = float(std_traj.mean())
            mc_confidence = 1.0 / (1.0 + mc_uncertainty * 5)

            return CalibDrivePrediction(
                trajectory=mean_traj,
                confidence=mc_confidence,
                per_dim_entropy=[],
                per_dim_max_prob=[],
                top_k_mass=0.0,
            )

        def _decode_to_trajectory(
            self,
            outputs,
            inputs,
            speed: float = 5.0,
        ) -> np.ndarray:
            """Decode generated action tokens to NAVSIM trajectory.

            OpenVLA produces 7 action dimensions (x,y,z,roll,pitch,yaw,gripper).
            For driving, we map the first 2 dimensions to x,y velocity
            and integrate to get trajectory.
            """
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]

            # Decode to continuous values via 256-bin dequantization
            vocab_size = self._processor.tokenizer.vocab_size
            bin_centers = np.linspace(-1, 1, 256)

            actions = np.zeros(min(7, len(generated_ids)))
            for i, token_id in enumerate(generated_ids[:7]):
                bin_idx = vocab_size - int(token_id) - 1
                bin_idx = np.clip(bin_idx, 0, 255)
                actions[i] = bin_centers[bin_idx]

            # Map actions to trajectory:
            # actions[0] → lateral velocity, actions[1] → longitudinal velocity
            dt = 0.1  # 10Hz
            poses = np.zeros((40, 3), dtype=np.float32)

            vx = actions[1] * speed if len(actions) > 1 else speed * 0.5
            vy = actions[0] * speed * 0.3 if len(actions) > 0 else 0.0
            yaw_rate = actions[5] * 0.5 if len(actions) > 5 else 0.0

            heading = 0.0
            x, y = 0.0, 0.0

            for t in range(40):
                heading += yaw_rate * dt
                x += vx * np.cos(heading) * dt
                y += vx * np.sin(heading) * dt + vy * dt
                poses[t] = [x, y, heading]

            return poses

        def get_calibration_data(self) -> dict:
            """Return all stored predictions for calibration analysis."""
            if not self.predictions_log:
                return {}

            return {
                "trajectories": np.array([p.trajectory for p in self.predictions_log]),
                "confidences": np.array([p.confidence for p in self.predictions_log]),
                "top_k_masses": np.array([p.top_k_mass for p in self.predictions_log]),
                "num_predictions": len(self.predictions_log),
            }
