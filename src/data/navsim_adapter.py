"""
NAVSIM Dataset Adapter for CalibDrive.

Loads NAVSIM data and converts it to the CalibDrive format for
VLA calibration evaluation. Supports NAVSIMv1 and v2.

Setup:
    git clone https://github.com/autonomousvision/navsim
    cd navsim && pip install -e .
    # Download data:
    cd download && ./download_maps && ./download_trainval

Environment variables:
    NUPLAN_MAP_VERSION=nuplan-maps-v1.0
    NUPLAN_MAPS_ROOT=~/navsim_workspace/dataset/maps
    NAVSIM_DEVKIT_ROOT=~/navsim_workspace/navsim
    OPENSCENE_DATA_ROOT=~/navsim_workspace/dataset
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

try:
    from navsim.common.dataclasses import (
        AgentInput,
        SensorConfig,
        Trajectory,
        TrajectorySampling,
    )
    from navsim.common.dataloader import SceneLoader
    HAS_NAVSIM = True
except ImportError:
    HAS_NAVSIM = False


# NAVSIM trajectory: 40 poses at 10Hz = 4 seconds
NAVSIM_NUM_POSES = 40
NAVSIM_DT = 0.1  # 10Hz


@dataclass(frozen=True)
class NAVSIMSample:
    """Processed NAVSIM sample for CalibDrive evaluation."""
    scene_token: str
    front_image: np.ndarray         # (H, W, 3) uint8
    ego_trajectory_gt: np.ndarray   # (40, 3) [x, y, heading] at 10Hz
    ego_speed: float                # m/s
    ego_velocity: np.ndarray        # (2,) [vx, vy]
    driving_command: str            # "left", "straight", "right", "unknown"
    difficulty: str                 # "easy", "medium", "hard"
    metadata: dict


class NAVSIMAdapter:
    """Adapter for loading and processing NAVSIM data.

    Uses NAVSIM's SceneLoader API with correct data access patterns:
    - agent_input.cameras[frame_idx].cam_f0.image for front camera
    - agent_input.ego_statuses[frame_idx].ego_velocity for speed
    - scene.get_future_trajectory() for ground truth
    """

    def __init__(
        self,
        sensor_blobs_path: str = "~/navsim_workspace/dataset/sensor_blobs",
        navsim_log_path: str = "~/navsim_workspace/dataset/navsim_logs",
        split: str = "trainval",
        max_samples: Optional[int] = None,
        scene_filter: Optional[list[str]] = None,
    ):
        if not HAS_NAVSIM:
            raise ImportError(
                "NAVSIM not installed. Run:\n"
                "  git clone https://github.com/autonomousvision/navsim\n"
                "  cd navsim && pip install -e ."
            )

        self.sensor_blobs_path = Path(sensor_blobs_path).expanduser()
        self.navsim_log_path = Path(navsim_log_path).expanduser()
        self.split = split
        self.max_samples = max_samples
        self.scene_filter = scene_filter

    def load(self) -> list[NAVSIMSample]:
        """Load and process NAVSIM scenes via SceneLoader."""
        # Build sensor config: only front camera for efficiency
        sensor_config = SensorConfig.build_no_sensors()
        sensor_config.cam_f0 = [3]  # current frame only (index 3 = latest)

        loader = SceneLoader(
            sensor_blobs_path=str(self.sensor_blobs_path),
            navsim_log_path=str(self.navsim_log_path),
            scene_filter=self.scene_filter,
            sensor_config=sensor_config,
        )

        tokens = loader.get_scene_tokens()
        if self.max_samples:
            tokens = tokens[:self.max_samples]

        samples = []
        for token in tokens:
            try:
                sample = self._process_token(loader, token)
                samples.append(sample)
            except Exception as e:
                print(f"Warning: skipping token {token}: {e}")

        return samples

    def _process_token(self, loader, token: str) -> NAVSIMSample:
        """Convert a NAVSIM scene token to CalibDrive format."""
        agent_input = loader.get_agent_input_from_token(token)

        # Front camera image (latest frame)
        front_image = agent_input.cameras[3].cam_f0.image  # np.ndarray

        # Ego state
        ego_status = agent_input.ego_statuses[3]
        ego_velocity = np.array(ego_status.ego_velocity)
        ego_speed = float(np.linalg.norm(ego_velocity))

        # Driving command
        command_map = {0: "unknown", 1: "left", 2: "straight", 3: "right"}
        driving_command = command_map.get(
            getattr(ego_status, "driving_command", 0), "unknown"
        )

        # Ground truth future trajectory
        scene = loader.get_scene_from_token(token)
        gt_trajectory = scene.get_future_trajectory()  # (40, 3)

        # Classify difficulty
        num_annotations = len(agent_input.cameras[3].cam_f0.image) if front_image is not None else 0
        difficulty = self._classify_difficulty(ego_speed, agent_input)

        return NAVSIMSample(
            scene_token=token,
            front_image=front_image,
            ego_trajectory_gt=gt_trajectory,
            ego_speed=ego_speed,
            ego_velocity=ego_velocity,
            driving_command=driving_command,
            difficulty=difficulty,
            metadata={
                "split": self.split,
                "has_camera": front_image is not None,
            },
        )

    def _classify_difficulty(
        self,
        ego_speed: float,
        agent_input: "AgentInput",
    ) -> str:
        """Classify scene difficulty heuristically."""
        # Fast speed or complex ego history → harder
        ego_statuses = agent_input.ego_statuses
        accel = np.array(ego_statuses[3].ego_acceleration)
        accel_mag = float(np.linalg.norm(accel))

        if ego_speed < 3 and accel_mag < 1:
            return "easy"
        if ego_speed < 10 and accel_mag < 3:
            return "medium"
        return "hard"

    def generate_vla_prompt(
        self,
        sample: NAVSIMSample,
        prompt_style: str = "default",
    ) -> str:
        """Generate a VLA prompt for the given sample."""
        if prompt_style == "openvla":
            return (
                "In: What action should the robot take to drive forward "
                f"at {sample.ego_speed:.1f} m/s safely?\nOut:"
            )
        if prompt_style == "cautious":
            return (
                f"You are driving at {sample.ego_speed:.1f} m/s. "
                f"Driving command: {sample.driving_command}. "
                "Carefully analyze the scene for hazards, then predict "
                "the safest trajectory for the next 4 seconds."
            )
        # default
        return (
            f"You are driving a vehicle at {sample.ego_speed:.1f} m/s. "
            f"Command: {sample.driving_command}. "
            "Predict the future trajectory for the next 4 seconds "
            "as (x, y) waypoints in the ego frame."
        )


def waypoints_to_navsim_trajectory(
    waypoints: np.ndarray,
    source_hz: float = 2.5,
) -> "Trajectory":
    """Convert VLA waypoint predictions to NAVSIM Trajectory format.

    NAVSIM expects 40 poses at 10Hz (4 seconds) with [x, y, heading].

    Args:
        waypoints: (N, 2) predicted waypoints at source_hz.
        source_hz: Prediction frequency of the VLA model.

    Returns:
        NAVSIM Trajectory object.
    """
    if not HAS_NAVSIM:
        raise ImportError("NAVSIM required for trajectory conversion")

    N = len(waypoints)
    if N < 2:
        poses = np.zeros((NAVSIM_NUM_POSES, 3), dtype=np.float32)
        return Trajectory(
            poses=poses,
            trajectory_sampling=TrajectorySampling(
                num_poses=NAVSIM_NUM_POSES,
                interval_length=NAVSIM_DT,
            ),
        )

    # Interpolate to 10Hz
    source_times = np.linspace(0, 4.0, N)
    target_times = np.linspace(NAVSIM_DT, 4.0, NAVSIM_NUM_POSES)

    interp_x = interp1d(
        source_times, waypoints[:, 0],
        kind="linear", fill_value="extrapolate",
    )
    interp_y = interp1d(
        source_times, waypoints[:, 1],
        kind="linear", fill_value="extrapolate",
    )

    x = interp_x(target_times)
    y = interp_y(target_times)

    # Compute heading from trajectory direction
    dx = np.diff(x, prepend=0)
    dy = np.diff(y, prepend=0)
    heading = np.arctan2(dy, dx)

    poses = np.stack([x, y, heading], axis=-1).astype(np.float32)

    return Trajectory(
        poses=poses,
        trajectory_sampling=TrajectorySampling(
            num_poses=NAVSIM_NUM_POSES,
            interval_length=NAVSIM_DT,
        ),
    )


def extract_calibration_signals_from_logits(
    action_logits: np.ndarray,
) -> dict:
    """Extract calibration-relevant signals from VLA action logits.

    OpenVLA uses 256-bin tokenization: each action dimension produces
    a (256,) logit vector. The distribution shape reveals confidence.

    Args:
        action_logits: (num_dims, 256) logit matrix.

    Returns:
        Dict with confidence metrics derived from logits.
    """
    from scipy.special import softmax as sp_softmax
    from scipy.stats import entropy as sp_entropy

    probs = sp_softmax(action_logits, axis=-1)  # (D, 256)

    # Per-dimension max probability
    max_probs = probs.max(axis=-1)  # (D,)

    # Per-dimension entropy
    entropies = sp_entropy(probs, axis=-1)  # (D,)

    # Aggregated confidence: geometric mean of max probs
    confidence_geomean = float(np.exp(np.log(max_probs + 1e-10).mean()))

    # Aggregated confidence: mean max prob
    confidence_mean = float(max_probs.mean())

    # Total entropy
    total_entropy = float(entropies.mean())

    # Distribution spread: std of the probability mass
    spread = float(probs.std(axis=-1).mean())

    # Top-k concentration: fraction of mass in top-5 bins
    top_k = 5
    sorted_probs = np.sort(probs, axis=-1)[:, -top_k:]
    top_k_mass = float(sorted_probs.sum(axis=-1).mean())

    return {
        "confidence_geomean": confidence_geomean,
        "confidence_mean": confidence_mean,
        "total_entropy": total_entropy,
        "per_dim_entropy": entropies.tolist(),
        "per_dim_max_prob": max_probs.tolist(),
        "distribution_spread": spread,
        "top_k_mass": top_k_mass,
    }
