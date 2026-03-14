"""
Driving Dataset Loaders for CalibDrive.

Provides unified data loading for driving benchmarks:
- NAVSIM (simulation)
- nuScenes (real-world)
- CARLA (controllable simulation)

Each loader returns standardized DrivingSample objects.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DrivingSample:
    """Immutable driving sample for evaluation."""
    scene_id: str
    images: np.ndarray              # (num_cameras, H, W, 3)
    ego_trajectory: np.ndarray      # (T, 2) ground truth future trajectory
    ego_speed: np.ndarray           # (T,) ground truth speeds
    ego_heading: np.ndarray         # (T,) ground truth headings
    obstacles: np.ndarray           # (M, 2) obstacle positions
    scenario_type: str              # "normal", "adverse_weather", "long_tail", etc.
    prompt: str                     # text prompt for VLA
    metadata: dict                  # additional info


@dataclass(frozen=True)
class ScenarioCategory:
    """Defines a driving scenario category for evaluation."""
    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"


# Standard scenario categories for CalibDrive benchmark
SCENARIO_CATEGORIES = {
    "normal_driving": ScenarioCategory(
        "normal_driving",
        "Standard highway and urban driving",
        "easy",
    ),
    "adverse_weather": ScenarioCategory(
        "adverse_weather",
        "Rain, fog, snow, night conditions",
        "medium",
    ),
    "construction_zone": ScenarioCategory(
        "construction_zone",
        "Construction zones with altered lanes",
        "medium",
    ),
    "pedestrian_crossing": ScenarioCategory(
        "pedestrian_crossing",
        "Pedestrians in crosswalks or jaywalking",
        "medium",
    ),
    "emergency_vehicle": ScenarioCategory(
        "emergency_vehicle",
        "Emergency vehicles requiring yield",
        "hard",
    ),
    "near_collision": ScenarioCategory(
        "near_collision",
        "Near-miss scenarios requiring emergency braking",
        "hard",
    ),
    "occluded_agent": ScenarioCategory(
        "occluded_agent",
        "Partially occluded agents entering roadway",
        "hard",
    ),
    "unusual_object": ScenarioCategory(
        "unusual_object",
        "Unusual objects on road (debris, animals, fallen cargo)",
        "hard",
    ),
}


class SyntheticDrivingDataset(Dataset):
    """Synthetic driving dataset for pipeline validation.

    Generates controllable driving scenarios with known ground truth
    for testing calibration methods before real data integration.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        prediction_horizon: int = 10,
        num_cameras: int = 1,
        image_size: tuple = (224, 224),
        seed: int = 42,
        difficulty_distribution: Optional[dict] = None,
    ):
        self.num_samples = num_samples
        self.prediction_horizon = prediction_horizon
        self.num_cameras = num_cameras
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

        # Default: 50% easy, 30% medium, 20% hard
        self.difficulty_dist = difficulty_distribution or {
            "easy": 0.5, "medium": 0.3, "hard": 0.2,
        }

        self.samples = self._generate_samples()

    def _generate_samples(self) -> list[DrivingSample]:
        """Generate synthetic driving samples."""
        samples = []
        categories = list(SCENARIO_CATEGORIES.values())

        for i in range(self.num_samples):
            # Select scenario category based on difficulty distribution
            difficulty = self.rng.choice(
                ["easy", "medium", "hard"],
                p=[
                    self.difficulty_dist["easy"],
                    self.difficulty_dist["medium"],
                    self.difficulty_dist["hard"],
                ],
            )
            matching = [c for c in categories if c.difficulty == difficulty]
            category = matching[self.rng.randint(len(matching))]

            # Generate trajectory (sinusoidal + noise)
            t = np.linspace(0, 2 * np.pi, self.prediction_horizon)
            amplitude = self.rng.uniform(1, 10)
            frequency = self.rng.uniform(0.5, 2)

            x = np.cumsum(self.rng.uniform(0.5, 2, self.prediction_horizon))
            y = amplitude * np.sin(frequency * t)
            trajectory = np.stack([x, y], axis=-1)

            # Add difficulty-dependent noise
            noise_scale = {"easy": 0.1, "medium": 0.5, "hard": 1.5}[difficulty]
            trajectory += self.rng.randn(*trajectory.shape) * noise_scale

            speed = np.diff(
                np.linalg.norm(np.diff(trajectory, axis=0, prepend=trajectory[:1]), axis=-1)
            )
            speed = np.concatenate([[5.0], np.abs(speed) + 3.0])

            heading = np.arctan2(
                np.diff(trajectory[:, 1], prepend=trajectory[0, 1]),
                np.diff(trajectory[:, 0], prepend=trajectory[0, 0]),
            )

            # Generate obstacles
            num_obstacles = self.rng.randint(0, 10)
            obstacles = trajectory[self.rng.randint(0, len(trajectory), num_obstacles)] + \
                self.rng.randn(num_obstacles, 2) * 5

            # Synthetic image (random noise — placeholder)
            images = self.rng.randint(
                0, 255,
                (self.num_cameras, *self.image_size, 3),
                dtype=np.uint8,
            )

            prompt = f"Drive safely in {category.name.replace('_', ' ')} scenario."

            sample = DrivingSample(
                scene_id=f"synthetic_{i:06d}",
                images=images,
                ego_trajectory=trajectory.astype(np.float32),
                ego_speed=speed.astype(np.float32),
                ego_heading=heading.astype(np.float32),
                obstacles=obstacles.astype(np.float32),
                scenario_type=category.name,
                prompt=prompt,
                metadata={
                    "difficulty": difficulty,
                    "noise_scale": noise_scale,
                    "category": category.name,
                    "index": i,
                },
            )
            samples.append(sample)

        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_by_scenario(self, scenario_type: str) -> list[DrivingSample]:
        """Get all samples of a specific scenario type."""
        return [s for s in self.samples if s.scenario_type == scenario_type]

    def get_by_difficulty(self, difficulty: str) -> list[DrivingSample]:
        """Get all samples of a specific difficulty level."""
        return [s for s in self.samples if s.metadata["difficulty"] == difficulty]

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        scenarios = {}
        difficulties = {}
        for s in self.samples:
            scenarios[s.scenario_type] = scenarios.get(s.scenario_type, 0) + 1
            d = s.metadata["difficulty"]
            difficulties[d] = difficulties.get(d, 0) + 1

        return {
            "total_samples": self.num_samples,
            "scenarios": scenarios,
            "difficulties": difficulties,
        }
