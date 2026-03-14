"""
VLA Model Wrapper for Driving Tasks.

Provides a unified interface for loading and running VLA models
with support for uncertainty quantification methods.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass(frozen=True)
class DrivingAction:
    """Immutable driving action prediction."""
    trajectory: torch.Tensor       # (T, 2) future waypoints [x, y]
    speed: torch.Tensor            # (T,) target speeds
    heading: torch.Tensor          # (T,) heading angles
    logits: Optional[torch.Tensor] = None  # raw model logits for calibration


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Immutable uncertainty estimate for a driving action."""
    action: DrivingAction
    mean_trajectory: torch.Tensor    # (T, 2) mean predicted trajectory
    std_trajectory: torch.Tensor     # (T, 2) std of predicted trajectory
    entropy: torch.Tensor            # scalar entropy of action distribution
    confidence: float                # aggregated confidence score [0, 1]
    num_samples: int                 # number of forward passes used


class VLADrivingModel(nn.Module):
    """Wrapper around VLA models for driving with uncertainty support."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.dropout_rate = dropout_rate

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

    def forward(
        self,
        images: torch.Tensor,
        prompt: str,
        **kwargs,
    ) -> DrivingAction:
        """Single forward pass producing a driving action."""
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        action_tokens = outputs.sequences[0]
        logits = torch.stack(outputs.scores, dim=0) if outputs.scores else None

        trajectory, speed, heading = self._decode_action(action_tokens)

        return DrivingAction(
            trajectory=trajectory,
            speed=speed,
            heading=heading,
            logits=logits,
        )

    def forward_with_dropout(
        self,
        images: torch.Tensor,
        prompt: str,
        num_samples: int = 10,
    ) -> UncertaintyEstimate:
        """MC Dropout: multiple forward passes with dropout enabled."""
        self._enable_dropout()
        trajectories = []

        for _ in range(num_samples):
            action = self.forward(images, prompt)
            trajectories.append(action.trajectory)

        self._disable_dropout()
        return self._aggregate_predictions(trajectories, num_samples)

    def _enable_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self):
        """Restore eval mode after MC Dropout."""
        self.model.eval()

    def _decode_action(self, tokens: torch.Tensor):
        """Decode model output tokens into driving actions."""
        decoded = self.processor.decode(tokens, skip_special_tokens=True)
        trajectory, speed, heading = self._parse_action_string(decoded)
        return trajectory, speed, heading

    def _parse_action_string(self, action_str: str):
        """Parse action string into trajectory, speed, heading tensors.

        Expected format: "trajectory: x1,y1;x2,y2;... speed: s1,s2,... heading: h1,h2,..."
        Specific parsing logic depends on the VLA model's output format.
        """
        # Default: return zeros (to be overridden per model)
        T = 10  # prediction horizon
        trajectory = torch.zeros(T, 2, device=self.device)
        speed = torch.zeros(T, device=self.device)
        heading = torch.zeros(T, device=self.device)
        return trajectory, speed, heading

    def _aggregate_predictions(
        self,
        trajectories: list[torch.Tensor],
        num_samples: int,
    ) -> UncertaintyEstimate:
        """Aggregate multiple predictions into uncertainty estimate."""
        stacked = torch.stack(trajectories, dim=0)  # (N, T, 2)
        mean_traj = stacked.mean(dim=0)
        std_traj = stacked.std(dim=0)

        # Entropy approximation from trajectory variance
        variance = std_traj.pow(2).sum()
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * variance + 1e-8)

        # Confidence: inverse of normalized variance
        confidence = 1.0 / (1.0 + variance.item())

        return UncertaintyEstimate(
            action=DrivingAction(
                trajectory=mean_traj,
                speed=torch.zeros(mean_traj.shape[0], device=self.device),
                heading=torch.zeros(mean_traj.shape[0], device=self.device),
            ),
            mean_trajectory=mean_traj,
            std_trajectory=std_traj,
            entropy=entropy,
            confidence=confidence,
            num_samples=num_samples,
        )
