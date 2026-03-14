"""
OpenVLA Driver Model for CalibDrive.

Wraps OpenVLA-7B for driving task evaluation with support for:
- Trajectory prediction from camera images
- Logit extraction for calibration analysis
- MC Dropout uncertainty estimation
- Prompt engineering for driving scenarios

Requires: pip install transformers accelerate torch
Model: openvla/openvla-7b (HuggingFace)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForVision2Seq, AutoProcessor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass(frozen=True)
class DrivingPrediction:
    """Immutable driving prediction with uncertainty info."""
    trajectory: np.ndarray        # (T, 2) predicted waypoints
    confidence: float             # aggregated confidence [0, 1]
    token_logits: Optional[np.ndarray] = None  # raw logits
    token_probs: Optional[np.ndarray] = None   # softmax probabilities


class OpenVLADriver:
    """OpenVLA-7B wrapper for autonomous driving trajectory prediction.

    Loads the model, processes driving scenes, and extracts both
    predictions and calibration-relevant signals (logits, probs).
    """

    # Driving-specific prompt templates
    PROMPTS = {
        "default": (
            "You are driving a vehicle. Observe the current scene and "
            "predict the future trajectory of the ego vehicle for the next "
            "4 seconds. Output waypoints as (x, y) coordinates in meters "
            "relative to the current ego position."
        ),
        "cautious": (
            "You are driving a vehicle in a potentially dangerous scenario. "
            "Carefully analyze the scene for any hazards, then predict the "
            "safest future trajectory for the next 4 seconds as (x, y) "
            "waypoints in meters."
        ),
        "speed_aware": (
            "You are driving at {speed:.1f} m/s. Based on the current scene, "
            "predict where the vehicle should be over the next 4 seconds. "
            "Output (x, y) waypoints in meters from current position."
        ),
    }

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 256,
    ):
        if not HAS_TORCH:
            raise ImportError("torch and transformers required for OpenVLADriver")

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        torch_dtype = getattr(torch, dtype, torch.float16)

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    def predict(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
        speed: float = 0.0,
        return_logits: bool = True,
    ) -> DrivingPrediction:
        """Generate trajectory prediction from a driving scene.

        Args:
            image: Front camera image (H, W, 3) uint8.
            prompt: Text prompt. If None, uses default driving prompt.
            speed: Current ego speed for speed-aware prompts.
            return_logits: Whether to return raw logits.

        Returns:
            DrivingPrediction with trajectory and confidence.
        """
        from PIL import Image

        if prompt is None:
            prompt = self.PROMPTS["default"]

        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=return_logits,
                return_dict_in_generate=True,
            )

        # Decode tokens to text
        generated_tokens = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text_output = self.processor.decode(
            generated_tokens, skip_special_tokens=True
        )

        # Parse trajectory from text
        trajectory = self._parse_trajectory(text_output)

        # Extract confidence from logits
        confidence = 1.0
        token_logits = None
        token_probs = None

        if return_logits and outputs.scores:
            scores = torch.stack(outputs.scores, dim=0)  # (T, 1, vocab)
            probs = torch.softmax(scores, dim=-1)

            # Token-level max probability
            max_probs = probs.max(dim=-1).values.squeeze()  # (T,)

            # Aggregate confidence: geometric mean of token probabilities
            confidence = float(max_probs.log().mean().exp())

            token_logits = scores.squeeze().cpu().numpy()
            token_probs = probs.squeeze().cpu().numpy()

        return DrivingPrediction(
            trajectory=trajectory,
            confidence=confidence,
            token_logits=token_logits,
            token_probs=token_probs,
        )

    def predict_with_mc_dropout(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
        num_samples: int = 20,
        dropout_rate: float = 0.1,
    ) -> dict:
        """MC Dropout prediction with uncertainty estimation.

        Args:
            image: Front camera image.
            prompt: Text prompt.
            num_samples: Number of forward passes.
            dropout_rate: Dropout rate.

        Returns:
            Dict with mean trajectory, std, entropy, confidence.
        """
        # Enable dropout
        self._set_dropout_training(True)

        trajectories = []
        confidences = []

        for _ in range(num_samples):
            pred = self.predict(image, prompt, return_logits=True)
            trajectories.append(pred.trajectory)
            confidences.append(pred.confidence)

        # Disable dropout
        self._set_dropout_training(False)

        trajectories = np.array(trajectories)  # (N, T, 2)
        mean_traj = trajectories.mean(axis=0)
        std_traj = trajectories.std(axis=0)

        variance = std_traj.mean()
        entropy = 0.5 * np.log(2 * np.pi * np.e * (variance ** 2 + 1e-8))

        return {
            "trajectory": mean_traj,
            "confidence": float(np.mean(confidences)),
            "std_trajectory": std_traj,
            "entropy": float(entropy),
            "mc_confidences": np.array(confidences),
            "all_trajectories": trajectories,
        }

    def predict_with_prompt_ensemble(
        self,
        image: np.ndarray,
        speed: float = 0.0,
        prompts: Optional[list[str]] = None,
    ) -> dict:
        """Prompt ensemble: predict with multiple prompts, measure variance.

        Novel UQ method for VLAs: use prompt diversity as uncertainty signal.
        """
        if prompts is None:
            prompts = [
                self.PROMPTS["default"],
                self.PROMPTS["cautious"],
                self.PROMPTS["speed_aware"].format(speed=speed),
                # Paraphrased versions
                "Predict the next 4 seconds of this vehicle's path as x,y coordinates.",
                "As the driver, where will this car be in 4 seconds? Give waypoints.",
            ]

        trajectories = []
        confidences = []

        for prompt in prompts:
            pred = self.predict(image, prompt, return_logits=True)
            trajectories.append(pred.trajectory)
            confidences.append(pred.confidence)

        trajectories = np.array(trajectories)
        mean_traj = trajectories.mean(axis=0)
        std_traj = trajectories.std(axis=0)

        # Prompt ensemble uncertainty: higher variance across prompts = more uncertain
        prompt_disagreement = std_traj.mean()

        return {
            "trajectory": mean_traj,
            "confidence": float(1.0 / (1.0 + prompt_disagreement)),
            "std_trajectory": std_traj,
            "prompt_disagreement": float(prompt_disagreement),
            "per_prompt_confidences": np.array(confidences),
        }

    def _parse_trajectory(
        self,
        text: str,
        num_points: int = 8,
    ) -> np.ndarray:
        """Parse trajectory from model text output.

        Attempts multiple parsing strategies for robustness.
        """
        import re

        trajectory = np.zeros((num_points, 2))

        # Strategy 1: Find (x, y) pairs
        pairs = re.findall(
            r'\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?',
            text
        )
        if pairs:
            for i, (x, y) in enumerate(pairs[:num_points]):
                trajectory[i] = [float(x), float(y)]
            return trajectory

        # Strategy 2: Find numbers in sequence
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if len(numbers) >= 2:
            for i in range(min(len(numbers) // 2, num_points)):
                trajectory[i] = [
                    float(numbers[2 * i]),
                    float(numbers[2 * i + 1]),
                ]
            return trajectory

        # Strategy 3: Return zeros (parsing failed)
        return trajectory

    def _set_dropout_training(self, training: bool):
        """Toggle dropout layers between train/eval mode."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train(training)
        if not training:
            self.model.eval()
