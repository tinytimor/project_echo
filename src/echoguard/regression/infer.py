"""Inference wrapper for the EF regression model.

This module provides the `EFRegressorInference` class that wraps the
full pipeline (SigLIP encoder + MLP head) into a single callable that
accepts raw PIL Images and returns an EF prediction. This is the
interface used by the agentic pipeline.

Usage in agent:
    from echoguard.regression.infer import EFRegressorInference

    ef_model = EFRegressorInference()  # uses PROJECT_ROOT-based defaults
    result = ef_model.predict(frames=[pil_img1, pil_img2, pil_img3, pil_img4])
    # result = {"ef_percent": 62.3, "confidence": 0.85, "category": "normal"}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from echoguard.config import PROJECT_ROOT, ef_category

logger = logging.getLogger(__name__)


class EFRegressorInference:
    """End-to-end EF prediction: PIL frames → SigLIP → MLP → EF%.

    This replaces the VLM text-generation EF prediction in the agentic
    pipeline. It's faster (~50ms vs ~6s), more accurate (proper regression
    vs text token sampling), and immune to mode collapse.

    Architecture:
        Input: 4 PIL Images (ED, mid-systole, ES, mid-diastole)
          ↓
        SigLIP Vision Encoder (frozen, from MedGemma)
          ↓
        4 × (1152,) embedding vectors
          ↓
        Concatenate → MLP → single EF% value

    The SigLIP encoder is loaded once and shared across predictions.
    The MLP head is tiny (~2.4M params) and runs in <1ms.
    """

    def __init__(
        self,
        medgemma_path: str = str(PROJECT_ROOT / "local_models" / "medgemma-4b"),
        checkpoint_path: str = str(PROJECT_ROOT / "checkpoints" / "regression" / "ef_regression_a4c" / "best_model.pt"),
        device: str = "cuda",
        load_vision: bool = True,
    ):
        """Initialize the inference pipeline.

        Args:
            medgemma_path: Path to MedGemma weights (for SigLIP encoder)
            checkpoint_path: Path to trained MLP checkpoint
            device: CUDA device
            load_vision: If False, skip loading SigLIP (for embedding-only inference)
        """
        self.device = device
        self._vision_tower = None
        self._processor = None

        # Load MLP head
        self._load_mlp(checkpoint_path)

        # Load SigLIP encoder
        if load_vision:
            self._load_vision(medgemma_path)

        logger.info("EFRegressorInference ready (device=%s)", device)

    def _load_mlp(self, checkpoint_path: str):
        """Load the trained MLP regression head."""
        from echoguard.regression.model import EFRegressor, EFRegressorWithMeta

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})
        self._embed_dim = ckpt.get("embed_dim", 1152)
        self._num_frames = ckpt.get("num_frames", 4)

        if config.get("use_metadata", False):
            self._model = EFRegressorWithMeta(
                embed_dim=self._embed_dim,
                num_frames=self._num_frames,
                hidden_dim=config.get("hidden_dim", 512),
                dropout=0.0,
            )
        else:
            self._model = EFRegressor(
                embed_dim=self._embed_dim,
                num_frames=self._num_frames,
                hidden_dim=config.get("hidden_dim", 512),
                dropout=0.0,
            )

        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model = self._model.to(self.device).eval()
        self._use_metadata = config.get("use_metadata", False)

        logger.info("MLP head loaded from %s (embed_dim=%d, frames=%d)",
                     checkpoint_path, self._embed_dim, self._num_frames)

    def _load_vision(self, medgemma_path: str):
        """Load the SigLIP vision encoder from MedGemma."""
        from echoguard.regression.extract_embeddings import load_siglip_encoder

        self._vision_tower, self._processor = load_siglip_encoder(medgemma_path)
        logger.info("SigLIP vision encoder loaded")

    @torch.no_grad()
    def predict(
        self,
        frames: Sequence[Image.Image],
        age: float = 8.0,
        sex: str = "U",
    ) -> dict:
        """Predict EF from raw PIL frame images.

        Args:
            frames: 4 PIL Images (ED, mid-systole, ES, mid-diastole)
            age: Patient age in years (for category classification)
            sex: Patient sex ("M"/"F"/"U")

        Returns:
            Dict with:
                - ef_percent: Predicted EF (0-100)
                - confidence: Rough confidence estimate (0-1)
                - category: Clinical category (normal/borderline/reduced/hyperdynamic)
                - age_group: Patient age group
        """
        if self._vision_tower is None:
            raise RuntimeError(
                "Vision encoder not loaded. Initialize with load_vision=True."
            )

        if len(frames) != self._num_frames:
            logger.warning(
                "Expected %d frames, got %d. Padding/truncating.",
                self._num_frames, len(frames),
            )
            if len(frames) < self._num_frames:
                frames = list(frames) + [frames[-1]] * (self._num_frames - len(frames))
            else:
                frames = list(frames[:self._num_frames])

        # Extract SigLIP embeddings
        from echoguard.regression.extract_embeddings import extract_single_video_embedding
        embedding = extract_single_video_embedding(
            self._vision_tower, self._processor, list(frames), self.device,
        )
        # embedding shape: (num_frames, embed_dim)
        embedding = embedding.unsqueeze(0).to(self.device)  # (1, frames, dim)

        # Run MLP
        if self._use_metadata:
            sex_m = 1.0 if sex.upper() == "M" else 0.0
            sex_f = 1.0 if sex.upper() == "F" else 0.0
            meta = torch.tensor([[age, sex_m, sex_f]], dtype=torch.float32, device=self.device)
            ef_pred = self._model(embedding, meta).item()
        else:
            ef_pred = self._model(embedding).item()

        # Clamp to valid range
        ef_pred = max(5.0, min(95.0, ef_pred))

        # Classify
        category = ef_category(ef_pred, age)

        # Rough confidence based on how central the prediction is within
        # its category range (higher = more confident)
        confidence = self._estimate_confidence(ef_pred, age)

        return {
            "ef_percent": round(ef_pred, 1),
            "confidence": round(confidence, 2),
            "category": category,
            "age_group": self._age_group_str(age),
        }

    @torch.no_grad()
    def predict_from_embedding(
        self,
        embedding: torch.Tensor,
        age: float = 8.0,
        sex: str = "U",
    ) -> dict:
        """Predict EF from pre-extracted SigLIP embeddings.

        Useful for batch evaluation or when embeddings are already cached.

        Args:
            embedding: Tensor of shape (num_frames, embed_dim)
            age: Patient age in years
            sex: Patient sex

        Returns:
            Same dict as predict()
        """
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        if self._use_metadata:
            sex_m = 1.0 if sex.upper() == "M" else 0.0
            sex_f = 1.0 if sex.upper() == "F" else 0.0
            meta = torch.tensor([[age, sex_m, sex_f]], dtype=torch.float32, device=self.device)
            ef_pred = self._model(embedding, meta).item()
        else:
            ef_pred = self._model(embedding).item()

        ef_pred = max(5.0, min(95.0, ef_pred))
        category = ef_category(ef_pred, age)
        confidence = self._estimate_confidence(ef_pred, age)

        return {
            "ef_percent": round(ef_pred, 1),
            "confidence": round(confidence, 2),
            "category": category,
            "age_group": self._age_group_str(age),
        }

    def _estimate_confidence(self, ef: float, age: float) -> float:
        """Estimate prediction confidence (heuristic).

        Based on how far the prediction is from category boundaries.
        Predictions deep within a category are more confident.
        """
        from echoguard.config import PEDIATRIC_EF_NORMS, age_group

        group = age_group(age)
        norms = PEDIATRIC_EF_NORMS[group]
        normal_lo, normal_hi = norms["normal"]
        border_lo = norms["borderline"][0]

        # Distance to nearest category boundary
        boundaries = [border_lo, normal_lo, normal_hi]
        min_dist = min(abs(ef - b) for b in boundaries)

        # Sigmoid-like mapping: 0-15% distance → 0.5-0.95 confidence
        confidence = 0.5 + 0.45 * (1.0 - np.exp(-min_dist / 8.0))
        return float(np.clip(confidence, 0.3, 0.95))

    @staticmethod
    def _age_group_str(age: float) -> str:
        from echoguard.config import age_group
        return age_group(age)

    @property
    def vram_usage_mb(self) -> float:
        """Estimate current VRAM usage."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1024**2
