"""VideoMAE-based EF regression pipeline + Model Garden.

This module provides the regression approach for EF prediction
using frozen VideoMAE spatiotemporal embeddings + lightweight heads
trained with Huber loss.

Architecture:
    AVI video → VideoMAE encoder (frozen) → 16 × 768 embeddings
        → Model Garden → EF prediction

Model Garden:
    model               — EFRegressor (MLP head) baseline + EFRegressorV2 (improved)
    model_garden        — EFMultiTaskModel, EFTemporalTransformer, TCN, EFEnsemble
    train               — MLP/v2 training with class-weighted sampling
    train_garden        — Multi-task / temporal / TCN / MLP training
    evaluate            — MLP evaluation
    evaluate_garden     — Rich evaluation with classification metrics
    extract_videomae    — Run VideoMAE on video frames, save embeddings to disk
    infer               — Inference wrapper for agentic pipeline
    geometric_ef        — DeepLabV3 LV segmentation + area-based EF
"""

from echoguard.regression.model import (
    EFRegressor,
    EFRegressorV2,
    EFRegressorWithMeta,
    ResidualBlock,
    FrameAttentionPooling,
    huber_loss,
    composite_loss,
    ordinal_boundary_loss,
    clinical_asymmetric_loss,
    range_penalty_loss,
)
from echoguard.regression.model_garden import (
    EFMultiTaskModel,
    EFTemporalTransformer,
    EFEnsemble,
    EFExplainer,
    EmbeddingAnalyzer,
    PredictionExplanation,
    multitask_loss,
    create_model,
)
from echoguard.regression.infer import EFRegressorInference

__all__ = [
    # Base models
    "EFRegressor",
    "EFRegressorV2",
    "EFRegressorWithMeta",
    # Building blocks
    "ResidualBlock",
    "FrameAttentionPooling",
    # Model Garden
    "EFMultiTaskModel",
    "EFTemporalTransformer",
    "EFEnsemble",
    "EFExplainer",
    "EmbeddingAnalyzer",
    "PredictionExplanation",
    "multitask_loss",
    "create_model",
    # Inference
    "EFRegressorInference",
    # Loss functions
    "huber_loss",
    "composite_loss",
    "ordinal_boundary_loss",
    "clinical_asymmetric_loss",
    "range_penalty_loss",
]
