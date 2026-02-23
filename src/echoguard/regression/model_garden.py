"""Model Garden: Multi-task, Temporal, and Ensemble architectures for EF prediction.

Extends the base EFRegressor with:
  1. EFMultiTaskModel — shared backbone + regression head + 4-class classifier
  2. EFTemporalTransformer — self-attention over frame sequence → regression
  3. EFEnsemble — fuses multiple model outputs with learned weights + consistency
  4. EFExplainer — generates per-prediction explanations with category probabilities
  5. EFBiLSTM — bidirectional LSTM over 4 key frames (drop-in Transformer replacement)
  6. EFLSTMFullSeq — BiLSTM + attention over ALL frames (~30/clip, variable-length)
  7. EFLSTMCRFPhase — LSTM-CRF that labels cardiac phases then computes EF geometrically

All models consume the same SigLIP embeddings (4 × 1152 CLS tokens) so there is
ZERO extraction overhead — just different heads on the same features.

LSTM variants (supervisor suggestion, Feb 2026):
  - EFBiLSTM:       drop-in benchmark vs Transformer on 4 key frames
  - EFLSTMFullSeq:  stronger temporal model using all frames per clip
  - EFLSTMCRFPhase: removes dependency on pre-labeled ED/ES frames by predicting
                    cardiac phase labels via CRF, then computing EF geometrically
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EF_CATEGORIES = ["reduced", "borderline", "normal", "hyperdynamic"]
N_CLASSES = len(EF_CATEGORIES)
CATEGORY_BOUNDARIES = {
    "reduced": (0, 45),
    "borderline": (45, 55),
    "normal": (55, 70),
    "hyperdynamic": (70, 100),
}


def ef_to_class_index(ef: float) -> int:
    """Map EF percentage to class index for classification head."""
    if ef < 45:
        return 0  # reduced
    elif ef < 55:
        return 1  # borderline
    elif ef <= 70:
        return 2  # normal
    else:
        return 3  # hyperdynamic


# ═══════════════════════════════════════════════════════════════════════════
# 1. Multi-Task Model (Regression + Classification)
# ═══════════════════════════════════════════════════════════════════════════

class EFMultiTaskModel(nn.Module):
    """Joint regression + classification with shared backbone.

    The shared backbone extracts features from concatenated frame embeddings.
    Two heads branch off:
      - Regression head → continuous EF prediction (Huber loss)
      - Classification head → 4-class probability (CE loss)

    The classifier acts as a guardrail: if the regressor says 65% but the
    classifier says P(reduced)=0.87, something is wrong → flag for review.

    Architecture:
        embed (4×1152=4608) → LayerNorm → FC(4608,512) → GELU → Dropout
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                            FC(512,128)→GELU         FC(512,128)→GELU
                            FC(128,1)                FC(128,4)→Softmax
                            → EF%                    → P(class)
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.n_classes = n_classes
        input_dim = embed_dim * num_frames

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Regression head → scalar EF%
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 1),
        )

        # Classification head → P(reduced, borderline, normal, hyperdynamic)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, n_classes),
        )

        # Initialize output layers
        nn.init.xavier_uniform_(self.regression_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.regression_head[-1].bias)
        nn.init.xavier_uniform_(self.classification_head[-1].weight, gain=0.5)
        nn.init.zeros_(self.classification_head[-1].bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            embeddings: (batch, num_frames, embed_dim) or (batch, input_dim)
            return_probs: If True, return (ef_pred, class_logits)

        Returns:
            If return_probs=False: (batch,) EF predictions
            If return_probs=True: ((batch,) EF, (batch, n_classes) logits)
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.flatten(1)

        features = self.backbone(embeddings)
        ef_pred = self.regression_head(features).squeeze(-1)

        if return_probs:
            class_logits = self.classification_head(features)
            return ef_pred, class_logits

        return ef_pred

    def predict_full(
        self, embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full prediction with regression + classification + consistency."""
        ef_pred, class_logits = self.forward(embeddings, return_probs=True)
        class_probs = F.softmax(class_logits, dim=-1)
        class_pred = class_logits.argmax(dim=-1)

        # Consistency check: does regression agree with classification?
        regression_class = torch.tensor(
            [ef_to_class_index(float(ef)) for ef in ef_pred],
            device=embeddings.device,
        )
        consistent = (regression_class == class_pred).float()

        return {
            "ef_percent": ef_pred,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "class_pred": class_pred,
            "consistent": consistent,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Temporal Transformer
# ═══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for frame sequence."""

    def __init__(self, d_model: int, max_len: int = 8):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class EFTemporalTransformer(nn.Module):
    """Self-attention over frame-level SigLIP embeddings.

    Instead of flattening 4 frame embeddings into one vector (losing temporal
    structure), this model treats them as a 4-token sequence. Self-attention
    learns pairwise relationships like "how does the LV change from ED→ES?"
    which is literally what EF measures.

    Architecture (with proj_dim, default 192):
        (T, 1152) → LayerNorm → Linear(1152, proj_dim) → PosEnc
                  → TransformerEncoder(n_layers, d_model=proj_dim)
                  → MeanPool → FC(proj_dim, 256) → GELU → FC(256, 1) → EF%

    The input projection is the critical change vs the original: it reduces the
    transformer from ~32M params (d_model=1152) to ~1.5M (d_model=192), which
    is far more appropriate for the ~2,500 training samples available.
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        proj_dim: int = 192,  # Project 1152→proj_dim before transformer
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.proj_dim = proj_dim

        # Input projection: 1152 → proj_dim (reduces transformer size ~36×)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Positional encoding for cardiac cycle phases
        self.pos_enc = PositionalEncoding(proj_dim, max_len=num_frames)

        # Transformer encoder — now operates on proj_dim, not 1152
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.transformer.enable_nested_tensor = False  # suppress UserWarning (norm_first=True)

        # Regression head after pooling
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

        # Attention weights storage for explainability
        self._attention_weights = None

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: (batch, num_frames, embed_dim)

        Returns:
            (batch,) predicted EF percentages
        """
        if embeddings.dim() == 2:
            # Unflatten if needed: (batch, T*1152) → (batch, T, 1152)
            embeddings = embeddings.view(-1, self.num_frames, self.embed_dim)

        x = self.input_proj(embeddings)   # (batch, T, proj_dim)
        x = self.pos_enc(x)

        # Self-attention over frames
        x = self.transformer(x)           # (batch, T, proj_dim)

        # Mean pooling over frames
        pooled = x.mean(dim=1)            # (batch, proj_dim)

        return self.head(pooled).squeeze(-1)

    def forward_with_attention(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning attention weights for explainability.

        Returns:
            (predictions, list_of_attention_weight_tensors per layer)
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.view(-1, self.num_frames, self.embed_dim)

        x = self.input_proj(embeddings)   # (batch, T, proj_dim)
        x = self.pos_enc(x)

        # Capture attention from each layer
        attention_maps = []
        for layer in self.transformer.layers:
            # Get attention weights
            x_normed = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(
                x_normed, x_normed, x_normed,
                need_weights=True,
                average_attn_weights=True,  # Average across heads
            )
            attention_maps.append(attn_weights.detach())
            # Complete the forward pass of this layer
            x = x + layer.dropout1(attn_out)
            x = x + layer._ff_block(x)

        pooled = x.mean(dim=1)
        preds = self.head(pooled).squeeze(-1)

        return preds, attention_maps


# ═══════════════════════════════════════════════════════════════════════════
# 3. Ensemble Fusion Model
# ═══════════════════════════════════════════════════════════════════════════

class EFEnsemble(nn.Module):
    """Learned fusion of multiple model predictions.

    Combines outputs from MLP regressor, multi-task classifier, and temporal
    transformer through a small learned fusion network. Also performs
    consistency checking across models for confidence estimation.

    Architecture:
        ┌── MLP pred (1) ─────────┐
        ├── MultiTask pred (1) ───┤
        ├── MultiTask probs (4) ──┼──► FC(9, 32) → GELU → FC(32, 1) → EF%
        ├── Temporal pred (1) ────┤       + consistency score
        └── Meta features (2) ────┘       + category probabilities
    """

    def __init__(
        self,
        n_models: int = 3,
        n_class_probs: int = N_CLASSES,
        n_meta: int = 2,  # age, sex_encoded
    ):
        super().__init__()
        # Input: n_models predictions + class_probs + meta
        input_dim = n_models + n_class_probs + n_meta

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Learnable model weights for simple weighted average fallback
        self.model_weights = nn.Parameter(torch.ones(n_models) / n_models)

        nn.init.xavier_uniform_(self.fusion[-1].weight, gain=0.1)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        model_preds: torch.Tensor,
        class_probs: torch.Tensor,
        meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            model_preds: (batch, n_models) — predictions from each model
            class_probs: (batch, n_classes) — classification probabilities
            meta: (batch, n_meta) — age, sex (optional, zeros if missing)

        Returns:
            (batch,) fused EF prediction
        """
        if meta is None:
            meta = torch.zeros(
                model_preds.size(0), 2, device=model_preds.device
            )

        features = torch.cat([model_preds, class_probs, meta], dim=-1)
        return self.fusion(features).squeeze(-1)

    def weighted_average(self, model_preds: torch.Tensor) -> torch.Tensor:
        """Simple weighted average (no learned fusion)."""
        weights = F.softmax(self.model_weights, dim=0)
        return (model_preds * weights.unsqueeze(0)).sum(dim=-1)

    def consistency_score(self, model_preds: torch.Tensor) -> torch.Tensor:
        """How much do the models agree? Lower std → higher consistency."""
        pred_std = model_preds.std(dim=-1)
        # Sigmoid mapping: std of 0 → score of 1.0, std of 10 → score of ~0.2
        return torch.sigmoid(3.0 - pred_std * 0.3)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Losses
# ═══════════════════════════════════════════════════════════════════════════

def multitask_loss(
    ef_pred: torch.Tensor,
    class_logits: torch.Tensor,
    ef_target: torch.Tensor,
    class_target: torch.Tensor,
    huber_delta: float = 5.0,
    classification_weight: float = 0.3,
    class_weights: torch.Tensor | None = None,
    ordinal_weight: float = 0.1,
    asymmetric_weight: float = 0.05,
    range_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined regression + classification loss with clinical penalties.

    L_total = L_composite(ef) + λ * L_CE(class_logits, class_target)

    where L_composite = L_huber + α·L_ordinal + β·L_asymmetric + γ·L_range

    Args:
        ef_pred: (batch,) predicted EF
        class_logits: (batch, n_classes) raw logits
        ef_target: (batch,) ground truth EF
        class_target: (batch,) ground truth class indices
        huber_delta: Huber loss delta
        classification_weight: λ weighting for classification loss
        class_weights: (n_classes,) optional class weights for CE
        ordinal_weight: α for ordinal boundary penalty
        asymmetric_weight: β for clinical asymmetric penalty
        range_weight: γ for range penalty

    Returns:
        (total_loss, loss_dict with individual components)
    """
    from echoguard.regression.model import composite_loss as _composite

    reg_total, reg_details = _composite(
        ef_pred, ef_target,
        huber_delta=huber_delta,
        ordinal_weight=ordinal_weight,
        asymmetric_weight=asymmetric_weight,
        range_weight=range_weight,
    )
    cls_loss = F.cross_entropy(class_logits, class_target, weight=class_weights)

    total = reg_total + classification_weight * cls_loss

    return total, {
        "regression_loss": reg_details["huber"],
        "ordinal_loss": reg_details["ordinal"],
        "asymmetric_loss": reg_details["asymmetric"],
        "classification_loss": cls_loss.item(),
        "total_loss": total.item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Explainability Engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionExplanation:
    """Rich, explainable prediction result from the model garden."""

    video_id: str
    # Regression
    ef_percent: float
    ef_percentile: float  # Where this EF falls in the training distribution
    # Classification
    category: str
    category_probs: dict[str, float]  # P(reduced), P(borderline), ...
    category_confidence: float  # max probability
    # Multi-model
    model_predictions: dict[str, float]  # {"mlp": 38.5, "temporal": 40.1, ...}
    ensemble_consistency: float  # Agreement across models (0-1)
    # Clinical context
    age_years: float
    age_group: str
    normal_ef_range: tuple[float, float]
    deviation_from_normal: float  # How far from normal center
    # Flags
    is_flagged: bool  # True if models disagree or confidence is low
    flag_reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "ef_percent": round(self.ef_percent, 1),
            "ef_percentile": round(self.ef_percentile, 1),
            "category": self.category,
            "category_probabilities": {
                k: round(v, 3) for k, v in self.category_probs.items()
            },
            "category_confidence": round(self.category_confidence, 3),
            "model_predictions": {
                k: round(v, 1) for k, v in self.model_predictions.items()
            },
            "ensemble_consistency": round(self.ensemble_consistency, 3),
            "age_years": round(self.age_years, 1),
            "age_group": self.age_group,
            "normal_ef_range": self.normal_ef_range,
            "deviation_from_normal_center": round(self.deviation_from_normal, 1),
            "flagged": self.is_flagged,
            "flag_reasons": self.flag_reasons,
        }

    def summary(self) -> str:
        """Human-readable clinical summary."""
        lines = [
            f"═══ EchoGuard Prediction: {self.video_id} ═══",
            f"  EF:         {self.ef_percent:.1f}% ({self.category})",
            f"  Percentile: P{self.ef_percentile:.0f}",
            f"  Confidence: {self.category_confidence:.0%}",
            f"  Category probabilities:",
        ]
        for cat, prob in sorted(
            self.category_probs.items(), key=lambda x: -x[1]
        ):
            bar = "█" * int(prob * 20)
            lines.append(f"    {cat:15s} {prob:5.1%} {bar}")

        lines.append(f"  Model agreement: {self.ensemble_consistency:.0%}")
        if self.model_predictions:
            for name, pred in self.model_predictions.items():
                lines.append(f"    {name:15s}: {pred:.1f}%")

        if self.is_flagged:
            lines.append(f"  ⚠️  FLAGGED: {', '.join(self.flag_reasons)}")

        lines.append(
            f"  Normal range (age {self.age_years:.0f}y, {self.age_group}): "
            f"{self.normal_ef_range[0]:.0f}–{self.normal_ef_range[1]:.0f}%"
        )
        return "\n".join(lines)


class EFExplainer:
    """Generates rich explanations for EF predictions.

    Combines outputs from multiple models into a single PredictionExplanation
    with category probabilities, percentiles, consistency checks, and flags.
    """

    def __init__(self, training_ef_distribution: np.ndarray | None = None):
        """
        Args:
            training_ef_distribution: Array of training set EF values for
                                       computing percentiles.
        """
        self.train_efs = training_ef_distribution
        if self.train_efs is not None:
            self.train_efs = np.sort(self.train_efs)

    def compute_percentile(self, ef: float) -> float:
        """Where does this EF fall in the training distribution?"""
        if self.train_efs is None:
            return 50.0
        return float(np.searchsorted(self.train_efs, ef) / len(self.train_efs) * 100)

    def explain(
        self,
        video_id: str,
        ef_pred: float,
        age: float,
        model_predictions: dict[str, float] | None = None,
        class_probs: dict[str, float] | None = None,
        attention_weights: list | None = None,
    ) -> PredictionExplanation:
        """Generate a full explanation for one prediction."""
        from echoguard.config import ef_category, age_group, PEDIATRIC_EF_NORMS

        cat = ef_category(ef_pred, age)
        ag = age_group(age)
        norms = PEDIATRIC_EF_NORMS.get(ag, PEDIATRIC_EF_NORMS["child"])
        normal_range = norms["normal"]
        normal_center = (normal_range[0] + normal_range[1]) / 2
        deviation = ef_pred - normal_center

        # Percentile
        percentile = self.compute_percentile(ef_pred)

        # Category probabilities — from classifier or estimated from regression
        if class_probs is None:
            # Estimate from Gaussian assumption around prediction
            class_probs = self._estimate_category_probs(ef_pred)

        confidence = max(class_probs.values())

        # Consistency
        if model_predictions and len(model_predictions) > 1:
            preds = list(model_predictions.values())
            consistency = self._consistency(preds)
        else:
            consistency = 1.0
            model_predictions = model_predictions or {}

        # Flags
        flags = []
        if confidence < 0.5:
            flags.append(f"Low classification confidence ({confidence:.0%})")
        if consistency < 0.7:
            pred_vals = list(model_predictions.values())
            flags.append(
                f"Model disagreement (std={np.std(pred_vals):.1f}%)"
            )
        if cat == "reduced" and confidence < 0.7:
            flags.append("Reduced EF with uncertain classification — urgent review")
        # Check if regression and classification disagree
        if class_probs:
            classifier_cat = max(class_probs, key=class_probs.get)
            if classifier_cat != cat:
                flags.append(
                    f"Regression→{cat} vs Classifier→{classifier_cat}"
                )

        return PredictionExplanation(
            video_id=video_id,
            ef_percent=ef_pred,
            ef_percentile=percentile,
            category=cat,
            category_probs=class_probs,
            category_confidence=confidence,
            model_predictions=model_predictions,
            ensemble_consistency=consistency,
            age_years=age,
            age_group=ag,
            normal_ef_range=normal_range,
            deviation_from_normal=deviation,
            is_flagged=len(flags) > 0,
            flag_reasons=flags,
        )

    @staticmethod
    def _estimate_category_probs(ef: float, sigma: float = 4.0) -> dict[str, float]:
        """Estimate category probabilities from regression output using Gaussian."""
        probs = {}
        for cat, (lo, hi) in CATEGORY_BOUNDARIES.items():
            # Probability mass in this category's range
            from scipy.stats import norm as sp_norm
            p = sp_norm.cdf(hi, loc=ef, scale=sigma) - sp_norm.cdf(
                lo, loc=ef, scale=sigma
            )
            probs[cat] = float(p)
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs

    @staticmethod
    def _consistency(preds: list[float]) -> float:
        """Compute consistency score from multiple model predictions."""
        if len(preds) < 2:
            return 1.0
        std = np.std(preds)
        # Sigmoid mapping: std=0 → 1.0, std=10 → ~0.2
        return float(1.0 / (1.0 + std / 3.0))


# ═══════════════════════════════════════════════════════════════════════════
# 6. Unsupervised Feature Analysis
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingAnalyzer:
    """Unsupervised analysis of SigLIP embeddings for data understanding.

    While unsupervised learning won't directly improve EF predictions (we have
    labels), it provides critical insights:
      - Cluster structure reveals natural patient groupings
      - Outlier detection identifies unusual studies
      - Embedding quality assessment validates SigLIP features
      - t-SNE/UMAP visualizations for challenge presentation
    """

    @staticmethod
    def compute_statistics(embeddings: torch.Tensor) -> dict:
        """Compute embedding distribution statistics.

        Args:
            embeddings: (N, num_frames, embed_dim) or (N, input_dim)
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.flatten(1)

        return {
            "mean": embeddings.mean(dim=0).numpy(),
            "std": embeddings.std(dim=0).numpy(),
            "global_mean": float(embeddings.mean()),
            "global_std": float(embeddings.std()),
            "n_samples": embeddings.size(0),
            "dim": embeddings.size(1),
            # Inter-sample cosine similarity statistics
            "cosine_mean": float(
                F.cosine_similarity(
                    embeddings[:100].unsqueeze(1),
                    embeddings[:100].unsqueeze(0),
                    dim=-1,
                ).mean()
            ) if len(embeddings) >= 2 else 1.0,
        }

    @staticmethod
    def find_outliers(
        embeddings: torch.Tensor,
        threshold_std: float = 3.0,
    ) -> list[int]:
        """Find outlier samples based on embedding norm."""
        if embeddings.dim() == 3:
            embeddings = embeddings.flatten(1)

        norms = embeddings.norm(dim=1)
        mean_norm = norms.mean()
        std_norm = norms.std()

        outlier_mask = (norms - mean_norm).abs() > threshold_std * std_norm
        return outlier_mask.nonzero(as_tuple=True)[0].tolist()

    @staticmethod
    def cluster_embeddings(
        embeddings: torch.Tensor,
        n_clusters: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """K-means clustering on embeddings.

        Returns (cluster_labels, cluster_centers).
        Useful for visualizing natural patient groupings.
        """
        from sklearn.cluster import KMeans

        if embeddings.dim() == 3:
            embeddings = embeddings.flatten(1)

        X = embeddings.numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        return labels, kmeans.cluster_centers_

    @staticmethod
    def compute_tsne(
        embeddings: torch.Tensor,
        perplexity: float = 30.0,
    ) -> np.ndarray:
        """t-SNE projection for 2D visualization."""
        from sklearn.manifold import TSNE

        if embeddings.dim() == 3:
            embeddings = embeddings.flatten(1)

        X = embeddings.numpy()
        tsne = TSNE(
            n_components=2, perplexity=perplexity,
            random_state=42, n_iter=1000,
        )
        return tsne.fit_transform(X)

    @staticmethod
    def temporal_similarity(embeddings: torch.Tensor) -> dict:
        """Analyze temporal patterns across the 4 cardiac phases.

        Measures how much the embedding changes between frames,
        which correlates with contractile function (EF).

        Args:
            embeddings: (N, 4, embed_dim)
        """
        if embeddings.dim() == 2:
            # Assume 4 frames, reshape
            embed_dim = embeddings.size(1) // 4
            embeddings = embeddings.view(-1, 4, embed_dim)

        # Frame-to-frame cosine similarity
        sims = []
        for i in range(3):
            sim = F.cosine_similarity(
                embeddings[:, i], embeddings[:, i + 1], dim=-1
            )
            sims.append(sim)

        sims = torch.stack(sims, dim=1)  # (N, 3)

        # ED-ES similarity (frame 0 vs frame 2)
        ed_es_sim = F.cosine_similarity(
            embeddings[:, 0], embeddings[:, 2], dim=-1
        )

        return {
            "frame_similarities": sims.numpy(),  # (N, 3)
            "ed_es_similarity": ed_es_sim.numpy(),  # (N,)
            "mean_temporal_change": float(1.0 - sims.mean()),
            "ed_es_change": float(1.0 - ed_es_sim.mean()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Temporal Convolutional Network (TCN)
# ═══════════════════════════════════════════════════════════════════════════

class SqueezeExcitation1d(nn.Module):
    """Channel attention for 1-D feature maps (B, C, T).

    Computes per-channel scaling factors from global average pooling across
    time, allowing the network to emphasise the most informative TCN feature
    channels.  Adds negligible parameters (2 FC layers on C//reduction nodes).
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),         # (B, C, T) → (B, C, 1)
            nn.Flatten(),                    # (B, C)
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — scale each channel by learned attention weight
        return x * self.se(x).unsqueeze(-1)


class EFTemporalCNN(nn.Module):
    """Dilated causal TCN over SigLIP frame embeddings for EF regression.

    Enhancements over baseline TCN:
      - Squeeze-excitation (SE) channel attention after every residual block,
        letting the model up-weight the most discriminative feature channels.
      - Multi-scale temporal pooling: concatenates mean-pool AND max-pool over
        the time axis, capturing both average trend AND peak activation.
        The head receives 2×hidden features instead of hidden.

    Architecture:
        [B, T, D] → input_proj → [B, hidden, T]
                  → N× (dilated residual conv + SE)
                  → mean-pool ⊕ max-pool → [B, 2*hidden]
                  → LayerNorm → Linear(2h→64) → GELU → Linear(64→1) → EF%

    Advantages over EFTemporalTransformer:
      - No positional encoding needed (convolutions are naturally order-aware)
      - More stable on short sequences (T=4 or T=8)
      - SE blocks add <1% extra parameters but consistently improve CNNs
      - Multi-scale pooling captures both trend and peak — critical for EF
        which is defined by the *extremes* of the cardiac cycle
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 8,
        hidden: int = 256,
        num_levels: int = 4,
        dropout: float = 0.1,
        use_se: bool = True,          # SE channel attention in residual blocks
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.use_se = use_se

        # Project from SigLIP embedding space into TCN hidden space
        self.input_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
        )

        # Dilated residual Conv1d blocks (causal: dilation=2^i, pad=dilation)
        self.tcn_blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList() if use_se else None
        for i in range(num_levels):
            dilation = 2 ** i
            self.tcn_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden, hidden,
                        kernel_size=3, padding=dilation, dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                )
            )
            if use_se:
                self.se_blocks.append(SqueezeExcitation1d(hidden, reduction=8))

        # Multi-scale head: concat mean-pool + max-pool → 2*hidden input
        head_in = hidden * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Small gain on final linear to keep initial preds near dataset mean
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, T, D) or (B, T*D) flat — both accepted.
        Returns:
            (B,) EF predictions.
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.view(-1, self.num_frames, self.embed_dim)

        x = self.input_proj(embeddings)  # (B, T, hidden)
        x = x.transpose(1, 2)           # (B, hidden, T)

        for i, block in enumerate(self.tcn_blocks):
            residual = x
            out = block(x)
            out = out[..., : x.size(-1)]      # trim causal padding back to T
            if self.use_se:
                out = self.se_blocks[i](out)  # channel attention
            x = out + residual

        # Multi-scale pooling: mean captures average trend, max captures peak
        mean_pool = x.mean(dim=-1)       # (B, hidden)
        max_pool  = x.max(dim=-1).values # (B, hidden)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*hidden)

        return self.head(pooled).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 8a. BiLSTM — drop-in 4-frame temporal model (supervisor suggestion)
# ═══════════════════════════════════════════════════════════════════════════

class EFBiLSTM(nn.Module):
    """Bidirectional LSTM over 4 key-frame SigLIP embeddings.

    Direct drop-in benchmark against EFTemporalTransformer on the same input.
    BiLSTM processes frames left→right AND right→left simultaneously, so both
    the forward path (ED→mid-sys→ES) and backward path (ES→mid-dia→ED) inform
    the final prediction — mirroring how a cardiologist mentally "plays" the
    cine loop both ways.

    Architecture:
        (T=4, 1152) → LayerNorm → Linear(1152, proj_dim)
                    → BiLSTM(proj_dim, hidden, bidirectional=True)
                    → Attention pool over T outputs
                    → FC(2*hidden, 128) → GELU → FC(128, 1) → EF%

    Trainable params with defaults: ~0.8M  (vs Transformer ~1.5M)
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # Project SigLIP embeddings into LSTM input space
        self.input_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Bidirectional LSTM — processes cardiac cycle forward and backward
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional doubles output

        # Attention pooling over T timesteps
        # Learns which frames (ED, mid-sys, ES, mid-dia) matter most for EF
        self.attn = nn.Linear(lstm_out_dim, 1)

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, T, embed_dim) or (batch, T*embed_dim)
        Returns:
            (batch,) predicted EF%
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.view(-1, self.num_frames, self.embed_dim)

        x = self.input_proj(embeddings)          # (B, T, proj_dim)
        out, _ = self.lstm(x)                    # (B, T, 2*hidden)

        # Attention pooling — softmax over T frames
        attn_w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        pooled = (out * attn_w).sum(dim=1)               # (B, 2*hidden)

        return self.head(pooled).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 8b. LSTM Full Sequence — all frames, variable-length input
# ═══════════════════════════════════════════════════════════════════════════

class EFLSTMFullSeq(nn.Module):
    """BiLSTM + attention over ALL frames in a clip (~20–40 frames).

    Unlike EFBiLSTM which only sees 4 pre-selected key frames, this model
    sees the complete cardiac cine loop. The LSTM can learn to identify the
    ED and ES frames by itself from context rather than relying on pre-labeling.

    This is the model your supervisor was pointing at: LSTMs naturally handle
    variable-length sequences and learn temporal dependencies across the full
    cardiac cycle.

    Input: packed variable-length sequences (use pack_padded_sequence)
    Output: scalar EF%

    Architecture:
        (T_variable, 1152) → project → BiLSTM(2 layers)
                           → scaled-dot-product attention pool → FC head

    Trainable params: ~1.2M (similar to Transformer)
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2

        # Scaled dot-product attention pool over variable T
        self.attn_score = nn.Linear(lstm_out_dim, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, T, embed_dim) — padded if variable length
            lengths:    (batch,) actual sequence lengths, or None for fixed T
        Returns:
            (batch,) predicted EF%
        """
        x = self.input_proj(embeddings)   # (B, T, proj_dim)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
        else:
            out, _ = self.lstm(x)        # (B, T, 2*hidden)

        # Attention pooling with padding mask
        scores = self.attn_score(out)    # (B, T, 1)

        if lengths is not None:
            mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) \
                   >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(scores, dim=1)
        pooled = (out * weights).sum(dim=1)    # (B, 2*hidden)

        return self.head(pooled).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 8c. LSTM-CRF — cardiac phase labeling → EF (supervisor's CRF suggestion)
# ═══════════════════════════════════════════════════════════════════════════

class LinearChainCRF(nn.Module):
    """Minimal linear-chain CRF for cardiac phase sequence labeling.

    States:
        0 = diastole   (filling phase)
        1 = ED         (end-diastole  — maximum LV volume)
        2 = systole    (ejection phase)
        3 = ES         (end-systole   — minimum LV volume)

    Transition constraints encode cardiac physiology:
        diastole → ED → systole → ES → diastole  (one valid cycle)
    Impossible transitions are initialised with a large negative score so the
    CRF must actively learn to override them — it won't produce illegal sequences
    unless the emission evidence overwhelmingly demands it.
    """

    N_TAGS   = 4
    DIASTOLE = 0
    ED       = 1
    SYSTOLE  = 2
    ES       = 3
    TAG_NAMES = ["diastole", "ED", "systole", "ES"]

    def __init__(self):
        super().__init__()
        self.transitions = nn.Parameter(torch.randn(self.N_TAGS, self.N_TAGS) * 0.1)
        with torch.no_grad():
            big = -10.0
            self.transitions[self.ES,      self.ED]       = big
            self.transitions[self.SYSTOLE, self.DIASTOLE] = big
            self.transitions[self.ED,      self.ES]       = big
            self.transitions[self.ED,      self.DIASTOLE] = big
            self.transitions[self.SYSTOLE, self.ED]       = big

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _log_sum_exp(x: torch.Tensor) -> torch.Tensor:
        m = x.max(dim=-1, keepdim=True).values
        return m.squeeze(-1) + (x - m).exp().sum(dim=-1).log()

    # ------------------------------------------------------------------ forward
    def forward_score(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Log-partition function Z via forward algorithm.

        Args:
            emissions: (B, T, N_TAGS)
            mask:      (B, T) bool
        Returns:
            (B,) log-Z
        """
        alpha = emissions[:, 0]
        for t in range(1, emissions.size(1)):
            trans  = self.transitions.unsqueeze(0)
            emit   = emissions[:, t].unsqueeze(1)
            scores = alpha.unsqueeze(2) + trans + emit
            new_alpha = self._log_sum_exp(scores)
            alpha = torch.where(mask[:, t].unsqueeze(1), new_alpha, alpha)
        return self._log_sum_exp(alpha)

    def score_sequence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score a gold tag sequence."""
        B, T, _ = emissions.shape
        score = torch.zeros(B, device=emissions.device)
        for t in range(T):
            score = score + emissions[:, t].gather(
                1, tags[:, t].unsqueeze(1)
            ).squeeze(1) * mask[:, t].float()
            if t < T - 1:
                score = score + self.transitions[
                    tags[:, t], tags[:, t + 1]
                ] * mask[:, t + 1].float()
        return score

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """CRF training loss = log Z − score(gold)."""
        return (self.forward_score(emissions, mask)
                - self.score_sequence(emissions, tags, mask)).mean()

    @torch.no_grad()
    def viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> list[list[int]]:
        """Viterbi decoding → best tag sequence per sample."""
        B, T, C = emissions.shape
        viterbi = emissions[:, 0].clone()
        backpointers: list[torch.Tensor] = []

        for t in range(1, T):
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_prev = scores.max(dim=1)
            new_viterbi = best_scores + emissions[:, t]
            viterbi = torch.where(
                mask[:, t].unsqueeze(1), new_viterbi, viterbi
            )
            backpointers.append(best_prev)

        results = []
        best_last = viterbi.argmax(dim=1)
        for b in range(B):
            seq = [int(best_last[b])]
            for bp in reversed(backpointers):
                seq.append(int(bp[b, seq[-1]]))
            seq.reverse()
            results.append(seq[:int(mask[b].sum())])
        return results


class EFLSTMCRFPhase(nn.Module):
    """LSTM-CRF for cardiac phase labeling → EF.

    Instead of predicting EF directly, this model:
      1. Runs BiLSTM over all frames to produce per-frame emission scores
      2. CRF decodes the globally optimal phase sequence
         (diastole → ED → systole → ES → diastole)
      3. Identifies the ED and ES frames from the decoded sequence
      4. Concatenates LSTM hidden states at those two frames and regresses EF

    This removes the dependency on expert-provided ED/ES frame indices from
    VolumeTracings.csv.  During training, phase labels derived from those
    indices are used to supervise the CRF (teacher forcing on the regression
    head).  At inference, the Viterbi algorithm self-identifies ED/ES.

    Two training loss terms:
        total = λ · CRF_NLL  +  (1−λ) · Huber(EF)

    Phase label construction (see `build_phase_tags()`):
        frame == ed_idx         → ED       (1)
        frame == es_idx         → ES       (3)
        ed_idx < frame < es_idx → systole  (2)
        otherwise               → diastole (0)
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.25,
        crf_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.crf_loss_weight = crf_loss_weight

        self.input_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        lstm_out = hidden_dim * 2

        # Per-frame emission head (→ 4 cardiac phase scores)
        self.emission_head = nn.Linear(lstm_out, LinearChainCRF.N_TAGS)

        # EF regression: concat LSTM outputs at ED and ES frames
        self.ef_head = nn.Sequential(
            nn.LayerNorm(lstm_out * 2),
            nn.Linear(lstm_out * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )
        nn.init.xavier_uniform_(self.ef_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.ef_head[-1].bias)

        self.crf = LinearChainCRF()

    # ------------------------------------------------------------------ helpers
    def _encode(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(embeddings)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x)
        emissions = self.emission_head(lstm_out)
        return lstm_out, emissions

    def _extract_ed_es(
        self,
        lstm_out: torch.Tensor,
        tag_seqs: list[list[int]],
    ) -> torch.Tensor:
        """Extract and concat LSTM hidden states at ED and ES frames."""
        ed_frames, es_frames = [], []
        for b, tags in enumerate(tag_seqs):
            ed_idx = next(
                (i for i, t in enumerate(tags) if t == LinearChainCRF.ED), 0
            )
            es_idx = next(
                (i for i, t in enumerate(tags) if t == LinearChainCRF.ES),
                max(1, len(tags) // 2),
            )
            ed_frames.append(lstm_out[b, ed_idx])
            es_frames.append(lstm_out[b, es_idx])
        return torch.cat(
            [torch.stack(ed_frames), torch.stack(es_frames)], dim=-1
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference: decode phases, identify ED/ES, predict EF.

        Args:
            embeddings: (B, T, embed_dim)
            lengths:    (B,) actual lengths or None
        Returns:
            (B,) EF%
        """
        B, T, _ = embeddings.shape
        if lengths is None:
            lengths = torch.full(
                (B,), T, dtype=torch.long, device=embeddings.device
            )
        mask = (
            torch.arange(T, device=embeddings.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        lstm_out, emissions = self._encode(embeddings, lengths)
        tag_seqs = self.crf.viterbi_decode(emissions, mask)
        combined = self._extract_ed_es(lstm_out, tag_seqs)
        return self.ef_head(combined).squeeze(-1)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        ef_targets: torch.Tensor,
        phase_tags: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Training loss = λ·CRF_NLL + (1−λ)·Huber(EF).

        Args:
            embeddings:  (B, T, embed_dim)
            ef_targets:  (B,) ground truth EF%
            phase_tags:  (B, T) LongTensor phase labels (0–3)
            lengths:     (B,) or None
        Returns:
            (total_loss, metrics_dict)
        """
        B, T, _ = embeddings.shape
        if lengths is None:
            lengths = torch.full(
                (B,), T, dtype=torch.long, device=embeddings.device
            )
        mask = (
            torch.arange(T, device=embeddings.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        lstm_out, emissions = self._encode(embeddings, lengths)
        crf_loss = self.crf.neg_log_likelihood(emissions, phase_tags, mask)

        # Teacher forcing: use ground-truth tags for regression head
        ed_frames, es_frames = [], []
        for b in range(B):
            tags = phase_tags[b, : int(lengths[b])].tolist()
            ed_idx = next(
                (i for i, t in enumerate(tags) if t == LinearChainCRF.ED), 0
            )
            es_idx = next(
                (i for i, t in enumerate(tags) if t == LinearChainCRF.ES),
                max(1, len(tags) // 2),
            )
            ed_frames.append(lstm_out[b, ed_idx])
            es_frames.append(lstm_out[b, es_idx])

        combined = torch.cat(
            [torch.stack(ed_frames), torch.stack(es_frames)], dim=-1
        )
        ef_pred = self.ef_head(combined).squeeze(-1)
        reg_loss = F.huber_loss(ef_pred, ef_targets, delta=5.0)
        total = (
            self.crf_loss_weight * crf_loss
            + (1.0 - self.crf_loss_weight) * reg_loss
        )
        return total, {
            "crf": crf_loss.item(),
            "regression": reg_loss.item(),
            "ef_pred": ef_pred.detach(),
        }

    @staticmethod
    def build_phase_tags(
        n_frames: int,
        ed_idx: int,
        es_idx: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Convert (ed_idx, es_idx) from VolumeTracings.csv into per-frame labels.

        Assignment:
            frame == ed_idx           → ED       (1)
            frame == es_idx           → ES       (3)
            between ed_idx and es_idx → systole  (2)
            otherwise                 → diastole (0)
        """
        tags = torch.zeros(n_frames, dtype=torch.long, device=device)
        lo, hi = min(ed_idx, es_idx), max(ed_idx, es_idx)
        for i in range(n_frames):
            if i == ed_idx:
                tags[i] = LinearChainCRF.ED
            elif i == es_idx:
                tags[i] = LinearChainCRF.ES
            elif lo < i < hi:
                tags[i] = LinearChainCRF.SYSTOLE
        return tags


# ═══════════════════════════════════════════════════════════════════════════
# 9. EFDeltaNet — Explicit ED-ES Delta Architecture
# ═══════════════════════════════════════════════════════════════════════════

class EFDeltaNet(nn.Module):
    """Explicit end-diastole / end-systole delta architecture for EF prediction.

    The 4 key frames are ordered: [ED=0, mid-systole=1, ES=2, mid-diastole=3].
    EF = (EDV − ESV) / EDV, so the *difference* between ED and ES embeddings
    is a direct proxy for stroke volume — which is exactly what EF measures.

    Whereas BiLSTM and Transformer must *discover* this relationship implicitly,
    EFDeltaNet hard-codes it as the primary input feature:

      δ = proj(ED) − proj(ES)          ← stroke-volume direction in feature space
      context = mean(proj(all 4 frames)) ← global cardiac cycle context
      cos_sim = cosine_similarity(ED_raw, ES_raw) ← normalised compression ratio
      ed_mag  = ‖ED_raw‖ / 100               ← EDV cavity size proxy
      es_mag  = ‖ES_raw‖ / 100               ← ESV cavity size proxy

    These are concatenated → MLP → EF %.

    Shared projection weights are applied to each frame independently, so the
    model must learn to project embeddings into a space where ED−ES is
    proportional to stroke volume. This provides a strong inductive bias
    that the standard BiLSTM / Transformer must learn purely from data.

    Architecture (~0.4M params):
        (4, 1152) → shared_proj (each frame) → δ, context (proj_dim each)
                  → concat [δ, context, cos_sim, ed_mag, es_mag] (2*proj+3,)
                  → LN → FC → GELU → Dropout → FC → GELU → FC → EF%
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        ed_frame_idx: int = 0,   # index of end-diastole frame
        es_frame_idx: int = 2,   # index of end-systole frame
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_frames  = num_frames
        self.ed_idx      = ed_frame_idx
        self.es_idx      = es_frame_idx

        # Shared projection applied identically to every frame
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # fusion_dim: δ (proj) + context (proj) + cos_sim (1) + ed_mag (1) + es_mag (1)
        fusion_dim = proj_dim * 2 + 3
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, T, embed_dim) — frames ordered ED, mid-sys, ES, mid-dia
        Returns:
            (B,) EF predictions in [0, 100]
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.view(-1, self.num_frames, self.embed_dim)

        # Project each frame with shared weights: (B, T, embed) → (B, T, proj)
        proj = self.frame_proj(embeddings)

        # Primary signal: ED minus ES — stroke-volume direction
        ed_proj = proj[:, self.ed_idx]   # (B, proj_dim)
        es_proj = proj[:, self.es_idx]   # (B, proj_dim)
        delta   = ed_proj - es_proj      # (B, proj_dim)

        # Context: mean cardiac cycle representation
        context = proj.mean(dim=1)       # (B, proj_dim)

        # Auxiliary scalars computed from raw (un-projected) embeddings
        ed_raw  = embeddings[:, self.ed_idx]   # (B, embed_dim)
        es_raw  = embeddings[:, self.es_idx]   # (B, embed_dim)
        cos_sim = F.cosine_similarity(ed_raw, es_raw, dim=-1, eps=1e-8).unsqueeze(-1)  # (B,1)
        ed_mag  = ed_raw.norm(dim=-1, keepdim=True) / 100.0  # (B,1) — EDV proxy
        es_mag  = es_raw.norm(dim=-1, keepdim=True) / 100.0  # (B,1) — ESV proxy

        features = torch.cat([delta, context, cos_sim, ed_mag, es_mag], dim=-1)
        return self.head(features).squeeze(-1).clamp(0.0, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# 10. EFOrdinalNet — Direct Clinical Boundary Optimisation
# ═══════════════════════════════════════════════════════════════════════════

class EFOrdinalNet(nn.Module):
    """Ordinal regression that directly learns the three clinical EF thresholds.

    Root problem with pure regression: the model predicts ~48% for patients
    with true EF ~29% (reduced) and ~61% for true ~51% (borderline), because
    MSE pulls predictions toward the majority-class mean (~61%).  The model
    never learns to separate minority classes from the dominant 73% "normal."

    Solution: Add a parallel head that predicts three *binary* decisions:
        P(EF > 45%)  — is this patient above the reduced threshold?
        P(EF > 55%)  — is this patient above the borderline threshold?
        P(EF > 70%)  — is this patient above the normal/hyperdynamic threshold?

    Each binary task is much easier to learn than a 4-class problem:
      - P(>45): 92.4% positive in train — easy but essential
      - P(>55): 85.1% positive in train — moderate
      - P(>70): 9.9% positive in train  — rare; focal loss helps greatly

    The focal BCE loss (default γ=2) suppresses easy examples and forces the
    model to concentrate gradient on the hard boundary-crossing patients.

    Architecture (~1.5M params, same as EFTemporalTransformer backbone):
        (4, 1152) → input_proj → PosEnc → TransformerEncoder
                  → mean-pool → pooled (proj_dim,)
                          ┌───────────┴───────────┐
                          ↓ Regression           ↓ Ordinal
                       FC → GELU → FC          FC → GELU → FC(3)
                       → EF% (for MAE)         → [logit_45, logit_55, logit_70]

    Loss = composite_loss(EF_pred, EF_true) + w * ordinal_bce_loss(logits, EF_true)

    Clinical category at inference:
        cat = sum( P(>b) > 0.5 for b in [45, 55, 70] )  → {0,1,2,3}
    """

    BOUNDARIES = [45.0, 55.0, 70.0]

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        proj_dim: int = 192,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_frames = num_frames

        # Shared Transformer backbone (same as EFTemporalTransformer)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_enc = PositionalEncoding(proj_dim, max_len=num_frames)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,     # pre-norm for stable gradients
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.transformer.enable_nested_tensor = False  # norm_first=True disables nested-tensor opt
        self.regression_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1),
        )

        # Head B: ordinal → P(EF>45), P(EF>55), P(EF>70)  [raw logits]
        self.ordinal_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 3),
        )

        nn.init.xavier_uniform_(self.regression_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.regression_head[-1].bias)
        nn.init.xavier_uniform_(self.ordinal_head[-1].weight, gain=0.3)
        # Initialise ordinal biases to reflect dataset marginals:
        #   P(>45) ≈ 92% → logit ≈ +2.4
        #   P(>55) ≈ 85% → logit ≈ +1.7
        #   P(>70) ≈ 10% → logit ≈ −2.2
        with torch.no_grad():
            self.ordinal_head[-1].bias.copy_(torch.tensor([2.4, 1.7, -2.2]))

    def forward(
        self,
        embeddings: torch.Tensor,
        return_ordinal: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        """
        Args:
            embeddings:    (B, T, embed_dim)
            return_ordinal: If True return (ef_pred, ordinal_logits)
        Returns:
            ef_pred: (B,) continuous EF in [0, 100]
            ordinal_logits: (B, 3) logits for P(>45), P(>55), P(>70)
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.view(-1, self.num_frames, self.embed_dim)

        x      = self.input_proj(embeddings)    # (B, T, proj_dim)
        x      = self.pos_enc(x)
        x      = self.transformer(x)
        pooled = x.mean(dim=1)                   # (B, proj_dim)

        ef_pred = self.regression_head(pooled).squeeze(-1).clamp(0.0, 100.0)

        if return_ordinal:
            return ef_pred, self.ordinal_head(pooled)  # (B,), (B, 3)
        return ef_pred

    def predict_category(self, embeddings: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        """Return clinical category index and EF prediction.

        Returns:
            (cat_idx, ef_pred) where cat_idx ∈ {0=reduced, 1=borderline, 2=normal, 3=hyperdynamic}
        """
        ef_pred, logits = self.forward(embeddings, return_ordinal=True)
        probs   = torch.sigmoid(logits)          # (B, 3)
        cat_idx = (probs > 0.5).long().sum(-1)   # (B,) ∈ {0,1,2,3}
        return cat_idx, ef_pred


# ═══════════════════════════════════════════════════════════════════════════
# 11. EFClassificationNet — Direct 4-class BiLSTM classifier
# ═══════════════════════════════════════════════════════════════════════════

# Approximate mean EF per class (used to convert argmax→EF for MAE computation).
# Derived from clinical norms; exact values don't affect ClinAcc at all.
CLASS_MIDPOINTS = [35.0, 50.0, 62.5, 71.5]   # reduced / borderline / normal / hyper


class EFClassificationNet(nn.Module):
    """BiLSTM + 4-class softmax — directly optimises clinical accuracy.

    Identical backbone to EFBiLSTM (B2 winner), but replaces the scalar
    regression head with a 4-class classification head trained with weighted
    cross-entropy (or focal CE).

    Why this works where OrdinalNet didn't:
      - Single loss → no conflicting regression + BCE gradient signals
      - Training is as stable as plain BiLSTM regression
      - Class weights handle the 75/8/8/8% imbalance directly

    Architecture:
        (T=4, 1152) → LayerNorm → Linear(1152, proj_dim)
                    → BiLSTM(proj_dim, hidden, bidirectional=True)
                    → Attention pool over T outputs
                    → FC(2*hidden, 128) → GELU → Dropout → FC(128, 4)

    Predicted EF% for MAE proxy:  CLASS_MIDPOINTS[argmax(logits)]
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Project SigLIP embeddings into LSTM input space (identical to EFBiLSTM)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Bidirectional LSTM (identical to EFBiLSTM)
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional doubles output

        # Attention pooling over T timesteps (identical to EFBiLSTM)
        self.attn = nn.Linear(lstm_out_dim, 1)

        # Classification head — replaces the scalar regression head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, T, embed_dim) or (batch, T*embed_dim)
        Returns:
            logits: (batch, n_classes) — raw logits (no softmax applied)
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.view(-1, self.num_frames, self.embed_dim)

        x = self.input_proj(embeddings)          # (B, T, proj_dim)
        out, _ = self.lstm(x)                    # (B, T, 2*hidden)

        # Attention pooling — softmax over T frames
        attn_w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        pooled = (out * attn_w).sum(dim=1)               # (B, 2*hidden)

        return self.head(pooled)                          # (B, n_classes)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Utility: Model Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_model(
    model_type: str,
    embed_dim: int = 1152,
    num_frames: int = 4,
    **kwargs,
) -> nn.Module:
    """Factory function to create any model in the garden.

    Args:
        model_type: One of "mlp", "v2", "multitask", "temporal", "tcn",
                    "lstm", "lstm_full", "lstm_crf", "delta", "ordinal",
                    "classify", "ensemble"
        embed_dim: SigLIP embedding dimension
        num_frames: Number of key frames

    Returns:
        nn.Module ready for training
    """
    if model_type == "mlp":
        from echoguard.regression.model import EFRegressor
        return EFRegressor(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_type == "v2":
        from echoguard.regression.model import EFRegressorV2
        return EFRegressorV2(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.2),
            attn_heads=kwargs.get("attn_heads", 4),
            clamp_output=kwargs.get("clamp_output", True),
        )
    elif model_type == "multitask":
        return EFMultiTaskModel(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_type == "temporal":
        return EFTemporalTransformer(
            embed_dim=embed_dim,
            num_frames=num_frames,
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            hidden_dim=kwargs.get("hidden_dim", 256),
            dropout=kwargs.get("dropout", 0.3),
            proj_dim=kwargs.get("proj_dim", 192),
        )
    elif model_type == "tcn":
        return EFTemporalCNN(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden=kwargs.get("hidden_dim", 256),
            num_levels=kwargs.get("num_levels", 4),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_type == "lstm":
        return EFBiLSTM(
            embed_dim=embed_dim,
            num_frames=num_frames,
            proj_dim=kwargs.get("proj_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("n_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_type == "lstm_full":
        return EFLSTMFullSeq(
            embed_dim=embed_dim,
            proj_dim=kwargs.get("proj_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("n_layers", 2),
            dropout=kwargs.get("dropout", 0.25),
        )
    elif model_type == "lstm_crf":
        return EFLSTMCRFPhase(
            embed_dim=embed_dim,
            proj_dim=kwargs.get("proj_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_lstm_layers=kwargs.get("n_layers", 2),
            dropout=kwargs.get("dropout", 0.25),
            crf_loss_weight=kwargs.get("crf_loss_weight", 0.3),
        )
    elif model_type == "delta":
        return EFDeltaNet(
            embed_dim=embed_dim,
            num_frames=num_frames,
            proj_dim=kwargs.get("proj_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_type == "ordinal":
        return EFOrdinalNet(
            embed_dim=embed_dim,
            num_frames=num_frames,
            proj_dim=kwargs.get("proj_dim", 192),
            hidden_dim=kwargs.get("hidden_dim", 256),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_type == "classify":
        return EFClassificationNet(
            embed_dim=embed_dim,
            num_frames=num_frames,
            proj_dim=kwargs.get("proj_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("n_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_type == "ensemble":
        return EFEnsemble(
            n_models=kwargs.get("n_models", 3),
            n_class_probs=N_CLASSES,
            n_meta=kwargs.get("n_meta", 2),
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: mlp, v2, multitask, temporal, tcn, "
            f"lstm, lstm_full, lstm_crf, delta, ordinal, classify, ensemble"
        )


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
