"""EF Regression model: SigLIP embeddings → MLP → EF prediction.

Architecture (v1):
    4 key frames → SigLIP encoder (frozen) → 4 × (embed_dim,) vectors
    → Concatenate → LayerNorm → Linear(4*d, 512) → GELU → Dropout(0.3)
    → Linear(512, 128) → GELU → Dropout(0.2) → Linear(128, 1) → EF%

Architecture (v2 — improved):
    4 key frames → SigLIP encoder (frozen) → 4 × (embed_dim,) vectors
    → Attention-weighted pooling (learn frame importance)
    → LayerNorm → ResidualBlock(d, 512) → ResidualBlock(512, 256)
    → ResidualBlock(256, 128) → Linear(128, 1) → Clamp[0, 100]

Loss: Huber (delta=5.0) — robust to outlier EF annotations.
      + Ordinal boundary penalty (penalizes crossing category boundaries)
      + Clinical asymmetric penalty (higher cost for missing reduced EF)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """FC → GELU → Dropout → FC with residual connection.

    If input_dim != output_dim, uses a learned projection for the skip path.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)
        # Skip connection — project if dimensions differ
        self.skip = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.block(x) + self.skip(x))


class FrameAttentionPooling(nn.Module):
    """Learn which frames matter most for EF prediction.

    Instead of naive concatenation, compute attention weights over frames
    and produce a weighted sum. ED and ES frames should naturally get
    higher weights since they define EF = (EDV - ESV) / EDV.

    Learns a query vector that attends to each frame embedding.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=n_heads, batch_first=True, dropout=0.1,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (batch, num_frames, embed_dim)

        Returns:
            (pooled (batch, embed_dim), attention_weights (batch, num_frames))
        """
        batch = embeddings.size(0)
        q = self.query.expand(batch, -1, -1)  # (B, 1, D)

        pooled, attn_weights = self.attn(
            q, embeddings, embeddings, need_weights=True
        )  # pooled: (B, 1, D), weights: (B, 1, num_frames)

        pooled = self.norm(pooled.squeeze(1))  # (B, D)
        attn_weights = attn_weights.squeeze(1)  # (B, num_frames)

        return pooled, attn_weights


class EFRegressor(nn.Module):
    """MLP regression head for ejection fraction prediction.

    Takes concatenated SigLIP embeddings from 4 key frames and predicts
    a single continuous EF value (0-100%).

    Args:
        embed_dim: Dimension of each SigLIP frame embedding (typically 1152).
        num_frames: Number of key frames per video (typically 4).
        hidden_dim: Hidden layer dimension (default 512).
        dropout: Dropout rate (default 0.3).
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        input_dim = embed_dim * num_frames

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),  # Lighter dropout in second layer
            nn.Linear(128, 1),
        )

        # Initialize output layer with small weights for stable start
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Tensor of shape (batch, num_frames, embed_dim)
                        or pre-flattened (batch, num_frames * embed_dim)

        Returns:
            Tensor of shape (batch,) — predicted EF percentages
        """
        if embeddings.dim() == 3:
            # (batch, num_frames, embed_dim) → (batch, num_frames * embed_dim)
            embeddings = embeddings.flatten(1)

        return self.net(embeddings).squeeze(-1)


class EFRegressorWithMeta(nn.Module):
    """Extended EF regressor that also accepts patient metadata.

    Takes SigLIP embeddings + optional metadata (age, sex) and predicts EF.
    This can improve accuracy since EF norms are age-dependent.

    Args:
        embed_dim: SigLIP embedding dimension
        num_frames: Number of key frames
        meta_dim: Number of metadata features (age, sex_male, sex_female)
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        meta_dim: int = 3,  # age, sex_male, sex_female
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        input_dim = embed_dim * num_frames

        # Image branch
        self.image_branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Metadata branch
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.GELU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 1),
        )

        nn.init.xavier_uniform_(self.fusion[-1].weight, gain=0.1)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: (batch, num_frames, embed_dim) or (batch, num_frames*embed_dim)
            metadata: (batch, meta_dim) — optional patient metadata

        Returns:
            (batch,) predicted EF percentages
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.flatten(1)

        img_feat = self.image_branch(embeddings)

        if metadata is not None:
            meta_feat = self.meta_branch(metadata)
            combined = torch.cat([img_feat, meta_feat], dim=1)
        else:
            # Zero-pad metadata if not provided
            batch_size = embeddings.size(0)
            device = embeddings.device
            meta_feat = self.meta_branch(torch.zeros(batch_size, 3, device=device))
            combined = torch.cat([img_feat, meta_feat], dim=1)

        return self.fusion(combined).squeeze(-1)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 5.0):
    """Huber loss (smooth L1) for EF regression.

    Delta=5.0 means:
      - For errors < 5% EF: MSE-like (quadratic, precise)
      - For errors > 5% EF: MAE-like (linear, robust to outliers)
    """
    return nn.functional.huber_loss(pred, target, delta=delta)


# ═══════════════════════════════════════════════════════════════════════════
# EFRegressorV2 — Improved Architecture
# ═══════════════════════════════════════════════════════════════════════════

class EFRegressorV2(nn.Module):
    """Improved EF regressor with attention pooling + residual blocks.

    Key improvements over v1:
      1. Attention-weighted frame pooling (learns frame importance)
      2. Residual blocks (better gradient flow, deeper network)
      3. Output clamped to [0, 100] (physically valid EF range)
      4. Frame attention weights available for explainability

    Architecture:
        (batch, 4, 1152) → FrameAttentionPooling → (batch, 1152)
        → ResidualBlock(1152, 512) → ResidualBlock(512, 256)
        → ResidualBlock(256, 128) → Linear(128, 1) → Clamp[0, 100]

    Args:
        embed_dim: SigLIP embedding dimension (1152)
        num_frames: Number of key frames (4)
        hidden_dim: First hidden layer dim (512)
        dropout: Dropout rate (0.2)
        attn_heads: Number of attention heads for frame pooling
        clamp_output: Whether to clamp output to [0, 100]
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_frames: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        attn_heads: int = 4,
        clamp_output: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.clamp_output = clamp_output

        # Attention-weighted frame pooling (replaces concat)
        self.frame_pool = FrameAttentionPooling(embed_dim, n_heads=attn_heads)

        # Residual backbone: 1152 → 512 → 256 → 128
        self.backbone = nn.Sequential(
            ResidualBlock(embed_dim, hidden_dim, dropout),
            ResidualBlock(hidden_dim, hidden_dim // 2, dropout),
            ResidualBlock(hidden_dim // 2, hidden_dim // 4, dropout * 0.67),
        )

        # Output head
        self.head = nn.Linear(hidden_dim // 4, 1)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

        # Store last attention weights for explainability
        self._last_frame_attn = None

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: (batch, num_frames, embed_dim) or (batch, num_frames*embed_dim)

        Returns:
            (batch,) predicted EF percentages
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.view(-1, self.num_frames, self.embed_dim)

        # Attention-weighted pooling over frames
        pooled, frame_attn = self.frame_pool(embeddings)
        self._last_frame_attn = frame_attn.detach()

        # Residual backbone
        features = self.backbone(pooled)

        # Output with physical clamp
        ef = self.head(features).squeeze(-1)
        if self.clamp_output:
            ef = ef.clamp(0.0, 100.0)

        return ef

    @property
    def frame_attention_weights(self) -> torch.Tensor | None:
        """Last computed frame attention weights (batch, num_frames).

        Shows which frames (ED, mid-systole, ES, mid-diastole) the model
        considers most important. Useful for explainability.
        """
        return self._last_frame_attn


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced Losses
# ═══════════════════════════════════════════════════════════════════════════

# EF category boundaries for ordinal loss
_CATEGORY_BOUNDARIES = torch.tensor([45.0, 55.0, 70.0])


def ordinal_boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundaries: torch.Tensor | None = None,
    margin: float = 3.0,
) -> torch.Tensor:
    """Penalize predictions that cross EF category boundaries from the target.

    If the true EF is 47% (borderline) and the model predicts 43% (reduced),
    this loss adds an extra penalty because crossing 45% changes the clinical
    category. Predictions that stay within the same category get zero penalty.

    The margin (default 3.0%) creates a "danger zone" near boundaries where
    the penalty ramps up smoothly.

    Args:
        pred: (batch,) predicted EF
        target: (batch,) ground truth EF
        boundaries: category boundary thresholds (default: [45, 55, 70])
        margin: width of the penalty zone around boundaries

    Returns:
        Scalar loss (mean over batch)
    """
    if boundaries is None:
        boundaries = _CATEGORY_BOUNDARIES.to(pred.device)

    total_penalty = torch.zeros_like(pred)

    for b in boundaries:
        # Did pred and target land on different sides of this boundary?
        pred_above = pred > b
        target_above = target > b
        crossed = (pred_above != target_above).float()

        # How far did it cross? Penalty proportional to crossing distance.
        dist_from_boundary = (pred - b).abs()
        # Smooth ramp: zero if far from boundary, linear near it
        penalty = crossed * F.relu(margin - dist_from_boundary + 1.0)

        total_penalty = total_penalty + penalty

    return total_penalty.mean()


def clinical_asymmetric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    under_weight: float = 3.5,
    hyper_weight: float = 2.0,
    danger_threshold: float = 45.0,
    hyper_threshold: float = 70.0,
) -> torch.Tensor:
    """Asymmetric loss with dual dangerous-miss penalties.

    Two dangerous miss types, both clinically harmful:
      1. Missed reduced EF: true < 45% predicted as normal (≥ 45%)  → weight 3.5×
      2. Missed hyperdynamic: true ≥ 70% predicted as normal (< 70%) → weight 2.0×
    Under-detecting reduced EF is the primary concern; hyperdynamic is secondary
    but also important (can indicate sepsis, anaemia, valve disease).

    Args:
        pred: (batch,) predicted EF
        target: (batch,) ground truth EF
        under_weight: multiplier for missed-reduced misses (default 3.5 — raised from 2.0)
        hyper_weight: multiplier for missed-hyperdynamic misses (default 2.0)
        danger_threshold: upper EF boundary of "reduced" category (default 45%)
        hyper_threshold: lower EF boundary of "hyperdynamic" category (default 70%)

    Returns:
        Scalar loss (mean over batch)
    """
    error = (pred - target).abs()
    weights = torch.ones_like(error)

    # Dangerous miss 1: true reduced, predicted normal
    missed_reduced = (target < danger_threshold) & (pred >= danger_threshold)
    weights[missed_reduced] = under_weight

    # Dangerous miss 2: true hyperdynamic, predicted normal
    missed_hyper = (target >= hyper_threshold) & (pred < hyper_threshold)
    # Only override if not already flagged (missed_reduced takes priority)
    weights[missed_hyper & ~missed_reduced] = hyper_weight

    return (error * weights).mean()


def range_penalty_loss(
    pred: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 100.0,
) -> torch.Tensor:
    """Soft penalty for predictions outside the valid EF range [0, 100].

    Even with output clamping, this gradient signal helps push the network
    away from producing extreme pre-clamp values.
    """
    below = F.relu(min_val - pred)
    above = F.relu(pred - max_val)
    return (below + above).mean()


def boundary_push_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    hyper_threshold: float = 70.0,
    reduced_threshold: float = 50.0,
) -> torch.Tensor:
    """Push predictions toward extreme EF class boundaries.

    Addresses hyperdynamic (EF ≥ 70%) and reduced (EF < 50%) recall by
    penalizing predictions that stop short of the clinical boundary:
      - Predicting < 70 when true EF ≥ 70  (missed hyperdynamic)
      - Predicting > 50 when true EF < 50  (missed reduced)

    Args:
        pred: (batch,) predicted EF values
        target: (batch,) ground truth EF values
        hyper_threshold: lower bound of hyperdynamic class (default 70.0%)
        reduced_threshold: upper bound of reduced class (default 50.0%)

    Returns:
        Scalar boundary push penalty
    """
    hyper_mask = target >= hyper_threshold
    reduced_mask = target < reduced_threshold

    hyper_penalty = (
        F.relu(hyper_threshold - pred[hyper_mask]).mean()
        if hyper_mask.any()
        else pred.new_zeros(1).squeeze()
    )
    reduced_penalty = (
        F.relu(pred[reduced_mask] - reduced_threshold).mean()
        if reduced_mask.any()
        else pred.new_zeros(1).squeeze()
    )
    return hyper_penalty * 0.6 + reduced_penalty * 0.6


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    huber_delta: float = 5.0,
    ordinal_weight: float = 0.15,
    asymmetric_weight: float = 0.15,
    range_weight: float = 0.01,
    boundary_push_weight: float = 0.3,
    ordinal_margin: float = 3.0,
    under_detection_weight: float = 3.5,
    hyper_detection_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss with all penalties for EF regression.

    L_total = L_huber + α·L_ordinal + β·L_asymmetric + γ·L_range + δ·L_boundary

    Defaults raised from original (α 0.1→0.15, β 0.05→0.15, δ 0.2→0.3,
    under_weight 2.0→3.5) to better penalise dangerous missed-reduced cases.

    Args:
        pred: (batch,) predicted EF
        target: (batch,) ground truth EF
        huber_delta: Huber delta (default 5.0)
        ordinal_weight: α — weight for ordinal boundary penalty
        asymmetric_weight: β — weight for clinical asymmetric penalty
        range_weight: γ — weight for range penalty
        boundary_push_weight: δ — weight for boundary push (hyperdynamic/reduced)
        ordinal_margin: margin around category boundaries
        under_detection_weight: multiplier for dangerous reduced misses
        hyper_detection_weight: multiplier for dangerous hyperdynamic misses

    Returns:
        (total_loss, loss_dict with individual components)
    """
    l_huber = F.huber_loss(pred, target, delta=huber_delta)
    l_ordinal = ordinal_boundary_loss(pred, target, margin=ordinal_margin)
    l_asymmetric = clinical_asymmetric_loss(
        pred, target,
        under_weight=under_detection_weight,
        hyper_weight=hyper_detection_weight,
    )
    l_range = range_penalty_loss(pred)
    l_boundary = boundary_push_loss(pred, target)

    total = (
        l_huber
        + ordinal_weight * l_ordinal
        + asymmetric_weight * l_asymmetric
        + range_weight * l_range
        + boundary_push_weight * l_boundary
    )

    return total, {
        "huber": l_huber.item(),
        "ordinal": l_ordinal.item(),
        "asymmetric": l_asymmetric.item(),
        "range": l_range.item(),
        "boundary": l_boundary.item(),
        "total": total.item(),
    }


# ---------------------------------------------------------------------------
# Ordinal BCE loss for EFOrdinalNet
# ---------------------------------------------------------------------------

# Per-boundary focal weights: P(>45) over-penalises missed-reduced cases,
# P(>70) over-penalises missed-hyperdynamic (the hardest boundary in this dataset).
_ORDINAL_BOUNDARIES  = [45.0, 55.0, 70.0]
_ORDINAL_BCE_WEIGHTS = [3.0,  2.0,  5.0]   # reduced / borderline / hyper boundaries


def ordinal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    boundaries: list | None = None,
    focal_weights: list | None = None,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Focal-weighted BCE at each clinical EF boundary.

    For each threshold b in [45, 55, 70] %:
        y_b      = (EF > b).float()                  ← binary label per sample
        focal_bce = focal_weight × BCE(logit_b, y_b)  ← hard-example focus
        total    += w_b × mean(focal_bce)

    γ=2 (default) suppresses easy well-classified examples, so the gradient
    concentrates on the boundary-hugging cases that determine ClinAcc.

    Args:
        logits:        (batch, 3)  raw logits for P(EF>45), P(EF>55), P(EF>70)
        targets:       (batch,)    ground-truth EF values
        boundaries:    clinical thresholds (default [45, 55, 70])
        focal_weights: per-boundary loss scale (default [3.0, 2.0, 5.0])
        focal_gamma:   focal exponent γ (default 2.0)

    Returns:
        Scalar loss
    """
    if boundaries is None:
        boundaries = _ORDINAL_BOUNDARIES
    if focal_weights is None:
        focal_weights = _ORDINAL_BCE_WEIGHTS

    total = logits.new_zeros(1).squeeze()
    for i, (b, w) in enumerate(zip(boundaries, focal_weights)):
        y_b   = (targets > b).float()
        log_b = logits[:, i]
        p_b   = torch.sigmoid(log_b)
        # Focal weight: downweight confident correct predictions
        focal_w = torch.where(
            y_b == 1,
            (1.0 - p_b).pow(focal_gamma),   # hard positives
            p_b.pow(focal_gamma),            # hard negatives
        )
        bce   = F.binary_cross_entropy_with_logits(log_b, y_b, reduction="none")
        total = total + w * (focal_w * bce).mean()

    return total


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
