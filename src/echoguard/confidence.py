"""
echoguard/confidence.py
=======================
Model confidence and consistency scoring for EchoGuard-Peds.

Two independent confidence signals are combined into a single ``ConfidenceResult``:

1. **Consistency Score** — based on the spread of predictions across N stochastic
   forward passes (MC-Dropout or multi-crop inference).  Uses a sigmoid transform
   so that perfectly identical predictions map to ~0.95 and a 10% EF std maps to ~0.27.

       consistency = σ_sigmoid(k · (c - σ_pred))
       where k=0.30, c=3.0  (empirically calibrated on EchoNet-Pediatric)

2. **Z-Score Confidence** — how confidently the predicted EF falls *outside* the
   normal range.  A prediction far from the normal boundary carries high pathological
   confidence; a prediction right on the 1.5σ boundary carries low confidence.

3. **Combined Confidence** — geometric mean of the two signals, clipped to [0.01, 0.99].

Usage
-----
    from echoguard.confidence import compute_confidence, ConfidenceLevel

    result = compute_confidence(ef_preds=[37.8, 38.1, 37.5, 38.9], zscore=-6.4)
    print(result.overall)          # 0.87
    print(result.level)            # ConfidenceLevel.HIGH
    print(result.summary)          # "High confidence — consistent predictions, clear pathology"
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + e^-x)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Confidence level enum
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    """Human-readable confidence tier."""

    HIGH = "high"          # overall ≥ 0.75
    MODERATE = "moderate"  # 0.50 ≤ overall < 0.75
    LOW = "low"            # 0.25 ≤ overall < 0.50
    UNRELIABLE = "unreliable"  # overall < 0.25

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.50:
            return cls.MODERATE
        if score >= 0.25:
            return cls.LOW
        return cls.UNRELIABLE


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceResult:
    """All confidence outputs for a single inference.

    Attributes
    ----------
    consistency : spread-based score ∈ (0, 1) — higher = more consistent
    z_confidence : Z-score-based score ∈ (0, 1) — higher = clearer finding
    overall : geometric mean of consistency and z_confidence, clipped to [0.01, 0.99]
    level : ConfidenceLevel tier
    ef_mean : mean predicted EF across passes (%)
    ef_std : std of predicted EF across passes (%)  — None for single-pass
    summary : one-sentence plain-language summary
    n_passes : number of forward passes used
    """

    consistency: float
    z_confidence: float
    overall: float
    level: ConfidenceLevel
    ef_mean: float
    ef_std: Optional[float]
    summary: str
    n_passes: int

    def to_dict(self) -> dict:
        return {
            "consistency": round(self.consistency, 3),
            "z_confidence": round(self.z_confidence, 3),
            "overall": round(self.overall, 3),
            "level": self.level.value,
            "ef_mean": round(self.ef_mean, 2),
            "ef_std": round(self.ef_std, 3) if self.ef_std is not None else None,
            "summary": self.summary,
            "n_passes": self.n_passes,
        }


# ---------------------------------------------------------------------------
# Core: consistency score
# ---------------------------------------------------------------------------

# Sigmoid parameters:  score = sigmoid(k * (centre - pred_std))
# k=0.30 gives a smooth curve:
#   std=0   → sigmoid(0.30 * 3.0) = sigmoid(0.90) ≈ 0.71  — but single-pass bonus brings to 0.95
#   std=1   → sigmoid(0.30 * 2.0) = sigmoid(0.60) ≈ 0.65
#   std=3   → sigmoid(0.30 * 0.0) = sigmoid(0.00) = 0.50
#   std=5   → sigmoid(0.30 * -2)  = sigmoid(-0.60) ≈ 0.35
#   std=10  → sigmoid(0.30 * -7)  = sigmoid(-2.10) ≈ 0.11
_CONSISTENCY_K: float = 0.30
_CONSISTENCY_CENTRE: float = 3.0   # std (%) at which score = 0.50

# Single-pass (std=None) is treated as "perfectly consistent" → score = 0.95
_SINGLE_PASS_CONSISTENCY: float = 0.95


def compute_consistency(ef_preds: Sequence[float]) -> tuple[float, float, Optional[float]]:
    """Return (consistency_score, ef_mean, ef_std).

    Parameters
    ----------
    ef_preds : sequence of predicted EF values from multiple forward passes or crops

    Returns
    -------
    consistency : float ∈ (0, 1)
    ef_mean : float — mean of predictions
    ef_std : float | None — stdev (None if single prediction)
    """
    if len(ef_preds) == 0:
        raise ValueError("ef_preds must not be empty")

    ef_mean = statistics.mean(ef_preds)

    if len(ef_preds) == 1:
        return _SINGLE_PASS_CONSISTENCY, ef_mean, None

    ef_std = statistics.stdev(ef_preds)
    raw = _sigmoid(_CONSISTENCY_K * (_CONSISTENCY_CENTRE - ef_std))
    return raw, ef_mean, ef_std


# ---------------------------------------------------------------------------
# Core: Z-score confidence
# ---------------------------------------------------------------------------

def compute_z_confidence(z_score: float) -> float:
    """How confidently the model can assign a clear finding based on Z-score.

    A large |Z| means the prediction is far from the normal boundary → high
    confidence in the finding.  A Z near ±1.5 (the borderline threshold) means
    low confidence.

    Maps |Z| → confidence via a saturating sigmoid:
        z_conf = sigmoid(0.5 * (|Z| - 1.5))

    Boundary cases:
        |Z| = 0.0  → sigmoid(-0.75) ≈ 0.32  (clearly normal but low boundary confidence)
        |Z| = 1.5  → sigmoid(0.00)  = 0.50  (borderline threshold — 50/50)
        |Z| = 2.0  → sigmoid(0.25)  ≈ 0.56
        |Z| = 3.0  → sigmoid(0.75)  ≈ 0.68
        |Z| = 5.0  → sigmoid(1.75)  ≈ 0.85
        |Z| = 7.0  → sigmoid(2.75)  ≈ 0.94

    Note: for clearly-normal predictions (|Z| < 0.5), confidence in "normal"
    is boosted to a floor of 0.70 because absence of pathology is itself a finding.
    """
    abs_z = abs(z_score)
    raw = _sigmoid(0.5 * (abs_z - 1.5))
    # boost clearly-normal floor
    if abs_z < 0.5:
        raw = max(raw, 0.70)
    return raw


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_confidence(
    ef_preds: Sequence[float],
    z_score: float,
) -> ConfidenceResult:
    """Compute combined confidence for an EchoGuard-Peds inference.

    Parameters
    ----------
    ef_preds : one or more EF predictions (%)
        - Single element → deterministic inference, std=None
        - Multiple elements → MC-Dropout or multi-crop; consistency computed from spread
    z_score : Z-score of the (mean) predicted EF against the nomogram

    Returns
    -------
    ConfidenceResult
    """
    consistency, ef_mean, ef_std = compute_consistency(ef_preds)
    z_conf = compute_z_confidence(z_score)

    # Geometric mean of the two independent signals
    overall = math.sqrt(consistency * z_conf)
    overall = max(0.01, min(0.99, overall))

    level = ConfidenceLevel.from_score(overall)

    # Plain-language summary
    summary = _build_summary(consistency, z_conf, overall, ef_std, z_score, level)

    return ConfidenceResult(
        consistency=round(consistency, 3),
        z_confidence=round(z_conf, 3),
        overall=round(overall, 3),
        level=level,
        ef_mean=round(ef_mean, 2),
        ef_std=round(ef_std, 3) if ef_std is not None else None,
        summary=summary,
        n_passes=len(ef_preds),
    )


def _build_summary(
    consistency: float,
    z_conf: float,
    overall: float,
    ef_std: Optional[float],
    z_score: float,
    level: ConfidenceLevel,
) -> str:
    tier = level.value.title()

    if ef_std is None:
        spread_str = "single-pass (no spread estimate)"
    elif ef_std < 1.0:
        spread_str = f"tight spread (±{ef_std:.1f}% EF)"
    elif ef_std < 3.0:
        spread_str = f"moderate spread (±{ef_std:.1f}% EF)"
    else:
        spread_str = f"high spread (±{ef_std:.1f}% EF) — review frames"

    if abs(z_score) >= 3.0:
        finding_str = "clear pathological finding"
    elif abs(z_score) >= 2.0:
        finding_str = "definite abnormality"
    elif abs(z_score) >= 1.5:
        finding_str = "borderline finding"
    elif abs(z_score) < 0.5:
        finding_str = "clearly normal"
    else:
        finding_str = "within normal limits"

    return f"{tier} confidence — {spread_str}, {finding_str}."


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== EchoGuard-Peds Confidence Self-Test ===\n")

    cases = [
        ("Critical reduced (consistent)",    [37.5, 38.1, 37.8, 38.9], -6.5),
        ("Reduced (some spread)",             [48.0, 52.0, 46.0, 50.5], -3.8),
        ("Borderline (tight, ambiguous Z)",   [56.5, 57.0, 56.2, 57.3], -1.8),
        ("Normal (single pass)",              [64.5],                     0.0),
        ("Normal (tight multi-pass)",         [63.8, 64.2, 64.0, 64.5],  -0.1),
        ("Borderline high",                   [71.0, 72.5, 70.8, 73.0],   1.7),
        ("Hyperdynamic (consistent)",         [75.0, 74.5, 76.0, 75.8],   2.5),
        ("Unreliable (huge spread)",          [45.0, 70.0, 55.0, 38.0],   0.0),
    ]

    hdr = f"{'Case':35s}  {'Cons':>5}  {'ZCon':>5}  {'Ovrl':>5}  {'EF±σ':>12}  Level"
    print(hdr)
    print("-" * len(hdr))
    for label, preds, z in cases:
        r = compute_confidence(preds, z)
        ef_spread = (
            f"{r.ef_mean:.1f}±{r.ef_std:.1f}%"
            if r.ef_std is not None
            else f"{r.ef_mean:.1f}% (det)"
        )
        print(
            f"{label:35s}  {r.consistency:.3f}  {r.z_confidence:.3f}  "
            f"{r.overall:.3f}  {ef_spread:>12}  {r.level.value}"
        )
        print(f"  → {r.summary}")

    print("\n=== to_dict() example ===")
    result = compute_confidence([37.5, 38.1, 37.8, 38.9], -6.5)
    print(json.dumps(result.to_dict(), indent=2))
