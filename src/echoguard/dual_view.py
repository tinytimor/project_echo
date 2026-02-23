"""
echoguard/dual_view.py
======================
Dual-view fusion for EchoGuard-Peds.

Combines an A4C report and a PSAX report into a single authoritative
ClinicalReport using a "conservative consensus" rule that mirrors clinical
practice: when two views disagree on cardiac function, the cardiologist acts
on the more pathological reading.

Clinical rationale
------------------
EchoNet-Pediatric provides both A4C and PSAX videos for the same patients.
The demo showed that on the dangerous silent-miss case (true EF=25.67%):
  - A4C: EF=58%  Z=-1.6  conf=0.52  [moderate]   ← dangerous false negative
  - PSAX: EF=47%  Z=-4.4  conf=0.46  [critical]   ← correct flag

Conservative fusion rule: take the view with the LOWER predicted EF
(i.e. the view detecting worse function) as the primary reading.
When views disagree strongly (|ΔEF| > 10%), surface both and raise uncertainty.

Usage
-----
    from echoguard.inference import EchoGuardInference
    from echoguard.dual_view import DualViewFusion, fuse_views

    engine = EchoGuardInference()
    report_a4c  = engine.run(..., view='A4C')
    report_psax = engine.run(..., view='PSAX')

    fused = fuse_views(report_a4c, report_psax)
    print(fused)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from echoguard.confidence import ConfidenceLevel, ConfidenceResult
from echoguard.inference import ClinicalReport
from echoguard.zscore import ZScoreFlag

logger = logging.getLogger(__name__)

# If the two views' EF estimates differ by more than this, flag disagreement.
_VIEW_DISAGREEMENT_THRESHOLD: float = 10.0


@dataclass
class FusedReport:
    """Combined dual-view clinical output.

    Attributes
    ----------
    primary_report : the report selected as primary (lower EF = worse function)
    secondary_report : the other view's report
    fused_ef : weighted mean of both views' EF estimates
    primary_view : 'A4C' or 'PSAX'
    ef_delta : |EF_A4C − EF_PSAX|
    views_agree : True when |ΔEF| ≤ _VIEW_DISAGREEMENT_THRESHOLD
    fused_confidence : geometric mean of both confidences, further penalised
                       by cross-view disagreement
    summary : human-readable fused narrative
    """

    primary_report: ClinicalReport
    secondary_report: ClinicalReport
    fused_ef: float
    primary_view: str
    ef_delta: float
    views_agree: bool
    fused_confidence: float
    summary: str

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "fused_ef": round(self.fused_ef, 1),
            "primary_view": self.primary_view,
            "ef_delta_pct": round(self.ef_delta, 1),
            "views_agree": self.views_agree,
            "fused_confidence": round(self.fused_confidence, 3),
            "fused_confidence_level": _confidence_level(self.fused_confidence).value,
            "summary": self.summary,
            "primary": self.primary_report.to_dict(),
            "secondary": self.secondary_report.to_dict(),
        }

    def __str__(self) -> str:
        agree_str = "✓ AGREE" if self.views_agree else "⚠ DISAGREE"
        lines = [
            "╔══ EchoGuard-Peds  Dual-View Fusion ══════════════════════════╗",
            f"  Patient:       {self.primary_report.patient_id}",
            f"  Fused EF:      {self.fused_ef:.1f}%  [{self.primary_report.ef_category}]",
            f"  Primary view:  {self.primary_view}  "
            f"(EF {self.primary_report.ef_predicted:.1f}%  ←  lower = more conservative)",
            f"  Secondary:     {self.secondary_report.view}  "
            f"(EF {self.secondary_report.ef_predicted:.1f}%)",
            f"  ΔEF:           {self.ef_delta:.1f}%  {agree_str}",
            f"  Z-score:       {self.primary_report.zscore.z_score:+.2f}"
            f"  [{self.primary_report.zscore.flag.value}]",
            f"  Fused conf:    {self.fused_confidence:.2f}"
            f"  [{_confidence_level(self.fused_confidence).value}]",
            "  ─────────────────────────────────────────────────────────────",
            f"  {self.summary}",
            "╚═════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence_level(score: float) -> ConfidenceLevel:
    if score >= 0.80:
        return ConfidenceLevel.HIGH
    elif score >= 0.60:
        return ConfidenceLevel.MODERATE
    elif score >= 0.35:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.UNRELIABLE


def _geometric_mean(a: float, b: float) -> float:
    return math.sqrt(max(a, 1e-6) * max(b, 1e-6))


def _build_summary(
    primary: ClinicalReport,
    secondary: ClinicalReport,
    fused_ef: float,
    ef_delta: float,
    views_agree: bool,
) -> str:
    """Build the fused clinical narrative."""
    parts = []

    # Lead with the primary (more conservative) finding
    parts.append(primary.zscore.interpretation)

    if views_agree:
        parts.append(
            f"Both {primary.view} (EF {primary.ef_predicted:.1f}%) and "
            f"{secondary.view} (EF {secondary.ef_predicted:.1f}%) are concordant."
        )
    else:
        parts.append(
            f"⚠ Cross-view disagreement: {primary.view} predicts "
            f"EF {primary.ef_predicted:.1f}% vs {secondary.view} "
            f"{secondary.ef_predicted:.1f}% (ΔEF = {ef_delta:.1f}%). "
            f"Conservative estimate ({primary.view}) used as primary; "
            f"repeat acquisition or biplane measurement recommended."
        )

    # Severity action line from primary
    if primary.zscore.flag in (ZScoreFlag.CRITICAL,):
        parts.append("URGENT: EF severely below age-adjusted normal — page cardiologist.")
    elif primary.zscore.flag is ZScoreFlag.REDUCED:
        parts.append("Correlation with clinical status and repeat assessment is advised.")
    elif primary.zscore.flag is ZScoreFlag.BORDERLINE_LOW:
        parts.append("Borderline function — close follow-up advised.")

    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_views(
    a4c: ClinicalReport,
    psax: ClinicalReport,
) -> FusedReport:
    """Fuse A4C and PSAX ClinicalReports into a single conservative estimate.

    Conservative fusion rule
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Select whichever view predicts the *lower* EF (more pathological reading)
    as the primary.  The fused EF is a confidence-weighted mean of both views.
    Cross-view disagreement further penalises the fused confidence.

    Parameters
    ----------
    a4c  : ClinicalReport from the A4C view
    psax : ClinicalReport from the PSAX view

    Returns
    -------
    FusedReport
    """
    ef_a4c = a4c.ef_predicted
    ef_psax = psax.ef_predicted
    ef_delta = abs(ef_a4c - ef_psax)
    views_agree = ef_delta <= _VIEW_DISAGREEMENT_THRESHOLD

    # Primary = the more pathological (lower EF) reading
    if ef_psax <= ef_a4c:
        primary, secondary = psax, a4c
    else:
        primary, secondary = a4c, psax

    # Confidence-weighted fused EF
    w_primary = primary.confidence.overall
    w_secondary = secondary.confidence.overall
    total_w = w_primary + w_secondary
    if total_w > 0:
        fused_ef = (w_primary * primary.ef_predicted + w_secondary * secondary.ef_predicted) / total_w
    else:
        fused_ef = (primary.ef_predicted + secondary.ef_predicted) / 2.0

    # Fused confidence: geometric mean, with cross-view disagreement penalty
    base_conf = _geometric_mean(primary.confidence.overall, secondary.confidence.overall)
    if not views_agree:
        # Penalise by how much they disagree beyond threshold
        excess_delta = ef_delta - _VIEW_DISAGREEMENT_THRESHOLD
        penalty = max(0.0, min(0.4, excess_delta * 0.02))   # up to −40% penalty at ΔEF=30%
        fused_conf = max(0.01, base_conf * (1.0 - penalty))
        logger.info(
            "Cross-view disagreement ΔEF=%.1f%% → confidence penalised %.3f → %.3f",
            ef_delta, base_conf, fused_conf,
        )
    else:
        fused_conf = base_conf

    summary = _build_summary(primary, secondary, fused_ef, ef_delta, views_agree)

    logger.info(
        "Fused [%s]: A4C=%.1f%%  PSAX=%.1f%%  δ=%.1f%%  primary=%s  fused=%.1f%%  conf=%.2f",
        primary.patient_id, ef_a4c, ef_psax, ef_delta, primary.view,
        fused_ef, fused_conf,
    )

    return FusedReport(
        primary_report=primary,
        secondary_report=secondary,
        fused_ef=round(fused_ef, 2),
        primary_view=primary.view,
        ef_delta=round(ef_delta, 1),
        views_agree=views_agree,
        fused_confidence=round(fused_conf, 3),
        summary=summary,
    )


class DualViewFusion:
    """Convenience wrapper: runs both views and fuses them in one call.

    Parameters
    ----------
    engine : EchoGuardInference instance (shared, loads models once)

    Example
    -------
        engine  = EchoGuardInference()
        fusion  = DualViewFusion(engine)
        fused   = fusion.run(
            a4c_embedding_path  = "...",
            psax_embedding_path = "...",
            age=8, sex='M', weight=27, height=130,
            patient_id='PEDS-001',
        )
        print(fused)
    """

    def __init__(self, engine) -> None:
        self._engine = engine

    def run(
        self,
        a4c_embedding_path: Optional[str] = None,
        psax_embedding_path: Optional[str] = None,
        a4c_embedding=None,
        psax_embedding=None,
        age: Optional[float] = None,
        sex: Optional[str] = None,
        weight: Optional[float] = None,
        height: Optional[float] = None,
        patient_id: str = "unknown",
    ) -> FusedReport:
        """Run inference on both views and return a FusedReport."""
        common = dict(age=age, sex=sex, weight=weight, height=height, patient_id=patient_id)
        a4c_report = self._engine.run(
            embedding=a4c_embedding, embedding_path=a4c_embedding_path,
            view='A4C', **common,
        )
        psax_report = self._engine.run(
            embedding=psax_embedding, embedding_path=psax_embedding_path,
            view='PSAX', **common,
        )
        return fuse_views(a4c_report, psax_report)
