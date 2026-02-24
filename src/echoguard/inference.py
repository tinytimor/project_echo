"""
echoguard/inference.py
======================
Unified clinical inference pipeline for EchoGuard-Peds.

Takes a pre-extracted SigLIP embedding (or .pt file path) plus patient
demographics and returns a structured ``ClinicalReport`` with:

- Predicted ejection fraction
- Age / sex / BSA-adjusted Z-score
- Flag severity (normal / borderline / reduced / critical / hyperdynamic)
- Confidence score (consistency + Z-score clarity)
- Human-readable interpretation

Both A4C (Temporal Transformer) and PSAX (TCN) views are supported.
MC-Dropout is available for consistency scoring when ``n_mc_passes > 1``.

Usage
-----
    from echoguard.inference import EchoGuardInference

    engine = EchoGuardInference()      # loads models once, auto-selects GPU

    report = engine.run(
        embedding_path="./data/embeddings_8f/pediatric_a4c/emb_001.pt",
        age=8.0,
        sex="M",
        weight=27.0,
        height=130.0,
        view="A4C",
        patient_id="PEDS-001",
    )
    print(report)
    print(report.to_dict())

For multiple patients:
    reports = engine.run_batch(records)   # records = list of dicts

Standalone CLI:
    python -m echoguard.inference \\
        --embedding path/to/emb.pt \\
        --age 8 --sex M --weight 27 --height 130 --view A4C
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from echoguard.config import ef_category, age_group, PROJECT_ROOT
from echoguard.confidence import ConfidenceResult, compute_confidence
from echoguard.zscore import ZScoreResult, compute_ef_zscore, ZScoreFlag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint root — all paths relative to project root
# ---------------------------------------------------------------------------
_CKPT_ROOT = str(PROJECT_ROOT / "checkpoints")

# ---------------------------------------------------------------------------
# Ensemble specialist specs  (VideoMAE backbone, embed_dim=768)
# ---------------------------------------------------------------------------
# Each view has an ordered list of specialists.  Ensemble weights use a
# composite score:  w = (1/MAE) * (1 + R²) * (1 + ClinAcc)  so that models
# scoring well on accuracy AND correlation dominate the weighted mean.
# The FIRST entry is the "primary" model used when ensemble is disabled.
#
# Architecture roles:
#   tcn       → Temporal Convolutional Network  (dilated causal convolutions)
#   temporal  → Transformer  (multi-head self-attention over frame sequence)
#   multitask → Multi-Task   (joint EF regression + clinical classification)
#   mlp       → MLP Baseline (2-layer on mean-pooled embeddings)
# ---------------------------------------------------------------------------

_ENSEMBLE_SPECS: dict[str, list[dict]] = {
    "A4C": [
        # Primary — best A4C model (VideoMAE TCN)
        {"role": "pattern_matcher",      "model_type": "tcn",       "val_mae": 5.49,
         "val_r2": 0.4368, "val_clin_acc": 0.762,
         "path": f"{_CKPT_ROOT}/regression_videomae_tcn_a4c/garden_tcn_a4c/best_model.pt"},
        # Second specialist — temporal transformer
        {"role": "motion_analyst",       "model_type": "temporal",  "val_mae": 5.78,
         "val_r2": 0.3622, "val_clin_acc": 0.741,
         "path": f"{_CKPT_ROOT}/regression_videomae_a4c/garden_temporal_a4c/best_model.pt"},
        # Third specialist — independent categorical head prevents anchoring
        {"role": "guardrail_classifier", "model_type": "multitask",  "val_mae": 6.14,
         "val_r2": 0.3153, "val_clin_acc": 0.679,
         "path": f"{_CKPT_ROOT}/regression_videomae_multitask_a4c/garden_multitask_a4c/best_model.pt"},
        # Baseline for maximum disagreement signal
        {"role": "sonographer_baseline", "model_type": "mlp",        "val_mae": 6.55,
         "val_r2": 0.2703, "val_clin_acc": 0.670,
         "path": f"{_CKPT_ROOT}/regression_videomae_mlp_a4c/garden_mlp_a4c/best_model.pt"},
    ],
    "PSAX": [
        # Primary — best PSAX model (VideoMAE Temporal)
        {"role": "motion_analyst",       "model_type": "temporal",  "val_mae": 5.08,
         "val_r2": 0.5363, "val_clin_acc": 0.748,
         "path": f"{_CKPT_ROOT}/regression_videomae_psax/garden_temporal_psax/best_model.pt"},
        # Second — TCN (nearly identical performance)
        {"role": "pattern_matcher",      "model_type": "tcn",       "val_mae": 5.14,
         "val_r2": 0.4988, "val_clin_acc": 0.746,
         "path": f"{_CKPT_ROOT}/regression_videomae_tcn_psax/garden_tcn_psax/best_model.pt"},
        # Third — categorical guardrail
        {"role": "guardrail_classifier", "model_type": "multitask",  "val_mae": 5.43,
         "val_r2": 0.4803, "val_clin_acc": 0.696,
         "path": f"{_CKPT_ROOT}/regression_videomae_multitask_psax/garden_multitask_psax/best_model.pt"},
        # Baseline
        {"role": "sonographer_baseline", "model_type": "mlp",        "val_mae": 5.64,
         "val_r2": 0.4394, "val_clin_acc": 0.672,
         "path": f"{_CKPT_ROOT}/regression_videomae_mlp_psax/garden_mlp_psax/best_model.pt"},
    ],
}

# Kept for backward-compatibility
_DEFAULT_CHECKPOINTS = {
    view: specs[0] for view, specs in _ENSEMBLE_SPECS.items()
}

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClinicalReport:
    """Structured output of a single EchoGuard-Peds inference.

    Attributes
    ----------
    patient_id : caller-supplied identifier
    view : echocardiographic view ('A4C' or 'PSAX')
    ef_predicted : predicted ejection fraction (%)
    ef_category : 'normal' | 'borderline' | 'reduced' | 'hyperdynamic'
    zscore : ZScoreResult object with Z-score, flag, normal range, etc.
    confidence : ConfidenceResult with overall score, level, consistency
    age : patient age in years
    sex : 'M' / 'F' / None
    weight : weight in kg (None if not provided)
    height : height in cm (None if not provided)
    bsa : body-surface area m² (None if anthropometrics not available)
    n_mc_passes : number of forward passes used for consistency scoring
    model_version : checkpoint path used
    timestamp : ISO-8601 UTC timestamp
    interpretation : concise multi-sentence clinical plain-language summary
    """

    patient_id: str
    view: str
    ef_predicted: float
    ef_category: str
    zscore: ZScoreResult
    confidence: ConfidenceResult
    age: Optional[float]
    sex: Optional[str]
    weight: Optional[float]
    height: Optional[float]
    bsa: Optional[float]
    n_mc_passes: int
    model_version: str
    timestamp: str
    interpretation: str
    # Specialist breakdown — one entry per model that ran
    # e.g. {'temporal': 38.1, 'tcn': 37.6, 'multitask': 39.2, 'mlp': 40.5}
    model_predictions: dict = field(default_factory=dict)
    models_used: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_normal(self) -> bool:
        return self.ef_category == "normal"

    @property
    def flag(self) -> ZScoreFlag:
        return self.zscore.flag

    @property
    def requires_attention(self) -> bool:
        return self.zscore.flag.requires_attention or self.ef_category != "normal"

    def to_dict(self) -> dict:
        d = {
            "patient_id": self.patient_id,
            "view": self.view,
            "ef_predicted": round(self.ef_predicted, 1),
            "ef_category": self.ef_category,
            "zscore": self.zscore.to_dict(),
            "confidence": self.confidence.to_dict(),
            "demographics": {
                "age": self.age,
                "sex": self.sex,
                "weight_kg": self.weight,
                "height_cm": self.height,
                "bsa_m2": round(self.bsa, 3) if self.bsa else None,
            },
            "n_mc_passes": self.n_mc_passes,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "interpretation": self.interpretation,
        }
        if self.model_predictions:
            d["specialist_roundtable"] = {
                role: round(ef, 2)
                for role, ef in self.model_predictions.items()
            }
            d["models_used"] = self.models_used
        return d

    def __str__(self) -> str:
        lines = [
            f"╔══ EchoGuard-Peds Clinical Report ═══════════════════════════╗",
            f"  Patient:     {self.patient_id}",
            f"  View:        {self.view}",
            f"  EF:          {self.ef_predicted:.1f}%  [{self.ef_category}]",
            f"  Z-score:     {self.zscore.z_score:+.2f}  [{self.zscore.flag.value}]",
            f"  Normal rng:  {self.zscore.normal_range[0]:.1f}–{self.zscore.normal_range[1]:.1f}%",
            f"  Confidence:  {self.confidence.overall:.2f}  [{self.confidence.level.value}]",
        ]
        if self.bsa:
            lines.append(f"  BSA:         {self.bsa:.2f} m²")
        if self.model_predictions:
            lines.append(f"  Specialists:")
            for role, ef in self.model_predictions.items():
                lines.append(f"    {role:<25s} → EF {ef:.1f}%")
            std = _prediction_std(list(self.model_predictions.values()))
            lines.append(f"  Agreement:   σ={std:.1f}%  (n={len(self.model_predictions)})")
        lines += [
            f"  ─────────────────────────────────────────────────────────────",
            f"  {self.interpretation}",
            f"╚═════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prediction_std(preds: list[float]) -> float:
    """Population std of a list of EF predictions (%).  Returns 0.0 for singletons."""
    import statistics
    return statistics.pstdev(preds) if len(preds) > 1 else 0.0


def _weighted_mean(preds: list[float], weights: list[float]) -> float:
    """Weighted mean: sum(w*v) / sum(w)."""
    total_w = sum(weights)
    return sum(w * v for w, v in zip(weights, preds)) / total_w


# If a specialist's prediction is more than this many EF-% away from the
# trimmed mean of the remaining specialists, it is excluded from the weighted
# mean used for the final EF estimate.  It is still included in ef_preds_for_conf
# so the inter-model σ (and therefore confidence) is correctly penalised.
_OUTLIER_THRESHOLD: float = 15.0


def _robust_weighted_mean(
    preds: list[float], weights: list[float], outlier_threshold: float = 15.0
) -> float:
    """Weighted mean with single-outlier exclusion.

    Algorithm
    ---------
    1. Compute the unweighted median of all predictions.
    2. If exactly one prediction is more than ``outlier_threshold`` EF-% away
       from the median, exclude it from the weighted mean.
    3. If two or more are outliers (or none are), use all predictions.

    The excluded prediction still contributes to the σ-based confidence score —
    the outlier itself is the signal of uncertainty.
    """
    import statistics
    if len(preds) <= 2:
        return _weighted_mean(preds, weights)

    median = statistics.median(preds)
    outlier_mask = [abs(p - median) > outlier_threshold for p in preds]
    n_outliers = sum(outlier_mask)

    if n_outliers == 1:
        # Exactly one outlier → exclude it from the mean
        filtered_preds = [p for p, out in zip(preds, outlier_mask) if not out]
        filtered_weights = [w for w, out in zip(weights, outlier_mask) if not out]
        outlier_val = next(p for p, out in zip(preds, outlier_mask) if out)
        logger.debug(
            "Outlier clipped from weighted mean: %.1f%%  (median=%.1f, threshold=%.1f)",
            outlier_val, median, outlier_threshold,
        )
        return _weighted_mean(filtered_preds, filtered_weights)

    # 0 or 2+ outliers → use all (cannot reliably identify which to drop)
    return _weighted_mean(preds, weights)


# ---------------------------------------------------------------------------
# Internal: build clinical interpretation paragraph
# ---------------------------------------------------------------------------

def _build_clinical_interpretation(
    ef: float,
    category: str,
    zscore: ZScoreResult,
    confidence: ConfidenceResult,
    age: Optional[float],
    sex: Optional[str],
    view: str,
) -> str:
    parts: list[str] = []

    # Primary finding
    parts.append(zscore.interpretation)

    # Category context
    if category == "reduced":
        parts.append(
            "This is consistent with reduced systolic function; "
            "correlation with clinical status and repeat assessment is recommended."
        )
    elif category == "critical" or zscore.flag is ZScoreFlag.CRITICAL:
        parts.append(
            "Severely reduced EF warrants urgent clinical evaluation."
        )
    elif category == "borderline":
        parts.append("Borderline function — close follow-up advised.")
    elif category == "hyperdynamic":
        parts.append(
            "Hyperdynamic function should be correlated with volume status and clinical context."
        )
    else:
        parts.append("No evidence of systolic dysfunction detected.")

    # Confidence qualifier
    if confidence.level.value == "unreliable":
        parts.append(
            f"⚠ Prediction confidence is low ({confidence.overall:.0%}) — "
            f"high inter-pass variability suggests poor image quality or unusual morphology; "
            f"manual review strongly recommended."
        )
    elif confidence.level.value == "low":
        parts.append(
            f"Prediction confidence is limited ({confidence.overall:.0%}); "
            f"consider additional views or repeat acquisition."
        )

    # View note
    parts.append(f"[{view} view]")

    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------

class EchoGuardInference:
    """Load-once, infer-many inference engine.

    Parameters
    ----------
    checkpoints : dict mapping view name → list of specialist specs, or view → single path.
                  Leave None to use _ENSEMBLE_SPECS defaults.
    device : 'cuda', 'cpu', or 'auto' (default: 'auto')
    use_ensemble : if True (default), run ALL specialists and compute real inter-model
                   consistency score.  If False, run only the primary (best) model.
    """

    def __init__(
        self,
        checkpoints: Optional[dict] = None,
        device: str = "auto",
        use_ensemble: bool = True,
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        self._use_ensemble = use_ensemble

        # Build per-view specialist specs
        self._specs: dict[str, list[dict]] = {
            view: list(specs) for view, specs in _ENSEMBLE_SPECS.items()
        }
        if checkpoints:
            for view, override in checkpoints.items():
                v = view.upper()
                if isinstance(override, str):
                    # Caller supplied a single path — replace primary, keep others
                    if v in self._specs:
                        self._specs[v][0] = dict(self._specs[v][0], path=override)
                    else:
                        self._specs[v] = [{"role": "custom", "model_type": "temporal",
                                           "val_mae": 6.0, "path": override}]
                elif isinstance(override, list):
                    self._specs[v] = override

        # model cache: keyed by (view, role)
        self._models: dict[tuple[str, str], nn.Module] = {}
        self._model_versions: dict[str, str] = {}   # view → primary path
        logger.info("EchoGuardInference initialised  device=%s  ensemble=%s",
                    self._device, use_ensemble)

    # ------------------------------------------------------------------ #
    # Model loading (lazy, keyed by view+role)                            #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self, view: str) -> None:
        """Lazy-load all available specialist models for a view."""
        from echoguard.regression.evaluate_garden import load_garden_model

        specs = self._specs.get(view, [])
        if not specs:
            raise ValueError(f"No specialists configured for view '{view}'")

        newly_loaded = 0
        for spec in specs:
            key = (view, spec["role"])
            if key in self._models:
                continue  # already cached from a previous call
            path = Path(spec["path"])
            if not path.exists():
                logger.warning("Specialist checkpoint missing, skipping: %s", path)
                continue
            try:
                model, ckpt = load_garden_model(path, self._device)
                self._models[key] = model
                if view not in self._model_versions:
                    self._model_versions[view] = str(path)
                newly_loaded += 1
                logger.info(
                    "  ✓ Loaded %s (%s)  val_mae=%.2f%%  epoch=%d",
                    spec["role"], spec["model_type"],
                    spec.get("val_mae", 0), ckpt.get("epoch", -1),
                )
            except Exception as exc:
                logger.warning("Could not load specialist %s: %s", spec["role"], exc)

        # Check that at least one model is available (loaded now OR previously cached)
        available = sum(1 for s in specs if (view, s["role"]) in self._models)
        if available == 0:
            raise RuntimeError(
                f"No specialist models could be loaded for view={view}. "
                f"Run training first."
            )
        if newly_loaded == 0:
            logger.debug("All specialists for view=%s already cached (%d models)", view, available)

    def _get_active_specs(self, view: str) -> list[dict]:
        """Return only the specs whose checkpoint is loaded."""
        return [
            s for s in self._specs.get(view, [])
            if (view, s["role"]) in self._models
        ]

    # ------------------------------------------------------------------ #
    # Forward pass helpers                                                 #
    # ------------------------------------------------------------------ #

    def _forward_once(self, model: nn.Module, embedding: torch.Tensor) -> float:
        """Single deterministic forward pass (eval mode, no dropout)."""
        model.eval()
        with torch.no_grad():
            emb = embedding.unsqueeze(0).to(self._device)  # (1, T, D)
            pred = model(emb)
            return float(pred.squeeze().cpu().item())

    def _forward_mc(
        self, model: nn.Module, embedding: torch.Tensor, n_passes: int
    ) -> list[float]:
        """MC-Dropout forward passes over a single model."""
        def _enable_dropout(m: nn.Module) -> None:
            if isinstance(m, nn.Dropout):
                m.train()

        model.eval()
        model.apply(_enable_dropout)
        preds = []
        with torch.no_grad():
            emb = embedding.unsqueeze(0).to(self._device)
            for _ in range(n_passes):
                preds.append(float(model(emb).squeeze().cpu().item()))
        model.eval()
        return preds

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        embedding: Optional[torch.Tensor] = None,
        embedding_path: Optional[str | Path] = None,
        age: Optional[float] = None,
        sex: Optional[str] = None,
        weight: Optional[float] = None,
        height: Optional[float] = None,
        view: str = "A4C",
        patient_id: str = "unknown",
        n_mc_passes: int = 1,
    ) -> ClinicalReport:
        """Run the full Clinical Roundtable for a single patient.

        When ``use_ensemble=True`` (default), ALL specialist models run on the
        same embedding and their predictions are aggregated using inverse-MAE
        weighting.  The spread across models becomes the consistency score.

        When ``use_ensemble=False``, only the primary (best) model runs.
        ``n_mc_passes > 1`` then enables MC-Dropout on that single model.

        Parameters
        ----------
        embedding : (num_frames, embed_dim) float32 tensor
        embedding_path : path to a .pt file (mutually exclusive with embedding)
        age : years
        sex : 'M' or 'F'
        weight : kg
        height : cm
        view : 'A4C' or 'PSAX'
        patient_id : caller-supplied identifier
        n_mc_passes : MC-Dropout passes (only used when use_ensemble=False)
        """
        view = view.upper()

        # ---- Load embedding ----
        if embedding is None and embedding_path is None:
            raise ValueError("Provide either 'embedding' or 'embedding_path'.")
        if embedding_path is not None:
            embedding = torch.load(str(embedding_path), weights_only=True)
        assert isinstance(embedding, torch.Tensor), "embedding must be a torch.Tensor"

        # ---- Lazy-load all specialists for this view ----
        self._ensure_loaded(view)
        active_specs = self._get_active_specs(view)

        # ---- Specialist Roundtable (Layer 1 + 2) ----
        if self._use_ensemble and len(active_specs) > 1:
            # Run every available specialist on the same embedding
            model_preds: dict[str, float] = {}
            for spec in active_specs:
                key = (view, spec["role"])
                model_preds[spec["role"]] = self._forward_once(
                    self._models[key], embedding
                )

            # Composite weighting: (1/MAE) * (1+R²) * (1+ClinAcc)
            # Better models dominate on ALL metrics, not just MAE.
            # Outlier clipping: if one specialist is >_OUTLIER_THRESHOLD% away
            # from the trimmed mean of the others, exclude it from the mean
            # (it still feeds ef_preds_for_conf so the σ / confidence is not inflated).
            weights = [
                (1.0 / max(s["val_mae"], 0.1))
                * (1.0 + s.get("val_r2", 0.0))
                * (1.0 + s.get("val_clin_acc", 0.0))
                for s in active_specs
            ]
            ef_preds_list = [model_preds[s["role"]] for s in active_specs]
            ef_mean = _robust_weighted_mean(ef_preds_list, weights,
                                            outlier_threshold=_OUTLIER_THRESHOLD)

            # ALL individual predictions feed the consistency score
            ef_preds_for_conf = ef_preds_list
            models_used = [s["role"] for s in active_specs]

            logger.debug(
                "Roundtable [%s]: %s → weighted EF=%.1f%%  σ=%.1f%%",
                view,
                {r: f"{v:.1f}" for r, v in model_preds.items()},
                ef_mean,
                _prediction_std(ef_preds_list),
            )
        else:
            # Single-model path (primary specialist only)
            primary_spec = active_specs[0]
            primary_model = self._models[(view, primary_spec["role"])]
            if n_mc_passes > 1:
                ef_preds_for_conf = self._forward_mc(primary_model, embedding, n_mc_passes)
            else:
                ef_preds_for_conf = [self._forward_once(primary_model, embedding)]
            ef_mean = sum(ef_preds_for_conf) / len(ef_preds_for_conf)
            model_preds = {primary_spec["role"]: ef_mean}
            models_used = [primary_spec["role"]]

        # ---- Z-score ----
        zscore = compute_ef_zscore(
            ef=ef_mean,
            age=age,
            sex=sex,
            weight=weight if (weight and weight > 0) else None,
            height=height if (height and height > 0) else None,
        )

        # ---- Confidence — fed by INTER-MODEL spread, not MC-Dropout ----
        confidence = compute_confidence(ef_preds_for_conf, zscore.z_score)

        # ---- Category ----
        category = ef_category(ef_mean, age or 8.0)

        # ---- Interpretation ----
        interpretation = _build_clinical_interpretation(
            ef_mean, category, zscore, confidence, age, sex, view
        )

        return ClinicalReport(
            patient_id=patient_id,
            view=view,
            ef_predicted=round(ef_mean, 2),
            ef_category=category,
            zscore=zscore,
            confidence=confidence,
            age=age,
            sex=sex,
            weight=weight,
            height=height,
            bsa=zscore.bsa,
            n_mc_passes=len(ef_preds_for_conf),
            model_version=self._model_versions.get(view, "not-loaded"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            interpretation=interpretation,
            model_predictions=model_preds,
            models_used=models_used,
        )

    def run_batch(
        self,
        records: list[dict],
        view: str = "A4C",
        n_mc_passes: int = 1,
    ) -> list[ClinicalReport]:
        """Run inference for a list of patient records.

        Each record dict should have keys matching ``run()`` parameters:
        ``embedding_path`` (or ``embedding``), ``age``, ``sex``,
        ``weight``, ``height``, ``patient_id``.

        Parameters
        ----------
        records : list of dicts, one per patient
        view : default view (overridden per-record if 'view' key present)
        n_mc_passes : MC-Dropout passes per patient (only used in single-model mode)

        Returns
        -------
        list[ClinicalReport]
        """
        reports = []
        for i, rec in enumerate(records):
            pid = rec.get("patient_id", f"patient_{i+1}")
            try:
                report = self.run(
                    embedding=rec.get("embedding"),
                    embedding_path=rec.get("embedding_path"),
                    age=rec.get("age"),
                    sex=rec.get("sex"),
                    weight=rec.get("weight"),
                    height=rec.get("height"),
                    view=rec.get("view", view),
                    patient_id=pid,
                    n_mc_passes=n_mc_passes,
                )
                reports.append(report)
            except Exception as exc:
                logger.error("Failed to process patient %s: %s", pid, exc)
                raise
        return reports

    def preload(self, views: Optional[Sequence[str]] = None) -> None:
        """Eagerly load models to avoid first-call latency."""
        for v in (views or list(self._specs)):
            self._ensure_loaded(v.upper())

    def unload(self, view: Optional[str] = None) -> None:
        """Free GPU memory for one or all specialists.

        Parameters
        ----------
        view : if given, unload only specialists for that view; otherwise unload all.
        """
        if view is not None:
            v = view.upper()
            keys = [(v2, role) for (v2, role) in list(self._models) if v2 == v]
        else:
            keys = list(self._models)
        for k in keys:
            if k in self._models:
                self._models[k].cpu()
                del self._models[k]
                torch.cuda.empty_cache()
        if keys:
            torch.cuda.empty_cache()
            logger.info("Unloaded %d specialist(s) for view=%s", len(keys), view or 'ALL')


# ---------------------------------------------------------------------------
# Convenience function (no-class interface)
# ---------------------------------------------------------------------------

_default_engine: Optional[EchoGuardInference] = None


def run_inference(
    embedding_path: str | Path,
    age: Optional[float] = None,
    sex: Optional[str] = None,
    weight: Optional[float] = None,
    height: Optional[float] = None,
    view: str = "A4C",
    patient_id: str = "unknown",
    n_mc_passes: int = 1,
    checkpoints: Optional[dict[str, str]] = None,
) -> ClinicalReport:
    """One-shot inference using a module-level singleton engine.

    Convenient for scripts that only need a few predictions.
    For high-throughput batch use, instantiate ``EchoGuardInference`` directly.
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = EchoGuardInference(checkpoints=checkpoints)
    return _default_engine.run(
        embedding_path=embedding_path,
        age=age, sex=sex, weight=weight, height=height,
        view=view, patient_id=patient_id, n_mc_passes=n_mc_passes,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="EchoGuard-Peds single-patient inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embedding", required=True, help="Path to .pt embedding file")
    parser.add_argument("--view", default="A4C", choices=["A4C", "PSAX"])
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--sex", default=None, choices=["M", "F"])
    parser.add_argument("--weight", type=float, default=None, help="kg")
    parser.add_argument("--height", type=float, default=None, help="cm")
    parser.add_argument("--patient-id", default="unknown")
    parser.add_argument("--mc-passes", type=int, default=1,
                        help="MC-Dropout forward passes (>1 for uncertainty)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument(
        "--checkpoint-a4c",
        default=_DEFAULT_CHECKPOINTS["A4C"]["path"],
        help="Override A4C checkpoint path",
    )
    parser.add_argument(
        "--checkpoint-psax",
        default=_DEFAULT_CHECKPOINTS["PSAX"]["path"],
        help="Override PSAX checkpoint path",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    checkpoints = {
        "A4C": args.checkpoint_a4c,
        "PSAX": args.checkpoint_psax,
    }
    engine = EchoGuardInference(checkpoints=checkpoints, device=args.device)
    report = engine.run(
        embedding_path=args.embedding,
        age=args.age,
        sex=args.sex,
        weight=args.weight,
        height=args.height,
        view=args.view,
        patient_id=args.patient_id,
        n_mc_passes=args.mc_passes,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report)


if __name__ == "__main__":
    _cli()
