"""
echoguard/vlm_critic.py
=======================
VLM-based visual critic for EchoGuard-Peds.

Uses MedGemma 1.5 4B (visual) to visually validate the EF prediction
produced by the embedding-based regression model.

Given:
- 3 key echo frames  (ED, mid-systole, ES) from the original AVI
- The regression model's EF prediction + Z-score + demographics

Outputs:
- Verdict: AGREE / DISAGREE / UNCERTAIN
- Confidence delta: how much the VLM adjusts confidence up/down
- LV description: free-text visual observation of the LV
- Final synthesized interpretation (merges regression + VLM signals)

Design Principle
----------------
The VLM does not predict a new EF number — it operates as a second-opinion
on whether the visual appearance is *consistent* with the regression estimate.
This prevents the VLM from overwriting a well-calibrated quantitative estimate
with a potentially noisy qualitative guess.

Usage
-----
    from echoguard.inference import EchoGuardInference
    from echoguard.vlm_critic import VLMCritic, CriticReport

    critic = VLMCritic()           # loads MedGemma 4B once
    report = engine.run(...)       # regression ClinicalReport

    critic_report = critic.evaluate(
        clinical_report=report,
        video_path="data/echonet_pediatric/A4C/Videos/CR32a7558-CR32a7585-000056.avi",
        ed_idx=39,
        es_idx=27,
    )
    print(critic_report.verdict)           # 'AGREE'
    print(critic_report.lv_description)   # 'LV cavity appears enlarged...'
    print(critic_report.final_interpretation)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from echoguard.config import PROJECT_ROOT
from echoguard.inference import ClinicalReport
from echoguard.zscore import ZScoreFlag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

class CriticVerdict(str, Enum):
    """VLM visual validation outcome."""

    AGREE = "agree"          # Visual appearance consistent with EF estimate
    UNCERTAIN = "uncertain"  # VLM cannot confidently confirm or deny
    DISAGREE = "disagree"    # Visual appearance inconsistent with EF estimate

    @property
    def confidence_multiplier(self) -> float:
        """Factor applied to regression confidence score."""
        return {
            CriticVerdict.AGREE: 1.10,
            CriticVerdict.UNCERTAIN: 0.85,
            CriticVerdict.DISAGREE: 0.60,
        }[self]


# ---------------------------------------------------------------------------
# Critic report
# ---------------------------------------------------------------------------

@dataclass
class CriticReport:
    """Combined output of the VLM critic.

    Attributes
    ----------
    verdict : AGREE / UNCERTAIN / DISAGREE
    lv_description : free-text description of LV appearance
    confidence_adjusted : original confidence × critic multiplier (clipped 0–1)
    final_interpretation : merged clinical narrative
    raw_vlm_output : unprocessed VLM text output
    frames_used : number of frames passed to VLM
    """

    verdict: CriticVerdict
    lv_description: str
    confidence_original: float
    confidence_adjusted: float
    final_interpretation: str
    raw_vlm_output: str
    frames_used: int

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "lv_description": self.lv_description,
            "confidence_original": round(self.confidence_original, 3),
            "confidence_adjusted": round(self.confidence_adjusted, 3),
            "final_interpretation": self.final_interpretation,
            "frames_used": self.frames_used,
        }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_CRITIC_PROMPT_TEMPLATE = """You are a senior paediatric cardiologist acting as the final reviewer for an AI-powered echocardiography system.

SPECIALIST ROUNDTABLE PREDICTIONS ({view} view):
{roundtable_str}
Ensemble EF (weighted): {ef:.1f}%  ({category})  |  Inter-specialist σ: {sigma:.1f}%

CLINICAL CONTEXT:
- Z-score: {z_score:+.2f}  (age/sex-adjusted reference range: {nr_lo:.1f}–{nr_hi:.1f}%)
- Patient: {age_sex_str}
{dissent_note}
I am showing you {n_frames} key frames from the echocardiogram ({view} view):
{frame_labels}

YOUR TASK:
1. Briefly describe what you see regarding left ventricular size and wall motion.
2. Given the specialist predictions above, state whether the visual appearance AGREES with, DISAGREES with, or is UNCERTAIN about the ensemble EF of {ef:.1f}%.{sigma_guidance}

Respond in this exact format:
LV_DESCRIPTION: <one or two sentences describing LV cavity size and wall motion>
VERDICT: <AGREE | DISAGREE | UNCERTAIN>
REASON: <one sentence justifying your verdict, referencing any specialist disagreement if relevant>"""

_FRAME_LABELS = {
    1: ["Representative frame"],
    2: ["End-diastole (ED)", "End-systole (ES)"],
    3: ["End-diastole (ED)", "Mid-systole", "End-systole (ES)"],
    4: ["End-diastole (ED)", "Mid-systole", "End-systole (ES)", "Mid-diastole"],
}


def _build_prompt(report: ClinicalReport, n_frames: int) -> str:
    """Build the VLM critic prompt from a regression ClinicalReport."""
    age_str = f"age {report.age:.0f}y" if report.age is not None else "age unknown"
    sex_str = report.sex or "sex unknown"
    if report.bsa:
        age_sex_str = f"{age_str}, {sex_str}, BSA {report.bsa:.2f} m²"
    else:
        age_sex_str = f"{age_str}, {sex_str}"

    labels = _FRAME_LABELS.get(n_frames, [f"Frame {i+1}" for i in range(n_frames)])
    frame_labels_str = "\n".join(f"  Image {i+1}: {lbl}" for i, lbl in enumerate(labels))

    # --- Specialist roundtable section ---
    _role_labels = {
        "motion_analyst":       "Motion Analyst (Temporal)",
        "pattern_matcher":      "Pattern Matcher (TCN)",
        "guardrail_classifier": "Guardrail Classifier (MultiTask)",
        "sonographer_baseline": "Sonographer Baseline (MLP)",
    }
    preds = report.model_predictions  # {role: ef_float}
    if preds:
        lines = []
        for role, ef_val in preds.items():
            label = _role_labels.get(role, role)
            lines.append(f"  - {label:<38s}: EF {ef_val:.1f}%")
        roundtable_str = "\n".join(lines)
        pred_vals = list(preds.values())
        import statistics
        sigma = statistics.pstdev(pred_vals) if len(pred_vals) > 1 else 0.0
        # Highlight if any specialist strongly disagrees (>10% from consensus)
        outliers = [
            f"{_role_labels.get(r, r)} ({v:.1f}%)"
            for r, v in preds.items()
            if abs(v - report.ef_predicted) > 10
        ]
        if outliers:
            dissent_note = (
                f"⚠ NOTE: {', '.join(outliers)} significantly disagree(s) with the ensemble "
                f"estimate. Please comment on whether the visual evidence supports or refutes "
                f"the dissenting view.\n"
            )
        else:
            dissent_note = ""
        sigma_guidance = (
            f"\n   Note: inter-specialist σ={sigma:.1f}% — "
            + ("high disagreement; use visual evidence to arbitrate." if sigma > 8 else
               "moderate disagreement; provide your independent assessment." if sigma > 4 else
               "strong specialist agreement.")
        )
    else:
        # Fallback: no ensemble breakdown available
        roundtable_str = f"  - Single model: EF {report.ef_predicted:.1f}%"
        sigma = 0.0
        dissent_note = ""
        sigma_guidance = ""

    return _CRITIC_PROMPT_TEMPLATE.format(
        ef=report.ef_predicted,
        category=report.ef_category,
        z_score=report.zscore.z_score,
        nr_lo=report.zscore.normal_range[0],
        nr_hi=report.zscore.normal_range[1],
        age_sex_str=age_sex_str,
        n_frames=n_frames,
        view=report.view,
        frame_labels=frame_labels_str,
        roundtable_str=roundtable_str,
        sigma=sigma,
        dissent_note=dissent_note,
        sigma_guidance=sigma_guidance,
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> tuple[CriticVerdict, str, str]:
    """Parse VLM output → (verdict, lv_description, reason).

    Returns UNCERTAIN if parsing fails.
    """
    text_upper = text.upper()

    # Verdict
    verdict = CriticVerdict.UNCERTAIN
    if "VERDICT: AGREE" in text_upper or "\nAGREE\n" in text_upper:
        verdict = CriticVerdict.AGREE
    elif "VERDICT: DISAGREE" in text_upper or "\nDISAGREE\n" in text_upper:
        verdict = CriticVerdict.DISAGREE
    # Handle inline VERDICT after description
    elif "VERDICT:" in text_upper:
        m = re.search(r"VERDICT:\s*(AGREE|DISAGREE|UNCERTAIN)", text, re.IGNORECASE)
        if m:
            v = m.group(1).upper()
            if v == "AGREE":
                verdict = CriticVerdict.AGREE
            elif v == "DISAGREE":
                verdict = CriticVerdict.DISAGREE

    # LV description — accept multiple labelling conventions the VLM may use
    lv_desc = ""
    for label in ("LV_DESCRIPTION", "LV DESCRIPTION", "FINDINGS", "DESCRIPTION"):
        m = re.search(
            rf"{label}:\s*(.+?)(?:\n(?:VERDICT|REASON)|$)",
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            lv_desc = m.group(1).strip()
            # Strip trailing VERDICT line if it leaked in
            lv_desc = re.sub(r"\s*VERDICT:.*", "", lv_desc, flags=re.IGNORECASE).strip()
            break
    if not lv_desc:
        # fallback: first 2 sentences before any keyword
        stripped = re.split(r"\n(?:VERDICT|REASON):", text, flags=re.IGNORECASE)[0]
        sentences = re.split(r"(?<=[.!?])\s+", stripped.strip())
        lv_desc = " ".join(sentences[:2]) if sentences else text[:200]

    # Reason
    reason = ""
    m = re.search(r"REASON:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()

    return verdict, lv_desc, reason


# ---------------------------------------------------------------------------
# Final interpretation synthesis
# ---------------------------------------------------------------------------

def _synthesize_interpretation(
    report: ClinicalReport, verdict: CriticVerdict, lv_desc: str, reason: str
) -> str:
    """Merge regression and VLM signals into a single clinical narrative."""
    parts = [report.interpretation]

    if verdict is CriticVerdict.AGREE:
        parts.append(
            f"Visual review confirms: {lv_desc}"
        )
    elif verdict is CriticVerdict.UNCERTAIN:
        parts.append(
            f"Visual review is inconclusive ({lv_desc}); "
            f"quantitative estimate should be prioritised."
        )
    else:  # DISAGREE
        parts.append(
            f"⚠ Visual review suggests a discrepancy: {lv_desc}  "
            f"Reason: {reason}  Manual re-measurement is advised."
        )

    return "  ".join(parts)


# ---------------------------------------------------------------------------
# VLM Critic
# ---------------------------------------------------------------------------

class VLMCritic:
    """Loads MedGemma 4B once and provides visual EF validation.

    Parameters
    ----------
    model_path : path to MedGemma 4B weights (default: local_models/medgemma-4b)
    video_dir  : base directory for AVI videos (default: data/echonet_pediatric/A4C/Videos)
    device     : 'auto', 'cuda', 'cpu', 'mps'
    n_frames   : how many key frames to send to the VLM (2 or 3, default 3)
    max_new_tokens : token budget for VLM response
    quantize   : None (full precision), '8bit', or '4bit' for quantized inference
                 8-bit uses ~4GB, 4-bit uses ~2.5GB (requires bitsandbytes)
    """

    def __init__(
        self,
        model_path: str = str(PROJECT_ROOT / "local_models" / "medgemma-4b"),
        video_dir: str = str(PROJECT_ROOT / "data" / "echonet_pediatric" / "A4C" / "Videos"),
        device: str = "auto",
        n_frames: int = 3,
        max_new_tokens: int = 256,
        quantize: str = None,
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

        self._model_path = model_path
        self._video_dir = Path(video_dir)
        self._n_frames = n_frames
        self._max_new_tokens = max_new_tokens
        self._quantize = quantize

        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ #
    # Lazy loading                                                         #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load MedGemma 4B into GPU memory."""
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading MedGemma 4B from %s ...", self._model_path)
        self._processor = AutoProcessor.from_pretrained(self._model_path)
        
        # MPS (Apple Silicon) doesn't support device_map="auto" — load to device directly
        if self._device == "mps":
            # Use float16 on MPS — float32 is more stable but 2× memory
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16,
            ).to("mps")
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self._model.eval()
        logger.info("MedGemma 4B loaded. Device: %s", self._model.device)

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            self._model.cpu()
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info("VLMCritic unloaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_frames(
        self,
        video_path: str | Path,
        ed_idx: int,
        es_idx: int,
        n: int,
    ) -> list[Image.Image]:
        """Extract key frames using video_utils.select_key_frames."""
        from echoguard.video_utils import extract_all_frames, select_key_frames, select_key_frames_extended

        frames = extract_all_frames(video_path)

        if n <= 4:
            all_keys = select_key_frames(
                frames, ed_idx, es_idx, target_size=448  # MedGemma 1.5 native size
            )
            return all_keys[:n]
        else:
            extended = select_key_frames_extended(
                frames, ed_idx, es_idx, num_frames=n, target_size=448
            )
            return extended

    def _vlm_forward(
        self,
        images: list[Image.Image],
        prompt: str,
    ) -> str:
        """Single VLM forward pass with images + text."""
        if self._model is None:
            raise RuntimeError("VLMCritic not loaded. Call .load() first.")

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=self._model.dtype)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            # On MPS with float16, sampling (softmax) can overflow to inf/nan.
            # Use greedy decoding for stability on Apple Silicon.
            is_mps = str(self._model.device).startswith("mps")
            gen_kwargs = dict(
                max_new_tokens=self._max_new_tokens,
                repetition_penalty=1.2,
            )
            if is_mps:
                gen_kwargs["do_sample"] = False  # greedy — avoids softmax overflow
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = 0.3
                gen_kwargs["top_p"] = 0.9

            outputs = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        generated = outputs[0][input_len:]
        return self._processor.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        clinical_report: ClinicalReport,
        video_path: Optional[str | Path] = None,
        ed_idx: Optional[int] = None,
        es_idx: Optional[int] = None,
        frames: Optional[list[Image.Image]] = None,
        n_frames: Optional[int] = None,
    ) -> CriticReport:
        """Visually validate a regression ClinicalReport.

        Parameters
        ----------
        clinical_report : output of EchoGuardInference.run()
        video_path : path to the AVI file.  Can be:
            - absolute path
            - filename only (looked up in self._video_dir)
        ed_idx : end-diastole frame index (from manifest)
        es_idx : end-systole frame index (from manifest)
        frames : pre-extracted PIL image list (overrides video_path + ed/es)
        n_frames : number of frames to use (overrides constructor default)

        Returns
        -------
        CriticReport
        """
        self.load()
        n = n_frames or self._n_frames

        # ---- Extract frames ----
        if frames is not None:
            key_frames = frames[:n]
        elif video_path is not None:
            vpath = Path(video_path)
            if not vpath.is_absolute():
                vpath = self._video_dir / vpath.name
            if not vpath.exists():
                raise FileNotFoundError(f"Video not found: {vpath}")
            ed = ed_idx if ed_idx is not None else 0
            es = es_idx if es_idx is not None else max(ed - 10, 0)
            key_frames = self._extract_frames(vpath, ed, es, n)
        else:
            raise ValueError("Provide either 'video_path' or 'frames'.")

        # ---- Build prompt ----
        prompt = _build_prompt(clinical_report, len(key_frames))
        self._last_prompt = prompt  # expose for demo API

        # ---- VLM forward ----
        raw_output = self._vlm_forward(key_frames, prompt)
        logger.debug("VLM raw output: %s", raw_output[:300])

        # ---- Parse response ----
        verdict, lv_desc, reason = _parse_response(raw_output)

        # ---- Adjusted confidence ----
        conf_orig = clinical_report.confidence.overall
        conf_adj = min(0.99, max(0.01, conf_orig * verdict.confidence_multiplier))

        # ---- Synthesize interpretation ----
        final_interp = _synthesize_interpretation(
            clinical_report, verdict, lv_desc, reason
        )

        return CriticReport(
            verdict=verdict,
            lv_description=lv_desc,
            confidence_original=conf_orig,
            confidence_adjusted=round(conf_adj, 3),
            final_interpretation=final_interp,
            raw_vlm_output=raw_output,
            frames_used=len(key_frames),
        )

    def evaluate_batch(
        self,
        records: list[dict],
        n_frames: Optional[int] = None,
    ) -> list[CriticReport]:
        """Run critic on multiple patients.

        Each record: {'clinical_report': ClinicalReport, 'video_path': str,
                       'ed_idx': int, 'es_idx': int}
        """
        self.load()
        results = []
        for i, rec in enumerate(records):
            try:
                cr = self.evaluate(
                    clinical_report=rec["clinical_report"],
                    video_path=rec.get("video_path"),
                    ed_idx=rec.get("ed_idx"),
                    es_idx=rec.get("es_idx"),
                    frames=rec.get("frames"),
                    n_frames=n_frames,
                )
                results.append(cr)
                logger.info(
                    "[%d/%d] %s → %s (conf %.2f → %.2f)",
                    i + 1, len(records),
                    rec.get("clinical_report", {}).patient_id if hasattr(rec.get("clinical_report"), "patient_id") else "?",
                    cr.verdict.value,
                    cr.confidence_original,
                    cr.confidence_adjusted,
                )
            except Exception as exc:
                logger.error("Error on record %d: %s", i, exc)
                raise
        return results


# ---------------------------------------------------------------------------
# Quick dry-run (no GPU needed) — validates frame extraction + prompt
# ---------------------------------------------------------------------------

def _dry_run(video_path: str, ed_idx: int, es_idx: int) -> None:
    """Validate frame extraction and prompt without loading MedGemma."""
    from echoguard.inference import EchoGuardInference
    import json as _json

    # Make a dummy ClinicalReport to test prompt formatting
    from echoguard.zscore import compute_ef_zscore
    from echoguard.confidence import compute_confidence
    from echoguard.config import ef_category
    from echoguard.inference import ClinicalReport, _build_clinical_interpretation
    from datetime import datetime, timezone

    ef = 38.5
    age = 8.0
    sex = "M"
    zr = compute_ef_zscore(ef, age, sex, 27.0, 130.0)
    conf = compute_confidence([ef], zr.z_score)
    cat = ef_category(ef, age)
    interp = _build_clinical_interpretation(ef, cat, zr, conf, age, sex, "A4C")

    dummy_report = ClinicalReport(
        patient_id="DRY-RUN",
        view="A4C",
        ef_predicted=ef,
        ef_category=cat,
        zscore=zr,
        confidence=conf,
        age=age,
        sex=sex,
        weight=27.0,
        height=130.0,
        bsa=zr.bsa,
        n_mc_passes=1,
        model_version="dry-run",
        timestamp=datetime.now(timezone.utc).isoformat(),
        interpretation=interp,
    )

    critic = VLMCritic()
    frames = critic._extract_frames(video_path, ed_idx, es_idx, n=3)
    print(f"✅ Extracted {len(frames)} frames: {[f.size for f in frames]}")

    prompt = _build_prompt(dummy_report, len(frames))
    print("\n=== Prompt Preview ===")
    print(prompt)
    print("=== End Prompt ===\n")
    print("To run full VLM inference, call critic.load() then critic.evaluate().")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLMCritic dry-run — validate frames + prompt")
    parser.add_argument("--video", required=True, help="Path to AVI file")
    parser.add_argument("--ed-idx", type=int, default=39, help="End-diastole frame index")
    parser.add_argument("--es-idx", type=int, default=27, help="End-systole frame index")
    args = parser.parse_args()

    _dry_run(args.video, args.ed_idx, args.es_idx)
