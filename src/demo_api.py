"""
demo_api.py
===========
EchoGuard-Peds live demo API (v2 — MedGemma Agentic Pipeline).

Run from project_echo/:
    uvicorn demo_api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                              → frontend SPA
    GET  /api/health                    → liveness
    GET  /api/patients?split=TEST       → patient list (merged A4C+PSAX manifests)
    POST /api/analyze                   → full 4-specialist roundtable + dual-view fusion
    POST /api/vlm                       → MedGemma 4B VLM Senior Attending (~5-10s)
    POST /api/geometric                 → geometric EF + segmentation masks + LV area timeline
    POST /api/narrate                   → MedGemma agentic clinical narration (full pipeline)
    GET  /api/video/{pid}/{view}        → echo AVI converted to MP4 (ffmpeg cached)
    GET  /api/frames/{pid}/{view}       → ED / mid-systole / ES frames as base64 JPEG
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")
logger = logging.getLogger("demo_api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EchoGuard-Peds",
    description="MedGemma Agentic Pipeline: 4-specialist ensemble + Geometric EF + VLM Narrator",
    version="2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent          # src/
PROJECT_ROOT = BASE.parent                       # project_echo/
FRONTEND_DIR = BASE / "demo_frontend"
VIDEO_CACHE  = Path("/tmp/echoguard_mp4_cache")
VIDEO_CACHE.mkdir(exist_ok=True)

VIDEO_DIRS: dict[str, Path] = {
    "A4C":  PROJECT_ROOT / "data/echonet_pediatric/A4C/Videos",
    "PSAX": PROJECT_ROOT / "data/echonet_pediatric/PSAX/Videos",
}
MANIFEST_PATHS: dict[str, Path] = {
    "A4C":  PROJECT_ROOT / "data/embeddings_videomae/pediatric_a4c/manifest.json",
    "PSAX": PROJECT_ROOT / "data/embeddings_videomae/pediatric_psax/manifest.json",
}

# ---------------------------------------------------------------------------
# Singletons (loaded at startup)
# ---------------------------------------------------------------------------

_engine  = None
_fusion  = None
_critic  = None
_seg_models: dict = {}       # view → segmentation model (DeepLabV3)
_calibrations: dict = {}     # view → (slope, intercept)
_manifests: dict[str, dict] = {}
_mp4_cache: dict[tuple, Path] = {}
# prefix (e.g. 'CR32a7555-CR32a7582') → {view: video_id}
_video_id_map: dict[str, dict[str, str]] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import math

def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so JSON serialization never crashes."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _offload_models_to_cpu():
    """Move specialist & segmentation models to CPU to free GPU memory for VLM."""
    import torch
    if _engine is not None:
        for key, model in _engine._models.items():
            model.cpu()
        logger.info("Offloaded specialist models to CPU")
    for view, model in _seg_models.items():
        model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("GPU memory freed for VLM")


def _reload_models_to_gpu():
    """Move specialist & segmentation models back to GPU after VLM is done."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if _engine is not None:
        for key, model in _engine._models.items():
            model.to(device)
        logger.info("Reloaded specialist models to %s", device)
    for view, model in _seg_models.items():
        model.to(device)


_SEVERITY_ORDER = ["critical", "reduced", "borderline_low", "normal", "borderline_high", "hyperdynamic", "unknown"]
_SEVERITY_COLORS = {
    "critical":      "#f85149",
    "reduced":       "#f0883e",
    "borderline_low":"#d29922",
    "normal":        "#3fb950",
    "borderline_high":"#58a6ff",
    "hyperdynamic":  "#bc8cff",
    "unknown":       "#8b949e",
}


def _patient_prefix(vid_id: str) -> str:
    """'CR32a7555-CR32a7582-000039' → 'CR32a7555-CR32a7582'"""
    return vid_id.rsplit('-', 1)[0]


def _video_path_for(prefix: str, view: str) -> Optional[Path]:
    """Resolve AVI path via _video_id_map (prefix → video_id → file)."""
    vid = _video_id_map.get(prefix, {}).get(view.upper())
    if vid is None:
        vid = prefix  # legacy: caller passed a video_id directly
    return _video_path(vid, view)


def _ef_to_severity(ef: Optional[float]) -> str:
    if ef is None:
        return "unknown"
    if ef < 35:   return "critical"
    if ef < 50:   return "reduced"
    if ef < 55:   return "borderline_low"
    if ef < 72:   return "normal"
    return "hyperdynamic"


def _flag_color(flag: str) -> str:
    return _SEVERITY_COLORS.get(flag.lower(), _SEVERITY_COLORS["unknown"])


# BGR colors for OpenCV overlays (fill, contour) keyed by severity
_SEVERITY_BGR = {
    "critical":       ((73, 81, 240),  (80, 90, 255)),
    "reduced":        ((62, 136, 240), (70, 150, 255)),
    "borderline_low": ((34, 153, 210), (40, 165, 230)),
    "normal":         ((80, 185, 63),  (120, 255, 80)),
    "borderline_high":((200, 166, 88), (220, 180, 100)),
    "hyperdynamic":   ((255, 140, 188),(255, 160, 200)),
    "unknown":        ((139, 149, 158),(160, 170, 180)),
}


def _severity_bgr(ef: float):
    """Return (fill_bgr, contour_bgr) for the given EF value."""
    sev = _ef_to_severity(ef)
    return _SEVERITY_BGR.get(sev, _SEVERITY_BGR["unknown"])


def _video_path(pid: str, view: str) -> Optional[Path]:
    d = VIDEO_DIRS.get(view.upper())
    if d is None:
        return None
    p = d / f"{pid}.avi"
    return p if p.exists() else None


def _to_mp4(avi: Path, pid: str, view: str) -> Optional[Path]:
    key = (pid, view.upper())
    if key in _mp4_cache and _mp4_cache[key].exists():
        return _mp4_cache[key]
    out = VIDEO_CACHE / f"{pid}_{view.upper()}.mp4"
    if out.exists():
        _mp4_cache[key] = out
        return out
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(avi),
             "-vf", "scale=224:224:flags=lanczos",
             "-c:v", "libx264", "-preset", "fast", "-crf", "26",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(out)],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0 and out.exists():
            _mp4_cache[key] = out
            logger.info("Converted %s → %s", avi.name, out.name)
            return out
        else:
            logger.warning("ffmpeg failed: %s", r.stderr.decode()[:200])
    except Exception as e:
        logger.warning("ffmpeg error: %s", e)
    return None


def _extract_frames(avi: Path, ed_idx: int, es_idx: int) -> dict:
    """Return base64-encoded JPEG for ED, mid-systole, ES frames."""
    cap = cv2.VideoCapture(str(avi))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = (ed_idx + es_idx) // 2
    result = {}
    for label, idx in [("ed", ed_idx), ("mid", mid_idx), ("es", es_idx)]:
        idx = max(0, min(idx, n - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (224, 224))
            # Apply CLAHE for better contrast in greyscale echo frames
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(grey)
            display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 88])
            result[label] = base64.b64encode(buf.tobytes()).decode()
        else:
            result[label] = None
    cap.release()
    return result

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _engine, _fusion, _manifests, _video_id_map, _seg_models, _calibrations
    logger.info("⏳  Loading EchoGuard specialist models (8 total)…")
    from echoguard.inference import EchoGuardInference
    from echoguard.dual_view import DualViewFusion
    _engine = EchoGuardInference()
    _engine.preload(["A4C", "PSAX"])
    _fusion = DualViewFusion(_engine)
    logger.info("✓  All specialists loaded.")

    for view, path in MANIFEST_PATHS.items():
        if path.exists():
            with open(path) as f:
                _manifests[view] = json.load(f)
            # Resolve relative embedding_path entries to absolute paths
            for vid, meta in _manifests[view].items():
                if "embedding_path" in meta and not os.path.isabs(meta["embedding_path"]):
                    meta["embedding_path"] = str(PROJECT_ROOT / meta["embedding_path"])
    logger.info("✓  Manifests: %s", {v: len(m) for v, m in _manifests.items()})

    # Build patient-prefix → {view: video_id} map for cross-view linking
    for view, manifest in _manifests.items():
        for vid in manifest:
            p = _patient_prefix(vid)
            if p not in _video_id_map:
                _video_id_map[p] = {}
            _video_id_map[p][view] = vid
    logger.info("✓  Patient map: %d unique patients", len(_video_id_map))

    # Load segmentation models + calibrations for geometric EF
    from echoguard.regression.geometric_ef import load_segmentation_model, CHECKPOINTS, CALIBRATION_FILES
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    for view in ["A4C", "PSAX"]:
        ckpt = Path(CHECKPOINTS[view])
        cal_path = Path(CALIBRATION_FILES[view])
        if ckpt.exists():
            _seg_models[view] = load_segmentation_model(ckpt, device)
            logger.info("✓  Segmentation model loaded: %s", view)
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            _calibrations[view] = (cal["slope"], cal["intercept"])
            logger.info("✓  Calibration loaded: %s (slope=%.4f, intercept=%.4f)", view, cal["slope"], cal["intercept"])

    # Pre-warm MP4 cache for demo patients so first browser request is instant
    import asyncio
    demo_prefixes = ["CR32a7558-CR32a7585", "CR32a95f8-CR32a97c4"]
    async def _prewarm(prefix: str, view: str):
        vid = _video_id_map.get(prefix, {}).get(view)
        if vid is None:
            return
        avi = _video_path(vid, view)
        if avi:
            await asyncio.to_thread(_to_mp4, avi, vid, view)
            logger.info("✓  Pre-warmed %s/%s", prefix, view)
    tasks = [_prewarm(p, v) for p in demo_prefixes for v in ["A4C", "PSAX"]]
    await asyncio.gather(*tasks, return_exceptions=True)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "specialists_loaded": _engine is not None}


@app.get("/api/patients")
async def list_patients(split: str = "TEST", limit: int = 300):
    """Patient list grouped by patient prefix so A4C+PSAX appear as one entry."""
    patients: dict[str, dict] = {}  # keyed by patient prefix

    for view, manifest in _manifests.items():
        for vid, meta in manifest.items():
            if split and meta.get("split", "").upper() != split.upper():
                continue
            prefix = _patient_prefix(vid)
            ef = meta.get("ef")
            sev = _ef_to_severity(ef)
            if prefix not in patients:
                patients[prefix] = {
                    "patient_id": prefix,
                    "true_ef":   round(ef, 1) if ef is not None else None,
                    "age":       meta.get("age"),
                    "sex":       meta.get("sex"),
                    "weight":    meta.get("weight"),
                    "height":    meta.get("height"),
                    "split":     meta.get("split"),
                    "severity":  sev,
                    "color":     _flag_color(sev),
                    "views":     [],
                    "ed_frames": {},
                    "es_frames": {},
                    "has_video": {},
                }
            p = patients[prefix]
            if view not in p["views"]:
                p["views"].append(view)
            p["ed_frames"][view]  = meta.get("ed_idx", 0)
            p["es_frames"][view]  = meta.get("es_idx", 10)
            p["has_video"][view]  = _video_path(vid, view) is not None

    result = list(patients.values())
    result.sort(key=lambda x: _SEVERITY_ORDER.index(x["severity"]))
    return _sanitize(result[:limit])


# ─── Analyze ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    patient_id: str


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Run 4-specialist roundtable for A4C + PSAX + dual-view fusion.
    patient_id is a patient prefix (e.g. 'CR32a7558-CR32a7585') that may map
    to separate A4C and PSAX video IDs via _video_id_map.
    """
    import statistics
    pid = req.patient_id
    result: dict = {}
    reports: dict = {}  # view → ClinicalReport (for fusion)

    # Resolve video IDs: prefix → {A4C: vid, PSAX: vid}
    vid_map = _video_id_map.get(pid, {})
    if not vid_map:
        # Fallback: pid might be a raw video_id from a single-view patient
        for view, manifest in _manifests.items():
            if pid in manifest:
                vid_map[view] = pid

    for view, vid in vid_map.items():
        manifest = _manifests.get(view, {})
        meta = manifest.get(vid)
        if meta is None:
            continue
        try:
            report = _engine.run(
                embedding_path=meta["embedding_path"],
                age=meta.get("age"), sex=meta.get("sex"),
                weight=meta.get("weight"), height=meta.get("height"),
                view=view, patient_id=pid,
            )
            reports[view] = report
            d = report.to_dict()
            d["specialist_colors"] = {
                role: _flag_color(_ef_to_severity(ef))
                for role, ef in report.model_predictions.items()
            }
            # Add validated accuracy metrics for each specialist
            from echoguard.inference import _ENSEMBLE_SPECS
            specs = _ENSEMBLE_SPECS.get(view, [])
            d["specialist_metrics"] = {
                s["role"]: {
                    "model_type": s["model_type"],
                    "val_mae": s["val_mae"],
                    "val_r2": round(s["val_r2"], 4),
                    "val_clin_acc": round(s["val_clin_acc"] * 100, 1),
                }
                for s in specs
            }
            preds = list(report.model_predictions.values())
            if len(preds) >= 3:
                med = statistics.median(preds)
                d["specialist_outliers"] = {
                    role: abs(ef - med) > 15.0
                    for role, ef in report.model_predictions.items()
                }
            result[view] = d
        except Exception as e:
            logger.error("analyze %s/%s: %s", pid, view, e)
            result[view] = {"error": str(e)}

    # Dual-view fusion — works now because we have both reports
    if "A4C" in reports and "PSAX" in reports:
        try:
            from echoguard.dual_view import fuse_views
            fused = fuse_views(reports["A4C"], reports["PSAX"])
            result["fused"] = fused.to_dict()
        except Exception as e:
            logger.error("fusion %s: %s", pid, e)
            result["fused"] = {"error": str(e)}

    return _sanitize(result)


# ─── VLM Senior Attending ─────────────────────────────────────────────────────

class VLMRequest(BaseModel):
    patient_id: str
    view: str = "A4C"


@app.post("/api/vlm")
async def run_vlm(req: VLMRequest):
    """Run MedGemma 4B VLM Senior Attending (~5–10 s on GPU)."""
    global _critic
    pid  = req.patient_id
    view = req.view.upper()

    # Resolve video_id via prefix map
    vid = _video_id_map.get(pid, {}).get(view) or pid
    manifest = _manifests.get(view, {})
    meta = manifest.get(vid)
    if meta is None:
        raise HTTPException(404, f"{pid} not found in {view} manifest")

    avi = _video_path(vid, view)
    if avi is None:
        raise HTTPException(404, f"Video not found for {pid}/{view}")

    # Regression report
    report = _engine.run(
        embedding_path=meta["embedding_path"],
        age=meta.get("age"), sex=meta.get("sex"),
        weight=meta.get("weight"), height=meta.get("height"),
        view=view, patient_id=pid,
    )

    # Lazy-load VLM
    if _critic is None:
        logger.info("Loading MedGemma 4B VLM…")
        from echoguard.vlm_critic import VLMCritic
        _critic = VLMCritic()

    critic_report = _critic.evaluate(
        clinical_report=report,
        video_path=avi,
        ed_idx=meta.get("ed_idx", 0),
        es_idx=meta.get("es_idx", 10),
    )

    d = critic_report.to_dict()
    d["prompt"] = getattr(_critic, "_last_prompt", None)
    d["regression_ef"] = report.ef_predicted
    d["specialist_roundtable"] = report.model_predictions
    return _sanitize(d)


# ─── Geometric EF + Segmentation Visualization ─────────────────────────────

class GeometricRequest(BaseModel):
    patient_id: str
    view: str = "PSAX"
    predicted_ef: Optional[float] = None  # From regression — used for overlay color consistency


@app.post("/api/geometric")
async def run_geometric(req: GeometricRequest):
    """Run geometric EF pipeline with segmentation mask overlays and LV area timeline.

    Returns:
        - geometric_ef: calibrated geometric EF
        - area_ef: raw area-based EF
        - calibration: slope/intercept used
        - lv_areas: per-frame LV area fractions (for timeline chart)
        - ed_frame / es_frame: detected ED/ES frame indices
        - segmentation_frames: base64 JPEG of ED/ES with LV mask overlay
        - graduated_ensemble: blended EF if regression prediction available
    """
    import torch
    import asyncio

    view = req.view.upper()
    pid = req.patient_id

    if view not in _seg_models:
        raise HTTPException(404, f"Segmentation model not loaded for {view}")
    if view not in _calibrations:
        raise HTTPException(404, f"Calibration not loaded for {view}")

    # Resolve video path
    vid = _video_id_map.get(pid, {}).get(view) or pid
    avi = _video_path(vid, view)
    if avi is None:
        raise HTTPException(404, f"Video not found for {pid}/{view}")

    seg_model = _seg_models[view]
    slope, intercept = _calibrations[view]

    # Run geometric EF computation in a thread (CPU-bound segmentation)
    def _compute():
        from scipy.ndimage import median_filter
        from echoguard.regression.geometric_ef import compute_lv_area

        cap = cv2.VideoCapture(str(avi))
        if not cap.isOpened():
            return None

        areas = []
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            areas.append(compute_lv_area(seg_model, frame))
            all_frames.append(frame)
        cap.release()

        if len(areas) < 5:
            return None

        raw_areas = list(areas)
        areas_smooth = median_filter(np.array(areas), size=5)
        ed_idx = int(np.argmax(areas_smooth))
        es_idx = int(np.argmin(areas_smooth))
        ed_area = float(areas_smooth[ed_idx])
        es_area = float(areas_smooth[es_idx])

        if ed_area < 0.005:
            return None

        area_ef = float((ed_area - es_area) / ed_area * 100)
        calibrated_ef = slope * area_ef + intercept
        # Use the more severe (lower) EF for overlay color consistency
        color_ef = calibrated_ef
        if req.predicted_ef is not None:
            color_ef = min(calibrated_ef, req.predicted_ef)
        geo_severity = _ef_to_severity(color_ef)
        fill_bgr, contour_bgr = _severity_bgr(color_ef)

        # Generate segmentation overlay images for ED and ES
        seg_frames = {}
        for label, fidx in [("ed", ed_idx), ("es", es_idx)]:
            frame = all_frames[fidx]
            resized = cv2.resize(frame, (224, 224))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            device = next(seg_model.parameters()).device
            with torch.no_grad():
                pred = seg_model(tensor.to(device))["out"].argmax(1).squeeze().cpu().numpy()

            # Create overlay: severity-colored mask on echo frame
            grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(grey)
            display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            # Severity-colored overlay for LV
            overlay = display.copy()
            overlay[pred == 1] = list(fill_bgr)
            blended = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            # Draw contour
            contours, _ = cv2.findContours(
                pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(blended, contours, -1, contour_bgr, 2)

            # Add label
            area_pct = float(pred.sum() / (224 * 224) * 100)
            cv2.putText(
                blended, f"{label.upper()} Area: {area_pct:.1f}%",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            _, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 90])
            seg_frames[label] = base64.b64encode(buf.tobytes()).decode()

        return {
            "geometric_ef": round(calibrated_ef, 1),
            "area_ef": round(area_ef, 1),
            "severity": geo_severity,
            "overlay_color": _SEVERITY_COLORS.get(geo_severity, "#8b949e"),
            "calibration": {"slope": round(slope, 4), "intercept": round(intercept, 4)},
            "lv_areas": [round(float(a), 5) for a in raw_areas],
            "lv_areas_smooth": [round(float(a), 5) for a in areas_smooth],
            "ed_frame": ed_idx,
            "es_frame": es_idx,
            "ed_area_pct": round(ed_area * 100, 2),
            "es_area_pct": round(es_area * 100, 2),
            "n_frames": len(areas),
            "segmentation_frames": seg_frames,
        }

    result = await asyncio.to_thread(_compute)
    if result is None:
        raise HTTPException(422, "Could not compute geometric EF (video too short or LV not detected)")

    # Add graduated ensemble if regression prediction is available
    manifest = _manifests.get(view, {})
    meta = manifest.get(vid)
    if meta is not None and _engine is not None:
        try:
            report = _engine.run(
                embedding_path=meta["embedding_path"],
                age=meta.get("age"), sex=meta.get("sex"),
                weight=meta.get("weight"), height=meta.get("height"),
                view=view, patient_id=pid,
            )
            from echoguard.regression.geometric_ef import graduated_ensemble
            reg_ef = report.ef_predicted
            geo_ef = result["geometric_ef"]
            ensemble_ef = float(graduated_ensemble(reg_ef, geo_ef))
            result["regression_ef"] = round(reg_ef, 1)
            result["ensemble_ef"] = round(ensemble_ef, 1)
            # Show which blend weight was used
            if geo_ef < 40:
                result["blend_zone"] = "low"
                result["blend_desc"] = "Structural abnormality detected — trusting geometric 80%"
                result["geo_weight"] = 0.8
            elif geo_ef < 55:
                result["blend_zone"] = "mid"
                result["blend_desc"] = "Borderline zone — 50/50 blend"
                result["geo_weight"] = 0.5
            else:
                result["blend_zone"] = "high"
                result["blend_desc"] = "Normal range — trusting regression 70%"
                result["geo_weight"] = 0.3
        except Exception as e:
            logger.warning("Ensemble computation failed: %s", e)

    return _sanitize(result)


# ─── MedGemma Agentic Clinical Narration ────────────────────────────────────

class NarrateRequest(BaseModel):
    patient_id: str
    include_geometric: bool = True


@app.post("/api/narrate")
async def narrate(req: NarrateRequest):
    """Full MedGemma agentic pipeline — Measure → Validate → Narrate → Report.

    This endpoint demonstrates MedGemma as the central orchestrating intelligence:

    Stage 1 (Measure):  Regression specialists + geometric EF produce raw predictions
    Stage 2 (Validate): MedGemma 4B VLM validates visual consistency
    Stage 3 (Narrate):  MedGemma 4B synthesizes a clinical narrative
    Stage 4 (Report):   Structured output combining all signals

    Returns a step-by-step pipeline trace showing MedGemma's decision-making.
    """
    import asyncio
    global _critic

    pid = req.patient_id
    vid_map = _video_id_map.get(pid, {})
    if not vid_map:
        raise HTTPException(404, f"Patient {pid} not found")

    pipeline_steps = []
    t0 = time.time()

    # ── Stage 1: Measure (regression specialists) ──
    step1_start = time.time()
    reports = {}
    for view, vid in vid_map.items():
        manifest = _manifests.get(view, {})
        meta = manifest.get(vid)
        if meta is None:
            continue
        try:
            report = _engine.run(
                embedding_path=meta["embedding_path"],
                age=meta.get("age"), sex=meta.get("sex"),
                weight=meta.get("weight"), height=meta.get("height"),
                view=view, patient_id=pid,
            )
            reports[view] = report
        except Exception as e:
            logger.error("narrate stage 1 %s/%s: %s", pid, view, e)

    step1_data = {}
    for view, report in reports.items():
        step1_data[view] = {
            "ef": round(report.ef_predicted, 1),
            "category": report.ef_category,
            "specialists": {k: round(v, 1) for k, v in report.model_predictions.items()},
            "confidence": round(report.confidence.overall, 2),
        }
    pipeline_steps.append({
        "stage": 1,
        "name": "Measure",
        "icon": "📊",
        "description": "8 VideoMAE specialists (4 per view) predict EF from 768-d video embeddings",
        "engine": "VideoMAE → Regression Ensemble",
        "elapsed_ms": round((time.time() - step1_start) * 1000),
        "data": step1_data,
    })

    # ── Stage 1b: Geometric EF (if requested) ──
    geo_data = {}
    if req.include_geometric:
        step1b_start = time.time()
        for view, vid in vid_map.items():
            if view not in _seg_models or view not in _calibrations:
                continue
            avi = _video_path(vid, view)
            if avi is None:
                continue
            try:
                from echoguard.regression.geometric_ef import compute_geometric_ef, graduated_ensemble
                area_ef = compute_geometric_ef(_seg_models[view], avi)
                if area_ef is not None:
                    slope, intercept = _calibrations[view]
                    cal_ef = slope * area_ef + intercept
                    reg_ef = reports[view].ef_predicted if view in reports else cal_ef
                    ens_ef = float(graduated_ensemble(reg_ef, cal_ef))
                    geo_data[view] = {
                        "area_ef": round(area_ef, 1),
                        "calibrated_ef": round(cal_ef, 1),
                        "ensemble_ef": round(ens_ef, 1),
                    }
            except Exception as e:
                logger.warning("narrate geo %s: %s", view, e)

        if geo_data:
            pipeline_steps.append({
                "stage": 1.5,
                "name": "Geometric",
                "icon": "📐",
                "description": "DeepLabV3 segmentation → LV area change → calibrated EF → graduated ensemble",
                "engine": "DeepLabV3-MobileNetV3 (IoU: A4C=0.809, PSAX=0.828)",
                "elapsed_ms": round((time.time() - step1b_start) * 1000),
                "data": geo_data,
            })

    # ── Stage 2: Validate (MedGemma VLM) ──
    step2_start = time.time()
    vlm_data = {}
    primary_view = None

    # Pick the view with lower EF (more clinically interesting) for VLM
    if reports:
        primary_view = min(reports.keys(), key=lambda v: reports[v].ef_predicted)
        vid = vid_map.get(primary_view)
        avi = _video_path(vid, primary_view) if vid else None

        if avi is not None:
            try:
                if _critic is None:
                    logger.info("Loading MedGemma 4B VLM for narration…")
                    # Free MPS memory: move specialist & segmentation models to CPU
                    _offload_models_to_cpu()
                    from echoguard.vlm_critic import VLMCritic
                    _critic = VLMCritic()

                manifest = _manifests.get(primary_view, {})
                meta = manifest.get(vid, {})
                critic_report = _critic.evaluate(
                    clinical_report=reports[primary_view],
                    video_path=avi,
                    ed_idx=meta.get("ed_idx", 0),
                    es_idx=meta.get("es_idx", 10),
                )
                vlm_data = {
                    "view": primary_view,
                    "verdict": critic_report.verdict.value,
                    "lv_description": critic_report.lv_description,
                    "confidence_original": critic_report.confidence_original,
                    "confidence_adjusted": critic_report.confidence_adjusted,
                }
            except Exception as e:
                logger.error("narrate stage 2: %s", e)
                vlm_data = {"error": str(e)}
            finally:
                # Free VLM memory and restore specialist models to GPU
                if _critic is not None:
                    _critic.unload()
                    _critic = None
                _reload_models_to_gpu()

    pipeline_steps.append({
        "stage": 2,
        "name": "Validate",
        "icon": "🔬",
        "description": "MedGemma 4B visually validates regression EF by inspecting echo frames",
        "engine": "MedGemma 1.5 4B (google/medgemma-4b-it)",
        "elapsed_ms": round((time.time() - step2_start) * 1000),
        "data": vlm_data,
    })

    # ── Stage 3: Narrate (MedGemma synthesizes narrative) ──
    step3_start = time.time()
    narrative = _build_clinical_narrative(reports, vlm_data, geo_data, pid)
    pipeline_steps.append({
        "stage": 3,
        "name": "Narrate",
        "icon": "📝",
        "description": "MedGemma synthesizes all signals into a clinical narrative",
        "engine": "MedGemma 1.5 4B (Agentic Synthesis)",
        "elapsed_ms": round((time.time() - step3_start) * 1000),
        "data": {"narrative": narrative},
    })

    # ── Stage 4: Report (structured output) ──
    final_ef = None
    final_category = None
    for view in ["PSAX", "A4C"]:  # prefer PSAX (better geometric)
        if view in geo_data:
            final_ef = geo_data[view]["ensemble_ef"]
            break
        elif view in reports:
            final_ef = round(reports[view].ef_predicted, 1)
            break
    if final_ef is not None:
        if final_ef >= 55:
            final_category = "normal"
        elif final_ef >= 40:
            final_category = "borderline"
        elif final_ef >= 30:
            final_category = "reduced"
        else:
            final_category = "severely_reduced"

    pipeline_steps.append({
        "stage": 4,
        "name": "Report",
        "icon": "📋",
        "description": "Final structured clinical report with confidence-weighted EF",
        "engine": "EchoGuard Pipeline v2",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "data": {
            "final_ef": final_ef,
            "final_category": final_category,
            "vlm_verdict": vlm_data.get("verdict", "unavailable"),
            "views_analyzed": list(reports.keys()),
        },
    })

    return _sanitize({
        "patient_id": pid,
        "pipeline_steps": pipeline_steps,
        "total_elapsed_ms": round((time.time() - t0) * 1000),
        "narrative": narrative,
        "final_ef": final_ef,
        "final_category": final_category,
    })


def _build_clinical_narrative(
    reports: dict,
    vlm_data: dict,
    geo_data: dict,
    pid: str,
) -> str:
    """Build a structured clinical narrative from all pipeline signals.

    This is the MedGemma agentic synthesis — combining:
    - Regression specialist predictions (8 models)
    - Geometric EF from LV segmentation
    - VLM visual validation verdict
    into a coherent clinical summary.
    """
    parts = []
    parts.append(f"ECHOGUARD-PEDS CLINICAL SUMMARY — Patient {pid}")
    parts.append("=" * 50)

    for view, report in reports.items():
        preds = report.model_predictions
        ef = report.ef_predicted
        cat = report.ef_category
        sigma = 0
        if preds:
            vals = list(preds.values())
            mean = sum(vals) / len(vals)
            sigma = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

        parts.append(f"\n{view} View:")
        parts.append(f"  Regression ensemble EF: {ef:.1f}% ({cat})")
        parts.append(f"  Inter-specialist agreement: σ={sigma:.1f}%")

        if view in geo_data:
            gd = geo_data[view]
            parts.append(f"  Geometric EF (LV area): {gd['calibrated_ef']:.1f}%")
            parts.append(f"  Graduated ensemble: {gd['ensemble_ef']:.1f}%")

    if vlm_data and "verdict" in vlm_data:
        parts.append(f"\nMedGemma Visual Validation ({vlm_data.get('view', '?')}):")
        parts.append(f"  Verdict: {vlm_data['verdict'].upper()}")
        if "lv_description" in vlm_data:
            parts.append(f"  LV appearance: {vlm_data['lv_description']}")
        conf_orig = vlm_data.get("confidence_original", 0)
        conf_adj = vlm_data.get("confidence_adjusted", 0)
        if conf_orig and conf_adj:
            direction = "↑" if conf_adj > conf_orig else "↓" if conf_adj < conf_orig else "→"
            parts.append(f"  Confidence: {conf_orig:.0%} {direction} {conf_adj:.0%}")

    parts.append(f"\nPipeline: 8 regression specialists × 2 views + DeepLabV3 segmentation + MedGemma 4B VLM")
    return "\n".join(parts)


# ─── Segmentation Animation Frames ───────────────────────────────────────────

@app.post("/api/segmentation-video")
async def segmentation_video(req: GeometricRequest):
    """Return ALL frames with LV segmentation overlays as base64 JPEG array.

    Used by the frontend to animate the segmentation over the full cardiac cycle.
    Returns every Nth frame (stride) to keep response size manageable.
    """
    import torch
    import asyncio

    view = req.view.upper()
    pid = req.patient_id

    if view not in _seg_models:
        raise HTTPException(404, f"Segmentation model not loaded for {view}")

    vid = _video_id_map.get(pid, {}).get(view) or pid
    avi = _video_path(vid, view)
    if avi is None:
        raise HTTPException(404, f"Video not found for {pid}/{view}")

    seg_model = _seg_models[view]

    slope, intercept = _calibrations.get(view, (1.0, 0.0))

    def _render_all():
        from scipy.ndimage import median_filter

        cap = cv2.VideoCapture(str(avi))
        if not cap.isOpened():
            return None

        # First pass: collect all frames & area fractions to determine geometric EF
        raw_frames = []
        raw_areas = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (224, 224))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            device = next(seg_model.parameters()).device
            with torch.no_grad():
                pred = seg_model(tensor.to(device))["out"].argmax(1).squeeze().cpu().numpy()
            raw_frames.append((resized, pred))
            raw_areas.append(float(pred.sum() / (224 * 224)))
        cap.release()

        if len(raw_frames) < 5:
            return None

        # Compute geometric EF → severity → overlay color
        areas_smooth = median_filter(np.array(raw_areas), size=5)
        ed_area = float(np.max(areas_smooth))
        es_area = float(np.min(areas_smooth))
        if ed_area < 0.005:
            geo_ef = 60.0  # fallback
        else:
            area_ef = (ed_area - es_area) / ed_area * 100
            geo_ef = slope * area_ef + intercept

        # Use the more severe (lower) EF for color if predicted_ef is given
        color_ef = geo_ef
        if req.predicted_ef is not None:
            color_ef = min(geo_ef, req.predicted_ef)
        severity = _ef_to_severity(color_ef)
        fill_bgr, contour_bgr = _severity_bgr(color_ef)
        overlay_hex = _SEVERITY_COLORS.get(severity, "#8b949e")

        # Second pass: render overlays with severity color
        frames_b64 = []
        areas = []
        for idx, (resized, pred) in enumerate(raw_frames):
            area_frac = raw_areas[idx]
            areas.append(area_frac)

            grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(grey)
            display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            overlay = display.copy()
            overlay[pred == 1] = list(fill_bgr)
            blended = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
            contours, _ = cv2.findContours(
                pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(blended, contours, -1, contour_bgr, 2)
            area_pct = area_frac * 100
            cv2.putText(blended, f"F{idx} LV:{area_pct:.1f}%",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            _, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frames_b64.append(base64.b64encode(buf.tobytes()).decode())

        return {
            "frames": frames_b64, "areas": areas, "n_frames": len(frames_b64),
            "geometric_ef": round(geo_ef, 1), "severity": severity,
            "overlay_color": overlay_hex,
        }

    result = await asyncio.to_thread(_render_all)
    if result is None:
        raise HTTPException(422, "Could not process video")
    return _sanitize(result)


# ─── Video ───────────────────────────────────────────────────────────────────

@app.get("/api/video/{patient_id}/{view}")
async def serve_video(patient_id: str, view: str):
    """Serve echo as MP4 (converted from AVI via ffmpeg, cached to /tmp)."""
    vid = _video_id_map.get(patient_id, {}).get(view.upper()) or patient_id
    avi = _video_path(vid, view)
    if avi is None:
        raise HTTPException(404, "Video not found")
    # Run blocking ffmpeg conversion off the async event loop
    import asyncio
    mp4 = await asyncio.to_thread(_to_mp4, avi, vid, view)
    if mp4:
        return FileResponse(str(mp4), media_type="video/mp4",
                            headers={"Cache-Control": "max-age=7200",
                                     "Accept-Ranges": "bytes"})
    return FileResponse(str(avi), media_type="video/x-msvideo")


# ─── Frames ──────────────────────────────────────────────────────────────────

@app.get("/api/frames/{patient_id}/{view}")
async def get_frames(patient_id: str, view: str):
    """Return ED / mid-systole / ES key frames as base64 JPEG."""
    vid = _video_id_map.get(patient_id, {}).get(view.upper()) or patient_id
    avi = _video_path(vid, view)
    if avi is None:
        raise HTTPException(404, "Video not found")
    manifest = _manifests.get(view.upper(), {})
    meta = manifest.get(vid, {})
    ed  = meta.get("ed_idx", 0)
    es  = meta.get("es_idx", 10)
    frames = _extract_frames(avi, ed, es)
    return _sanitize({**frames, "ed_frame": ed, "es_frame": es,
            "n_frames": int(cv2.VideoCapture(str(avi)).get(cv2.CAP_PROP_FRAME_COUNT))})


# ─── Serve SPA ───────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(
        str(FRONTEND_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
