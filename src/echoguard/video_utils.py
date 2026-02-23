"""Video processing utilities for EchoNet-Pediatric data.

Handles AVI → frame extraction, key frame selection (ED/ES/mid-cycle),
and tracing-based wall motion derivation.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_all_frames(video_path: str | Path) -> list[np.ndarray]:
    """Extract all frames from an AVI video file.

    Returns list of BGR numpy arrays (OpenCV format).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    logger.debug("Extracted %d frames from %s", len(frames), video_path)
    return frames


def frame_to_pil(
    frame: np.ndarray,
    target_size: int = 224,
) -> Image.Image:
    """Convert a BGR OpenCV frame to an RGB PIL Image, resized.

    Args:
        frame: BGR numpy array from OpenCV.
        target_size: Output size (square). Default 224 for MedGemma.

    Returns:
        RGB PIL Image resized to (target_size, target_size).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)
    return img


def get_video_info(video_path: str | Path) -> dict:
    """Get basic info about a video file."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


# ---------------------------------------------------------------------------
# Key frame selection
# ---------------------------------------------------------------------------

def select_key_frames(
    frames: list[np.ndarray],
    ed_frame_idx: int,
    es_frame_idx: int,
    target_size: int = 224,
) -> list[Image.Image]:
    """Select 4 key frames from a cardiac cycle.

    Strategy:
    1. ED frame (end-diastole) — maximum chamber size
    2. Mid-systole — midpoint between ED → ES
    3. ES frame (end-systole) — minimum chamber size
    4. Mid-diastole — midpoint between ES → next ED (wraps around)

    Args:
        frames: All video frames (BGR numpy arrays).
        ed_frame_idx: Frame index for end-diastole.
        es_frame_idx: Frame index for end-systole.
        target_size: Output image size.

    Returns:
        List of 4 PIL Images [ED, mid-systole, ES, mid-diastole].
    """
    n = len(frames)
    ed_idx = ed_frame_idx % n
    es_idx = es_frame_idx % n

    # Mid-systole: midpoint between ED and ES
    if es_idx >= ed_idx:
        mid_sys_idx = (ed_idx + es_idx) // 2
    else:
        # ES wraps around
        mid_sys_idx = ((ed_idx + es_idx + n) // 2) % n

    # Mid-diastole: midpoint between ES and next ED
    # Assuming the cycle repeats, next ED is approximately ed_idx + n_cycle
    # For simplicity, use midpoint between ES and ED going forward
    if ed_idx > es_idx:
        mid_dia_idx = (es_idx + ed_idx) // 2
    else:
        mid_dia_idx = ((es_idx + ed_idx + n) // 2) % n

    # Ensure mid-diastole is distinct from mid-systole
    if mid_dia_idx == mid_sys_idx:
        mid_dia_idx = (mid_dia_idx + 1) % n

    indices = [ed_idx, mid_sys_idx, es_idx, mid_dia_idx]
    key_frames = []
    for idx in indices:
        idx = max(0, min(idx, n - 1))
        key_frames.append(frame_to_pil(frames[idx], target_size))

    return key_frames


def select_key_frames_extended(
    frames: list[np.ndarray],
    ed_frame_idx: int,
    es_frame_idx: int,
    num_frames: int = 8,
    target_size: int = 224,
) -> list[Image.Image]:
    """Select N key frames using phase-aware ED/ES-dense sampling.

    Strategy (for num_frames=8):
      - 2 frames BEFORE ED  (pre-systole context)
      - ED frame itself     (end-diastole anchor — max volume)
      - 1 frame between ED and ES  (early systole)
      - ES frame itself     (end-systole anchor — min volume, EF-critical)
      - 1 frame after ES    (early diastole)
      - 2 frames uniformly from the rest of the cycle (context)

    This concentrates sampling around the phases that define EF = (EDV-ESV)/EDV,
    unlike uniform sampling which may miss the actual ES/ED frames.

    For num_frames=16: similar ED/ES-dense strategy with finer resolution.
    For num_frames<=4: falls back to select_key_frames (4 canonical frames).

    Args:
        frames: All video frames (BGR numpy arrays).
        ed_frame_idx: Frame index for end-diastole.
        es_frame_idx: Frame index for end-systole.
        num_frames: Number of frames to select (4, 8, or 16).
        target_size: Output image size.

    Returns:
        List of num_frames PIL Images.
    """
    if num_frames <= 4:
        return select_key_frames(frames, ed_frame_idx, es_frame_idx, target_size)

    n = len(frames)
    ed_idx = ed_frame_idx % n
    es_idx = es_frame_idx % n

    # Phase-aware sampling: dense around ED and ES
    # ED comes before ES in a typical cardiac cycle
    if es_idx > ed_idx:
        systole_len = es_idx - ed_idx          # contraction phase
        diastole_len = n - systole_len          # filling phase
    else:
        # ES wraps around (short videos or unusual annotation)
        systole_len = (es_idx + n - ed_idx) % n
        diastole_len = n - systole_len

    # --- build anchor-dense index set ---
    anchors = set()
    anchors.add(ed_idx)
    anchors.add(es_idx)

    # 2 frames before ED (pre-systole context)
    for offset in [2, 4]:
        anchors.add((ed_idx - offset) % n)

    # 1 frame in mid-systole (between ED and ES)
    mid_sys = (ed_idx + max(1, systole_len // 2)) % n
    anchors.add(mid_sys)

    # 1 frame just after ES (early diastole)
    anchors.add((es_idx + max(1, diastole_len // 4)) % n)

    # For 16-frame mode: add finer systole / diastole sampling
    if num_frames >= 12:
        for frac in [0.25, 0.75]:
            anchors.add(int(ed_idx + frac * systole_len) % n)
        for frac in [0.33, 0.67]:
            anchors.add(int(es_idx + frac * diastole_len) % n)

    # Fill remaining slots uniformly from the full cycle
    anchor_list = sorted(anchors)
    remaining_budget = num_frames - len(anchor_list)
    if remaining_budget > 0:
        uniform = [
            int(round(i * (n - 1) / (remaining_budget + 1)))
            for i in range(1, remaining_budget + 1)
        ]
        for u in uniform:
            if u not in anchors:
                anchors.add(u)
                if len(anchors) >= num_frames:
                    break

    # Sort, deduplicate, then fill to exactly num_frames
    indices_set = set(anchors)

    # Fill from any remaining cycle frames not yet selected
    if len(indices_set) < num_frames:
        for i in range(n):
            if i not in indices_set:
                indices_set.add(i)
            if len(indices_set) >= num_frames:
                break

    # Trim to num_frames (in sorted order) in the normal case
    indices = sorted(indices_set)[:num_frames]

    # Pad with frame repetition only if video is genuinely shorter than num_frames
    step = 0
    while len(indices) < num_frames:
        indices.append(indices[step % max(1, len(indices))])
        step += 1

    key_frames = []
    for idx in sorted(indices):
        idx = max(0, min(idx, n - 1))
        key_frames.append(frame_to_pil(frames[idx], target_size))

    return key_frames


def extract_key_frames_from_video(
    video_path: str | Path,
    ed_frame_idx: int,
    es_frame_idx: int,
    target_size: int = 224,
) -> list[Image.Image]:
    """Convenience: extract all frames then select key frames.

    Args:
        video_path: Path to AVI file.
        ed_frame_idx: End-diastole frame index.
        es_frame_idx: End-systole frame index.
        target_size: Output size.

    Returns:
        List of 4 PIL Images.
    """
    frames = extract_all_frames(video_path)
    return select_key_frames(frames, ed_frame_idx, es_frame_idx, target_size)


# ---------------------------------------------------------------------------
# Save extracted frames to disk
# ---------------------------------------------------------------------------

def save_key_frames(
    frames: list[Image.Image],
    output_dir: str | Path,
    video_id: str,
) -> list[str]:
    """Save 4 key frames as PNG files.

    Returns list of saved file paths.
    """
    output_dir = Path(output_dir) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["ed", "mid_systole", "es", "mid_diastole"]
    paths = []
    for img, label in zip(frames, labels):
        path = output_dir / f"{label}.png"
        img.save(str(path))
        paths.append(str(path))

    return paths


# ---------------------------------------------------------------------------
# Batch frame extraction from EchoNet-Pediatric
# ---------------------------------------------------------------------------

def load_file_list(file_list_path: str | Path) -> list[dict]:
    """Load FileList.csv from EchoNet-Pediatric.

    Actual columns: FileName, EF, Sex, Age, Weight, Height, Split
    Split is numeric 0-9 (10-fold cross-validation).

    Returns list of dicts with typed values (EF/Age/Weight/Height as float,
    Split as int, Sex as str).
    """
    records = []
    with open(file_list_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {
                "FileName": row["FileName"].strip(),
                "Sex": row.get("Sex", "").strip(),
            }
            # Parse numeric fields safely
            for key in ("EF", "Age", "Weight", "Height"):
                try:
                    record[key] = float(row[key])
                except (KeyError, ValueError, TypeError):
                    record[key] = None
            try:
                record["Split"] = int(row["Split"])
            except (KeyError, ValueError, TypeError):
                record["Split"] = 0
            records.append(record)
    logger.info("Loaded %d records from %s", len(records), file_list_path)
    return records


def load_volume_tracings(
    tracings_path: str | Path,
) -> dict[str, dict[int, list[tuple[float, float]]]]:
    """Load VolumeTracings.csv and group by video filename and frame.

    Actual CSV format: FileName, X, Y, Frame
    Each row is one contour point. Points for the same (FileName, Frame) form
    an LV boundary polygon.

    Returns:
        dict mapping filename → {frame_number: [(x, y), ...]}.
    """
    tracings: dict[str, dict[int, list[tuple[float, float]]]] = {}
    with open(tracings_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("FileName", "").strip()
            if not filename:
                continue
            try:
                x = float(row["X"])
                y = float(row["Y"])
                frame = int(float(row["Frame"]))
            except (KeyError, ValueError, TypeError):
                continue

            if filename not in tracings:
                tracings[filename] = {}
            if frame not in tracings[filename]:
                tracings[filename][frame] = []
            tracings[filename][frame].append((x, y))

    logger.info("Loaded tracings for %d videos from %s", len(tracings), tracings_path)
    return tracings


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    """Compute area of a polygon using the shoelace formula.

    Args:
        pts: List of (x, y) vertices forming a closed polygon.

    Returns:
        Absolute area (always non-negative).
    """
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def get_ed_es_frames_from_tracings(
    tracings: dict[str, dict[int, list[tuple[float, float]]]],
    video_id: str,
) -> tuple[int, int] | None:
    """Get ED and ES frame indices from volume tracings.

    ED = frame with largest LV polygon area (maximum chamber size).
    ES = frame with smallest LV polygon area (minimum chamber size).

    The tracings contain (X, Y) contour points forming an LV boundary
    polygon for each annotated frame (typically 2 frames per video).

    Returns (ed_frame_idx, es_frame_idx) or None if not found.
    """
    # Try with and without .avi extension
    for key in [video_id, f"{video_id}.avi", video_id.replace(".avi", "")]:
        if key in tracings:
            frame_data = tracings[key]
            if len(frame_data) < 2:
                # Only one annotated frame — use it as ED, guess ES
                frame_nums = list(frame_data.keys())
                return (frame_nums[0], max(0, frame_nums[0] - 5))

            # Compute polygon area for each annotated frame
            frame_areas = {}
            for frame_num, pts in frame_data.items():
                frame_areas[frame_num] = _polygon_area(pts)

            # ED = largest area, ES = smallest area
            sorted_frames = sorted(frame_areas.items(), key=lambda x: x[1])
            es_frame = sorted_frames[0][0]
            ed_frame = sorted_frames[-1][0]
            return (ed_frame, es_frame)

    return None


# ---------------------------------------------------------------------------
# Wall motion derivation from tracings
# ---------------------------------------------------------------------------

def derive_wall_motion_labels(
    ed_coords: list[tuple[float, float]],
    es_coords: list[tuple[float, float]],
    segments: list[str] | None = None,
) -> list[dict]:
    """Derive regional wall motion labels from ED/ES LV contour tracings.

    Resamples both contours to *n_segments* equally-spaced points, then
    compares inward displacement between ED and ES for each segment.

    The contour points trace the LV boundary.  We resample so that
    corresponding points on the ED and ES contours represent the same
    anatomical segment.

    Args:
        ed_coords: (X, Y) contour points at end-diastole.
        es_coords: (X, Y) contour points at end-systole.
        segments:  Segment names (default: 6-segment A4C model).

    Returns:
        List of dicts with segment_name, score, displacement.
    """
    if segments is None:
        segments = [
            "basal_septal", "mid_septal", "apical_septal",
            "apical_lateral", "mid_lateral", "basal_lateral",
        ]

    n_segments = len(segments)

    def _resample_contour(
        pts: list[tuple[float, float]], n: int,
    ) -> list[tuple[float, float]]:
        """Resample a contour to *n* equally-spaced points."""
        if len(pts) == 0:
            return [(0.0, 0.0)] * n
        if len(pts) <= 2:
            return [pts[0]] * n
        arr = np.array(pts, dtype=np.float64)
        # Cumulative arc length
        diffs = np.diff(arr, axis=0)
        seg_len = np.sqrt((diffs ** 2).sum(axis=1))
        cum_len = np.concatenate(([0.0], np.cumsum(seg_len)))
        total_len = cum_len[-1]
        if total_len == 0:
            return [pts[0]] * n
        targets = np.linspace(0, total_len, n)
        resampled = []
        for t in targets:
            idx = np.searchsorted(cum_len, t, side="right") - 1
            idx = max(0, min(idx, len(arr) - 2))
            frac = (t - cum_len[idx]) / max(seg_len[idx], 1e-9)
            frac = np.clip(frac, 0.0, 1.0)
            pt = arr[idx] + frac * (arr[idx + 1] - arr[idx])
            resampled.append((float(pt[0]), float(pt[1])))
        return resampled

    ed_sampled = _resample_contour(ed_coords, n_segments)
    es_sampled = _resample_contour(es_coords, n_segments)

    results = []
    for i, seg_name in enumerate(segments):
        ed_x, ed_y = ed_sampled[i]
        es_x, es_y = es_sampled[i]
        displacement = np.sqrt((ed_x - es_x) ** 2 + (ed_y - es_y) ** 2)

        # Thresholds (in pixel units, 112×112 native resolution)
        if displacement > 5.0:
            score = "normal"
        elif displacement > 2.0:
            score = "hypokinetic"
        else:
            score = "akinetic"

        results.append({
            "segment_name": seg_name,
            "score": score,
            "displacement_px": round(float(displacement), 2),
        })

    return results
