#!/usr/bin/env python3
"""Generate README demo MP4 videos: raw echo + LV segmentation overlay.

Creates 3 MP4 videos:
  1) a4c_segmentation.mp4 — A4C normal vs reduced EF, side-by-side raw + overlay
  2) psax_segmentation.mp4 — PSAX normal vs reduced EF, side-by-side raw + overlay
  3) combined_demo.mp4 — Both views together in one showcase video

Each video shows the full cardiac cycle with real-time LV segmentation,
ED/ES markers, and EF annotation — giving viewers an immediate visual
understanding of what the AI is doing.

Usage:
    python generate_videos.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models.segmentation as seg_models

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
A4C_DIR = PROJECT_ROOT / "data" / "echonet_pediatric" / "A4C"
PSAX_DIR = PROJECT_ROOT / "data" / "echonet_pediatric" / "PSAX"
A4C_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "lv_seg_deeplabv3.pt"
PSAX_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "lv_seg_psax_deeplabv3.pt"
OUTPUT_DIR = PROJECT_ROOT / "figures"

SEG_SIZE = 224       # segmentation model input
DISPLAY_SIZE = 300   # upscaled size for each panel in the output video
FPS = 28             # native echo frame rate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_seg_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = seg_models.deeplabv3_mobilenet_v3_large(
        weights_backbone=None, num_classes=2,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def segment_frame(model: torch.nn.Module, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Return binary mask (H, W) for a single BGR frame."""
    resized = cv2.resize(frame_bgr, (SEG_SIZE, SEG_SIZE))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = model(tensor.to(device))["out"].argmax(1).squeeze().cpu().numpy()
    return pred.astype(np.uint8)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------
def read_all_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_ed_es_from_tracings(tracings_csv: Path, filename: str) -> tuple[int, int]:
    """ED = largest LV area, ES = smallest LV area from expert tracings."""
    df = pd.read_csv(tracings_csv)
    vt = df[df["FileName"] == filename]
    if vt.empty:
        raise ValueError(f"No tracings for {filename}")
    frame_areas = {}
    for frame_idx, grp in vt.groupby("Frame"):
        xs, ys = grp["X"].values, grp["Y"].values
        area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
        frame_areas[frame_idx] = area
    return int(max(frame_areas, key=frame_areas.get)), int(min(frame_areas, key=frame_areas.get))


def pick_patients(file_list_csv: Path, tracings_csv: Path) -> tuple[dict, dict]:
    """Pick one reduced-EF and one normal-EF patient with tracings."""
    fl = pd.read_csv(file_list_csv)
    tracings = pd.read_csv(tracings_csv)
    traced = set(tracings["FileName"].unique())
    fl = fl[fl["FileName"].isin(traced)]
    reduced = fl[(fl["EF"] < 35) & (fl["EF"] > 15)].sort_values("EF")
    normal = fl[(fl["EF"] > 60) & (fl["EF"] < 70)].sort_values("EF", ascending=False)
    if reduced.empty or normal.empty:
        reduced = fl.nsmallest(5, "EF")
        normal = fl.nlargest(5, "EF")
    r = reduced.iloc[len(reduced) // 2]
    n = normal.iloc[len(normal) // 2]
    to_dict = lambda row: {"filename": row["FileName"], "ef": row["EF"],
                           "age": row["Age"], "sex": row["Sex"]}
    return to_dict(r), to_dict(n)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def create_overlay(frame_bgr: np.ndarray, mask: np.ndarray,
                   color=(0, 255, 100), alpha=0.35) -> np.ndarray:
    """Green LV segmentation overlay on upscaled frame (returns BGR)."""
    resized = cv2.resize(frame_bgr, (DISPLAY_SIZE, DISPLAY_SIZE))
    mask_up = cv2.resize(mask, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)

    overlay = resized.copy()
    colored = np.zeros_like(resized)
    colored[mask_up == 1] = color
    overlay = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)

    contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def put_text(img: np.ndarray, text: str, pos: tuple, scale=0.55, color=(255, 255, 255),
             thickness=1, bg=True):
    """Put white text with dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def add_border(img: np.ndarray, color: tuple, thickness=3) -> np.ndarray:
    """Add a colored border around the image."""
    bordered = img.copy()
    cv2.rectangle(bordered, (0, 0), (img.shape[1]-1, img.shape[0]-1), color, thickness)
    return bordered


# ---------------------------------------------------------------------------
# Per-view video generation
# ---------------------------------------------------------------------------
def generate_view_video(view: str, device: torch.device) -> Path:
    """Generate an MP4 for a single view showing normal + reduced side-by-side."""
    data_dir = A4C_DIR if view == "A4C" else PSAX_DIR
    ckpt = A4C_CHECKPOINT if view == "A4C" else PSAX_CHECKPOINT

    print(f"\n{'='*50}")
    print(f"Generating {view} segmentation video")
    print(f"{'='*50}")

    model = load_seg_model(ckpt, device)

    fl_csv = data_dir / "FileList.csv"
    tr_csv = data_dir / "VolumeTracings.csv"
    reduced_p, normal_p = pick_patients(fl_csv, tr_csv)
    print(f"  Normal:  {normal_p['filename']} (EF={normal_p['ef']:.1f}%)")
    print(f"  Reduced: {reduced_p['filename']} (EF={reduced_p['ef']:.1f}%)")

    # Load frames
    normal_frames = read_all_frames(data_dir / "Videos" / normal_p["filename"])
    reduced_frames = read_all_frames(data_dir / "Videos" / reduced_p["filename"])

    # Get ED/ES
    n_ed, n_es = get_ed_es_from_tracings(tr_csv, normal_p["filename"])
    r_ed, r_es = get_ed_es_from_tracings(tr_csv, reduced_p["filename"])

    # Segment all frames
    print("  Segmenting normal patient frames...")
    normal_masks = [segment_frame(model, f, device) for f in normal_frames]
    print("  Segmenting reduced patient frames...")
    reduced_masks = [segment_frame(model, f, device) for f in reduced_frames]

    # Compute LV areas for area bar
    def lv_area_pct(mask):
        return mask.sum() / (SEG_SIZE * SEG_SIZE) * 100

    normal_areas = [lv_area_pct(m) for m in normal_masks]
    reduced_areas = [lv_area_pct(m) for m in reduced_masks]

    # Match video lengths: loop shorter to length of longer, then do 2 full loops
    max_len = max(len(normal_frames), len(reduced_frames))
    total_frames = max_len * 3  # 3 loops for visual effect

    # Layout:
    # Title bar (40px)
    # [Normal Raw | Normal+Seg] label (30px) + panels (DISPLAY_SIZE)
    # gap (10px)
    # [Reduced Raw | Reduced+Seg] label (30px) + panels (DISPLAY_SIZE)
    # Bottom info bar (50px)

    panel_w = DISPLAY_SIZE
    canvas_w = panel_w * 2 + 10  # 2 panels + 10px gap per row
    title_h = 50
    label_h = 28
    row_h = DISPLAY_SIZE
    gap_h = 12
    info_h = 55
    canvas_h = title_h + label_h + row_h + gap_h + label_h + row_h + info_h

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{view.lower()}_segmentation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (canvas_w, canvas_h))

    view_full = "Apical Four-Chamber" if view == "A4C" else "Parasternal Short-Axis"

    print(f"  Rendering {total_frames} frames...")
    for i in range(total_frames):
        n_idx = i % len(normal_frames)
        r_idx = i % len(reduced_frames)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # -- Title bar --
        put_text(canvas, f"project_echo  |  LV Segmentation  |  {view} ({view_full})",
                 (10, 32), scale=0.6, color=(200, 220, 255), thickness=1)

        # -- Normal row --
        y_label_n = title_h
        y_panel_n = y_label_n + label_h
        status_str = f"Normal Function (EF={normal_p['ef']:.0f}%)"
        put_text(canvas, status_str, (10, y_label_n + 20), scale=0.55,
                 color=(100, 255, 100), thickness=1, bg=False)

        # Raw frame
        raw_n = cv2.resize(normal_frames[n_idx], (panel_w, panel_w))
        canvas[y_panel_n:y_panel_n + row_h, 0:panel_w] = raw_n

        # Overlay
        ov_n = create_overlay(normal_frames[n_idx], normal_masks[n_idx])
        canvas[y_panel_n:y_panel_n + row_h, panel_w + 10:panel_w * 2 + 10] = ov_n

        # ED/ES marker on normal
        if n_idx == n_ed:
            put_text(canvas, "ED", (panel_w + 15, y_panel_n + 25), scale=0.7,
                     color=(100, 255, 100), thickness=2)
        elif n_idx == n_es:
            put_text(canvas, "ES", (panel_w + 15, y_panel_n + 25), scale=0.7,
                     color=(100, 100, 255), thickness=2)

        # Area text on overlay panel
        area_n = normal_areas[n_idx]
        put_text(canvas, f"LV area: {area_n:.1f}%",
                 (panel_w + 15, y_panel_n + row_h - 15), scale=0.45, color=(200, 255, 200))

        # Labels
        put_text(canvas, "Raw Echo", (panel_w // 2 - 40, y_panel_n + 20),
                 scale=0.45, color=(200, 200, 200))
        put_text(canvas, "AI Segmentation", (panel_w + panel_w // 2 - 50, y_panel_n + 20),
                 scale=0.45, color=(200, 200, 200))

        # -- Reduced row --
        y_label_r = y_panel_n + row_h + gap_h
        y_panel_r = y_label_r + label_h
        status_str_r = f"Reduced Function (EF={reduced_p['ef']:.0f}%)"
        put_text(canvas, status_str_r, (10, y_label_r + 20), scale=0.55,
                 color=(100, 100, 255), thickness=1, bg=False)

        # Raw frame
        raw_r = cv2.resize(reduced_frames[r_idx], (panel_w, panel_w))
        canvas[y_panel_r:y_panel_r + row_h, 0:panel_w] = raw_r

        # Overlay
        ov_r = create_overlay(reduced_frames[r_idx], reduced_masks[r_idx])
        canvas[y_panel_r:y_panel_r + row_h, panel_w + 10:panel_w * 2 + 10] = ov_r

        # ED/ES marker on reduced
        if r_idx == r_ed:
            put_text(canvas, "ED", (panel_w + 15, y_panel_r + 25), scale=0.7,
                     color=(100, 255, 100), thickness=2)
        elif r_idx == r_es:
            put_text(canvas, "ES", (panel_w + 15, y_panel_r + 25), scale=0.7,
                     color=(100, 100, 255), thickness=2)

        # Area text
        area_r = reduced_areas[r_idx]
        put_text(canvas, f"LV area: {area_r:.1f}%",
                 (panel_w + 15, y_panel_r + row_h - 15), scale=0.45, color=(200, 200, 255))

        # Labels
        put_text(canvas, "Raw Echo", (panel_w // 2 - 40, y_panel_r + 20),
                 scale=0.45, color=(200, 200, 200))
        put_text(canvas, "AI Segmentation", (panel_w + panel_w // 2 - 50, y_panel_r + 20),
                 scale=0.45, color=(200, 200, 200))

        # -- Bottom info bar --
        y_info = y_panel_r + row_h + 5
        normal_age_str = f"{normal_p['age']:.0f}y" if normal_p['age'] >= 1 else f"{normal_p['age']*12:.0f}mo"
        reduced_age_str = f"{reduced_p['age']:.0f}y" if reduced_p['age'] >= 1 else f"{reduced_p['age']*12:.0f}mo"
        put_text(canvas, f"Normal: {normal_p['sex']}, {normal_age_str}  |  Reduced: {reduced_p['sex']}, {reduced_age_str}",
                 (10, y_info + 18), scale=0.45, color=(180, 180, 180), bg=False)
        put_text(canvas, "DeepLabV3-MobileNetV3  |  Green = AI-detected LV boundary",
                 (10, y_info + 38), scale=0.42, color=(140, 140, 140), bg=False)

        writer.write(canvas)

    writer.release()

    # Re-encode with ffmpeg for H.264 (GitHub compatible)
    h264_path = out_path.with_suffix(".h264.mp4")
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(out_path),
        "-c:v", "libx264", "-preset", "slow", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(h264_path),
    ], capture_output=True, check=True)
    # Replace the original
    h264_path.rename(out_path)

    file_size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {out_path} ({file_size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate per-view videos
    a4c_path = generate_view_video("A4C", device)
    psax_path = generate_view_video("PSAX", device)

    print(f"\n{'='*50}")
    print("All videos generated!")
    print(f"  {a4c_path}")
    print(f"  {psax_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
