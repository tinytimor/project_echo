#!/usr/bin/env python3
"""Generate README figures: raw echo frames + LV segmentation overlays.

Creates a 2-row, 4-column figure:
  Row 1: Normal EF patient — ED frame, ES frame, ED+seg overlay, ES+seg overlay
  Row 2: Reduced EF patient — ED frame, ES frame, ED+seg overlay, ES+seg overlay

Also generates a separate PSAX segmentation figure.

Usage:
    python generate_figures.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_seg_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = seg_models.deeplabv3_mobilenet_v3_large(
        weights_backbone=None, num_classes=2,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def segment_frame(model: torch.nn.Module, frame: np.ndarray, device: torch.device) -> np.ndarray:
    """Return binary mask (H, W) for a single BGR frame."""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = model(tensor.to(device))["out"].argmax(1).squeeze().cpu().numpy()
    return pred.astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def get_ed_es_from_tracings(tracings_csv: Path, filename: str) -> tuple[int, int]:
    """Get ED and ES frame indices from VolumeTracings.csv."""
    df = pd.read_csv(tracings_csv)
    video_tracings = df[df["FileName"] == filename]
    if video_tracings.empty:
        raise ValueError(f"No tracings for {filename}")

    frames = video_tracings.groupby("Frame")
    # ED = frame with most enclosed area, ES = frame with least
    frame_areas = {}
    for frame_idx, group in frames:
        xs = group["X"].values
        ys = group["Y"].values
        # Shoelace formula for polygon area
        area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
        frame_areas[frame_idx] = area

    ed_frame = max(frame_areas, key=frame_areas.get)
    es_frame = min(frame_areas, key=frame_areas.get)
    return int(ed_frame), int(es_frame)


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Read a specific frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def create_overlay(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 100), alpha=0.4) -> np.ndarray:
    """Create a green segmentation overlay on the frame."""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    overlay = rgb.copy()
    colored = np.zeros_like(rgb)
    colored[mask == 1] = color
    overlay = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)

    # Draw contour edge
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


# ---------------------------------------------------------------------------
# Pick patients
# ---------------------------------------------------------------------------
def pick_patients(file_list_csv: Path, tracings_csv: Path) -> tuple[dict, dict]:
    """Pick one reduced-EF and one normal-EF patient from the dataset."""
    fl = pd.read_csv(file_list_csv)
    tracings = pd.read_csv(tracings_csv)
    traced_files = set(tracings["FileName"].unique())

    # Only files that have tracings
    fl = fl[fl["FileName"].isin(traced_files)]

    # Reduced: EF < 35%, prefer something dramatic
    reduced = fl[(fl["EF"] < 35) & (fl["EF"] > 15)].sort_values("EF")
    # Normal: EF > 60%
    normal = fl[(fl["EF"] > 60) & (fl["EF"] < 70)].sort_values("EF", ascending=False)

    if reduced.empty or normal.empty:
        # Fallback
        reduced = fl.nsmallest(5, "EF")
        normal = fl.nlargest(5, "EF")

    reduced_row = reduced.iloc[len(reduced) // 2]
    normal_row = normal.iloc[len(normal) // 2]

    return (
        {"filename": reduced_row["FileName"], "ef": reduced_row["EF"],
         "age": reduced_row["Age"], "sex": reduced_row["Sex"]},
        {"filename": normal_row["FileName"], "ef": normal_row["EF"],
         "age": normal_row["Age"], "sex": normal_row["Sex"]},
    )


# ---------------------------------------------------------------------------
# Main figure generation
# ---------------------------------------------------------------------------
def generate_segmentation_figure(view: str = "A4C"):
    """Generate the main segmentation comparison figure."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = A4C_DIR if view == "A4C" else PSAX_DIR
    ckpt = A4C_CHECKPOINT if view == "A4C" else PSAX_CHECKPOINT

    print(f"Loading {view} segmentation model...")
    model = load_seg_model(ckpt, device)

    file_list_csv = data_dir / "FileList.csv"
    tracings_csv = data_dir / "VolumeTracings.csv"

    print("Picking example patients...")
    reduced_patient, normal_patient = pick_patients(file_list_csv, tracings_csv)
    print(f"  Reduced: {reduced_patient['filename']} (EF={reduced_patient['ef']:.1f}%)")
    print(f"  Normal:  {normal_patient['filename']} (EF={normal_patient['ef']:.1f}%)")

    rows = []
    for patient in [normal_patient, reduced_patient]:
        fname = patient["filename"]
        video_path = data_dir / "Videos" / fname

        ed_idx, es_idx = get_ed_es_from_tracings(tracings_csv, fname)
        ed_frame = read_frame(video_path, ed_idx)
        es_frame = read_frame(video_path, es_idx)

        ed_mask = segment_frame(model, ed_frame, device)
        es_mask = segment_frame(model, es_frame, device)

        ed_rgb = cv2.cvtColor(cv2.resize(ed_frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        es_rgb = cv2.cvtColor(cv2.resize(es_frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)

        ed_overlay = create_overlay(ed_frame, ed_mask)
        es_overlay = create_overlay(es_frame, es_mask)

        ed_area = ed_mask.sum() / (IMG_SIZE * IMG_SIZE) * 100
        es_area = es_mask.sum() / (IMG_SIZE * IMG_SIZE) * 100

        rows.append({
            "patient": patient,
            "ed_rgb": ed_rgb, "es_rgb": es_rgb,
            "ed_overlay": ed_overlay, "es_overlay": es_overlay,
            "ed_area": ed_area, "es_area": es_area,
        })

    # Create figure
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    fig.suptitle(
        f"project_echo — LV Segmentation ({view} View)\n"
        "DeepLabV3-MobileNetV3 identifies the left ventricle (green overlay) at end-diastole and end-systole",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for row_idx, row_data in enumerate(rows):
        p = row_data["patient"]
        ef = p["ef"]
        age = p["age"]
        sex = p["sex"]
        status = "Normal" if ef >= 55 else ("Borderline" if ef >= 40 else "Reduced")
        status_color = "#2ecc71" if ef >= 55 else ("#f39c12" if ef >= 40 else "#e74c3c")

        label = f"{status} Function (EF={ef:.0f}%) — {sex}, Age {age:.0f}y"

        axes[row_idx, 0].imshow(row_data["ed_rgb"])
        axes[row_idx, 0].set_title("End-Diastole\n(max filling)", fontsize=11)
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight="bold", color=status_color)

        axes[row_idx, 1].imshow(row_data["es_rgb"])
        axes[row_idx, 1].set_title("End-Systole\n(max contraction)", fontsize=11)

        axes[row_idx, 2].imshow(row_data["ed_overlay"])
        axes[row_idx, 2].set_title(f"ED + LV Segmentation\n(area={row_data['ed_area']:.1f}%)", fontsize=11)

        axes[row_idx, 3].imshow(row_data["es_overlay"])
        axes[row_idx, 3].set_title(f"ES + LV Segmentation\n(area={row_data['es_area']:.1f}%)", fontsize=11)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_path = OUTPUT_DIR / f"segmentation_{view.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def generate_cardiac_cycle_figure(view: str = "A4C"):
    """Generate a figure showing LV area across the cardiac cycle."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = A4C_DIR if view == "A4C" else PSAX_DIR
    ckpt = A4C_CHECKPOINT if view == "A4C" else PSAX_CHECKPOINT

    model = load_seg_model(ckpt, device)

    file_list_csv = data_dir / "FileList.csv"
    tracings_csv = data_dir / "VolumeTracings.csv"
    _, normal_patient = pick_patients(file_list_csv, tracings_csv)

    fname = normal_patient["filename"]
    video_path = data_dir / "Videos" / fname

    # Read all frames and compute LV area per frame
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    areas = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        mask = segment_frame(model, frame, device)
        areas.append(mask.sum() / (IMG_SIZE * IMG_SIZE) * 100)
    cap.release()

    areas = np.array(areas)
    ed_idx = np.argmax(areas)
    es_idx = np.argmin(areas)

    # Create figure
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, (ax_plot, ax_ed, ax_es) = plt.subplots(1, 3, figsize=(14, 4.5),
                                                  gridspec_kw={"width_ratios": [2, 1, 1]})
    fig.suptitle(
        f"project_echo — Cardiac Cycle Analysis ({view} View)",
        fontsize=13, fontweight="bold",
    )

    # Plot LV area over time
    ax_plot.plot(areas, color="#3498db", linewidth=2, label="LV Area")
    ax_plot.axvline(ed_idx, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"ED (frame {ed_idx})")
    ax_plot.axvline(es_idx, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"ES (frame {es_idx})")
    ax_plot.fill_between(range(len(areas)), areas, alpha=0.15, color="#3498db")
    ax_plot.set_xlabel("Frame Number", fontsize=11)
    ax_plot.set_ylabel("LV Area (% of image)", fontsize=11)
    ax_plot.set_title("LV Area Over Cardiac Cycle", fontsize=11)
    ax_plot.legend(fontsize=9)
    ax_plot.grid(True, alpha=0.3)

    # ED overlay
    ed_frame = frames[ed_idx]
    ed_mask = segment_frame(model, ed_frame, device)
    ed_overlay = create_overlay(ed_frame, ed_mask)
    ax_ed.imshow(ed_overlay)
    ax_ed.set_title(f"End-Diastole (frame {ed_idx})\nMax LV area", fontsize=10)
    ax_ed.set_xticks([]); ax_ed.set_yticks([])

    # ES overlay
    es_frame = frames[es_idx]
    es_mask = segment_frame(model, es_frame, device)
    es_overlay = create_overlay(es_frame, es_mask)
    ax_es.imshow(es_overlay)
    ax_es.set_title(f"End-Systole (frame {es_idx})\nMin LV area", fontsize=10)
    ax_es.set_xticks([]); ax_es.set_yticks([])

    geo_ef = (areas[ed_idx] - areas[es_idx]) / areas[ed_idx] * 100
    fig.text(0.5, 0.01,
             f"Geometric EF = (ED area − ES area) / ED area = ({areas[ed_idx]:.2f} − {areas[es_idx]:.2f}) / {areas[ed_idx]:.2f} = {geo_ef:.1f}%   |   Ground Truth EF = {normal_patient['ef']:.1f}%",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    out_path = OUTPUT_DIR / f"cardiac_cycle_{view.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Generating project_echo README figures")
    print("=" * 60)

    # A4C segmentation comparison (normal vs reduced)
    generate_segmentation_figure("A4C")
    # PSAX segmentation comparison
    generate_segmentation_figure("PSAX")
    # Cardiac cycle plot
    generate_cardiac_cycle_figure("A4C")

    print("\nAll figures saved to figures/")
    print("Add to README.md:")
    print('  ![LV Segmentation](figures/segmentation_a4c.png)')
    print('  ![Cardiac Cycle](figures/cardiac_cycle_a4c.png)')
