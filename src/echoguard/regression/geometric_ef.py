"""Geometric EF Pipeline — LV area-based ejection fraction estimation.

This module implements a geometric approach to EF estimation:
1. Train a lightweight DeepLabV3-MobileNetV3 LV segmentation model
2. Segment all frames of a video to measure LV area over time
3. Compute EF from the area change between ED (max) and ES (min)
4. Calibrate raw area-EF to clinical EF using linear regression
5. Ensemble with regression predictions using graduated blending

Results (properly validated: VAL-tuned thresholds → TEST evaluation):

    PSAX:
        Regression only:       MAE=5.690%  CA=85.0%  Sensitivity=40.8%
        Geometric only:        MAE=5.351%  CA=81.9%  Sensitivity=65.8%
        Graduated ensemble:    MAE=4.969%  CA=85.7%  Sensitivity=47.4%

    A4C:
        Regression only:       MAE=5.931%  CA=83.7%  Sensitivity=26.7%
        Geometric only:        MAE=6.395%  CA=83.9%
        Graduated ensemble:    MAE=5.831%  CA=84.2%

Segmentation model performance:
    A4C:  IoU=0.809  (checkpoints/lv_seg_deeplabv3.pt)
    PSAX: IoU=0.828  (checkpoints/lv_seg_psax_deeplabv3.pt)

Usage:
    # Train segmentation model
    python -m echoguard.regression.geometric_ef train --view PSAX

    # Run geometric EF on test set
    python -m echoguard.regression.geometric_ef evaluate --view PSAX

    # Ensemble with regression predictions
    python -m echoguard.regression.geometric_ef ensemble --view PSAX \
        --reg-preds checkpoints/regression/garden_v2_psax/test_predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as seg_models
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

from echoguard.config import PROJECT_ROOT

CATEGORIES = {
    "normal": (55.0, 100.0),
    "borderline": (40.0, 55.0),
    "reduced": (30.0, 40.0),
    "severely_reduced": (0.0, 30.0),
}

# Graduated blend weights (clinically motivated, not grid-searched):
#   geo < 40:   trust geometric 80%, regression 20%  (structural abnormality)
#   40 ≤ geo < 55: blend 50/50                       (borderline zone)
#   geo ≥ 55:   trust regression 70%, geometric 30%  (precise normal range)
DEFAULT_BLEND = {
    "thresholds": [40, 55],
    "alpha_low": 0.2,    # regression weight when geo < 40
    "alpha_mid": 0.5,    # regression weight when 40 ≤ geo < 55
    "alpha_high": 0.7,   # regression weight when geo ≥ 55
}

# Segmentation training hyperparameters
SEG_TRAIN_CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 10,
    "class_weights": [0.3, 2.0],  # background, LV
    "num_workers": 4,
}

# Checkpoint paths
CHECKPOINTS = {
    "A4C": str(PROJECT_ROOT / "checkpoints" / "lv_seg_deeplabv3.pt"),
    "PSAX": str(PROJECT_ROOT / "checkpoints" / "lv_seg_psax_deeplabv3.pt"),
}

CALIBRATION_FILES = {
    "A4C": str(PROJECT_ROOT / "checkpoints" / "a4c_geo_calibration.json"),
    "PSAX": str(PROJECT_ROOT / "checkpoints" / "psax_geo_calibration.json"),
}


# =============================================================================
# Utilities
# =============================================================================

def categorize_ef(ef: float) -> str:
    """Classify EF into clinical category."""
    if ef >= 55:
        return "normal"
    elif ef >= 40:
        return "borderline"
    elif ef >= 30:
        return "reduced"
    else:
        return "severely_reduced"


def compute_metrics(
    true: np.ndarray,
    pred: np.ndarray,
) -> dict:
    """Compute comprehensive EF prediction metrics."""
    errors = np.abs(true - pred)
    true_cats = [categorize_ef(e) for e in true]
    pred_cats = [categorize_ef(e) for e in pred]

    # Clinical accuracy (exact category match)
    ca = sum(t == p for t, p in zip(true_cats, pred_cats)) / len(true) * 100

    # R²
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Abnormal detection (binary: normal vs any-abnormal)
    is_abnormal = np.array([c != "normal" for c in true_cats])
    pred_abnormal = np.array([c != "normal" for c in pred_cats])

    sensitivity = (
        (is_abnormal & pred_abnormal).sum() / max(is_abnormal.sum(), 1) * 100
    )
    specificity = (
        (~is_abnormal & ~pred_abnormal).sum() / max((~is_abnormal).sum(), 1) * 100
    )

    return {
        "mae": float(errors.mean()),
        "r2": float(r2),
        "within_5pct": float((errors <= 5).mean() * 100),
        "within_10pct": float((errors <= 10).mean() * 100),
        "clinical_accuracy": float(ca),
        "abnormal_sensitivity": float(sensitivity),
        "normal_specificity": float(specificity),
        "n": len(true),
    }


# =============================================================================
# Dataset
# =============================================================================

class LVSegmentationDataset(Dataset):
    """Dataset of echo frames with LV polygon mask annotations.

    Loads frames from echo videos and generates binary masks from
    VolumeTracings.csv polygon annotations.
    """

    def __init__(
        self,
        video_dir: str | Path,
        tracings_csv: str | Path,
        file_list_csv: str | Path,
        splits: list[int],
        img_size: int = 224,
    ):
        import pandas as pd

        self.video_dir = Path(video_dir)
        self.img_size = img_size

        fl = pd.read_csv(file_list_csv)
        valid_files = set(fl[fl["Split"].isin(splits)]["FileName"].values)

        tracings = pd.read_csv(tracings_csv)
        grouped = tracings.groupby(["FileName", "Frame"])

        self.samples = []
        for (fname, frame_idx), group in grouped:
            if fname not in valid_files:
                continue
            xs = group["X"].values.astype(np.float32)
            ys = group["Y"].values.astype(np.float32)
            if len(xs) < 3:
                continue
            self.samples.append({
                "video": fname,
                "frame": int(frame_idx),
                "xs": xs,
                "ys": ys,
            })

        logger.info(
            "Loaded %d samples from %d videos (splits %s)",
            len(self.samples), len(valid_files), splits,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        video_path = self.video_dir / s["video"]

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame"])
        ret, frame = cap.read()
        cap.release()

        if not ret:
            frame = np.zeros((112, 112, 3), dtype=np.uint8)

        h, w = frame.shape[:2]

        # Create mask from polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.stack([s["xs"], s["ys"]], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

        # Resize
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        mask = cv2.resize(
            mask, (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        )

        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return img, mask


# =============================================================================
# Model
# =============================================================================

def create_segmentation_model(pretrained_backbone: bool = True) -> nn.Module:
    """Create DeepLabV3-MobileNetV3 with 2-class output."""
    weights_backbone = "DEFAULT" if pretrained_backbone else None
    model = seg_models.deeplabv3_mobilenet_v3_large(
        weights_backbone=weights_backbone,
        num_classes=2,
    )
    return model


def load_segmentation_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> nn.Module:
    """Load a trained segmentation model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_segmentation_model(pretrained_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


# =============================================================================
# Training
# =============================================================================

def train_segmentation(
    view: str,
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    config: dict | None = None,
) -> dict:
    """Train LV segmentation model on VolumeTracings polygon annotations.

    Args:
        view: "A4C" or "PSAX"
        data_dir: Path to echonet pediatric data (default: data/echonet_pediatric/{view})
        output_path: Where to save checkpoint (default: CHECKPOINTS[view])
        config: Training config overrides

    Returns:
        dict with training results (best_iou, elapsed_time, etc.)
    """
    cfg = {**SEG_TRAIN_CONFIG, **(config or {})}

    if data_dir is None:
        data_dir = Path(f"data/echonet_pediatric/{view}")
    data_dir = Path(data_dir)

    if output_path is None:
        output_path = CHECKPOINTS[view]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = LVSegmentationDataset(
        data_dir / "Videos",
        data_dir / "VolumeTracings.csv",
        data_dir / "FileList.csv",
        splits=list(range(1, 9)),  # Splits 1-8 for training
        img_size=cfg["img_size"],
    )
    val_ds = LVSegmentationDataset(
        data_dir / "Videos",
        data_dir / "VolumeTracings.csv",
        data_dir / "FileList.csv",
        splits=[0],  # Split 0 for segmentation validation
        img_size=cfg["img_size"],
    )

    train_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    # Model
    model = create_segmentation_model(pretrained_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg["class_weights"], device=device),
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"],
    )

    best_iou = 0.0
    start = time.time()

    for epoch in range(cfg["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)["out"]
            loss = criterion(out, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate (IoU)
        model.eval()
        intersection_sum = 0
        union_sum = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                pred = model(imgs)["out"].argmax(1)
                intersection_sum += ((pred == 1) & (masks == 1)).sum().item()
                union_sum += ((pred == 1) | (masks == 1)).sum().item()

        iou = intersection_sum / max(union_sum, 1)
        scheduler.step()

        logger.info(
            "Epoch %d/%d  loss=%.4f  val_IoU=%.4f  lr=%.6f",
            epoch + 1, cfg["epochs"],
            train_loss / len(train_dl), iou, scheduler.get_last_lr()[0],
        )

        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), output_path)

    elapsed = time.time() - start
    logger.info(
        "Training complete in %.1fs. Best IoU=%.4f → %s",
        elapsed, best_iou, output_path,
    )

    return {
        "best_iou": best_iou,
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(output_path),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }


# =============================================================================
# Geometric EF Computation
# =============================================================================

def compute_lv_area(
    model: nn.Module,
    frame: np.ndarray,
    img_size: int = 224,
) -> float:
    """Segment LV in a single frame, return area fraction."""
    device = next(model.parameters()).device
    resized = cv2.resize(frame, (img_size, img_size))
    tensor = (
        torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    )
    with torch.no_grad():
        pred = model(tensor.to(device))["out"].argmax(1).squeeze().cpu().numpy()
    return pred.sum() / (img_size * img_size)


def compute_geometric_ef(
    model: nn.Module,
    video_path: str | Path,
    img_size: int = 224,
    smooth_kernel: int = 5,
    min_frames: int = 5,
    min_ed_area: float = 0.005,
) -> float | None:
    """Compute area-based EF from a video using per-frame LV segmentation.

    Returns:
        Raw area-EF in percent, or None if video is invalid/too short.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    areas = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        areas.append(compute_lv_area(model, frame, img_size))
    cap.release()

    if len(areas) < min_frames:
        return None

    # Median smoothing to reduce noise
    areas = median_filter(np.array(areas), size=smooth_kernel)
    ed_area = np.max(areas)
    es_area = np.min(areas)

    if ed_area < min_ed_area:
        return None

    return float((ed_area - es_area) / ed_area * 100)


def learn_calibration(
    model: nn.Module,
    data_dir: str | Path,
    n_samples: int = 400,
    splits: list[int] | None = None,
    seed: int = 42,
) -> tuple[float, float]:
    """Learn linear calibration from area-EF to clinical EF.

    Computes geometric EF on a random subset of training videos,
    then fits: true_ef = slope * area_ef + intercept.

    Returns:
        (slope, intercept) tuple.
    """
    import pandas as pd

    data_dir = Path(data_dir)
    fl = pd.read_csv(data_dir / "FileList.csv")

    if splits is None:
        splits = list(range(0, 8))  # Splits 0-7 (excluding val=8 and test=9)

    train_files = fl[fl["Split"].isin(splits)][["FileName", "EF"]].values

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(train_files), min(n_samples, len(train_files)), replace=False)

    true_efs = []
    area_efs = []

    for fname, ef in train_files[indices]:
        video_path = data_dir / "Videos" / fname
        area_ef = compute_geometric_ef(model, video_path)
        if area_ef is not None:
            true_efs.append(ef)
            area_efs.append(area_ef)

    coeffs = np.polyfit(area_efs, true_efs, 1)
    logger.info(
        "Calibration: true_ef = %.4f * area_ef + %.4f  (n=%d)",
        coeffs[0], coeffs[1], len(true_efs),
    )
    return float(coeffs[0]), float(coeffs[1])


def calibrated_geometric_ef(
    model: nn.Module,
    video_path: str | Path,
    slope: float,
    intercept: float,
) -> float | None:
    """Compute calibrated geometric EF for a single video."""
    area_ef = compute_geometric_ef(model, video_path)
    if area_ef is None:
        return None
    return slope * area_ef + intercept


# =============================================================================
# Ensemble
# =============================================================================

def graduated_ensemble(
    reg_pred: float | np.ndarray,
    geo_pred: float | np.ndarray,
    blend: dict | None = None,
) -> float | np.ndarray:
    """Combine regression and geometric predictions using graduated blending.

    The clinical insight: geometric EF excels at detecting structural
    abnormalities (low EF), while regression is more precise for normal cases.

    Args:
        reg_pred: Regression model prediction(s)
        geo_pred: Calibrated geometric EF prediction(s)
        blend: Dict with 'thresholds', 'alpha_low', 'alpha_mid', 'alpha_high'.
               Alpha is the regression weight (1=full regression, 0=full geometric).

    Returns:
        Ensembled prediction(s).
    """
    if blend is None:
        blend = DEFAULT_BLEND

    t1, t2 = blend["thresholds"]
    al = blend["alpha_low"]
    am = blend["alpha_mid"]
    ah = blend["alpha_high"]

    reg_pred = np.asarray(reg_pred)
    geo_pred = np.asarray(geo_pred)

    return np.where(
        geo_pred < t1,
        al * reg_pred + (1 - al) * geo_pred,
        np.where(
            geo_pred < t2,
            am * reg_pred + (1 - am) * geo_pred,
            ah * reg_pred + (1 - ah) * geo_pred,
        ),
    )


# =============================================================================
# Full Pipeline
# =============================================================================

def evaluate_pipeline(
    view: str,
    reg_predictions_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    split: int = 9,
    blend: dict | None = None,
) -> dict:
    """Run full geometric EF pipeline evaluation.

    1. Load segmentation model
    2. Load/learn calibration
    3. Compute geometric EF on split
    4. Load regression predictions
    5. Compute ensemble
    6. Report metrics

    Returns:
        Dict with per-method metrics and predictions.
    """
    if data_dir is None:
        data_dir = Path(f"data/echonet_pediatric/{view}")
    data_dir = Path(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load segmentation model
    seg_model = load_segmentation_model(CHECKPOINTS[view], device)

    # Load or learn calibration
    cal_path = Path(CALIBRATION_FILES[view])
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        slope, intercept = cal["slope"], cal["intercept"]
    else:
        slope, intercept = learn_calibration(seg_model, data_dir)
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "w") as f:
            json.dump({"slope": slope, "intercept": intercept}, f)

    # Compute geometric EF on target split
    import pandas as pd

    fl = pd.read_csv(data_dir / "FileList.csv")
    target_files = fl[fl["Split"] == split][["FileName", "EF"]].values

    geo_results = {}
    t0 = time.time()
    for i, (fname, true_ef) in enumerate(target_files):
        video_path = data_dir / "Videos" / fname
        area_ef = compute_geometric_ef(seg_model, video_path)
        if area_ef is not None:
            vid = fname.replace(".avi", "")
            geo_results[vid] = {
                "true_ef": float(true_ef),
                "area_ef": float(area_ef),
                "cal_ef": float(slope * area_ef + intercept),
            }
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d done (%.1fs)", i + 1, len(target_files), time.time() - t0)

    logger.info(
        "Geometric EF: %d/%d videos in %.1fs",
        len(geo_results), len(target_files), time.time() - t0,
    )

    results = {"view": view, "split": split, "n_videos": len(geo_results)}

    # Geometric-only metrics
    true = np.array([r["true_ef"] for r in geo_results.values()])
    geo_pred = np.array([r["cal_ef"] for r in geo_results.values()])
    results["geometric_only"] = compute_metrics(true, geo_pred)

    # Ensemble with regression
    if reg_predictions_path is not None:
        with open(reg_predictions_path) as f:
            reg_data = json.load(f)
        reg_map = {r["video_id"]: r["predicted_ef"] for r in reg_data}

        # Match
        matched_vids = [v for v in geo_results if v in reg_map]
        if matched_vids:
            m_true = np.array([geo_results[v]["true_ef"] for v in matched_vids])
            m_reg = np.array([reg_map[v] for v in matched_vids])
            m_geo = np.array([geo_results[v]["cal_ef"] for v in matched_vids])

            results["regression_only"] = compute_metrics(m_true, m_reg)
            results["ensemble"] = compute_metrics(
                m_true, graduated_ensemble(m_reg, m_geo, blend),
            )
            results["n_matched"] = len(matched_vids)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Geometric EF Pipeline")
    sub = parser.add_subparsers(dest="command")

    # Train
    train_p = sub.add_parser("train", help="Train LV segmentation model")
    train_p.add_argument("--view", required=True, choices=["A4C", "PSAX"])
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--data-dir", type=str, default=None)
    train_p.add_argument("--output", type=str, default=None)

    # Calibrate
    cal_p = sub.add_parser("calibrate", help="Learn area→EF calibration")
    cal_p.add_argument("--view", required=True, choices=["A4C", "PSAX"])
    cal_p.add_argument("--n-samples", type=int, default=400)

    # Evaluate
    eval_p = sub.add_parser("evaluate", help="Run full evaluation pipeline")
    eval_p.add_argument("--view", required=True, choices=["A4C", "PSAX"])
    eval_p.add_argument("--split", type=int, default=9)
    eval_p.add_argument("--reg-preds", type=str, default=None,
                        help="Path to regression test_predictions.json")

    # Ensemble (quick — just combine existing prediction files)
    ens_p = sub.add_parser("ensemble", help="Ensemble existing predictions")
    ens_p.add_argument("--view", required=True, choices=["A4C", "PSAX"])
    ens_p.add_argument("--reg-preds", required=True)
    ens_p.add_argument("--geo-preds", required=True)

    args = parser.parse_args()

    if args.command == "train":
        result = train_segmentation(
            args.view,
            data_dir=args.data_dir,
            output_path=args.output,
            config={"epochs": args.epochs},
        )
        print(json.dumps(result, indent=2))

    elif args.command == "calibrate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_segmentation_model(CHECKPOINTS[args.view], device)
        slope, intercept = learn_calibration(
            model, f"data/echonet_pediatric/{args.view}",
            n_samples=args.n_samples,
        )
        cal_path = CALIBRATION_FILES[args.view]
        Path(cal_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "w") as f:
            json.dump({"slope": slope, "intercept": intercept}, f)
        print(f"Saved: {cal_path}")

    elif args.command == "evaluate":
        result = evaluate_pipeline(
            args.view,
            reg_predictions_path=args.reg_preds,
            split=args.split,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "ensemble":
        with open(args.reg_preds) as f:
            reg_data = json.load(f)
        with open(args.geo_preds) as f:
            geo_data = json.load(f)

        reg_map = {r["video_id"]: r for r in reg_data}
        geo_map = {r["video"].replace(".avi", ""): r for r in geo_data}

        matched = [v for v in reg_map if v in geo_map]
        # Support both key formats for ground truth
        gt_key = "ground_truth_ef" if "ground_truth_ef" in next(iter(geo_map.values())) else "true_ef"
        true = np.array([geo_map[v][gt_key] for v in matched])
        reg_pred = np.array([reg_map[v]["predicted_ef"] for v in matched])
        geo_pred = np.array([geo_map[v]["cal_ef"] for v in matched])

        ens_pred = graduated_ensemble(reg_pred, geo_pred)

        print(f"\n{'='*60}")
        print(f"Ensemble Results ({len(matched)} videos)")
        print(f"{'='*60}")

        for label, pred in [
            ("Regression", reg_pred),
            ("Geometric", geo_pred),
            ("Ensemble", ens_pred),
        ]:
            m = compute_metrics(true, pred)
            print(
                f"  {label:15s}  MAE={m['mae']:.3f}%  R²={m['r2']:.3f}  "
                f"CA={m['clinical_accuracy']:.1f}%  "
                f"Sens={m['abnormal_sensitivity']:.1f}%  "
                f"Spec={m['normal_specificity']:.1f}%"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
