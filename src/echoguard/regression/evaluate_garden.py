"""Evaluate Model Garden architectures with rich explainability output.

Usage:
    python -m echoguard.regression.evaluate_garden --view A4C --model-type multitask
    python -m echoguard.regression.evaluate_garden --view A4C --model-type temporal
    python -m echoguard.regression.evaluate_garden --view A4C --model-type ensemble
    python -m echoguard.regression.evaluate_garden --view A4C --all  # Evaluate all models

Produces:
  - Standard regression metrics (MAE, R², ±5%, ±10%)
  - Classification metrics (accuracy, precision, recall, F1 per category)
  - Category probability analysis
  - Percentile distribution
  - Model consistency analysis (ensemble only)
  - Flagged cases report
  - Per-sample explainability output
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from echoguard.config import PROJECT_ROOT, ef_category, age_group, PEDIATRIC_EF_NORMS
from echoguard.confidence import compute_confidence
from echoguard.zscore import ZScoreFlag, compute_ef_zscore
from echoguard.regression.model import EFRegressor, EFRegressorV2, EFRegressorWithMeta
from echoguard.regression.model_garden import (
    EFMultiTaskModel,
    EFTemporalTransformer,
    EFEnsemble,
    EFExplainer,
    PredictionExplanation,
    EF_CATEGORIES,
    N_CLASSES,
    ef_to_class_index,
    count_parameters,
)
from echoguard.regression.train import EFEmbeddingDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_garden_model(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> tuple[torch.nn.Module, dict]:
    """Load any Model Garden model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    embed_dim = ckpt.get("embed_dim", 1152)
    num_frames = ckpt.get("num_frames", 4)
    model_type = ckpt.get("model_type", "mlp")

    if model_type == "multitask":
        model = EFMultiTaskModel(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.get("hidden_dim", 512),
            dropout=0.0,
        )
    elif model_type == "temporal":
        model = EFTemporalTransformer(
            embed_dim=embed_dim,
            num_frames=num_frames,
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            hidden_dim=config.get("hidden_dim", 256),
            dropout=0.0,
            proj_dim=config.get("proj_dim", 192),  # new arch default
        )
    elif model_type == "tcn":
        from echoguard.regression.model_garden import EFTemporalCNN
        model = EFTemporalCNN(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden=config.get("hidden_dim", 256),
            num_levels=config.get("num_levels", 4),
            dropout=0.0,
        )
    elif model_type == "v2":
        model = EFRegressorV2(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.get("hidden_dim", 512),
            dropout=0.0,
        )
    elif model_type == "ensemble":
        model = EFEnsemble(
            n_models=len(ckpt.get("component_models", ["mlp", "multitask", "temporal"])),
            n_class_probs=N_CLASSES,
            n_meta=2,
        )
    elif config.get("use_metadata", False):
        model = EFRegressorWithMeta(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.get("hidden_dim", 512),
            dropout=0.0,
        )
    else:
        model = EFRegressor(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.get("hidden_dim", 512),
            dropout=0.0,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    logger.info(
        "Loaded %s model from %s (epoch %d, val_mae=%.2f%%)",
        model_type, checkpoint_path,
        ckpt.get("epoch", -1), ckpt.get("val_mae", -1),
    )

    return model, ckpt


# ---------------------------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    view: str,
    model_type: str = "multitask",
    embeddings_dir: str = str(PROJECT_ROOT / "data" / "embeddings"),
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints" / "regression"),
    checkpoint_name: str = "best_model.pt",
    output_dir: str | None = None,
    device: str = "cuda",
) -> dict:
    """Run full evaluation on a Model Garden model.

    Returns dict with all metrics + per-sample predictions.
    """
    # Find checkpoint
    if model_type == "mlp":
        ckpt_subdir = f"ef_regression_{view.lower()}"
    else:
        ckpt_subdir = f"garden_{model_type}_{view.lower()}"

    ckpt_path = Path(checkpoint_dir) / ckpt_subdir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, ckpt = load_garden_model(ckpt_path, device)
    ckpt_config = ckpt.get("config", {})

    # Load test data
    manifest_path = (
        Path(embeddings_dir) / f"pediatric_{view.lower()}" / "manifest.json"
    )
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build demographic lookup for Z-score computation  vid → (sex, weight, height)
    demo_map: dict[str, tuple] = {
        vid: (
            meta.get("sex"),
            float(meta["weight"]) if meta.get("weight") else None,
            float(meta["height"]) if meta.get("height") else None,
        )
        for vid, meta in manifest.items()
    }

    test_dataset = EFEmbeddingDataset(manifest, split="TEST")
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    logger.info(
        "Evaluating %s on %d test samples (%s view)",
        model_type, len(test_dataset), view,
    )

    # Collect predictions
    all_preds = []
    all_targets = []
    all_ages = []
    all_ids = []
    all_class_probs = []  # For multitask models

    # For ensemble, load component models
    component_models = {}
    if model_type == "ensemble":
        ckpt_config = ckpt.get("config", {})
        component_names = ckpt.get("component_models", ["mlp", "multitask", "temporal"])
        for comp_name in component_names:
            if comp_name == "mlp":
                comp_subdir = f"ef_regression_{view.lower()}"
            else:
                comp_subdir = f"garden_{comp_name}_{view.lower()}"
            comp_path = Path(checkpoint_dir) / comp_subdir / "best_model.pt"
            if comp_path.exists():
                comp_model, _ = load_garden_model(comp_path, device)
                component_models[comp_name] = comp_model
                logger.info("  Loaded ensemble component: %s", comp_name)
            else:
                logger.warning("  Missing ensemble component: %s (%s)", comp_name, comp_path)
        if len(component_models) < 2:
            raise RuntimeError(
                f"Ensemble requires at least 2 component models, found {len(component_models)}"
            )

    for embeddings, targets, metadata, video_ids in test_loader:
        embeddings = embeddings.to(device)
        metadata = metadata.to(device)

        if model_type == "ensemble":
            # Get predictions from each component model
            preds_list = []
            class_probs_batch = torch.zeros(
                embeddings.size(0), N_CLASSES, device=device
            )
            for comp_name in component_models:
                m = component_models[comp_name]
                if comp_name == "multitask":
                    ef_p, logits = m(embeddings, return_probs=True)
                    preds_list.append(ef_p)
                    class_probs_batch = F.softmax(logits, dim=-1)
                else:
                    preds_list.append(m(embeddings))
            model_preds = torch.stack(preds_list, dim=-1)
            meta_input = metadata[:, :2]  # age, sex_male
            preds = model(model_preds, class_probs_batch, meta_input)
            all_class_probs.append(class_probs_batch.cpu().numpy())
        elif model_type == "multitask":
            ef_pred, class_logits = model(embeddings, return_probs=True)
            probs = F.softmax(class_logits, dim=-1).cpu().numpy()
            all_class_probs.append(probs)
            preds = ef_pred
        elif model_type == "temporal":
            preds = model(embeddings)
        else:
            preds = model(embeddings)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_ages.extend(metadata[:, 0].cpu().numpy())
        all_ids.extend(video_ids)

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    ages = np.array(all_ages)

    if all_class_probs:
        class_probs = np.concatenate(all_class_probs, axis=0)
    else:
        class_probs = None

    # ----- Standard regression metrics -----
    from echoguard.evaluation import evaluate, print_evaluation
    result = evaluate(y_true, y_pred, ages, tier=f"garden_{model_type}_{view.lower()}")
    print_evaluation(result)

    # ----- Classification metrics -----
    cls_metrics = _compute_classification_metrics(y_true, y_pred, ages, class_probs)
    _print_classification_report(cls_metrics)

    # ----- Percentile analysis -----
    _print_percentile_analysis(y_true, y_pred, ages)

    # ----- Prediction diversity -----
    _print_diversity_check(y_pred)

    # ----- Worst cases -----
    _print_worst_cases(y_true, y_pred, ages, all_ids, class_probs)

    # ----- Z-score summary -----
    _print_zscore_summary(y_true, y_pred, ages, all_ids, demo_map)

    # ----- Save results -----
    save_dir = Path(output_dir or checkpoint_dir) / ckpt_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample predictions with rich metadata
    predictions = []
    for i, vid in enumerate(all_ids):
        age_i = float(ages[i])
        sex_i, wt_i, ht_i = demo_map.get(vid, (None, None, None))

        zr = compute_ef_zscore(
            ef=float(y_pred[i]),
            age=age_i,
            sex=sex_i,
            weight=wt_i if (wt_i and wt_i > 0) else None,
            height=ht_i if (ht_i and ht_i > 0) else None,
        )
        zr_true = compute_ef_zscore(
            ef=float(y_true[i]),
            age=age_i,
            sex=sex_i,
            weight=wt_i if (wt_i and wt_i > 0) else None,
            height=ht_i if (ht_i and ht_i > 0) else None,
        )
        conf = compute_confidence([float(y_pred[i])], zr.z_score)

        entry = {
            "video_id": vid,
            "ground_truth_ef": round(float(y_true[i]), 1),
            "predicted_ef": round(float(y_pred[i]), 1),
            "error": round(float(abs(y_true[i] - y_pred[i])), 1),
            "patient_age_years": round(age_i, 1),
            "true_category": ef_category(float(y_true[i]), age_i),
            "pred_category": ef_category(float(y_pred[i]), age_i),
            "category_correct": (
                ef_category(float(y_true[i]), age_i)
                == ef_category(float(y_pred[i]), age_i)
            ),
            # Z-score fields
            "z_score": zr.z_score,
            "z_score_true": zr_true.z_score,
            "z_flag": zr.flag.value,
            "z_flag_true": zr_true.flag.value,
            "z_normal_range_lower": zr.normal_range[0],
            "z_normal_range_upper": zr.normal_range[1],
            "mu_adjusted": zr.mu_adjusted,
            "bsa": zr.bsa,
            "confidence": conf.overall,
            "confidence_level": conf.level.value,
        }
        if class_probs is not None:
            entry["category_probabilities"] = {
                EF_CATEGORIES[j]: round(float(class_probs[i, j]), 4)
                for j in range(N_CLASSES)
            }
            entry["classifier_category"] = EF_CATEGORIES[int(class_probs[i].argmax())]
            entry["classifier_confidence"] = round(float(class_probs[i].max()), 3)

        # EF percentile within test set
        entry["ef_percentile"] = round(
            float(np.searchsorted(np.sort(y_pred), y_pred[i]) / len(y_pred) * 100), 1
        )

        predictions.append(entry)

    with open(save_dir / "test_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    # Summary metrics
    summary = {
        "model_type": model_type,
        "view": view,
        "n_test": len(y_true),
        "regression": result.to_dict(),
        "classification": cls_metrics,
        "prediction_diversity": {
            "pred_std": round(float(np.std(y_pred)), 2),
            "n_unique": len(set(np.round(y_pred, 1))),
            "pred_range": [round(float(y_pred.min()), 1), round(float(y_pred.max()), 1)],
        },
    }
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved to %s", save_dir)
    return summary


# ---------------------------------------------------------------------------
# Z-Score Summary
# ---------------------------------------------------------------------------

def _print_zscore_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ages: np.ndarray,
    video_ids: list[str],
    demo_map: dict,
) -> None:
    """Print Z-score flag distribution and flag-agreement statistics."""
    from collections import Counter

    true_flags, pred_flags = [], []
    for i, vid in enumerate(video_ids):
        age_i = float(ages[i])
        sex_i, wt_i, ht_i = demo_map.get(vid, (None, None, None))
        kw = dict(
            age=age_i,
            sex=sex_i,
            weight=wt_i if (wt_i and wt_i > 0) else None,
            height=ht_i if (ht_i and ht_i > 0) else None,
        )
        true_flags.append(compute_ef_zscore(float(y_true[i]), **kw).flag)
        pred_flags.append(compute_ef_zscore(float(y_pred[i]), **kw).flag)

    flag_order = [f.value for f in ZScoreFlag]
    true_ctr = Counter(f.value for f in true_flags)
    pred_ctr = Counter(f.value for f in pred_flags)
    agree = sum(t == p for t, p in zip(true_flags, pred_flags))

    print(f"\n{'═'*70}")
    print("  Z-Score Flag Distribution")
    print(f"{'═'*70}")
    print(f"  {'Flag':<18s}  {'Ground Truth':>12s}  {'Predicted':>10s}")
    print(f"  {'─'*44}")
    for flag in flag_order:
        n_true = true_ctr.get(flag, 0)
        n_pred = pred_ctr.get(flag, 0)
        bar = "▓" * min(int(n_true / len(y_true) * 30), 30)
        print(f"  {flag:<18s}  {n_true:>7d} ({n_true/len(y_true):.0%})  "
              f"{n_pred:>5d} ({n_pred/len(y_pred):.0%})  {bar}")
    print(f"  {'─'*44}")
    print(f"  Z-Flag agreement: {agree}/{len(y_true)} ({agree/len(y_true):.1%})")

    # Dangerous mismatches: true normal flagged as reduced or vice-versa
    danger = sum(
        1 for t, p in zip(true_flags, pred_flags)
        if {t, p} == {ZScoreFlag.NORMAL, ZScoreFlag.CRITICAL}
        or {t, p} == {ZScoreFlag.NORMAL, ZScoreFlag.REDUCED}
        or {t, p} == {ZScoreFlag.NORMAL, ZScoreFlag.HYPERDYNAMIC}
    )
    if danger:
        print(f"  ⚠ Dangerous flag mismatches (normal↔abnormal): {danger}")
    print(f"{'═'*70}")


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ages: np.ndarray,
    class_probs: np.ndarray | None = None,
) -> dict:
    """Compute per-category classification metrics."""
    true_cats = [ef_category(float(ef), float(a)) for ef, a in zip(y_true, ages)]
    pred_cats = [ef_category(float(ef), float(a)) for ef, a in zip(y_pred, ages)]

    # If we have classifier output, also evaluate that
    classifier_cats = None
    if class_probs is not None:
        classifier_cats = [EF_CATEGORIES[int(p.argmax())] for p in class_probs]

    categories = EF_CATEGORIES
    metrics = {
        "overall_accuracy": float(np.mean([t == p for t, p in zip(true_cats, pred_cats)])),
        "per_category": {},
    }

    for cat in categories:
        true_mask = [t == cat for t in true_cats]
        pred_mask = [p == cat for p in pred_cats]
        n_true = sum(true_mask)
        n_pred = sum(pred_mask)

        # True positives
        tp = sum(t and p for t, p in zip(true_mask, pred_mask))
        precision = tp / max(n_pred, 1)
        recall = tp / max(n_true, 1)
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-8)
            if tp > 0 else 0.0
        )

        cat_metrics = {
            "n_true": n_true,
            "n_predicted": n_pred,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

        # Per-category MAE
        true_indices = [i for i, t in enumerate(true_mask) if t]
        if true_indices:
            cat_mae = float(np.mean(np.abs(y_true[true_indices] - y_pred[true_indices])))
            cat_metrics["mae"] = round(cat_mae, 2)

        # Classifier accuracy (if available)
        if classifier_cats and n_true > 0:
            cls_correct = sum(
                classifier_cats[i] == cat
                for i in range(len(true_cats)) if true_cats[i] == cat
            )
            cat_metrics["classifier_accuracy"] = round(cls_correct / n_true, 3)

        metrics["per_category"][cat] = cat_metrics

    # Classifier overall accuracy (if available)
    if classifier_cats:
        metrics["classifier_overall_accuracy"] = float(
            np.mean([t == c for t, c in zip(true_cats, classifier_cats)])
        )

    return metrics


def _print_classification_report(metrics: dict) -> None:
    """Print classification metrics in a formatted table."""
    print(f"\n{'═'*70}")
    print("  Classification Report")
    print(f"{'═'*70}")
    print(
        f"  Overall accuracy (regression-derived): "
        f"{metrics['overall_accuracy']:.1%}"
    )
    if "classifier_overall_accuracy" in metrics:
        print(
            f"  Overall accuracy (classifier head):    "
            f"{metrics['classifier_overall_accuracy']:.1%}"
        )

    print(f"\n  {'Category':<15s} {'N':>4s} {'Prec':>6s} {'Recall':>7s} "
          f"{'F1':>6s} {'MAE':>6s}", end="")
    if any("classifier_accuracy" in v for v in metrics["per_category"].values()):
        print(f" {'Cls Acc':>8s}", end="")
    print()
    print(f"  {'─'*60}")

    for cat in EF_CATEGORIES:
        m = metrics["per_category"].get(cat, {})
        line = (
            f"  {cat:<15s} {m.get('n_true', 0):>4d} "
            f"{m.get('precision', 0):>6.1%} {m.get('recall', 0):>7.1%} "
            f"{m.get('f1', 0):>6.3f} {m.get('mae', 0):>5.1f}%"
        )
        if "classifier_accuracy" in m:
            line += f" {m['classifier_accuracy']:>7.1%}"
        print(line)

    print(f"{'═'*70}")


# ---------------------------------------------------------------------------
# Percentile Analysis
# ---------------------------------------------------------------------------

def _print_percentile_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ages: np.ndarray,
) -> None:
    """Print percentile-based analysis of predictions."""
    errors = np.abs(y_true - y_pred)

    print(f"\n{'─'*60}")
    print("  Percentile Analysis")
    print(f"{'─'*60}")

    # Error percentiles
    print("  Absolute Error Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(errors, p)
        bar = "█" * int(val * 2)
        print(f"    P{p:2d}: {val:5.1f}% {bar}")

    # EF distribution comparison
    print("\n  EF Distribution:")
    print(f"    {'':10s} {'P10':>6s} {'P25':>6s} {'P50':>6s} "
          f"{'P75':>6s} {'P90':>6s}")
    for label, arr in [("Ground Truth", y_true), ("Predicted", y_pred)]:
        pcts = [np.percentile(arr, p) for p in [10, 25, 50, 75, 90]]
        print(f"    {label:10s} " + " ".join(f"{v:5.1f}%" for v in pcts))

    # Which EF ranges have worst errors?
    print("\n  Error by True EF Range:")
    ranges = [(0, 30, "<30%"), (30, 45, "30-45%"), (45, 55, "45-55%"),
              (55, 70, "55-70%"), (70, 100, ">70%")]
    for lo, hi, label in ranges:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            range_mae = float(np.mean(errors[mask]))
            print(f"    {label:8s}: N={mask.sum():3d}, MAE={range_mae:.2f}%")

    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Diversity Check
# ---------------------------------------------------------------------------

def _print_diversity_check(y_pred: np.ndarray) -> None:
    """Check prediction diversity (critical after GRPO collapse history)."""
    pred_std = float(np.std(y_pred))
    n_unique = len(set(np.round(y_pred, 1)))

    print(f"\n{'─'*60}")
    print("  Prediction Diversity Check")
    print(f"{'─'*60}")
    print(f"  Prediction std: {pred_std:.2f}%")
    print(f"  Unique values (0.1% resolution): {n_unique}")
    print(f"  Range: [{y_pred.min():.1f}%, {y_pred.max():.1f}%]")

    if pred_std < 3.0:
        print("  ⚠️  WARNING: Low prediction diversity — possible collapse!")
    elif pred_std < 5.0:
        print("  ⚡ Moderate diversity — watch for convergence")
    else:
        print("  ✓  Prediction diversity looks healthy")

    # Histogram
    print("\n  Distribution:")
    bins = np.arange(0, 101, 10)
    counts, _ = np.histogram(y_pred, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(len(counts)):
        bar = "█" * int(counts[i] / max_count * 30)
        print(f"    {bins[i]:3.0f}-{bins[i+1]:3.0f}%: {counts[i]:4d} {bar}")

    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Worst Cases
# ---------------------------------------------------------------------------

def _print_worst_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ages: np.ndarray,
    ids: list[str],
    class_probs: np.ndarray | None = None,
    n: int = 10,
) -> None:
    """Print worst predictions with clinical context."""
    errors = np.abs(y_true - y_pred)
    worst_idx = np.argsort(errors)[-n:][::-1]

    print(f"\n{'═'*80}")
    print(f"  Top {n} Worst Predictions")
    print(f"{'═'*80}")
    print(
        f"  {'Video':<20s} {'True':>6s} {'Pred':>6s} {'Err':>5s} "
        f"{'True Cat':<13s} {'Pred Cat':<13s}", end=""
    )
    if class_probs is not None:
        print(f" {'Confidence':>10s}", end="")
    print()
    print(f"  {'─'*75}")

    for idx in worst_idx:
        true_cat = ef_category(float(y_true[idx]), float(ages[idx]))
        pred_cat = ef_category(float(y_pred[idx]), float(ages[idx]))
        mismatch = "⚠️" if true_cat != pred_cat else "  "

        line = (
            f"  {ids[idx]:<20s} {y_true[idx]:5.1f}% {y_pred[idx]:5.1f}% "
            f"{errors[idx]:4.1f}% {true_cat:<13s} {pred_cat:<13s}"
        )
        if class_probs is not None:
            conf = float(class_probs[idx].max())
            line += f" {conf:9.1%}"
        line += f" {mismatch}"
        print(line)

    # How many are dangerous misclassifications?
    dangerous = 0
    for idx in worst_idx:
        true_cat = ef_category(float(y_true[idx]), float(ages[idx]))
        pred_cat = ef_category(float(y_pred[idx]), float(ages[idx]))
        if (true_cat == "reduced" and pred_cat == "normal") or \
           (true_cat == "normal" and pred_cat == "reduced"):
            dangerous += 1

    if dangerous > 0:
        print(f"\n  🚨 {dangerous} DANGEROUS misclassifications "
              f"(normal↔reduced confusion)")

    print(f"{'═'*80}")


# ---------------------------------------------------------------------------
# Multi-Model Comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def compare_all_models(
    view: str,
    embeddings_dir: str = str(PROJECT_ROOT / "data" / "embeddings"),
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints" / "regression"),
    device: str = "cuda",
) -> dict:
    """Evaluate and compare all available Model Garden models for a view.

    Returns summary comparison dict.
    """
    results = {}
    model_types_to_try = ["mlp", "v2", "multitask", "temporal", "tcn", "ensemble"]

    for mt in model_types_to_try:
        try:
            logger.info("\n" + "=" * 60)
            logger.info("  Evaluating %s model...", mt)
            logger.info("=" * 60)
            r = run_evaluation(
                view=view,
                model_type=mt,
                embeddings_dir=embeddings_dir,
                checkpoint_dir=checkpoint_dir,
                device=device,
            )
            results[mt] = r
        except FileNotFoundError:
            logger.info("No %s checkpoint found — skipping", mt)
        except Exception as e:
            logger.error("Error evaluating %s: %s", mt, e)

    if len(results) < 1:
        logger.warning("No models evaluated!")
        return {}

    # Print comparison table
    print(f"\n{'═'*70}")
    print(f"  Model Garden Comparison — {view}")
    print(f"{'═'*70}")
    print(f"  {'Model':<12s} {'MAE':>6s} {'R²':>7s} {'±5%':>6s} "
          f"{'±10%':>6s} {'ClinAcc':>8s} {'PredStd':>8s}")
    print(f"  {'─'*65}")

    for mt, r in results.items():
        reg = r.get("regression", {})
        mae = reg.get("mae", -1)
        r2 = reg.get("r2", -1)
        w5 = reg.get("within_5", -1)
        w10 = reg.get("within_10", -1)
        ca = r.get("classification", {}).get("overall_accuracy", -1)
        ps = r.get("prediction_diversity", {}).get("pred_std", -1)

        print(
            f"  {mt:<12s} {mae:5.2f}% {r2:7.4f} "
            f"{w5*100 if w5 > 0 else -1:5.1f}% "
            f"{w10*100 if w10 > 0 else -1:5.1f}% "
            f"{ca*100 if ca > 0 else -1:7.1f}% "
            f"{ps:7.2f}"
        )

    print(f"{'═'*70}")

    # Save comparison
    save_path = Path(checkpoint_dir) / f"garden_comparison_{view.lower()}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Comparison saved to %s", save_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Model Garden architectures"
    )
    parser.add_argument("--view", default="A4C", choices=["A4C", "PSAX"])
    parser.add_argument(
        "--model-type", default="multitask",
        choices=["mlp", "v2", "multitask", "temporal", "tcn", "ensemble"],
    )
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all available models and compare")
    parser.add_argument("--embeddings-dir", default=str(PROJECT_ROOT / "data" / "embeddings"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints" / "regression"))
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.all:
        compare_all_models(
            view=args.view,
            embeddings_dir=args.embeddings_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
        )
    else:
        run_evaluation(
            view=args.view,
            model_type=args.model_type,
            embeddings_dir=args.embeddings_dir,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
