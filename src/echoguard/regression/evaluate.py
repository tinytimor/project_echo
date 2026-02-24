"""Evaluate the EF regression model on the test set.

Usage:
    python -m echoguard.regression.evaluate --view A4C
    python -m echoguard.regression.evaluate --view PSAX
    python -m echoguard.regression.evaluate --view A4C --checkpoint best_model.pt

Loads the trained MLP regression head, runs inference on test fold (9),
and reports metrics using the shared echoguard.evaluation framework.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from echoguard.config import DataConfig, PROJECT_ROOT, ef_category
from echoguard.regression.model import EFRegressor, EFRegressorV2, EFRegressorWithMeta
from echoguard.regression.train import EFEmbeddingDataset

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> tuple[torch.nn.Module, dict]:
    """Load a trained EF regression model from checkpoint.

    Automatically detects model type from checkpoint metadata:
      - "v2" → EFRegressorV2 (attention pooling + residual blocks)
      - "multitask" → EFMultiTaskModel
      - "temporal" → EFTemporalTransformer
      - default → EFRegressor (base MLP)

    Returns (model, checkpoint_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    embed_dim = ckpt.get("embed_dim", 1152)
    num_frames = ckpt.get("num_frames", 4)
    model_type = ckpt.get("model_type", config.get("model_type", "mlp"))

    if model_type == "v2":
        model = EFRegressorV2(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.get("hidden_dim", 512),
            dropout=0.0,
        )
    elif model_type == "multitask":
        from echoguard.regression.model_garden import EFMultiTaskModel
        model = EFMultiTaskModel(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.get("hidden_dim", 512),
            dropout=0.0,
        )
    elif model_type == "temporal":
        from echoguard.regression.model_garden import EFTemporalTransformer
        model = EFTemporalTransformer(
            embed_dim=embed_dim,
            num_frames=num_frames,
            n_heads=config.get("n_heads", 8),
            n_layers=config.get("n_layers", 2),
            dropout=0.0,
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
        "Loaded model from %s (epoch %d, val_mae=%.2f%%)",
        checkpoint_path,
        ckpt.get("epoch", -1),
        ckpt.get("val_mae", -1),
    )

    return model, ckpt


@torch.no_grad()
def run_evaluation(
    view: str,
    embeddings_dir: str = str(PROJECT_ROOT / "data" / "embeddings"),
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints" / "regression"),
    checkpoint_name: str = "best_model.pt",
    output_dir: str | None = None,
    device: str = "cuda",
) -> EvaluationResult:
    """Run full evaluation on the test set.

    Args:
        view: "A4C" or "PSAX"
        embeddings_dir: Directory with pre-extracted embeddings
        checkpoint_dir: Directory containing the checkpoint
        checkpoint_name: Checkpoint file name
        output_dir: Where to save results (defaults to checkpoint_dir)
        device: CUDA device

    Returns:
        EvaluationResult with all metrics
    """
    ckpt_path = Path(checkpoint_dir) / f"ef_regression_{view.lower()}" / checkpoint_name
    if not ckpt_path.exists():
        # Try direct path
        ckpt_path = Path(checkpoint_dir) / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, ckpt = load_model(ckpt_path, device)
    config = ckpt.get("config", {})

    # Load test data
    manifest_path = Path(embeddings_dir) / f"pediatric_{view.lower()}" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    test_dataset = EFEmbeddingDataset(manifest, split="TEST")
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    logger.info("Evaluating on %d test samples (%s view)", len(test_dataset), view)

    # Run inference
    all_preds = []
    all_targets = []
    all_ages = []
    all_ids = []
    all_metadata = []

    for embeddings, targets, metadata, video_ids in test_loader:
        embeddings = embeddings.to(device)
        metadata = metadata.to(device)

        if config.get("use_metadata", False) and hasattr(model, "meta_branch"):
            preds = model(embeddings, metadata)
        else:
            preds = model(embeddings)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_ages.extend(metadata[:, 0].cpu().numpy())  # First element is age
        all_ids.extend(video_ids)

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    ages = np.array(all_ages)

    # Run evaluation using shared framework
    from echoguard.evaluation import evaluate, print_evaluation, EvaluationResult
    result = evaluate(y_true, y_pred, ages, tier=f"regression_{view.lower()}")
    print_evaluation(result)

    # Additional analysis
    _print_prediction_analysis(y_true, y_pred, ages)

    # Save results
    save_dir = Path(output_dir or checkpoint_dir) / f"ef_regression_{view.lower()}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save per-sample predictions
    predictions = []
    for i, vid in enumerate(all_ids):
        predictions.append({
            "video_id": vid,
            "ground_truth_ef": float(y_true[i]),
            "predicted_ef": float(y_pred[i]),
            "patient_age_years": float(ages[i]),
            "error": float(abs(y_true[i] - y_pred[i])),
            "true_category": ef_category(float(y_true[i]), float(ages[i])),
            "pred_category": ef_category(float(y_pred[i]), float(ages[i])),
        })

    with open(save_dir / "test_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    # Save summary metrics
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info("Results saved to %s", save_dir)

    return result


def _print_prediction_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ages: np.ndarray,
) -> None:
    """Print detailed prediction analysis including diversity checks."""
    print(f"\n{'─'*60}")
    print("  Prediction Analysis")
    print(f"{'─'*60}")

    # Distribution stats
    print(f"  Ground truth: mean={np.mean(y_true):.1f}%, std={np.std(y_true):.1f}%, "
          f"range=[{np.min(y_true):.1f}, {np.max(y_true):.1f}]")
    print(f"  Predictions:  mean={np.mean(y_pred):.1f}%, std={np.std(y_pred):.1f}%, "
          f"range=[{np.min(y_pred):.1f}, {np.max(y_pred):.1f}]")

    # Diversity check (critical — VLM approach failed here)
    pred_std = np.std(y_pred)
    n_unique = len(np.unique(np.round(y_pred, 1)))
    print(f"  Prediction diversity: std={pred_std:.2f}%, unique_vals={n_unique}")
    if pred_std < 3.0:
        print("  ⚠️  WARNING: Low prediction diversity — possible collapse")
    else:
        print("  ✓  Prediction diversity looks healthy")

    # Per-category breakdown
    true_cats = [ef_category(float(ef), float(age)) for ef, age in zip(y_true, ages)]
    pred_cats = [ef_category(float(ef), float(age)) for ef, age in zip(y_pred, ages)]

    categories = sorted(set(true_cats))
    print(f"\n  Per-category MAE:")
    for cat in categories:
        indices = [i for i, c in enumerate(true_cats) if c == cat]
        if indices:
            cat_mae = np.mean(np.abs(y_true[indices] - y_pred[indices]))
            cat_correct = sum(pred_cats[i] == cat for i in indices)
            print(f"    {cat:20s}: MAE={cat_mae:.2f}% | "
                  f"Classified={cat_correct}/{len(indices)} "
                  f"({100*cat_correct/len(indices):.0f}%)")

    # Error percentiles
    errors = np.abs(y_true - y_pred)
    print(f"\n  Error percentiles:")
    for p in [25, 50, 75, 90, 95]:
        print(f"    P{p}: {np.percentile(errors, p):.2f}%")

    # Worst predictions
    worst_idx = np.argsort(errors)[-5:][::-1]
    print(f"\n  5 worst predictions:")
    for idx in worst_idx:
        print(f"    True={y_true[idx]:.1f}%, Pred={y_pred[idx]:.1f}%, "
              f"Error={errors[idx]:.1f}%")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate EF regression model")
    parser.add_argument("--view", default="A4C", choices=["A4C", "PSAX"],
                        help="Echo view to evaluate")
    parser.add_argument("--embeddings-dir", default=str(PROJECT_ROOT / "data" / "embeddings"),
                        help="Directory with pre-extracted embeddings")
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints" / "regression"),
                        help="Directory containing checkpoints")
    parser.add_argument("--checkpoint", default="best_model.pt",
                        help="Checkpoint file name")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_evaluation(
        view=args.view,
        embeddings_dir=args.embeddings_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
