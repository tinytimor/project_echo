"""Train the EF regression model (SigLIP embeddings → MLP → EF).

Usage:
    python -m echoguard.regression.train --view A4C
    python -m echoguard.regression.train --view A4C --use-adult-curriculum
    python -m echoguard.regression.train --view PSAX

This loads pre-extracted SigLIP embeddings from disk, trains an MLP
regression head with Huber loss, class-weighted sampling, and cosine
LR scheduling. Validates on fold 8, tests on fold 9.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from echoguard.config import DataConfig, PROJECT_ROOT, SPLIT_MAP, ef_category
from echoguard.regression.model import (
    EFRegressor, EFRegressorV2, EFRegressorWithMeta,
    huber_loss, composite_loss, count_parameters,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EFEmbeddingDataset(Dataset):
    """Dataset of (embedding, ef_target, age, metadata) tuples.

    Loads pre-extracted SigLIP embeddings from .pt files on disk.
    """

    def __init__(
        self,
        manifest: dict,
        split: str = "TRAIN",
        transform_noise: float = 0.0,
    ):
        self.entries = []
        for vid, meta in manifest.items():
            if meta.get("split", "") == split:
                emb_path = Path(meta["embedding_path"])
                if emb_path.exists():
                    self.entries.append({
                        "video_id": vid,
                        "embedding_path": str(emb_path),
                        "ef": float(meta["ef"]),
                        "age": float(meta.get("age", 0.0)),
                        "sex": meta.get("sex", "U"),
                        "weight": float(meta.get("weight", 0.0)),
                        "height": float(meta.get("height", 0.0)),
                    })

        self.split = split
        self.transform_noise = transform_noise

        logger.info("Loaded %d samples for split=%s", len(self.entries), split)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load embedding
        embedding = torch.load(entry["embedding_path"], weights_only=True).contiguous()
        # embedding shape: (num_frames, embed_dim)

        # Add small Gaussian noise during training (regularization)
        if self.transform_noise > 0 and self.split == "TRAIN":
            embedding = embedding + torch.randn_like(embedding) * self.transform_noise

        # Target
        ef = torch.tensor(entry["ef"], dtype=torch.float32)

        # Metadata
        age = torch.tensor(entry["age"], dtype=torch.float32)
        sex_male = 1.0 if entry["sex"].upper() == "M" else 0.0
        sex_female = 1.0 if entry["sex"].upper() == "F" else 0.0
        metadata = torch.tensor([age, sex_male, sex_female], dtype=torch.float32)

        return embedding, ef, metadata, entry["video_id"]


def compute_class_weights(dataset: EFEmbeddingDataset) -> list[float]:
    """Compute per-sample weights for WeightedRandomSampler.

    Upweights underrepresented EF categories (borderline, reduced, hyperdynamic)
    to counter the ~86% normal class imbalance.
    """
    # Assign category to each sample
    categories = []
    for entry in dataset.entries:
        cat = ef_category(entry["ef"], entry["age"])
        categories.append(cat)

    # Count category frequencies
    from collections import Counter
    cat_counts = Counter(categories)
    total = len(categories)

    # Inverse frequency weights
    cat_weights = {}
    for cat, count in cat_counts.items():
        cat_weights[cat] = total / (len(cat_counts) * count)

    # Per-sample weight
    sample_weights = [cat_weights[cat] for cat in categories]

    # Log distribution
    for cat in sorted(cat_counts.keys()):
        logger.info(
            "  %s: %d (%.1f%%) → weight=%.2f",
            cat, cat_counts[cat],
            100 * cat_counts[cat] / total,
            cat_weights[cat],
        )

    return sample_weights


def _create_demo_split(
    manifest: dict,
    n_demo: int,
    seed: int,
    output_path: Path,
) -> dict:
    """Carve out a demo holdout from the TEST split.

    Selects n_demo samples from TEST, prioritizing diversity across
    EF categories so the demo showcases normal, borderline, AND reduced cases.

    Returns a modified copy of the manifest with DEMO split entries.
    """
    import copy

    test_entries = [
        (vid, meta) for vid, meta in manifest.items()
        if meta.get("split") == "TEST"
    ]

    if len(test_entries) <= n_demo:
        logger.warning(
            "Not enough test samples (%d) for demo holdout (%d). Skipping.",
            len(test_entries), n_demo,
        )
        return manifest

    # Group by EF category for stratified selection
    by_category: dict[str, list] = {}
    for vid, meta in test_entries:
        cat = ef_category(float(meta["ef"]), float(meta.get("age", 8.0)))
        by_category.setdefault(cat, []).append((vid, meta))

    rng = random.Random(seed)
    demo_ids = set()

    # First pass: guarantee at least 1 from each category (if available)
    for cat in ["reduced", "borderline", "hyperdynamic", "normal"]:
        if cat in by_category and len(by_category[cat]) > 0:
            selected = rng.choice(by_category[cat])
            demo_ids.add(selected[0])
            by_category[cat].remove(selected)

    # Second pass: fill remaining slots from all categories proportionally
    remaining = n_demo - len(demo_ids)
    all_remaining = [
        (vid, meta) for cat_list in by_category.values()
        for vid, meta in cat_list
        if vid not in demo_ids
    ]
    rng.shuffle(all_remaining)
    for vid, meta in all_remaining[:remaining]:
        demo_ids.add(vid)

    # Create modified manifest
    new_manifest = copy.deepcopy(manifest)
    for vid in demo_ids:
        new_manifest[vid]["split"] = "DEMO"

    # Save demo set details for reference
    demo_info = []
    for vid in sorted(demo_ids):
        meta = new_manifest[vid]
        demo_info.append({
            "video_id": vid,
            "ef": meta["ef"],
            "age": meta.get("age", -1),
            "category": ef_category(float(meta["ef"]), float(meta.get("age", 8.0))),
        })

    demo_path = output_path / "demo_holdout.json"
    with open(demo_path, "w") as f:
        json.dump(demo_info, f, indent=2)

    logger.info(
        "Demo holdout: %d samples carved from TEST → %s",
        len(demo_ids), demo_path,
    )
    for item in demo_info:
        logger.info(
            "  DEMO: %s | EF=%.1f%% | category=%s",
            item["video_id"], item["ef"], item["category"],
        )

    logger.info("Remaining TEST samples: %d", len(test_entries) - len(demo_ids))

    return new_manifest


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class RegressionTrainConfig:
    """Hyperparameters for MLP regression training."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    patience: int = 15  # Early stopping patience
    warmup_epochs: int = 5
    huber_delta: float = 5.0
    noise_std: float = 0.01  # Embedding noise augmentation
    use_metadata: bool = False  # Whether to use EFRegressorWithMeta
    model_type: str = "mlp"  # "mlp" or "v2"
    hidden_dim: int = 512
    dropout: float = 0.3
    class_weighted: bool = True  # Use WeightedRandomSampler
    # Composite loss weights (v2 model uses these by default)
    ordinal_weight: float = 0.1       # α — ordinal boundary penalty
    asymmetric_weight: float = 0.05   # β — clinical asymmetric penalty
    range_weight: float = 0.01        # γ — range [0, 100] penalty
    use_composite_loss: bool = False   # True = composite_loss, False = huber_loss
    # Adult curriculum pre-training
    adult_epochs: int = 20  # Epochs for adult pre-training phase
    adult_lr: float = 1e-3  # LR during adult phase
    pediatric_lr_factor: float = 0.3  # LR multiplier when switching to pediatric
    # Demo holdout
    demo_holdout: int = 0  # Number of test samples to reserve for demo
    demo_seed: int = 42  # Seed for reproducible demo selection


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    view: str,
    embeddings_dir: str = str(PROJECT_ROOT / "data" / "embeddings"),
    output_dir: str = str(PROJECT_ROOT / "checkpoints" / "regression"),
    config: RegressionTrainConfig | None = None,
    adult_embeddings_dir: str | None = None,
    device: str = "cuda",
) -> dict:
    """Train the EF regression model.

    Args:
        view: "A4C" or "PSAX"
        embeddings_dir: Directory with pre-extracted embeddings
        output_dir: Where to save checkpoints
        config: Training hyperparameters
        adult_embeddings_dir: Optional adult data for curriculum pre-training
        device: CUDA device

    Returns:
        dict with training history and final metrics
    """
    if config is None:
        config = RegressionTrainConfig()

    output_path = Path(output_dir) / f"ef_regression_{view.lower()}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # ----- Load manifest -----
    manifest_path = Path(embeddings_dir) / f"pediatric_{view.lower()}" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run extract_embeddings.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Detect embedding dimension from first entry
    first_entry = next(iter(manifest.values()))
    sample_emb = torch.load(first_entry["embedding_path"], weights_only=True)
    num_frames, embed_dim = sample_emb.shape
    logger.info("Detected embedding shape: (%d frames, %d dim)", num_frames, embed_dim)

    # ----- Datasets -----
    # If demo holdout requested, reclassify some TEST entries as DEMO in a copy
    if config.demo_holdout > 0:
        manifest = _create_demo_split(manifest, config.demo_holdout, config.demo_seed, output_path)

    train_dataset = EFEmbeddingDataset(
        manifest, split="TRAIN", transform_noise=config.noise_std,
    )
    val_dataset = EFEmbeddingDataset(manifest, split="VAL")
    test_dataset = EFEmbeddingDataset(manifest, split="TEST")
    demo_dataset = EFEmbeddingDataset(manifest, split="DEMO") if config.demo_holdout > 0 else None

    logger.info("Splits: train=%d, val=%d, test=%d%s",
                len(train_dataset), len(val_dataset), len(test_dataset),
                f", demo={len(demo_dataset)}" if demo_dataset else "")

    # ----- Optional adult curriculum pre-training data -----
    adult_dataset = None
    if adult_embeddings_dir:
        adult_manifest_path = Path(adult_embeddings_dir) / "manifest.json"
        if adult_manifest_path.exists():
            with open(adult_manifest_path) as f:
                adult_manifest = json.load(f)
            adult_dataset = EFEmbeddingDataset(
                adult_manifest, split="TRAIN", transform_noise=config.noise_std,
            )
            logger.info("Loaded %d adult curriculum samples", len(adult_dataset))

    # ----- Sampler -----
    if config.class_weighted:
        sample_weights = compute_class_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ----- Model -----
    if config.use_metadata:
        model = EFRegressorWithMeta(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    elif config.model_type == "v2":
        model = EFRegressorV2(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        # v2 defaults to composite loss unless explicitly disabled
        if not config.use_composite_loss:
            config.use_composite_loss = True
            logger.info("v2 model: enabling composite loss automatically")
    else:
        model = EFRegressor(
            embed_dim=embed_dim,
            num_frames=num_frames,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    model = model.to(device)
    params = count_parameters(model)
    logger.info("Model: %s params (trainable: %s)",
                f"{params['total']:,}", f"{params['trainable']:,}")

    # ----- Optimizer & Scheduler -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Cosine annealing with warmup
    total_steps = config.num_epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- TensorBoard -----
    tb_writer = None
    if HAS_TENSORBOARD:
        tb_dir = output_path / "tensorboard"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info("TensorBoard logging to %s", tb_dir)

    # ----- Adult curriculum (optional Stage 1) -----
    if adult_dataset is not None and len(adult_dataset) > 0:
        n_adult_epochs = config.adult_epochs
        logger.info("=== Stage 1: Adult curriculum pre-training (%d epochs, %d samples) ===", n_adult_epochs, len(adult_dataset))
    elif adult_dataset is not None and len(adult_dataset) == 0:
        logger.warning("Adult curriculum requested but 0 samples loaded — skipping")
        adult_dataset = None
    if adult_dataset is not None and len(adult_dataset) > 0:

        # Class-weighted sampling for adult data (broader EF distribution)
        adult_weights = compute_class_weights(adult_dataset)
        adult_sampler = WeightedRandomSampler(
            weights=adult_weights,
            num_samples=len(adult_dataset),
            replacement=True,
        )
        adult_loader = DataLoader(
            adult_dataset,
            batch_size=config.batch_size,
            sampler=adult_sampler,
            num_workers=4,
            pin_memory=True,
        )

        # Adult optimizer (may use different LR)
        adult_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.adult_lr,
            weight_decay=config.weight_decay,
        )

        best_adult_loss = float("inf")
        for adult_epoch in range(n_adult_epochs):
            adult_loss, adult_grad = _train_epoch(
                model, adult_loader, adult_optimizer, config, device,
                epoch=adult_epoch, prefix="adult",
            )
            # Validate on pediatric val set to monitor transfer
            adult_val = _validate(model, val_loader, config, device)
            logger.info(
                "  Adult %2d/%d | loss=%.4f | peds_val_MAE=%.2f%% | "
                "peds_val_R²=%.4f | grad=%.3f",
                adult_epoch + 1, n_adult_epochs,
                adult_loss, adult_val["mae"], adult_val["r2"],
                adult_grad,
            )
            if adult_loss < best_adult_loss:
                best_adult_loss = adult_loss

            # Log adult phase to TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("adult/train_loss", adult_loss, adult_epoch + 1)
                tb_writer.add_scalar("adult/peds_val_mae", adult_val["mae"], adult_epoch + 1)
                tb_writer.add_scalar("adult/peds_val_r2", adult_val["r2"], adult_epoch + 1)

        logger.info("Adult curriculum complete (best_loss=%.4f).", best_adult_loss)

        # Reduce LR for pediatric specialization
        pediatric_lr = config.learning_rate * config.pediatric_lr_factor
        logger.info(
            "Switching to pediatric phase: LR %.2e → %.2e (factor=%.1f)",
            config.learning_rate, pediatric_lr, config.pediatric_lr_factor,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = pediatric_lr

    # ----- Training loop -----
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
        "val_within_5": [],
        "val_pred_std": [],
        "val_n_unique": [],
        "grad_norm": [],
        "learning_rates": [],
    }
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    # ----- CSV epoch log -----
    csv_path = output_path / "epoch_log.csv"
    csv_fields = [
        "epoch", "train_loss", "val_loss", "val_mae", "val_r2",
        "val_within_5", "val_pred_std", "val_n_unique", "grad_norm", "lr",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    logger.info("=== Training %s regression model (%d epochs) ===", view, config.num_epochs)

    for epoch in range(config.num_epochs):
        t0 = time.time()

        # Train
        train_loss, epoch_grad_norm = _train_epoch(
            model, train_loader, optimizer, config, device,
            epoch=epoch, scheduler=scheduler,
        )

        # Validate
        val_metrics = _validate(model, val_loader, config, device)

        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_within_5"].append(val_metrics["within_5"])
        history["val_pred_std"].append(val_metrics["pred_std"])
        history["val_n_unique"].append(val_metrics["n_unique"])
        history["grad_norm"].append(epoch_grad_norm)
        history["learning_rates"].append(current_lr)

        # TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalars("loss", {
                "train": train_loss,
                "val": val_metrics["loss"],
            }, epoch + 1)
            tb_writer.add_scalar("metrics/val_mae", val_metrics["mae"], epoch + 1)
            tb_writer.add_scalar("metrics/val_r2", val_metrics["r2"], epoch + 1)
            tb_writer.add_scalar("metrics/val_within_5", val_metrics["within_5"], epoch + 1)
            tb_writer.add_scalar("metrics/val_pred_std", val_metrics["pred_std"], epoch + 1)
            tb_writer.add_scalar("metrics/val_n_unique", val_metrics["n_unique"], epoch + 1)
            tb_writer.add_scalar("train/grad_norm", epoch_grad_norm, epoch + 1)
            tb_writer.add_scalar("train/lr", current_lr, epoch + 1)

        # CSV log
        csv_writer.writerow({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_mae": f"{val_metrics['mae']:.4f}",
            "val_r2": f"{val_metrics['r2']:.6f}",
            "val_within_5": f"{val_metrics['within_5']:.4f}",
            "val_pred_std": f"{val_metrics['pred_std']:.4f}",
            "val_n_unique": val_metrics["n_unique"],
            "grad_norm": f"{epoch_grad_norm:.4f}",
            "lr": f"{current_lr:.2e}",
        })
        csv_file.flush()

        elapsed = time.time() - t0

        # Logging
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_MAE=%.2f%% | val_R²=%.4f | val_±5%%=%.1f%% | "
            "pred_std=%.1f | grad=%.3f | lr=%.2e | %.1fs",
            epoch + 1, config.num_epochs,
            train_loss, val_metrics["loss"],
            val_metrics["mae"], val_metrics["r2"],
            val_metrics["within_5"] * 100,
            val_metrics["pred_std"], epoch_grad_norm,
            current_lr, elapsed,
        )

        # Checkpointing
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": best_val_mae,
                "val_r2": val_metrics["r2"],
                "config": config.__dict__,
                "embed_dim": embed_dim,
                "num_frames": num_frames,
                "model_type": config.model_type,
            }, output_path / "best_model.pt")
            logger.info("  ✓ New best model (MAE=%.2f%%)", best_val_mae)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("  Early stopping at epoch %d (patience=%d)",
                            epoch + 1, config.patience)
                break

    # Close loggers
    if tb_writer is not None:
        tb_writer.close()
    csv_file.close()
    logger.info("Epoch log saved to %s", csv_path)

    # Save final model
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "embed_dim": embed_dim,
        "num_frames": num_frames,
        "model_type": config.model_type,
    }, output_path / "final_model.pt")

    # Save training history
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "Training complete. Best MAE=%.2f%% at epoch %d. "
        "Saved to %s",
        best_val_mae, best_epoch, output_path,
    )

    return {
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "output_dir": str(output_path),
        "history": history,
    }


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: RegressionTrainConfig,
    device: str,
    epoch: int = 0,
    scheduler=None,
    prefix: str = "",
) -> tuple[float, float]:
    """Run one training epoch. Returns (average_loss, average_grad_norm)."""
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    for embeddings, targets, metadata, _ in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)
        metadata = metadata.to(device)

        optimizer.zero_grad()

        if config.use_metadata and hasattr(model, "meta_branch"):
            preds = model(embeddings, metadata)
        else:
            preds = model(embeddings)

        if config.use_composite_loss:
            loss, _ = composite_loss(
                preds, targets,
                huber_delta=config.huber_delta,
                ordinal_weight=config.ordinal_weight,
                asymmetric_weight=config.asymmetric_weight,
                range_weight=config.range_weight,
            )
        else:
            loss = huber_loss(preds, targets, delta=config.huber_delta)
        loss.backward()

        # Gradient clipping + norm tracking
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_grad_norm += grad_norm.item()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_grad_norm = total_grad_norm / max(n_batches, 1)
    return avg_loss, avg_grad_norm


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    config: RegressionTrainConfig,
    device: str,
) -> dict:
    """Run validation. Returns dict with loss, mae, r2, within_5."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    for embeddings, targets, metadata, _ in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)
        metadata = metadata.to(device)

        if config.use_metadata and hasattr(model, "meta_branch"):
            preds = model(embeddings, metadata)
        else:
            preds = model(embeddings)

        if config.use_composite_loss:
            loss, _ = composite_loss(
                preds, targets,
                huber_delta=config.huber_delta,
                ordinal_weight=config.ordinal_weight,
                asymmetric_weight=config.asymmetric_weight,
                range_weight=config.range_weight,
            )
        else:
            loss = huber_loss(preds, targets, delta=config.huber_delta)
        total_loss += loss.item()
        n_batches += 1

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))
    within_5 = float(np.mean(np.abs(y_true - y_pred) <= 5.0))

    # Prediction diversity check
    pred_std = float(np.std(y_pred))
    n_unique = len(set(np.round(y_pred, 1)))
    if pred_std < 2.0:
        logger.warning(
            "⚠️  Low prediction diversity: std=%.2f, unique_vals=%d",
            pred_std, n_unique,
        )

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": mae,
        "r2": r2,
        "within_5": within_5,
        "pred_std": pred_std,
        "n_unique": n_unique,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EF regression model")
    parser.add_argument("--view", default="A4C", choices=["A4C", "PSAX"],
                        help="Echo view to train on")
    parser.add_argument("--embeddings-dir", default=str(PROJECT_ROOT / "data" / "embeddings"),
                        help="Directory with pre-extracted embeddings")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "regression"),
                        help="Checkpoint output directory")
    parser.add_argument("--model-type", default="mlp", choices=["mlp", "v2"],
                        help="Model architecture: mlp (baseline) or v2 (improved)")
    parser.add_argument("--use-adult-curriculum", action="store_true",
                        help="Pre-train on adult EchoNet-Dynamic embeddings")
    parser.add_argument("--adult-epochs", type=int, default=20,
                        help="Epochs for adult pre-training phase")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use-metadata", action="store_true",
                        help="Include patient age/sex in model")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Embedding noise augmentation std")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class-weighted sampling")
    parser.add_argument("--composite-loss", action="store_true",
                        help="Use composite loss (auto-enabled for v2)")
    parser.add_argument("--ordinal-weight", type=float, default=0.1,
                        help="α for ordinal boundary loss")
    parser.add_argument("--asymmetric-weight", type=float, default=0.05,
                        help="β for clinical asymmetric loss")
    parser.add_argument("--range-weight", type=float, default=0.01,
                        help="γ for range penalty loss")
    parser.add_argument("--demo-holdout", type=int, default=0,
                        help="Reserve N test samples for live demo (0=disabled)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = RegressionTrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        use_metadata=args.use_metadata,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        noise_std=args.noise,
        class_weighted=not args.no_class_weights,
        use_composite_loss=args.composite_loss,
        ordinal_weight=args.ordinal_weight,
        asymmetric_weight=args.asymmetric_weight,
        range_weight=args.range_weight,
        adult_epochs=args.adult_epochs,
        demo_holdout=args.demo_holdout,
    )

    adult_dir = None
    if args.use_adult_curriculum:
        adult_dir = str(Path(args.embeddings_dir) / "adult_a4c")

    results = train(
        view=args.view,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        config=config,
        adult_embeddings_dir=adult_dir,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"  Training Complete — {args.view} EF Regression")
    print(f"{'='*60}")
    print(f"  Best Val MAE:  {results['best_val_mae']:.2f}%")
    print(f"  Best Epoch:    {results['best_epoch']}/{results['total_epochs']}")
    print(f"  Output:        {results['output_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
