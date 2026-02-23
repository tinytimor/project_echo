"""Train Model Garden architectures (multi-task, temporal, ensemble).

Usage:
    python -m echoguard.regression.train_garden --view A4C --model-type multitask
    python -m echoguard.regression.train_garden --view A4C --model-type temporal
    python -m echoguard.regression.train_garden --view A4C --model-type ensemble

Multi-task and temporal models consume the same SigLIP embeddings as the base MLP.
Ensemble training requires pre-trained MLP + multitask + temporal checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


def _safe_collate(batch):
    """Custom collate that avoids PyTorch shared-memory pre-allocation.

    The default collate_tensor_fn tries to pre-allocate an output tensor in
    shared memory (for num_workers > 0) and then resize it — this fails with
    'Trying to resize storage that is not resizable' on newer PyTorch / Python
    3.13 for larger tensors (e.g. 16-frame embeddings).  Using torch.stack()
    directly bypasses that shared-memory path entirely.
    """
    embeddings, efs, metas, vids = zip(*batch)
    return (
        torch.stack([e.contiguous() for e in embeddings]),
        torch.stack(list(efs)),
        torch.stack(list(metas)),
        list(vids),
    )

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from echoguard.config import PROJECT_ROOT, SPLIT_MAP, ef_category, age_group
from echoguard.regression.model import (
    EFRegressor, EFRegressorV2, EFRegressorWithMeta,
    huber_loss, composite_loss, ordinal_bce_loss, count_parameters,
)
from echoguard.regression.model_garden import (
    EFMultiTaskModel,
    EFTemporalTransformer,
    EFEnsemble,
    EFExplainer,
    N_CLASSES,
    ef_to_class_index,
    CLASS_MIDPOINTS,
    multitask_loss,
    create_model,
)
from echoguard.regression.train import EFEmbeddingDataset, compute_class_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GardenTrainConfig:
    """Hyperparameters for Model Garden training."""
    model_type: str = "multitask"  # multitask | temporal | tcn | v2 | lstm | lstm_full | lstm_crf | ensemble | classify
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    patience: int = 15
    warmup_epochs: int = 5
    huber_delta: float = 5.0
    noise_std: float = 0.01
    hidden_dim: int = 512
    dropout: float = 0.3
    class_weighted: bool = True
    # Composite loss weights
    ordinal_weight: float = 0.1       # α — ordinal boundary penalty
    asymmetric_weight: float = 0.05   # β — clinical asymmetric penalty
    range_weight: float = 0.01        # γ — range [0, 100] penalty
    boundary_push_weight: float = 0.2 # δ — boundary push (hyperdynamic/reduced)
    # Multi-task specific
    classification_weight: float = 0.3  # λ for CE loss in joint training
    # Temporal specific
    n_heads: int = 8
    n_layers: int = 2
    # TCN specific
    num_levels: int = 4  # Dilation levels (TCN only)
    # Temporal specific — input projection before transformer
    proj_dim: int = 192  # 1152 → proj_dim reduces params ~36× vs raw d_model=1152
    # LSTM specific (BiLSTM / LSTMFullSeq / LSTM-CRF)
    lstm_hidden: int = 256    # BiLSTM hidden size (doubled when bidirectional=True)
    lstm_layers: int = 2      # Number of stacked LSTM layers
    crf_loss_weight: float = 0.3  # λ: weight of CRF NLL loss vs EF regression loss
    all_frames: bool = False  # Use full frame sequence for lstm_full / lstm_crf
    # Ordinal model specific
    ordinal_bce_weight: float = 1.0  # Weight for ordinal BCE loss (ordinal model only)
    # Classification model specific
    classify_focal_gamma: float = 0.0  # 0=standard weighted CE; >0=focal CE (γ=2 recommended)
    # Ensemble specific
    mlp_checkpoint: str = ""  # Required for ensemble training
    multitask_checkpoint: str = ""
    temporal_checkpoint: str = ""
    v2_checkpoint: str = ""
    tcn_checkpoint: str = ""   # TCN can be used as ensemble component
    # Adult curriculum
    adult_epochs: int = 0  # 0 = no adult curriculum
    adult_lr: float = 1e-3
    pediatric_lr_factor: float = 0.3
    # Resume / init from checkpoint (for co-training)
    init_checkpoint: str = ""  # Path to checkpoint to load weights from
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine" | "plateau"  (plateau = ReduceLROnPlateau)
    # Mixup embedding augmentation
    use_mixup: bool = False
    mixup_alpha: float = 0.2      # Beta distribution α — 0.2 gives mild mixing
    # Demo
    demo_holdout: int = 0
    demo_seed: int = 42


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    view: str,
    embeddings_dir: str = str(PROJECT_ROOT / "data" / "embeddings"),
    output_dir: str = str(PROJECT_ROOT / "checkpoints" / "regression"),
    config: GardenTrainConfig | None = None,
    adult_embeddings_dir: str | None = None,
    device: str = "cuda",
) -> dict:
    """Train a Model Garden model.

    Args:
        view: "A4C" or "PSAX"
        embeddings_dir: Directory with pre-extracted embeddings
        output_dir: Where to save checkpoints
        config: Training hyperparameters
        adult_embeddings_dir: Optional adult data for curriculum pre-training
        device: CUDA device

    Returns:
        dict with training results
    """
    if config is None:
        config = GardenTrainConfig()

    model_type = config.model_type
    output_path = Path(output_dir) / f"garden_{model_type}_{view.lower()}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # ----- Load manifest -----
    manifest_path = Path(embeddings_dir) / f"pediatric_{view.lower()}" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Detect embedding dimensions
    first_entry = next(iter(manifest.values()))
    sample_emb = torch.load(first_entry["embedding_path"], weights_only=True)
    num_frames, embed_dim = sample_emb.shape
    logger.info("Embedding shape: (%d frames, %d dim)", num_frames, embed_dim)

    # ----- Datasets -----
    if config.demo_holdout > 0:
        from echoguard.regression.train import _create_demo_split
        manifest = _create_demo_split(
            manifest, config.demo_holdout, config.demo_seed, output_path
        )

    train_dataset = EFEmbeddingDataset(
        manifest, split="TRAIN", transform_noise=config.noise_std,
    )
    val_dataset = EFEmbeddingDataset(manifest, split="VAL")

    if len(train_dataset) == 0:
        raise RuntimeError("No training samples found!")

    # ----- Sampler -----
    if config.class_weighted:
        sample_weights = compute_class_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            sampler=sampler, num_workers=4, pin_memory=False,
            collate_fn=_safe_collate,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=4, pin_memory=False,
            collate_fn=_safe_collate,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=4, pin_memory=False,
        collate_fn=_safe_collate,
    )

    # ----- Model -----
    if model_type == "ensemble":
        return _train_ensemble(
            view, config, train_loader, val_loader,
            embed_dim, num_frames, output_path, device,
        )

    # LSTM and delta models use lstm_hidden (defaults to 256) rather than the
    # Transformer/MLP hidden_dim (defaults to 512), avoiding accidentally large models.
    # Ordinal and temporal models use hidden_dim for their MLP heads.
    _hidden = (
        config.lstm_hidden
        if model_type in ("lstm", "lstm_full", "lstm_crf", "delta", "classify")
        else config.hidden_dim
    )
    model = create_model(
        model_type=model_type,
        embed_dim=embed_dim,
        num_frames=num_frames,
        hidden_dim=_hidden,
        dropout=config.dropout,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        num_levels=config.num_levels,
        proj_dim=config.proj_dim,
        # LSTM-specific — passed via kwargs, ignored by non-LSTM models
        crf_loss_weight=config.crf_loss_weight,
    )
    model = model.to(device)
    params = count_parameters(model)
    logger.info(
        "Model: %s (%s params, trainable: %s)",
        model_type, f"{params['total']:,}", f"{params['trainable']:,}",
    )

    # ----- Optional: load pre-trained weights (for co-training) -----
    if config.init_checkpoint and Path(config.init_checkpoint).exists():
        ckpt = torch.load(config.init_checkpoint, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        logger.info(
            "Loaded init checkpoint: %s (epoch %s, MAE %.2f%%)",
            config.init_checkpoint,
            ckpt.get("epoch", "?"),
            ckpt.get("val_mae", 0.0),
        )

    # ----- Classification label weights (for multi-task) -----
    class_label_weights = None
    if model_type == "multitask":
        # Compute class-level weights from training data
        from collections import Counter
        cats = [
            ef_to_class_index(entry["ef"])
            for entry in train_dataset.entries
        ]
        cat_counts = Counter(cats)
        total = len(cats)
        weights = torch.zeros(N_CLASSES, device=device)
        for cls_idx in range(N_CLASSES):
            count = cat_counts.get(cls_idx, 1)
            weights[cls_idx] = total / (N_CLASSES * count)
        class_label_weights = weights
        logger.info("Classification class weights: %s", weights.tolist())

    # ----- Optimizer & Scheduler -----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = config.num_epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)

    if config.scheduler_type == "plateau":
        # ReduceLROnPlateau: halves LR when val_MAE doesn't improve for 5 epochs.
        # Much more stable than cosine-warmup for oscillating Temporal models.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
            min_lr=1e-6,
        )
    else:
        # Default: cosine annealing with linear warmup
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

    # ----- Optional adult curriculum -----
    if adult_embeddings_dir and config.adult_epochs > 0:
        adult_manifest_path = Path(adult_embeddings_dir) / "manifest.json"
        if adult_manifest_path.exists():
            with open(adult_manifest_path) as f:
                adult_manifest = json.load(f)
            adult_dataset = EFEmbeddingDataset(
                adult_manifest, split="TRAIN", transform_noise=config.noise_std,
            )
            if len(adult_dataset) > 0:
                logger.info(
                    "=== Adult curriculum (%d epochs, %d samples) ===",
                    config.adult_epochs, len(adult_dataset),
                )
                adult_weights = compute_class_weights(adult_dataset)
                adult_sampler = WeightedRandomSampler(
                    weights=adult_weights,
                    num_samples=len(adult_dataset),
                    replacement=True,
                )
                adult_loader = DataLoader(
                    adult_dataset, batch_size=config.batch_size,
                    sampler=adult_sampler, num_workers=4, pin_memory=False,
                    collate_fn=_safe_collate,
                )
                adult_optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config.adult_lr,
                    weight_decay=config.weight_decay,
                )

                for ae in range(config.adult_epochs):
                    adult_loss = _train_epoch(
                        model, adult_loader, adult_optimizer,
                        config, device, model_type=model_type,
                        class_label_weights=class_label_weights,
                    )
                    val_metrics = _validate(
                        model, val_loader, config, device,
                        model_type=model_type,
                        class_label_weights=class_label_weights,
                    )
                    logger.info(
                        "  Adult %2d/%d | loss=%.4f | peds_val_MAE=%.2f%%",
                        ae + 1, config.adult_epochs, adult_loss, val_metrics["mae"],
                    )

                # Lower LR for pediatric
                pediatric_lr = config.learning_rate * config.pediatric_lr_factor
                for pg in optimizer.param_groups:
                    pg["lr"] = pediatric_lr
                logger.info("Switching to pediatric LR: %.2e", pediatric_lr)

    # ----- Training loop -----
    history = {
        "train_loss": [], "val_loss": [], "val_mae": [], "val_r2": [],
        "val_within_5": [], "val_pred_std": [], "val_n_unique": [],
    }
    if model_type == "multitask":
        history["val_cls_acc"] = []
        history["val_cls_loss"] = []

    best_val_mae = float("inf")
    best_val_r2 = 0.0
    best_val_clin_acc = 0.0
    # composite = MAE - 5.0*ClinAcc - 3.0*max(R²,0)  →  lower is better
    # 1pp ClinAcc ≈ 0.05% MAE benefit; 0.1 R² ≈ 0.30% MAE benefit
    best_composite_score = float("inf")
    best_epoch = 0
    patience_counter = 0

    # CSV log
    csv_path = output_path / "epoch_log.csv"
    csv_fields = [
        "epoch", "train_loss", "val_loss", "val_mae", "val_r2",
        "val_within_5", "val_pred_std", "val_clin_acc", "lr",
    ]
    if model_type == "multitask":
        csv_fields += ["val_cls_acc", "val_cls_loss"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    logger.info(
        "=== Training %s %s model (%d epochs) ===",
        view, model_type, config.num_epochs,
    )

    for epoch in range(config.num_epochs):
        t0 = time.time()

        train_loss = _train_epoch(
            model, train_loader, optimizer, config, device,
            model_type=model_type,
            class_label_weights=class_label_weights,
            scheduler=scheduler if config.scheduler_type != "plateau" else None,
        )

        val_metrics = _validate(
            model, val_loader, config, device,
            model_type=model_type,
            class_label_weights=class_label_weights,
        )
        # ReduceLROnPlateau steps on epoch-level val_MAE (not per batch)
        if config.scheduler_type == "plateau":
            scheduler.step(val_metrics["mae"])

        current_lr = optimizer.param_groups[0]["lr"]

        # Record
        history["train_loss"].append(train_loss)
        for k in ["val_loss", "val_mae", "val_r2", "val_within_5",
                   "val_pred_std", "val_n_unique"]:
            key = k.replace("val_", "")
            if key in val_metrics:
                history[k].append(val_metrics[key])
            elif k in val_metrics:
                history[k].append(val_metrics[k])

        if model_type == "multitask":
            history["val_cls_acc"].append(val_metrics.get("cls_acc", 0))
            history["val_cls_loss"].append(val_metrics.get("cls_loss", 0))

        # TensorBoard
        if tb_writer:
            tb_writer.add_scalar("loss/train", train_loss, epoch + 1)
            tb_writer.add_scalar("loss/val", val_metrics["loss"], epoch + 1)
            tb_writer.add_scalar("metrics/val_mae", val_metrics["mae"], epoch + 1)
            tb_writer.add_scalar("metrics/val_r2", val_metrics["r2"], epoch + 1)
            if model_type == "multitask":
                tb_writer.add_scalar(
                    "metrics/val_cls_acc",
                    val_metrics.get("cls_acc", 0), epoch + 1,
                )

        # CSV
        row = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_mae": f"{val_metrics['mae']:.4f}",
            "val_r2": f"{val_metrics['r2']:.6f}",
            "val_within_5": f"{val_metrics['within_5']:.4f}",
            "val_pred_std": f"{val_metrics['pred_std']:.4f}",
            "val_clin_acc": f"{val_metrics.get('clin_acc', 0):.4f}",
            "lr": f"{current_lr:.2e}",
        }
        if model_type == "multitask":
            row["val_cls_acc"] = f"{val_metrics.get('cls_acc', 0):.4f}"
            row["val_cls_loss"] = f"{val_metrics.get('cls_loss', 0):.6f}"
        csv_writer.writerow(row)
        csv_file.flush()

        elapsed = time.time() - t0

        log_msg = (
            f"Epoch {epoch+1:3d}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | val_MAE={val_metrics['mae']:.2f}% | "
            f"val_R²={val_metrics['r2']:.4f} | val_±5%={val_metrics['within_5']*100:.1f}% | "
            f"ClinAcc={val_metrics.get('clin_acc', 0)*100:.1f}%"
        )
        if model_type == "multitask":
            log_msg += f" | cls_acc={val_metrics.get('cls_acc', 0)*100:.1f}%"
        log_msg += f" | lr={current_lr:.2e} | {elapsed:.1f}s"
        logger.info(log_msg)

        # Composite score (lower is better):
        #   score = MAE - 5.0 * ClinAcc - 3.0 * max(R², 0)
        #   Each 1pp  ClinAcc ≈ 0.05% MAE benefit
        #   Each 0.10 R²      ≈ 0.30% MAE benefit
        # This jointly rewards low MAE, high clinical-category accuracy, AND high R².
        clin_acc = val_metrics.get("clin_acc", 0.0)
        r2       = val_metrics["r2"]
        composite_score = val_metrics["mae"] - 5.0 * clin_acc - 3.0 * max(r2, 0.0)
        if composite_score < best_composite_score:
            best_composite_score = composite_score
            best_val_mae = val_metrics["mae"]
            best_val_r2 = r2
            best_val_clin_acc = clin_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": best_val_mae,
                "val_clin_acc": clin_acc,
                "val_r2": r2,
                "composite_score": composite_score,
                "config": config.__dict__,
                "embed_dim": embed_dim,
                "num_frames": num_frames,
                "model_type": model_type,
            }, output_path / "best_model.pt")
            logger.info(
                "  ✓ New best model (MAE=%.2f%%, ClinAcc=%.1f%%, R²=%.4f, score=%.3f)",
                best_val_mae, clin_acc * 100, r2, composite_score,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if tb_writer:
        tb_writer.close()
    csv_file.close()

    # Save final model
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "embed_dim": embed_dim,
        "num_frames": num_frames,
        "model_type": model_type,
    }, output_path / "final_model.pt")

    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "Training complete. Best MAE=%.2f%% at epoch %d → %s",
        best_val_mae, best_epoch, output_path,
    )

    return {
        "best_val_mae": best_val_mae,
        "best_val_r2": best_val_r2,
        "best_val_clin_acc": best_val_clin_acc,
        "best_composite_score": best_composite_score,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "output_dir": str(output_path),
        "model_type": model_type,
    }


# ---------------------------------------------------------------------------
# Epoch functions
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: GardenTrainConfig,
    device: str,
    model_type: str = "multitask",
    class_label_weights: torch.Tensor | None = None,
    scheduler=None,
) -> float:
    """One training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for embeddings, targets, metadata, _ in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)
        metadata = metadata.to(device)

        # Mixup embedding augmentation (disabled for multitask — class labels hard to mix)
        if config.use_mixup and model_type != "multitask" and torch.rand(1).item() < 0.5:
            lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
            idx = torch.randperm(embeddings.size(0), device=device)
            embeddings = lam * embeddings + (1.0 - lam) * embeddings[idx]
            targets    = lam * targets    + (1.0 - lam) * targets[idx]

        optimizer.zero_grad()

        if model_type == "multitask":
            ef_pred, class_logits = model(embeddings, return_probs=True)
            # Create class labels from EF targets
            class_targets = torch.tensor(
                [ef_to_class_index(float(t)) for t in targets],
                device=device, dtype=torch.long,
            )
            loss, _ = multitask_loss(
                ef_pred, class_logits, targets, class_targets,
                huber_delta=config.huber_delta,
                classification_weight=config.classification_weight,
                class_weights=class_label_weights,
                ordinal_weight=config.ordinal_weight,
                asymmetric_weight=config.asymmetric_weight,
                range_weight=config.range_weight,
            )
        elif model_type in ("temporal", "v2", "tcn", "lstm", "lstm_full", "lstm_crf", "delta"):
            preds = model(embeddings)
            loss, _ = composite_loss(
                preds, targets,
                huber_delta=config.huber_delta,
                ordinal_weight=config.ordinal_weight,
                asymmetric_weight=config.asymmetric_weight,
                range_weight=config.range_weight,
                boundary_push_weight=config.boundary_push_weight,
            )
        elif model_type == "ordinal":
            # Ordinal model returns (ef_pred, ordinal_logits)
            ef_pred, ord_logits = model(embeddings, return_ordinal=True)
            reg_loss, _ = composite_loss(
                ef_pred, targets,
                huber_delta=config.huber_delta,
                ordinal_weight=config.ordinal_weight,
                asymmetric_weight=config.asymmetric_weight,
                range_weight=config.range_weight,
                boundary_push_weight=config.boundary_push_weight,
            )
            loss = reg_loss + config.ordinal_bce_weight * ordinal_bce_loss(ord_logits, targets)
        elif model_type == "classify":
            # Direct 4-class classification — single loss, no conflicting gradients
            class_targets = torch.tensor(
                [ef_to_class_index(float(t)) for t in targets],
                device=device, dtype=torch.long,
            )
            logits = model(embeddings)          # (B, 4)
            if config.classify_focal_gamma > 0:
                # Focal CE: down-weight easy (normal) samples, up-weight hard minority
                import torch.nn.functional as _F
                log_probs = _F.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)
                nll = -log_probs.gather(1, class_targets.unsqueeze(1)).squeeze(1)
                pt  = probs.gather(1, class_targets.unsqueeze(1)).squeeze(1)
                focal_w = (1.0 - pt) ** config.classify_focal_gamma
                if class_label_weights is not None:
                    focal_w = focal_w * class_label_weights[class_targets]
                loss = (focal_w * nll).mean()
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits, class_targets,
                    weight=class_label_weights,
                )
        else:
            preds = model(embeddings)
            loss = huber_loss(preds, targets, delta=config.huber_delta)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    config: GardenTrainConfig,
    device: str,
    model_type: str = "multitask",
    class_label_weights: torch.Tensor | None = None,
) -> dict:
    """Validate. Returns dict with metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_ord_cats: list[int] = []   # ordinal model only — stores predicted category indices
    total_loss = 0.0
    total_cls_correct = 0
    total_cls_count = 0
    total_cls_loss = 0.0
    n_batches = 0

    for embeddings, targets, metadata, _ in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)

        if model_type == "multitask":
            ef_pred, class_logits = model(embeddings, return_probs=True)
            class_targets = torch.tensor(
                [ef_to_class_index(float(t)) for t in targets],
                device=device, dtype=torch.long,
            )
            loss, loss_dict = multitask_loss(
                ef_pred, class_logits, targets, class_targets,
                huber_delta=config.huber_delta,
                classification_weight=config.classification_weight,
                class_weights=class_label_weights,
                ordinal_weight=config.ordinal_weight,
                asymmetric_weight=config.asymmetric_weight,
                range_weight=config.range_weight,
            )
            # Classification accuracy
            cls_pred = class_logits.argmax(dim=-1)
            total_cls_correct += (cls_pred == class_targets).sum().item()
            total_cls_count += len(targets)
            total_cls_loss += loss_dict["classification_loss"]
            preds = ef_pred
        elif model_type in ("temporal", "v2", "tcn", "lstm", "lstm_full", "lstm_crf", "delta"):
            preds = model(embeddings)
            loss, _ = composite_loss(
                preds, targets,
                huber_delta=config.huber_delta,
                ordinal_weight=config.ordinal_weight,
                asymmetric_weight=config.asymmetric_weight,
                range_weight=config.range_weight,
                boundary_push_weight=config.boundary_push_weight,
            )
        elif model_type == "ordinal":
            # Ordinal model: regression output for metrics, ordinal logits for ClinAcc
            ef_pred, ord_logits = model(embeddings, return_ordinal=True)
            reg_loss, _ = composite_loss(
                ef_pred, targets,
                huber_delta=config.huber_delta,
                ordinal_weight=config.ordinal_weight,
                asymmetric_weight=config.asymmetric_weight,
                range_weight=config.range_weight,
                boundary_push_weight=config.boundary_push_weight,
            )
            loss = reg_loss + config.ordinal_bce_weight * ordinal_bce_loss(ord_logits, targets)
            preds = ef_pred
            # Store ordinal category predictions (sum of exceeded thresholds)
            ord_cats = (torch.sigmoid(ord_logits) > 0.5).long().sum(dim=-1)
            all_ord_cats.extend(ord_cats.cpu().numpy())
        elif model_type == "classify":
            class_targets = torch.tensor(
                [ef_to_class_index(float(t)) for t in targets],
                device=device, dtype=torch.long,
            )
            logits = model(embeddings)          # (B, 4)
            if config.classify_focal_gamma > 0:
                import torch.nn.functional as _F
                log_probs = _F.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)
                nll = -log_probs.gather(1, class_targets.unsqueeze(1)).squeeze(1)
                pt  = probs.gather(1, class_targets.unsqueeze(1)).squeeze(1)
                focal_w = (1.0 - pt) ** config.classify_focal_gamma
                if class_label_weights is not None:
                    focal_w = focal_w * class_label_weights[class_targets]
                loss = (focal_w * nll).mean()
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits, class_targets,
                    weight=class_label_weights,
                )
            # Convert argmax prediction → proxy EF using class midpoints
            pred_cats = logits.argmax(dim=-1)
            _midpoints = torch.tensor(CLASS_MIDPOINTS, device=device)
            preds = _midpoints[pred_cats]               # (B,) proxy EF%
            # Store predicted categories for ClinAcc (overrides regression mapping)
            all_ord_cats.extend(pred_cats.cpu().numpy())
        else:
            preds = model(embeddings)
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
    pred_std = float(np.std(y_pred))
    n_unique = len(set(np.round(y_pred, 1)))

    # Clinical accuracy: correct EF category (reduced/borderline/normal/hyperdynamic)
    # For ordinal/classify models: use the learned category predictions directly
    # For all others: map regression EF output through fixed 45/55/70 thresholds
    if model_type in ("ordinal", "classify") and all_ord_cats:
        cat_pred = np.array(all_ord_cats, dtype=np.int64)
    else:
        cat_pred = np.array([ef_to_class_index(float(p)) for p in y_pred])
    cat_true = np.array([ef_to_class_index(float(t)) for t in y_true])
    clin_acc = float(np.mean(cat_pred == cat_true))

    result = {
        "loss": total_loss / max(n_batches, 1),
        "mae": mae,
        "r2": r2,
        "within_5": within_5,
        "pred_std": pred_std,
        "n_unique": n_unique,
        "clin_acc": clin_acc,
    }

    if model_type == "multitask" and total_cls_count > 0:
        result["cls_acc"] = total_cls_correct / total_cls_count
        result["cls_loss"] = total_cls_loss / max(n_batches, 1)

    return result


# ---------------------------------------------------------------------------
# Ensemble training (requires pre-trained components)
# ---------------------------------------------------------------------------

def _train_ensemble(
    view: str,
    config: GardenTrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    embed_dim: int,
    num_frames: int,
    output_path: Path,
    device: str,
) -> dict:
    """Train ensemble fusion layer on top of frozen component models.

    Loads MLP, multi-task, and temporal model checkpoints, runs inference
    to get predictions, then trains a small fusion network.
    """
    from echoguard.regression.evaluate import load_model as load_base_model

    # Load component models
    models = {}
    model_names = []

    for name, ckpt_path_str in [
        ("mlp", config.mlp_checkpoint),
        ("v2", config.v2_checkpoint),
        ("multitask", config.multitask_checkpoint),
        ("temporal", config.temporal_checkpoint),
        ("tcn", config.tcn_checkpoint),
    ]:
        if not ckpt_path_str:
            logger.info("No checkpoint for %s — skipping in ensemble", name)
            continue
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found for %s: %s", name, ckpt_path)
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_config = ckpt.get("config", {})
        ckpt_model_type = ckpt.get("model_type", "mlp")

        if ckpt_model_type == "multitask":
            m = EFMultiTaskModel(
                embed_dim=embed_dim, num_frames=num_frames,
                hidden_dim=ckpt_config.get("hidden_dim", 512),
                dropout=0.0,
            )
        elif ckpt_model_type == "temporal":
            m = EFTemporalTransformer(
                embed_dim=embed_dim, num_frames=num_frames,
                n_heads=ckpt_config.get("n_heads", 4),
                n_layers=ckpt_config.get("n_layers", 2),
                hidden_dim=ckpt_config.get("hidden_dim", 256),
                dropout=0.0,
                proj_dim=ckpt_config.get("proj_dim", 192),
            )
        elif ckpt_model_type == "tcn":
            from echoguard.regression.model_garden import EFTemporalCNN
            m = EFTemporalCNN(
                embed_dim=embed_dim, num_frames=num_frames,
                hidden=ckpt_config.get("hidden_dim", 256),
                num_levels=ckpt_config.get("num_levels", 4),
                dropout=0.0,
            )
        elif ckpt_model_type == "v2":
            m = EFRegressorV2(
                embed_dim=embed_dim, num_frames=num_frames,
                hidden_dim=ckpt_config.get("hidden_dim", 512),
                dropout=0.0,
            )
        else:
            m = EFRegressor(
                embed_dim=embed_dim, num_frames=num_frames,
                hidden_dim=ckpt_config.get("hidden_dim", 512),
                dropout=0.0,
            )

        m.load_state_dict(ckpt["model_state_dict"])
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False

        models[name] = m
        model_names.append(name)
        logger.info("Loaded frozen %s model from %s", name, ckpt_path)

    if len(models) < 2:
        raise RuntimeError(
            f"Ensemble requires at least 2 component models, got {len(models)}. "
            "Train MLP, multitask, and/or temporal models first."
        )

    # Create ensemble
    has_multitask = "multitask" in models
    n_class_probs = N_CLASSES if has_multitask else 0

    ensemble = EFEnsemble(
        n_models=len(models),
        n_class_probs=n_class_probs,
        n_meta=2,
    ).to(device)

    logger.info(
        "Ensemble fusion: %d models (%s), %s params",
        len(models), ", ".join(model_names),
        f"{count_parameters(ensemble)['trainable']:,}",
    )

    optimizer = torch.optim.AdamW(
        ensemble.parameters(), lr=config.learning_rate * 10,  # Faster for small model
        weight_decay=config.weight_decay,
    )

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    csv_path = output_path / "epoch_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "val_mae", "val_r2", "lr",
    ])
    csv_writer.writeheader()

    logger.info("=== Training ensemble (%d epochs) ===", config.num_epochs)

    for epoch in range(config.num_epochs):
        t0 = time.time()

        # Train
        ensemble.train()
        total_loss = 0.0
        n = 0
        for embeddings, targets, metadata, _ in train_loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            metadata = metadata.to(device)

            # Get predictions from all component models
            preds_list = []
            class_probs = torch.zeros(
                embeddings.size(0), n_class_probs, device=device
            )
            for name in model_names:
                m = models[name]
                if name == "multitask":
                    ef_p, logits = m(embeddings, return_probs=True)
                    preds_list.append(ef_p)
                    class_probs = F.softmax(logits, dim=-1)
                else:
                    preds_list.append(m(embeddings))

            model_preds = torch.stack(preds_list, dim=-1)  # (batch, n_models)
            meta_input = metadata[:, :2]  # age, sex_male

            optimizer.zero_grad()
            fused = ensemble(model_preds, class_probs, meta_input)
            loss = huber_loss(fused, targets, delta=config.huber_delta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        train_loss = total_loss / max(n, 1)

        # Validate
        ensemble.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for embeddings, targets, metadata, _ in val_loader:
                embeddings = embeddings.to(device)
                targets = targets.to(device)
                metadata = metadata.to(device)

                preds_list = []
                class_probs = torch.zeros(
                    embeddings.size(0), n_class_probs, device=device
                )
                for name in model_names:
                    m = models[name]
                    if name == "multitask":
                        ef_p, logits = m(embeddings, return_probs=True)
                        preds_list.append(ef_p)
                        class_probs = F.softmax(logits, dim=-1)
                    else:
                        preds_list.append(m(embeddings))

                model_preds = torch.stack(preds_list, dim=-1)
                meta_input = metadata[:, :2]
                fused = ensemble(model_preds, class_probs, meta_input)

                all_preds.extend(fused.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | loss=%.4f | MAE=%.2f%% | R²=%.4f | %.1fs",
            epoch + 1, config.num_epochs, train_loss, mae, r2, elapsed,
        )

        csv_writer.writerow({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_mae": f"{mae:.4f}",
            "val_r2": f"{r2:.6f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        csv_file.flush()

        if mae < best_val_mae:
            best_val_mae = mae
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": ensemble.state_dict(),
                "val_mae": best_val_mae,
                "val_r2": r2,
                "config": config.__dict__,
                "embed_dim": embed_dim,
                "num_frames": num_frames,
                "model_type": "ensemble",
                "component_models": model_names,
            }, output_path / "best_model.pt")
            logger.info("  ✓ New best ensemble (MAE=%.2f%%)", best_val_mae)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    csv_file.close()

    # Save final
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": ensemble.state_dict(),
        "config": config.__dict__,
        "embed_dim": embed_dim,
        "num_frames": num_frames,
        "model_type": "ensemble",
        "component_models": model_names,
    }, output_path / "final_model.pt")

    logger.info(
        "Ensemble complete. Best MAE=%.2f%% at epoch %d", best_val_mae, best_epoch,
    )

    return {
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "output_dir": str(output_path),
        "model_type": "ensemble",
        "component_models": model_names,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Model Garden architectures"
    )
    parser.add_argument("--view", default="A4C", choices=["A4C", "PSAX"])
    parser.add_argument(
        "--model-type", default="multitask",
        choices=["mlp", "multitask", "temporal", "v2", "tcn",
                 "lstm", "lstm_full", "lstm_crf", "delta", "ordinal",
                 "classify", "ensemble"],
        help="Model architecture to train",
    )
    parser.add_argument("--embeddings-dir", default=str(PROJECT_ROOT / "data" / "embeddings"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "regression"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--demo-holdout", type=int, default=0)
    # Loss weights
    parser.add_argument("--ordinal-weight", type=float, default=0.1,
                        help="α for ordinal boundary loss")
    parser.add_argument("--asymmetric-weight", type=float, default=0.05,
                        help="β for clinical asymmetric loss")
    parser.add_argument("--range-weight", type=float, default=0.01,
                        help="γ for range penalty loss")
    parser.add_argument("--boundary-push-weight", type=float, default=0.2,
                        help="δ for boundary push loss (hyperdynamic/reduced recall)")
    # Multi-task specific
    parser.add_argument("--cls-weight", type=float, default=0.3,
                        help="Classification loss weight λ (multitask only)")
    # Temporal specific
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--num-levels", type=int, default=4,
                        help="TCN dilation levels (tcn only)")
    parser.add_argument("--proj-dim", type=int, default=192,
                        help="Input projection dim for Temporal (reduces ~36x params)")
    # LSTM specific
    parser.add_argument("--lstm-hidden", type=int, default=256,
                        help="BiLSTM hidden size (doubled when bidirectional=True; default 256)")
    parser.add_argument("--lstm-layers", type=int, default=2,
                        help="Number of stacked LSTM layers (default 2)")
    parser.add_argument("--crf-loss-weight", type=float, default=0.3,
                        help="λ weight of CRF NLL loss vs EF regression (lstm_crf only; default 0.3)")
    parser.add_argument("--ordinal-bce-weight", type=float, default=1.0,
                        help="Weight for ordinal focal-BCE loss component (ordinal model only; default 1.0)")
    parser.add_argument("--classify-focal-gamma", type=float, default=0.0,
                        help="Focal CE gamma (classify only; 0=standard weighted CE, 2=focal; default 0.0)")
    parser.add_argument("--all-frames", action="store_true",
                        help="Use full frame sequence instead of 4 key frames "
                             "(required for lstm_full and lstm_crf)")
    # Ensemble specific
    parser.add_argument("--mlp-checkpoint", default="",
                        help="Path to trained MLP checkpoint")
    parser.add_argument("--v2-checkpoint", default="",
                        help="Path to trained v2 checkpoint")
    parser.add_argument("--multitask-checkpoint", default="",
                        help="Path to trained multitask checkpoint")
    parser.add_argument("--temporal-checkpoint", default="",
                        help="Path to trained temporal checkpoint")
    parser.add_argument("--tcn-checkpoint", default="",
                        help="Path to trained TCN checkpoint (used as ensemble component)")
    # Adult curriculum
    parser.add_argument("--use-adult-curriculum", action="store_true")
    parser.add_argument("--adult-epochs", type=int, default=20)
    parser.add_argument("--adult-lr", type=float, default=1e-3,
                        help="Learning rate for adult curriculum phase (default 1e-3)")
    parser.add_argument("--init-checkpoint", default="",
                        help="Resume/fine-tune from this checkpoint path")
    parser.add_argument("--scheduler-type", default="cosine",
                        choices=["cosine", "plateau"],
                        help="LR scheduler: cosine (default) or plateau (ReduceLROnPlateau)")
    parser.add_argument("--use-mixup", action="store_true",
                        help="Enable Mixup embedding augmentation (alpha=--mixup-alpha)")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="Beta distribution alpha for Mixup (default 0.2)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = GardenTrainConfig(
        model_type=args.model_type,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        noise_std=args.noise,
        class_weighted=not args.no_class_weights,
        ordinal_weight=args.ordinal_weight,
        asymmetric_weight=args.asymmetric_weight,
        range_weight=args.range_weight,
        boundary_push_weight=args.boundary_push_weight,
        classification_weight=args.cls_weight,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_levels=args.num_levels,
        proj_dim=args.proj_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        crf_loss_weight=args.crf_loss_weight,
        ordinal_bce_weight=args.ordinal_bce_weight,
        classify_focal_gamma=args.classify_focal_gamma,
        all_frames=args.all_frames,
        mlp_checkpoint=args.mlp_checkpoint,
        v2_checkpoint=args.v2_checkpoint,
        multitask_checkpoint=args.multitask_checkpoint,
        temporal_checkpoint=args.temporal_checkpoint,
        tcn_checkpoint=args.tcn_checkpoint,
        adult_epochs=args.adult_epochs if args.use_adult_curriculum else 0,
        adult_lr=args.adult_lr,
        init_checkpoint=args.init_checkpoint,
        scheduler_type=args.scheduler_type,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
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
    print(f"  {args.model_type.upper()} Training Complete — {args.view}")
    print(f"{'='*60}")
    print(f"  Best Val MAE:       {results['best_val_mae']:.2f}%")
    print(f"  Best Val R²:        {results['best_val_r2']:.4f}")
    print(f"  Best ClinAcc:       {results['best_val_clin_acc']*100:.1f}%")
    print(f"  Composite Score:    {results['best_composite_score']:.3f}")
    print(f"  Best Epoch:         {results['best_epoch']}/{results['total_epochs']}")
    print(f"  Output:             {results['output_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
