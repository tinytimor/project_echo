"""Configuration for EchoGuard-Peds pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Project root — all data/checkpoint/model paths are relative to this.
# Computed from this file's location: src/echoguard/config.py → ../../ = project root
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Pediatric EF Normal Ranges (age-adjusted)
# ---------------------------------------------------------------------------

PEDIATRIC_EF_NORMS: dict[str, dict[str, tuple[float, float]]] = {
    "neonate": {"normal": (60.0, 75.0), "borderline": (50.0, 59.9), "reduced_upper": (50.0,)},
    "infant": {"normal": (60.0, 75.0), "borderline": (50.0, 59.9), "reduced_upper": (50.0,)},
    "toddler": {"normal": (58.0, 72.0), "borderline": (48.0, 57.9), "reduced_upper": (48.0,)},
    "child": {"normal": (56.0, 70.0), "borderline": (46.0, 55.9), "reduced_upper": (46.0,)},
    "adolescent": {"normal": (55.0, 70.0), "borderline": (45.0, 54.9), "reduced_upper": (45.0,)},
}


def age_group(age_years: float) -> str:
    """Map a patient age (in years) to a clinical age group."""
    if age_years < 28 / 365.25:
        return "neonate"
    if age_years < 1.0:
        return "infant"
    if age_years < 4.0:
        return "toddler"
    if age_years < 13.0:
        return "child"
    return "adolescent"


def ef_category(ef_pct: float, age_years: float) -> str:
    """Classify an EF value as normal / borderline / reduced for a given age."""
    group = age_group(age_years)
    norms = PEDIATRIC_EF_NORMS[group]
    if norms["normal"][0] <= ef_pct <= norms["normal"][1]:
        return "normal"
    if norms["borderline"][0] <= ef_pct <= norms["borderline"][1]:
        return "borderline"
    if ef_pct > norms["normal"][1]:
        return "hyperdynamic"
    return "reduced"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Where to find MedGemma weights."""

    specialist_id: str = "google/medgemma-1.5-4b-it"
    specialist_local: str = str(PROJECT_ROOT / "local_models" / "medgemma-4b")
    orchestrator_id: str = "google/medgemma-27b-text-it"
    orchestrator_local: str = str(PROJECT_ROOT / "local_models" / "medgemma-27b")

    ef_lora_path: str = str(PROJECT_ROOT / "checkpoints" / "ef_lora")
    wall_motion_lora_path: str = str(PROJECT_ROOT / "checkpoints" / "wall_motion_lora")

    @property
    def specialist_path(self) -> str:
        """Resolve specialist model path (local first, then Hub)."""
        if Path(self.specialist_local).exists():
            return self.specialist_local
        return self.specialist_id

    @property
    def orchestrator_path(self) -> str:
        """Resolve orchestrator model path (local first, then Hub)."""
        if Path(self.orchestrator_local).exists():
            return self.orchestrator_local
        return self.orchestrator_id


# ---------------------------------------------------------------------------
# Split mapping (10-fold CV → TRAIN / VAL / TEST)
# ---------------------------------------------------------------------------

# EchoNet-Pediatric uses 10 folds (0-9). We map:
#   Folds 0-7 → TRAIN, Fold 8 → VAL, Fold 9 → TEST
SPLIT_MAP: dict[int, str] = {
    0: "TRAIN", 1: "TRAIN", 2: "TRAIN", 3: "TRAIN",
    4: "TRAIN", 5: "TRAIN", 6: "TRAIN", 7: "TRAIN",
    8: "VAL",
    9: "TEST",
}

VIEWS: list[str] = ["A4C", "PSAX"]


def map_split(fold: int | str) -> str:
    """Convert a numeric fold (0-9) to a canonical split name."""
    return SPLIT_MAP.get(int(fold), "TRAIN")


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Paths for EchoNet-Pediatric dataset.

    The dataset is organised by *view* (A4C / PSAX). Each view directory
    contains its own Videos/, FileList.csv, and VolumeTracings.csv.

    Directory layout::

        data_root/
        ├── A4C/
        │   ├── Videos/
        │   ├── FileList.csv
        │   └── VolumeTracings.csv
        └── PSAX/
            ├── Videos/
            ├── FileList.csv
            └── VolumeTracings.csv
    """

    data_root: str = str(PROJECT_ROOT / "data" / "echonet_pediatric")

    # Processed outputs
    frames_dir: str = str(PROJECT_ROOT / "data" / "echonet_pediatric" / "frames")
    training_jsonl: str = str(PROJECT_ROOT / "data" / "echonet_pediatric" / "training.jsonl")

    # Frame extraction settings
    frame_size: int = 224  # Upscale from 112×112 to 224×224
    num_key_frames: int = 4  # ED, mid-systole, ES, mid-diastole

    # --- per-view path helpers ------------------------------------------------

    def view_dir(self, view: str) -> Path:
        """Return the directory for a given view (A4C or PSAX)."""
        return Path(self.data_root) / view.upper()

    def videos_dir(self, view: str) -> Path:
        return self.view_dir(view) / "Videos"

    def file_list(self, view: str) -> Path:
        return self.view_dir(view) / "FileList.csv"

    def volume_tracings(self, view: str) -> Path:
        return self.view_dir(view) / "VolumeTracings.csv"

    def available_views(self) -> list[str]:
        """Return views that actually exist on disk."""
        return [v for v in VIEWS if self.view_dir(v).exists()]


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for LoRA fine-tuning."""

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "all-linear"

    # Training settings
    batch_size: int = 2
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    scheduler: str = "cosine"

    # Quantization
    load_in_4bit: bool = True
    bnb_compute_dtype: str = "bfloat16"

    # Output
    output_dir: str = str(PROJECT_ROOT / "checkpoints")
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Top-level configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.2


def load_config() -> PipelineConfig:
    """Load configuration, overriding with environment variables if set."""
    model_kwargs = {}
    local_4b = os.getenv("MEDGEMMA_4B_PATH")
    if local_4b:
        model_kwargs["specialist_local"] = local_4b
    local_27b = os.getenv("MEDGEMMA_27B_PATH")
    if local_27b:
        model_kwargs["orchestrator_local"] = local_27b

    data_kwargs = {}
    data_root = os.getenv("ECHONET_DATA_ROOT")
    if data_root:
        data_kwargs["data_root"] = data_root
        data_kwargs["frames_dir"] = f"{data_root}/frames"

    return PipelineConfig(
        model=ModelConfig(**model_kwargs),
        data=DataConfig(**data_kwargs),
    )
