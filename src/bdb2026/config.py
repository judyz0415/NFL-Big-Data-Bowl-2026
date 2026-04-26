"""Hyperparameters and runtime configuration for the physics-informed transformer."""

from dataclasses import dataclass, field
from typing import List

import torch


# ---------------------------------------------------------------------------
# Continuous + categorical feature definitions
# ---------------------------------------------------------------------------

CONTINUOUS_COLUMNS: List[str] = [
    "x", "y", "s", "a",
    "dir", "o",
    "ball_land_x", "ball_land_y",
    "absolute_yardline_number",
]
"""Numeric per-frame columns standardized via z-score using training statistics.

Note: ``dir`` and ``o`` are stored here for normalization-stats computation, but
are converted to (sin, cos) pairs before being fed to the model.
"""

CATEGORICAL_COLUMNS: List[str] = [
    "player_position",
    "player_side",
    "player_role",
    "play_direction",
]
"""Per-player / per-play categorical columns turned into learned embeddings."""


@dataclass
class TrainConfig:
    """Container for hyperparameters and runtime knobs."""

    # I/O ------------------------------------------------------------------
    data_dir: str = "data/raw"
    """Directory containing input_2023_w*.csv and output_2023_w*.csv files."""

    checkpoint_path: str = "checkpoints/bdb2026_phys_transformer.pt"
    """Where to save the trained model + vocabularies + normalization stats."""

    # Splits ---------------------------------------------------------------
    train_weeks: List[int] = field(default_factory=lambda: list(range(1, 14)))
    """Weeks 1-13 are used for training."""

    val_weeks: List[int] = field(default_factory=lambda: list(range(14, 19)))
    """Weeks 14-18 form the held-out validation set (mirrors Kaggle protocol)."""

    # Optimization ---------------------------------------------------------
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 30

    # Architecture ---------------------------------------------------------
    hidden_dim: int = 96
    num_layers: int = 1
    dropout: float = 0.1
    embedding_dim: int = 6
    num_heads: int = 2
    max_time: int = 300
    """Maximum number of pre-throw frames the time embedding can index."""

    # Physics --------------------------------------------------------------
    lambda_phys: float = 0.003
    """Weight on the physics regularizer added to the masked RMSE loss."""

    max_speed: float = 12.0
    """Soft upper bound on per-frame speed (yards/s) used by the regularizer."""

    max_acc: float = 8.0
    """Soft upper bound on per-frame acceleration (yards/s^2)."""

    # Repro / hardware -----------------------------------------------------
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    """DataLoader worker count. Keep 0 on Colab; bump up on local multi-core."""
