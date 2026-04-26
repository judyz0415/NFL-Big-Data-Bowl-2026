"""Data loading, sample construction, and PyTorch Dataset for the BDB 2026 task.

The pipeline produces one sample per (game_id, play_id, nfl_id) target player:

* ``cont``     : (T_in, 11) standardized continuous features per pre-throw frame
* ``cat``      : (4,) integer-coded categorical attributes for the player/play
* ``y_future`` : (T_out, 2) ground-truth future (x, y) coordinates
* ``num_out``  : declared number of future frames for the play

Variable-length sequences are padded inside :func:`collate_fn` with masks so the
transformer can ignore padding positions.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def angle_to_sincos(deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert degree angles to (sin, cos) pairs to remove the 0/360 wrap discontinuity."""
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)


def get_week_num(path: str) -> int:
    """Extract the week number from filenames like ``input_2023_w14.csv`` → 14."""
    base = os.path.basename(path)
    return int(base.split("_w")[1].split(".")[0])


def load_week_files(data_dir: str, prefix: str) -> List[str]:
    """Return sorted weekly CSV paths matching ``{prefix}_2023_w*.csv``."""
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_2023_w*.csv")))
    if not files:
        raise FileNotFoundError(f"No {prefix} files found in {data_dir}")
    return files


def build_vocab(series: pd.Series) -> Dict[str, int]:
    """Map each unique categorical value to a 1-indexed integer (0 reserved for unknown)."""
    uniq = sorted(series.dropna().unique().tolist())
    return {v: i + 1 for i, v in enumerate(uniq)}


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def make_samples(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    vocabs: Dict[str, Dict],
    norm_stats: Dict[str, Dict[str, float]],
) -> List[Dict]:
    """Build per-target-player samples from a week's input/output CSV pair.

    Only rows with ``player_to_predict == True`` become targets, matching the
    Kaggle scoring protocol.
    """
    targets = input_df[input_df["player_to_predict"] == True].copy()  # noqa: E712
    if len(targets) == 0:
        return []

    targets = targets.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    output_df = output_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

    in_groups = targets.groupby(["game_id", "play_id", "nfl_id"])
    out_groups = output_df.groupby(["game_id", "play_id", "nfl_id"])

    samples: List[Dict] = []
    for key, g_in in in_groups:
        if key not in out_groups.groups:
            continue
        g_out = out_groups.get_group(key)

        def zscore(arr: np.ndarray, key: str) -> np.ndarray:
            return (arr - norm_stats["mean"][key]) / norm_stats["std"][key]

        # Standardized continuous features
        x = zscore(g_in["x"].values, "x")
        y = zscore(g_in["y"].values, "y")
        s = zscore(g_in["s"].values, "s")
        a = zscore(g_in["a"].values, "a")
        ball_x = zscore(g_in["ball_land_x"].values, "ball_land_x")
        ball_y = zscore(g_in["ball_land_y"].values, "ball_land_y")
        yardline = zscore(g_in["absolute_yardline_number"].values, "absolute_yardline_number")

        # Angles → (sin, cos) — never z-scored
        dir_sin, dir_cos = angle_to_sincos(g_in["dir"].values)
        o_sin, o_cos = angle_to_sincos(g_in["o"].values)

        cont = np.stack(
            [x, y, s, a, dir_sin, dir_cos, o_sin, o_cos, ball_x, ball_y, yardline],
            axis=1,
        ).astype(np.float32)

        # Categorical attributes (constant per player/play → take first row)
        def map_cat(col: str, vocab: Dict[str, int]) -> int:
            val = g_in[col].iloc[0] if col in g_in else None
            return vocab.get(val, 0)

        cat = np.array(
            [map_cat(col, vocabs[col]) for col in CATEGORICAL_COLUMNS],
            dtype=np.int64,
        )

        # Future trajectory (truncated to declared length if necessary)
        y_future = g_out[["x", "y"]].values.astype(np.float32)
        num_out = int(g_in["num_frames_output"].iloc[0])
        if len(y_future) != num_out:
            y_future = y_future[:num_out]

        samples.append({
            "cont": cont,
            "cat": cat,
            "y_future": y_future,
            "num_out": num_out,
        })

    return samples


def compute_norm_stats(train_in_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute per-column mean/std on the training inputs only (avoid leakage)."""
    means = train_in_df[CONTINUOUS_COLUMNS].mean()
    stds = train_in_df[CONTINUOUS_COLUMNS].std().replace(0.0, 1.0)
    return {"mean": means.to_dict(), "std": stds.to_dict()}


def build_vocabs(train_in_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Build vocabs and per-column embedding sizes (size = max_index + 1)."""
    vocabs = {col: build_vocab(train_in_df[col]) for col in CATEGORICAL_COLUMNS}
    cat_sizes = {col: len(v) + 1 for col, v in vocabs.items()}
    return vocabs, cat_sizes


# ---------------------------------------------------------------------------
# PyTorch Dataset + collate
# ---------------------------------------------------------------------------

class BDBDataset(Dataset):
    """Thin wrapper around a list of pre-built sample dicts."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        return (
            torch.from_numpy(s["cont"]),
            torch.from_numpy(s["cat"]),
            torch.from_numpy(s["y_future"]),
            s["num_out"],
        )


def collate_fn(batch):
    """Pad variable T_in and T_out across a batch and return padding masks."""
    cont_list, cat_list, y_list, _ = zip(*batch)

    max_t_in = max(c.shape[0] for c in cont_list)
    max_t_out = max(y.shape[0] for y in y_list)

    cont_dim = cont_list[0].shape[1]
    cat_dim = cat_list[0].shape[0]
    B = len(batch)

    cont_pad = torch.zeros(B, max_t_in, cont_dim)
    tin_mask = torch.zeros(B, max_t_in)

    y_pad = torch.zeros(B, max_t_out, 2)
    tout_mask = torch.zeros(B, max_t_out)

    cat_tensor = torch.zeros(B, cat_dim, dtype=torch.long)

    for i, (c, cat, y) in enumerate(zip(cont_list, cat_list, y_list)):
        cont_pad[i, : c.shape[0]] = c
        tin_mask[i, : c.shape[0]] = 1.0
        y_pad[i, : y.shape[0]] = y
        tout_mask[i, : y.shape[0]] = 1.0
        cat_tensor[i] = cat

    return cont_pad, tin_mask, cat_tensor, y_pad, tout_mask
