"""Training entry point.

Usage::

    python -m bdb2026.train --data-dir data/raw --epochs 30

The CLI mirrors the dataclass fields in :class:`bdb2026.config.TrainConfig`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import (
    BDBDataset,
    build_vocabs,
    collate_fn,
    compute_norm_stats,
    get_week_num,
    load_week_files,
    make_samples,
)
from .losses import masked_rmse_loss, physics_regularizer
from .model import PhysicsInformedTrajectoryTransformer


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Device: {cfg.device}")

    # ------------------------------------------------------------------
    # Resolve weekly files and split by week index (1-13 train, 14-18 val).
    # ------------------------------------------------------------------
    in_files = load_week_files(cfg.data_dir, "input")
    out_files = load_week_files(cfg.data_dir, "output")

    train_pairs, val_pairs = [], []
    for fin, fout in zip(in_files, out_files):
        w = get_week_num(fin)
        if w in cfg.train_weeks:
            train_pairs.append((fin, fout))
        elif w in cfg.val_weeks:
            val_pairs.append((fin, fout))

    print("Train weeks:", [get_week_num(p[0]) for p in train_pairs])
    print("Val weeks  :", [get_week_num(p[0]) for p in val_pairs])

    # ------------------------------------------------------------------
    # Compute normalization stats and build vocabularies on TRAIN ONLY.
    # ------------------------------------------------------------------
    train_in_df = pd.concat(
        [pd.read_csv(fin) for fin, _ in train_pairs], ignore_index=True
    )
    norm_stats = compute_norm_stats(train_in_df)
    vocabs, cat_sizes = build_vocabs(train_in_df)

    # ------------------------------------------------------------------
    # Build per-target-player samples.
    # ------------------------------------------------------------------
    train_samples, val_samples = [], []
    for fin, fout in train_pairs:
        train_samples += make_samples(pd.read_csv(fin), pd.read_csv(fout), vocabs, norm_stats)
    for fin, fout in val_pairs:
        val_samples += make_samples(pd.read_csv(fin), pd.read_csv(fout), vocabs, norm_stats)

    print("Train samples:", len(train_samples))
    print("Val samples  :", len(val_samples))

    train_loader = DataLoader(
        BDBDataset(train_samples),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        BDBDataset(val_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
    )

    # ------------------------------------------------------------------
    # Model + optimizer.
    # ------------------------------------------------------------------
    cont_dim = train_samples[0]["cont"].shape[1]   # 11
    model = PhysicsInformedTrajectoryTransformer(cont_dim, cat_sizes, cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    for ep in range(1, cfg.epochs + 1):
        # ---- train pass --------------------------------------------------
        model.train()
        tr_loss = tr_data = tr_phys = 0.0
        for cont, tin_mask, cat, y, tout_mask in train_loader:
            cont = cont.to(cfg.device); tin_mask = tin_mask.to(cfg.device)
            cat = cat.to(cfg.device); y = y.to(cfg.device); tout_mask = tout_mask.to(cfg.device)

            pred = model(cont, tin_mask, cat, y.shape[1])
            data_loss = masked_rmse_loss(pred, y, tout_mask)
            phys_loss = physics_regularizer(
                pred, tout_mask, max_speed=cfg.max_speed, max_acc=cfg.max_acc
            )
            loss = data_loss + cfg.lambda_phys * phys_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item(); tr_data += data_loss.item(); tr_phys += phys_loss.item()

        n = len(train_loader)
        tr_loss /= n; tr_data /= n; tr_phys /= n

        # ---- validation pass --------------------------------------------
        model.eval()
        va_loss = va_data = va_phys = 0.0
        with torch.no_grad():
            for cont, tin_mask, cat, y, tout_mask in val_loader:
                cont = cont.to(cfg.device); tin_mask = tin_mask.to(cfg.device)
                cat = cat.to(cfg.device); y = y.to(cfg.device); tout_mask = tout_mask.to(cfg.device)

                pred = model(cont, tin_mask, cat, y.shape[1])
                data_loss = masked_rmse_loss(pred, y, tout_mask)
                phys_loss = physics_regularizer(
                    pred, tout_mask, max_speed=cfg.max_speed, max_acc=cfg.max_acc
                )
                loss = data_loss + cfg.lambda_phys * phys_loss

                va_loss += loss.item(); va_data += data_loss.item(); va_phys += phys_loss.item()

        nv = len(val_loader)
        va_loss /= nv; va_data /= nv; va_phys /= nv

        print(
            f"Epoch {ep:02d} | "
            f"train RMSE={tr_data:.4f} (phys={tr_phys:.4f}) | "
            f"val RMSE={va_data:.4f} (phys={va_phys:.4f})"
        )

        # ---- best checkpoint --------------------------------------------
        if va_data < best_val:
            best_val = va_data
            os.makedirs(os.path.dirname(cfg.checkpoint_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocabs": vocabs,
                    "norm_stats": norm_stats,
                    "config": asdict(cfg),
                    "epoch": ep,
                    "val_rmse": va_data,
                },
                cfg.checkpoint_path,
            )

    print(f"Best validation RMSE: {best_val:.4f}")
    print(f"Saved best checkpoint to {cfg.checkpoint_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train physics-informed transformer for BDB 2026.")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--checkpoint-path", default="checkpoints/bdb2026_phys_transformer.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--lambda-phys", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lambda_phys=args.lambda_phys,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()
