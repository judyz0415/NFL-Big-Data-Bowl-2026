"""Loss functions for the physics-informed transformer.

* :func:`masked_rmse_loss` — the metric used for grading; matches Kaggle's
  variant of RMSE over all valid (x, y) frames.
* :func:`physics_regularizer` — soft penalty that discourages predictions whose
  inferred speed exceeds ``max_speed`` (yards/s) or whose inferred acceleration
  exceeds ``max_acc`` (yards/s^2). Both bounds are calibrated to roughly match
  elite NFL player ceilings.
"""

from __future__ import annotations

import torch


def masked_rmse_loss(
    pred: torch.Tensor,    # [B, T, 2]
    target: torch.Tensor,  # [B, T, 2]
    mask: torch.Tensor,    # [B, T]   (1 = valid, 0 = pad)
) -> torch.Tensor:
    """RMSE over valid frames, matching the Big Data Bowl 2026 scoring formula."""
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.unsqueeze(-1)
    mse = diff2.sum() / (mask.sum() * 2.0 + 1e-8)
    return torch.sqrt(mse + 1e-8)


def physics_regularizer(
    pred_xy: torch.Tensor,   # [B, T, 2]
    tout_mask: torch.Tensor, # [B, T]
    dt: float = 1.0,
    max_speed: float = 12.0,
    max_acc: float = 8.0,
) -> torch.Tensor:
    """Penalize predicted trajectories that exceed plausible kinematic limits.

    We compute finite-difference velocities and accelerations on the predictions
    and penalize the squared magnitude of any *excess* over the configured
    thresholds, masked so padded positions contribute zero loss.
    """
    B, T, _ = pred_xy.shape
    if T < 3:
        return torch.tensor(0.0, device=pred_xy.device)

    # First-order finite difference: velocity
    v = (pred_xy[:, 1:] - pred_xy[:, :-1]) / dt           # [B, T-1, 2]
    mask_v = tout_mask[:, 1:] * tout_mask[:, :-1]         # [B, T-1]

    speed = torch.linalg.norm(v, dim=-1)                  # [B, T-1]
    speed_excess = torch.clamp(speed - max_speed, min=0.0)
    speed_pen = (speed_excess ** 2 * mask_v).sum() / (mask_v.sum() + 1e-8)

    # Second-order finite difference: acceleration
    a = (v[:, 1:] - v[:, :-1]) / dt                       # [B, T-2, 2]
    mask_a = mask_v[:, 1:] * mask_v[:, :-1]               # [B, T-2]

    acc_mag = torch.linalg.norm(a, dim=-1)                # [B, T-2]
    acc_excess = torch.clamp(acc_mag - max_acc, min=0.0)
    acc_pen = (acc_excess ** 2 * mask_a).sum() / (mask_a.sum() + 1e-8)

    return speed_pen + acc_pen
