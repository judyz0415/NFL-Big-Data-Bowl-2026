"""Physics-informed transformer for NFL player-trajectory forecasting.

Architecture
------------
* **Encoder.** A standard ``nn.TransformerEncoder`` over per-frame inputs
  formed from standardized continuous features concatenated with learned
  categorical embeddings, plus a learned time embedding.

* **Decoder.** A ``GRUCell`` that consumes ``[xy_t, v_t, cat_embed, enc_summary]``
  at each future timestep and predicts an *acceleration* ``a_t``. Velocity and
  position are then integrated explicitly via Euler steps:

      v_{t+1}  = v_t  + a_t  * dt
      xy_{t+1} = xy_t + v_{t+1} * dt

  The integration timestep ``dt`` is itself a learnable scalar (parameterized
  in log-space to keep it positive).

This "predict accelerations and integrate" design is the *physics-informed*
piece: the model can never produce trajectories that violate basic kinematic
continuity, regardless of what the network outputs.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .config import TrainConfig


class PhysicsInformedTrajectoryTransformer(nn.Module):
    """Transformer encoder + acceleration-predicting GRU decoder."""

    def __init__(self, cont_dim: int, cat_sizes: Dict[str, int], cfg: TrainConfig):
        super().__init__()

        # ---- categorical embeddings ----------------------------------------
        self.emb_pos = nn.Embedding(cat_sizes["player_position"], cfg.embedding_dim)
        self.emb_side = nn.Embedding(cat_sizes["player_side"], cfg.embedding_dim)
        self.emb_role = nn.Embedding(cat_sizes["player_role"], cfg.embedding_dim)
        self.emb_dir = nn.Embedding(cat_sizes["play_direction"], cfg.embedding_dim)
        self.cat_dim = cfg.embedding_dim * 4

        # ---- input projection + time embedding -----------------------------
        self.input_dim = cont_dim + self.cat_dim
        self.input_proj = nn.Linear(self.input_dim, cfg.hidden_dim)

        self.max_time = cfg.max_time
        self.time_embed = nn.Embedding(cfg.max_time, cfg.hidden_dim)

        # ---- temporal transformer encoder ----------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # ---- physics-aware decoder cell ------------------------------------
        # Decoder input concatenates: [xy(2), v(2), cat_embed(cat_dim), enc_summary(hidden)]
        self.dec_in_dim = 2 + 2 + self.cat_dim + cfg.hidden_dim
        self.decoder_cell = nn.GRUCell(self.dec_in_dim, cfg.hidden_dim)

        self.acc_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 2),
        )

        # Learnable integration timestep (parameterized in log-space → positive).
        self.dt_log = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def forward(
        self,
        cont: torch.Tensor,        # [B, T_in, cont_dim]
        tin_mask: torch.Tensor,    # [B, T_in]    (1 = valid, 0 = pad)
        cat: torch.Tensor,         # [B, 4]
        max_t_out: int,
    ) -> torch.Tensor:
        """Auto-regressively roll out ``max_t_out`` future (x, y) positions."""
        B, T_in, _ = cont.shape
        device = cont.device

        # ---- categorical embedding (per player) ----------------------------
        e = torch.cat(
            [
                self.emb_pos(cat[:, 0]),
                self.emb_side(cat[:, 1]),
                self.emb_role(cat[:, 2]),
                self.emb_dir(cat[:, 3]),
            ],
            dim=-1,
        )  # [B, cat_dim]

        # Broadcast cat embedding to every input frame and concat with continuous features.
        e_in = e.unsqueeze(1).expand(B, T_in, self.cat_dim)
        enc_in = torch.cat([cont, e_in], dim=-1)
        x = self.input_proj(enc_in)

        # Add learned time-step embedding.
        time_idx = torch.arange(T_in, device=device).unsqueeze(0).expand(B, T_in)
        x = x + self.time_embed(time_idx)

        # PyTorch transformer expects True = pad.
        src_key_padding_mask = tin_mask == 0
        x_enc = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # ---- summarize encoder output at the last *valid* frame ------------
        lengths = tin_mask.sum(dim=1).long().clamp(min=1)
        idx = torch.arange(B, device=device)
        enc_summary = x_enc[idx, lengths - 1]  # [B, hidden]

        # ---- physics initial state from last observed frame ----------------
        last_xy = cont[idx, lengths - 1, 0:2]                     # [B, 2]
        # cont columns: [x, y, s, a, dir_sin, dir_cos, o_sin, o_cos, ball_x, ball_y, yardline]
        s_last = cont[idx, lengths - 1, 2]
        dir_sin_last = cont[idx, lengths - 1, 4]
        dir_cos_last = cont[idx, lengths - 1, 5]
        v_last = torch.stack(
            [s_last * dir_cos_last, s_last * dir_sin_last], dim=-1
        )  # [B, 2]

        h = enc_summary
        dt = torch.exp(self.dt_log)

        xy_t, v_t = last_xy, v_last
        cat_rep = e

        preds = []
        for _ in range(max_t_out):
            dec_in = torch.cat([xy_t, v_t, cat_rep, enc_summary], dim=-1)
            h = self.decoder_cell(dec_in, h)
            a_t = self.acc_head(h)             # predicted acceleration
            v_t = v_t + a_t * dt               # integrate velocity
            xy_t = xy_t + v_t * dt             # integrate position
            preds.append(xy_t.unsqueeze(1))

        return torch.cat(preds, dim=1)         # [B, T_out, 2]
