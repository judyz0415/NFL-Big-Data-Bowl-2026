# NFL Big Data Bowl 2026 — Physics-Informed Transformer

> Predicting NFL player trajectories at 0.1-second resolution during the airborne portion of a pass play. **Test RMSE: 1.172 yards** — the best of six architectures we benchmarked.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Big Data Bowl 2026](https://img.shields.io/badge/Kaggle-NFL%20Big%20Data%20Bowl%202026-20BEFF.svg)](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)

## TL;DR

The Big Data Bowl 2026 task asks competitors to forecast every player's
two-dimensional position in 0.1-second steps after the ball is released. We
benchmarked six modeling approaches on team **HODL** and the **Physics-Informed
Transformer** in this repo achieved the best score:

| Model | Test RMSE (yards) |
| --- | --- |
| XGBoost (baseline) | 2.406 |
| Physics-Informed ResNet | 1.290 |
| LSTM | 1.456 |
| Transformer | 1.249 |
| Physics-Informed LSTM | 1.535 |
| **Physics-Informed Transformer** *(this repo)* | **1.172** |

The full report with all models is in [`reports/HODL_Final_Report.pdf`](reports/HODL_Final_Report.pdf).

## What makes this model "physics-informed"

The decoder predicts **accelerations**, not raw coordinates. Velocity and
position are then integrated explicitly inside the network at each future
timestep:

```
v_{t+1}  = v_t  + a_t      * dt        # Euler integrate velocity
xy_{t+1} = xy_t + v_{t+1}  * dt        # Euler integrate position
```

with a **learnable timestep** `dt` (parameterized in log-space so it stays
positive). This means the network can never produce trajectories that violate
basic kinematic continuity — even before any loss-function penalty kicks in.

A soft regularizer adds a second layer of physical realism by penalizing
predicted speeds above 12 yd/s and accelerations above 8 yd/s², keeping
forecasts inside the envelope of plausible NFL athletic limits.

## Architecture

```
              ┌─────────────────────────────────────────────┐
              │  Pre-throw frames (variable T_in)           │
              │   x, y, s, a, dir, o, ball_land_x/y, yardline│
              └─────────────────────────────────────────────┘
                              │
                cont features │   ┌──────────────┐
                              ├──▶│  z-score     │
                              │   │  normalize   │
              ┌──────────────┐│   └──────┬───────┘
              │ player_pos    │          │
              │ player_side   ├──┐  ┌────▼───────┐
              │ player_role   │  └─▶│ embeddings │
              │ play_direction│     └────┬───────┘
              └───────────────┘          │  + time embedding
                                ┌────────▼─────────┐
                                │  Transformer enc │
                                │  (attention)     │
                                └────────┬─────────┘
                                         │
                                  enc_summary  ────────────┐
                                         │                 │
                                  ┌──────▼──────┐          │
                          xy_t ─▶│  GRUCell    │◀─ cat_emb │
                          v_t  ─▶│  (decoder)  │           │
                                  └──────┬──────┘          │
                                         │                 │
                                  ┌──────▼──────┐          │
                                  │  acc head   │ → a_t    │
                                  └─────────────┘          │
                                         │                 │
                              integrate v, xy ──── repeat for T_out steps
```

Encoder defaults: 1 layer, 2 heads, `d_model=96`, dropout 0.1.
Decoder is a single GRUCell with the same hidden width.

## Repository structure

```
nfl-big-data-bowl-2026/
├── src/bdb2026/
│   ├── __init__.py
│   ├── config.py         # TrainConfig dataclass + feature column lists
│   ├── data.py           # Loading, normalization, sample building, padding
│   ├── model.py          # PhysicsInformedTrajectoryTransformer
│   ├── losses.py         # Masked RMSE + physics regularizer
│   └── train.py          # Training loop + CLI entry point
├── notebooks/
│   ├── physics_informed_transformer.ipynb   # Reviewer-friendly walk-through
│   └── _original_colab_notebook.ipynb       # Untouched original for transparency
├── reports/
│   └── HODL_Final_Report.pdf                # Full project report (all 6 models)
├── configs/                                  # (reserved for YAML configs)
├── results/                                  # (training curves / figures)
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── .gitignore
└── README.md
```

## Quick start

```bash
git clone https://github.com/judyz0415/nfl-big-data-bowl-2026.git
cd nfl-big-data-bowl-2026

python -m venv .venv
source .venv/bin/activate
pip install -e .

# Drop the official competition CSVs into data/raw/
#   data/raw/input_2023_w01.csv ... input_2023_w18.csv
#   data/raw/output_2023_w01.csv ... output_2023_w18.csv

python -m bdb2026.train \
    --data-dir data/raw \
    --epochs 30 \
    --batch-size 128
```

The best checkpoint is saved to `checkpoints/bdb2026_phys_transformer.pt` along
with the trained vocabularies, normalization statistics, and the full
`TrainConfig` used.

### Reading the code

If you only have time to read one file, read `src/bdb2026/model.py` — it is
under 130 lines and contains the entire architecture, including the explicit
Euler integration in `forward()`.

After that, `src/bdb2026/train.py` ties data, model, and losses together with a
CLI; `src/bdb2026/data.py` shows the per-target-player sample construction and
the padding-aware collate function.

## Data

Competition page: <https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction>

The 2023-season NFL Next Gen Stats tracking data is provided by the NFL on
Kaggle. We split by week (1-13 train, 14-18 validation) to mirror the
competition's held-out scoring protocol. CSVs are not committed here — see
`.gitignore`.

## Team

This was the final project for our team **HODL** in our deep learning course
(Fall 2025), submitted to the NFL Big Data Bowl 2026:

| Member | Primary contribution |
| --- | --- |
| Evan Hoch | (see `reports/HODL_Final_Report.pdf`) |
| Yue Ran Kang | (see `reports/HODL_Final_Report.pdf`) |
| Colton Mikolajczyk | (see `reports/HODL_Final_Report.pdf`) |
| **Judy Zhu** | **Physics-Informed Transformer** (this repo's winning model) |

The final report covers all six models the team built (XGBoost, Physics-Informed
ResNet, LSTM, Transformer, Physics-Informed LSTM, Physics-Informed Transformer).
This repository isolates the architecture I personally built and tuned.

## License

Released under the [MIT License](LICENSE).

## Contact

Judy Zhu — judy.zhu6052@gmail.com — [@judyz0415](https://github.com/judyz0415)
