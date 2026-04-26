# Repo settings to apply on GitHub (via the web UI)

After creating the repo at https://github.com/judyz0415/nfl-big-data-bowl-2026 and pushing, configure these settings.

## About section (gear icon next to "About")

**Description**:
```
Physics-informed transformer for NFL Big Data Bowl 2026 player-trajectory prediction. Test RMSE 1.172 yards (best of 6 architectures benchmarked).
```

**Website**: <https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction>

**Topics** (paste, space-separated):
```
nfl big-data-bowl big-data-bowl-2026 sports-analytics nfl-analytics
trajectory-prediction transformer physics-informed deep-learning
sequence-to-sequence pytorch kaggle player-tracking
```

## Pin this repo

https://github.com/judyz0415 → "Customize your pins" → check this repo. This is
likely your strongest single project — make sure it's pinned.

## Social preview (optional but recommended)

Settings → General → Social preview → upload a custom image. A clean trajectory
plot from the validation set works well; this is what shows when the link is
shared on LinkedIn / X.

## Verify before going public

- [ ] No competition CSVs committed (`git ls-files | grep -i 'input_2023\|output_2023'` should be empty)
- [ ] No model checkpoints committed (`git ls-files | grep -E '\.pt$|\.ckpt$'` should be empty)
- [ ] `reports/HODL_Final_Report.pdf` is in the repo (your team's full writeup)
- [ ] `notebooks/_original_colab_notebook.ipynb` preserved for transparency
- [ ] README "Team" section accurately credits all four HODL members
