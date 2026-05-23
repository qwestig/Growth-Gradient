# Current Best Submission

## Best public result

- File: `submissions/test_best_gru_momentum_bias_8965.csv`
- Public score: `89.65`
- Previous GRU-only file: `submissions/test_gru_alpha070.csv`
- Previous GRU-only public score: `89.52`

## What was rejected

- Global scale calibration around the best GRU forecast: dropped to about `89.28`.
- GRU + CatBoost blend: `89.49`, worse than the standalone GRU.
- Multi-month GRU residual: worse than the March-specific GRU on local March-2024 diagnostics.
- Transformer residual: public score was lower than the GRU branch.
- DLinear residual inspired by the expert TS/DL notebook: local MAPE `3.4779`, worse than current GRU `3.4417`.
- GRU seed ensemble / higher alpha variants: did not improve public score.
- Group-stat-heavy CatBoost feature expansion: did not transfer to public score.
- Weaker momentum correction `k500_b075`: improved to `89.63`, but was superseded by `k500_b100`.
- Softer momentum correction `k1000_b075`: `89.62`, worse than `k500_b100`.

## Current interpretation

The best public signal comes from the March-specific GRU residual correction over the handcrafted seasonal heuristic, followed by a segment-level correction for `momentum_raw_bin`. The failed scale and blend tests suggest the gap is not a global under/over-prediction problem. The useful residual structure is tied to February momentum: stores with unusually weak February need a small upward correction, while stores with unusually strong February need a small downward correction.
