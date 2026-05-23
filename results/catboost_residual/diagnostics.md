# CatBoost Residual Diagnostics

## What was tested

The model is not a direct RTO forecaster. It learns a residual over the current heuristic baseline:

`target = log(actual March RTO / heuristic baseline)`

The current baseline already includes:

1. February RTO.
2. Global March / February seasonal factor.
3. Conservative momentum correction.

The residual model is validated on March 2024 with a 5-fold cross-sectional split by store ID.

## Main finding

Residual CatBoost is useful because it learns where the heuristic baseline is systematically biased.

Top feature-importance signals:

| Feature | Importance |
|---|---:|
| `momentum_raw` | 28.71 |
| `volatility_rel3` | 18.79 |
| `volatility_rel6` | 9.86 |
| `diff1` | 8.62 |
| `lag1_to_lag12` | 4.72 |
| `momentum_clip` | 4.71 |

The gain is mainly about recent dynamics, volatility, and YoY positioning, not replacing the baseline level.

## OOF alpha / clip comparison

| Variant | OOF MAPE | Gain vs heuristic | Mean abs correction | Max correction |
|---|---:|---:|---:|---:|
| `alpha=0.50`, no clip | 3.6498 | 0.4782 | 1.04% | 1.55x |
| `alpha=0.60`, no clip | 3.5773 | 0.5506 | 1.26% | 1.69x |
| `alpha=0.70`, no clip | 3.5142 | 0.6138 | 1.48% | 1.84x |
| `alpha=0.80`, no clip | 3.4612 | 0.6667 | 1.71% | 2.01x |
| `alpha=0.80`, clip +/-10% | 3.5960 | 0.5320 | 1.36% | 1.10x |

Cross-sectional OOF prefers stronger residuals, but the tail risk grows quickly for `alpha >= 0.7`.

## Segment check

For `alpha=0.60`, the residual improves every checked quintile:

- baseline RTO level;
- raw momentum;
- three-month volatility;
- current February level versus last March.

The largest gains appear where the pure heuristic is weakest:

- extreme momentum buckets;
- high volatility stores;
- stores far from their prior-year March scale.

## Next submission choice

The next moderate candidate is:

`submissions/test_next.csv`

It uses:

`alpha=0.60`, no clip.

Rationale:

- `alpha=0.50` already improved public score;
- `alpha=0.60` improves OOF across checked segments;
- `alpha=0.60` is less aggressive than the OOF-best `alpha=0.80`, whose max local correction exceeds `2x`.

The stronger `alpha=0.70` or `alpha=0.80` branch is worth testing only after seeing how `alpha=0.60` transfers to leaderboard.
