# Segment Bias Correction

## Hypothesis

The best GRU forecast may still have systematic bias in specific store segments. Since global scaling failed on public, the next useful check is not a universal multiplier but a segment-specific multiplier.

## Method

- Start from OOF March-2024 predictions of `GRU alpha=0.70`.
- Compute log error: `log(actual / pred_gru)`.
- For each candidate segment, estimate group-level mean log error using only other folds.
- Shrink group corrections toward zero: `mean_log_error * n / (n + k)`.
- Clip corrections to avoid aggressive movement.
- Apply correction multiplicatively to the GRU forecast.

## Best Local Result

- Base GRU MAPE: `3.4417`.
- Best segment: `momentum_raw_bin`.
- Parameters: `k=500`, `beta=0.75`, `clip=0.02`.
- Corrected local MAPE: `3.4113`.
- Local gain: `0.0304` MAPE points.

## Interpretation

`momentum_raw = lag1 / mean(lag1, lag2, lag3)` measures how unusual February is compared with the recent three-month level.

Learned correction factors:

| Segment | Meaning | Factor |
|---|---:|---:|
| `q0` | weakest February momentum | `1.0042` |
| `q1` | low-medium momentum | `0.9991` |
| `q2` | middle momentum | `0.9960` |
| `q3` | high-medium momentum | `0.9953` |
| `q4` | strongest February momentum | `0.9924` |

This says: when February is weak relative to the last three months, the GRU slightly underpredicts March; when February is unusually strong, the GRU slightly overpredicts March.

## Candidate

- `submissions/test_gru_segment_momentum_k500_b075_c020.csv`

This is a stronger local signal than previous GRU/CatBoost or GRU/DLinear blends, but still needs public validation because March-2025 has a different momentum-bin distribution than March-2024.

## Local Sweep Around Public Winner

After `k500_b075_c020` improved public score to `89.63`, the next sweep stayed close to this correction:

| File | Meaning |
|---|---|
| `test_gru_segment_momentum_k500_b050_c020.csv` | weaker correction |
| `test_gru_segment_momentum_k500_b100_c020.csv` | stronger correction |
| `test_gru_segment_momentum_k250_b075_c020.csv` | less shrinkage, slightly stronger group factors |
| `test_gru_segment_momentum_k1000_b075_c020.csv` | more shrinkage, slightly softer group factors |

Public results:

| Variant | Public score |
|---|---:|
| `k500_b075_c020` | `89.63` |
| `k500_b100_c020` | `89.65` |
| `k1000_b075_c020` | `89.62` |

The current best file is now:

`submissions/test_best_gru_momentum_bias_8965.csv`
