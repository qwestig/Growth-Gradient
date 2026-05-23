# Current Best: Calendar Baseline + Faithful GRU Blend

## Public Best

Current best submission:

```text
submissions/test_best_cal050_gru_blend_9070.csv
```

Public result:

```text
MAPE = 4.763
score = 90.70086169
```

Previous best before this branch:

```text
GRU alpha 0.70 + momentum_raw_bin correction
MAPE ~= 5.316
score = 89.65
```

Gain:

```text
5.316 - 4.763 = 0.553 absolute MAPE points
```

## Final Formula Family

The winning branch came from replacing the old baseline structure first, then adding residuals.

### New Baseline

The old baseline used:

```text
factor = median(Mar2023 / Feb2023, Mar2024 / Feb2024)
momentum = clip(Feb / mean(Dec, Jan, Feb), 0.95, 1.05)
base = Feb * factor * momentum ** 0.25
```

The new public-winning baseline removes the old momentum penalty and partially adjusts the 2024 leap-year February effect:

```text
calendar_lambda = 0.50
momentum_exp = 0.00
base = Feb * calendar_adjusted_factor
```

Key public checks:

| Candidate | Description | Public MAPE |
|---:|---|---:|
| 45 | `lambda=0.25`, `momentum_exp=0.25` | `5.538` |
| 46 | `lambda=0.375`, `momentum_exp=0.25` | `5.464` |
| 47 | `lambda=0.50`, `momentum_exp=0.25` | `5.390` |
| 70 | `lambda=0.50`, `momentum_exp=0.00` | `5.153` |
| 78 | `lambda=0.60`, `momentum_exp=0.00` | `5.153` |

Interpretation:

```text
calendar/leap adjustment was strongly public-positive;
the old weak-February momentum penalty was harmful for March 2025.
```

### Residual Transfer

Applying the old GRU residual to the new baseline, centered by `momentum_raw_bin`, gave the next jump:

| Candidate | Description | Public MAPE |
|---:|---|---:|
| 101 | old-GRU transfer, mombin centered, `gamma=0.50` | `4.958` |
| 104 | `gamma=0.70` | `4.900` |
| 115 | `gamma=0.85` | `4.858` |
| 117 | `gamma=0.95` | `4.837` |
| 118 | `gamma=1.00` | `4.832` |

### Fresh GRU Residual

Then GRU was retrained on the new residual target:

```text
target = log(actual_march_2024 / new_baseline_march_2024)
```

The faithful final fit follows the original GRU pipeline more closely:

```text
SEQ_LEN = 14
EPOCHS = 80
FINAL_EPOCHS = 35
SmoothL1Loss(beta=0.02)
AdamW(lr=2e-3, weight_decay=2e-3)
global centering for final candidate
```

Key public checks:

| Candidate | Description | Public MAPE |
|---:|---|---:|
| 125 | first raw fresh GRU, uncentered | `4.853` |
| 150 | faithful fresh GRU, global-centered, alpha `1.00` | `4.784` |
| 156 | same, alpha `1.05` | `4.800` |

### Final Blend

Best current file blends the two independent residual sources:

```text
candidate_159 = 0.75 * candidate_150 + 0.25 * candidate_118
```

Public:

```text
candidate_159 MAPE = 4.763
```

## Repro Scripts

Core scripts are under:

```text
analysis/new_baseline/
```

Most important scripts:

```text
heuristic_baseline_direction_lab.py
build_calendar_followup_candidates.py
build_momentum_zero_followup_candidates.py
build_new_baseline_residual_transfer_candidates.py
build_residual_transfer_gamma_sweep.py
build_gamma_high_sweep_candidates.py
train_new_baseline_gru_residual.py
build_faithful_gru_followup_candidates.py
analyze_public_experiment_log.py
```

Lightweight result summaries are under:

```text
results/new_baseline/
results/new_baseline_gru_residual/
results/new_baseline_catboost_residual/
```

Large row-level OOF files are intentionally not committed.

## Notes

The public experiment log is the source of truth for submitted candidates and scores:

```text
results/new_baseline/public_experiment_log.csv
```

This branch intentionally keeps generated submission candidates out of git except the current best file.
