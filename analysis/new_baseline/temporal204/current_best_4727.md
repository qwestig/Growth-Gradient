# Current Best: Temporal/Q1 Baseline + GRU + Second Residual

## Public Best

Current best submission:

```text
submissions/test_best_temporal204_gru_second_9076.csv
```

Submitted milestone files kept in the repo:

```text
submissions/test_best_temporal204_pure_9012.csv       # candidate 204, pure baseline, MAPE 5.068
submissions/test_best_temporal204_gru_9072.csv        # candidate 210, one-layer fresh GRU, MAPE 4.748
submissions/test_best_temporal204_gru_blend_9075.csv  # candidate 215, one-layer GRU + transfer blend, MAPE 4.732
submissions/test_best_temporal204_gru_second_9076.csv # candidate 220, second residual layer, MAPE 4.727
```

Equivalent source file outside the repo:

```text
submissions/stage2_candidate_220_baseline_temporal204_second_cb_on215_global_g010_clip005.csv
```

Public result:

```text
MAPE = 4.727
score = 90.76944529
```

Candidate `221` tied by reported public MAPE, but `220` is kept as current best because it uses the smaller second-layer correction.

## Public Ladder

| Candidate | Description | Public MAPE |
|---:|---|---:|
| 159 | previous full best, `0.75 * 150 + 0.25 * 118` | 4.763 |
| 204 | best pure temporal/Q1 baseline | 5.068 |
| 210 | fresh GRU residual on candidate `204` | 4.748 |
| 215 | `0.75 * 210 + 0.25 * 118` | 4.732 |
| 220 | tiny CatBoost second residual on `215` | 4.727 |
| 221 | stronger CatBoost second residual on `215` | 4.727 |

## Main Breakthrough

The largest new pure-baseline gain came from a temporal aggregation / Q1-style anchor:

```text
Mar2025_anchor = Mar2024 * (Jan2025 + Feb2025) / (Jan2024 + Feb2024)
```

Candidate `204` blends this anchor into the sign-asymmetric store-seasonality baseline with clipped log-ratio:

```text
base_204 = base_198 * exp(0.12 * clip(log(anchor / base_198), -0.06, 0.06))
```

This moved the best pure baseline from `5.126` to `5.068`.

## Residual Stack

### First Residual Layer

Fresh GRU was trained directly on the temporal/Q1 baseline target:

```text
target = log(actual_Mar2024 / base_204_Mar2024)
```

Best OOF variant:

| Variant | Alpha | Centering | OOF MAPE |
|---|---:|---|---:|
| `temporal204_gru_global` | 1.00 | global | 3.335572 |

Public:

```text
candidate_210 = base_204 * exp(global_centered_GRU)
MAPE = 4.748
```

### Transfer Blend

The previous old-transfer residual `candidate_118` still carried independent signal:

```text
candidate_215 = 0.75 * candidate_210 + 0.25 * candidate_118
MAPE = 4.732
```

### Second Residual Layer

An OOF-safe CatBoost second residual was trained on top of `candidate_210` OOF predictions:

```text
target_2 = log(actual / pred_210_oof)
```

Best public candidate applies a tiny clipped correction to `candidate_215`:

```text
candidate_220 = candidate_215 * exp(clip(0.10 * cb_second_residual_global, -0.005, 0.005))
MAPE = 4.727
```

Candidate `221` used a stronger correction and tied by public MAPE; the axis is likely near plateau.

## Repro Scripts

Core scripts for this stage:

```text
analysis/new_baseline/temporal204/search_structural_baselines_v6.py
analysis/new_baseline/temporal204/build_structural_baseline_v6_candidates.py
analysis/new_baseline/temporal204/train_temporal204_gru_residual.py
analysis/new_baseline/temporal204/train_temporal204_second_residual.py
```

Lightweight summaries:

```text
results/structural_baselines_v6/
results/temporal204_gru_residual/
results/temporal204_second_residual/
results/public_experiment_log.csv
results/public_experiment_direction_summary.csv
```

Large row-level OOF files are intentionally not committed.

## Recommended Handoff

For the ML teammate:

1. Use `candidate_220` / `test_best_temporal204_gru_second_9076.csv` as the current best submission.
2. If continuing residual work, start from the temporal/Q1 baseline and candidate `210/215/220` stack, not from old `159`.
3. Avoid spending more submissions on stronger second-layer CatBoost corrections unless there is a new validation signal; `220` and `221` tied.
4. Most promising next work is independent residual modeling around the temporal/Q1 baseline or a more principled stack using OOF predictions from `210`, `118`, and handcrafted temporal features.
