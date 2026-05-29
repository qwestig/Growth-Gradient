# Deseasonalized Normal-Level Baseline Review

Goal: test a bare independent baseline before any GRU residual training.

Decision rule:

```text
Only train GRU residual if the bare baseline beats the current clean heuristic.
```

## Current Clean Heuristic Anchor

```text
cal050_median_mom000
March forecast = February * median March/February factor
calendar_lambda = 0.50
momentum_exp = 0.00
```

Local March-2024 OOF:

```text
current clean heuristic MAPE = 3.872334
best known clean heuristic MAPE = 3.848401
```

## Tested Strategy

Script:

```text
analysis/new_strategy/train_deseasonalized_normal_baseline.py
```

Outputs:

```text
results/new_strategy_deseasonalized_normal/deseasonalized_normal_summary.csv
results/new_strategy_deseasonalized_normal/submission_candidates.csv
```

Core idea:

```text
1. estimate monthly seasonal indices from previous full years;
2. divide recent months by their seasonal indices;
3. estimate normal store level from deseasonalized recent months;
4. multiply normal level by March seasonal index.
```

Tested normal level variants:

- February only;
- median of Dec-Jan-Feb;
- weighted Dec-Jan-Feb;
- median of Sep-Feb;
- weighted Sep-Feb.

Tested seasonal index variants:

- global;
- region;
- large city;
- area;
- age;
- region-area;
- region-age;
- area-age;
- volume cohorts;
- shrunk store-specific seasonal indices.

## Result

Best deseasonalized normal-level baseline:

```text
best OOF MAPE = 4.223340
delta vs current clean heuristic = +0.351005 worse
delta vs best known clean        = +0.374939 worse
```

Best candidate:

```text
deseason_feb_only_city_big_k1000_sw0p0
```

No candidate beat the current clean heuristic, so no GRU residual training is justified for this anchor under the current strategy.

No submission CSV was created.

## Interpretation

The best variant still uses February only after deseasonalization. Multi-month normal levels were worse.

This suggests:

```text
for this task, the latest February level is more informative
than a smoothed deseasonalized normal level.
```

The current clean heuristic already captures the strongest useful part:

```text
fresh February level + March/February seasonal transition
```

The deseasonalized formulation loses accuracy because seasonal indices are too noisy with only two years of monthly history.

## Decision

Do not train GRU residual on this baseline yet.

The bare anchor does not clear the required threshold.
