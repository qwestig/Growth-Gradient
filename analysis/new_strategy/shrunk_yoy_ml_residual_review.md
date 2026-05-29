# Shrunk YoY + ML Residual Review

Goal: test the fair version of Shrunk YoY:

```text
independent Shrunk YoY baseline + learned residual model
```

`pred265` is still used only as a benchmark, not as an input to the forecast formula.

## Script

```text
analysis/new_strategy/train_shrunk_yoy_ml_residual.py
```

Outputs:

```text
results/new_strategy_shrunk_yoy_ml/shrunk_yoy_ml_summary.csv
results/new_strategy_shrunk_yoy_ml/submission_candidates.csv
```

## Setup

Baseline:

```text
Mar(previous year) * shrunk YoY growth
```

Residual target:

```text
target_residual = log(actual_march_2024 / shrunk_yoy_pred_2024)
```

Models checked:

- HistGradientBoostingRegressor;
- Ridge with one-hot categorical features.

Validation:

```text
fold OOF on March-2024 stores
```

## Result

Raw Shrunk YoY:

```text
OOF MAPE = 5.296252
```

Best Shrunk YoY + ML:

```text
OOF MAPE = 4.105828
delta vs raw Shrunk YoY = -1.190424
```

Current comparison anchor:

```text
pred265 OOF MAPE = 3.314574
```

So the best ML-corrected Shrunk YoY result is still:

```text
delta vs pred265 = +0.791254 worse
```

Best candidate:

```text
ridge_a1.0_c0.1
```

No submission candidate was created because no model beat the current anchor.

## Interpretation

ML clearly helps this baseline:

```text
5.296 -> 4.106
```

That means the residual model learns meaningful corrections around Shrunk YoY.

But the baseline is too far from the target mechanism. Even after ML correction, it remains much weaker than the current March-specific `Feb -> Mar` anchor.

Conclusion:

```text
Shrunk YoY is not competitive as a primary strategy,
even after a supervised residual layer.
```

The idea can still be useful as a feature family in a broader direct model, but not as the main baseline anchor.
