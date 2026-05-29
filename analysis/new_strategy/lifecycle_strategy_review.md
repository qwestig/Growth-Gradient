# Lifecycle Strategy Review

Goal: test a genuinely separate lifecycle-based approach, not a residual or blend on top of `pred265`.

`pred265` is used only as a comparison anchor in reports.

## Public / Local Anchor

```text
current public score ~= 90.78
local pred265 OOF MAPE = 3.314574
```

## Attempt 1: March Lifecycle Ratio

Script:

```text
analysis/new_strategy/train_lifecycle_heuristic.py
```

Idea:

```text
forecast March as base * ratio(lifecycle_state)
```

Lifecycle states used:

- opening-age category;
- store area category;
- February vs January momentum;
- volume bucket;
- volatility bucket;
- local competition pressure from nearby Pyaterochka / grocery density;
- basket / promo / cancellation / working-hours bins where available.

Validation setup:

- March-2024 forecast uses only data up to February-2024;
- lifecycle March ratios are estimated from March-2023;
- March-2025 inference can use March-2023 and March-2024 histories.

Best result:

```text
best lifecycle OOF MAPE = 3.840006
delta vs pred265        = +0.525432
```

Result: not competitive as a full replacement.

Bucket diagnostics also showed no large segment where this standalone lifecycle formula beats `pred265`.

## Attempt 2: Monthly Lifecycle Multiplier

Script:

```text
analysis/new_strategy/train_lifecycle_monthly_residual.py
```

Idea:

Instead of learning lifecycle effect from March-2023 only, use all one-step monthly transitions available before the target date:

```text
actual_ratio = current_month_rto / previous_month_rto
global_month_ratio = median actual_ratio for that calendar month
lifecycle_multiplier = actual_ratio / global_month_ratio
```

Then for March:

```text
forecast = February * March_global_ratio * lifecycle_multiplier(state)
```

This is still independent from `pred265`; it learns lifecycle multipliers from raw panel history.

Best result:

```text
best monthly lifecycle OOF MAPE = 4.083171
delta vs pred265                = +0.768597
```

Result: worse than Attempt 1 and not competitive.

## Interpretation

The lifecycle direction is theoretically reasonable, but in this dataset it does not provide enough signal as a standalone forecasting mechanism.

Main reason:

```text
lifecycle state explains store regimes weakly,
but March prediction is dominated by very specific seasonal/calendar behavior.
```

Observed lifecycle states such as age, area, momentum, volume, volatility, and local competition do not recover the March-2024 target well enough.

## Decision

Do not submit lifecycle candidates.

No candidate CSV was created because no lifecycle variant passed the acceptance threshold.

This direction is not dead forever, but it should not be the next main path unless it is reframed:

1. lifecycle as an error-atlas segmentation tool;
2. lifecycle only for rare pathological stores;
3. lifecycle combined with a stronger independent direct model;
4. lifecycle used to explain public/private risk, not to replace the forecast.

For a genuinely new next strategy, the stronger candidates are:

- direct panel model predicting `log(March / February)` without `pred265`;
- share-of-market forecasting inside city/region cohorts;
- residual-shape similarity, but matching on error profiles rather than raw RTO;
- target-month-specific model trained across all month transitions with stronger calendar features.
