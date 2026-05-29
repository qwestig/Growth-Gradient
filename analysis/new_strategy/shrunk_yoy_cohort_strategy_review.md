# Shrunk YoY Store-Cohort Baseline Review

Goal: test a genuinely independent YoY baseline:

```text
pred(Mar) = Mar(previous year) * g*
g* = w * store_growth + (1 - w) * cohort_growth
```

`pred265` is used only as a comparison anchor, not inside the forecast formula.

## Script

```text
analysis/new_strategy/train_shrunk_yoy_cohort_baseline.py
```

Outputs:

```text
results/new_strategy_shrunk_yoy/shrunk_yoy_summary.csv
results/new_strategy_shrunk_yoy/shrunk_yoy_bucket_diagnostics.csv
results/new_strategy_shrunk_yoy/submission_candidates.csv
```

## Tested Variants

Growth windows:

- February YoY;
- Jan-Feb YoY.

Calendar modes:

- raw YoY;
- day-adjusted YoY for leap-year comparability.

Cohorts:

- global;
- region;
- large city;
- area category;
- opening-age category;
- region × area;
- region × age;
- area × age;
- region × large city;
- region × volume;
- age × volume;
- area × volume;
- region × stability;
- region × competition.

Shrinkage / trust rules:

- cohort only;
- constant store weights;
- volume-scaled store weight;
- stability-scaled store weight;
- volume × stability scaled store weight.

Both linear and log-space shrinkage were checked. The linear version corresponds directly to the formula suggested by the research note.

## Result

Local anchor:

```text
pred265 OOF MAPE = 3.314574
```

Best Shrunk YoY result:

```text
best OOF MAPE = 5.296252
delta         = +1.981678 worse
```

Best candidate:

```text
shrunk_yoy_feb_day_adjusted_age_const1.00_linear_k20_c0p3
```

Interpretation of best candidate:

```text
Mar2024 * day-adjusted Feb YoY growth,
grouped by opening-age category,
with high trust in store-level growth but clipped around cohort growth.
```

No submission CSV was created because no candidate passed acceptance criteria.

## Segment Diagnostics

The best candidate is worse in all large diagnostic buckets.

Even the best-looking buckets are still worse:

```text
Самара г:
pred265 MAPE = 3.363023
candidate    = 4.429469
delta        = +1.066446

Ростов-на-Дону г:
pred265 MAPE = 4.162885
candidate    = 5.298914
delta        = +1.136028
```

Worst examples:

```text
Московская обл:
delta = +2.755589 worse

Москва г | Большой:
delta = +2.961205 worse

Ленинградская обл:
delta = +3.729126 worse
```

## Interpretation

This direction is statistically reasonable in theory, but weak for this specific March task.

The likely reason:

```text
YoY store growth from Jan-Feb or Feb does not transfer cleanly into March.
```

The current `Feb * March/February` style anchor is much closer to the target mechanism than:

```text
previous March * recent YoY growth
```

Shrinkage improves the raw YoY idea somewhat, but not nearly enough to compete with the existing March-specific seasonal anchor.

## Decision

Do not submit Shrunk YoY candidates.

This strategy should not be the next main path unless it is reframed as:

1. a diagnostic feature for a direct model;
2. a fallback for rare cases where February itself is corrupted;
3. part of a conservative anchor selector, not a standalone baseline.

Standalone Shrunk YoY is not competitive here.
