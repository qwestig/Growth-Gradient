# Share-of-Market Strategy Review

Goal: test a separate forecasting strategy that does not use `pred265` in the formula.

The strategy forecasts March in two stages:

```text
1. forecast March total for a cohort;
2. allocate that total between stores by current / smoothed / trend store shares.
```

`pred265` is used only as the comparison anchor.

## Script

```text
analysis/new_strategy/train_share_of_market_strategy.py
```

Outputs:

```text
results/new_strategy_share_of_market/share_of_market_summary.csv
results/new_strategy_share_of_market/share_bucket_diagnostics.csv
results/new_strategy_share_of_market/submission_candidates.csv
```

## Tested Cohorts

- global;
- region;
- large city;
- area category;
- opening-age category;
- region × area;
- region × age;
- area × age;
- region × large city;
- region × competition pressure.

## Tested Cohort Total Forecasts

```text
seasonal_feb_total:
    cohort_total_Mar = cohort_total_Feb * historical Mar/Feb ratio

prev_march_janfeb_growth:
    cohort_total_Mar = previous_year_Mar * current Jan-Feb YoY growth

blend_total:
    0.65 * seasonal_feb_total + 0.35 * prev_march_janfeb_growth
```

## Tested Store Share Allocation

- February share;
- Jan-Feb share;
- weighted last-3-month share;
- 6-month share trend;
- blend of current February share and previous March share.

## Result

Local anchor:

```text
pred265 OOF MAPE = 3.314574
```

Best share-of-market result:

```text
best OOF MAPE = 4.149334
delta         = +0.834759 worse
```

No candidate passed submission criteria.

## Segment Diagnostics

There are isolated groups where share-of-market is better, for example:

```text
Ростов-на-Дону г:
pred265 MAPE = 4.162885
candidate    = 3.833176
delta        = -0.329710
```

But most large groups are worse, especially Moscow / Moscow region and several region-area buckets.

Examples:

```text
Московская обл:
delta = +1.521536 worse

Ленинградская обл:
delta = +2.015133 worse

Москва г | Большой:
delta = +1.600804 worse
```

## Interpretation

The share-of-market idea is structurally reasonable, but not strong enough here as a standalone forecast.

The likely reason:

```text
store identity and store-specific March/Feb behavior matter more
than smooth redistribution of aggregate cohort totals.
```

The best variants were the simplest:

```text
cohort total from seasonal Feb total
allocation by February share
```

More complex share dynamics did not help.

## Decision

Do not submit share-of-market candidates.

This strategy may still be useful only as:

1. a diagnostic for regional allocation errors;
2. a specialist for a tiny region/city subset;
3. a feature family for a direct model;
4. a private-risk check for cohort total consistency.

It is not a good replacement for the current March-specific seasonal anchor.
