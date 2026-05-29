# New Heuristic Plateau Review

Current public anchor:

```text
public score ~= 90.78
public MAPE  ~= 4.72
local pred265 OOF MAPE = 3.314574
```

The current `pred265` family looks plateaued under small residual/gate corrections, so this check focused on changing the baseline formula itself rather than tuning coefficients around it.

## Tested Alternative Heuristic Families

Implemented in:

```text
analysis/new_baseline/alternative_heuristic_baseline_lab.py
```

Outputs:

```text
results/new_baseline_alt_heuristics/heuristic_summary.csv
results/new_baseline_alt_heuristics/top20_heuristic_summary.csv
results/new_baseline_alt_heuristics/stress_bucket_summary.csv
results/new_baseline_alt_heuristics/major_region_summary.csv
results/new_baseline_alt_heuristics/submission_candidates.csv
```

### 1. Previous March + YoY Growth

Hypothesis:

```text
Mar2025 ~= Mar2024 * observed YoY growth from Jan-Feb or Feb
```

This is a different anchor from `Feb * Mar/Feb seasonal factor`: it assumes last March is the base, and current Jan-Feb tells us how much the store grew year over year.

Result:

```text
Best variant OOF MAPE = 3.340444
Delta vs pred265      = +0.025870
```

Conclusion: worse than `pred265`. The previous March anchor does not beat the current February-seasonality anchor.

### 2. Q1 Total Residual

Hypothesis:

```text
Forecast Q1 total, then March = Q1 - Jan - Feb
```

This tests whether March is better treated as the missing part of the quarter rather than as a direct next-month forecast.

Result:

```text
Best variant OOF MAPE = 3.340444
Delta vs pred265      = +0.025870
```

Conclusion: no gain. At store level, Q1 residual is too unstable.

### 3. Normal Jan-Feb Level Instead Of Raw February

Hypothesis:

```text
Mar ~= normal_level(Jan, Feb) * seasonal_factor
```

This checks whether February alone is too noisy, and whether a Jan-Feb level is a better baseline.

Result:

```text
Best variant OOF MAPE = 3.340444
Delta vs pred265      = +0.025870
```

Conclusion: smoothing February with January hurts locally. The current solution benefits from using February directly.

### 4. Explicit Calendar Intensity

Hypothesis:

```text
Convert Jan/Feb/March into pseudo-daily intensity,
including leap-year and weekend-count effects.
```

Result:

```text
Best variant OOF MAPE = 3.340733
Delta vs pred265      = +0.026159
```

Conclusion: exact calendar/day-count handling is not better than the current partial calendar correction.

### 5. Top-Down Share Allocation

Hypothesis:

```text
Forecast global or regional March total,
then distribute it by current Feb or Jan-Feb store shares.
```

This tests whether the main error is not the aggregate March uplift, but the allocation across stores.

Result:

```text
Best variant OOF MAPE = 3.337217
Delta vs pred265      = +0.022643
```

Conclusion: this was the best new family, but still worse than `pred265`. It did not improve any of the 5 stress buckets.

### 6. Rank / Quantile Transfer

Hypothesis:

```text
Keep the prior-year March distribution,
but assign March values by current Feb rank.
```

This tests whether ranks are more stable than store-level ratios.

Result: worse than `pred265`, including in clipped blends.

Conclusion: rank transfer loses too much store identity.

## Diagnostic: Inverse Signal

After all direct alternatives worsened OOF, tested whether these heuristics at least point in a useful opposite direction.

Best inverse diagnostic:

```text
OOF MAPE = 3.319895
Delta    = +0.005320
```

Conclusion: even inverse corrections are worse. These alternatives do not contain a useful hidden direction around `pred265`.

## Operational Feature Signal Check

Checked simple monotonic relation between March-2024 `pred265` residual and Jan-Feb operational/static features:

```text
max |Spearman rho| ~= 0.035
```

Largest signals were from household count, nearby stores, population, and basket size, but the effects are too weak for a deterministic heuristic. They may help a learned model slightly, but they do not justify a new hand-written baseline.

## Current Conclusion

The most natural replacements for the current base formula failed:

```text
previous March anchor       -> worse
Q1 residual                 -> worse
Jan-Feb normal level        -> worse
exact calendar intensity    -> worse
top-down share allocation   -> worse
rank transfer               -> worse
feature-pressure heuristic  -> too weak
```

So the plateau is real for obvious deterministic alternatives. The current `pred265` is not just over-tuned by coefficients; it is locally a very strong March-specific anchor.

## Most Promising Next Heuristic Directions

1. Error-atlas heuristic, not global formula replacement.
   Build a map of where `pred265` fails by store regimes and only invent a new formula for those regimes. The key is not "one better baseline for all", but "which stores should not use the same February-seasonality anchor".

2. Lifecycle / cannibalization heuristic.
   Use opening age, nearby Pyaterochka density, nearby grocery density, and settlement scale to identify stores where March growth may be structurally suppressed or boosted. The raw feature correlation is weak globally, so this should be regime-based, not global.

3. Multi-window robustness check before new submit.
   Any new heuristic should be validated on more than March-2024 if possible. A single March OOF can reject bad formulas, but it is dangerous for selecting tiny improvements.

4. Cross-similarity v2 only if it changes the matching object.
   The first similarity expert failed. A better version should match on residual shape versus `pred265`, not raw RTO trajectories.

No submission CSV was created from this lab because no candidate passed the minimum acceptance threshold.
