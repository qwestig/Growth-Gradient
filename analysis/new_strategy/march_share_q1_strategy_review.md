# March Share Of Q1 Baseline Review

Goal: test a bare independent baseline before any GRU residual training.

Decision rule:

```text
Only train GRU residual if the bare baseline beats the current clean heuristic.
```

## Current Clean Heuristic Anchor

Current branch clean heuristic:

```text
cal050_median_mom000
March forecast = February * median March/February factor
calendar_lambda = 0.50
momentum_exp = 0.00
```

Local March-2024 OOF:

```text
current clean heuristic MAPE = 3.872334
```

Best known local clean follow-up reference:

```text
best known clean heuristic MAPE = 3.848401
```

## Tested Strategy

Script:

```text
analysis/new_strategy/train_march_share_q1_baseline.py
```

Outputs:

```text
results/new_strategy_march_share_q1/march_share_q1_summary.csv
results/new_strategy_march_share_q1/submission_candidates.csv
```

Core idea:

```text
March = Q1 * March_share_of_Q1
```

Because January and February are already known, the strongest tested form was:

```text
March = (Jan + Feb) * share / (1 - share)
```

where `share` is a shrunk store/cohort estimate of:

```text
historical March / historical Q1
```

Tested variants:

- known Jan-Feb to Q1-share;
- Q1 growth from Jan-Feb YoY;
- Q1 growth from Feb YoY;
- blend of known Jan-Feb and growth-based Q1;
- global / region / city / area / age / volume / stability cohorts;
- cohort-only and store-cohort weighted shares.

## Result

Best March-share-of-Q1 result:

```text
best OOF MAPE = 4.044685
delta vs current clean heuristic = +0.172351 worse
delta vs best known clean        = +0.196284 worse
```

No candidate beat the current clean heuristic, so no GRU residual training is justified for this anchor under the current strategy.

No submission CSV was created.

## Interpretation

This heuristic is much better than Shrunk YoY, but still weaker than direct February-to-March seasonality.

What it tells us:

```text
March share of Q1 is a reasonable signal,
but knowing Jan+Feb and historical Q1 share is still less precise
than directly modeling the February -> March transition.
```

The best variants were cohort-only regional/global share formulas. Store-level historical Q1 share did not help enough, likely because each store has too little history.

## Decision

Do not train GRU residual on this baseline yet.

The bare anchor does not clear the required threshold:

```text
new bare baseline must beat current clean heuristic first.
```

Next independent heuristic candidates:

1. Deseasonalized normal-level baseline with shrunk monthly seasonal indices.
2. True calendar/trading-day intensity baseline with weekday and holiday weights.
3. Theta/SES-with-drift style baseline on log seasonally adjusted levels.
