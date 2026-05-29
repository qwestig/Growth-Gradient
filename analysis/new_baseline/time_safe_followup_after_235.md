# Time-Safe Follow-Up After Candidate 235

## Public Reference

- Current confirmed best: `test_best_temporal204_gru_second_9076.csv`
- Public score: `90.76944529` (`MAPE = 4.727`)
- Rejected candidate: `235.csv`
- Public score of rejected candidate: `90.73`

Candidate 235 used a four-seed ensemble of the March-only GRU and looked better
on store-fold OOF for March 2024. Its public deterioration shows that this OOF
does not reliably select small corrections for March 2025.

## Validation Change

Further experiments use rolling-origin validation:

- A forecast for each observed target month uses only months already available
  before that target month.
- A calendar/leap adjustment is applied only when forecasting March. The
  previous multiperiod helper applied a February leap adjustment to non-March
  target months, so it is not used as the final selection rule.

## Experiments Not Submitted

### Recent Observed Residual State

The temporal204 baseline was corrected using each store's latest known
centered residuals.

Best average result:

| Variant | Avg Delta MAPE | Wins / Periods | Worst Monthly Delta |
|---|---:|---:|---:|
| last residual, alpha 0.35, cap 0.030 | -0.0389 | 7 / 11 | +0.2635 |

The average improves, but the monthly instability is too large for a
correction on top of the `90.77` solution.

### Expanding-Window CatBoost Residual

CatBoost was trained at each cutoff only on earlier revealed monthly
residuals, with latest residual state included as input features.

Best average result:

| Variant | Avg Delta MAPE | Wins / Periods | Last 3 Wins | Worst Monthly Delta |
|---|---:|---:|---:|---:|
| alpha 0.35, cap 0.030 | -0.0574 | 4 / 8 | 1 / 3 | +0.2004 |

This model improves the mean through a few strong periods, but it is not
stable near the final forecast origin. It must not be submitted.

## Public-Feedback Probe

Candidate 235 moved away from the confirmed best and worsened public score.
The next probe tests only the opposite direction at half strength:

```text
delta_235 = log(pred_235 / pred_best220)
candidate_237 = pred_best220 * exp(-0.50 * delta_235)
```

Properties of candidate 237:

| Mean Absolute Change vs Best | P95 Change | Maximum Change |
|---:|---:|---:|
| 0.1841% | 0.5302% | 1.0050% |

This is leaderboard-feedback calibration, not a validation-confirmed modeling
improvement. Only candidate 237 should be tested; a stronger reverse step
should be considered only after observing its public result.
