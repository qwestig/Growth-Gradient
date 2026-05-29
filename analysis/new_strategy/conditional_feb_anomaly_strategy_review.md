# Conditional February Anomaly Baseline

## Idea

The current clean heuristic is strong because it uses February as the freshest level:

```text
March forecast = February RTO * global March/February seasonal factor
```

The weak point is also February. If February is an abnormal local dip after a stable December-January pattern, this anchor can underpredict March by a lot. This experiment keeps the same March/February seasonal logic, but conditionally replaces February with a normalized level only for stores where February looks broken.

## Rule

Best family:

```text
normal_level = January * clip(January / December, 0.90, 1.10)
trigger = log(February / normal_level) <= -threshold
          and abs(log(January / December)) <= 0.04

if trigger:
    level = normal_level
else:
    level = February

forecast = level * March/February factor
```

The capped variant additionally limits the corrected level:

```text
level <= February * 1.75
```

## Local Results

Reference:

```text
current clean heuristic cal050_median_mom000 MAPE = 3.872334
best known clean heuristic reference MAPE        = 3.848401
```

Best uncapped variant:

```text
candidate = feb_anom_trend_dec_jan_low_after_stable_t0p13_w1p0_cal0p6
OOF MAPE  = 3.654878
delta vs current clean = -0.217456
validation trigger rate = 0.7986%
test trigger rate       = 1.2703%
test mean shift         = 0.9180%
test max shift          = 319.48%
```

Best safer capped variant:

```text
candidate = feb_anom_trend_dec_jan_low_after_stable_t0p13_w1p0_cal0p6_cap1p75
OOF MAPE  = 3.694642
delta vs current clean = -0.177693
delta vs best clean reference = -0.153759
validation trigger rate = 0.7986%
test trigger rate       = 1.2703%
test mean shift         = 0.7112%
test max shift          = 75.31%
```

## Interpretation

This is the first new bare heuristic in this branch that materially beats the current clean baseline. It does not improve by retuning the global seasonal coefficient; it finds a specific failure mode: February is too low, December-January is stable, and March historically rebounds.

The uncapped version is locally strongest but risky because a few test stores move by more than 300% relative to the current clean prediction. The capped version keeps most of the OOF gain while making the test-side behavior less aggressive.

## Files

- Script: `analysis/new_strategy/train_conditional_feb_anomaly_baseline.py`
- Full summary: `results/new_strategy_feb_anomaly/conditional_feb_anomaly_summary.csv`
- Safe candidates: `results/new_strategy_feb_anomaly/submission_candidates.csv`
- Recommended capped CSV: `submissions/stage2_feb_anom_trend_dec_jan_low_after_stable_t0p13_w1p0_cal0p6_cap1p75.csv`

## Decision

This heuristic passes the current rule for escalation:

- bare heuristic beats current clean baseline;
- improvement is much larger than micro-noise;
- correction applies only to a small subset of stores;
- capped variant is safer for public/private shift.

Next logical step is to use this capped heuristic as the base anchor for GRU residual training, then compare that against the current public stack.
