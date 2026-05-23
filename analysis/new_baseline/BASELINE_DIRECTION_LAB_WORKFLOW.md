# Baseline Direction Lab Workflow

## Goal

The current target is not another tiny proxy improvement. The goal is to find a new heuristic baseline direction that can eventually beat the current public-best stack.

Current reference:

```text
GRU alpha 0.70 + momentum_raw_bin correction
public score 89.65
implied MAPE ~= 5.3163
```

Absolute target:

```text
MAPE improvement: 0.5-0.6 points
target score: about 90.60-90.79
```

This likely requires a better baseline or data-generating heuristic, not just more ML.

## Tools Added

### Direction Lab

Run:

```bash
.venv/bin/python scripts/heuristic_baseline_direction_lab.py
```

Outputs:

```text
results/heuristic_baseline_direction_candidates.csv
results/heuristic_direction_shap_summary.csv
results/heuristic_direction_feature_bins.csv
results/heuristic_public_experiment_template.csv
submissions/stage2_candidate_43_baseline_*.csv ... submissions/stage2_candidate_62_baseline_*.csv
```

The script does not read any `submissions/*.csv`. It writes candidate files through Python only.

### Public Experiment Tracker

Run after adding public scores to `results/public_experiment_log.csv`:

```bash
.venv/bin/python scripts/analyze_public_experiment_log.py
```

Outputs:

```text
results/public_experiment_log.csv
results/public_experiment_direction_summary.csv
```

This is not intended to create a proxy. It turns public submissions into a directional map.

## First Direction Lab Result

The strongest baseline-level signal is partial leap/calendar normalization.

Current raw baseline diagnostic:

```text
raw_median_mom025 March-2024 MAPE = 4.128017
```

Best heuristic candidates in the first lab run:

| Candidate | Name | March-2024 MAPE | Final factor | Change vs raw baseline |
|---:|---|---:|---:|---:|
| 46 | `cal0375_median_mom025` | `3.966277` | `1.126372` | `1.293%` |
| 48 | `cal025_mape_wmedian_mom025` | `3.967991` | `1.122306` | `0.997%` |
| 50 | `cal025_median_mom0125` | `3.972902` | `1.123960` | `0.441%` |
| 45 | `cal025_median_mom025` | `3.975748` | `1.123960` | `0.862%` |
| 52 | `cal025_tempdip_lift010` | `3.975765` | `1.123960` | `0.861%` |

Interpretation:

```text
partial calendar/leap adjustment is a real candidate direction;
temporary-dip corrections did not yet add meaningful local gain on top of calendar adjustment;
cohort factors are not the first public-test priority.
```

## SHAP Diagnostic Takeaway

Top diagnostic features for residual over raw baseline:

| Feature | mean_abs_shap |
|---|---:|
| `Количество домохозяйств` | `0.005288` |
| `momentum_raw` | `0.004292` |
| `Регион` | `0.003185` |
| `dip_score` | `0.003177` |
| `momentum_clip` | `0.003044` |
| `volatility_rel3` | `0.002936` |
| `store_mar_feb_ratio_prev` | `0.002274` |

Interpretation:

```text
momentum/dip/volatility are still meaningful;
region and household count may matter, but they are dangerous as direct corrections;
store-level previous March factor appears but is known to transfer poorly, so use it only for diagnostics.
```

## How To Use 50+ Public Experiments

Do not try to pick the best candidate by local metric. Use public submissions as controlled experiments.

Recommended first public batch from this lab:

| Submit order | Candidate | Question |
|---:|---|---|
| 1 | `stage2_candidate_45_baseline_cal025_median_mom025.csv` | Does partial calendar normalization help at moderate strength? |
| 2 | `stage2_candidate_46_baseline_cal0375_median_mom025.csv` | Is the best local calendar strength also public-positive? |
| 3 | `stage2_candidate_48_baseline_cal025_mape_wmedian_mom025.csv` | Does MAPE-optimal factor selection help vs normal median? |
| 4 | `stage2_candidate_50_baseline_cal025_median_mom0125.csv` | Is the current momentum exponent too strong under calendar normalization? |
| 5 | `stage2_candidate_52_baseline_cal025_tempdip_lift010.csv` | Does light temporary-dip rebound add public signal? |

These are baseline-only candidates, so they will likely score worse than GRU-best in absolute terms. Their value is directional:

```text
If calendar candidates improve materially over raw baseline/public baseline references,
then retrain GRU residual around calendar-adjusted baseline.
```

## Decision Rules

Use public feedback this way:

| Public outcome | Decision |
|---|---|
| Calendar candidates clearly beat raw baseline direction | Retrain GRU on calendar-adjusted baseline immediately. |
| Stronger calendar is better than weaker calendar | Expand `calendar_lambda` upward around winning value. |
| MAPE-weighted factor beats normal median | Use MAPE-aligned factor objective in all future baseline design. |
| Momentum exponent `0.125` beats `0.25` | Rebuild baseline with weaker momentum and retrain GRU. |
| Temporary-dip lift helps baseline-only public | Build a two-regime rebound heuristic before GRU retraining. |
| Cohort candidates help baseline-only public | Add cohort factors only with heavy shrink and stability filters. |

## Important Caution

Baseline-only candidates are not expected to beat `89.65`. The near-term question is whether their directional public movement is large enough to justify a full retrain:

```text
new heuristic baseline -> new residual target -> retrained GRU -> momentum correction
```

If a baseline-only candidate moves public in the right direction by a lot, it is more important than a tiny GRU-best micro-correction.
