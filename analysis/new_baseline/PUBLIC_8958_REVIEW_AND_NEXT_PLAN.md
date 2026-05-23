# Public 89.58 Review And Next Plan

## Constraint

Do not inspect `submissions/*.csv` with context-loading tools. Submission files are too large and should only be touched by scripts when needed.

## Notebooks Reviewed

### `notebooks/stage2_best_baseline_catboost_residual.ipynb`

This is the cleanest notebook-backed strong model.

Core stack:

```text
global median March/February seasonal factor
-> conservative February momentum
-> CatBoost residual on log(actual / baseline)
```

Confirmed leaderboard path:

| Model | Public MAPE |
|---|---:|
| best baseline / `test_best.csv` | `~5.72` |
| candidate 14: mild CatBoost residual | `5.707` |
| candidate 15: aggressive CatBoost residual | `5.58` |

Key local result:

| Model | March-2024 OOF MAPE |
|---|---:|
| clean best baseline | `4.1280` |
| CatBoost residual `alpha=0.03`, clip +/-2% | `~4.09` |
| CatBoost residual `alpha=0.50`, no clip | `3.6498` |

Useful conclusion: CatBoost residual proved that the baseline still had learnable multiplicative residual signal. But it is now behind the GRU branch.

### `Growth-Gradient/notebooks/heuristic_baseline.ipynb`

This notebook validates the core baseline:

```text
forecast = February RTO * global March/February factor * conservative momentum correction
```

March-2024 local result:

| Method | MAPE |
|---|---:|
| seasonal + conservative momentum | `4.1280` |
| seasonal baseline | `4.3554` |

Useful conclusion: the best model should keep this baseline. Store-level March/February ratios are too noisy; global seasonality is robust.

### `notebooks/stage2_momentum_heuristic.ipynb`

Important because it exposed proxy weakness.

The old offline proxy preferred `delta_yoy` / blends, but public leaderboard favored multiplicative momentum. This was one of the first signs that non-March or poorly aligned proxy metrics are dangerous.

### `notebooks/stage2_momentum_catboost_residual.ipynb`

This was a conservative CatBoost correction over an earlier momentum baseline.

Useful conclusion: CatBoost can improve a baseline offline, but offline ranking was not reliable enough to select aggressive settings.

### `notebooks/stage2_proxy_rank_diagnostics.ipynb`

Important finding: old candidate generation had lag alignment errors. Scores from lag-shifted candidates should not be used as honest calibration points.

### Old November notebooks

`final_gpu_blend_submission.ipynb`, `final_segment_blend_submission.ipynb`, and `gpu_ensemble_search.ipynb` belong to an older November-style task. They are not directly reusable for March 2025 because their validation windows were `8, 9, 10` and their baseline was mostly `lag1`.

Reusable idea only:

```text
segment-specific blending/gating can work if validation matches the target month
```

## Current Best Is Not Notebook-Backed

The current public best comes from scripts in `Growth-Gradient/origin/codex-momentum-bias-solution`, not from a notebook.

Current proven stack:

```text
best heuristic baseline
-> March-specific GRU residual alpha 0.70
-> momentum_raw_bin bias correction
```

Known public ladder:

| Model | Public score | Implied MAPE |
|---|---:|---:|
| GRU alpha 0.70 | `89.52` | `5.3850` |
| GRU + momentum bias correction | `89.65` | `5.3163` |
| candidate 36 second-level correction | `89.58` | `5.3533` |

The public drop from `89.65` to `89.58` is approximately `+0.0370` MAPE points worse.

## Updated Interpretation After Public 89.58

The previous proxy liked candidate 36:

```text
store_mar_feb_ratio_bin second-level correction
k=1000, beta=0.50, clip=0.005
```

But public score dropped. That changes the conclusion:

```text
Do not trust second-level OOF micro-corrections as submission selectors.
```

More specifically:

| Hypothesis | Status after public 89.58 |
|---|---|
| Momentum_raw_bin correction | Still proven. It improved public from `89.52` to `89.65`. |
| Extra store_mar_feb_ratio_bin correction | Rejected or at least deprioritized. Public fell to `89.58`. |
| More fine-grained OOF segment corrections | High overfit risk. Avoid as primary path. |
| Robust proxy ranker | Useful as diagnostics, not as final ranking oracle. |
| Global blending | Still rejected. GRU+CatBoost blend was worse. |

The current public evidence says the remaining gap is not solved by adding another small historical-ratio segment correction. We need either:

1. a better public-guided calibration of the already-proven momentum bias, or
2. a genuinely stronger residual model than the current GRU.

## Target Math

Current score `89.65` implies:

```text
MAPE ~= 5.3163
```

If the target means `0.5` absolute MAPE points better:

```text
target MAPE ~= 4.8163
target score ~= 90.60
```

If the target means `0.6` absolute MAPE points better:

```text
target MAPE ~= 4.7163
target score ~= 90.79
```

If the target means `0.5%` relative MAPE improvement:

```text
target MAPE ~= 5.2897
target score ~= 89.70
```

If the target means `0.6%` relative MAPE improvement:

```text
target MAPE ~= 5.2844
target score ~= 89.71
```

These are very different difficulty levels. A `0.5` absolute MAPE-point improvement likely requires a new model-level gain, not a micro-correction. A `0.5%` relative improvement can plausibly come from better calibration or a small gated correction.

## What To Do Next

### Track A: Public-Guided Momentum Calibration

Priority: highest immediate ROI.

Reason: the only post-GRU correction proven on public is `momentum_raw_bin`.

Stop using OOF winner as the main criterion. Use public feedback to probe the shape around the current correction.

Candidate ideas:

```text
q0 beta: 0.90, 1.05, 1.10, 1.15
q1 beta: 0.90, 1.00, 1.10
q4 beta: 0.75, 0.90, 1.00, 1.10
clip: keep 0.02 first
shrink_k: keep 500 first
```

Why not jump to many segment corrections: candidate 36 showed that proxy-positive tiny corrections can still hurt public.

Expected gain: small, but plausible. This is the route if the target is relative `0.5%` MAPE improvement.

### Track B: Tail And Disagreement Safety, But Only As Ablation

Priority: medium.

The current GRU has tail corrections where `GRU / baseline` can exceed `2.7` on weak-February stores. Tail caps are conceptually attractive, but local proxy disliked them.

Use them only as information-gathering submissions:

```text
very mild cap_up 2.50 / cap_down 0.75
maybe cap only when momentum_raw < extreme threshold
avoid broad caps that touch many rows
```

Expected gain: uncertain. Good for reducing catastrophic private/public subset risk, not for guaranteed public gain.

### Track C: Save CatBoost OOF And Test True GRU-CatBoost Gates

Priority: medium-high.

We currently have test-side CatBoost predictions but not row-level March-2024 CatBoost OOF predictions in a form that lets the robust ranker evaluate gate candidates.

Next step:

```text
save March-2024 CatBoost OOF residual predictions aligned by new_id and fold
evaluate candidate_33/34/35 style gates honestly
```

Important: CatBoost should not be globally blended. Use it only where GRU and CatBoost disagree strongly.

Expected gain: small-to-medium. More credible than arbitrary second-level segment corrections because model disagreement can flag uncertainty.

### Track D: Model-Level Improvement To GRU

Priority: necessary if target is absolute `0.5` MAPE point.

Current GRU details:

```text
SEQ_LEN = 14
hidden_dim = 32
dropout = 0.15
loss = SmoothL1Loss(beta=0.02)
alpha = 0.70
sequence features include normalized log RTO and log growth
static features include numeric store context and categorical embeddings
```

High-upside changes:

1. Pretrain encoder on all one-step monthly transitions, then fine-tune a March-only head.
2. Train with a more MAPE-aligned residual loss instead of only SmoothL1 on log ratio.
3. Try direct residual quantile/asymmetric training by momentum segment.
4. Try longer and shorter sequence lengths around `8`, `14`, `18`, `24`; current `14` may miss full yearly pattern plus recent shock.
5. Train several GRU variants but use seed/model variance as uncertainty, not simple averaging.

Expected gain: only track with enough upside for a large MAPE move, but more engineering cost.

### Track E: Transformer / DLinear / NHITS Only If They Change The Data View

Priority: lower unless implemented as diagnostics.

Known facts:

```text
Transformer was locally strong but public-worse.
DLinear local MAPE 3.4779 was worse than GRU 3.4417.
GRU/DLinear local blend improved only from 3.4417 to about 3.4406, too tiny.
```

Do not submit raw architecture outputs. Use them for:

```text
agreement flags
uncertainty flags
segment diagnostics
pretraining comparisons
```

## Recommended Immediate Sequence

1. Treat public `89.58` as a rejection of candidate 36 and store-ratio second-level correction.
2. Do a small public-guided momentum calibration sweep around current best, not around OOF-best.
3. Rebuild/surface CatBoost March-2024 OOF predictions so GRU-CatBoost gates can be validated instead of guessed.
4. Start a model-level GRU improvement branch: all-transition encoder pretraining plus March-only head.
5. Use robust proxy only as a stress dashboard. Do not let it pick final submissions alone.

## Bottom Line

For a small relative MAPE improvement, continue around public-proven `momentum_raw_bin` calibration.

For an absolute `0.5-0.6` MAPE-point improvement, micro-corrections are very unlikely to be enough. We need a stronger residual model, most plausibly:

```text
pretrained sequence encoder -> March-specific head -> public-proven momentum bias correction
```
