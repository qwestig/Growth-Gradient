# Current Best 4.721 And Post-Robust-Validation Handoff

Last updated: 2026-05-27

## Current Best

Current public best is `MAPE=4.721`, tied by two v11 submissions:

| Candidate | File | Public MAPE | Role |
|---:|---|---:|---|
| 261 | `stage2_candidate_261_full_v11_stack_huber030_g0p50_c0p005.csv` | 4.721 | Stronger pure v11 meta-correction |
| 265 | `stage2_candidate_265_full_v11_stack_avg_huber030_segment_g0p75_c0p005.csv` | 4.721 | Safer anchor; smaller movement vs 220 |

Recommended anchor for private safety: `candidate_265`.

Previous best:

| Candidate | Public MAPE | Notes |
|---:|---:|---|
| 220 | 4.727 | Previous full best: temporal204 + GRU + transfer + tiny second residual |
| 215 | 4.732 | GRU + old transfer blend |
| 210 | 4.748 | Clean GRU on 204 |
| 204 | 5.068 | Best pure temporal/Q1 baseline |

## Core Stack

`candidate_204` pure baseline:

```text
Mar2025_anchor = Mar2024 * (Jan2025 + Feb2025) / (Jan2024 + Feb2024)
candidate_204 = base_198 * exp(0.12 * clip(log(anchor / base_198), -0.06, 0.06))
```

`candidate_210`:

```text
candidate_210 = candidate_204 * exp(global_centered_GRU_residual)
```

`candidate_215`:

```text
candidate_215 = 0.75 * candidate_210 + 0.25 * candidate_118
```

`candidate_220`:

```text
candidate_220 = candidate_215 * exp(clip(0.10 * second_cb_global, -0.005, 0.005))
```

`candidate_261/265` add constrained OOF-safe v11 meta-corrections on top of the `204/210/118/159/215/220` stack.

## Robust Validation Takeaways

- `204` beats `198` on `4/5` proxy periods; average proxy delta about `-0.1005` MAPE.
- `210` is the most robust first-layer model: OOF `3.836920 -> 3.335572`, bootstrap and group-bootstrap `prob_improve=1.0`.
- `220` was preferred over `221`: same public MAPE, smaller correction.
- March-2025 target is shifted: adversarial validation historical proxy vs target AUC about `0.924`; single proxy is unreliable.
- OOF/proxy can mislead: `222` and `245` looked interesting offline but failed public/follow-up.

## Public Experiments Since Robust Lab

### Confirmed Positive

| Candidate | Public MAPE | Delta vs 220 | Notes |
|---:|---:|---:|---|
| 261 | 4.721 | -0.006 | v11 constrained Huber meta-correction; confirmed new best |
| 265 | 4.721 | -0.006 | v11 Huber + segment; ties 261 with smaller movement; safer anchor |
| 282 | 4.721 | -0.006 | Bolder v11 Huber+segment; ties but does not improve |

### Confirmed Negative / Stop These Axes

| Candidate | Public MAPE | Result | What to do |
|---:|---:|---|---|
| 222 | 4.769 | Multi-seed GRU OOF gain did not transfer | Do not submit `223/224/226` |
| 227 | 5.0684 | Robust v7 baseline tied/slightly worse than 204 | Do not retrain on 227 |
| 250 | 4.758 | v10 panel-ratio overlay on 220 failed | Do not submit `248/251/252` |
| 283 | 4.727 | Too-aggressive v11 amplification reverted to 220 level | Stop broad v11 amplification |

## Experiment Families And Decisions

### v7: Robust Temporal Anchor And Rebound

Files in root workspace:

- `scripts/search_strategy_v7_lab.py`
- `scripts/build_strategy_v7_candidates.py`
- `results/strategy_v7_lab/`

Result:

- `227` public `5.0684`, rejected/neutral.
- `228` prepared, not submitted.

Decision: stop v7 baseline axis.

### Multi-Seed GRU Ensemble

File:

- `scripts/train_temporal204_gru_multiseed_ensemble.py`

Result:

- `222` public `4.769`, worse than `210/215/220`.

Decision: do not submit `223/224/226`; seed bagging alone does not transfer.

### v8: Temporal Decomposition / Q-share

Files:

- `scripts/search_strategy_v8_temporal_decomposition.py`
- `scripts/build_strategy_v8_candidates.py`

Prepared but not submitted:

- `229`, `230`, `231`.

Decision: deltas too small for current submission budget unless probing pure-baseline behavior.

### v9: Full-Stack Micro-Blends

File:

- `scripts/search_full_stack_blend_v9.py`

Result: useful candidates moved `220` only about `0.03-0.05%`.

Decision: too small; not worth public submissions.

### v10: Multi-Period Panel Residual Baseline

Files:

- `scripts/train_temporal204_panel_residual_v10.py`
- `scripts/build_v10_full_from_panel_ratio.py`
- `scripts/train_v10panel245_gru_full_stack.py`

Result:

- Direct overlay on `220` failed: `250` public `4.758`.
- True GRU retrain on v10 baseline `245` failed offline: best OOF `3.393` vs old `210` OOF `3.336`.

Decision: stop v10 overlay and v10-baseline retrain axis.

### v11: Constrained OOF-Safe Stacking

Files:

- `scripts/build_stacking_v11_candidates.py`
- `scripts/audit_stacking_v11_261.py`

Candidates:

- `261`: public `4.721`, new best.
- `265`: public `4.721`, safer tied best.
- `262`: prepared, not submitted.

Robustness for `261`:

- OOF delta vs OOF-220: `-0.01255`.
- Row bootstrap p95 still negative, `prob_improve=1.0`.
- Group bootstrap by region/area/opening/momentum/volume: all `prob_improve=1.0`.
- Test correction small and symmetric: mean abs `0.182%`, p95 `0.489%`, max `0.501%`.
- Correlation with rejected `250` correction: `0.041`, so signal is different.

Decision: v11 confirmed, but public gain is small; `261/265` are best anchors.

### v11 Bold Follow-Up

File:

- `scripts/build_v11_bold_followup.py`

Public:

- `282`: public `4.721`, ties but no improvement.
- `283`: public `4.727`, too aggressive.

Decision: stop broad v11 amplification.

### v12: CatBoost First Residual On 204

File:

- `scripts/train_temporal204_catboost_first_residual_v12.py`

Result:

- CatBoost-first residual creates independent signal and can improve the `210` region in OOF.
- It does not beat `220/265` OOF.

Decision: do not submit `267-271` as-is. Use v12 predictions only as possible features in future broader meta-stack.

### v13: Group Total Reconciliation

File:

- `scripts/build_v13_group_total_reconciliation.py`

Result: too weak; best OOF delta near zero; movements vs `265` tiny.

Decision: do not submit `272-277`.

### v14: Selective v11 Correction

File:

- `scripts/build_v14_selective_v11_correction.py`

Idea:

- Start from safe `265` anchor.
- Strengthen only rows with high-confidence v11 combo signal.

Prepared candidates:

| Candidate | File | OOF delta vs 265 | Notes |
|---:|---|---:|---|
| 284 | `stage2_candidate_284_full_v14_selective_top50_g1p25_c0p015.csv` | -0.00330 | Best selective follow-up; changes ~43% test rows; mean abs vs 265 `0.082%` |
| 285 | `stage2_candidate_285_full_v14_selective_agree_top50_g1p25_c0p015.csv` | -0.00321 | More conservative, requires Huber/segment sign agreement |

Decision: if continuing v11-derived exploration, submit `284` first. If `284` is worse, stop v14 and search a new independent axis.

## What Not To Repeat

- Do not retrain GRU on `245` or v10 panel baseline variants without a new baseline validation result.
- Do not submit `223/224/226`: same failed multi-seed branch as `222`.
- Do not submit `248/251/252`: same failed panel-ratio overlay as `250`.
- Do not submit `267-271` yet: v12 CatBoost-first residual does not beat current OOF anchor.
- Do not submit `272-277`: group-total reconciliation too weak.
- Do not push aggressive global scale changes: OOF best scale for `210/215/220` is essentially `1.0`.
- Do not spend many submissions around `261/265/282`; v11 plateau is visible.

## Promising Next Directions

1. Submit or evaluate `284` as the next selective-v11 probe.
2. Build a broader meta-stack that includes v12 CatBoost-first residual predictions as features, not direct predictions.
3. Try a genuinely different first residual model on `204`: MLP/tabular NN, DLinear-style residual, or GRU with a different data view, not just seeds.
4. Search for public-scale segment corrections only if selective and high-confidence; broad group-total correction was too weak.
5. Use AutoTS/StatsForecast/sktime-style ideas as candidate features or small corrections first; pure baseline submissions have had small deltas recently.

## Root Workspace Files To Sync If Needed

Important root docs:

- `docs/CURRENT_BEST_4721.md`
- `docs/CURRENT_STRATEGY_AFTER_250.md`
- `docs/TEAM_HANDOFF_AFTER_ROBUST_VALIDATION.md`

Important root scripts:

- `scripts/build_stacking_v11_candidates.py`
- `scripts/audit_stacking_v11_261.py`
- `scripts/build_v11_bold_followup.py`
- `scripts/build_v14_selective_v11_correction.py`
- `scripts/train_temporal204_catboost_first_residual_v12.py`
- `scripts/build_v13_group_total_reconciliation.py`

Important lightweight summaries:

- `results/stacking_v11/submission_candidates.csv`
- `results/stacking_v11/bootstrap_selected.csv`
- `results/stacking_v11/candidate261_overall_robustness.csv`
- `results/stacking_v11/candidate261_group_bootstrap.csv`
- `results/stacking_v11_bold_followup/submission_candidates.csv`
- `results/v14_selective_v11_correction/submission_candidates.csv`

Avoid committing row-level OOF/test diagnostic CSVs unless explicitly needed.
