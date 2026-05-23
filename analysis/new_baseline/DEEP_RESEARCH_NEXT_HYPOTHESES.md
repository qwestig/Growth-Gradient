# Deep Research Next Hypotheses

## Scope

This memo adapts `docs/deep-research-report.md` to the current proven competition stack:

```text
best heuristic baseline -> GRU residual alpha 0.70 -> momentum_raw_bin bias correction
```

Current public-best reference:

```text
Growth-Gradient/submissions/test_best_gru_momentum_bias_8965.csv
```

Public score is `89.65`, roughly `MAPE ~= 5.32`. The most relevant local challenger from the audit pool is:

```text
submissions/stage2_candidate_36_gru_second_store_mar_feb_ratio_bin_k1000_b050_c005.csv
```

It improves the March-2024 OOF/stress proxy versus the current momentum-corrected GRU:

| Model | Global OOF MAPE | Momentum-weighted MAPE | Robust score |
|---|---:|---:|---:|
| Current GRU-best proxy | `3.413660` | `3.787455` | `3.737374` |
| Candidate 36 | `3.406647` | `3.779636` | `3.729342` |

## Main Takeaway

The report supports the direction we are already in: strong simple baseline, global learning across many related store series, residual modeling, and conservative post-model calibration.

For this contest, the next high-ROI work is not a raw replacement of GRU with a fashionable TS architecture. The best use of additional models is:

```text
diagnostic signal -> uncertainty gate -> tiny residual correction -> robust stress ranking
```

Not:

```text
average many forecasts globally and submit the blend
```

Reason: naive GRU + CatBoost blending already degraded public score to about `89.49`, while segment-specific momentum correction improved public score from `89.52` to `89.65`.

## Does Stacking Make Sense?

Yes, but only in a restricted form.

| Stacking type | Recommendation | Why |
|---|---|---|
| Global average of GRU, CatBoost, Transformer, DLinear | Avoid as first choice | Already failed for GRU + CatBoost; single March validation makes global weights fragile. |
| OOF residual meta-model after GRU-best | Try, with hard shrink and clips | Residual structure remains by momentum, volatility, store ratio, region, area, opening age. |
| Segment/gated stacking | Best stacking direction | Uses alternative models only where they are likely informative or where GRU is uncertain. |
| Rank/uncertainty stacking | Good diagnostic layer | Model disagreement can flag tail-risk stores without changing normal stores. |
| Architecture ensemble as raw submit | Risky | Local Transformer and seed ensemble looked good but did not transfer to public. |

Practical rule:

```text
If a stacking layer changes most rows, it is probably too risky.
If it changes only known weak/stress segments with OOF shrinkage, it is a plausible candidate.
```

## Boosting Direction

Tree boosting is still useful, but not as a standalone raw predictor.

Best uses:

| Use | Priority | Implementation |
|---|---:|---|
| Residual feature importance | High | Use CatBoost SHAP to find correction segments. |
| Safety gate | High | If GRU and CatBoost strongly disagree, shrink only the extreme GRU correction. |
| OOF residual micro-model | Medium | Predict `log(actual / gru_best)` with strict cross-fitting, then multiply by `exp(clipped_residual)`. |
| Direct forecast replacement | Low | Baseline + CatBoost residual was weaker than GRU branch. |

Candidate family already created for the safety-gate idea:

```text
stage2_candidate_33_gru_cb_gate_15pct_085_115.csv
stage2_candidate_34_gru_cb_gate_25pct_080_120.csv
stage2_candidate_35_gru_base_gate_15pct_080_125.csv
```

## Time-Series Architecture Hypotheses

### 1. GRU-Best With Second-Level Micro Correction

Status: already implemented.

Recommended first submit from the audit-generated pool:

```text
submissions/stage2_candidate_36_gru_second_store_mar_feb_ratio_bin_k1000_b050_c005.csv
```

Why: it improves global OOF, weak momentum, high volatility, and March-2025 momentum-weighted proxy simultaneously.

Risk: the gain is small, so public/private subset noise can dominate.

### 2. Gated Stacking With Transformer

Priority: high-upside, high-risk.

Rationale: Transformer had strong local MAPE but worse public behavior. That means it may contain signal, but validation transfer is unreliable.

Use it only as a small correction:

```text
pred = gru_best * exp(w * clipped(transformer_log_residual - gru_log_residual))
```

Suggested constraints:

| Parameter | Value |
|---|---:|
| `w` | `0.05`, `0.10`, `0.15` |
| clip | `0.005`, `0.010` |
| gate | only when Transformer and GRU agree on correction direction |
| fallback | keep GRU-best unchanged |

Best test segments:

```text
high volatility
extreme momentum_raw
large GRU/CatBoost disagreement
large GRU/base ratio tail
```

### 3. Pretrain Encoder On All Monthly Transitions, Fine-Tune March Head

Priority: radical but conceptually strong.

The failed multi-month GRU likely lost March specificity. A better design is:

```text
Stage A: train sequence encoder on all one-step residual transitions.
Stage B: freeze or lightly fine-tune encoder.
Stage C: train March-only head on March residual.
Stage D: calibrate with momentum_raw_bin bias correction.
```

Why this matches the report: global neural models benefit from cross-series learning and more transitions, but horizon-specific behavior needs direct calibration.

Risk: engineering cost is higher than micro-corrections, and there is still only one true March OOF target.

### 4. NHITS / DLinear / NLinear As Counter-Baselines

Priority: medium.

The report emphasizes DLinear/NLinear/NHITS as cheap and strong counter-baselines. Here they should be used to produce residual diagnostics, not immediate replacements.

Useful outputs:

```text
nhits_residual
dlinear_residual
agreement_with_gru
disagreement_tail_flag
model_rank_by_segment
```

Submit only if a model improves robust stress slices after being gated or shrunk into GRU-best.

### 5. Foundation Models As Diagnostics

Priority: low to medium.

Chronos/TimesFM/TimeGPT-like models can be useful as zero-shot diagnostics or cold-start features, but direct submission is unlikely to be best because:

```text
the current solution already uses contest-specific March seasonality, store metadata, and public-tested segment calibration
```

Potential use:

```text
foundation forecast / baseline ratio
foundation vs GRU disagreement
foundation quantile width as uncertainty
```

### 6. Hierarchical / Cohort Reconciliation-Like Corrections

Priority: high if implemented conservatively.

There is no official aggregate target in the submission, but retail hierarchy still matters. Use reconciliation-like ideas as small cohort corrections:

```text
region
area category
opening category
store-level historical Mar/Feb ratio bin
normalized trajectory cluster
```

Candidate 36 is already an example of this approach using `store_mar_feb_ratio_bin`.

## Robust Proxy Suite Needed Next

The report repeatedly stresses validation. Our largest current weakness is that the true target for residual learning is effectively one March: `2024-03`.

Implemented script:

```bash
.venv/bin/python scripts/rank_gru_audit_candidates_proxy.py
```

Outputs:

```text
results/audit_gru_robust_proxy_rank.csv
results/audit_gru_robust_proxy_skipped.csv
```

Create a robust ranker that scores each candidate by:

| Slice | Purpose |
|---|---|
| Global March-2024 OOF | Main local quality. |
| March-2025 momentum-weighted OOF | Approximate public distribution shift. |
| Weak `momentum_raw` rows | Current test is dominated by weak February momentum. |
| High `volatility_rel3` rows | Known hard segment. |
| Low/high store `Mar/Feb` ratio | Captures historical seasonality heterogeneity. |
| Region stress | Detect geographic overfit. |
| Big area and old opening stores | Known weaker OOF segments. |
| Tail penalty | Penalize candidates that make very large GRU/base corrections. |

This should become the default gate before spending submissions.

Current `robust_score_v2` ranking keeps candidate 36 as the best OOF/stress candidate among reproducible non-CatBoost-gate variants:

| Rank | Model | Global MAPE | Momentum-weighted MAPE | Robust score v2 |
|---:|---|---:|---:|---:|
| 1 | `c36_second_store_ratio_k1000_b050_c005` | `3.406647` | `3.779636` | `3.893986` |
| 2 | `c38_second_store_ratio_k1000_b050_c003` | `3.407364` | `3.780253` | `3.894719` |
| 3 | `c39_second_store_ratio_k2000_b050_c005` | `3.407701` | `3.780875` | `3.895289` |
| 7 | `old_best_b100` | `3.413660` | `3.787455` | `3.902055` |

The GRU-CatBoost gate files are intentionally skipped by this ranker until row-level CatBoost March-2024 OOF predictions are saved.

## Recommended Experiment Order

1. Submit or evaluate `stage2_candidate_36_gru_second_store_mar_feb_ratio_bin_k1000_b050_c005.csv` as the cleanest current incremental candidate.
2. Use `scripts/rank_gru_audit_candidates_proxy.py` as the default robust proxy ranker so future candidates are not judged by a single local metric.
3. Try GRU-CatBoost disagreement gates, starting with `stage2_candidate_33_gru_cb_gate_15pct_085_115.csv`.
4. Generate asymmetric momentum variants guided by public feedback, but be careful: local proxy disliked stronger q0 beta even though public might differ.
5. Rehabilitate Transformer only as a gated 5-15% correction, never as raw global blend.
6. Implement encoder pretraining on all transitions if there is time for a more radical model iteration.
7. Use NHITS/DLinear/NLinear/foundation models as diagnostic features unless they beat the robust proxy after strict shrinkage.

## Decision Summary

Stacking and boosting are worth doing, but the correct unit is not a full-forecast ensemble. The correct unit is a small, interpretable, OOF-shrunk multiplicative correction around GRU-best.

The strongest immediate path remains:

```text
current GRU-best -> robust proxy ranker -> targeted micro-corrections/gates -> selective submissions
```
