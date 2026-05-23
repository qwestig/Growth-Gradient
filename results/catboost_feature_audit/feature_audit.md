# CatBoost Feature Audit

## Setup

- Target: March 2024 residual over the current heuristic baseline.
- Validation: five cross-sectional folds by `new_id % 5`.
- Residual correction: `heuristic * exp(0.70 * catboost_log_residual)`, without residual clipping.
- Current feature families: level lags, current dynamics, static numeric features, categorical features, and lagged behavior features.
- Added feature families: explicit trend, last-month anomaly, prior seasonal context, recent stability, heuristic-trust, Q1 transition, alternative March-base disagreement, log-transform, first log-difference, seasonal log-difference, small second-difference, and group-level scale/dynamics features. Every added value is built from data available before the target month.

## Ablation summary

| config | features | mape_base | mape_alpha070 | mape_std | gain_vs_base | gain_vs_current |
| --- | --- | --- | --- | --- | --- | --- |
| targeted_plus_group_stats | 197 | 4.12797 | 3.46839 | 0.0926 | 0.65959 | 0.04389 |
| targeted_group_minus_behavior | 169 | 4.12797 | 3.46974 | 0.09373 | 0.65823 | 0.04253 |
| all_new_plus_group_stats | 168 | 4.12797 | 3.47107 | 0.0938 | 0.6569 | 0.0412 |
| all_new_plus_targeted | 133 | 4.12797 | 3.47214 | 0.09297 | 0.65584 | 0.04014 |
| targeted_plus_log_diff | 158 | 4.12797 | 3.47328 | 0.09223 | 0.6547 | 0.039 |
| targeted_plus_alt_bases | 157 | 4.12797 | 3.47393 | 0.09281 | 0.65404 | 0.03835 |
| targeted_plus_log_diff_accel | 163 | 4.12797 | 3.47524 | 0.09487 | 0.65274 | 0.03704 |
| heuristic_trust_plus_alt_bases | 138 | 4.12797 | 3.47583 | 0.0958 | 0.65215 | 0.03645 |
| all_new_plus_heuristic_trust | 114 | 4.12797 | 3.47588 | 0.09796 | 0.6521 | 0.0364 |
| all_new_plus_alt_bases | 128 | 4.12797 | 3.4759 | 0.09522 | 0.65207 | 0.03637 |
| targeted_minus_behavior | 105 | 4.12797 | 3.47739 | 0.09317 | 0.65058 | 0.03488 |
| targeted_log_diff_minus_behavior | 135 | 4.12797 | 3.47766 | 0.09446 | 0.65031 | 0.03461 |
| all_new_plus_log_diff | 129 | 4.12797 | 3.47907 | 0.09812 | 0.64891 | 0.03321 |
| all_new_plus_stability | 114 | 4.12797 | 3.48012 | 0.09905 | 0.64786 | 0.03216 |
| all_new_plus_q1_context | 113 | 4.12797 | 3.48081 | 0.09805 | 0.64716 | 0.03146 |
| current_plus_all_new | 104 | 4.12797 | 3.48122 | 0.10061 | 0.64676 | 0.03106 |
| all_new_minus_behavior | 76 | 4.12797 | 3.48632 | 0.09524 | 0.64165 | 0.02595 |
| current_plus_trend_anomaly | 96 | 4.12797 | 3.48799 | 0.09543 | 0.63998 | 0.02429 |
| current_plus_anomaly | 85 | 4.12797 | 3.49236 | 0.0951 | 0.63561 | 0.01991 |
| current_plus_trend | 89 | 4.12797 | 3.49258 | 0.09657 | 0.6354 | 0.0197 |
| current_plus_seasonal_context | 86 | 4.12797 | 3.5093 | 0.09295 | 0.61868 | 0.00298 |
| current_all | 78 | 4.12797 | 3.51227 | 0.09357 | 0.6157 | 0.0 |
| current_minus_behavior | 50 | 4.12797 | 3.51489 | 0.09649 | 0.61308 | -0.00262 |
| current_minus_level | 57 | 4.12797 | 3.52021 | 0.09215 | 0.60777 | -0.00793 |
| current_minus_static | 62 | 4.12797 | 3.57021 | 0.09377 | 0.55776 | -0.05794 |
| current_minus_dynamics | 65 | 4.12797 | 3.68729 | 0.10768 | 0.44069 | -0.17501 |

## Best config

Best local config: `targeted_plus_group_stats` with **197** features.

Candidate generated from this feature set:

`submissions/test_group_stats.csv`

## Top feature importances for the best config

| feature | importance |
| --- | --- |
| jump1_abs | 11.2607 |
| lag1_vs_prev5_mean | 5.6863 |
| store_momentum_vs_area | 5.5064 |
| range_rel6 | 5.3629 |
| Количество домохозяйств | 4.9862 |
| store_momentum_vs_opening | 4.7472 |
| momentum_clip_delta | 4.3522 |
| momentum_gap_abs | 4.1812 |
| volatility_rel3 | 3.7304 |
| log_slope3 | 3.4871 |
| current_feb_jan_ratio | 3.4053 |
| store_volatility_vs_area | 2.7352 |
| momentum_raw | 2.6977 |
| q1_scale_pred_to_base | 2.5368 |
| log_slope3_minus6 | 2.5052 |
| log_slope6 | 2.3866 |
| recent_level_to_year | 2.318 |
| lag1_position_prev5 | 2.2402 |
| lag1_z_prev5 | 1.664 |
| store_momentum_vs_region_area | 1.2702 |

## Notes

- `current_minus_*` rows test whether existing feature families carry useful signal.
- `current_plus_*` rows test whether a new time-series feature family adds signal beyond the current residual model.
- True two-year stability of an individual March/February coefficient cannot be validated on March 2024 without changing the validation target; March 2024 only has the prior March 2023 transition available.
