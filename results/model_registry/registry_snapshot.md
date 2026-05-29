# Model Registry Snapshot

Anchor public score: 90.78
Anchor public MAPE: 4.7214609684
Anchor local pred265 OOF MAPE: 3.314574

## Best Local Rows By Family

| family_id | verdict | best_local_mape | best_local_delta | best_local_candidate |
|---|---:|---:|---:|---|
| micro_overlays_on_265 | public_neutral | 3.299217 | -0.015357 | v14_284_plus_feb410_a020_c008 |
| multimonth_gru_seasonal | tested_but_not_final_anchor | 3.304390 | -0.010184 | none_a0.25_c0.020_reg_cov0.65 |
| multimonth_gru | tested_but_not_final_anchor | 3.308853 | -0.005722 | q0.40_a0.30_c0.003 |
| similarity_overlays_on_265 | candidate_not_yet_public | 3.309672 | -0.004903 | v14_284_knn50_low40_a005_c005 |
| feb_anomaly_gru | public_failed | 3.312633 | nan |  |
| cross_similarity_expert | small_local_signal | 3.313853 | -0.000721 | knn_seasonal_pred_k20__a0.20__c0.003__close_low_std |
| anchor_temporal204_v11_gru_stack | current_best_public_anchor | 3.314574 | 0.000000 | pred265_oof / stage2 candidate 265 |
| hierarchical_cohort_expert | rejected_locally | 3.316120 | 0.001546 | cohort_share_pred__a0.15__c0.003__large_shift_stable |
| early_gru_residual | kept_for_history | 3.330653 | nan |  |
| alternative_heuristic_lab | rejected_locally | 3.337217 | 0.022643 | topdown_Регион_growth_feb_share_feb__blend_a0.25_c0.006 |
| catboost_residual | not_anchor | 3.409351 | nan |  |
| conditional_feb_anomaly | promising_as_bare_heuristic_but_failed_after_gru | 3.654878 | -0.193523 | feb_anom_trend_dec_jan_low_after_stable_t0p13_w1p0_cal0p6 |
| lifecycle_heuristic | rejected_locally | 3.840006 | 0.525432 | lifecycle_ensemble_top2 |
| march_share_q1 | rejected_locally | 4.044685 | 0.196284 | q1share_ensemble_top5 |
| shrunk_yoy_ml_residual | rejected_locally | 4.105828 | 0.791254 | ridge_a1.0_c0.1 |
| share_of_market | rejected_locally | 4.149334 | 0.834759 | som_ensemble_top7 |
| deseasonalized_normal_level | rejected_locally | 4.223340 | 0.374939 | deseason_feb_only_city_big_k1000_sw0p0 |
| shrunk_yoy_cohort | rejected_locally | 5.296252 | 1.981678 | shrunk_yoy_feb_day_adjusted_age_const1.00_linear_k20_c0p3 |
| legacy_stage2_stack_201_211 | historically_useful_but_obsolete |  |  |  |

## Top 25 Indexed Candidate Rows By Local MAPE

| rank | family_id | candidate_id | local_mape | local_delta | public_score | file/candidate |
|---:|---|---:|---:|---:|---:|---|
| 1 | micro_overlays_on_265 | 472 | 3.299217 | -0.015357 | 90.78 | stage2_candidate_472_micro265_v14_284_plus_feb410_a020_c008.csv |
| 2 | micro_overlays_on_265 | 474 | 3.299274 | -0.015300 |  | stage2_candidate_474_micro265_v14_285_plus_feb410_a020_c008.csv |
| 3 | micro_overlays_on_265 | 473 | 3.299985 | -0.014590 |  | stage2_candidate_473_micro265_v14_284_plus_feb410_a020_c005.csv |
| 4 | micro_overlays_on_265 | 475 | 3.299989 | -0.014585 |  | stage2_candidate_475_micro265_v14_284_plus_feb430_a020_c008.csv |
| 5 | micro_overlays_on_265 | 467 | 3.304258 | -0.010316 | 90.78 | stage2_candidate_467_micro265_v14_284_plus_feb410_a010_c005.csv |
| 6 | micro_overlays_on_265 | 468 | 3.304361 | -0.010214 |  | stage2_candidate_468_micro265_v14_285_plus_feb410_a010_c005.csv |
| 7 | multimonth_gru_seasonal |  | 3.304390 | -0.010184 |  | none_a0.25_c0.020_reg_cov0.65 |
| 8 | micro_overlays_on_265 | 471 | 3.304502 | -0.010073 |  | stage2_candidate_471_micro265_v14_284_plus_feb450_a010_c005.csv |
| 9 | micro_overlays_on_265 | 470 | 3.304598 | -0.009977 |  | stage2_candidate_470_micro265_v14_284_plus_feb430_a010_c005.csv |
| 10 | multimonth_gru_seasonal |  | 3.304740 | -0.009834 |  | none_a0.25_c0.020_reg_cov0.80 |
| 11 | micro_overlays_on_265 | 469 | 3.304785 | -0.009789 |  | stage2_candidate_469_micro265_v14_284_plus_feb410_a010_c003.csv |
| 12 | multimonth_gru_seasonal |  | 3.305031 | -0.009543 |  | none_a0.25_c0.020_prob_cov0.80 |
| 13 | multimonth_gru_seasonal |  | 3.305234 | -0.009340 |  | none_a0.25_c0.020_hybrid_cov0.80 |
| 14 | multimonth_gru_seasonal |  | 3.305237 | -0.009338 |  | none_a0.25_c0.020_reg_cov0.50 |
| 15 | multimonth_gru_seasonal |  | 3.305328 | -0.009247 |  | none_a0.25_c0.020_hybrid_cov0.65 |
| 16 | multimonth_gru_seasonal |  | 3.305328 | -0.009246 |  | median_a0.25_c0.020_reg_cov0.65 |
| 17 | multimonth_gru_seasonal |  | 3.305381 | -0.009193 |  | median_a0.25_c0.020_reg_cov0.80 |
| 18 | multimonth_gru_seasonal |  | 3.305411 | -0.009164 |  | none_a0.25_c0.020_hybrid_cov0.35 |
| 19 | multimonth_gru_seasonal |  | 3.305575 | -0.009000 |  | median_a0.20_c0.020_reg_cov0.80 |
| 20 | multimonth_gru_seasonal |  | 3.305658 | -0.008917 |  | median_a0.25_c0.020_reg_cov0.50 |
| 21 | multimonth_gru_seasonal |  | 3.305781 | -0.008793 |  | none_a0.25_c0.020_hybrid_cov0.50 |
| 22 | multimonth_gru_seasonal |  | 3.305798 | -0.008776 |  | median_a0.20_c0.012_reg_cov0.65 |
| 23 | multimonth_gru_seasonal |  | 3.305828 | -0.008747 |  | none_a0.25_c0.012_reg_cov0.80 |
| 24 | multimonth_gru_seasonal |  | 3.305902 | -0.008672 |  | none_a0.20_c0.020_reg_cov0.50 |
| 25 | multimonth_gru_seasonal |  | 3.305946 | -0.008628 |  | none_a0.25_c0.020_reg_cov0.35 |
