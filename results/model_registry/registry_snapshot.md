# Model Registry Snapshot

Anchor public score: 90.78
Anchor public MAPE: 4.7214609684
Anchor local pred265 OOF MAPE: 3.314574

## Best Local Rows By Family

| family_id | verdict | best_local_mape | best_local_delta | best_local_candidate |
|---|---:|---:|---:|---|
| v15_meta_residual_on_265 | best_known_public_9078_local_mape | 3.270457 | -0.044118 | huber030_g1.00_c0.012 |
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
| 1 | v15_meta_residual_on_265 |  | 3.270457 | -0.044118 |  | huber030_g1.00_c0.012 |
| 2 | v15_meta_residual_on_265 | 310 | 3.270457 | -0.044118 |  | stage2_candidate_310_v15_meta265_huber030_g1p00_c0p012.csv |
| 3 | v15_meta_residual_on_265 |  | 3.270798 | -0.043776 |  | huber100_g1.00_c0.012 |
| 4 | v15_meta_residual_on_265 | 311 | 3.270798 | -0.043776 |  | stage2_candidate_311_v15_meta265_huber100_g1p00_c0p012.csv |
| 5 | v15_meta_residual_on_265 |  | 3.270942 | -0.043632 |  | huber010_g1.00_c0.012 |
| 6 | v15_meta_residual_on_265 | 312 | 3.270942 | -0.043632 |  | stage2_candidate_312_v15_meta265_huber010_g1p00_c0p012.csv |
| 7 | v15_meta_residual_on_265 |  | 3.271148 | -0.043426 |  | huber030_g0.80_c0.012 |
| 8 | v15_meta_residual_on_265 | 313 | 3.271148 | -0.043426 |  | stage2_candidate_313_v15_meta265_huber030_g0p80_c0p012.csv |
| 9 | v15_meta_residual_on_265 |  | 3.271412 | -0.043163 |  | huber010_g0.80_c0.012 |
| 10 | v15_meta_residual_on_265 | 314 | 3.271412 | -0.043163 |  | stage2_candidate_314_v15_meta265_huber010_g0p80_c0p012.csv |
| 11 | v15_meta_residual_on_265 |  | 3.271454 | -0.043121 |  | huber100_g0.80_c0.012 |
| 12 | v15_meta_residual_on_265 | 315 | 3.271454 | -0.043121 |  | stage2_candidate_315_v15_meta265_huber100_g0p80_c0p012.csv |
| 13 | v15_meta_residual_on_265 |  | 3.274564 | -0.040010 |  | huber030_g1.00_c0.008 |
| 14 | v15_meta_residual_on_265 | 316 | 3.274564 | -0.040010 |  | stage2_candidate_316_v15_meta265_huber030_g1p00_c0p008.csv |
| 15 | v15_meta_residual_on_265 | 317 | 3.274858 | -0.039716 | 90.78 | stage2_candidate_317_v15_meta265_huber030_g0p80_c0p008.csv |
| 16 | v15_meta_residual_on_265 |  | 3.274858 | -0.039716 |  | huber030_g0.80_c0.008 |
| 17 | v15_meta_residual_on_265 |  | 3.275021 | -0.039554 |  | huber010_g1.00_c0.008 |
| 18 | v15_meta_residual_on_265 |  | 3.275067 | -0.039507 |  | huber100_g1.00_c0.008 |
| 19 | v15_meta_residual_on_265 |  | 3.275149 | -0.039426 |  | huber010_g0.80_c0.008 |
| 20 | v15_meta_residual_on_265 |  | 3.275177 | -0.039397 |  | huber100_g0.80_c0.008 |
| 21 | v15_meta_residual_on_265 |  | 3.275593 | -0.038982 |  | huber030_g0.60_c0.012 |
| 22 | v15_meta_residual_on_265 |  | 3.275659 | -0.038915 |  | huber010_g0.60_c0.012 |
| 23 | v15_meta_residual_on_265 |  | 3.275816 | -0.038758 |  | huber100_g0.60_c0.012 |
| 24 | v15_meta_residual_on_265 |  | 3.277827 | -0.036747 |  | huber030_g0.60_c0.008 |
| 25 | v15_meta_residual_on_265 |  | 3.277918 | -0.036656 |  | huber010_g0.60_c0.008 |
