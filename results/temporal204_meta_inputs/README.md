# temporal204 Meta Inputs

Row-level inputs for searching variants around the current `candidate_265` anchor.

Generate from the full workspace:

```bash
PYTHONPATH=scripts .venv/bin/python Growth-Gradient/analysis/new_baseline/temporal204/export_temporal204_meta_inputs.py --workspace-root .
```

Main outputs:

- `oof_meta_inputs_v11_v14.csv`: March-2024 OOF table with `pred204`, `pred210`, `pred215`, `pred220`, reconstructed `pred261_oof`, reconstructed `pred265_oof`, actual residuals vs `265`, v11/v14 signals, and meta-layer features.
- `test_meta_inputs_v11_v14.csv`: March-2025 test table with the same prediction stack/signals/features, plus exact submission predictions where available.
- `meta_input_oof_metrics.csv`: quick OOF MAPE check for exported prediction columns.

Important columns:

- `actual_log_residual_vs_265`: `log(target_rto / pred265_oof)`.
- `actual_pct_residual_vs_265`: `target_rto / pred265_oof - 1`.
- `v11_huber030_centered`: raw centered Huber v11 correction signal.
- `v11_segment_centered`: raw centered segment residual signal.
- `v11_combo_raw`: v14-style combo, `0.5 * centered_huber030 + 0.5 * centered_segment`.
- `v11_combo265_centered`: exact candidate-265 combo, `(0.5 * raw_huber030 + 0.5 * raw_segment)` centered by its own median.
- `v11_corr261_log`: candidate-261 style correction, `clip(0.50 * huber030, -0.005, 0.005)`.
- `v11_corr265_log`: exact candidate-265 correction, `clip(0.75 * v11_combo265_centered, -0.005, 0.005)`.
- `v14_internal_corr265_log`: v14 script's internal safe-anchor approximation.
- `v14_conf_abs_combo`: `abs(v11_combo_raw)` selective confidence.
- `v14_agree_huber_segment`: sign agreement between Huber and segment signals.
- `v14_top50_mask`: candidate-284 selective mask.
- `v14_agree_top50_mask`: candidate-285 selective mask.
- `v14_delta284_vs_265_log`: row-level extra correction of v14 candidate 284 over 265.
- `v14_delta285_vs_265_log`: row-level extra correction of v14 candidate 285 over 265.
- `v14_delta284_vs_internal265_log`: row-level extra correction of v14 candidate 284 over v14 internal safe anchor.
- `v14_delta285_vs_internal265_log`: row-level extra correction of v14 candidate 285 over v14 internal safe anchor.
- `feature__*`: row-level features used by the v11/v14 meta-layer.

Use `pred265_oof` / `pred265_submission` as the current safe anchor when testing new local corrections.
