from __future__ import annotations

import numpy as np
import pandas as pd

from build_stacking_v11_candidates import add_stack_features, build_prediction_tables, crossfit_meta, feature_columns, segment_correction
from heuristic_baseline_direction_lab import RESULTS_DIR, SUBMISSIONS_DIR, TARGET, mape


OUT_DIR = RESULTS_DIR / "stacking_v11_bold_followup"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_prediction_tables()
    train = add_stack_features(train)
    test = add_stack_features(test)
    numeric, cats = feature_columns(train)
    train_corr_df, test_corr = crossfit_meta(train, test, numeric, cats)
    train_corrs = {col: train_corr_df[col].to_numpy(float) for col in train_corr_df.columns if col not in {"new_id", "fold"}}
    seg_train, seg_test = segment_correction(train, test, ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный", "momentum_train_bin", "volume_bin"], shrink_k=300.0)
    train_corrs["segment"] = seg_train
    test_corr["segment"] = seg_test

    specs = [
        (278, "huber030_g100_c005", train_corrs["huber030"], test_corr["huber030"], 1.00, 0.005),
        (279, "huber030_g100_c008", train_corrs["huber030"], test_corr["huber030"], 1.00, 0.008),
        (280, "huber030_g125_c008", train_corrs["huber030"], test_corr["huber030"], 1.25, 0.008),
        (281, "huber030_g150_c010", train_corrs["huber030"], test_corr["huber030"], 1.50, 0.010),
        (282, "avg_huber030_segment_g100_c008", 0.5 * train_corrs["huber030"] + 0.5 * train_corrs["segment"], 0.5 * test_corr["huber030"] + 0.5 * test_corr["segment"], 1.00, 0.008),
        (283, "avg_huber030_segment_g125_c010", 0.5 * train_corrs["huber030"] + 0.5 * train_corrs["segment"], 0.5 * test_corr["huber030"] + 0.5 * test_corr["segment"], 1.25, 0.010),
    ]
    y = train[TARGET].to_numpy(float)
    base_train = train["pred220"].to_numpy(float)
    base_test = test["pred220"].to_numpy(float)
    rows = []
    for candidate_id, name, tr_raw, te_raw, gamma, clip in specs:
        tr_corr = np.clip(gamma * (tr_raw - np.median(tr_raw)), -clip, clip)
        te_corr = np.clip(gamma * (te_raw - np.median(te_raw)), -clip, clip)
        pred_train = base_train * np.exp(tr_corr)
        pred_test = base_test * np.exp(te_corr)
        ratio = pred_test / base_test - 1.0
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred_test, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_full_v11_bold_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "gamma": gamma,
                "clip": clip,
                "oof_mape": mape(y, pred_train),
                "delta_vs_220_oof": mape(y, pred_train) - mape(y, base_train),
                "mean_abs_vs_220_pct": float(np.mean(np.abs(ratio)) * 100),
                "p95_abs_vs_220_pct": float(np.quantile(np.abs(ratio), 0.95) * 100),
                "max_abs_vs_220_pct": float(np.max(np.abs(ratio)) * 100),
                "mean_delta_vs_220_pct": float(np.mean(ratio) * 100),
            }
        )
    summary = pd.DataFrame(rows).sort_values("oof_mape")
    summary.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
