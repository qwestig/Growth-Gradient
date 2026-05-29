from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "new_baseline_multimonth_gru"
SUBMISSIONS_DIR = ROOT / "submissions"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0)


def main() -> None:
    oof = pd.read_csv(RESULTS_DIR / "oof.csv")
    test = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    y = oof["target_rto"].to_numpy(float)
    base_oof = oof["pred265_oof"].to_numpy(float)
    raw_oof = oof["pred_mmgru_raw"].to_numpy(float)
    base_test = test["pred265_submission"].to_numpy(float)
    raw_test = test["pred_mmgru_raw"].to_numpy(float)

    delta_oof = np.log(np.clip(raw_oof, 1.0, None) / np.clip(base_oof, 1.0, None))
    delta_test = np.log(np.clip(raw_test, 1.0, None) / np.clip(base_test, 1.0, None))
    delta_oof = delta_oof - np.nanmedian(delta_oof)
    delta_test = delta_test - np.nanmedian(delta_test)
    base_mape = mape(y, base_oof)

    rows = []
    preds = {}
    for q in [0.20, 0.40, 0.50, 0.60, 0.70]:
        threshold = float(np.quantile(np.abs(delta_oof), q))
        oof_mask = np.abs(delta_oof) >= threshold
        test_threshold = float(np.quantile(np.abs(delta_test), q))
        test_mask = np.abs(delta_test) >= test_threshold
        for alpha in [0.15, 0.20, 0.25, 0.30, 0.40]:
            for clip in [0.002, 0.003, 0.005, 0.008]:
                corr_oof = np.where(oof_mask, np.clip(alpha * delta_oof, -clip, clip), 0.0)
                corr_test = np.where(test_mask, np.clip(alpha * delta_test, -clip, clip), 0.0)
                pred_oof = base_oof * np.exp(corr_oof)
                pred_test = base_test * np.exp(corr_test)
                ratio = pred_test / base_test - 1.0
                name = f"q{q:.2f}_a{alpha:.2f}_c{clip:.3f}"
                preds[name] = pred_test
                rows.append(
                    {
                        "name": name,
                        "q": q,
                        "alpha": alpha,
                        "clip": clip,
                        "oof_mape": mape(y, pred_oof),
                        "delta_vs_265_oof": mape(y, pred_oof) - base_mape,
                        "oof_coverage": float(oof_mask.mean()),
                        "test_coverage": float(test_mask.mean()),
                        "test_mean_abs_shift_pct": float(np.mean(np.abs(ratio)) * 100),
                        "test_max_abs_shift_pct": float(np.max(np.abs(ratio)) * 100),
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["delta_vs_265_oof", "test_mean_abs_shift_pct"]).reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "gated_candidates_summary.csv", index=False)

    selected = summary.head(8)
    out_rows = []
    for candidate_id, (_, row) in enumerate(selected.iterrows(), start=360):
        pred = preds[row["name"]]
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = (
            f"stage2_candidate_{candidate_id}_multimonth_gru_gate_"
            f"q{str(row['q']).replace('.', 'p')}_a{str(row['alpha']).replace('.', 'p')}_c{str(row['clip']).replace('.', 'p')}.csv"
        )
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        out_rows.append({"candidate_id": candidate_id, "file": file_name, **row.to_dict()})

    submissions = pd.DataFrame(out_rows)
    submissions.to_csv(RESULTS_DIR / "gated_submission_candidates.csv", index=False)
    print("Gated candidates:")
    print(submissions.to_string(index=False))


if __name__ == "__main__":
    main()
