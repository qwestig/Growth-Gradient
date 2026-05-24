from __future__ import annotations

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import (
    RESULTS_DIR,
    SUBMISSIONS_DIR,
    HeuristicSpec,
    STATIC_CAT_COLS,
    build_target_frame,
    factors_for_spec,
    load_data,
)
from search_structural_baselines_v6 import (
    CURRENT_LAMBDA,
    blend_anchor,
    current_198_march,
    group_total_correction,
    two_month_temporal_anchor,
)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val = build_target_frame(rto, static, 2024, 3, include_target=True)
    test = build_target_frame(rto, static, 2025, 3, include_target=False)
    spec = HeuristicSpec(0, "current", CURRENT_LAMBDA, momentum_exp=0.0)
    _vf, _tf, _val_factor, final_factor = factors_for_spec(spec, val, test, rto, static)
    current_pred = current_198_march(test, final_factor, target_non_leap=True)
    current_198 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_198_baseline_v4_followup_signasym_lam0550_wp02_wn14_cp002_cn003.csv").rename(columns={"rto": "rto_198"})

    candidate_specs = [
        {
            "candidate_id": 203,
            "family": "temporal_two_month_growth",
            "name": "temporal2_raw_clip085_135_a012_lc004",
            "daily_adjust": False,
            "clip_lo": 0.85,
            "clip_hi": 1.35,
            "alpha": 0.12,
            "log_clip": 0.04,
            "selection_reason": "best March proxy for temporal/Q1-style two-month growth anchor",
        },
        {
            "candidate_id": 204,
            "family": "temporal_two_month_growth",
            "name": "temporal2_raw_clip085_135_a012_lc006",
            "daily_adjust": False,
            "clip_lo": 0.85,
            "clip_hi": 1.35,
            "alpha": 0.12,
            "log_clip": 0.06,
            "selection_reason": "best multi-period temporal/Q1-style two-month growth anchor",
        },
        {
            "candidate_id": 205,
            "family": "hierarchical_group_total_correction",
            "name": "group_opening_total_a035_lc001_k1000",
            "group_col": "Дата открытия, категориальный",
            "alpha": 0.35,
            "log_clip": 0.01,
            "shrink_k": 1000,
            "selection_reason": "best lightweight hierarchical group-total diagnostic; lower priority than temporal candidates",
        },
    ]

    rows = []
    for cfg in candidate_specs:
        if cfg["family"] == "temporal_two_month_growth":
            anchor = two_month_temporal_anchor(rto, test, 2025, 3, cfg["daily_adjust"], cfg["clip_lo"], cfg["clip_hi"])
            pred = blend_anchor(current_pred, anchor, cfg["alpha"], cfg["log_clip"])
        else:
            pred = group_total_correction(
                test,
                current_pred,
                cfg["group_col"],
                2025,
                3,
                [2023, 2024],
                rto,
                cfg["alpha"],
                cfg["log_clip"],
                cfg["shrink_k"],
            )

        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0.0), 2)
        file_name = f"stage2_candidate_{cfg['candidate_id']:03d}_baseline_v6_{cfg['name']}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        merged = out.merge(current_198, on="new_id")
        ratio = merged["rto"].to_numpy(float) / merged["rto_198"].to_numpy(float) - 1.0
        row = {
            "candidate_id": cfg["candidate_id"],
            "file": file_name,
            "family": cfg["family"],
            "name": cfg["name"],
            "selection_reason": cfg["selection_reason"],
            "test_mean_rto": float(np.mean(pred)),
            "test_total_rto": float(np.sum(pred)),
            "delta_mean_rto_vs_198_pct": float((np.mean(pred) / merged["rto_198"].mean() - 1.0) * 100),
            "mean_abs_change_vs_198_pct": float(np.mean(np.abs(ratio)) * 100),
            "p95_abs_change_vs_198_pct": float(np.quantile(np.abs(ratio), 0.95) * 100),
            "max_abs_change_vs_198_pct": float(np.max(np.abs(ratio)) * 100),
        }
        row.update({k: v for k, v in cfg.items() if k not in row})
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "structural_baselines_v6_candidates.csv", index=False)

    log_path = RESULTS_DIR / "public_experiment_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        existing = set(log["candidate_id"].astype(str))
        new_rows = []
        for _, row in summary.iterrows():
            if str(row["candidate_id"]) in existing:
                continue
            new_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "file": row["file"],
                    "family": row["family"],
                    "hypothesis": row["selection_reason"],
                    "public_score": "",
                    "notes": "fill public_score after submission",
                    "public_mape": "",
                    "delta_mape_vs_current_best": "",
                    "delta_score_vs_current_best": "",
                }
            )
        if new_rows:
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
