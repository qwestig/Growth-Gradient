from __future__ import annotations

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import (
    RESULTS_DIR,
    SUBMISSIONS_DIR,
    TARGET,
    HeuristicSpec,
    build_target_frame,
    factors_for_spec,
    load_data,
    mape,
    predict_from_spec,
)


def specs() -> list[HeuristicSpec]:
    return [
        HeuristicSpec(76, "cal045_median_mom000", 0.45, momentum_exp=0.0),
        HeuristicSpec(77, "cal055_median_mom000", 0.55, momentum_exp=0.0),
        HeuristicSpec(78, "cal060_median_mom000", 0.60, momentum_exp=0.0),
        HeuristicSpec(79, "cal0675_median_mom000", 0.675, momentum_exp=0.0),
        HeuristicSpec(80, "cal075_median_mom000", 0.75, momentum_exp=0.0),
        HeuristicSpec(81, "cal050_median_mom0050", 0.50, momentum_exp=0.05),
        HeuristicSpec(82, "cal050_median_mom0100", 0.50, momentum_exp=0.10),
        HeuristicSpec(83, "cal055_median_mom0050", 0.55, momentum_exp=0.05),
        HeuristicSpec(84, "cal060_median_mom0050", 0.60, momentum_exp=0.05),
        HeuristicSpec(85, "cal0625_median_mom0050", 0.625, momentum_exp=0.05),
        HeuristicSpec(86, "cal050_median_momneg0050", 0.50, momentum_exp=-0.05),
        HeuristicSpec(87, "cal050_median_momneg0100", 0.50, momentum_exp=-0.10),
        HeuristicSpec(88, "cal055_median_momneg0050", 0.55, momentum_exp=-0.05),
        HeuristicSpec(89, "cal060_median_momneg0050", 0.60, momentum_exp=-0.05),
        HeuristicSpec(90, "cal0625_median_momneg0050", 0.625, momentum_exp=-0.05),
        HeuristicSpec(91, "cal050_mape_wmedian_mom000", 0.50, factor_method="mape_weighted_median", momentum_exp=0.0),
        HeuristicSpec(92, "cal055_mape_wmedian_mom000", 0.55, factor_method="mape_weighted_median", momentum_exp=0.0),
        HeuristicSpec(93, "cal0625_mape_wmedian_mom000", 0.625, factor_method="mape_weighted_median", momentum_exp=0.0),
    ]


def direction_question(spec: HeuristicSpec) -> str:
    if spec.momentum_exp < 0:
        return "Does explicitly reversing weak-February momentum improve the new winning baseline?"
    if spec.momentum_exp == 0:
        return "Where is the best calendar lambda after removing the old momentum penalty?"
    if spec.momentum_exp < 0.25:
        return "Is a tiny positive momentum exponent better than zero after candidate 70?"
    return "Momentum/calendar follow-up around candidate 70"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val_frame = build_target_frame(rto, static, 2024, 3, include_target=True)
    test_frame = build_target_frame(rto, static, 2025, 3, include_target=False)

    rows = []
    for spec in specs():
        val_factor, test_factor, val_global, final_global = factors_for_spec(spec, val_frame, test_frame, rto, static)
        val_pred = predict_from_spec(val_frame, val_factor, spec)
        test_pred = predict_from_spec(test_frame, test_factor, spec)

        out = test_frame[["new_id"]].copy()
        out["rto"] = np.round(test_pred, 2)
        file_name = f"stage2_candidate_{spec.candidate_id:02d}_baseline_{spec.name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)

        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "file": file_name,
                "name": spec.name,
                "family": "momentum_zero_followup_baseline",
                "calendar_lambda": spec.calendar_lambda,
                "factor_method": spec.factor_method,
                "momentum_exp": spec.momentum_exp,
                "val_global_factor": val_global,
                "final_global_factor": final_global,
                "val_mape_march2024": mape(val_frame[TARGET], val_pred),
                "test_mean_rto": float(np.mean(test_pred)),
                "test_total_rto": float(np.sum(test_pred)),
                "direction_question": direction_question(spec),
                "public_status": "not_submitted",
                "public_score": "",
                "public_mape": "",
            }
        )

    summary = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "momentum_zero_followup_candidates.csv", index=False)

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
                    "hypothesis": row["direction_question"],
                    "public_score": "",
                    "notes": "fill public_score after submission",
                    "public_mape": "",
                    "delta_mape_vs_current_best": "",
                    "delta_score_vs_current_best": "",
                }
            )
        if new_rows:
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)

    print("Wrote momentum-zero follow-up candidates:")
    print(summary[["candidate_id", "name", "val_mape_march2024", "final_global_factor", "test_mean_rto"]].to_string(index=False))


if __name__ == "__main__":
    main()
