from __future__ import annotations

import subprocess
from io import StringIO

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


REMOTE_BRANCH = "origin/codex-momentum-bias-solution"
MOMENTUM_TRAIN_BINS = [
    -np.inf,
    0.935766244455909,
    0.9524077370639855,
    0.9656151000316019,
    0.9815696834781942,
    np.inf,
]


def read_remote_csv(path: str) -> pd.DataFrame:
    result = subprocess.run(
        ["git", "show", f"{REMOTE_BRANCH}:{path}"],
        cwd=RESULTS_DIR.parents[0],
        check=True,
        text=True,
        capture_output=True,
    )
    return pd.read_csv(StringIO(result.stdout))


def momentum_bin(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame["momentum_raw"],
        bins=MOMENTUM_TRAIN_BINS,
        labels=[f"q{i}" for i in range(5)],
        include_lowest=True,
    ).astype(str)


def centered_log_residual(log_residual: pd.Series, groups: pd.Series | None) -> pd.Series:
    if groups is None:
        return log_residual - float(log_residual.median())
    med = log_residual.groupby(groups, observed=True).transform("median")
    return log_residual - med.fillna(float(log_residual.median()))


def build_baseline(frame: pd.DataFrame, rto: pd.DataFrame, static: pd.DataFrame, spec: HeuristicSpec, is_test: bool) -> tuple[np.ndarray, float]:
    # factors_for_spec needs both frames but returns the requested side. Use the same frame for the unused side.
    val_factor, test_factor, val_global, final_global = factors_for_spec(spec, frame, frame, rto, static)
    factor = test_factor if is_test else val_factor
    global_factor = final_global if is_test else val_global
    return predict_from_spec(frame, factor, spec), global_factor


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val_frame = build_target_frame(rto, static, 2024, 3, include_target=True)
    test_frame = build_target_frame(rto, static, 2025, 3, include_target=False)

    old_spec = HeuristicSpec(0, "old_raw_mom025", 0.0, momentum_exp=0.25)
    new_specs = {
        "cal050_mom000": HeuristicSpec(0, "cal050_mom000", 0.50, momentum_exp=0.0),
        "cal060_mom000": HeuristicSpec(0, "cal060_mom000", 0.60, momentum_exp=0.0),
    }

    old_val_base, _ = build_baseline(val_frame, rto, static, old_spec, is_test=False)
    old_test_base, _ = build_baseline(test_frame, rto, static, old_spec, is_test=True)

    oof = read_remote_csv("results/gru_residual/gru_oof.csv")[["new_id", "pred_log_gru"]]
    val = val_frame.merge(oof, on="new_id", how="inner")
    val_old_base = pd.Series(old_val_base, index=val_frame["new_id"]).reindex(val["new_id"]).to_numpy(dtype=float)
    val["old_gru_ratio_log"] = 0.70 * val["pred_log_gru"].to_numpy(dtype=float)
    val["momentum_train_bin"] = momentum_bin(val)

    test_gru = read_remote_csv("submissions/test_gru_alpha070.csv").rename(columns={"rto": "old_gru"})
    test = test_frame.merge(test_gru, on="new_id", how="inner")
    test_old_base = pd.Series(old_test_base, index=test_frame["new_id"]).reindex(test["new_id"]).to_numpy(dtype=float)
    test["old_gru_ratio_log"] = np.log(test["old_gru"].to_numpy(dtype=float) / test_old_base)
    test["momentum_train_bin"] = momentum_bin(test)

    specs = [
        (94, "cal050_mom000_gru_global_g010", "cal050_mom000", 0.10, "global"),
        (95, "cal050_mom000_gru_global_g020", "cal050_mom000", 0.20, "global"),
        (96, "cal050_mom000_gru_global_g035", "cal050_mom000", 0.35, "global"),
        (97, "cal050_mom000_gru_mombin_g020", "cal050_mom000", 0.20, "momentum_bin"),
        (98, "cal050_mom000_gru_mombin_g035", "cal050_mom000", 0.35, "momentum_bin"),
        (99, "cal060_mom000_gru_global_g020", "cal060_mom000", 0.20, "global"),
        (100, "cal060_mom000_gru_mombin_g020", "cal060_mom000", 0.20, "momentum_bin"),
        (101, "cal050_mom000_gru_mombin_g050", "cal050_mom000", 0.50, "momentum_bin"),
    ]

    rows = []
    for candidate_id, name, base_key, gamma, centering in specs:
        new_val_base, _ = build_baseline(val_frame, rto, static, new_specs[base_key], is_test=False)
        new_test_base, _ = build_baseline(test_frame, rto, static, new_specs[base_key], is_test=True)
        val_new_base = pd.Series(new_val_base, index=val_frame["new_id"]).reindex(val["new_id"]).to_numpy(dtype=float)
        test_new_base = pd.Series(new_test_base, index=test_frame["new_id"]).reindex(test["new_id"]).to_numpy(dtype=float)

        if centering == "momentum_bin":
            val_centered = centered_log_residual(val["old_gru_ratio_log"], val["momentum_train_bin"])
            test_centered = centered_log_residual(test["old_gru_ratio_log"], test["momentum_train_bin"])
        else:
            val_centered = centered_log_residual(val["old_gru_ratio_log"], None)
            test_centered = centered_log_residual(test["old_gru_ratio_log"], None)

        val_pred = val_new_base * np.exp(gamma * val_centered.to_numpy(dtype=float))
        test_pred = test_new_base * np.exp(gamma * test_centered.to_numpy(dtype=float))

        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(test_pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)

        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "base_key": base_key,
                "gamma": gamma,
                "centering": centering,
                "val_mape_march2024": mape(val[TARGET], val_pred),
                "test_mean_rto": float(np.mean(test_pred)),
                "test_total_rto": float(np.sum(test_pred)),
                "mean_abs_change_vs_new_base_pct": float(np.mean(np.abs(test_pred / test_new_base - 1.0)) * 100),
                "p95_abs_change_vs_new_base_pct": float(np.quantile(np.abs(test_pred / test_new_base - 1.0), 0.95) * 100),
            }
        )

    summary = pd.DataFrame(rows).sort_values("val_mape_march2024").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "new_baseline_residual_transfer_candidates.csv", index=False)

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
                    "family": "new_baseline_residual_transfer",
                    "hypothesis": "Can a small centered old-GRU residual improve the new calendar/momentum-zero baseline?",
                    "public_score": "",
                    "notes": "fill public_score after submission",
                    "public_mape": "",
                    "delta_mape_vs_current_best": "",
                    "delta_score_vs_current_best": "",
                }
            )
        if new_rows:
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)

    print("Wrote residual-transfer candidates:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
