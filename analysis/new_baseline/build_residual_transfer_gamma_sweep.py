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


def centered(log_residual: pd.Series, groups: pd.Series | None) -> pd.Series:
    if groups is None:
        return log_residual - float(log_residual.median())
    med = log_residual.groupby(groups, observed=True).transform("median")
    return log_residual - med.fillna(float(log_residual.median()))


def make_base(frame: pd.DataFrame, rto: pd.DataFrame, static: pd.DataFrame, spec: HeuristicSpec, is_test: bool) -> np.ndarray:
    val_factor, test_factor, _val_global, _final_global = factors_for_spec(spec, frame, frame, rto, static)
    factor = test_factor if is_test else val_factor
    return predict_from_spec(frame, factor, spec)


def specs() -> list[tuple[int, str, HeuristicSpec, float, str, float | None]]:
    base050 = HeuristicSpec(0, "cal050_mom000", 0.50, momentum_exp=0.0)
    base055 = HeuristicSpec(0, "cal055_mom000", 0.55, momentum_exp=0.0)
    base060 = HeuristicSpec(0, "cal060_mom000", 0.60, momentum_exp=0.0)
    return [
        (102, "cal050_mombin_g040", base050, 0.40, "momentum_bin", None),
        (103, "cal050_mombin_g060", base050, 0.60, "momentum_bin", None),
        (104, "cal050_mombin_g070", base050, 0.70, "momentum_bin", None),
        (105, "cal050_mombin_g080", base050, 0.80, "momentum_bin", None),
        (106, "cal050_mombin_g100", base050, 1.00, "momentum_bin", None),
        (107, "cal055_mombin_g050", base055, 0.50, "momentum_bin", None),
        (108, "cal060_mombin_g050", base060, 0.50, "momentum_bin", None),
        (109, "cal055_mombin_g060", base055, 0.60, "momentum_bin", None),
        (110, "cal060_mombin_g060", base060, 0.60, "momentum_bin", None),
        (111, "cal050_mombin_g060_clip015", base050, 0.60, "momentum_bin", 0.015),
        (112, "cal050_mombin_g070_clip015", base050, 0.70, "momentum_bin", 0.015),
        (113, "cal055_mombin_g060_clip015", base055, 0.60, "momentum_bin", 0.015),
    ]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val_frame = build_target_frame(rto, static, 2024, 3, include_target=True)
    test_frame = build_target_frame(rto, static, 2025, 3, include_target=False)

    old_spec = HeuristicSpec(0, "old_raw_mom025", 0.0, momentum_exp=0.25)
    old_test_base = make_base(test_frame, rto, static, old_spec, is_test=True)
    old_val_base = make_base(val_frame, rto, static, old_spec, is_test=False)

    oof = read_remote_csv("results/gru_residual/gru_oof.csv")[["new_id", "pred_log_gru"]]
    val = val_frame.merge(oof, on="new_id", how="inner")
    val["old_gru_ratio_log"] = 0.70 * val["pred_log_gru"].to_numpy(dtype=float)
    val["momentum_train_bin"] = momentum_bin(val)

    test_gru = read_remote_csv("submissions/test_gru_alpha070.csv").rename(columns={"rto": "old_gru"})
    test = test_frame.merge(test_gru, on="new_id", how="inner")
    test_old_base = pd.Series(old_test_base, index=test_frame["new_id"]).reindex(test["new_id"]).to_numpy(dtype=float)
    test["old_gru_ratio_log"] = np.log(test["old_gru"].to_numpy(dtype=float) / test_old_base)
    test["momentum_train_bin"] = momentum_bin(test)

    rows = []
    for candidate_id, name, base_spec, gamma, centering, clip in specs():
        val_base = make_base(val_frame, rto, static, base_spec, is_test=False)
        test_base = make_base(test_frame, rto, static, base_spec, is_test=True)
        val_base = pd.Series(val_base, index=val_frame["new_id"]).reindex(val["new_id"]).to_numpy(dtype=float)
        test_base = pd.Series(test_base, index=test_frame["new_id"]).reindex(test["new_id"]).to_numpy(dtype=float)

        groups_val = val["momentum_train_bin"] if centering == "momentum_bin" else None
        groups_test = test["momentum_train_bin"] if centering == "momentum_bin" else None
        val_resid = centered(val["old_gru_ratio_log"], groups_val).to_numpy(dtype=float)
        test_resid = centered(test["old_gru_ratio_log"], groups_test).to_numpy(dtype=float)
        if clip is not None:
            val_resid = np.clip(val_resid, -clip, clip)
            test_resid = np.clip(test_resid, -clip, clip)

        val_pred = val_base * np.exp(gamma * val_resid)
        test_pred = test_base * np.exp(gamma * test_resid)

        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_{name}.csv"
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(test_pred, 0), 2)
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)

        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "calendar_lambda": base_spec.calendar_lambda,
                "gamma": gamma,
                "centering": centering,
                "clip": clip if clip is not None else "",
                "val_mape_march2024": mape(val[TARGET], val_pred),
                "test_mean_rto": float(np.mean(test_pred)),
                "test_total_rto": float(np.sum(test_pred)),
                "mean_abs_change_vs_base_pct": float(np.mean(np.abs(test_pred / test_base - 1.0)) * 100),
                "p95_abs_change_vs_base_pct": float(np.quantile(np.abs(test_pred / test_base - 1.0), 0.95) * 100),
            }
        )

    summary = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "residual_transfer_gamma_sweep.csv", index=False)

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
                    "family": "residual_transfer_gamma_sweep",
                    "hypothesis": "Tune gamma/base lambda for momentum-bin-centered old-GRU residual transfer on new baseline",
                    "public_score": "",
                    "notes": "fill public_score after submission",
                    "public_mape": "",
                    "delta_mape_vs_current_best": "",
                    "delta_score_vs_current_best": "",
                }
            )
        if new_rows:
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)

    print("Wrote residual-transfer gamma sweep:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
