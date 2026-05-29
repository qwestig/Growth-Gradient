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
MOMENTUM_TRAIN_BINS = [-np.inf, 0.935766244455909, 0.9524077370639855, 0.9656151000316019, 0.9815696834781942, np.inf]


def read_remote_csv(path: str) -> pd.DataFrame:
    local_path = RESULTS_DIR.parents[0] / path
    if local_path.exists():
        return pd.read_csv(local_path)
    result = subprocess.run(
        ["git", "show", f"{REMOTE_BRANCH}:{path}"],
        cwd=RESULTS_DIR.parents[0],
        check=True,
        text=True,
        capture_output=True,
    )
    return pd.read_csv(StringIO(result.stdout))


def momentum_bin(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame["momentum_raw"], bins=MOMENTUM_TRAIN_BINS, labels=[f"q{i}" for i in range(5)], include_lowest=True).astype(str)


def centered(log_residual: pd.Series, groups: pd.Series) -> pd.Series:
    med = log_residual.groupby(groups, observed=True).transform("median")
    return log_residual - med.fillna(float(log_residual.median()))


def make_base(frame: pd.DataFrame, rto: pd.DataFrame, static: pd.DataFrame, spec: HeuristicSpec, is_test: bool) -> np.ndarray:
    val_factor, test_factor, _val_global, _final_global = factors_for_spec(spec, frame, frame, rto, static)
    return predict_from_spec(frame, test_factor if is_test else val_factor, spec)


def candidate_specs() -> list[tuple[int, str, HeuristicSpec, float, float | None]]:
    base050 = HeuristicSpec(0, "cal050_mom000", 0.50, momentum_exp=0.0)
    base055 = HeuristicSpec(0, "cal055_mom000", 0.55, momentum_exp=0.0)
    return [
        (114, "cal050_mombin_g075", base050, 0.75, None),
        (115, "cal050_mombin_g085", base050, 0.85, None),
        (116, "cal050_mombin_g090", base050, 0.90, None),
        (117, "cal050_mombin_g095", base050, 0.95, None),
        (118, "cal050_mombin_g100", base050, 1.00, None),
        (119, "cal055_mombin_g070", base055, 0.70, None),
        (120, "cal055_mombin_g080", base055, 0.80, None),
        (121, "cal050_mombin_g080_clip020", base050, 0.80, 0.020),
        (122, "cal050_mombin_g090_clip020", base050, 0.90, 0.020),
        (123, "cal050_mombin_g100_clip020", base050, 1.00, 0.020),
    ]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val_frame = build_target_frame(rto, static, 2024, 3, include_target=True)
    test_frame = build_target_frame(rto, static, 2025, 3, include_target=False)

    old_spec = HeuristicSpec(0, "old_raw_mom025", 0.0, momentum_exp=0.25)
    old_test_base = make_base(test_frame, rto, static, old_spec, is_test=True)

    oof = read_remote_csv("results/gru_residual/gru_oof.csv")[["new_id", "pred_log_gru"]]
    val = val_frame.merge(oof, on="new_id", how="inner")
    val["old_gru_ratio_log"] = 0.70 * val["pred_log_gru"].to_numpy(dtype=float)
    val["momentum_train_bin"] = momentum_bin(val)

    test_gru = read_remote_csv("submissions/test_gru_alpha070.csv").rename(columns={"rto": "old_gru"})
    test = test_frame.merge(test_gru, on="new_id", how="inner")
    test_old_base = pd.Series(old_test_base, index=test_frame["new_id"]).reindex(test["new_id"]).to_numpy(dtype=float)
    test["old_gru_ratio_log"] = np.log(test["old_gru"].to_numpy(dtype=float) / test_old_base)
    test["momentum_train_bin"] = momentum_bin(test)

    val_resid_raw = centered(val["old_gru_ratio_log"], val["momentum_train_bin"]).to_numpy(dtype=float)
    test_resid_raw = centered(test["old_gru_ratio_log"], test["momentum_train_bin"]).to_numpy(dtype=float)

    rows = []
    for candidate_id, name, base_spec, gamma, clip in candidate_specs():
        val_base = make_base(val_frame, rto, static, base_spec, is_test=False)
        test_base = make_base(test_frame, rto, static, base_spec, is_test=True)
        val_base = pd.Series(val_base, index=val_frame["new_id"]).reindex(val["new_id"]).to_numpy(dtype=float)
        test_base = pd.Series(test_base, index=test_frame["new_id"]).reindex(test["new_id"]).to_numpy(dtype=float)
        val_resid = val_resid_raw.copy()
        test_resid = test_resid_raw.copy()
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
                "clip": clip if clip is not None else "",
                "val_mape_march2024": mape(val[TARGET], val_pred),
                "test_mean_rto": float(np.mean(test_pred)),
                "test_total_rto": float(np.sum(test_pred)),
                "mean_abs_change_vs_base_pct": float(np.mean(np.abs(test_pred / test_base - 1.0)) * 100),
                "p95_abs_change_vs_base_pct": float(np.quantile(np.abs(test_pred / test_base - 1.0), 0.95) * 100),
            }
        )

    summary = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "gamma_high_sweep_candidates.csv", index=False)

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
                    "family": "gamma_high_sweep",
                    "hypothesis": "Find public optimum above gamma 0.70 for mombin-centered residual transfer",
                    "public_score": "",
                    "notes": "fill public_score after submission",
                    "public_mape": "",
                    "delta_mape_vs_current_best": "",
                    "delta_score_vs_current_best": "",
                }
            )
        if new_rows:
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)

    print("Wrote high-gamma sweep candidates:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
