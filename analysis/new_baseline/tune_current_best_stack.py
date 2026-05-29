from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[2]
REMOTE_OLD_GRU_BRANCH = "origin/codex-momentum-bias-solution"
NEW_BASE_SPEC = HeuristicSpec(0, "cal050_mom000", 0.50, momentum_exp=0.0)
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
        ["git", "show", f"{REMOTE_OLD_GRU_BRANCH}:{path}"],
        cwd=ROOT,
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


def center_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    med = values.groupby(groups, observed=True).transform("median")
    return values - med.fillna(float(values.median()))


def build_base_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    _raw, rto, static = load_data()
    train = build_target_frame(rto, static, 2024, 3, include_target=True).sort_values("new_id").reset_index(drop=True)
    test = build_target_frame(rto, static, 2025, 3, include_target=False).sort_values("new_id").reset_index(drop=True)
    train_factor, test_factor, _val_global, _final_global = factors_for_spec(NEW_BASE_SPEC, train, test, rto, static)
    train["new_base"] = predict_from_spec(train, train_factor, NEW_BASE_SPEC)
    test["new_base"] = predict_from_spec(test, test_factor, NEW_BASE_SPEC)
    train["momentum_train_bin"] = momentum_bin(train)
    test["momentum_train_bin"] = momentum_bin(test)
    return train, test


def ensure_fresh_gru_oof() -> pd.DataFrame:
    oof_path = RESULTS_DIR / "new_baseline_gru_residual" / "oof.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            "Missing results/new_baseline_gru_residual/oof.csv. "
            "Run analysis/new_baseline/train_new_baseline_gru_residual.py first."
        )
    return pd.read_csv(oof_path)[["new_id", "pred_log_gru"]].rename(columns={"pred_log_gru": "fresh_log"})


def old_gru_oof() -> pd.DataFrame:
    return read_remote_csv("results/gru_residual/gru_oof.csv")[["new_id", "pred_log_gru"]].rename(
        columns={"pred_log_gru": "old_log"}
    )


def build_fresh_test_log() -> pd.Series:
    final_best = pd.read_csv(SUBMISSIONS_DIR / "test_best_cal050_gru_blend_9070.csv").sort_values("new_id")
    transfer = build_transfer_test_prediction(gamma=1.0)
    # current_best = 0.75 * fresh150 + 0.25 * transfer118
    fresh = (final_best["rto"].to_numpy(dtype=float) - 0.25 * transfer["rto"].to_numpy(dtype=float)) / 0.75
    test = transfer[["new_id", "new_base"]].copy()
    test["fresh_pred"] = fresh
    return pd.Series(np.log(test["fresh_pred"].to_numpy(dtype=float) / test["new_base"].to_numpy(dtype=float)), index=test["new_id"])


def build_transfer_test_prediction(gamma: float) -> pd.DataFrame:
    _train, test = build_base_frames()
    test_gru = read_remote_csv("submissions/test_gru_alpha070.csv").rename(columns={"rto": "old_gru"})
    old_spec = HeuristicSpec(0, "old_raw_mom025", 0.0, momentum_exp=0.25)
    _raw, rto, static = load_data()
    _val_factor, test_factor, _val_global, _final_global = factors_for_spec(old_spec, test, test, rto, static)
    old_base = predict_from_spec(test, test_factor, old_spec)
    frame = test.merge(test_gru, on="new_id", how="inner").sort_values("new_id").reset_index(drop=True)
    frame["old_ratio_log"] = np.log(frame["old_gru"].to_numpy(dtype=float) / old_base)
    frame["transfer_log"] = center_by_group(frame["old_ratio_log"], frame["momentum_train_bin"])
    frame["rto"] = frame["new_base"].to_numpy(dtype=float) * np.exp(gamma * frame["transfer_log"].to_numpy(dtype=float))
    return frame[["new_id", "new_base", "rto"]]


def build_train_components() -> pd.DataFrame:
    train, _test = build_base_frames()
    fresh = ensure_fresh_gru_oof()
    old = old_gru_oof()
    frame = train[["new_id", TARGET, "new_base", "momentum_train_bin"]].merge(fresh, on="new_id").merge(old, on="new_id")
    frame["fresh_global_log"] = frame["fresh_log"] - float(frame["fresh_log"].median())
    frame["old_ratio_log"] = 0.70 * frame["old_log"]
    frame["transfer_log"] = center_by_group(frame["old_ratio_log"], frame["momentum_train_bin"])
    return frame


def make_predictions(base: np.ndarray, fresh_log: np.ndarray, transfer_log: np.ndarray, fresh_alpha: float, transfer_gamma: float, transfer_weight: float) -> np.ndarray:
    fresh_pred = base * np.exp(fresh_alpha * fresh_log)
    transfer_pred = base * np.exp(transfer_gamma * transfer_log)
    return (1 - transfer_weight) * fresh_pred + transfer_weight * transfer_pred


def main() -> None:
    out_dir = RESULTS_DIR / "new_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    train = build_train_components()
    y = train[TARGET].to_numpy(dtype=float)
    base = train["new_base"].to_numpy(dtype=float)
    fresh_log = train["fresh_global_log"].to_numpy(dtype=float)
    transfer_log = train["transfer_log"].to_numpy(dtype=float)

    configs = [
        (201, "fresh100_transfer_g100_w010", 1.00, 1.00, 0.10),
        (202, "fresh100_transfer_g100_w015", 1.00, 1.00, 0.15),
        (203, "fresh100_transfer_g110_w015", 1.00, 1.10, 0.15),
        (204, "fresh100_transfer_g115_w015", 1.00, 1.15, 0.15),
        (205, "fresh100_transfer_g115_w020", 1.00, 1.15, 0.20),
        (206, "fresh0975_transfer_g110_w015", 0.975, 1.10, 0.15),
        (207, "fresh1025_transfer_g110_w015", 1.025, 1.10, 0.15),
    ]

    # Reconstruct test-side fresh log from the committed public-best blend and generated transfer component.
    transfer_test = build_transfer_test_prediction(gamma=1.0)
    fresh_test_log = build_fresh_test_log()
    test = transfer_test[["new_id", "new_base"]].copy()
    test["fresh_log"] = test["new_id"].map(fresh_test_log).to_numpy(dtype=float)
    # Build a reusable transfer log at gamma=1.0; other gamma values scale this log.
    test["transfer_log"] = np.log(transfer_test["rto"].to_numpy(dtype=float) / test["new_base"].to_numpy(dtype=float))
    current = pd.read_csv(SUBMISSIONS_DIR / "test_best_cal050_gru_blend_9070.csv").rename(columns={"rto": "current_rto"})

    rows = []
    for candidate_id, name, fresh_alpha, transfer_gamma, transfer_weight in configs:
        train_pred = make_predictions(base, fresh_log, transfer_log, fresh_alpha, transfer_gamma, transfer_weight)
        test_pred = make_predictions(
            test["new_base"].to_numpy(dtype=float),
            test["fresh_log"].to_numpy(dtype=float),
            test["transfer_log"].to_numpy(dtype=float),
            fresh_alpha,
            transfer_gamma,
            transfer_weight,
        )
        submission = test[["new_id"]].copy()
        submission["rto"] = np.round(np.maximum(test_pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id}_cal050_stack_{name}.csv"
        submission.to_csv(SUBMISSIONS_DIR / file_name, index=False)

        merged = submission.merge(current, on="new_id")
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "fresh_alpha": fresh_alpha,
                "transfer_gamma": transfer_gamma,
                "transfer_weight": transfer_weight,
                "proxy_mape": mape(y, train_pred),
                "test_mean_rto": float(submission["rto"].mean()),
                "mean_ratio_vs_current": float((merged["rto"] / merged["current_rto"]).mean()),
                "mean_abs_delta_pct_vs_current": float(
                    (np.abs(merged["rto"] - merged["current_rto"]) / merged["current_rto"] * 100).mean()
                ),
                "p95_abs_delta_pct_vs_current": float(
                    (np.abs(merged["rto"] - merged["current_rto"]) / merged["current_rto"] * 100).quantile(0.95)
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values("proxy_mape").reset_index(drop=True)
    summary.to_csv(out_dir / "current_best_stack_tuning_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
