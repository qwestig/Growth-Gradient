from __future__ import annotations

import numpy as np
import pandas as pd

from build_stacking_v11_candidates import add_stack_features, build_prediction_tables, crossfit_meta, feature_columns
from heuristic_baseline_direction_lab import RESULTS_DIR, SUBMISSIONS_DIR, TARGET, mape


OUT_DIR = RESULTS_DIR / "stacking_v11"
SEED = 2026


def load_sub(candidate_id: int, col: str) -> pd.DataFrame:
    path = sorted(SUBMISSIONS_DIR.glob(f"stage2_candidate_{candidate_id:03d}_*.csv"))[0]
    return pd.read_csv(path).rename(columns={"rto": col})[["new_id", col]]


def bootstrap_delta(y: np.ndarray, ref: np.ndarray, pred: np.ndarray, n_boot: int = 5000) -> dict:
    rng = np.random.default_rng(SEED)
    delta = np.abs((y - pred) / y) * 100 - np.abs((y - ref) / y) * 100
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    boot = delta[idx].mean(axis=1)
    return {
        "boot_p01": float(np.quantile(boot, 0.01)),
        "boot_p05": float(np.quantile(boot, 0.05)),
        "boot_p50": float(np.quantile(boot, 0.50)),
        "boot_p95": float(np.quantile(boot, 0.95)),
        "boot_p99": float(np.quantile(boot, 0.99)),
        "prob_improve": float((boot < 0).mean()),
    }


def group_bootstrap_delta(frame: pd.DataFrame, pred_col: str, group_col: str, n_boot: int = 3000) -> dict:
    rng = np.random.default_rng(SEED)
    groups = frame[group_col].astype(str).fillna("__nan__").unique()
    group_delta = []
    group_weight = []
    for group, grp in frame.groupby(group_col, observed=True):
        delta = mape(grp[TARGET], grp[pred_col]) - mape(grp[TARGET], grp["pred220"])
        group_delta.append(delta)
        group_weight.append(len(grp))
    group_delta = np.array(group_delta, dtype=float)
    group_weight = np.array(group_weight, dtype=float)
    sampled = rng.integers(0, len(group_delta), size=(n_boot, len(group_delta)))
    boot = (group_delta[sampled] * group_weight[sampled]).sum(axis=1) / group_weight[sampled].sum(axis=1)
    return {
        "group_col": group_col,
        "groups": int(len(groups)),
        "boot_p05": float(np.quantile(boot, 0.05)),
        "boot_p50": float(np.quantile(boot, 0.50)),
        "boot_p95": float(np.quantile(boot, 0.95)),
        "prob_improve": float((boot < 0).mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_prediction_tables()
    train = add_stack_features(train)
    test = add_stack_features(test)
    numeric, cats = feature_columns(train)
    train_corr_df, _test_corr = crossfit_meta(train, test, numeric, cats)

    corr_raw = train_corr_df["huber030"].to_numpy(float)
    corr = np.clip(0.50 * (corr_raw - np.median(corr_raw)), -0.005, 0.005)
    train["pred261_oof"] = train["pred220"].to_numpy(float) * np.exp(corr)
    train["corr261_log"] = corr

    overall = {
        "mape220_oof": mape(train[TARGET], train["pred220"]),
        "mape261_oof": mape(train[TARGET], train["pred261_oof"]),
        "delta_oof": mape(train[TARGET], train["pred261_oof"]) - mape(train[TARGET], train["pred220"]),
        **bootstrap_delta(train[TARGET].to_numpy(float), train["pred220"].to_numpy(float), train["pred261_oof"].to_numpy(float)),
    }
    pd.Series(overall).to_csv(OUT_DIR / "candidate261_overall_robustness.csv")

    segments = []
    for col in ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный", "momentum_train_bin", "volume_bin"]:
        if col not in train.columns:
            continue
        for segment, grp in train.groupby(col, observed=True):
            if len(grp) < 100:
                continue
            segments.append(
                {
                    "segment_col": col,
                    "segment": segment,
                    "n": len(grp),
                    "mape220": mape(grp[TARGET], grp["pred220"]),
                    "mape261": mape(grp[TARGET], grp["pred261_oof"]),
                    "delta": mape(grp[TARGET], grp["pred261_oof"]) - mape(grp[TARGET], grp["pred220"]),
                    "mean_abs_corr_pct": float(np.mean(np.abs(np.exp(grp["corr261_log"]) - 1.0)) * 100),
                }
            )
    seg_df = pd.DataFrame(segments).sort_values(["delta", "segment_col"])
    seg_df.to_csv(OUT_DIR / "candidate261_segment_robustness.csv", index=False)

    group_boot = []
    for col in ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный", "momentum_train_bin", "volume_bin"]:
        if col in train.columns:
            group_boot.append(group_bootstrap_delta(train, "pred261_oof", col))
    pd.DataFrame(group_boot).to_csv(OUT_DIR / "candidate261_group_bootstrap.csv", index=False)

    test_diag = load_sub(220, "pred220_sub").merge(load_sub(261, "pred261"), on="new_id")
    test_diag = test_diag.merge(test[["new_id", "momentum_raw", "volatility_rel3", "dip_score", "lag1", "Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный"]], on="new_id")
    test_diag["corr261_log"] = np.log(test_diag["pred261"] / test_diag["pred220_sub"])
    test_diag["corr261_pct"] = (test_diag["pred261"] / test_diag["pred220_sub"] - 1.0) * 100
    test_summary = {
        "mean_abs_corr_pct": float(test_diag["corr261_pct"].abs().mean()),
        "p50_corr_pct": float(test_diag["corr261_pct"].quantile(0.50)),
        "p95_abs_corr_pct": float(test_diag["corr261_pct"].abs().quantile(0.95)),
        "p99_abs_corr_pct": float(test_diag["corr261_pct"].abs().quantile(0.99)),
        "max_abs_corr_pct": float(test_diag["corr261_pct"].abs().max()),
        "share_positive_corr": float((test_diag["corr261_pct"] > 0).mean()),
    }
    pd.Series(test_summary).to_csv(OUT_DIR / "candidate261_test_correction_summary.csv")
    test_diag.to_csv(OUT_DIR / "candidate261_test_correction_rows.csv", index=False)

    print("Overall robustness:")
    print(pd.Series(overall).to_string())
    print("\nWorst segments:")
    print(seg_df.sort_values("delta", ascending=False).head(15).to_string(index=False))
    print("\nGroup bootstrap:")
    print(pd.DataFrame(group_boot).to_string(index=False))
    print("\nTest correction summary:")
    print(pd.Series(test_summary).to_string())


if __name__ == "__main__":
    main()
