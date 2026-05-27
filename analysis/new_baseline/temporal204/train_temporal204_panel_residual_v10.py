from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from heuristic_baseline_direction_lab import RESULTS_DIR, STATIC_CAT_COLS, STATIC_NUM_COLS, SUBMISSIONS_DIR, TARGET, build_target_frame, load_data, mape
from search_strategy_v7_lab import CURRENT_LAMBDA, label, temporal204
from validate_baseline_heuristics_multiperiod import historical_factor


OUT_DIR = RESULTS_DIR / "temporal204_panel_residual_v10"
SEED = 2026
TRAIN_PERIODS = [(2024, m) for m in range(3, 13)] + [(2025, 1), (2025, 2)]


def add_features(frame: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    out = frame.copy()
    out["target_year"] = year
    out["target_month"] = month
    out["period"] = label(year, month)
    out["log_lag1"] = np.log(np.clip(out["lag1"], 1.0, None))
    out["log_ref204"] = np.log(np.clip(out["ref204"], 1.0, None))
    out["ref204_to_lag1"] = out["ref204"] / out["lag1"]
    out["ref204_to_lag12"] = out["ref204"] / out["lag12"]
    out["ref204_to_lag13"] = out["ref204"] / out["lag13"]
    out["lag2_to_lag12"] = out["lag2"] / out["lag12"]
    out["lag3_to_lag13"] = out["lag3"] / out["lag13"]
    out["dip_x_yoy"] = out["dip_score"].fillna(0.0) * out["lag1_to_lag12"].fillna(1.0)
    out["mom_x_vol"] = out["momentum_raw"].fillna(1.0) * out["volatility_rel3"].fillna(0.0)
    out["weak_score"] = ((0.98 - out["momentum_raw"].fillna(1.0)) / 0.08).clip(0.0, 1.0)
    out["rebound_score"] = out["weak_score"] * out["dip_score"].fillna(0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = [
        "target_month",
        "lag1",
        "lag2",
        "lag3",
        "lag12",
        "lag13",
        "rolling3_mean",
        "rolling2_pre_mean",
        "momentum_raw",
        "momentum_clip",
        "feb_to_jan",
        "jan_to_dec",
        "lag1_to_lag12",
        "lag1_to_lag13",
        "store_mar_feb_ratio_prev",
        "volatility_rel3",
        "dip_depth",
        "dip_score",
        "temporary_dip_score",
        "structural_decline_score",
        "ref204",
        "log_lag1",
        "log_ref204",
        "ref204_to_lag1",
        "ref204_to_lag12",
        "ref204_to_lag13",
        "lag2_to_lag12",
        "lag3_to_lag13",
        "dip_x_yoy",
        "mom_x_vol",
        "weak_score",
        "rebound_score",
    ]
    numeric += [col for col in STATIC_NUM_COLS if col in frame.columns]
    cats = ["period"] + [col for col in STATIC_CAT_COLS if col in frame.columns]
    return [col for col in numeric + cats if col in frame.columns]


def model_params() -> dict:
    return dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=450,
        learning_rate=0.025,
        depth=3,
        l2_leaf_reg=120.0,
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
    )


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    _raw, rto, static = load_data()
    parts = []
    for year, month in TRAIN_PERIODS:
        frame = build_target_frame(rto, static, year, month, include_target=True)
        factor = historical_factor(rto, year, month, CURRENT_LAMBDA)
        ref204, _base, _anchor = temporal204(frame, rto, factor, year, month)
        frame["ref204"] = ref204
        frame["target_log_ratio"] = np.log(frame[TARGET].to_numpy(float) / np.maximum(ref204, 1.0))
        parts.append(add_features(frame, year, month))
    train = pd.concat(parts, ignore_index=True)

    test = build_target_frame(rto, static, 2025, 3, include_target=False)
    sub204 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_204_baseline_v6_temporal2_raw_clip085_135_a012_lc006.csv").rename(columns={"rto": "ref204"})
    test = test.merge(sub204, on="new_id", how="inner")
    test = add_features(test, 2025, 3)

    for col in ["period"] + STATIC_CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype(str).fillna("__nan__")
            test[col] = test[col].astype(str).fillna("__nan__")
    return train, test


def train_lpo(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cat_idx = [features.index(col) for col in ["period"] + STATIC_CAT_COLS if col in features]
    rows = []
    for period in sorted(train["period"].unique()):
        tr = train[train["period"] != period]
        va = train[train["period"] == period].copy()
        model = CatBoostRegressor(**model_params())
        model.fit(Pool(tr[features], tr["target_log_ratio"], cat_features=cat_idx))
        va["pred_panel_log"] = model.predict(Pool(va[features], cat_features=cat_idx))
        rows.append(va[["new_id", "period", TARGET, "ref204", "target_log_ratio", "pred_panel_log"]])
    return pd.concat(rows, ignore_index=True)


def evaluate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    centered = oof["pred_panel_log"] - oof.groupby("period")["pred_panel_log"].transform("median")
    global_centered = oof["pred_panel_log"] - float(oof["pred_panel_log"].median())
    rows = []
    for center_name, corr_base in [("raw", oof["pred_panel_log"]), ("period", centered), ("global", global_centered)]:
        for gamma in [0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00]:
            for clip in [0.003, 0.005, 0.008, 0.012, 0.020, 0.030, None]:
                corr = gamma * corr_base.to_numpy(float)
                if clip is not None:
                    corr = np.clip(corr, -clip, clip)
                pred = oof["ref204"].to_numpy(float) * np.exp(corr)
                period_delta = []
                for period, grp in oof.assign(pred=pred).groupby("period"):
                    period_delta.append(mape(grp[TARGET], grp["pred"]) - mape(grp[TARGET], grp["ref204"]))
                rows.append(
                    {
                        "center": center_name,
                        "gamma": gamma,
                        "clip": "" if clip is None else clip,
                        "mape": mape(oof[TARGET], pred),
                        "delta_vs_204": mape(oof[TARGET], pred) - mape(oof[TARGET], oof["ref204"]),
                        "avg_period_delta": float(np.mean(period_delta)),
                        "worst_period_delta": float(np.max(period_delta)),
                        "wins": int((np.array(period_delta) < 0).sum()),
                        **{f"p_{p}_delta": float(d) for p, d in zip(sorted(oof["period"].unique()), period_delta)},
                    }
                )
    return pd.DataFrame(rows).sort_values(["avg_period_delta", "worst_period_delta", "mape"]).reset_index(drop=True)


def train_final_write(train: pd.DataFrame, test: pd.DataFrame, features: list[str], top_specs: pd.DataFrame) -> pd.DataFrame:
    cat_idx = [features.index(col) for col in ["period"] + STATIC_CAT_COLS if col in features]
    model = CatBoostRegressor(**model_params())
    model.fit(Pool(train[features], train["target_log_ratio"], cat_features=cat_idx))
    train_pred = model.predict(Pool(train[features], cat_features=cat_idx))
    train_pred_median = float(np.median(train_pred))
    test = test.copy()
    test["pred_panel_log"] = model.predict(Pool(test[features], cat_features=cat_idx))
    test["pred_panel_period"] = test["pred_panel_log"] - float(test["pred_panel_log"].median())
    test["pred_panel_global"] = test["pred_panel_log"] - train_pred_median

    specs = [
        (240, "panel_raw_g010_clip005", "pred_panel_log", 0.10, 0.005),
        (241, "panel_global_g015_clip008", "pred_panel_global", 0.15, 0.008),
        (242, "panel_global_g030_clip012", "pred_panel_global", 0.30, 0.012),
        (243, "panel_global_g045_clip020", "pred_panel_global", 0.45, 0.020),
        (244, "panel_period_g045_clip020", "pred_panel_period", 0.45, 0.020),
        (245, "panel_period_g060_clip020", "pred_panel_period", 0.60, 0.020),
        (246, "panel_period_g080_clip030", "pred_panel_period", 0.80, 0.030),
        (247, "panel_period_g100_clip030", "pred_panel_period", 1.00, 0.030),
    ]
    rows = []
    ref = test["ref204"].to_numpy(float)
    for candidate_id, name, corr_col, gamma, clip in specs:
        corr = np.clip(gamma * test[corr_col].to_numpy(float), -clip, clip)
        pred = ref * np.exp(corr)
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_v10_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        ratio = pred / ref - 1.0
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "gamma": gamma,
                "clip": clip,
                "test_mean_abs_change_vs_204_pct": float(np.mean(np.abs(ratio)) * 100),
                "test_p95_abs_change_vs_204_pct": float(np.quantile(np.abs(ratio), 0.95) * 100),
                "test_max_abs_change_vs_204_pct": float(np.max(np.abs(ratio)) * 100),
                "test_mean_delta_pct": float(np.mean(ratio) * 100),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_panel()
    features = feature_columns(train)
    oof = train_lpo(train, features)
    oof.to_csv(OUT_DIR / "oof_lpo.csv", index=False)
    summary = evaluate_oof(oof)
    summary.to_csv(OUT_DIR / "oof_lpo_summary.csv", index=False)
    pd.Series(features).to_csv(OUT_DIR / "features.csv", index=False, header=["feature"])
    candidates = train_final_write(train, test, features, summary.head(10))
    candidates.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    print("Top LPO specs:")
    cols = ["center", "gamma", "clip", "mape", "delta_vs_204", "avg_period_delta", "worst_period_delta", "wins", "p_2024-03_delta", "p_2025-02_delta"]
    print(summary[cols].head(30).to_string(index=False))
    print("\nCandidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
