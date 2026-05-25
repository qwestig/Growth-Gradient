from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.append(str(Path(__file__).resolve().parents[1]))

from heuristic_baseline_direction_lab import RESULTS_DIR, STATIC_CAT_COLS, STATIC_NUM_COLS, SUBMISSIONS_DIR, TARGET, mape
from train_temporal204_gru_residual import prepare_data


SEED = 2026
OUT_DIR = RESULTS_DIR / "temporal204_second_residual"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_lag1"] = np.log(np.clip(out["lag1"], 1.0, None))
    out["log_new_base"] = np.log(np.clip(out["new_base"], 1.0, None))
    out["base_to_lag12"] = out["new_base"] / out["lag12"]
    out["base_to_lag13"] = out["new_base"] / out["lag13"]
    out["dip_x_yoy"] = out["dip_score"] * out["lag1_to_lag12"].fillna(1.0)
    out["mom_x_vol"] = out["momentum_raw"] * out["volatility_rel3"]
    out["first_corr_abs"] = out["first_corr_log"].abs()
    out["first_corr_x_mom"] = out["first_corr_log"] * out["momentum_raw"].fillna(1.0)
    out["first_corr_x_vol"] = out["first_corr_log"] * out["volatility_rel3"].fillna(0.0)
    out["pred210_to_lag12"] = out["pred210"] / out["lag12"]
    out["pred210_to_lag13"] = out["pred210"] / out["lag13"]
    out["pred210_to_base"] = out["pred210"] / out["new_base"]
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = [
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
        "new_base",
        "pred210",
        "first_corr_log",
        "first_corr_abs",
        "log_lag1",
        "log_new_base",
        "base_to_lag12",
        "base_to_lag13",
        "dip_x_yoy",
        "mom_x_vol",
        "first_corr_x_mom",
        "first_corr_x_vol",
        "pred210_to_lag12",
        "pred210_to_lag13",
        "pred210_to_base",
    ]
    numeric += [col for col in STATIC_NUM_COLS if col in frame.columns]
    cats = [col for col in STATIC_CAT_COLS if col in frame.columns]
    return numeric + cats


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepare_data()
    train = data.train_rows.copy()
    test = data.test_rows.copy()

    oof = pd.read_csv(RESULTS_DIR / "temporal204_gru_residual" / "oof.csv")[["new_id", "fold", "pred_log_gru"]]
    oof["first_corr_log"] = oof["pred_log_gru"] - float(oof["pred_log_gru"].median())
    train = train.merge(oof, on="new_id", how="inner", suffixes=("", "_oof"))
    train["pred210"] = train["new_base"].to_numpy(float) * np.exp(train["first_corr_log"].to_numpy(float))
    train["target_second_log_ratio"] = np.log(train[TARGET].to_numpy(float) / train["pred210"].to_numpy(float))
    train["fold"] = train["fold_oof"] if "fold_oof" in train.columns else train["fold"]

    sub204 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_204_baseline_v6_temporal2_raw_clip085_135_a012_lc006.csv").rename(columns={"rto": "base204"})
    sub210 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_210_baseline_temporal204_newgru_global_a100.csv").rename(columns={"rto": "pred210_sub"})
    test = test.merge(sub204, on="new_id").merge(sub210, on="new_id")
    test["first_corr_log"] = np.log(test["pred210_sub"].to_numpy(float) / test["base204"].to_numpy(float))
    test["pred210"] = test["pred210_sub"].to_numpy(float)

    train = add_features(train)
    test = add_features(test)
    for col in STATIC_CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype(str).fillna("__nan__")
            test[col] = test[col].astype(str).fillna("__nan__")
    return train, test


def model_params() -> dict:
    return dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=300,
        learning_rate=0.025,
        depth=3,
        l2_leaf_reg=80.0,
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
    )


def train_oof(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cat_idx = [features.index(col) for col in STATIC_CAT_COLS if col in features]
    parts = []
    for fold in range(5):
        tr = train[train["fold"] != fold]
        va = train[train["fold"] == fold].copy()
        model = CatBoostRegressor(**model_params())
        model.fit(Pool(tr[features], tr["target_second_log_ratio"], cat_features=cat_idx))
        va["pred_second_log_cb"] = model.predict(Pool(va[features], cat_features=cat_idx))
        parts.append(va[["new_id", TARGET, "new_base", "pred210", "fold", "target_second_log_ratio", "pred_second_log_cb"]])
    return pd.concat(parts, ignore_index=True)


def evaluate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = [{"variant": "pred210", "gamma": 0.0, "clip": "", "mape": mape(oof[TARGET], oof["pred210"])}]
    centered = oof["pred_second_log_cb"] - float(oof["pred_second_log_cb"].median())
    for center_name, corr_base in [("raw", oof["pred_second_log_cb"]), ("global", centered)]:
        for gamma in [0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60]:
            for clip in [None, 0.003, 0.005, 0.008, 0.012, 0.020]:
                corr = gamma * corr_base.to_numpy(float)
                if clip is not None:
                    corr = np.clip(corr, -clip, clip)
                pred = oof["pred210"].to_numpy(float) * np.exp(corr)
                rows.append({"variant": f"second_cb_{center_name}", "gamma": gamma, "clip": "" if clip is None else clip, "mape": mape(oof[TARGET], pred)})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def train_final_and_write(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cat_idx = [features.index(col) for col in STATIC_CAT_COLS if col in features]
    model = CatBoostRegressor(**model_params())
    model.fit(Pool(train[features], train["target_second_log_ratio"], cat_features=cat_idx))
    test = test.copy()
    test["pred_second_log_cb"] = model.predict(Pool(test[features], cat_features=cat_idx))
    test["pred_second_log_global"] = test["pred_second_log_cb"] - float(test["pred_second_log_cb"].median())

    sub215 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_215_baseline_temporal204_newgru_global_a100_blend75_transfer118.csv").rename(columns={"rto": "pred215"})
    test = test.merge(sub215, on="new_id")

    specs = [
        (217, "second_cb_on210_global_g010_clip005", "pred210", "pred_second_log_global", 0.10, 0.005),
        (218, "second_cb_on210_global_g020_clip008", "pred210", "pred_second_log_global", 0.20, 0.008),
        (219, "second_cb_on210_global_g030_clip012", "pred210", "pred_second_log_global", 0.30, 0.012),
        (220, "second_cb_on215_global_g010_clip005", "pred215", "pred_second_log_global", 0.10, 0.005),
        (221, "second_cb_on215_global_g020_clip008", "pred215", "pred_second_log_global", 0.20, 0.008),
    ]
    rows = []
    for candidate_id, name, base_col, corr_col, gamma, clip in specs:
        corr = gamma * test[corr_col].to_numpy(float)
        corr = np.clip(corr, -clip, clip)
        pred = test[base_col].to_numpy(float) * np.exp(corr)
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_temporal204_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        ratio_base = pred / test[base_col].to_numpy(float) - 1.0
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "base_col": base_col,
                "gamma": gamma,
                "clip": clip,
                "test_mean_rto": float(np.mean(pred)),
                "test_total_rto": float(np.sum(pred)),
                "mean_abs_change_vs_base_pct": float(np.mean(np.abs(ratio_base)) * 100),
                "p95_abs_change_vs_base_pct": float(np.quantile(np.abs(ratio_base), 0.95) * 100),
                "max_abs_change_vs_base_pct": float(np.max(np.abs(ratio_base)) * 100),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "submission_candidates.csv", index=False)

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
                    "family": "temporal204_second_residual",
                    "hypothesis": "OOF-safe small CatBoost second residual layer on top of temporal204 GRU residual predictions.",
                    "public_score": "",
                    "notes": "fill public_score after submission",
                    "public_mape": "",
                    "delta_mape_vs_current_best": "",
                    "delta_score_vs_current_best": "",
                }
            )
        if new_rows:
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_frames()
    features = feature_columns(train)
    oof = train_oof(train, features)
    oof.to_csv(OUT_DIR / "oof.csv", index=False)
    oof_summary = evaluate_oof(oof)
    oof_summary.to_csv(OUT_DIR / "oof_summary.csv", index=False)
    candidates = train_final_and_write(train, test, features)
    pd.Series(features).to_csv(OUT_DIR / "features.csv", index=False, header=["feature"])
    print("OOF summary:")
    print(oof_summary.head(30).to_string(index=False))
    print("\nSubmission candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
