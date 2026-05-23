from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from heuristic_baseline_direction_lab import (
    RESULTS_DIR,
    SUBMISSIONS_DIR,
    STATIC_CAT_COLS,
    STATIC_NUM_COLS,
    TARGET,
    HeuristicSpec,
    build_target_frame,
    factors_for_spec,
    load_data,
    mape,
    predict_from_spec,
)


SEED = 2026
OUT_DIR = RESULTS_DIR / "new_baseline_catboost_residual"
BASE_SPEC = HeuristicSpec(0, "cal050_mom000", 0.50, momentum_exp=0.0)


def add_extra_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_lag1"] = np.log(np.clip(out["lag1"], 1.0, None))
    out["log_new_base"] = np.log(np.clip(out["new_base"], 1.0, None))
    out["base_to_lag12"] = out["new_base"] / out["lag12"]
    out["base_to_lag13"] = out["new_base"] / out["lag13"]
    out["dip_x_yoy"] = out["dip_score"] * out["lag1_to_lag12"].fillna(1.0)
    out["mom_x_vol"] = out["momentum_raw"] * out["volatility_rel3"]
    return out.replace([np.inf, -np.inf], np.nan)


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    _raw, rto, static = load_data()
    train = build_target_frame(rto, static, 2024, 3, include_target=True)
    test = build_target_frame(rto, static, 2025, 3, include_target=False)
    train_factor, test_factor, _val_global, _final_global = factors_for_spec(BASE_SPEC, train, test, rto, static)
    train["new_base"] = predict_from_spec(train, train_factor, BASE_SPEC)
    test["new_base"] = predict_from_spec(test, test_factor, BASE_SPEC)
    train = train[(train[TARGET] > 0) & (train["new_base"] > 0)].copy()
    train["target_log_ratio"] = np.log(train[TARGET] / train["new_base"])
    train["fold"] = train["new_id"] % 5
    train = add_extra_features(train)
    test = add_extra_features(test)
    for col in STATIC_CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype(str).fillna("__nan__")
            test[col] = test[col].astype(str).fillna("__nan__")
    return train, test


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
        "log_lag1",
        "log_new_base",
        "base_to_lag12",
        "base_to_lag13",
        "dip_x_yoy",
        "mom_x_vol",
    ]
    numeric += [col for col in STATIC_NUM_COLS if col in frame.columns]
    cats = [col for col in STATIC_CAT_COLS if col in frame.columns]
    return numeric + cats


def train_oof(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cat_idx = [features.index(col) for col in STATIC_CAT_COLS if col in features]
    params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=450,
        learning_rate=0.035,
        depth=4,
        l2_leaf_reg=30.0,
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
    )
    parts = []
    for fold in range(5):
        tr = train[train["fold"] != fold]
        va = train[train["fold"] == fold].copy()
        model = CatBoostRegressor(**params)
        model.fit(Pool(tr[features], tr["target_log_ratio"], cat_features=cat_idx))
        va["pred_log_cb"] = model.predict(Pool(va[features], cat_features=cat_idx))
        parts.append(va[["new_id", TARGET, "new_base", "fold", "pred_log_cb"]])
    return pd.concat(parts, ignore_index=True)


def evaluate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = [{"variant": "new_base", "alpha": 0.0, "clip": "", "mape": mape(oof[TARGET], oof["new_base"])}]
    for alpha in [0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.10]:
        for clip in [None, 0.01, 0.02, 0.03, 0.05]:
            corr = alpha * oof["pred_log_cb"].to_numpy(dtype=float)
            if clip is not None:
                corr = np.clip(corr, -clip, clip)
            pred = oof["new_base"].to_numpy(dtype=float) * np.exp(corr)
            rows.append({"variant": "catboost", "alpha": alpha, "clip": "" if clip is None else clip, "mape": mape(oof[TARGET], pred)})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def train_final_and_write(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cat_idx = [features.index(col) for col in STATIC_CAT_COLS if col in features]
    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=450,
        learning_rate=0.035,
        depth=4,
        l2_leaf_reg=30.0,
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(Pool(train[features], train["target_log_ratio"], cat_features=cat_idx))
    test = test.copy()
    test["pred_log_cb"] = model.predict(Pool(test[features], cat_features=cat_idx))
    specs = [
        (132, "newcb_a020", 0.20, None),
        (133, "newcb_a035", 0.35, None),
        (134, "newcb_a050", 0.50, None),
        (135, "newcb_a070", 0.70, None),
        (139, "newcb_a090", 0.90, None),
        (140, "newcb_a110", 1.10, None),
        (136, "newcb_a035_clip020", 0.35, 0.02),
        (137, "newcb_a050_clip020", 0.50, 0.02),
        (138, "newcb_a070_clip030", 0.70, 0.03),
        (141, "newcb_a090_clip050", 0.90, 0.05),
        (142, "newcb_a110_clip050", 1.10, 0.05),
    ]
    rows = []
    for candidate_id, name, alpha, clip in specs:
        corr = alpha * test["pred_log_cb"].to_numpy(dtype=float)
        if clip is not None:
            corr = np.clip(corr, -clip, clip)
        pred = test["new_base"].to_numpy(dtype=float) * np.exp(corr)
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append({"candidate_id": candidate_id, "file": file_name, "name": name, "alpha": alpha, "clip": clip if clip is not None else "", "test_mean_rto": float(np.mean(pred)), "test_total_rto": float(np.sum(pred))})
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
            new_rows.append({"candidate_id": row["candidate_id"], "file": row["file"], "family": "new_baseline_catboost_residual", "hypothesis": "CatBoost residual trained directly on new calendar/momentum-zero baseline", "public_score": "", "notes": "fill public_score after submission", "public_mape": "", "delta_mape_vs_current_best": "", "delta_score_vs_current_best": ""})
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
    submissions = train_final_and_write(train, test, features)
    pd.Series(features).to_csv(OUT_DIR / "features.csv", index=False, header=["feature"])
    print("OOF summary:")
    print(oof_summary.head(20).to_string(index=False))
    print("\nSubmission candidates:")
    print(submissions.to_string(index=False))


if __name__ == "__main__":
    main()
