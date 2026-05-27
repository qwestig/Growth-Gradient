from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from heuristic_baseline_direction_lab import RESULTS_DIR, STATIC_CAT_COLS, STATIC_NUM_COLS, SUBMISSIONS_DIR, TARGET, mape
from train_temporal204_gru_residual import prepare_data


SEED = 2026
OUT_DIR = RESULTS_DIR / "temporal204_catboost_first_residual_v12"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_lag1"] = np.log(np.clip(out["lag1"], 1.0, None))
    out["log_lag2"] = np.log(np.clip(out["lag2"], 1.0, None))
    out["log_lag12"] = np.log(np.clip(out["lag12"], 1.0, None))
    out["log_new_base"] = np.log(np.clip(out["new_base"], 1.0, None))
    out["base_to_lag1"] = out["new_base"] / out["lag1"]
    out["base_to_lag12"] = out["new_base"] / out["lag12"]
    out["base_to_lag13"] = out["new_base"] / out["lag13"]
    out["lag2_to_lag12"] = out["lag2"] / out["lag12"]
    out["lag3_to_lag13"] = out["lag3"] / out["lag13"]
    out["dip_x_yoy"] = out["dip_score"].fillna(0.0) * out["lag1_to_lag12"].fillna(1.0)
    out["mom_x_vol"] = out["momentum_raw"].fillna(1.0) * out["volatility_rel3"].fillna(0.0)
    out["weak_score"] = ((0.98 - out["momentum_raw"].fillna(1.0)) / 0.08).clip(0.0, 1.0)
    out["rebound_score"] = out["weak_score"] * out["dip_score"].fillna(0.0)
    out["stable_score"] = (1.0 - out["volatility_rel3"].fillna(0.0) / 0.16).clip(0.0, 1.0)
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
        "log_lag1",
        "log_lag2",
        "log_lag12",
        "log_new_base",
        "base_to_lag1",
        "base_to_lag12",
        "base_to_lag13",
        "lag2_to_lag12",
        "lag3_to_lag13",
        "dip_x_yoy",
        "mom_x_vol",
        "weak_score",
        "rebound_score",
        "stable_score",
    ]
    numeric += [col for col in STATIC_NUM_COLS if col in frame.columns]
    cats = [col for col in STATIC_CAT_COLS if col in frame.columns]
    return [col for col in numeric + cats if col in frame.columns]


def configs() -> dict[str, dict]:
    base = dict(loss_function="MAE", eval_metric="MAE", random_seed=SEED, allow_writing_files=False, verbose=False)
    return {
        "cb_d3_l80": {**base, "iterations": 450, "learning_rate": 0.025, "depth": 3, "l2_leaf_reg": 80.0},
        "cb_d4_l120": {**base, "iterations": 450, "learning_rate": 0.025, "depth": 4, "l2_leaf_reg": 120.0},
        "cb_d5_l200": {**base, "iterations": 500, "learning_rate": 0.020, "depth": 5, "l2_leaf_reg": 200.0},
    }


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data = prepare_data()
    train = add_features(data.train_rows.copy())
    test = add_features(data.test_rows.copy())
    train["fold"] = train["new_id"] % 5
    for col in STATIC_CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype(str).fillna("__nan__")
            test[col] = test[col].astype(str).fillna("__nan__")
    features = feature_columns(train)
    return train, test, features


def train_oof_and_test(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_idx = [features.index(col) for col in STATIC_CAT_COLS if col in features]
    oof = train[["new_id", TARGET, "new_base", "fold"]].copy()
    test_pred = test[["new_id", "new_base"]].copy()
    for name, params in configs().items():
        pred_oof = np.zeros(len(train), dtype=float)
        fold_losses = []
        for fold in range(5):
            tr = train[train["fold"] != fold]
            va = train[train["fold"] == fold]
            model = CatBoostRegressor(**params)
            model.fit(Pool(tr[features], tr["target_log_ratio"], cat_features=cat_idx))
            pred_oof[train["fold"] == fold] = model.predict(Pool(va[features], cat_features=cat_idx))
            fold_losses.append({"config": name, "fold": fold, "base_mape": mape(va[TARGET], va["new_base"])})
        oof[name] = pred_oof
        final = CatBoostRegressor(**params)
        final.fit(Pool(train[features], train["target_log_ratio"], cat_features=cat_idx))
        test_pred[name] = final.predict(Pool(test[features], cat_features=cat_idx))
        pd.DataFrame(fold_losses).to_csv(OUT_DIR / f"fold_summary_{name}.csv", index=False)
    return oof, test_pred


def centered(series: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    return arr - float(np.median(arr))


def evaluate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = [{"name": "base204", "mape": mape(oof[TARGET], oof["new_base"]), "delta_vs_210": np.nan, "delta_vs_220": np.nan}]
    ref210 = pd.read_csv(RESULTS_DIR / "temporal204_gru_residual" / "oof.csv")[["new_id", "new_base", "pred_log_gru"]]
    ref210["pred210"] = ref210["new_base"] * np.exp(ref210["pred_log_gru"] - float(ref210["pred_log_gru"].median()))
    second = pd.read_csv(RESULTS_DIR / "stacking_v11" / "oof_prediction_diagnostics.csv")
    m210 = mape(ref210.merge(oof[["new_id", TARGET]], on="new_id")[TARGET], ref210["pred210"])
    m220 = float(second.loc[second["prediction"] == "pred220", "oof_mape"].iloc[0]) if not second.empty else np.nan
    frame = oof.merge(ref210[["new_id", "pred210"]], on="new_id", how="inner")
    for cfg in configs():
        corr = centered(frame[cfg])
        for alpha in [0.50, 0.75, 1.00, 1.25]:
            for clip in [None, 0.02, 0.04, 0.06]:
                c = alpha * corr
                if clip is not None:
                    c = np.clip(c, -clip, clip)
                pred_cb = frame["new_base"].to_numpy(float) * np.exp(c)
                rows.append({"name": f"{cfg}_a{alpha:.2f}_c{'' if clip is None else clip}", "mape": mape(frame[TARGET], pred_cb), "delta_vs_210": mape(frame[TARGET], pred_cb) - m210, "delta_vs_220": mape(frame[TARGET], pred_cb) - m220})
                for w in [0.15, 0.25, 0.35, 0.50]:
                    blend = (1.0 - w) * frame["pred210"].to_numpy(float) + w * pred_cb
                    rows.append({"name": f"blend210_{cfg}_w{w:.2f}_a{alpha:.2f}_c{'' if clip is None else clip}", "mape": mape(frame[TARGET], blend), "delta_vs_210": mape(frame[TARGET], blend) - m210, "delta_vs_220": mape(frame[TARGET], blend) - m220})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def write_candidates(train: pd.DataFrame, test: pd.DataFrame, oof: pd.DataFrame, test_pred: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    test_frame = test[["new_id", "new_base"]].copy().merge(test_pred.drop(columns=["new_base"]), on="new_id")
    sub210 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_210_baseline_temporal204_newgru_global_a100.csv").rename(columns={"rto": "pred210"})
    sub220 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_220_baseline_temporal204_second_cb_on215_global_g010_clip005.csv").rename(columns={"rto": "pred220"})
    sub265 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_265_full_v11_stack_avg_huber030_segment_g0p75_c0p005.csv").rename(columns={"rto": "pred265"})
    test_frame = test_frame.merge(sub210, on="new_id").merge(sub220, on="new_id").merge(sub265, on="new_id")

    specs = [
        (267, "cb_d3_l80_a100_c004", "cb_d3_l80", 1.00, 0.04, None, None),
        (268, "blend210_cb_d3_l80_w025_a100_c004", "cb_d3_l80", 1.00, 0.04, "pred210", 0.25),
        (269, "blend220_cb_d3_l80_w025_a100_c004", "cb_d3_l80", 1.00, 0.04, "pred220", 0.25),
        (270, "blend265_cb_d3_l80_w025_a100_c004", "cb_d3_l80", 1.00, 0.04, "pred265", 0.25),
        (271, "blend265_cb_d4_l120_w025_a100_c004", "cb_d4_l120", 1.00, 0.04, "pred265", 0.25),
    ]
    rows = []
    for candidate_id, name, cfg, alpha, clip, blend_col, w in specs:
        corr = alpha * centered(test_frame[cfg])
        corr = np.clip(corr, -clip, clip)
        pred_cb = test_frame["new_base"].to_numpy(float) * np.exp(corr)
        pred = pred_cb if blend_col is None else (1.0 - w) * test_frame[blend_col].to_numpy(float) + w * pred_cb
        out = test_frame[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_full_v12_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        ratio220 = pred / test_frame["pred220"].to_numpy(float) - 1.0
        ratio265 = pred / test_frame["pred265"].to_numpy(float) - 1.0
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "cfg": cfg,
                "alpha": alpha,
                "clip": clip,
                "blend_col": blend_col or "none",
                "blend_weight_cb": w if w is not None else 1.0,
                "mean_abs_vs_220_pct": float(np.mean(np.abs(ratio220)) * 100),
                "p95_abs_vs_220_pct": float(np.quantile(np.abs(ratio220), 0.95) * 100),
                "mean_abs_vs_265_pct": float(np.mean(np.abs(ratio265)) * 100),
                "p95_abs_vs_265_pct": float(np.quantile(np.abs(ratio265), 0.95) * 100),
            }
        )
    candidates = pd.DataFrame(rows)
    candidates.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    return candidates


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, features = build_frames()
    oof, test_pred = train_oof_and_test(train, test, features)
    oof.to_csv(OUT_DIR / "oof.csv", index=False)
    test_pred.to_csv(OUT_DIR / "test_pred_log.csv", index=False)
    pd.Series(features).to_csv(OUT_DIR / "features.csv", index=False, header=["feature"])
    summary = evaluate_oof(oof)
    summary.to_csv(OUT_DIR / "oof_search.csv", index=False)
    candidates = write_candidates(train, test, oof, test_pred, summary)
    print("OOF search top:")
    print(summary.head(40).to_string(index=False))
    print("\nCandidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
