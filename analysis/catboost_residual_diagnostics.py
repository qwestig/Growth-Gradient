from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    CatBoostRegressor = None
    Pool = None


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train_2.csv"
RESULTS_DIR = ROOT / "results" / "catboost_residual"
SUBMISSIONS_DIR = ROOT / "submissions"
TARGET = "РТО"
SEED = 2026

CAT_COLS = [
    "Дата открытия, категориальный",
    "Торговая площадь, категориальный",
    "Населенный пункт",
    "Регион",
]

STATIC_NUM_COLS = [
    "Численность населения",
    "Количество домохозяйств",
    "Трафик пеший, в час",
    "Трафик авто, в час",
    "Маркетплейсы, доставки, постаматы (100 м)",
    "Медицинские уч. и аптеки (300 м)",
    "Школы (300 м)",
    "Остановки (300 м)",
    "Продуктовые магазины (500 м)",
    "Пятерочки (500 м)",
    "Количество касс",
    "Флаг алкогольной лицензии",
]

BEHAVIOR_COLS = [
    "Среднее количество промо товаров в чеке",
    "Среднее количество товаров в чеке",
    "Среднее количество отмен",
    "Рабочие часы в день",
]

MODEL_PARAMS = dict(
    loss_function="MAE",
    eval_metric="MAE",
    iterations=250,
    learning_rate=0.04,
    depth=5,
    l2_leaf_reg=12.0,
    random_seed=SEED,
    allow_writing_files=False,
    verbose=False,
)

ALPHA_GRID = [0.0, 0.03, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CLIP_GRID = [None, 0.10, 0.05, 0.02]


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def make_panel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["period"] = out["Год"] * 12 + out["Месяц"]
    out = out.sort_values(["new_id", "period"]).reset_index(drop=True)
    for col in CAT_COLS:
        out[col] = out[col].astype(str)

    target_group = out.groupby("new_id")[TARGET]
    for lag in [1, 2, 3, 4, 5, 6, 12, 13]:
        out[f"lag{lag}"] = target_group.shift(lag)

    out["rolling3_mean"] = out[["lag1", "lag2", "lag3"]].mean(axis=1)
    out["momentum_raw"] = out["lag1"] / out["rolling3_mean"]
    out["momentum_clip"] = out["momentum_raw"].clip(0.95, 1.05)
    out["momentum_corr"] = out["momentum_clip"] ** 0.25

    out["store_mar_feb_ratio"] = out["lag12"] / out["lag13"]
    out["delta_yoy"] = out["lag12"] + (out["lag1"] - out["lag13"])
    out["trend_yoy"] = out["lag12"] * (out["lag1"] / out["lag13"])
    out["diff1"] = out["lag1"] - out["lag2"]
    out["diff2"] = out["lag2"] - out["lag3"]
    out["accel1"] = out["diff1"] - out["diff2"]
    out["lag1_to_lag12"] = out["lag1"] / out["lag12"]
    out["lag1_to_lag13"] = out["lag1"] / out["lag13"]

    shifted = target_group.shift(1)
    for window in [3, 6, 12]:
        rolled = shifted.groupby(out["new_id"]).rolling(window, min_periods=2)
        out[f"roll{window}_mean"] = rolled.mean().reset_index(level=0, drop=True)
        out[f"roll{window}_std"] = rolled.std().reset_index(level=0, drop=True)
        out[f"roll{window}_min"] = rolled.min().reset_index(level=0, drop=True)
        out[f"roll{window}_max"] = rolled.max().reset_index(level=0, drop=True)

    out["volatility_rel3"] = out["roll3_std"] / out["roll3_mean"]
    out["volatility_rel6"] = out["roll6_std"] / out["roll6_mean"]

    for col in BEHAVIOR_COLS:
        behavior_group = out.groupby("new_id")[col]
        for lag in [1, 2, 3, 12, 13]:
            out[f"{col}_lag{lag}"] = behavior_group.shift(lag)
        out[f"{col}_diff1"] = out[f"{col}_lag1"] - out[f"{col}_lag2"]
        out[f"{col}_yoy_shift"] = out[f"{col}_lag1"] - out[f"{col}_lag13"]

    return out.replace([np.inf, -np.inf], np.nan)


def feature_cols() -> list[str]:
    cols = [
        "lag1",
        "lag2",
        "lag3",
        "lag4",
        "lag5",
        "lag6",
        "lag12",
        "lag13",
        "rolling3_mean",
        "momentum_raw",
        "momentum_clip",
        "momentum_corr",
        "store_mar_feb_ratio",
        "delta_yoy",
        "trend_yoy",
        "diff1",
        "diff2",
        "accel1",
        "lag1_to_lag12",
        "lag1_to_lag13",
        "roll3_mean",
        "roll3_std",
        "roll3_min",
        "roll3_max",
        "roll6_mean",
        "roll6_std",
        "roll6_min",
        "roll6_max",
        "roll12_mean",
        "roll12_std",
        "roll12_min",
        "roll12_max",
        "volatility_rel3",
        "volatility_rel6",
    ]
    cols += STATIC_NUM_COLS + CAT_COLS
    for col in BEHAVIOR_COLS:
        cols += [
            f"{col}_lag1",
            f"{col}_lag2",
            f"{col}_lag3",
            f"{col}_lag12",
            f"{col}_lag13",
            f"{col}_diff1",
            f"{col}_yoy_shift",
        ]
    return cols


def add_base(frame: pd.DataFrame, seasonal_factor: float) -> pd.DataFrame:
    out = frame.copy()
    out["best_base"] = out["lag1"] * seasonal_factor * out["momentum_corr"]
    return out


def apply_residual(base: np.ndarray, pred_log: np.ndarray, alpha: float, clip: float | None) -> np.ndarray:
    raw = base * np.exp(alpha * pred_log)
    if clip is None:
        return raw
    return base * np.clip(raw / base, 1 - clip, 1 + clip)


def bin_summary(frame: pd.DataFrame, key: str, bins: int = 5) -> pd.DataFrame:
    cols = list(dict.fromkeys(["fold", key, TARGET, "best_base", "pred_alpha_050", "pred_alpha_060"]))
    work = frame[cols].copy()
    work["bucket"] = pd.qcut(work[key], q=bins, duplicates="drop")
    rows = []
    for bucket, group in work.groupby("bucket", observed=True):
        rows.append(
            {
                "segment": key,
                "bucket": str(bucket),
                "stores": len(group),
                "base_mape": mape(group[TARGET], group["best_base"]),
                "alpha_050_mape": mape(group[TARGET], group["pred_alpha_050"]),
                "alpha_060_mape": mape(group[TARGET], group["pred_alpha_060"]),
                "gain_050": mape(group[TARGET], group["best_base"]) - mape(group[TARGET], group["pred_alpha_050"]),
                "gain_060": mape(group[TARGET], group["best_base"]) - mape(group[TARGET], group["pred_alpha_060"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    period_label = pd.to_datetime(
        raw["Год"].astype(str) + "-" + raw["Месяц"].astype(str).str.zfill(2) + "-01"
    ).dt.strftime("%Y-%m")
    rto = raw.assign(period_label=period_label).pivot_table(
        index="new_id", columns="period_label", values=TARGET, aggfunc="mean"
    )

    ratio_2023 = (rto["2023-03"] / rto["2023-02"]).replace([np.inf, -np.inf], np.nan)
    ratio_2024 = (rto["2024-03"] / rto["2024-02"]).replace([np.inf, -np.inf], np.nan)
    validation_factor = float(ratio_2023.median())
    final_factor = float(pd.concat([ratio_2023, ratio_2024]).dropna().median())

    features = feature_cols()
    cat_feature_idx = [features.index(col) for col in CAT_COLS]
    panel = add_base(make_panel(raw), validation_factor)
    march_2024 = panel[
        (panel["Год"] == 2024)
        & (panel["Месяц"] == 3)
        & (panel[TARGET] > 0)
        & (panel["best_base"] > 0)
    ].copy()
    march_2024["target_log_ratio"] = np.log(march_2024[TARGET] / march_2024["best_base"])

    cv_rows = []
    oof_parts = []
    for fold in range(5):
        valid_mask = (march_2024["new_id"] % 5) == fold
        train = march_2024[~valid_mask].copy()
        valid = march_2024[valid_mask].copy()

        model = CatBoostRegressor(**MODEL_PARAMS)
        model.fit(Pool(train[features], train["target_log_ratio"], cat_features=cat_feature_idx))
        pred_log = model.predict(valid[features])
        y = valid[TARGET].to_numpy(dtype=float)
        base = valid["best_base"].to_numpy(dtype=float)

        valid_oof = valid[
            [
                "new_id",
                TARGET,
                "best_base",
                "momentum_raw",
                "momentum_clip",
                "volatility_rel3",
                "volatility_rel6",
                "lag1_to_lag12",
                "Регион",
                "Торговая площадь, категориальный",
            ]
        ].copy()
        valid_oof["fold"] = fold
        valid_oof["pred_log"] = pred_log
        valid_oof["correction_alpha_050"] = np.exp(0.5 * pred_log)
        valid_oof["pred_alpha_050"] = apply_residual(base, pred_log, 0.5, None)
        valid_oof["pred_alpha_060"] = apply_residual(base, pred_log, 0.6, None)
        oof_parts.append(valid_oof)

        for alpha in ALPHA_GRID:
            for clip in CLIP_GRID:
                pred = apply_residual(base, pred_log, alpha, clip)
                correction = pred / base
                cv_rows.append(
                    {
                        "fold": fold,
                        "alpha": alpha,
                        "clip": "no_clip" if clip is None else f"clip_{clip:.2f}",
                        "mape_ml": mape(y, pred),
                        "mape_base": mape(y, base),
                        "mean_abs_correction_pct": float(np.mean(np.abs(correction - 1)) * 100),
                        "p99_correction": float(pd.Series(correction).quantile(0.99)),
                        "max_correction": float(pd.Series(correction).max()),
                    }
                )

    cv = pd.DataFrame(cv_rows)
    cv_summary = (
        cv.groupby(["alpha", "clip"], as_index=False)
        .agg(
            mape_ml=("mape_ml", "mean"),
            mape_base=("mape_base", "mean"),
            mean_abs_correction_pct=("mean_abs_correction_pct", "mean"),
            p99_correction=("p99_correction", "mean"),
            max_correction=("max_correction", "max"),
        )
        .sort_values("mape_ml")
        .reset_index(drop=True)
    )
    cv_summary["gain_vs_base"] = cv_summary["mape_base"] - cv_summary["mape_ml"]
    oof = pd.concat(oof_parts, ignore_index=True)

    segment_summary = pd.concat(
        [
            bin_summary(oof, "best_base"),
            bin_summary(oof, "momentum_raw"),
            bin_summary(oof, "volatility_rel3"),
            bin_summary(oof, "lag1_to_lag12"),
        ],
        ignore_index=True,
    )
    region_summary = []
    for region, group in oof.groupby("Регион"):
        if len(group) >= 150:
            base_mape = mape(group[TARGET], group["best_base"])
            region_summary.append(
                {
                    "Регион": region,
                    "stores": len(group),
                    "base_mape": base_mape,
                    "alpha_050_mape": mape(group[TARGET], group["pred_alpha_050"]),
                    "alpha_060_mape": mape(group[TARGET], group["pred_alpha_060"]),
                    "gain_050": base_mape - mape(group[TARGET], group["pred_alpha_050"]),
                    "gain_060": base_mape - mape(group[TARGET], group["pred_alpha_060"]),
                }
            )
    region_summary = pd.DataFrame(region_summary).sort_values("gain_050", ascending=False)

    tail_summary = oof["correction_alpha_050"].describe(
        percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    ).to_frame("alpha_050_oof_correction")
    tail_stores = oof.assign(abs_corr=(oof["correction_alpha_050"] - 1).abs()).sort_values(
        "abs_corr", ascending=False
    ).head(50)

    final_model = CatBoostRegressor(**MODEL_PARAMS)
    final_model.fit(Pool(march_2024[features], march_2024["target_log_ratio"], cat_features=cat_feature_idx))

    feb_2025 = raw[(raw["Год"] == 2025) & (raw["Месяц"] == 2)].copy()
    mar_2025 = feb_2025.copy()
    mar_2025["Год"] = 2025
    mar_2025["Месяц"] = 3
    mar_2025[TARGET] = np.nan
    pred_panel = add_base(make_panel(pd.concat([raw, mar_2025], ignore_index=True)), final_factor)
    pred_rows = pred_panel[(pred_panel["Год"] == 2025) & (pred_panel["Месяц"] == 3)].copy()
    pred_log = final_model.predict(pred_rows[features])
    base = pred_rows["best_base"].to_numpy(dtype=float)

    candidate_specs = [
        ("candidate_cb_alpha050_noclip", 0.5, None),
        ("candidate_cb_alpha060_noclip", 0.6, None),
        ("candidate_cb_alpha050_clip010", 0.5, 0.10),
    ]
    candidate_rows = []
    candidate_predictions = {}
    for name, alpha, clip in candidate_specs:
        pred = np.maximum(apply_residual(base, pred_log, alpha, clip), 0)
        candidate_predictions[name] = pred
        correction = pred / base
        candidate_rows.append(
            {
                "candidate": name,
                "alpha": alpha,
                "clip": "no_clip" if clip is None else clip,
                "rows": len(pred_rows),
                "mean_abs_correction_pct": float(np.mean(np.abs(correction - 1)) * 100),
                "p99_correction": float(pd.Series(correction).quantile(0.99)),
                "max_correction": float(pd.Series(correction).max()),
            }
        )
    candidate_summary = pd.DataFrame(candidate_rows)

    pick = cv_summary[(cv_summary["clip"] == "no_clip") & (cv_summary["alpha"].isin([0.5, 0.6]))].sort_values(
        "mape_ml"
    ).iloc[0]
    recommended_name = "candidate_cb_alpha060_noclip" if float(pick["alpha"]) == 0.6 else "candidate_cb_alpha050_noclip"
    recommended_submission = pred_rows[["new_id"]].copy()
    recommended_submission["rto"] = np.round(candidate_predictions[recommended_name], 2)
    recommended_submission = recommended_submission.sort_values("new_id").reset_index(drop=True)
    recommended_path = SUBMISSIONS_DIR / "test_next.csv"
    recommended_submission.to_csv(recommended_path, index=False)

    cv.to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    cv_summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    oof.to_csv(RESULTS_DIR / "oof_march_2024.csv", index=False)
    segment_summary.to_csv(RESULTS_DIR / "segment_summary.csv", index=False)
    region_summary.to_csv(RESULTS_DIR / "region_summary.csv", index=False)
    tail_summary.to_csv(RESULTS_DIR / "correction_tail_summary.csv")
    tail_stores.to_csv(RESULTS_DIR / "largest_oof_corrections.csv", index=False)
    candidate_summary.to_csv(RESULTS_DIR / "candidate_summary.csv", index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": final_model.get_feature_importance(
                Pool(march_2024[features], march_2024["target_log_ratio"], cat_features=cat_feature_idx),
                type="FeatureImportance",
            ),
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    report = {
        "validation_factor": validation_factor,
        "final_factor": final_factor,
        "march_2024_rows": len(march_2024),
        "cv_top": cv_summary.head(12).to_dict(orient="records"),
        "candidate_summary": candidate_summary.to_dict(orient="records"),
        "recommended_submission": str(recommended_path),
        "recommended_candidate": recommended_name,
        "recommended_basis": {
            "alpha": float(pick["alpha"]),
            "clip": str(pick["clip"]),
            "oof_mape": float(pick["mape_ml"]),
            "oof_gain_vs_base": float(pick["gain_vs_base"]),
        },
        "feature_importance_top": feature_importance.head(15).to_dict(orient="records"),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
