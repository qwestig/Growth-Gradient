from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
RESULTS_DIR = ROOT / "results"
SUBMISSIONS_DIR = ROOT / "submissions"
TARGET = "РТО"
SEED = 2026

STATIC_CAT_COLS = [
    "Регион",
    "Торговая площадь, категориальный",
    "Дата открытия, категориальный",
]

STATIC_NUM_COLS = [
    "Численность населения",
    "Количество домохозяйств",
    "Трафик пеший, в час",
    "Трафик авто, в час",
    "Маркетплейсы, доставки, постаматы (100 м)",
    "Медицинские уч. и аптеки (100 м)",
    "Остановки (100 м)",
    "Оптики (100 м)",
    "ПВЗ (100 м)",
    "Спорт (100 м)",
    "Цветы (100 м)",
    "Алкоголь (100 м)",
    "Кафе и рестораны (100 м)",
    "Косметика (100 м)",
    "Продуктовые магазины (100 м)",
    "Промтовары магазины (100 м)",
    "Алкоголь (500 м)",
    "Кафе и рестораны (500 м)",
    "Косметика (500 м)",
    "Продуктовые магазины (500 м)",
    "Промтовары магазины (500 м)",
]


@dataclass(frozen=True)
class HeuristicSpec:
    candidate_id: int
    name: str
    calendar_lambda: float
    factor_method: str = "median"
    momentum_exp: float = 0.25
    rebound_lift: float = 0.0
    structural_cut: float = 0.0
    group_col: str | None = None
    group_shrink_k: float = 2000.0


def mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.sort_values("value")
    cutoff = frame["weight"].sum() * 0.5
    return float(frame.loc[frame["weight"].cumsum() >= cutoff, "value"].iloc[0])


def period_label(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        frame["Год"].astype(str) + "-" + frame["Месяц"].astype(str).str.zfill(2) + "-01"
    ).dt.strftime("%Y-%m")


def label(year: int, month: int, months_back: int = 0) -> str:
    return (pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=months_back)).strftime("%Y-%m")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).replace([np.inf, -np.inf], np.nan)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw["period_label"] = period_label(raw)
    rto = raw.pivot_table(index="new_id", columns="period_label", values=TARGET, aggfunc="mean")
    static_cols = ["new_id"] + [c for c in STATIC_CAT_COLS + STATIC_NUM_COLS if c in raw.columns]
    static = raw.sort_values(["new_id", "Год", "Месяц"]).groupby("new_id", as_index=False)[static_cols].first()
    return raw, rto, static.set_index("new_id")


def build_target_frame(rto: pd.DataFrame, static: pd.DataFrame, year: int, month: int, include_target: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=rto.index)
    out["new_id"] = out.index.astype(int)
    out["lag1"] = rto[label(year, month, 1)]
    out["lag2"] = rto[label(year, month, 2)]
    out["lag3"] = rto[label(year, month, 3)]
    out["lag12"] = rto[label(year, month, 12)]
    out["lag13"] = rto[label(year, month, 13)]
    out["rolling3_mean"] = out[["lag1", "lag2", "lag3"]].mean(axis=1)
    out["rolling2_pre_mean"] = out[["lag2", "lag3"]].mean(axis=1)
    out["momentum_raw"] = safe_div(out["lag1"], out["rolling3_mean"])
    out["momentum_clip"] = out["momentum_raw"].fillna(1.0).clip(0.95, 1.05)
    out["feb_to_jan"] = safe_div(out["lag1"], out["lag2"])
    out["jan_to_dec"] = safe_div(out["lag2"], out["lag3"])
    out["lag1_to_lag12"] = safe_div(out["lag1"], out["lag12"])
    out["lag1_to_lag13"] = safe_div(out["lag1"], out["lag13"])
    out["store_mar_feb_ratio_prev"] = safe_div(out["lag12"], out["lag13"])
    out["volatility_rel3"] = out[["lag1", "lag2", "lag3"]].std(axis=1) / out["rolling3_mean"]
    out["dip_depth"] = ((out["rolling2_pre_mean"] - out["lag1"]) / out["rolling2_pre_mean"]).clip(lower=0, upper=0.25).fillna(0)
    out["dip_score"] = (out["dip_depth"] / 0.25).clip(0, 1)
    out["temporary_dip_score"] = out["dip_score"] * (
        out["jan_to_dec"].fillna(1.0).ge(0.98)
        & out["lag1_to_lag12"].fillna(1.0).ge(1.00)
    ).astype(float)
    out["structural_decline_score"] = (
        out["feb_to_jan"].fillna(1.0).lt(0.97)
        & out["jan_to_dec"].fillna(1.0).lt(0.99)
        & out["lag1_to_lag12"].fillna(1.0).lt(1.02)
    ).astype(float)
    if include_target:
        out[TARGET] = rto[label(year, month)]
    out = out.join(static, how="left")
    for col in STATIC_CAT_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).fillna("__nan__")
    return out.reset_index(drop=True)


def ratio_frame(rto: pd.DataFrame, static: pd.DataFrame, year: int, calendar_lambda: float, target_non_leap: bool) -> pd.DataFrame:
    ratio = safe_div(rto[f"{year}-03"], rto[f"{year}-02"])
    # February 2024 had 29 days. Convert it partially toward a non-leap February ratio for 2025,
    # or partially down when using 2023 to predict leap-year March 2024 validation.
    if year == 2024 and target_non_leap:
        ratio = ratio * ((1.0 - calendar_lambda) + calendar_lambda * (29.0 / 28.0))
    elif year == 2023 and not target_non_leap:
        ratio = ratio * ((1.0 - calendar_lambda) + calendar_lambda * (28.0 / 29.0))
    frame = pd.DataFrame({"new_id": ratio.index.astype(int), "ratio": ratio}).dropna().reset_index(drop=True)
    return frame.join(static, on="new_id", how="left")


def global_factor(ratios: pd.Series, method: str) -> float:
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    if method == "median":
        return float(ratios.median())
    if method == "mape_weighted_median":
        return weighted_median(ratios, 1.0 / ratios)
    if method == "aggregate_mean":
        return float(ratios.mean())
    raise ValueError(f"Unknown factor method: {method}")


def shrink_group_factors(
    ratio_data: pd.DataFrame,
    target_frame: pd.DataFrame,
    group_col: str,
    fallback_factor: float,
    shrink_k: float,
) -> pd.Series:
    stats = ratio_data.dropna(subset=["ratio"]).groupby(group_col, observed=True)["ratio"].agg(["median", "count"])
    correction = np.log(stats["median"] / fallback_factor) * (stats["count"] / (stats["count"] + shrink_k))
    mapped = target_frame[group_col].map(correction).fillna(0.0).to_numpy(dtype=float)
    return pd.Series(fallback_factor * np.exp(mapped), index=target_frame.index)


def factors_for_spec(
    spec: HeuristicSpec,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    rto: pd.DataFrame,
    static: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, float, float]:
    val_ratio_data = ratio_frame(rto, static, 2023, spec.calendar_lambda, target_non_leap=False)
    final_ratio_2023 = ratio_frame(rto, static, 2023, spec.calendar_lambda, target_non_leap=True)
    final_ratio_2024 = ratio_frame(rto, static, 2024, spec.calendar_lambda, target_non_leap=True)
    final_ratio_data = pd.concat([final_ratio_2023, final_ratio_2024], ignore_index=True)

    val_global = global_factor(val_ratio_data["ratio"], spec.factor_method)
    final_global = global_factor(final_ratio_data["ratio"], spec.factor_method)

    if spec.group_col is None:
        return (
            pd.Series(val_global, index=val_frame.index),
            pd.Series(final_global, index=test_frame.index),
            val_global,
            final_global,
        )

    val_factor = shrink_group_factors(val_ratio_data, val_frame, spec.group_col, val_global, spec.group_shrink_k)
    final_factor = shrink_group_factors(final_ratio_data, test_frame, spec.group_col, final_global, spec.group_shrink_k)
    return val_factor, final_factor, val_global, final_global


def predict_from_spec(frame: pd.DataFrame, factor: pd.Series, spec: HeuristicSpec) -> np.ndarray:
    pred = frame["lag1"].to_numpy(dtype=float) * factor.to_numpy(dtype=float)
    pred *= frame["momentum_clip"].to_numpy(dtype=float) ** spec.momentum_exp
    if spec.rebound_lift:
        pred *= np.exp(spec.rebound_lift * frame["temporary_dip_score"].to_numpy(dtype=float))
    if spec.structural_cut:
        pred *= np.exp(-spec.structural_cut * frame["structural_decline_score"].to_numpy(dtype=float))
    return np.maximum(pred, 0.0)


def specs() -> list[HeuristicSpec]:
    return [
        HeuristicSpec(43, "raw_median_mom025", 0.0),
        HeuristicSpec(44, "cal0125_median_mom025", 0.125),
        HeuristicSpec(45, "cal025_median_mom025", 0.25),
        HeuristicSpec(46, "cal0375_median_mom025", 0.375),
        HeuristicSpec(47, "cal050_median_mom025", 0.50),
        HeuristicSpec(48, "cal025_mape_wmedian_mom025", 0.25, factor_method="mape_weighted_median"),
        HeuristicSpec(49, "cal025_median_mom000", 0.25, momentum_exp=0.0),
        HeuristicSpec(50, "cal025_median_mom0125", 0.25, momentum_exp=0.125),
        HeuristicSpec(51, "cal025_median_mom0375", 0.25, momentum_exp=0.375),
        HeuristicSpec(52, "cal025_tempdip_lift010", 0.25, rebound_lift=0.010),
        HeuristicSpec(53, "cal025_tempdip_lift020", 0.25, rebound_lift=0.020),
        HeuristicSpec(54, "cal025_tempdip_lift030", 0.25, rebound_lift=0.030),
        HeuristicSpec(55, "cal025_tempdip020_structcut010", 0.25, rebound_lift=0.020, structural_cut=0.010),
        HeuristicSpec(56, "cal0125_tempdip020_structcut010", 0.125, rebound_lift=0.020, structural_cut=0.010),
        HeuristicSpec(57, "cal0375_tempdip020_structcut010", 0.375, rebound_lift=0.020, structural_cut=0.010),
        HeuristicSpec(58, "cal025_region_k2000", 0.25, group_col="Регион", group_shrink_k=2000),
        HeuristicSpec(59, "cal025_area_k2000", 0.25, group_col="Торговая площадь, категориальный", group_shrink_k=2000),
        HeuristicSpec(60, "cal025_opening_k2000", 0.25, group_col="Дата открытия, категориальный", group_shrink_k=2000),
        HeuristicSpec(61, "cal025_region_k500", 0.25, group_col="Регион", group_shrink_k=500),
        HeuristicSpec(62, "cal025_opening_k500", 0.25, group_col="Дата открытия, категориальный", group_shrink_k=500),
    ]


def add_shrunk_factor_features(frame: pd.DataFrame, rto: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    base_spec = HeuristicSpec(0, "cal025", 0.25)
    for group_col, shrink_k in [
        ("Регион", 2000),
        ("Торговая площадь, категориальный", 2000),
        ("Дата открытия, категориальный", 2000),
    ]:
        spec = HeuristicSpec(0, "factor", 0.25, group_col=group_col, group_shrink_k=shrink_k)
        val_factor, _, val_global, _ = factors_for_spec(spec, out, out, rto, static)
        out[f"factor_{group_col}_k{shrink_k}"] = val_factor / val_global
    _, _, val_global, _ = factors_for_spec(base_spec, out, out, rto, static)
    out["cal025_global_factor"] = val_global
    return out


def run_shap_diagnostic(val_frame: pd.DataFrame, current_pred: np.ndarray, rto: pd.DataFrame, static: pd.DataFrame) -> None:
    frame = add_shrunk_factor_features(val_frame, rto, static)
    frame["target_log_residual"] = np.log(frame[TARGET].to_numpy(dtype=float) / current_pred)
    feature_cols = [
        "momentum_raw",
        "momentum_clip",
        "feb_to_jan",
        "jan_to_dec",
        "lag1_to_lag12",
        "lag1_to_lag13",
        "store_mar_feb_ratio_prev",
        "volatility_rel3",
        "dip_score",
        "temporary_dip_score",
        "structural_decline_score",
        "factor_Регион_k2000",
        "factor_Торговая площадь, категориальный_k2000",
        "factor_Дата открытия, категориальный_k2000",
    ] + [col for col in STATIC_NUM_COLS if col in frame.columns] + STATIC_CAT_COLS

    cat_idx = [feature_cols.index(col) for col in STATIC_CAT_COLS if col in feature_cols]
    pool = Pool(frame[feature_cols], frame["target_log_residual"], cat_features=cat_idx)
    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=350,
        learning_rate=0.035,
        depth=3,
        l2_leaf_reg=30.0,
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(pool)
    shap_values = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    shap_summary = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_summary.to_csv(RESULTS_DIR / "heuristic_direction_shap_summary.csv", index=False)

    bin_rows = []
    numeric_cols = [col for col in feature_cols if col not in STATIC_CAT_COLS and pd.api.types.is_numeric_dtype(frame[col])]
    for col in numeric_cols:
        values = frame[col].replace([np.inf, -np.inf], np.nan)
        try:
            bins = pd.qcut(values, q=5, duplicates="drop")
        except ValueError:
            continue
        tmp = frame.assign(_bin=bins)
        for bin_name, part in tmp.groupby("_bin", observed=True):
            if len(part) < 50:
                continue
            bin_rows.append(
                {
                    "feature": col,
                    "bin": str(bin_name),
                    "n": len(part),
                    "mean_feature": float(part[col].mean()),
                    "mean_log_residual": float(part["target_log_residual"].mean()),
                    "mape_current_base": mape(part[TARGET], current_pred[part.index]),
                }
            )
    pd.DataFrame(bin_rows).to_csv(RESULTS_DIR / "heuristic_direction_feature_bins.csv", index=False)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val_frame = build_target_frame(rto, static, 2024, 3, include_target=True)
    test_frame = build_target_frame(rto, static, 2025, 3, include_target=False)

    rows = []
    current_pred = None
    for spec in specs():
        val_factor, test_factor, val_global, final_global = factors_for_spec(spec, val_frame, test_frame, rto, static)
        val_pred = predict_from_spec(val_frame, val_factor, spec)
        test_pred = predict_from_spec(test_frame, test_factor, spec)

        if spec.name == "raw_median_mom025":
            current_pred = val_pred.copy()

        out = test_frame[["new_id"]].copy()
        out["rto"] = np.round(test_pred, 2)
        file_name = f"stage2_candidate_{spec.candidate_id:02d}_baseline_{spec.name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)

        ratio_to_current = val_pred / current_pred if current_pred is not None else np.ones_like(val_pred)
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "file": file_name,
                "name": spec.name,
                "family": "heuristic_baseline",
                "calendar_lambda": spec.calendar_lambda,
                "factor_method": spec.factor_method,
                "momentum_exp": spec.momentum_exp,
                "rebound_lift": spec.rebound_lift,
                "structural_cut": spec.structural_cut,
                "group_col": spec.group_col or "",
                "group_shrink_k": spec.group_shrink_k if spec.group_col else np.nan,
                "val_global_factor": val_global,
                "final_global_factor": final_global,
                "val_mape_march2024": mape(val_frame[TARGET], val_pred),
                "val_weak_momentum_mape": mape(
                    val_frame.loc[val_frame["momentum_raw"] < 0.935766244455909, TARGET],
                    val_pred[val_frame["momentum_raw"] < 0.935766244455909],
                ),
                "val_temp_dip_mape": mape(
                    val_frame.loc[val_frame["temporary_dip_score"] > 0, TARGET],
                    val_pred[val_frame["temporary_dip_score"] > 0],
                )
                if int((val_frame["temporary_dip_score"] > 0).sum())
                else np.nan,
                "val_struct_decline_mape": mape(
                    val_frame.loc[val_frame["structural_decline_score"] > 0, TARGET],
                    val_pred[val_frame["structural_decline_score"] > 0],
                )
                if int((val_frame["structural_decline_score"] > 0).sum())
                else np.nan,
                "test_mean_rto": float(np.mean(test_pred)),
                "test_total_rto": float(np.sum(test_pred)),
                "val_mean_abs_change_vs_current_pct": float(np.mean(np.abs(ratio_to_current - 1.0)) * 100),
                "val_p95_abs_change_vs_current_pct": float(np.quantile(np.abs(ratio_to_current - 1.0), 0.95) * 100),
            }
        )

    summary = pd.DataFrame(rows).sort_values("val_mape_march2024").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "heuristic_baseline_direction_candidates.csv", index=False)

    current_row = summary[summary["name"] == "raw_median_mom025"].iloc[0]
    diagnostic_rows = []
    for _, row in summary.iterrows():
        diagnostic_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "name": row["name"],
                "delta_val_mape_vs_raw_median_mom025": row["val_mape_march2024"] - current_row["val_mape_march2024"],
                "direction_question": direction_question(str(row["name"])),
                "public_status": "not_submitted",
                "public_score": "",
                "public_mape": "",
                "decision_after_public": "",
            }
        )
    pd.DataFrame(diagnostic_rows).to_csv(RESULTS_DIR / "heuristic_public_experiment_template.csv", index=False)

    if current_pred is not None:
        run_shap_diagnostic(val_frame, current_pred, rto, static)

    print("Wrote heuristic candidates:")
    print(summary[["candidate_id", "name", "val_mape_march2024", "final_global_factor", "val_mean_abs_change_vs_current_pct"]].to_string(index=False))
    print("\nTop SHAP summary:")
    shap_path = RESULTS_DIR / "heuristic_direction_shap_summary.csv"
    if shap_path.exists():
        print(pd.read_csv(shap_path).head(20).to_string(index=False))


def direction_question(name: str) -> str:
    if name.startswith("cal") and "tempdip" not in name and "region" not in name and "area" not in name and "opening" not in name:
        return "Does partial leap/calendar normalization improve public baseline direction?"
    if "tempdip" in name:
        return "Does weak-February temporary-dip rebound improve public direction?"
    if "region" in name or "area" in name or "opening" in name:
        return "Do heavily shrunk cohort seasonal factors transfer to public?"
    if "mom" in name:
        return "Does changing conservative momentum exponent improve public direction?"
    return "General heuristic direction test"


if __name__ == "__main__":
    main()
