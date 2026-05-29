from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = ROOT / "analysis" / "new_strategy"
sys.path.append(str(STRATEGY_DIR))

import train_shrunk_yoy_cohort_baseline as sy  # noqa: E402


OUT_DIR = ROOT / "results" / "new_strategy_shrunk_yoy_ml"
SUBMISSIONS_DIR = ROOT / "submissions"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    alpha: float
    clip: float
    learning_rate: float = 0.04
    max_leaf_nodes: int = 15
    l2_regularization: float = 1.0


def make_baseline_specs() -> list[sy.Spec]:
    return [
        sy.Spec(
            name="shrunk_yoy_feb_day_adjusted_age_const1_linear_k20_c0p3",
            growth_window="feb",
            calendar_mode="day_adjusted",
            group_col="age",
            store_weight_mode="const1.00",
            blend_space="linear",
            group_shrink_k=20.0,
            outlier_clip=0.30,
            min_group_size=40,
        ),
        sy.Spec(
            name="shrunk_yoy_feb_day_adjusted_area_age_const1_linear_k20_c0p3",
            growth_window="feb",
            calendar_mode="day_adjusted",
            group_col="area_age",
            store_weight_mode="const1.00",
            blend_space="linear",
            group_shrink_k=20.0,
            outlier_clip=0.30,
            min_group_size=80,
        ),
        sy.Spec(
            name="shrunk_yoy_janfeb_raw_age_volume_const1_linear_k300_c0p3",
            growth_window="janfeb",
            calendar_mode="raw",
            group_col="age_volume",
            store_weight_mode="const1.00",
            blend_space="linear",
            group_shrink_k=300.0,
            outlier_clip=0.30,
            min_group_size=80,
        ),
        sy.Spec(
            name="shrunk_yoy_feb_day_adjusted_region_age_const1_linear_k20_c0p3",
            growth_window="feb",
            calendar_mode="day_adjusted",
            group_col="region_age",
            store_weight_mode="const1.00",
            blend_space="linear",
            group_shrink_k=20.0,
            outlier_clip=0.30,
            min_group_size=80,
        ),
    ]


def model_specs() -> list[ModelSpec]:
    specs = []
    for clip in [0.03, 0.06, 0.10, 0.16]:
        for alpha in [0.35, 0.55, 0.75, 1.00]:
            specs.append(ModelSpec(name=f"hgb_a{alpha}_c{clip}", kind="hgb", alpha=alpha, clip=clip))
    for clip in [0.03, 0.06, 0.10]:
        for alpha in [0.35, 0.55, 0.75, 1.00]:
            specs.append(ModelSpec(name=f"ridge_a{alpha}_c{clip}", kind="ridge", alpha=alpha, clip=clip))
    return specs


def build_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rto, static, meta_oof, meta_test = sy.load_data()
    val = sy.target_frame(rto, static, meta_oof, target_year=2024, is_train=True)
    test = sy.target_frame(rto, static, meta_test, target_year=2025, is_train=False)
    if "fold" not in val.columns:
        val = val.merge(meta_oof[["new_id", "fold"]], on="new_id", how="left")
    specs = make_baseline_specs()
    for idx, spec_val in enumerate(specs):
        spec_test = sy.Spec(
            name=spec_val.name.replace("2024", "2025"),
            growth_window=spec_val.growth_window,
            calendar_mode=spec_val.calendar_mode,
            group_col=spec_val.group_col,
            store_weight_mode=spec_val.store_weight_mode,
            blend_space=spec_val.blend_space,
            group_shrink_k=spec_val.group_shrink_k,
            outlier_clip=spec_val.outlier_clip,
            min_group_size=spec_val.min_group_size,
        )
        pred_val, store_g_val, cohort_g_val, group_size_val = sy.forecast_for_year(val, spec_val, 2024)
        pred_test, store_g_test, cohort_g_test, group_size_test = sy.forecast_for_year(test, spec_test, 2025)
        prefix = f"sy{idx}"
        val[f"{prefix}_pred"] = pred_val
        test[f"{prefix}_pred"] = pred_test
        val[f"{prefix}_store_growth"] = store_g_val
        test[f"{prefix}_store_growth"] = store_g_test
        val[f"{prefix}_cohort_growth"] = cohort_g_val
        test[f"{prefix}_cohort_growth"] = cohort_g_test
        val[f"{prefix}_growth_delta_log"] = np.log(np.clip(store_g_val, 1e-6, None) / np.clip(cohort_g_val, 1e-6, None))
        test[f"{prefix}_growth_delta_log"] = np.log(np.clip(store_g_test, 1e-6, None) / np.clip(cohort_g_test, 1e-6, None))
        val[f"{prefix}_group_size"] = group_size_val
        test[f"{prefix}_group_size"] = group_size_test

    val["base_pred"] = val["sy0_pred"]
    test["base_pred"] = test["sy0_pred"]
    val["target_residual_log"] = np.log(np.clip(val["target"], 1.0, None) / np.clip(val["base_pred"], 1.0, None))
    val["pred265_residual_log"] = np.log(np.clip(val["target"], 1.0, None) / np.clip(val["pred265"], 1.0, None))
    return val, test, meta_oof, meta_test


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        "jan",
        "feb",
        "dec_prev",
        "mar_prev_year",
        "jan_prev_year",
        "feb_prev_year",
        "feb_to_jan",
        "volatility_rel3",
        "growth_stability_proxy",
    ]
    for idx in range(4):
        numeric.extend(
            [
                f"sy{idx}_pred",
                f"sy{idx}_store_growth",
                f"sy{idx}_cohort_growth",
                f"sy{idx}_growth_delta_log",
                f"sy{idx}_group_size",
            ]
        )
    numeric.extend(
        [
            "Численность населения",
            "Трафик пеший, в час",
            "Трафик авто, в час",
            "Продуктовые магазины (500 м)",
            "Пятерочки (500 м)",
            "Количество касс",
            "Флаг алкогольной лицензии",
        ]
    )
    categorical = [
        "region",
        "city_big",
        "area",
        "age",
        "region_area",
        "region_age",
        "area_age",
        "volume_bin",
        "current_momentum_bin",
        "volatility_bin",
        "stability_bin",
        "pyat_bin",
        "grocery_bin",
    ]
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def make_model(spec: ModelSpec, numeric: list[str], categorical: list[str]) -> Pipeline:
    if spec.kind == "hgb":
        pre = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), numeric),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                        ]
                    ),
                    categorical,
                ),
            ],
            verbose_feature_names_out=False,
        )
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=spec.learning_rate,
            max_iter=260,
            max_leaf_nodes=spec.max_leaf_nodes,
            min_samples_leaf=80,
            l2_regularization=spec.l2_regularization,
            random_state=42,
        )
        return Pipeline([("pre", pre), ("model", model)])
    if spec.kind == "ridge":
        pre = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                    numeric,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=True),
                    categorical,
                ),
            ]
        )
        return Pipeline([("pre", pre), ("model", Ridge(alpha=20.0))])
    raise ValueError(spec.kind)


def safe_apply_residual(base: np.ndarray, residual: np.ndarray, alpha: float, clip: float) -> np.ndarray:
    return np.clip(base, 1.0, None) * np.exp(np.clip(alpha * residual, -clip, clip))


def run_oof(val: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    numeric, categorical = feature_columns(val)
    features = numeric + categorical
    oof_residual = np.zeros(len(val), dtype=float)
    target = val["target_residual_log"].to_numpy(float)
    folds = sorted(val["fold"].dropna().unique().astype(int))
    for fold in folds:
        train_mask = val["fold"].astype(int).ne(fold).to_numpy()
        valid_mask = val["fold"].astype(int).eq(fold).to_numpy()
        model = make_model(spec, numeric, categorical)
        model.fit(val.loc[train_mask, features], target[train_mask])
        oof_residual[valid_mask] = model.predict(val.loc[valid_mask, features])

    final_model = make_model(spec, numeric, categorical)
    final_model.fit(val[features], target)
    test_residual = final_model.predict(test[features])
    return oof_residual, test_residual


def evaluate_candidate(val: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec, oof_resid: np.ndarray, test_resid: np.ndarray) -> dict[str, object]:
    pred_val = safe_apply_residual(val["base_pred"].to_numpy(float), oof_resid, spec.alpha, spec.clip)
    pred_test = safe_apply_residual(test["base_pred"].to_numpy(float), test_resid, spec.alpha, spec.clip)
    pred265_mape = sy.mape(val["target"], val["pred265"])
    base_mape = sy.mape(val["target"], val["base_pred"])
    candidate_mape = sy.mape(val["target"], pred_val)
    shift = np.abs(pred_test / np.clip(test["pred265"].to_numpy(float), 1.0, None) - 1.0) * 100.0
    return {
        "candidate": spec.name,
        "kind": spec.kind,
        "alpha": spec.alpha,
        "clip": spec.clip,
        "base_oof_mape": base_mape,
        "pred265_oof_mape": pred265_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_shrunk_yoy_base": candidate_mape - base_mape,
        "delta_vs_pred265": candidate_mape - pred265_mape,
        "residual_mae": mean_absolute_error(val["target_residual_log"], oof_resid),
        "test_mean_abs_shift_pct_vs_pred265": float(np.mean(shift)),
        "test_max_abs_shift_pct_vs_pred265": float(np.max(shift)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    val, test, _, _ = build_data()
    rows = []
    pred_store = {}
    for spec in model_specs():
        print(f"training {spec.name}")
        oof_resid, test_resid = run_oof(val, test, spec)
        row = evaluate_candidate(val, test, spec, oof_resid, test_resid)
        rows.append(row)
        pred_store[spec.name] = (
            safe_apply_residual(val["base_pred"].to_numpy(float), oof_resid, spec.alpha, spec.clip),
            safe_apply_residual(test["base_pred"].to_numpy(float), test_resid, spec.alpha, spec.clip),
        )

    summary = pd.DataFrame(rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"]).reset_index(drop=True)
    summary["accepted_for_submission"] = (
        (summary["delta_vs_pred265"] <= -0.015)
        & (summary["test_mean_abs_shift_pct_vs_pred265"] <= 6.0)
        & (summary["test_max_abs_shift_pct_vs_pred265"] <= 140.0)
    )
    summary.to_csv(OUT_DIR / "shrunk_yoy_ml_summary.csv", index=False)

    accepted_rows = []
    for _, row in summary.loc[summary["accepted_for_submission"]].head(3).iterrows():
        name = str(row["candidate"])
        _, pred_test = pred_store[name]
        filename = f"stage2_shrunk_yoy_ml_{name}.csv".replace(".", "p")
        sub = pd.DataFrame({"new_id": test["new_id"].astype(int), "rto": np.clip(pred_test, 1.0, None)})
        if len(sub) != 18657 or sub["rto"].isna().any() or (sub["rto"] <= 0).any():
            raise RuntimeError(f"Bad submission format for {filename}")
        sub.to_csv(SUBMISSIONS_DIR / filename, index=False)
        accepted = row.to_dict()
        accepted["filename"] = filename
        accepted_rows.append(accepted)
    pd.DataFrame(accepted_rows).to_csv(OUT_DIR / "submission_candidates.csv", index=False)

    print("Top Shrunk YoY + ML residual candidates:")
    print(
        summary[
            [
                "candidate",
                "base_oof_mape",
                "candidate_oof_mape",
                "delta_vs_shrunk_yoy_base",
                "delta_vs_pred265",
                "test_mean_abs_shift_pct_vs_pred265",
                "test_max_abs_shift_pct_vs_pred265",
                "accepted_for_submission",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )
    print(f"Accepted submission candidates: {len(accepted_rows)}")


if __name__ == "__main__":
    main()
