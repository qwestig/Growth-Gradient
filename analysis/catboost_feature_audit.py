from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from catboost_residual_diagnostics import (
    BEHAVIOR_COLS,
    CAT_COLS,
    MODEL_PARAMS,
    STATIC_NUM_COLS,
    SUBMISSIONS_DIR,
    TARGET,
    TRAIN_PATH,
    add_base,
    apply_residual,
    make_panel,
    mape,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "catboost_feature_audit"
ALPHA = 0.70

LEVEL_COLS = [
    "lag1",
    "lag2",
    "lag3",
    "lag4",
    "lag5",
    "lag6",
    "lag12",
    "lag13",
    "rolling3_mean",
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
]

CURRENT_DYNAMICS_COLS = [
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
    "volatility_rel3",
    "volatility_rel6",
]

BEHAVIOR_FEATURE_COLS = [
    feature
    for col in BEHAVIOR_COLS
    for feature in [
        f"{col}_lag1",
        f"{col}_lag2",
        f"{col}_lag3",
        f"{col}_lag12",
        f"{col}_lag13",
        f"{col}_diff1",
        f"{col}_yoy_shift",
    ]
]

TREND_COLS = [
    "growth1",
    "growth2",
    "growth3",
    "recent_mean3_to_prev3",
    "recent_mean2_to_prev2",
    "slope3_rel",
    "slope6_rel",
    "slope12_rel",
    "log_slope3",
    "log_slope6",
    "log_slope12",
]

ANOMALY_COLS = [
    "lag1_vs_prev5_median",
    "lag1_vs_prev5_mean",
    "lag1_z_prev5",
    "lag1_position_prev5",
    "range_rel6",
    "jump1_abs",
    "jump1_abs_vs_prev_jumps",
]

SEASONAL_CONTEXT_COLS = [
    "prior_feb_jan_ratio",
    "prior_mar_jan_ratio",
    "feb_yoy_vs_jan_yoy",
    "region_prev_mar_feb_ratio",
    "area_prev_mar_feb_ratio",
    "opening_prev_mar_feb_ratio",
    "store_vs_region_prev_ratio",
    "store_vs_area_prev_ratio",
]

STABILITY_COLS = [
    "volatility_rel12",
    "range_rel12",
    "recent_level_to_year",
    "median_abs_growth3",
    "mean_abs_growth5",
    "max_abs_growth5",
    "positive_growth_share5",
    "slope3_minus_slope6",
    "slope6_minus_slope12",
    "log_slope3_minus6",
]

HEURISTIC_TRUST_COLS = [
    "momentum_gap_abs",
    "momentum_clip_delta",
    "momentum_was_clipped",
    "short_volatility_to_long",
    "short_range_to_long",
    "recent_mean_to_recent_median",
    "store_vs_global_prev_ratio",
    "store_prev_ratio_abs_dev_global",
    "store_prev_ratio_abs_dev_region",
    "store_prev_ratio_abs_dev_area",
]

Q1_CONTEXT_COLS = [
    "current_feb_jan_ratio",
    "jan_yoy",
    "feb_yoy",
    "current_janfeb_yoy",
    "feb_jan_transition_shift",
    "prior_mar_to_janfeb_mean",
    "prior_q1_curvature",
    "q1_scale_pred",
    "q1_scale_pred_to_base",
]

ALT_BASE_COLS = [
    "base_global_without_momentum",
    "base_store_prev_mar_feb",
    "base_store_prev_mar_jan",
    "base_prior_mar_scaled_jan_yoy",
    "base_prior_mar_scaled_feb_yoy",
    "base_prior_mar_scaled_q1_yoy",
    "base_prior_mar_scaled_yoy_mean",
    "base_janfeb_blend_prev_q1",
    "base_global_without_momentum_to_base",
    "base_store_prev_mar_feb_to_base",
    "base_store_prev_mar_jan_to_base",
    "base_prior_mar_scaled_jan_yoy_to_base",
    "base_prior_mar_scaled_feb_yoy_to_base",
    "base_prior_mar_scaled_q1_yoy_to_base",
    "base_prior_mar_scaled_yoy_mean_to_base",
    "base_janfeb_blend_prev_q1_to_base",
    "alt_base_ratio_mean",
    "alt_base_ratio_median",
    "alt_base_ratio_std",
    "alt_base_ratio_range",
    "alt_base_ratio_min",
    "alt_base_ratio_max",
    "jan_based_vs_feb_based_base",
    "prior_mar_scaled_yoy_spread",
]

LOG_DIFF_COLS = [
    "log_lag1",
    "log_lag2",
    "log_lag3",
    "log_lag12",
    "log_lag13",
    "log_growth1",
    "log_growth2",
    "log_growth3",
    "log_growth4",
    "log_growth5",
    "log_growth_mean3",
    "log_growth_mean5",
    "log_growth_std5",
    "log_growth_abs_mean5",
    "log_yoy_jan",
    "log_yoy_feb",
    "log_yoy_janfeb",
    "log_yoy_feb_minus_jan",
    "log_prior_mar_feb_ratio",
    "log_prior_mar_jan_ratio",
    "log_current_feb_jan_ratio",
    "log_feb_jan_transition_shift",
    "log_q1_scale_pred_to_base",
    "log_store_vs_region_prev_ratio",
    "log_store_vs_global_prev_ratio",
]

LOG_ACCEL_COLS = [
    "log_accel1",
    "log_accel2",
    "log_accel_abs1",
    "log_accel_mean_abs",
    "log_slope_delta_short_long",
]

GROUP_PREFIXES = ["region", "area", "opening", "region_area"]
GROUP_STATS_COLS = [
    feature
    for prefix in GROUP_PREFIXES
    for feature in [
        f"{prefix}_group_size",
        f"{prefix}_lag1_mean",
        f"{prefix}_lag1_median",
        f"{prefix}_lag1_q25",
        f"{prefix}_lag1_q75",
        f"{prefix}_lag1_iqr_rel",
        f"store_lag1_to_{prefix}_median",
        f"store_lag1_rank_pct_{prefix}",
        f"{prefix}_momentum_median",
        f"store_momentum_vs_{prefix}",
        f"{prefix}_growth1_median",
        f"store_growth1_vs_{prefix}",
        f"{prefix}_feb_yoy_median",
        f"store_feb_yoy_vs_{prefix}",
        f"{prefix}_volatility_rel3_median",
        f"store_volatility_vs_{prefix}",
    ]
]


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def safe_log(value: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(value.where(value > 0))


def relative_slope(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].to_numpy(dtype=float)
    x = np.arange(len(columns), dtype=float)
    x -= x.mean()
    slope = np.nansum(values * x, axis=1) / np.sum(x**2)
    valid_count = np.sum(~np.isnan(values), axis=1)
    level = np.divide(
        np.nansum(values, axis=1),
        valid_count,
        out=np.full(len(frame), np.nan, dtype=float),
        where=valid_count > 0,
    )
    return pd.Series(slope / level, index=frame.index)


def log_slope(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].to_numpy(dtype=float)
    logs = np.log(np.clip(values, 1.0, None))
    x = np.arange(len(columns), dtype=float)
    x -= x.mean()
    slope = np.nansum(logs * x, axis=1) / np.sum(x**2)
    return pd.Series(slope, index=frame.index)


def add_feature_audit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    target_group = out.groupby("new_id")[TARGET]
    for lag in [7, 8, 9, 10, 11, 14]:
        out[f"lag{lag}"] = target_group.shift(lag)

    out["growth1"] = ratio(out["lag1"], out["lag2"])
    out["growth2"] = ratio(out["lag2"], out["lag3"])
    out["growth3"] = ratio(out["lag3"], out["lag4"])
    out["growth4"] = ratio(out["lag4"], out["lag5"])
    out["growth5"] = ratio(out["lag5"], out["lag6"])
    out["recent_mean3_to_prev3"] = ratio(
        out[["lag1", "lag2", "lag3"]].mean(axis=1),
        out[["lag4", "lag5", "lag6"]].mean(axis=1),
    )
    out["recent_mean2_to_prev2"] = ratio(
        out[["lag1", "lag2"]].mean(axis=1),
        out[["lag3", "lag4"]].mean(axis=1),
    )
    out["slope3_rel"] = relative_slope(out, ["lag3", "lag2", "lag1"])
    out["slope6_rel"] = relative_slope(out, ["lag6", "lag5", "lag4", "lag3", "lag2", "lag1"])
    out["slope12_rel"] = relative_slope(
        out,
        ["lag12", "lag11", "lag10", "lag9", "lag8", "lag7", "lag6", "lag5", "lag4", "lag3", "lag2", "lag1"],
    )
    out["log_slope3"] = log_slope(out, ["lag3", "lag2", "lag1"])
    out["log_slope6"] = log_slope(out, ["lag6", "lag5", "lag4", "lag3", "lag2", "lag1"])
    out["log_slope12"] = log_slope(
        out,
        ["lag12", "lag11", "lag10", "lag9", "lag8", "lag7", "lag6", "lag5", "lag4", "lag3", "lag2", "lag1"],
    )

    previous5 = out[["lag2", "lag3", "lag4", "lag5", "lag6"]]
    previous5_mean = previous5.mean(axis=1)
    previous5_median = previous5.median(axis=1)
    previous5_std = previous5.std(axis=1)
    previous5_min = previous5.min(axis=1)
    previous5_max = previous5.max(axis=1)
    out["lag1_vs_prev5_median"] = ratio(out["lag1"], previous5_median)
    out["lag1_vs_prev5_mean"] = ratio(out["lag1"], previous5_mean)
    out["lag1_z_prev5"] = (out["lag1"] - previous5_mean) / previous5_std.replace(0, np.nan)
    out["lag1_position_prev5"] = (out["lag1"] - previous5_min) / (
        previous5_max - previous5_min
    ).replace(0, np.nan)
    out["range_rel6"] = ratio(
        out[["lag1", "lag2", "lag3", "lag4", "lag5", "lag6"]].max(axis=1)
        - out[["lag1", "lag2", "lag3", "lag4", "lag5", "lag6"]].min(axis=1),
        out[["lag1", "lag2", "lag3", "lag4", "lag5", "lag6"]].mean(axis=1),
    )
    out["jump1_abs"] = (out["growth1"] - 1).abs()
    previous_jumps = pd.concat(
        [(out["growth2"] - 1).abs(), (out["growth3"] - 1).abs()], axis=1
    ).mean(axis=1)
    out["jump1_abs_vs_prev_jumps"] = ratio(out["jump1_abs"], previous_jumps)

    out["prior_feb_jan_ratio"] = ratio(out["lag13"], out["lag14"])
    out["prior_mar_jan_ratio"] = ratio(out["lag12"], out["lag14"])
    jan_yoy = ratio(out["lag2"], out["lag14"])
    feb_yoy = ratio(out["lag1"], out["lag13"])
    out["feb_yoy_vs_jan_yoy"] = ratio(feb_yoy, jan_yoy)

    ratio_by_period = out.groupby("period")["store_mar_feb_ratio"]
    out["global_prev_mar_feb_ratio"] = ratio_by_period.transform("median")
    seasonal_group_specs = {
        "region": "Регион",
        "area": "Торговая площадь, категориальный",
        "opening": "Дата открытия, категориальный",
    }
    for prefix, group_col in seasonal_group_specs.items():
        feature = f"{prefix}_prev_mar_feb_ratio"
        out[feature] = out.groupby(["period", group_col])["store_mar_feb_ratio"].transform("median")

    out["store_vs_region_prev_ratio"] = ratio(
        out["store_mar_feb_ratio"], out["region_prev_mar_feb_ratio"]
    )
    out["store_vs_area_prev_ratio"] = ratio(out["store_mar_feb_ratio"], out["area_prev_mar_feb_ratio"])

    last12 = [
        "lag12",
        "lag11",
        "lag10",
        "lag9",
        "lag8",
        "lag7",
        "lag6",
        "lag5",
        "lag4",
        "lag3",
        "lag2",
        "lag1",
    ]
    last12_mean = out[last12].mean(axis=1)
    last12_std = out[last12].std(axis=1)
    last12_range = out[last12].max(axis=1) - out[last12].min(axis=1)
    last6_median = out[["lag1", "lag2", "lag3", "lag4", "lag5", "lag6"]].median(axis=1)
    growth_abs = pd.concat(
        [(out[f"growth{i}"] - 1).abs() for i in range(1, 6)],
        axis=1,
    )
    growth_signed = pd.concat(
        [(out[f"growth{i}"] - 1) > 0 for i in range(1, 6)],
        axis=1,
    )
    out["volatility_rel12"] = ratio(last12_std, last12_mean)
    out["range_rel12"] = ratio(last12_range, last12_mean)
    out["recent_level_to_year"] = ratio(out["roll3_mean"], last12_mean)
    out["median_abs_growth3"] = growth_abs.iloc[:, :3].median(axis=1)
    out["mean_abs_growth5"] = growth_abs.mean(axis=1)
    out["max_abs_growth5"] = growth_abs.max(axis=1)
    out["positive_growth_share5"] = growth_signed.mean(axis=1)
    out["slope3_minus_slope6"] = out["slope3_rel"] - out["slope6_rel"]
    out["slope6_minus_slope12"] = out["slope6_rel"] - out["slope12_rel"]
    out["log_slope3_minus6"] = out["log_slope3"] - out["log_slope6"]

    out["momentum_gap_abs"] = (out["momentum_raw"] - 1).abs()
    out["momentum_clip_delta"] = (out["momentum_raw"] - out["momentum_clip"]).abs()
    out["momentum_was_clipped"] = (out["momentum_clip_delta"] > 0).astype(float)
    out["short_volatility_to_long"] = ratio(out["volatility_rel3"], out["volatility_rel12"])
    out["short_range_to_long"] = ratio(out["range_rel6"], out["range_rel12"])
    out["recent_mean_to_recent_median"] = ratio(out["roll3_mean"], last6_median)
    out["store_vs_global_prev_ratio"] = ratio(
        out["store_mar_feb_ratio"], out["global_prev_mar_feb_ratio"]
    )
    out["store_prev_ratio_abs_dev_global"] = (out["store_vs_global_prev_ratio"] - 1).abs()
    out["store_prev_ratio_abs_dev_region"] = (out["store_vs_region_prev_ratio"] - 1).abs()
    out["store_prev_ratio_abs_dev_area"] = (out["store_vs_area_prev_ratio"] - 1).abs()

    out["current_feb_jan_ratio"] = out["growth1"]
    out["jan_yoy"] = jan_yoy
    out["feb_yoy"] = feb_yoy
    out["current_janfeb_yoy"] = ratio(
        out[["lag1", "lag2"]].mean(axis=1),
        out[["lag13", "lag14"]].mean(axis=1),
    )
    out["feb_jan_transition_shift"] = ratio(
        out["current_feb_jan_ratio"], out["prior_feb_jan_ratio"]
    )
    out["prior_mar_to_janfeb_mean"] = ratio(
        out["lag12"], out[["lag13", "lag14"]].mean(axis=1)
    )
    out["prior_q1_curvature"] = ratio(out["store_mar_feb_ratio"], out["prior_feb_jan_ratio"])
    out["q1_scale_pred"] = out["lag12"] * out["current_janfeb_yoy"]
    out["q1_scale_pred_to_base"] = ratio(out["q1_scale_pred"], out["best_base"])

    out["base_global_without_momentum"] = out["lag1"] * out["global_prev_mar_feb_ratio"]
    out["base_store_prev_mar_feb"] = out["lag1"] * out["store_mar_feb_ratio"]
    out["base_store_prev_mar_jan"] = out["lag2"] * out["prior_mar_jan_ratio"]
    out["base_prior_mar_scaled_jan_yoy"] = out["lag12"] * out["jan_yoy"]
    out["base_prior_mar_scaled_feb_yoy"] = out["lag12"] * out["feb_yoy"]
    out["base_prior_mar_scaled_q1_yoy"] = out["q1_scale_pred"]
    out["base_prior_mar_scaled_yoy_mean"] = out["lag12"] * out[["jan_yoy", "feb_yoy"]].mean(
        axis=1
    )
    out["base_janfeb_blend_prev_q1"] = out[["lag1", "lag2"]].mean(axis=1) * out[
        "prior_mar_to_janfeb_mean"
    ]

    alternative_bases = [
        "base_global_without_momentum",
        "base_store_prev_mar_feb",
        "base_store_prev_mar_jan",
        "base_prior_mar_scaled_jan_yoy",
        "base_prior_mar_scaled_feb_yoy",
        "base_prior_mar_scaled_q1_yoy",
        "base_prior_mar_scaled_yoy_mean",
        "base_janfeb_blend_prev_q1",
    ]
    ratio_cols = []
    for base_col in alternative_bases:
        ratio_col = f"{base_col}_to_base"
        out[ratio_col] = ratio(out[base_col], out["best_base"])
        ratio_cols.append(ratio_col)
    out["alt_base_ratio_mean"] = out[ratio_cols].mean(axis=1)
    out["alt_base_ratio_median"] = out[ratio_cols].median(axis=1)
    out["alt_base_ratio_std"] = out[ratio_cols].std(axis=1)
    out["alt_base_ratio_range"] = out[ratio_cols].max(axis=1) - out[ratio_cols].min(axis=1)
    out["alt_base_ratio_min"] = out[ratio_cols].min(axis=1)
    out["alt_base_ratio_max"] = out[ratio_cols].max(axis=1)
    out["jan_based_vs_feb_based_base"] = ratio(
        out["base_store_prev_mar_jan"], out["base_store_prev_mar_feb"]
    )
    out["prior_mar_scaled_yoy_spread"] = ratio(
        out[["base_prior_mar_scaled_jan_yoy", "base_prior_mar_scaled_feb_yoy"]].max(axis=1),
        out[["base_prior_mar_scaled_jan_yoy", "base_prior_mar_scaled_feb_yoy"]].min(axis=1),
    )

    for lag in [1, 2, 3, 12, 13]:
        out[f"log_lag{lag}"] = safe_log(out[f"lag{lag}"])
    for idx in range(1, 6):
        out[f"log_growth{idx}"] = safe_log(out[f"growth{idx}"])
    growth_logs = [f"log_growth{i}" for i in range(1, 6)]
    out["log_growth_mean3"] = out[growth_logs[:3]].mean(axis=1)
    out["log_growth_mean5"] = out[growth_logs].mean(axis=1)
    out["log_growth_std5"] = out[growth_logs].std(axis=1)
    out["log_growth_abs_mean5"] = out[growth_logs].abs().mean(axis=1)

    out["log_yoy_jan"] = safe_log(out["jan_yoy"])
    out["log_yoy_feb"] = safe_log(out["feb_yoy"])
    out["log_yoy_janfeb"] = safe_log(out["current_janfeb_yoy"])
    out["log_yoy_feb_minus_jan"] = out["log_yoy_feb"] - out["log_yoy_jan"]
    out["log_prior_mar_feb_ratio"] = safe_log(out["store_mar_feb_ratio"])
    out["log_prior_mar_jan_ratio"] = safe_log(out["prior_mar_jan_ratio"])
    out["log_current_feb_jan_ratio"] = safe_log(out["current_feb_jan_ratio"])
    out["log_feb_jan_transition_shift"] = safe_log(out["feb_jan_transition_shift"])
    out["log_q1_scale_pred_to_base"] = safe_log(out["q1_scale_pred_to_base"])
    out["log_store_vs_region_prev_ratio"] = safe_log(out["store_vs_region_prev_ratio"])
    out["log_store_vs_global_prev_ratio"] = safe_log(out["store_vs_global_prev_ratio"])

    out["log_accel1"] = out["log_growth1"] - out["log_growth2"]
    out["log_accel2"] = out["log_growth2"] - out["log_growth3"]
    out["log_accel_abs1"] = out["log_accel1"].abs()
    out["log_accel_mean_abs"] = out[["log_accel1", "log_accel2"]].abs().mean(axis=1)
    out["log_slope_delta_short_long"] = out["log_slope3"] - out["log_slope12"]

    out["region_area_group"] = out["Регион"].astype(str) + " | " + out[
        "Торговая площадь, категориальный"
    ].astype(str)
    group_specs = {
        "region": "Регион",
        "area": "Торговая площадь, категориальный",
        "opening": "Дата открытия, категориальный",
        "region_area": "region_area_group",
    }
    for prefix, group_col in group_specs.items():
        grouped = out.groupby(["period", group_col], observed=True)
        lag1_group = grouped["lag1"]
        out[f"{prefix}_group_size"] = lag1_group.transform("count")
        out[f"{prefix}_lag1_mean"] = lag1_group.transform("mean")
        out[f"{prefix}_lag1_median"] = lag1_group.transform("median")
        out[f"{prefix}_lag1_q25"] = lag1_group.transform(lambda values: values.quantile(0.25))
        out[f"{prefix}_lag1_q75"] = lag1_group.transform(lambda values: values.quantile(0.75))
        out[f"{prefix}_lag1_iqr_rel"] = ratio(
            out[f"{prefix}_lag1_q75"] - out[f"{prefix}_lag1_q25"],
            out[f"{prefix}_lag1_median"],
        )
        out[f"store_lag1_to_{prefix}_median"] = ratio(out["lag1"], out[f"{prefix}_lag1_median"])
        out[f"store_lag1_rank_pct_{prefix}"] = grouped["lag1"].rank(pct=True)

        out[f"{prefix}_momentum_median"] = grouped["momentum_raw"].transform("median")
        out[f"store_momentum_vs_{prefix}"] = out["momentum_raw"] - out[f"{prefix}_momentum_median"]
        out[f"{prefix}_growth1_median"] = grouped["growth1"].transform("median")
        out[f"store_growth1_vs_{prefix}"] = out["growth1"] - out[f"{prefix}_growth1_median"]
        out[f"{prefix}_feb_yoy_median"] = grouped["feb_yoy"].transform("median")
        out[f"store_feb_yoy_vs_{prefix}"] = out["feb_yoy"] - out[f"{prefix}_feb_yoy_median"]
        out[f"{prefix}_volatility_rel3_median"] = grouped["volatility_rel3"].transform("median")
        out[f"store_volatility_vs_{prefix}"] = out["volatility_rel3"] - out[
            f"{prefix}_volatility_rel3_median"
        ]
    return out.replace([np.inf, -np.inf], np.nan)


def current_blocks() -> dict[str, list[str]]:
    return {
        "level": LEVEL_COLS,
        "dynamics": CURRENT_DYNAMICS_COLS,
        "static_num": STATIC_NUM_COLS,
        "categorical": CAT_COLS,
        "behavior": BEHAVIOR_FEATURE_COLS,
    }


def unique_features(blocks: list[list[str]]) -> list[str]:
    return list(dict.fromkeys(feature for block in blocks for feature in block))


def config_features() -> dict[str, list[str]]:
    blocks = current_blocks()
    current = unique_features(list(blocks.values()))
    current_without_dynamics = unique_features(
        [blocks["level"], blocks["static_num"], blocks["categorical"], blocks["behavior"]]
    )
    current_without_behavior = unique_features(
        [blocks["level"], blocks["dynamics"], blocks["static_num"], blocks["categorical"]]
    )
    current_without_static = unique_features([blocks["level"], blocks["dynamics"], blocks["behavior"]])
    current_without_level = unique_features(
        [blocks["dynamics"], blocks["static_num"], blocks["categorical"], blocks["behavior"]]
    )
    all_new = unique_features([current, TREND_COLS, ANOMALY_COLS, SEASONAL_CONTEXT_COLS])
    all_new_without_behavior = unique_features(
        [current_without_behavior, TREND_COLS, ANOMALY_COLS, SEASONAL_CONTEXT_COLS]
    )
    return {
        "current_all": current,
        "current_minus_dynamics": current_without_dynamics,
        "current_minus_behavior": current_without_behavior,
        "current_minus_static": current_without_static,
        "current_minus_level": current_without_level,
        "current_plus_trend": unique_features([current, TREND_COLS]),
        "current_plus_anomaly": unique_features([current, ANOMALY_COLS]),
        "current_plus_seasonal_context": unique_features([current, SEASONAL_CONTEXT_COLS]),
        "current_plus_trend_anomaly": unique_features([current, TREND_COLS, ANOMALY_COLS]),
        "current_plus_all_new": all_new,
        "all_new_minus_behavior": all_new_without_behavior,
        "all_new_plus_stability": unique_features([all_new, STABILITY_COLS]),
        "all_new_plus_heuristic_trust": unique_features([all_new, HEURISTIC_TRUST_COLS]),
        "all_new_plus_q1_context": unique_features([all_new, Q1_CONTEXT_COLS]),
        "all_new_plus_targeted": unique_features(
            [all_new, STABILITY_COLS, HEURISTIC_TRUST_COLS, Q1_CONTEXT_COLS]
        ),
        "all_new_plus_alt_bases": unique_features([all_new, ALT_BASE_COLS]),
        "targeted_plus_alt_bases": unique_features(
            [all_new, STABILITY_COLS, HEURISTIC_TRUST_COLS, Q1_CONTEXT_COLS, ALT_BASE_COLS]
        ),
        "heuristic_trust_plus_alt_bases": unique_features(
            [all_new, HEURISTIC_TRUST_COLS, ALT_BASE_COLS]
        ),
        "targeted_minus_behavior": unique_features(
            [
                all_new_without_behavior,
                STABILITY_COLS,
                HEURISTIC_TRUST_COLS,
                Q1_CONTEXT_COLS,
            ]
        ),
        "targeted_plus_log_diff": unique_features(
            [all_new, STABILITY_COLS, HEURISTIC_TRUST_COLS, Q1_CONTEXT_COLS, LOG_DIFF_COLS]
        ),
        "targeted_plus_log_diff_accel": unique_features(
            [
                all_new,
                STABILITY_COLS,
                HEURISTIC_TRUST_COLS,
                Q1_CONTEXT_COLS,
                LOG_DIFF_COLS,
                LOG_ACCEL_COLS,
            ]
        ),
        "targeted_log_diff_minus_behavior": unique_features(
            [
                all_new_without_behavior,
                STABILITY_COLS,
                HEURISTIC_TRUST_COLS,
                Q1_CONTEXT_COLS,
                LOG_DIFF_COLS,
                LOG_ACCEL_COLS,
            ]
        ),
        "all_new_plus_log_diff": unique_features([all_new, LOG_DIFF_COLS]),
        "all_new_plus_group_stats": unique_features([all_new, GROUP_STATS_COLS]),
        "targeted_plus_group_stats": unique_features(
            [all_new, STABILITY_COLS, HEURISTIC_TRUST_COLS, Q1_CONTEXT_COLS, GROUP_STATS_COLS]
        ),
        "targeted_group_minus_behavior": unique_features(
            [
                all_new_without_behavior,
                STABILITY_COLS,
                HEURISTIC_TRUST_COLS,
                Q1_CONTEXT_COLS,
                GROUP_STATS_COLS,
            ]
        ),
    }


def fit_pool(frame: pd.DataFrame, features: list[str], target: str) -> Pool:
    cat_idx = [features.index(col) for col in CAT_COLS if col in features]
    return Pool(frame[features], frame[target], cat_features=cat_idx)


def predict_pool(model: CatBoostRegressor, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    cat_idx = [features.index(col) for col in CAT_COLS if col in features]
    return model.predict(Pool(frame[features], cat_features=cat_idx))


def evaluate_config(frame: pd.DataFrame, features: list[str], name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    oof_parts = []
    for fold in range(5):
        valid_mask = (frame["new_id"] % 5) == fold
        train = frame[~valid_mask]
        valid = frame[valid_mask].copy()

        model = CatBoostRegressor(**MODEL_PARAMS)
        model.fit(fit_pool(train, features, "target_log_ratio"))
        pred_log = predict_pool(model, valid, features)
        pred = apply_residual(
            valid["best_base"].to_numpy(dtype=float),
            pred_log,
            ALPHA,
            None,
        )
        rows.append(
            {
                "config": name,
                "fold": fold,
                "features": len(features),
                "mape_base": mape(valid[TARGET], valid["best_base"]),
                "mape_alpha070": mape(valid[TARGET], pred),
            }
        )
        valid_oof = valid[["new_id", TARGET, "best_base"]].copy()
        valid_oof["config"] = name
        valid_oof["fold"] = fold
        valid_oof["pred_log"] = pred_log
        valid_oof["pred_alpha070"] = pred
        oof_parts.append(valid_oof)
    return pd.DataFrame(rows), pd.concat(oof_parts, ignore_index=True)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows_"
    values = frame.astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, separator, *rows])


def prepare_target_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
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

    panel = add_feature_audit_columns(add_base(make_panel(raw), validation_factor))
    target_rows = panel[
        (panel["Год"] == 2024)
        & (panel["Месяц"] == 3)
        & (panel[TARGET] > 0)
        & (panel["best_base"] > 0)
    ].copy()
    target_rows["target_log_ratio"] = np.log(target_rows[TARGET] / target_rows["best_base"])
    return target_rows, validation_factor, final_factor


def make_final_prediction(
    raw: pd.DataFrame, train_rows: pd.DataFrame, final_factor: float, features: list[str], config_name: str
) -> Path:
    final_model = CatBoostRegressor(**MODEL_PARAMS)
    final_model.fit(fit_pool(train_rows, features, "target_log_ratio"))

    feb_2025 = raw[(raw["Год"] == 2025) & (raw["Месяц"] == 2)].copy()
    mar_2025 = feb_2025.copy()
    mar_2025["Год"] = 2025
    mar_2025["Месяц"] = 3
    mar_2025[TARGET] = np.nan
    pred_panel = add_feature_audit_columns(
        add_base(make_panel(pd.concat([raw, mar_2025], ignore_index=True)), final_factor)
    )
    pred_rows = pred_panel[(pred_panel["Год"] == 2025) & (pred_panel["Месяц"] == 3)].copy()
    pred_log = predict_pool(final_model, pred_rows, features)
    pred = apply_residual(pred_rows["best_base"].to_numpy(dtype=float), pred_log, ALPHA, None)

    submission = pred_rows[["new_id"]].copy()
    submission["rto"] = np.round(np.maximum(pred, 0), 2)
    submission = submission.sort_values("new_id").reset_index(drop=True)
    safe_name = config_name.replace("current_plus_", "").replace("all_new_plus_", "").replace("targeted_plus_", "")
    path = SUBMISSIONS_DIR / f"test_{safe_name}.csv"
    submission.to_csv(path, index=False)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    train_rows, validation_factor, final_factor = prepare_target_rows(raw)
    configs = config_features()

    fold_parts = []
    oof_parts = []
    for name, features in configs.items():
        folds, oof = evaluate_config(train_rows, features, name)
        fold_parts.append(folds)
        oof_parts.append(oof)

    folds = pd.concat(fold_parts, ignore_index=True)
    oof = pd.concat(oof_parts, ignore_index=True)
    summary = (
        folds.groupby("config", as_index=False)
        .agg(
            features=("features", "first"),
            mape_base=("mape_base", "mean"),
            mape_alpha070=("mape_alpha070", "mean"),
            mape_std=("mape_alpha070", "std"),
        )
        .sort_values("mape_alpha070")
        .reset_index(drop=True)
    )
    summary["gain_vs_base"] = summary["mape_base"] - summary["mape_alpha070"]
    current_mape = float(summary.loc[summary["config"] == "current_all", "mape_alpha070"].iloc[0])
    summary["gain_vs_current"] = current_mape - summary["mape_alpha070"]

    best_config = str(summary.iloc[0]["config"])
    best_features = configs[best_config]
    submission_path = make_final_prediction(raw, train_rows, final_factor, best_features, best_config)

    final_model = CatBoostRegressor(**MODEL_PARAMS)
    final_model.fit(fit_pool(train_rows, best_features, "target_log_ratio"))
    feature_importance = pd.DataFrame(
        {
            "feature": best_features,
            "importance": final_model.get_feature_importance(
                fit_pool(train_rows, best_features, "target_log_ratio"),
                type="FeatureImportance",
            ),
        }
    ).sort_values("importance", ascending=False)

    folds.to_csv(RESULTS_DIR / "ablation_folds.csv", index=False)
    summary.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)
    oof.to_csv(RESULTS_DIR / "ablation_oof.csv", index=False)
    feature_importance.to_csv(RESULTS_DIR / "best_feature_importance.csv", index=False)

    display = summary.copy()
    for col in ["mape_base", "mape_alpha070", "mape_std", "gain_vs_base", "gain_vs_current"]:
        display[col] = display[col].round(5)
    top_features = feature_importance.head(20).copy()
    top_features["importance"] = top_features["importance"].round(4)

    report = f"""# CatBoost Feature Audit

## Setup

- Target: March 2024 residual over the current heuristic baseline.
- Validation: five cross-sectional folds by `new_id % 5`.
- Residual correction: `heuristic * exp({ALPHA:.2f} * catboost_log_residual)`, without residual clipping.
- Current feature families: level lags, current dynamics, static numeric features, categorical features, and lagged behavior features.
- Added feature families: explicit trend, last-month anomaly, prior seasonal context, recent stability, heuristic-trust, Q1 transition, alternative March-base disagreement, log-transform, first log-difference, seasonal log-difference, small second-difference, and group-level scale/dynamics features. Every added value is built from data available before the target month.

## Ablation summary

{markdown_table(display)}

## Best config

Best local config: `{best_config}` with **{len(best_features)}** features.

Candidate generated from this feature set:

`{submission_path.relative_to(ROOT)}`

## Top feature importances for the best config

{markdown_table(top_features)}

## Notes

- `current_minus_*` rows test whether existing feature families carry useful signal.
- `current_plus_*` rows test whether a new time-series feature family adds signal beyond the current residual model.
- True two-year stability of an individual March/February coefficient cannot be validated on March 2024 without changing the validation target; March 2024 only has the prior March 2023 transition available.
"""
    (RESULTS_DIR / "feature_audit.md").write_text(report, encoding="utf-8")
    run_summary = {
        "validation_factor": validation_factor,
        "final_factor": final_factor,
        "alpha": ALPHA,
        "rows": len(train_rows),
        "best_config": best_config,
        "best_feature_count": len(best_features),
        "current_mape": current_mape,
        "best_mape": float(summary.iloc[0]["mape_alpha070"]),
        "candidate_submission": str(submission_path),
        "ablation_summary": summary.to_dict(orient="records"),
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
