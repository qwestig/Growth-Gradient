from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import (
    RESULTS_DIR,
    TARGET,
    HeuristicSpec,
    STATIC_CAT_COLS,
    STATIC_NUM_COLS,
    build_target_frame,
    factors_for_spec,
    load_data,
    mape,
    predict_from_spec,
    safe_div,
)
from search_baseline_heuristics_v3 import store_gate, store_seasonal_pred


RU_HOLIDAYS = {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4)}


def month_exposure(year: int, month: int, weekend_weight: float, holiday_weight: float) -> float:
    exposure = 0.0
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        dt = date(year, month, day)
        if (month, day) in RU_HOLIDAYS and dt.weekday() < 5:
            exposure += holiday_weight
        elif dt.weekday() >= 5:
            exposure += weekend_weight
        else:
            exposure += 1.0
    return exposure


def exposure_factor(rto: pd.DataFrame, target_year: int, weekend_weight: float, holiday_weight: float) -> float:
    ratios = []
    target_exp = month_exposure(target_year, 3, weekend_weight, holiday_weight) / month_exposure(target_year, 2, weekend_weight, holiday_weight)
    for year in [2023, 2024]:
        if year >= target_year:
            continue
        ratio = safe_div(rto[f"{year}-03"], rto[f"{year}-02"])
        source_exp = month_exposure(year, 3, weekend_weight, holiday_weight) / month_exposure(year, 2, weekend_weight, holiday_weight)
        ratios.append(ratio / source_exp)
    return float(pd.concat(ratios).dropna().median() * target_exp)


def current_191(frame: pd.DataFrame, factor: float, target_non_leap: bool) -> np.ndarray:
    base = frame["lag1"].to_numpy(float) * factor
    gate = store_gate(frame, None, 0.12, avoid_structural=True)
    return store_seasonal_pred(
        frame,
        base,
        factor,
        target_non_leap=target_non_leap,
        calendar_lambda=0.575,
        weight=0.70,
        log_clip=0.03,
        rel_mode="stable_volume",
        volume_k=100_000_000.0,
        extra_gate=gate,
    )


def factor_for_lambda(rto: pd.DataFrame, static: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, calendar_lambda: float) -> tuple[float, float]:
    spec = HeuristicSpec(0, "base", calendar_lambda, momentum_exp=0.0)
    _vf, _tf, val_global, final_global = factors_for_spec(spec, val, test, rto, static)
    return val_global, final_global


def residual_diagnostics(val: pd.DataFrame, pred: np.ndarray) -> None:
    frame = val.copy()
    frame["pred"] = pred
    frame["log_residual"] = np.log(frame[TARGET].to_numpy(float) / pred)
    frame["ape"] = np.abs((frame[TARGET].to_numpy(float) - pred) / frame[TARGET].to_numpy(float)) * 100
    numeric_features = [
        "momentum_raw",
        "volatility_rel3",
        "lag1_to_lag12",
        "lag1_to_lag13",
        "store_mar_feb_ratio_prev",
        "temporary_dip_score",
        "structural_decline_score",
        "lag1",
    ]
    for col in [c for c in STATIC_NUM_COLS if c in frame.columns]:
        numeric_features.append(col)
    rows = []
    for col in numeric_features:
        values = frame[col].replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() < 500 or values.nunique(dropna=True) < 5:
            continue
        try:
            bins = pd.qcut(values, 5, duplicates="drop")
        except ValueError:
            continue
        tmp = frame.assign(_bin=bins)
        for bin_name, part in tmp.groupby("_bin", observed=True):
            if len(part) < 100:
                continue
            rows.append(
                {
                    "feature": col,
                    "bin": str(bin_name),
                    "n": len(part),
                    "mean_feature": float(part[col].mean()),
                    "mean_log_residual": float(part["log_residual"].mean()),
                    "median_log_residual": float(part["log_residual"].median()),
                    "mape": mape(part[TARGET], part["pred"]),
                    "mean_ape": float(part["ape"].mean()),
                }
            )
    for col in [c for c in STATIC_CAT_COLS if c in frame.columns]:
        stats = frame.groupby(col, observed=True).agg(
            n=(TARGET, "size"),
            mean_log_residual=("log_residual", "mean"),
            median_log_residual=("log_residual", "median"),
            mean_ape=("ape", "mean"),
        )
        stats = stats[stats["n"] >= 200].sort_values("mean_ape", ascending=False).head(40).reset_index()
        stats.insert(0, "feature", col)
        stats.to_csv(RESULTS_DIR / f"baseline_v4_residual_cat_{col}.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baseline_v4_residual_numeric_bins.csv", index=False)


def add_row(rows: list[dict], name: str, family: str, val_pred: np.ndarray, test_pred: np.ndarray, val: pd.DataFrame, ref_val: np.ndarray, ref_test: np.ndarray, params: dict) -> None:
    rows.append(
        {
            "name": name,
            "family": family,
            "proxy_mape_march2024": mape(val[TARGET], val_pred),
            "delta_vs_191_proxy": mape(val[TARGET], val_pred) - mape(val[TARGET], ref_val),
            "val_mean_abs_change_vs_191_pct": float(np.mean(np.abs(val_pred / ref_val - 1.0)) * 100),
            "test_mean_abs_change_vs_191_pct": float(np.mean(np.abs(test_pred / ref_test - 1.0)) * 100),
            "test_p95_abs_change_vs_191_pct": float(np.quantile(np.abs(test_pred / ref_test - 1.0), 0.95) * 100),
            "test_mean_rto": float(np.mean(test_pred)),
            **params,
        }
    )


def sign_asym_store_pred(frame: pd.DataFrame, base: np.ndarray, factor: float, target_non_leap: bool, calendar_lambda: float, w_pos: float, w_neg: float, clip_pos: float, clip_neg: float, vol_max: float | None) -> np.ndarray:
    ratio = safe_div(frame["lag12"], frame["lag13"])
    if target_non_leap:
        ratio *= (1.0 - calendar_lambda) + calendar_lambda * (29.0 / 28.0)
    else:
        ratio *= (1.0 - calendar_lambda) + calendar_lambda * (28.0 / 29.0)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(factor).clip(0.85, 1.45)
    delta = np.log(ratio.to_numpy(float) / factor)
    pos = np.clip(delta, 0.0, clip_pos)
    neg = np.clip(delta, -clip_neg, 0.0)
    volume = (frame["lag13"].to_numpy(float) / (frame["lag13"].to_numpy(float) + 100_000_000.0)).clip(0.0, 1.0)
    stable = (1.0 - frame["volatility_rel3"].fillna(0.0).to_numpy(float) / 0.12).clip(0.0, 1.0)
    rel = volume * stable
    if vol_max is not None:
        rel *= (frame["volatility_rel3"].fillna(0.0).to_numpy(float) <= vol_max).astype(float)
    rel *= (frame["structural_decline_score"].to_numpy(float) <= 0).astype(float)
    return base * np.exp(rel * (w_pos * pos + w_neg * neg))


def group_ratio_factor(frame: pd.DataFrame, base_factor: float, group_col: str, shrink_k: float, calendar_lambda: float, target_non_leap: bool) -> pd.Series:
    ratio = safe_div(frame["lag12"], frame["lag13"])
    if target_non_leap:
        ratio *= (1.0 - calendar_lambda) + calendar_lambda * (29.0 / 28.0)
    else:
        ratio *= (1.0 - calendar_lambda) + calendar_lambda * (28.0 / 29.0)
    tmp = pd.DataFrame({"ratio": ratio, group_col: frame[group_col]})
    stats = tmp.replace([np.inf, -np.inf], np.nan).dropna().groupby(group_col, observed=True)["ratio"].agg(["median", "count"])
    correction = np.log(stats["median"] / base_factor) * stats["count"] / (stats["count"] + shrink_k)
    return pd.Series(np.exp(frame[group_col].map(correction).fillna(0.0).to_numpy(float)), index=frame.index)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val = build_target_frame(rto, static, 2024, 3, include_target=True)
    test = build_target_frame(rto, static, 2025, 3, include_target=False)
    val_factor_0575, final_factor_0575 = factor_for_lambda(rto, static, val, test, 0.575)
    ref_val = current_191(val, val_factor_0575, target_non_leap=False)
    ref_test = current_191(test, final_factor_0575, target_non_leap=True)
    residual_diagnostics(val, ref_val)

    rows: list[dict] = []
    base_cache: dict[float, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for lam in [0.525, 0.55, 0.575, 0.60, 0.625]:
        vf, ff = factor_for_lambda(rto, static, val, test, lam)
        base_cache[lam] = (val["lag1"].to_numpy(float) * vf, test["lag1"].to_numpy(float) * ff, vf, ff)

    # 1) Sign-asymmetric store-seasonality: public-positive signal may be stronger in one direction.
    for lam in [0.55, 0.575, 0.60]:
        base_val, base_test, vf, ff = base_cache[lam]
        for w_pos in [0.4, 0.7, 1.0]:
            for w_neg in [0.2, 0.5, 0.7, 1.0]:
                for clip_pos, clip_neg in [(0.03, 0.03), (0.04, 0.02), (0.02, 0.04), (0.05, 0.03)]:
                    val_pred = sign_asym_store_pred(val, base_val, vf, False, lam, w_pos, w_neg, clip_pos, clip_neg, 0.12)
                    test_pred = sign_asym_store_pred(test, base_test, ff, True, lam, w_pos, w_neg, clip_pos, clip_neg, 0.12)
                    add_row(rows, f"signasym_lam{lam:.3f}_wp{w_pos:.1f}_wn{w_neg:.1f}_cp{clip_pos:.2f}_cn{clip_neg:.2f}", "sign_asymmetric_storeseason", val_pred, test_pred, val, ref_val, ref_test, {"calendar_lambda": lam, "w_pos": w_pos, "w_neg": w_neg, "clip_pos": clip_pos, "clip_neg": clip_neg})

    # 2) Exposure global anchor combined with the confirmed store-seasonality correction.
    for weekend_weight, holiday_weight in [(1.0, 1.0), (1.1, 0.8), (1.2, 0.6), (1.2, 1.0), (1.2, 1.2)]:
        vf = exposure_factor(rto, 2024, weekend_weight, holiday_weight)
        ff = exposure_factor(rto, 2025, weekend_weight, holiday_weight)
        base_val = val["lag1"].to_numpy(float) * vf
        base_test = test["lag1"].to_numpy(float) * ff
        for weight in [0.5, 0.7]:
            val_pred = store_seasonal_pred(val, base_val, vf, False, 0.575, weight, 0.03, "stable_volume", 100_000_000.0, extra_gate=store_gate(val, None, 0.12, True))
            test_pred = store_seasonal_pred(test, base_test, ff, True, 0.575, weight, 0.03, "stable_volume", 100_000_000.0, extra_gate=store_gate(test, None, 0.12, True))
            add_row(rows, f"exposure_ww{weekend_weight:.1f}_hw{holiday_weight:.1f}_store_w{weight:.1f}", "exposure_plus_storeseason", val_pred, test_pred, val, ref_val, ref_test, {"weekend_weight": weekend_weight, "holiday_weight": holiday_weight, "weight": weight})

    # 3) Group-level previous-year seasonal memory, heavily shrunk and blended with 191.
    for group_col in [c for c in ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный"] if c in val.columns]:
        for shrink_k in [500, 1000, 2000, 5000]:
            mult_val = group_ratio_factor(val, val_factor_0575, group_col, shrink_k, 0.575, False)
            mult_test = group_ratio_factor(test, final_factor_0575, group_col, shrink_k, 0.575, True)
            for blend in [0.10, 0.20, 0.30]:
                val_pred = ref_val * np.exp(blend * np.log(mult_val.to_numpy(float)))
                test_pred = ref_test * np.exp(blend * np.log(mult_test.to_numpy(float)))
                add_row(rows, f"group_{group_col}_k{shrink_k}_b{blend:.2f}", "group_seasonality_blend", val_pred, test_pred, val, ref_val, ref_test, {"group_col": group_col, "shrink_k": shrink_k, "blend": blend})

    # 4) Residual state adjustments only if they survive proxy. These diagnose remaining high-error segments.
    for weak_cut in [0.935, 0.945, 0.955]:
        weak = (val["momentum_raw"].fillna(1.0).to_numpy(float) < weak_cut).astype(float)
        weak_t = (test["momentum_raw"].fillna(1.0).to_numpy(float) < weak_cut).astype(float)
        for lift in [-0.015, -0.010, 0.010, 0.015]:
            val_pred = ref_val * np.exp(lift * weak)
            test_pred = ref_test * np.exp(lift * weak_t)
            add_row(rows, f"weak_mom_cut{weak_cut:.3f}_lift{lift:+.3f}", "state_residual_adjustment", val_pred, test_pred, val, ref_val, ref_test, {"weak_cut": weak_cut, "lift": lift})

    summary = pd.DataFrame(rows).sort_values("proxy_mape_march2024").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "baseline_heuristics_v4_search.csv", index=False)
    summary.head(60).to_csv(RESULTS_DIR / "baseline_heuristics_v4_top60.csv", index=False)

    print("Current 191 proxy MAPE:", mape(val[TARGET], ref_val))
    print("Top v4 heuristics:")
    print(summary[["name", "family", "proxy_mape_march2024", "delta_vs_191_proxy", "test_mean_abs_change_vs_191_pct", "test_p95_abs_change_vs_191_pct", "test_mean_rto"]].head(40).to_string(index=False))
    print("\nFamily best:")
    print(summary.groupby("family", as_index=False).first()[["family", "name", "proxy_mape_march2024", "delta_vs_191_proxy", "test_mean_abs_change_vs_191_pct"]].sort_values("proxy_mape_march2024").to_string(index=False))


if __name__ == "__main__":
    main()
