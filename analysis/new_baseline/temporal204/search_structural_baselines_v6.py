from __future__ import annotations

import calendar
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from heuristic_baseline_direction_lab import (
    RESULTS_DIR,
    STATIC_CAT_COLS,
    TARGET,
    HeuristicSpec,
    build_target_frame,
    factors_for_spec,
    load_data,
    mape,
    predict_from_spec,
    safe_div,
)
from search_baseline_heuristics_v4 import sign_asym_store_pred
from validate_baseline_heuristics_multiperiod import VALID_PERIODS, historical_factor, label, shift, target_frame


CURRENT_LAMBDA = 0.55


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    params: dict


def current_198_march(frame: pd.DataFrame, factor: float, target_non_leap: bool) -> np.ndarray:
    base = frame["lag1"].to_numpy(float) * factor
    return sign_asym_store_pred(
        frame,
        base,
        factor,
        target_non_leap=target_non_leap,
        calendar_lambda=CURRENT_LAMBDA,
        w_pos=0.2,
        w_neg=1.4,
        clip_pos=0.02,
        clip_neg=0.03,
        vol_max=0.12,
    )


def current_like_generic(frame: pd.DataFrame, factor: float, year: int, month: int) -> np.ndarray:
    base = frame["lag1"].to_numpy(float) * factor
    target_non_leap = month == 3 and calendar.monthrange(year, 2)[1] == 28
    return sign_asym_store_pred(
        frame,
        base,
        factor,
        target_non_leap=target_non_leap,
        calendar_lambda=CURRENT_LAMBDA,
        w_pos=0.2,
        w_neg=1.4,
        clip_pos=0.02,
        clip_neg=0.03,
        vol_max=0.12,
    )


def days_in_months(year: int, months: list[int]) -> int:
    return sum(calendar.monthrange(year, m)[1] for m in months)


def two_month_temporal_anchor(rto: pd.DataFrame, frame: pd.DataFrame, year: int, month: int, daily_adjust: bool, clip_lo: float, clip_hi: float) -> np.ndarray:
    prev1_y, prev1_m = shift(year, month, 1)
    prev2_y, prev2_m = shift(year, month, 2)
    py, pm = shift(year, month, 12)
    pprev1_y, pprev1_m = shift(year, month, 13)
    pprev2_y, pprev2_m = shift(year, month, 14)

    current_two = rto[label(prev1_y, prev1_m)] + rto[label(prev2_y, prev2_m)]
    previous_two = rto[label(pprev1_y, pprev1_m)] + rto[label(pprev2_y, pprev2_m)]
    growth = safe_div(current_two, previous_two)
    if daily_adjust:
        current_days = days_in_months(prev1_y, [prev1_m]) + days_in_months(prev2_y, [prev2_m])
        previous_days = days_in_months(pprev1_y, [pprev1_m]) + days_in_months(pprev2_y, [pprev2_m])
        growth *= previous_days / current_days
    growth = growth.replace([np.inf, -np.inf], np.nan).fillna(growth.median()).clip(clip_lo, clip_hi)
    anchor = rto[label(py, pm)] * growth
    return pd.Series(anchor, index=rto.index).reindex(frame["new_id"]).to_numpy(float)


def blend_anchor(base: np.ndarray, anchor: np.ndarray, alpha: float, log_clip: float) -> np.ndarray:
    ratio = np.log(np.maximum(anchor, 1.0) / np.maximum(base, 1.0))
    return base * np.exp(alpha * np.clip(ratio, -log_clip, log_clip))


def group_total_correction(
    frame: pd.DataFrame,
    pred: np.ndarray,
    group_col: str,
    target_year: int,
    target_month: int,
    source_years: list[int],
    rto: pd.DataFrame,
    alpha: float,
    log_clip: float,
    shrink_k: float,
) -> np.ndarray:
    tmp = frame[["new_id", group_col, "lag1"]].copy()
    tmp["pred"] = pred
    group_current = tmp.groupby(group_col, observed=True).agg(current_pred=("pred", "sum"), current_lag1=("lag1", "sum"), n=("new_id", "size"))

    ratio_rows = []
    static_group = frame.set_index("new_id")[group_col]
    for source_year in source_years:
        source = label(source_year, target_month)
        prev_y, prev_m = shift(source_year, target_month, 1)
        prev = label(prev_y, prev_m)
        if source not in rto.columns or prev not in rto.columns:
            continue
        source_frame = pd.DataFrame(
            {
                "new_id": rto.index.astype(int),
                "target": rto[source].to_numpy(float),
                "prev": rto[prev].to_numpy(float),
            }
        ).join(static_group, on="new_id", how="inner")
        group_sum = source_frame.groupby(group_col, observed=True).agg(target=("target", "sum"), prev=("prev", "sum"))
        ratio = safe_div(group_sum["target"], group_sum["prev"])
        if target_month == 3:
            source_feb_days = calendar.monthrange(source_year, 2)[1]
            target_feb_days = calendar.monthrange(target_year, 2)[1]
            ratio *= (1.0 - CURRENT_LAMBDA) + CURRENT_LAMBDA * (source_feb_days / target_feb_days)
        ratio_rows.append(ratio.rename(source_year))
    if not ratio_rows:
        return pred.copy()
    group_factor = pd.concat(ratio_rows, axis=1).median(axis=1)
    group_target = group_current["current_lag1"] * group_factor.reindex(group_current.index)
    corr = np.log(group_target / group_current["current_pred"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shrink = group_current["n"] / (group_current["n"] + shrink_k)
    corr = np.clip(corr * shrink, -log_clip, log_clip)
    mapped = frame[group_col].map(corr).fillna(0.0).to_numpy(float)
    return pred * np.exp(alpha * mapped)


def bootstrap_delta(y: np.ndarray, ref: np.ndarray, pred: np.ndarray) -> dict:
    rng = np.random.default_rng(2026)
    ref_ape = np.abs((y - ref) / y) * 100
    ape = np.abs((y - pred) / y) * 100
    delta = ape - ref_ape
    idx = rng.integers(0, len(y), size=(1000, len(y)))
    boot = delta[idx].mean(axis=1)
    return {
        "boot_p05": float(np.quantile(boot, 0.05)),
        "boot_p50": float(np.quantile(boot, 0.50)),
        "boot_p95": float(np.quantile(boot, 0.95)),
        "prob_improve": float((boot < 0).mean()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val = build_target_frame(rto, static, 2024, 3, include_target=True)
    test = build_target_frame(rto, static, 2025, 3, include_target=False)
    spec = HeuristicSpec(0, "current", CURRENT_LAMBDA, momentum_exp=0.0)
    _vf_series, _tf_series, val_factor, final_factor = factors_for_spec(spec, val, test, rto, static)
    ref_val = current_198_march(val, val_factor, target_non_leap=False)
    ref_test = current_198_march(test, final_factor, target_non_leap=True)

    rows = []
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Temporal aggregation: target month from same month last year scaled by previous two months growth.
    for daily_adjust in [False, True]:
        for clip_lo, clip_hi in [(0.85, 1.35), (0.90, 1.25), (0.95, 1.20)]:
            anchor_val = two_month_temporal_anchor(rto, val, 2024, 3, daily_adjust, clip_lo, clip_hi)
            anchor_test = two_month_temporal_anchor(rto, test, 2025, 3, daily_adjust, clip_lo, clip_hi)
            for alpha in [0.03, 0.05, 0.08, 0.12, 0.18]:
                for log_clip in [0.02, 0.04, 0.06, 0.10]:
                    pred_val = blend_anchor(ref_val, anchor_val, alpha, log_clip)
                    pred_test = blend_anchor(ref_test, anchor_test, alpha, log_clip)
                    name = f"temporal2_daily{int(daily_adjust)}_clip{clip_lo:.2f}_{clip_hi:.2f}_a{alpha:.2f}_lc{log_clip:.2f}"
                    prediction_cache[name] = (pred_val, pred_test)
                    rows.append(
                        {
                            "name": name,
                            "family": "temporal_two_month_growth",
                            "proxy_mape_march2024": mape(val[TARGET], pred_val),
                            "delta_vs_198_proxy": mape(val[TARGET], pred_val) - mape(val[TARGET], ref_val),
                            "test_mean_abs_change_vs_198_pct": float(np.mean(np.abs(pred_test / ref_test - 1.0)) * 100),
                            "test_p95_abs_change_vs_198_pct": float(np.quantile(np.abs(pred_test / ref_test - 1.0), 0.95) * 100),
                            "test_mean_rto": float(np.mean(pred_test)),
                            "daily_adjust": daily_adjust,
                            "clip_lo": clip_lo,
                            "clip_hi": clip_hi,
                            "alpha": alpha,
                            "log_clip": log_clip,
                        }
                    )

    # Hierarchical/group total correction as a small multiplicative correction to current best baseline.
    for group_col in [c for c in STATIC_CAT_COLS if c in val.columns]:
        for alpha in [0.10, 0.20, 0.35, 0.50]:
            for log_clip in [0.01, 0.02, 0.035, 0.05]:
                for shrink_k in [200, 500, 1000, 2000]:
                    pred_val = group_total_correction(val, ref_val, group_col, 2024, 3, [2023], rto, alpha, log_clip, shrink_k)
                    pred_test = group_total_correction(test, ref_test, group_col, 2025, 3, [2023, 2024], rto, alpha, log_clip, shrink_k)
                    name = f"groupcorr_{group_col}_a{alpha:.2f}_lc{log_clip:.3f}_k{shrink_k}"
                    prediction_cache[name] = (pred_val, pred_test)
                    rows.append(
                        {
                            "name": name,
                            "family": "hierarchical_group_total_correction",
                            "proxy_mape_march2024": mape(val[TARGET], pred_val),
                            "delta_vs_198_proxy": mape(val[TARGET], pred_val) - mape(val[TARGET], ref_val),
                            "test_mean_abs_change_vs_198_pct": float(np.mean(np.abs(pred_test / ref_test - 1.0)) * 100),
                            "test_p95_abs_change_vs_198_pct": float(np.quantile(np.abs(pred_test / ref_test - 1.0), 0.95) * 100),
                            "test_mean_rto": float(np.mean(pred_test)),
                            "group_col": group_col,
                            "alpha": alpha,
                            "log_clip": log_clip,
                            "shrink_k": shrink_k,
                        }
                    )

    summary = pd.DataFrame(rows).sort_values("proxy_mape_march2024").reset_index(drop=True)
    # Bootstrap top candidates and multi-period sanity for the best few per family.
    top_names = []
    for family, part in summary.groupby("family", observed=True):
        top_names.extend(part.head(12)["name"].tolist())
    top_names = list(dict.fromkeys(top_names))

    boot_rows = []
    y = val[TARGET].to_numpy(float)
    for name in top_names:
        pred_val, _pred_test = prediction_cache[name]
        boot_rows.append({"name": name, **bootstrap_delta(y, ref_val, pred_val)})
    boot = pd.DataFrame(boot_rows)

    mp_rows = []
    for name in top_names:
        row = summary.loc[summary["name"] == name].iloc[0]
        vals = []
        for year, month in VALID_PERIODS:
            fr = target_frame(rto, year, month).join(static, on="new_id", how="left")
            factor = historical_factor(rto, year, month, CURRENT_LAMBDA)
            ref = current_like_generic(fr, factor, year, month)
            if row["family"] == "temporal_two_month_growth":
                anchor = two_month_temporal_anchor(rto, fr, year, month, bool(row["daily_adjust"]), float(row["clip_lo"]), float(row["clip_hi"]))
                pred = blend_anchor(ref, anchor, float(row["alpha"]), float(row["log_clip"]))
            else:
                # Multi-period group correction uses all available prior years for that target month.
                source_years = [sy for sy in [2023, 2024] if sy < year]
                pred = group_total_correction(fr, ref, str(row["group_col"]), year, month, source_years, rto, float(row["alpha"]), float(row["log_clip"]), float(row["shrink_k"]))
            vals.append(mape(fr[TARGET], pred))
        mp_rows.append({"name": name, "multiperiod_avg_mape": float(np.mean(vals)), "multiperiod_worst_mape": float(np.max(vals)), "multiperiod_march2024_mape": vals[0]})
    mp = pd.DataFrame(mp_rows)

    final = summary.merge(boot, on="name", how="left").merge(mp, on="name", how="left")
    final.to_csv(RESULTS_DIR / "structural_baselines_v6_search.csv", index=False)
    final.head(80).to_csv(RESULTS_DIR / "structural_baselines_v6_top80.csv", index=False)
    print("Current 198 proxy MAPE:", mape(val[TARGET], ref_val))
    print("Family best:")
    print(final.groupby("family", as_index=False).first()[["family", "name", "proxy_mape_march2024", "delta_vs_198_proxy", "test_mean_abs_change_vs_198_pct", "multiperiod_avg_mape", "boot_p95", "prob_improve"]].sort_values("proxy_mape_march2024").to_string(index=False))
    print("\nTop candidates:")
    print(final[["name", "family", "proxy_mape_march2024", "delta_vs_198_proxy", "test_mean_abs_change_vs_198_pct", "test_p95_abs_change_vs_198_pct", "multiperiod_avg_mape", "boot_p95", "prob_improve"]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
