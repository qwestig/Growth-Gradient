from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import RESULTS_DIR, TARGET, load_data, mape, safe_div


VALID_PERIODS = [(2024, 3), (2024, 11), (2024, 12), (2025, 1), (2025, 2)]


@dataclass(frozen=True)
class StoreSeasonSpec:
    name: str
    calendar_lambda: float
    weight: float
    log_clip: float
    vol_max: float | None
    avoid_structural: bool


def label(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def shift(year: int, month: int, months_back: int) -> tuple[int, int]:
    stamp = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=months_back)
    return int(stamp.year), int(stamp.month)


def adjusted_ratio(ratio: pd.Series, source_year: int, target_year: int, target_month: int, calendar_lambda: float) -> pd.Series:
    if target_month != 3:
        return ratio
    source_feb_days = calendar.monthrange(source_year, 2)[1]
    target_feb_days = calendar.monthrange(target_year, 2)[1]
    adjustment = (1.0 - calendar_lambda) + calendar_lambda * (source_feb_days / target_feb_days)
    return ratio * adjustment


def target_frame(rto: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    lag1_y, lag1_m = shift(year, month, 1)
    lag2_y, lag2_m = shift(year, month, 2)
    lag3_y, lag3_m = shift(year, month, 3)
    lag12_y, lag12_m = shift(year, month, 12)
    lag13_y, lag13_m = shift(year, month, 13)
    frame = pd.DataFrame(index=rto.index)
    frame["new_id"] = frame.index.astype(int)
    frame["lag1"] = rto[label(lag1_y, lag1_m)]
    frame["lag2"] = rto[label(lag2_y, lag2_m)]
    frame["lag3"] = rto[label(lag3_y, lag3_m)]
    frame["lag12"] = rto[label(lag12_y, lag12_m)]
    frame["lag13"] = rto[label(lag13_y, lag13_m)]
    frame[TARGET] = rto[label(year, month)]
    frame["rolling3_mean"] = frame[["lag1", "lag2", "lag3"]].mean(axis=1)
    frame["rolling2_pre_mean"] = frame[["lag2", "lag3"]].mean(axis=1)
    frame["momentum_raw"] = safe_div(frame["lag1"], frame["rolling3_mean"])
    frame["momentum_clip"] = frame["momentum_raw"].fillna(1.0).clip(0.95, 1.05)
    frame["feb_to_jan"] = safe_div(frame["lag1"], frame["lag2"])
    frame["jan_to_dec"] = safe_div(frame["lag2"], frame["lag3"])
    frame["lag1_to_lag12"] = safe_div(frame["lag1"], frame["lag12"])
    frame["volatility_rel3"] = frame[["lag1", "lag2", "lag3"]].std(axis=1) / frame["rolling3_mean"]
    frame["structural_decline_score"] = (
        frame["feb_to_jan"].fillna(1.0).lt(0.97)
        & frame["jan_to_dec"].fillna(1.0).lt(0.99)
        & frame["lag1_to_lag12"].fillna(1.0).lt(1.02)
    ).astype(float)
    return frame.reset_index(drop=True)


def historical_factor(rto: pd.DataFrame, target_year: int, target_month: int, calendar_lambda: float) -> float:
    ratios = []
    for source_year in sorted({int(c[:4]) for c in rto.columns if c.endswith(f"-{target_month:02d}")}):
        if source_year >= target_year:
            continue
        prev_y, prev_m = shift(source_year, target_month, 1)
        source = label(source_year, target_month)
        prev = label(prev_y, prev_m)
        if source not in rto.columns or prev not in rto.columns:
            continue
        ratio = safe_div(rto[source], rto[prev])
        ratios.append(adjusted_ratio(ratio, source_year, target_year, target_month, calendar_lambda))
    if not ratios:
        raise ValueError(f"No historical ratios for {target_year}-{target_month:02d}")
    return float(pd.concat(ratios).replace([np.inf, -np.inf], np.nan).dropna().median())


def seasonal_pred(frame: pd.DataFrame, factor: float, momentum_exp: float) -> np.ndarray:
    return frame["lag1"].to_numpy(float) * factor * frame["momentum_clip"].to_numpy(float) ** momentum_exp


def storeseason_pred(frame: pd.DataFrame, base_pred: np.ndarray, global_factor: float, year: int, month: int, spec: StoreSeasonSpec) -> np.ndarray:
    source_year = year - 1
    store_ratio = adjusted_ratio(safe_div(frame["lag12"], frame["lag13"]), source_year, year, month, spec.calendar_lambda)
    store_ratio = store_ratio.replace([np.inf, -np.inf], np.nan).fillna(global_factor).clip(0.85, 1.45)
    log_delta = np.clip(np.log(store_ratio.to_numpy(float) / global_factor), -spec.log_clip, spec.log_clip)
    volume = (frame["lag13"].to_numpy(float) / (frame["lag13"].to_numpy(float) + 100_000_000.0)).clip(0.0, 1.0)
    stable = (1.0 - frame["volatility_rel3"].fillna(0.0).to_numpy(float) / 0.12).clip(0.0, 1.0)
    rel = volume * stable
    if spec.vol_max is not None:
        rel *= (frame["volatility_rel3"].fillna(0.0).to_numpy(float) <= spec.vol_max).astype(float)
    if spec.avoid_structural:
        rel *= (frame["structural_decline_score"].to_numpy(float) <= 0).astype(float)
    return base_pred * np.exp(spec.weight * rel * log_delta)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    _raw, rto, _static = load_data()
    store_specs = [
        StoreSeasonSpec("store_v3_188_like", 0.60, 0.70, 0.03, 0.12, True),
        StoreSeasonSpec("store_v3_189_like", 0.60, 0.50, 0.05, 0.12, True),
        StoreSeasonSpec("store_v3_190_like", 0.60, 0.50, 0.05, None, True),
    ]
    rows = []
    for year, month in VALID_PERIODS:
        frame = target_frame(rto, year, month)
        y = frame[TARGET]
        factors = {
            "seasonal_lam000_mom025": historical_factor(rto, year, month, 0.00),
            "seasonal_lam050_mom000": historical_factor(rto, year, month, 0.50),
            "seasonal_lam060_mom000": historical_factor(rto, year, month, 0.60),
            "seasonal_lam060_mom025": historical_factor(rto, year, month, 0.60),
        }
        preds = {
            "seasonal_lam000_mom025": seasonal_pred(frame, factors["seasonal_lam000_mom025"], 0.25),
            "seasonal_lam050_mom000": seasonal_pred(frame, factors["seasonal_lam050_mom000"], 0.0),
            "seasonal_lam060_mom000": seasonal_pred(frame, factors["seasonal_lam060_mom000"], 0.0),
            "seasonal_lam060_mom025": seasonal_pred(frame, factors["seasonal_lam060_mom025"], 0.25),
            "delta_yoy": frame["lag12"].to_numpy(float) + (frame["lag1"].to_numpy(float) - frame["lag13"].to_numpy(float)),
            "trend_yoy": frame["lag12"].to_numpy(float) * safe_div(frame["lag1"], frame["lag13"]).fillna(1.0).to_numpy(float),
            "lag1": frame["lag1"].to_numpy(float),
        }
        base60 = preds["seasonal_lam060_mom000"]
        for spec in store_specs:
            preds[spec.name] = storeseason_pred(frame, base60, factors["seasonal_lam060_mom000"], year, month, spec)
        for name, pred in preds.items():
            rows.append(
                {
                    "period": f"{year}-{month:02d}",
                    "name": name,
                    "rows": len(frame),
                    "factor": factors.get(name, np.nan),
                    "mape": mape(y, pred),
                    "mean_pred": float(np.mean(pred)),
                    "mean_abs_change_vs_lam060_mom000_pct": float(np.mean(np.abs(pred / base60 - 1.0)) * 100),
                }
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS_DIR / "baseline_multiperiod_validation_detail.csv", index=False)
    summary = (
        detail.groupby("name", as_index=False)
        .agg(
            avg_mape=("mape", "mean"),
            worst_mape=("mape", "max"),
            avg_abs_change_vs_lam060=("mean_abs_change_vs_lam060_mom000_pct", "mean"),
        )
        .sort_values("avg_mape")
        .reset_index(drop=True)
    )
    march = detail[detail["period"] == "2024-03"][["name", "mape"]].rename(columns={"mape": "march2024_mape"})
    summary = summary.merge(march, on="name", how="left")
    summary.to_csv(RESULTS_DIR / "baseline_multiperiod_validation_summary.csv", index=False)
    pivot = detail.pivot(index="name", columns="period", values="mape").reset_index()
    pivot.to_csv(RESULTS_DIR / "baseline_multiperiod_validation_pivot.csv", index=False)
    print("Multi-period baseline validation summary:")
    print(summary.to_string(index=False))
    print("\nPivot:")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
