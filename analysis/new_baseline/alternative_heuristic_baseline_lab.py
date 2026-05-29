from __future__ import annotations

import calendar
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_baseline_alt_heuristics"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"

MAJOR_REGIONS = [
    "Москва г",
    "Московская обл",
    "Краснодарский край",
    "Татарстан Респ",
    "Ростовская обл",
]


@dataclass(frozen=True)
class AltForecast:
    name: str
    family: str
    val_pred: np.ndarray
    test_pred: np.ndarray
    notes: str


def read_remote_csv(path: str) -> pd.DataFrame:
    local = ROOT / path
    if local.exists():
        return pd.read_csv(local)
    result = subprocess.run(
        ["git", "show", f"{REMOTE_BRANCH}:{path}"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return pd.read_csv(StringIO(result.stdout))


def period_id(year: int, month: int) -> int:
    return year * 12 + month


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    return float(np.mean(np.abs((p[mask] - y[mask]) / y[mask])) * 100.0)


def month_days(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def weekend_days(year: int, month: int) -> int:
    _, ndays = calendar.monthrange(year, month)
    return sum(1 for day in range(1, ndays + 1) if calendar.weekday(year, month, day) >= 5)


def intensity(year: int, month: int, weekend_weight: float = 0.0) -> float:
    return float(month_days(year, month) + weekend_weight * weekend_days(year, month))


def period_values(rto: pd.DataFrame, year: int, month: int) -> np.ndarray:
    return rto[period_id(year, month)].to_numpy(dtype=float)


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.asarray(num, dtype=float) / np.clip(np.asarray(den, dtype=float), 1.0, None)


def positive(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1.0, None)


def safe_blend(anchor: np.ndarray, alt: np.ndarray, alpha: float, clip: float) -> np.ndarray:
    anchor = positive(anchor)
    alt = positive(alt)
    correction = alpha * np.log(alt / anchor)
    return anchor * np.exp(np.clip(correction, -clip, clip))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    rto = raw.pivot_table(index="new_id", columns="period", values=TARGET, aggfunc="mean").sort_index()

    static_cols = [
        "new_id",
        "Регион",
        "Населенный пункт",
        "Торговая площадь, категориальный",
        "Дата открытия, категориальный",
    ]
    static = (
        raw.sort_values(["new_id", "Год", "Месяц"])
        .groupby("new_id", as_index=False)[[c for c in static_cols if c in raw.columns]]
        .first()
        .set_index("new_id")
    )
    for col in static.columns:
        static[col] = static[col].astype("string").fillna("__NA__")

    meta_oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    meta_test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    return rto, static, meta_oof, meta_test


def target_frame(
    rto: pd.DataFrame,
    static: pd.DataFrame,
    meta: pd.DataFrame,
    target_year: int,
    is_train: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame({"new_id": rto.index.to_numpy(dtype=int)})
    frame["jan"] = period_values(rto, target_year, 1)
    frame["feb"] = period_values(rto, target_year, 2)
    frame["dec_prev"] = period_values(rto, target_year - 1, 12)
    frame["jan_prev_year"] = period_values(rto, target_year - 1, 1)
    frame["feb_prev_year"] = period_values(rto, target_year - 1, 2)
    frame["mar_prev_year"] = period_values(rto, target_year - 1, 3)
    frame["q1_prev_year"] = (
        period_values(rto, target_year - 1, 1)
        + period_values(rto, target_year - 1, 2)
        + period_values(rto, target_year - 1, 3)
    )
    frame["last3_mean"] = frame[["dec_prev", "jan", "feb"]].mean(axis=1)
    frame["janfeb_mean"] = frame[["jan", "feb"]].mean(axis=1)
    frame["volatility_rel3"] = frame[["dec_prev", "jan", "feb"]].std(axis=1) / np.clip(frame["last3_mean"], 1.0, None)
    frame["prev_mar_feb_ratio"] = frame["mar_prev_year"] / np.clip(frame["feb_prev_year"], 1.0, None)
    frame["prev_ratio_absdev"] = np.abs(np.log(np.clip(frame["prev_mar_feb_ratio"], 1e-6, None)))
    frame = frame.join(static, on="new_id", how="left")
    if is_train:
        frame = frame.merge(meta[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
        frame["anchor"] = frame["pred265_oof"]
        frame["target"] = frame[META_TARGET]
    else:
        frame = frame.merge(meta[["new_id", "pred265_submission"]], on="new_id", how="left")
        frame["anchor"] = frame["pred265_submission"]
    return frame.replace([np.inf, -np.inf], np.nan)


def add_alt_forecasts(val: pd.DataFrame, test: pd.DataFrame) -> list[AltForecast]:
    out: list[AltForecast] = []
    val = val.copy()
    test = test.copy()
    val["region_area"] = val["Регион"].astype(str) + "|" + val["Торговая площадь, категориальный"].astype(str)
    test["region_area"] = test["Регион"].astype(str) + "|" + test["Торговая площадь, категориальный"].astype(str)

    # 1. Year-over-year March heuristic: March follows observed Jan/Feb or Feb YoY growth.
    for growth_name, val_growth, test_growth, notes in [
        (
            "feb_yoy_raw",
            ratio(val["feb"], val["feb_prev_year"]),
            ratio(test["feb"], test["feb_prev_year"]),
            "Mar(prev year) multiplied by raw February YoY growth.",
        ),
        (
            "feb_yoy_day_adjusted",
            ratio(val["feb"], val["feb_prev_year"]) * (28.0 / 29.0),
            ratio(test["feb"], test["feb_prev_year"]) * (29.0 / 28.0),
            "Same, but neutralizes leap-year February day count.",
        ),
        (
            "janfeb_yoy_raw",
            ratio(val["jan"] + val["feb"], val["jan_prev_year"] + val["feb_prev_year"]),
            ratio(test["jan"] + test["feb"], test["jan_prev_year"] + test["feb_prev_year"]),
            "Mar(prev year) multiplied by raw Jan+Feb YoY growth.",
        ),
        (
            "janfeb_yoy_day_adjusted",
            ratio(val["jan"] + val["feb"], val["jan_prev_year"] + val["feb_prev_year"]) * (59.0 / 60.0),
            ratio(test["jan"] + test["feb"], test["jan_prev_year"] + test["feb_prev_year"]) * (60.0 / 59.0),
            "Same, but adjusts Jan+Feb for leap-year day count.",
        ),
    ]:
        out.append(
            AltForecast(
                name=f"yoy_march_{growth_name}",
                family="prev_march_yoy_growth",
                val_pred=positive(val["mar_prev_year"].to_numpy(float) * val_growth),
                test_pred=positive(test["mar_prev_year"].to_numpy(float) * test_growth),
                notes=notes,
            )
        )

    # 2. Q1 total heuristic: forecast full Q1 and subtract already known Jan+Feb.
    for growth_name, val_growth, test_growth, notes in [
        (
            "feb_yoy_raw",
            ratio(val["feb"], val["feb_prev_year"]),
            ratio(test["feb"], test["feb_prev_year"]),
            "Q1(prev year) follows raw February YoY growth; March is residual after Jan+Feb.",
        ),
        (
            "feb_yoy_day_adjusted",
            ratio(val["feb"], val["feb_prev_year"]) * (28.0 / 29.0),
            ratio(test["feb"], test["feb_prev_year"]) * (29.0 / 28.0),
            "Same, with leap-year February adjustment.",
        ),
        (
            "janfeb_yoy_raw",
            ratio(val["jan"] + val["feb"], val["jan_prev_year"] + val["feb_prev_year"]),
            ratio(test["jan"] + test["feb"], test["jan_prev_year"] + test["feb_prev_year"]),
            "Q1(prev year) follows raw Jan+Feb YoY growth; March is residual after Jan+Feb.",
        ),
        (
            "janfeb_yoy_day_adjusted",
            ratio(val["jan"] + val["feb"], val["jan_prev_year"] + val["feb_prev_year"]) * (59.0 / 60.0),
            ratio(test["jan"] + test["feb"], test["jan_prev_year"] + test["feb_prev_year"]) * (60.0 / 59.0),
            "Same, with Jan+Feb leap-year day-count adjustment.",
        ),
    ]:
        val_pred = val["q1_prev_year"].to_numpy(float) * val_growth - val["jan"].to_numpy(float) - val["feb"].to_numpy(float)
        test_pred = test["q1_prev_year"].to_numpy(float) * test_growth - test["jan"].to_numpy(float) - test["feb"].to_numpy(float)
        out.append(
            AltForecast(
                name=f"q1_residual_{growth_name}",
                family="q1_total_residual",
                val_pred=positive(val_pred),
                test_pred=positive(test_pred),
                notes=notes,
            )
        )

    # 3. Normal-level March seasonality: March factor is measured against Jan/Feb normal level, not only February.
    for weight_feb in [0.50, 0.65, 0.80]:
        val_norm_cur = weight_feb * val["feb"].to_numpy(float) + (1.0 - weight_feb) * val["jan"].to_numpy(float)
        val_norm_prev = weight_feb * val["feb_prev_year"].to_numpy(float) + (1.0 - weight_feb) * val["jan_prev_year"].to_numpy(float)
        test_norm_cur = weight_feb * test["feb"].to_numpy(float) + (1.0 - weight_feb) * test["jan"].to_numpy(float)
        test_norm_prev = weight_feb * test["feb_prev_year"].to_numpy(float) + (1.0 - weight_feb) * test["jan_prev_year"].to_numpy(float)

        raw_val_ratio = ratio(val["mar_prev_year"], val_norm_prev)
        raw_test_ratio = ratio(test["mar_prev_year"], test_norm_prev)
        out.append(
            AltForecast(
                name=f"normal_level_janfeb_w{int(weight_feb * 100):02d}",
                family="normal_level_seasonality",
                val_pred=positive(val_norm_cur * raw_val_ratio),
                test_pred=positive(test_norm_cur * raw_test_ratio),
                notes=f"March factor is Mar(prev year) / weighted Jan-Feb normal level, Feb weight={weight_feb:.2f}.",
            )
        )

    # 4. Explicit calendar intensity heuristic: convert source and current base months to pseudo-daily intensity.
    for weekend_weight in [0.0, 0.25, 0.50]:
        for base_name, val_base_cur, val_base_prev, test_base_cur, test_base_prev, val_base_int_cur, val_base_int_prev, test_base_int_cur, test_base_int_prev in [
            (
                "feb",
                val["feb"].to_numpy(float),
                val["feb_prev_year"].to_numpy(float),
                test["feb"].to_numpy(float),
                test["feb_prev_year"].to_numpy(float),
                intensity(2024, 2, weekend_weight),
                intensity(2023, 2, weekend_weight),
                intensity(2025, 2, weekend_weight),
                intensity(2024, 2, weekend_weight),
            ),
            (
                "janfeb",
                val["jan"].to_numpy(float) + val["feb"].to_numpy(float),
                val["jan_prev_year"].to_numpy(float) + val["feb_prev_year"].to_numpy(float),
                test["jan"].to_numpy(float) + test["feb"].to_numpy(float),
                test["jan_prev_year"].to_numpy(float) + test["feb_prev_year"].to_numpy(float),
                intensity(2024, 1, weekend_weight) + intensity(2024, 2, weekend_weight),
                intensity(2023, 1, weekend_weight) + intensity(2023, 2, weekend_weight),
                intensity(2025, 1, weekend_weight) + intensity(2025, 2, weekend_weight),
                intensity(2024, 1, weekend_weight) + intensity(2024, 2, weekend_weight),
            ),
        ]:
            val_factor = (val["mar_prev_year"].to_numpy(float) / intensity(2023, 3, weekend_weight)) / np.clip(
                val_base_prev / val_base_int_prev,
                1.0,
                None,
            )
            test_factor = (test["mar_prev_year"].to_numpy(float) / intensity(2024, 3, weekend_weight)) / np.clip(
                test_base_prev / test_base_int_prev,
                1.0,
                None,
            )
            val_pred = (val_base_cur / val_base_int_cur) * val_factor * intensity(2024, 3, weekend_weight)
            test_pred = (test_base_cur / test_base_int_cur) * test_factor * intensity(2025, 3, weekend_weight)
            out.append(
                AltForecast(
                    name=f"calendar_intensity_{base_name}_weekend{int(weekend_weight * 100):02d}",
                    family="calendar_intensity",
                    val_pred=positive(val_pred),
                    test_pred=positive(test_pred),
                    notes=f"Pseudo-daily intensity with weekend weight={weekend_weight:.2f}, base={base_name}.",
                )
            )

    # 5. Top-down share heuristics: forecast the aggregate March movement, then allocate by current shares.
    def topdown_global(
        frame: pd.DataFrame,
        growth_base: str,
        share_base: str,
        day_adjustment: float,
    ) -> np.ndarray:
        if growth_base == "feb":
            cur_growth_base = frame["feb"].to_numpy(float)
            prev_growth_base = frame["feb_prev_year"].to_numpy(float)
        elif growth_base == "janfeb":
            cur_growth_base = (frame["jan"] + frame["feb"]).to_numpy(float)
            prev_growth_base = (frame["jan_prev_year"] + frame["feb_prev_year"]).to_numpy(float)
        else:
            raise ValueError(growth_base)

        if share_base == "feb":
            share_values = frame["feb"].to_numpy(float)
        elif share_base == "janfeb":
            share_values = (frame["jan"] + frame["feb"]).to_numpy(float)
        elif share_base == "blend_feb_prev_mar":
            current_share = frame["feb"].to_numpy(float) / max(float(frame["feb"].sum()), 1.0)
            prev_share = frame["mar_prev_year"].to_numpy(float) / max(float(frame["mar_prev_year"].sum()), 1.0)
            total = frame["mar_prev_year"].sum() * ratio(cur_growth_base.sum(), prev_growth_base.sum()) * day_adjustment
            return positive(total * (0.75 * current_share + 0.25 * prev_share))
        else:
            raise ValueError(share_base)

        growth = ratio(np.array([cur_growth_base.sum()]), np.array([prev_growth_base.sum()]))[0] * day_adjustment
        total = float(frame["mar_prev_year"].sum()) * growth
        share = share_values / max(float(share_values.sum()), 1.0)
        return positive(total * share)

    def topdown_group(
        frame: pd.DataFrame,
        group_col: str,
        growth_base: str,
        share_base: str,
        day_adjustment: float,
    ) -> np.ndarray:
        pred = np.zeros(len(frame), dtype=float)
        if growth_base == "feb":
            cur_growth = frame["feb"].to_numpy(float)
            prev_growth = frame["feb_prev_year"].to_numpy(float)
        elif growth_base == "janfeb":
            cur_growth = (frame["jan"] + frame["feb"]).to_numpy(float)
            prev_growth = (frame["jan_prev_year"] + frame["feb_prev_year"]).to_numpy(float)
        else:
            raise ValueError(growth_base)

        if share_base == "feb":
            share_values = frame["feb"].to_numpy(float)
        elif share_base == "janfeb":
            share_values = (frame["jan"] + frame["feb"]).to_numpy(float)
        else:
            raise ValueError(share_base)

        groups = frame[group_col].astype(str).to_numpy()
        for group in pd.unique(groups):
            mask = groups == group
            cur_sum = float(cur_growth[mask].sum())
            prev_sum = float(prev_growth[mask].sum())
            if mask.sum() < 20 or cur_sum <= 0 or prev_sum <= 0:
                continue
            total = float(frame.loc[mask, "mar_prev_year"].sum()) * (cur_sum / prev_sum) * day_adjustment
            denom = max(float(share_values[mask].sum()), 1.0)
            pred[mask] = total * (share_values[mask] / denom)
        fallback = topdown_global(frame, growth_base=growth_base, share_base=share_base, day_adjustment=day_adjustment)
        pred[pred <= 0] = fallback[pred <= 0]
        return positive(pred)

    topdown_specs = [
        ("feb", "feb", 28.0 / 29.0, 29.0 / 28.0),
        ("feb", "blend_feb_prev_mar", 28.0 / 29.0, 29.0 / 28.0),
        ("janfeb", "janfeb", 59.0 / 60.0, 60.0 / 59.0),
        ("janfeb", "blend_feb_prev_mar", 59.0 / 60.0, 60.0 / 59.0),
    ]
    for growth_base, share_base, val_adj, test_adj in topdown_specs:
        out.append(
            AltForecast(
                name=f"topdown_global_growth_{growth_base}_share_{share_base}",
                family="topdown_share",
                val_pred=topdown_global(val, growth_base, share_base, val_adj),
                test_pred=topdown_global(test, growth_base, share_base, test_adj),
                notes=f"Global March total from {growth_base} YoY, allocated by {share_base} share.",
            )
        )
        for group_col in ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный", "region_area"]:
            if share_base == "blend_feb_prev_mar":
                continue
            out.append(
                AltForecast(
                    name=f"topdown_{group_col}_growth_{growth_base}_share_{share_base}",
                    family="topdown_share",
                    val_pred=topdown_group(val, group_col, growth_base, share_base, val_adj),
                    test_pred=topdown_group(test, group_col, growth_base, share_base, test_adj),
                    notes=f"{group_col} March totals from {growth_base} YoY, allocated by {share_base} share inside group.",
                )
            )

    # 6. Cross-sectional rank transfer: current stores receive prior-year March values by current rank.
    def quantile_transfer(frame: pd.DataFrame, base: str, day_adjustment: float) -> np.ndarray:
        if base == "feb":
            cur_base = frame["feb"].to_numpy(float)
            prev_base = frame["feb_prev_year"].to_numpy(float)
            growth = ratio(np.array([cur_base.sum()]), np.array([prev_base.sum()]))[0] * day_adjustment
        elif base == "janfeb":
            cur_base = (frame["jan"] + frame["feb"]).to_numpy(float)
            prev_base = (frame["jan_prev_year"] + frame["feb_prev_year"]).to_numpy(float)
            growth = ratio(np.array([cur_base.sum()]), np.array([prev_base.sum()]))[0] * day_adjustment
        else:
            raise ValueError(base)
        source = frame["mar_prev_year"].to_numpy(float) * growth
        cur_order = np.argsort(cur_base, kind="mergesort")
        prev_order = np.argsort(prev_base, kind="mergesort")
        pred = np.empty(len(frame), dtype=float)
        pred[cur_order] = source[prev_order]
        return positive(pred)

    for base, val_adj, test_adj in [("feb", 28.0 / 29.0, 29.0 / 28.0), ("janfeb", 59.0 / 60.0, 60.0 / 59.0)]:
        out.append(
            AltForecast(
                name=f"rank_transfer_{base}",
                family="rank_quantile_transfer",
                val_pred=quantile_transfer(val, base, val_adj),
                test_pred=quantile_transfer(test, base, test_adj),
                notes=f"Prior-year March distribution transferred by current {base} rank, scaled by aggregate growth.",
            )
        )

    return out


def stress_masks(val: pd.DataFrame, pred: np.ndarray) -> dict[str, np.ndarray]:
    shift = np.abs(np.log(positive(pred) / positive(val["anchor"].to_numpy(float))))
    return {
        "high_volume": val["feb"].rank(pct=True).to_numpy(float) >= 0.80,
        "low_volume": val["feb"].rank(pct=True).to_numpy(float) <= 0.20,
        "high_volatility": val["volatility_rel3"].rank(pct=True).to_numpy(float) >= 0.80,
        "unstable_mar_feb": val["prev_ratio_absdev"].rank(pct=True).to_numpy(float) >= 0.80,
        "high_shift_vs_pred265": pd.Series(shift).rank(pct=True).to_numpy(float) >= 0.80,
    }


def evaluate_candidate(
    name: str,
    family: str,
    notes: str,
    val: pd.DataFrame,
    test: pd.DataFrame,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    pred265_mape: float,
    transform: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    y = val["target"].to_numpy(float)
    anchor_val = val["anchor"].to_numpy(float)
    anchor_test = test["anchor"].to_numpy(float)
    candidate_mape = mape(y, val_pred)
    delta = candidate_mape - pred265_mape
    test_shift = np.abs(test_pred / np.clip(anchor_test, 1.0, None) - 1.0) * 100.0
    val_shift = np.abs(val_pred / np.clip(anchor_val, 1.0, None) - 1.0) * 100.0

    stress_rows = []
    improved_buckets = 0
    for bucket, mask in stress_masks(val, val_pred).items():
        base_bucket = mape(y[mask], anchor_val[mask])
        cand_bucket = mape(y[mask], val_pred[mask])
        bucket_delta = cand_bucket - base_bucket
        improved_buckets += int(bucket_delta < 0)
        stress_rows.append(
            {
                "candidate": name,
                "family": family,
                "transform": transform,
                "bucket": bucket,
                "rows": int(mask.sum()),
                "pred265_mape": base_bucket,
                "candidate_mape": cand_bucket,
                "delta_vs_pred265": bucket_delta,
            }
        )

    region_rows = []
    major_region_worse = False
    if "Регион" in val.columns:
        for region in MAJOR_REGIONS:
            mask = val["Регион"].astype(str).eq(region).to_numpy()
            if mask.sum() == 0:
                continue
            base_region = mape(y[mask], anchor_val[mask])
            cand_region = mape(y[mask], val_pred[mask])
            region_delta = cand_region - base_region
            major_region_worse |= region_delta > 0
            region_rows.append(
                {
                    "candidate": name,
                    "family": family,
                    "transform": transform,
                    "region": region,
                    "rows": int(mask.sum()),
                    "pred265_mape": base_region,
                    "candidate_mape": cand_region,
                    "delta_vs_pred265": region_delta,
                }
            )

    accepted = (
        delta <= -0.015
        and improved_buckets >= 3
        and float(np.mean(test_shift)) <= 0.25
        and float(np.max(test_shift)) <= 1.50
        and not major_region_worse
    )

    row = {
        "candidate": name,
        "family": family,
        "transform": transform,
        "notes": notes,
        "pred265_oof_mape": pred265_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_pred265": delta,
        "improved_stress_buckets": improved_buckets,
        "val_mean_abs_shift_pct": float(np.mean(val_shift)),
        "val_max_abs_shift_pct": float(np.max(val_shift)),
        "test_mean_abs_shift_pct": float(np.mean(test_shift)),
        "test_max_abs_shift_pct": float(np.max(test_shift)),
        "major_region_worse": bool(major_region_worse),
        "accepted_for_candidate_csv": bool(accepted),
    }
    return row, stress_rows, region_rows


def write_submission(path: Path, ids: np.ndarray, pred: np.ndarray) -> None:
    sub = pd.DataFrame({"new_id": ids.astype(int), "rto": positive(pred)})
    if len(sub) != 18657:
        raise RuntimeError(f"{path.name}: expected 18657 rows, got {len(sub)}")
    if sub["rto"].isna().any() or (sub["rto"] <= 0).any():
        raise RuntimeError(f"{path.name}: invalid rto values")
    sub.to_csv(path, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    rto, static, meta_oof, meta_test = load_data()
    val = target_frame(rto, static, meta_oof, target_year=2024, is_train=True)
    test = target_frame(rto, static, meta_test, target_year=2025, is_train=False)

    y = val["target"].to_numpy(float)
    anchor_val = val["anchor"].to_numpy(float)
    anchor_test = test["anchor"].to_numpy(float)
    pred265_mape = mape(y, anchor_val)

    summary_rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    accepted_rows: list[dict[str, object]] = []

    alt_forecasts = add_alt_forecasts(val, test)
    transforms: list[tuple[str, float | None, float | None]] = [("raw_alt", None, None)]
    for alpha in [0.25, 0.50, 0.75, 1.00]:
        for clip in [0.006, 0.012, 0.020, 0.040]:
            transforms.append((f"blend_a{alpha:.2f}_c{clip:.3f}", alpha, clip))

    candidate_id = 400
    test_prediction_rows = []
    for alt in alt_forecasts:
        for transform_name, alpha, clip in transforms:
            if alpha is None or clip is None:
                val_pred = positive(alt.val_pred)
                test_pred = positive(alt.test_pred)
            else:
                val_pred = safe_blend(anchor_val, alt.val_pred, alpha=alpha, clip=clip)
                test_pred = safe_blend(anchor_test, alt.test_pred, alpha=alpha, clip=clip)

            name = f"{alt.name}__{transform_name}"
            row, stress, regions = evaluate_candidate(
                name=name,
                family=alt.family,
                notes=alt.notes,
                val=val,
                test=test,
                val_pred=val_pred,
                test_pred=test_pred,
                pred265_mape=pred265_mape,
                transform=transform_name,
            )
            summary_rows.append(row)
            stress_rows.extend(stress)
            region_rows.extend(regions)
            test_prediction_rows.append(
                pd.DataFrame(
                    {
                        "new_id": test["new_id"].to_numpy(dtype=int),
                        "candidate": name,
                        "rto": test_pred,
                    }
                )
            )

            if row["accepted_for_candidate_csv"]:
                filename = f"stage2_candidate_{candidate_id}_alt_heuristic_{alt.name}_{transform_name}.csv"
                write_submission(SUBMISSIONS_DIR / filename, test["new_id"].to_numpy(dtype=int), test_pred)
                accepted = dict(row)
                accepted["filename"] = filename
                accepted_rows.append(accepted)
                candidate_id += 1

    summary = pd.DataFrame(summary_rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct"])
    stress_summary = pd.DataFrame(stress_rows)
    region_summary = pd.DataFrame(region_rows)
    accepted = pd.DataFrame(accepted_rows)

    summary.to_csv(OUT_DIR / "heuristic_summary.csv", index=False)
    stress_summary.to_csv(OUT_DIR / "stress_bucket_summary.csv", index=False)
    region_summary.to_csv(OUT_DIR / "major_region_summary.csv", index=False)
    accepted.to_csv(OUT_DIR / "submission_candidates.csv", index=False)

    best = summary.head(20).copy()
    best.to_csv(OUT_DIR / "top20_heuristic_summary.csv", index=False)

    print(f"pred265 OOF MAPE: {pred265_mape:.6f}")
    print("Top 10 alternative heuristic variants:")
    cols = [
        "candidate",
        "candidate_oof_mape",
        "delta_vs_pred265",
        "improved_stress_buckets",
        "test_mean_abs_shift_pct",
        "test_max_abs_shift_pct",
        "accepted_for_candidate_csv",
    ]
    print(summary[cols].head(10).to_string(index=False))
    print(f"Accepted candidate CSVs: {len(accepted_rows)}")
    if accepted_rows:
        print(accepted[["filename", "candidate_oof_mape", "delta_vs_pred265"]].to_string(index=False))


if __name__ == "__main__":
    main()
