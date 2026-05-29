from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_strategy_deseasonalized_normal"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"


@dataclass(frozen=True)
class Spec:
    name: str
    group_col: str
    level_method: str
    seasonality_mode: str
    group_shrink_k: float
    store_season_weight: float
    min_group_size: int


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


def positive(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1.0, None)


def qbin(series: pd.Series, q: int, prefix: str) -> pd.Series:
    ranked = series.rank(method="first")
    labels = [f"{prefix}{i}" for i in range(q)]
    return pd.Series(pd.qcut(ranked, q, labels=labels), index=series.index).astype("string").fillna(f"{prefix}NA")


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
        "Пятерочки (500 м)",
        "Продуктовые магазины (500 м)",
    ]
    static = (
        raw.sort_values(["new_id", "Год", "Месяц"])
        .groupby("new_id", as_index=False)[[c for c in static_cols if c in raw.columns]]
        .first()
        .set_index("new_id")
    )
    for col in static.select_dtypes(include=["object", "string"]).columns:
        static[col] = static[col].astype("string").fillna("__NA__")
    meta_oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    meta_test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    return rto, static, meta_oof, meta_test


def add_group_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["global"] = "all"
    out["region"] = out["Регион"].astype(str)
    out["city"] = out["Населенный пункт"].astype(str)
    city_counts = out["city"].value_counts()
    out["city_big"] = out["city"].where(out["city"].map(city_counts).fillna(0) >= 80, "__SMALL_CITY__")
    out["area"] = out["Торговая площадь, категориальный"].astype(str)
    out["age"] = out["Дата открытия, категориальный"].astype(str)
    out["region_area"] = out["region"] + "|" + out["area"]
    out["region_age"] = out["region"] + "|" + out["age"]
    out["area_age"] = out["area"] + "|" + out["age"]
    out["volume_bin"] = qbin(out["feb"], 5, "vol")
    out["feb_to_jan"] = positive(out["feb"]) / positive(out["jan"])
    out["momentum_bin"] = pd.Series(
        pd.cut(
            np.log(np.clip(out["feb_to_jan"].to_numpy(float), 1e-6, None)),
            bins=[-np.inf, -0.10, -0.04, 0.04, 0.10, np.inf],
            labels=["sharp_drop", "soft_drop", "flat", "soft_growth", "sharp_growth"],
        ),
        index=out.index,
    ).astype("string").fillna("flat")
    out["region_volume"] = out["region"] + "|" + out["volume_bin"].astype(str)
    out["age_volume"] = out["age"] + "|" + out["volume_bin"].astype(str)
    out["area_volume"] = out["area"] + "|" + out["volume_bin"].astype(str)
    return out


def target_frame(rto: pd.DataFrame, static: pd.DataFrame, meta: pd.DataFrame, target_year: int, is_train: bool) -> pd.DataFrame:
    frame = pd.DataFrame({"new_id": rto.index.to_numpy(dtype=int)}).set_index("new_id")
    for label, year, month in [
        ("sep_prev", target_year - 1, 9),
        ("oct_prev", target_year - 1, 10),
        ("nov_prev", target_year - 1, 11),
        ("dec_prev", target_year - 1, 12),
        ("jan", target_year, 1),
        ("feb", target_year, 2),
    ]:
        frame[label] = rto[period_id(year, month)].astype(float)
    frame = frame.join(static, how="left").reset_index()
    frame = add_group_keys(frame)
    if is_train:
        frame = frame.merge(meta[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
        frame["target"] = frame[META_TARGET]
        frame["pred265"] = frame["pred265_oof"]
    else:
        frame = frame.merge(meta[["new_id", "pred265_submission"]], on="new_id", how="left")
        frame["pred265"] = frame["pred265_submission"]
    return frame.replace([np.inf, -np.inf], np.nan)


def history_long(rto: pd.DataFrame, static: pd.DataFrame, target_year: int) -> pd.DataFrame:
    rows = []
    for year in range(2023, target_year):
        periods = [period_id(year, m) for m in range(1, 13)]
        values = rto[periods].copy()
        annual_mean = values.mean(axis=1).clip(lower=1.0)
        for month, period in enumerate(periods, start=1):
            part = pd.DataFrame(
                {
                    "new_id": rto.index.to_numpy(dtype=int),
                    "source_year": year,
                    "month": month,
                    "rto": values[period].to_numpy(float),
                    "annual_mean": annual_mean.to_numpy(float),
                }
            )
            part["store_season_index"] = part["rto"] / np.clip(part["annual_mean"], 1.0, None)
            rows.append(part)
    hist = pd.concat(rows, ignore_index=True)
    hist = hist.join(static, on="new_id", how="left")
    # Historical group bins use historical RTO only where target-year bins are unavailable.
    hist["jan"] = hist["annual_mean"]
    hist["feb"] = hist["annual_mean"]
    hist = add_group_keys(hist)
    return hist.replace([np.inf, -np.inf], np.nan)


def current_heuristic_pred(frame: pd.DataFrame, rto: pd.DataFrame, target_year: int, calendar_lambda: float = 0.50) -> np.ndarray:
    ratios = []
    for source_year in range(2023, target_year):
        ratio = positive(rto[period_id(source_year, 3)]) / positive(rto[period_id(source_year, 2)])
        if source_year == 2023 and target_year == 2024:
            ratio = ratio * ((1.0 - calendar_lambda) + calendar_lambda * (28.0 / 29.0))
        elif source_year == 2024 and target_year == 2025:
            ratio = ratio * ((1.0 - calendar_lambda) + calendar_lambda * (29.0 / 28.0))
        ratios.append(ratio[np.isfinite(ratio)])
    factor = float(np.nanmedian(np.concatenate(ratios)))
    return positive(frame["feb"]) * factor


def group_season_indices(hist: pd.DataFrame, target: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    work = target.copy()
    group_size = work.groupby(spec.group_col, observed=True)["new_id"].transform("size")
    work.loc[group_size < spec.min_group_size, spec.group_col] = "fallback_all"

    hist_work = hist.copy()
    known_groups = set(work[spec.group_col].astype(str))
    hist_work.loc[~hist_work[spec.group_col].astype(str).isin(known_groups), spec.group_col] = "fallback_all"

    global_month = hist_work.groupby("month", observed=True).agg(
        month_sum=("rto", "sum"),
        annual_sum=("annual_mean", "sum"),
    )
    global_index = global_month["month_sum"] / global_month["annual_sum"].clip(lower=1.0)

    stats = hist_work.groupby([spec.group_col, "month"], observed=True).agg(
        month_sum=("rto", "sum"),
        annual_sum=("annual_mean", "sum"),
        median_store_idx=("store_season_index", "median"),
        count=("store_season_index", "size"),
    )
    raw = 0.70 * (stats["month_sum"] / stats["annual_sum"].clip(lower=1.0)) + 0.30 * stats["median_store_idx"]
    gidx = stats.index.get_level_values("month").map(global_index)
    shrink = stats["count"] / (stats["count"] + spec.group_shrink_k)
    stats["season_index"] = np.exp(np.log(gidx) + shrink * np.log(np.clip(raw, 1e-6, None) / gidx))
    return stats.reset_index()


def map_group_index(target: pd.DataFrame, index_table: pd.DataFrame, spec: Spec, month: int) -> pd.Series:
    sub = index_table[index_table["month"].eq(month)][[spec.group_col, "season_index"]]
    return target[[spec.group_col]].merge(sub, on=spec.group_col, how="left")["season_index"].fillna(index_table[index_table["month"].eq(month)]["season_index"].median()).astype(float)


def map_store_index(target: pd.DataFrame, hist: pd.DataFrame, month: int) -> pd.Series:
    store_idx = hist.loc[hist["month"].eq(month)].groupby("new_id", observed=True)["store_season_index"].median()
    fallback = float(hist.loc[hist["month"].eq(month), "store_season_index"].median())
    return target["new_id"].map(store_idx).fillna(fallback).astype(float)


def seasonal_index(target: pd.DataFrame, hist: pd.DataFrame, index_table: pd.DataFrame, spec: Spec, month: int) -> pd.Series:
    group_idx = map_group_index(target, index_table, spec, month)
    if spec.store_season_weight <= 0:
        return group_idx
    store_idx = map_store_index(target, hist, month)
    return np.exp((1.0 - spec.store_season_weight) * np.log(group_idx) + spec.store_season_weight * np.log(np.clip(store_idx, 1e-6, None)))


def normal_level(frame: pd.DataFrame, hist: pd.DataFrame, index_table: pd.DataFrame, spec: Spec) -> np.ndarray:
    month_cols = [
        ("sep_prev", 9),
        ("oct_prev", 10),
        ("nov_prev", 11),
        ("dec_prev", 12),
        ("jan", 1),
        ("feb", 2),
    ]
    deseasonalized = {}
    for col, month in month_cols:
        deseasonalized[col] = positive(frame[col]) / np.clip(seasonal_index(frame, hist, index_table, spec, month), 1e-6, None)
    d = pd.DataFrame(deseasonalized, index=frame.index)
    if spec.level_method == "median_3":
        return d[["dec_prev", "jan", "feb"]].median(axis=1).to_numpy(float)
    if spec.level_method == "weighted_3":
        return (0.20 * d["dec_prev"] + 0.30 * d["jan"] + 0.50 * d["feb"]).to_numpy(float)
    if spec.level_method == "median_6":
        return d[["sep_prev", "oct_prev", "nov_prev", "dec_prev", "jan", "feb"]].median(axis=1).to_numpy(float)
    if spec.level_method == "weighted_6":
        weights = np.array([0.05, 0.07, 0.10, 0.18, 0.25, 0.35])
        return (d[["sep_prev", "oct_prev", "nov_prev", "dec_prev", "jan", "feb"]].to_numpy(float) @ weights).astype(float)
    if spec.level_method == "feb_only":
        return d["feb"].to_numpy(float)
    raise ValueError(spec.level_method)


def forecast(frame: pd.DataFrame, hist: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    index_table = group_season_indices(hist, frame, spec)
    level = normal_level(frame, hist, index_table, spec)
    march_index = seasonal_index(frame, hist, index_table, spec, 3)
    pred = positive(level * march_index)
    return pred, level


def make_specs() -> list[Spec]:
    groups = [
        "global",
        "region",
        "city_big",
        "area",
        "age",
        "region_area",
        "region_age",
        "area_age",
        "region_volume",
        "age_volume",
        "area_volume",
    ]
    level_methods = ["feb_only", "median_3", "weighted_3", "median_6", "weighted_6"]
    specs = []
    for group in groups:
        for method in level_methods:
            for shrink_k in [20.0, 100.0, 300.0, 1000.0]:
                for store_weight in [0.0, 0.15, 0.35]:
                    min_group_size = 1 if group == "global" else (40 if group in {"region", "area", "age"} else 80)
                    specs.append(
                        Spec(
                            name=f"deseason_{method}_{group}_k{int(shrink_k)}_sw{str(store_weight).replace('.', 'p')}",
                            group_col=group,
                            level_method=method,
                            seasonality_mode="multiplicative",
                            group_shrink_k=shrink_k,
                            store_season_weight=store_weight,
                            min_group_size=min_group_size,
                        )
                    )
    return specs


def evaluate(val: pd.DataFrame, test: pd.DataFrame, pred_val: np.ndarray, pred_test: np.ndarray, spec: Spec | None, current_mape: float, best_clean_mape: float) -> dict[str, object]:
    candidate_mape = mape(val["target"], pred_val)
    shift = np.abs(pred_test / positive(test["current_heuristic_pred"]) - 1.0) * 100.0
    return {
        "candidate": spec.name if spec else "ensemble",
        "group_col": spec.group_col if spec else "ensemble",
        "level_method": spec.level_method if spec else "",
        "group_shrink_k": spec.group_shrink_k if spec else np.nan,
        "store_season_weight": spec.store_season_weight if spec else np.nan,
        "current_heuristic_mape": current_mape,
        "best_known_clean_mape": best_clean_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_current_heuristic": candidate_mape - current_mape,
        "delta_vs_best_known_clean": candidate_mape - best_clean_mape,
        "test_mean_abs_shift_pct_vs_current": float(np.mean(shift)),
        "test_max_abs_shift_pct_vs_current": float(np.max(shift)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    rto, static, meta_oof, meta_test = load_data()
    val = target_frame(rto, static, meta_oof, 2024, is_train=True)
    test = target_frame(rto, static, meta_test, 2025, is_train=False)
    hist_val = history_long(rto, static, 2024)
    hist_test = history_long(rto, static, 2025)

    val["current_heuristic_pred"] = current_heuristic_pred(val, rto, 2024, calendar_lambda=0.50)
    test["current_heuristic_pred"] = current_heuristic_pred(test, rto, 2025, calendar_lambda=0.50)
    current_mape = mape(val["target"], val["current_heuristic_pred"])
    best_clean_mape = 3.848401

    rows = []
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    for spec in make_specs():
        pred_val, _ = forecast(val, hist_val, spec)
        pred_test, _ = forecast(test, hist_test, spec)
        val_preds[spec.name] = pred_val
        test_preds[spec.name] = pred_test
        rows.append(evaluate(val, test, pred_val, pred_test, spec, current_mape, best_clean_mape))

    summary = pd.DataFrame(rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_current"]).reset_index(drop=True)
    top_names = summary.head(5)["candidate"].tolist()
    for n in [2, 3, 5]:
        name = f"deseason_ensemble_top{n}"
        pred_val = np.mean([val_preds[x] for x in top_names[:n]], axis=0)
        pred_test = np.mean([test_preds[x] for x in top_names[:n]], axis=0)
        val_preds[name] = pred_val
        test_preds[name] = pred_test
        row = evaluate(val, test, pred_val, pred_test, None, current_mape, best_clean_mape)
        row["candidate"] = name
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

    summary = summary.sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_current"]).reset_index(drop=True)
    summary["beats_current_heuristic"] = summary["delta_vs_current_heuristic"] < 0
    summary["beats_best_known_clean"] = summary["delta_vs_best_known_clean"] < 0
    summary.to_csv(OUT_DIR / "deseasonalized_normal_summary.csv", index=False)

    accepted_rows = []
    for _, row in summary.loc[summary["beats_current_heuristic"]].head(3).iterrows():
        name = str(row["candidate"])
        filename = f"stage2_{name}.csv".replace("|", "_").replace(" ", "_").replace(",", "")
        sub = pd.DataFrame({"new_id": test["new_id"].astype(int), "rto": positive(test_preds[name])})
        if len(sub) != 18657 or sub["rto"].isna().any() or (sub["rto"] <= 0).any():
            raise RuntimeError(f"Bad submission format for {filename}")
        sub.to_csv(SUBMISSIONS_DIR / filename, index=False)
        accepted = row.to_dict()
        accepted["filename"] = filename
        accepted_rows.append(accepted)
    pd.DataFrame(accepted_rows).to_csv(OUT_DIR / "submission_candidates.csv", index=False)

    print(f"Current clean heuristic cal050_median_mom000 MAPE: {current_mape:.6f}")
    print(f"Best known clean heuristic local MAPE reference: {best_clean_mape:.6f}")
    print("Top deseasonalized normal-level candidates:")
    print(
        summary[
            [
                "candidate",
                "candidate_oof_mape",
                "delta_vs_current_heuristic",
                "delta_vs_best_known_clean",
                "test_mean_abs_shift_pct_vs_current",
                "test_max_abs_shift_pct_vs_current",
                "beats_current_heuristic",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )
    print(f"Candidate CSVs beating current clean heuristic: {len(accepted_rows)}")


if __name__ == "__main__":
    main()
