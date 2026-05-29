from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_strategy_lifecycle"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"


@dataclass(frozen=True)
class Spec:
    name: str
    group_cols: tuple[str, ...]
    shrink_k: float
    min_count: int
    clip: float


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


def ym_from_period(period: int) -> tuple[int, int]:
    year = period // 12
    month = period % 12
    if month == 0:
        return year - 1, 12
    return year, month


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


def momentum_bin(lag1_to_lag2: pd.Series) -> pd.Series:
    values = np.log(np.clip(lag1_to_lag2.to_numpy(dtype=float), 1e-6, None))
    bins = [-np.inf, -0.10, -0.04, 0.04, 0.10, np.inf]
    labels = ["sharp_drop", "soft_drop", "flat", "soft_growth", "sharp_growth"]
    return pd.Series(pd.cut(values, bins=bins, labels=labels), index=lag1_to_lag2.index).astype("string").fillna("flat")


def leap_ratio_adjustment(source_year: int, target_year: int) -> float:
    if source_year == 2023 and target_year == 2024:
        return 28.0 / 29.0
    if source_year == 2024 and target_year != 2024:
        return 29.0 / 28.0
    return 1.0


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    rto = raw.pivot_table(index="new_id", columns="period", values=TARGET, aggfunc="mean").sort_index()
    static_cols = [
        "new_id",
        "Дата открытия, категориальный",
        "Торговая площадь, категориальный",
        "Регион",
        "Численность населения",
        "Трафик пеший, в час",
        "Трафик авто, в час",
        "Продуктовые магазины (500 м)",
        "Пятерочки (500 м)",
        "Количество касс",
        "Флаг алкогольной лицензии",
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


def add_static_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["age"] = out["Дата открытия, категориальный"].astype("string").fillna("__NA__")
    out["area"] = out["Торговая площадь, категориальный"].astype("string").fillna("__NA__")
    out["region"] = out["Регион"].astype("string").fillna("__NA__")
    out["pyat_bin"] = qbin(out["Пятерочки (500 м)"].fillna(0), 4, "pyat")
    out["grocery_bin"] = qbin(out["Продуктовые магазины (500 м)"].fillna(0), 4, "gro")
    out["population_bin"] = qbin(out["Численность населения"].fillna(0), 4, "pop")
    out["traffic_bin"] = qbin((out["Трафик пеший, в час"].fillna(0) + out["Трафик авто, в час"].fillna(0)), 4, "tr")
    out["cashbox_bin"] = qbin(out["Количество касс"].fillna(out["Количество касс"].median()), 4, "cash")
    return out


def state_for_period(rto: pd.DataFrame, static: pd.DataFrame, target_period: int) -> pd.DataFrame:
    ids = rto.index.to_numpy(dtype=int)
    frame = pd.DataFrame({"new_id": ids}).set_index("new_id")
    frame["target_period"] = target_period
    _, target_month = ym_from_period(target_period)
    frame["target_month"] = target_month
    frame["lag1"] = rto[target_period - 1].astype(float)
    frame["lag2"] = rto[target_period - 2].astype(float)
    frame["lag3"] = rto[target_period - 3].astype(float) if target_period - 3 in rto.columns else frame["lag2"]
    frame["lag1_to_lag2"] = positive(frame["lag1"]) / positive(frame["lag2"])
    frame["momentum_state"] = momentum_bin(frame["lag1_to_lag2"])
    frame["volume_bin"] = qbin(frame["lag1"], 5, "vol")
    frame["volatility_rel3"] = frame[["lag1", "lag2", "lag3"]].std(axis=1) / np.clip(frame[["lag1", "lag2", "lag3"]].mean(axis=1), 1.0, None)
    frame["volatility_bin"] = qbin(frame["volatility_rel3"], 4, "vr")
    frame = frame.join(add_static_bins(static), how="left")
    frame["lifecycle_core"] = frame["age"].astype(str) + "|" + frame["momentum_state"].astype(str)
    frame["lifecycle_area"] = frame["age"].astype(str) + "|" + frame["area"].astype(str) + "|" + frame["momentum_state"].astype(str)
    frame["lifecycle_volume"] = frame["age"].astype(str) + "|" + frame["volume_bin"].astype(str) + "|" + frame["momentum_state"].astype(str)
    frame["lifecycle_volatility"] = frame["age"].astype(str) + "|" + frame["momentum_state"].astype(str) + "|" + frame["volatility_bin"].astype(str)
    frame["lifecycle_competition"] = (
        frame["age"].astype(str)
        + "|"
        + frame["momentum_state"].astype(str)
        + "|"
        + frame["pyat_bin"].astype(str)
        + "|"
        + frame["grocery_bin"].astype(str)
    )
    return frame.reset_index().replace([np.inf, -np.inf], np.nan)


def month_global_ratios(rto: pd.DataFrame, train_periods: list[int], target_year: int | None = None) -> dict[int, float]:
    ratios_by_month: dict[int, list[np.ndarray]] = {}
    for period in train_periods:
        year, month = ym_from_period(period)
        if period - 1 not in rto.columns:
            continue
        values = positive(rto[period]) / positive(rto[period - 1])
        if target_year is not None and month == 3:
            values = values * leap_ratio_adjustment(year, target_year)
        ratios_by_month.setdefault(month, []).append(values[np.isfinite(values)])
    out = {}
    for month, parts in ratios_by_month.items():
        out[month] = float(np.nanmedian(np.concatenate(parts)))
    return out


def build_examples(rto: pd.DataFrame, static: pd.DataFrame, target_period: int) -> pd.DataFrame:
    train_periods = [p for p in sorted(rto.columns) if p >= period_id(2023, 3) and p < target_period and p - 2 in rto.columns]
    global_ratios = month_global_ratios(rto, train_periods)
    rows = []
    for period in train_periods:
        _, month = ym_from_period(period)
        if month not in global_ratios:
            continue
        frame = state_for_period(rto, static, period)
        frame["actual_ratio"] = positive(rto[period].to_numpy(float)) / positive(frame["lag1"])
        frame["global_month_ratio"] = global_ratios[month]
        frame["lifecycle_multiplier"] = frame["actual_ratio"] / frame["global_month_ratio"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def target_base_ratio(rto: pd.DataFrame, target_year: int, target_month: int) -> float:
    parts = []
    for source_year in range(2023, target_year):
        period = period_id(source_year, target_month)
        if period in rto.columns and period - 1 in rto.columns:
            values = positive(rto[period]) / positive(rto[period - 1])
            values = values * leap_ratio_adjustment(source_year, target_year) if target_month == 3 else values
            parts.append(values[np.isfinite(values)])
    if not parts:
        return 1.0
    return float(np.nanmedian(np.concatenate(parts)))


def multiplier_map(examples: pd.DataFrame, group_cols: tuple[str, ...], shrink_k: float, min_count: int, clip: float) -> pd.DataFrame:
    stats = examples.groupby(list(group_cols), observed=True).agg(
        multiplier_median=("lifecycle_multiplier", "median"),
        multiplier_mean=("lifecycle_multiplier", "mean"),
        count=("lifecycle_multiplier", "size"),
        iqr=("lifecycle_multiplier", lambda s: float(np.nanquantile(s, 0.75) - np.nanquantile(s, 0.25))),
    )
    raw = 0.75 * stats["multiplier_median"] + 0.25 * stats["multiplier_mean"]
    shrink = stats["count"] / (stats["count"] + shrink_k)
    stats["multiplier"] = np.exp(shrink * np.log(np.clip(raw, 1e-6, None)))
    stats["multiplier"] = np.clip(stats["multiplier"], np.exp(-clip), np.exp(clip))
    stats.loc[stats["count"] < min_count, "multiplier"] = 1.0
    return stats.reset_index()


def predict(target: pd.DataFrame, examples: pd.DataFrame, spec: Spec, base_ratio: float) -> tuple[np.ndarray, pd.Series, pd.Series]:
    mapping = multiplier_map(examples, spec.group_cols, spec.shrink_k, spec.min_count, spec.clip)
    merged = target[list(spec.group_cols)].merge(mapping, on=list(spec.group_cols), how="left")
    multiplier = merged["multiplier"].fillna(1.0).to_numpy(float)
    return positive(target["lag1"].to_numpy(float) * base_ratio * multiplier), merged["count"].fillna(0), merged["iqr"].fillna(mapping["iqr"].median())


def make_specs() -> list[Spec]:
    group_sets = [
        ("age",),
        ("age", "area"),
        ("lifecycle_core",),
        ("lifecycle_area",),
        ("lifecycle_volume",),
        ("lifecycle_volatility",),
        ("lifecycle_competition",),
        ("age", "area", "volume_bin"),
        ("age", "area", "momentum_state", "pyat_bin"),
        ("age", "area", "momentum_state", "grocery_bin"),
    ]
    specs = []
    for group_cols in group_sets:
        for shrink_k in [100.0, 300.0, 800.0]:
            for clip in [0.006, 0.012, 0.020, 0.040]:
                min_count = 300 if len(group_cols) <= 2 else 600
                specs.append(
                    Spec(
                        name=f"monthly_lifecycle__{'_'.join(group_cols)}__k{int(shrink_k)}__c{str(clip).replace('.', 'p')}",
                        group_cols=group_cols,
                        shrink_k=shrink_k,
                        min_count=min_count,
                        clip=clip,
                    )
                )
    return specs


def evaluate(
    val: pd.DataFrame,
    test: pd.DataFrame,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    name: str,
    spec: Spec | None,
    pred265_mape: float,
    counts: pd.Series | None = None,
    iqr: pd.Series | None = None,
) -> dict[str, object]:
    candidate_mape = mape(val["target"], pred_val)
    shift = np.abs(pred_test / positive(test["pred265"]) - 1.0) * 100.0
    return {
        "candidate": name,
        "group_cols": "|".join(spec.group_cols) if spec else "",
        "shrink_k": spec.shrink_k if spec else np.nan,
        "clip": spec.clip if spec else np.nan,
        "min_count": spec.min_count if spec else np.nan,
        "pred265_oof_mape": pred265_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_pred265": candidate_mape - pred265_mape,
        "test_mean_abs_shift_pct_vs_pred265": float(np.mean(shift)),
        "test_max_abs_shift_pct_vs_pred265": float(np.max(shift)),
        "avg_group_count": float(counts.mean()) if counts is not None else np.nan,
        "avg_group_iqr": float(iqr.mean()) if iqr is not None else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    rto, static, meta_oof, meta_test = load_data()

    val_period = period_id(2024, 3)
    test_period = period_id(2025, 3)
    val = state_for_period(rto, static, val_period)
    test = state_for_period(rto, static, test_period)
    val = val.merge(meta_oof[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
    val["target"] = val[META_TARGET]
    val["pred265"] = val["pred265_oof"]
    test = test.merge(meta_test[["new_id", "pred265_submission"]], on="new_id", how="left")
    test["pred265"] = test["pred265_submission"]

    examples_val = build_examples(rto, static, val_period)
    examples_test = build_examples(rto, static, test_period)
    base_ratio_val = target_base_ratio(rto, 2024, 3)
    base_ratio_test = target_base_ratio(rto, 2025, 3)
    pred265_mape = mape(val["target"], val["pred265"])

    rows = []
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    for spec in make_specs():
        pred_val, counts, iqr = predict(val, examples_val, spec, base_ratio_val)
        pred_test, _, _ = predict(test, examples_test, spec, base_ratio_test)
        val_preds[spec.name] = pred_val
        test_preds[spec.name] = pred_test
        rows.append(evaluate(val, test, pred_val, pred_test, spec.name, spec, pred265_mape, counts, iqr))

    summary = pd.DataFrame(rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"]).reset_index(drop=True)
    top_names = summary.head(5)["candidate"].tolist()
    for n in [2, 3, 5]:
        name = f"monthly_lifecycle_ensemble_top{n}"
        pred_val = np.mean([val_preds[x] for x in top_names[:n]], axis=0)
        pred_test = np.mean([test_preds[x] for x in top_names[:n]], axis=0)
        val_preds[name] = pred_val
        test_preds[name] = pred_test
        summary = pd.concat([summary, pd.DataFrame([evaluate(val, test, pred_val, pred_test, name, None, pred265_mape)])])

    summary = summary.sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"]).reset_index(drop=True)
    summary["accepted_for_submission"] = (
        (summary["delta_vs_pred265"] <= -0.015)
        & (summary["test_mean_abs_shift_pct_vs_pred265"] <= 3.0)
        & (summary["test_max_abs_shift_pct_vs_pred265"] <= 40.0)
    )
    summary.to_csv(OUT_DIR / "monthly_lifecycle_summary.csv", index=False)

    accepted_rows = []
    for _, row in summary.loc[summary["accepted_for_submission"]].head(3).iterrows():
        name = str(row["candidate"])
        filename = f"stage2_{name}.csv".replace("|", "_").replace(" ", "_").replace(",", "")
        sub = pd.DataFrame({"new_id": test["new_id"].astype(int), "rto": positive(test_preds[name])})
        if len(sub) != 18657 or sub["rto"].isna().any() or (sub["rto"] <= 0).any():
            raise RuntimeError(f"Bad submission format for {filename}")
        sub.to_csv(SUBMISSIONS_DIR / filename, index=False)
        accepted = row.to_dict()
        accepted["filename"] = filename
        accepted_rows.append(accepted)
    pd.DataFrame(accepted_rows).to_csv(OUT_DIR / "monthly_lifecycle_submission_candidates.csv", index=False)

    print(f"pred265 OOF MAPE: {pred265_mape:.6f}")
    print(f"base_ratio_val: {base_ratio_val:.6f}, base_ratio_test: {base_ratio_test:.6f}")
    print("Top monthly lifecycle candidates:")
    print(
        summary[
            [
                "candidate",
                "candidate_oof_mape",
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
