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
    base: str
    group_cols: tuple[str, ...]
    shrink_k: float
    min_count: int


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


def ratio(num: np.ndarray | pd.Series, den: np.ndarray | pd.Series) -> np.ndarray:
    return positive(num) / positive(den)


def leap_ratio_adjustment(source_year: int, target_year: int) -> float:
    if source_year == 2023 and target_year == 2024:
        return 28.0 / 29.0
    if source_year == 2024 and target_year != 2024:
        return 29.0 / 28.0
    return 1.0


def janfeb_ratio_adjustment(source_year: int, target_year: int) -> float:
    if source_year == 2023 and target_year == 2024:
        return 59.0 / 60.0
    if source_year == 2024 and target_year != 2024:
        return 60.0 / 59.0
    return 1.0


def fixed_momentum_bin(feb_to_jan: pd.Series) -> pd.Series:
    values = np.log(np.clip(feb_to_jan.to_numpy(dtype=float), 1e-6, None))
    bins = [-np.inf, -0.10, -0.04, 0.04, 0.10, np.inf]
    labels = ["sharp_drop", "soft_drop", "flat", "soft_growth", "sharp_growth"]
    return pd.Series(pd.cut(values, bins=bins, labels=labels), index=feb_to_jan.index).astype("string").fillna("flat")


def qbin(series: pd.Series, q: int, prefix: str) -> pd.Series:
    ranked = series.rank(method="first")
    labels = [f"{prefix}{i}" for i in range(q)]
    return pd.Series(pd.qcut(ranked, q, labels=labels), index=series.index).astype("string").fillna(f"{prefix}NA")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    rto = raw.pivot_table(index="new_id", columns="period", values=TARGET, aggfunc="mean").sort_index()

    static_cols = [
        "new_id",
        "Дата открытия, категориальный",
        "Торговая площадь, категориальный",
        "Населенный пункт",
        "Регион",
        "Численность населения",
        "Количество домохозяйств",
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

    monthly_cols = [
        "new_id",
        "Год",
        "Месяц",
        "Среднее количество промо товаров в чеке",
        "Среднее количество товаров в чеке",
        "Среднее количество отмен",
        "Рабочие часы в день",
    ]
    monthly = raw[[c for c in monthly_cols if c in raw.columns]].copy()
    monthly["period"] = monthly["Год"] * 12 + monthly["Месяц"]
    meta_oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    meta_test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    return raw, rto, static, monthly, meta_oof, meta_test


def add_static_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["age"] = out["Дата открытия, категориальный"].astype("string").fillna("__NA__")
    out["area"] = out["Торговая площадь, категориальный"].astype("string").fillna("__NA__")
    out["region"] = out["Регион"].astype("string").fillna("__NA__")
    out["alcohol"] = out["Флаг алкогольной лицензии"].fillna(-1).astype(int).astype(str)
    out["cashbox_bin"] = qbin(out["Количество касс"].fillna(out["Количество касс"].median()), 4, "cash")
    out["pyat_bin"] = qbin(out["Пятерочки (500 м)"].fillna(0), 4, "pyat")
    out["grocery_bin"] = qbin(out["Продуктовые магазины (500 м)"].fillna(0), 4, "gro")
    out["population_bin"] = qbin(out["Численность населения"].fillna(0), 4, "pop")
    out["traffic_bin"] = qbin((out["Трафик пеший, в час"].fillna(0) + out["Трафик авто, в час"].fillna(0)), 4, "tr")
    out["competition_pressure"] = (
        out["pyat_bin"].astype(str) + "|" + out["grocery_bin"].astype(str) + "|" + out["population_bin"].astype(str)
    )
    return out


def monthly_feature(monthly: pd.DataFrame, col: str, year: int, month: int) -> pd.Series:
    period = period_id(year, month)
    part = monthly.loc[monthly["period"].eq(period), ["new_id", col]]
    return part.set_index("new_id")[col]


def build_state_frame(
    rto: pd.DataFrame,
    static: pd.DataFrame,
    monthly: pd.DataFrame,
    state_year: int,
) -> pd.DataFrame:
    ids = rto.index.to_numpy(dtype=int)
    frame = pd.DataFrame({"new_id": ids}).set_index("new_id")
    frame["jan"] = rto[period_id(state_year, 1)].astype(float)
    frame["feb"] = rto[period_id(state_year, 2)].astype(float)
    frame["janfeb_mean"] = frame[["jan", "feb"]].mean(axis=1)
    frame["feb_to_jan"] = ratio(frame["feb"], frame["jan"])
    frame["momentum_state"] = fixed_momentum_bin(frame["feb_to_jan"])
    frame["volume_bin"] = qbin(frame["feb"], 5, "vol")
    frame["janfeb_volume_bin"] = qbin(frame["janfeb_mean"], 5, "jfv")
    frame["jan_feb_gap_bin"] = qbin(np.abs(np.log(np.clip(frame["feb_to_jan"], 1e-6, None))), 4, "gap")

    for col, out_col in [
        ("Среднее количество промо товаров в чеке", "promo_janfeb"),
        ("Среднее количество товаров в чеке", "basket_janfeb"),
        ("Среднее количество отмен", "cancel_janfeb"),
        ("Рабочие часы в день", "hours_janfeb"),
    ]:
        if col not in monthly.columns:
            continue
        jan = monthly_feature(monthly, col, state_year, 1)
        feb = monthly_feature(monthly, col, state_year, 2)
        frame[out_col] = pd.concat([jan, feb], axis=1).mean(axis=1)
        frame[f"{out_col}_bin"] = qbin(frame[out_col].fillna(frame[out_col].median()), 4, out_col[:3])

    frame = frame.join(add_static_bins(static), how="left")
    frame["lifecycle_core"] = frame["age"].astype(str) + "|" + frame["momentum_state"].astype(str)
    frame["lifecycle_area"] = frame["age"].astype(str) + "|" + frame["area"].astype(str) + "|" + frame["momentum_state"].astype(str)
    frame["lifecycle_volume"] = frame["age"].astype(str) + "|" + frame["volume_bin"].astype(str) + "|" + frame["momentum_state"].astype(str)
    frame["lifecycle_competition"] = (
        frame["age"].astype(str) + "|" + frame["momentum_state"].astype(str) + "|" + frame["competition_pressure"].astype(str)
    )
    frame["lifecycle_basket"] = (
        frame["age"].astype(str)
        + "|"
        + frame["momentum_state"].astype(str)
        + "|"
        + frame.get("basket_janfeb_bin", pd.Series("__NA__", index=frame.index)).astype(str)
    )
    frame["region_lifecycle"] = frame["region"].astype(str) + "|" + frame["age"].astype(str) + "|" + frame["momentum_state"].astype(str)
    return frame.reset_index().replace([np.inf, -np.inf], np.nan)


def build_history(
    rto: pd.DataFrame,
    static: pd.DataFrame,
    monthly: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    parts = []
    for source_year in range(2023, target_year):
        if period_id(source_year, 3) not in rto.columns:
            continue
        state = build_state_frame(rto, static, monthly, source_year)
        mar = rto[period_id(source_year, 3)].rename("source_mar").reset_index()
        state = state.merge(mar, on="new_id", how="left")
        state["source_year"] = source_year
        state["target_year"] = target_year
        state["target_feb_ratio"] = ratio(state["source_mar"], state["feb"]) * leap_ratio_adjustment(source_year, target_year)
        state["target_janfeb_ratio"] = ratio(state["source_mar"], state["janfeb_mean"]) * janfeb_ratio_adjustment(source_year, target_year)
        parts.append(state)
    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def target_frame(
    rto: pd.DataFrame,
    static: pd.DataFrame,
    monthly: pd.DataFrame,
    meta: pd.DataFrame,
    target_year: int,
    is_train: bool,
) -> pd.DataFrame:
    frame = build_state_frame(rto, static, monthly, target_year)
    if is_train:
        frame = frame.merge(meta[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
        frame["target"] = frame[META_TARGET]
        frame["pred265"] = frame["pred265_oof"]
    else:
        frame = frame.merge(meta[["new_id", "pred265_submission"]], on="new_id", how="left")
        frame["pred265"] = frame["pred265_submission"]
    return frame


def group_ratio_map(
    history: pd.DataFrame,
    group_cols: tuple[str, ...],
    ratio_col: str,
    shrink_k: float,
    min_count: int,
) -> pd.DataFrame:
    global_ratio = float(np.average(history[ratio_col], weights=positive(history["feb"])))
    stats = history.groupby(list(group_cols), observed=True).agg(
        ratio_median=(ratio_col, "median"),
        ratio_mean=(ratio_col, "mean"),
        ratio_count=(ratio_col, "size"),
        feb_sum=("feb", "sum"),
        mar_sum=("source_mar", "sum"),
        ratio_iqr=(ratio_col, lambda s: float(np.nanquantile(s, 0.75) - np.nanquantile(s, 0.25))),
    )
    raw_ratio = 0.65 * (stats["mar_sum"] / stats["feb_sum"].clip(lower=1.0)) + 0.35 * stats["ratio_median"]
    shrink = stats["ratio_count"] / (stats["ratio_count"] + shrink_k)
    stats["ratio"] = np.exp(np.log(global_ratio) + shrink * np.log(np.clip(raw_ratio, 1e-6, None) / global_ratio))
    stats.loc[stats["ratio_count"] < min_count, "ratio"] = global_ratio
    return stats.reset_index()


def predict_with_spec(target: pd.DataFrame, history: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, pd.Series, pd.Series]:
    ratio_col = "target_feb_ratio" if spec.base == "feb" else "target_janfeb_ratio"
    base_values = target["feb"].to_numpy(float) if spec.base == "feb" else target["janfeb_mean"].to_numpy(float)
    mapping = group_ratio_map(history, spec.group_cols, ratio_col, spec.shrink_k, spec.min_count)
    merged = target[list(spec.group_cols)].merge(mapping, on=list(spec.group_cols), how="left")
    global_ratio = float(np.average(history[ratio_col], weights=positive(history["feb"])))
    ratio_values = merged["ratio"].fillna(global_ratio).to_numpy(float)
    counts = merged["ratio_count"].fillna(0)
    iqr = merged["ratio_iqr"].fillna(mapping["ratio_iqr"].median())
    return positive(base_values * ratio_values), counts, iqr


def make_specs() -> list[Spec]:
    groups = [
        ("age",),
        ("age", "area"),
        ("lifecycle_core",),
        ("lifecycle_area",),
        ("lifecycle_volume",),
        ("lifecycle_competition",),
        ("lifecycle_basket",),
        ("region_lifecycle",),
        ("age", "area", "volume_bin"),
        ("age", "area", "momentum_state", "pyat_bin"),
        ("age", "area", "momentum_state", "grocery_bin"),
        ("age", "area", "momentum_state", "basket_janfeb_bin"),
    ]
    specs = []
    for base in ["feb", "janfeb"]:
        for group_cols in groups:
            for shrink_k in [25.0, 75.0, 200.0]:
                min_count = 20 if len(group_cols) <= 2 else 35
                name = f"lifecycle_{base}__{'_'.join(group_cols)}__k{int(shrink_k)}"
                specs.append(Spec(name=name, base=base, group_cols=group_cols, shrink_k=shrink_k, min_count=min_count))
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
    y = val["target"].to_numpy(float)
    candidate_mape = mape(y, pred_val)
    pred265_test = test["pred265"].to_numpy(float)
    shift = np.abs(pred_test / positive(pred265_test) - 1.0) * 100.0
    row = {
        "candidate": name,
        "base": spec.base if spec else "ensemble",
        "group_cols": "|".join(spec.group_cols) if spec else "",
        "shrink_k": spec.shrink_k if spec else np.nan,
        "min_count": spec.min_count if spec else np.nan,
        "pred265_oof_mape": pred265_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_pred265": candidate_mape - pred265_mape,
        "test_mean_abs_shift_pct_vs_pred265": float(np.mean(shift)),
        "test_max_abs_shift_pct_vs_pred265": float(np.max(shift)),
        "avg_group_count": float(counts.mean()) if counts is not None else np.nan,
        "avg_group_iqr": float(iqr.mean()) if iqr is not None else np.nan,
    }
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _, rto, static, monthly, meta_oof, meta_test = load_data()
    val = target_frame(rto, static, monthly, meta_oof, target_year=2024, is_train=True)
    test = target_frame(rto, static, monthly, meta_test, target_year=2025, is_train=False)
    hist_val = build_history(rto, static, monthly, target_year=2024)
    hist_test = build_history(rto, static, monthly, target_year=2025)
    pred265_mape = mape(val["target"], val["pred265"])

    rows = []
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    for spec in make_specs():
        pred_val, counts, iqr = predict_with_spec(val, hist_val, spec)
        pred_test, _, _ = predict_with_spec(test, hist_test, spec)
        val_preds[spec.name] = pred_val
        test_preds[spec.name] = pred_test
        rows.append(evaluate(val, test, pred_val, pred_test, spec.name, spec, pred265_mape, counts, iqr))

    summary = pd.DataFrame(rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"])

    # Ensemble only independent lifecycle forecasts. This is still not a pred265 blend.
    top_names = summary.head(5)["candidate"].tolist()
    for n in [2, 3, 5]:
        chosen = top_names[:n]
        pred_val = np.mean([val_preds[name] for name in chosen], axis=0)
        pred_test = np.mean([test_preds[name] for name in chosen], axis=0)
        name = f"lifecycle_ensemble_top{n}"
        summary = pd.concat(
            [
                summary,
                pd.DataFrame([evaluate(val, test, pred_val, pred_test, name, None, pred265_mape)]),
            ],
            ignore_index=True,
        )
        val_preds[name] = pred_val
        test_preds[name] = pred_test

    summary = summary.sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"]).reset_index(drop=True)
    summary["accepted_for_submission"] = (
        (summary["delta_vs_pred265"] <= -0.015)
        & (summary["test_mean_abs_shift_pct_vs_pred265"] <= 3.0)
        & (summary["test_max_abs_shift_pct_vs_pred265"] <= 40.0)
    )
    summary.to_csv(OUT_DIR / "lifecycle_summary.csv", index=False)

    # Segment diagnostics for the best standalone lifecycle candidate.
    best_name = str(summary.iloc[0]["candidate"])
    best_val = val_preds[best_name]
    y = val["target"].to_numpy(float)
    pred265 = val["pred265"].to_numpy(float)
    bucket_rows = []
    for col in ["age", "area", "momentum_state", "volume_bin", "pyat_bin", "grocery_bin"]:
        for level, idx in val.groupby(col, observed=True).groups.items():
            mask = val.index.isin(idx)
            if mask.sum() < 200:
                continue
            bucket_rows.append(
                {
                    "candidate": best_name,
                    "bucket_col": col,
                    "bucket": str(level),
                    "rows": int(mask.sum()),
                    "pred265_mape": mape(y[mask], pred265[mask]),
                    "candidate_mape": mape(y[mask], best_val[mask]),
                    "delta_vs_pred265": mape(y[mask], best_val[mask]) - mape(y[mask], pred265[mask]),
                }
            )
    pd.DataFrame(bucket_rows).sort_values(["bucket_col", "delta_vs_pred265"]).to_csv(
        OUT_DIR / "lifecycle_bucket_diagnostics.csv",
        index=False,
    )

    accepted_rows = []
    for _, row in summary.loc[summary["accepted_for_submission"]].head(3).iterrows():
        name = str(row["candidate"])
        filename = f"stage2_lifecycle_{name}.csv".replace("|", "_").replace(" ", "_").replace(",", "")
        sub = pd.DataFrame({"new_id": test["new_id"].astype(int), "rto": positive(test_preds[name])})
        if len(sub) != 18657 or sub["rto"].isna().any() or (sub["rto"] <= 0).any():
            raise RuntimeError(f"Bad submission format for {filename}")
        sub.to_csv(SUBMISSIONS_DIR / filename, index=False)
        accepted = row.to_dict()
        accepted["filename"] = filename
        accepted_rows.append(accepted)
    pd.DataFrame(accepted_rows).to_csv(OUT_DIR / "submission_candidates.csv", index=False)

    print(f"pred265 OOF MAPE: {pred265_mape:.6f}")
    print("Top lifecycle candidates:")
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
