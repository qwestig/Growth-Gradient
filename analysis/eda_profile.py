from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "analysis"
REPORT_PATH = OUT_DIR / "eda_report.md"
PROFILE_PATH = OUT_DIR / "eda_profile.json"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Нет данных_"
    return df.to_markdown(index=False)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["period"] = pd.to_datetime(
        df["Год"].astype(str) + "-" + df["Месяц"].astype(str).str.zfill(2) + "-01"
    )
    df["period_label"] = df["period"].dt.strftime("%Y-%m")

    id_col = "new_id"
    target = "РТО"
    static_cols = [
        "Дата открытия, категориальный",
        "Торговая площадь, категориальный",
        "Населенный пункт",
        "Регион",
        "Численность населения",
        "Количество домохозяйств",
        "Трафик пеший, в час",
        "Трафик авто, в час",
        "Маркетплейсы, доставки, постаматы (100 м)",
        "Медицинские уч. и аптеки (300 м)",
        "Школы (300 м)",
        "Остановки (300 м)",
        "Продуктовые магазины (500 м)",
        "Пятерочки (500 м)",
        "Количество касс",
        "Флаг алкогольной лицензии",
    ]

    rows, cols = df.shape
    stores = df[id_col].nunique()
    periods = (
        df[["Год", "Месяц", "period_label"]]
        .drop_duplicates()
        .sort_values(["Год", "Месяц"])["period_label"]
        .tolist()
    )
    duplicate_keys = int(df.duplicated([id_col, "Год", "Месяц"]).sum())
    store_month_counts = df.groupby(id_col)["period_label"].nunique()
    month_count_dist = (
        store_month_counts.value_counts()
        .sort_index()
        .rename_axis("months")
        .reset_index(name="stores")
    )

    rows_by_month = (
        df.groupby("period_label")
        .agg(
            stores=(id_col, "nunique"),
            rto_sum=(target, "sum"),
            rto_mean=(target, "mean"),
            rto_median=(target, "median"),
        )
        .reset_index()
    )
    rows_by_month["rto_sum_mln"] = (rows_by_month["rto_sum"] / 1_000_000).round(2)
    rows_by_month["rto_mean_mln"] = (rows_by_month["rto_mean"] / 1_000_000).round(2)
    rows_by_month["rto_median_mln"] = (rows_by_month["rto_median"] / 1_000_000).round(2)
    month_table = rows_by_month[
        ["period_label", "stores", "rto_sum_mln", "rto_mean_mln", "rto_median_mln"]
    ].copy()
    month_table.columns = ["месяц", "магазинов", "сумма РТО, млн", "средний РТО, млн", "медианный РТО, млн"]

    missing = (
        df.drop(columns=["period", "period_label"])
        .isna()
        .sum()
        .rename("missing")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing = missing[missing["missing"] > 0].sort_values("missing", ascending=False)
    missing["missing_pct"] = (missing["missing"] / rows * 100).round(2).astype(str) + "%"
    missing.columns = ["столбец", "пропусков", "доля"]

    object_cols = [c for c in df.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns if c != "period_label"]
    category_summary = pd.DataFrame(
        {
            "категориальный столбец": object_cols,
            "уникальных значений": [df[c].nunique(dropna=True) for c in object_cols],
        }
    ).sort_values("уникальных значений", ascending=False)

    rto = df[target]
    rto_summary = {
        "mean": float(rto.mean()),
        "median": float(rto.median()),
        "std": float(rto.std()),
        "min": float(rto.min()),
        "p05": float(rto.quantile(0.05)),
        "p25": float(rto.quantile(0.25)),
        "p75": float(rto.quantile(0.75)),
        "p95": float(rto.quantile(0.95)),
        "p99": float(rto.quantile(0.99)),
        "max": float(rto.max()),
        "zero_or_negative": int((rto <= 0).sum()),
    }
    rto_table = pd.DataFrame(
        {
            "метрика": ["mean", "median", "p05", "p25", "p75", "p95", "p99", "max"],
            "РТО, млн": [
                round(rto_summary["mean"] / 1_000_000, 2),
                round(rto_summary["median"] / 1_000_000, 2),
                round(rto_summary["p05"] / 1_000_000, 2),
                round(rto_summary["p25"] / 1_000_000, 2),
                round(rto_summary["p75"] / 1_000_000, 2),
                round(rto_summary["p95"] / 1_000_000, 2),
                round(rto_summary["p99"] / 1_000_000, 2),
                round(rto_summary["max"] / 1_000_000, 2),
            ],
        }
    )

    region_summary = (
        df.groupby("Регион")
        .agg(stores=(id_col, "nunique"), rows=(id_col, "size"), mean_rto=(target, "mean"), median_rto=(target, "median"))
        .sort_values("stores", ascending=False)
        .head(15)
        .reset_index()
    )
    region_summary["mean_rto"] = (region_summary["mean_rto"] / 1_000_000).round(2)
    region_summary["median_rto"] = (region_summary["median_rto"] / 1_000_000).round(2)
    region_summary.columns = ["регион", "магазинов", "строк", "средний РТО, млн", "медианный РТО, млн"]

    rto_pivot = df.pivot_table(index=id_col, columns="period_label", values=target, aggfunc="mean")
    lag_pairs = [("2025-02", "2025-01"), ("2025-01", "2024-12"), ("2024-03", "2024-02"), ("2025-02", "2024-02")]
    lag_rows = []
    for later, earlier in lag_pairs:
        pair = rto_pivot[[later, earlier]].dropna()
        lag_rows.append(
            {
                "сравнение": f"{later} vs {earlier}",
                "магазинов": len(pair),
                "Pearson": round(pair[later].corr(pair[earlier], method="pearson"), 3),
                "Spearman": round(pair[later].corr(pair[earlier], method="spearman"), 3),
                "медианная abs %-дельта": f"{(((pair[later] - pair[earlier]).abs() / pair[earlier]).median() * 100):.2f}%",
            }
        )
    lag_summary = pd.DataFrame(lag_rows)

    growth = rto_pivot.sort_index(axis=1).pct_change(axis=1).replace([np.inf, -np.inf], np.nan).stack()
    growth_summary = {
        "p01": float(growth.quantile(0.01)),
        "p05": float(growth.quantile(0.05)),
        "median": float(growth.median()),
        "p95": float(growth.quantile(0.95)),
        "p99": float(growth.quantile(0.99)),
    }

    feature_numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in [id_col, "Год", "Месяц", target]
    ]
    corr_rows = []
    for col in feature_numeric:
        pair = df[[col, target]].dropna()
        corr_rows.append(
            {
                "признак": col,
                "Pearson": round(pair[col].corr(pair[target], method="pearson"), 3),
                "Spearman": round(pair[col].corr(pair[target], method="spearman"), 3),
            }
        )
    corr_summary = pd.DataFrame(corr_rows).sort_values("Spearman", key=lambda s: s.abs(), ascending=False).head(15)

    static_drift = []
    for col in static_cols:
        nunique = df.groupby(id_col)[col].nunique(dropna=False)
        static_drift.append(
            {
                "column": col,
                "stores_with_changes": int((nunique > 1).sum()),
                "max_unique_per_store": int(nunique.max()),
            }
        )
    static_drift = pd.DataFrame(static_drift)

    baseline_context = {}
    for year in [2023, 2024]:
        ratios = (rto_pivot[f"{year}-03"] / rto_pivot[f"{year}-02"]).replace([np.inf, -np.inf], np.nan).dropna()
        baseline_context[str(year)] = {
            "stores": int(len(ratios)),
            "median_mar_feb_ratio": float(ratios.median()),
            "mean_mar_feb_ratio": float(ratios.mean()),
        }

    profile = {
        "shape": {"rows": rows, "columns": cols},
        "stores": stores,
        "periods": periods,
        "duplicate_store_month_keys": duplicate_keys,
        "missing": missing.to_dict(orient="records"),
        "rto_summary": rto_summary,
        "store_month_count_distribution": month_count_dist.to_dict(orient="records"),
        "period_counts": rows_by_month.to_dict(orient="records"),
        "lag_correlations": lag_summary.to_dict(orient="records"),
        "growth_summary": growth_summary,
        "static_feature_drift": static_drift.to_dict(orient="records"),
        "baseline_context": baseline_context,
    }
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# EDA train_2.csv

## Графики

- `analysis/figures/monthly_rto_trend.png` — сезонная динамика среднего и медианного РТО.
- `analysis/figures/rto_distribution.png` — распределение РТО.
- `analysis/figures/numeric_corr_spearman.png` — числовые признаки с максимальной ранговой корреляцией с РТО.

## Что это за таблица

- Строк: **{rows:,}**.
- Столбцов: **{cols}**.
- Уникальных магазинов `new_id`: **{stores:,}**.
- Периоды: **{periods[0]} ... {periods[-1]}**, всего **{len(periods)}** месяцев.
- Дубликатов по ключу `new_id` + `Год` + `Месяц`: **{duplicate_keys}**.
- У каждого магазина ровно **{store_month_counts.min()}** месяцев истории.

## Покрытие по месяцам

{markdown_table(month_table)}

## Целевая переменная РТО

{markdown_table(rto_table)}

- Нулевых или отрицательных значений РТО: **{rto_summary["zero_or_negative"]}**.
- Распределение имеет длинный правый хвост: среднее выше медианы, p99 заметно выше типичного магазина.

## История магазинов

{markdown_table(month_count_dist.rename(columns={"months": "месяцев истории", "stores": "магазинов"}))}

## Пропуски

{markdown_table(missing)}

## Категориальные признаки

{markdown_table(category_summary)}

## Топ регионов по числу магазинов

{markdown_table(region_summary)}

## Насколько прошлый РТО объясняет будущий

{markdown_table(lag_summary)}

Сводка соседних месячных темпов роста по магазинам:

- p01: **{growth_summary["p01"] * 100:.2f}%**
- p05: **{growth_summary["p05"] * 100:.2f}%**
- median: **{growth_summary["median"] * 100:.2f}%**
- p95: **{growth_summary["p95"] * 100:.2f}%**
- p99: **{growth_summary["p99"] * 100:.2f}%**

## Самые сильные числовые корреляции с РТО

{markdown_table(corr_summary)}

## Сезонный контекст для марта

- Median `March / February` в 2023: **{baseline_context["2023"]["median_mar_feb_ratio"]:.4f}**.
- Median `March / February` в 2024: **{baseline_context["2024"]["median_mar_feb_ratio"]:.4f}**.

## Первичные выводы

1. Это сбалансированная панель магазин-месяц без пропусков РТО.
2. Главный сигнал для прогноза — история РТО конкретного магазина.
3. Мартовский прогноз должен учитывать сезонный переход февраль → март.
4. Признаки операционной активности текущего месяца нельзя подставлять в март 2025, если они заранее неизвестны.
5. MAPE требует следить за относительной ошибкой, особенно на магазинах с меньшим оборотом.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)
    print(PROFILE_PATH)


if __name__ == "__main__":
    main()
