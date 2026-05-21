from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "analysis"
REPORT_PATH = OUT_DIR / "modeling_probe.md"
PROFILE_PATH = OUT_DIR / "modeling_probe.json"


def mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100)


def score_from_mape(value: float) -> float:
    return 100 * ((100 - min(value, 100)) / 100) ** 2


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
    df = df.sort_values(["new_id", "period"]).reset_index(drop=True)
    rto = df.pivot_table(index="new_id", columns="period_label", values="РТО", aggfunc="mean")

    ratio_2023 = (rto["2023-03"] / rto["2023-02"]).replace([np.inf, -np.inf], np.nan)
    ratio_2024 = (rto["2024-03"] / rto["2024-02"]).replace([np.inf, -np.inf], np.nan)
    validation_factor = ratio_2023.median()

    y_true = rto["2024-03"]
    feb_2024 = rto["2024-02"]
    pred_v1 = feb_2024 * validation_factor
    rolling_3 = (rto["2023-12"] + rto["2024-01"] + rto["2024-02"]) / 3
    momentum = (feb_2024 / rolling_3).replace([np.inf, -np.inf], np.nan).fillna(1).clip(0.95, 1.05)
    pred_momentum = pred_v1 * (momentum ** 0.25)
    pred_store_ratio = feb_2024 * ratio_2023

    comparisons = []
    for method, pred in [
        ("lag1: February equals March", feb_2024),
        ("seasonal global median Mar/Feb 2023", pred_v1),
        ("seasonal store Mar/Feb 2023", pred_store_ratio),
        ("seasonal + conservative momentum", pred_momentum),
        ("same month last year: March 2023", rto["2023-03"]),
    ]:
        method_mape = mape(y_true, pred)
        comparisons.append({"method": method, "MAPE": method_mape, "score": score_from_mape(method_mape)})
    comparison = pd.DataFrame(comparisons).sort_values("MAPE").reset_index(drop=True)
    display_comparison = comparison.copy()
    display_comparison["MAPE"] = display_comparison["MAPE"].round(4)
    display_comparison["score"] = display_comparison["score"].round(4)

    ratio_stability = pd.DataFrame({"ratio_2023": ratio_2023, "ratio_2024": ratio_2024}).dropna()
    ratio_summary = {
        "ratio_2023_median": float(ratio_stability["ratio_2023"].median()),
        "ratio_2024_median": float(ratio_stability["ratio_2024"].median()),
        "ratio_2024_p05": float(ratio_stability["ratio_2024"].quantile(0.05)),
        "ratio_2024_p95": float(ratio_stability["ratio_2024"].quantile(0.95)),
        "ratio_year_to_year_pearson": float(ratio_stability["ratio_2023"].corr(ratio_stability["ratio_2024"], method="pearson")),
        "ratio_year_to_year_spearman": float(ratio_stability["ratio_2023"].corr(ratio_stability["ratio_2024"], method="spearman")),
    }

    final_ratios = pd.concat([ratio_2023, ratio_2024]).dropna()
    feb_2025 = rto["2025-02"]
    forecast_v1 = feb_2025 * final_ratios.median()
    forecast_momentum = forecast_v1 * (
        (feb_2025 / ((rto["2024-12"] + rto["2025-01"] + rto["2025-02"]) / 3))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1)
        .clip(0.95, 1.05)
        ** 0.25
    )
    forecast_summary = pd.DataFrame(
        [
            {
                "method": "seasonal baseline",
                "mean_rto_mln": forecast_v1.mean() / 1_000_000,
                "median_rto_mln": forecast_v1.median() / 1_000_000,
                "p05_rto_mln": forecast_v1.quantile(0.05) / 1_000_000,
                "p95_rto_mln": forecast_v1.quantile(0.95) / 1_000_000,
                "total_rto_mln": forecast_v1.sum() / 1_000_000,
            },
            {
                "method": "seasonal + conservative momentum",
                "mean_rto_mln": forecast_momentum.mean() / 1_000_000,
                "median_rto_mln": forecast_momentum.median() / 1_000_000,
                "p05_rto_mln": forecast_momentum.quantile(0.05) / 1_000_000,
                "p95_rto_mln": forecast_momentum.quantile(0.95) / 1_000_000,
                "total_rto_mln": forecast_momentum.sum() / 1_000_000,
            },
        ]
    )
    forecast_display = forecast_summary.copy()
    for col in forecast_display.columns[1:]:
        forecast_display[col] = forecast_display[col].round(2)

    profile = {
        "validation": comparison.to_dict(orient="records"),
        "ratio_stability": ratio_summary,
        "forecast_summary": forecast_summary.to_dict(orient="records"),
    }
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Modeling probe

## Цель

Проверить простые мартовские baseline на честной имитации финальной задачи:

- считаем, что доступны данные до февраля 2024;
- прогнозируем март 2024;
- сравниваем с истинным мартом 2024 по MAPE.

## Сравнение baseline

{markdown_table(display_comparison)}

## Стабильность мартовского коэффициента

- Median `Mar/Feb 2023`: **{ratio_summary["ratio_2023_median"]:.4f}**.
- Median `Mar/Feb 2024`: **{ratio_summary["ratio_2024_median"]:.4f}**.
- 5-95% диапазон `Mar/Feb 2024`: **{ratio_summary["ratio_2024_p05"]:.4f} ... {ratio_summary["ratio_2024_p95"]:.4f}**.
- Spearman личных коэффициентов магазинов между 2023 и 2024: **{ratio_summary["ratio_year_to_year_spearman"]:.3f}**.

Глобальная мартовская сезонность устойчива, но индивидуальный коэффициент конкретного магазина шумный.

## Как читается momentum-поправка

`momentum = February RTO / mean(December RTO, January RTO, February RTO)`

Поправка ограничена `clip(0.95, 1.05)` и берется в степени `0.25`, поэтому она только слегка сдвигает сезонный прогноз.

## Масштаб финальных прогнозов для марта 2025

{markdown_table(forecast_display)}

## Вывод

1. Сильный интерпретируемый baseline — февральский РТО, умноженный на сезонный мартовский коэффициент.
2. Осторожная momentum-поправка улучшает локальную мартовскую проверку, но остается эвристикой.
3. Любая следующая ML-модель должна сравниваться с этим baseline на временной валидации.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)
    print(PROFILE_PATH)


if __name__ == "__main__":
    main()

