from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis"
FIG_DIR = OUT_DIR / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = ROOT / "train_2.csv"
EDA_PROFILE = OUT_DIR / "eda_profile.json"
MODELING_PROFILE = OUT_DIR / "modeling_probe.json"


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    df = pd.read_csv(DATA_PATH)
    df["period_label"] = df["Год"].astype(str) + "-" + df["Месяц"].astype(str).str.zfill(2)
    monthly = (
        df.groupby("period_label")
        .agg(mean_rto=("РТО", "mean"), median_rto=("РТО", "median"))
        .reset_index()
    )
    monthly["mean_rto_mln"] = monthly["mean_rto"] / 1_000_000
    monthly["median_rto_mln"] = monthly["median_rto"] / 1_000_000

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["period_label"], monthly["mean_rto_mln"], marker="o", label="Mean")
    ax.plot(monthly["period_label"], monthly["median_rto_mln"], marker="o", label="Median")
    ax.set_title("Monthly RTO per store")
    ax.set_xlabel("Month")
    ax.set_ylabel("RTO, mln")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    save(fig, "monthly_rto_trend.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["РТО"] / 1_000_000, bins=80, color="#4C72B0", alpha=0.8)
    ax.set_xlim(0, (df["РТО"] / 1_000_000).quantile(0.995))
    ax.set_title("RTO distribution")
    ax.set_xlabel("RTO, mln")
    ax.set_ylabel("Rows")
    save(fig, "rto_distribution.png")

    eda = json.loads(EDA_PROFILE.read_text(encoding="utf-8"))
    corr = pd.DataFrame(eda["lag_correlations"])
    model = json.loads(MODELING_PROFILE.read_text(encoding="utf-8"))
    validation = pd.DataFrame(model["validation"]).sort_values("MAPE", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot = validation.sort_values("MAPE", ascending=False)
    ax.barh(plot["method"], plot["MAPE"], color="#4C72B0")
    ax.set_title("March 2024 validation MAPE")
    ax.set_xlabel("MAPE, %")
    ax.set_ylabel("")
    save(fig, "baseline_mape.png")

    numeric = pd.read_csv(DATA_PATH).select_dtypes(include="number")
    corr_rows = []
    for col in numeric.columns:
        if col not in ["new_id", "Год", "Месяц", "РТО"]:
            corr_rows.append({"feature": col, "spearman": numeric[col].corr(numeric["РТО"], method="spearman")})
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman", key=lambda s: s.abs()).tail(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(corr_df["feature"], corr_df["spearman"], color="#55A868")
    ax.set_title("Top numeric feature Spearman correlation with RTO")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("")
    save(fig, "numeric_corr_spearman.png")

    print(FIG_DIR)


if __name__ == "__main__":
    main()
