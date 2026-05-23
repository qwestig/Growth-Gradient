from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from catboost_residual_diagnostics import CAT_COLS, TARGET, add_base, apply_residual, make_panel, mape
from gru_residual_experiment import period_id, prepare_data


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "segment_bias"
SUBMISSIONS_DIR = ROOT / "submissions"
GRU_OOF_PATH = ROOT / "results" / "gru_residual" / "gru_oof.csv"
BEST_SUBMISSION_PATH = ROOT / "submissions" / "test_gru_alpha070.csv"
ALPHA = 0.70

NUMERIC_SEGMENTS = [
    "best_base",
    "lag1",
    "rolling3_mean",
    "momentum_raw",
    "lag1_to_lag12",
    "lag1_to_lag13",
    "store_mar_feb_ratio",
    "volatility_rel3",
    "volatility_rel6",
]

CATEGORICAL_SEGMENTS = [
    ["Регион"],
    ["Торговая площадь, категориальный"],
    ["Дата открытия, категориальный"],
    ["Регион", "Торговая площадь, категориальный"],
    ["Регион", "Дата открытия, категориальный"],
]

SHRINK_K_GRID = [100, 250, 500, 1000]
BETA_GRID = [0.25, 0.50, 0.75, 1.00]
CLIP_GRID = [0.005, 0.010, 0.015, 0.020, 0.030]


def prepare_march_2024_frame() -> pd.DataFrame:
    data = prepare_data()
    oof = pd.read_csv(GRU_OOF_PATH)
    target = data.target_rows.copy()
    frame = target.merge(oof[["new_id", "fold", "pred_log_gru"]], on="new_id", how="inner")
    frame["pred_gru"] = apply_residual(
        frame["best_base"].to_numpy(dtype=float),
        frame["pred_log_gru"].to_numpy(dtype=float),
        ALPHA,
        None,
    )
    frame["log_error"] = np.log(frame[TARGET] / frame["pred_gru"])
    frame["abs_pct_error_gru"] = np.abs(frame[TARGET] - frame["pred_gru"]) / frame[TARGET] * 100
    return frame.replace([np.inf, -np.inf], np.nan)


def make_march_2025_frame() -> pd.DataFrame:
    data = prepare_data()
    feb_2025 = data.raw[(data.raw["Год"] == 2025) & (data.raw["Месяц"] == 2)].copy()
    mar_2025 = feb_2025.copy()
    mar_2025["Год"] = 2025
    mar_2025["Месяц"] = 3
    mar_2025["period"] = period_id(2025, 3)
    mar_2025[TARGET] = np.nan
    panel = add_base(make_panel(pd.concat([data.raw, mar_2025], ignore_index=True)), data.final_factor)
    pred_rows = panel[(panel["Год"] == 2025) & (panel["Месяц"] == 3)].copy()
    best = pd.read_csv(BEST_SUBMISSION_PATH).rename(columns={"rto": "pred_gru"})
    pred_rows = pred_rows.merge(best, on="new_id", how="inner")
    return pred_rows.sort_values("new_id").reset_index(drop=True).replace([np.inf, -np.inf], np.nan)


def add_numeric_bins(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[float]]]:
    train = train.copy()
    test = test.copy()
    bins_by_col: dict[str, list[float]] = {}
    for col in NUMERIC_SEGMENTS:
        values = train[col].replace([np.inf, -np.inf], np.nan).dropna()
        _, bins = pd.qcut(values, q=5, retbins=True, duplicates="drop")
        bins[0] = -np.inf
        bins[-1] = np.inf
        bins = np.unique(bins)
        labels = [f"{col}_q{i}" for i in range(len(bins) - 1)]
        segment_col = f"{col}_bin"
        train[segment_col] = pd.cut(train[col], bins=bins, labels=labels, include_lowest=True).astype(str)
        test[segment_col] = pd.cut(test[col], bins=bins, labels=labels, include_lowest=True).astype(str)
        train[segment_col] = train[segment_col].replace("nan", "__missing__")
        test[segment_col] = test[segment_col].replace("nan", "__missing__")
        bins_by_col[col] = [float(value) if np.isfinite(value) else str(value) for value in bins]
    return train, test, bins_by_col


def add_interaction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for cols in CATEGORICAL_SEGMENTS:
        name = "__".join(cols)
        out[name] = out[cols].astype(str).agg(" | ".join, axis=1)
    return out


def segment_names() -> list[str]:
    names = [f"{col}_bin" for col in NUMERIC_SEGMENTS]
    names += ["__".join(cols) for cols in CATEGORICAL_SEGMENTS]
    return names


def group_corrections(train: pd.DataFrame, segment: str, shrink_k: float, clip: float) -> pd.Series:
    stats = train.groupby(segment, observed=True)["log_error"].agg(["mean", "count"])
    correction = stats["mean"] * (stats["count"] / (stats["count"] + shrink_k))
    return correction.clip(-clip, clip)


def apply_segment_correction(
    frame: pd.DataFrame,
    corrections: pd.Series,
    segment: str,
    beta: float,
    pred_col: str = "pred_gru",
) -> np.ndarray:
    corr = frame[segment].map(corrections).fillna(0.0).to_numpy(dtype=float)
    return frame[pred_col].to_numpy(dtype=float) * np.exp(beta * corr)


def evaluate_candidate(frame: pd.DataFrame, segment: str, shrink_k: float, beta: float, clip: float) -> float:
    parts = []
    for fold in sorted(frame["fold"].unique()):
        train = frame[frame["fold"] != fold]
        valid = frame[frame["fold"] == fold]
        corrections = group_corrections(train, segment, shrink_k, clip)
        pred = apply_segment_correction(valid, corrections, segment, beta)
        part = valid[["new_id", TARGET]].copy()
        part["pred"] = pred
        parts.append(part)
    oof = pd.concat(parts, ignore_index=True)
    return mape(oof[TARGET], oof["pred"])


def evaluate_all(frame: pd.DataFrame) -> pd.DataFrame:
    base_score = mape(frame[TARGET], frame["pred_gru"])
    rows = []
    for segment in segment_names():
        non_missing_groups = frame[segment].nunique(dropna=True)
        if non_missing_groups < 2:
            continue
        for shrink_k in SHRINK_K_GRID:
            for beta in BETA_GRID:
                for clip in CLIP_GRID:
                    score = evaluate_candidate(frame, segment, shrink_k, beta, clip)
                    rows.append(
                        {
                            "segment": segment,
                            "groups": non_missing_groups,
                            "shrink_k": shrink_k,
                            "beta": beta,
                            "clip": clip,
                            "mape": score,
                            "gain_vs_gru": base_score - score,
                        }
                    )
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def make_segment_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment in segment_names():
        for group, part in frame.groupby(segment, observed=True):
            if len(part) < 50:
                continue
            rows.append(
                {
                    "segment": segment,
                    "group": str(group),
                    "rows": len(part),
                    "gru_mape": mape(part[TARGET], part["pred_gru"]),
                    "mean_log_error": part["log_error"].mean(),
                    "median_log_error": part["log_error"].median(),
                    "mean_error_pct": ((part[TARGET] - part["pred_gru"]) / part[TARGET] * 100).mean(),
                }
            )
    diagnostics = pd.DataFrame(rows)
    diagnostics["abs_mean_log_error"] = diagnostics["mean_log_error"].abs()
    return diagnostics.sort_values(["abs_mean_log_error", "rows"], ascending=[False, False]).reset_index(drop=True)


def make_submission(train: pd.DataFrame, test: pd.DataFrame, best: pd.Series) -> Path:
    corrections = group_corrections(train, str(best["segment"]), float(best["shrink_k"]), float(best["clip"]))
    pred = apply_segment_correction(test, corrections, str(best["segment"]), float(best["beta"]))
    out = test[["new_id"]].copy()
    out["rto"] = np.round(np.maximum(pred, 0), 2)
    segment = str(best["segment"]).replace("__", "_").replace(",", "_").replace(" ", "")
    path = SUBMISSIONS_DIR / (
        f"test_gru_segment_{segment}_k{int(best['shrink_k'])}"
        f"_b{int(float(best['beta']) * 100):03d}_c{int(float(best['clip']) * 1000):03d}.csv"
    )
    out.to_csv(path, index=False)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    train = prepare_march_2024_frame()
    test = make_march_2025_frame()
    train, test, bins_by_col = add_numeric_bins(train, test)
    train = add_interaction_columns(train)
    test = add_interaction_columns(test)

    summary = evaluate_all(train)
    diagnostics = make_segment_diagnostics(train)
    base_score = mape(train[TARGET], train["pred_gru"])
    best = summary.iloc[0]

    submission_path = None
    if float(best["gain_vs_gru"]) > 0:
        submission_path = make_submission(train, test, best)

    summary.to_csv(RESULTS_DIR / "segment_correction_summary.csv", index=False)
    diagnostics.to_csv(RESULTS_DIR / "segment_bias_diagnostics.csv", index=False)
    (RESULTS_DIR / "numeric_bins.json").write_text(json.dumps(bins_by_col, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "base_gru_mape": base_score,
        "best": best.to_dict(),
        "submission": str(submission_path) if submission_path else None,
        "top10": summary.head(10).to_dict(orient="records"),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
