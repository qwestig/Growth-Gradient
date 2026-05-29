from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_strategy_feb_anomaly"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"


@dataclass(frozen=True)
class Spec:
    name: str
    normal_method: str
    trigger_mode: str
    threshold: float
    correction_weight: float
    factor_lambda: float
    factor_method: str
    max_level_ratio: float = 999.0


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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cutoff = w.sum() * 0.5
    return float(v[np.searchsorted(np.cumsum(w), cutoff)])


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
        .groupby("new_id", as_index=False)[static_cols]
        .first()
        .set_index("new_id")
    )
    for col in static.select_dtypes(include=["object", "string"]).columns:
        static[col] = static[col].astype("string").fillna("__NA__")
    meta_oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    meta_test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    return rto, static, meta_oof, meta_test


def target_frame(rto: pd.DataFrame, static: pd.DataFrame, meta: pd.DataFrame, target_year: int, is_train: bool) -> pd.DataFrame:
    frame = pd.DataFrame({"new_id": rto.index.to_numpy(dtype=int)}).set_index("new_id")
    for label, year, month in [
        ("sep_prev", target_year - 1, 9),
        ("oct_prev", target_year - 1, 10),
        ("nov_prev", target_year - 1, 11),
        ("dec_prev", target_year - 1, 12),
        ("jan", target_year, 1),
        ("feb", target_year, 2),
        ("mar_prev_year", target_year - 1, 3),
        ("feb_prev_year", target_year - 1, 2),
    ]:
        frame[label] = rto[period_id(year, month)].astype(float)
    frame["feb_to_jan"] = positive(frame["feb"]) / positive(frame["jan"])
    frame["jan_to_dec"] = positive(frame["jan"]) / positive(frame["dec_prev"])
    frame["feb_to_prev_feb"] = positive(frame["feb"]) / positive(frame["feb_prev_year"])
    frame["rolling3_mean"] = frame[["dec_prev", "jan", "feb"]].mean(axis=1)
    frame["pre2_mean"] = frame[["dec_prev", "jan"]].mean(axis=1)
    frame["pre3_mean"] = frame[["nov_prev", "dec_prev", "jan"]].mean(axis=1)
    frame["pre5_median"] = frame[["sep_prev", "oct_prev", "nov_prev", "dec_prev", "jan"]].median(axis=1)
    frame = frame.join(static, how="left").reset_index()
    if is_train:
        frame = frame.merge(meta[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
        frame["target"] = frame[META_TARGET]
        frame["pred265"] = frame["pred265_oof"]
    else:
        frame = frame.merge(meta[["new_id", "pred265_submission"]], on="new_id", how="left")
        frame["pred265"] = frame["pred265_submission"]
    return frame.replace([np.inf, -np.inf], np.nan)


def ratio_values(rto: pd.DataFrame, source_year: int, target_year: int, calendar_lambda: float) -> np.ndarray:
    ratio = positive(rto[period_id(source_year, 3)]) / positive(rto[period_id(source_year, 2)])
    if source_year == 2023 and target_year == 2024:
        ratio = ratio * ((1.0 - calendar_lambda) + calendar_lambda * (28.0 / 29.0))
    elif source_year == 2024 and target_year == 2025:
        ratio = ratio * ((1.0 - calendar_lambda) + calendar_lambda * (29.0 / 28.0))
    return ratio[np.isfinite(ratio)]


def seasonal_factor(rto: pd.DataFrame, target_year: int, calendar_lambda: float, method: str) -> float:
    values = []
    for source_year in range(2023, target_year):
        values.append(ratio_values(rto, source_year, target_year, calendar_lambda))
    ratios = np.concatenate(values)
    if method == "median":
        return float(np.nanmedian(ratios))
    if method == "mape_weighted_median":
        return weighted_median(ratios, 1.0 / np.clip(ratios, 1e-6, None))
    raise ValueError(method)


def normal_level(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method == "pre2_mean":
        return positive(frame["pre2_mean"])
    if method == "pre3_mean":
        return positive(frame["pre3_mean"])
    if method == "pre5_median":
        return positive(frame["pre5_median"])
    if method == "weighted_dec_jan":
        return positive(0.35 * frame["dec_prev"] + 0.65 * frame["jan"])
    if method == "weighted_pre3":
        return positive(0.15 * frame["nov_prev"] + 0.30 * frame["dec_prev"] + 0.55 * frame["jan"])
    if method == "trend_dec_jan":
        return positive(frame["jan"] * np.clip(frame["jan_to_dec"], 0.90, 1.10))
    if method == "trend_pre3":
        trend = np.log(positive(frame["jan"]) / positive(frame["nov_prev"])) / 2.0
        return positive(frame["jan"] * np.exp(np.clip(trend, -0.08, 0.08)))
    raise ValueError(method)


def trigger_mask(frame: pd.DataFrame, normal: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    log_gap = np.log(positive(frame["feb"]) / positive(normal))
    if mode == "symmetric":
        return np.abs(log_gap) >= threshold
    if mode == "low_only":
        return log_gap <= -threshold
    if mode == "high_only":
        return log_gap >= threshold
    if mode == "low_after_stable":
        return (log_gap <= -threshold) & (np.abs(np.log(positive(frame["jan_to_dec"]))) <= 0.04)
    if mode == "high_after_stable":
        return (log_gap >= threshold) & (np.abs(np.log(positive(frame["jan_to_dec"]))) <= 0.04)
    if mode == "low_not_yoy_decline":
        return (log_gap <= -threshold) & (frame["feb_to_prev_feb"].to_numpy(float) >= 0.98)
    raise ValueError(mode)


def forecast(frame: pd.DataFrame, rto: pd.DataFrame, target_year: int, spec: Spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = seasonal_factor(rto, target_year, spec.factor_lambda, spec.factor_method)
    normal = normal_level(frame, spec.normal_method)
    mask = trigger_mask(frame, normal, spec.trigger_mode, spec.threshold)
    feb = positive(frame["feb"])
    corrected_level = feb.copy()
    corrected_level[mask] = np.exp(
        (1.0 - spec.correction_weight) * np.log(feb[mask])
        + spec.correction_weight * np.log(np.clip(normal[mask], 1.0, None))
    )
    corrected_level[mask] = np.minimum(corrected_level[mask], feb[mask] * spec.max_level_ratio)
    return positive(corrected_level * factor), corrected_level, mask


def make_specs() -> list[Spec]:
    specs = []
    normal_methods = ["pre2_mean", "pre3_mean", "pre5_median", "weighted_dec_jan", "weighted_pre3", "trend_dec_jan", "trend_pre3"]
    trigger_modes = ["symmetric", "low_only", "high_only", "low_after_stable", "high_after_stable", "low_not_yoy_decline"]
    for normal_method in normal_methods:
        for trigger_mode in trigger_modes:
            for threshold in [0.035, 0.055, 0.075, 0.100, 0.130]:
                for weight in [0.15, 0.30, 0.50, 0.75, 1.00]:
                    for factor_lambda in [0.50, 0.55, 0.60, 0.625]:
                        for max_level_ratio in [1.75, 2.50, 3.00, 999.0]:
                            cap_label = "" if max_level_ratio >= 999 else f"_cap{str(max_level_ratio).replace('.', 'p')}"
                            specs.append(
                                Spec(
                                    name=(
                                        f"feb_anom_{normal_method}_{trigger_mode}_"
                                        f"t{str(threshold).replace('.', 'p')}_w{str(weight).replace('.', 'p')}_"
                                        f"cal{str(factor_lambda).replace('.', 'p')}{cap_label}"
                                    ),
                                    normal_method=normal_method,
                                    trigger_mode=trigger_mode,
                                    threshold=threshold,
                                    correction_weight=weight,
                                    factor_lambda=factor_lambda,
                                    factor_method="median",
                                    max_level_ratio=max_level_ratio,
                                )
                            )
    return specs


def evaluate(
    val: pd.DataFrame,
    test: pd.DataFrame,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    mask_val: np.ndarray,
    mask_test: np.ndarray,
    spec: Spec,
    current_mape: float,
    best_clean_mape: float,
) -> dict[str, object]:
    candidate_mape = mape(val["target"], pred_val)
    shift = np.abs(pred_test / positive(test["current_heuristic_pred"]) - 1.0) * 100.0
    return {
        "candidate": spec.name,
        "normal_method": spec.normal_method,
        "trigger_mode": spec.trigger_mode,
        "threshold": spec.threshold,
        "correction_weight": spec.correction_weight,
        "factor_lambda": spec.factor_lambda,
        "max_level_ratio": spec.max_level_ratio,
        "current_heuristic_mape": current_mape,
        "best_known_clean_mape": best_clean_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_current_heuristic": candidate_mape - current_mape,
        "delta_vs_best_known_clean": candidate_mape - best_clean_mape,
        "val_trigger_rate": float(mask_val.mean()),
        "test_trigger_rate": float(mask_test.mean()),
        "test_mean_abs_shift_pct_vs_current": float(np.mean(shift)),
        "test_max_abs_shift_pct_vs_current": float(np.max(shift)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    rto, static, meta_oof, meta_test = load_data()
    val = target_frame(rto, static, meta_oof, 2024, is_train=True)
    test = target_frame(rto, static, meta_test, 2025, is_train=False)

    current_spec = Spec(
        name="current_cal050_median_mom000",
        normal_method="pre2_mean",
        trigger_mode="symmetric",
        threshold=999.0,
        correction_weight=0.0,
        factor_lambda=0.50,
        factor_method="median",
        max_level_ratio=999.0,
    )
    val["current_heuristic_pred"], _, _ = forecast(val, rto, 2024, current_spec)
    test["current_heuristic_pred"], _, _ = forecast(test, rto, 2025, current_spec)
    current_mape = mape(val["target"], val["current_heuristic_pred"])
    best_clean_mape = 3.848401

    rows = []
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    for spec in make_specs():
        pred_val, _, mask_val = forecast(val, rto, 2024, spec)
        pred_test, _, mask_test = forecast(test, rto, 2025, spec)
        val_preds[spec.name] = pred_val
        test_preds[spec.name] = pred_test
        rows.append(evaluate(val, test, pred_val, pred_test, mask_val, mask_test, spec, current_mape, best_clean_mape))

    summary = pd.DataFrame(rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_current"]).reset_index(drop=True)
    summary["beats_current_heuristic"] = summary["delta_vs_current_heuristic"] < 0
    summary["beats_best_known_clean"] = summary["delta_vs_best_known_clean"] < 0
    summary.to_csv(OUT_DIR / "conditional_feb_anomaly_summary.csv", index=False)

    safe_summary = summary[
        (summary["beats_current_heuristic"])
        & (summary["test_max_abs_shift_pct_vs_current"] <= 100.0)
    ]
    candidate_source = safe_summary if len(safe_summary) else summary.loc[summary["beats_current_heuristic"]]

    accepted_rows = []
    for _, row in candidate_source.head(5).iterrows():
        name = str(row["candidate"])
        filename = f"stage2_{name}.csv".replace("|", "_").replace(" ", "_").replace(",", "")
        sub = pd.DataFrame({"new_id": test["new_id"].astype(int), "rto": np.round(positive(test_preds[name]), 2)})
        if len(sub) != 18657 or sub["rto"].isna().any() or (sub["rto"] <= 0).any():
            raise RuntimeError(f"Bad submission format for {filename}")
        sub.to_csv(SUBMISSIONS_DIR / filename, index=False)
        accepted = row.to_dict()
        accepted["filename"] = filename
        accepted_rows.append(accepted)
    pd.DataFrame(accepted_rows).to_csv(OUT_DIR / "submission_candidates.csv", index=False)

    print(f"Current clean heuristic cal050_median_mom000 MAPE: {current_mape:.6f}")
    print(f"Best known clean heuristic local MAPE reference: {best_clean_mape:.6f}")
    print("Top conditional February anomaly candidates:")
    print(
        summary[
            [
                "candidate",
                "candidate_oof_mape",
                "delta_vs_current_heuristic",
                "delta_vs_best_known_clean",
                "max_level_ratio",
                "val_trigger_rate",
                "test_trigger_rate",
                "test_mean_abs_shift_pct_vs_current",
                "test_max_abs_shift_pct_vs_current",
                "beats_current_heuristic",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(f"Safe candidate CSVs beating current clean heuristic: {len(accepted_rows)}")
    if accepted_rows:
        print(pd.DataFrame(accepted_rows)[["filename", "candidate_oof_mape", "delta_vs_current_heuristic"]].to_string(index=False))


if __name__ == "__main__":
    main()
