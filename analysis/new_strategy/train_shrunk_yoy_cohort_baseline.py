from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_strategy_shrunk_yoy"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"


@dataclass(frozen=True)
class Spec:
    name: str
    growth_window: str
    calendar_mode: str
    group_col: str
    store_weight_mode: str
    blend_space: str
    group_shrink_k: float
    outlier_clip: float
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


def growth_day_adjustment(target_year: int, window: str) -> float:
    if window == "feb":
        if target_year == 2024:
            return 28.0 / 29.0
        if target_year == 2025:
            return 29.0 / 28.0
    if window == "janfeb":
        if target_year == 2024:
            return 59.0 / 60.0
        if target_year == 2025:
            return 60.0 / 59.0
    return 1.0


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
    out["region_city_big"] = out["region"] + "|" + out["city_big"]
    out["volume_bin"] = qbin(out["feb"], 5, "vol")
    out["prev_volume_bin"] = qbin(out["feb_prev_year"], 5, "pvol")
    out["feb_to_jan"] = positive(out["feb"]) / positive(out["jan"])
    out["current_momentum_bin"] = pd.Series(
        pd.cut(
            np.log(np.clip(out["feb_to_jan"].to_numpy(float), 1e-6, None)),
            bins=[-np.inf, -0.10, -0.04, 0.04, 0.10, np.inf],
            labels=["sharp_drop", "soft_drop", "flat", "soft_growth", "sharp_growth"],
        ),
        index=out.index,
    ).astype("string").fillna("flat")
    out["volatility_rel3"] = out[["dec_prev", "jan", "feb"]].std(axis=1) / np.clip(out[["dec_prev", "jan", "feb"]].mean(axis=1), 1.0, None)
    out["volatility_bin"] = qbin(out["volatility_rel3"], 4, "vr")
    out["growth_stability_proxy"] = np.abs(np.log(np.clip(out["feb_to_jan"].to_numpy(float), 1e-6, None)))
    out["stability_bin"] = qbin(-out["growth_stability_proxy"], 4, "st")
    out["pyat_bin"] = qbin(out["Пятерочки (500 м)"].fillna(0), 4, "pyat")
    out["grocery_bin"] = qbin(out["Продуктовые магазины (500 м)"].fillna(0), 4, "gro")
    out["region_volume"] = out["region"] + "|" + out["volume_bin"].astype(str)
    out["age_volume"] = out["age"] + "|" + out["volume_bin"].astype(str)
    out["area_volume"] = out["area"] + "|" + out["volume_bin"].astype(str)
    out["region_stability"] = out["region"] + "|" + out["stability_bin"].astype(str)
    out["region_competition"] = out["region"] + "|" + out["pyat_bin"].astype(str) + "|" + out["grocery_bin"].astype(str)
    return out


def target_frame(
    rto: pd.DataFrame,
    static: pd.DataFrame,
    meta: pd.DataFrame,
    target_year: int,
    is_train: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame({"new_id": rto.index.to_numpy(dtype=int)}).set_index("new_id")
    for label, year, month in [
        ("jan", target_year, 1),
        ("feb", target_year, 2),
        ("dec_prev", target_year - 1, 12),
        ("mar_prev_year", target_year - 1, 3),
        ("jan_prev_year", target_year - 1, 1),
        ("feb_prev_year", target_year - 1, 2),
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


def growth_values(frame: pd.DataFrame, window: str, calendar_factor: float) -> pd.Series:
    if window == "feb":
        growth = positive(frame["feb"]) / positive(frame["feb_prev_year"])
    elif window == "janfeb":
        growth = positive(frame["jan"] + frame["feb"]) / positive(frame["jan_prev_year"] + frame["feb_prev_year"])
    else:
        raise ValueError(window)
    return pd.Series(growth * calendar_factor, index=frame.index)


def cohort_growth(frame: pd.DataFrame, group_col: str, window: str, calendar_factor: float, shrink_k: float, min_group_size: int) -> tuple[pd.Series, pd.Series]:
    work = frame.copy()
    group_size = work.groupby(group_col, observed=True)["new_id"].transform("size")
    work.loc[group_size < min_group_size, group_col] = "fallback_all"

    if window == "feb":
        current = work["feb"]
        previous = work["feb_prev_year"]
    elif window == "janfeb":
        current = work["jan"] + work["feb"]
        previous = work["jan_prev_year"] + work["feb_prev_year"]
    else:
        raise ValueError(window)

    global_growth = float(current.sum() / max(previous.sum(), 1.0) * calendar_factor)
    group_current = current.groupby(work[group_col], observed=True).transform("sum")
    group_previous = previous.groupby(work[group_col], observed=True).transform("sum").clip(lower=1.0)
    raw_group_growth = group_current / group_previous * calendar_factor
    group_size = work.groupby(group_col, observed=True)["new_id"].transform("size")
    shrink = group_size / (group_size + shrink_k)
    shrunk = np.exp(np.log(global_growth) + shrink * np.log(np.clip(raw_group_growth, 1e-6, None) / global_growth))
    return pd.Series(shrunk, index=frame.index), group_size


def store_weight(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "cohort_only":
        return pd.Series(0.0, index=frame.index)
    if mode.startswith("const"):
        return pd.Series(float(mode.replace("const", "")), index=frame.index)
    volume_pct = frame["feb"].rank(pct=True)
    stability = np.exp(-np.abs(np.log(np.clip(frame["feb_to_jan"].to_numpy(float), 1e-6, None))) / 0.10)
    if mode == "volume_scaled":
        return (0.10 + 0.60 * volume_pct).clip(0.05, 0.75)
    if mode == "stability_scaled":
        return pd.Series((0.15 + 0.55 * stability).clip(0.05, 0.75), index=frame.index)
    if mode == "volume_stability_scaled":
        return pd.Series((0.10 + 0.55 * volume_pct.to_numpy(float) * stability).clip(0.05, 0.75), index=frame.index)
    raise ValueError(mode)


def forecast(frame: pd.DataFrame, spec: Spec, target_year: int) -> tuple[np.ndarray, pd.Series, pd.Series, pd.Series]:
    calendar_factor = growth_day_adjustment(target_year, spec.growth_window) if spec.calendar_mode == "day_adjusted" else 1.0
    store_g = growth_values(frame, spec.growth_window, calendar_factor)
    cohort_g, group_size = cohort_growth(
        frame,
        spec.group_col,
        spec.growth_window,
        calendar_factor,
        spec.group_shrink_k,
        spec.min_group_size,
    )
    log_delta = np.log(np.clip(store_g, 1e-6, None) / np.clip(cohort_g, 1e-6, None))
    clipped_store_g = cohort_g * np.exp(np.clip(log_delta, -spec.outlier_clip, spec.outlier_clip))
    w = store_weight(frame, spec.store_weight_mode)
    if spec.blend_space == "linear":
        growth = w * clipped_store_g + (1.0 - w) * cohort_g
    elif spec.blend_space == "log":
        growth = np.exp(w * np.log(np.clip(clipped_store_g, 1e-6, None)) + (1.0 - w) * np.log(np.clip(cohort_g, 1e-6, None)))
    else:
        raise ValueError(spec.blend_space)
    pred = positive(frame["mar_prev_year"] * growth)
    return pred, store_g, cohort_g, group_size


def make_specs(target_year: int) -> list[Spec]:
    group_cols = [
        "global",
        "region",
        "city_big",
        "area",
        "age",
        "region_area",
        "region_age",
        "area_age",
        "region_city_big",
        "region_volume",
        "age_volume",
        "area_volume",
        "region_stability",
        "region_competition",
    ]
    weight_modes = [
        "cohort_only",
        "const0.15",
        "const0.30",
        "const0.50",
        "const0.70",
        "const1.00",
        "volume_scaled",
        "stability_scaled",
        "volume_stability_scaled",
    ]
    specs = []
    for window in ["feb", "janfeb"]:
        for calendar_mode in ["raw", "day_adjusted"]:
            for group_col in group_cols:
                for weight_mode in weight_modes:
                    for blend_space in ["linear", "log"]:
                        for shrink_k in [20.0, 100.0, 300.0]:
                            for clip in [0.04, 0.08, 0.15, 0.30, 0.60]:
                                min_group_size = 1 if group_col == "global" else (40 if group_col in {"region", "area", "age"} else 80)
                                specs.append(
                                    Spec(
                                        name=(
                                            f"shrunk_yoy_y{target_year}_{window}_{calendar_mode}_{group_col}_"
                                            f"{weight_mode}_{blend_space}_k{int(shrink_k)}_c{str(clip).replace('.', 'p')}"
                                        ),
                                        growth_window=window,
                                        calendar_mode=calendar_mode,
                                        group_col=group_col,
                                        store_weight_mode=weight_mode,
                                        blend_space=blend_space,
                                        group_shrink_k=shrink_k,
                                        outlier_clip=clip,
                                        min_group_size=min_group_size,
                                    )
                                )
    return specs


def forecast_for_year(frame: pd.DataFrame, spec: Spec, target_year: int) -> tuple[np.ndarray, pd.Series, pd.Series, pd.Series]:
    return forecast(frame, spec, target_year)


def evaluate(
    val: pd.DataFrame,
    test: pd.DataFrame,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    spec: Spec | None,
    name: str,
    pred265_mape: float,
    group_size: pd.Series | None = None,
) -> dict[str, object]:
    candidate_mape = mape(val["target"], pred_val)
    shift = np.abs(pred_test / positive(test["pred265"]) - 1.0) * 100.0
    return {
        "candidate": name,
        "growth_window": spec.growth_window if spec else "",
        "calendar_mode": spec.calendar_mode if spec else "",
        "group_col": spec.group_col if spec else "ensemble",
        "store_weight_mode": spec.store_weight_mode if spec else "",
        "blend_space": spec.blend_space if spec else "",
        "group_shrink_k": spec.group_shrink_k if spec else np.nan,
        "outlier_clip": spec.outlier_clip if spec else np.nan,
        "pred265_oof_mape": pred265_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_pred265": candidate_mape - pred265_mape,
        "test_mean_abs_shift_pct_vs_pred265": float(np.mean(shift)),
        "test_max_abs_shift_pct_vs_pred265": float(np.max(shift)),
        "avg_group_size": float(group_size.mean()) if group_size is not None else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    rto, static, meta_oof, meta_test = load_data()
    val = target_frame(rto, static, meta_oof, target_year=2024, is_train=True)
    test = target_frame(rto, static, meta_test, target_year=2025, is_train=False)
    pred265_mape = mape(val["target"], val["pred265"])

    rows = []
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    for spec_val in make_specs(2024):
        spec_test = Spec(
            name=spec_val.name.replace("y2024", "y2025"),
            growth_window=spec_val.growth_window,
            calendar_mode=spec_val.calendar_mode,
            group_col=spec_val.group_col,
            store_weight_mode=spec_val.store_weight_mode,
            blend_space=spec_val.blend_space,
            group_shrink_k=spec_val.group_shrink_k,
            outlier_clip=spec_val.outlier_clip,
            min_group_size=spec_val.min_group_size,
        )
        pred_val, _, _, group_size = forecast_for_year(val, spec_val, 2024)
        pred_test, _, _, _ = forecast_for_year(test, spec_test, 2025)
        name = spec_val.name.replace("y2024_", "")
        val_preds[name] = pred_val
        test_preds[name] = pred_test
        rows.append(evaluate(val, test, pred_val, pred_test, spec_val, name, pred265_mape, group_size))

    summary = pd.DataFrame(rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"]).reset_index(drop=True)
    top_names = summary.head(7)["candidate"].tolist()
    for n in [2, 3, 5, 7]:
        name = f"shrunk_yoy_ensemble_top{n}"
        pred_val = np.mean([val_preds[x] for x in top_names[:n]], axis=0)
        pred_test = np.mean([test_preds[x] for x in top_names[:n]], axis=0)
        val_preds[name] = pred_val
        test_preds[name] = pred_test
        summary = pd.concat([summary, pd.DataFrame([evaluate(val, test, pred_val, pred_test, None, name, pred265_mape)])])

    summary = summary.sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"]).reset_index(drop=True)
    summary["accepted_for_submission"] = (
        (summary["delta_vs_pred265"] <= -0.015)
        & (summary["test_mean_abs_shift_pct_vs_pred265"] <= 5.0)
        & (summary["test_max_abs_shift_pct_vs_pred265"] <= 120.0)
    )
    summary.to_csv(OUT_DIR / "shrunk_yoy_summary.csv", index=False)

    best_name = str(summary.iloc[0]["candidate"])
    bucket_rows = []
    y = val["target"].to_numpy(float)
    pred265 = val["pred265"].to_numpy(float)
    best_val = val_preds[best_name]
    for col in ["region", "city_big", "area", "age", "volume_bin", "stability_bin", "region_area"]:
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
    pd.DataFrame(bucket_rows).sort_values("delta_vs_pred265").to_csv(OUT_DIR / "shrunk_yoy_bucket_diagnostics.csv", index=False)

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
    pd.DataFrame(accepted_rows).to_csv(OUT_DIR / "submission_candidates.csv", index=False)

    print(f"pred265 OOF MAPE: {pred265_mape:.6f}")
    print("Top shrunk YoY cohort candidates:")
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
        .head(15)
        .to_string(index=False)
    )
    print(f"Accepted submission candidates: {len(accepted_rows)}")


if __name__ == "__main__":
    main()
