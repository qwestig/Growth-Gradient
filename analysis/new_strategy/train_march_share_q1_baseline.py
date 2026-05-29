from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_strategy_march_share_q1"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"


@dataclass(frozen=True)
class Spec:
    name: str
    method: str
    group_col: str
    store_weight_mode: str
    group_shrink_k: float
    share_clip_low: float
    share_clip_high: float
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
    out["stability_bin"] = qbin(-np.abs(np.log(np.clip(out["feb_to_jan"], 1e-6, None))), 4, "st")
    out["pyat_bin"] = qbin(out["Пятерочки (500 м)"].fillna(0), 4, "pyat")
    out["grocery_bin"] = qbin(out["Продуктовые магазины (500 м)"].fillna(0), 4, "gro")
    out["region_volume"] = out["region"] + "|" + out["volume_bin"].astype(str)
    out["age_volume"] = out["age"] + "|" + out["volume_bin"].astype(str)
    out["region_stability"] = out["region"] + "|" + out["stability_bin"].astype(str)
    out["region_competition"] = out["region"] + "|" + out["pyat_bin"].astype(str) + "|" + out["grocery_bin"].astype(str)
    return out


def target_frame(rto: pd.DataFrame, static: pd.DataFrame, meta: pd.DataFrame, target_year: int, is_train: bool) -> pd.DataFrame:
    frame = pd.DataFrame({"new_id": rto.index.to_numpy(dtype=int)}).set_index("new_id")
    for label, year, month in [
        ("jan", target_year, 1),
        ("feb", target_year, 2),
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


def history_frame(rto: pd.DataFrame, static: pd.DataFrame, target_year: int) -> pd.DataFrame:
    rows = []
    for source_year in range(2023, target_year):
        part = pd.DataFrame({"new_id": rto.index.to_numpy(dtype=int)}).set_index("new_id")
        part["hist_jan"] = rto[period_id(source_year, 1)].astype(float)
        part["hist_feb"] = rto[period_id(source_year, 2)].astype(float)
        part["hist_mar"] = rto[period_id(source_year, 3)].astype(float)
        part["hist_q1"] = part["hist_jan"] + part["hist_feb"] + part["hist_mar"]
        part["hist_share"] = part["hist_mar"] / np.clip(part["hist_q1"], 1.0, None)
        part["source_year"] = source_year
        part = part.join(static, how="left").reset_index()
        # Use historical February as the volume state for historical group keys.
        part["jan"] = part["hist_jan"]
        part["feb"] = part["hist_feb"]
        part = add_group_keys(part)
        rows.append(part)
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)


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


def store_weight(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "cohort_only":
        return pd.Series(0.0, index=frame.index)
    if mode.startswith("const"):
        return pd.Series(float(mode.replace("const", "")), index=frame.index)
    volume_pct = frame["feb"].rank(pct=True).to_numpy(float)
    stability = np.exp(-np.abs(np.log(np.clip(frame["feb_to_jan"].to_numpy(float), 1e-6, None))) / 0.10)
    if mode == "volume_scaled":
        return pd.Series((0.10 + 0.60 * volume_pct).clip(0.05, 0.75), index=frame.index)
    if mode == "stability_scaled":
        return pd.Series((0.15 + 0.55 * stability).clip(0.05, 0.75), index=frame.index)
    if mode == "volume_stability_scaled":
        return pd.Series((0.10 + 0.55 * volume_pct * stability).clip(0.05, 0.75), index=frame.index)
    raise ValueError(mode)


def cohort_share(frame: pd.DataFrame, hist: pd.DataFrame, spec: Spec) -> tuple[pd.Series, pd.Series]:
    work = frame.copy()
    group_size = work.groupby(spec.group_col, observed=True)["new_id"].transform("size")
    work.loc[group_size < spec.min_group_size, spec.group_col] = "fallback_all"
    hist_work = hist.copy()
    known_groups = set(work[spec.group_col].astype(str))
    hist_work.loc[~hist_work[spec.group_col].astype(str).isin(known_groups), spec.group_col] = "fallback_all"

    global_share = float(hist_work["hist_mar"].sum() / max(hist_work["hist_q1"].sum(), 1.0))
    stats = hist_work.groupby(spec.group_col, observed=True).agg(
        mar_sum=("hist_mar", "sum"),
        q1_sum=("hist_q1", "sum"),
        share_median=("hist_share", "median"),
        count=("hist_share", "size"),
    )
    raw = 0.70 * (stats["mar_sum"] / stats["q1_sum"].clip(lower=1.0)) + 0.30 * stats["share_median"]
    shrink = stats["count"] / (stats["count"] + spec.group_shrink_k)
    stats["share"] = global_share + shrink * (raw - global_share)
    mapped = work[spec.group_col].map(stats["share"]).fillna(global_share).astype(float)
    counts = work[spec.group_col].map(stats["count"]).fillna(0)
    return mapped, counts


def store_share_from_history(frame: pd.DataFrame, hist: pd.DataFrame) -> pd.Series:
    stats = hist.groupby("new_id", observed=True)["hist_share"].median()
    return frame["new_id"].map(stats).fillna(float(hist["hist_mar"].sum() / max(hist["hist_q1"].sum(), 1.0))).astype(float)


def q1_growth_factor(frame: pd.DataFrame, window: str, target_year: int) -> pd.Series:
    if window == "janfeb":
        growth = positive(frame["jan"] + frame["feb"]) / positive(frame["jan_prev_year"] + frame["feb_prev_year"])
        if target_year == 2024:
            growth = growth * (59.0 / 60.0)
        elif target_year == 2025:
            growth = growth * (60.0 / 59.0)
        return pd.Series(growth, index=frame.index)
    if window == "feb":
        growth = positive(frame["feb"]) / positive(frame["feb_prev_year"])
        if target_year == 2024:
            growth = growth * (28.0 / 29.0)
        elif target_year == 2025:
            growth = growth * (29.0 / 28.0)
        return pd.Series(growth, index=frame.index)
    raise ValueError(window)


def forecast(frame: pd.DataFrame, hist: pd.DataFrame, spec: Spec, target_year: int) -> tuple[np.ndarray, pd.Series, pd.Series]:
    cshare, counts = cohort_share(frame, hist, spec)
    sshare = store_share_from_history(frame, hist)
    w = store_weight(frame, spec.store_weight_mode)
    share = w * sshare + (1.0 - w) * cshare
    share = pd.Series(share, index=frame.index).clip(spec.share_clip_low, spec.share_clip_high)

    if spec.method == "known_janfeb_to_q1_share":
        pred = positive(frame["jan"] + frame["feb"]) * share / np.clip(1.0 - share, 0.05, None)
    elif spec.method == "q1_growth_janfeb_share":
        q1_prev = positive(frame["jan_prev_year"] + frame["feb_prev_year"] + frame["mar_prev_year"])
        pred = q1_prev * q1_growth_factor(frame, "janfeb", target_year) * share
    elif spec.method == "q1_growth_feb_share":
        q1_prev = positive(frame["jan_prev_year"] + frame["feb_prev_year"] + frame["mar_prev_year"])
        pred = q1_prev * q1_growth_factor(frame, "feb", target_year) * share
    elif spec.method == "blend_known_and_growth":
        pred_known = positive(frame["jan"] + frame["feb"]) * share / np.clip(1.0 - share, 0.05, None)
        q1_prev = positive(frame["jan_prev_year"] + frame["feb_prev_year"] + frame["mar_prev_year"])
        pred_growth = q1_prev * q1_growth_factor(frame, "janfeb", target_year) * share
        pred = 0.60 * pred_known + 0.40 * pred_growth
    else:
        raise ValueError(spec.method)
    return positive(pred), share, counts


def make_specs() -> list[Spec]:
    group_cols = [
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
        "region_stability",
        "region_competition",
    ]
    methods = ["known_janfeb_to_q1_share", "q1_growth_janfeb_share", "q1_growth_feb_share", "blend_known_and_growth"]
    weights = ["cohort_only", "const0.20", "const0.50", "const0.80", "volume_scaled", "stability_scaled", "volume_stability_scaled"]
    specs = []
    for method in methods:
        for group_col in group_cols:
            for weight in weights:
                for shrink_k in [20.0, 100.0, 300.0]:
                    min_group_size = 1 if group_col == "global" else (40 if group_col in {"region", "area", "age"} else 80)
                    specs.append(
                        Spec(
                            name=f"q1share_{method}_{group_col}_{weight}_k{int(shrink_k)}",
                            method=method,
                            group_col=group_col,
                            store_weight_mode=weight,
                            group_shrink_k=shrink_k,
                            share_clip_low=0.22,
                            share_clip_high=0.46,
                            min_group_size=min_group_size,
                        )
                    )
    return specs


def evaluate(val: pd.DataFrame, test: pd.DataFrame, pred_val: np.ndarray, pred_test: np.ndarray, spec: Spec | None, current_mape: float, current_best_known_mape: float) -> dict[str, object]:
    candidate_mape = mape(val["target"], pred_val)
    shift = np.abs(pred_test / positive(test["current_heuristic_pred"]) - 1.0) * 100.0
    return {
        "candidate": spec.name if spec else "ensemble",
        "method": spec.method if spec else "",
        "group_col": spec.group_col if spec else "ensemble",
        "store_weight_mode": spec.store_weight_mode if spec else "",
        "group_shrink_k": spec.group_shrink_k if spec else np.nan,
        "current_heuristic_mape": current_mape,
        "current_best_known_clean_mape": current_best_known_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_current_heuristic": candidate_mape - current_mape,
        "delta_vs_best_known_clean": candidate_mape - current_best_known_mape,
        "test_mean_abs_shift_pct_vs_current": float(np.mean(shift)),
        "test_max_abs_shift_pct_vs_current": float(np.max(shift)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    rto, static, meta_oof, meta_test = load_data()
    val = target_frame(rto, static, meta_oof, 2024, is_train=True)
    test = target_frame(rto, static, meta_test, 2025, is_train=False)
    hist_val = history_frame(rto, static, 2024)
    hist_test = history_frame(rto, static, 2025)

    val["current_heuristic_pred"] = current_heuristic_pred(val, rto, 2024, calendar_lambda=0.50)
    test["current_heuristic_pred"] = current_heuristic_pred(test, rto, 2025, calendar_lambda=0.50)
    current_mape = mape(val["target"], val["current_heuristic_pred"])
    current_best_known_clean_mape = 3.848401

    rows = []
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    for spec in make_specs():
        pred_val, _, _ = forecast(val, hist_val, spec, 2024)
        pred_test, _, _ = forecast(test, hist_test, spec, 2025)
        val_preds[spec.name] = pred_val
        test_preds[spec.name] = pred_test
        rows.append(evaluate(val, test, pred_val, pred_test, spec, current_mape, current_best_known_clean_mape))

    summary = pd.DataFrame(rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_current"]).reset_index(drop=True)
    top_names = summary.head(5)["candidate"].tolist()
    for n in [2, 3, 5]:
        name = f"q1share_ensemble_top{n}"
        pred_val = np.mean([val_preds[x] for x in top_names[:n]], axis=0)
        pred_test = np.mean([test_preds[x] for x in top_names[:n]], axis=0)
        val_preds[name] = pred_val
        test_preds[name] = pred_test
        row = evaluate(val, test, pred_val, pred_test, None, current_mape, current_best_known_clean_mape)
        row["candidate"] = name
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

    summary = summary.sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_current"]).reset_index(drop=True)
    summary["beats_current_heuristic"] = summary["delta_vs_current_heuristic"] < 0
    summary["beats_best_known_clean"] = summary["delta_vs_best_known_clean"] < 0
    summary.to_csv(OUT_DIR / "march_share_q1_summary.csv", index=False)

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
    print(f"Best known clean heuristic local MAPE reference: {current_best_known_clean_mape:.6f}")
    print("Top March-share-of-Q1 candidates:")
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
