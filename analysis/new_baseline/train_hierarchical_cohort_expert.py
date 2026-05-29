from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_baseline_hierarchy"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"

MAJOR_REGIONS = [
    "Москва г",
    "Московская обл",
    "Краснодарский край",
    "Татарстан Респ",
    "Ростовская обл",
]


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
    mask = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return float(np.mean(np.abs((p[mask] - y[mask]) / y[mask])) * 100.0)


def safe_log_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.log(np.clip(num.astype(float), 1.0, None) / np.clip(den.astype(float), 1.0, None))


def leap_adjust_ratio(ratio: np.ndarray, source_year: int, target_year: int, target_month: int) -> np.ndarray:
    # March/February ratio is mechanically affected by February length.
    if target_month != 3:
        return ratio
    if source_year == 2023 and target_year == 2024:
        return ratio * (28.0 / 29.0)
    if source_year == 2024 and target_year != 2024:
        return ratio * (29.0 / 28.0)
    return ratio


def load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        .groupby("new_id", as_index=False)[[c for c in static_cols if c in raw.columns]]
        .first()
        .set_index("new_id")
    )
    meta_oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    meta_test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    return raw, rto, static, meta_oof, meta_test


def frame_for_target(
    rto: pd.DataFrame,
    static: pd.DataFrame,
    meta: pd.DataFrame,
    target_year: int,
    target_month: int,
    is_train: bool,
) -> pd.DataFrame:
    target_period = period_id(target_year, target_month)
    ids = rto.index.to_numpy(dtype=int)
    frame = pd.DataFrame({"new_id": ids})
    frame["target_period"] = target_period
    frame["lag1"] = rto[target_period - 1].to_numpy(dtype=float)
    frame["lag2"] = rto[target_period - 2].to_numpy(dtype=float)
    frame["lag3"] = rto[target_period - 3].to_numpy(dtype=float)
    frame["lag12"] = rto[target_period - 12].to_numpy(dtype=float)
    frame["lag13"] = rto[target_period - 13].to_numpy(dtype=float)
    frame["rolling3_mean"] = frame[["lag1", "lag2", "lag3"]].mean(axis=1)
    frame["volume_rank"] = pd.qcut(frame["lag1"].rank(method="first"), 10, labels=False)
    frame["volatility_rel3"] = frame[["lag1", "lag2", "lag3"]].std(axis=1) / np.maximum(frame["rolling3_mean"], 1.0)
    frame["volatility_bin"] = pd.qcut(frame["volatility_rel3"].rank(method="first"), 5, labels=False).astype(str)
    frame["volume_bin"] = pd.qcut(frame["lag1"].rank(method="first"), 5, labels=False).astype(str)
    frame["store_mar_feb_ratio_prev"] = frame["lag12"] / np.maximum(frame["lag13"], 1.0)
    frame["ratio_deviation_abs"] = np.abs(np.log(np.clip(frame["store_mar_feb_ratio_prev"], 1e-6, None)))
    frame["stability_bin"] = pd.qcut(frame["ratio_deviation_abs"].rank(method="first"), 5, labels=False).astype(str)
    frame = frame.join(static, on="new_id", how="left")
    for col in ["Регион", "Населенный пункт", "Торговая площадь, категориальный", "Дата открытия, категориальный"]:
        if col in frame.columns:
            frame[col] = frame[col].astype("string").fillna("__NA__")
    if is_train:
        frame = frame.merge(meta[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
        frame["anchor"] = frame["pred265_oof"]
        frame["target"] = frame[META_TARGET]
    else:
        frame = frame.merge(meta[["new_id", "pred265_submission"]], on="new_id", how="left")
        frame["anchor"] = frame["pred265_submission"]
    return frame.replace([np.inf, -np.inf], np.nan)


def ratio_history(
    rto: pd.DataFrame,
    static: pd.DataFrame,
    target_year: int,
    target_month: int,
) -> pd.DataFrame:
    parts = []
    for source_year in range(2023, target_year):
        cur = period_id(source_year, target_month)
        prev = cur - 1
        if cur not in rto.columns or prev not in rto.columns:
            continue
        ratio = rto[cur].to_numpy(dtype=float) / np.maximum(rto[prev].to_numpy(dtype=float), 1.0)
        ratio = leap_adjust_ratio(ratio, source_year, target_year, target_month)
        part = pd.DataFrame(
            {
                "new_id": rto.index.to_numpy(dtype=int),
                "source_year": source_year,
                "hist_feb": rto[prev].to_numpy(dtype=float),
                "hist_mar": rto[cur].to_numpy(dtype=float),
                "ratio": ratio,
            }
        ).join(static, on="new_id", how="left")
        parts.append(part)
    if not parts:
        raise RuntimeError(f"No ratio history for {target_year}-{target_month:02d}")
    out = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio"])
    for col in ["Регион", "Населенный пункт", "Торговая площадь, категориальный", "Дата открытия, категориальный"]:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__")
    out["volume_bin"] = pd.qcut(out["hist_feb"].rank(method="first"), 5, labels=False).astype(str)
    out["store_ratio_absdev"] = np.abs(np.log(np.clip(out["ratio"], 1e-6, None)))
    out["stability_bin"] = pd.qcut(out["store_ratio_absdev"].rank(method="first"), 5, labels=False).astype(str)
    return out


def add_group_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    counts_city = out["Населенный пункт"].value_counts(dropna=False)
    out["city_big"] = out["Населенный пункт"].where(out["Населенный пункт"].map(counts_city).fillna(0) >= 80, "__SMALL_CITY__")
    out["region_area"] = out["Регион"].astype(str) + "|" + out["Торговая площадь, категориальный"].astype(str)
    out["region_age"] = out["Регион"].astype(str) + "|" + out["Дата открытия, категориальный"].astype(str)
    out["area_age"] = out["Торговая площадь, категориальный"].astype(str) + "|" + out["Дата открытия, категориальный"].astype(str)
    out["vol_volume"] = out["volatility_bin"].astype(str) + "|" + out["volume_bin"].astype(str)
    return out


def add_history_group_keys(hist: pd.DataFrame, target_like: pd.DataFrame) -> pd.DataFrame:
    out = hist.copy()
    city_big_values = set(target_like.loc[target_like["city_big"].ne("__SMALL_CITY__"), "city_big"].astype(str))
    out["city_big"] = out["Населенный пункт"].where(out["Населенный пункт"].astype(str).isin(city_big_values), "__SMALL_CITY__")
    out["region_area"] = out["Регион"].astype(str) + "|" + out["Торговая площадь, категориальный"].astype(str)
    out["region_age"] = out["Регион"].astype(str) + "|" + out["Дата открытия, категориальный"].astype(str)
    out["area_age"] = out["Торговая площадь, категориальный"].astype(str) + "|" + out["Дата открытия, категориальный"].astype(str)
    # For historical rows this is an endogenous stability/volume proxy from the source window.
    out["vol_volume"] = out["stability_bin"].astype(str) + "|" + out["volume_bin"].astype(str)
    return out


def group_ratio_prediction(
    target: pd.DataFrame,
    hist: pd.DataFrame,
    group_col: str,
    shrink_k: float,
    min_count: int,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    global_ratio = float(hist["hist_mar"].sum() / max(hist["hist_feb"].sum(), 1.0))
    stats = hist.groupby(group_col, observed=True).agg(
        ratio_sum=("hist_mar", "sum"),
        feb_sum=("hist_feb", "sum"),
        ratio_median=("ratio", "median"),
        ratio_count=("ratio", "size"),
        ratio_iqr=("ratio", lambda s: float(np.nanquantile(s, 0.75) - np.nanquantile(s, 0.25))),
    )
    agg_ratio = stats["ratio_sum"] / stats["feb_sum"].clip(lower=1.0)
    raw_ratio = 0.65 * agg_ratio + 0.35 * stats["ratio_median"]
    shrink = stats["ratio_count"] / (stats["ratio_count"] + shrink_k)
    shrunk_ratio = np.exp(np.log(global_ratio) + shrink * np.log(raw_ratio / global_ratio))
    ratio_map = shrunk_ratio.where(stats["ratio_count"] >= min_count, global_ratio)
    mapped = target[group_col].map(ratio_map).fillna(global_ratio).to_numpy(dtype=float)
    counts = target[group_col].map(stats["ratio_count"]).fillna(0)
    iqr = target[group_col].map(stats["ratio_iqr"]).fillna(float(stats["ratio_iqr"].median()))
    return target["lag1"].to_numpy(dtype=float) * mapped, counts, iqr


def group_share_prediction(
    rto: pd.DataFrame,
    target: pd.DataFrame,
    hist: pd.DataFrame,
    group_col: str,
    target_year: int,
    target_month: int,
    shrink_k: float,
    min_count: int,
) -> tuple[np.ndarray, pd.Series]:
    target_period = period_id(target_year, target_month)
    global_ratio = float(hist["hist_mar"].sum() / max(hist["hist_feb"].sum(), 1.0))
    stats = hist.groupby(group_col, observed=True).agg(
        mar_sum=("hist_mar", "sum"),
        feb_sum=("hist_feb", "sum"),
        count=("ratio", "size"),
    )
    group_ratio = stats["mar_sum"] / stats["feb_sum"].clip(lower=1.0)
    shrink = stats["count"] / (stats["count"] + shrink_k)
    group_ratio = np.exp(np.log(global_ratio) + shrink * np.log(group_ratio / global_ratio))
    group_ratio = group_ratio.where(stats["count"] >= min_count, global_ratio)

    lag_periods = [target_period - 3, target_period - 2, target_period - 1]
    share_frame = target[["new_id", group_col]].copy()
    for period in lag_periods:
        values = rto.loc[share_frame["new_id"].to_numpy(dtype=int), period].to_numpy(dtype=float)
        share_frame[f"v_{period}"] = values
        denom = share_frame.groupby(group_col, observed=True)[f"v_{period}"].transform("sum").clip(lower=1.0)
        share_frame[f"s_{period}"] = share_frame[f"v_{period}"] / denom
    share_cols = [f"s_{period}" for period in lag_periods]
    share_frame["share_mean"] = share_frame[share_cols].mean(axis=1)
    share_frame["share_std"] = share_frame[share_cols].std(axis=1).fillna(0.0)
    share_frame["share_cv"] = share_frame["share_std"] / share_frame["share_mean"].clip(lower=1e-9)

    current_group_total = target.groupby(group_col, observed=True)["lag1"].transform("sum").clip(lower=1.0)
    pred_group_total = current_group_total * target[group_col].map(group_ratio).fillna(global_ratio).to_numpy(dtype=float)
    pred = pred_group_total * share_frame["share_mean"].to_numpy(dtype=float)
    return pred, share_frame["share_cv"]


def build_experts(
    rto: pd.DataFrame,
    frame: pd.DataFrame,
    hist: pd.DataFrame,
    target_year: int,
    target_month: int,
) -> pd.DataFrame:
    out = add_group_keys(frame)
    hist = add_history_group_keys(hist, out)
    group_specs = [
        ("Регион", 250.0, 60),
        ("city_big", 200.0, 80),
        ("Торговая площадь, категориальный", 400.0, 100),
        ("Дата открытия, категориальный", 400.0, 100),
        ("region_area", 180.0, 40),
        ("region_age", 180.0, 40),
        ("area_age", 300.0, 80),
        ("volume_bin", 300.0, 100),
        ("stability_bin", 300.0, 100),
        ("vol_volume", 180.0, 60),
    ]
    ratio_preds = []
    share_preds = []
    gate_count = []
    gate_iqr = []
    gate_share_cv = []
    for group_col, shrink_k, min_count in group_specs:
        ratio_pred, counts, iqr = group_ratio_prediction(out, hist, group_col, shrink_k, min_count)
        share_pred, share_cv = group_share_prediction(rto, out, hist, group_col, target_year, target_month, shrink_k, min_count)
        out[f"ratio_pred__{group_col}"] = ratio_pred
        out[f"share_pred__{group_col}"] = share_pred
        ratio_preds.append(ratio_pred)
        share_preds.append(share_pred)
        gate_count.append(counts.to_numpy(dtype=float))
        gate_iqr.append(iqr.to_numpy(dtype=float))
        gate_share_cv.append(share_cv.to_numpy(dtype=float))

    out["cohort_ratio_pred"] = np.nanmedian(np.vstack(ratio_preds), axis=0)
    out["cohort_share_pred"] = np.nanmedian(np.vstack(share_preds), axis=0)
    out["mixed_hierarchy_pred"] = np.sqrt(np.clip(out["cohort_ratio_pred"], 1.0, None) * np.clip(out["cohort_share_pred"], 1.0, None))
    out["hier_group_count"] = np.nanmedian(np.vstack(gate_count), axis=0)
    out["hier_ratio_iqr"] = np.nanmedian(np.vstack(gate_iqr), axis=0)
    out["hier_share_cv"] = np.nanmedian(np.vstack(gate_share_cv), axis=0)
    return out.replace([np.inf, -np.inf], np.nan)


def stress_bucket_summary(train: pd.DataFrame, candidate: str, pred: np.ndarray) -> pd.DataFrame:
    y = train["target"].to_numpy(dtype=float)
    base = train["anchor"].to_numpy(dtype=float)
    shift = np.abs(np.log(np.clip(pred, 1.0, None) / np.clip(base, 1.0, None)))
    buckets = {
        "high_volume": train["lag1"] >= train["lag1"].quantile(0.80),
        "low_volume": train["lag1"] <= train["lag1"].quantile(0.20),
        "high_volatility": train["volatility_rel3"] >= train["volatility_rel3"].quantile(0.80),
        "unstable_mar_feb": train["ratio_deviation_abs"] >= train["ratio_deviation_abs"].quantile(0.80),
        "high_shift_vs_pred265": shift >= np.quantile(shift, 0.80),
    }
    rows = []
    for name, mask in buckets.items():
        mask_arr = np.asarray(mask, dtype=bool)
        base_m = mape(y[mask_arr], base[mask_arr])
        pred_m = mape(y[mask_arr], pred[mask_arr])
        rows.append({"candidate": candidate, "bucket": name, "n": int(mask_arr.sum()), "base_mape": base_m, "candidate_mape": pred_m, "delta": pred_m - base_m, "improved": pred_m < base_m})
    return pd.DataFrame(rows)


def major_region_summary(train: pd.DataFrame, candidate: str, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for region in MAJOR_REGIONS:
        mask = train["Регион"].astype(str).eq(region).to_numpy()
        if not mask.any():
            continue
        base_m = mape(train.loc[mask, "target"], train.loc[mask, "anchor"])
        pred_m = mape(train.loc[mask, "target"], pred[mask])
        rows.append({"candidate": candidate, "region": region, "n": int(mask.sum()), "base_mape": base_m, "candidate_mape": pred_m, "delta": pred_m - base_m, "not_worse": pred_m <= base_m})
    return pd.DataFrame(rows)


def evaluate_candidates(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    y = train["target"].to_numpy(dtype=float)
    base = train["anchor"].to_numpy(dtype=float)
    base_test = test["anchor"].to_numpy(dtype=float)
    base_m = mape(y, base)
    rows = []
    stress_parts = []
    region_parts = []
    test_preds: dict[str, np.ndarray] = {}
    expert_cols = ["cohort_ratio_pred", "cohort_share_pred", "mixed_hierarchy_pred"]
    for expert_col in expert_cols:
        train_delta = safe_log_ratio(train[expert_col].to_numpy(dtype=float), base)
        test_delta = safe_log_ratio(test[expert_col].to_numpy(dtype=float), base_test)
        stable_train = (
            (train["hier_group_count"].fillna(0) >= 80)
            & (train["hier_ratio_iqr"].fillna(999) <= train["hier_ratio_iqr"].quantile(0.75))
            & (train["hier_share_cv"].fillna(999) <= train["hier_share_cv"].quantile(0.75))
        ).to_numpy(dtype=bool)
        stable_test = (
            (test["hier_group_count"].fillna(0) >= 80)
            & (test["hier_ratio_iqr"].fillna(999) <= train["hier_ratio_iqr"].quantile(0.75))
            & (test["hier_share_cv"].fillna(999) <= train["hier_share_cv"].quantile(0.75))
        ).to_numpy(dtype=bool)
        for alpha in [0.15, 0.25, 0.35, 0.50, 0.70, 1.00]:
            for clip in [0.003, 0.005, 0.008, 0.012, 0.015]:
                for gate_name, gate_train, gate_test in [
                    ("all", np.ones(len(train), dtype=bool), np.ones(len(test), dtype=bool)),
                    ("stable", stable_train, stable_test),
                    ("large_shift_stable", stable_train & (np.abs(train_delta) >= np.quantile(np.abs(train_delta), 0.50)), stable_test & (np.abs(test_delta) >= np.quantile(np.abs(test_delta), 0.50))),
                ]:
                    corr_train = np.where(gate_train, np.clip(alpha * train_delta, -clip, clip), 0.0)
                    corr_test = np.where(gate_test, np.clip(alpha * test_delta, -clip, clip), 0.0)
                    pred = base * np.exp(corr_train)
                    pred_test = base_test * np.exp(corr_test)
                    name = f"{expert_col}__a{alpha:.2f}__c{clip:.3f}__{gate_name}"
                    test_preds[name] = pred_test
                    stress = stress_bucket_summary(train, name, pred)
                    regions = major_region_summary(train, name, pred)
                    stress_ok = int(stress["improved"].sum())
                    regions_ok = bool(regions["not_worse"].all()) if len(regions) else True
                    ratio = pred_test / base_test - 1.0
                    rows.append(
                        {
                            "candidate": name,
                            "expert": expert_col,
                            "alpha": alpha,
                            "clip": clip,
                            "gate": gate_name,
                            "oof_mape": mape(y, pred),
                            "delta_vs_265_oof": mape(y, pred) - base_m,
                            "stress_buckets_improved": stress_ok,
                            "major_regions_not_worse": regions_ok,
                            "train_applied_share": float(gate_train.mean()),
                            "test_applied_share": float(gate_test.mean()),
                            "test_mean_abs_shift_pct": float(np.mean(np.abs(ratio)) * 100.0),
                            "test_max_abs_shift_pct": float(np.max(np.abs(ratio)) * 100.0),
                        }
                    )
                    stress_parts.append(stress)
                    region_parts.append(regions)
    summary = pd.DataFrame(rows).sort_values(["delta_vs_265_oof", "test_mean_abs_shift_pct"]).reset_index(drop=True)
    stress_summary = pd.concat(stress_parts, ignore_index=True)
    region_summary = pd.concat(region_parts, ignore_index=True)
    return summary, stress_summary, region_summary, test_preds


def add_reference_comparison() -> pd.DataFrame:
    rows = [{"reference": "pred265_oof", "oof_mape": 3.314574426593587, "delta_vs_265_oof": 0.0}]
    seasonal = ROOT / "results" / "new_baseline_multimonth_gru_seasonal" / "gated_submission_candidates.csv"
    if seasonal.exists():
        df = pd.read_csv(seasonal)
        if not df.empty:
            best = df.sort_values("delta_vs_265_oof").iloc[0]
            rows.append({"reference": "seasonal_multimonth_gru_best", "oof_mape": float(best["oof_mape"]), "delta_vs_265_oof": float(best["delta_vs_265_oof"])})
    ml_gate = ROOT / "results" / "new_baseline_multimonth_gru_seasonal" / "ml_gate_submission_candidates.csv"
    if ml_gate.exists():
        df = pd.read_csv(ml_gate)
        if not df.empty:
            best = df.sort_values("delta_vs_265_oof").iloc[0]
            rows.append({"reference": "ml_gate_candidate_best", "oof_mape": float(best["oof_mape"]), "delta_vs_265_oof": float(best["delta_vs_265_oof"])})
    return pd.DataFrame(rows)


def write_submissions(test: pd.DataFrame, summary: pd.DataFrame, test_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    accepted = summary[
        (summary["delta_vs_265_oof"] <= -0.015)
        & (summary["stress_buckets_improved"] >= 3)
        & (summary["test_mean_abs_shift_pct"] <= 0.25)
        & (summary["test_max_abs_shift_pct"] <= 1.50)
        & (summary["major_regions_not_worse"])
    ].head(8)
    columns = [
        "candidate_id",
        "file",
        *summary.columns.tolist(),
    ]
    rows = []
    for candidate_id, (_, row) in enumerate(accepted.iterrows(), start=400):
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(test_preds[row["candidate"]], 0), 2)
        token = (
            str(row["candidate"])
            .replace("__", "_")
            .replace(".", "p")
            .replace(",", "")
            .replace(" ", "_")
        )
        file_name = f"stage2_candidate_{candidate_id}_hierarchy_{token}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append({"candidate_id": candidate_id, "file": file_name, **row.to_dict()})
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _raw, rto, static, meta_oof, meta_test = load_base()

    train = frame_for_target(rto, static, meta_oof, 2024, 3, is_train=True)
    test = frame_for_target(rto, static, meta_test, 2025, 3, is_train=False)
    train_hist = ratio_history(rto, static, 2024, 3)
    test_hist = ratio_history(rto, static, 2025, 3)
    train = build_experts(rto, train, train_hist, 2024, 3)
    test = build_experts(rto, test, test_hist, 2025, 3)

    summary, stress_summary, region_summary, test_preds = evaluate_candidates(train, test)
    submissions = write_submissions(test, summary, test_preds)
    reference = add_reference_comparison()

    summary.to_csv(OUT_DIR / "cohort_expert_summary.csv", index=False)
    stress_summary.to_csv(OUT_DIR / "stress_bucket_summary.csv", index=False)
    region_summary.to_csv(OUT_DIR / "major_region_summary.csv", index=False)
    submissions.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    reference.to_csv(OUT_DIR / "reference_comparison.csv", index=False)

    print("Reference comparison:")
    print(reference.to_string(index=False))
    print("\nTop hierarchy candidates:")
    print(summary.head(25).to_string(index=False))
    print("\nAccepted submissions:")
    if submissions.empty:
        print("No candidates met acceptance criteria.")
    else:
        print(submissions.to_string(index=False))


if __name__ == "__main__":
    main()
