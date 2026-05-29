from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_strategy_share_of_market"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"


@dataclass(frozen=True)
class Spec:
    name: str
    group_col: str
    total_method: str
    share_method: str
    min_group_size: int
    shrink_k: float
    trend_clip: float


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


def leap_ratio_adjustment(source_year: int, target_year: int) -> float:
    if source_year == 2023 and target_year == 2024:
        return 28.0 / 29.0
    if source_year == 2024 and target_year != 2024:
        return 29.0 / 28.0
    return 1.0


def janfeb_adjustment(source_year: int, target_year: int) -> float:
    if source_year == 2023 and target_year == 2024:
        return 59.0 / 60.0
    if source_year == 2024 and target_year != 2024:
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
        "Продуктовые магазины (500 м)",
        "Пятерочки (500 м)",
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


def add_group_keys(static: pd.DataFrame) -> pd.DataFrame:
    out = static.copy()
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
    out["pyat_bin"] = pd.qcut(out["Пятерочки (500 м)"].fillna(0).rank(method="first"), 4, labels=[f"pyat{i}" for i in range(4)]).astype("string")
    out["grocery_bin"] = pd.qcut(
        out["Продуктовые магазины (500 м)"].fillna(0).rank(method="first"),
        4,
        labels=[f"gro{i}" for i in range(4)],
    ).astype("string")
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
        ("nov_prev", target_year - 1, 11),
        ("oct_prev", target_year - 1, 10),
        ("sep_prev", target_year - 1, 9),
        ("mar_prev_year", target_year - 1, 3),
        ("jan_prev_year", target_year - 1, 1),
        ("feb_prev_year", target_year - 1, 2),
    ]:
        frame[label] = rto[period_id(year, month)].astype(float)
    frame = frame.join(add_group_keys(static), how="left").reset_index()
    if is_train:
        frame = frame.merge(meta[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
        frame["target"] = frame[META_TARGET]
        frame["pred265"] = frame["pred265_oof"]
    else:
        frame = frame.merge(meta[["new_id", "pred265_submission"]], on="new_id", how="left")
        frame["pred265"] = frame["pred265_submission"]
    return frame.replace([np.inf, -np.inf], np.nan)


def grouped_sum(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    return frame.groupby(group_col, observed=True)[value_col].transform("sum").clip(lower=1.0)


def group_total_forecast(frame: pd.DataFrame, group_col: str, target_year: int, method: str, shrink_k: float) -> pd.Series:
    group_size = frame.groupby(group_col, observed=True)["new_id"].transform("size")
    global_feb_ratio = frame["mar_prev_year"].sum() / max(frame["feb_prev_year"].sum(), 1.0)
    global_janfeb_growth = (frame["jan"].sum() + frame["feb"].sum()) / max(frame["jan_prev_year"].sum() + frame["feb_prev_year"].sum(), 1.0)

    if method == "seasonal_feb_total":
        raw_ratio = grouped_sum(frame, group_col, "mar_prev_year") / grouped_sum(frame, group_col, "feb_prev_year")
        raw_ratio = raw_ratio * leap_ratio_adjustment(target_year - 1, target_year)
        global_ratio = global_feb_ratio * leap_ratio_adjustment(target_year - 1, target_year)
        shrink = group_size / (group_size + shrink_k)
        ratio = np.exp(np.log(global_ratio) + shrink * np.log(np.clip(raw_ratio, 1e-6, None) / global_ratio))
        return grouped_sum(frame, group_col, "feb") * ratio

    if method == "prev_march_janfeb_growth":
        group_growth = (grouped_sum(frame, group_col, "jan") + grouped_sum(frame, group_col, "feb")) / (
            grouped_sum(frame, group_col, "jan_prev_year") + grouped_sum(frame, group_col, "feb_prev_year")
        )
        group_growth = group_growth * janfeb_adjustment(target_year - 1, target_year)
        global_growth = global_janfeb_growth * janfeb_adjustment(target_year - 1, target_year)
        shrink = group_size / (group_size + shrink_k)
        growth = np.exp(np.log(global_growth) + shrink * np.log(np.clip(group_growth, 1e-6, None) / global_growth))
        return grouped_sum(frame, group_col, "mar_prev_year") * growth

    if method == "blend_total":
        a = group_total_forecast(frame, group_col, target_year, "seasonal_feb_total", shrink_k)
        b = group_total_forecast(frame, group_col, target_year, "prev_march_janfeb_growth", shrink_k)
        return 0.65 * a + 0.35 * b

    raise ValueError(method)


def store_share(frame: pd.DataFrame, group_col: str, method: str, trend_clip: float) -> pd.Series:
    def share(col: str) -> pd.Series:
        return frame[col] / grouped_sum(frame, group_col, col)

    if method == "feb_share":
        raw = share("feb")
    elif method == "janfeb_share":
        raw = (frame["jan"] + frame["feb"]) / (grouped_sum(frame, group_col, "jan") + grouped_sum(frame, group_col, "feb"))
    elif method == "last3_weighted_share":
        value = 0.20 * frame["dec_prev"] + 0.30 * frame["jan"] + 0.50 * frame["feb"]
        denom = frame.groupby(group_col, observed=True).apply(
            lambda g: 0.20 * g["dec_prev"].sum() + 0.30 * g["jan"].sum() + 0.50 * g["feb"].sum()
        )
        raw = value / frame[group_col].map(denom).clip(lower=1.0)
    elif method == "share_trend_6m":
        shares = pd.concat(
            [
                share("sep_prev").rename("s0"),
                share("oct_prev").rename("s1"),
                share("nov_prev").rename("s2"),
                share("dec_prev").rename("s3"),
                share("jan").rename("s4"),
                share("feb").rename("s5"),
            ],
            axis=1,
        ).clip(lower=1e-12)
        x = np.arange(6, dtype=float)
        centered = x - x.mean()
        log_shares = np.log(shares.to_numpy(dtype=float))
        slope = (log_shares * centered).sum(axis=1) / (centered**2).sum()
        raw = share("feb") * np.exp(np.clip(slope, -trend_clip, trend_clip))
    elif method == "current_prior_march_blend":
        current = share("feb")
        prior_march = frame["mar_prev_year"] / grouped_sum(frame, group_col, "mar_prev_year")
        raw = 0.80 * current + 0.20 * prior_march
    else:
        raise ValueError(method)

    raw = pd.Series(np.asarray(raw, dtype=float), index=frame.index).clip(lower=1e-12)
    norm = raw.groupby(frame[group_col], observed=True).transform("sum").clip(lower=1e-12)
    return raw / norm


def forecast(frame: pd.DataFrame, spec: Spec, target_year: int) -> np.ndarray:
    group_size = frame.groupby(spec.group_col, observed=True)["new_id"].transform("size")
    use_group = group_size >= spec.min_group_size
    work = frame.copy()
    if not use_group.all():
        work.loc[~use_group, spec.group_col] = "fallback_all"

    total = group_total_forecast(work, spec.group_col, target_year, spec.total_method, spec.shrink_k)
    share = store_share(work, spec.group_col, spec.share_method, spec.trend_clip)
    return positive(total * share)


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
        "region_city_big",
        "region_competition",
    ]
    total_methods = ["seasonal_feb_total", "prev_march_janfeb_growth", "blend_total"]
    share_methods = ["feb_share", "janfeb_share", "last3_weighted_share", "share_trend_6m", "current_prior_march_blend"]
    specs = []
    for group_col in group_cols:
        for total_method in total_methods:
            for share_method in share_methods:
                for shrink_k in [20.0, 80.0, 250.0]:
                    min_group_size = 1 if group_col == "global" else (40 if group_col in {"region", "area", "age"} else 80)
                    specs.append(
                        Spec(
                            name=f"som_{group_col}_{total_method}_{share_method}_k{int(shrink_k)}",
                            group_col=group_col,
                            total_method=total_method,
                            share_method=share_method,
                            min_group_size=min_group_size,
                            shrink_k=shrink_k,
                            trend_clip=0.06,
                        )
                    )
    return specs


def evaluate(val: pd.DataFrame, test: pd.DataFrame, pred_val: np.ndarray, pred_test: np.ndarray, spec: Spec | None, name: str, pred265_mape: float) -> dict[str, object]:
    candidate_mape = mape(val["target"], pred_val)
    shift = np.abs(pred_test / positive(test["pred265"]) - 1.0) * 100.0
    return {
        "candidate": name,
        "group_col": spec.group_col if spec else "ensemble",
        "total_method": spec.total_method if spec else "",
        "share_method": spec.share_method if spec else "",
        "shrink_k": spec.shrink_k if spec else np.nan,
        "pred265_oof_mape": pred265_mape,
        "candidate_oof_mape": candidate_mape,
        "delta_vs_pred265": candidate_mape - pred265_mape,
        "test_mean_abs_shift_pct_vs_pred265": float(np.mean(shift)),
        "test_max_abs_shift_pct_vs_pred265": float(np.max(shift)),
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
    for spec in make_specs():
        pred_val = forecast(val, spec, 2024)
        pred_test = forecast(test, spec, 2025)
        val_preds[spec.name] = pred_val
        test_preds[spec.name] = pred_test
        rows.append(evaluate(val, test, pred_val, pred_test, spec, spec.name, pred265_mape))

    summary = pd.DataFrame(rows).sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"]).reset_index(drop=True)
    top_names = summary.head(7)["candidate"].tolist()
    for n in [2, 3, 5, 7]:
        name = f"som_ensemble_top{n}"
        pred_val = np.mean([val_preds[x] for x in top_names[:n]], axis=0)
        pred_test = np.mean([test_preds[x] for x in top_names[:n]], axis=0)
        val_preds[name] = pred_val
        test_preds[name] = pred_test
        summary = pd.concat([summary, pd.DataFrame([evaluate(val, test, pred_val, pred_test, None, name, pred265_mape)])])

    summary = summary.sort_values(["candidate_oof_mape", "test_mean_abs_shift_pct_vs_pred265"]).reset_index(drop=True)
    summary["accepted_for_submission"] = (
        (summary["delta_vs_pred265"] <= -0.015)
        & (summary["test_mean_abs_shift_pct_vs_pred265"] <= 3.0)
        & (summary["test_max_abs_shift_pct_vs_pred265"] <= 40.0)
    )
    summary.to_csv(OUT_DIR / "share_of_market_summary.csv", index=False)

    best_name = str(summary.iloc[0]["candidate"])
    bucket_rows = []
    y = val["target"].to_numpy(float)
    pred265 = val["pred265"].to_numpy(float)
    best_val = val_preds[best_name]
    for col in ["region", "city_big", "area", "age", "region_area"]:
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
    pd.DataFrame(bucket_rows).sort_values(["delta_vs_pred265"]).to_csv(OUT_DIR / "share_bucket_diagnostics.csv", index=False)

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
    print("Top share-of-market candidates:")
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
