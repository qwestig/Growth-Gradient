from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
OUT_DIR = ROOT / "results" / "new_baseline_similarity"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "РТО"
META_TARGET = "target_rto"
SEQ_LEN = 6
MAJOR_REGIONS = ["Москва г", "Московская обл", "Краснодарский край", "Татарстан Респ", "Ростовская обл"]


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
    for col in static.columns:
        static[col] = static[col].astype("string").fillna("__NA__")
    meta_oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    meta_test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    return rto, static, meta_oof, meta_test


def seasonal_factor(rto: pd.DataFrame, target_period: int) -> float:
    target_year, target_month = ym_from_period(target_period)
    ratios = []
    for source_year in range(2023, target_year):
        source_period = period_id(source_year, target_month)
        if source_period not in rto.columns or source_period - 1 not in rto.columns:
            continue
        ratio = rto[source_period].to_numpy(float) / np.clip(rto[source_period - 1].to_numpy(float), 1.0, None)
        if target_month == 3 and source_year == 2023 and target_year == 2024:
            ratio = ratio * (28.0 / 29.0)
        elif target_month == 3 and source_year == 2024 and target_year != 2024:
            ratio = ratio * (29.0 / 28.0)
        ratios.append(ratio[np.isfinite(ratio)])
    if not ratios:
        return 1.0
    return float(np.nanmedian(np.concatenate(ratios)))


def base_features(rto: pd.DataFrame, static: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = []
    aux_rows = []
    for target_period, group in rows.groupby("target_period", sort=False):
        ids = group["new_id"].to_numpy(dtype=int)
        hist_periods = list(range(int(target_period) - SEQ_LEN, int(target_period)))
        values = rto.loc[ids, hist_periods].to_numpy(dtype=float)
        logv = np.log(np.clip(values, 1.0, None))
        rel = logv - logv[:, [-1]]
        growth = np.diff(logv, axis=1)
        rolling3 = values[:, -3:].mean(axis=1)
        vol3 = values[:, -3:].std(axis=1) / np.clip(rolling3, 1.0, None)
        target_month = ym_from_period(int(target_period))[1]
        month_sin = np.full(len(ids), np.sin(2 * np.pi * target_month / 12))
        month_cos = np.full(len(ids), np.cos(2 * np.pi * target_month / 12))
        numeric = np.column_stack(
            [
                rel,
                growth,
                safe_log_ratio(values[:, -1], rolling3),
                safe_log_ratio(values[:, -1], values[:, -2]),
                safe_log_ratio(values[:, -2], values[:, -3]),
                vol3,
                np.log(np.clip(values[:, -1], 1.0, None)),
                month_sin,
                month_cos,
            ]
        )
        features.append(numeric)
        aux_rows.append(
            pd.DataFrame(
                {
                    "new_id": ids,
                    "target_period": int(target_period),
                    "lag1": values[:, -1],
                    "rolling3_mean": rolling3,
                    "volatility_rel3": vol3,
                }
            )
        )
    aux = pd.concat(aux_rows, ignore_index=True)
    aux = aux.join(static, on="new_id", how="left")
    aux["volume_bin"] = pd.qcut(aux["lag1"].rank(method="first"), 5, labels=False).astype(str)
    aux["volatility_bin"] = pd.qcut(aux["volatility_rel3"].rank(method="first"), 5, labels=False).astype(str)
    cat = pd.get_dummies(aux[["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный"]], dtype=float)
    x = np.concatenate([np.concatenate(features, axis=0), cat.to_numpy(dtype=float)], axis=1)
    return aux, np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sample_rows(rto: pd.DataFrame, target_periods: list[int]) -> pd.DataFrame:
    parts = []
    all_ids = rto.index.to_numpy(dtype=int)
    for target_period in target_periods:
        needed = list(range(target_period - SEQ_LEN, target_period + 1))
        ok = rto.reindex(columns=needed).notna().all(axis=1)
        parts.append(pd.DataFrame({"new_id": all_ids[ok.to_numpy()], "target_period": target_period}))
    return pd.concat(parts, ignore_index=True)


def query_rows(rto: pd.DataFrame, target_period: int) -> pd.DataFrame:
    needed = list(range(target_period - SEQ_LEN, target_period))
    ok = rto.reindex(columns=needed).notna().all(axis=1)
    return pd.DataFrame({"new_id": rto.index.to_numpy(dtype=int)[ok.to_numpy()], "target_period": target_period})


def reference_targets(rto: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    target = []
    lag1 = []
    seasonal = []
    for target_period, group in out.groupby("target_period", sort=False):
        ids = group["new_id"].to_numpy(dtype=int)
        target.append(rto.loc[ids, int(target_period)].to_numpy(float))
        lag1.append(rto.loc[ids, int(target_period) - 1].to_numpy(float))
        seasonal.append(np.full(len(ids), seasonal_factor(rto, int(target_period))))
    out["target"] = np.concatenate(target)
    out["lag1"] = np.concatenate(lag1)
    out["seasonal_factor"] = np.concatenate(seasonal)
    out["log_growth_target"] = safe_log_ratio(out["target"].to_numpy(float), out["lag1"].to_numpy(float))
    out["log_seasonal_residual_target"] = np.log(
        np.clip(out["target"].to_numpy(float), 1.0, None)
        / np.clip(out["lag1"].to_numpy(float) * out["seasonal_factor"].to_numpy(float), 1.0, None)
    )
    return out


def knn_predict(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_query: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_ref_scaled = scaler.fit_transform(x_ref)
    x_query_scaled = scaler.transform(x_query)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(x_ref_scaled)
    dist, idx = nn.kneighbors(x_query_scaled, return_distance=True)
    weights = 1.0 / np.square(dist + 1e-3)
    weights = weights / weights.sum(axis=1, keepdims=True)
    pred = np.sum(y_ref[idx] * weights, axis=1)
    neighbor_std = np.sqrt(np.sum(np.square(y_ref[idx] - pred[:, None]) * weights, axis=1))
    mean_dist = dist.mean(axis=1)
    return pred, neighbor_std, mean_dist


def build_similarity_predictions(
    rto: pd.DataFrame,
    static: pd.DataFrame,
    query_period: int,
    reference_end_period: int,
    meta: pd.DataFrame,
    is_train: bool,
) -> pd.DataFrame:
    ref_periods = list(range(period_id(2023, 7), reference_end_period + 1))
    ref_rows = sample_rows(rto, ref_periods)
    query = query_rows(rto, query_period)
    ref_aux, x_ref = base_features(rto, static, ref_rows)
    query_aux, x_query = base_features(rto, static, query)
    ref_target = reference_targets(rto, ref_aux[["new_id", "target_period"]])
    out = query_aux.copy()
    sf = seasonal_factor(rto, query_period)
    for k in [20, 50, 100, 200]:
        pred_growth, std_growth, dist_growth = knn_predict(x_ref, ref_target["log_growth_target"].to_numpy(float), x_query, k)
        pred_resid, std_resid, dist_resid = knn_predict(x_ref, ref_target["log_seasonal_residual_target"].to_numpy(float), x_query, k)
        out[f"knn_growth_pred_k{k}"] = out["lag1"].to_numpy(float) * np.exp(pred_growth)
        out[f"knn_seasonal_pred_k{k}"] = out["lag1"].to_numpy(float) * sf * np.exp(pred_resid)
        out[f"knn_growth_std_k{k}"] = std_growth
        out[f"knn_seasonal_std_k{k}"] = std_resid
        out[f"knn_dist_k{k}"] = 0.5 * (dist_growth + dist_resid)
    if is_train:
        out = out.merge(meta[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
        out["target"] = out[META_TARGET]
        out["anchor"] = out["pred265_oof"]
    else:
        out = out.merge(meta[["new_id", "pred265_submission"]], on="new_id", how="left")
        out["anchor"] = out["pred265_submission"]
    return out.replace([np.inf, -np.inf], np.nan)


def stress_bucket_summary(train: pd.DataFrame, candidate: str, pred: np.ndarray) -> pd.DataFrame:
    y = train["target"].to_numpy(float)
    base = train["anchor"].to_numpy(float)
    shift = np.abs(np.log(np.clip(pred, 1.0, None) / np.clip(base, 1.0, None)))
    buckets = {
        "high_volume": train["lag1"] >= train["lag1"].quantile(0.80),
        "low_volume": train["lag1"] <= train["lag1"].quantile(0.20),
        "high_volatility": train["volatility_rel3"] >= train["volatility_rel3"].quantile(0.80),
        "low_neighbor_std": train.filter(like="knn_seasonal_std").median(axis=1) <= train.filter(like="knn_seasonal_std").median(axis=1).quantile(0.40),
        "high_shift_vs_pred265": shift >= np.quantile(shift, 0.80),
    }
    rows = []
    for bucket, mask in buckets.items():
        mask_arr = np.asarray(mask, dtype=bool)
        base_m = mape(y[mask_arr], base[mask_arr])
        pred_m = mape(y[mask_arr], pred[mask_arr])
        rows.append({"candidate": candidate, "bucket": bucket, "n": int(mask_arr.sum()), "base_mape": base_m, "candidate_mape": pred_m, "delta": pred_m - base_m, "improved": pred_m < base_m})
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


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    y = train["target"].to_numpy(float)
    base = train["anchor"].to_numpy(float)
    base_test = test["anchor"].to_numpy(float)
    base_m = mape(y, base)
    pred_cols = [c for c in train.columns if c.startswith("knn_") and "_pred_k" in c]
    rows = []
    stress_parts = []
    region_parts = []
    test_preds: dict[str, np.ndarray] = {}
    for pred_col in pred_cols:
        k = pred_col.rsplit("k", 1)[-1]
        std_col = f"knn_seasonal_std_k{k}" if "seasonal" in pred_col else f"knn_growth_std_k{k}"
        dist_col = f"knn_dist_k{k}"
        delta = safe_log_ratio(train[pred_col].to_numpy(float), base)
        delta_test = safe_log_ratio(test[pred_col].to_numpy(float), base_test)
        low_std_train = train[std_col].le(train[std_col].quantile(0.50)).to_numpy()
        low_std_test = test[std_col].le(test[std_col].quantile(0.50)).to_numpy()
        close_train = train[dist_col].le(train[dist_col].quantile(0.50)).to_numpy()
        close_test = test[dist_col].le(test[dist_col].quantile(0.50)).to_numpy()
        for alpha in [0.10, 0.15, 0.20, 0.30, 0.40, 0.60]:
            for clip in [0.003, 0.005, 0.008, 0.012, 0.018]:
                for gate, mask_train, mask_test in [
                    ("all", np.ones(len(train), dtype=bool), np.ones(len(test), dtype=bool)),
                    ("low_std", low_std_train, low_std_test),
                    ("close", close_train, close_test),
                    ("close_low_std", close_train & low_std_train, close_test & low_std_test),
                    ("large_shift_close", close_train & (np.abs(delta) >= np.quantile(np.abs(delta), 0.50)), close_test & (np.abs(delta_test) >= np.quantile(np.abs(delta_test), 0.50))),
                ]:
                    corr = np.where(mask_train, np.clip(alpha * delta, -clip, clip), 0.0)
                    corr_test = np.where(mask_test, np.clip(alpha * delta_test, -clip, clip), 0.0)
                    pred = base * np.exp(corr)
                    pred_test = base_test * np.exp(corr_test)
                    ratio = pred_test / base_test - 1.0
                    name = f"{pred_col}__a{alpha:.2f}__c{clip:.3f}__{gate}"
                    test_preds[name] = pred_test
                    stress = stress_bucket_summary(train, name, pred)
                    regions = major_region_summary(train, name, pred)
                    rows.append(
                        {
                            "candidate": name,
                            "expert": pred_col,
                            "alpha": alpha,
                            "clip": clip,
                            "gate": gate,
                            "oof_mape": mape(y, pred),
                            "delta_vs_265_oof": mape(y, pred) - base_m,
                            "stress_buckets_improved": int(stress["improved"].sum()),
                            "major_regions_not_worse": bool(regions["not_worse"].all()) if len(regions) else True,
                            "train_applied_share": float(mask_train.mean()),
                            "test_applied_share": float(mask_test.mean()),
                            "test_mean_abs_shift_pct": float(np.mean(np.abs(ratio)) * 100.0),
                            "test_max_abs_shift_pct": float(np.max(np.abs(ratio)) * 100.0),
                        }
                    )
                    stress_parts.append(stress)
                    region_parts.append(regions)
    summary = pd.DataFrame(rows).sort_values(["delta_vs_265_oof", "test_mean_abs_shift_pct"]).reset_index(drop=True)
    return summary, pd.concat(stress_parts, ignore_index=True), pd.concat(region_parts, ignore_index=True), test_preds


def write_submissions(test: pd.DataFrame, summary: pd.DataFrame, test_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    accepted = summary[
        (summary["delta_vs_265_oof"] <= -0.015)
        & (summary["stress_buckets_improved"] >= 3)
        & (summary["test_mean_abs_shift_pct"] <= 0.25)
        & (summary["test_max_abs_shift_pct"] <= 1.50)
        & (summary["major_regions_not_worse"])
    ].head(8)
    columns = ["candidate_id", "file", *summary.columns.tolist()]
    rows = []
    for candidate_id, (_, row) in enumerate(accepted.iterrows(), start=420):
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(test_preds[row["candidate"]], 0), 2)
        token = str(row["candidate"]).replace("__", "_").replace(".", "p")
        file_name = f"stage2_candidate_{candidate_id}_similarity_{token}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append({"candidate_id": candidate_id, "file": file_name, **row.to_dict()})
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    rto, static, meta_oof, meta_test = load_data()
    train = build_similarity_predictions(
        rto,
        static,
        query_period=period_id(2024, 3),
        reference_end_period=period_id(2024, 2),
        meta=meta_oof,
        is_train=True,
    )
    test = build_similarity_predictions(
        rto,
        static,
        query_period=period_id(2025, 3),
        reference_end_period=period_id(2025, 2),
        meta=meta_test,
        is_train=False,
    )
    train.to_csv(OUT_DIR / "oof_similarity_predictions.csv", index=False)
    test.to_csv(OUT_DIR / "test_similarity_predictions.csv", index=False)
    summary, stress, regions, test_preds = evaluate(train, test)
    submissions = write_submissions(test, summary, test_preds)
    summary.to_csv(OUT_DIR / "similarity_expert_summary.csv", index=False)
    stress.to_csv(OUT_DIR / "stress_bucket_summary.csv", index=False)
    regions.to_csv(OUT_DIR / "major_region_summary.csv", index=False)
    submissions.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    print("Base pred265 OOF MAPE:", mape(train["target"], train["anchor"]))
    print("\nTop similarity candidates:")
    print(summary.head(25).to_string(index=False))
    print("\nAccepted submissions:")
    if submissions.empty:
        print("No candidates met acceptance criteria.")
    else:
        print(submissions.to_string(index=False))


if __name__ == "__main__":
    main()
