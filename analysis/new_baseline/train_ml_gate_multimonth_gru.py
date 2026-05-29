from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "new_baseline_multimonth_gru_seasonal"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"


def read_remote_csv(path: str) -> pd.DataFrame:
    result = subprocess.run(
        ["git", "show", f"{REMOTE_BRANCH}:{path}"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return pd.read_csv(StringIO(result.stdout))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0)


def add_gate_features(base: pd.DataFrame, meta: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = base.merge(meta, on="new_id", how="left", suffixes=("", "_meta"))
    anchor = "pred265_oof" if is_train else "pred265_submission"
    out["mmgru_delta_log"] = np.log(
        np.clip(out["pred_mmgru_raw"].to_numpy(float), 1.0, None)
        / np.clip(out[anchor].to_numpy(float), 1.0, None)
    )
    out["mmgru_delta_log_centered"] = out["mmgru_delta_log"] - float(np.nanmedian(out["mmgru_delta_log"]))
    out["mmgru_abs_delta_log"] = np.abs(out["mmgru_delta_log"])
    out["mmgru_abs_delta_centered"] = np.abs(out["mmgru_delta_log_centered"])
    out["mmgru_raw_to_seasonal_log"] = np.log(
        np.clip(out["pred_mmgru_raw"].to_numpy(float), 1.0, None)
        / np.clip(out["seasonal_baseline"].to_numpy(float), 1.0, None)
    )
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(train: pd.DataFrame) -> list[str]:
    prefixes = ("feature__", "v11_", "v14_", "mmgru_")
    blocked = {
        "new_id",
        "target_period",
        "target_year",
        "target_month",
        "fold",
        "target_rto",
        "pred265_oof",
        "pred265_submission",
        "pred_mmgru_raw",
        "seasonal_baseline",
        "pred_log_growth_mmgru",
    }
    features = []
    for col in train.columns:
        if col in blocked:
            continue
        if col.startswith(prefixes) and is_numeric_dtype(train[col]):
            features.append(col)
    return features


def correction(delta: np.ndarray, alpha: float, clip: float, center: str) -> np.ndarray:
    d = delta.copy()
    if center == "median":
        d = d - float(np.nanmedian(d))
    return np.clip(alpha * d, -clip, clip)


def make_classifier(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.045,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=0.20,
            min_samples_leaf=80,
            random_state=seed,
        ),
    )


def make_regressor(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.045,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=0.20,
            min_samples_leaf=80,
            random_state=seed,
        ),
    )


def crossfit_gate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    alpha: float,
    clip: float,
    center: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    y = train["target_rto"].to_numpy(float)
    base = train["pred265_oof"].to_numpy(float)
    base_test = test["pred265_submission"].to_numpy(float)
    corr_oof = correction(train["mmgru_delta_log"].to_numpy(float), alpha, clip, center)
    corr_test = correction(test["mmgru_delta_log"].to_numpy(float), alpha, clip, center)
    cand = base * np.exp(corr_oof)
    improve = (np.abs((cand - y) / y) < np.abs((base - y) / y)).astype(int)
    delta_ape = np.abs((cand - y) / y) * 100.0 - np.abs((base - y) / y) * 100.0

    p_oof = np.zeros(len(train), dtype=float)
    r_oof = np.zeros(len(train), dtype=float)
    p_test_folds = []
    r_test_folds = []
    for fold in sorted(train["fold"].unique()):
        tr = train["fold"] != fold
        va = train["fold"] == fold
        clf = make_classifier(2026 + int(fold))
        reg = make_regressor(3026 + int(fold))
        clf.fit(train.loc[tr, features], improve[tr])
        reg.fit(train.loc[tr, features], delta_ape[tr])
        p_oof[va] = clf.predict_proba(train.loc[va, features])[:, 1]
        r_oof[va] = reg.predict(train.loc[va, features])
        p_test_folds.append(clf.predict_proba(test[features])[:, 1])
        r_test_folds.append(reg.predict(test[features]))

    p_test = np.mean(np.stack(p_test_folds, axis=0), axis=0)
    r_test = np.mean(np.stack(r_test_folds, axis=0), axis=0)

    rows = []
    test_preds: dict[str, np.ndarray] = {}
    for gate_type in ["prob", "reg", "hybrid"]:
        for coverage in [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]:
            if gate_type == "prob":
                score_oof = p_oof
                score_test = p_test
                high_is_good = True
            elif gate_type == "reg":
                score_oof = r_oof
                score_test = r_test
                high_is_good = False
            else:
                score_oof = p_oof - 0.08 * r_oof
                score_test = p_test - 0.08 * r_test
                high_is_good = True

            if coverage >= 0.999:
                mask_oof = np.ones(len(train), dtype=bool)
                mask_test = np.ones(len(test), dtype=bool)
            elif high_is_good:
                thr = float(np.quantile(score_oof, 1.0 - coverage))
                thrt = float(np.quantile(score_test, 1.0 - coverage))
                mask_oof = score_oof >= thr
                mask_test = score_test >= thrt
            else:
                thr = float(np.quantile(score_oof, coverage))
                thrt = float(np.quantile(score_test, coverage))
                mask_oof = score_oof <= thr
                mask_test = score_test <= thrt

            pred_oof = base * np.exp(np.where(mask_oof, corr_oof, 0.0))
            pred_test = base_test * np.exp(np.where(mask_test, corr_test, 0.0))
            ratio = pred_test / base_test - 1.0
            name = f"{center}_a{alpha:.2f}_c{clip:.3f}_{gate_type}_cov{coverage:.2f}"
            test_preds[name] = pred_test
            rows.append(
                {
                    "name": name,
                    "center": center,
                    "alpha": alpha,
                    "clip": clip,
                    "gate_type": gate_type,
                    "coverage": coverage,
                    "oof_mape": mape(y, pred_oof),
                    "delta_vs_265_oof": mape(y, pred_oof) - mape(y, base),
                    "oof_applied_share": float(mask_oof.mean()),
                    "test_applied_share": float(mask_test.mean()),
                    "test_mean_abs_shift_pct": float(np.mean(np.abs(ratio)) * 100.0),
                    "test_max_abs_shift_pct": float(np.max(np.abs(ratio)) * 100.0),
                    "improve_rate_oof": float(improve.mean()),
                }
            )
    return pd.DataFrame(rows), p_test, test_preds


def write_candidates(test: pd.DataFrame, summary: pd.DataFrame, pred_map: dict[str, np.ndarray]) -> pd.DataFrame:
    selected = summary[
        (summary["delta_vs_265_oof"] < -0.004)
        & (summary["test_mean_abs_shift_pct"] <= 0.22)
        & (summary["test_max_abs_shift_pct"] <= 1.25)
    ].head(10)
    if selected.empty:
        selected = summary.head(8)

    rows = []
    for candidate_id, (_, row) in enumerate(selected.iterrows(), start=380):
        pred = pred_map[row["name"]]
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        token = (
            str(row["name"])
            .replace(".", "p")
            .replace("_c0p", "_c")
            .replace("_a0p", "_a")
            .replace("_cov0p", "_cov")
        )
        file_name = f"stage2_candidate_{candidate_id}_mlgate_mmgru_{token}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append({"candidate_id": candidate_id, "file": file_name, **row.to_dict()})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(RESULTS_DIR / "oof.csv")
    test_pred = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    meta_oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    meta_test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    train = add_gate_features(oof, meta_oof, is_train=True)
    test = add_gate_features(test_pred, meta_test, is_train=False)
    features = feature_columns(train)

    all_rows = []
    all_preds: dict[str, np.ndarray] = {}
    for center in ["median", "none"]:
        for alpha in [0.10, 0.15, 0.20, 0.25]:
            for clip in [0.005, 0.008, 0.012, 0.020]:
                rows, _p_test, preds = crossfit_gate(train, test, features, alpha, clip, center)
                all_rows.append(rows)
                all_preds.update(preds)

    summary = pd.concat(all_rows, ignore_index=True).sort_values(
        ["delta_vs_265_oof", "test_mean_abs_shift_pct"]
    ).reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "ml_gate_search_summary.csv", index=False)
    candidates = write_candidates(test, summary, all_preds)
    candidates.to_csv(RESULTS_DIR / "ml_gate_submission_candidates.csv", index=False)

    print("Features:", len(features))
    print("\nTop ML-gate rows:")
    print(summary.head(30).to_string(index=False))
    print("\nWritten candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
