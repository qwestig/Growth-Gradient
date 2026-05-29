from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "new_baseline"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
TARGET = "target_rto"


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
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)


def add_meta_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    anchor = "pred265_oof" if "pred265_oof" in out.columns else "pred265_submission"
    for col in ["pred204", "pred210", "pred118", "pred159", "pred215", "pred220"]:
        if col in out.columns:
            out[f"meta_log_{col}_to_265"] = np.log(
                np.maximum(out[col].to_numpy(float), 1.0)
                / np.maximum(out[anchor].to_numpy(float), 1.0)
            )
    if "pred284_submission" in out.columns and "pred265_submission" in out.columns:
        out["meta_log_284_to_265"] = np.log(
            np.maximum(out["pred284_submission"].to_numpy(float), 1.0)
            / np.maximum(out["pred265_submission"].to_numpy(float), 1.0)
        )
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    blocked = {
        "new_id",
        "fold",
        TARGET,
        "pred204",
        "pred210",
        "pred118",
        "pred159",
        "pred215",
        "pred220",
        "pred261_oof",
        "pred265_oof",
        "actual_log_residual_vs_265",
        "actual_pct_residual_vs_265",
        "ape_265",
    }
    candidates = [
        col
        for col in train.columns
        if col not in blocked
        and (
            col.startswith("feature__")
            or col.startswith("v11_")
            or col.startswith("v14_")
            or col.startswith("meta_log_")
        )
    ]
    cats = [col for col in candidates if not is_numeric_dtype(train[col])]
    numeric = [col for col in candidates if col not in cats]
    return numeric, cats


def make_model(kind: str, numeric: list[str], cats: list[str]):
    pre = ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False),
                ),
                cats,
            ),
        ],
        remainder="drop",
    )
    if kind.startswith("ridge"):
        model = Ridge(alpha=float(kind.replace("ridge", "")))
    elif kind == "huber010":
        model = HuberRegressor(epsilon=1.20, alpha=0.10, max_iter=500)
    elif kind == "huber030":
        model = HuberRegressor(epsilon=1.35, alpha=0.30, max_iter=500)
    elif kind == "huber100":
        model = HuberRegressor(epsilon=1.50, alpha=1.00, max_iter=500)
    else:
        raise ValueError(kind)
    return make_pipeline(pre, model)


def crossfit_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    cats: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y = train["actual_log_residual_vs_265"].to_numpy(float)
    kinds = ["ridge10", "ridge30", "ridge100", "ridge300", "huber010", "huber030", "huber100"]
    oof = pd.DataFrame({"new_id": train["new_id"], "fold": train["fold"]})
    test_pred: dict[str, np.ndarray] = {}
    x_cols = numeric + cats
    for kind in kinds:
        pred = np.zeros(len(train), dtype=float)
        for fold in sorted(train["fold"].unique()):
            tr = train["fold"] != fold
            va = train["fold"] == fold
            model = make_model(kind, numeric, cats)
            model.fit(train.loc[tr, x_cols], y[tr])
            pred[va] = model.predict(train.loc[va, x_cols])
        final = make_model(kind, numeric, cats)
        final.fit(train[x_cols], y)
        oof[kind] = pred
        test_pred[kind] = final.predict(test[x_cols])
    return oof, test_pred


def bootstrap_delta(
    y: np.ndarray,
    ref: np.ndarray,
    pred: np.ndarray,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = np.abs((y - pred) / y) * 100.0 - np.abs((y - ref) / y) * 100.0
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[idx].mean(axis=1)
    return {
        "boot_p05": float(np.quantile(boot, 0.05)),
        "boot_p50": float(np.quantile(boot, 0.50)),
        "boot_p95": float(np.quantile(boot, 0.95)),
        "prob_improve": float((boot < 0).mean()),
    }


def evaluate_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_oof: pd.DataFrame,
    raw_test: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y = train[TARGET].to_numpy(float)
    base_oof = train["pred265_oof"].to_numpy(float)
    base_test = test["pred265_submission"].to_numpy(float)
    base_mape = mape(y, base_oof)
    rows = []
    test_predictions: dict[str, np.ndarray] = {}

    specs: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []
    for name in [col for col in raw_oof.columns if col not in {"new_id", "fold"}]:
        for gamma in [0.25, 0.40, 0.60, 0.80, 1.00]:
            for clip in [0.002, 0.003, 0.005, 0.008, 0.012]:
                specs.append((name, raw_oof[name].to_numpy(float), raw_test[name], gamma, clip))

    # Include v14 deltas as a reference family in the same summary.
    for delta_col in ["v14_delta284_vs_265_log", "v14_delta285_vs_265_log"]:
        if delta_col in train.columns and delta_col in test.columns:
            for gamma in [0.50, 0.75, 1.00, 1.25]:
                specs.append((delta_col, train[delta_col].to_numpy(float), test[delta_col].to_numpy(float), gamma, 0.020))

    for name, tr_raw, te_raw, gamma, clip in specs:
        tr_center = tr_raw - float(np.nanmedian(tr_raw))
        te_center = te_raw - float(np.nanmedian(te_raw))
        tr_corr = np.clip(gamma * tr_center, -clip, clip)
        te_corr = np.clip(gamma * te_center, -clip, clip)
        pred_oof = base_oof * np.exp(tr_corr)
        pred_test = base_test * np.exp(te_corr)
        ratio = pred_test / base_test - 1.0
        candidate_name = f"{name}_g{gamma:.2f}_c{clip:.3f}"
        rows.append(
            {
                "name": candidate_name,
                "source": name,
                "gamma": gamma,
                "clip": clip,
                "oof_mape": mape(y, pred_oof),
                "delta_vs_265_oof": mape(y, pred_oof) - base_mape,
                "mean_abs_vs_265_pct": float(np.mean(np.abs(ratio)) * 100.0),
                "p95_abs_vs_265_pct": float(np.quantile(np.abs(ratio), 0.95) * 100.0),
                "max_abs_vs_265_pct": float(np.max(np.abs(ratio)) * 100.0),
                "mean_delta_vs_265_pct": float(np.mean(ratio) * 100.0),
                **bootstrap_delta(y, base_oof, pred_oof, n_boot=800),
            }
        )
        test_predictions[candidate_name] = pred_test

    summary = pd.DataFrame(rows).sort_values(["delta_vs_265_oof", "mean_abs_vs_265_pct"]).reset_index(drop=True)
    return summary, test_predictions


def write_candidates(test: pd.DataFrame, summary: pd.DataFrame, test_predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    auto_selected = summary[
        (summary["delta_vs_265_oof"] < -0.0010)
        & (summary["mean_abs_vs_265_pct"] <= 0.35)
        & (summary["p95_abs_vs_265_pct"] <= 1.00)
        & (summary["prob_improve"] >= 0.95)
    ].head(8)
    if auto_selected.empty:
        auto_selected = summary.head(6)

    tier_names = [
        "huber030_g0.40_c0.003",  # conservative: small public-test shift, still solid OOF gain
        "huber030_g0.60_c0.005",  # safe: larger OOF gain, bounded by about 0.5% max shift
        "huber030_g0.80_c0.008",  # balanced: best middle ground before aggressive moves
        "huber030_g1.00_c0.012",  # aggressive: best OOF row, highest public-test risk
    ]
    tier_selected = summary[summary["name"].isin(tier_names)].copy()
    tier_selected["tier_order"] = tier_selected["name"].map({name: idx for idx, name in enumerate(tier_names)})
    tier_selected = tier_selected.sort_values("tier_order").drop(columns=["tier_order"])

    selected = pd.concat([auto_selected, tier_selected], ignore_index=True)
    selected = selected.drop_duplicates(subset=["name"], keep="first")

    rows = []
    for candidate_id, (_, row) in enumerate(selected.iterrows(), start=310):
        out = test[["new_id"]].copy()
        pred = test_predictions[row["name"]]
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        safe_name = str(row["name"]).replace(".", "p")
        file_name = f"stage2_candidate_{candidate_id}_v15_meta265_{safe_name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append({"candidate_id": candidate_id, "file": file_name, **row.to_dict()})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    train = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    train = add_meta_ratios(train)
    test = add_meta_ratios(test)
    numeric, cats = feature_columns(train)

    raw_oof, raw_test = crossfit_predictions(train, test, numeric, cats)
    raw_oof.to_csv(RESULTS_DIR / "v15_meta265_raw_oof.csv", index=False)

    summary, test_predictions = evaluate_candidates(train, test, raw_oof, raw_test)
    summary.to_csv(RESULTS_DIR / "v15_meta265_search_summary.csv", index=False)
    candidates = write_candidates(test, summary, test_predictions)
    candidates.to_csv(RESULTS_DIR / "v15_meta265_submission_candidates.csv", index=False)

    print("Features:", len(numeric), "numeric,", len(cats), "categorical")
    print("\nTop search rows:")
    print(summary.head(20).to_string(index=False))
    print("\nWritten candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
