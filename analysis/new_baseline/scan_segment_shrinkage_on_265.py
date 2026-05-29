from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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


def bootstrap_delta(
    y: np.ndarray,
    ref: np.ndarray,
    pred: np.ndarray,
    n_boot: int = 800,
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


def feature_candidates(train: pd.DataFrame) -> list[str]:
    cols = []
    for col in train.columns:
        if col.startswith("feature__"):
            cols.append(col)
        elif col in {
            "v14_conf_abs_combo",
            "v14_agree_huber_segment",
            "v14_top50_mask",
            "v14_agree_top50_mask",
        }:
            cols.append(col)
    return cols


def label_specs(train: pd.DataFrame, test: pd.DataFrame, col: str) -> list[tuple[str, pd.Series, pd.Series]]:
    tr = train[col]
    te = test[col]
    specs: list[tuple[str, pd.Series, pd.Series]] = []

    if (not is_numeric_dtype(tr)) or tr.nunique(dropna=True) <= 20:
        specs.append(
            (
                "cat",
                tr.astype("string").fillna("__NA__"),
                te.astype("string").fillna("__NA__"),
            )
        )
        return specs

    tr_num = pd.to_numeric(tr, errors="coerce")
    te_num = pd.to_numeric(te, errors="coerce")
    for q in [4, 6, 10]:
        quantiles = np.unique(np.nanquantile(tr_num.to_numpy(float), np.linspace(0, 1, q + 1)))
        if len(quantiles) < 3:
            continue
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        tr_lab = pd.cut(tr_num, bins=quantiles, include_lowest=True).astype("string").fillna("__NA__")
        te_lab = pd.cut(te_num, bins=quantiles, include_lowest=True).astype("string").fillna("__NA__")
        specs.append((f"q{len(quantiles) - 1}", tr_lab, te_lab))
    return specs


def build_correction(
    labels_train: pd.Series,
    labels_test: pd.Series,
    folds: pd.Series,
    residual: np.ndarray,
    gamma: float,
    clip: float,
    shrink_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    oof_corr = np.zeros(len(labels_train), dtype=float)
    work = pd.DataFrame(
        {
            "label": labels_train.astype("string").fillna("__NA__"),
            "fold": folds.to_numpy(),
            "residual": residual,
        }
    )

    for fold in sorted(work["fold"].unique()):
        tr = work["fold"] != fold
        va = work["fold"] == fold
        global_center = float(np.nanmedian(work.loc[tr, "residual"]))
        stats = work.loc[tr].groupby("label")["residual"].agg(["median", "size"])
        raw = (stats["median"] - global_center) * stats["size"] / (stats["size"] + shrink_k)
        corr = np.clip(gamma * raw, -clip, clip)
        oof_corr[va.to_numpy()] = work.loc[va, "label"].map(corr).fillna(0.0).to_numpy(float)

    global_center = float(np.nanmedian(work["residual"]))
    stats = work.groupby("label")["residual"].agg(["median", "size"])
    raw = (stats["median"] - global_center) * stats["size"] / (stats["size"] + shrink_k)
    corr = np.clip(gamma * raw, -clip, clip)
    test_corr = labels_test.astype("string").fillna("__NA__").map(corr).fillna(0.0).to_numpy(float)
    return oof_corr, test_corr


def get_labels(train: pd.DataFrame, test: pd.DataFrame, col: str, label_mode: str) -> tuple[pd.Series, pd.Series]:
    for mode, labels_train, labels_test in label_specs(train, test, col):
        if mode == label_mode:
            return labels_train, labels_test
    raise ValueError(f"Unknown label mode {label_mode!r} for {col!r}")


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y = train[TARGET].to_numpy(float)
    base_oof = train["pred265_oof"].to_numpy(float)
    base_test = test["pred265_submission"].to_numpy(float)
    residual = train["actual_log_residual_vs_265"].to_numpy(float)
    base_mape = mape(y, base_oof)

    rows = []
    for col in feature_candidates(train):
        if col not in test.columns:
            continue
        for label_mode, labels_train, labels_test in label_specs(train, test, col):
            n_groups = int(labels_train.nunique(dropna=False))
            if n_groups < 2:
                continue
            for shrink_k in [100, 300, 700, 1500]:
                for gamma in [0.25, 0.50, 0.75, 1.00]:
                    for clip in [0.002, 0.003, 0.005, 0.008]:
                        corr_oof, corr_test = build_correction(
                            labels_train,
                            labels_test,
                            train["fold"],
                            residual,
                            gamma=gamma,
                            clip=clip,
                            shrink_k=shrink_k,
                        )
                        pred_oof = base_oof * np.exp(corr_oof)
                        pred_test = base_test * np.exp(corr_test)
                        ratio = pred_test / base_test - 1.0
                        name = f"{col}__{label_mode}__k{shrink_k:g}__g{gamma:.2f}__c{clip:.3f}"
                        score = mape(y, pred_oof)
                        rows.append(
                            {
                                "name": name,
                                "feature": col,
                                "label_mode": label_mode,
                                "n_groups": n_groups,
                                "shrink_k": shrink_k,
                                "gamma": gamma,
                                "clip": clip,
                                "oof_mape": score,
                                "delta_vs_265_oof": score - base_mape,
                                "mean_abs_vs_265_pct": float(np.mean(np.abs(ratio)) * 100.0),
                                "p95_abs_vs_265_pct": float(np.quantile(np.abs(ratio), 0.95) * 100.0),
                                "max_abs_vs_265_pct": float(np.max(np.abs(ratio)) * 100.0),
                                "mean_delta_vs_265_pct": float(np.mean(ratio) * 100.0),
                            }
                        )

    summary = pd.DataFrame(rows).sort_values(["delta_vs_265_oof", "max_abs_vs_265_pct"]).reset_index(drop=True)
    return summary


def safe_file_token(name: str) -> str:
    replacements = {
        "feature__": "",
        "Торговая площадь, категориальный": "area_cat",
        "Дата открытия, категориальный": "open_cat",
        "Регион": "region",
        ".": "p",
        " ": "_",
        ",": "",
        "__": "_",
    }
    out = name
    for old, new in replacements.items():
        out = out.replace(old, new)
    return "".join(ch for ch in out if ch.isalnum() or ch in "_-")[:140]


def write_candidates(train: pd.DataFrame, test: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[
        (summary["delta_vs_265_oof"] < -0.002)
        & (summary["mean_abs_vs_265_pct"] <= 0.40)
        & (summary["p95_abs_vs_265_pct"] <= 0.90)
        & (summary["max_abs_vs_265_pct"] <= 1.10)
    ].head(10)
    if selected.empty:
        selected = summary.head(6)

    rows = []
    y = train[TARGET].to_numpy(float)
    base_oof = train["pred265_oof"].to_numpy(float)
    base_test = test["pred265_submission"].to_numpy(float)
    residual = train["actual_log_residual_vs_265"].to_numpy(float)
    for candidate_id, (_, row) in enumerate(selected.iterrows(), start=330):
        labels_train, labels_test = get_labels(train, test, str(row["feature"]), str(row["label_mode"]))
        corr_oof, corr_test = build_correction(
            labels_train,
            labels_test,
            train["fold"],
            residual,
            gamma=float(row["gamma"]),
            clip=float(row["clip"]),
            shrink_k=float(row["shrink_k"]),
        )
        pred_oof = base_oof * np.exp(corr_oof)
        pred_test = base_test * np.exp(corr_test)
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred_test, 0), 2)
        file_name = f"stage2_candidate_{candidate_id}_seg265_{safe_file_token(str(row['name']))}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                **row.to_dict(),
                **bootstrap_delta(y, base_oof, pred_oof),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    train = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    summary = evaluate(train, test)
    summary.to_csv(RESULTS_DIR / "segment_shrinkage_265_search_summary.csv", index=False)
    candidates = write_candidates(train, test, summary)
    candidates.to_csv(RESULTS_DIR / "segment_shrinkage_265_submission_candidates.csv", index=False)

    print("Current public anchor: score=90.78, mape=4.7214609684")
    print("\nTop search rows:")
    print(summary.head(25).to_string(index=False))
    print("\nWritten candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
