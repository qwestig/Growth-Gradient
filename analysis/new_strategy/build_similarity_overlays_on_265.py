from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
SIM_DIR = ROOT / "results" / "new_baseline_similarity"
OUT_DIR = ROOT / "results" / "new_strategy_similarity_overlays_265"
SUBMISSIONS_DIR = ROOT / "submissions"
MAJOR_REGIONS = [
    "Москва г",
    "Московская обл",
    "Краснодарский край",
    "Татарстан Респ",
    "Ростовская обл",
]


@dataclass(frozen=True)
class Spec:
    candidate_id: int
    name: str
    base: str
    expert: str
    mask: str
    rel_to: str
    alpha: float
    clip: float


def read_remote_csv(path: str) -> pd.DataFrame:
    result = subprocess.run(
        ["git", "show", f"{REMOTE_BRANCH}:{path}"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return pd.read_csv(StringIO(result.stdout))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return float(np.mean(np.abs((p[mask] - y[mask]) / y[mask])) * 100.0)


def log_ratio(num: np.ndarray | pd.Series, den: np.ndarray | pd.Series) -> np.ndarray:
    return np.log(np.clip(np.asarray(num, dtype=float), 1.0, None) / np.clip(np.asarray(den, dtype=float), 1.0, None))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv")
    meta_test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv")
    sim_oof = pd.read_csv(SIM_DIR / "oof_similarity_predictions.csv")
    sim_test = pd.read_csv(SIM_DIR / "test_similarity_predictions.csv")
    train = meta_oof.merge(sim_oof, on="new_id", suffixes=("", "_sim"), validate="one_to_one")
    test = meta_test.merge(sim_test, on="new_id", suffixes=("", "_sim"), validate="one_to_one")
    return train, test


def base_arrays(train: pd.DataFrame, test: pd.DataFrame, base: str) -> tuple[np.ndarray, np.ndarray]:
    if base == "265":
        return train["pred265_oof"].to_numpy(float), test["pred265_submission"].to_numpy(float)
    if base == "284":
        return (
            train["pred265_oof"].to_numpy(float) * np.exp(train["v14_delta284_vs_265_log"].to_numpy(float)),
            test["pred284_submission"].to_numpy(float),
        )
    if base == "285":
        return (
            train["pred265_oof"].to_numpy(float) * np.exp(train["v14_delta285_vs_265_log"].to_numpy(float)),
            test["pred285_submission"].to_numpy(float),
        )
    raise ValueError(f"Unknown base: {base}")


def mask_arrays(train: pd.DataFrame, test: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    k = spec.expert.rsplit("k", 1)[-1]
    std_col = f"knn_seasonal_std_k{k}" if "seasonal" in spec.expert else f"knn_growth_std_k{k}"
    dist_col = f"knn_dist_k{k}"
    if spec.mask == "low40":
        return (
            train[std_col].le(train[std_col].quantile(0.40)).to_numpy(),
            test[std_col].le(test[std_col].quantile(0.40)).to_numpy(),
        )
    if spec.mask == "low_std":
        return (
            train[std_col].le(train[std_col].quantile(0.50)).to_numpy(),
            test[std_col].le(test[std_col].quantile(0.50)).to_numpy(),
        )
    if spec.mask == "close_low_std":
        train_low = train[std_col].le(train[std_col].quantile(0.50)).to_numpy()
        test_low = test[std_col].le(test[std_col].quantile(0.50)).to_numpy()
        train_close = train[dist_col].le(train[dist_col].quantile(0.50)).to_numpy()
        test_close = test[dist_col].le(test[dist_col].quantile(0.50)).to_numpy()
        return train_low & train_close, test_low & test_close
    raise ValueError(f"Unknown mask: {spec.mask}")


def stress_bucket_count(train: pd.DataFrame, pred: np.ndarray) -> int:
    y = train["target_rto"].to_numpy(float)
    base = train["pred265_oof"].to_numpy(float)
    lag = train["lag1"].to_numpy(float)
    vol = train["volatility_rel3"].to_numpy(float)
    shift = np.abs(log_ratio(pred, base))
    neighbor_std = train.filter(like="knn_seasonal_std").median(axis=1).to_numpy(float)
    buckets = [
        lag >= np.nanquantile(lag, 0.80),
        lag <= np.nanquantile(lag, 0.20),
        vol >= np.nanquantile(vol, 0.80),
        neighbor_std <= np.nanquantile(neighbor_std, 0.40),
        shift >= np.nanquantile(shift, 0.80),
    ]
    return int(sum(mape(y[mask], pred[mask]) < mape(y[mask], base[mask]) for mask in buckets))


def major_region_deltas(train: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    y = train["target_rto"].to_numpy(float)
    base = train["pred265_oof"].to_numpy(float)
    out: dict[str, float] = {}
    for region in MAJOR_REGIONS:
        mask = train["Регион"].astype(str).eq(region).to_numpy()
        out[region] = mape(y[mask], pred[mask]) - mape(y[mask], base[mask])
    return out


def apply_spec(train: pd.DataFrame, test: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    base_train, base_test = base_arrays(train, test, spec.base)
    anchor_train = train["pred265_oof"].to_numpy(float)
    anchor_test = test["pred265_submission"].to_numpy(float)
    ref_train = anchor_train if spec.rel_to == "265" else base_train
    ref_test = anchor_test if spec.rel_to == "265" else base_test
    mask_train, mask_test = mask_arrays(train, test, spec)
    delta_train = log_ratio(train[spec.expert], ref_train)
    delta_test = log_ratio(test[spec.expert], ref_test)
    corr_train = np.where(mask_train, np.clip(spec.alpha * delta_train, -spec.clip, spec.clip), 0.0)
    corr_test = np.where(mask_test, np.clip(spec.alpha * delta_test, -spec.clip, spec.clip), 0.0)
    pred_train = base_train * np.exp(corr_train)
    pred_test = base_test * np.exp(corr_test)
    shift = pred_test / anchor_test - 1.0
    stats = {
        "train_applied_share": float(mask_train.mean()),
        "test_applied_share": float(mask_test.mean()),
        "test_mean_abs_shift_vs_265_pct": float(np.mean(np.abs(shift)) * 100.0),
        "test_p95_abs_shift_vs_265_pct": float(np.quantile(np.abs(shift), 0.95) * 100.0),
        "test_max_abs_shift_vs_265_pct": float(np.max(np.abs(shift)) * 100.0),
        "test_mean_shift_vs_265_pct": float(np.mean(shift) * 100.0),
    }
    return pred_train, pred_test, stats


def specs() -> list[Spec]:
    return [
        Spec(
            candidate_id=476,
            name="v14_284_knn50_low40_a005_c005",
            base="284",
            expert="knn_seasonal_pred_k50",
            mask="low40",
            rel_to="base",
            alpha=0.05,
            clip=0.005,
        ),
        Spec(
            candidate_id=477,
            name="v14_285_knn200_low40_a010_c002_safe_regions",
            base="285",
            expert="knn_seasonal_pred_k200",
            mask="low40",
            rel_to="265",
            alpha=0.10,
            clip=0.002,
        ),
        Spec(
            candidate_id=478,
            name="v14_285_knn100_lowstd_a005_c002_safe_regions",
            base="285",
            expert="knn_seasonal_pred_k100",
            mask="low_std",
            rel_to="265",
            alpha=0.05,
            clip=0.002,
        ),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_inputs()
    y = train["target_rto"].to_numpy(float)
    anchor_train = train["pred265_oof"].to_numpy(float)
    anchor_test = test["pred265_submission"].to_numpy(float)
    base_mape = mape(y, anchor_train)
    rows = []
    for spec in specs():
        pred_train, pred_test, stats = apply_spec(train, test, spec)
        candidate_mape = mape(y, pred_train)
        region_deltas = major_region_deltas(train, pred_train)
        file_name = f"stage2_candidate_{spec.candidate_id}_similarity265_{spec.name}.csv"
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred_test, 0), 2)
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        out.to_csv(SUBMISSIONS_DIR / f"{spec.candidate_id}.csv", index=False)
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "file": file_name,
                "name": spec.name,
                "base": spec.base,
                "expert": spec.expert,
                "mask": spec.mask,
                "rel_to": spec.rel_to,
                "alpha": spec.alpha,
                "clip": spec.clip,
                "oof_mape": candidate_mape,
                "delta_vs_265_oof": candidate_mape - base_mape,
                "stress_buckets_improved": stress_bucket_count(train, pred_train),
                "major_regions_not_worse": bool(all(v <= 0 for v in region_deltas.values())),
                "max_major_region_delta": max(region_deltas.values()),
                **stats,
                **{f"region_delta_{k}": v for k, v in region_deltas.items()},
            }
        )
        if len(out) != 18657 or list(out.columns) != ["new_id", "rto"] or out["rto"].isna().any() or (out["rto"] < 0).any():
            raise RuntimeError(f"Bad submission format for candidate {spec.candidate_id}")
    summary = pd.DataFrame(rows).sort_values(["delta_vs_265_oof", "test_mean_abs_shift_vs_265_pct"])
    summary.to_csv(OUT_DIR / "similarity_overlay_summary.csv", index=False)
    print("Base pred265 OOF MAPE:", f"{base_mape:.6f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
