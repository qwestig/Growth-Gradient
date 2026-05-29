from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "new_strategy_micro_overlays_265"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
META_TARGET = "target_rto"


@dataclass(frozen=True)
class OverlaySpec:
    candidate_id: int
    name: str
    base: str
    source: str
    alpha: float
    clip: float
    mask: str = "all"
    direction: int = 1


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
    mask = (y > 0) & np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs((p[mask] - y[mask]) / y[mask])) * 100.0)


def feb_gru_prediction(result_dir: str, target: str) -> pd.DataFrame:
    if target == "oof":
        frame = pd.read_csv(ROOT / "results" / result_dir / "oof.csv").sort_values("new_id").reset_index(drop=True)
        delta = frame["pred_log_gru"].to_numpy(float) - float(frame["pred_log_gru"].median())
        pred = frame["new_base"].to_numpy(float) * np.exp(delta)
        return frame[["new_id"]].assign(pred=pred)
    if target == "test":
        frame = pd.read_csv(ROOT / "results" / result_dir / "test_predictions.csv").sort_values("new_id").reset_index(drop=True)
        delta = frame["pred_log_gru"].to_numpy(float) - float(frame["pred_log_gru"].median())
        pred = frame["new_base"].to_numpy(float) * np.exp(delta)
        return frame[["new_id"]].assign(pred=pred)
    raise ValueError(target)


def base_predictions(meta: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    if target == "oof":
        pred265 = meta["pred265_oof"].to_numpy(float)
        return {
            "265": pred265,
            "284": pred265 * np.exp(meta["v14_delta284_vs_265_log"].to_numpy(float)),
            "285": pred265 * np.exp(meta["v14_delta285_vs_265_log"].to_numpy(float)),
        }
    pred265 = meta["pred265_submission"].to_numpy(float)
    return {
        "265": pred265,
        "284": meta["pred284_submission"].to_numpy(float),
        "285": meta["pred285_submission"].to_numpy(float),
    }


def source_predictions(target: str) -> dict[str, pd.DataFrame]:
    return {
        "feb410": feb_gru_prediction("new_strategy_feb_anomaly_gru_cap3p0", target),
        "feb430": feb_gru_prediction("new_strategy_feb_anomaly_gru_uncapped", target),
        "feb450": feb_gru_prediction("new_strategy_feb_anomaly_gru_symmetric_pre5", target),
    }


def mask_values(meta: pd.DataFrame, source_delta: np.ndarray, name: str) -> np.ndarray:
    if name == "all":
        return np.ones(len(meta), dtype=bool)
    if name == "v14top50":
        return meta["v14_top50_mask"].astype(bool).to_numpy()
    if name == "agree":
        return meta["v14_agree_top50_mask"].astype(bool).to_numpy()
    if name == "same_sign_v11":
        return np.sign(source_delta) == np.sign(meta["v11_combo265_centered"].to_numpy(float))
    if name == "large_vol":
        vol = meta["feature__volatility_rel3"].to_numpy(float)
        return vol >= np.nanmedian(vol)
    raise ValueError(name)


def apply_overlay(
    meta: pd.DataFrame,
    bases: dict[str, np.ndarray],
    source_frames: dict[str, pd.DataFrame],
    spec: OverlaySpec,
) -> np.ndarray:
    base = bases[spec.base]
    pred265 = bases["265"]
    source = source_frames[spec.source].sort_values("new_id")["pred"].to_numpy(float)
    delta = np.log(np.clip(source, 1.0, None) / np.clip(pred265, 1.0, None))
    mask = mask_values(meta, delta, spec.mask)
    corr = np.zeros(len(meta), dtype=float)
    corr[mask] = np.clip(spec.direction * spec.alpha * delta[mask], -spec.clip, spec.clip)
    return base * np.exp(corr)


def make_specs() -> list[OverlaySpec]:
    return [
        OverlaySpec(466, "v14_284_exact", "284", "feb410", 0.0, 0.0),
        OverlaySpec(467, "v14_284_plus_feb410_a010_c005", "284", "feb410", 0.10, 0.005),
        OverlaySpec(468, "v14_285_plus_feb410_a010_c005", "285", "feb410", 0.10, 0.005),
        OverlaySpec(469, "v14_284_plus_feb410_a010_c003", "284", "feb410", 0.10, 0.003),
        OverlaySpec(470, "v14_284_plus_feb430_a010_c005", "284", "feb430", 0.10, 0.005),
        OverlaySpec(471, "v14_284_plus_feb450_a010_c005", "284", "feb450", 0.10, 0.005),
        OverlaySpec(472, "v14_284_plus_feb410_a020_c008", "284", "feb410", 0.20, 0.008),
        OverlaySpec(473, "v14_284_plus_feb410_a020_c005", "284", "feb410", 0.20, 0.005),
        OverlaySpec(474, "v14_285_plus_feb410_a020_c008", "285", "feb410", 0.20, 0.008),
        OverlaySpec(475, "v14_284_plus_feb430_a020_c008", "284", "feb430", 0.20, 0.008),
    ]


def write_submission(candidate_id: int, name: str, new_id: pd.Series, pred: np.ndarray) -> str:
    file_name = f"stage2_candidate_{candidate_id}_micro265_{name}.csv"
    sub = pd.DataFrame({"new_id": new_id.astype(int), "rto": np.round(np.maximum(pred, 1.0), 2)})
    if len(sub) != 18657 or sub["rto"].isna().any() or (sub["rto"] <= 0).any():
        raise RuntimeError(f"Bad submission format for {file_name}")
    sub.to_csv(SUBMISSIONS_DIR / file_name, index=False)
    sub.to_csv(SUBMISSIONS_DIR / f"{candidate_id}.csv", index=False)
    return file_name


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    oof = read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv").sort_values("new_id").reset_index(drop=True)
    test = read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv").sort_values("new_id").reset_index(drop=True)
    oof_bases = base_predictions(oof, "oof")
    test_bases = base_predictions(test, "test")
    oof_sources = source_predictions("oof")
    test_sources = source_predictions("test")

    y = oof[META_TARGET].to_numpy(float)
    base_mape = mape(y, oof_bases["265"])
    rows = []
    for spec in make_specs():
        if spec.alpha == 0.0:
            pred_oof = oof_bases[spec.base]
            pred_test = test_bases[spec.base]
        else:
            pred_oof = apply_overlay(oof, oof_bases, oof_sources, spec)
            pred_test = apply_overlay(test, test_bases, test_sources, spec)
        shift = pred_test / test_bases["265"] - 1.0
        file_name = write_submission(spec.candidate_id, spec.name, test["new_id"], pred_test)
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "file": file_name,
                "name": spec.name,
                "base": spec.base,
                "source": spec.source,
                "alpha": spec.alpha,
                "clip": spec.clip,
                "mask": spec.mask,
                "oof_mape": mape(y, pred_oof),
                "delta_vs_265_oof": mape(y, pred_oof) - base_mape,
                "test_mean_abs_shift_vs_265_pct": float(np.mean(np.abs(shift)) * 100.0),
                "test_p95_abs_shift_vs_265_pct": float(pd.Series(np.abs(shift)).quantile(0.95) * 100.0),
                "test_max_abs_shift_vs_265_pct": float(np.max(np.abs(shift)) * 100.0),
                "test_mean_shift_vs_265_pct": float(np.mean(shift) * 100.0),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["oof_mape", "test_mean_abs_shift_vs_265_pct"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "micro_overlay_summary.csv", index=False)
    print(f"265 OOF MAPE: {base_mape:.6f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
