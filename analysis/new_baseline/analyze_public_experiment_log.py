from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
LOG_PATH = RESULTS_DIR / "new_baseline" / "public_experiment_log.csv"
OUT_PATH = RESULTS_DIR / "new_baseline" / "public_experiment_direction_summary.csv"
CURRENT_BEST_SCORE = 89.65


def score_to_mape(score: float) -> float:
    return float(100 * (1 - np.sqrt(score / 100)))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_iter: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_iter, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.10)), float(np.quantile(samples, 0.90))


def ensure_log_exists() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        return

    rows = [
        {
            "candidate_id": "best_gru_momentum_bias_8965",
            "file": "Growth-Gradient/submissions/test_best_gru_momentum_bias_8965.csv",
            "family": "current_best_reference",
            "hypothesis": "Current public-best GRU plus momentum_raw_bin correction",
            "public_score": CURRENT_BEST_SCORE,
            "notes": "reference",
        },
        {
            "candidate_id": 36,
            "file": "stage2_candidate_36_gru_second_store_mar_feb_ratio_bin_k1000_b050_c005.csv",
            "family": "rejected_second_level_oof_segment",
            "hypothesis": "Extra store_mar_feb_ratio_bin second-level correction after GRU-best",
            "public_score": 89.58,
            "notes": "proxy-positive but public-negative",
        },
    ]

    template_path = RESULTS_DIR / "heuristic_public_experiment_template.csv"
    if template_path.exists():
        template = pd.read_csv(template_path)
        for _, row in template.iterrows():
            rows.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "file": row.get("name", ""),
                    "family": "heuristic_baseline_direction",
                    "hypothesis": row.get("direction_question", ""),
                    "public_score": "",
                    "notes": "fill public_score after submission",
                }
            )

    pd.DataFrame(rows).to_csv(LOG_PATH, index=False)


def main() -> None:
    ensure_log_exists()
    log = pd.read_csv(LOG_PATH)
    log["public_score"] = pd.to_numeric(log["public_score"], errors="coerce")
    log["public_mape"] = log["public_score"].map(lambda value: score_to_mape(value) if pd.notna(value) else np.nan)
    ref_mape = score_to_mape(CURRENT_BEST_SCORE)
    log["delta_mape_vs_current_best"] = log["public_mape"] - ref_mape
    log["delta_score_vs_current_best"] = log["public_score"] - CURRENT_BEST_SCORE

    submitted = log.dropna(subset=["public_score"]).copy()
    rng = np.random.default_rng(2026)
    rows = []
    for family, part in submitted.groupby("family", observed=True):
        deltas = part["delta_mape_vs_current_best"].to_numpy(dtype=float)
        lo, hi = bootstrap_mean(deltas, rng)
        rows.append(
            {
                "family": family,
                "n_submitted": len(part),
                "best_score": float(part["public_score"].max()),
                "best_mape": float(part.loc[part["public_score"].idxmax(), "public_mape"]),
                "best_candidate_id": part.loc[part["public_score"].idxmax(), "candidate_id"],
                "mean_delta_mape_vs_current": float(np.mean(deltas)),
                "p10_bootstrap_mean_delta": lo,
                "p90_bootstrap_mean_delta": hi,
                "direction_read": direction_read(deltas),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["best_score", "mean_delta_mape_vs_current"], ascending=[False, True])
    log.to_csv(LOG_PATH, index=False)
    summary.to_csv(OUT_PATH, index=False)

    print(f"Updated {LOG_PATH}")
    print(f"Wrote {OUT_PATH}")
    print("\nSubmitted experiments:")
    print(
        submitted[["candidate_id", "family", "public_score", "public_mape", "delta_mape_vs_current_best", "hypothesis"]]
        .sort_values("public_score", ascending=False)
        .to_string(index=False)
    )
    print("\nDirection summary:")
    print(summary.to_string(index=False))


def direction_read(deltas: np.ndarray) -> str:
    if len(deltas) == 0:
        return "no signal"
    best = float(np.min(deltas))
    mean = float(np.mean(deltas))
    if best < -0.05:
        return "strong positive: expand this family"
    if best < -0.015:
        return "weak positive: test nearby variants"
    if mean > 0.02 and best > 0:
        return "negative: deprioritize"
    return "mixed/neutral: use only for differential diagnostics"


if __name__ == "__main__":
    main()
