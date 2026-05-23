from __future__ import annotations

import subprocess
from io import StringIO

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import RESULTS_DIR, SUBMISSIONS_DIR


REMOTE_BRANCH = "origin/codex-momentum-bias-solution"


def read_remote_csv(path: str) -> pd.DataFrame:
    result = subprocess.run(
        ["git", "show", f"{REMOTE_BRANCH}:{path}"],
        cwd=RESULTS_DIR.parents[0],
        check=True,
        text=True,
        capture_output=True,
    )
    return pd.read_csv(StringIO(result.stdout))


def main() -> None:
    # Reuse generated submission candidates rather than retraining. These are small CSV reads done by script only.
    base = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_70_baseline_cal050_median_mom000.csv").rename(columns={"rto": "base"})
    g070 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_148_baseline_newgru_faithful_global_a070.csv").rename(columns={"rto": "g070"})
    g085 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_149_baseline_newgru_faithful_global_a085.csv").rename(columns={"rto": "g085"})
    g100 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_150_baseline_newgru_faithful_global_a100.csv").rename(columns={"rto": "g100"})
    transfer = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_118_baseline_cal050_mombin_g100.csv").rename(columns={"rto": "transfer118"})
    frame = base.merge(g070, on="new_id").merge(g085, on="new_id").merge(g100, on="new_id").merge(transfer, on="new_id")

    log_r070 = np.log(frame["g070"] / frame["base"])
    log_r085 = np.log(frame["g085"] / frame["base"])
    log_r100 = np.log(frame["g100"] / frame["base"])
    # Extrapolate/interpolate using the already-generated alpha line.
    unit_log = log_r100
    specs = [
        (155, "faithful_global_a0925", frame["base"].to_numpy(float) * np.exp(0.925 * unit_log)),
        (156, "faithful_global_a1050", frame["base"].to_numpy(float) * np.exp(1.050 * unit_log)),
        (157, "faithful_global_a1100", frame["base"].to_numpy(float) * np.exp(1.100 * unit_log)),
        (158, "faithful_global_a1150", frame["base"].to_numpy(float) * np.exp(1.150 * unit_log)),
        (159, "faithful_global_blend_75_25_transfer", 0.75 * frame["g100"].to_numpy(float) + 0.25 * frame["transfer118"].to_numpy(float)),
        (160, "faithful_global_blend_50_50_transfer", 0.50 * frame["g100"].to_numpy(float) + 0.50 * frame["transfer118"].to_numpy(float)),
        (161, "faithful_global_a1050_clip030", frame["base"].to_numpy(float) * np.exp(np.clip(1.050 * unit_log, -0.03, 0.03))),
        (162, "faithful_global_a1100_clip030", frame["base"].to_numpy(float) * np.exp(np.clip(1.100 * unit_log, -0.03, 0.03))),
    ]

    rows = []
    for candidate_id, name, pred in specs:
        out = frame[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "test_mean_rto": float(np.mean(pred)),
                "test_total_rto": float(np.sum(pred)),
                "mean_abs_change_vs_150_pct": float(np.mean(np.abs(pred / frame["g100"].to_numpy(float) - 1.0)) * 100),
                "p95_abs_change_vs_150_pct": float(np.quantile(np.abs(pred / frame["g100"].to_numpy(float) - 1.0), 0.95) * 100),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "faithful_gru_followup_candidates.csv", index=False)

    log_path = RESULTS_DIR / "public_experiment_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        existing = set(log["candidate_id"].astype(str))
        new_rows = []
        for _, row in summary.iterrows():
            if str(row["candidate_id"]) in existing:
                continue
            new_rows.append({"candidate_id": row["candidate_id"], "file": row["file"], "family": "faithful_gru_followup", "hypothesis": "Fine tune faithful globally-centered GRU alpha/blend around candidate 150", "public_score": "", "notes": "fill public_score after submission", "public_mape": "", "delta_mape_vs_current_best": "", "delta_score_vs_current_best": ""})
        if new_rows:
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
