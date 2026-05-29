from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = ROOT / "submissions"
RESULTS_DIR = ROOT / "results" / "new_baseline"
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


def pct_diagnostics(candidate: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float]:
    merged = candidate.rename(columns={"rto": "candidate"}).merge(
        anchor.rename(columns={"rto": "anchor"}),
        on="new_id",
        how="inner",
    )
    ratio = merged["candidate"].to_numpy(float) / merged["anchor"].to_numpy(float) - 1.0
    return {
        "rows": int(len(merged)),
        "mean_abs_vs_265_pct": float(np.mean(np.abs(ratio)) * 100),
        "p95_abs_vs_265_pct": float(np.quantile(np.abs(ratio), 0.95) * 100),
        "max_abs_vs_265_pct": float(np.max(np.abs(ratio)) * 100),
        "mean_delta_vs_265_pct": float(np.mean(ratio) * 100),
        "share_up": float((ratio > 0).mean()),
    }


def geometric_blend(anchor: pd.DataFrame, target: pd.DataFrame, alpha: float) -> pd.DataFrame:
    frame = anchor.rename(columns={"rto": "anchor"}).merge(
        target.rename(columns={"rto": "target"}),
        on="new_id",
        how="inner",
    )
    log_step = np.log(frame["target"].to_numpy(float) / frame["anchor"].to_numpy(float))
    pred = frame["anchor"].to_numpy(float) * np.exp(alpha * log_step)
    out = frame[["new_id"]].copy()
    out["rto"] = np.round(np.maximum(pred, 0), 2)
    return out


def main() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    anchor265 = read_remote_csv("submissions/test_best_temporal204_stack_v11_safe_4721.csv")
    probe284 = read_remote_csv("submissions/test_probe_v14_selective_284.csv")

    rows = []
    for candidate_id, alpha in [
        (292, 0.25),
        (293, 0.50),
        (294, 0.75),
        (295, 1.00),
        (296, 1.25),
    ]:
        out = geometric_blend(anchor265, probe284, alpha)
        file_name = f"stage2_candidate_{candidate_id}_v15_blend265_to_284_a{str(alpha).replace('.', 'p')}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "alpha_to_284": alpha,
                **pct_diagnostics(out, anchor265),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "v15_blend_265_284_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
