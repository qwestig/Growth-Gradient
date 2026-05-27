from __future__ import annotations

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import RESULTS_DIR, SUBMISSIONS_DIR


OUT_DIR = RESULTS_DIR / "temporal204_panel_residual_v10"


def load(file_name: str, col: str) -> pd.DataFrame:
    return pd.read_csv(SUBMISSIONS_DIR / file_name).rename(columns={"rto": col})[["new_id", col]]


def main() -> None:
    ref204 = load("stage2_candidate_204_baseline_v6_temporal2_raw_clip085_135_a012_lc006.csv", "pred204")
    best220 = load("stage2_candidate_220_baseline_temporal204_second_cb_on215_global_g010_clip005.csv", "pred220")
    frame = ref204.merge(best220, on="new_id")
    sources = [
        (244, "period_g045_clip020"),
        (245, "period_g060_clip020"),
        (246, "period_g080_clip030"),
    ]
    specs = [
        (248, 244, 0.50, "current220_panel244_ratio_g050"),
        (249, 244, 0.75, "current220_panel244_ratio_g075"),
        (250, 245, 0.50, "current220_panel245_ratio_g050"),
        (251, 245, 0.75, "current220_panel245_ratio_g075"),
        (252, 246, 0.35, "current220_panel246_ratio_g035"),
    ]
    source_files = {
        244: "stage2_candidate_244_baseline_v10_panel_period_g045_clip020.csv",
        245: "stage2_candidate_245_baseline_v10_panel_period_g060_clip020.csv",
        246: "stage2_candidate_246_baseline_v10_panel_period_g080_clip030.csv",
    }
    for source_id, _name in sources:
        src = load(source_files[source_id], f"pred{source_id}")
        frame = frame.merge(src, on="new_id")

    rows = []
    pred220 = frame["pred220"].to_numpy(float)
    pred204 = frame["pred204"].to_numpy(float)
    for candidate_id, source_id, gamma, name in specs:
        src = frame[f"pred{source_id}"].to_numpy(float)
        corr = np.log(np.maximum(src, 1.0) / np.maximum(pred204, 1.0))
        pred = pred220 * np.exp(gamma * corr)
        out = frame[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_full_v10_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        ratio220 = pred / pred220 - 1.0
        ratio204 = pred / pred204 - 1.0
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "source_baseline": source_id,
                "gamma_on_panel_ratio": gamma,
                "mean_abs_vs_220_pct": float(np.mean(np.abs(ratio220)) * 100),
                "p95_abs_vs_220_pct": float(np.quantile(np.abs(ratio220), 0.95) * 100),
                "max_abs_vs_220_pct": float(np.max(np.abs(ratio220)) * 100),
                "mean_delta_vs_220_pct": float(np.mean(ratio220) * 100),
                "mean_abs_vs_204_pct": float(np.mean(np.abs(ratio204)) * 100),
                "mean_delta_vs_204_pct": float(np.mean(ratio204) * 100),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "full_from_panel_ratio_candidates.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
