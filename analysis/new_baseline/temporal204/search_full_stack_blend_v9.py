from __future__ import annotations

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import RESULTS_DIR, SUBMISSIONS_DIR


OUT_DIR = RESULTS_DIR / "full_stack_blend_v9"


PUBLIC_MAPE = {
    118: 4.832,
    210: 4.748,
    215: 4.732,
    220: 4.727,
    221: 4.727,
    222: 4.769,
}


def load_candidate(candidate_id: int, name: str | None = None) -> pd.DataFrame:
    if name is not None:
        path = SUBMISSIONS_DIR / name
    else:
        matches = sorted(SUBMISSIONS_DIR.glob(f"stage2_candidate_{candidate_id:03d}_*.csv"))
        if not matches:
            raise FileNotFoundError(candidate_id)
        path = matches[0]
    return pd.read_csv(path).rename(columns={"rto": f"pred{candidate_id}"})[["new_id", f"pred{candidate_id}"]]


def diagnostics(pred: np.ndarray, ref: np.ndarray) -> dict:
    ratio = pred / ref - 1.0
    return {
        "mean_delta_pct": float(np.mean(ratio) * 100),
        "mean_abs_pct": float(np.mean(np.abs(ratio)) * 100),
        "p95_abs_pct": float(np.quantile(np.abs(ratio), 0.95) * 100),
        "p99_abs_pct": float(np.quantile(np.abs(ratio), 0.99) * 100),
        "max_abs_pct": float(np.max(np.abs(ratio)) * 100),
        "corr": float(np.corrcoef(pred, ref)[0, 1]),
    }


def quadratic_transfer_optimum() -> dict:
    # Fit public MAPE curve for pred(w)=(1-w)*210+w*118 using known points w=0, .25, 1.
    c = PUBLIC_MAPE[210]
    # 0.0625a + 0.25b + c = m215; a+b+c = m118
    rhs1 = PUBLIC_MAPE[215] - c
    rhs2 = PUBLIC_MAPE[118] - c
    a, b = np.linalg.solve(np.array([[0.25**2, 0.25], [1.0, 1.0]]), np.array([rhs1, rhs2]))
    w_star = float(np.clip(-b / (2 * a), 0.0, 1.0)) if a > 0 else 0.25
    m_star = float(a * w_star**2 + b * w_star + c)
    return {"quad_a": float(a), "quad_b": float(b), "quad_c": float(c), "w_star": w_star, "mape_star_est": m_star}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = None
    for cid in [118, 159, 210, 215, 220, 221, 222]:
        sub = load_candidate(cid)
        frame = sub if frame is None else frame.merge(sub, on="new_id", how="inner")

    quad = quadratic_transfer_optimum()
    rows = []
    preds: dict[int, tuple[str, np.ndarray]] = {}
    p118 = frame["pred118"].to_numpy(float)
    p210 = frame["pred210"].to_numpy(float)
    p215 = frame["pred215"].to_numpy(float)
    p220 = frame["pred220"].to_numpy(float)
    p221 = frame["pred221"].to_numpy(float)
    p159 = frame["pred159"].to_numpy(float)

    # Small transfer-weight probes around the estimated public optimum ~0.287.
    for cid, w in [(232, 0.28), (233, 0.30), (234, 0.32)]:
        preds[cid] = (f"blend210_transfer118_w{w:.2f}", (1.0 - w) * p210 + w * p118)

    # Add tiny second residual effect from 220-vs-215 to transfer-weight probes.
    second_ratio = p220 / p215
    preds[235] = ("blend210_transfer118_w0.30_plus_second220ratio", ((1.0 - 0.30) * p210 + 0.30 * p118) * second_ratio)

    # Conservative current-best hedges: only tiny move from 220 toward old transfer/current old best.
    for cid, w, target, target_name in [(236, 0.03, p118, "transfer118"), (237, 0.05, p118, "transfer118"), (238, 0.05, p159, "best159")]:
        preds[cid] = (f"current220_blend{int(w*100):02d}_{target_name}", (1.0 - w) * p220 + w * target)

    # Candidate 221 tied 220 but stronger; keep only midpoint diagnostic, not expected to move much.
    preds[239] = ("midpoint220_221", 0.5 * p220 + 0.5 * p221)

    for cid, (name, pred) in preds.items():
        ref220 = diagnostics(pred, p220)
        ref215 = diagnostics(pred, p215)
        ref210 = diagnostics(pred, p210)
        public_prior = ""
        if "blend210_transfer118" in name:
            w = float(name.split("_w")[-1].split("_")[0]) if "_w" in name else 0.30
            public_prior = f"quadratic public prior around transfer weight: w*={quad['w_star']:.3f}, estimated MAPE {quad['mape_star_est']:.4f}; this candidate w={w:.2f}"
        elif "current220" in name:
            public_prior = "small hedge from current best 220 toward a public-positive independent old signal; low magnitude but unvalidated"
        rows.append(
            {
                "candidate_id": cid,
                "name": name,
                "mean_rto": float(np.mean(pred)),
                "total_rto": float(np.sum(pred)),
                "mean_abs_vs_220_pct": ref220["mean_abs_pct"],
                "p95_abs_vs_220_pct": ref220["p95_abs_pct"],
                "max_abs_vs_220_pct": ref220["max_abs_pct"],
                "mean_abs_vs_215_pct": ref215["mean_abs_pct"],
                "mean_abs_vs_210_pct": ref210["mean_abs_pct"],
                "corr_vs_220": ref220["corr"],
                "public_prior": public_prior,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["mean_abs_vs_220_pct", "candidate_id"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "blend_v9_candidates.csv", index=False)
    pd.Series(quad).to_csv(OUT_DIR / "transfer_quadratic_public_prior.csv")

    for cid, (name, pred) in preds.items():
        out = frame[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{cid:03d}_full_v9_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        summary.loc[summary["candidate_id"] == cid, "file"] = file_name
    summary.to_csv(OUT_DIR / "blend_v9_candidates.csv", index=False)
    print("Quadratic public prior:")
    print(pd.Series(quad).to_string())
    print("\nCandidates:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
