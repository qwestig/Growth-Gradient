from __future__ import annotations

import numpy as np
import pandas as pd

from build_stacking_v11_candidates import add_stack_features, build_prediction_tables, crossfit_meta, feature_columns, segment_correction
from heuristic_baseline_direction_lab import RESULTS_DIR, SUBMISSIONS_DIR, TARGET, mape


OUT_DIR = RESULTS_DIR / "v14_selective_v11_correction"


def load_sub(candidate_id: int, col: str) -> pd.DataFrame:
    path = sorted(SUBMISSIONS_DIR.glob(f"stage2_candidate_{candidate_id:03d}_*.csv"))[0]
    return pd.read_csv(path).rename(columns={"rto": col})[["new_id", col]]


def bootstrap_delta(y: np.ndarray, ref: np.ndarray, pred: np.ndarray, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(2026)
    delta = np.abs((y - pred) / y) * 100 - np.abs((y - ref) / y) * 100
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    boot = delta[idx].mean(axis=1)
    return {"boot_p05": float(np.quantile(boot, 0.05)), "boot_p50": float(np.quantile(boot, 0.50)), "boot_p95": float(np.quantile(boot, 0.95)), "prob_improve": float((boot < 0).mean())}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_prediction_tables()
    train = add_stack_features(train)
    test = add_stack_features(test)
    numeric, cats = feature_columns(train)
    train_corr_df, test_corr = crossfit_meta(train, test, numeric, cats)
    train_corrs = {col: train_corr_df[col].to_numpy(float) for col in train_corr_df.columns if col not in {"new_id", "fold"}}
    seg_train, seg_test = segment_correction(train, test, ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный", "momentum_train_bin", "volume_bin"], shrink_k=300.0)

    hub_tr = train_corrs["huber030"] - np.median(train_corrs["huber030"])
    hub_te = test_corr["huber030"] - np.median(test_corr["huber030"])
    seg_tr = seg_train - np.median(seg_train)
    seg_te = seg_test - np.median(seg_test)
    combo_tr = 0.5 * hub_tr + 0.5 * seg_tr
    combo_te = 0.5 * hub_te + 0.5 * seg_te

    # Public-safe anchor 265: avg_huber030_segment, gamma=0.75, clip=0.005.
    c265_tr = np.clip(0.75 * combo_tr, -0.005, 0.005)
    c265_te = np.clip(0.75 * combo_te, -0.005, 0.005)

    conf_tr = np.abs(combo_tr)
    conf_te = np.abs(combo_te)
    agree_tr = np.sign(hub_tr) == np.sign(seg_tr)
    agree_te = np.sign(hub_te) == np.sign(seg_te)

    specs = []
    for q in [0.50, 0.60, 0.70, 0.80]:
        thr = float(np.quantile(conf_tr, q))
        for gamma in [1.00, 1.25, 1.50, 1.75]:
            for clip in [0.008, 0.010, 0.012, 0.015]:
                specs.append((f"top{int((1-q)*100):02d}_g{gamma:.2f}_c{clip:.3f}", thr, gamma, clip, False))
                specs.append((f"agree_top{int((1-q)*100):02d}_g{gamma:.2f}_c{clip:.3f}", thr, gamma, clip, True))

    y = train[TARGET].to_numpy(float)
    base_train = train["pred220"].to_numpy(float)
    base_test = test["pred220"].to_numpy(float)
    pred265_train = base_train * np.exp(c265_tr)
    rows = []
    test_preds = {}
    for name, thr, gamma, clip, require_agree in specs:
        mask_tr = conf_tr >= thr
        mask_te = conf_te >= thr
        if require_agree:
            mask_tr &= agree_tr
            mask_te &= agree_te
        c_tr = c265_tr.copy()
        c_te = c265_te.copy()
        c_tr[mask_tr] = np.clip(gamma * combo_tr[mask_tr], -clip, clip)
        c_te[mask_te] = np.clip(gamma * combo_te[mask_te], -clip, clip)
        pred_tr = base_train * np.exp(c_tr)
        pred_te = base_test * np.exp(c_te)
        ratio265 = pred_te / (base_test * np.exp(c265_te)) - 1.0
        boot = bootstrap_delta(y, pred265_train, pred_tr, n_boot=1000)
        rows.append(
            {
                "name": name,
                "oof_mape": mape(y, pred_tr),
                "delta_vs_265_oof": mape(y, pred_tr) - mape(y, pred265_train),
                "delta_vs_220_oof": mape(y, pred_tr) - mape(y, base_train),
                "train_share_changed": float(mask_tr.mean()),
                "test_share_changed": float(mask_te.mean()),
                "mean_abs_vs_265_pct": float(np.mean(np.abs(ratio265)) * 100),
                "p95_abs_vs_265_pct": float(np.quantile(np.abs(ratio265), 0.95) * 100),
                "max_abs_vs_265_pct": float(np.max(np.abs(ratio265)) * 100),
                **boot,
            }
        )
        test_preds[name] = pred_te

    summary = pd.DataFrame(rows).sort_values(["delta_vs_265_oof", "mean_abs_vs_265_pct"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "v14_search.csv", index=False)
    chosen = summary[
        (summary["delta_vs_265_oof"] < -0.0005)
        & (summary["mean_abs_vs_265_pct"].between(0.05, 0.60))
        & (summary["p95_abs_vs_265_pct"] <= 1.50)
        & (summary["boot_p95"] < 0.001)
    ].head(8)
    if chosen.empty:
        chosen = summary.head(8)
    out_rows = []
    for candidate_id, (_, row) in enumerate(chosen.iterrows(), start=284):
        pred = test_preds[row["name"]]
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        safe = row["name"].replace(".", "p")
        file_name = f"stage2_candidate_{candidate_id:03d}_full_v14_selective_{safe}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        out_rows.append({"candidate_id": candidate_id, "file": file_name, **row.to_dict()})
    candidates = pd.DataFrame(out_rows)
    candidates.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    print("Top v14 search:")
    print(summary.head(30).to_string(index=False))
    print("\nCandidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
