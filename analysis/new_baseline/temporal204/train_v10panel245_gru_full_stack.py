from __future__ import annotations

import json

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import RESULTS_DIR, SUBMISSIONS_DIR, TARGET, build_target_frame, load_data, mape
from train_new_baseline_gru_residual import center_by_group, momentum_bin
import train_temporal204_gru_residual as temporal_gru


OUT_DIR = RESULTS_DIR / "v10panel245_gru_full_stack"
BASELINE_FILE = "stage2_candidate_245_baseline_v10_panel_period_g060_clip020.csv"


def prepare_data() -> temporal_gru.PreparedData:
    raw, rto_label_wide, static_for_base = load_data()
    raw = raw.copy()
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    rto_wide = raw.pivot_table(index="new_id", columns="period", values=TARGET, aggfunc="mean")
    behavior_wide = {
        col: raw.pivot_table(index="new_id", columns="period", values=col, aggfunc="mean")
        for col in temporal_gru.BEHAVIOR_COLS
        if col in raw.columns
    }
    static_frame = temporal_gru.build_static(raw)

    train_features = build_target_frame(rto_label_wide, static_for_base, 2024, 3, include_target=True)
    test_features = build_target_frame(rto_label_wide, static_for_base, 2025, 3, include_target=False)

    # Use the leave-period-out v10 panel residual for train March-2024 to avoid target leakage.
    panel_oof = pd.read_csv(RESULTS_DIR / "temporal204_panel_residual_v10" / "oof_lpo.csv")
    train_panel = panel_oof[panel_oof["period"] == "2024-03"].copy()
    train_panel["panel_period_corr"] = train_panel["pred_panel_log"] - float(train_panel["pred_panel_log"].median())
    train_panel["new_base"] = train_panel["ref204"] * np.exp(np.clip(0.60 * train_panel["panel_period_corr"], -0.020, 0.020))
    train_features = train_features.merge(train_panel[["new_id", "new_base"]], on="new_id", how="inner")

    test_base = pd.read_csv(SUBMISSIONS_DIR / BASELINE_FILE).rename(columns={"rto": "new_base"})
    test_features = test_features.merge(test_base[["new_id", "new_base"]], on="new_id", how="inner")

    train_features = train_features[(train_features[TARGET] > 0) & (train_features["new_base"] > 0)].copy()
    train_features["target_log_ratio"] = np.log(train_features[TARGET] / train_features["new_base"])
    train_features["momentum_train_bin"] = momentum_bin(train_features)
    test_features["momentum_train_bin"] = momentum_bin(test_features)

    return temporal_gru.PreparedData(
        raw=raw,
        rto_wide=rto_wide,
        behavior_wide=behavior_wide,
        static_frame=static_frame,
        train_rows=train_features.sort_values("new_id").reset_index(drop=True),
        test_rows=test_features.sort_values("new_id").reset_index(drop=True),
    )


def write_candidates(data: temporal_gru.PreparedData, pred_log: np.ndarray) -> pd.DataFrame:
    rows = []
    test = data.test_rows.copy()
    test["pred_log_gru"] = pred_log
    test["pred_log_global"] = test["pred_log_gru"] - float(test["pred_log_gru"].median())
    test["pred_log_mombin"] = center_by_group(test["pred_log_gru"], test["momentum_train_bin"])
    base = test["new_base"].to_numpy(dtype=float)

    preds: dict[int, tuple[str, np.ndarray, str, float | None]] = {}
    for candidate_id, name, col, alpha in [
        (253, "v10panel245_newgru_global_a085", "pred_log_global", 0.85),
        (254, "v10panel245_newgru_global_a100", "pred_log_global", 1.00),
        (255, "v10panel245_newgru_global_a105", "pred_log_global", 1.05),
        (256, "v10panel245_newgru_mombin_a100", "pred_log_mombin", 1.00),
        (257, "v10panel245_newgru_raw_a100", "pred_log_gru", 1.00),
    ]:
        preds[candidate_id] = (name, base * np.exp(alpha * test[col].to_numpy(dtype=float)), col, alpha)

    transfer = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_118_baseline_cal050_mombin_g100.csv").rename(columns={"rto": "transfer118"})
    best159 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_159_baseline_faithful_global_blend_75_25_transfer.csv").rename(columns={"rto": "best159"})
    frame_blend = test[["new_id"]].merge(transfer, on="new_id").merge(best159, on="new_id")
    g100 = preds[254][1]
    preds[258] = ("v10panel245_newgru_global_a100_blend85_transfer118", 0.85 * g100 + 0.15 * frame_blend["transfer118"].to_numpy(float), "blend_transfer118", None)
    preds[259] = ("v10panel245_newgru_global_a100_blend75_transfer118", 0.75 * g100 + 0.25 * frame_blend["transfer118"].to_numpy(float), "blend_transfer118", None)
    preds[260] = ("v10panel245_newgru_global_a100_blend85_best159", 0.85 * g100 + 0.15 * frame_blend["best159"].to_numpy(float), "blend_best159", None)

    ref204 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_204_baseline_v6_temporal2_raw_clip085_135_a012_lc006.csv").rename(columns={"rto": "pred204"})
    ref220 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_220_baseline_temporal204_second_cb_on215_global_g010_clip005.csv").rename(columns={"rto": "pred220"})
    refs = test[["new_id"]].merge(ref204, on="new_id").merge(ref220, on="new_id")

    for candidate_id, (name, pred, center, alpha) in preds.items():
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        pred_frame = out.rename(columns={"rto": "pred"}).merge(refs, on="new_id")
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "alpha": alpha if alpha is not None else np.nan,
                "center": center,
                "test_mean_rto": float(np.mean(pred)),
                "test_total_rto": float(np.sum(pred)),
                "mean_abs_change_vs_base245_pct": float(np.mean(np.abs(pred / base - 1.0)) * 100),
                "mean_abs_change_vs_204_pct": float(np.mean(np.abs(pred_frame["pred"] / pred_frame["pred204"] - 1.0)) * 100),
                "mean_abs_change_vs_220_pct": float(np.mean(np.abs(pred_frame["pred"] / pred_frame["pred220"] - 1.0)) * 100),
                "p95_abs_change_vs_220_pct": float(np.quantile(np.abs(pred_frame["pred"] / pred_frame["pred220"] - 1.0), 0.95) * 100),
            }
        )
    out_summary = pd.DataFrame(rows)
    out_summary.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    return out_summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporal_gru.OUT_DIR = OUT_DIR
    print(f"Using device: {temporal_gru.DEVICE}")
    data = prepare_data()
    oof = temporal_gru.run_oof(data)
    oof.to_csv(OUT_DIR / "oof.csv", index=False)
    summary = temporal_gru.evaluate_oof(oof)
    summary.to_csv(OUT_DIR / "oof_summary.csv", index=False)
    pred_log = temporal_gru.train_final_and_predict(data)
    candidates = write_candidates(data, pred_log)
    metadata = {
        "base_test_file": BASELINE_FILE,
        "train_base": "v10 panel LPO period-centered gamma=0.60 clip=0.020 for 2024-03",
        "seq_len": temporal_gru.SEQ_LEN,
        "epochs": temporal_gru.EPOCHS,
        "final_epochs": temporal_gru.FINAL_EPOCHS,
        "seed": temporal_gru.SEED,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("OOF summary:")
    print(summary.head(30).to_string(index=False))
    print("\nSubmission candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
