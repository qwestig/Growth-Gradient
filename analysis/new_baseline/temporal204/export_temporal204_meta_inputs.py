from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def add_import_paths(workspace_root: Path | None) -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    paths = [here.parent, here.parent.parent, repo_root / "analysis" / "new_baseline", repo_root / "analysis" / "new_baseline" / "temporal204"]
    if workspace_root is not None:
        paths.append(workspace_root / "scripts")
    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def safe_log_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(num.astype(float), 1.0) / np.maximum(den.astype(float), 1.0))


def load_sub(submissions_dir: Path, candidate_id: int, col: str) -> pd.DataFrame:
    paths = sorted(submissions_dir.glob(f"stage2_candidate_{candidate_id:03d}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No submission for candidate {candidate_id:03d} in {submissions_dir}")
    return pd.read_csv(paths[0]).rename(columns={"rto": col})[["new_id", col]]


def build_tables(output_dir: Path, include_full_features: bool) -> tuple[Path, Path, Path]:
    from build_stacking_v11_candidates import add_stack_features, build_prediction_tables, crossfit_meta, feature_columns, segment_correction
    from heuristic_baseline_direction_lab import SUBMISSIONS_DIR, TARGET, mape

    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = build_prediction_tables()
    train = add_stack_features(train)
    test = add_stack_features(test)
    numeric, cats = feature_columns(train)

    train_corr_df, test_corr = crossfit_meta(train, test, numeric, cats)
    train_corrs = {col: train_corr_df[col].to_numpy(float) for col in train_corr_df.columns if col not in {"new_id", "fold"}}
    seg_train, seg_test = segment_correction(
        train,
        test,
        ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный", "momentum_train_bin", "volume_bin"],
        shrink_k=300.0,
    )

    hub_tr = train_corrs["huber030"] - np.median(train_corrs["huber030"])
    hub_te = test_corr["huber030"] - np.median(test_corr["huber030"])
    seg_tr = seg_train - np.median(seg_train)
    seg_te = seg_test - np.median(seg_test)
    combo_tr = 0.5 * hub_tr + 0.5 * seg_tr
    combo_te = 0.5 * hub_te + 0.5 * seg_te
    combo265_raw_tr = 0.5 * train_corrs["huber030"] + 0.5 * seg_train
    combo265_raw_te = 0.5 * test_corr["huber030"] + 0.5 * seg_test
    combo265_tr = combo265_raw_tr - np.median(combo265_raw_tr)
    combo265_te = combo265_raw_te - np.median(combo265_raw_te)

    c261_tr = np.clip(0.50 * hub_tr, -0.005, 0.005)
    c261_te = np.clip(0.50 * hub_te, -0.005, 0.005)
    c265_tr = np.clip(0.75 * combo265_tr, -0.005, 0.005)
    c265_te = np.clip(0.75 * combo265_te, -0.005, 0.005)
    c265_v14_internal_tr = np.clip(0.75 * combo_tr, -0.005, 0.005)
    c265_v14_internal_te = np.clip(0.75 * combo_te, -0.005, 0.005)

    pred220_tr = train["pred220"].to_numpy(float)
    pred220_te = test["pred220"].to_numpy(float)
    pred261_tr = pred220_tr * np.exp(c261_tr)
    pred261_te_calc = pred220_te * np.exp(c261_te)
    pred265_tr = pred220_tr * np.exp(c265_tr)
    pred265_te_calc = pred220_te * np.exp(c265_te)

    conf_tr = np.abs(combo_tr)
    conf_te = np.abs(combo_te)
    agree_tr = np.sign(hub_tr) == np.sign(seg_tr)
    agree_te = np.sign(hub_te) == np.sign(seg_te)
    top50_threshold = float(np.quantile(conf_tr, 0.50))
    top50_tr = conf_tr >= top50_threshold
    top50_te = conf_te >= top50_threshold
    agree_top50_tr = top50_tr & agree_tr
    agree_top50_te = top50_te & agree_te

    # Candidate 284/285-style v14 signals over safe 265 anchor.
    c284_tr = c265_v14_internal_tr.copy()
    c284_te = c265_v14_internal_te.copy()
    c284_tr[top50_tr] = np.clip(1.25 * combo_tr[top50_tr], -0.015, 0.015)
    c284_te[top50_te] = np.clip(1.25 * combo_te[top50_te], -0.015, 0.015)
    c285_tr = c265_v14_internal_tr.copy()
    c285_te = c265_v14_internal_te.copy()
    c285_tr[agree_top50_tr] = np.clip(1.25 * combo_tr[agree_top50_tr], -0.015, 0.015)
    c285_te[agree_top50_te] = np.clip(1.25 * combo_te[agree_top50_te], -0.015, 0.015)

    base_cols = [
        "new_id",
        "fold",
        TARGET,
        "pred204",
        "pred210",
        "pred118",
        "pred159",
        "pred215",
        "pred220",
    ]
    oof = train[base_cols].copy()
    oof = oof.rename(columns={TARGET: "target_rto"})
    oof["pred261_oof"] = pred261_tr
    oof["pred265_oof"] = pred265_tr
    oof["actual_log_residual_vs_265"] = safe_log_ratio(oof["target_rto"].to_numpy(float), oof["pred265_oof"].to_numpy(float))
    oof["actual_pct_residual_vs_265"] = oof["target_rto"].to_numpy(float) / np.maximum(oof["pred265_oof"].to_numpy(float), 1.0) - 1.0
    oof["ape_265"] = np.abs((oof["target_rto"].to_numpy(float) - oof["pred265_oof"].to_numpy(float)) / oof["target_rto"].to_numpy(float)) * 100.0

    test_out = test[["new_id", "pred204", "pred210", "pred118", "pred159", "pred215", "pred220"]].copy()
    test_out["pred261_calc"] = pred261_te_calc
    test_out["pred265_calc"] = pred265_te_calc
    for candidate_id, col in [(261, "pred261_submission"), (265, "pred265_submission"), (284, "pred284_submission"), (285, "pred285_submission")]:
        try:
            test_out = test_out.merge(load_sub(SUBMISSIONS_DIR, candidate_id, col), on="new_id", how="left")
        except FileNotFoundError:
            test_out[col] = np.nan

    signal_cols = {
        "v11_huber030_centered": (hub_tr, hub_te),
        "v11_segment_centered": (seg_tr, seg_te),
        "v11_combo_raw": (combo_tr, combo_te),
        "v11_combo265_centered": (combo265_tr, combo265_te),
        "v11_corr261_log": (c261_tr, c261_te),
        "v11_corr265_log": (c265_tr, c265_te),
        "v14_internal_corr265_log": (c265_v14_internal_tr, c265_v14_internal_te),
        "v14_conf_abs_combo": (conf_tr, conf_te),
        "v14_agree_huber_segment": (agree_tr.astype(int), agree_te.astype(int)),
        "v14_top50_mask": (top50_tr.astype(int), top50_te.astype(int)),
        "v14_agree_top50_mask": (agree_top50_tr.astype(int), agree_top50_te.astype(int)),
        "v14_corr284_log": (c284_tr, c284_te),
        "v14_corr285_log": (c285_tr, c285_te),
        "v14_delta284_vs_265_log": (c284_tr - c265_tr, c284_te - c265_te),
        "v14_delta285_vs_265_log": (c285_tr - c265_tr, c285_te - c265_te),
        "v14_delta284_vs_internal265_log": (c284_tr - c265_v14_internal_tr, c284_te - c265_v14_internal_te),
        "v14_delta285_vs_internal265_log": (c285_tr - c265_v14_internal_tr, c285_te - c265_v14_internal_te),
    }
    for col, (tr_values, te_values) in signal_cols.items():
        oof[col] = tr_values
        test_out[col] = te_values

    # Include original meta-layer row-level features so teammates can run new stacking/searches directly.
    meta_features = [c for c in numeric + cats if c in train.columns]
    if not include_full_features:
        keep = {
            "log_pred204_to_220",
            "log_pred210_to_220",
            "log_pred118_to_220",
            "log_pred159_to_220",
            "log_pred215_to_220",
            "log_215_to_210",
            "log_220_to_215",
            "abs_210_118_disagree",
            "momentum_raw",
            "feb_to_jan",
            "jan_to_dec",
            "lag1_to_lag12",
            "lag1_to_lag13",
            "store_mar_feb_ratio_prev",
            "volatility_rel3",
            "dip_depth",
            "dip_score",
            "temporary_dip_score",
            "structural_decline_score",
            "weak_score",
            "rebound_score",
            "momentum_train_bin",
            "volume_bin",
            "Регион",
            "Торговая площадь, категориальный",
            "Дата открытия, категориальный",
        }
        meta_features = [c for c in meta_features if c in keep]
    feature_prefix = "feature__"
    for col in meta_features:
        oof[f"{feature_prefix}{col}"] = train[col].to_numpy()
        test_out[f"{feature_prefix}{col}"] = test[col].to_numpy()

    oof = oof.sort_values("new_id").reset_index(drop=True)
    test_out = test_out.sort_values("new_id").reset_index(drop=True)
    oof_path = output_dir / "oof_meta_inputs_v11_v14.csv"
    test_path = output_dir / "test_meta_inputs_v11_v14.csv"
    oof.to_csv(oof_path, index=False)
    test_out.to_csv(test_path, index=False)

    metrics = pd.DataFrame(
        [
            {"variant": "pred204", "oof_mape": mape(train[TARGET], train["pred204"])},
            {"variant": "pred210", "oof_mape": mape(train[TARGET], train["pred210"])},
            {"variant": "pred215", "oof_mape": mape(train[TARGET], train["pred215"])},
            {"variant": "pred220", "oof_mape": mape(train[TARGET], train["pred220"])},
            {"variant": "pred261_oof", "oof_mape": mape(train[TARGET], pred261_tr)},
            {"variant": "pred265_oof", "oof_mape": mape(train[TARGET], pred265_tr)},
            {"variant": "v14_284_oof", "oof_mape": mape(train[TARGET], pred220_tr * np.exp(c284_tr))},
            {"variant": "v14_285_oof", "oof_mape": mape(train[TARGET], pred220_tr * np.exp(c285_tr))},
        ]
    )
    metrics["delta_vs_pred265"] = metrics["oof_mape"] - float(metrics.loc[metrics["variant"] == "pred265_oof", "oof_mape"].iloc[0])
    metrics_path = output_dir / "meta_input_oof_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    return oof_path, test_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export temporal204 v11/v14 row-level meta inputs for teammate search.")
    parser.add_argument("--workspace-root", type=Path, default=None, help="Full workspace root with scripts/results/submissions. Use when running from Growth-Gradient but relying on root artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[3] / "results" / "temporal204_meta_inputs")
    parser.add_argument("--full-features", action="store_true", help="Export every feature from feature_columns instead of the compact selected set.")
    args = parser.parse_args()
    add_import_paths(args.workspace_root)
    oof_path, test_path, metrics_path = build_tables(args.output_dir, include_full_features=args.full_features)
    print("Wrote:")
    print(oof_path)
    print(test_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
