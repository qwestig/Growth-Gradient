from __future__ import annotations

import subprocess
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heuristic_baseline_direction_lab import RESULTS_DIR, STATIC_CAT_COLS, STATIC_NUM_COLS, SUBMISSIONS_DIR, TARGET, HeuristicSpec, build_target_frame, factors_for_spec, load_data, mape, predict_from_spec
from train_new_baseline_gru_residual import center_by_group
from build_new_baseline_residual_transfer_candidates import MOMENTUM_TRAIN_BINS, read_remote_csv


OUT_DIR = RESULTS_DIR / "stacking_v11"
REMOTE_BRANCH = "origin/codex-momentum-bias-solution"


def read_remote(path: str) -> pd.DataFrame:
    local_path = RESULTS_DIR.parents[0] / path
    if local_path.exists():
        return pd.read_csv(local_path)
    result = subprocess.run(
        ["git", "-C", "Growth-Gradient", "show", f"{REMOTE_BRANCH}:{path}"],
        check=True,
        text=True,
        capture_output=True,
    )
    return pd.read_csv(StringIO(result.stdout))


def momentum_bin(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame["momentum_raw"],
        bins=MOMENTUM_TRAIN_BINS,
        labels=[f"q{i}" for i in range(5)],
        include_lowest=True,
    ).astype(str)


def make_base(frame: pd.DataFrame, rto: pd.DataFrame, static: pd.DataFrame, spec: HeuristicSpec, is_test: bool) -> np.ndarray:
    val_factor, test_factor, _val_global, _final_global = factors_for_spec(spec, frame, frame, rto, static)
    factor = test_factor if is_test else val_factor
    return predict_from_spec(frame, factor, spec)


def load_submission(candidate_id: int, col: str) -> pd.DataFrame:
    path = sorted(SUBMISSIONS_DIR.glob(f"stage2_candidate_{candidate_id:03d}_*.csv"))[0]
    return pd.read_csv(path).rename(columns={"rto": col})[["new_id", col]]


def build_prediction_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    _raw, rto, static = load_data()
    val_frame = build_target_frame(rto, static, 2024, 3, include_target=True)
    test_frame = build_target_frame(rto, static, 2025, 3, include_target=False)

    val = val_frame.copy()
    test = test_frame.copy()
    val["momentum_train_bin"] = momentum_bin(val)
    test["momentum_train_bin"] = momentum_bin(test)

    # Candidate 204 / 210 OOF.
    gru_oof = pd.read_csv(RESULTS_DIR / "temporal204_gru_residual" / "oof.csv")
    gru_oof["pred204"] = gru_oof["new_base"]
    gru_oof["pred210"] = gru_oof["new_base"] * np.exp(gru_oof["pred_log_gru"] - float(gru_oof["pred_log_gru"].median()))
    val = val.merge(gru_oof[["new_id", "pred204", "pred210", "fold"]], on="new_id", how="inner")

    # Candidate 118 OOF: old GRU transfer residual on cal050_mom000 baseline, momentum-bin centered gamma=1.
    old_oof = read_remote_csv("results/gru_residual/gru_oof.csv")[["new_id", "pred_log_gru"]]
    old_spec = HeuristicSpec(0, "old_raw_mom025", 0.0, momentum_exp=0.25)
    cal050_spec = HeuristicSpec(0, "cal050_mom000", 0.50, momentum_exp=0.0)
    cal050_val = make_base(val_frame, rto, static, cal050_spec, is_test=False)
    old_base_val = make_base(val_frame, rto, static, old_spec, is_test=False)
    old_oof = val_frame[["new_id"]].assign(cal050_base=cal050_val, old_base=old_base_val).merge(old_oof, on="new_id", how="inner")
    old_oof = old_oof.merge(val[["new_id", "momentum_train_bin"]], on="new_id", how="left")
    old_oof["old_ratio_log"] = 0.70 * old_oof["pred_log_gru"].to_numpy(float)
    old_oof["old_ratio_log_mombin"] = center_by_group(old_oof["old_ratio_log"], old_oof["momentum_train_bin"])
    old_oof["pred118"] = old_oof["cal050_base"] * np.exp(old_oof["old_ratio_log_mombin"])
    val = val.merge(old_oof[["new_id", "pred118"]], on="new_id", how="inner")

    # Candidate 159 OOF approximation: 75% old new-baseline GRU global + 25% candidate 118 transfer.
    nb_oof = pd.read_csv(RESULTS_DIR / "new_baseline_gru_residual" / "oof.csv")
    nb_oof["pred150"] = nb_oof["new_base"] * np.exp(nb_oof["pred_log_gru"] - float(nb_oof["pred_log_gru"].median()))
    val = val.merge(nb_oof[["new_id", "pred150"]], on="new_id", how="inner")
    val["pred159"] = 0.75 * val["pred150"] + 0.25 * val["pred118"]

    # Candidate 215 and 220 OOF approximations.
    val["pred215"] = 0.75 * val["pred210"] + 0.25 * val["pred118"]
    second_oof = pd.read_csv(RESULTS_DIR / "temporal204_second_residual" / "oof.csv")[["new_id", "pred_second_log_cb"]]
    second_oof["second_global"] = second_oof["pred_second_log_cb"] - float(second_oof["pred_second_log_cb"].median())
    val = val.merge(second_oof[["new_id", "second_global"]], on="new_id", how="inner")
    val["pred220"] = val["pred215"] * np.exp(np.clip(0.10 * val["second_global"].to_numpy(float), -0.005, 0.005))

    for cid, col in [(204, "pred204"), (210, "pred210"), (118, "pred118"), (159, "pred159"), (215, "pred215"), (220, "pred220")]:
        test = test.merge(load_submission(cid, col), on="new_id", how="inner")
    test["fold"] = test["new_id"] % 5
    return val, test


def add_stack_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["pred204", "pred210", "pred118", "pred159", "pred215"]:
        out[f"log_{col}_to_220"] = np.log(np.maximum(out[col].to_numpy(float), 1.0) / np.maximum(out["pred220"].to_numpy(float), 1.0))
    out["log_215_to_210"] = np.log(np.maximum(out["pred215"].to_numpy(float), 1.0) / np.maximum(out["pred210"].to_numpy(float), 1.0))
    out["log_220_to_215"] = np.log(np.maximum(out["pred220"].to_numpy(float), 1.0) / np.maximum(out["pred215"].to_numpy(float), 1.0))
    out["abs_210_118_disagree"] = np.abs(np.log(np.maximum(out["pred210"].to_numpy(float), 1.0) / np.maximum(out["pred118"].to_numpy(float), 1.0)))
    out["log_lag1"] = np.log(np.clip(out["lag1"], 1.0, None))
    out["weak_score"] = ((0.98 - out["momentum_raw"].fillna(1.0)) / 0.08).clip(0.0, 1.0)
    out["rebound_score"] = out["weak_score"] * out["dip_score"].fillna(0.0)
    out["volume_bin"] = pd.qcut(out["lag1"].rank(method="first"), 5, labels=[f"v{i}" for i in range(5)]).astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        "log_pred204_to_220",
        "log_pred210_to_220",
        "log_pred118_to_220",
        "log_pred159_to_220",
        "log_pred215_to_220",
        "log_215_to_210",
        "log_220_to_215",
        "abs_210_118_disagree",
        "lag1",
        "lag2",
        "lag3",
        "lag12",
        "lag13",
        "rolling3_mean",
        "rolling2_pre_mean",
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
        "log_lag1",
        "weak_score",
        "rebound_score",
    ]
    numeric += [col for col in STATIC_NUM_COLS if col in frame.columns]
    cats = ["momentum_train_bin", "volume_bin"] + [col for col in STATIC_CAT_COLS if col in frame.columns]
    return [col for col in numeric if col in frame.columns], [col for col in cats if col in frame.columns]


def make_model(kind: str, numeric: list[str], cats: list[str]):
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)
    pre = ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), encoder), cats),
        ],
        remainder="drop",
    )
    if kind.startswith("ridge"):
        alpha = float(kind.replace("ridge", ""))
        model = Ridge(alpha=alpha, random_state=2026)
    elif kind == "huber010":
        model = HuberRegressor(epsilon=1.20, alpha=0.10, max_iter=300)
    elif kind == "huber030":
        model = HuberRegressor(epsilon=1.35, alpha=0.30, max_iter=300)
    else:
        raise ValueError(kind)
    return make_pipeline(pre, model)


def crossfit_meta(train: pd.DataFrame, test: pd.DataFrame, numeric: list[str], cats: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y = np.log(train[TARGET].to_numpy(float) / np.maximum(train["pred220"].to_numpy(float), 1.0))
    kinds = ["ridge10", "ridge30", "ridge100", "huber010", "huber030"]
    train_corr = pd.DataFrame({"new_id": train["new_id"], "fold": train["fold"]})
    test_corr = {}
    for kind in kinds:
        oof = np.zeros(len(train), dtype=float)
        for fold in sorted(train["fold"].unique()):
            tr_mask = train["fold"] != fold
            va_mask = train["fold"] == fold
            model = make_model(kind, numeric, cats)
            model.fit(train.loc[tr_mask, numeric + cats], y[tr_mask])
            oof[va_mask] = model.predict(train.loc[va_mask, numeric + cats])
        final = make_model(kind, numeric, cats)
        final.fit(train[numeric + cats], y)
        train_corr[kind] = oof
        test_corr[kind] = final.predict(test[numeric + cats])
    return train_corr, test_corr


def segment_correction(train: pd.DataFrame, test: pd.DataFrame, segment_cols: list[str], shrink_k: float) -> tuple[np.ndarray, np.ndarray]:
    resid = np.log(train[TARGET].to_numpy(float) / np.maximum(train["pred220"].to_numpy(float), 1.0))
    train_out = np.zeros(len(train), dtype=float)
    test_parts = []
    global_med = float(np.median(resid))
    for col in segment_cols:
        oof_col = np.zeros(len(train), dtype=float)
        for fold in sorted(train["fold"].unique()):
            tr = train[train["fold"] != fold].copy()
            va = train[train["fold"] == fold].copy()
            tr_resid = pd.Series(resid[train["fold"] != fold], index=tr.index)
            stats = tr.assign(_resid=tr_resid).groupby(col, observed=True)["_resid"].agg(["median", "count"])
            shrink = stats["count"] / (stats["count"] + shrink_k)
            mapping = (stats["median"] * shrink).to_dict()
            oof_col[train["fold"] == fold] = va[col].map(mapping).fillna(global_med).to_numpy(float)
        stats_all = train.assign(_resid=resid).groupby(col, observed=True)["_resid"].agg(["median", "count"])
        shrink_all = stats_all["count"] / (stats_all["count"] + shrink_k)
        mapping_all = (stats_all["median"] * shrink_all).to_dict()
        test_part = test[col].map(mapping_all).fillna(global_med).to_numpy(float)
        train_out += oof_col / len(segment_cols)
        test_parts.append(test_part)
    return train_out, np.mean(test_parts, axis=0)


def evaluate_and_write(train: pd.DataFrame, test: pd.DataFrame, train_corrs: dict[str, np.ndarray], test_corrs: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    test_preds: dict[str, np.ndarray] = {}
    base_oof = train["pred220"].to_numpy(float)
    y = train[TARGET].to_numpy(float)
    base_test = test["pred220"].to_numpy(float)
    rejected250 = load_submission(250, "pred250") if list(SUBMISSIONS_DIR.glob("stage2_candidate_250_*.csv")) else None
    rejected_corr = None
    if rejected250 is not None:
        tmp = test[["new_id", "pred220"]].merge(rejected250, on="new_id", how="inner")
        rejected_corr = np.log(tmp["pred250"].to_numpy(float) / tmp["pred220"].to_numpy(float))

    specs = []
    for name in train_corrs:
        for gamma in [0.25, 0.50, 0.75]:
            for clip in [0.002, 0.003, 0.005]:
                specs.append((f"{name}_g{gamma:.2f}_c{clip:.3f}", train_corrs[name], test_corrs[name], gamma, clip))
    for name_a, name_b in [("ridge30", "segment"), ("huber030", "segment")]:
        if name_a in train_corrs and name_b in train_corrs:
            specs.append((f"avg_{name_a}_{name_b}_g0.50_c0.003", 0.5 * train_corrs[name_a] + 0.5 * train_corrs[name_b], 0.5 * test_corrs[name_a] + 0.5 * test_corrs[name_b], 0.50, 0.003))
            specs.append((f"avg_{name_a}_{name_b}_g0.75_c0.005", 0.5 * train_corrs[name_a] + 0.5 * train_corrs[name_b], 0.5 * test_corrs[name_a] + 0.5 * test_corrs[name_b], 0.75, 0.005))

    for name, tr_corr_raw, te_corr_raw, gamma, clip in specs:
        tr_corr = np.clip(gamma * (tr_corr_raw - np.median(tr_corr_raw)), -clip, clip)
        te_corr = np.clip(gamma * (te_corr_raw - np.median(te_corr_raw)), -clip, clip)
        pred_oof = base_oof * np.exp(tr_corr)
        pred_test = base_test * np.exp(te_corr)
        ratio = pred_test / base_test - 1.0
        row = {
            "name": name,
            "oof_mape": mape(y, pred_oof),
            "delta_vs_220_oof": mape(y, pred_oof) - mape(y, base_oof),
            "mean_abs_vs_220_pct": float(np.mean(np.abs(ratio)) * 100),
            "p95_abs_vs_220_pct": float(np.quantile(np.abs(ratio), 0.95) * 100),
            "max_abs_vs_220_pct": float(np.max(np.abs(ratio)) * 100),
            "mean_delta_vs_220_pct": float(np.mean(ratio) * 100),
            "corr_with_250_corr": float(np.corrcoef(te_corr, rejected_corr)[0, 1]) if rejected_corr is not None and np.std(te_corr) > 0 else np.nan,
        }
        rows.append(row)
        test_preds[name] = pred_test
    summary = pd.DataFrame(rows).sort_values(["delta_vs_220_oof", "mean_abs_vs_220_pct"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "stacking_v11_search.csv", index=False)

    chosen = summary[
        (summary["delta_vs_220_oof"] < -0.0005)
        & (summary["mean_abs_vs_220_pct"] >= 0.05)
        & (summary["mean_abs_vs_220_pct"] <= 0.40)
        & (summary["p95_abs_vs_220_pct"] <= 1.20)
    ].head(6).copy()
    if chosen.empty:
        chosen = summary.head(6).copy()
    rows_out = []
    for offset, (_, row) in enumerate(chosen.iterrows(), start=261):
        pred = test_preds[row["name"]]
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        safe_name = row["name"].replace(".", "p")
        file_name = f"stage2_candidate_{offset:03d}_full_v11_stack_{safe_name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows_out.append({"candidate_id": offset, "file": file_name, **row.to_dict()})
    candidates = pd.DataFrame(rows_out)
    candidates.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    return candidates


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_prediction_tables()
    train = add_stack_features(train)
    test = add_stack_features(test)
    numeric, cats = feature_columns(train)
    train_corr_df, test_corr = crossfit_meta(train, test, numeric, cats)
    train_corrs = {col: train_corr_df[col].to_numpy(float) for col in train_corr_df.columns if col not in {"new_id", "fold"}}
    seg_train, seg_test = segment_correction(train, test, ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный", "momentum_train_bin", "volume_bin"], shrink_k=300.0)
    train_corrs["segment"] = seg_train
    test_corr["segment"] = seg_test

    diagnostics = []
    base_mape = mape(train[TARGET], train["pred220"])
    for col in ["pred204", "pred210", "pred118", "pred159", "pred215", "pred220"]:
        diagnostics.append({"prediction": col, "oof_mape": mape(train[TARGET], train[col]), "delta_vs_220": mape(train[TARGET], train[col]) - base_mape})
    pd.DataFrame(diagnostics).to_csv(OUT_DIR / "oof_prediction_diagnostics.csv", index=False)
    candidates = evaluate_and_write(train, test, train_corrs, test_corr)
    print("OOF prediction diagnostics:")
    print(pd.DataFrame(diagnostics).to_string(index=False))
    print("\nSubmission candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
