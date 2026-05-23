from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from catboost_residual_diagnostics import TARGET, add_base, apply_residual, make_panel, mape
from gru_residual_experiment import (
    BATCH_SIZE,
    GRUResidual,
    SEED,
    SEQ_LEN,
    SMALL_CAT_COLS,
    STATIC_NUM_COLS,
    SUBMISSIONS_DIR,
    apply_sequence_scaler,
    build_sequence,
    fit_sequence_scaler,
    fit_static_encoders,
    period_id,
    prepare_data,
    set_seed,
    train_one_fold,
    transform_static,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "gru_seed_ensemble"
SEEDS = [2026, 3026, 4026]
ALPHAS = [0.60, 0.65, 0.70, 0.75]


def evaluate_alpha(frame: pd.DataFrame, pred_col: str, alpha: float) -> float:
    pred = apply_residual(
        frame["best_base"].to_numpy(dtype=float),
        frame[pred_col].to_numpy(dtype=float),
        alpha,
        None,
    )
    return mape(frame[TARGET], pred)


def run_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepare_data()
    target = data.target_rows.copy()
    store_ids = target["new_id"].to_numpy()
    y = target["target_log_ratio"].to_numpy(dtype=np.float32)
    full_seq = build_sequence(data, store_ids, period_id(2024, 3))

    fold_rows = []
    oof_parts = []
    for fold in range(5):
        valid_mask = (store_ids % 5) == fold
        train_ids = store_ids[~valid_mask]
        valid_ids = store_ids[valid_mask]
        train_seq_raw = full_seq[~valid_mask]
        valid_seq_raw = full_seq[valid_mask]
        seq_mean, seq_std = fit_sequence_scaler(train_seq_raw)
        train_seq = apply_sequence_scaler(train_seq_raw, seq_mean, seq_std)
        valid_seq = apply_sequence_scaler(valid_seq_raw, seq_mean, seq_std)

        num_mean, num_std, encoders = fit_static_encoders(data.static_frame, train_ids)
        train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders)
        valid_num, valid_cat = transform_static(data.static_frame, valid_ids, num_mean, num_std, encoders)
        cat_cardinalities = [len(encoders[col]) for col in SMALL_CAT_COLS]

        valid_rows = target.loc[valid_mask, ["new_id", TARGET, "best_base"]].copy()
        seed_preds = []
        seed_mapes = {}
        for seed in SEEDS:
            pred_log, best_loss = train_one_fold(
                (train_seq, train_num, train_cat, y[~valid_mask]),
                (valid_seq, valid_num, valid_cat, y[valid_mask]),
                cat_cardinalities,
                seed + fold,
            )
            col = f"pred_log_seed_{seed}"
            valid_rows[col] = pred_log
            seed_preds.append(pred_log)
            seed_mapes[f"seed_{seed}_alpha070_mape"] = evaluate_alpha(
                valid_rows.rename(columns={col: "pred_log_tmp"}), "pred_log_tmp", 0.70
            )
            seed_mapes[f"seed_{seed}_valid_loss"] = best_loss

        valid_rows["pred_log_ensemble"] = np.mean(np.stack(seed_preds, axis=0), axis=0)
        valid_rows["fold"] = fold
        row = {"fold": fold, "base_mape": mape(valid_rows[TARGET], valid_rows["best_base"]), **seed_mapes}
        for alpha in ALPHAS:
            row[f"ensemble_alpha_{alpha:.2f}_mape"] = evaluate_alpha(valid_rows, "pred_log_ensemble", alpha)
        fold_rows.append(row)
        oof_parts.append(valid_rows)

    return pd.DataFrame(fold_rows), pd.concat(oof_parts, ignore_index=True)


def train_final_model(seed: int, data):
    set_seed(seed)
    train_target = data.target_rows.copy()
    train_ids = train_target["new_id"].to_numpy()
    y_train = train_target["target_log_ratio"].to_numpy(dtype=np.float32)
    train_seq_raw = build_sequence(data, train_ids, period_id(2024, 3))
    seq_mean, seq_std = fit_sequence_scaler(train_seq_raw)
    train_seq = apply_sequence_scaler(train_seq_raw, seq_mean, seq_std)
    num_mean, num_std, encoders = fit_static_encoders(data.static_frame, train_ids)
    train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders)
    cat_cardinalities = [len(encoders[col]) for col in SMALL_CAT_COLS]

    model = GRUResidual(train_seq.shape[-1], train_num.shape[1], cat_cardinalities)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)
    dataset = TensorDataset(
        torch.from_numpy(train_seq),
        torch.from_numpy(train_num),
        torch.from_numpy(train_cat),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(35):
        model.train()
        for batch_seq, batch_num, batch_cat, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_seq, batch_num, batch_cat), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model, (seq_mean, seq_std), (num_mean, num_std, encoders)


def predict_test_logs(model_bundle, data, pred_rows: pd.DataFrame, pred_ids: np.ndarray) -> np.ndarray:
    model, (seq_mean, seq_std), (num_mean, num_std, encoders) = model_bundle
    pred_seq = apply_sequence_scaler(build_sequence(data, pred_ids, period_id(2025, 3)), seq_mean, seq_std)
    pred_num, pred_cat = transform_static(data.static_frame, pred_ids, num_mean, num_std, encoders)
    with torch.no_grad():
        return model(torch.from_numpy(pred_seq), torch.from_numpy(pred_num), torch.from_numpy(pred_cat)).numpy()


def make_test_submissions(best_alpha: float) -> list[str]:
    data = prepare_data()
    feb_2025 = data.raw[(data.raw["Год"] == 2025) & (data.raw["Месяц"] == 2)].copy()
    mar_2025 = feb_2025.copy()
    mar_2025["Год"] = 2025
    mar_2025["Месяц"] = 3
    mar_2025["period"] = period_id(2025, 3)
    mar_2025[TARGET] = np.nan
    pred_panel = add_base(make_panel(pd.concat([data.raw, mar_2025], ignore_index=True)), data.final_factor)
    pred_rows = pred_panel[(pred_panel["Год"] == 2025) & (pred_panel["Месяц"] == 3)].copy()
    pred_rows = pred_rows.sort_values("new_id").reset_index(drop=True)
    pred_ids = pred_rows["new_id"].to_numpy()

    pred_logs = []
    for seed in SEEDS:
        bundle = train_final_model(seed, data)
        pred_logs.append(predict_test_logs(bundle, data, pred_rows, pred_ids))
    pred_log = np.mean(np.stack(pred_logs, axis=0), axis=0)

    candidate_alphas = sorted({0.70, best_alpha})
    paths = []
    rows = []
    base = pred_rows["best_base"].to_numpy(dtype=float)
    for alpha in candidate_alphas:
        pred = apply_residual(base, pred_log, alpha, None)
        submission = pred_rows[["new_id"]].copy()
        submission["rto"] = np.round(np.maximum(pred, 0), 2)
        path = SUBMISSIONS_DIR / f"test_gru_ensemble_alpha{int(alpha * 100):03d}.csv"
        submission.to_csv(path, index=False)
        corr = pred / base
        rows.append(
            {
                "alpha": alpha,
                "file": str(path),
                "rows": len(submission),
                "missing": int(submission.isna().sum().sum()),
                "nonpositive": int((submission["rto"] <= 0).sum()),
                "mean_abs_corr_pct": float(np.mean(np.abs(corr - 1)) * 100),
                "p99_corr": float(pd.Series(corr).quantile(0.99)),
                "max_corr": float(corr.max()),
                "min_corr": float(corr.min()),
            }
        )
        paths.append(str(path))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "test_candidates.csv", index=False)
    return paths


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    folds, oof = run_oof()
    folds.to_csv(RESULTS_DIR / "ensemble_folds.csv", index=False)
    oof.to_csv(RESULTS_DIR / "ensemble_oof.csv", index=False)

    rows = []
    for alpha in ALPHAS:
        rows.append({"model": "gru_seed_ensemble", "alpha": alpha, "mape": evaluate_alpha(oof, "pred_log_ensemble", alpha)})
    for seed in SEEDS:
        rows.append({"model": f"single_seed_{seed}", "alpha": 0.70, "mape": evaluate_alpha(oof, f"pred_log_seed_{seed}", 0.70)})
    comparison = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
    comparison.to_csv(RESULTS_DIR / "ensemble_comparison.csv", index=False)
    best_alpha = float(comparison[comparison["model"] == "gru_seed_ensemble"].sort_values("mape").iloc[0]["alpha"])
    paths = make_test_submissions(best_alpha)

    report = {
        "seeds": SEEDS,
        "alphas": ALPHAS,
        "comparison": comparison.to_dict(orient="records"),
        "candidate_submissions": paths,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
