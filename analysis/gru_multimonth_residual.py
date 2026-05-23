from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from catboost_residual_diagnostics import (
    BEHAVIOR_COLS,
    STATIC_NUM_COLS,
    SUBMISSIONS_DIR,
    TARGET,
    TRAIN_PATH,
    apply_residual,
    mape,
)
from gru_residual_experiment import GRUResidual, SMALL_CAT_COLS, fit_static_encoders, set_seed, transform_static


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "gru_multimonth"
SEED = 2026
SEQ_LEN = 14
BATCH_SIZE = 1024
EPOCHS = 35
PATIENCE = 5
TARGET_PERIODS = [(2024, month) for month in range(3, 13)] + [(2025, 1), (2025, 2)]
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]


@dataclass
class MultiMonthData:
    raw: pd.DataFrame
    rto_wide: pd.DataFrame
    behavior_wide: dict[str, pd.DataFrame]
    static_frame: pd.DataFrame
    examples: pd.DataFrame


def period_id(year: int, month: int) -> int:
    return year * 12 + month


def year_month(period: int) -> tuple[int, int]:
    return period // 12, period % 12


def build_wide(raw: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return raw.pivot_table(index="new_id", columns="period", values=value_col, aggfunc="mean")


def make_base_for_period(rto_wide: pd.DataFrame, target_period: int) -> pd.DataFrame:
    prev = target_period - 1
    prev_year_same = target_period - 12
    prev_year_prev = target_period - 13
    recent = [target_period - 1, target_period - 2, target_period - 3]

    seasonal = (rto_wide[prev_year_same] / rto_wide[prev_year_prev]).replace([np.inf, -np.inf], np.nan)
    global_factor = float(seasonal.dropna().median())
    rolling3 = rto_wide[recent].mean(axis=1)
    momentum = (rto_wide[prev] / rolling3).replace([np.inf, -np.inf], np.nan).fillna(1).clip(0.95, 1.05)
    base = rto_wide[prev] * global_factor * (momentum**0.25)
    return pd.DataFrame(
        {
            "new_id": rto_wide.index,
            "target_period": target_period,
            "target_year": target_period // 12,
            "target_month": target_period % 12,
            "actual": rto_wide[target_period],
            "base": base,
            "global_factor": global_factor,
            "momentum_raw": rto_wide[prev] / rolling3,
            "momentum_clip": momentum,
        }
    ).reset_index(drop=True)


def prepare_data() -> MultiMonthData:
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    rto_wide = build_wide(raw, TARGET)
    behavior_wide = {col: build_wide(raw, col) for col in BEHAVIOR_COLS}
    static_cols = ["new_id"] + STATIC_NUM_COLS + SMALL_CAT_COLS
    static_frame = (
        raw.sort_values(["new_id", "period"])
        .groupby("new_id", as_index=False)[static_cols]
        .first()
        .set_index("new_id")
    )

    parts = []
    for year, month in TARGET_PERIODS:
        period = period_id(year, month)
        needed = [period, period - 1, period - 2, period - 3, period - 12, period - 13]
        if not set(needed).issubset(set(rto_wide.columns)):
            continue
        part = make_base_for_period(rto_wide, period)
        parts.append(part)
    examples = pd.concat(parts, ignore_index=True)
    examples = examples[(examples["actual"] > 0) & (examples["base"] > 0)].copy()
    examples["target_log_ratio"] = np.log(examples["actual"] / examples["base"])
    return MultiMonthData(raw, rto_wide, behavior_wide, static_frame, examples)


def build_sequence(data: MultiMonthData, frame: pd.DataFrame) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    seq = np.empty((len(frame), SEQ_LEN, 6 + len(BEHAVIOR_COLS)), dtype=np.float32)
    for target_period, index in frame.groupby("target_period", sort=False).groups.items():
        positions = np.asarray(index, dtype=int)
        rows = frame.iloc[positions]
        store_ids = rows["new_id"].to_numpy()
        hist_periods = list(range(target_period - SEQ_LEN, target_period))
        rto = data.rto_wide.loc[store_ids, hist_periods].to_numpy(dtype=float)
        log_rto = np.log(np.clip(rto, 1.0, None))
        rel_log_rto = log_rto - log_rto[:, [-1]]
        log_growth = np.zeros_like(log_rto)
        log_growth[:, 1:] = log_rto[:, 1:] - log_rto[:, :-1]
        months = np.array([((period - 1) % 12) + 1 for period in hist_periods], dtype=float)
        month_sin = np.repeat(np.sin(2 * np.pi * months / 12)[None, :], len(rows), axis=0)
        month_cos = np.repeat(np.cos(2 * np.pi * months / 12)[None, :], len(rows), axis=0)
        target_month = int(rows["target_month"].iloc[0])
        target_sin = np.full((len(rows), SEQ_LEN), np.sin(2 * np.pi * target_month / 12))
        target_cos = np.full((len(rows), SEQ_LEN), np.cos(2 * np.pi * target_month / 12))
        behavior_cols = [
            data.behavior_wide[col].loc[store_ids, hist_periods].to_numpy(dtype=float)
            for col in BEHAVIOR_COLS
        ]
        seq[positions] = np.stack(
            [rel_log_rto, log_growth, month_sin, month_cos, target_sin, target_cos, *behavior_cols],
            axis=2,
        )
    return seq


def fit_sequence_scaler(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = seq.reshape(-1, seq.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_sequence_scaler(seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (seq - mean[None, None, :]) / std[None, None, :]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def train_model(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cat_cardinalities: list[int],
    seed: int,
) -> tuple[GRUResidual, np.ndarray, float]:
    set_seed(seed)
    train_seq, train_num, train_cat, y_train = train_data
    valid_seq, valid_num, valid_cat, y_valid = valid_data
    model = GRUResidual(train_seq.shape[-1], train_num.shape[1], cat_cardinalities, hidden_dim=32, dropout=0.18)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_seq),
            torch.from_numpy(train_num),
            torch.from_numpy(train_cat),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    valid_tensors = (
        torch.from_numpy(valid_seq),
        torch.from_numpy(valid_num),
        torch.from_numpy(valid_cat),
        torch.from_numpy(y_valid.astype(np.float32)),
    )
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(EPOCHS):
        model.train()
        for batch_seq, batch_num, batch_cat, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_seq, batch_num, batch_cat), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_tensors[0], valid_tensors[1], valid_tensors[2]), valid_tensors[3]))
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(valid_tensors[0], valid_tensors[1], valid_tensors[2]).numpy()
    return model, pred, best_loss


def evaluate_predictions(frame: pd.DataFrame, pred_log: np.ndarray, alpha: float) -> float:
    pred = apply_residual(frame["base"].to_numpy(dtype=float), pred_log, alpha, None)
    return mape(frame["actual"], pred)


def run_oof(data: MultiMonthData) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = data.examples.copy().reset_index(drop=True)
    y = examples["target_log_ratio"].to_numpy(dtype=np.float32)
    fold_rows = []
    oof_parts = []
    for fold in range(5):
        valid_mask = (examples["new_id"].to_numpy() % 5) == fold
        train_frame = examples[~valid_mask].copy()
        valid_frame = examples[valid_mask].copy()
        train_seq_raw = build_sequence(data, train_frame)
        valid_seq_raw = build_sequence(data, valid_frame)
        seq_mean, seq_std = fit_sequence_scaler(train_seq_raw)
        train_seq = apply_sequence_scaler(train_seq_raw, seq_mean, seq_std)
        valid_seq = apply_sequence_scaler(valid_seq_raw, seq_mean, seq_std)

        train_ids = train_frame["new_id"].to_numpy()
        valid_ids = valid_frame["new_id"].to_numpy()
        num_mean, num_std, encoders = fit_static_encoders(data.static_frame, train_ids)
        train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders)
        valid_num, valid_cat = transform_static(data.static_frame, valid_ids, num_mean, num_std, encoders)
        cat_cardinalities = [len(encoders[col]) for col in SMALL_CAT_COLS]

        _, pred_log, best_loss = train_model(
            (train_seq, train_num, train_cat, y[~valid_mask]),
            (valid_seq, valid_num, valid_cat, y[valid_mask]),
            cat_cardinalities,
            SEED + fold,
        )
        valid_out = valid_frame[["new_id", "target_period", "target_year", "target_month", "actual", "base"]].copy()
        valid_out["fold"] = fold
        valid_out["pred_log"] = pred_log
        oof_parts.append(valid_out)
        row = {
            "fold": fold,
            "valid_loss": best_loss,
            "base_mape": mape(valid_frame["actual"], valid_frame["base"]),
        }
        for alpha in ALPHAS:
            row[f"alpha_{alpha:.2f}_mape"] = evaluate_predictions(valid_frame, pred_log, alpha)
        fold_rows.append(row)
    return pd.DataFrame(fold_rows), pd.concat(oof_parts, ignore_index=True)


def run_march_holdout(data: MultiMonthData) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = data.examples.copy().reset_index(drop=True)
    valid_mask = examples["target_period"].to_numpy() == period_id(2024, 3)
    train_frame = examples[~valid_mask].copy()
    valid_frame = examples[valid_mask].copy()
    print(
        f"march_holdout: train_rows={len(train_frame)} valid_rows={len(valid_frame)}",
        flush=True,
    )

    train_seq_raw = build_sequence(data, train_frame)
    valid_seq_raw = build_sequence(data, valid_frame)
    seq_mean, seq_std = fit_sequence_scaler(train_seq_raw)
    train_seq = apply_sequence_scaler(train_seq_raw, seq_mean, seq_std)
    valid_seq = apply_sequence_scaler(valid_seq_raw, seq_mean, seq_std)

    train_ids = train_frame["new_id"].to_numpy()
    valid_ids = valid_frame["new_id"].to_numpy()
    num_mean, num_std, encoders = fit_static_encoders(data.static_frame, train_ids)
    train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders)
    valid_num, valid_cat = transform_static(data.static_frame, valid_ids, num_mean, num_std, encoders)
    cat_cardinalities = [len(encoders[col]) for col in SMALL_CAT_COLS]
    y_train = train_frame["target_log_ratio"].to_numpy(dtype=np.float32)
    y_valid = valid_frame["target_log_ratio"].to_numpy(dtype=np.float32)

    _, pred_log, best_loss = train_model(
        (train_seq, train_num, train_cat, y_train),
        (valid_seq, valid_num, valid_cat, y_valid),
        cat_cardinalities,
        SEED,
    )
    valid_out = valid_frame[["new_id", "target_period", "target_year", "target_month", "actual", "base"]].copy()
    valid_out["pred_log"] = pred_log
    rows = [
        {
            "scope": "march2024_holdout",
            "alpha": alpha,
            "mape": evaluate_predictions(valid_frame, pred_log, alpha),
            "base_mape": mape(valid_frame["actual"], valid_frame["base"]),
            "valid_loss": best_loss,
        }
        for alpha in ALPHAS
    ]
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True), valid_out


def summarize_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha in ALPHAS:
        rows.append({"scope": "all_months", "alpha": alpha, "mape": evaluate_predictions(oof, oof["pred_log"].to_numpy(), alpha)})
        march = oof[oof["target_month"] == 3]
        rows.append({"scope": "march_only", "alpha": alpha, "mape": evaluate_predictions(march, march["pred_log"].to_numpy(), alpha)})
        feb2025 = oof[oof["target_period"] == period_id(2025, 2)]
        if len(feb2025):
            rows.append({"scope": "feb2025", "alpha": alpha, "mape": evaluate_predictions(feb2025, feb2025["pred_log"].to_numpy(), alpha)})
    return pd.DataFrame(rows).sort_values(["scope", "mape"]).reset_index(drop=True)


def make_prediction_rows(data: MultiMonthData, target_year: int, target_month: int) -> pd.DataFrame:
    target_period = period_id(target_year, target_month)
    rto_wide = data.rto_wide.copy()
    if target_period not in rto_wide.columns:
        rto_wide[target_period] = np.nan
    base_frame = make_base_for_period(rto_wide, target_period)
    return base_frame[base_frame["base"] > 0].copy().reset_index(drop=True)


def train_final_and_submit(data: MultiMonthData, alpha: float) -> Path:
    examples = data.examples.copy().reset_index(drop=True)
    train_seq_raw = build_sequence(data, examples)
    seq_mean, seq_std = fit_sequence_scaler(train_seq_raw)
    train_seq = apply_sequence_scaler(train_seq_raw, seq_mean, seq_std)
    train_ids = examples["new_id"].to_numpy()
    num_mean, num_std, encoders = fit_static_encoders(data.static_frame, train_ids)
    train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders)
    cat_cardinalities = [len(encoders[col]) for col in SMALL_CAT_COLS]
    y = examples["target_log_ratio"].to_numpy(dtype=np.float32)

    # Last target month as early-stopping validation, all previous months as train.
    valid_mask = examples["target_period"].to_numpy() == examples["target_period"].max()
    model, _, _ = train_model(
        (train_seq[~valid_mask], train_num[~valid_mask], train_cat[~valid_mask], y[~valid_mask]),
        (train_seq[valid_mask], train_num[valid_mask], train_cat[valid_mask], y[valid_mask]),
        cat_cardinalities,
        SEED,
    )

    pred_rows = make_prediction_rows(data, 2025, 3).sort_values("new_id").reset_index(drop=True)
    pred_seq = apply_sequence_scaler(build_sequence(data, pred_rows), seq_mean, seq_std)
    pred_num, pred_cat = transform_static(data.static_frame, pred_rows["new_id"].to_numpy(), num_mean, num_std, encoders)
    model.eval()
    with torch.no_grad():
        pred_log = model(torch.from_numpy(pred_seq), torch.from_numpy(pred_num), torch.from_numpy(pred_cat)).numpy()
    pred = apply_residual(pred_rows["base"].to_numpy(dtype=float), pred_log, alpha, None)
    submission = pred_rows[["new_id"]].copy()
    submission["rto"] = np.round(np.maximum(pred, 0), 2)
    path = SUBMISSIONS_DIR / f"test_gru_multimonth_alpha{int(alpha * 100):03d}.csv"
    submission.to_csv(path, index=False)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    print(f"prepared examples={len(data.examples)} periods={data.examples['target_period'].nunique()}", flush=True)
    summary, holdout = run_march_holdout(data)
    holdout.to_csv(RESULTS_DIR / "march2024_holdout.csv", index=False)
    summary.to_csv(RESULTS_DIR / "march2024_holdout_summary.csv", index=False)

    best_march = summary.sort_values("mape").iloc[0]
    candidate_alpha = float(best_march["alpha"])
    paths = [str(train_final_and_submit(data, 0.70))]
    if abs(candidate_alpha - 0.70) > 1e-9:
        paths.append(str(train_final_and_submit(data, candidate_alpha)))

    report = {
        "target_periods": TARGET_PERIODS,
        "examples": len(data.examples),
        "best_march": best_march.to_dict(),
        "summary": summary.to_dict(orient="records"),
        "candidate_submissions": paths,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
