from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from catboost_residual_diagnostics import SUBMISSIONS_DIR, TARGET, add_base, apply_residual, make_panel, mape
from gru_residual_experiment import (
    ALPHA_GRID,
    BATCH_SIZE,
    PATIENCE,
    SEED,
    SEQ_LEN,
    SMALL_CAT_COLS,
    PreparedData,
    apply_sequence_scaler,
    build_sequence,
    fit_sequence_scaler,
    fit_static_encoders,
    mape_score_from_log_residual,
    period_id,
    prepare_data,
    set_seed,
    transform_static,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "dlinear_residual"
EPOCHS = 80


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1) // 2, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearResidual(nn.Module):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        static_num_dim: int,
        cat_cardinalities: list[int],
        kernel_size: int,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.seasonal_linear = nn.Linear(seq_len, 1)
        self.trend_linear = nn.Linear(seq_len, 1)
        self.seasonal_linear.weight = nn.Parameter((1 / seq_len) * torch.ones([1, seq_len]))
        self.trend_linear.weight = nn.Parameter((1 / seq_len) * torch.ones([1, seq_len]))

        self.embeddings = nn.ModuleList()
        emb_dims = []
        for cardinality in cat_cardinalities:
            dim = min(12, max(2, int(np.sqrt(cardinality + 1)) + 1))
            self.embeddings.append(nn.Embedding(cardinality + 1, dim, padding_idx=0))
            emb_dims.append(dim)

        static_in = static_num_dim + sum(emb_dims)
        self.static_net = nn.Sequential(
            nn.Linear(static_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(seq_dim + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq: torch.Tensor, static_num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(seq)
        seasonal = self.seasonal_linear(seasonal.permute(0, 2, 1)).squeeze(-1)
        trend = self.trend_linear(trend.permute(0, 2, 1)).squeeze(-1)
        seq_state = seasonal + trend

        emb = [embedding(cat[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        static = torch.cat([static_num, *emb], dim=1)
        static_state = self.static_net(static)
        return self.head(torch.cat([seq_state, static_state], dim=1)).squeeze(1)


def train_one_fold(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cat_cardinalities: list[int],
    seed: int,
    kernel_size: int,
) -> tuple[np.ndarray, float]:
    set_seed(seed)
    train_seq, train_num, train_cat, y_train = train_data
    valid_seq, valid_num, valid_cat, y_valid = valid_data
    model = DLinearResidual(
        seq_len=train_seq.shape[1],
        seq_dim=train_seq.shape[2],
        static_num_dim=train_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        kernel_size=kernel_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
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
    return pred, best_loss


def run_oof(data: PreparedData, kernel_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = data.target_rows.copy()
    store_ids = target["new_id"].to_numpy()
    full_seq = build_sequence(data, store_ids, period_id(2024, 3))
    y = target["target_log_ratio"].to_numpy(dtype=np.float32)

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

        pred_log, best_loss = train_one_fold(
            (train_seq, train_num, train_cat, y[~valid_mask]),
            (valid_seq, valid_num, valid_cat, y[valid_mask]),
            cat_cardinalities,
            SEED + 100 + fold,
            kernel_size,
        )
        valid_rows = target.loc[valid_mask].copy()
        valid_rows["fold"] = fold
        valid_rows["pred_log_dlinear"] = pred_log
        valid_rows["kernel_size"] = kernel_size
        oof_parts.append(valid_rows[["new_id", TARGET, "best_base", "fold", "kernel_size", "pred_log_dlinear"]])
        row = {
            "fold": fold,
            "kernel_size": kernel_size,
            "valid_loss": best_loss,
            "base_mape": mape(valid_rows[TARGET], valid_rows["best_base"]),
        }
        for alpha in ALPHA_GRID:
            row[f"dlinear_alpha_{alpha:.1f}_mape"] = mape_score_from_log_residual(valid_rows, pred_log, alpha)
        fold_rows.append(row)
        print(f"kernel={kernel_size} fold={fold} done", flush=True)

    return pd.DataFrame(fold_rows), pd.concat(oof_parts, ignore_index=True)


def make_final_submission(data: PreparedData, kernel_size: int, alpha: float) -> Path:
    set_seed(SEED + 1000 + kernel_size)
    train_target = data.target_rows.copy()
    train_ids = train_target["new_id"].to_numpy()
    y_train = train_target["target_log_ratio"].to_numpy(dtype=np.float32)
    train_seq_raw = build_sequence(data, train_ids, period_id(2024, 3))
    seq_mean, seq_std = fit_sequence_scaler(train_seq_raw)
    train_seq = apply_sequence_scaler(train_seq_raw, seq_mean, seq_std)
    num_mean, num_std, encoders = fit_static_encoders(data.static_frame, train_ids)
    train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders)
    cat_cardinalities = [len(encoders[col]) for col in SMALL_CAT_COLS]

    model = DLinearResidual(train_seq.shape[1], train_seq.shape[2], train_num.shape[1], cat_cardinalities, kernel_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
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
    for _epoch in range(35):
        model.train()
        for batch_seq, batch_num, batch_cat, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_seq, batch_num, batch_cat), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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

    pred_seq = apply_sequence_scaler(build_sequence(data, pred_ids, period_id(2025, 3)), seq_mean, seq_std)
    pred_num, pred_cat = transform_static(data.static_frame, pred_ids, num_mean, num_std, encoders)
    model.eval()
    with torch.no_grad():
        pred_log = model(torch.from_numpy(pred_seq), torch.from_numpy(pred_num), torch.from_numpy(pred_cat)).numpy()
    pred = apply_residual(pred_rows["best_base"].to_numpy(dtype=float), pred_log, alpha, None)
    submission = pred_rows[["new_id"]].copy()
    submission["rto"] = np.round(np.maximum(pred, 0), 2)
    path = SUBMISSIONS_DIR / f"test_dlinear_k{kernel_size}_alpha{int(alpha * 100):03d}.csv"
    submission.to_csv(path, index=False)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()

    fold_parts = []
    oof_parts = []
    for kernel_size in [3, 5, 7]:
        folds, oof = run_oof(data, kernel_size)
        fold_parts.append(folds)
        oof_parts.append(oof)

    folds = pd.concat(fold_parts, ignore_index=True)
    oof = pd.concat(oof_parts, ignore_index=True)
    rows = []
    for kernel_size, part in oof.groupby("kernel_size"):
        for alpha in ALPHA_GRID:
            pred = apply_residual(part["best_base"].to_numpy(dtype=float), part["pred_log_dlinear"].to_numpy(), alpha, None)
            rows.append({"kernel_size": int(kernel_size), "alpha": alpha, "mape": mape(part[TARGET], pred)})
    summary = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
    best = summary.iloc[0]
    submission_path = make_final_submission(data, int(best["kernel_size"]), float(best["alpha"]))

    folds.to_csv(RESULTS_DIR / "dlinear_folds.csv", index=False)
    oof.to_csv(RESULTS_DIR / "dlinear_oof.csv", index=False)
    summary.to_csv(RESULTS_DIR / "dlinear_summary.csv", index=False)
    report = {
        "best": best.to_dict(),
        "summary": summary.to_dict(orient="records"),
        "submission": str(submission_path),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
