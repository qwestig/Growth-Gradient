from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from catboost_residual_diagnostics import (
    SUBMISSIONS_DIR,
    TARGET,
    add_base,
    apply_residual,
    make_panel,
    mape,
)
from gru_residual_experiment import (
    ALPHA_GRID,
    BATCH_SIZE,
    CATBOOST_OOF_PATH,
    EPOCHS,
    PATIENCE,
    SEED,
    SEQ_LEN,
    SMALL_CAT_COLS,
    STATIC_NUM_COLS,
    apply_sequence_scaler,
    build_sequence,
    fit_sequence_scaler,
    fit_static_encoders,
    period_id,
    prepare_data,
    set_seed,
    transform_static,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "transformer_residual"
GRU_OOF_PATH = ROOT / "results" / "gru_residual" / "gru_oof.csv"


class TransformerResidual(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_num_dim: int,
        cat_cardinalities: list[int],
        seq_len: int = SEQ_LEN,
        d_model: int = 32,
        nhead: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(seq_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
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
            nn.Linear(d_model * 2 + 16, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, seq: torch.Tensor, static_num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        tokens = self.input_projection(seq) + self.position[:, : seq.shape[1], :]
        encoded = self.encoder(tokens)
        seq_state = torch.cat([encoded[:, -1, :], encoded.mean(dim=1)], dim=1)
        emb = [embedding(cat[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        static = torch.cat([static_num, *emb], dim=1)
        static_state = self.static_net(static)
        return self.head(torch.cat([seq_state, static_state], dim=1)).squeeze(1)


def mape_score_from_log_residual(target_rows: pd.DataFrame, pred_log: np.ndarray, alpha: float) -> float:
    pred = apply_residual(
        target_rows["best_base"].to_numpy(dtype=float),
        pred_log,
        alpha,
        None,
    )
    return mape(target_rows[TARGET], pred)


def train_one_fold(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cat_cardinalities: list[int],
    seed: int,
) -> tuple[np.ndarray, float]:
    set_seed(seed)
    train_seq, train_num, train_cat, y_train = train_data
    valid_seq, valid_num, valid_cat, y_valid = valid_data
    model = TransformerResidual(
        seq_dim=train_seq.shape[-1],
        static_num_dim=train_num.shape[1],
        cat_cardinalities=cat_cardinalities,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    train_dataset = TensorDataset(
        torch.from_numpy(train_seq),
        torch.from_numpy(train_num),
        torch.from_numpy(train_cat),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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


def run_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepare_data()
    target = data.target_rows.copy()
    store_ids = target["new_id"].to_numpy()
    target_period = period_id(2024, 3)
    full_seq = build_sequence(data, store_ids, target_period)
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
        )
        valid_rows = target.loc[valid_mask].copy()
        valid_rows["fold"] = fold
        valid_rows["pred_log_transformer"] = pred_log
        oof_parts.append(valid_rows[["new_id", TARGET, "best_base", "fold", "pred_log_transformer"]])

        row = {"fold": fold, "valid_loss": best_loss, "base_mape": mape(valid_rows[TARGET], valid_rows["best_base"])}
        for alpha in ALPHA_GRID + [0.8, 0.9, 1.0]:
            row[f"transformer_alpha_{alpha:.1f}_mape"] = mape_score_from_log_residual(valid_rows, pred_log, alpha)
        fold_rows.append(row)

    return pd.DataFrame(fold_rows), pd.concat(oof_parts, ignore_index=True)


def compare(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha in sorted(set(ALPHA_GRID + [0.8, 0.9, 1.0])):
        pred = apply_residual(
            oof["best_base"].to_numpy(dtype=float),
            oof["pred_log_transformer"].to_numpy(dtype=float),
            alpha,
            None,
        )
        rows.append({"model": "transformer", "alpha": alpha, "blend": "", "mape": mape(oof[TARGET], pred)})

    if GRU_OOF_PATH.exists():
        gru = pd.read_csv(GRU_OOF_PATH)[["new_id", "pred_log_gru"]]
        merged = oof.merge(gru, on="new_id", how="inner")
        for weight_transformer in [0.1, 0.2, 0.3, 0.4, 0.5]:
            blended_log = (1 - weight_transformer) * merged["pred_log_gru"].to_numpy(dtype=float) + weight_transformer * merged[
                "pred_log_transformer"
            ].to_numpy(dtype=float)
            pred = apply_residual(merged["best_base"].to_numpy(dtype=float), blended_log, 0.70, None)
            rows.append(
                {
                    "model": "gru_transformer_blend",
                    "alpha": 0.70,
                    "blend": f"transformer_weight_{weight_transformer:.1f}",
                    "mape": mape(merged[TARGET], pred),
                }
            )
        gru_pred = apply_residual(
            merged["best_base"].to_numpy(dtype=float),
            merged["pred_log_gru"].to_numpy(dtype=float),
            0.70,
            None,
        )
        rows.append({"model": "gru_reference", "alpha": 0.70, "blend": "", "mape": mape(merged[TARGET], gru_pred)})

    if CATBOOST_OOF_PATH.exists():
        cb = pd.read_csv(CATBOOST_OOF_PATH)
        cb = cb[cb["config"] == "all_new_plus_targeted"][["new_id", "pred_log"]].rename(
            columns={"pred_log": "pred_log_catboost"}
        )
        merged = oof.merge(cb, on="new_id", how="inner")
        cb_pred = apply_residual(
            merged["best_base"].to_numpy(dtype=float),
            merged["pred_log_catboost"].to_numpy(dtype=float),
            0.70,
            None,
        )
        rows.append({"model": "catboost_targeted_reference", "alpha": 0.70, "blend": "", "mape": mape(merged[TARGET], cb_pred)})

    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def train_final_model(data, epochs: int = 35) -> tuple[TransformerResidual, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]]:
    set_seed(SEED + 100)
    train_target = data.target_rows.copy()
    train_ids = train_target["new_id"].to_numpy()
    y_train = train_target["target_log_ratio"].to_numpy(dtype=np.float32)
    train_seq_raw = build_sequence(data, train_ids, period_id(2024, 3))
    seq_mean, seq_std = fit_sequence_scaler(train_seq_raw)
    train_seq = apply_sequence_scaler(train_seq_raw, seq_mean, seq_std)
    num_mean, num_std, encoders = fit_static_encoders(data.static_frame, train_ids)
    train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders)
    cat_cardinalities = [len(encoders[col]) for col in SMALL_CAT_COLS]

    model = TransformerResidual(train_seq.shape[-1], train_num.shape[1], cat_cardinalities)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)
    dataset = TensorDataset(
        torch.from_numpy(train_seq),
        torch.from_numpy(train_num),
        torch.from_numpy(train_cat),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch_seq, batch_num, batch_cat, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_seq, batch_num, batch_cat), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model, (seq_mean, seq_std), (num_mean, num_std, encoders)


def make_submission(data, model, seq_scaler, static_scaler, alpha: float) -> Path:
    seq_mean, seq_std = seq_scaler
    num_mean, num_std, encoders = static_scaler
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
    path = SUBMISSIONS_DIR / f"test_transformer_alpha{int(alpha * 100):03d}.csv"
    submission.to_csv(path, index=False)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    folds, oof = run_oof()
    comparison = compare(oof)
    folds.to_csv(RESULTS_DIR / "transformer_folds.csv", index=False)
    oof.to_csv(RESULTS_DIR / "transformer_oof.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "transformer_comparison.csv", index=False)

    transformer_rows = comparison[comparison["model"] == "transformer"].copy()
    best_alpha = float(transformer_rows.sort_values("mape").iloc[0]["alpha"])
    data = prepare_data()
    model, seq_scaler, static_scaler = train_final_model(data)
    candidate_paths = [str(make_submission(data, model, seq_scaler, static_scaler, 0.70))]
    if abs(best_alpha - 0.70) > 1e-9:
        candidate_paths.append(str(make_submission(data, model, seq_scaler, static_scaler, best_alpha)))

    report = {
        "seq_len": SEQ_LEN,
        "best_transformer_alpha": best_alpha,
        "comparison": comparison.to_dict(orient="records"),
        "candidate_submissions": candidate_paths,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
