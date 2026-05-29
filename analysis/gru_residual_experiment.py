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
    CAT_COLS,
    STATIC_NUM_COLS,
    SUBMISSIONS_DIR,
    TARGET,
    TRAIN_PATH,
    add_base,
    apply_residual,
    make_panel,
    mape,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "gru_residual"
CATBOOST_OOF_PATH = ROOT / "results" / "catboost_feature_audit" / "ablation_oof.csv"
SEED = 2026
ALPHA_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SEQ_LEN = 14
EPOCHS = 80
BATCH_SIZE = 512
PATIENCE = 10

SMALL_CAT_COLS = [
    "Дата открытия, категориальный",
    "Торговая площадь, категориальный",
    "Регион",
]


@dataclass
class PreparedData:
    raw: pd.DataFrame
    target_rows: pd.DataFrame
    validation_factor: float
    final_factor: float
    rto_wide: pd.DataFrame
    behavior_wide: dict[str, pd.DataFrame]
    static_frame: pd.DataFrame


class GRUResidual(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_num_dim: int,
        cat_cardinalities: list[int],
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden_dim, num_layers=1, batch_first=True)
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
            nn.Linear(hidden_dim + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq: torch.Tensor, static_num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(seq)
        seq_state = hidden[-1]
        emb = [embedding(cat[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        static = torch.cat([static_num, *emb], dim=1)
        static_state = self.static_net(static)
        return self.head(torch.cat([seq_state, static_state], dim=1)).squeeze(1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def period_id(year: int, month: int) -> int:
    return year * 12 + month


def mape_score_from_log_residual(
    target_rows: pd.DataFrame, pred_log: np.ndarray, alpha: float
) -> float:
    pred = apply_residual(
        target_rows["best_base"].to_numpy(dtype=float),
        pred_log,
        alpha,
        None,
    )
    return mape(target_rows[TARGET], pred)


def prepare_data() -> PreparedData:
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    period_label = pd.to_datetime(
        raw["Год"].astype(str) + "-" + raw["Месяц"].astype(str).str.zfill(2) + "-01"
    ).dt.strftime("%Y-%m")
    rto_wide = raw.assign(period_label=period_label).pivot_table(
        index="new_id", columns="period", values=TARGET, aggfunc="mean"
    )
    ratio_2023 = (
        rto_wide[period_id(2023, 3)] / rto_wide[period_id(2023, 2)]
    ).replace([np.inf, -np.inf], np.nan)
    ratio_2024 = (
        rto_wide[period_id(2024, 3)] / rto_wide[period_id(2024, 2)]
    ).replace([np.inf, -np.inf], np.nan)
    validation_factor = float(ratio_2023.median())
    final_factor = float(pd.concat([ratio_2023, ratio_2024]).dropna().median())

    panel = add_base(make_panel(raw), validation_factor)
    target_rows = panel[
        (panel["Год"] == 2024)
        & (panel["Месяц"] == 3)
        & (panel[TARGET] > 0)
        & (panel["best_base"] > 0)
    ].copy()
    target_rows["target_log_ratio"] = np.log(target_rows[TARGET] / target_rows["best_base"])
    target_rows = target_rows.sort_values("new_id").reset_index(drop=True)

    behavior_wide = {
        col: raw.pivot_table(index="new_id", columns="period", values=col, aggfunc="mean")
        for col in BEHAVIOR_COLS
    }
    static_cols = ["new_id"] + STATIC_NUM_COLS + SMALL_CAT_COLS
    static_frame = (
        raw.sort_values(["new_id", "period"])
        .groupby("new_id", as_index=False)[static_cols]
        .first()
        .set_index("new_id")
    )
    return PreparedData(
        raw=raw,
        target_rows=target_rows,
        validation_factor=validation_factor,
        final_factor=final_factor,
        rto_wide=rto_wide,
        behavior_wide=behavior_wide,
        static_frame=static_frame,
    )


def build_sequence(data: PreparedData, store_ids: np.ndarray, target_period: int) -> np.ndarray:
    hist_periods = list(range(target_period - SEQ_LEN, target_period))
    rto = data.rto_wide.loc[store_ids, hist_periods].to_numpy(dtype=float)
    log_rto = np.log(np.clip(rto, 1.0, None))
    last_log = log_rto[:, [-1]]
    rel_log_rto = log_rto - last_log
    log_growth = np.zeros_like(log_rto)
    log_growth[:, 1:] = log_rto[:, 1:] - log_rto[:, :-1]

    months = np.array([((period - 1) % 12) + 1 for period in hist_periods], dtype=float)
    month_sin = np.sin(2 * np.pi * months / 12)[None, :, None]
    month_cos = np.cos(2 * np.pi * months / 12)[None, :, None]
    calendar = np.repeat(np.concatenate([month_sin, month_cos], axis=2), len(store_ids), axis=0)

    behavior_parts = []
    for col in BEHAVIOR_COLS:
        values = data.behavior_wide[col].loc[store_ids, hist_periods].to_numpy(dtype=float)
        behavior_parts.append(values[:, :, None])
    behavior = np.concatenate(behavior_parts, axis=2)
    seq = np.concatenate([rel_log_rto[:, :, None], log_growth[:, :, None], calendar, behavior], axis=2)
    return seq.astype(np.float32)


def fit_sequence_scaler(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = seq.reshape(-1, seq.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_sequence_scaler(seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (seq - mean[None, None, :]) / std[None, None, :]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_static_encoders(
    static_frame: pd.DataFrame, train_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    train_static = static_frame.loc[train_ids]
    num_mean = train_static[STATIC_NUM_COLS].astype(float).mean().to_numpy(dtype=np.float32)
    num_std = train_static[STATIC_NUM_COLS].astype(float).std().replace(0, 1).to_numpy(dtype=np.float32)
    encoders = {}
    for col in SMALL_CAT_COLS:
        values = train_static[col].astype(str).fillna("__nan__").unique().tolist()
        encoders[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return num_mean, num_std, encoders


def transform_static(
    static_frame: pd.DataFrame,
    store_ids: np.ndarray,
    num_mean: np.ndarray,
    num_std: np.ndarray,
    encoders: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    static = static_frame.loc[store_ids]
    numeric = static[STATIC_NUM_COLS].astype(float).to_numpy(dtype=np.float32)
    numeric = (numeric - num_mean[None, :]) / num_std[None, :]
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    cats = []
    for col in SMALL_CAT_COLS:
        mapping = encoders[col]
        cats.append(static[col].astype(str).fillna("__nan__").map(mapping).fillna(0).to_numpy(dtype=np.int64))
    return numeric, np.stack(cats, axis=1)


def train_one_fold(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cat_cardinalities: list[int],
    seed: int,
) -> tuple[np.ndarray, float]:
    set_seed(seed)
    train_seq, train_num, train_cat, y_train = train_data
    valid_seq, valid_num, valid_cat, y_valid = valid_data
    model = GRUResidual(
        seq_dim=train_seq.shape[-1],
        static_num_dim=train_num.shape[1],
        cat_cardinalities=cat_cardinalities,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
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


def run_oof(data: PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            SEED + fold,
        )
        valid_rows = target.loc[valid_mask].copy()
        valid_rows["fold"] = fold
        valid_rows["pred_log_gru"] = pred_log
        oof_parts.append(valid_rows[["new_id", TARGET, "best_base", "fold", "pred_log_gru"]])
        row = {"fold": fold, "valid_loss": best_loss, "base_mape": mape(valid_rows[TARGET], valid_rows["best_base"])}
        for alpha in ALPHA_GRID:
            row[f"gru_alpha_{alpha:.1f}_mape"] = mape_score_from_log_residual(valid_rows, pred_log, alpha)
        fold_rows.append(row)

    oof = pd.concat(oof_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    return folds, oof


def compare_with_catboost(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha in ALPHA_GRID:
        pred = apply_residual(oof["best_base"].to_numpy(dtype=float), oof["pred_log_gru"].to_numpy(dtype=float), alpha, None)
        rows.append({"model": "gru", "alpha": alpha, "blend_weight_gru": 1.0, "mape": mape(oof[TARGET], pred)})

    if CATBOOST_OOF_PATH.exists():
        cb = pd.read_csv(CATBOOST_OOF_PATH)
        cb = cb[cb["config"] == "all_new_plus_targeted"][["new_id", "pred_log"]].rename(
            columns={"pred_log": "pred_log_catboost"}
        )
        merged = oof.merge(cb, on="new_id", how="inner")
        for weight in [0.10, 0.20, 0.30, 0.40, 0.50]:
            blended_log = (1 - weight) * merged["pred_log_catboost"].to_numpy(dtype=float) + weight * merged[
                "pred_log_gru"
            ].to_numpy(dtype=float)
            pred = apply_residual(merged["best_base"].to_numpy(dtype=float), blended_log, 0.70, None)
            rows.append({"model": "catboost_gru_blend", "alpha": 0.70, "blend_weight_gru": weight, "mape": mape(merged[TARGET], pred)})
        cb_pred = apply_residual(
            merged["best_base"].to_numpy(dtype=float),
            merged["pred_log_catboost"].to_numpy(dtype=float),
            0.70,
            None,
        )
        rows.append({"model": "catboost_targeted_reference", "alpha": 0.70, "blend_weight_gru": 0.0, "mape": mape(merged[TARGET], cb_pred)})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def make_final_gru_submission(data: PreparedData, alpha: float) -> Path:
    set_seed(SEED)
    train_target = data.target_rows.copy()
    train_ids = train_target["new_id"].to_numpy()
    y_train = train_target["target_log_ratio"].to_numpy(dtype=np.float32)
    train_seq_raw = build_sequence(data, train_ids, period_id(2024, 3))
    seq_mean, seq_std = fit_sequence_scaler(train_seq_raw)
    train_seq = apply_sequence_scaler(train_seq_raw, seq_mean, seq_std)
    num_mean, num_std, encoders = fit_static_encoders(data.static_frame, train_ids)
    train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders)
    cat_cardinalities = [len(encoders[col]) for col in SMALL_CAT_COLS]

    # Keep a small validation slice for early stopping in the final fit.
    valid_mask = (train_ids % 5) == 0
    pred_dummy, _ = train_one_fold(
        (train_seq[~valid_mask], train_num[~valid_mask], train_cat[~valid_mask], y_train[~valid_mask]),
        (train_seq[valid_mask], train_num[valid_mask], train_cat[valid_mask], y_train[valid_mask]),
        cat_cardinalities,
        SEED,
    )
    del pred_dummy

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
    path = SUBMISSIONS_DIR / f"test_gru_alpha{int(alpha * 100):03d}.csv"
    submission.to_csv(path, index=False)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    folds, oof = run_oof(data)
    comparison = compare_with_catboost(oof)
    best_gru = comparison[comparison["model"] == "gru"].sort_values("mape").iloc[0]
    submission_path = make_final_gru_submission(data, float(best_gru["alpha"]))

    folds.to_csv(RESULTS_DIR / "gru_folds.csv", index=False)
    oof.to_csv(RESULTS_DIR / "gru_oof.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "gru_comparison.csv", index=False)
    report = {
        "seq_len": SEQ_LEN,
        "epochs": EPOCHS,
        "best_gru": best_gru.to_dict(),
        "comparison": comparison.to_dict(orient="records"),
        "submission": str(submission_path),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
