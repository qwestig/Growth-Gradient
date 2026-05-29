from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train_2.csv"
RESULTS_DIR = ROOT / "results" / "new_baseline_multimonth_gru_seasonal"
SUBMISSIONS_DIR = ROOT / "submissions"
REMOTE_BRANCH = "origin/codex-new-baseline-gru-4763"
SEED = 2026
SEQ_LEN = 14
DEFAULT_EPOCHS = 24
DEFAULT_FINAL_EPOCHS = 18
BATCH_SIZE = 1024
PATIENCE = 5
TARGET = "РТО"
META_TARGET = "target_rto"

STATIC_CAT_COLS = [
    "Регион",
    "Торговая площадь, категориальный",
    "Дата открытия, категориальный",
]

STATIC_NUM_COLS = [
    "Численность населения",
    "Количество домохозяйств",
    "Трафик пеший, в час",
    "Трафик авто, в час",
    "Маркетплейсы, доставки, постаматы (100 м)",
    "Медицинские уч. и аптеки (300 м)",
    "Школы (300 м)",
    "Остановки (300 м)",
    "Продуктовые магазины (500 м)",
    "Пятерочки (500 м)",
    "Количество касс",
    "Флаг алкогольной лицензии",
]

BEHAVIOR_COLS = [
    "Среднее количество промо товаров в чеке",
    "Среднее количество товаров в чеке",
    "Среднее количество отмен",
    "Рабочие часы в день",
]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


@dataclass
class PreparedData:
    raw: pd.DataFrame
    rto_wide: pd.DataFrame
    behavior_wide: dict[str, pd.DataFrame]
    static_frame: pd.DataFrame
    meta_oof: pd.DataFrame
    meta_test: pd.DataFrame


class MultiMonthGRU(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        context_dim: int,
        static_num_dim: int,
        cat_cardinalities: list[int],
        hidden_dim: int = 48,
        dropout: float = 0.12,
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
            nn.Linear(static_in, 40),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(40, 16),
            nn.ReLU(),
        )
        self.context_net = nn.Sequential(nn.Linear(context_dim, 24), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 24, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, seq: torch.Tensor, context: torch.Tensor, static_num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(seq)
        seq_state = hidden[-1]
        emb = [embedding(cat[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        static_state = self.static_net(torch.cat([static_num, *emb], dim=1))
        context_state = self.context_net(context)
        return self.head(torch.cat([seq_state, static_state, context_state], dim=1)).squeeze(1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_raw_data() -> pd.DataFrame:
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    return raw


def period_id(year: int, month: int) -> int:
    return year * 12 + month


def ym_from_period(period: int) -> tuple[int, int]:
    year = period // 12
    month = period % 12
    if month == 0:
        return year - 1, 12
    return year, month


def read_remote_csv(path: str) -> pd.DataFrame:
    result = subprocess.run(
        ["git", "show", f"{REMOTE_BRANCH}:{path}"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return pd.read_csv(StringIO(result.stdout))


def build_static(raw: pd.DataFrame) -> pd.DataFrame:
    existing_num = [col for col in STATIC_NUM_COLS if col in raw.columns]
    existing_cat = [col for col in STATIC_CAT_COLS if col in raw.columns]
    static = raw.sort_values(["new_id", "Год", "Месяц"]).groupby("new_id", as_index=False)[["new_id"] + existing_num + existing_cat].first()
    static = static.set_index("new_id")
    for col in existing_num:
        static[col] = pd.to_numeric(static[col], errors="coerce")
    for col in existing_cat:
        static[col] = static[col].astype(str).fillna("__nan__")
    return static


def prepare_data() -> PreparedData:
    raw = load_raw_data()
    raw = raw.copy()
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    rto_wide = raw.pivot_table(index="new_id", columns="period", values=TARGET, aggfunc="mean").sort_index()
    behavior_wide = {
        col: raw.pivot_table(index="new_id", columns="period", values=col, aggfunc="mean").sort_index()
        for col in BEHAVIOR_COLS
        if col in raw.columns
    }
    return PreparedData(
        raw=raw,
        rto_wide=rto_wide,
        behavior_wide=behavior_wide,
        static_frame=build_static(raw),
        meta_oof=read_remote_csv("results/temporal204_meta_inputs/oof_meta_inputs_v11_v14.csv").sort_values("new_id").reset_index(drop=True),
        meta_test=read_remote_csv("results/temporal204_meta_inputs/test_meta_inputs_v11_v14.csv").sort_values("new_id").reset_index(drop=True),
    )


def sample_index(data: PreparedData, target_periods: list[int]) -> pd.DataFrame:
    rows = []
    all_ids = data.rto_wide.index.to_numpy(dtype=int)
    for target_period in target_periods:
        hist_periods = list(range(target_period - SEQ_LEN, target_period))
        needed = hist_periods + [target_period]
        ok = data.rto_wide.reindex(columns=needed).notna().all(axis=1)
        ids = all_ids[ok.to_numpy()]
        year, month = ym_from_period(target_period)
        rows.append(pd.DataFrame({"new_id": ids, "target_period": target_period, "target_year": year, "target_month": month}))
    return pd.concat(rows, ignore_index=True)


def build_sequence(data: PreparedData, sample_rows: pd.DataFrame) -> np.ndarray:
    parts = []
    for target_period, group in sample_rows.groupby("target_period", sort=False):
        store_ids = group["new_id"].to_numpy(dtype=int)
        hist_periods = list(range(int(target_period) - SEQ_LEN, int(target_period)))
        rto = data.rto_wide.loc[store_ids, hist_periods].to_numpy(dtype=float)
        log_rto = np.log(np.clip(rto, 1.0, None))
        last_log = log_rto[:, [-1]]
        rel_log_rto = log_rto - last_log
        log_growth = np.zeros_like(log_rto)
        log_growth[:, 1:] = log_rto[:, 1:] - log_rto[:, :-1]
        months = np.array([ym_from_period(p)[1] for p in hist_periods], dtype=float)
        month_sin = np.sin(2 * np.pi * months / 12)[None, :, None]
        month_cos = np.cos(2 * np.pi * months / 12)[None, :, None]
        calendar = np.repeat(np.concatenate([month_sin, month_cos], axis=2), len(store_ids), axis=0)
        behavior_parts = []
        for col in data.behavior_wide:
            values = data.behavior_wide[col].loc[store_ids, hist_periods].to_numpy(dtype=float)
            behavior_parts.append(values[:, :, None])
        behavior = np.concatenate(behavior_parts, axis=2) if behavior_parts else np.zeros((len(store_ids), SEQ_LEN, 0))
        parts.append(np.concatenate([rel_log_rto[:, :, None], log_growth[:, :, None], calendar, behavior], axis=2))
    return np.concatenate(parts, axis=0).astype(np.float32)


def seasonal_factor_for_period(data: PreparedData, target_period: int) -> float:
    target_year, target_month = ym_from_period(target_period)
    factors = []
    for source_year in range(2023, target_year):
        source_period = period_id(source_year, target_month)
        source_prev = source_period - 1
        if source_period not in data.rto_wide.columns or source_prev not in data.rto_wide.columns:
            continue
        ratio = (
            data.rto_wide[source_period].to_numpy(dtype=float)
            / np.clip(data.rto_wide[source_prev].to_numpy(dtype=float), 1.0, None)
        )
        if target_month == 3 and source_year == 2023 and target_year == 2024:
            ratio = ratio * (28.0 / 29.0)
        elif target_month == 3 and source_year == 2024 and target_year != 2024:
            ratio = ratio * (29.0 / 28.0)
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio):
            factors.append(float(np.nanmedian(ratio)))
    if not factors:
        return 1.0
    return float(np.nanmedian(factors))


def seasonal_baseline(data: PreparedData, sample_rows: pd.DataFrame) -> np.ndarray:
    out = []
    for target_period, group in sample_rows.groupby("target_period", sort=False):
        store_ids = group["new_id"].to_numpy(dtype=int)
        lag1 = data.rto_wide.loc[store_ids, int(target_period) - 1].to_numpy(dtype=float)
        out.append(lag1 * seasonal_factor_for_period(data, int(target_period)))
    return np.concatenate(out).astype(float)


def build_context(data: PreparedData, sample_rows: pd.DataFrame) -> np.ndarray:
    rows = []
    for target_period, group in sample_rows.groupby("target_period", sort=False):
        store_ids = group["new_id"].to_numpy(dtype=int)
        hist = list(range(int(target_period) - SEQ_LEN, int(target_period)))
        rto = data.rto_wide.loc[store_ids, hist].to_numpy(dtype=float)
        lag1 = rto[:, -1]
        lag2 = rto[:, -2]
        lag3 = rto[:, -3]
        lag12 = rto[:, -12]
        lag13 = rto[:, -13]
        rolling3 = np.nanmean(rto[:, -3:], axis=1)
        volatility = np.nanstd(rto[:, -3:], axis=1) / np.maximum(rolling3, 1.0)
        target_month = ym_from_period(int(target_period))[1]
        seasonal_factor = seasonal_factor_for_period(data, int(target_period))
        month_sin = np.full(len(store_ids), np.sin(2 * np.pi * target_month / 12))
        month_cos = np.full(len(store_ids), np.cos(2 * np.pi * target_month / 12))
        ctx = np.column_stack(
            [
                np.log(np.clip(lag1, 1.0, None)),
                np.log(np.clip(lag1, 1.0, None) / np.clip(lag12, 1.0, None)),
                np.log(np.clip(lag1, 1.0, None) / np.clip(lag13, 1.0, None)),
                np.log(np.clip(lag1, 1.0, None) / np.clip(rolling3, 1.0, None)),
                np.log(np.clip(lag1, 1.0, None) / np.clip(lag2, 1.0, None)),
                np.log(np.clip(lag2, 1.0, None) / np.clip(lag3, 1.0, None)),
                np.log(np.clip(lag12, 1.0, None) / np.clip(lag13, 1.0, None)),
                volatility,
                np.full(len(store_ids), np.log(max(seasonal_factor, 1e-6))),
                month_sin,
                month_cos,
            ]
        )
        rows.append(ctx)
    return np.nan_to_num(np.concatenate(rows, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def target_log_residual(data: PreparedData, sample_rows: pd.DataFrame) -> np.ndarray:
    y = []
    baseline = seasonal_baseline(data, sample_rows)
    start = 0
    for target_period, group in sample_rows.groupby("target_period", sort=False):
        store_ids = group["new_id"].to_numpy(dtype=int)
        target = data.rto_wide.loc[store_ids, int(target_period)].to_numpy(dtype=float)
        end = start + len(store_ids)
        y.append(np.log(np.clip(target, 1.0, None) / np.clip(baseline[start:end], 1.0, None)))
        start = end
    return np.concatenate(y).astype(np.float32)


def fit_scaler(x: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=axes)
    std = np.nanstd(x, axis=axes)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        out = (x - mean[None, None, :]) / std[None, None, :]
    else:
        out = (x - mean[None, :]) / std[None, :]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_static_encoders(static_frame: pd.DataFrame, train_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]], list[str], list[str]]:
    existing_num = [col for col in STATIC_NUM_COLS if col in static_frame.columns]
    existing_cat = [col for col in STATIC_CAT_COLS if col in static_frame.columns]
    train_static = static_frame.loc[train_ids]
    num_mean = train_static[existing_num].astype(float).mean().to_numpy(dtype=np.float32)
    num_std = train_static[existing_num].astype(float).std().replace(0, 1).fillna(1).to_numpy(dtype=np.float32)
    encoders = {}
    for col in existing_cat:
        values = train_static[col].astype(str).fillna("__nan__").unique().tolist()
        encoders[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return num_mean, num_std, encoders, existing_num, existing_cat


def transform_static(
    static_frame: pd.DataFrame,
    store_ids: np.ndarray,
    num_mean: np.ndarray,
    num_std: np.ndarray,
    encoders: dict[str, dict[str, int]],
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    static = static_frame.loc[store_ids]
    numeric = static[num_cols].astype(float).to_numpy(dtype=np.float32)
    numeric = np.nan_to_num((numeric - num_mean[None, :]) / num_std[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    cats = []
    for col in cat_cols:
        cats.append(static[col].astype(str).fillna("__nan__").map(encoders[col]).fillna(0).to_numpy(dtype=np.int64))
    cat_arr = np.stack(cats, axis=1) if cats else np.zeros((len(store_ids), 0), dtype=np.int64)
    return numeric.astype(np.float32), cat_arr


def predict_model(model: nn.Module, seq: np.ndarray, context: np.ndarray, static_num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(seq), torch.from_numpy(context), torch.from_numpy(static_num), torch.from_numpy(cat)),
        batch_size=4096,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_seq, batch_ctx, batch_num, batch_cat in loader:
            pred = model(
                batch_seq.to(DEVICE),
                batch_ctx.to(DEVICE),
                batch_num.to(DEVICE),
                batch_cat.to(DEVICE),
            )
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds)


def train_model(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    cat_cardinalities: list[int],
    epochs: int,
    seed: int,
) -> tuple[MultiMonthGRU, float]:
    set_seed(seed)
    train_seq, train_ctx, train_num, train_cat, y_train = train_data
    model = MultiMonthGRU(train_seq.shape[-1], train_ctx.shape[1], train_num.shape[1], cat_cardinalities).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.015)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_seq),
            torch.from_numpy(train_ctx),
            torch.from_numpy(train_num),
            torch.from_numpy(train_cat),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type == "cuda",
    )
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(epochs):
        model.train()
        for batch_seq, batch_ctx, batch_num, batch_cat, batch_y in loader:
            batch_seq = batch_seq.to(DEVICE, non_blocking=True)
            batch_ctx = batch_ctx.to(DEVICE, non_blocking=True)
            batch_num = batch_num.to(DEVICE, non_blocking=True)
            batch_cat = batch_cat.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_seq, batch_ctx, batch_num, batch_cat), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if valid_data is None:
            continue
        valid_seq, valid_ctx, valid_num, valid_cat, y_valid = valid_data
        valid_pred = predict_model(model, valid_seq, valid_ctx, valid_num, valid_cat)
        valid_loss = float(np.mean(np.abs(valid_pred - y_valid)))
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
    return model, best_loss


def build_arrays(
    data: PreparedData,
    rows: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    seq_raw = build_sequence(data, rows)
    ctx_raw = build_context(data, rows)
    y = target_log_residual(data, rows)
    seq_mean, seq_std = fit_scaler(seq_raw[train_mask], axes=(0, 1))
    ctx_mean, ctx_std = fit_scaler(ctx_raw[train_mask], axes=(0,))
    seq = apply_scaler(seq_raw, seq_mean, seq_std)
    ctx = apply_scaler(ctx_raw, ctx_mean, ctx_std)
    train_ids = rows.loc[train_mask, "new_id"].to_numpy(dtype=int)
    num_mean, num_std, encoders, num_cols, cat_cols = fit_static_encoders(data.static_frame, train_ids)
    static_num, cat = transform_static(data.static_frame, rows["new_id"].to_numpy(dtype=int), num_mean, num_std, encoders, num_cols, cat_cols)
    meta = {
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "ctx_mean": ctx_mean,
        "ctx_std": ctx_std,
        "num_mean": num_mean,
        "num_std": num_std,
        "encoders": encoders,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    return seq, ctx, static_num, cat, y, meta


def build_inference_arrays(data: PreparedData, rows: pd.DataFrame, meta: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq = apply_scaler(build_sequence(data, rows), meta["seq_mean"], meta["seq_std"])
    ctx = apply_scaler(build_context(data, rows), meta["ctx_mean"], meta["ctx_std"])
    static_num, cat = transform_static(
        data.static_frame,
        rows["new_id"].to_numpy(dtype=int),
        meta["num_mean"],
        meta["num_std"],
        meta["encoders"],
        meta["num_cols"],
        meta["cat_cols"],
    )
    return seq, ctx, static_num, cat


def run_oof(data: PreparedData, epochs: int) -> pd.DataFrame:
    target_periods = [period_id(2024, month) for month in range(3, 13)] + [period_id(2025, month) for month in range(1, 3)]
    rows = sample_index(data, target_periods).sort_values(["target_period", "new_id"]).reset_index(drop=True)
    march_period = period_id(2024, 3)
    march_mask = rows["target_period"].eq(march_period).to_numpy()
    store_ids = rows["new_id"].to_numpy(dtype=int)
    parts = []
    fold_rows = []
    for fold in range(5):
        valid_store = store_ids % 5 == fold
        train_mask = ~valid_store
        valid_mask = valid_store & march_mask
        seq, ctx, static_num, cat, y, _meta = build_arrays(data, rows, train_mask)
        model, valid_loss = train_model(
            (seq[train_mask], ctx[train_mask], static_num[train_mask], cat[train_mask], y[train_mask]),
            (seq[valid_mask], ctx[valid_mask], static_num[valid_mask], cat[valid_mask], y[valid_mask]),
            [len(_meta["encoders"][col]) for col in _meta["cat_cols"]],
            epochs=epochs,
            seed=SEED + fold,
        )
        pred = predict_model(model, seq[valid_mask], ctx[valid_mask], static_num[valid_mask], cat[valid_mask])
        fold_rows.append({"fold": fold, "valid_log_growth_mae": valid_loss, "n_train": int(train_mask.sum()), "n_valid_march": int(valid_mask.sum())})
        part = rows.loc[valid_mask, ["new_id", "target_period", "target_year", "target_month"]].copy()
        part["fold"] = fold
        part["pred_log_growth_mmgru"] = pred
        parts.append(part)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "fold_summary.csv", index=False)
    oof = pd.concat(parts, ignore_index=True).sort_values("new_id").reset_index(drop=True)
    oof = oof.merge(data.meta_oof[["new_id", META_TARGET, "pred265_oof"]], on="new_id", how="left")
    feb_2024 = data.rto_wide.loc[oof["new_id"].to_numpy(dtype=int), period_id(2024, 2)].to_numpy(dtype=float)
    oof["seasonal_baseline"] = seasonal_baseline(data, oof[["new_id", "target_period", "target_year", "target_month"]])
    oof["pred_mmgru_raw"] = oof["seasonal_baseline"].to_numpy(float) * np.exp(oof["pred_log_growth_mmgru"].to_numpy(float))
    return oof


def evaluate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    y = oof[META_TARGET].to_numpy(float)
    anchor = oof["pred265_oof"].to_numpy(float)
    raw = oof["pred_mmgru_raw"].to_numpy(float)
    raw_delta = np.log(np.clip(raw, 1.0, None) / np.clip(anchor, 1.0, None))
    raw_delta_centered = raw_delta - np.nanmedian(raw_delta)
    rows = [
        {"variant": "pred265_oof", "alpha": 0.0, "clip": "", "center": "none", "mape": mape(y, anchor), "mean_abs_shift_pct": 0.0, "max_abs_shift_pct": 0.0},
        {"variant": "mmgru_raw", "alpha": 1.0, "clip": "", "center": "none", "mape": mape(y, raw), "mean_abs_shift_pct": float(np.mean(np.abs(raw / anchor - 1)) * 100), "max_abs_shift_pct": float(np.max(np.abs(raw / anchor - 1)) * 100)},
    ]
    for center_name, delta in [("median", raw_delta_centered), ("none", raw_delta)]:
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]:
            for clip in [0.003, 0.005, 0.008, 0.012, 0.020]:
                corr = np.clip(alpha * delta, -clip, clip)
                pred = anchor * np.exp(corr)
                ratio = pred / anchor - 1.0
                rows.append(
                    {
                        "variant": f"pred265_to_mmgru_{center_name}",
                        "alpha": alpha,
                        "clip": clip,
                        "center": center_name,
                        "mape": mape(y, pred),
                        "mean_abs_shift_pct": float(np.mean(np.abs(ratio)) * 100),
                        "max_abs_shift_pct": float(np.max(np.abs(ratio)) * 100),
                    }
                )
    out = pd.DataFrame(rows)
    out["delta_vs_265_oof"] = out["mape"] - float(out.loc[out["variant"].eq("pred265_oof"), "mape"].iloc[0])
    return out.sort_values(["mape", "mean_abs_shift_pct"]).reset_index(drop=True)


def train_final_predict(data: PreparedData, final_epochs: int, seeds: list[int]) -> pd.DataFrame:
    target_periods = [period_id(2024, month) for month in range(3, 13)] + [period_id(2025, month) for month in range(1, 3)]
    rows = sample_index(data, target_periods).sort_values(["target_period", "new_id"]).reset_index(drop=True)
    train_mask = np.ones(len(rows), dtype=bool)
    seq, ctx, static_num, cat, y, meta = build_arrays(data, rows, train_mask)
    test_rows = pd.DataFrame(
        {
            "new_id": data.meta_test["new_id"].to_numpy(dtype=int),
            "target_period": period_id(2025, 3),
            "target_year": 2025,
            "target_month": 3,
        }
    )
    test_seq, test_ctx, test_num, test_cat = build_inference_arrays(data, test_rows, meta)
    preds = []
    for seed in seeds:
        model, _ = train_model(
            (seq, ctx, static_num, cat, y),
            None,
            [len(meta["encoders"][col]) for col in meta["cat_cols"]],
            epochs=final_epochs,
            seed=seed,
        )
        preds.append(predict_model(model, test_seq, test_ctx, test_num, test_cat))
    pred_log_growth = np.mean(np.stack(preds, axis=0), axis=0)
    out = test_rows[["new_id"]].copy()
    out["pred_log_growth_mmgru"] = pred_log_growth
    out["seasonal_baseline"] = seasonal_baseline(data, test_rows)
    out["pred_mmgru_raw"] = out["seasonal_baseline"].to_numpy(float) * np.exp(pred_log_growth)
    out = out.merge(data.meta_test[["new_id", "pred265_submission"]], on="new_id", how="left")
    return out


def write_candidates(test_pred: pd.DataFrame, oof: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    best = summary[summary["variant"].str.startswith("pred265_to_mmgru")].head(8).copy()
    if best.empty:
        best = pd.DataFrame(
            [
                {"variant": "pred265_to_mmgru_median", "alpha": 0.10, "clip": 0.005, "center": "median"},
                {"variant": "pred265_to_mmgru_median", "alpha": 0.20, "clip": 0.008, "center": "median"},
            ]
        )
    test_delta = np.log(
        np.clip(test_pred["pred_mmgru_raw"].to_numpy(float), 1.0, None)
        / np.clip(test_pred["pred265_submission"].to_numpy(float), 1.0, None)
    )
    rows = []
    for candidate_id, (_, row) in enumerate(best.iterrows(), start=350):
        delta = test_delta.copy()
        if row["center"] == "median":
            delta = delta - np.nanmedian(delta)
        corr = np.clip(float(row["alpha"]) * delta, -float(row["clip"]), float(row["clip"]))
        pred = test_pred["pred265_submission"].to_numpy(float) * np.exp(corr)
        file_name = f"stage2_candidate_{candidate_id}_multimonth_gru_seasonal_a{str(row['alpha']).replace('.', 'p')}_c{str(row['clip']).replace('.', 'p')}_{row['center']}.csv"
        out = test_pred[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        ratio = pred / test_pred["pred265_submission"].to_numpy(float) - 1.0
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                **row.to_dict(),
                "test_mean_abs_shift_pct": float(np.mean(np.abs(ratio)) * 100),
                "test_max_abs_shift_pct": float(np.max(np.abs(ratio)) * 100),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--final-epochs", type=int, default=DEFAULT_FINAL_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED, SEED + 17])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    data = prepare_data()
    oof = run_oof(data, epochs=args.epochs)
    oof.to_csv(RESULTS_DIR / "oof.csv", index=False)
    summary = evaluate_oof(oof)
    summary.to_csv(RESULTS_DIR / "oof_summary.csv", index=False)
    test_pred = train_final_predict(data, final_epochs=args.final_epochs, seeds=args.seeds)
    test_pred.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    candidates = write_candidates(test_pred, oof, summary)
    candidates.to_csv(RESULTS_DIR / "submission_candidates.csv", index=False)
    metadata = {
        "current_public_anchor": {"score": 90.78, "mape": 4.7214609684},
        "seq_len": SEQ_LEN,
        "epochs": args.epochs,
        "final_epochs": args.final_epochs,
        "seeds": args.seeds,
        "device": str(DEVICE),
        "train_target_periods": "2024-03..2025-02",
        "target": "next-month log residual versus month-specific seasonal baseline; candidates are clipped blends from pred265 toward raw multimonth GRU",
    }
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OOF summary:")
    print(summary.head(25).to_string(index=False))
    print("\nSubmission candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
