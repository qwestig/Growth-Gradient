from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from heuristic_baseline_direction_lab import (
    RESULTS_DIR,
    SUBMISSIONS_DIR,
    TARGET,
    HeuristicSpec,
    build_target_frame,
    factors_for_spec,
    load_data,
    mape,
    predict_from_spec,
)


SEED = 2026
SEQ_LEN = 14
EPOCHS = 80
FINAL_EPOCHS = 35
BATCH_SIZE = 512
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = RESULTS_DIR / "new_baseline_gru_residual"
NEW_BASE_SPEC = HeuristicSpec(0, "cal050_mom000", 0.50, momentum_exp=0.0)
MOMENTUM_TRAIN_BINS = [-np.inf, 0.935766244455909, 0.9524077370639855, 0.9656151000316019, 0.9815696834781942, np.inf]

STATIC_CAT_COLS = [
    "Дата открытия, категориальный",
    "Торговая площадь, категориальный",
    "Регион",
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


@dataclass
class PreparedData:
    raw: pd.DataFrame
    rto_wide: pd.DataFrame
    behavior_wide: dict[str, pd.DataFrame]
    static_frame: pd.DataFrame
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame


class GRUResidual(nn.Module):
    def __init__(self, seq_dim: int, static_num_dim: int, cat_cardinalities: list[int], hidden_dim: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden_dim, num_layers=1, batch_first=True)
        self.embeddings = nn.ModuleList()
        emb_dims = []
        for cardinality in cat_cardinalities:
            dim = min(12, max(2, int(np.sqrt(cardinality + 1)) + 1))
            self.embeddings.append(nn.Embedding(cardinality + 1, dim, padding_idx=0))
            emb_dims.append(dim)
        static_in = static_num_dim + sum(emb_dims)
        self.static_net = nn.Sequential(nn.Linear(static_in, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden_dim + 16, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

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


def build_static(raw: pd.DataFrame) -> pd.DataFrame:
    existing_num = [col for col in STATIC_NUM_COLS if col in raw.columns]
    existing_cat = [col for col in STATIC_CAT_COLS if col in raw.columns]
    static = raw.sort_values(["new_id", "period"]).groupby("new_id", as_index=False)[["new_id"] + existing_num + existing_cat].first().set_index("new_id")
    for col in existing_num:
        static[col] = pd.to_numeric(static[col], errors="coerce")
    for col in existing_cat:
        static[col] = static[col].astype(str).fillna("__nan__")
    return static


def prepare_data() -> PreparedData:
    raw, rto_label_wide, static_for_base = load_data()
    raw = raw.copy()
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    rto_wide = raw.pivot_table(index="new_id", columns="period", values=TARGET, aggfunc="mean")
    behavior_wide = {
        col: raw.pivot_table(index="new_id", columns="period", values=col, aggfunc="mean")
        for col in BEHAVIOR_COLS
        if col in raw.columns
    }
    static_frame = build_static(raw)

    train_features = build_target_frame(rto_label_wide, static_for_base, 2024, 3, include_target=True)
    test_features = build_target_frame(rto_label_wide, static_for_base, 2025, 3, include_target=False)
    train_factor, test_factor, _val_global, _final_global = factors_for_spec(NEW_BASE_SPEC, train_features, test_features, rto_label_wide, static_for_base)
    train_features["new_base"] = predict_from_spec(train_features, train_factor, NEW_BASE_SPEC)
    test_features["new_base"] = predict_from_spec(test_features, test_factor, NEW_BASE_SPEC)
    train_features = train_features[(train_features[TARGET] > 0) & (train_features["new_base"] > 0)].copy()
    train_features["target_log_ratio"] = np.log(train_features[TARGET] / train_features["new_base"])
    train_features["momentum_train_bin"] = momentum_bin(train_features)
    test_features["momentum_train_bin"] = momentum_bin(test_features)

    return PreparedData(
        raw=raw,
        rto_wide=rto_wide,
        behavior_wide=behavior_wide,
        static_frame=static_frame,
        train_rows=train_features.sort_values("new_id").reset_index(drop=True),
        test_rows=test_features.sort_values("new_id").reset_index(drop=True),
    )


def momentum_bin(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame["momentum_raw"], bins=MOMENTUM_TRAIN_BINS, labels=[f"q{i}" for i in range(5)], include_lowest=True).astype(str)


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
    for col in data.behavior_wide:
        values = data.behavior_wide[col].loc[store_ids, hist_periods].to_numpy(dtype=float)
        behavior_parts.append(values[:, :, None])
    behavior = np.concatenate(behavior_parts, axis=2) if behavior_parts else np.zeros((len(store_ids), len(hist_periods), 0))
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


def transform_static(static_frame: pd.DataFrame, store_ids: np.ndarray, num_mean: np.ndarray, num_std: np.ndarray, encoders: dict[str, dict[str, int]], num_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    static = static_frame.loc[store_ids]
    numeric = static[num_cols].astype(float).to_numpy(dtype=np.float32)
    numeric = np.nan_to_num((numeric - num_mean[None, :]) / num_std[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    cats = []
    for col in cat_cols:
        mapping = encoders[col]
        cats.append(static[col].astype(str).fillna("__nan__").map(mapping).fillna(0).to_numpy(dtype=np.int64))
    cat_arr = np.stack(cats, axis=1) if cats else np.zeros((len(store_ids), 0), dtype=np.int64)
    return numeric.astype(np.float32), cat_arr


def train_one_fold(train_data, valid_data, cat_cardinalities: list[int], seed: int) -> tuple[np.ndarray, float]:
    set_seed(seed)
    train_seq, train_num, train_cat, y_train = train_data
    valid_seq, valid_num, valid_cat, y_valid = valid_data
    model = GRUResidual(train_seq.shape[-1], train_num.shape[1], cat_cardinalities).to(DEVICE)
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
        pin_memory=DEVICE.type == "cuda",
    )
    valid_tensors = (
        torch.from_numpy(valid_seq).to(DEVICE),
        torch.from_numpy(valid_num).to(DEVICE),
        torch.from_numpy(valid_cat).to(DEVICE),
        torch.from_numpy(y_valid.astype(np.float32)).to(DEVICE),
    )
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(EPOCHS):
        model.train()
        for batch_seq, batch_num, batch_cat, batch_y in loader:
            batch_seq = batch_seq.to(DEVICE, non_blocking=True)
            batch_num = batch_num.to(DEVICE, non_blocking=True)
            batch_cat = batch_cat.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_seq, batch_num, batch_cat), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_tensors[0], valid_tensors[1], valid_tensors[2]), valid_tensors[3]).detach().cpu().item())
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
        pred = model(valid_tensors[0], valid_tensors[1], valid_tensors[2]).detach().cpu().numpy()
    return pred, best_loss


def run_oof(data: PreparedData) -> pd.DataFrame:
    target = data.train_rows.copy()
    store_ids = target["new_id"].to_numpy()
    full_seq = build_sequence(data, store_ids, period_id(2024, 3))
    y = target["target_log_ratio"].to_numpy(dtype=np.float32)
    parts = []
    fold_rows = []
    for fold in range(5):
        valid_mask = (store_ids % 5) == fold
        train_ids = store_ids[~valid_mask]
        valid_ids = store_ids[valid_mask]
        seq_mean, seq_std = fit_sequence_scaler(full_seq[~valid_mask])
        train_seq = apply_sequence_scaler(full_seq[~valid_mask], seq_mean, seq_std)
        valid_seq = apply_sequence_scaler(full_seq[valid_mask], seq_mean, seq_std)
        num_mean, num_std, encoders, num_cols, cat_cols = fit_static_encoders(data.static_frame, train_ids)
        train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders, num_cols, cat_cols)
        valid_num, valid_cat = transform_static(data.static_frame, valid_ids, num_mean, num_std, encoders, num_cols, cat_cols)
        pred_log, best_loss = train_one_fold((train_seq, train_num, train_cat, y[~valid_mask]), (valid_seq, valid_num, valid_cat, y[valid_mask]), [len(encoders[col]) for col in cat_cols], SEED + fold)
        valid_rows = target.loc[valid_mask].copy()
        valid_rows["fold"] = fold
        valid_rows["pred_log_gru"] = pred_log
        parts.append(valid_rows[["new_id", TARGET, "new_base", "momentum_train_bin", "fold", "pred_log_gru"]])
        fold_rows.append({"fold": fold, "valid_loss": best_loss, "base_mape": mape(valid_rows[TARGET], valid_rows["new_base"])})
    oof = pd.concat(parts, ignore_index=True)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    return oof


def center_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    med = values.groupby(groups, observed=True).transform("median")
    return values - med.fillna(float(values.median()))


def evaluate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = [{"variant": "new_base", "alpha": 0.0, "center": "none", "mape": mape(oof[TARGET], oof["new_base"])}]
    for center in ["none", "global", "momentum_bin"]:
        if center == "none":
            pred_log = oof["pred_log_gru"]
        elif center == "global":
            pred_log = oof["pred_log_gru"] - float(oof["pred_log_gru"].median())
        else:
            pred_log = center_by_group(oof["pred_log_gru"], oof["momentum_train_bin"])
        for alpha in [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3]:
            pred = oof["new_base"].to_numpy(dtype=float) * np.exp(alpha * pred_log.to_numpy(dtype=float))
            rows.append({"variant": f"new_gru_{center}", "alpha": alpha, "center": center, "mape": mape(oof[TARGET], pred)})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def train_final_and_predict(data: PreparedData) -> np.ndarray:
    set_seed(SEED)
    train = data.train_rows.copy()
    store_ids = train["new_id"].to_numpy()
    y = train["target_log_ratio"].to_numpy(dtype=np.float32)
    seq_raw = build_sequence(data, store_ids, period_id(2024, 3))
    seq_mean, seq_std = fit_sequence_scaler(seq_raw)
    seq = apply_sequence_scaler(seq_raw, seq_mean, seq_std)
    num_mean, num_std, encoders, num_cols, cat_cols = fit_static_encoders(data.static_frame, store_ids)
    num, cat = transform_static(data.static_frame, store_ids, num_mean, num_std, encoders, num_cols, cat_cols)
    valid_mask = (store_ids % 5) == 0
    pred_dummy, _ = train_one_fold(
        (seq[~valid_mask], num[~valid_mask], cat[~valid_mask], y[~valid_mask]),
        (seq[valid_mask], num[valid_mask], cat[valid_mask], y[valid_mask]),
        [len(encoders[col]) for col in cat_cols],
        SEED,
    )
    del pred_dummy
    model = GRUResidual(seq.shape[-1], num.shape[1], [len(encoders[col]) for col in cat_cols]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(seq), torch.from_numpy(num), torch.from_numpy(cat), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type == "cuda",
    )
    for _epoch in range(FINAL_EPOCHS):
        model.train()
        for batch_seq, batch_num, batch_cat, batch_y in loader:
            batch_seq = batch_seq.to(DEVICE, non_blocking=True)
            batch_num = batch_num.to(DEVICE, non_blocking=True)
            batch_cat = batch_cat.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_seq, batch_num, batch_cat), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    test_ids = data.test_rows["new_id"].to_numpy()
    test_seq = apply_sequence_scaler(build_sequence(data, test_ids, period_id(2025, 3)), seq_mean, seq_std)
    test_num, test_cat = transform_static(data.static_frame, test_ids, num_mean, num_std, encoders, num_cols, cat_cols)
    model.eval()
    with torch.no_grad():
        return (
            model(
                torch.from_numpy(test_seq).to(DEVICE),
                torch.from_numpy(test_num).to(DEVICE),
                torch.from_numpy(test_cat).to(DEVICE),
            )
            .detach()
            .cpu()
            .numpy()
        )


def write_candidates(data: PreparedData, pred_log: np.ndarray, oof: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = data.test_rows.copy()
    test["pred_log_gru"] = pred_log
    test["pred_log_global"] = test["pred_log_gru"] - float(test["pred_log_gru"].median())
    test["pred_log_mombin"] = center_by_group(test["pred_log_gru"], test["momentum_train_bin"])
    candidate_specs = [
        (145, "newgru_faithful_raw_a070", "pred_log_gru", 0.70),
        (146, "newgru_faithful_raw_a085", "pred_log_gru", 0.85),
        (147, "newgru_faithful_raw_a100", "pred_log_gru", 1.00),
        (148, "newgru_faithful_global_a070", "pred_log_global", 0.70),
        (149, "newgru_faithful_global_a085", "pred_log_global", 0.85),
        (150, "newgru_faithful_global_a100", "pred_log_global", 1.00),
        (151, "newgru_faithful_mombin_a070", "pred_log_mombin", 0.70),
        (152, "newgru_faithful_mombin_a100", "pred_log_mombin", 1.00),
        (153, "newgru_faithful_mombin_a115", "pred_log_mombin", 1.15),
        (154, "newgru_faithful_mombin_a130", "pred_log_mombin", 1.30),
    ]
    for candidate_id, name, col, alpha in candidate_specs:
        pred = test["new_base"].to_numpy(dtype=float) * np.exp(alpha * test[col].to_numpy(dtype=float))
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append({"candidate_id": candidate_id, "file": file_name, "name": name, "alpha": alpha, "center": col, "test_mean_rto": float(np.mean(pred)), "test_total_rto": float(np.sum(pred))})
    out_summary = pd.DataFrame(rows)
    out_summary.to_csv(OUT_DIR / "submission_candidates.csv", index=False)

    log_path = RESULTS_DIR / "public_experiment_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        existing = set(log["candidate_id"].astype(str))
        new_rows = []
        for _, row in out_summary.iterrows():
            if str(row["candidate_id"]) in existing:
                continue
            new_rows.append({"candidate_id": row["candidate_id"], "file": row["file"], "family": "new_baseline_gru_retrained", "hypothesis": "Fresh GRU residual trained directly on new calendar/momentum-zero baseline", "public_score": "", "notes": "fill public_score after submission", "public_mape": "", "delta_mape_vs_current_best": "", "delta_score_vs_current_best": ""})
        if new_rows:
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)
    return out_summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")
    data = prepare_data()
    oof = run_oof(data)
    oof.to_csv(OUT_DIR / "oof.csv", index=False)
    summary = evaluate_oof(oof)
    summary.to_csv(OUT_DIR / "oof_summary.csv", index=False)
    pred_log = train_final_and_predict(data)
    candidates = write_candidates(data, pred_log, oof, summary)
    metadata = {"seq_len": SEQ_LEN, "epochs": EPOCHS, "final_epochs": FINAL_EPOCHS, "base_spec": NEW_BASE_SPEC.__dict__}
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OOF summary:")
    print(summary.head(20).to_string(index=False))
    print("\nSubmission candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
