from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = ROOT / "analysis" / "new_strategy"
sys.path.append(str(STRATEGY_DIR))

import train_conditional_feb_anomaly_baseline as fb  # noqa: E402


OUT_DIR = ROOT / "results" / "new_strategy_feb_anomaly_gru"
SUBMISSIONS_DIR = ROOT / "submissions"
TRAIN_PATH = ROOT / "train_2.csv"
TARGET = "РТО"
META_TARGET = "target_rto"
SEED = 2026
SEQ_LEN = 14
DEFAULT_EPOCHS = 80
DEFAULT_FINAL_EPOCHS = 35
BATCH_SIZE = 512
PATIENCE = 10

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
        static_state = self.static_net(torch.cat([static_num, *emb], dim=1))
        return self.head(torch.cat([seq_state, static_state], dim=1)).squeeze(1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def base_spec(
    normal_method: str = "trend_dec_jan",
    trigger_mode: str = "low_after_stable",
    threshold: float = 0.13,
    correction_weight: float = 1.0,
    factor_lambda: float = 0.60,
    max_level_ratio: float = 1.75,
) -> fb.Spec:
    cap_label = "" if max_level_ratio >= 999.0 else f"_cap{str(max_level_ratio).replace('.', 'p')}"
    return fb.Spec(
        name=(
            f"feb_anom_{normal_method}_{trigger_mode}_"
            f"t{str(threshold).replace('.', 'p')}_w{str(correction_weight).replace('.', 'p')}_"
            f"cal{str(factor_lambda).replace('.', 'p')}{cap_label}"
        ),
        normal_method=normal_method,
        trigger_mode=trigger_mode,
        threshold=threshold,
        correction_weight=correction_weight,
        factor_lambda=factor_lambda,
        factor_method="median",
        max_level_ratio=max_level_ratio,
    )


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


def prepare_data(spec: fb.Spec) -> PreparedData:
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw["period"] = raw["Год"] * 12 + raw["Месяц"]
    rto_wide = raw.pivot_table(index="new_id", columns="period", values=TARGET, aggfunc="mean").sort_index()
    behavior_wide = {
        col: raw.pivot_table(index="new_id", columns="period", values=col, aggfunc="mean").sort_index()
        for col in BEHAVIOR_COLS
        if col in raw.columns
    }
    rto_for_base, static_for_base, meta_oof, meta_test = fb.load_data()
    train_rows = fb.target_frame(rto_for_base, static_for_base, meta_oof, target_year=2024, is_train=True)
    test_rows = fb.target_frame(rto_for_base, static_for_base, meta_test, target_year=2025, is_train=False)
    train_rows["new_base"], _level, train_rows["anom_trigger"] = fb.forecast(train_rows, rto_for_base, 2024, spec)
    test_rows["new_base"], _level, test_rows["anom_trigger"] = fb.forecast(test_rows, rto_for_base, 2025, spec)
    train_rows = train_rows[(train_rows["target"] > 0) & (train_rows["new_base"] > 0)].copy()
    train_rows["target_log_ratio"] = np.log(train_rows["target"] / train_rows["new_base"])
    train_rows["anomaly_bin"] = np.where(train_rows["anom_trigger"], "triggered", "normal")
    test_rows["anomaly_bin"] = np.where(test_rows["anom_trigger"], "triggered", "normal")
    return PreparedData(
        raw=raw,
        rto_wide=rto_wide,
        behavior_wide=behavior_wide,
        static_frame=build_static(raw),
        train_rows=train_rows.sort_values("new_id").reset_index(drop=True),
        test_rows=test_rows.sort_values("new_id").reset_index(drop=True),
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
    for col in data.behavior_wide:
        values = data.behavior_wide[col].loc[store_ids, hist_periods].to_numpy(dtype=float)
        behavior_parts.append(values[:, :, None])
    behavior = np.concatenate(behavior_parts, axis=2) if behavior_parts else np.zeros((len(store_ids), len(hist_periods), 0))
    seq = np.concatenate([rel_log_rto[:, :, None], log_growth[:, :, None], calendar, behavior], axis=2)
    return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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


def predict_model(model: nn.Module, seq: np.ndarray, static_num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(seq), torch.from_numpy(static_num), torch.from_numpy(cat)),
        batch_size=4096,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_seq, batch_num, batch_cat in loader:
            pred = model(batch_seq.to(DEVICE), batch_num.to(DEVICE), batch_cat.to(DEVICE))
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds)


def train_model(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    cat_cardinalities: list[int],
    epochs: int,
    seed: int,
) -> tuple[GRUResidual, float]:
    set_seed(seed)
    train_seq, train_num, train_cat, y_train = train_data
    model = GRUResidual(train_seq.shape[-1], train_num.shape[1], cat_cardinalities).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.02)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_num), torch.from_numpy(train_cat), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type == "cuda",
    )
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(epochs):
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
        if valid_data is None:
            continue
        valid_seq, valid_num, valid_cat, y_valid = valid_data
        pred = predict_model(model, valid_seq, valid_num, valid_cat)
        valid_loss = float(np.mean(np.abs(pred - y_valid)))
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


def run_oof(data: PreparedData, epochs: int) -> pd.DataFrame:
    target = data.train_rows.copy()
    store_ids = target["new_id"].to_numpy(dtype=int)
    full_seq = build_sequence(data, store_ids, fb.period_id(2024, 3))
    y = target["target_log_ratio"].to_numpy(dtype=np.float32)
    parts = []
    fold_rows = []
    for fold in range(5):
        valid_mask = store_ids % 5 == fold
        train_ids = store_ids[~valid_mask]
        valid_ids = store_ids[valid_mask]
        seq_mean, seq_std = fit_sequence_scaler(full_seq[~valid_mask])
        train_seq = apply_sequence_scaler(full_seq[~valid_mask], seq_mean, seq_std)
        valid_seq = apply_sequence_scaler(full_seq[valid_mask], seq_mean, seq_std)
        num_mean, num_std, encoders, num_cols, cat_cols = fit_static_encoders(data.static_frame, train_ids)
        train_num, train_cat = transform_static(data.static_frame, train_ids, num_mean, num_std, encoders, num_cols, cat_cols)
        valid_num, valid_cat = transform_static(data.static_frame, valid_ids, num_mean, num_std, encoders, num_cols, cat_cols)
        model, valid_loss = train_model(
            (train_seq, train_num, train_cat, y[~valid_mask]),
            (valid_seq, valid_num, valid_cat, y[valid_mask]),
            [len(encoders[col]) for col in cat_cols],
            epochs=epochs,
            seed=SEED + fold,
        )
        pred_log = predict_model(model, valid_seq, valid_num, valid_cat)
        valid_rows = target.loc[valid_mask].copy()
        valid_rows["fold"] = fold
        valid_rows["pred_log_gru"] = pred_log
        parts.append(valid_rows[["new_id", "target", "new_base", "anomaly_bin", "anom_trigger", "fold", "pred_log_gru"]])
        fold_rows.append({"fold": fold, "valid_log_mae": valid_loss, "base_mape": fb.mape(valid_rows["target"], valid_rows["new_base"])})
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    return pd.concat(parts, ignore_index=True).sort_values("new_id").reset_index(drop=True)


def center_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    med = values.groupby(groups, observed=True).transform("median")
    return values - med.fillna(float(values.median()))


def evaluate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    y = oof["target"].to_numpy(float)
    base = oof["new_base"].to_numpy(float)
    rows = [
        {
            "variant": "feb_anomaly_base",
            "alpha": 0.0,
            "clip": "",
            "center": "none",
            "mape": fb.mape(y, base),
            "mean_abs_shift_pct": 0.0,
            "max_abs_shift_pct": 0.0,
        }
    ]
    centered = {
        "none": oof["pred_log_gru"].to_numpy(float),
        "global": (oof["pred_log_gru"] - float(oof["pred_log_gru"].median())).to_numpy(float),
        "anomaly_bin": center_by_group(oof["pred_log_gru"], oof["anomaly_bin"]).to_numpy(float),
    }
    for center_name, pred_log in centered.items():
        for alpha in [0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 1.00]:
            for clip in [0.003, 0.005, 0.008, 0.012, 0.020, 0.040, 999.0]:
                corr = alpha * pred_log
                if clip < 999.0:
                    corr = np.clip(corr, -clip, clip)
                pred = base * np.exp(corr)
                ratio = pred / base - 1.0
                rows.append(
                    {
                        "variant": "feb_anomaly_gru",
                        "alpha": alpha,
                        "clip": "" if clip >= 999.0 else clip,
                        "center": center_name,
                        "mape": fb.mape(y, pred),
                        "mean_abs_shift_pct": float(np.mean(np.abs(ratio)) * 100),
                        "max_abs_shift_pct": float(np.max(np.abs(ratio)) * 100),
                    }
                )
    out = pd.DataFrame(rows)
    out["delta_vs_base"] = out["mape"] - float(out.loc[out["variant"].eq("feb_anomaly_base"), "mape"].iloc[0])
    return out.sort_values(["mape", "mean_abs_shift_pct"]).reset_index(drop=True)


def train_final_predict(data: PreparedData, final_epochs: int, seeds: list[int]) -> pd.DataFrame:
    train = data.train_rows.copy()
    store_ids = train["new_id"].to_numpy(dtype=int)
    y = train["target_log_ratio"].to_numpy(dtype=np.float32)
    seq_raw = build_sequence(data, store_ids, fb.period_id(2024, 3))
    seq_mean, seq_std = fit_sequence_scaler(seq_raw)
    seq = apply_sequence_scaler(seq_raw, seq_mean, seq_std)
    num_mean, num_std, encoders, num_cols, cat_cols = fit_static_encoders(data.static_frame, store_ids)
    num, cat = transform_static(data.static_frame, store_ids, num_mean, num_std, encoders, num_cols, cat_cols)
    test_ids = data.test_rows["new_id"].to_numpy(dtype=int)
    test_seq = apply_sequence_scaler(build_sequence(data, test_ids, fb.period_id(2025, 3)), seq_mean, seq_std)
    test_num, test_cat = transform_static(data.static_frame, test_ids, num_mean, num_std, encoders, num_cols, cat_cols)
    preds = []
    for seed in seeds:
        model, _ = train_model(
            (seq, num, cat, y),
            None,
            [len(encoders[col]) for col in cat_cols],
            epochs=final_epochs,
            seed=seed,
        )
        preds.append(predict_model(model, test_seq, test_num, test_cat))
    out = data.test_rows[["new_id", "new_base", "anomaly_bin", "anom_trigger"]].copy()
    out["pred_log_gru"] = np.mean(np.stack(preds, axis=0), axis=0)
    return out


def write_candidates(test_pred: pd.DataFrame, summary: pd.DataFrame, candidate_start: int, file_tag: str) -> pd.DataFrame:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    best_aggressive = summary[summary["variant"].eq("feb_anomaly_gru")].head(8).copy()
    best_aggressive["candidate_group"] = "aggressive"
    best_safe = (
        summary[(summary["variant"].eq("feb_anomaly_gru")) & (summary["max_abs_shift_pct"] <= 5.0)]
        .head(8)
        .copy()
    )
    best_safe["candidate_group"] = "safe"
    best = pd.concat([best_aggressive, best_safe], ignore_index=True)
    test_base = test_pred["new_base"].to_numpy(float)
    test_logs = {
        "none": test_pred["pred_log_gru"].to_numpy(float),
        "global": (test_pred["pred_log_gru"] - float(test_pred["pred_log_gru"].median())).to_numpy(float),
        "anomaly_bin": center_by_group(test_pred["pred_log_gru"], test_pred["anomaly_bin"]).to_numpy(float),
    }
    rows = []
    for idx, (_, row) in enumerate(best.iterrows()):
        candidate_id = candidate_start + idx if row["candidate_group"] == "aggressive" else candidate_start + 8 + idx - len(best_aggressive)
        delta = float(row["alpha"]) * test_logs[str(row["center"])]
        clip = row["clip"]
        if clip != "":
            delta = np.clip(delta, -float(clip), float(clip))
        pred = test_base * np.exp(delta)
        out = test_pred[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 1.0), 2)
        if row["candidate_group"] == "safe":
            clip_label = str(row["clip"]).replace(".", "p")
            file_name = (
                f"stage2_candidate_{candidate_id}_febanom_gru_safe_"
                f"{file_tag}_"
                f"a{str(row['alpha']).replace('.', 'p')}_"
                f"c{clip_label}_{row['center']}.csv"
            )
        else:
            file_name = (
                f"stage2_candidate_{candidate_id}_febanom_gru_"
                f"{file_tag}_"
                f"a{str(row['alpha']).replace('.', 'p')}_"
                f"c_{row['center']}.csv"
            )
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        ratio = pred / test_base - 1.0
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                **row.to_dict(),
                "test_mean_abs_shift_pct_vs_febanom_base": float(np.mean(np.abs(ratio)) * 100),
                "test_max_abs_shift_pct_vs_febanom_base": float(np.max(np.abs(ratio)) * 100),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--final-epochs", type=int, default=DEFAULT_FINAL_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED, SEED + 17])
    parser.add_argument("--normal-method", type=str, default="trend_dec_jan")
    parser.add_argument("--trigger-mode", type=str, default="low_after_stable")
    parser.add_argument("--threshold", type=float, default=0.13)
    parser.add_argument("--correction-weight", type=float, default=1.0)
    parser.add_argument("--factor-lambda", type=float, default=0.60)
    parser.add_argument("--max-level-ratio", type=float, default=1.75)
    parser.add_argument("--candidate-start", type=int, default=390)
    parser.add_argument("--output-name", type=str, default="new_strategy_feb_anomaly_gru")
    parser.add_argument("--file-tag", type=str, default="cap1p75")
    args = parser.parse_args()

    OUT_DIR = ROOT / "results" / args.output_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    spec = base_spec(
        normal_method=args.normal_method,
        trigger_mode=args.trigger_mode,
        threshold=args.threshold,
        correction_weight=args.correction_weight,
        factor_lambda=args.factor_lambda,
        max_level_ratio=args.max_level_ratio,
    )
    print(f"Base spec: {spec.name}")
    data = prepare_data(spec)
    oof = run_oof(data, args.epochs)
    oof.to_csv(OUT_DIR / "oof.csv", index=False)
    summary = evaluate_oof(oof)
    summary.to_csv(OUT_DIR / "oof_summary.csv", index=False)
    test_pred = train_final_predict(data, args.final_epochs, args.seeds)
    test_pred.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    candidates = write_candidates(test_pred, summary, args.candidate_start, args.file_tag)
    candidates.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    metadata = {
        "base": spec.__dict__,
        "seq_len": SEQ_LEN,
        "epochs": args.epochs,
        "final_epochs": args.final_epochs,
        "seeds": args.seeds,
        "device": str(DEVICE),
        "target": "log(actual / capped conditional February anomaly baseline)",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OOF summary:")
    print(summary.head(25).to_string(index=False))
    print("\nSubmission candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
