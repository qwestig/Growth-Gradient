from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from heuristic_baseline_direction_lab import (
    RESULTS_DIR,
    SUBMISSIONS_DIR,
    TARGET,
    HeuristicSpec,
    build_target_frame,
    factors_for_spec,
    load_data,
    mape,
)
from search_structural_baselines_v6 import (
    CURRENT_LAMBDA,
    blend_anchor,
    current_198_march,
    two_month_temporal_anchor,
)
from train_new_baseline_gru_residual import (
    BATCH_SIZE,
    BEHAVIOR_COLS,
    DEVICE,
    EPOCHS,
    FINAL_EPOCHS,
    GRUResidual,
    MOMENTUM_TRAIN_BINS,
    PATIENCE,
    SEED,
    SEQ_LEN,
    STATIC_CAT_COLS,
    STATIC_NUM_COLS,
    build_sequence,
    build_static,
    center_by_group,
    fit_sequence_scaler,
    fit_static_encoders,
    momentum_bin,
    period_id,
    set_seed,
    transform_static,
)


OUT_DIR = RESULTS_DIR / "temporal204_gru_residual"
BASE_SPEC = HeuristicSpec(0, "temporal204", CURRENT_LAMBDA, momentum_exp=0.0)


class PreparedData:
    def __init__(self, raw, rto_wide, behavior_wide, static_frame, train_rows, test_rows):
        self.raw = raw
        self.rto_wide = rto_wide
        self.behavior_wide = behavior_wide
        self.static_frame = static_frame
        self.train_rows = train_rows
        self.test_rows = test_rows


def temporal204_base(frame: pd.DataFrame, rto_label_wide: pd.DataFrame, factor: float, year: int, month: int, target_non_leap: bool) -> np.ndarray:
    base_198 = current_198_march(frame, factor, target_non_leap=target_non_leap)
    anchor = two_month_temporal_anchor(
        rto_label_wide,
        frame,
        year,
        month,
        daily_adjust=False,
        clip_lo=0.85,
        clip_hi=1.35,
    )
    return blend_anchor(base_198, anchor, alpha=0.12, log_clip=0.06)


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
    _train_factor, _test_factor, val_global, final_global = factors_for_spec(BASE_SPEC, train_features, test_features, rto_label_wide, static_for_base)
    train_features["new_base"] = temporal204_base(train_features, rto_label_wide, val_global, 2024, 3, target_non_leap=False)
    test_features["new_base"] = temporal204_base(test_features, rto_label_wide, final_global, 2025, 3, target_non_leap=True)
    train_features = train_features[(train_features[TARGET] > 0) & (train_features["new_base"] > 0)].copy()
    train_features["target_log_ratio"] = np.log(train_features[TARGET] / train_features["new_base"])
    train_features["momentum_train_bin"] = momentum_bin(train_features)
    test_features["momentum_train_bin"] = momentum_bin(test_features)

    # Sanity: test baseline should reproduce candidate 204 to rounding-level precision.
    cand204 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_204_baseline_v6_temporal2_raw_clip085_135_a012_lc006.csv").rename(columns={"rto": "rto_204"})
    check = test_features[["new_id", "new_base"]].merge(cand204, on="new_id")
    max_abs_diff = float(np.max(np.abs(np.round(check["new_base"].to_numpy(float), 2) - check["rto_204"].to_numpy(float))))
    if max_abs_diff > 0.02:
        raise RuntimeError(f"Temporal204 base mismatch vs candidate 204: max_abs_diff={max_abs_diff}")

    return PreparedData(
        raw=raw,
        rto_wide=rto_wide,
        behavior_wide=behavior_wide,
        static_frame=static_frame,
        train_rows=train_features.sort_values("new_id").reset_index(drop=True),
        test_rows=test_features.sort_values("new_id").reset_index(drop=True),
    )


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
        train_seq = fit_apply_sequence(full_seq[~valid_mask], seq_mean, seq_std)
        valid_seq = fit_apply_sequence(full_seq[valid_mask], seq_mean, seq_std)
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


def fit_apply_sequence(seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (seq - mean[None, None, :]) / std[None, None, :]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def evaluate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = [{"variant": "temporal204_base", "alpha": 0.0, "center": "none", "mape": mape(oof[TARGET], oof["new_base"])}]
    for center in ["none", "global", "momentum_bin"]:
        if center == "none":
            pred_log = oof["pred_log_gru"]
        elif center == "global":
            pred_log = oof["pred_log_gru"] - float(oof["pred_log_gru"].median())
        else:
            pred_log = center_by_group(oof["pred_log_gru"], oof["momentum_train_bin"])
        for alpha in [0.3, 0.5, 0.7, 0.85, 1.0, 1.05, 1.15, 1.3]:
            pred = oof["new_base"].to_numpy(dtype=float) * np.exp(alpha * pred_log.to_numpy(dtype=float))
            rows.append({"variant": f"temporal204_gru_{center}", "alpha": alpha, "center": center, "mape": mape(oof[TARGET], pred)})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def train_final_and_predict(data: PreparedData) -> np.ndarray:
    set_seed(SEED)
    train = data.train_rows.copy()
    store_ids = train["new_id"].to_numpy()
    y = train["target_log_ratio"].to_numpy(dtype=np.float32)
    seq_raw = build_sequence(data, store_ids, period_id(2024, 3))
    seq_mean, seq_std = fit_sequence_scaler(seq_raw)
    seq = fit_apply_sequence(seq_raw, seq_mean, seq_std)
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
    test_seq = fit_apply_sequence(build_sequence(data, test_ids, period_id(2025, 3)), seq_mean, seq_std)
    test_num, test_cat = transform_static(data.static_frame, test_ids, num_mean, num_std, encoders, num_cols, cat_cols)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(test_seq).to(DEVICE), torch.from_numpy(test_num).to(DEVICE), torch.from_numpy(test_cat).to(DEVICE)).detach().cpu().numpy()


def write_candidates(data: PreparedData, pred_log: np.ndarray, oof: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = data.test_rows.copy()
    test["pred_log_gru"] = pred_log
    test["pred_log_global"] = test["pred_log_gru"] - float(test["pred_log_gru"].median())
    test["pred_log_mombin"] = center_by_group(test["pred_log_gru"], test["momentum_train_bin"])
    base = test["new_base"].to_numpy(dtype=float)
    preds: dict[int, tuple[str, np.ndarray, str, float | None]] = {}
    for candidate_id, name, col, alpha in [
        (206, "temporal204_newgru_raw_a070", "pred_log_gru", 0.70),
        (207, "temporal204_newgru_raw_a100", "pred_log_gru", 1.00),
        (208, "temporal204_newgru_global_a070", "pred_log_global", 0.70),
        (209, "temporal204_newgru_global_a085", "pred_log_global", 0.85),
        (210, "temporal204_newgru_global_a100", "pred_log_global", 1.00),
        (211, "temporal204_newgru_global_a105", "pred_log_global", 1.05),
        (212, "temporal204_newgru_mombin_a100", "pred_log_mombin", 1.00),
        (213, "temporal204_newgru_mombin_a115", "pred_log_mombin", 1.15),
    ]:
        preds[candidate_id] = (name, base * np.exp(alpha * test[col].to_numpy(dtype=float)), col, alpha)

    # Include the previously useful independent old-transfer signal as a small blend candidate.
    transfer = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_118_baseline_cal050_mombin_g100.csv").rename(columns={"rto": "transfer118"})
    best159 = pd.read_csv(SUBMISSIONS_DIR / "stage2_candidate_159_baseline_faithful_global_blend_75_25_transfer.csv").rename(columns={"rto": "best159"})
    frame_blend = test[["new_id"]].merge(transfer, on="new_id").merge(best159, on="new_id")
    g100 = preds[210][1]
    preds[214] = ("temporal204_newgru_global_a100_blend85_transfer118", 0.85 * g100 + 0.15 * frame_blend["transfer118"].to_numpy(float), "blend_transfer118", None)
    preds[215] = ("temporal204_newgru_global_a100_blend75_transfer118", 0.75 * g100 + 0.25 * frame_blend["transfer118"].to_numpy(float), "blend_transfer118", None)
    preds[216] = ("temporal204_newgru_global_a100_blend85_best159", 0.85 * g100 + 0.15 * frame_blend["best159"].to_numpy(float), "blend_best159", None)

    for candidate_id, (name, pred, center, alpha) in preds.items():
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        file_name = f"stage2_candidate_{candidate_id:03d}_baseline_{name}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        rows.append(
            {
                "candidate_id": candidate_id,
                "file": file_name,
                "name": name,
                "alpha": alpha if alpha is not None else np.nan,
                "center": center,
                "test_mean_rto": float(np.mean(pred)),
                "test_total_rto": float(np.sum(pred)),
                "mean_abs_change_vs_base204_pct": float(np.mean(np.abs(pred / base - 1.0)) * 100),
            }
        )
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
            new_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "file": row["file"],
                    "family": "temporal204_gru_residual",
                    "hypothesis": "Fresh faithful GRU residual trained directly on candidate 204 temporal/Q1 pure baseline; includes best centering and transfer-blend variants.",
                    "public_score": "",
                    "notes": "fill public_score after submission",
                    "public_mape": "",
                    "delta_mape_vs_current_best": "",
                    "delta_score_vs_current_best": "",
                }
            )
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
    metadata = {
        "seq_len": SEQ_LEN,
        "epochs": EPOCHS,
        "final_epochs": FINAL_EPOCHS,
        "base": "candidate_204 temporal2_raw_clip085_135_a012_lc006",
        "base_spec": BASE_SPEC.__dict__,
        "device": str(DEVICE),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OOF summary:")
    print(summary.head(24).to_string(index=False))
    print("\nSubmission candidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
