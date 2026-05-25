from __future__ import annotations

import numpy as np
import pandas as pd

from heuristic_baseline_direction_lab import (
    RESULTS_DIR,
    TARGET,
    HeuristicSpec,
    build_target_frame,
    factors_for_spec,
    load_data,
    mape,
    predict_from_spec,
    safe_div,
)


BASE_SPEC = HeuristicSpec(70, "cal050_median_mom000", 0.50, momentum_exp=0.0)


def adjusted_store_ratio(frame: pd.DataFrame, target_non_leap: bool, calendar_lambda: float) -> pd.Series:
    ratio = safe_div(frame["lag12"], frame["lag13"])
    if target_non_leap:
        return ratio * ((1.0 - calendar_lambda) + calendar_lambda * (29.0 / 28.0))
    return ratio * ((1.0 - calendar_lambda) + calendar_lambda * (28.0 / 29.0))


def reliability(frame: pd.DataFrame, mode: str, volume_k: float) -> np.ndarray:
    if mode == "constant":
        return np.ones(len(frame), dtype=float)
    if mode == "volume":
        return (frame["lag13"].to_numpy(float) / (frame["lag13"].to_numpy(float) + volume_k)).clip(0.0, 1.0)
    if mode == "stable_volume":
        volume = (frame["lag13"].to_numpy(float) / (frame["lag13"].to_numpy(float) + volume_k)).clip(0.0, 1.0)
        stable = (1.0 - frame["volatility_rel3"].fillna(0.0).to_numpy(float) / 0.12).clip(0.0, 1.0)
        return volume * stable
    raise ValueError(mode)


def store_seasonal_pred(
    frame: pd.DataFrame,
    base_pred: np.ndarray,
    global_factor: float,
    target_non_leap: bool,
    calendar_lambda: float,
    weight: float,
    log_clip: float,
    rel_mode: str,
    volume_k: float,
    extra_gate: np.ndarray | None = None,
) -> np.ndarray:
    ratio = adjusted_store_ratio(frame, target_non_leap, calendar_lambda).fillna(global_factor).clip(0.85, 1.45)
    log_delta = np.log(ratio.to_numpy(float) / global_factor)
    log_delta = np.clip(log_delta, -log_clip, log_clip)
    rel = reliability(frame, rel_mode, volume_k)
    if extra_gate is not None:
        rel = rel * extra_gate.astype(float)
    return base_pred * np.exp(weight * rel * log_delta)


def store_gate(frame: pd.DataFrame, momentum_min: float | None, vol_max: float | None, avoid_structural: bool) -> np.ndarray:
    gate = np.ones(len(frame), dtype=bool)
    if momentum_min is not None:
        gate &= frame["momentum_raw"].fillna(1.0).to_numpy(float) >= momentum_min
    if vol_max is not None:
        gate &= frame["volatility_rel3"].fillna(0.0).to_numpy(float) <= vol_max
    if avoid_structural:
        gate &= frame["structural_decline_score"].to_numpy(float) <= 0
    return gate.astype(float)


def yoy_modulation_pred(
    frame: pd.DataFrame,
    base_pred: np.ndarray,
    target_non_leap: bool,
    beta: float,
    log_clip: float,
) -> np.ndarray:
    if target_non_leap:
        growth = safe_div(frame["lag1"], frame["lag13"]) * (29.0 / 28.0)
    else:
        growth = safe_div(frame["lag1"], frame["lag13"]) * (28.0 / 29.0)
    growth = growth.replace([np.inf, -np.inf], np.nan).fillna(growth.median()).clip(0.75, 1.60)
    centered = np.log(growth / growth.median())
    return base_pred * np.exp(beta * np.clip(centered.to_numpy(float), -log_clip, log_clip))


def state_adjustment_pred(base_pred: np.ndarray, frame: pd.DataFrame, dip_lift: float, structural_cut: float, vol_cut: float) -> np.ndarray:
    pred = base_pred.copy()
    pred *= np.exp(dip_lift * frame["temporary_dip_score"].to_numpy(float))
    pred *= np.exp(-structural_cut * frame["structural_decline_score"].to_numpy(float))
    if vol_cut:
        vol = frame["volatility_rel3"].fillna(0.0).to_numpy(float)
        pred *= np.exp(-vol_cut * np.clip(vol - 0.07, 0.0, 0.20))
    return pred


def add_row(rows: list[dict], name: str, family: str, val_pred: np.ndarray, test_pred: np.ndarray, val_frame: pd.DataFrame, base_val: np.ndarray, base_test: np.ndarray, params: dict) -> None:
    rows.append(
        {
            "name": name,
            "family": family,
            "proxy_mape_march2024": mape(val_frame[TARGET], val_pred),
            "proxy_delta_vs_base": mape(val_frame[TARGET], val_pred) - mape(val_frame[TARGET], base_val),
            "val_mean_abs_change_vs_base_pct": float(np.mean(np.abs(val_pred / base_val - 1.0)) * 100),
            "val_p95_abs_change_vs_base_pct": float(np.quantile(np.abs(val_pred / base_val - 1.0), 0.95) * 100),
            "test_mean_abs_change_vs_base_pct": float(np.mean(np.abs(test_pred / base_test - 1.0)) * 100),
            "test_p95_abs_change_vs_base_pct": float(np.quantile(np.abs(test_pred / base_test - 1.0), 0.95) * 100),
            "test_mean_rto": float(np.mean(test_pred)),
            **params,
        }
    )


def segment_diagnostics(summary: pd.DataFrame, val_frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    segments = {
        "all": np.ones(len(val_frame), dtype=bool),
        "weak_momentum_q20": val_frame["momentum_raw"] <= val_frame["momentum_raw"].quantile(0.20),
        "strong_momentum_q80": val_frame["momentum_raw"] >= val_frame["momentum_raw"].quantile(0.80),
        "high_vol_q80": val_frame["volatility_rel3"] >= val_frame["volatility_rel3"].quantile(0.80),
        "low_vol_q20": val_frame["volatility_rel3"] <= val_frame["volatility_rel3"].quantile(0.20),
        "temp_dip": val_frame["temporary_dip_score"] > 0,
        "struct_decline": val_frame["structural_decline_score"] > 0,
    }
    rows = []
    for name, pred in predictions.items():
        for seg_name, mask in segments.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() < 50:
                continue
            rows.append(
                {
                    "name": name,
                    "segment": seg_name,
                    "n": int(mask.sum()),
                    "proxy_mape": mape(val_frame.loc[mask, TARGET], pred[mask]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    _raw, rto, static = load_data()
    val_frame = build_target_frame(rto, static, 2024, 3, include_target=True)
    test_frame = build_target_frame(rto, static, 2025, 3, include_target=False)

    val_factor, test_factor, val_global, final_global = factors_for_spec(BASE_SPEC, val_frame, test_frame, rto, static)
    base_val = predict_from_spec(val_frame, val_factor, BASE_SPEC)
    base_test = predict_from_spec(test_frame, test_factor, BASE_SPEC)

    rows: list[dict] = []
    keep_predictions = {"base_cal050_mom000": base_val}

    # Store-level March/February seasonality memory, shrunk toward the global factor.
    for calendar_lambda in [0.50, 0.60]:
        val_factor_l, test_factor_l, val_global_l, final_global_l = factors_for_spec(
            HeuristicSpec(0, "base", calendar_lambda, momentum_exp=0.0), val_frame, test_frame, rto, static
        )
        base_val_l = predict_from_spec(val_frame, val_factor_l, HeuristicSpec(0, "base", calendar_lambda, momentum_exp=0.0))
        base_test_l = predict_from_spec(test_frame, test_factor_l, HeuristicSpec(0, "base", calendar_lambda, momentum_exp=0.0))
        keep_predictions[f"base_cal{calendar_lambda:.2f}_mom000"] = base_val_l
        for weight in [0.10, 0.20, 0.30, 0.40, 0.50]:
            for log_clip in [0.03, 0.05, 0.08, 0.12]:
                for rel_mode, volume_k in [("constant", 1.0), ("volume", 50_000_000.0), ("volume", 150_000_000.0), ("stable_volume", 100_000_000.0)]:
                    val_pred = store_seasonal_pred(val_frame, base_val_l, val_global_l, False, calendar_lambda, weight, log_clip, rel_mode, volume_k)
                    test_pred = store_seasonal_pred(test_frame, base_test_l, final_global_l, True, calendar_lambda, weight, log_clip, rel_mode, volume_k)
                    name = f"cal{calendar_lambda:.2f}_storeseason_w{weight:.2f}_clip{log_clip:.2f}_{rel_mode}_k{int(volume_k/1e6)}m"
                    add_row(
                        rows,
                        name,
                        "store_seasonality_shrink",
                        val_pred,
                        test_pred,
                        val_frame,
                        base_val,
                        base_test,
                        {
                            "calendar_lambda": calendar_lambda,
                            "weight": weight,
                            "log_clip": log_clip,
                            "rel_mode": rel_mode,
                            "volume_k": volume_k,
                        },
                    )
                    keep_predictions[name] = val_pred

        for weight in [0.30, 0.50, 0.70]:
            for log_clip in [0.03, 0.05, 0.08]:
                for momentum_min in [0.96, 0.98, 1.00, 1.02, None]:
                    for vol_max in [0.08, 0.10, 0.12, None]:
                        if momentum_min is None and vol_max is None:
                            continue
                        val_gate = store_gate(val_frame, momentum_min, vol_max, avoid_structural=True)
                        test_gate = store_gate(test_frame, momentum_min, vol_max, avoid_structural=True)
                        val_pred = store_seasonal_pred(
                            val_frame,
                            base_val_l,
                            val_global_l,
                            False,
                            calendar_lambda,
                            weight,
                            log_clip,
                            "stable_volume",
                            100_000_000.0,
                            extra_gate=val_gate,
                        )
                        test_pred = store_seasonal_pred(
                            test_frame,
                            base_test_l,
                            final_global_l,
                            True,
                            calendar_lambda,
                            weight,
                            log_clip,
                            "stable_volume",
                            100_000_000.0,
                            extra_gate=test_gate,
                        )
                        mom_label = "none" if momentum_min is None else f"{momentum_min:.2f}"
                        vol_label = "none" if vol_max is None else f"{vol_max:.2f}"
                        name = f"cal{calendar_lambda:.2f}_storeseason_gated_w{weight:.2f}_clip{log_clip:.2f}_mom{mom_label}_vol{vol_label}"
                        add_row(
                            rows,
                            name,
                            "store_seasonality_gated",
                            val_pred,
                            test_pred,
                            val_frame,
                            base_val,
                            base_test,
                            {
                                "calendar_lambda": calendar_lambda,
                                "weight": weight,
                                "log_clip": log_clip,
                                "momentum_min": momentum_min if momentum_min is not None else np.nan,
                                "vol_max": vol_max if vol_max is not None else np.nan,
                                "avoid_structural": True,
                                "val_gate_share": float(val_gate.mean()),
                                "test_gate_share": float(test_gate.mean()),
                            },
                        )
                        keep_predictions[name] = val_pred

    # Small modulation by February YoY growth. This tests whether growth stores need a different Mar/Feb factor.
    for beta in [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]:
        for log_clip in [0.05, 0.10, 0.20]:
            val_pred = yoy_modulation_pred(val_frame, base_val, False, beta, log_clip)
            test_pred = yoy_modulation_pred(test_frame, base_test, True, beta, log_clip)
            name = f"cal050_yoymod_beta{beta:+.2f}_clip{log_clip:.2f}"
            add_row(
                rows,
                name,
                "feb_yoy_modulation",
                val_pred,
                test_pred,
                val_frame,
                base_val,
                base_test,
                {"beta": beta, "log_clip": log_clip},
            )
            keep_predictions[name] = val_pred

    # State-based adjustments around the public-safe no-momentum baseline.
    for dip_lift in [0.00, 0.01, 0.02, 0.03, 0.05]:
        for structural_cut in [0.00, 0.01, 0.02, 0.03]:
            for vol_cut in [0.00, 0.05, 0.10]:
                if dip_lift == 0 and structural_cut == 0 and vol_cut == 0:
                    continue
                val_pred = state_adjustment_pred(base_val, val_frame, dip_lift, structural_cut, vol_cut)
                test_pred = state_adjustment_pred(base_test, test_frame, dip_lift, structural_cut, vol_cut)
                name = f"cal050_state_dip{dip_lift:.2f}_struct{structural_cut:.2f}_vol{vol_cut:.2f}"
                add_row(
                    rows,
                    name,
                    "state_adjustment",
                    val_pred,
                    test_pred,
                    val_frame,
                    base_val,
                    base_test,
                    {"dip_lift": dip_lift, "structural_cut": structural_cut, "vol_cut": vol_cut},
                )
                keep_predictions[name] = val_pred

    summary = pd.DataFrame(rows).sort_values("proxy_mape_march2024").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "baseline_heuristics_v3_search.csv", index=False)
    summary.head(50).to_csv(RESULTS_DIR / "baseline_heuristics_v3_top50.csv", index=False)

    best_names = ["base_cal050_mom000"] + summary.head(12)["name"].tolist()
    for row in summary.head(12).itertuples(index=False):
        if row.name not in keep_predictions:
            # Recompute only if the top prediction was not retained during grid generation.
            pass
    seg = segment_diagnostics(summary, val_frame, {k: v for k, v in keep_predictions.items() if k in best_names or k == "base_cal050_mom000"})
    seg.to_csv(RESULTS_DIR / "baseline_heuristics_v3_segment_diagnostics.csv", index=False)

    print("Top v3 heuristic baselines:")
    print(
        summary[
            [
                "name",
                "family",
                "proxy_mape_march2024",
                "proxy_delta_vs_base",
                "val_mean_abs_change_vs_base_pct",
                "test_mean_abs_change_vs_base_pct",
                "test_mean_rto",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )
    print("\nFamily best:")
    print(
        summary.groupby("family", as_index=False)
        .first()[["family", "name", "proxy_mape_march2024", "proxy_delta_vs_base", "test_mean_abs_change_vs_base_pct"]]
        .sort_values("proxy_mape_march2024")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
