from __future__ import annotations

import numpy as np
import pandas as pd

from build_stacking_v11_candidates import add_stack_features, build_prediction_tables
from heuristic_baseline_direction_lab import RESULTS_DIR, SUBMISSIONS_DIR, TARGET, load_data, mape, safe_div


OUT_DIR = RESULTS_DIR / "v13_group_total_reconciliation"


def label(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def load_sub(candidate_id: int, col: str) -> pd.DataFrame:
    path = sorted(SUBMISSIONS_DIR.glob(f"stage2_candidate_{candidate_id:03d}_*.csv"))[0]
    return pd.read_csv(path).rename(columns={"rto": col})[["new_id", col]]


def group_key(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        return frame[cols[0]].astype(str).fillna("__nan__")
    return frame[cols].astype(str).fillna("__nan__").agg("|".join, axis=1)


def group_ratio_map(rto: pd.DataFrame, static: pd.DataFrame, year: int, month: int, cols: list[str]) -> dict[str, float]:
    keys = group_key(static.reset_index(), cols)
    ids = static.reset_index()["new_id"]
    key_by_id = pd.Series(keys.to_numpy(), index=ids.to_numpy())
    source_years = [sy for sy in sorted({int(c[:4]) for c in rto.columns if c.endswith(f"-{month:02d}")}) if sy < year and label(sy, month) in rto.columns and label(sy, month - 1) in rto.columns and label(sy, month - 2) in rto.columns]
    rows = []
    for sy in source_years:
        tmp = pd.DataFrame(
            {
                "key": key_by_id,
                "target": rto[label(sy, month)],
                "prevsum": rto[label(sy, month - 1)] + rto[label(sy, month - 2)],
            }
        ).dropna()
        grp = tmp.groupby("key", observed=True)[["target", "prevsum"]].sum()
        ratio = safe_div(grp["target"], grp["prevsum"]).replace([np.inf, -np.inf], np.nan)
        rows.append(ratio.rename(sy))
    if not rows:
        return {}
    ratios = pd.concat(rows, axis=1)
    return ratios.median(axis=1).dropna().to_dict()


def current_group_anchor(frame: pd.DataFrame, rto: pd.DataFrame, static: pd.DataFrame, year: int, month: int, cols: list[str]) -> pd.Series:
    ratios = group_ratio_map(rto, static, year, month, cols)
    keys = group_key(frame, cols)
    prevsum = frame["lag1"].to_numpy(float) + frame["lag2"].to_numpy(float)
    global_ratio = np.nanmedian(list(ratios.values())) if ratios else 0.5
    mapped = keys.map(ratios).fillna(global_ratio).to_numpy(float)
    return pd.Series(prevsum * mapped, index=frame.index)


def correction_by_group(frame: pd.DataFrame, pred_col: str, anchor_col: str, cols: list[str], gamma: float, clip: float, shrink_k: float) -> np.ndarray:
    keys = group_key(frame, cols)
    tmp = pd.DataFrame({"key": keys, "pred": frame[pred_col].to_numpy(float), "anchor": frame[anchor_col].to_numpy(float)})
    grp = tmp.groupby("key", observed=True).agg(pred=("pred", "sum"), anchor=("anchor", "sum"), n=("pred", "size"))
    raw = np.log(np.maximum(grp["anchor"], 1.0) / np.maximum(grp["pred"], 1.0))
    shrink = grp["n"] / (grp["n"] + shrink_k)
    corr_map = (gamma * shrink * raw.clip(-clip, clip)).to_dict()
    return keys.map(corr_map).fillna(0.0).to_numpy(float)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _raw, rto, static = load_data()
    train, test = build_prediction_tables()
    train = add_stack_features(train)
    test = add_stack_features(test)
    test = test.merge(load_sub(265, "pred265"), on="new_id", how="inner")
    train["pred265_proxy"] = train["pred220"]  # use confirmed OOF-220 anchor for validation; 265 OOF is only a small meta-correction.
    specs = []
    group_sets = [
        ["Регион"],
        ["Торговая площадь, категориальный"],
        ["Дата открытия, категориальный"],
        ["Регион", "Торговая площадь, категориальный"],
        ["Регион", "Дата открытия, категориальный"],
    ]
    for cols in group_sets:
        key = "+".join(c[:6] for c in cols).replace(" ", "")
        train[f"anchor_{key}"] = current_group_anchor(train, rto, static, 2024, 3, cols)
        test[f"anchor_{key}"] = current_group_anchor(test, rto, static, 2025, 3, cols)
        for gamma in [0.25, 0.50, 0.75, 1.00]:
            for clip in [0.005, 0.010, 0.020, 0.030]:
                for shrink_k in [50.0, 200.0, 800.0]:
                    specs.append((key, cols, gamma, clip, shrink_k, f"anchor_{key}"))
    rows = []
    test_preds = {}
    y = train[TARGET].to_numpy(float)
    base_train = train["pred265_proxy"].to_numpy(float)
    base_test = test["pred265"].to_numpy(float)
    for key, cols, gamma, clip, shrink_k, anchor_col in specs:
        corr_train = correction_by_group(train, "pred265_proxy", anchor_col, cols, gamma, clip, shrink_k)
        corr_test = correction_by_group(test, "pred265", anchor_col, cols, gamma, clip, shrink_k)
        pred_train = base_train * np.exp(corr_train)
        pred_test = base_test * np.exp(corr_test)
        ratio = pred_test / base_test - 1.0
        rows.append(
            {
                "name": f"{key}_g{gamma:.2f}_c{clip:.3f}_k{int(shrink_k)}",
                "cols": "+".join(cols),
                "gamma": gamma,
                "clip": clip,
                "shrink_k": shrink_k,
                "oof_mape": mape(y, pred_train),
                "delta_vs_proxy": mape(y, pred_train) - mape(y, base_train),
                "mean_abs_vs_265_pct": float(np.mean(np.abs(ratio)) * 100),
                "p95_abs_vs_265_pct": float(np.quantile(np.abs(ratio), 0.95) * 100),
                "max_abs_vs_265_pct": float(np.max(np.abs(ratio)) * 100),
                "mean_delta_vs_265_pct": float(np.mean(ratio) * 100),
            }
        )
        test_preds[rows[-1]["name"]] = pred_test
    summary = pd.DataFrame(rows).sort_values(["delta_vs_proxy", "mean_abs_vs_265_pct"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "v13_search.csv", index=False)
    chosen = summary[(summary["delta_vs_proxy"] < -0.001) & (summary["mean_abs_vs_265_pct"].between(0.05, 0.80)) & (summary["p95_abs_vs_265_pct"] <= 2.0)].head(6)
    if chosen.empty:
        chosen = summary.head(6)
    out_rows = []
    for candidate_id, (_, row) in enumerate(chosen.iterrows(), start=272):
        pred = test_preds[row["name"]]
        out = test[["new_id"]].copy()
        out["rto"] = np.round(np.maximum(pred, 0), 2)
        safe = row["name"].replace(".", "p")
        file_name = f"stage2_candidate_{candidate_id:03d}_full_v13_group_total_{safe}.csv"
        out.to_csv(SUBMISSIONS_DIR / file_name, index=False)
        out_rows.append({"candidate_id": candidate_id, "file": file_name, **row.to_dict()})
    candidates = pd.DataFrame(out_rows)
    candidates.to_csv(OUT_DIR / "submission_candidates.csv", index=False)
    print("Top v13 search:")
    print(summary.head(30).to_string(index=False))
    print("\nCandidates:")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
