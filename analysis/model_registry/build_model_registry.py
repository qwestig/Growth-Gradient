from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "model_registry"

ANCHOR_LOCAL_OOF_MAPE = 3.314574426593587
ANCHOR_PUBLIC_SCORE = 90.78
ANCHOR_PUBLIC_MAPE = 4.7214609684


PUBLIC_SCORE_LOG = {
    204: {
        "public_score": 89.65,
        "public_mape": np.nan,
        "note": "Reported best pre-90 heuristic/stack attempt.",
    },
    235: {
        "public_score": 90.73,
        "public_mape": np.nan,
        "note": "Reported regression from later 90.77/90.78 line.",
    },
    265: {
        "public_score": ANCHOR_PUBLIC_SCORE,
        "public_mape": ANCHOR_PUBLIC_MAPE,
        "note": "Current public anchor: temporal204 v11 / safer GRU stack.",
    },
    317: {
        "public_score": 90.78,
        "public_mape": np.nan,
        "note": "Reported same visible score as current best.",
    },
    319: {
        "public_score": 90.78,
        "public_mape": np.nan,
        "note": "Reported same visible score as current best.",
    },
    390: {
        "public_score": 90.59,
        "public_mape": np.nan,
        "note": "Aggressive feb-anomaly GRU residual; public worse than anchor.",
    },
    398: {
        "public_score": 90.45,
        "public_mape": np.nan,
        "note": "Safe feb-anomaly GRU residual; public worse than anchor.",
    },
    430: {
        "public_score": 90.58,
        "public_mape": np.nan,
        "note": "Uncapped feb-anomaly GRU residual; public worse than anchor.",
    },
    467: {
        "public_score": 90.78,
        "public_mape": np.nan,
        "note": "Micro overlay on 265; same visible public score.",
    },
    472: {
        "public_score": 90.78,
        "public_mape": np.nan,
        "note": "Stronger micro overlay on 265; same visible public score.",
    },
}


FAMILIES = [
    {
        "family_id": "anchor_temporal204_v11_gru_stack",
        "stage": "current_anchor",
        "approach": "Temporal204 feature stack + GRU residual + conservative postprocessing",
        "training": "GRU residual / stacked heuristic",
        "primary_script": "remote:analysis/new_baseline/temporal204/*",
        "result_dir": "remote:results/temporal204_meta_inputs",
        "summary_files": "",
        "public_candidate_ids": "265",
        "verdict": "current_best_public_anchor",
        "notes": "Main anchor for all later comparisons. Public score 90.78, public MAPE 4.7214609684, local pred265 OOF MAPE 3.314574.",
    },
    {
        "family_id": "legacy_stage2_stack_201_211",
        "stage": "legacy_heuristic_stack",
        "approach": "Coefficient/stack search before the current temporal204 solution",
        "training": "Mostly deterministic stack / calibration",
        "primary_script": "analysis/new_baseline/tune_current_best_stack.py",
        "result_dir": "submissions/stage2_candidate_201_*.csv ... stage2_candidate_211_*.csv",
        "summary_files": "results/new_baseline/current_best_stack_tuning_summary.csv;results/new_baseline/current_best_stack_gamma_followup_summary.csv",
        "public_candidate_ids": "204",
        "verdict": "historically_useful_but_obsolete",
        "notes": "Useful as a defense baseline: improved over simple heuristics but later beaten by GRU/temporal204 stack.",
    },
    {
        "family_id": "early_gru_residual",
        "stage": "ml_residual",
        "approach": "First GRU residual over heuristic baseline",
        "training": "GRU",
        "primary_script": "analysis/new_baseline/train_multimonth_gru_residual.py",
        "result_dir": "results/new_baseline_gru_residual",
        "summary_files": "results/new_baseline_gru_residual/oof_summary.csv;results/new_baseline_gru_residual/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "kept_for_history",
        "notes": "Earlier GRU branch. Useful to show why later temporal204/v11 stack became the stronger anchor.",
    },
    {
        "family_id": "catboost_residual",
        "stage": "ml_residual",
        "approach": "CatBoost-style residual modeling over baseline",
        "training": "CatBoost when available",
        "primary_script": "analysis/new_baseline/temporal204/train_temporal204_catboost_first_residual_v12.py",
        "result_dir": "results/new_baseline_catboost_residual",
        "summary_files": "results/new_baseline_catboost_residual/oof_summary.csv;results/new_baseline_catboost_residual/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "not_anchor",
        "notes": "Useful as an ML comparison. Local summaries suggest it did not become the current public anchor.",
    },
    {
        "family_id": "multimonth_gru",
        "stage": "ml_residual",
        "approach": "Multi-month GRU residual, non-seasonal variant",
        "training": "GRU",
        "primary_script": "analysis/new_baseline/train_multimonth_gru_residual.py",
        "result_dir": "results/new_baseline_multimonth_gru",
        "summary_files": "results/new_baseline_multimonth_gru/oof_summary.csv;results/new_baseline_multimonth_gru/gated_candidates_summary.csv;results/new_baseline_multimonth_gru/submission_candidates.csv",
        "public_candidate_ids": "380,381,382,383,384,385,386,387,388,389",
        "verdict": "tested_but_not_final_anchor",
        "notes": "Generated many gated candidates. Kept for local selection and defense discussion.",
    },
    {
        "family_id": "multimonth_gru_seasonal",
        "stage": "ml_residual",
        "approach": "Multi-month GRU residual with seasonal target/normalization",
        "training": "GRU",
        "primary_script": "analysis/new_baseline/train_multimonth_gru_seasonal_residual.py",
        "result_dir": "results/new_baseline_multimonth_gru_seasonal",
        "summary_files": "results/new_baseline_multimonth_gru_seasonal/oof_summary.csv;results/new_baseline_multimonth_gru_seasonal/gated_candidates_summary.csv;results/new_baseline_multimonth_gru_seasonal/ml_gate_search_summary.csv;results/new_baseline_multimonth_gru_seasonal/submission_candidates.csv",
        "public_candidate_ids": "370,371,372,373,374,375,376,377",
        "verdict": "tested_but_not_final_anchor",
        "notes": "Seasonal GRU/gating family. Useful to compare local OOF vs public behavior.",
    },
    {
        "family_id": "hierarchical_cohort_expert",
        "stage": "new_heuristic",
        "approach": "Cohort/hierarchy March-Feb ratio and share expert",
        "training": "Deterministic/shrinkage heuristic",
        "primary_script": "analysis/new_baseline/train_hierarchical_cohort_expert.py",
        "result_dir": "results/new_baseline_hierarchy",
        "summary_files": "results/new_baseline_hierarchy/cohort_expert_summary.csv;results/new_baseline_hierarchy/stress_bucket_summary.csv;results/new_baseline_hierarchy/major_region_summary.csv;results/new_baseline_hierarchy/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "rejected_locally",
        "notes": "Important negative result: hierarchy/cohort idea was theoretically plausible but did not beat the 265 anchor locally.",
    },
    {
        "family_id": "cross_similarity_expert",
        "stage": "new_heuristic",
        "approach": "KNN/cross-similarity forecast from similar historical store profiles",
        "training": "Nearest-neighbor deterministic expert",
        "primary_script": "analysis/new_baseline/train_cross_similarity_expert.py",
        "result_dir": "results/new_baseline_similarity",
        "summary_files": "results/new_baseline_similarity/similarity_expert_summary.csv;results/new_baseline_similarity/stress_bucket_summary.csv;results/new_baseline_similarity/major_region_summary.csv;results/new_baseline_similarity/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "small_local_signal",
        "notes": "Only independent heuristic family with a small local gain vs pred265; later used in candidates 476-478 as overlay.",
    },
    {
        "family_id": "alternative_heuristic_lab",
        "stage": "new_heuristic",
        "approach": "Same-month YoY, quarter/Q1, calendar intensity, top-down share, normal-level alternatives",
        "training": "Deterministic heuristic lab",
        "primary_script": "analysis/new_baseline/alternative_heuristic_baseline_lab.py",
        "result_dir": "results/new_baseline_alt_heuristics",
        "summary_files": "results/new_baseline_alt_heuristics/heuristic_summary.csv;results/new_baseline_alt_heuristics/top20_heuristic_summary.csv;results/new_baseline_alt_heuristics/stress_bucket_summary.csv;results/new_baseline_alt_heuristics/major_region_summary.csv",
        "public_candidate_ids": "",
        "verdict": "rejected_locally",
        "notes": "Good defense material: many theoretically reasonable baselines were worse than current 265 anchor.",
    },
    {
        "family_id": "shrunk_yoy_cohort",
        "stage": "new_heuristic",
        "approach": "Previous March multiplied by store/cohort-shrunk YoY growth",
        "training": "Deterministic/shrinkage heuristic",
        "primary_script": "analysis/new_strategy/train_shrunk_yoy_cohort_baseline.py",
        "result_dir": "results/new_strategy_shrunk_yoy",
        "summary_files": "results/new_strategy_shrunk_yoy/shrunk_yoy_summary.csv;results/new_strategy_shrunk_yoy/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "rejected_locally",
        "notes": "Tested because it changes the anchor from Feb->Mar to YoY; local MAPE was much worse than pred265.",
    },
    {
        "family_id": "shrunk_yoy_ml_residual",
        "stage": "new_heuristic_plus_ml",
        "approach": "ML residual on shrunk-YoY baseline",
        "training": "Ridge/HGB residual",
        "primary_script": "analysis/new_strategy/train_shrunk_yoy_ml_residual.py",
        "result_dir": "results/new_strategy_shrunk_yoy_ml",
        "summary_files": "results/new_strategy_shrunk_yoy_ml/shrunk_yoy_ml_summary.csv;results/new_strategy_shrunk_yoy_ml/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "rejected_locally",
        "notes": "ML improved the weak YoY baseline but still remained far behind pred265.",
    },
    {
        "family_id": "deseasonalized_normal_level",
        "stage": "new_heuristic",
        "approach": "Rolling deseasonalized normal level + March seasonality",
        "training": "Deterministic heuristic",
        "primary_script": "analysis/new_strategy/train_deseasonalized_normal_baseline.py",
        "result_dir": "results/new_strategy_deseasonalized_normal",
        "summary_files": "results/new_strategy_deseasonalized_normal/deseasonalized_normal_summary.csv;results/new_strategy_deseasonalized_normal/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "rejected_locally",
        "notes": "Robust decomposition idea; did not beat the current clean heuristic or pred265 line.",
    },
    {
        "family_id": "march_share_q1",
        "stage": "new_heuristic",
        "approach": "Q1 total / March share of Q1 baseline",
        "training": "Deterministic temporal aggregation heuristic",
        "primary_script": "analysis/new_strategy/train_march_share_q1_baseline.py",
        "result_dir": "results/new_strategy_march_share_q1",
        "summary_files": "results/new_strategy_march_share_q1/march_share_q1_summary.csv;results/new_strategy_march_share_q1/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "rejected_locally",
        "notes": "Orthogonal quarter-level idea; useful negative result for defense.",
    },
    {
        "family_id": "share_of_market",
        "stage": "new_heuristic",
        "approach": "Forecast aggregate/cohort total and distribute by store share",
        "training": "Deterministic share allocation",
        "primary_script": "analysis/new_strategy/train_share_of_market_strategy.py",
        "result_dir": "results/new_strategy_share_of_market",
        "summary_files": "results/new_strategy_share_of_market/share_of_market_summary.csv;results/new_strategy_share_of_market/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "rejected_locally",
        "notes": "The share-allocation idea was too unstable at store level in local OOF.",
    },
    {
        "family_id": "lifecycle_heuristic",
        "stage": "new_heuristic",
        "approach": "Store lifecycle/maturity grouped March-Feb correction",
        "training": "Deterministic grouped heuristic",
        "primary_script": "analysis/new_strategy/train_lifecycle_heuristic.py;analysis/new_strategy/train_lifecycle_monthly_residual.py",
        "result_dir": "results/new_strategy_lifecycle",
        "summary_files": "results/new_strategy_lifecycle/lifecycle_summary.csv;results/new_strategy_lifecycle/monthly_lifecycle_summary.csv;results/new_strategy_lifecycle/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "rejected_locally",
        "notes": "Useful for explanation: maturity effect exists conceptually, but this implementation did not improve pred265.",
    },
    {
        "family_id": "conditional_feb_anomaly",
        "stage": "new_heuristic",
        "approach": "Detect abnormal February and replace/repair local level before March jump",
        "training": "Deterministic heuristic",
        "primary_script": "analysis/new_strategy/train_conditional_feb_anomaly_baseline.py",
        "result_dir": "results/new_strategy_feb_anomaly",
        "summary_files": "results/new_strategy_feb_anomaly/conditional_feb_anomaly_summary.csv;results/new_strategy_feb_anomaly/submission_candidates.csv",
        "public_candidate_ids": "",
        "verdict": "promising_as_bare_heuristic_but_failed_after_gru",
        "notes": "Bare heuristic improved a clean baseline locally, but GRU versions later underperformed badly on public.",
    },
    {
        "family_id": "feb_anomaly_gru",
        "stage": "new_heuristic_plus_ml",
        "approach": "GRU residual trained on conditional February-anomaly baseline",
        "training": "GRU residual",
        "primary_script": "analysis/new_strategy/train_feb_anomaly_gru_residual.py",
        "result_dir": "results/new_strategy_feb_anomaly_gru*",
        "summary_files": "results/new_strategy_feb_anomaly_gru/oof_summary.csv;results/new_strategy_feb_anomaly_gru_cap3p0/oof_summary.csv;results/new_strategy_feb_anomaly_gru_uncapped/oof_summary.csv;results/new_strategy_feb_anomaly_gru_symmetric_pre5/oof_summary.csv",
        "public_candidate_ids": "390,398,430",
        "verdict": "public_failed",
        "notes": "Important lesson: locally plausible new anchor did not transfer to public. Do not use as final unless later private/local protocol says otherwise.",
    },
    {
        "family_id": "micro_overlays_on_265",
        "stage": "posthoc_overlay",
        "approach": "Small v14/feb-anomaly overlays on current pred265 anchor",
        "training": "No new training; deterministic blend/clip",
        "primary_script": "analysis/new_strategy/build_micro_overlays_on_265.py",
        "result_dir": "results/new_strategy_micro_overlays_265",
        "summary_files": "results/new_strategy_micro_overlays_265/micro_overlay_summary.csv",
        "public_candidate_ids": "467,472",
        "verdict": "public_neutral",
        "notes": "OOF improved, but two public submissions tied visible 90.78. Probably too small or not aligned with public rows.",
    },
    {
        "family_id": "similarity_overlays_on_265",
        "stage": "posthoc_overlay",
        "approach": "v14 plus KNN similarity overlay on current pred265 anchor",
        "training": "No new training; deterministic blend/clip",
        "primary_script": "analysis/new_strategy/build_similarity_overlays_on_265.py",
        "result_dir": "results/new_strategy_similarity_overlays_265",
        "summary_files": "results/new_strategy_similarity_overlays_265/similarity_overlay_summary.csv",
        "public_candidate_ids": "476,477,478",
        "verdict": "candidate_not_yet_public",
        "notes": "Independent small local signal; candidate 476 had best local delta but one major region worse.",
    },
]


METRIC_COLUMNS = [
    "oof_mape",
    "candidate_oof_mape",
    "mape",
    "valid_mape",
    "overall_mape",
]

DELTA_COLUMNS = [
    "delta_vs_265_oof",
    "delta_vs_pred265",
    "delta_vs_best_known_clean",
    "delta_vs_current_heuristic",
    "delta_vs_shrunk_yoy_base",
    "delta",
]


def parse_candidate_id(row: pd.Series) -> float:
    for col in ["candidate_id", "id"]:
        if col in row and pd.notna(row[col]):
            try:
                return float(int(row[col]))
            except Exception:
                pass
    for col in ["file", "candidate", "name"]:
        if col in row and pd.notna(row[col]):
            match = re.search(r"(?:candidate_|^)(\d{3})(?:_|\.|$)", str(row[col]))
            if match:
                return float(int(match.group(1)))
    return np.nan


def best_metric_column(df: pd.DataFrame) -> str | None:
    for col in METRIC_COLUMNS:
        if col in df.columns:
            return col
    return None


def best_delta_column(df: pd.DataFrame) -> str | None:
    for col in DELTA_COLUMNS:
        if col in df.columns:
            return col
    return None


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size <= 1:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_family_candidates(family: dict[str, str]) -> pd.DataFrame:
    parts = []
    for rel in filter(None, family["summary_files"].split(";")):
        path = ROOT / rel
        df = safe_read_csv(path)
        if df is None or df.empty:
            continue
        metric_col = best_metric_column(df)
        delta_col = best_delta_column(df)
        for rank, (_, row) in enumerate(df.sort_values(metric_col).iterrows() if metric_col else df.iterrows(), start=1):
            candidate_id = parse_candidate_id(row)
            cid_int = int(candidate_id) if pd.notna(candidate_id) else None
            parts.append(
                {
                    "family_id": family["family_id"],
                    "source_file": rel,
                    "rank_in_source": rank,
                    "candidate_id": cid_int,
                    "candidate": row.get("candidate", row.get("name", "")),
                    "file": row.get("file", ""),
                    "metric_col": metric_col or "",
                    "local_mape": row.get(metric_col, np.nan) if metric_col else np.nan,
                    "delta_col": delta_col or "",
                    "local_delta": row.get(delta_col, np.nan) if delta_col else np.nan,
                    "public_score": PUBLIC_SCORE_LOG.get(cid_int or -1, {}).get("public_score", np.nan),
                    "public_mape": PUBLIC_SCORE_LOG.get(cid_int or -1, {}).get("public_mape", np.nan),
                    "public_note": PUBLIC_SCORE_LOG.get(cid_int or -1, {}).get("note", ""),
                    "accepted_flag": row.get("accepted_for_submission", row.get("accepted_for_candidate_csv", "")),
                    "params_json_like": "; ".join(
                        f"{col}={row[col]}"
                        for col in [
                            "alpha",
                            "clip",
                            "gate",
                            "mask",
                            "base",
                            "expert",
                            "normal_method",
                            "trigger_mode",
                            "threshold",
                            "correction_weight",
                            "factor_lambda",
                            "group_col",
                            "growth_window",
                            "calendar_mode",
                            "store_weight_mode",
                            "blend_space",
                            "group_shrink_k",
                        ]
                        if col in row and pd.notna(row[col])
                    ),
                }
            )
    return pd.DataFrame(parts)


def build_registries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_rows = []
    candidate_parts = []
    for family in FAMILIES:
        cands = load_family_candidates(family)
        if not cands.empty:
            candidate_parts.append(cands)
            best = cands.sort_values(["local_mape", "rank_in_source"], na_position="last").iloc[0]
            best_local_candidate = best["candidate"]
            best_local_file = best["file"]
            best_local_mape = best["local_mape"]
            best_local_delta = best["local_delta"]
            n_candidates = len(cands)
        elif family["family_id"] == "anchor_temporal204_v11_gru_stack":
            best_local_candidate = "pred265_oof / stage2 candidate 265"
            best_local_file = "remote candidate 265"
            best_local_mape = ANCHOR_LOCAL_OOF_MAPE
            best_local_delta = 0.0
            n_candidates = 1
        else:
            best_local_candidate = ""
            best_local_file = ""
            best_local_mape = np.nan
            best_local_delta = np.nan
            n_candidates = 0

        family_rows.append(
            {
                **family,
                "n_indexed_candidate_rows": n_candidates,
                "best_local_candidate": best_local_candidate,
                "best_local_file": best_local_file,
                "best_local_mape": best_local_mape,
                "best_local_delta": best_local_delta,
            }
        )

    family_registry = pd.DataFrame(family_rows)
    candidate_registry = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    public_registry = pd.DataFrame(
        [
            {"candidate_id": cid, **payload}
            for cid, payload in sorted(PUBLIC_SCORE_LOG.items())
        ]
    )
    return family_registry, candidate_registry, public_registry


def write_markdown_snapshot(family_registry: pd.DataFrame, candidate_registry: pd.DataFrame) -> None:
    top = candidate_registry.dropna(subset=["local_mape"]).sort_values("local_mape").head(25)
    lines = [
        "# Model Registry Snapshot",
        "",
        f"Anchor public score: {ANCHOR_PUBLIC_SCORE}",
        f"Anchor public MAPE: {ANCHOR_PUBLIC_MAPE}",
        f"Anchor local pred265 OOF MAPE: {ANCHOR_LOCAL_OOF_MAPE:.6f}",
        "",
        "## Best Local Rows By Family",
        "",
        "| family_id | verdict | best_local_mape | best_local_delta | best_local_candidate |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in family_registry.sort_values("best_local_mape", na_position="last").iterrows():
        lines.append(
            f"| {row['family_id']} | {row['verdict']} | {row['best_local_mape']:.6f} | "
            f"{row['best_local_delta']:.6f} | {str(row['best_local_candidate'])[:120]} |"
            if pd.notna(row["best_local_mape"])
            else f"| {row['family_id']} | {row['verdict']} |  |  | {row['best_local_candidate']} |"
        )
    lines += [
        "",
        "## Top 25 Indexed Candidate Rows By Local MAPE",
        "",
        "| rank | family_id | candidate_id | local_mape | local_delta | public_score | file/candidate |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        label = row["file"] if isinstance(row["file"], str) and row["file"] else row["candidate"]
        cid = "" if pd.isna(row["candidate_id"]) else int(row["candidate_id"])
        public = "" if pd.isna(row["public_score"]) else f"{row['public_score']:.2f}"
        delta = "" if pd.isna(row["local_delta"]) else f"{row['local_delta']:.6f}"
        lines.append(
            f"| {rank} | {row['family_id']} | {cid} | {row['local_mape']:.6f} | {delta} | {public} | {str(label)[:140]} |"
        )
    (OUT_DIR / "registry_snapshot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    family_registry, candidate_registry, public_registry = build_registries()
    family_registry.to_csv(OUT_DIR / "family_registry.csv", index=False)
    candidate_registry.to_csv(OUT_DIR / "candidate_registry.csv", index=False)
    public_registry.to_csv(OUT_DIR / "public_score_log.csv", index=False)
    write_markdown_snapshot(family_registry, candidate_registry)
    print(f"Wrote {OUT_DIR / 'family_registry.csv'} ({len(family_registry)} families)")
    print(f"Wrote {OUT_DIR / 'candidate_registry.csv'} ({len(candidate_registry)} candidate rows)")
    print(f"Wrote {OUT_DIR / 'public_score_log.csv'} ({len(public_registry)} public score rows)")
    print(f"Wrote {OUT_DIR / 'registry_snapshot.md'}")


if __name__ == "__main__":
    main()
