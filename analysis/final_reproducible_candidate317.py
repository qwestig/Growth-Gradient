from __future__ import annotations

import os
import random
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
SELECTED_V15_NAME = "huber030_g0.80_c0.008"
FINAL_OUTPUT = ROOT / "test.csv"


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def add_import_paths() -> None:
    paths = [
        ROOT / "analysis",
        ROOT / "analysis" / "new_baseline",
        ROOT / "analysis" / "new_baseline" / "temporal204",
    ]
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def ensure_input_file() -> Path:
    canonical = ROOT / "train_2.csv"
    alternate = ROOT / "train.csv"
    if canonical.exists():
        return canonical
    if alternate.exists():
        shutil.copyfile(alternate, canonical)
        return canonical
    raise FileNotFoundError(
        "Expected input file train_2.csv in project root. "
        "If the checker provides train.csv, place it in the project root and rerun."
    )


def run_script(rel_path: str) -> None:
    path = ROOT / rel_path
    print(f"\n=== Running {rel_path} ===")
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(path)]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv


def verify_submission(path: Path) -> None:
    frame = pd.read_csv(path)
    assert frame.shape == (18657, 2), frame.shape
    assert list(frame.columns) == ["new_id", "rto"], frame.columns.tolist()
    assert frame.isna().sum().sum() == 0
    assert (frame["rto"] < 0).sum() == 0


def main() -> Path:
    set_seed()
    input_path = ensure_input_file()
    add_import_paths()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "submissions").mkdir(exist_ok=True)
    print(f"Input file: {input_path}")

    steps = [
        "analysis/gru_residual_experiment.py",
        "analysis/new_baseline/build_calendar_followup_candidates.py",
        "analysis/new_baseline/train_new_baseline_gru_residual.py",
        "analysis/new_baseline/build_gamma_high_sweep_candidates.py",
        "analysis/new_baseline/build_faithful_gru_followup_candidates.py",
        "analysis/new_baseline/temporal204/build_structural_baseline_v6_candidates.py",
        "analysis/new_baseline/temporal204/train_temporal204_gru_residual.py",
        "analysis/new_baseline/temporal204/train_temporal204_second_residual.py",
        "analysis/new_baseline/temporal204/build_stacking_v11_candidates.py",
        "analysis/new_baseline/temporal204/build_v14_selective_v11_correction.py",
        "analysis/new_baseline/temporal204/export_temporal204_meta_inputs.py",
        "analysis/new_baseline/train_v15_meta_residual_on_265.py",
    ]
    for rel_path in steps:
        run_script(rel_path)

    candidates = pd.read_csv(ROOT / "results" / "new_baseline" / "v15_meta265_submission_candidates.csv")
    selected = candidates.loc[candidates["name"].eq(SELECTED_V15_NAME)]
    if selected.empty:
        raise RuntimeError(f"Selected v15 configuration not found: {SELECTED_V15_NAME}")
    final_candidate = ROOT / "submissions" / str(selected.iloc[0]["file"])
    if not final_candidate.exists():
        raise FileNotFoundError(final_candidate)
    final = pd.read_csv(final_candidate)
    final.to_csv(FINAL_OUTPUT, index=False)
    verify_submission(FINAL_OUTPUT)
    print(f"\nSelected v15 config: {SELECTED_V15_NAME}")
    print(f"Source candidate file: {final_candidate}")
    print(f"Final submission written to {FINAL_OUTPUT}")
    return FINAL_OUTPUT


if __name__ == "__main__":
    main()
