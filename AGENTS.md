# Agent Handoff: Growth Gradient

This repository is an active forecasting research workspace for the Pyaterochka March 2025 RTO task.

The branch is intentionally artifact-heavy. Do not clean, delete, or rewrite experiment outputs unless the user explicitly asks for that. The purpose of this branch is to preserve all useful model families, negative results, local metrics, public submission notes, and generated CSV files so the team can choose the final model later.

## Current Anchor

- Current public best visible score: `90.78`.
- Current public MAPE anchor: `4.7214609684`.
- Current local `pred265` OOF MAPE anchor: `3.314574`.
- The current best public anchor is candidate `265`, referred to as `pred265` / temporal204 v11 GRU stack.

Any new candidate should be compared against `pred265`, not against older clean heuristics.

## Main Navigation Files

Start here:

- `analysis/model_registry/README.md`
- `analysis/model_registry/MODEL_FAMILY_NOTES.md`
- `analysis/model_registry/REPRODUCIBILITY_TODO.md`
- `results/model_registry/family_registry.csv`
- `results/model_registry/candidate_registry.csv`
- `results/model_registry/public_score_log.csv`
- `results/model_registry/registry_snapshot.md`

Regenerate the registry after adding new experiment summaries:

```bash
python3 analysis/model_registry/build_model_registry.py
```

If a new public submission result is known, update `PUBLIC_SCORE_LOG` in `analysis/model_registry/build_model_registry.py`, then rerun the registry script.

## Repository Layout

- `analysis/new_baseline/` - baseline and ML research scripts from the earlier phase.
- `analysis/new_strategy/` - later alternative heuristic/model-family experiments.
- `analysis/model_registry/` - experiment index and handoff documentation.
- `results/` - summary tables, OOF predictions, diagnostics, and generated registry files.
- `submissions/` - generated candidate CSV files. Keep both short aliases such as `472.csv` and full descriptive names.
- `notebooks/` - exploratory notebooks.

## Experiment Rules

When adding a new experiment family:

1. Put the script under `analysis/new_strategy/` or `analysis/new_baseline/`.
2. Write outputs to a dedicated `results/<family_name>/` directory.
3. Save at least one machine-readable summary CSV with local MAPE, delta versus anchor, parameters, and candidate file names.
4. Save submission CSVs under `submissions/` with names like `stage2_candidate_<id>_<description>.csv`.
5. If useful, also create a short alias `submissions/<id>.csv`.
6. Update `analysis/model_registry/build_model_registry.py` so the family appears in the global registry.
7. Rerun `python3 analysis/model_registry/build_model_registry.py`.

## Known Public Results

Known public outcomes are recorded in `results/model_registry/public_score_log.csv`.

Important examples:

- `265`: `90.78`, current anchor.
- `390`: `90.59`, feb-anomaly GRU, worse than anchor.
- `398`: `90.45`, safe feb-anomaly GRU, worse than anchor.
- `430`: `90.58`, uncapped feb-anomaly GRU, worse than anchor.
- `467`: `90.78`, micro overlay, public-neutral.
- `472`: `90.78`, stronger micro overlay, public-neutral.

## What Not To Do

- Do not remove old result folders just because a model lost. Negative results are useful for model selection and defense.
- Do not rely on public leaderboard alone. The next phase should rank models by local MAPE and known public behavior.
- Do not assume all local MAPE columns are comparable without checking the protocol in the source script.
- Do not run server commands unless the user explicitly re-enables server usage. Current workflow is local.
- Do not commit virtual environments, OS metadata, or caches.

## Final Model Reproducibility

Before final delivery, fill in `analysis/model_registry/REPRODUCIBILITY_TODO.md` with:

- final candidate id and CSV path;
- exact run command;
- Python and package versions;
- seed/fold/training parameters;
- input files and expected sizes/hashes;
- output files;
- submission format check.
