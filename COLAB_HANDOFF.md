# Colab handoff runbook

This is the only remaining external execution gate in `IMPLEMENTATION_PLAN_V3.md`.
The repository is pinned by Notebook 00 to commit `dde0da0` and the frozen data
must be uploaded manually; no LSEG credentials or LSEG API calls are used in
Colab.

## Before opening Colab

Upload the complete frozen package to:

```text
MyDrive/kltn/frozen/vn_v3_lseg_2026-08-03/
```

The package must preserve:

```text
data/lseg_v3/**
data/external/risk_free_daily.parquet
data/external/risk_free_daily.metadata.json
```

## Notebook order

Run these notebooks from the cloned repository in order:

```text
00_colab_setup_and_validate.ipynb
01_build_tensor_cache.ipynb
02_run_forecast_baselines.ipynb
03_run_risk_optimizer_backtest.ipynb
04_train_ptcst.ipynb
05_run_main_ablations.ipynb
06_locked_test_and_inference.ipynb
07_generate_tables.ipynb
```

Notebook 00 copies the Drive package to local Colab storage and aborts if any
of the 276 freeze-manifest checksums fail. Later notebooks write only to the
workspace `runs/` and `artifacts/` directories. Notebook 07 validates the
completed method, assembles `runs/v3_final_handoff/`, then calls
`scripts/sync_v3_handoff.py` to copy the handoff and exact notebook files to
Drive. The destination is persistent, so the sync manifest is the archive
receipt for the run.

Do not change `PINNED_COMMIT`, the freeze ID, the feature list, seed list,
cost rule or split protocol after the protocol lock is created.
