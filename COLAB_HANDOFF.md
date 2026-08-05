# Colab handoff runbook

This is the only remaining external execution gate in `IMPLEMENTATION_PLAN_V3.md`.
The repository is pinned by Notebook 00 to immutable SHA
`c52e00ebab2cb0025881e6a4b80b9ed672edbf06` via tag `v3-colab-checkpoint`, and the frozen data
must be uploaded manually; no LSEG credentials or LSEG API calls are used in
Colab.

## Before opening Colab

Upload the complete frozen package to:

```text
MyDrive/kltn/frozen/vn_v3_lseg_2026-08-03/
```

To create a verified upload archive from the local frozen inputs, run
`python scripts/package_v3_drive.py`. The generated ZIP is ignored by Git and
contains 276 frozen files plus the two risk-free files and
`FROZEN_DATASET_V3.md`; upload/extract it so the directory layout above is
preserved.

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

Notebook 04 persists deep-model checkpoints under
`MyDrive/kltn/runs/checkpoints/`. Each model/seed keeps `last.pt` (optimizer,
epoch, best score and training history) plus `best.pt`; rerunning Notebook 04
with the same pinned code resumes from `last.pt` when a prior runtime stopped
before `metrics.json` was written.
