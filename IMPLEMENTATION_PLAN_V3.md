# Implementation Plan — Vietnam V3 Experimental Pipeline

## 1. Objective and locked scope

Implement the two-stage experiment described in `deep-research-report.md`:

1. Forecast five-session excess returns for a frozen weekly cross-section of 100 HOSE/HNX securities.
2. Convert forecasts into long-only portfolio weights using a Ledoit–Wolf risk model and frictionless/cost-aware mean–variance optimization.
3. Execute at the next market close, account for drift and transaction costs, and evaluate forecast and portfolio performance out of sample.

Dataset V3 is immutable. All code must verify `data/lseg_v3/reports/freeze_manifest_v3.json` before a run and must never write inside `data/lseg_v3` or `data/external`. Derived caches go to `artifacts/`; experiment outputs go to `runs/`.

Main-market scope is Vietnam only. The US dataset is excluded. Claims prohibited by `FROZEN_DATASET_V3.md`, including “survivorship-bias-free” and observed historical bid/ask costs, remain prohibited.

Execution platform is Google Colab. Each notebook starts from a clean Colab runtime, clones `https://github.com/maiphuowng205/kltn.git`, checks out a pinned commit, installs the Colab requirements and calls reusable Python modules/scripts from the cloned repository. Notebooks are orchestration layers; research logic remains in `src/` so results do not depend on hidden notebook state.

## 2. Frozen empirical contract

| Item | Locked value |
|---|---|
| Freeze ID | `vn_v3_lseg_2026-08-03` |
| Weekly rows | 39,900 = 399 dates × 100 RICs |
| Exact labels | 39,642; 258 rows remain masked |
| Train | 2019–2022: 20,700 rows |
| Validation | 2023: 5,200 rows |
| Test | 2024–2025: 10,400 rows |
| Warm-up/excluded | 3,600 rows |
| Forecast target | `target_excess_return_5d_bps` |
| Input lookback | 60 master-market sessions |
| Signal | Final observed session of each W-FRI week |
| Execution | Close of the next master-market session |
| Holding horizon | Five master-market sessions |
| Main cost | Assumed proportional cost: 10 bps per unit of absolute weight traded |
| Main covariance | 252-session Ledoit–Wolf shrinkage |
| Main portfolio | Long-only, fully invested, max 5% per asset, L1 turnover cap 40% |

`target_available=False` is a loss/evaluation mask only. It may not alter the frozen 100-name cross-section.

## 2A. Google Colab and GitHub workflow

### Code distribution

- GitHub repository: `https://github.com/maiphuowng205/kltn.git`.
- GitHub contains notebooks, source code, configurations, tests and small metadata files.
- Every final run checks out a commit SHA, not an unpinned moving branch.
- Colab records the commit SHA in `environment.json` and every result manifest.
- If the repository becomes private, authenticate with a read-only GitHub token stored in Colab Secrets; never place a token in notebook cells or committed files.

### Frozen-data distribution

Colab cannot use the local LSEG Workspace Desktop session. Therefore, Colab must not call LSEG download scripts. Prepare one immutable archive outside Colab containing these exact paths:

```text
data/lseg_v3/**
data/external/risk_free_daily.parquet
data/external/risk_free_daily.metadata.json
FROZEN_DATASET_V3.md
```

Recommended storage is Google Drive:

```text
MyDrive/kltn/frozen/vn_v3_lseg_2026-08-03/
```

The student/user manually uploads this frozen V3 package to their own Google Drive before running the notebooks. The implementation does not upload data, request LSEG credentials, or recreate LSEG extracts from Colab. The Drive location is supplied through one setup variable such as `V3_DRIVE_ROOT`; it is not hard-coded to a personal account path.

The current V3 directory is approximately 186 MB and is not tracked by the Git repository. Do not assume `git clone` downloads V3. Do not publish licensed LSEG raw data to a public GitHub repository. The user-uploaded Drive copy must preserve the relative paths expected by the freeze manifest.

### Runtime layout

```text
/content/temp/                         # cloned GitHub code
/content/vn_v3_workspace/data/         # local read-only copy of frozen data
/content/vn_v3_workspace/artifacts/    # local derived tensor/cache files
/content/vn_v3_workspace/runs/         # local run outputs
/content/drive/MyDrive/kltn/runs/      # persistent checkpoints/results
```

Copy the frozen dataset from Drive to the Colab local SSD at notebook startup for faster Parquet access. Validate all checksums after copying. Train and backtest locally, then sync checkpoints and completed artifacts to Drive after each stage/epoch checkpoint. A runtime restart must be resumable from Drive.

### Standard setup cell

The setup notebook will implement the equivalent of:

```python
from google.colab import drive
drive.mount('/content/drive')

# Shell commands executed by the notebook:
# git clone --depth 1 https://github.com/maiphuowng205/kltn.git /content/kltn
# cd /content/kltn
# git fetch --depth 1 origin <PINNED_COMMIT>
# git checkout <PINNED_COMMIT>
# python -m pip install -r requirements-colab.txt

# User-supplied location of the manually uploaded frozen dataset:
# V3_DRIVE_ROOT = '/content/drive/MyDrive/kltn/frozen/vn_v3_lseg_2026-08-03'
```

Notebook code must use `pathlib.Path` and configuration values; it must not contain Windows paths such as `D:\\kltn`.

### Runtime selection

- CPU/high-RAM runtime: validation, cache construction, Ridge/XGBoost, covariance, CVXPY, backtest and statistics.
- GPU runtime: vanilla Transformer, PatchTST and PTCST training/inference.
- Mixed precision is optional and must be fixed in the config. Deterministic settings and device information are recorded.
- Long stages checkpoint to Drive so Colab disconnections do not force a restart from zero.

## 3. Repository architecture to implement

```text
notebooks/
  00_colab_setup_and_validate.ipynb
  01_build_tensor_cache.ipynb
  02_run_forecast_baselines.ipynb
  03_run_risk_optimizer_backtest.ipynb
  04_train_transformers.ipynb
  05_run_main_ablations.ipynb
  06_run_locked_test_and_inference.ipynb
  07_generate_tables.ipynb
configs/
  v3_main.yaml
  v3_robustness.yaml
requirements-colab.txt
src/
  data/
    contracts.py
    v3_loader.py
    tensor_cache.py
    transforms.py
  models/
    baselines.py
    temporal_transformer.py
    ptcst.py
    training.py
  portfolio/
    covariance.py
    optimizer.py
    accounting.py
    backtest.py
  evaluation/
    forecast_metrics.py
    portfolio_metrics.py
    statistical_tests.py
    reporting.py
  utils/
    config.py
    io.py
    reproducibility.py
scripts/
  validate_v3_contract.py
  build_v3_tensor_cache.py
  run_v3_baselines.py
  run_v3_deep_models.py
  run_v3_backtest.py
  run_v3_inference.py
  run_v3_experiment.py
tests/
  test_v3_contract.py
  test_no_leakage.py
  test_tensor_alignment.py
  test_covariance.py
  test_optimizer.py
  test_backtest_accounting.py
artifacts/
  v3_tensor_cache/
runs/
  <run_id>/
```

Notebooks call the scripts/modules shown above and may display tables/plots, but they must not duplicate model, optimizer or metric implementations inside cells.

## 4. Dependency-ordered milestones

### M0 — Reproducibility and freeze guard

Implement the Colab clone/setup flow, configuration parsing, deterministic seeds, environment capture and SHA-256 validation against the freeze manifest.

Deliverables:

- `configs/v3_main.yaml` with every locked protocol value.
- `requirements-colab.txt` with pinned major dependencies compatible with the selected Colab Python/PyTorch runtime.
- `notebooks/00_colab_setup_and_validate.ipynb` that clones the pinned Git commit, mounts Drive, copies V3 and runs validation.
- `validate_v3_contract.py` and freeze-manifest validator.
- Run metadata: config hash, freeze ID, Git commit, Python/package versions, seed and UTC timestamps.

Definition of done:

- All 276 frozen files match the manifest.
- Any changed/missing frozen file aborts the run before training.
- No code path writes into the frozen dataset directories.
- A clean Colab runtime can reproduce setup using only the notebook, GitHub repository and authorized Drive dataset archive.
- Restarting the runtime can resume from persistent Drive checkpoints without silently changing the Git commit or config.

### M1 — Dataset loader and tensor cache

Construct one sample per forecast date with shapes:

```text
X:             [date, 100, 60, F]
y:             [date, 100]
target_mask:   [date, 100]
ric:           [date, 100]
signal_date:   [date]
execution_date:[date, 100]
```

Use daily-panel rows ending at signal close. Preserve the weekly universe order by `market_cap_rank`. Do not bridge missing master sessions. Fit imputation, robust scaling and clipping parameters on training sequence rows only; persist them per run.

Initial feature set:

- Returns: 1, 5, 10, 20 and 60 sessions.
- Volatility: 5, 20 and 60 sessions.
- Liquidity/scale: log volume, log dollar volume, log price, log market cap and Amihud.
- Range/calendar: high–low proxy, day of week, month-end and quarter-end.

Definition of done:

- Exactly 100 assets and 60 ordered timesteps per forecast date.
- Every feature timestamp is `<= signal_date`.
- Changing validation/test data cannot change training scaler/imputer statistics.
- Cache records source hashes and is reproducible byte-for-byte where supported.

### M2 — Common forecast evaluation and simple baselines

Implement a shared model interface and these baselines before deep learning:

1. Zero forecast.
2. Expanding historical-mean forecast.
3. Pooled Ridge.
4. XGBoost using last-step and sequence-summary features.

The existing `run_v3_ridge_baseline.py` is preliminary only; rerun Ridge through the common loader/evaluator and do not treat its current metrics as final results.

Forecast metrics are computed per forecast date and then summarized across dates: Spearman rank IC, Pearson IC, MAE, RMSE/Huber, directional accuracy and top-minus-bottom realized spread. Missing targets use the frozen mask.

Definition of done:

- One `forecasts.parquet` schema for every model.
- Metrics are date-level; stock rows from one date are not treated as independent observations.
- Ridge and XGBoost beat or diagnose against zero/historical-mean baselines on validation without accessing test labels.

### M3 — Risk engine

For each signal date, load the preceding 252 master-market sessions ending at the signal close and estimate covariance only from past returns.

Main estimator: `sklearn.covariance.LedoitWolf`.

Pre-specified missing-history policy:

- An asset without the required covariance history is non-tradeable for that rebalance and retains its pre-trade weight.
- At portfolio inception its weight is zero.
- No rank-101 replacement is permitted.
- If fewer than 20 assets are tradeable or the covariance solve fails validation, use the logged fallback `w = w_pre`; at inception use equal weight across the valid set only if at least 20 assets are valid.

Before final runs, produce a train/validation-only risk-coverage report to confirm that the fallback is not dominant. Any policy revision after this audit creates a new protocol config before test is opened.

Definition of done:

- Covariance uses no timestamp after the signal date.
- Matrices are symmetric positive semidefinite within tolerance.
- Conditioning, available-asset count and fallback status are logged for every date.

### M4 — Optimizer and accounting kernel

Implement CVXPY optimizers:

- Equal weight.
- Minimum variance.
- Frictionless MVO.
- Cost-aware MVO.

Main cost-aware objective:

```text
maximize  w' mu - lambda/2 * w' Sigma w - c * sum(abs(w - w_pre))
subject to sum(w)=1
           0 <= w_i <= 0.05
           sum(abs(w - w_pre)) <= 0.40
```

Solver order: CLARABEL, then OSQP. A failed solve returns `w_pre` and logs status; it never silently substitutes equal weight.

Accounting must distinguish target weights, drifted pre-trade weights, executed weights, trades, one-way/L1 turnover, gross return, cost and net return.

Definition of done:

- Unit tests verify sum-to-one, long-only, 5% cap and turnover cap.
- Zero trade produces zero cost.
- Independent hand calculation matches at least one stored portfolio week.

### M5 — Deterministic non-deep end-to-end benchmark

Run these strategies first:

| Strategy | Forecast | Allocation |
|---|---|---|
| EW | None | Weekly equal weight |
| EW-BH | None | Initial equal weight, drift only |
| MinVar | None | Ledoit–Wolf minimum variance |
| HM-MVO | Historical mean | Frictionless MVO |
| Ridge-MVO | Ridge | Frictionless MVO |
| XGB-MVO | XGBoost | Frictionless MVO |
| XGB-CA-MVO | XGBoost | Cost-aware MVO |

Every strategy, including cost-unaware strategies, pays the same realized assumed cost in evaluation. “Cost-aware” means the optimizer sees the penalty before deciding trades.

Definition of done:

- Validation table contains forecast and gross/net portfolio metrics.
- Stored outputs include forecasts, weights, trades, costs, solver states and missing-price events.
- Backtest is deterministic across repeated runs with the same config.

### M6 — Deep forecast models

Minimum deep-model ladder:

1. Vanilla temporal Transformer.
2. PatchTST-style temporal-only model.
3. Proposed PTCST: patch-based temporal encoder plus one cross-sectional attention layer.

Optional LSTM/TCN baselines are added only if compute permits and do not delay the core comparison.

Initial PTCST configuration:

```yaml
lookback: 60
patch_length: 5
patch_stride: 5
d_model: 64
temporal_layers: 2
cross_sectional_layers: 1
n_heads: 4
ffn_dim: 128
dropout: 0.10
optimizer: AdamW
learning_rate: 0.0003
weight_decay: 0.0001
loss: Huber
max_epochs: 100
early_stopping_patience: 10
gradient_clip_norm: 1.0
seeds: [7, 19, 43, 71, 101]
selection_metric: validation_mean_weekly_spearman_ic
```

Lock a small tuning budget before runs: at most 12 validation configurations per deep architecture. Report mean and standard deviation across five seeds; never select the best test seed.

Definition of done:

- Checkpoints, learning curves and validation selection records are stored.
- Same target mask and features are used across models.
- PTCST ablations isolate patches and cross-sectional attention.

### M7 — Main portfolio ablations

Run:

- PTCST-Top20 equal weight.
- PTCST-MVO.
- PTCST-CA-MVO.
- XGB-CA-MVO.
- EW and MinVar.

These comparisons identify separately:

- Forecast value: PTCST versus Ridge/XGBoost/vanilla Transformer.
- Optimizer value: PTCST-MVO versus PTCST-Top20.
- Cost-awareness value: PTCST-CA-MVO versus PTCST-MVO.
- Complexity value: PTCST-CA-MVO versus XGB-CA-MVO/EW/MinVar.

### M8 — Locked test and expanding walk-forward

Pilot: fixed split, tune only on 2023 validation and evaluate 2024–2025 test once after configuration lock.

Main robustness: expanding walk-forward retraining quarterly during test, using only labels whose five-session outcomes are fully realized before each retraining origin. Architecture, feature list, cost rule, search space and selection metric remain locked.

Definition of done:

- A protocol-lock JSON is written before the first test evaluation.
- Test labels are not read by tuning/model-selection code.
- Every forecast records model training cutoff and checkpoint hash.

### M9 — Metrics and statistical inference

Portfolio metrics:

- Gross/net annualized return and volatility.
- Net Sharpe and Sortino.
- Maximum drawdown, Calmar and CVaR.
- Certainty-equivalent return.
- L1/one-way turnover, cost drag, concentration and maximum weight.

Statistical tests:

- DM with HLN adjustment on forecast-date average loss differentials.
- Paired stationary/block bootstrap with 4–12 week blocks for IC, net return, Sharpe, CE and turnover differences.
- Multiple-testing adjustment only after the main pre-specified comparisons.

Inference uses weekly/date-level observations, not 100 stock rows as independent samples.

### M10 — Pre-specified robustness

Run only after the main result is frozen:

| Dimension | Grid |
|---|---|
| Assumed cost | 0, 5, 10, 20, 30, 50 bps |
| Covariance | Ledoit–Wolf, sample, EWMA half-life 60 |
| Lookback | 20, 60, 120 sessions |
| Turnover cap | 20%, 40%, 80% |
| Max weight | 3%, 5%, 10% |
| Risk aversion | Validation-locked low/base/high values |

Do not select a new “winner” from robustness results.

## 5. Mandatory run artifacts

Each `runs/<run_id>/` contains:

```text
config.yaml
protocol_lock.json
environment.json
data_freeze.json
scaler.json
imputer.json
checkpoints/
forecasts.parquet
forecast_metrics_by_date.parquet
weights.parquet
trades.parquet
portfolio_returns.parquet
solver_log.parquet
missing_price_events.parquet
metrics.json
statistical_tests.json
run.log
```

Required keys include `date`, `ric`, model, seed and strategy where applicable. Saving cumulative wealth alone is insufficient.

## 6. Mandatory leakage and accounting tests

1. All sequence timestamps and covariance timestamps are no later than signal close.
2. Execution is the next master-market close, never the signal close.
3. Future target values cannot change weekly-universe membership.
4. Validation/test mutations cannot change training preprocessing statistics.
5. Missing-target assets remain in the 100-name universe and are not replaced.
6. Quarterly retraining uses only fully realized labels available at its cutoff.
7. Cost-unaware and cost-aware strategies pay the same realized cost rule.
8. Drifted pre-trade weights reconcile to previous holdings and realized returns.
9. Optimizer constraints hold within numerical tolerance.
10. Solver failures use the documented `w_pre` fallback.

## 7. Execution order and stop gates

```text
Notebook 00: clone GitHub + mount Drive + M0 freeze guard
  -> Notebook 01: M1 tensor/data contract
  -> Notebook 02: M2 simple forecasts
  -> Notebook 03: M3 risk + M4 optimizer + M5 non-deep benchmark
  -> Notebook 04: M6 deep models
  -> Notebook 05: M7 main ablations
  -> protocol lock saved to the persistent Drive run folder and tied to the pinned Git commit
  -> Notebook 06: M8 test/walk-forward + M9 inference
  -> Notebook 07: M10 robustness/reporting
```

Do not begin deep-model tuning until M5 is deterministic. Do not evaluate test until the protocol lock exists. Do not run robustness grids until the main test result and its configuration are frozen.

If Colab disconnects, rerun Notebook 00 with the same pinned commit/config and resume the interrupted stage from Drive. A new runtime does not authorize a new hyperparameter search or a changed protocol.

## 8. Practical delivery slices

| Slice | Milestones | Runnable outcome |
|---|---|---|
| A | Notebooks 00–02 / M0–M2 | Colab setup, validated tensor loader and zero/HM/Ridge/XGBoost forecast table |
| B | Notebook 03 / M3–M5 | Deterministic end-to-end non-deep portfolio benchmark |
| C | Notebooks 04–05 / M6–M7 | Five-seed Transformer/PTCST forecasts and main portfolio ablations |
| D | Notebook 06 / M8–M9 | Locked test, walk-forward results and statistical inference |
| E | Notebook 07 / M10 | Robustness appendix and final generated tables |

The recommended first implementation target is Slice A. It validates the frozen data contract and forecast evaluation before optimization or GPU training introduces additional failure modes.

## 9. Execution checklist

Use this checklist in the Colab/Drive run folder. Check an item only after its
artifact and acceptance tests are present; do not mark a stage complete merely
because a notebook cell finished.

Local implementation snapshot (2026-08-05): the V3 contract validator, full
100×60×17 tensor loader, common zero/HM/Ridge/XGBoost baselines, non-deep
portfolio ladder, protocol-locked seed-7 PTCST fixed-split method (run
`runs/v3_ptcst_main_seed7_v11`), five-seed PTCST aggregation, PTCST portfolio
ablations, quarterly Ridge walk-forward and report-table/statistics scripts
have been implemented and smoke/full-tested under ignored local `runs/`
directories. The method forecast rows carry model, seed, execution date,
selection cutoff and checkpoint SHA-256 metadata; the method run includes
`config.yaml` and `run_manifest.json`.
Notebook 00 now records package/device metadata, copies the freeze to a local
read-only input tree and persists setup metadata to Drive; Notebook 07 includes
the filesystem-only `sync_v3_handoff.py` archive step.
Notebook 01 now delegates cache construction to
`scripts/build_v3_tensor_cache.py`, which writes train-fitted preprocessing,
freeze-manifest SHA-256 and per-cache-file hashes in `cache_manifest.json`.
The unchecked boxes below remain the authoritative Colab/Drive handoff gates;
in particular, a clean Colab execution, Drive synchronization and archival of
the exact notebook outputs have not been claimed complete.

### Repository and data

- [ ] Clone `https://github.com/maiphuowng205/kltn.git`.
- [ ] Record the pinned Git commit SHA.
- [ ] Mount Google Drive and set `V3_DRIVE_ROOT`.
- [ ] Copy the user-uploaded frozen V3 package to Colab local storage.
- [ ] Verify all 276 freeze-manifest checksums.
- [ ] Confirm V3 and RF paths are read-only during the run.
- [ ] Capture Python, package, device and runtime metadata.

### M0–M2: data and forecast baselines

- [ ] Run Notebook 00 setup and contract validation.
- [ ] Build the 100 × 60 × F tensor cache.
- [ ] Verify timestamps are no later than signal close.
- [ ] Verify exactly 100 assets per forecast date.
- [x] Fit imputer/scaler on train rows only (verified in the seed-7 method run).
- [x] Run zero and historical-mean forecasts (full local baseline run).
- [x] Run Ridge forecast (full local baseline run).
- [x] Run XGBoost forecast (full local baseline run).
- [x] Save date-level forecast metrics and predictions.
- [x] Complete leakage tests for the loader and preprocessing.

### M3–M5: risk, optimizer and non-deep portfolio benchmark

- [x] Produce 252-session risk-coverage report (train/validation only; zero fallback dates).
- [x] Run Ledoit–Wolf covariance checks.
- [x] Run EW and EW-BH (full local benchmark run).
- [x] Run MinVar (full local benchmark run).
- [x] Run HM-MVO, Ridge-MVO and XGB-MVO (full local benchmark run).
- [x] Run XGB-CA-MVO (full local benchmark run).
- [x] Verify optimizer constraints and deterministic fallback.
- [x] Verify drift, turnover, cost and net-return accounting.
- [x] Save weights, trades, solver log and portfolio returns.
- [x] Confirm repeated run produces identical non-deep outputs (forecast and portfolio SHA-256 comparison).

### M6–M7: deep models and main ablations

- [x] Run vanilla temporal Transformer (seed-7 full local run).
- [x] Run PatchTST temporal-only baseline (seed-7 full local run).
- [x] Run PTCST with seeds 7, 19, 43, 71 and 101 (local sweep).
- [x] Save checkpoints, learning curves and validation selection records (per seed).
- [x] Run PTCST-Top20 (seed-7 local run).
- [x] Run PTCST-MVO (seed-7 local run).
- [x] Run PTCST-CA-MVO (seed-7 local run).
- [x] Complete forecast-value, optimizer-value and cost-awareness ablations (local fixed-split runs).

### M8–M10: locked evaluation and reporting

- [x] Write protocol-lock JSON before opening test labels.
- [x] Run fixed-split test once for the locked seed-7 method run.
- [x] Run expanding quarterly walk-forward (Ridge, eight quarterly retraining points).
- [x] Verify retraining uses only realized labels (quarterly walk-forward cutoff audit).
- [x] Compute forecast and portfolio metrics by date.
- [x] Run DM/HLN on date-level forecast losses.
- [x] Run paired block/stationary bootstrap.
- [x] Run the pre-specified cost/covariance/lookback robustness grid (21 one-dimension-at-a-time variants; full local run).
- [x] Generate final tables and run manifest (figures remain a reporting-layer task).
- [ ] Sync completed artifacts/checkpoints to Drive.
- [ ] Archive the exact notebook outputs and environment metadata.

### Final handoff gate

- [x] All required local run artifacts listed in Section 5 exist in `runs/v3_final_handoff`.
- [x] No test result was used for model selection or tuning.
- [x] Every strategy paid the same realized cost rule.
- [x] Claim restrictions in `FROZEN_DATASET_V3.md` are reproduced in the report.
- [x] Final report identifies the Git commit, freeze ID and run ID.
