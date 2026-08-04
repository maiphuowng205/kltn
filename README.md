# Vietnam equity dataset audit

This repository audits a reproducible HOSE + HNX daily OHLCV extract before any model training.  It uses Vnstock only for price data and treats its present-day reference list as a possible source of survivorship bias.

## Current Vietnam V3 experiment

The main research dataset is the frozen LSEG Vietnam V3 release. The experiment
plan is in `IMPLEMENTATION_PLAN_V3.md`; the freeze policy is in
`FROZEN_DATASET_V3.md`. The V3 data and generated runs are intentionally not
committed to Git because the LSEG extract is supplied separately by the user
through Google Drive for Colab.

Colab workflow:

1. Clone the repository at a pinned commit.
2. Mount Google Drive and set `V3_DRIVE_ROOT` to the user-uploaded frozen V3 package.
3. Copy the package to the Colab local runtime and verify `freeze_manifest_v3.json`.
4. Run the notebooks in the order specified by `IMPLEMENTATION_PLAN_V3.md`.

The old Vnstock V2 audit scripts remain available for provenance; they are not
the main V3 experimental protocol.

## V3 runnable stages

The reusable method code is in `src/v3_method.py`; it never writes inside the
frozen data directory. The main method is PTCST forecast + 252-session
Ledoit–Wolf covariance + cost-aware long-only MVO. The runner writes protocol,
freeze, environment, preprocessing, forecasts, weights, trades, solver and
missing-return artifacts under a supplied `runs/` directory.

```powershell
python scripts/validate_v3_contract.py --workspace-root .
python scripts/validate_v3_method.py --data-root data/lseg_v3
python scripts/run_v3_forecast_baselines.py --run-dir runs/v3_forecast_baselines_main
python scripts/run_v3_portfolio_benchmarks.py --forecast-run runs/v3_forecast_baselines_main --run-dir runs/v3_portfolio_benchmarks_main
python scripts/run_v3_ptcst_method.py --model-type PTCST --run-dir runs/v3_ptcst_main_seed7
python scripts/run_v3_ptcst_ablations.py --forecast-run runs/v3_ptcst_main_seed7 --run-dir runs/v3_ptcst_ablations
python scripts/run_v3_walk_forward.py --run-dir runs/v3_walk_forward
```

The local full method evidence currently uses 207 train dates, 52 validation
dates and 104 test dates with train-only median/IQR imputation. Primary
portfolio metrics exclude dates whose five-session outcomes are not fully
realized; those rows are retained in `missing_price_events.parquet`.

## Install

```powershell
python -m pip install -r requirements.txt
```

To use a Community key without putting it in code, create a local `.env` file (already ignored by Git):

```text
VNSTOCK_API_KEY=your_key_here
```

The crawler registers that key at startup. Without it, the script uses a conservative guest-safe 7 seconds between tickers. With a registered Community key, use `--sleep-seconds 2.2` after a small pilot confirms the actual limit.

## Run

```powershell
# Fetch/resume one Parquet per current HOSE/HNX symbol, then audit it.
python scripts/dataset_audit_v1.py all --start 2018-01-01 --end 2025-12-31

# Run only the deterministic audit after an interrupted crawl has resumed.
python scripts/dataset_audit_v1.py audit

# A one-ticker API and schema smoke test, isolated from the main data folder.
python scripts/dataset_audit_v1.py all --symbols FPT --data-root data/smoke
```

The crawl saves `data/raw/universe_current.csv`, `data/raw/ohlcv/<TICKER>.parquet`, and ticker-level failures in `data/raw/crawl_errors.csv`. Existing ticker files are skipped, so the crawl is resumable; pass `--overwrite` only to download them again.

The audit writes:

- `data/dataset_audit_v1.json` — the paper-ready counts and explicit limitations.
- `data/coverage_report.csv` — first/last observation, history length, zero-volume and missing-field counts per ticker.
- `data/forecast_dates.csv` — every weekly forecast date, candidate count and whether a full Top-100 cross-section is available.

`final_stock_week_targets` is not the raw row count: it is the sum of 100 targets for each weekly date that has a full eligible Top-100 universe. Eligibility requires 252 valid observed OHLCV rows for risk estimation, 60-day median `close * volume`, and a valid future 5-session close for the target.

Vnstock 4.0.5 defaults to `count=100`, even when `start` and `end` are supplied. The script therefore requests `--bar-count 5000` by default; never remove that parameter when auditing multi-year coverage.

## Dataset V2: calendar-aware model-development data

V2 retains the V1 raw cache and corrects the supervised-label timing: a weekly signal at market close `t` enters at close `t+1` and exits at close `t+6`, which is a five-market-session holding period. A ticker without a close on either exact execution date is excluded; the script never substitutes its fifth subsequent ticker row.

```powershell
& "$env:USERPROFILE\.venv\Scripts\Activate.ps1"
python scripts/build_dataset_v2.py
```

It writes `data/processed/market_calendar.parquet`, `universe_weekly.parquet`, `features.parquet`, `targets.parquet`, `split_manifest.csv`, `corporate_action_flags.csv`, and `dataset_v2_report.json`. `targets.parquet` always contains `raw_return_5d`; it leaves `excess_return_5d` unavailable until a documented daily risk-free series is supplied:

```powershell
python scripts/build_dataset_v2.py --risk-free-file data/external/risk_free_daily.parquet
```

The risk-free input must have `date` and `rf_daily`, where `rf_daily` is the daily simple return. V2 is explicitly not frozen for paper claims until point-in-time/delisted universe coverage and the adjustment audit are resolved.
