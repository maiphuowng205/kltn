# ASEAN protocol V2

V2 is a new development pipeline; it does not overwrite or reinterpret the
frozen V1 result tables.

1. `build_asean_v2_dataset.py` builds an EOD-safe, variable-N dataset.
   The signal is observed after close `t`, execution is at close `t+1`, and
   the five-session label begins on `t+2` and ends on `t+6`.  It also adds a
   cross-sectionally demeaned target, market-regime features, pure Top-100
   membership, a 60-session feature gate, and a past-only covariance-history
   gate.
2. `run_asean_v2_forecasts.py` trains one shared ASEAN PTCST with country
   embeddings and country-specific heads. Cross-sectional attention operates
   only within one country/date; padding is masked. It uses a Huber plus
   pairwise ranking loss, runs five seeds by default, and calibrates scores on
   2023 validation only before handing them to an optimizer.
3. `run_asean_v2_daily_backtest.py` applies a continuous daily state
   backtest: P&L accrues before close execution, missing valuations are
   retained and logged, covariance requires a configurable 90/126/180/252
   sessions, and transaction cost is each stock's lagged EOD half-spread
   (with explicit 10 bps fallback).

`2024–2025` are development data in V2 because they have already been
observed. A new final test must use a pre-registered 2026 holdout once it has
been extracted from LSEG. Until then, report purged development or
walk-forward evidence only.

The remaining research choice that must be selected *without a new holdout*
is the risk-aversion/volatility-target configuration. The backtest requires
an explicit `--risk-aversion` instead of silently choosing one after viewing a
final test outcome.
