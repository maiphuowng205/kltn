# ASEAN V2.1-RC1 audit package

The daily backtest now writes both Parquet and CSV ledgers for each cost
scenario. The CSV files are intended for independent recomputation and the
Parquet files are intended for downstream analysis.

## Required ledger files

- `daily_portfolio_returns.csv`: one row per country and evaluation date.
- `portfolio_weights.csv`: pre-trade and target weight for every name in the
  prior/current-universe union at each rebalance.
- `portfolio_trades.csv`: signed and absolute trade, with entry/exit labels.
- `portfolio_costs.csv`: per-name cost rate, trade amount and cost.
- `turnover_decomposition.csv`: continuing-name, forced-entry and forced-exit
  turnover. `reported_turnover` is their full-L1 sum.
- `risk_coverage.csv`: covariance history length, valid risk assets and risk
  fallback status.
- `solver_log.csv` and `fallback_log.csv`: optimizer status and all fallback
  events.

The first portfolio allocation is labelled `initial_deployment` and is not
subject to the recurring turnover cap. The cap applies from the second
rebalance onward, so an initial 100% deployment is not reported as an
infeasible 250% cap utilization.

`portfolio_metrics_summary.csv` and `reliability_metrics.csv` are derived
summary tables, not substitutes for the ledgers.

The reliability table reports `missing_valuation_event_rate_per_asset_day`,
whose denominator is held asset-days. It also reports the raw event count and
events per evaluation day. `cumulative_cost_drag` is the cumulative cost over
the evaluation period; it is not annualized.

The same engine can be run with `--alpha-mode zero` to create a risk-only
MVO/CA benchmark. This benchmark is required when country calibration maps a
forecast beta to zero.

## Turnover definition

For each execution date, turnover is computed over the union of the previous
and current holdings:

```text
turnover = continuing_name_turnover
         + forced_entry_turnover
         + forced_exit_turnover
```

This avoids hiding a forced exit when a name leaves the investable universe.
The cost ledger uses the same trade amounts. The report therefore makes it
possible to check whether a high Philippines turnover is caused by signal
changes, universe churn or forced exits.

## Release status

`freeze_asean_v21_rc1.py` creates an immutable `V2.1-RC1` directory and refuses
to overwrite an existing freeze. This release candidate is not a final
holdout result. Before a final claim, replay the V1 forecasts through the
corrected engine, audit Philippines turnover, and run the pre-registered
holdout.
