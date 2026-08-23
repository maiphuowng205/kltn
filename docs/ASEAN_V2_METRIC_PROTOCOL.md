# Locked PTCST-v2 metric protocol

Every country and every forecast model uses the same table. Forecast claims
are made from mean Spearman IC, median IC, IC hit rate, ICIR and top-minus-
bottom return. Pearson IC, MAE, RMSE and directional accuracy are secondary.
Constant forecasts, including Zero, have rank metrics recorded as `N/A`.

Forecast diagnostics are median cross-sectional forecast dispersion,
dispersion ratio `SD(prediction) / SD(target)`, and the validation calibration
slope. These diagnose scale; they are not a reason to claim ranking skill.

The daily-state portfolio table reports annualized net Sharpe, annualized net
excess return, annualized volatility, maximum drawdown, turnover, cost drag,
and return per unit turnover. It uses `sqrt(252)` only. Legacy weekly V1
results retain `sqrt(52)` and must not be combined with V2 values.

Reliability is always reported separately: evaluation coverage, eligible N,
valid risk-history N, covariance fallback, solver fallback and missing-return
rate. ASEAN summaries are equal-country averages, with ASEAN-5 and
valid-risk-market summaries shown separately.

All confidence intervals bootstrap dates/rebalance blocks. Model selection is
validation-only and lexicographic: Mean Spearman IC, then ICIR, then
top-minus-bottom; a model cannot be declared superior because it wins one
secondary metric.
