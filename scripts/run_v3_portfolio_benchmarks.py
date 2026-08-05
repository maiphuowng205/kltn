"""Run the locked non-deep portfolio benchmark ladder on V3 test dates."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.v3_method import cost_aware_mvo, ledoit_covariance


def future_returns(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ric, group in daily.sort_values(["ric", "session_id"]).groupby("ric", sort=False):
        g = group.copy(); contiguous = pd.Series(True, index=g.index); stock = pd.Series(1.0, index=g.index); rf = pd.Series(1.0, index=g.index)
        for k in range(1, 6):
            contiguous &= g["session_id"].shift(-k).eq(g["session_id"] + k)
            stock *= 1 + g["return"].shift(-k); rf *= 1 + g["rf_daily"].shift(-k)
        g["future_raw_5d"] = np.where(contiguous, stock - 1, np.nan); g["future_excess_5d"] = np.where(contiguous, stock - rf, np.nan)
        out.append(g[["date", "ric", "future_raw_5d", "future_excess_5d"]])
    return pd.concat(out, ignore_index=True)


def equal_target(w_pre: np.ndarray, valid: np.ndarray) -> np.ndarray:
    target = w_pre.copy(); fixed = float(target[~valid].sum()); n = int(valid.sum())
    if n < 1 or 1 - fixed < -1e-8:
        return w_pre.copy()
    target[valid] = (1 - fixed) / n
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/lseg_v3")
    parser.add_argument("--forecast-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/v3_portfolio_benchmarks")
    args = parser.parse_args(); args.run_dir.mkdir(parents=True, exist_ok=True)

    forecasts = pd.read_parquet(args.forecast_run / "forecasts.parquet")
    forecasts["date"] = pd.to_datetime(forecasts["date"]).dt.normalize()
    forecasts = forecasts.loc[forecasts["split"].eq("test")].copy()
    model_names = sorted(forecasts["model"].unique())
    weekly = pd.read_parquet(args.data_root / "curated" / "universe_weekly.parquet"); weekly["date"] = pd.to_datetime(weekly["date"]).dt.normalize()
    weekly = weekly.loc[weekly["split"].eq("test")].sort_values(["date", "market_cap_rank"]).groupby("date", sort=True).head(100)
    daily = pd.read_parquet(args.data_root / "curated" / "daily_panel.parquet"); daily["date"] = pd.to_datetime(daily["date"]).dt.normalize(); daily["session_id"] = daily["session_id"].astype(int); daily["rf_daily"] = pd.to_numeric(daily["rf_daily"], errors="coerce").fillna(0.0)
    daily = daily.merge(future_returns(daily), on=["date", "ric"], how="left", validate="one_to_one"); daily_map = daily.set_index(["date", "ric"])
    prediction = {(model, date, str(ric)): float(value) for model, date, ric, value in forecasts[["model", "date", "ric", "prediction"]].itertuples(index=False, name=None)}
    dates = sorted(weekly["date"].unique())
    strategies = ["EW", "EW-BH", "MinVar", "HM-MVO", "Ridge-MVO", "XGB-MVO", "XGB-CA-MVO"]
    returns, weights, trades, solver_rows, missing_events = [], [], [], [], []

    # Risk history depends only on the date's fixed 100-name universe, not on
    # the strategy.  Caching avoids repeating seven identical Ledoit-Wolf
    # fits per rebalance and makes the Colab benchmark stage substantially
    # more predictable without changing the locked estimator or policy.
    risk_cache: dict[tuple[pd.Timestamp, tuple[str, ...]], tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}

    for strategy in strategies:
        w_pre_map: dict[str, float] = {}
        for index, date_value in enumerate(dates):
            date = pd.Timestamp(date_value); rics = weekly.loc[weekly["date"].eq(date), "ric"].astype(str).tolist(); n = len(rics)
            risk_key = (date, tuple(rics))
            if risk_key not in risk_cache:
                risk_cache[risk_key] = ledoit_covariance(daily, date, rics)
            covariance, valid, risk_status = risk_cache[risk_key]
            exited = {ric: weight for ric, weight in w_pre_map.items() if ric not in set(rics)}; exited_weight = float(sum(exited.values()))
            w_pre = np.asarray([w_pre_map.get(ric, 0.0) for ric in rics], dtype=float)
            if index == 0 and w_pre.sum() == 0 and valid.sum() >= 20:
                w_pre[valid] = 1.0 / valid.sum()
            if strategy in ("EW", "EW-BH"):
                target = equal_target(w_pre, valid) if strategy == "EW" or index == 0 else w_pre.copy()
                if strategy == "EW-BH" and exited_weight > 0 and valid.any():
                    target[valid] += exited_weight / valid.sum()
                solver = {"solver": None, "status": "direct_equal_weight", "fallback": None}
            else:
                model = "historical_mean" if strategy == "HM-MVO" else "ridge" if strategy == "Ridge-MVO" else "xgb"
                mu = np.asarray([prediction[(model, date, ric)] for ric in rics], dtype=float) / 10000.0 if model in model_names else np.zeros(n)
                if strategy == "MinVar":
                    mu = np.zeros(n); target, solver = cost_aware_mvo(mu, covariance, w_pre, valid, risk_aversion=1.0, cost=0.0, turnover_fixed=exited_weight)
                else:
                    target, solver = cost_aware_mvo(mu, covariance, w_pre, valid, cost=0.001 if strategy == "XGB-CA-MVO" else 0.0, turnover_fixed=exited_weight)
            turnover = float(np.abs(target - w_pre).sum() + exited_weight); realized=[]; raw=[]
            for ric in rics:
                try:
                    row = daily_map.loc[(date, ric)]; realized.append(float(row["future_excess_5d"])); raw.append(float(row["future_raw_5d"]))
                except KeyError:
                    realized.append(np.nan); raw.append(np.nan)
            realized_arr = np.asarray(realized, dtype=float); raw_arr = np.asarray(raw, dtype=float); complete = np.isfinite(realized_arr) & np.isfinite(raw_arr)
            gross = float(target @ realized_arr) if complete.all() else float("nan"); net = gross - 0.001 * turnover if complete.all() else float("nan")
            for ric, ok in zip(rics, complete):
                if not ok: missing_events.append({"strategy": strategy, "date": date, "ric": ric, "reason": "future_5_session_return_unavailable"})
            returns.append({"strategy": strategy, "date": date, "gross_excess_return_5d": gross, "turnover_l1": turnover, "turnover_exited": exited_weight, "cost": 0.001 * turnover, "net_excess_return_5d": net, "evaluation_available": bool(complete.all()), "available_realized_assets": int(complete.sum()), "risk_fallback": risk_status.get("fallback"), "solver": solver.get("solver"), "solver_status": solver.get("status"), "solver_fallback": solver.get("fallback")})
            solver_rows.append({"strategy": strategy, "date": date, "turnover_exited": exited_weight, **risk_status, **solver})
            for ric, pre in exited.items(): trades.append({"strategy": strategy, "date": date, "ric": ric, "trade": -float(pre), "exited_universe": True})
            for ric, weight, pre in zip(rics, target, w_pre):
                weights.append({"strategy": strategy, "date": date, "ric": ric, "weight": float(weight), "w_pre": float(pre)})
                trades.append({"strategy": strategy, "date": date, "ric": ric, "trade": float(weight - pre), "exited_universe": False})
            post = target * (1 + np.nan_to_num(raw_arr, nan=0.0)); total = post.sum(); w_pre_map = {ric: float(weight / total) for ric, weight in zip(rics, post)} if total > 0 else {ric: float(weight) for ric, weight in zip(rics, target)}

    returns_df = pd.DataFrame(returns); weights_df = pd.DataFrame(weights); trades_df = pd.DataFrame(trades); solver_df = pd.DataFrame(solver_rows); missing_df = pd.DataFrame(missing_events, columns=["strategy", "date", "ric", "reason"])
    returns_df.to_parquet(args.run_dir / "portfolio_returns.parquet", index=False); weights_df.to_parquet(args.run_dir / "weights.parquet", index=False); trades_df.to_parquet(args.run_dir / "trades.parquet", index=False); solver_df.to_parquet(args.run_dir / "solver_log.parquet", index=False); missing_df.to_parquet(args.run_dir / "missing_price_events.parquet", index=False)
    eval_df = returns_df.loc[returns_df.evaluation_available]
    summary = eval_df.groupby("strategy").agg(evaluation_dates=("date", "count"), mean_net_excess_5d=("net_excess_return_5d", "mean"), mean_turnover=("turnover_l1", "mean"), net_sharpe_annualized=("net_excess_return_5d", lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(52)) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan)).reset_index()
    summary.to_parquet(args.run_dir / "portfolio_metrics_summary.parquet", index=False)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "strategies": strategies, "test_dates": len(dates), "forecast_run": str(args.forecast_run), "models_available": model_names}
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(summary.to_string(index=False)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
