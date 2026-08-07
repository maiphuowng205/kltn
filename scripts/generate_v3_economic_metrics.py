"""Generate wealth and economic metrics for the extension portfolio runs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def load_strategy(run_dir: Path, strategy: str, method_run: bool = False) -> pd.DataFrame:
    frame = pd.read_parquet(run_dir / "portfolio_returns.parquet")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if not method_run:
        frame = frame.loc[frame["strategy"].astype(str).eq(strategy)].copy()
    if "evaluation_available" in frame:
        frame = frame.loc[frame["evaluation_available"].astype(bool)].copy()
    return frame[["date", "net_excess_return_5d", "turnover_l1"]].sort_values("date")


def max_drawdown(values: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + values)
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)) if len(wealth) else np.nan


def summarize(frame: pd.DataFrame, label: str, period: str) -> dict[str, object]:
    values = frame["net_excess_return_5d"].to_numpy(float)
    turnover = frame["turnover_l1"].to_numpy(float)
    n = len(values)
    std = float(values.std(ddof=1)) if n > 1 else np.nan
    mean = float(values.mean()) if n else np.nan
    annualized_geo = float(np.prod(1.0 + values) ** (52.0 / n) - 1.0) if n and np.all(1.0 + values > 0) else np.nan
    downside = values[values < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sharpe = float(mean / std * np.sqrt(52)) if np.isfinite(std) and std > 0 else np.nan
    sortino = float(mean / downside_std * np.sqrt(52)) if np.isfinite(downside_std) and downside_std > 0 else np.nan
    drawdown = max_drawdown(values)
    cvar_cutoff = float(np.quantile(values, 0.05)) if n else np.nan
    cvar = float(values[values <= cvar_cutoff].mean()) if n else np.nan
    rolling = pd.Series(values).rolling(26).mean() / pd.Series(values).rolling(26).std(ddof=1) * np.sqrt(52)
    return {
        "strategy": label,
        "period": period,
        "evaluation_dates": n,
        "mean_net_excess_5d": mean,
        "net_sharpe_annualized": sharpe,
        "annualized_geometric_excess_return": annualized_geo,
        "annualized_volatility": float(std * np.sqrt(52)) if np.isfinite(std) else np.nan,
        "mean_turnover": float(turnover.mean()) if n else np.nan,
        "max_drawdown": drawdown,
        "sortino_annualized": sortino,
        "calmar_approx": float(annualized_geo / abs(drawdown)) if np.isfinite(annualized_geo) and np.isfinite(drawdown) and drawdown < 0 else np.nan,
        "cvar_5pct_per_5d": cvar,
        "hit_rate": float(np.mean(values > 0)) if n else np.nan,
        "worst_5d_return": float(values.min()) if n else np.nan,
        "best_5d_return": float(values.max()) if n else np.nan,
        "min_rolling_26w_sharpe": float(rolling.dropna().min()) if rolling.notna().any() else np.nan,
        "mean_rolling_26w_sharpe": float(rolling.dropna().mean()) if rolling.notna().any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-run", type=Path, required=True)
    parser.add_argument("--benchmark-run", type=Path, required=True)
    parser.add_argument("--ablation-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v3_extension_p1/economic_metrics"))
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("PTCST-CA-MVO", args.method_run, True),
        ("EW", args.benchmark_run, False),
        ("EW-BH", args.benchmark_run, False),
        ("MinVar", args.benchmark_run, False),
        ("Ridge-MVO", args.benchmark_run, False),
        ("XGB-CA-MVO", args.benchmark_run, False),
        ("PTCST-MVO", args.ablation_run, False),
        ("PTCST-Top20", args.ablation_run, False),
    ]
    rows = []
    wealth_rows = []
    for label, run_dir, method_run in specs:
        frame = load_strategy(run_dir, label, method_run=method_run)
        rows.append(summarize(frame, label, "2024-2025"))
        for period, group in frame.assign(year=frame["date"].dt.year).groupby("year"):
            rows.append(summarize(group, label, str(int(period))))
        wealth = frame.copy()
        wealth["strategy"] = label
        wealth["wealth_index"] = (1.0 + wealth["net_excess_return_5d"]).cumprod()
        wealth_rows.append(wealth)
    summary = pd.DataFrame(rows).sort_values(["period", "strategy"])
    summary.to_csv(args.run_dir / "economic_metrics.csv", index=False)
    pd.concat(wealth_rows, ignore_index=True).to_parquet(args.run_dir / "portfolio_wealth.parquet", index=False)
    summary.loc[summary["period"].isin(["2024", "2025"])].to_csv(args.run_dir / "year_split_portfolio_summary.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategies": [label for label, _, _ in specs],
        "periods": ["2024-2025", "2024", "2025"],
        "artifacts": ["economic_metrics.csv", "year_split_portfolio_summary.csv", "portfolio_wealth.parquet"],
        "annualization": "52 five-session periods per year; geometric return computed from observed evaluation sequence",
        "status": "COMPLETED_FROM_VERIFIED_LOCAL_RUNS",
    }
    (args.run_dir / "economic_metrics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
