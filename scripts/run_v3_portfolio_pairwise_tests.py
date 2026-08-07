"""Paired block-bootstrap inference for portfolio strategy differences."""
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
        if "strategy" not in frame:
            raise ValueError(f"{run_dir}: strategy column is required")
        frame = frame.loc[frame["strategy"].astype(str).eq(strategy)].copy()
    if "evaluation_available" in frame:
        frame = frame.loc[frame["evaluation_available"].astype(bool)].copy()
    required = {"date", "net_excess_return_5d", "turnover_l1"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{run_dir}: missing columns {missing}")
    out = frame[["date", "net_excess_return_5d", "turnover_l1"]].copy()
    if out.duplicated("date").any():
        raise ValueError(f"{run_dir}/{strategy}: duplicate dates")
    return out.set_index("date").sort_index()


def sharpe(values: np.ndarray) -> float:
    return float(values.mean() / values.std(ddof=1) * np.sqrt(52)) if len(values) > 1 and values.std(ddof=1) > 0 else np.nan


def max_drawdown(values: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(values, dtype=float))
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)) if len(wealth) else np.nan


def metric_vector(frame: pd.DataFrame) -> dict[str, float]:
    returns = frame["net_excess_return_5d"].to_numpy(float)
    turnover = frame["turnover_l1"].to_numpy(float)
    return {
        "mean_net_excess_5d": float(returns.mean()),
        "net_sharpe_annualized": sharpe(returns),
        "mean_turnover": float(turnover.mean()),
        "max_drawdown": max_drawdown(returns),
    }


def block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    indices: list[int] = []
    while len(indices) < n:
        start = int(rng.integers(0, n))
        indices.extend(((start + np.arange(block_length)) % n).tolist())
    return np.asarray(indices[:n], dtype=int)


def bootstrap_pair(left: pd.DataFrame, right: pd.DataFrame, seed: int, repetitions: int) -> tuple[dict[str, object], pd.DataFrame]:
    joined = left.join(right, how="inner", lsuffix="_left", rsuffix="_right")
    if joined.empty:
        raise ValueError("No overlapping evaluation dates")
    joined = joined.sort_index()
    observed_left = metric_vector(joined.rename(columns={"net_excess_return_5d_left": "net_excess_return_5d", "turnover_l1_left": "turnover_l1"}))
    observed_right = metric_vector(joined.rename(columns={"net_excess_return_5d_right": "net_excess_return_5d", "turnover_l1_right": "turnover_l1"}))
    observed_diff = {key: float(observed_left[key] - observed_right[key]) for key in observed_left}
    n = len(joined)
    block_length = min(12, max(4, n // 8))
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in observed_diff}
    for _ in range(repetitions):
        idx = block_indices(n, block_length, rng)
        l = joined.iloc[idx]
        left_metrics = metric_vector(l.rename(columns={"net_excess_return_5d_left": "net_excess_return_5d", "turnover_l1_left": "turnover_l1"}))
        right_metrics = metric_vector(l.rename(columns={"net_excess_return_5d_right": "net_excess_return_5d", "turnover_l1_right": "turnover_l1"}))
        for key in samples:
            samples[key].append(float(left_metrics[key] - right_metrics[key]))
    summary: dict[str, object] = {"n_dates": n, "block_length": block_length, "repetitions": repetitions, "observed_difference": observed_diff, "metrics": {}}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        metric_result: dict[str, object] = {
            "difference": observed_diff[key],
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }
        if key == "mean_net_excess_5d":
            null = np.asarray([
                float(np.mean((joined["net_excess_return_5d_left"] - joined["net_excess_return_5d_right"]).to_numpy(float) * rng.choice(np.array([-1.0, 1.0]), size=n)))
                for _ in range(repetitions)
            ])
            metric_result["sign_flip_p_value"] = float(np.mean(np.abs(null) >= abs(observed_diff[key])))
        else:
            metric_result["sign_flip_p_value"] = None
        summary["metrics"][key] = metric_result
    by_date = pd.DataFrame({
        "date": joined.index,
        "net_return_difference": joined["net_excess_return_5d_left"].to_numpy(float) - joined["net_excess_return_5d_right"].to_numpy(float),
        "turnover_difference": joined["turnover_l1_left"].to_numpy(float) - joined["turnover_l1_right"].to_numpy(float),
    })
    return summary, by_date


def holm_adjust(values: list[float | None]) -> list[float | None]:
    valid = [(i, float(value)) for i, value in enumerate(values) if value is not None and np.isfinite(value)]
    adjusted: list[float | None] = [None] * len(values)
    previous = 0.0
    for rank, (index, value) in enumerate(sorted(valid, key=lambda pair: pair[1])):
        previous = max(previous, min(1.0, (len(valid) - rank) * value))
        adjusted[index] = previous
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-run", type=Path, required=True, help="PTCST-CA-MVO run")
    parser.add_argument("--benchmark-run", type=Path, required=True)
    parser.add_argument("--ablation-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v3_extension_p1/portfolio_inference"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    strategies = {
        "PTCST-CA-MVO": load_strategy(args.method_run, "PTCST-CA-MVO", method_run=True),
        "EW": load_strategy(args.benchmark_run, "EW"),
        "EW-BH": load_strategy(args.benchmark_run, "EW-BH"),
        "MinVar": load_strategy(args.benchmark_run, "MinVar"),
        "Ridge-MVO": load_strategy(args.benchmark_run, "Ridge-MVO"),
        "XGB-CA-MVO": load_strategy(args.benchmark_run, "XGB-CA-MVO"),
        "PTCST-MVO": load_strategy(args.ablation_run, "PTCST-MVO"),
        "PTCST-Top20": load_strategy(args.ablation_run, "PTCST-Top20"),
    }
    pairs = [
        ("PTCST-CA-MVO", "EW"),
        ("PTCST-CA-MVO", "EW-BH"),
        ("PTCST-CA-MVO", "MinVar"),
        ("PTCST-CA-MVO", "Ridge-MVO"),
        ("PTCST-CA-MVO", "XGB-CA-MVO"),
        ("PTCST-MVO", "PTCST-Top20"),
        ("PTCST-CA-MVO", "PTCST-MVO"),
    ]
    results = []
    daily = []
    for left_name, right_name in pairs:
        summary, by_date = bootstrap_pair(strategies[left_name], strategies[right_name], args.seed, args.repetitions)
        summary["left"] = left_name
        summary["right"] = right_name
        results.append(summary)
        by_date.insert(0, "left", left_name)
        by_date.insert(1, "right", right_name)
        daily.append(by_date)
    p_values = [r["metrics"]["mean_net_excess_5d"]["sign_flip_p_value"] for r in results]
    adjusted = holm_adjust(p_values)
    for result, adjusted_p in zip(results, adjusted):
        result["metrics"]["mean_net_excess_5d"]["holm_adjusted_p_value"] = adjusted_p
    pd.concat(daily, ignore_index=True).to_parquet(args.run_dir / "portfolio_pairwise_by_date.parquet", index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_run": str(args.method_run),
        "benchmark_run": str(args.benchmark_run),
        "ablation_run": str(args.ablation_run),
        "inference_unit": "evaluation date",
        "bootstrap": "circular paired blocks",
        "seed": args.seed,
        "repetitions": args.repetitions,
        "multiple_comparison_correction": "Holm on mean-return sign-flip p-values",
        "comparisons": results,
    }
    (args.run_dir / "portfolio_pairwise_tests.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
