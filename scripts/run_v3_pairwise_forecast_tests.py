"""Pairwise date-level forecast-loss inference for each baseline model.

Unlike the original aggregate test, this runner filters the baseline forecast
file by model before aligning it with PTCST. It writes one comparison per
baseline model and applies Holm correction separately to DM/HLN and bootstrap
p-values.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_forecasts(path: Path, label: str, model: str | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "split" in frame:
        frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if model is not None:
        if "model" not in frame:
            raise ValueError(f"{label}: model column is required to filter {model}")
        frame = frame.loc[frame["model"].astype(str).eq(model)].copy()
    prediction = "prediction" if "prediction" in frame else "prediction_excess_return_5d_bps"
    target = "target" if "target" in frame else "target_excess_return_5d_bps"
    required = {"date", "ric", prediction, target, "target_available"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    out = frame[["date", "ric", prediction, target, "target_available"]].rename(
        columns={prediction: "prediction_bps", target: "target_bps"}
    )
    out["ric"] = out["ric"].astype(str)
    if out.duplicated(["date", "ric"]).any():
        duplicate_count = int(out.duplicated(["date", "ric"]).sum())
        raise ValueError(f"{label}: {duplicate_count} duplicate date/RIC rows")
    return out


def dm_hln(difference: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return {"n_dates": n, "hac_lag": None, "mean_loss_difference": None, "t_stat": None, "p_value": None}
    lag = max(1, min(12, int(np.floor(n ** (1 / 3)))))
    centered = values - values.mean()
    long_var = float(np.dot(centered, centered) / n)
    for k in range(1, min(lag, n - 1) + 1):
        long_var += 2 * (1 - k / (lag + 1)) * float(np.dot(centered[k:], centered[:-k]) / n)
    standard_error = np.sqrt(max(long_var, 1e-16) / n)
    statistic = float(values.mean() / standard_error)
    hln = np.sqrt((n + 1 - 2 + 1) / n)
    corrected = float(statistic * hln)
    p_value = float(2 * stats.t.sf(abs(corrected), df=max(n - 1, 1)))
    return {
        "n_dates": n,
        "hac_lag": lag,
        "mean_loss_difference": float(values.mean()),
        "t_stat": corrected,
        "p_value": p_value,
    }


def block_bootstrap(difference: np.ndarray, seed: int, repetitions: int) -> dict[str, float | int]:
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 4:
        return {"n_dates": n, "repetitions": repetitions, "block_length": 0, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed)
    block_length = min(12, max(4, n // 8))
    samples: list[float] = []
    for _ in range(repetitions):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend(((start + np.arange(block_length)) % n).tolist())
        samples.append(float(values[np.asarray(indices[:n])].mean()))
    bootstrap = np.asarray(samples)
    null = np.asarray([
        float(np.mean(values * rng.choice(np.array([-1.0, 1.0]), size=n)))
        for _ in range(repetitions)
    ])
    return {
        "n_dates": n,
        "repetitions": repetitions,
        "block_length": block_length,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "p_value": float(np.mean(np.abs(null) >= abs(values.mean()))),
    }


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    valid = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    adjusted: list[float | None] = [None] * len(p_values)
    previous = 0.0
    for rank, (index, p_value) in enumerate(sorted(valid, key=lambda pair: pair[1])):
        corrected = min(1.0, (len(valid) - rank) * p_value)
        previous = max(previous, corrected)
        adjusted[index] = previous
    return adjusted


def compare_model(baseline: pd.DataFrame, ptcst: pd.DataFrame, model: str, seed: int, repetitions: int) -> tuple[dict[str, object], pd.DataFrame]:
    left = baseline.loc[baseline["model"].eq(model)].copy()
    if left.empty:
        raise ValueError(f"Baseline model not found: {model}")
    left = left.rename(columns={"prediction_bps": "prediction_left", "target_bps": "target_left", "target_available": "available_left"})
    right = ptcst.rename(columns={"prediction_bps": "prediction_right", "target_bps": "target_right", "target_available": "available_right"})
    merged = left.merge(right, on=["date", "ric"], how="inner", validate="one_to_one")
    available = merged["available_left"].astype(bool) & merged["available_right"].astype(bool)
    finite = np.isfinite(merged[["prediction_left", "prediction_right", "target_left", "target_right"]]).all(axis=1)
    merged = merged.loc[available & finite].copy()
    if merged.empty:
        raise ValueError(f"No complete aligned observations for {model}")
    if not np.allclose(merged["target_left"].to_numpy(float), merged["target_right"].to_numpy(float), atol=1e-6):
        raise ValueError(f"Target mismatch between baseline and PTCST for {model}")
    merged["loss_difference"] = (merged["prediction_left"] - merged["target_left"]) ** 2 - (merged["prediction_right"] - merged["target_right"]) ** 2
    by_date = merged.groupby("date", as_index=False).agg(
        mean_loss_difference=("loss_difference", "mean"),
        paired_assets=("ric", "count"),
    )
    dm = dm_hln(by_date["mean_loss_difference"].to_numpy())
    bootstrap = block_bootstrap(by_date["mean_loss_difference"].to_numpy(), seed, repetitions)
    result = {
        "model": model,
        "n_observations": int(len(merged)),
        "n_dates": int(len(by_date)),
        "mean_paired_assets_per_date": float(by_date["paired_assets"].mean()),
        "dm_hln": dm,
        "block_bootstrap": bootstrap,
    }
    by_date.insert(0, "model", model)
    return result, by_date


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--ptcst-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v3_pairwise_forecast_tests"))
    parser.add_argument("--models", nargs="+", default=["zero", "historical_mean", "ridge", "xgb"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = []
    for model in args.models:
        frame = load_forecasts(args.baseline_run / "forecasts.parquet", f"baseline/{model}", model=model)
        frame["model"] = model
        baseline_rows.append(frame)
    baseline = pd.concat(baseline_rows, ignore_index=True)
    ptcst = load_forecasts(args.ptcst_run / "forecasts.parquet", "ptcst")
    baseline_test = baseline.assign(split="test")
    # The input files contain validation and test rows. Filter by the date
    # range that intersects the PTCST test rows after loading the same schema.
    ptcst_dates = set(ptcst["date"])
    baseline_test = baseline_test.loc[baseline_test["date"].isin(ptcst_dates)].copy()
    results = []
    daily = []
    for model in args.models:
        result, by_date = compare_model(baseline_test, ptcst, model, args.seed, args.repetitions)
        results.append(result)
        daily.append(by_date)
    dm_adjusted = holm_adjust([r["dm_hln"]["p_value"] for r in results])
    bootstrap_adjusted = holm_adjust([r["block_bootstrap"]["p_value"] for r in results])
    for result, dm_p, bootstrap_p in zip(results, dm_adjusted, bootstrap_adjusted):
        result["dm_hln"]["holm_adjusted_p_value"] = dm_p
        result["block_bootstrap"]["holm_adjusted_p_value"] = bootstrap_p
    pd.concat(daily, ignore_index=True).to_parquet(args.run_dir / "forecast_pairwise_loss_by_date.parquet", index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": str(args.baseline_run),
        "ptcst_run": str(args.ptcst_run),
        "models": args.models,
        "inference_unit": "forecast date",
        "loss": "squared error in bps",
        "correction": "Holm correction across requested baseline models",
        "seed": args.seed,
        "repetitions": args.repetitions,
        "comparisons": results,
    }
    (args.run_dir / "forecast_pairwise_tests.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
