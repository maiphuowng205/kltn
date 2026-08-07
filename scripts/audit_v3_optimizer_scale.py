"""Audit optimizer units and objective-term magnitudes for a stored run."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v3_method import ledoit_covariance, l1_turnover


def normalize_forecasts(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "split" in frame:
        frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    prediction = "prediction" if "prediction" in frame else "prediction_excess_return_5d_bps"
    if prediction not in frame:
        raise ValueError(f"Missing forecast column in {path}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ric"] = frame["ric"].astype(str)
    return frame[["date", "ric", prediction]].rename(columns={prediction: "prediction_bps"})


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/lseg_v3"))
    parser.add_argument("--method-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/v3_optimizer_scale_audit/optimizer_scale_diagnostic.json"))
    parser.add_argument("--date", type=str, default=None, help="Optional signal date; defaults to first complete test date")
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument("--cost", type=float, default=0.001)
    args = parser.parse_args()

    forecasts = normalize_forecasts(args.method_run / "forecasts.parquet")
    returns = pd.read_parquet(args.method_run / "portfolio_returns.parquet")
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    if "evaluation_available" in returns:
        complete = returns.loc[returns["evaluation_available"].astype(bool)]
    else:
        complete = returns
    if complete.empty:
        raise ValueError("No complete portfolio-return date available for audit")
    date = pd.Timestamp(args.date).normalize() if args.date else pd.Timestamp(complete["date"].iloc[0])

    weights = pd.read_parquet(args.method_run / "weights.parquet")
    weights["date"] = pd.to_datetime(weights["date"]).dt.normalize()
    weights["ric"] = weights["ric"].astype(str)
    current_weights = weights.loc[weights["date"].eq(date)].copy()
    if current_weights.empty:
        raise ValueError(f"No stored weights for audit date {date.date()}")
    required_weight_columns = {"ric", "weight", "w_pre"}
    missing = required_weight_columns.difference(current_weights.columns)
    if missing:
        raise ValueError(f"Weights missing columns: {sorted(missing)}")

    daily = pd.read_parquet(args.data_root / "curated" / "daily_panel.parquet")
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    rics = current_weights["ric"].tolist()
    forecast = forecasts.loc[forecasts["date"].eq(date)].set_index("ric").reindex(rics)
    if forecast["prediction_bps"].isna().any():
        missing_rics = forecast.index[forecast["prediction_bps"].isna()].tolist()
        raise ValueError(f"Forecast missing {len(missing_rics)} current-universe RICs")
    covariance, valid, risk_status = ledoit_covariance(daily, date, rics, window=252)
    mu_bps = forecast["prediction_bps"].to_numpy(dtype=float)
    mu_decimal = mu_bps / 10000.0
    target = current_weights["weight"].to_numpy(dtype=float)
    pre = current_weights["w_pre"].to_numpy(dtype=float)
    return_row = complete.loc[complete["date"].eq(date)]
    exited_weight = float(return_row["turnover_exited"].iloc[0]) if "turnover_exited" in return_row and not return_row.empty else 0.0
    turnover = l1_turnover(target, pre, exited_weight)
    sigma = np.asarray(covariance, dtype=float)
    expected_term = float(target @ mu_decimal)
    risk_term = float(args.risk_aversion / 2.0 * (target @ sigma @ target))
    cost_term = float(args.cost * turnover)
    objective = expected_term - risk_term - cost_term
    diag = np.diag(sigma)
    offdiag = sigma - np.diag(diag)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "method_run": str(args.method_run),
        "date": date.date().isoformat(),
        "units": {
            "prediction_input": "basis points",
            "mu_optimizer": "decimal return = prediction_bps / 10000",
            "covariance": "decimal daily-return squared",
            "cost": "decimal cost per L1 turnover",
        },
        "configuration": {"risk_aversion": args.risk_aversion, "cost": args.cost, "covariance_window_sessions": 252},
        "universe": {"assets": len(rics), "valid_covariance_assets": int(valid.sum()), "risk_status": risk_status},
        "prediction_bps": describe(mu_bps),
        "mu_decimal": describe(mu_decimal),
        "covariance_diagonal": describe(diag),
        "covariance_offdiagonal_abs": describe(np.abs(offdiag[offdiag != 0])),
        "weights": {"sum": float(target.sum()), "max": float(target.max()), "min": float(target.min()), "pre_sum": float(pre.sum())},
        "turnover": {"current_l1": float(np.abs(target - pre).sum()), "exited_weight": exited_weight, "total_l1": turnover},
        "objective_terms": {
            "expected_return_term": expected_term,
            "risk_term": risk_term,
            "turnover_penalty_term": cost_term,
            "objective_value": objective,
            "absolute_term_sum": float(abs(expected_term) + abs(risk_term) + abs(cost_term)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
