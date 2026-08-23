"""Align legacy ASEAN/V1 forecasts to the corrected V2.1 portfolio engine.

This adapter does not retrain or alter the legacy forecast. It only maps a
forecast table keyed by country/date/ric to the V2 variable-N universe and
serializes the NPZ contract consumed by ``run_asean_v2_daily_backtest.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pick(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"Could not find {label}; tried {candidates}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--forecast-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="development")
    p.add_argument("--prediction-column", default=None)
    p.add_argument("--prediction-unit", choices=("bps", "decimal"), default="bps")
    args = p.parse_args()

    weekly_path = args.data_root / "model_ready" / "weekly_features_targets_v2"
    if not weekly_path.exists():
        weekly_path = weekly_path.with_suffix(".parquet")
    weekly = pd.read_parquet(weekly_path)
    weekly["date"] = pd.to_datetime(weekly["date"]).dt.normalize()
    weekly = weekly.loc[weekly["split_v2"].eq(args.split) & weekly["model_eligible_v2"]].copy()
    weekly = weekly.sort_values(["country", "date", "market_cap_rank"])

    forecasts = pd.read_parquet(args.forecast_file)
    date_col = pick(forecasts, ["signal_date", "date"], "forecast date")
    country_col = pick(forecasts, ["country", "market"], "country")
    ric_col = pick(forecasts, ["ric", "instrument", "ticker"], "RIC")
    pred_col = args.prediction_column or pick(
        forecasts,
        ["prediction", "prediction_excess_return_5d_bps", "forecast", "score", "raw_score"],
        "prediction",
    )
    forecasts = forecasts.rename(
        columns={date_col: "date", country_col: "country", ric_col: "ric", pred_col: "prediction"}
    )
    forecasts["date"] = pd.to_datetime(forecasts["date"]).dt.normalize()
    forecasts["country"] = forecasts["country"].astype(str)
    forecasts["ric"] = forecasts["ric"].astype(str)
    if "split" in forecasts.columns:
        forecasts = forecasts.loc[forecasts["split"].eq(args.split)]
    forecasts = forecasts.drop_duplicates(["country", "date", "ric"], keep="last")
    prediction_map = forecasts.set_index(["country", "date", "ric"])["prediction"].to_dict()

    dates: list[pd.Timestamp] = []
    countries: list[str] = []
    rics_rows: list[np.ndarray] = []
    alpha_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    target_mask_rows: list[np.ndarray] = []
    asset_mask_rows: list[np.ndarray] = []
    for (country, date), group in weekly.groupby(["country", "date"], sort=True):
        group = group.head(100)
        rics = np.full(100, "", dtype="U64")
        alpha = np.zeros(100, dtype="float64")
        target = np.zeros(100, dtype="float64")
        target_mask = np.zeros(100, dtype=bool)
        asset_mask = np.zeros(100, dtype=bool)
        for index, row in enumerate(group.itertuples(index=False)):
            ric = str(row.ric)
            value = prediction_map.get((str(country), pd.Timestamp(date), ric))
            if value is None or not np.isfinite(float(value)):
                continue
            rics[index] = ric
            alpha[index] = float(value) / 10000.0 if args.prediction_unit == "bps" else float(value)
            target[index] = float(row.target_cs_excess_return_5d_bps_v2) if pd.notna(row.target_cs_excess_return_5d_bps_v2) else 0.0
            target_mask[index] = bool(row.target_available_v2)
            asset_mask[index] = True
        if asset_mask.sum() < 3:
            continue
        dates.append(pd.Timestamp(date))
        countries.append(str(country))
        rics_rows.append(rics)
        alpha_rows.append(alpha)
        target_rows.append(target)
        target_mask_rows.append(target_mask)
        asset_mask_rows.append(asset_mask)

    if not dates:
        raise RuntimeError("No legacy forecasts could be aligned to the selected V2 split/universe.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dates=np.asarray(dates, dtype="datetime64[ns]"),
        countries=np.asarray(countries),
        rics=np.asarray(rics_rows),
        raw_score=np.asarray(alpha_rows, dtype="float64"),
        calibrated_alpha_decimal=np.asarray(alpha_rows, dtype="float64"),
        target_bps=np.asarray(target_rows, dtype="float64"),
        target_mask=np.asarray(target_mask_rows, dtype=bool),
        asset_mask=np.asarray(asset_mask_rows, dtype=bool),
    )
    print(f"Aligned {len(dates)} dates to {args.output}")


if __name__ == "__main__":
    main()
