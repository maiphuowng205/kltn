"""Audit historical end-of-day BID/ASK coverage and implied spread costs.

This is deliberately a quote-spread audit, not a tick-level implementation-
shortfall estimate.  The V3 portfolio policy uses a fixed one-way cost of
0.001 (10 bps); this report compares that assumption with half of the
observed end-of-day quoted spread on each country's execution date.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def audit_country(data_root: Path, out_dir: Path, country: str) -> dict[str, object]:
    weekly = pd.read_parquet(data_root / "curated" / "universe_weekly.parquet")
    weekly["date"] = pd.to_datetime(weekly["date"]).dt.normalize()
    weekly = weekly.sort_values(["date", "market_cap_rank"]).groupby("date", sort=True).head(100)
    daily = pd.read_parquet(
        data_root / "curated" / "daily_panel.parquet",
        columns=["date", "ric", "execution_date", "bid", "ask", "quoted_spread_bps", "quote_observed"],
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["execution_date"] = pd.to_datetime(daily["execution_date"]).dt.normalize()
    # Signal date -> next tradable execution date.  The value is the same for
    # all rows on a signal date, so taking the first is deterministic.
    exec_map = daily.loc[daily["date"].isin(weekly["date"]), ["date", "execution_date"]].drop_duplicates("date")
    weekly = weekly.merge(exec_map, on="date", how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for signal_date, group in weekly.groupby("date", sort=True):
        execution_date = group["execution_date"].iloc[0]
        rics = group["ric"].astype(str).tolist()
        q = daily.loc[(daily["date"].eq(execution_date)) & (daily["ric"].astype(str).isin(rics))].copy()
        spread = pd.to_numeric(q["quoted_spread_bps"], errors="coerce")
        valid = spread.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "country": country,
                "signal_date": signal_date,
                "execution_date": execution_date,
                "universe_assets": len(rics),
                "quote_assets": int(valid.size),
                "quote_coverage": float(valid.size / len(rics)) if rics else np.nan,
                "median_quoted_spread_bps": float(valid.median()) if len(valid) else np.nan,
                "p90_quoted_spread_bps": float(valid.quantile(0.90)) if len(valid) else np.nan,
                "median_half_spread_bps": float(valid.median() / 2.0) if len(valid) else np.nan,
                "assumed_one_way_cost_bps": 10.0,
                "median_half_spread_above_assumption": bool(valid.median() / 2.0 > 10.0) if len(valid) else None,
            }
        )
    by_date = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_date.to_parquet(out_dir / "quote_cost_by_date.parquet", index=False)
    coverage = by_date["quote_coverage"]
    spread = by_date["median_half_spread_bps"]
    summary = {
        "country": country,
        "signal_dates": int(len(by_date)),
        "mean_quote_coverage": float(coverage.mean()),
        "min_quote_coverage": float(coverage.min()),
        "median_half_spread_bps": float(spread.median()),
        "p90_median_half_spread_bps": float(spread.quantile(0.90)),
        "fraction_dates_half_spread_above_10bps": float((spread > 10.0).mean()),
        "note": "BID/ASK are end-of-day fields; this does not measure tick-level execution or implementation shortfall.",
    }
    (out_dir / "quote_cost_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country-root", type=Path, required=True, help="artifacts/asean_v1_country_runs")
    parser.add_argument("--countries", nargs="+", default=["indonesia", "malaysia", "philippines", "singapore", "thailand"])
    args = parser.parse_args()
    summaries = []
    for country in args.countries:
        root = args.country_root / country
        summaries.append(audit_country(root / "data" / "lseg_v3", root / "runs" / "v3_quote_cost_audit", country))
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "assumed_one_way_cost_bps": 10.0, "countries": summaries}
    out = args.country_root / "asean_quote_cost_audit_summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
