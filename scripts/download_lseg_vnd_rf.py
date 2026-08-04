"""Create the versioned VND cash-proxy input required by Dataset V3.

Source: LSEG RIC VND1MD=, the composite wholesale VND one-month deposit
mid-rate. This is a documented cash proxy, not a claim that it is a sovereign
risk-free security. Its annual percentage quote is converted to a daily simple
return under an Actual/365 convention.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lseg.data as ld
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RIC = "VND1MD="
FIELD = "MID_PRICE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar", type=Path, default=ROOT / "data/processed/market_calendar.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/risk_free_daily.parquet")
    args = parser.parse_args()

    calendar = pd.read_parquet(args.calendar)[["time"]].copy()
    calendar["date"] = pd.to_datetime(calendar["time"]).dt.normalize()
    start_date = calendar["date"].min()
    # Request a short buffer because the desktop historical endpoint can omit
    # the first requested business date for a deposit-rate series.
    request_start = (start_date - pd.Timedelta("7D")).strftime("%Y-%m-%d")
    start = start_date.strftime("%Y-%m-%d")
    end = calendar["date"].max().strftime("%Y-%m-%d")

    try:
        ld.open_session(name="desktop.workspace")
        history = ld.get_history([RIC], fields=[FIELD], start=request_start, end=end, interval="1D")
    finally:
        try:
            ld.close_session()
        except Exception:
            pass

    rate = history.iloc[:, 0].rename("annual_rate_pct").reset_index()
    rate.columns = ["date", "annual_rate_pct"]
    rate["date"] = pd.to_datetime(rate["date"]).dt.normalize()
    rate["annual_rate_pct"] = pd.to_numeric(rate["annual_rate_pct"], errors="coerce")
    rf = calendar[["date"]].merge(rate, on="date", how="left", validate="one_to_one")
    if rf["annual_rate_pct"].isna().any():
        missing = rf.loc[rf["annual_rate_pct"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"RF coverage is incomplete: {len(missing)} missing sessions, first={missing[:5]}")
    # The LSEG quote is an annual percentage rate.  The research data contract
    # needs a daily simple return, while the target builder compounds sessions.
    rf["rf_daily"] = (1.0 + rf["annual_rate_pct"] / 100.0) ** (1.0 / 365.0) - 1.0
    rf["source"] = "LSEG VND1MD= MID_PRICE"
    rf["yield_convention"] = "annual percentage rate converted to daily simple return: (1 + y/100)^(1/365)-1"
    rf["published_at"] = pd.NaT
    rf["published_at_note"] = "Not supplied by the historical endpoint; do not claim a point-in-time publication timestamp."

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rf.to_parquet(args.output, index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_ric": RIC,
        "source_field": FIELD,
        "definition": "VND one-month wholesale-deposit mid-rate cash proxy; not asserted to be a sovereign risk-free rate.",
        "yield_convention": "Actual/365 annual quote converted to daily simple return",
        "coverage": {"first": start, "last": end, "sessions": len(rf), "missing_sessions": int(rf["rf_daily"].isna().sum())},
        "publication_time_limitation": "Historical endpoint does not provide publication timestamps; this artifact supports V3 measurement but not a point-in-time RF publication claim.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"WROTE {args.output} rows={len(rf)}")


if __name__ == "__main__":
    main()
