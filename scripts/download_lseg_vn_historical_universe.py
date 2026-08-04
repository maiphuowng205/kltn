"""Retrieve point-in-time HOSE/HNX constituent snapshots from LSEG.

The current Vnstock reference list is retained untouched.  This script creates
a separate LSEG V3 raw layer, including instruments that appeared historically
but may be absent from the current snapshot.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lseg.data as ld
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INDEXES = {"HOSE": ".VNI", "HNX": ".HNXI"}
FIELDS = ["TR.RIC", "TR.CommonName", "TR.CompanyMarketCap", "TR.TRBCEconomicSector", "TR.ISIN"]


def snapshot_dates(calendar_path: Path) -> list[pd.Timestamp]:
    calendar = pd.read_parquet(calendar_path)[["time"]].copy()
    calendar["time"] = pd.to_datetime(calendar["time"]).dt.normalize()
    return calendar.groupby(calendar["time"].dt.to_period("M"))["time"].max().tolist()


def retrieve(index_ric: str, date: pd.Timestamp) -> pd.DataFrame:
    text_date = date.strftime("%Y-%m-%d")
    chain = f"0#{index_ric}({date.strftime('%Y%m%d')})"
    df = ld.get_data(
        universe=[chain],
        fields=FIELDS,
        parameters={"SDate": text_date, "EDate": text_date, "Frq": "D", "Curn": "VND"},
    )
    df = df.rename(
        columns={
            "Instrument": "instrument",
            "RIC": "ric",
            "Company Common Name": "company_name",
            "Company Market Cap": "market_cap_vnd",
            "TRBC Economic Sector Name": "trbc_sector",
            "ISIN": "isin",
        }
    )
    df["snapshot_date"] = date
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar", type=Path, default=ROOT / "data/processed/market_calendar.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/lseg_v3/raw/universe_monthly")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dates = snapshot_dates(args.calendar)
    frames: list[pd.DataFrame] = []
    try:
        ld.open_session(name="desktop.workspace")
        for date in dates:
            for exchange, index_ric in INDEXES.items():
                output = args.output_dir / f"{exchange}_{date:%Y-%m-%d}.parquet"
                if output.exists():
                    frame = pd.read_parquet(output)
                else:
                    frame = retrieve(index_ric, date)
                    frame["exchange"] = exchange
                    frame["source_index_ric"] = index_ric
                    frame["extract_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                    frame.to_parquet(output, index=False)
                frames.append(frame)
                print(f"{exchange} {date:%Y-%m-%d}: {len(frame)}")
    finally:
        try:
            ld.close_session()
        except Exception:
            pass

    combined = pd.concat(frames, ignore_index=True)
    combined_path = args.output_dir.parent / "historical_universe_monthly.parquet"
    combined.to_parquet(combined_path, index=False)
    ledger = (
        combined.groupby(["exchange", "ric"], dropna=False)
        .agg(
            first_snapshot=("snapshot_date", "min"),
            last_snapshot=("snapshot_date", "max"),
            months_observed=("snapshot_date", "nunique"),
            company_name=("company_name", "last"),
            isin=("isin", "last"),
        )
        .reset_index()
    )
    ledger_path = args.output_dir.parent / "security_presence_ledger_monthly.parquet"
    ledger.to_parquet(ledger_path, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "LSEG Workspace Desktop Session",
        "indices": INDEXES,
        "snapshot_count": len(dates),
        "rows": len(combined),
        "unique_exchange_ric_pairs": len(ledger),
        "limitation": "First/last monthly snapshot is a membership-presence ledger, not an official delisting-reason/event ledger.",
    }
    (args.output_dir.parent / "historical_universe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE {combined_path} rows={len(combined)}")


if __name__ == "__main__":
    main()
