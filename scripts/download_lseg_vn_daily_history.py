"""Download LSEG daily OHLCV and total-return data for historical Vietnamese RICs.

Outputs are kept in the LSEG V3 raw layer and never overwrite Vnstock/V2 data.
The historical RIC universe comes from monthly .VNI/.HNXI snapshots.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lseg.data as ld
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 200
START_YEAR = 2016  # supplies a 252-session risk lookback before the 2018 sample
END_YEAR = 2025


def retry(call, *args, **kwargs):
    for attempt in range(4):
        try:
            return call(*args, **kwargs)
        except Exception as exc:
            if "permission" in str(exc).lower() or attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def batches(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def rics_from_universe(path: Path) -> list[str]:
    universe = pd.read_parquet(path)
    rics = universe["ric"].dropna().astype(str).str.strip()
    return sorted(ric for ric in rics.unique() if ric and ric.lower() != "nan")


def normalise_prices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(
        columns={
            "Instrument": "ric",
            "Date": "date",
            "Price Open": "open",
            "Price High": "high",
            "Price Low": "low",
            "Price Close": "close",
            "Volume": "volume",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates(["ric", "date"], keep="last")


def normalise_returns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={"Instrument": "ric", "Calc Date": "date", "Total Return": "total_return_pct"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["total_return_pct"] = pd.to_numeric(frame["total_return_pct"], errors="coerce")
    return frame.drop_duplicates(["ric", "date"], keep="last")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, default=ROOT / "data/lseg_v3/raw/historical_universe_monthly.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/lseg_v3/raw/daily_history")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rics = rics_from_universe(args.universe)
    print(f"HISTORICAL_RICS={len(rics)}")

    written: list[Path] = []
    try:
        ld.open_session(name="desktop.workspace")
        for year in range(args.start_year, args.end_year + 1):
            for batch_no, batch in enumerate(batches(rics, BATCH_SIZE), start=1):
                output = args.output_dir / f"daily_{year}_batch_{batch_no:02d}.parquet"
                written.append(output)
                if output.exists():
                    print(f"SKIP {output.name}")
                    continue
                start, end = f"{year}-01-01", f"{year}-12-31"
                prices = retry(
                    ld.get_data,
                    batch,
                    ["TR.PriceClose.Date", "TR.PriceOpen", "TR.PriceHigh", "TR.PriceLow", "TR.PriceClose", "TR.Volume"],
                    {"SDate": start, "EDate": end, "Frq": "D", "Curn": "VND"},
                )
                returns = retry(
                    ld.get_data,
                    batch,
                    ["TR.TotalReturn.CalcDate", "TR.TotalReturn"],
                    {"SDate": start, "EDate": end, "Frq": "D"},
                )
                daily = normalise_prices(prices).merge(normalise_returns(returns), on=["ric", "date"], how="outer", validate="one_to_one")
                daily["extract_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                daily.to_parquet(output, index=False)
                print(f"WROTE {output.name} rows={len(daily)}")
    finally:
        try:
            ld.close_session()
        except Exception:
            pass

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "LSEG Workspace Desktop Session",
        "historical_ric_count": len(rics),
        "years": [args.start_year, args.end_year],
        "fields": ["open", "high", "low", "close", "volume", "total_return_pct"],
        "partition_count_expected": len(written),
        "partition_count_present": sum(path.exists() for path in written),
        "limitation": "RIC history and LSEG adjustment semantics must be audited around corporate actions before total-return claims are frozen.",
    }
    (args.output_dir / "daily_history_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
