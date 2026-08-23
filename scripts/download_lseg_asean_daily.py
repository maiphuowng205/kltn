"""Download ASEAN daily price/volume/quote fields from the open Workspace session.

This is an auditable, resumable raw extract. It deliberately uses TR.*
historical fields for BID/ASK rather than the snapshot BID/ASK fields: the
latter are not historical when requested through the data-grid endpoint.
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
PRICE_FIELDS = [
    "TR.PriceClose.Date",
    "TR.PriceOpen",
    "TR.PriceHigh",
    "TR.PriceLow",
    "TR.PriceClose",
    "TR.Volume",
    "TR.BidPrice",
    "TR.AskPrice",
]
RETURN_FIELDS = ["TR.TotalReturn.CalcDate", "TR.TotalReturn"]
CURRENCIES = {
    "Singapore": "SGD",
    "Malaysia": "MYR",
    "Indonesia": "IDR",
    "Thailand": "THB",
    "Philippines": "PHP",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=Path, default=ROOT / "artifacts/asean_preflight/historical_primary_universe.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/asean_v1/raw/daily_history")
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--max-files", type=int, default=0, help="Only for a resumable smoke test; 0 means all files.")
    p.add_argument("--countries", default="", help="Comma-separated country subset for parallel/resumable runs.")
    return p.parse_args()


def batches(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def normalise_prices(raw: pd.DataFrame, country: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    rename = {
        "Instrument": "ric",
        "Date": "date",
        "Price Open": "open",
        "Price High": "high",
        "Price Low": "low",
        "Price Close": "close",
        "Volume": "volume",
        "Bid Price": "bid",
        "Ask Price": "ask",
    }
    df = df.rename(columns=rename)
    # Some responses contain the qualified output names instead of display names.
    for source, target in {
        "TR.PriceClose.Date": "date",
        "TR.BidPrice": "bid",
        "TR.AskPrice": "ask",
    }.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})
    if "ric" not in df.columns:
        return pd.DataFrame()
    df["ric"] = df["ric"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume", "bid", "ask"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["country"] = country
    df["currency"] = CURRENCIES[country]
    df["extract_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    df = df.loc[df["ric"].ne("") & df["date"].notna()].copy()
    return df.drop_duplicates(["ric", "date"], keep="last")


def normalise_returns(raw: pd.DataFrame, country: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["ric", "date", "total_return_pct"])
    df = raw.rename(columns={"Instrument": "ric", "Calc Date": "date", "Total Return": "total_return_pct"}).copy()
    if "TR.TotalReturn.CalcDate" in df.columns:
        df = df.rename(columns={"TR.TotalReturn.CalcDate": "date"})
    if "TR.TotalReturn" in df.columns:
        df = df.rename(columns={"TR.TotalReturn": "total_return_pct"})
    if "ric" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=["ric", "date", "total_return_pct"])
    df["ric"] = df["ric"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["total_return_pct"] = pd.to_numeric(df.get("total_return_pct"), errors="coerce")
    df = df.loc[df["ric"].ne("") & df["date"].notna(), ["ric", "date", "total_return_pct"]]
    return df.drop_duplicates(["ric", "date"], keep="last")


def request_with_split(rics: list[str], country: str, start: str, end: str, currency: str) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch a batch; split recursively if one bad instrument poisons it."""
    errors: list[dict] = []
    try:
        # Keep price and total-return requests separate.  Workspace's data-grid
        # response does not preserve row alignment when fields expose different
        # date columns (``Date`` versus ``Calc Date``); a combined request can
        # therefore attach a return from one day to the price of another day.
        # We normalise each response independently and merge on RIC/date below.
        raw_prices = ld.get_data(
            rics,
            PRICE_FIELDS,
            {"SDate": start, "EDate": end, "Frq": "D", "Curn": currency},
        )
        raw_returns = ld.get_data(
            rics,
            RETURN_FIELDS,
            {"SDate": start, "EDate": end, "Frq": "D"},
        )
        prices = normalise_prices(raw_prices, country)
        returns = normalise_returns(raw_returns, country)
        if prices.empty:
            return prices, errors
        frame = prices.merge(returns, on=["ric", "date"], how="left", validate="one_to_one")
        return frame, errors
    except Exception as exc:
        if len(rics) == 1:
            errors.append({"country": country, "ric": rics[0], "start": start, "end": end,
                           "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            return pd.DataFrame(), errors
        mid = len(rics) // 2
        time.sleep(1.0)
        left, left_errors = request_with_split(rics[:mid], country, start, end, currency)
        right, right_errors = request_with_split(rics[mid:], country, start, end, currency)
        return pd.concat([left, right], ignore_index=True), left_errors + right_errors


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(a.universe)
    universe["RIC"] = universe["RIC"].astype(str).str.strip()
    universe = universe.loc[universe["RIC"].ne("") & universe["RIC"].ne("nan")].copy()
    files_written: list[str] = []
    errors: list[dict] = []
    file_count = 0
    try:
        ld.open_session(name="desktop.workspace")
        selected = [c.strip() for c in a.countries.split(",") if c.strip()] if a.countries else list(CURRENCIES)
        unknown = sorted(set(selected) - set(CURRENCIES))
        if unknown:
            raise ValueError(f"Unknown countries: {unknown}")
        for country in selected:
            currency = CURRENCIES[country]
            rics = sorted(universe.loc[universe["Country"].eq(country), "RIC"].drop_duplicates().tolist())
            for year in range(a.start_year, a.end_year + 1):
                for batch_no, batch in enumerate(batches(rics, a.batch_size), start=1):
                    out = a.output_dir / f"daily_{country.lower()}_{year}_batch_{batch_no:03d}.parquet"
                    if out.exists():
                        files_written.append(out.name)
                        print("SKIP", out.name, flush=True)
                        continue
                    if a.max_files and file_count >= a.max_files:
                        break
                    print(f"FETCH {country} {year} batch={batch_no} n={len(batch)}", flush=True)
                    start, end = f"{year}-01-01", f"{year}-12-31"
                    frame, batch_errors = request_with_split(batch, country, start, end, currency)
                    errors.extend(batch_errors)
                    frame.to_parquet(out, index=False)
                    files_written.append(out.name)
                    file_count += 1
                    print("WROTE", out.name, "rows=", len(frame), "errors=", len(batch_errors), flush=True)
                if a.max_files and file_count >= a.max_files:
                    break
            if a.max_files and file_count >= a.max_files:
                break
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "LSEG Workspace DesktopSession",
        "countries": list(CURRENCIES),
        "price_fields": PRICE_FIELDS,
        "return_fields": RETURN_FIELDS,
        "start_year": a.start_year,
        "end_year": a.end_year,
        "batch_size": a.batch_size,
        "files_written": len(files_written),
        "errors": len(errors),
        "quote_field_note": "TR.BidPrice/TR.AskPrice are historical data-grid fields; snapshot BID/ASK are not used.",
        "request_note": "Price and total-return fields are fetched in separate requests and merged on RIC/date because the Workspace response has independent date columns.",
        "errors_detail": errors[:500],
    }
    (a.output_dir / "daily_history_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if errors:
        pd.DataFrame(errors).to_csv(a.output_dir / "daily_history_errors.csv", index=False)
    print(json.dumps({k: v for k, v in manifest.items() if k != "errors_detail"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
