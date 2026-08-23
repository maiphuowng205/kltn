"""Download point-in-time monthly market capitalisation for the ASEAN universe.

Market-cap snapshots are stored separately from the daily extract because they
are used only for the lagged, point-in-time Top-100 universe.  A month-end
snapshot is made available to the model in the following month by the dataset
builder; this script only downloads the observations.
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
FIELDS = ["TR.CompanyMarketCap.Date", "TR.CompanyMarketCap"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=Path, default=ROOT / "artifacts/asean_preflight/historical_primary_universe.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/asean_v1/raw/market_cap_monthly")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--max-files", type=int, default=0)
    return p.parse_args()


def batches(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def normalise(raw: pd.DataFrame, country: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["ric", "snapshot_date", "market_cap_usd", "country"])
    df = raw.rename(
        columns={"Instrument": "ric", "Date": "snapshot_date", "Company Market Cap": "market_cap_usd"}
    ).copy()
    for source, target in {"TR.CompanyMarketCap.Date": "snapshot_date", "TR.CompanyMarketCap": "market_cap_usd"}.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})
    if not {"ric", "snapshot_date", "market_cap_usd"}.issubset(df.columns):
        return pd.DataFrame(columns=["ric", "snapshot_date", "market_cap_usd", "country"])
    df["ric"] = df["ric"].astype(str).str.strip()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.normalize()
    df["market_cap_usd"] = pd.to_numeric(df["market_cap_usd"], errors="coerce")
    df["country"] = country
    df = df.loc[df["ric"].ne("") & df["snapshot_date"].notna()].copy()
    return df[["ric", "snapshot_date", "market_cap_usd", "country"]].drop_duplicates(
        ["ric", "snapshot_date"], keep="last"
    )


def request_with_split(rics: list[str], country: str, start: str, end: str) -> tuple[pd.DataFrame, list[dict]]:
    errors: list[dict] = []
    try:
        raw = ld.get_data(
            rics,
            FIELDS,
            {"SDate": start, "EDate": end, "Frq": "M", "Curn": "USD"},
        )
        return normalise(raw, country), errors
    except Exception as exc:
        if len(rics) == 1:
            errors.append({"country": country, "ric": rics[0], "start": start, "end": end,
                           "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            return pd.DataFrame(), errors
        mid = len(rics) // 2
        time.sleep(1.0)
        left, le = request_with_split(rics[:mid], country, start, end)
        right, re = request_with_split(rics[mid:], country, start, end)
        return pd.concat([left, right], ignore_index=True), le + re


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
        start, end = f"{a.start_year}-01-01", f"{a.end_year}-12-31"
        for country in sorted(universe["Country"].dropna().unique()):
            rics = sorted(universe.loc[universe["Country"].eq(country), "RIC"].drop_duplicates().tolist())
            for batch_no, batch in enumerate(batches(rics, a.batch_size), start=1):
                out = a.output_dir / f"market_cap_{country.lower()}_{a.start_year}_{a.end_year}_batch_{batch_no:03d}.parquet"
                if out.exists():
                    files_written.append(out.name); print("SKIP", out.name, flush=True); continue
                if a.max_files and file_count >= a.max_files:
                    break
                print(f"FETCH {country} {a.start_year}-{a.end_year} batch={batch_no} n={len(batch)}", flush=True)
                frame, batch_errors = request_with_split(batch, country, start, end)
                errors.extend(batch_errors)
                frame.to_parquet(out, index=False)
                files_written.append(out.name); file_count += 1
                print("WROTE", out.name, "rows=", len(frame), "errors=", len(batch_errors), flush=True)
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
        "fields": FIELDS,
        "start_year": a.start_year, "end_year": a.end_year, "batch_size": a.batch_size,
        "files_written": len(files_written), "errors": len(errors),
        "currency": "USD conversion requested from Workspace",
        "errors_detail": errors[:500],
    }
    (a.output_dir / "market_cap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if errors:
        pd.DataFrame(errors).to_csv(a.output_dir / "market_cap_errors.csv", index=False)
    print(json.dumps({k: v for k, v in manifest.items() if k != "errors_detail"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
