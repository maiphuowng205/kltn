"""Download daily one-month cash-rate proxies for the five ASEAN markets."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lseg.data as ld
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RF_RICS = {
    "Singapore": ("SGD1MD=", False),
    "Malaysia": ("MYR1MD=", False),
    "Indonesia": ("IDR1MD=", False),
    "Thailand": ("THB1MD=", False),
    # No PHP 1-month deposit RIC was resolvable in this Workspace.  The
    # Philippines mid-market 1-month proxy is retained and flagged explicitly.
    "Philippines": ("PHP1MID=PHR", True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/asean_v1/raw/risk_free_daily.parquet")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2025-12-31")
    return p.parse_args()


def main() -> None:
    a = parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    try:
        ld.open_session(name="desktop.workspace")
        for country, (ric, is_proxy) in RF_RICS.items():
            try:
                raw = ld.get_history(universe=ric, fields=["MID_PRICE"], interval="1D", start=a.start, end=a.end)
                if raw is None or raw.empty:
                    errors.append({"country": country, "ric": ric, "error": "empty response"}); continue
                x = raw.reset_index()
                x.columns = ["date", "rate_annual_pct"]
                x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
                x["rate_annual_pct"] = pd.to_numeric(x["rate_annual_pct"], errors="coerce")
                x["rf_daily"] = (1.0 + x["rate_annual_pct"] / 100.0) ** (1.0 / 365.0) - 1.0
                x["country"] = country; x["ric"] = ric; x["rf_proxy"] = int(is_proxy)
                x["rf_source"] = "PHP1MID=PHR Philippines 1-month implied proxy" if is_proxy else f"{ric} LSEG 1-month rate"
                rows.append(x[["date", "country", "ric", "rate_annual_pct", "rf_daily", "rf_proxy", "rf_source"]])
                print("FETCHED", country, ric, "rows=", len(x), flush=True)
            except Exception as exc:
                errors.append({"country": country, "ric": ric, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
    finally:
        try: ld.close_session()
        except Exception: pass
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_parquet(a.output, index=False)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "start": a.start, "end": a.end,
                "rows": len(out), "errors": errors, "proxy_note": "Philippines uses PHP1MID=PHR because PHP1MD= was unavailable."}
    a.output.with_name("risk_free_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
