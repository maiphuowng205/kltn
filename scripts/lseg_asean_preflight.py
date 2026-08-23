"""Small LSEG Desktop-session preflight for ASEAN quote coverage.

This deliberately downloads a small, auditable sample before any full-universe
extract. It uses the Workspace desktop session opened on the same machine.
"""
from pathlib import Path
import time

import pandas as pd
import lseg.data as ld


SAMPLE = {
    "Singapore": ["DBSM.SI", "OCBC.SI", "UOBH.SI"],
    "Malaysia": ["MBBM.KL", "CIMB.KL", "PUBM.KL"],
    "Indonesia": ["BBCA.JK", "BBRI.JK", "BMRI.JK"],
    "Thailand": ["PTT.BK", "CPALL.BK", "AOT.BK"],
    "Philippines": ["BDO.PS", "BPI.PS", "SM.PS"],
}
FIELDS = [
    "BID", "ASK", "TRDPRC_1", "BIDSIZE", "ASKSIZE", "MID_PRICE",
    "ACVOL_UNS", "TRNOVR_UNS",
]
START = "2020-01-01"
END = "2025-12-31"


def main() -> None:
    out = Path("artifacts") / "asean_preflight"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    ld.open_session()
    try:
        for country, rics in SAMPLE.items():
            for ric in rics:
                try:
                    df = ld.get_history(
                        universe=ric,
                        fields=FIELDS,
                        start=START,
                        end=END,
                        interval="daily",
                    ).reset_index()
                    if "RIC" not in df.columns:
                        df.insert(0, "RIC", ric)
                    df.insert(0, "Country", country)
                    rows.append(df)
                    print(country, ric, "rows=", len(df), "columns=", list(df.columns))
                except Exception as exc:  # keep the rest of the sample running
                    errors.append({"Country": country, "RIC": ric,
                                   "error_type": type(exc).__name__,
                                   "error": str(exc)[:1000]})
                    print(country, ric, "ERROR", type(exc).__name__, str(exc)[:500])
                time.sleep(0.25)
    finally:
        try:
            ld.close_session()
        except Exception:
            pass

    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    data.to_csv(out / "daily_quotes.csv", index=False)
    data.to_parquet(out / "daily_quotes.parquet", index=False)

    summary_rows = []
    for country, rics in SAMPLE.items():
        for ric in rics:
            part = data[data.get("RIC", pd.Series(dtype=str)).astype(str).eq(ric)] if not data.empty else pd.DataFrame()
            summary_rows.append({
                "Country": country,
                "RIC": ric,
                "rows": len(part),
                "bid_nonnull": int(part["BID"].notna().sum()) if "BID" in part else 0,
                "ask_nonnull": int(part["ASK"].notna().sum()) if "ASK" in part else 0,
                "trade_nonnull": int(part["TRDPRC_1"].notna().sum()) if "TRDPRC_1" in part else 0,
                "bid_ask_complete": int((part["BID"].notna() & part["ASK"].notna()).sum()) if {"BID", "ASK"}.issubset(part.columns) else 0,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "coverage_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "errors.csv", index=False)
    print("\nOUTPUT", out.resolve())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
