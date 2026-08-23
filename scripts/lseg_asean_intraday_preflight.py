"""Intraday quote preflight for a small ASEAN sample via LSEG Desktop."""
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
DATES = ["2020-06-15", "2023-06-15", "2025-06-16"]
FIELDS = ["BID", "ASK", "TRDPRC_1", "BIDSIZE", "ASKSIZE", "MID_PRICE"]


def main() -> None:
    out = Path("artifacts") / "asean_preflight"
    out.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    ld.open_session()
    try:
        for country, rics in SAMPLE.items():
            for ric in rics:
                for date in DATES:
                    start = f"{date}T00:00"
                    end = f"{date}T23:59"
                    try:
                        df = ld.get_history(
                            universe=ric,
                            fields=FIELDS,
                            start=start,
                            end=end,
                            interval="1min",
                        ).reset_index()
                        if "RIC" not in df.columns:
                            df.insert(0, "RIC", ric)
                        df.insert(0, "Country", country)
                        df.insert(2, "RequestedDate", date)
                        rows.append(df)
                        print(country, ric, date, "rows=", len(df), "columns=", list(df.columns))
                    except Exception as exc:
                        errors.append({"Country": country, "RIC": ric, "RequestedDate": date,
                                       "error_type": type(exc).__name__, "error": str(exc)[:1000]})
                        print(country, ric, date, "ERROR", type(exc).__name__, str(exc)[:500])
                    time.sleep(0.25)
    finally:
        try:
            ld.close_session()
        except Exception:
            pass

    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    data.to_csv(out / "intraday_quotes.csv", index=False)
    data.to_parquet(out / "intraday_quotes.parquet", index=False)
    pd.DataFrame(errors).to_csv(out / "intraday_errors.csv", index=False)

    summary_rows = []
    for country, rics in SAMPLE.items():
        for ric in rics:
            for date in DATES:
                part = data[(data.get("Country", pd.Series(dtype=str)).eq(country)) &
                            (data.get("RIC", pd.Series(dtype=str)).eq(ric)) &
                            (data.get("RequestedDate", pd.Series(dtype=str)).eq(date))]
                has_bid = part["BID"].notna() if "BID" in part else pd.Series(dtype=bool)
                has_ask = part["ASK"].notna() if "ASK" in part else pd.Series(dtype=bool)
                summary_rows.append({
                    "Country": country, "RIC": ric, "RequestedDate": date,
                    "rows": len(part),
                    "bid_nonnull_share": float(has_bid.mean()) if len(part) else 0.0,
                    "ask_nonnull_share": float(has_ask.mean()) if len(part) else 0.0,
                    "both_nonnull_share": float((has_bid & has_ask).mean()) if len(part) else 0.0,
                    "columns_present": ",".join(part.columns),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "intraday_coverage_summary.csv", index=False)
    print("\nOUTPUT", out.resolve())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
