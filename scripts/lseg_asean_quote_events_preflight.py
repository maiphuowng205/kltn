"""Historical quote-event preflight for a small ASEAN sample."""
from pathlib import Path
import time

import pandas as pd
import lseg.data as ld
from lseg.data.content import historical_pricing


SAMPLE = {
    "Singapore": ["DBSM.SI", "OCBC.SI", "UOBH.SI"],
    "Malaysia": ["MBBM.KL", "CIMB.KL", "PUBM.KL"],
    "Indonesia": ["BBCA.JK", "BBRI.JK", "BMRI.JK"],
    "Thailand": ["PTT.BK", "CPALL.BK", "AOT.BK"],
    "Philippines": ["BDO.PS", "BPI.PS", "SM.PS"],
}
DATES = ["2026-06-16", "2026-08-14"]
FIELDS = ["BID", "ASK", "BIDSIZE", "ASKSIZE", "MID_PRICE", "TRDPRC_1"]


def main() -> None:
    out = Path("artifacts") / "asean_preflight"
    out.mkdir(parents=True, exist_ok=True)
    rows, errors, summary_rows = [], [], []
    ld.open_session()
    try:
        for country, rics in SAMPLE.items():
            for ric in rics:
                for date in DATES:
                    try:
                        response = historical_pricing.events.Definition(
                            universe=ric,
                            eventTypes="quote",
                            start=f"{date}T00:00:00",
                            end=f"{date}T23:59:59",
                            fields=FIELDS,
                            count=5000,
                        ).get_data()
                        df = response.data.df.reset_index()
                        if not df.empty:
                            df.insert(0, "RIC", ric)
                            df.insert(0, "Country", country)
                            df.insert(2, "RequestedDate", date)
                            rows.append(df)
                        present = set(df.columns)
                        summary_rows.append({
                            "Country": country, "RIC": ric, "RequestedDate": date,
                            "rows": len(df),
                            "bid_nonnull": int(df["BID"].notna().sum()) if "BID" in present else 0,
                            "ask_nonnull": int(df["ASK"].notna().sum()) if "ASK" in present else 0,
                            "bidsize_nonnull": int(df["BIDSIZE"].notna().sum()) if "BIDSIZE" in present else 0,
                            "asksize_nonnull": int(df["ASKSIZE"].notna().sum()) if "ASKSIZE" in present else 0,
                            "columns": ",".join(df.columns),
                        })
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
    data.to_csv(out / "quote_events.csv", index=False)
    data.to_parquet(out / "quote_events.parquet", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "quote_event_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "quote_event_errors.csv", index=False)
    print("\nOUTPUT", out.resolve())
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
