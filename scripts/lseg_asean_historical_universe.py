"""Retrieve active-or-inactive primary equity universe for five ASEAN markets."""
from pathlib import Path

import pandas as pd
import lseg.data as ld


COUNTRIES = {"SG": "Singapore", "MY": "Malaysia", "ID": "Indonesia", "TH": "Thailand", "PH": "Philippines"}
FIELDS = ["TR.RIC", "TR.CommonName", "TR.ExchangeCountry", "TR.InstrumentType", "TR.PrimaryRICCode", "TR.ISIN"]


def main() -> None:
    out = Path("artifacts") / "asean_preflight"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    ld.open_session()
    try:
        for code, country in COUNTRIES.items():
            expression = (
                "SCREEN(U(IN(Equity(active or inactive,public,primary))), "
                f'IN(TR.ExchangeCountryCode,"{code}"), CURN=USD)'
            )
            try:
                df = ld.get_data(expression, FIELDS)
                df["Country"] = country
                df["ExchangeCountryCode"] = code
                frames.append(df)
                print(country, "rows=", len(df))
            except Exception as exc:
                print(country, "ERROR", type(exc).__name__, str(exc)[:1000])
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
    result = pd.concat(frames, ignore_index=True)
    if "RIC" in result:
        result = result.drop_duplicates(subset=["RIC"]).reset_index(drop=True)
    result.to_csv(out / "historical_primary_universe.csv", index=False)
    result.to_parquet(out / "historical_primary_universe.parquet", index=False)
    print(result.groupby("Country").size().to_string())
    print("total", len(result))
    print("saved", (out / "historical_primary_universe.csv").resolve())


if __name__ == "__main__":
    main()
