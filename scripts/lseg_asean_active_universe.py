"""Export active primary ordinary-share RICs for five ASEAN exchanges."""
from pathlib import Path
import lseg.data as ld
import pandas as pd


SUFFIX_TO_COUNTRY = {
    "SI": "Singapore",
    "KL": "Malaysia",
    "JK": "Indonesia",
    "BK": "Thailand",
    "PS": "Philippines",
}


def main() -> None:
    out = Path("artifacts") / "asean_preflight"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    ld.open_session()
    try:
        for suffix, country in SUFFIX_TO_COUNTRY.items():
            df = ld.discovery.search(
                view="EquityQuotes",
                filter=(
                    "AssetState ne 'DC' and AssetType eq 'EQUITY' "
                    "and IsPrimaryIssueRIC eq true and RIC eq '*.%s'" % suffix
                ),
                select="RIC,DocumentTitle,AssetState,IsPrimaryIssueRIC",
                order_by="RIC asc",
                top=10000,
            )
            df["RIC"] = df["RIC"].astype(str)
            df = df[df["DocumentTitle"].astype(str).str.contains("Ordinary Share", case=False, na=False)]
            df = df[df["RIC"].map(lambda x: not any(ch.islower() for ch in x.split(".")[0]))]
            df.insert(0, "Country", country)
            df.insert(1, "Suffix", suffix)
            frames.append(df)
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out / "active_primary_ordinary_universe.csv", index=False)
    result.to_parquet(out / "active_primary_ordinary_universe.parquet", index=False)
    print(result.groupby("Country").size().to_string())
    print("total", len(result))
    print("saved", (out / "active_primary_ordinary_universe.csv").resolve())


if __name__ == "__main__":
    main()
