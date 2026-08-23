"""Assemble the V1 daily panels already used in Colab into a V2 builder source."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

COUNTRIES=("indonesia","malaysia","philippines","singapore","thailand")

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--country-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--load-all",action="store_true",help="Create one combined ASEAN parquet; requires high-RAM Colab."); a=p.parse_args()
    paths=[a.country_root/country/"data/lseg_v3/curated/daily_panel.parquet" for country in COUNTRIES]
    missing=[str(path) for path in paths if not path.exists()]
    if missing: raise FileNotFoundError(f"Missing country V1 panels: {missing}")
    output=a.output_root/"curated"
    if a.load_all:
        # Explicit high-RAM path requested by the user. It is useful when one
        # wants a single complete ASEAN source for audit/debugging.
        daily=pd.concat([pd.read_parquet(path) for path in paths],ignore_index=True)
        output.mkdir(parents=True,exist_ok=True); daily.to_parquet(output/"daily_panel.parquet",index=False)
        print({"output":str(output/"daily_panel.parquet"),"rows":len(daily),"mode":"load_all"})
        return
    output=output/"daily_panel"; output.mkdir(parents=True,exist_ok=True)
    # Keep each market separate on disk.  Concatenating all five panels is an
    # unnecessary >GB in-memory allocation in a Colab runtime.
    rows={}
    for country,path in zip(COUNTRIES,paths):
        daily=pd.read_parquet(path)
        target=output/f"country={country}"; target.mkdir(parents=True,exist_ok=True)
        daily.to_parquet(target/"part.parquet",index=False)
        rows[country]=len(daily)
    print({"output":str(output),"rows_by_country":rows})

if __name__=="__main__":main()
