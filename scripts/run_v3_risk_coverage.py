"""Produce a train/validation-only covariance-history coverage report."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/lseg_v3"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v3_risk_coverage"))
    args = parser.parse_args(); args.run_dir.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_parquet(args.data_root / "curated" / "universe_weekly.parquet")
    weekly["date"] = pd.to_datetime(weekly["date"]).dt.normalize()
    weekly = weekly.loc[weekly["split"].isin(["train", "validation"])].sort_values(["date", "market_cap_rank"]).groupby("date", sort=True).head(100)
    daily = pd.read_parquet(args.data_root / "curated" / "daily_panel.parquet", columns=["date", "ric", "return"])
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    # Use the same pandas Timestamp key type on both sides.  NumPy datetime64
    # keys can hash differently across Colab/Pandas versions and cause a
    # false KeyError for valid weekly dates.
    calendar = pd.DatetimeIndex(daily["date"].drop_duplicates()).sort_values()
    positions = {pd.Timestamp(value).normalize(): index for index, value in enumerate(calendar)}
    returns = daily.pivot_table(index="date", columns="ric", values="return", aggfunc="last").reindex(index=calendar)
    rows=[]
    for date, group in weekly.groupby("date", sort=True):
        rics = group["ric"].astype(str).tolist(); date_key = pd.Timestamp(date).normalize()
        if date_key not in positions:
            raise ValueError(f"Weekly date {date_key.date()} is absent from daily_panel calendar")
        end = positions[date_key]; window_dates = calendar[max(0, end - 252 + 1):end + 1]
        valid = returns.reindex(index=window_dates, columns=rics).notna().all(axis=0)
        rows.append({"split": str(group["split"].iloc[0]), "date": date, "window_rows": len(window_dates), "universe_assets": len(rics), "valid_covariance_assets": int(valid.sum()), "fallback": bool(valid.sum() < 20)})
    report_df=pd.DataFrame(rows); report_df.to_parquet(args.run_dir/'risk_coverage_by_date.parquet',index=False)
    summary=report_df.groupby('split').agg(dates=('date','count'),mean_valid_assets=('valid_covariance_assets','mean'),min_valid_assets=('valid_covariance_assets','min'),fallback_dates=('fallback','sum')).reset_index(); summary.to_parquet(args.run_dir/'risk_coverage_summary.parquet',index=False)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'window_sessions':252,'splits':['train','validation'],'fallback_dates_total':int(report_df.fallback.sum()),'dates_total':len(report_df),'fallback_fraction':float(report_df.fallback.mean()),'summary':summary.to_dict(orient='records')}; (args.run_dir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(summary.to_string(index=False)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
