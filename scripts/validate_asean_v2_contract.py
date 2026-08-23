"""Validate the V2 timing, purge and investability contract before modelling."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,default=ROOT/"artifacts"/"asean_v2"); p.add_argument("--report",type=Path,default=None); a=p.parse_args()
    weekly=pd.read_parquet(a.data_root/"model_ready"/"weekly_features_targets_v2.parquet")
    for col in ("date","execution_date_v2","label_start_date_v2","label_end_date_v2"): weekly[col]=pd.to_datetime(weekly[col]).dt.normalize()
    errors=[]
    if weekly.market_cap_rank.gt(100).any(): errors.append("pure Top-100 universe contains rank above 100")
    counts=weekly.groupby(["country","date"]).ric.nunique()
    if counts.gt(100).any(): errors.append("variable-N universe exceeds 100 assets")
    purged=weekly.loc[weekly.split_v2.eq("train") & weekly.label_end_date_v2.ge("2023-01-01")]
    if purged.model_eligible_v2.any(): errors.append("purged train label crosses development boundary")
    timing=weekly.dropna(subset=["execution_date_v2","label_start_date_v2","label_end_date_v2"])
    if (timing.execution_date_v2.le(timing.date)).any() or (timing.label_start_date_v2.le(timing.execution_date_v2)).any() or (timing.label_end_date_v2.le(timing.label_start_date_v2)).any(): errors.append("signal/execution/label ordering invalid")
    means=weekly.groupby(["country","date"])["target_cs_excess_return_5d_bps_v2"].mean().dropna()
    if not np.allclose(means.to_numpy(),0.,atol=1e-8): errors.append("cross-sectional target does not demean to zero")
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if not errors else "FAIL","errors":errors,"weekly_dates":int(len(counts)),"assets_per_date":{"min":int(counts.min()),"median":float(counts.median()),"max":int(counts.max())},"purged_train_rows":int((weekly.split_v2.eq("train") & weekly.purged_from_split).sum()),"timing":"signal close t -> execution close t+1 -> label t+2 through t+6","final_holdout":"not available in V1 source; 2024-2025 are development only"}
    target=a.report or a.data_root/"reports"/"validation_v2.json"; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
    if errors: raise SystemExit(1)

if __name__=="__main__": main()
