"""Build the single locked forecast-metric table for any V2 model output."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.asean_v2 import summarize_forecasts

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--input",action="append",required=True,help="MODEL=path/to/development_predictions.npz; repeat for every model."); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    pieces=[]
    for item in a.input:
        model,path=item.split("=",1); blob=np.load(Path(path),allow_pickle=False); dates=pd.to_datetime(blob["dates"]); countries=blob["countries"].astype(str); rics=blob["rics"].astype(str); prediction=blob["raw_score"]*100.0; target=blob["target_bps"]; available=blob["target_mask"]&blob["asset_mask"]
        for i,(date,country) in enumerate(zip(dates,countries)):
            pieces.append(pd.DataFrame({"country":country,"date":date,"ric":rics[i],"model":model,"prediction_bps":prediction[i],"target_bps":target[i],"target_available":available[i] & (rics[i] != "")}))
    rows=pd.concat(pieces,ignore_index=True); daily,summary=summarize_forecasts(rows); rows.to_parquet(a.output_dir/"forecast_rows.parquet",index=False); daily.to_parquet(a.output_dir/"forecast_metrics_by_date.parquet",index=False); summary.to_csv(a.output_dir/"forecast_metrics_summary.csv",index=False); print(summary.to_string(index=False))
if __name__=="__main__": main()
