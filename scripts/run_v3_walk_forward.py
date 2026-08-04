"""Expanding quarterly walk-forward Ridge forecast without label leakage."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.v3_method import build_tensor_bundle

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=ROOT/'data/lseg_v3'); p.add_argument('--run-dir',type=Path,default=ROOT/'runs/v3_walk_forward'); p.add_argument('--alpha',type=float,default=10.0); a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=True)
    train,median,iqr=build_tensor_bundle(a.data_root,'train'); val,_,_=build_tensor_bundle(a.data_root,'validation',median,iqr); test,_,_=build_tensor_bundle(a.data_root,'test',median,iqr)
    def features(bundle): return bundle.x.reshape(len(bundle.dates)*100,-1)
    base_x=np.concatenate([features(train),features(val)]); base_y=np.concatenate([train.y.reshape(-1),val.y.reshape(-1)]); base_mask=np.concatenate([train.mask.reshape(-1),val.mask.reshape(-1)])
    learned_blocks=[base_x[base_mask]]; learned_y_blocks=[base_y[base_mask]]; learned_exec=list(np.concatenate([train.execution_dates.reshape(-1),val.execution_dates.reshape(-1)])[base_mask]); test_features=features(test)
    rows=[]; checkpoints=[]; model=None; current_quarter=None
    for i,date64 in enumerate(test.dates):
        date=pd.Timestamp(date64); quarter=(date.year,date.quarter)
        if model is None or quarter != current_quarter:
            # Only labels fully realized by this signal date are eligible.
            learned_x=np.concatenate(learned_blocks,axis=0); learned_y=np.concatenate(learned_y_blocks,axis=0); realized=np.asarray([pd.notna(x) and pd.Timestamp(x)<=date for x in learned_exec])
            model=Ridge(alpha=a.alpha).fit(learned_x[realized],learned_y[realized]); current_quarter=quarter; checkpoints.append({'date':date,'quarter':f'{date.year}Q{date.quarter}','training_rows':int(realized.sum()),'training_cutoff':str(max(pd.Timestamp(x) for x in np.asarray(learned_exec,dtype=object)[realized])) if realized.any() else None})
        pred=model.predict(test_features[i * 100:(i + 1) * 100]).reshape(100)
        for j,ric in enumerate(test.rics[i]): rows.append({'date':date,'ric':str(ric),'split':'test','prediction':float(pred[j]),'target':float(test.y[i,j]),'target_available':bool(test.mask[i,j]),'model_training_quarter':f'{date.year}Q{date.quarter}'})
        # Add this date's labels only after its forecast has been produced; at
        # the next date they are filtered again by execution_date.
        day_mask=test.mask[i]
        learned_blocks.append(test_features[i * 100:(i + 1) * 100][day_mask]); learned_y_blocks.append(test.y[i][day_mask]); learned_exec.extend(list(test.execution_dates[i][day_mask]))
    f=pd.DataFrame(rows); metrics=[]
    for date,g in f.groupby('date'):
        g=g.loc[g.target_available]; pvals=g.prediction.to_numpy(); y=g.target.to_numpy(); metrics.append({'date':date,'n_assets':len(g),'spearman_ic':float(pd.Series(pvals).rank().corr(pd.Series(y).rank())) if len(g)>=3 else np.nan,'mae_bps':float(np.mean(np.abs(pvals-y))) if len(g) else np.nan})
    f.to_parquet(a.run_dir/'forecasts.parquet',index=False); pd.DataFrame(metrics).to_parquet(a.run_dir/'forecast_metrics_by_date.parquet',index=False); pd.DataFrame(checkpoints).to_parquet(a.run_dir/'retraining_checkpoints.parquet',index=False)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'model':'Ridge(alpha=%s)'%a.alpha,'retraining':'expanding quarterly','test_dates':len(test.dates),'retraining_points':len(checkpoints),'label_rule':'target_available and execution_date <= signal date'}; (a.run_dir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
