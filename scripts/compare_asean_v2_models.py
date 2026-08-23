"""Time-block bootstrap inference for forecast-model differences."""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

def ci_difference(x: np.ndarray, seed: int=7, draws: int=2000) -> tuple[float,float,float]:
    x=x[np.isfinite(x)]
    if len(x)<2:return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed); values=rng.choice(x,(draws,len(x)),replace=True).mean(1)
    return float(x.mean()),float(np.quantile(values,.025)),float(np.quantile(values,.975))

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input',action='append',required=True,help='MODEL=forecast_metrics_by_date.parquet'); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); frames=[]
    for item in a.input:
        name,path=item.split('=',1); frame=pd.read_parquet(path); frame['model']=name; frames.append(frame)
    data=pd.concat(frames,ignore_index=True); rows=[]
    for country,g in data.groupby('country',sort=True):
        for left,right in combinations(sorted(g.model.unique()),2):
            aligned=g.loc[g.model.eq(left),['date','spearman_ic']].merge(g.loc[g.model.eq(right),['date','spearman_ic']],on='date',suffixes=('_left','_right'))
            mean,low,high=ci_difference((aligned.spearman_ic_left-aligned.spearman_ic_right).to_numpy())
            rows.append({'country':country,'left_model':left,'right_model':right,'metric':'delta_mean_spearman_ic','aligned_dates':len(aligned),'estimate':mean,'ci_low':low,'ci_high':high})
    result=pd.DataFrame(rows); a.output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(a.output,index=False); print(result.to_string(index=False))
if __name__=='__main__':main()
