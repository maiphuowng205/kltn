"""Paired daily inference and turnover-reduction metrics for V2 portfolios."""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

def sharpe(x: np.ndarray) -> float:
    return float(x.mean()/x.std(ddof=1)*np.sqrt(252)) if len(x)>1 and x.std(ddof=1)>0 else np.nan
def bootstrap_delta_sharpe(left: np.ndarray,right: np.ndarray,seed=7,draws=2000):
    if len(left)<3:return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed); ids=rng.integers(0,len(left),(draws,len(left))); values=np.asarray([sharpe(left[i])-sharpe(right[i]) for i in ids]); return float(sharpe(left)-sharpe(right)),float(np.nanquantile(values,.025)),float(np.nanquantile(values,.975))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',action='append',required=True,help='LABEL=run_directory'); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); returns={}; rebalances={}
    for item in a.input:
        label,path=item.split('=',1); root=Path(path); returns[label]=pd.read_parquet(root/'daily_portfolio_returns.parquet'); rebalances[label]=pd.read_parquet(root/'rebalance_log.parquet')
    rows=[]
    for country in sorted(set().union(*(set(x.country.unique()) for x in returns.values()))):
        for left,right in combinations(sorted(returns),2):
            l=returns[left].loc[returns[left].country.eq(country),['date','net_return']]; r=returns[right].loc[returns[right].country.eq(country),['date','net_return']]; merged=l.merge(r,on='date',suffixes=('_left','_right')); estimate,low,high=bootstrap_delta_sharpe(merged.net_return_left.to_numpy(),merged.net_return_right.to_numpy())
            lt=float(rebalances[left].loc[rebalances[left].country.eq(country),'turnover'].mean()); rt=float(rebalances[right].loc[rebalances[right].country.eq(country),'turnover'].mean())
            rows.append({'country':country,'left_strategy':left,'right_strategy':right,'aligned_days':len(merged),'delta_annualized_net_sharpe':estimate,'delta_sharpe_ci_low':low,'delta_sharpe_ci_high':high,'left_mean_turnover':lt,'right_mean_turnover':rt,'turnover_reduction_left_vs_right':float(1-lt/rt) if np.isfinite(lt) and np.isfinite(rt) and rt>0 else np.nan})
    output=pd.DataFrame(rows); a.output.parent.mkdir(parents=True,exist_ok=True); output.to_csv(a.output,index=False); print(output.to_string(index=False))
if __name__=='__main__':main()
