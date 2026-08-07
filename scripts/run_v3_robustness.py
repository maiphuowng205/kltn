"""Run the pre-specified one-dimension-at-a-time robustness grid."""
from __future__ import annotations
import argparse, itertools, json
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.v3_method import cost_aware_mvo, l1_turnover, ledoit_covariance

def future_returns(daily):
    out=[]
    for ric,g in daily.sort_values(['ric','session_id']).groupby('ric',sort=False):
        g=g.copy(); contiguous=pd.Series(True,index=g.index); stock=pd.Series(1.0,index=g.index); rf=pd.Series(1.0,index=g.index)
        for k in range(1,6): contiguous &= g.session_id.shift(-k).eq(g.session_id+k); stock*=1+g['return'].shift(-k); rf*=1+g.rf_daily.shift(-k)
        g['future_raw_5d']=np.where(contiguous,stock-1,np.nan); g['future_excess_5d']=np.where(contiguous,stock-rf,np.nan); out.append(g[['date','ric','future_raw_5d','future_excess_5d']])
    return pd.concat(out,ignore_index=True)

def covariance_variant(daily,date,rics,kind,window):
    if kind=='ledoit_wolf': return ledoit_covariance(daily,date,rics,window)
    frame=daily.loc[daily.ric.isin(rics)].copy(); calendar=np.sort(daily.date.unique()); positions={d:i for i,d in enumerate(calendar)}; end=positions.get(pd.Timestamp(date).to_datetime64())
    if end is None: return np.eye(len(rics))*1e-4,np.zeros(len(rics),dtype=bool),{'fallback':'missing_signal_date'}
    dates=calendar[max(0,end-window+1):end+1]; pivot=frame.loc[frame.date.isin(dates)].pivot_table(index='date',columns='ric',values='return',aggfunc='last').reindex(index=dates,columns=rics); valid=pivot.notna().all(axis=0).to_numpy(); cov=np.eye(len(rics))*1e-4
    if valid.sum()<20: return cov,valid,{'fallback':'fewer_than_20_complete_assets','valid_assets':int(valid.sum()),'window_rows':len(pivot)}
    values=pivot.loc[:,valid].to_numpy(float); centered=values-values.mean(axis=0)
    if kind=='sample': block=np.cov(centered,rowvar=False)
    elif kind=='ewma_half_life_60':
        weights=np.exp(np.log(.5)*np.arange(len(values)-1,-1,-1)/60); weights/=weights.sum(); mean=np.sum(values*weights[:,None],axis=0); z=values-mean; block=(z*weights[:,None]).T@z
    else: raise ValueError(kind)
    cov[np.ix_(valid,valid)]=np.atleast_2d(block); return (cov+cov.T)/2,valid,{'fallback':None,'valid_assets':int(valid.sum()),'window_rows':len(pivot)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=ROOT/'data/lseg_v3'); p.add_argument('--forecast-run',type=Path,required=True); p.add_argument('--run-dir',type=Path,default=ROOT/'runs/v3_robustness'); p.add_argument('--max-variants',type=int,default=21); a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=True)
    fc=pd.read_parquet(a.forecast_run/'forecasts.parquet'); fc['date']=pd.to_datetime(fc.date).dt.normalize(); fc=fc.loc[fc.split.eq('test')].copy(); pred={(pd.Timestamp(d),str(r)):float(v) for d,r,v in fc[['date','ric','prediction_excess_return_5d_bps']].itertuples(index=False,name=None)}
    weekly=pd.read_parquet(a.data_root/'curated'/'universe_weekly.parquet'); weekly['date']=pd.to_datetime(weekly.date).dt.normalize(); weekly=weekly.loc[weekly.split.eq('test')].sort_values(['date','market_cap_rank']).groupby('date').head(100); dates=sorted(weekly.date.unique())
    daily=pd.read_parquet(a.data_root/'curated'/'daily_panel.parquet'); daily['date']=pd.to_datetime(daily.date).dt.normalize(); daily['session_id']=daily.session_id.astype(int); daily['rf_daily']=pd.to_numeric(daily.rf_daily,errors='coerce').fillna(0.0); daily=daily.merge(future_returns(daily),on=['date','ric'],how='left',validate='one_to_one'); daily_map=daily.set_index(['date','ric'])
    base={'cost':.001,'covariance':'ledoit_wolf','lookback':252,'turnover_cap':.40,'max_weight':.05,'risk_aversion':10.0}; variants=[]
    for dim,values in [('cost',[0,.0005,.001,.002,.003,.005]),('covariance',['ledoit_wolf','sample','ewma_half_life_60']),('lookback',[20,60,120]),('turnover_cap',[.20,.40,.80]),('max_weight',[.03,.05,.10]),('risk_aversion',[5.,10.,20.])]:
        for value in values:
            config=base.copy(); config[dim]=value; config['variant']=f'{dim}={value}'; variants.append(config)
    variants=variants[:a.max_variants]; rows=[]
    for config in variants:
        hold={}; net_values=[]; turnovers=[]
        for i,dv in enumerate(dates):
            date=pd.Timestamp(dv); rics=weekly.loc[weekly.date.eq(date),'ric'].astype(str).tolist(); cov,valid,risk=covariance_variant(daily,date,rics,config['covariance'],int(config['lookback'])); exited={r:w for r,w in hold.items() if r not in set(rics)}; exit_weight=sum(exited.values()); pre=np.asarray([hold.get(r,0.) for r in rics]);
            if i==0 and pre.sum()==0 and valid.sum()>=20: pre[valid]=1/valid.sum()
            mu=np.asarray([pred[(date,r)] for r in rics])/10000.; target,solver=cost_aware_mvo(mu,cov,pre,valid,risk_aversion=float(config['risk_aversion']),cost=float(config['cost']),max_weight=float(config['max_weight']),turnover_cap=float(config['turnover_cap']),turnover_fixed=float(exit_weight)); turnover=l1_turnover(target, pre, exit_weight); rr=[]; raw=[]
            for r in rics:
                try: row=daily_map.loc[(date,r)]; rr.append(float(row.future_excess_5d)); raw.append(float(row.future_raw_5d))
                except KeyError: rr.append(np.nan); raw.append(np.nan)
            rr=np.asarray(rr); raw=np.asarray(raw); ok=np.isfinite(rr)&np.isfinite(raw); net_values.append(float(target@rr-float(config['cost'])*turnover) if ok.all() else np.nan); turnovers.append(turnover); post=target*(1+np.nan_to_num(raw,nan=0)); total=post.sum(); hold={r:float(w/total) for r,w in zip(rics,post)} if total>0 else {r:float(w) for r,w in zip(rics,target)}
        vals=np.asarray(net_values); vals=vals[np.isfinite(vals)]; rows.append({**config,'evaluation_dates':len(vals),'mean_net_excess_5d':float(vals.mean()) if len(vals) else np.nan,'net_sharpe_annualized':float(vals.mean()/vals.std(ddof=1)*np.sqrt(52)) if len(vals)>1 and vals.std(ddof=1)>0 else np.nan,'mean_turnover':float(np.mean(turnovers)),'solver_fallbacks':None})
    out=pd.DataFrame(rows); out.to_parquet(a.run_dir/'robustness_results.parquet',index=False); report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'variant_count':len(variants),'grid_policy':'one dimension at a time; base values held fixed','base':base}; (a.run_dir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(out.to_string(index=False)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
