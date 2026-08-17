"""Run PTCST-Top20, PTCST-MVO and PTCST-CA-MVO on a PTCST forecast run."""
from __future__ import annotations

import argparse, json
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
        for k in range(1,6):
            contiguous &= g['session_id'].shift(-k).eq(g['session_id']+k); stock *= 1+g['return'].shift(-k); rf *= 1+g['rf_daily'].shift(-k)
        g['future_raw_5d']=np.where(contiguous,stock-1,np.nan); g['future_excess_5d']=np.where(contiguous,stock-rf,np.nan); out.append(g[['date','ric','future_raw_5d','future_excess_5d']])
    return pd.concat(out,ignore_index=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=ROOT/'data/lseg_v3'); p.add_argument('--forecast-run',type=Path,required=True); p.add_argument('--run-dir',type=Path,default=ROOT/'runs/v3_ptcst_ablations'); p.add_argument('--cost-bps',type=float,default=10.0,help='one-way transaction cost used by the CA-MVO branch'); a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=True)
    fc=pd.read_parquet(a.forecast_run/'forecasts.parquet'); fc['date']=pd.to_datetime(fc['date']).dt.normalize(); fc=fc.loc[fc.split.eq('test')].copy(); fc['prediction']=fc['prediction_excess_return_5d_bps'].astype(float)
    weekly=pd.read_parquet(a.data_root/'curated'/'universe_weekly.parquet'); weekly['date']=pd.to_datetime(weekly.date).dt.normalize(); weekly=weekly.loc[weekly.split.eq('test')].sort_values(['date','market_cap_rank']).groupby('date',sort=True).head(100)
    daily=pd.read_parquet(a.data_root/'curated'/'daily_panel.parquet'); daily['date']=pd.to_datetime(daily.date).dt.normalize(); daily['session_id']=daily.session_id.astype(int); daily['rf_daily']=pd.to_numeric(daily.rf_daily,errors='coerce').fillna(0.0); daily=daily.merge(future_returns(daily),on=['date','ric'],how='left',validate='one_to_one'); daily_map=daily.set_index(['date','ric'])
    pred={(pd.Timestamp(d),str(r)):float(v) for d,r,v in fc[['date','ric','prediction']].itertuples(index=False,name=None)}; dates=sorted(weekly.date.unique()); strategies=['PTCST-Top20','PTCST-MVO','PTCST-CA-MVO']; returns=[]; weights=[]; trades=[]; solvers=[]; missing=[]
    for strategy in strategies:
        hold={}
        for i,dv in enumerate(dates):
            date=pd.Timestamp(dv); rics=weekly.loc[weekly.date.eq(date),'ric'].astype(str).tolist(); mu=np.asarray([pred[(date,ric)] for ric in rics])/10000.0; cov,valid,risk=ledoit_covariance(daily,date,rics); exited={r:w for r,w in hold.items() if r not in set(rics)}; exited_weight=sum(exited.values()); w_pre=np.asarray([hold.get(r,0.0) for r in rics])
            if i==0 and w_pre.sum()==0 and valid.sum()>=20: w_pre[valid]=1/valid.sum()
            if strategy=='PTCST-Top20':
                target=np.zeros(100); top=np.argsort(mu)[-20:]; target[top]=0.05
                # If a current universe change removes held assets, the top-20
                # target already liquidates them and the extra exit is logged.
                solver={'solver':None,'status':'direct_top20','fallback':None}
            else: target,solver=cost_aware_mvo(mu,cov,w_pre,valid,cost=(a.cost_bps / 10000.0) if strategy.endswith('CA-MVO') else 0.0,turnover_fixed=float(exited_weight))
            turnover=l1_turnover(target, w_pre, exited_weight); realized=[]; raw=[]
            for ric in rics:
                try: row=daily_map.loc[(date,ric)]; realized.append(float(row.future_excess_5d)); raw.append(float(row.future_raw_5d))
                except KeyError: realized.append(np.nan); raw.append(np.nan)
            rr=np.asarray(realized); raw_arr=np.asarray(raw); ok=np.isfinite(rr)&np.isfinite(raw_arr); gross=float(target@rr) if ok.all() else np.nan; net=gross-0.001*turnover if ok.all() else np.nan
            for ric,isok in zip(rics,ok):
                if not isok: missing.append({'strategy':strategy,'date':date,'ric':ric,'reason':'future_5_session_return_unavailable'})
            returns.append({'strategy':strategy,'date':date,'gross_excess_return_5d':gross,'turnover_l1':turnover,'cost':0.001*turnover,'net_excess_return_5d':net,'evaluation_available':bool(ok.all()),'available_realized_assets':int(ok.sum()),'solver':solver.get('solver'),'solver_status':solver.get('status'),'solver_fallback':solver.get('fallback'),'risk_fallback':risk.get('fallback')})
            solvers.append({'strategy':strategy,'date':date,**risk,**solver})
            for ric,pre in exited.items(): trades.append({'strategy':strategy,'date':date,'ric':ric,'trade':-float(pre),'exited_universe':True})
            for ric,w,pre in zip(rics,target,w_pre): weights.append({'strategy':strategy,'date':date,'ric':ric,'weight':float(w),'w_pre':float(pre)}); trades.append({'strategy':strategy,'date':date,'ric':ric,'trade':float(w-pre),'exited_universe':False})
            post=target*(1+np.nan_to_num(raw_arr,nan=0.0)); total=post.sum(); hold={r:float(w/total) for r,w in zip(rics,post)} if total>0 else {r:float(w) for r,w in zip(rics,target)}
    rdf=pd.DataFrame(returns); rdf.to_parquet(a.run_dir/'portfolio_returns.parquet',index=False); pd.DataFrame(weights).to_parquet(a.run_dir/'weights.parquet',index=False); pd.DataFrame(trades).to_parquet(a.run_dir/'trades.parquet',index=False); pd.DataFrame(solvers).to_parquet(a.run_dir/'solver_log.parquet',index=False); pd.DataFrame(missing,columns=['strategy','date','ric','reason']).to_parquet(a.run_dir/'missing_price_events.parquet',index=False)
    ev=rdf.loc[rdf.evaluation_available]; summary=ev.groupby('strategy').agg(evaluation_dates=('date','count'),mean_net_excess_5d=('net_excess_return_5d','mean'),mean_turnover=('turnover_l1','mean'),net_sharpe_annualized=('net_excess_return_5d',lambda x:float(x.mean()/x.std(ddof=1)*np.sqrt(52)) if len(x)>1 and x.std(ddof=1)>0 else np.nan)).reset_index(); summary.to_parquet(a.run_dir/'portfolio_metrics_summary.parquet',index=False)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'strategies':strategies,'forecast_run':str(a.forecast_run),'test_dates':len(dates),'cost_bps_one_way':float(a.cost_bps)}; (a.run_dir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(summary.to_string(index=False)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
