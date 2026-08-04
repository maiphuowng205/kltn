"""Run the frozen Vietnam V3 method: PTCST forecast + cost-aware MVO."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.v3_method import build_tensor_bundle, cost_aware_mvo, ledoit_covariance, predict_ptcst, train_ptcst


def future_returns(daily: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for ric,g in daily.sort_values(["ric","session_id"]).groupby("ric",sort=False):
        g=g.copy(); contiguous=pd.Series(True,index=g.index)
        stock=pd.Series(1.0,index=g.index); rf=pd.Series(1.0,index=g.index)
        for k in range(1,6):
            contiguous &= g["session_id"].shift(-k).eq(g["session_id"]+k)
            stock *= 1+g["return"].shift(-k); rf *= 1+g["rf_daily"].shift(-k)
        g["future_raw_5d"]=np.where(contiguous,stock-1,np.nan)
        g["future_excess_5d"]=np.where(contiguous,stock-rf,np.nan)
        out.append(g[["date","ric","future_raw_5d","future_excess_5d"]])
    return pd.concat(out,ignore_index=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=ROOT/'data/lseg_v3'); p.add_argument('--run-dir',type=Path,default=ROOT/'runs/v3_ptcst_ca_mvo'); p.add_argument('--epochs',type=int,default=100); p.add_argument('--seed',type=int,default=7); p.add_argument('--batch-dates',type=int,default=16); p.add_argument('--early-stopping-patience',type=int,default=10); p.add_argument('--max-test-dates',type=int,default=0); p.add_argument('--device',default=None); args=p.parse_args()
    out=args.run_dir; out.mkdir(parents=True,exist_ok=True)
    train,median,scale=build_tensor_bundle(args.data_root,'train')
    val,_,_=build_tensor_bundle(args.data_root,'validation',median,scale)
    test,_,_=build_tensor_bundle(args.data_root,'test',median,scale)
    (out/'preprocessing.json').write_text(json.dumps({'features':train.x.shape[-1],'lookback':train.x.shape[-2],'median':median.tolist(),'iqr':scale.tolist()},indent=2),encoding='utf-8')
    train_report=train_ptcst(train,val,out,args.epochs,args.seed,args.batch_dates,args.device,args.early_stopping_patience)
    val_pred=predict_ptcst(val,out/'best.pt',args.device); test_pred=predict_ptcst(test,out/'best.pt',args.device)
    forecast_rows=[]
    for bundle,pred,name in ((val,val_pred,'validation'),(test,test_pred,'test')):
        for i,date in enumerate(bundle.dates):
            for j,ric in enumerate(bundle.rics[i]): forecast_rows.append({'date':pd.Timestamp(date),'ric':ric,'split':name,'prediction_excess_return_5d_bps':float(pred[i,j]),'target_excess_return_5d_bps':float(bundle.y[i,j]),'target_available':bool(bundle.mask[i,j])})
    forecasts=pd.DataFrame(forecast_rows); forecasts.to_parquet(out/'forecasts.parquet',index=False)
    daily=pd.read_parquet(args.data_root/'curated/daily_panel.parquet'); daily['date']=pd.to_datetime(daily['date']).dt.normalize(); daily['session_id']=daily['session_id'].astype(int); daily['rf_daily']=pd.to_numeric(daily['rf_daily'],errors='coerce').fillna(0.0); daily=daily.merge(future_returns(daily),on=['date','ric'],how='left',validate='one_to_one')
    daily_map=daily.set_index(['date','ric'])
    weights=[]; trades=[]; returns=[]; solver_rows=[]; missing_events=[]
    # Carry holdings by RIC rather than by array position: the weekly
    # universe can change membership and market-cap order.
    w_pre_map={}
    dates=list(test.dates); dates=dates[:args.max_test_dates] if args.max_test_dates else dates
    for i,date64 in enumerate(dates):
        date=pd.Timestamp(date64); rics=[str(x) for x in test.rics[i]]; pred=test_pred[i]; covariance,valid,risk_status=ledoit_covariance(daily,date,rics)
        exited={ric:weight for ric,weight in w_pre_map.items() if ric not in set(rics)}
        exited_weight=float(sum(exited.values()))
        w_pre=np.asarray([w_pre_map.get(ric,0.0) for ric in rics],dtype=float)
        if i == 0 and w_pre.sum() == 0 and valid.sum() >= 20:
            w_pre[valid] = 1.0 / valid.sum()
        w_new,solver=cost_aware_mvo(pred/10000.0,covariance,w_pre,valid,turnover_fixed=exited_weight)
        turn=float(np.abs(w_new-w_pre).sum()+exited_weight); realized=[]; raw=[]
        for ric in rics:
            try: row=daily_map.loc[(date,ric)]; realized.append(float(row['future_excess_5d'])); raw.append(float(row['future_raw_5d']))
            except KeyError: realized.append(np.nan); raw.append(np.nan)
        realized_arr=np.asarray(realized,dtype=float); raw_arr=np.asarray(raw,dtype=float)
        realized_ok=np.isfinite(realized_arr) & np.isfinite(raw_arr)
        if realized_ok.all():
            gross=float(w_new@realized_arr); net=float(gross-0.001*turn)
        else:
            gross=float('nan'); net=float('nan')
            for ric,ok in zip(rics,realized_ok):
                if not ok: missing_events.append({'date':date,'ric':ric,'reason':'future_5_session_return_unavailable'})
        returns.append({'date':date,'gross_excess_return_5d':gross,'turnover_l1':turn,'turnover_exited':exited_weight,'cost':0.001*turn,'net_excess_return_5d':net,'evaluation_available':bool(realized_ok.all()),'available_realized_assets':int(realized_ok.sum()),'risk_fallback':risk_status.get('fallback'),'risk_valid_assets':risk_status.get('valid_assets'),'risk_window_rows':risk_status.get('window_rows'),'solver':solver.get('solver'),'solver_status':solver.get('status'),'solver_fallback':solver.get('fallback')})
        solver_rows.append({'date':date,'turnover_exited':exited_weight,**risk_status,**solver})
        for ric,pre in exited.items():
            trades.append({'date':date,'ric':ric,'trade':float(-pre),'exited_universe':True})
        for ric,weight,pre in zip(rics,w_new,w_pre): weights.append({'date':date,'ric':ric,'weight':float(weight),'w_pre':float(pre)}); trades.append({'date':date,'ric':ric,'trade':float(weight-pre)})
        # Drift the current holdings and align them by RIC for the next date.
        # Missing future returns are neutral only for state carry-forward and
        # are excluded from primary performance metrics above.
        drift_returns=np.nan_to_num(raw_arr,nan=0.0)
        post=w_new*(1+drift_returns); post_total=post.sum()
        w_pre_map={ric:float(weight/post_total) for ric,weight in zip(rics,post)} if post_total>0 else {ric:float(weight) for ric,weight in zip(rics,w_new)}
    pd.DataFrame(weights).to_parquet(out/'weights.parquet',index=False); pd.DataFrame(trades).to_parquet(out/'trades.parquet',index=False); returns_df=pd.DataFrame(returns); returns_df.to_parquet(out/'portfolio_returns.parquet',index=False); pd.DataFrame(solver_rows).to_parquet(out/'solver_log.parquet',index=False); pd.DataFrame(missing_events, columns=['date','ric','reason']).to_parquet(out/'missing_price_events.parquet',index=False)
    eval_returns=returns_df.loc[returns_df.evaluation_available] if len(returns_df) else returns_df
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'method':'PTCST forecast + Ledoit-Wolf covariance + cost-aware long-only MVO','train':train_report,'test_dates':len(returns_df),'evaluation_dates':len(eval_returns),'excluded_incomplete_dates':int(len(returns_df)-len(eval_returns)),'mean_net_excess_5d':float(eval_returns.net_excess_return_5d.mean()) if len(eval_returns) else None,'net_sharpe_annualized':float(eval_returns.net_excess_return_5d.mean()/eval_returns.net_excess_return_5d.std(ddof=1)*np.sqrt(52)) if len(eval_returns)>1 and eval_returns.net_excess_return_5d.std(ddof=1)>0 else None,'mean_turnover':float(eval_returns.turnover_l1.mean()) if len(eval_returns) else None,'epochs':args.epochs,'seed':args.seed}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
