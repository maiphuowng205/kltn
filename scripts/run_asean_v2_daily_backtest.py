"""Continuous daily-state V2 portfolio backtest using calibrated forecasts.

Unlike V1, a missing future return never discards an entire portfolio date.
The position is carried at zero return until the next observable valuation and
the event is explicitly logged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.asean_v2 import cost_aware_mvo_vector_cost, ledoit_covariance_min_history, summarize_daily_portfolio


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,default=ROOT/"artifacts"/"asean_v2"); p.add_argument("--prediction-file",type=Path,required=True); p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--risk-min-history",type=int,default=126,choices=(90,126,180,252)); p.add_argument("--risk-aversion",type=float,required=True,help="Must be selected and locked using development-only validation."); p.add_argument("--turnover-cap",type=float,default=.40); a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(a.data_root/"curated"/"daily_panel_v2.parquet"); weekly=pd.read_parquet(a.data_root/"model_ready"/"weekly_features_targets_v2.parquet")
    for frame in (panel,weekly): frame["date"]=pd.to_datetime(frame.date).dt.normalize()
    blob=np.load(a.prediction_file, allow_pickle=False); pred_dates=pd.to_datetime(blob["dates"]).normalize(); pred_countries=blob["countries"].astype(str); pred_rics=blob["rics"].astype(str); alpha=blob["calibrated_alpha_decimal"].astype(float); asset_mask=blob["asset_mask"].astype(bool)
    schedule={}
    for i,(country,date) in enumerate(zip(pred_countries,pred_dates)):
        signal=weekly.loc[(weekly.country.eq(country))&(weekly.date.eq(date))].sort_values("market_cap_rank")
        if signal.empty: continue
        execution=pd.Timestamp(signal.execution_date_v2.iloc[0]).normalize()
        entries=[]
        for ric,value,ok in zip(pred_rics[i],alpha[i],asset_mask[i]):
            if ok and ric: entries.append((ric,float(value)))
        schedule[(country,execution)]={"signal_date":date,"entries":entries}
    daily_records=[]; rebalance_records=[]; missing=[]
    for country, country_panel in panel.groupby("country",sort=True):
        country_panel=country_panel.sort_values(["date","ric"]); holdings={}; last_cost={}; previous_date=None
        execution_dates=[execution for (scheduled_country,execution) in schedule if scheduled_country==country]
        if not execution_dates:
            continue
        country_panel=country_panel.loc[country_panel.date.ge(min(execution_dates))]
        for date, day in country_panel.groupby("date",sort=True):
            returns=day.set_index("ric")["return"].to_dict()
            missing_assets=[ric for ric in holdings if ric not in returns or not np.isfinite(returns.get(ric,np.nan))]
            pnl=sum(weight*float(returns.get(ric,0.0)) for ric,weight in holdings.items() if ric not in missing_assets)
            for ric in missing_assets: missing.append({"country":country,"date":date,"ric":ric,"reason":"missing_daily_valuation_carried_zero"})
            # Drift first: orders submitted at the prior close are owned over
            # this close-to-close interval.  A scheduled trade occurs after
            # today's P&L, at today's close.
            if holdings:
                post={ric:weight*(1+float(returns.get(ric,0.0) if np.isfinite(returns.get(ric,np.nan)) else 0.0)) for ric,weight in holdings.items()}; total=sum(post.values()); holdings={ric:value/total for ric,value in post.items()} if total>0 else holdings
            cost=0.; turnover=0.; status="no_rebalance"; risk_fallback=None
            task=schedule.get((country,pd.Timestamp(date).normalize()))
            if task:
                signal_date=task["signal_date"]; rics=[ric for ric,_ in task["entries"]]; score=np.asarray([v for _,v in task["entries"]])
                signal_rows=country_panel.loc[country_panel.date.eq(signal_date)].set_index("ric")
                costs=np.asarray([max(float(signal_rows.loc[ric,"quoted_spread_bps"])/20000.0,0.0) if ric in signal_rows.index and pd.notna(signal_rows.loc[ric,"quoted_spread_bps"]) else .001 for ric in rics])
                covariance,valid,risk=ledoit_covariance_min_history(country_panel,signal_date,rics,a.risk_min_history); risk_fallback=risk.get("fallback")
                exited={ric:w for ric,w in holdings.items() if ric not in set(rics)}; exited_turnover=sum(exited.values()); exited_cost=sum(w*last_cost.get(ric,.001) for ric,w in exited.items())
                w_pre=np.asarray([holdings.get(ric,0.) for ric in rics]); max_weight=max(.05,1/max(len(rics),1))
                target,info=cost_aware_mvo_vector_cost(score,covariance,w_pre,valid,costs,a.risk_aversion,max_weight,a.turnover_cap,exited_turnover,exited_cost)
                if info.get("fallback") and valid.any(): target=np.where(valid,1/valid.sum(),0.)
                trades=np.abs(target-w_pre); turnover=float(trades.sum()+exited_turnover); cost=float(costs@trades+exited_cost); holdings={ric:float(weight) for ric,weight in zip(rics,target) if weight>0}; last_cost={ric:float(c) for ric,c in zip(rics,costs)}; status=info.get("status")
                rebalance_records.append({"country":country,"signal_date":signal_date,"execution_date":date,"assets":len(rics),"valid_risk_assets":risk.get("valid_assets"),"risk_fallback":risk_fallback,"turnover":turnover,"cost":cost,"solver_status":status,"solver_fallback":info.get("fallback")})
            daily_records.append({"country":country,"date":date,"gross_return":pnl,"cost":cost,"net_return":pnl-cost,"turnover":turnover,"positions":len(holdings),"missing_valuations":len(missing_assets),"rebalance_status":status,"risk_fallback":risk_fallback})
    daily=pd.DataFrame(daily_records); rebalance=pd.DataFrame(rebalance_records); missing_df=pd.DataFrame(missing,columns=["country","date","ric","reason"])
    daily.to_parquet(a.run_dir/"daily_portfolio_returns.parquet",index=False); rebalance.to_parquet(a.run_dir/"rebalance_log.parquet",index=False); missing_df.to_parquet(a.run_dir/"missing_valuation_events.parquet",index=False)
    summary,reliability=summarize_daily_portfolio(daily,rebalance); summary.to_csv(a.run_dir/"portfolio_metrics_summary.csv",index=False); reliability.to_csv(a.run_dir/"reliability_metrics.csv",index=False)
    (a.run_dir/"protocol.json").write_text(json.dumps({"timing":"daily state; P&L is applied before close execution; signal t, execute close t+1, label t+2:t+6","cost":"stock-specific lagged end-of-day half-spread, fallback 10 bps","evaluation":"no date dropped for a missing asset valuation","risk_min_history":a.risk_min_history,"risk_aversion":a.risk_aversion},indent=2),encoding="utf-8"); print(summary.to_string(index=False))

if __name__=="__main__": main()
