"""Leakage-safe Ridge forecast baseline and equal-weight Top-20 backtest for VN V3."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lseg_v3"
OUT = ROOT / "runs" / "v3_ridge_top20"
FEATURES = ["return_1d", "return_5d", "return_10d", "return_20d", "return_60d", "vol_5d", "vol_20d", "vol_60d", "log_volume", "log_dollar_volume", "log_price", "log_market_cap", "high_low_proxy", "amihud", "day_of_week", "is_month_end", "is_quarter_end"]

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA / "model_ready" / "weekly_features_targets.parquet")
    train = df.loc[(df.split == "train") & df.target_available].copy()
    # Imputation statistics are estimated from training data only.
    medians = train[FEATURES].median()
    train_x = train[FEATURES].fillna(medians)
    full_x = df[FEATURES].fillna(medians)
    scaler = RobustScaler().fit(train_x)
    model = Ridge(alpha=10.0).fit(scaler.transform(train_x), train.target_excess_return_5d_bps)
    df["prediction_excess_return_5d_bps"] = model.predict(scaler.transform(full_x))
    df.to_parquet(OUT / "forecasts.parquet", index=False)
    rows=[]; previous=set()
    for date, g in df.loc[df.split.isin(["validation", "test"])].groupby("date", sort=True):
        tradeable=g.loc[g.target_available].nlargest(20,"prediction_excess_return_5d_bps")
        if tradeable.empty:
            continue
        names=set(tradeable.ric)
        turnover=len(names.symmetric_difference(previous))/20 if previous else 1.0
        gross=tradeable.target_excess_return_5d_bps.mean()/10000.0
        cost=0.001*turnover
        rows.append({"date":date,"split":tradeable.split.iloc[0],"assets":len(tradeable),"gross_excess_return_5d":gross,"turnover_l1":turnover,"cost":cost,"net_excess_return_5d":gross-cost})
        previous=names
    bt=pd.DataFrame(rows); bt.to_parquet(OUT / "portfolio_returns.parquet",index=False)
    summary={}
    for key,g in bt.groupby("split"):
        r=g.net_excess_return_5d
        summary[key]={"weeks":len(g),"mean_net_5d":float(r.mean()),"annualized_net_return_approx":float((1+r.mean())**52-1),"annualized_sharpe_approx":float(r.mean()/r.std(ddof=1)*np.sqrt(52)) if r.std(ddof=1)>0 else None,"mean_turnover":float(g.turnover_l1.mean()),"mean_cost":float(g.cost.mean())}
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"model":"Ridge(alpha=10), median imputation and RobustScaler fit only on train labels","strategy":"weekly equal-weight top 20 predicted assets; all returns net of 10 bps one-way assumed cost","features":FEATURES,"results":summary}
    (OUT / "metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__ == "__main__": main()
