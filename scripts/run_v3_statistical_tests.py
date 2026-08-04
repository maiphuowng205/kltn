"""Date-level DM/HLN-style and paired block-bootstrap comparisons."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def load_forecast(spec: str) -> pd.DataFrame:
    name, raw = spec.split("=", 1); frame = pd.read_parquet(Path(raw) / "forecasts.parquet"); frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "prediction" not in frame: frame["prediction"] = frame["prediction_excess_return_5d_bps"]
    target_col = "target" if "target" in frame else "target_excess_return_5d_bps"
    frame["target_excess_return_5d_bps"] = frame[target_col]
    frame["label"] = name
    return frame.loc[frame["split"].eq("test"), ["label", "date", "ric", "prediction", "target_excess_return_5d_bps", "target_available"]]

def dm_hln(difference: np.ndarray) -> dict[str, float | int | None]:
    d = np.asarray(difference, dtype=float); d = d[np.isfinite(d)]; n = len(d)
    if n < 3: return {"n_dates": n, "mean_loss_difference": None, "t_stat": None, "p_value": None}
    lag = max(1, min(12, int(np.floor(n ** (1 / 3)))))
    centered = d - d.mean(); long_var = float(np.dot(centered, centered) / n)
    for k in range(1, min(lag, n - 1) + 1): long_var += 2 * (1 - k / (lag + 1)) * float(np.dot(centered[k:], centered[:-k]) / n)
    se = np.sqrt(max(long_var, 1e-16) / n); t_stat = float(d.mean() / se)
    # Harvey-Leybourne-Newbold small-sample correction for a one-step/date loss.
    hln = np.sqrt((n + 1 - 2 * 1 + 1 * 0) / n) if n > 1 else 1.0
    corrected = t_stat * hln; p = float(2 * stats.t.sf(abs(corrected), df=max(n - 1, 1)))
    return {"n_dates": n, "hac_lag": lag, "mean_loss_difference": float(d.mean()), "t_stat": corrected, "p_value": p}

def block_bootstrap(difference: np.ndarray, seed: int, repetitions: int = 2000) -> dict[str, float | int]:
    d = np.asarray(difference, dtype=float); d = d[np.isfinite(d)]; n = len(d)
    if n < 4: return {"n_dates": n, "repetitions": repetitions, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed); block = min(12, max(4, n // 8)); samples=[]
    for _ in range(repetitions):
        take=[]
        while len(take)<n:
            start=int(rng.integers(0,n)); take.extend(d[(start+np.arange(block))%n].tolist())
        samples.append(np.mean(take[:n]))
    samples=np.asarray(samples)
    null=np.asarray([np.mean(d*rng.choice(np.array([-1.0,1.0]),size=n)) for _ in range(repetitions)])
    return {"n_dates":n,"repetitions":repetitions,"block_length":block,"ci_low":float(np.quantile(samples,.025)),"ci_high":float(np.quantile(samples,.975)),"p_value":float(np.mean(np.abs(null)>=abs(d.mean())))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--forecast-runs',nargs='+',required=True,help='label=run_dir'); p.add_argument('--run-dir',type=Path,default=Path('runs/v3_statistical_tests')); p.add_argument('--seed',type=int,default=7); p.add_argument('--repetitions',type=int,default=2000); a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=True)
    frames=[]
    for spec in a.forecast_runs: frames.append(load_forecast(spec))
    data=pd.concat(frames,ignore_index=True); labels=data.label.unique().tolist(); results=[]
    for i,left in enumerate(labels):
        for right in labels[i+1:]:
            l=data.loc[data.label.eq(left),['date','ric','prediction','target_excess_return_5d_bps','target_available']].rename(columns={'prediction':'p_left','target_excess_return_5d_bps':'target_left'}); r=data.loc[data.label.eq(right),['date','ric','prediction','target_excess_return_5d_bps','target_available']].rename(columns={'prediction':'p_right','target_excess_return_5d_bps':'target_right'}); m=l.merge(r,on=['date','ric'],suffixes=('','_r')); m=m.loc[m.target_available & m.target_available_r]; m['loss_difference']=(m.p_left-m.target_left)**2-(m.p_right-m.target_right)**2; date_diff=m.groupby('date').loss_difference.mean().to_numpy(); results.append({'left':left,'right':right,'dm_hln':dm_hln(date_diff),'block_bootstrap':block_bootstrap(date_diff,a.seed,a.repetitions)})
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'comparisons':results,'inference_unit':'forecast date','loss':'squared error in bps'}; (a.run_dir/'statistical_tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
