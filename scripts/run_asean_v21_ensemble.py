"""Build the V2.1 rank-normalized seed ensemble and validation-only calibration."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri

def rank_normalize(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out=np.zeros_like(scores,dtype=float)
    for i in range(scores.shape[0]):
        valid=np.flatnonzero(mask[i]); n=len(valid)
        if n<2: continue
        order=valid[np.argsort(scores[i,valid],kind="mergesort")]
        ranks=np.empty(n,dtype=float); ranks[np.argsort(np.argsort(scores[i,valid],kind="mergesort"),kind="mergesort")]=(np.arange(n)+.5)/n
        out[i,valid]=ndtri(np.clip(ranks,.001,.999))
    return out

def load_seeds(root: Path, split: str, seeds: list[int]):
    blobs=[np.load(root/f"seed_{seed}"/f"{split}_predictions.npz",allow_pickle=False) for seed in seeds]
    first=blobs[0]; scores=np.stack([b["raw_score"].astype(float) for b in blobs],axis=-1)
    masks=np.stack([b["asset_mask"].astype(bool) for b in blobs],axis=-1)
    common=masks.all(axis=-1)
    if not np.array_equal(first["dates"],blobs[-1]["dates"]): raise ValueError(f"Seed date grids differ for {split}")
    if not np.array_equal(first["rics"],blobs[-1]["rics"]): raise ValueError(f"Seed asset grids differ for {split}")
    normalized=np.stack([rank_normalize(scores[...,s],masks[...,s]) for s in range(scores.shape[-1])],axis=-1)
    ensemble=normalized.mean(axis=-1)
    return first,ensemble,common

def country_calibration(validation, val_score, val_mask):
    rows=[]; betas={}
    for country in sorted(set(validation["countries"].astype(str))):
        z_values=[]; y_values=[]
        for i,name in enumerate(validation["countries"].astype(str)):
            if name!=country: continue
            m=val_mask[i]; p=val_score[i,m]; y=validation["target_bps"][i,m]
            if len(p)<3 or p.std()<=1e-12: continue
            z=(p-p.mean())/p.std(); z_values.extend(z); y_values.extend(y/10000.0)
        z=np.asarray(z_values); y=np.asarray(y_values); beta=float((z@y)/(z@z)) if len(z) and z@z>0 else 0.0
        usable=bool(beta>0); betas[country]=max(beta,0.0); rows.append({"country":country,"beta_decimal_per_z":beta,"beta_positive":max(beta,0.0),"usable_positive_signal":usable,"validation_observations":int(len(z))})
    return betas,pd.DataFrame(rows)

def apply_country_calibration(bundle, score, mask, betas):
    alpha=np.zeros_like(score,dtype=float)
    for i,country in enumerate(bundle["countries"].astype(str)):
        m=mask[i]; p=score[i,m]
        if m.sum()>=2 and p.std()>1e-12: alpha[i,m]=betas.get(country,0.0)*(p-p.mean())/p.std()
    return alpha

def deciles(bundle, score, mask):
    rows=[]
    for i,(country,date) in enumerate(zip(bundle["countries"].astype(str),pd.to_datetime(bundle["dates"]))):
        m=mask[i]; n=int(m.sum())
        if n<10: continue
        order=np.flatnonzero(m)[np.argsort(score[i,m],kind="mergesort")]
        for decile,indices in enumerate(np.array_split(order,10),1):
            rows.append({"country":country,"date":date,"decile":decile,"n_assets":len(indices),"mean_target_bps":float(np.nanmean(bundle["target_bps"][i,indices]))})
    table=pd.DataFrame(rows)
    if table.empty:return table,pd.DataFrame()
    summary=[]
    for country,g in table.groupby("country"):
        pivot=g.groupby("decile").mean(numeric_only=True).sort_index(); d1=float(pivot.loc[1,"mean_target_bps"]); d10=float(pivot.loc[10,"mean_target_bps"]); observed=pivot["mean_target_bps"].to_numpy(float); expected=np.arange(1,len(observed)+1,dtype=float); mono=float(np.corrcoef(observed,expected)[0,1]) if len(observed)>1 and np.std(observed)>0 else np.nan
        summary.append({"country":country,"d10_minus_d1_bps":d10-d1,"decile_monotonicity":mono})
    return table,pd.DataFrame(summary)

def save_bundle(path, bundle, score, mask, alpha=None):
    values={"dates":bundle["dates"],"countries":bundle["countries"],"rics":bundle["rics"],"raw_score":score,"target_bps":bundle["target_bps"],"target_mask":bundle["target_mask"],"asset_mask":mask}
    if alpha is not None: values["calibrated_alpha_decimal"]=alpha
    np.savez_compressed(path,**values)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--seeds",default="7,19,31,43,59"); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(); seeds=[int(x) for x in a.seeds.split(",")]; a.output_dir.mkdir(parents=True,exist_ok=True)
    val,val_score,val_mask=load_seeds(a.run_root,"validation",seeds); dev,dev_score,dev_mask=load_seeds(a.run_root,"development",seeds)
    betas,calibration=country_calibration(val,val_score,val["target_mask"].astype(bool)&val_mask); dev_alpha=apply_country_calibration(dev,dev_score,dev["target_mask"].astype(bool)&dev_mask,betas); val_alpha=apply_country_calibration(val,val_score,val["target_mask"].astype(bool)&val_mask,betas)
    save_bundle(a.output_dir/"validation_ensemble.npz",val,val_score,val_mask,val_alpha); save_bundle(a.output_dir/"development_ensemble.npz",dev,dev_score,dev_mask,dev_alpha)
    val_decile,val_decile_summary=deciles(val,val_score,val["target_mask"].astype(bool)&val_mask); dev_decile,dev_decile_summary=deciles(dev,dev_score,dev["target_mask"].astype(bool)&dev_mask)
    calibration.to_csv(a.output_dir/"calibration_summary.csv",index=False); val_decile.to_parquet(a.output_dir/"validation_decile_returns.parquet",index=False); dev_decile.to_parquet(a.output_dir/"development_decile_returns.parquet",index=False); val_decile_summary.to_csv(a.output_dir/"validation_decile_summary.csv",index=False); dev_decile_summary.to_csv(a.output_dir/"development_decile_summary.csv",index=False)
    (a.output_dir/"ensemble_manifest.json").write_text(json.dumps({"created_at_utc":datetime.now(timezone.utc).isoformat(),"seeds":seeds,"rank_normalization":"within country/date/seed; normal-score inverse CDF then average","calibration":"country-specific positive slope fit on validation only; beta<=0 mapped to unusable/no-alpha","status":"development_outputs_only"},indent=2),encoding="utf-8")
    print(calibration.to_string(index=False)); print(dev_decile_summary.to_string(index=False))
if __name__=="__main__": main()
