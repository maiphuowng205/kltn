"""Run and aggregate the locked five-seed PTCST forecast sweep."""
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=ROOT/'data/lseg_v3'); p.add_argument('--run-root',type=Path,default=ROOT/'runs/v3_ptcst_seed_sweep'); p.add_argument('--seeds',nargs='+',type=int,default=[7,19,43,71,101]); p.add_argument('--epochs',type=int,default=100); p.add_argument('--checkpoint-sync-root',type=Path,default=None); a=p.parse_args(); a.run_root.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        run=a.run_root/f'seed_{seed}'
        if not (run/'metrics.json').exists():
            command=[sys.executable,str(ROOT/'scripts'/'run_v3_ptcst_method.py'),'--model-type','PTCST','--data-root',str(a.data_root),'--run-dir',str(run),'--epochs',str(a.epochs),'--early-stopping-patience','10','--batch-dates','32','--seed',str(seed)]
            if a.checkpoint_sync_root is not None:
                sync_dir=a.checkpoint_sync_root/f'seed_{seed}'; command += ['--checkpoint-sync-dir',str(sync_dir)]
                resume=sync_dir/'last.pt'
                if resume.exists(): command += ['--resume-checkpoint',str(resume)]
            subprocess.run(command,check=True)
    reports=[]
    for seed in a.seeds:
        report=json.loads((a.run_root/f'seed_{seed}'/'metrics.json').read_text()); reports.append({'seed':seed,'best_validation_spearman_ic':report['train']['best_validation_spearman_ic'],'mean_net_excess_5d':report['mean_net_excess_5d'],'net_sharpe_annualized':report['net_sharpe_annualized'],'mean_turnover':report['mean_turnover'],'evaluation_dates':report['evaluation_dates']})
    table=pd.DataFrame(reports); table.to_parquet(a.run_root/'seed_summary.parquet',index=False); summary=table.drop(columns='seed').agg(['mean','std']).reset_index(); summary.to_parquet(a.run_root/'seed_mean_std.parquet',index=False); out={'created_at_utc':datetime.now(timezone.utc).isoformat(),'seeds':a.seeds,'completed':len(table),'mean_std_artifact':'seed_mean_std.parquet'}; (a.run_root/'metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(table.to_string(index=False)); print(summary.to_string(index=False)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
