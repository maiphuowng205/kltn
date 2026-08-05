"""Run the locked vanilla Transformer and PatchTST deep baselines."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=ROOT/'data/lseg_v3'); p.add_argument('--run-root',type=Path,default=ROOT/'runs/v3_deep_baselines'); p.add_argument('--epochs',type=int,default=100); p.add_argument('--checkpoint-sync-root',type=Path,default=None); a=p.parse_args(); a.run_root.mkdir(parents=True,exist_ok=True)
    for model in ('TemporalTransformer','PatchTST'):
        run=a.run_root/model
        if not (run/'metrics.json').exists():
            command=[sys.executable,str(ROOT/'scripts'/'run_v3_ptcst_method.py'),'--model-type',model,'--data-root',str(a.data_root),'--run-dir',str(run),'--epochs',str(a.epochs),'--early-stopping-patience','10','--batch-dates','32','--seed','7']
            if a.checkpoint_sync_root is not None:
                sync_dir=a.checkpoint_sync_root/model; command += ['--checkpoint-sync-dir',str(sync_dir)]
                resume=sync_dir/'last.pt'
                if resume.exists(): command += ['--resume-checkpoint',str(resume)]
            subprocess.run(command,check=True)
    print('completed', [str(a.run_root/m/'metrics.json') for m in ('TemporalTransformer','PatchTST')])
if __name__=='__main__': main()
