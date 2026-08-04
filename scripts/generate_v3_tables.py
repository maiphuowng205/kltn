"""Collect generated forecast/portfolio summaries into report-ready tables."""
from __future__ import annotations
import argparse, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,default=Path('runs/v3_tables')); p.add_argument('--baseline-run',type=Path,required=True); p.add_argument('--portfolio-run',type=Path,required=True); p.add_argument('--method-run',type=Path,required=True); p.add_argument('--ablation-run',type=Path,required=True); a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=True)
    forecast=pd.read_parquet(a.baseline_run/'forecast_metrics_summary.parquet'); portfolio=pd.read_parquet(a.portfolio_run/'portfolio_metrics_summary.parquet'); method=pd.read_json(a.method_run/'metrics.json',typ='series').to_frame('value').reset_index(names='metric'); ablation=pd.read_parquet(a.ablation_run/'portfolio_metrics_summary.parquet')
    forecast.to_csv(a.run_dir/'forecast_summary.csv',index=False); portfolio.to_csv(a.run_dir/'portfolio_summary.csv',index=False); ablation.to_csv(a.run_dir/'ptcst_ablation_summary.csv',index=False); method.to_csv(a.run_dir/'ptcst_method_summary.csv',index=False)
    try: git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    except Exception: git_commit=None
    freeze=json.loads((a.method_run/'data_freeze.json').read_text(encoding='utf-8')) if (a.method_run/'data_freeze.json').exists() else {}
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'git_commit':git_commit,'freeze_id':freeze.get('freeze_id'),'sources':{'baseline':str(a.baseline_run),'portfolio':str(a.portfolio_run),'method':str(a.method_run),'ablation':str(a.ablation_run)},'tables':['forecast_summary.csv','portfolio_summary.csv','ptcst_ablation_summary.csv','ptcst_method_summary.csv']}; (a.run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
