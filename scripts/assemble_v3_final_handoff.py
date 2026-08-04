"""Assemble a local, report-ready handoff directory from verified runs."""
from __future__ import annotations
import argparse, json, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--method-run',type=Path,default=ROOT/'runs/v3_ptcst_main_seed7_v6'); p.add_argument('--baseline-run',type=Path,default=ROOT/'runs/v3_forecast_baselines_main'); p.add_argument('--portfolio-run',type=Path,default=ROOT/'runs/v3_portfolio_benchmarks_main'); p.add_argument('--ablation-run',type=Path,default=ROOT/'runs/v3_ptcst_ablations_main'); p.add_argument('--stats-run',type=Path,default=ROOT/'runs/v3_statistical_tests_final'); p.add_argument('--robustness-run',type=Path,default=ROOT/'runs/v3_robustness_main'); p.add_argument('--seed-run',type=Path,default=ROOT/'runs/v3_ptcst_seed_sweep'); p.add_argument('--tables-run',type=Path,default=ROOT/'runs/v3_tables_final2'); p.add_argument('--output',type=Path,default=ROOT/'runs/v3_final_handoff'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    sources=[(a.method_run,['protocol_lock.json','environment.json','data_freeze.json','preprocessing.json','imputer.json','scaler.json','best.pt','forecasts.parquet','forecast_metrics_by_date.parquet','weights.parquet','trades.parquet','portfolio_returns.parquet','solver_log.parquet','missing_price_events.parquet','metrics.json','run.log']), (a.stats_run,['statistical_tests.json']), (a.robustness_run,['robustness_results.parquet']), (a.seed_run,['seed_summary.parquet','seed_mean_std.parquet']), (a.tables_run,['run_manifest.json','forecast_summary.csv','portfolio_summary.csv','ptcst_ablation_summary.csv','ptcst_method_summary.csv']), (ROOT/'configs',['v3_main.yaml','v3_robustness.yaml'])]
    copied=[]
    for source,names in sources:
        for name in names:
            src=source/name; dst=a.output/name
            if src.exists(): shutil.copy2(src,dst); copied.append(name)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'method_run':str(a.method_run),'baseline_run':str(a.baseline_run),'portfolio_run':str(a.portfolio_run),'ablation_run':str(a.ablation_run),'files':sorted(set(copied)),'status':'LOCAL_HANDOFF_READY'}; (a.output/'handoff_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
