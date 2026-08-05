"""Assemble a local, report-ready handoff directory from verified runs."""
from __future__ import annotations
import argparse, json, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """Return a workspace-relative path whenever the source is inside ROOT."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--method-run',type=Path,default=ROOT/'runs/v3_ptcst_main_seed7_v6'); p.add_argument('--baseline-run',type=Path,default=ROOT/'runs/v3_forecast_baselines_main'); p.add_argument('--portfolio-run',type=Path,default=ROOT/'runs/v3_portfolio_benchmarks_main'); p.add_argument('--ablation-run',type=Path,default=ROOT/'runs/v3_ptcst_ablations_main'); p.add_argument('--stats-run',type=Path,default=ROOT/'runs/v3_statistical_tests_final'); p.add_argument('--robustness-run',type=Path,default=ROOT/'runs/v3_robustness_main'); p.add_argument('--seed-run',type=Path,default=ROOT/'runs/v3_ptcst_seed_sweep'); p.add_argument('--tables-run',type=Path,default=ROOT/'runs/v3_tables_final2'); p.add_argument('--leakage-report',type=Path,default=ROOT/'runs/v3_leakage_report.json'); p.add_argument('--determinism-report',type=Path,default=ROOT/'runs/v3_determinism_report.json'); p.add_argument('--risk-run',type=Path,default=ROOT/'runs/v3_risk_coverage_main'); p.add_argument('--cache-run',type=Path,default=ROOT/'artifacts/v3_tensor_cache'); p.add_argument('--output',type=Path,default=ROOT/'runs/v3_final_handoff'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    if not a.method_run.exists() and (ROOT/'runs/v3_ptcst_main_seed7_v8').exists():
        a.method_run = ROOT/'runs/v3_ptcst_main_seed7_v8'
    sources=[
        (a.method_run, [(name, name) for name in ['protocol_lock.json','environment.json','data_freeze.json','config.yaml','preprocessing.json','imputer.json','scaler.json','best.pt','training_history.json','forecasts.parquet','forecast_metrics_by_date.parquet','weights.parquet','trades.parquet','portfolio_returns.parquet','solver_log.parquet','missing_price_events.parquet','metrics.json','run.log']]),
        (a.method_run, [('run_manifest.json', 'method_run_manifest.json')]),
        (a.stats_run, [( 'statistical_tests.json', 'statistical_tests.json')]),
        (a.robustness_run, [('robustness_results.parquet', 'robustness_results.parquet')]),
        (a.seed_run, [('seed_summary.parquet', 'seed_summary.parquet'), ('seed_mean_std.parquet', 'seed_mean_std.parquet')]),
        (a.tables_run, [(name, name) for name in ['run_manifest.json','forecast_summary.csv','portfolio_summary.csv','ptcst_ablation_summary.csv','ptcst_method_summary.csv']]),
        (a.leakage_report.parent, [('v3_leakage_report.json', 'v3_leakage_report.json')]),
        (a.determinism_report.parent, [('v3_determinism_report.json', 'v3_determinism_report.json')]),
        (a.risk_run, [('risk_coverage_by_date.parquet', 'risk_coverage_by_date.parquet'), ('risk_coverage_summary.parquet', 'risk_coverage_summary.parquet'), ('metrics.json', 'risk_coverage_metrics.json')]),
        (a.cache_run, [('cache_manifest.json', 'tensor_cache_manifest.json'), ('preprocessing.json', 'tensor_cache_preprocessing.json')]),
        (ROOT/'configs', [('v3_main.yaml', 'v3_main.yaml'), ('v3_robustness.yaml', 'v3_robustness.yaml')]),
    ]
    copied=[]
    for source,names in sources:
        for source_name, output_name in names:
            src=source/source_name; dst=a.output/output_name
            if src.exists():
                shutil.copy2(src, dst)
                copied.append(output_name)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'method_run':portable_path(a.method_run),'baseline_run':portable_path(a.baseline_run),'portfolio_run':portable_path(a.portfolio_run),'ablation_run':portable_path(a.ablation_run),'files':sorted(set(copied)),'status':'LOCAL_HANDOFF_READY'}; (a.output/'handoff_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
