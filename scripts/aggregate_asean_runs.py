"""Create compact ASEAN experiment tables and a reproducibility manifest."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


COUNTRIES = ["indonesia", "malaysia", "philippines", "singapore", "thailand"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--country-root", type=Path, default=Path("artifacts/asean_v1_country_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/asean_v1_aggregate"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecast_rows, portfolio_rows, ablation_rows, observed_cost_rows, deep_rows, risk_rows, quote_rows = [], [], [], [], [], [], []
    required: list[Path] = []
    for country in COUNTRIES:
        root = args.country_root / country
        fp = root / "runs" / "v3_forecast_baselines" / "forecast_metrics_summary.parquet"
        if fp.exists():
            df = pd.read_parquet(fp); df.insert(0, "country", country); forecast_rows.append(df)
            required.append(fp)
        for rel, bucket, kind in [
            ("runs/v3_portfolio_benchmarks/portfolio_metrics_summary.parquet", portfolio_rows, "benchmark"),
            ("runs/v3_ptcst_ablations/portfolio_metrics_summary.parquet", ablation_rows, "ablation"),
        ]:
            p = root / rel
            if p.exists():
                df = pd.read_parquet(p); df.insert(0, "country", country); df.insert(1, "experiment", kind); bucket.append(df); required.append(p)
        p = root / "runs" / "v3_ptcst_cost_observed" / "portfolio_metrics_summary.parquet"
        if p.exists():
            df = pd.read_parquet(p); df.insert(0, "country", country); df.insert(1, "experiment", "observed_half_spread_cost"); observed_cost_rows.append(df); required.append(p)
        for model in ("TemporalTransformer", "PatchTST"):
            p = root / "runs" / "v3_deep_baselines" / model / "metrics.json"
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8")); row = {"country": country, "model": model, **{k: obj.get(k) for k in ("test_dates", "evaluation_dates", "excluded_incomplete_dates", "mean_net_excess_5d", "net_sharpe_annualized", "mean_turnover")}, "best_epoch": obj.get("train", {}).get("best_epoch"), "best_validation_spearman_ic": obj.get("train", {}).get("best_validation_spearman_ic")}; deep_rows.append(row); required.append(p)
        p = root / "runs" / "v3_ptcst_ca_mvo" / "metrics.json"
        obj = json.loads(p.read_text(encoding="utf-8")); deep_rows.append({"country": country, "model": "PTCST", **{k: obj.get(k) for k in ("test_dates", "evaluation_dates", "excluded_incomplete_dates", "mean_net_excess_5d", "net_sharpe_annualized", "mean_turnover")}, "best_epoch": obj.get("train", {}).get("best_epoch"), "best_validation_spearman_ic": obj.get("train", {}).get("best_validation_spearman_ic")}); required.append(p)
        p = root / "runs" / "v3_risk_coverage" / "metrics.json"; obj = json.loads(p.read_text(encoding="utf-8")); risk_rows.append({"country": country, "dates_total": obj.get("dates_total"), "fallback_dates_total": obj.get("fallback_dates_total"), "fallback_fraction": obj.get("fallback_fraction")}); required.append(p)
        p = root / "runs" / "v3_quote_cost_audit" / "quote_cost_metrics.json"; obj = json.loads(p.read_text(encoding="utf-8")); quote_rows.append(obj); required.append(p)

    tables = {
        "forecast_summary.csv": pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame(),
        "portfolio_benchmarks.csv": pd.concat(portfolio_rows, ignore_index=True) if portfolio_rows else pd.DataFrame(),
        "ptcst_ablations.csv": pd.concat(ablation_rows, ignore_index=True) if ablation_rows else pd.DataFrame(),
        "ptcst_observed_cost_sensitivity.csv": pd.concat(observed_cost_rows, ignore_index=True) if observed_cost_rows else pd.DataFrame(),
        "deep_model_summary.csv": pd.DataFrame(deep_rows),
        "risk_coverage_summary.csv": pd.DataFrame(risk_rows),
        "quote_cost_summary.csv": pd.DataFrame(quote_rows),
    }
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / name, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "countries": COUNTRIES,
        "main_dataset": "ASEAN availability-aware Top-100, staged one country at a time",
        "cost_rule": "portfolio policy assumes 10 bps one-way; quote audit uses end-of-day BID/ASK half-spread",
        "files": {str(p.relative_to(args.country_root.parent)): sha256(p) for p in required if p.exists()},
        "outputs": sorted(name for name in tables),
        "limitations": ["Philippines and Singapore have covariance-history fallback dates; inspect risk_coverage_summary.csv.", "BID/ASK are end-of-day and do not establish tick-level implementation shortfall.", "Zero forecast rank-spread is not economically meaningful because predictions are tied."],
    }
    (args.output_dir / "asean_experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "tables": list(tables), "required_files": len(required)}, indent=2))


if __name__ == "__main__":
    main()
