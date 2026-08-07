"""Collect comparable temporal architecture runs into an extension table."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def read_run(model: str, run_dir: Path) -> list[dict[str, object]]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    forecast = pd.read_parquet(run_dir / "forecast_metrics_by_date.parquet")
    test = forecast.loc[forecast["split"].astype(str).eq("test")]
    test = test.loc[test["n_assets"] >= 3]
    portfolio = pd.read_parquet(run_dir / "portfolio_returns.parquet")
    portfolio = portfolio.loc[portfolio["evaluation_available"].astype(bool)]
    row = {
        "model": model,
        "source_run": str(run_dir),
        "seed": int(metrics.get("seed", metrics.get("train", {}).get("seed", 7))),
        "device": metrics.get("train", {}).get("device"),
        "best_epoch": int(metrics.get("train", {}).get("best_epoch", 0)),
        "validation_spearman_ic": float(metrics.get("train", {}).get("best_validation_spearman_ic")),
        "test_dates": int(metrics.get("test_dates", len(test))),
        "evaluation_dates": int(metrics.get("evaluation_dates", len(portfolio))),
        "test_spearman_ic": float(test["spearman_ic"].mean()),
        "test_pearson_ic": float(test["pearson_ic"].mean()),
        "test_mae_bps": float(test["mae_bps"].mean()),
        "test_rmse_bps": float(test["rmse_bps"].mean()),
        "test_directional_accuracy": float(test["directional_accuracy"].mean()),
        "test_top_minus_bottom_bps": float(test["top_minus_bottom_bps"].mean()),
        "mean_net_excess_5d": float(portfolio["net_excess_return_5d"].mean()),
        "net_sharpe_annualized": float(portfolio["net_excess_return_5d"].mean() / portfolio["net_excess_return_5d"].std(ddof=1) * 52 ** 0.5),
        "mean_turnover": float(portfolio["turnover_l1"].mean()),
    }
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptcst-run", type=Path, required=True)
    parser.add_argument("--temporal-run", type=Path, required=True)
    parser.add_argument("--patchtst-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v3_extension_p0/architecture_ablation"))
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    specs = [("TemporalTransformer", args.temporal_run), ("PatchTST", args.patchtst_run), ("PTCST", args.ptcst_run)]
    rows = [row for model, run_dir in specs for row in read_run(model, run_dir)]
    summary = pd.DataFrame(rows).sort_values("model")
    summary.to_csv(args.run_dir / "architecture_ablation_summary.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "same frozen V3 data, fixed train/validation/test split, seed 7 and 100-epoch budget",
        "models": [model for model, _ in specs],
        "source_runs": {model: str(run_dir) for model, run_dir in specs},
        "artifacts": ["architecture_ablation_summary.csv"],
        "status": "COMPLETED_FROM_VERIFIED_LOCAL_RUNS",
        "note": "This is a separate extension handoff; it does not overwrite the Colab V3 final handoff.",
    }
    (args.run_dir / "architecture_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
