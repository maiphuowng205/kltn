"""Prepare one V3-compatible staging root per ASEAN country.

The existing V3 experiment scripts operate on one fixed 100-name market at a
time. This adapter keeps the ASEAN panel intact while giving each country its
own contract-compatible root; no country rows are mixed in a covariance or
cross-sectional tensor.
"""
from __future__ import annotations

import hashlib
import json
import argparse
import gc
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = ["Indonesia", "Malaysia", "Philippines", "Singapore", "Thailand"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", choices=COUNTRIES, default=None)
    args = parser.parse_args()
    source = ROOT / "artifacts/asean_v1"
    output = ROOT / "artifacts/asean_v1_country_runs"
    daily_path = source / "curated/daily_panel.parquet"
    model_path = source / "model_ready/weekly_features_targets_full_top100.parquet"
    daily_columns = pd.read_parquet(daily_path, columns=None).columns.tolist()
    model_columns = pd.read_parquet(model_path, columns=None).columns.tolist()
    rows = []
    countries = [args.country] if args.country else COUNTRIES
    for country in countries:
        root = output / country.lower()
        data = root / "data/lseg_v3"
        for p in [data / "curated", data / "model_ready", data / "reports"]:
            p.mkdir(parents=True, exist_ok=True)
        daily = pd.read_parquet(daily_path, columns=daily_columns, filters=[["country", "=", country]])
        model = pd.read_parquet(model_path, columns=model_columns, filters=[["country", "=", country]])
        if daily.empty or model.empty:
            raise RuntimeError(f"No staging rows for {country}: daily={len(daily)} model={len(model)}")
        weekly_cols = [c for c in ["date", "ric", "market_cap_usd", "market_cap_rank", "split", "target_available", "target_status", "country"] if c in model.columns]
        weekly = model[weekly_cols].copy()
        daily_out = data / "curated/daily_panel.parquet"; weekly_out = data / "curated/universe_weekly.parquet"; model_out = data / "model_ready/weekly_features_targets.parquet"
        daily.to_parquet(daily_out, index=False); weekly.to_parquet(weekly_out, index=False); model.to_parquet(model_out, index=False)
        files = []
        for path in [daily_out, weekly_out, model_out]:
            files.append({"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)})
        manifest = {"freeze_id": f"ASEAN_{country.upper()}_TOP100_{datetime.now(timezone.utc).date().isoformat()}", "file_count": len(files), "files": files}
        (data / "reports/freeze_manifest_v3.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        rows.append({"country": country, "workspace": str(root), "daily_rows": len(daily), "weekly_rows": len(weekly), "model_rows": len(model), "daily_rics": daily.ric.nunique(), "weekly_dates": weekly.date.nunique()})
        print(country, rows[-1], flush=True)
        del daily, model, weekly
        gc.collect()
    (output / "staging_manifest.json").write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(), "countries": rows}, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
