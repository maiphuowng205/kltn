"""Materialise a pure market-cap Top-100 ASEAN forecast dataset.

This reuses the already-built daily panel (features and targets are unchanged)
and selects only lagged market-cap ranks 1--100.  It intentionally does not
replace the availability-aware Top-100 dataset.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, default=ROOT / "artifacts/asean_v1")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts/asean_v1_pure_top100")
    return p.parse_args()


def split(dates: pd.Series) -> np.ndarray:
    return np.select(
        [dates.between("2019-01-01", "2022-12-31"), dates.between("2023-01-01", "2023-12-31"), dates.between("2024-01-01", "2025-12-31")],
        ["train", "validation", "test"], default="excluded_boundary_or_warmup"
    )


def main() -> None:
    a = parse_args(); source = a.source_root; out = a.output_root
    curated, model_dir, reports = out / "curated", out / "model_ready", out / "reports"
    for p in (curated, model_dir, reports): p.mkdir(parents=True, exist_ok=True)
    panel_path = source / "curated" / "daily_panel.parquet"
    cols = ["ric", "date", "country", "currency", "close", "volume", "bid", "ask", "return", "return_source", "quote_observed", "mid", "quoted_spread", "quoted_spread_bps", "dollar_volume", "session_id", "week", "is_weekly_signal", "rf_daily", "rf_observed", "return_1d", "return_5d", "return_10d", "return_20d", "return_60d", "vol_5d", "vol_20d", "vol_60d", "continuous_60", "log_volume", "log_dollar_volume", "log_price", "high_low_proxy", "amihud", "day_of_week", "is_month_end", "is_quarter_end", "target_excess_return_5d_bps", "execution_date", "universe_month", "market_cap_usd", "market_cap_rank", "eligible", "log_market_cap"]
    panel = pd.read_parquet(panel_path, columns=cols)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["market_cap_rank"] = pd.to_numeric(panel["market_cap_rank"], errors="coerce")
    selected = panel.loc[panel["is_weekly_signal"].eq(True) & panel["market_cap_rank"].le(100)].copy()
    selected["pure_top100"] = True
    selected["split"] = split(selected["date"])
    feature_required = ["return_60d", "vol_60d", "log_dollar_volume", "log_market_cap"]
    selected["feature_complete"] = selected[feature_required].notna().all(axis=1)
    selected["target_available"] = selected[["target_excess_return_5d_bps", "execution_date"]].notna().all(axis=1)
    selected["target_status"] = np.where(selected["target_available"], "available", "missing_exact_next_five_sessions")
    usable = selected.loc[selected["feature_complete"]].copy()
    counts = usable.groupby(["country", "date"])["ric"].nunique()
    full_keys = counts.loc[counts.eq(100)].index
    usable["full_pure_top100"] = pd.MultiIndex.from_frame(usable[["country", "date"]]).isin(full_keys)
    selected.to_parquet(curated / "pure_top100_weekly_all.parquet", index=False)
    usable.to_parquet(model_dir / "weekly_features_targets_pure_top100.parquet", index=False)
    usable.loc[usable["full_pure_top100"]].to_parquet(model_dir / "weekly_features_targets_full_pure_top100.parquet", index=False)
    usable[["country", "date", "ric", "market_cap_usd", "market_cap_rank", "split", "feature_complete", "target_available", "target_status", "full_pure_top100"]].to_parquet(curated / "pure_top100_universe_weekly.parquet", index=False)
    coverage = []
    for country, g in selected.groupby("country"):
        u = usable.loc[usable["country"].eq(country)]
        n_by_date = u.groupby("date")["ric"].nunique()
        coverage.append({"country": country, "pure_selected_rows": len(g), "feature_complete_rows": len(u), "unique_rics": g["ric"].nunique(), "weekly_dates": g["date"].nunique(), "full_100_dates": int(n_by_date.eq(100).sum()), "feature_complete_rate": float(g["feature_complete"].mean()), "target_rate_usable": float(u["target_available"].mean()) if len(u) else 0.0, "median_usable_n": float(n_by_date.median()) if len(n_by_date) else 0.0, "min_usable_n": int(n_by_date.min()) if len(n_by_date) else 0})
    pd.DataFrame(coverage).to_csv(reports / "pure_top100_coverage.csv", index=False)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PURE_MARKET_CAP_TOP100_BUILT", "selection": "exact lagged market-cap ranks 1-100 within each country/month; no availability substitution", "source_panel": str(panel_path), "outputs": {"pure_selected_rows": len(selected), "feature_complete_rows": len(usable), "full_100_rows": int(usable["full_pure_top100"].sum()), "full_100_dates": int(usable.loc[usable["full_pure_top100"], ["country", "date"]].drop_duplicates().shape[0]), "target_rows_usable": int(usable["target_available"].sum())}, "limitations": ["Feature-complete sample can have fewer than 100 names on a signal date; full_100 file excludes those dates.", "No imputation or lower-ranked substitution is used.", "BID/ASK remain historical end-of-day fields, not tick history."]}
    (reports / "pure_top100_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
