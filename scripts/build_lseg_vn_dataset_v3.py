"""Build a separate, point-in-time-aware Vietnam V3 model-ready dataset.

Inputs are the LSEG historical HOSE/HNX monthly snapshots, daily OHLCV / total
return extracts, and the VND cash-proxy RF series.  V2 is never read or
overwritten.  A month-end constituent / market-cap snapshot only becomes
available in the following calendar month, preventing look-ahead.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START, END = pd.Timestamp("2018-01-02"), pd.Timestamp("2025-12-31")
TOP_N, COST_ONE_WAY = 100, 0.001  # Explicit assumed 10 bps one-way cost.


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Vietnam LSEG Dataset V3 without changing V2.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "lseg_v3")
    parser.add_argument("--rf-file", type=Path, default=ROOT / "data" / "external" / "risk_free_daily.parquet")
    return parser.parse_args()


def read_daily(root: Path) -> pd.DataFrame:
    files = sorted((root / "raw" / "daily_history").glob("daily_*.parquet"))
    if len(files) != 50:
        raise RuntimeError(f"Expected 50 daily LSEG partitions, found {len(files)}")
    cols = ["ric", "date", "open", "high", "low", "close", "volume", "total_return_pct", "extract_timestamp_utc"]
    df = pd.concat([pd.read_parquet(path, columns=cols) for path in files], ignore_index=True)
    df["ric"] = df["ric"].fillna("").astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.loc[df["ric"].ne("") & df["date"].between(START, END)].copy()
    for col in ("open", "high", "low", "close", "volume", "total_return_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # LSEG occasionally returns repeated response rows.  Keep one deterministic
    # row per RIC-date before feature or label construction.
    df = df.sort_values(["ric", "date", "extract_timestamp_utc"]).drop_duplicates(["ric", "date"], keep="last")
    valid = df[["open", "high", "low", "close", "volume"]].notna().all(axis=1) & df["close"].gt(0) & df["volume"].ge(0)
    df = df.loc[valid].copy()
    df["return"] = df["total_return_pct"] / 100.0
    return df.sort_values(["ric", "date"]).reset_index(drop=True)


def build_universe(root: Path) -> pd.DataFrame:
    path = root / "raw" / "historical_universe_monthly.parquet"
    u = pd.read_parquet(path)
    u["ric"] = u["ric"].fillna("").astype(str).str.strip()
    u["snapshot_date"] = pd.to_datetime(u["snapshot_date"], errors="coerce").dt.normalize()
    u = u.loc[u["ric"].ne("") & u["snapshot_date"].notna()].copy()
    u["market_cap_vnd"] = pd.to_numeric(u["market_cap_vnd"], errors="coerce")
    # Snapshot at the end of M may be known only during M+1.
    u["universe_month"] = (u["snapshot_date"].dt.to_period("M") + 1).astype(str)
    u = u.sort_values(["universe_month", "ric", "market_cap_vnd"]).drop_duplicates(["universe_month", "ric"], keep="last")
    u["market_cap_rank"] = u.groupby("universe_month")["market_cap_vnd"].rank(ascending=False, method="first")
    u["eligible"] = u["market_cap_vnd"].gt(0)
    u["selected_top100"] = u["eligible"] & u["market_cap_rank"].le(TOP_N)
    cols = ["instrument", "ric", "company_name", "market_cap_vnd", "trbc_sector", "isin", "snapshot_date", "exchange", "source_index_ric", "extract_timestamp_utc", "universe_month", "market_cap_rank", "eligible", "selected_top100"]
    return u.loc[:, cols].sort_values(["universe_month", "market_cap_rank"]).reset_index(drop=True)


def master_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    cal = pd.DataFrame({"date": sorted(daily["date"].unique())})
    cal["session_id"] = np.arange(len(cal), dtype="int32")
    cal["week"] = cal["date"].dt.to_period("W-FRI").astype(str)
    cal["is_weekly_signal"] = cal.groupby("week")["date"].transform("max").eq(cal["date"])
    return cal


def features_and_targets(daily: pd.DataFrame, calendar: pd.DataFrame, rf_path: Path) -> pd.DataFrame:
    if not rf_path.exists():
        raise FileNotFoundError(rf_path)
    rf = pd.read_parquet(rf_path, columns=["date", "rf_daily"])
    rf["date"] = pd.to_datetime(rf["date"], errors="coerce").dt.normalize()
    rf["rf_daily"] = pd.to_numeric(rf["rf_daily"], errors="coerce")
    df = daily.merge(calendar[["date", "session_id"]], on="date", how="left", validate="many_to_one")
    df = df.merge(rf, on="date", how="left", validate="many_to_one")
    df["rf_daily"] = df["rf_daily"].fillna(0.0)
    df["dollar_volume"] = df["close"] * df["volume"]
    parts = []
    for ric, g in df.groupby("ric", sort=False):
        g = g.sort_values("date").copy()
        contiguous = g["session_id"].diff().eq(1)
        g["return_1d"] = g["return"]
        for window in (5, 10, 20, 60):
            g[f"return_{window}d"] = (1.0 + g["return"]).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        for window in (5, 20, 60):
            g[f"vol_{window}d"] = g["return"].rolling(window, min_periods=window).std(ddof=0)
        # Reject rolling values that bridge a suspension/missing master session.
        g["continuous_60"] = contiguous.astype("int16").rolling(60, min_periods=60).sum().eq(60)
        g.loc[~g["continuous_60"], ["return_60d", "vol_60d"]] = np.nan
        g["log_volume"] = np.log1p(g["volume"])
        g["log_dollar_volume"] = np.log1p(g["dollar_volume"])
        g["log_price"] = np.log(g["close"])
        g["high_low_proxy"] = g["high"] / g["low"].where(g["low"].gt(0)) - 1.0
        g["amihud"] = g["return"].abs() / g["dollar_volume"].replace(0, np.nan)
        g["day_of_week"] = g["date"].dt.dayofweek.astype("int8")
        g["is_month_end"] = g["date"].dt.is_month_end.astype("int8")
        g["is_quarter_end"] = g["date"].dt.is_quarter_end.astype("int8")
        # Target is exactly the next five *master-market* sessions; suspensions
        # are not bridged or imputed.
        future_stock = pd.Series(1.0, index=g.index)
        future_rf = pd.Series(1.0, index=g.index)
        exact_future = pd.Series(True, index=g.index)
        for step in range(1, 6):
            future_stock *= 1.0 + g["return"].shift(-step)
            future_rf *= 1.0 + g["rf_daily"].shift(-step)
            exact_future &= g["session_id"].shift(-step).eq(g["session_id"] + step)
        g["target_excess_return_5d_bps"] = np.where(exact_future, (future_stock - future_rf) * 10000.0, np.nan)
        g["execution_date"] = g["date"].shift(-1).where(g["session_id"].shift(-1).eq(g["session_id"] + 1))
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def split(date: pd.Series) -> np.ndarray:
    return np.select(
        [date.between("2019-01-01", "2022-12-31"), date.between("2023-01-01", "2023-12-31"), date.between("2024-01-01", "2025-12-31")],
        ["train", "validation", "test"], default="excluded_boundary_or_warmup"
    )


def main() -> None:
    a = args(); root = a.data_root
    curated, model_dir, reports = root / "curated", root / "model_ready", root / "reports"
    for p in (curated, model_dir, reports): p.mkdir(parents=True, exist_ok=True)
    daily = read_daily(root)
    universe = build_universe(root)
    calendar = master_calendar(daily)
    enriched = features_and_targets(daily, calendar, a.rf_file)
    enriched["universe_month"] = enriched["date"].dt.to_period("M").astype(str)
    # Retain every point-in-time eligible constituent here.  The weekly Top-100
    # is chosen below only from information known at the signal date (features
    # and lagged market capitalisation), rather than from future label quality.
    eligible = universe.loc[universe["eligible"], ["universe_month", "ric", "market_cap_vnd", "market_cap_rank", "trbc_sector", "exchange"]]
    panel = enriched.merge(eligible, on=["universe_month", "ric"], how="inner", validate="many_to_one")
    panel["log_market_cap"] = np.log(panel["market_cap_vnd"].where(panel["market_cap_vnd"].gt(0)))
    panel["bid"] = np.nan; panel["ask"] = np.nan; panel["mid"] = np.nan; panel["quoted_spread"] = np.nan
    panel["cost_one_way"] = COST_ONE_WAY; panel["cost_is_imputed"] = np.int8(1)
    panel["rf_source"] = "LSEG VND1MD= composite 1M VND deposit cash proxy"
    panel = panel.sort_values(["ric", "date"]).reset_index(drop=True)
    model = panel.loc[panel["date"].isin(calendar.loc[calendar["is_weekly_signal"], "date"])].copy()
    # Eligibility to enter the forecast cross-section is strictly past-only.
    feature_required = ["return_60d", "vol_60d", "log_dollar_volume", "log_market_cap"]
    candidates = model.loc[model[feature_required].notna().all(axis=1)].copy()
    candidates = candidates.sort_values(["date", "market_cap_rank"], ascending=[True, True]).groupby("date", group_keys=False).head(TOP_N)
    full_dates = candidates.groupby("date")["ric"].nunique()
    full_dates = full_dates.loc[full_dates.eq(TOP_N)].index
    model = candidates.loc[candidates["date"].isin(full_dates)].copy()
    model["split"] = split(model["date"])
    model["target_available"] = model[["target_excess_return_5d_bps", "execution_date"]].notna().all(axis=1)
    model["target_status"] = np.where(model["target_available"], "available", "missing_exact_next_five_sessions")
    calendar.to_parquet(curated / "market_calendar.parquet", index=False)
    universe.to_parquet(curated / "universe_monthly.parquet", index=False)
    panel.to_parquet(curated / "daily_panel.parquet", index=False)
    model[["date", "ric", "market_cap_vnd", "market_cap_rank", "exchange", "split", "target_available", "target_status"]].to_parquet(curated / "universe_weekly.parquet", index=False)
    model.to_parquet(model_dir / "weekly_features_targets.parquet", index=False)
    per_date = model.groupby("date")["ric"].nunique()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "V3_MODEL_READY_WITH_LIMITATIONS",
        "source": "LSEG Workspace historical HOSE/HNX snapshots, LSEG daily OHLCV/total return, VND1MD= cash-proxy RF",
        "timing": "month-end snapshot M is used only in M+1; weekly signal is final observed session of W-FRI; target uses next five master-market sessions",
        "cost": {"cost_one_way": COST_ONE_WAY, "cost_is_imputed": True, "note": "No historical bid/ask was extracted; this is a protocol assumption, not observed spread."},
        "outputs": {"calendar_rows": len(calendar), "universe_rows": len(universe), "selected_universe_rows": int(universe["selected_top100"].sum()), "daily_panel_rows": len(panel), "weekly_model_rows": len(model), "weekly_dates": int(model["date"].nunique()), "weekly_dates_with_100_assets": int((per_date == TOP_N).sum()), "exact_target_rows": int(model["target_available"].sum())},
        "features": ["return_1d", "return_5d", "return_10d", "return_20d", "return_60d", "vol_5d", "vol_20d", "vol_60d", "log_volume", "log_dollar_volume", "log_price", "log_market_cap", "high_low_proxy", "amihud", "day_of_week", "is_month_end", "is_quarter_end"],
        "limitations": ["Monthly index constituents reduce but do not eliminate survivorship bias; listing/delisting event dates and reasons are unavailable.", "Total-return adjustment semantics require corporate-action validation before final economic claims.", "RF is a VND 1-month deposit cash proxy and its historical publication timestamp is unavailable.", "Transaction costs are assumed because no historical bid/ask or tick history is present."],
    }
    (reports / "dataset_v3_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
