"""Build the ASEAN daily panel and weekly excess-return forecast dataset.

This step constructs data only; it does not run forecasting or portfolio
experiments. All joins are point-in-time: month-end market-cap snapshots are
available only from the following month, and five-session targets use each
country's own observed trading calendar.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "artifacts/asean_v1")
    p.add_argument("--universe", type=Path, default=ROOT / "artifacts/asean_preflight/historical_primary_universe.csv")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--selection-pool-n", type=int, default=None,
                   help="Point-in-time market-cap pool from which top-n feature-complete assets are selected.")
    return p.parse_args()


def read_daily(root: Path) -> pd.DataFrame:
    paths = sorted((root / "raw" / "daily_history").glob("daily_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No daily partitions found under {root / 'raw' / 'daily_history'}")
    cols = ["ric", "date", "open", "high", "low", "close", "volume", "bid", "ask",
            "total_return_pct", "country", "currency", "extract_timestamp_utc"]
    frames = []
    for path in paths:
        available = pd.read_parquet(path, columns=None).columns.tolist()
        use = [c for c in cols if c in available]
        frames.append(pd.read_parquet(path, columns=use))
    df = pd.concat(frames, ignore_index=True)
    df["ric"] = df["ric"].fillna("").astype(str).str.strip()
    df["country"] = df["country"].fillna("").astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.loc[df["ric"].ne("") & df["date"].between(START, END)].copy()
    for col in ["open", "high", "low", "close", "volume", "bid", "ask", "total_return_pct"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["country", "ric", "date", "extract_timestamp_utc"], na_position="last")
    df = df.drop_duplicates(["country", "ric", "date"], keep="last")
    df = df.loc[df["close"].gt(0) & df["volume"].ge(0)].copy()
    df["close_return"] = df.groupby(["country", "ric"])["close"].pct_change()
    df["return"] = df["total_return_pct"].div(100.0)
    df["return_source"] = np.where(df["return"].notna(), "LSEG_total_return", "close_return_fallback")
    df["return"] = df["return"].fillna(df["close_return"])
    mid = (df["bid"] + df["ask"]) / 2.0
    quote_ok = (df["bid"].ge(0) & df["ask"].ge(df["bid"]) & mid.gt(0)).fillna(False)
    df["quote_observed"] = quote_ok.astype("int8")
    df["mid"] = mid.where(quote_ok)
    df["quoted_spread"] = ((df["ask"] - df["bid"]) / mid).where(quote_ok)
    df["quoted_spread_bps"] = df["quoted_spread"] * 10000.0
    df["dollar_volume"] = df["close"] * df["volume"]
    return df.sort_values(["country", "ric", "date"]).reset_index(drop=True)


def read_universe(path: Path) -> pd.DataFrame:
    u = pd.read_csv(path)
    u["RIC"] = u["RIC"].fillna("").astype(str).str.strip()
    u["Country"] = u["Country"].fillna("").astype(str).str.strip()
    return u.loc[u["RIC"].ne("") & u["RIC"].ne("nan")].copy()


def read_market_cap(root: Path) -> pd.DataFrame:
    paths = sorted((root / "raw" / "market_cap_monthly").glob("market_cap_*.parquet"))
    if not paths:
        raise FileNotFoundError("Monthly market-cap partitions are missing; run download_lseg_asean_marketcap.py first.")
    x = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    x["ric"] = x["ric"].astype(str).str.strip()
    x["country"] = x["country"].astype(str).str.strip()
    x["snapshot_date"] = pd.to_datetime(x["snapshot_date"], errors="coerce").dt.normalize()
    x["market_cap_usd"] = pd.to_numeric(x["market_cap_usd"], errors="coerce")
    x = x.loc[x["snapshot_date"].notna() & x["market_cap_usd"].gt(0)].copy()
    x["universe_month"] = (x["snapshot_date"].dt.to_period("M") + 1).astype(str)
    x = x.sort_values(["country", "universe_month", "ric", "snapshot_date"])
    x = x.drop_duplicates(["country", "universe_month", "ric"], keep="last")
    x["market_cap_rank"] = x.groupby(["country", "universe_month"])["market_cap_usd"].rank(method="first", ascending=False)
    x["eligible"] = x["market_cap_usd"].gt(0)
    return x


def add_calendar_and_rf(daily: pd.DataFrame, rf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar_parts = []
    for country, g in daily.groupby("country", sort=True):
        dates = pd.DataFrame({"date": sorted(g["date"].unique())})
        dates["country"] = country
        dates["session_id"] = np.arange(len(dates), dtype="int32")
        dates["week"] = dates["date"].dt.to_period("W-FRI").astype(str)
        dates["is_weekly_signal"] = dates.groupby("week")["date"].transform("max").eq(dates["date"])
        calendar_parts.append(dates)
    calendar = pd.concat(calendar_parts, ignore_index=True)
    rf = pd.read_parquet(rf_path)
    rf["date"] = pd.to_datetime(rf["date"], errors="coerce").dt.normalize()
    rf["rf_daily"] = pd.to_numeric(rf["rf_daily"], errors="coerce")
    rf = rf[["country", "date", "rf_daily", "rf_proxy", "rf_source"]].drop_duplicates(["country", "date"])
    cal_parts = []
    for country, g in calendar.groupby("country", sort=False):
        c = g.sort_values("date").merge(rf.loc[rf["country"].eq(country)], on=["country", "date"], how="left")
        c["rf_daily"] = c["rf_daily"].ffill()
        c["rf_observed"] = c["rf_daily"].notna().astype("int8")
        cal_parts.append(c)
    calendar = pd.concat(cal_parts, ignore_index=True)
    daily = daily.merge(calendar[["country", "date", "session_id", "week", "is_weekly_signal", "rf_daily", "rf_observed"]],
                        on=["country", "date"], how="left", validate="many_to_one")
    daily["rf_daily"] = daily["rf_daily"].fillna(0.0)
    return daily, calendar


def features_targets(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (country, ric), g in daily.groupby(["country", "ric"], sort=False):
        g = g.sort_values("date").copy()
        contiguous = g["session_id"].diff().eq(1)
        g["return_1d"] = g["return"]
        for window in (5, 10, 20, 60):
            g[f"return_{window}d"] = (1.0 + g["return"]).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        for window in (5, 20, 60):
            g[f"vol_{window}d"] = g["return"].rolling(window, min_periods=window).std(ddof=0)
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
        future_stock = pd.Series(1.0, index=g.index); future_rf = pd.Series(1.0, index=g.index)
        exact_future = pd.Series(True, index=g.index)
        for step in range(1, 6):
            future_stock *= 1.0 + g["return"].shift(-step)
            future_rf *= 1.0 + g["rf_daily"].shift(-step)
            exact_future &= g["session_id"].shift(-step).eq(g["session_id"] + step)
        g["target_excess_return_5d_bps"] = np.where(exact_future, (future_stock - future_rf) * 10000.0, np.nan)
        g["execution_date"] = g["date"].shift(-1).where(g["session_id"].shift(-1).eq(g["session_id"] + 1))
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def split(dates: pd.Series) -> np.ndarray:
    return np.select([dates.between("2019-01-01", "2022-12-31"), dates.between("2023-01-01", "2023-12-31"), dates.between("2024-01-01", "2025-12-31")],
                     ["train", "validation", "test"], default="excluded_boundary_or_warmup")


def main() -> None:
    a = parse_args(); root = a.root
    curated, model_dir, reports = root / "curated", root / "model_ready", root / "reports"
    for p in (curated, model_dir, reports): p.mkdir(parents=True, exist_ok=True)
    daily = read_daily(root)
    universe = read_universe(a.universe)
    daily, calendar = add_calendar_and_rf(daily, root / "raw" / "risk_free_daily.parquet")
    enriched = features_targets(daily)
    caps = read_market_cap(root)
    caps["selected_top_n"] = caps["market_cap_rank"].le(a.top_n)
    pool_n = a.selection_pool_n or a.top_n
    enriched["universe_month"] = enriched["date"].dt.to_period("M").astype(str)
    panel = enriched.merge(caps[["country", "ric", "universe_month", "market_cap_usd", "market_cap_rank", "eligible", "selected_top_n"]], on=["country", "ric", "universe_month"], how="inner", validate="many_to_one")
    panel["log_market_cap"] = np.log(panel["market_cap_usd"].where(panel["market_cap_usd"].gt(0)))
    signal = panel.loc[panel["is_weekly_signal"].eq(True)].copy()
    required = ["return_60d", "vol_60d", "log_dollar_volume", "log_market_cap"]
    # Availability-aware but point-in-time selection: use only the lagged
    # market-cap pool and features known at the signal date, then take the
    # highest-ranked top-n names that have all required history.
    candidates = signal.loc[signal["market_cap_rank"].le(pool_n) & signal[required].notna().all(axis=1)].copy()
    candidates = candidates.sort_values(["country", "date", "market_cap_rank"]).groupby(["country", "date"], group_keys=False).head(a.top_n)
    counts = candidates.groupby(["country", "date"])["ric"].nunique()
    full_keys = counts.loc[counts.eq(a.top_n)].index
    candidates["full_top_n"] = pd.MultiIndex.from_frame(candidates[["country", "date"]]).isin(full_keys)
    candidates["split"] = split(candidates["date"])
    candidates["target_available"] = candidates[["target_excess_return_5d_bps", "execution_date"]].notna().all(axis=1)
    candidates["target_status"] = np.where(candidates["target_available"], "available", "missing_exact_next_five_sessions")
    calendar.to_parquet(curated / "market_calendar.parquet", index=False)
    universe.to_parquet(curated / "historical_primary_universe.parquet", index=False)
    caps.to_parquet(curated / "universe_monthly.parquet", index=False)
    panel.to_parquet(curated / "daily_panel.parquet", index=False)
    candidates[["country", "date", "ric", "market_cap_usd", "market_cap_rank", "split", "full_top_n", "target_available", "target_status"]].to_parquet(curated / "universe_weekly.parquet", index=False)
    candidates.to_parquet(model_dir / "weekly_features_targets.parquet", index=False)
    candidates.loc[candidates["full_top_n"]].to_parquet(model_dir / f"weekly_features_targets_full_top{a.top_n}.parquet", index=False)
    coverage = []
    for country, g in daily.groupby("country"):
        cg = candidates.loc[candidates["country"].eq(country)]
        coverage.append({"country": country, "rows": len(g), "unique_rics": g["ric"].nunique(), "start": str(g["date"].min().date()), "end": str(g["date"].max().date()), "close_coverage": float(g["close"].notna().mean()), "volume_coverage": float(g["volume"].notna().mean()), "bid_coverage": float(g["bid"].notna().mean()), "ask_coverage": float(g["ask"].notna().mean()), "quote_observed": float(g["quote_observed"].mean()), "total_return_coverage": float(g["total_return_pct"].notna().mean()), "target_coverage": float(cg["target_available"].mean()) if len(cg) else 0.0})
    pd.DataFrame(coverage).to_csv(reports / "coverage_report.csv", index=False)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "ASEAN_FORECAST_DATASET_BUILT", "source": "LSEG Workspace daily OHLCV/TR BidPrice/TR AskPrice, total-return fields, monthly market cap, and country cash proxies", "countries": sorted(daily["country"].unique().tolist()), "top_n": a.top_n, "selection_pool_n": pool_n, "selection_rule": "At each country-specific weekly signal date, select the highest lagged-market-cap-ranked top-n assets with complete 60-session feature history from the pre-specified selection pool.", "timing": "month-end market-cap snapshot M is used in M+1; weekly signal is final observed session per country; target is next five country-specific sessions", "outputs": {"daily_rows": len(daily), "daily_unique_rics": int(daily[["country", "ric"]].drop_duplicates().shape[0]), "calendar_rows": len(calendar), "panel_rows": len(panel), "weekly_candidate_rows": len(candidates), "weekly_full_top_n_rows": int(candidates["full_top_n"].sum()), "weekly_dates": int(candidates[["country", "date"]].drop_duplicates().shape[0]), "full_top_n_dates": int(candidates.loc[candidates["full_top_n"], ["country", "date"]].drop_duplicates().shape[0]), "target_rows": int(candidates["target_available"].sum())}, "features": ["return_1d", "return_5d", "return_10d", "return_20d", "return_60d", "vol_5d", "vol_20d", "vol_60d", "log_volume", "log_dollar_volume", "log_price", "log_market_cap", "high_low_proxy", "amihud", "quoted_spread_bps", "day_of_week", "is_month_end", "is_quarter_end"], "quote_note": "BID/ASK are historical end-of-day TR.BidPrice/TR.AskPrice fields; quote events and implementation shortfall are not evaluated in this step.", "rf_note": "Philippines uses PHP1MID=PHR as an explicitly flagged one-month proxy because PHP1MD= was unavailable.", "limitations": ["The Screener universe is a historical primary-security candidate universe, not a complete delisting-event ledger.", "Total return falls back to close return only where LSEG total-return data is missing; return_source is retained.", "Dates with fewer than top_n feature-complete assets are retained in the variable-size file but excluded from the full_top_n file."]}
    (reports / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
