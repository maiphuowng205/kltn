"""Crawl and audit the free Vnstock OHLCV dataset for the VN equity study.

The script deliberately snapshots the *current* HOSE/HNX reference universe.
That makes the resulting survivorship limitation observable and reportable; it
does not claim to reconstruct the historical listed universe.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
REQUIRED_COLUMNS = ["time", *OHLCV_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl Vnstock daily OHLCV and create Dataset Audit v1."
    )
    parser.add_argument("command", choices=("crawl", "audit", "all"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--liquidity-window", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--risk-window", type=int, default=252)
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument(
        "--bar-count",
        type=int,
        default=5000,
        help="Requested historical bars per ticker. Vnstock defaults to only 100 if omitted.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=7.0,
        help="Delay between tickers. 7s is conservative for guest access because one OHLCV call may make multiple upstream requests.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--symbols",
        help="Optional comma-separated ticker subset, useful for a smoke test.",
    )
    parser.add_argument(
        "--max-symbols", type=int, help="Optional cap after deterministic symbol sorting."
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dotenv(env_file: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def paths(data_root: Path) -> dict[str, Path]:
    raw = data_root / "raw"
    return {
        "raw": raw,
        "ohlcv": raw / "ohlcv",
        "universe": raw / "universe_current.csv",
        "errors": raw / "crawl_errors.csv",
        "coverage": data_root / "coverage_report.csv",
        "forecast_dates": data_root / "forecast_dates.csv",
        "audit": data_root / "dataset_audit_v1.json",
    }


def normalise_symbols(value: Any, exchange: str) -> pd.DataFrame:
    """Accept Vnstock v4's Series output as well as older DataFrame outputs."""
    if isinstance(value, pd.Series):
        frame = value.to_frame(name="symbol")
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
        if "symbol" not in frame.columns:
            raise ValueError(f"{exchange} reference response has no 'symbol' column")
    else:
        raise TypeError(f"Unexpected {exchange} reference type: {type(value).__name__}")

    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame = frame.loc[frame["symbol"].str.fullmatch(r"[A-Z0-9]{2,10}", na=False)]
    frame = frame.drop_duplicates("symbol")
    frame["exchange"] = exchange
    return frame


def fetch_universe() -> pd.DataFrame:
    from vnstock import Reference

    reference = Reference()
    frames = []
    for exchange in ("HOSE", "HNX"):
        # Version 4.0.5 returns a Series named symbol.  Keeping this adapter
        # also makes the script work with releases that return a DataFrame.
        response = reference.equity.list_by_group(group=exchange)
        frames.append(normalise_symbols(response, exchange))
    universe = pd.concat(frames, ignore_index=True).drop_duplicates("symbol")
    universe["reference_as_of_utc"] = utc_now()
    return universe.sort_values(["exchange", "symbol"]).reset_index(drop=True)


def clean_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: response lacks columns {sorted(missing)}")
    clean = frame.loc[:, REQUIRED_COLUMNS].copy()
    clean["time"] = pd.to_datetime(clean["time"], errors="coerce", utc=True)
    clean["time"] = clean["time"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.normalize().dt.tz_localize(None)
    for column in OHLCV_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean["symbol"] = symbol
    clean = clean.dropna(subset=["time"]).drop_duplicates(["symbol", "time"])
    return clean.sort_values("time").reset_index(drop=True)


def crawl(args: argparse.Namespace, p: dict[str, Path]) -> None:
    from vnstock import Market, register_user

    load_dotenv()
    api_key = os.environ.get("VNSTOCK_API_KEY")
    if api_key:
        if not register_user(api_key):
            raise RuntimeError("Vnstock API key registration failed; check VNSTOCK_API_KEY in .env")
        print("Vnstock Community API key registered from .env.")
    else:
        print("No VNSTOCK_API_KEY found; using guest API quota.")
    p["ohlcv"].mkdir(parents=True, exist_ok=True)
    universe = fetch_universe()
    if args.symbols:
        requested = {item.strip().upper() for item in args.symbols.split(",") if item.strip()}
        universe = universe.loc[universe["symbol"].isin(requested)].copy()
        missing = requested.difference(universe["symbol"])
        if missing:
            print(f"Not in the current HOSE/HNX reference universe: {sorted(missing)}")
    if args.max_symbols is not None:
        universe = universe.sort_values("symbol").head(args.max_symbols).copy()
    if universe.empty:
        raise ValueError("No symbols selected for crawl")
    universe.to_csv(p["universe"], index=False)
    market = Market()
    errors: list[dict[str, str]] = []

    for row in tqdm(universe.itertuples(index=False), total=len(universe), desc="Downloading"):
        symbol = row.symbol
        output = p["ohlcv"] / f"{symbol}.parquet"
        if output.exists() and not args.overwrite:
            continue

        for attempt in range(1, args.retries + 1):
            try:
                frame = market.equity(symbol).ohlcv(
                    start=args.start,
                    end=args.end,
                    interval="1D",
                    count=args.bar_count,
                )
                if frame is None or frame.empty:
                    raise ValueError("empty response")
                clean_ohlcv(frame, symbol).to_parquet(output, index=False)
                break
            except (Exception, SystemExit) as exc:  # Vnstock may use SystemExit for rate limits.
                error_message = f"{type(exc).__name__}: {exc}"
                rate_limited = "rate limit" in error_message.lower() or "giới hạn api" in error_message.lower()
                if attempt == args.retries:
                    errors.append(
                        {
                            "symbol": symbol,
                            "exchange": row.exchange,
                            "attempts": str(attempt),
                            "error": error_message,
                        }
                    )
                else:
                    # The provider explicitly requests a one-minute pause on
                    # rate limiting. Preserve completed Parquet files and retry.
                    time.sleep(65.0 if rate_limited else max(args.sleep_seconds, 1.0) * attempt)
            finally:
                time.sleep(args.sleep_seconds)

    pd.DataFrame(errors, columns=["symbol", "exchange", "attempts", "error"]).to_csv(
        p["errors"], index=False
    )
    print(f"Universe snapshot: {len(universe)} current symbols")
    print(f"Failed downloads this run: {len(errors)}")


def read_panel(ohlcv_dir: Path) -> pd.DataFrame:
    files = sorted(ohlcv_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {ohlcv_dir}")
    frames = []
    for file in tqdm(files, desc="Reading Parquet"):
        frame = pd.read_parquet(file)
        if "symbol" not in frame.columns:
            frame["symbol"] = file.stem
        frames.append(clean_ohlcv(frame, file.stem))
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["symbol", "time"])
        .sort_values(["symbol", "time"])
        .reset_index(drop=True)
    )


def weekly_last_dates(dates: pd.Series) -> pd.DatetimeIndex:
    series = pd.Series(pd.DatetimeIndex(dates).unique()).sort_values()
    return pd.DatetimeIndex(series.groupby(series.dt.to_period("W-FRI")).max())


def monthly_last_dates(dates: pd.Series) -> pd.DatetimeIndex:
    series = pd.Series(pd.DatetimeIndex(dates).unique()).sort_values()
    return pd.DatetimeIndex(series.groupby(series.dt.to_period("M")).max())


def audit(args: argparse.Namespace, p: dict[str, Path]) -> None:
    panel = read_panel(p["ohlcv"])
    panel["valid_ohlcv"] = panel[OHLCV_COLUMNS].notna().all(axis=1)
    panel["dollar_volume"] = panel["close"] * panel["volume"]

    if p["universe"].exists():
        universe = pd.read_csv(p["universe"])
    else:
        universe = pd.DataFrame({"symbol": panel["symbol"].unique(), "exchange": "UNKNOWN"})
    exchange_by_symbol = universe.drop_duplicates("symbol").set_index("symbol")["exchange"]
    panel["exchange"] = panel["symbol"].map(exchange_by_symbol).fillna("UNKNOWN")

    coverage = (
        panel.groupby(["symbol", "exchange"], as_index=False)
        .agg(
            first_date=("time", "min"),
            last_date=("time", "max"),
            observations=("time", "size"),
            valid_ohlcv_observations=("valid_ohlcv", "sum"),
            zero_volume_days=("volume", lambda s: int((s == 0).sum())),
            missing_ohlcv_values=(OHLCV_COLUMNS[0], "size"),
        )
    )
    # A field-level count is more useful than a row-level count for data quality.
    missing_by_symbol = panel.assign(
        missing_ohlcv_fields=panel[OHLCV_COLUMNS].isna().sum(axis=1)
    ).groupby("symbol")["missing_ohlcv_fields"].sum()
    coverage["missing_ohlcv_fields"] = coverage["symbol"].map(missing_by_symbol).astype(int)
    coverage = coverage.drop(columns="missing_ohlcv_values").sort_values("observations", ascending=False)

    # Missing-grid rate uses exchange calendars inferred from observed data.  It
    # excludes dates before a ticker's first record and after its last record.
    expected_rows = 0
    observed_rows = 0
    for row in coverage.itertuples(index=False):
        exchange_days = panel.loc[panel["exchange"] == row.exchange, "time"].unique()
        expected_rows += int(((exchange_days >= row.first_date) & (exchange_days <= row.last_date)).sum())
        observed_rows += int(row.observations)

    panel = panel.sort_values(["symbol", "time"]).copy()
    group = panel.groupby("symbol", group_keys=False)
    panel["liquidity_60d"] = group["dollar_volume"].transform(
        lambda s: s.rolling(args.liquidity_window, min_periods=args.liquidity_window).median()
    )
    panel["valid_history"] = group["valid_ohlcv"].transform(
        lambda s: s.astype("int64").rolling(args.risk_window, min_periods=args.risk_window).sum()
    )
    panel["future_close"] = group["close"].shift(-args.target_horizon)
    panel["target_ready"] = panel["close"].gt(0) & panel["future_close"].gt(0)
    panel["eligible"] = (
        panel["valid_history"].eq(args.risk_window)
        & panel["liquidity_60d"].notna()
        & panel["target_ready"]
    )

    all_dates = pd.Series(panel["time"].unique()).sort_values()
    weekly_dates = weekly_last_dates(all_dates)
    monthly_dates = monthly_last_dates(all_dates)
    candidates = panel.loc[panel["eligible"], ["time", "symbol", "liquidity_60d"]]
    weekly_candidates = candidates.loc[candidates["time"].isin(weekly_dates)]
    weekly_counts = weekly_candidates.groupby("time")["symbol"].nunique()
    selected_weekly = (
        weekly_candidates.sort_values(["time", "liquidity_60d"], ascending=[True, False])
        .groupby("time", group_keys=False)
        .head(args.top_k)
        if not weekly_candidates.empty
        else weekly_candidates
    )
    selected_counts = selected_weekly.groupby("time")["symbol"].nunique()
    forecast = pd.DataFrame({"time": weekly_dates})
    forecast["eligible_stocks"] = forecast["time"].map(weekly_counts).fillna(0).astype(int)
    forecast["selected_stocks"] = forecast["time"].map(selected_counts).fillna(0).astype(int)
    forecast["usable_top_k"] = forecast["selected_stocks"].eq(args.top_k)
    forecast.to_csv(p["forecast_dates"], index=False)

    monthly_counts = candidates.loc[candidates["time"].isin(monthly_dates)].groupby("time")["symbol"].nunique()
    source_counts = universe.groupby("exchange")["symbol"].nunique().to_dict()
    threshold_counts = {
        str(threshold): int((coverage["observations"] >= threshold).sum())
        for threshold in (252, 500, 1000)
    }
    total_ohlcv_fields = len(panel) * len(OHLCV_COLUMNS)
    report = {
        "generated_at_utc": utc_now(),
        "scope": {
            "exchanges": ["HOSE", "HNX"],
            "frequency": "1D",
            "requested_bars_per_ticker": args.bar_count,
            "top_k": args.top_k,
            "liquidity_definition": f"{args.liquidity_window}-day median(close * volume)",
            "model_lookback_days": args.lookback,
            "risk_window_days": args.risk_window,
            "target": f"future {args.target_horizon}-trading-session close return",
            "rebalance": "weekly, last observed trading day of each W-FRI period",
        },
        "important_limitations": [
            "The universe is a current Vnstock HOSE/HNX reference snapshot, not a point-in-time historical membership file.",
            "Missing-grid rate is inferred from each static exchange's observed trading calendar; it is not proof of vendor completeness.",
        ],
        "universe": {
            "current_reference_symbols_by_exchange": {k: int(v) for k, v in source_counts.items()},
            "current_reference_symbols_total": int(len(universe.drop_duplicates("symbol"))),
            "successfully_downloaded_symbols": int(panel["symbol"].nunique()),
            "failed_symbols_recorded": int(len(pd.read_csv(p["errors"])) if p["errors"].exists() else 0),
        },
        "raw_ohlcv": {
            "stock_day_rows": int(len(panel)),
            "unique_trading_days": int(panel["time"].nunique()),
            "earliest_date": str(panel["time"].min().date()),
            "latest_date": str(panel["time"].max().date()),
            "stocks_by_minimum_observations": threshold_counts,
            "missing_ohlcv_field_rate": float(panel[OHLCV_COLUMNS].isna().sum().sum() / total_ohlcv_fields),
            "zero_volume_rate": float((panel["volume"] == 0).sum() / len(panel)),
            "inferred_missing_stock_day_rate": float(1 - observed_rows / expected_rows) if expected_rows else None,
        },
        "model_ready_panel": {
            "eligible_top_k_months": int((monthly_counts >= args.top_k).sum()),
            "weekly_forecast_dates_with_top_k": int(forecast["usable_top_k"].sum()),
            "final_stock_week_targets": int(forecast.loc[forecast["usable_top_k"], "selected_stocks"].sum()),
            "note": "A forecast date is one cross-sectional Transformer input of top_k x lookback x features; each selected stock supplies one 5-day target.",
        },
    }
    p["coverage"].parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(p["coverage"], index=False)
    p["audit"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Coverage report: {p['coverage']}")
    print(f"Forecast dates: {p['forecast_dates']}")
    print(f"Audit report: {p['audit']}")


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.target_horizon <= 0:
        raise ValueError("--top-k and --target-horizon must be positive")
    p = paths(args.data_root)
    if args.command in {"crawl", "all"}:
        crawl(args, p)
    if args.command in {"audit", "all"}:
        audit(args, p)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
