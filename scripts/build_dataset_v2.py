"""Build the calendar-aware Vietnam equity dataset used for model development.

V2 preserves the V1 raw OHLCV cache, but fixes two semantic problems in the
supervised sample construction: targets use a market-wide session calendar and
execution occurs one session after the signal.  It deliberately does *not*
claim to solve survivorship bias or infer a risk-free series from OHLCV.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


OHLCV = ["open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "intraday_range",
    "open_to_close_return",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "close_to_sma_5",
    "close_to_sma_20",
    "close_to_sma_60",
    "volume_to_ma_5",
    "volume_to_ma_20",
    "volume_to_ma_60",
    "log_volume",
    "log_liquidity_proxy",
    "amihud_20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calendar-aware Dataset V2.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--liquidity-window", type=int, default=60)
    parser.add_argument("--risk-window", type=int, default=252)
    parser.add_argument("--execution-lag", type=int, default=1)
    parser.add_argument("--holding-sessions", type=int, default=5)
    parser.add_argument(
        "--risk-free-file",
        type=Path,
        help="Optional Parquet/CSV with date and rf_daily (daily simple return).",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_time(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert("Asia/Ho_Chi_Minh").dt.normalize().dt.tz_localize(None)


def read_panel(data_root: Path) -> pd.DataFrame:
    files = sorted((data_root / "raw" / "ohlcv").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No raw OHLCV Parquet files found. Run Dataset Audit V1 first.")
    frames: list[pd.DataFrame] = []
    for file in tqdm(files, desc="Reading raw OHLCV"):
        frame = pd.read_parquet(file)
        if "symbol" not in frame:
            frame["symbol"] = file.stem
        missing = set(["time", *OHLCV]).difference(frame.columns)
        if missing:
            raise ValueError(f"{file.name} is missing {sorted(missing)}")
        frame = frame.loc[:, ["time", "symbol", *OHLCV]].copy()
        frame["time"] = normalise_time(frame["time"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        for column in OHLCV:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["time"])
        .drop_duplicates(["symbol", "time"])
        .sort_values(["symbol", "time"])
        .reset_index(drop=True)
    )


def read_universe(data_root: Path, symbols: pd.Series) -> pd.DataFrame:
    path = data_root / "raw" / "universe_current.csv"
    if not path.exists():
        return pd.DataFrame({"symbol": sorted(symbols.unique()), "exchange": "UNKNOWN"})
    universe = pd.read_csv(path)
    universe["symbol"] = universe["symbol"].astype(str).str.upper()
    return universe.drop_duplicates("symbol")


def make_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(panel["time"].drop_duplicates().sort_values())
    calendar = pd.DataFrame({"time": dates})
    calendar["session_id"] = np.arange(len(calendar), dtype=np.int32)
    calendar["week"] = calendar["time"].dt.to_period("W-FRI").astype(str)
    calendar["is_weekly_signal"] = calendar.groupby("week")["time"].transform("max").eq(calendar["time"])
    return calendar


def prepare_symbol(raw: pd.DataFrame, calendar: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Reindex one ticker to market sessions and compute strictly past-only features."""
    indexed = raw.set_index("time").reindex(pd.DatetimeIndex(calendar["time"])).copy()
    indexed.index.name = "time"
    indexed["symbol"] = raw["symbol"].iloc[0]
    valid = indexed[OHLCV].notna().all(axis=1) & indexed["close"].gt(0) & indexed["volume"].ge(0)
    indexed["valid_ohlcv"] = valid
    indexed["liquidity_proxy_60d"] = (indexed["close"] * indexed["volume"]).rolling(
        args.liquidity_window, min_periods=args.liquidity_window
    ).median()
    indexed["complete_history_252"] = valid.astype("int16").rolling(
        args.risk_window, min_periods=args.risk_window
    ).sum().eq(args.risk_window)

    close = indexed["close"]
    volume = indexed["volume"]
    simple_return = close.pct_change(fill_method=None)
    indexed["return_1d"] = simple_return
    for window in (5, 20, 60):
        indexed[f"return_{window}d"] = close.pct_change(window, fill_method=None)
        indexed[f"volatility_{window}d"] = simple_return.rolling(window, min_periods=window).std(ddof=0)
        indexed[f"close_to_sma_{window}"] = close / close.rolling(window, min_periods=window).mean() - 1.0
        indexed[f"volume_to_ma_{window}"] = volume / volume.rolling(window, min_periods=window).mean() - 1.0
    indexed["intraday_range"] = indexed["high"] / indexed["low"] - 1.0
    indexed["open_to_close_return"] = close / indexed["open"] - 1.0
    indexed["log_volume"] = np.log1p(volume)
    indexed["log_liquidity_proxy"] = np.log1p(close * volume)
    indexed["amihud_20"] = (
        simple_return.abs() / (close * volume).replace(0, np.nan)
    ).rolling(20, min_periods=20).mean()
    indexed["feature_complete"] = indexed[FEATURE_COLUMNS].notna().all(axis=1)
    return indexed.reset_index()


def split_for_period(signal_date: pd.Timestamp, exit_date: pd.Timestamp) -> str:
    if signal_date >= pd.Timestamp("2019-01-01") and exit_date <= pd.Timestamp("2022-12-31"):
        return "train"
    if signal_date >= pd.Timestamp("2023-01-01") and exit_date <= pd.Timestamp("2023-12-31"):
        return "validation"
    if signal_date >= pd.Timestamp("2024-01-01") and exit_date <= pd.Timestamp("2025-12-31"):
        return "test"
    return "excluded_boundary_or_warmup"


def candidate_table(
    panel: pd.DataFrame, calendar: pd.DataFrame, exchange: pd.Series, args: argparse.Namespace
) -> pd.DataFrame:
    session_count = len(calendar)
    weekly_sessions = set(calendar.loc[calendar["is_weekly_signal"], "session_id"])
    candidates: list[pd.DataFrame] = []
    for symbol, raw in tqdm(panel.groupby("symbol", sort=True), desc="Screening weekly universe"):
        prepared = prepare_symbol(raw, calendar, args)
        prepared["session_id"] = calendar["session_id"].to_numpy()
        # These IDs describe the later realized holding period.  They are not
        # used to decide whether a stock belongs to the Top-K universe at t.
        prepared["entry_session_id"] = prepared["session_id"] + args.execution_lag
        prepared["exit_session_id"] = prepared["entry_session_id"] + args.holding_sessions
        mask = (
            prepared["session_id"].isin(weekly_sessions)
            & prepared["complete_history_252"]
            & prepared["feature_complete"]
            & prepared["liquidity_proxy_60d"].notna()
        )
        if mask.any():
            item = prepared.loc[mask, ["time", "symbol", "session_id", "entry_session_id", "exit_session_id", "liquidity_proxy_60d"]]
            item["exchange"] = exchange.get(symbol, "UNKNOWN")
            candidates.append(item)
    if not candidates:
        return pd.DataFrame()
    return pd.concat(candidates, ignore_index=True)


def select_universe(candidates: pd.DataFrame, calendar: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = (
        candidates.sort_values(["time", "liquidity_proxy_60d"], ascending=[True, False])
        .groupby("time", group_keys=False)
        .head(args.top_k)
    )
    counts = selected.groupby("time")["symbol"].nunique()
    selected = selected.loc[selected["time"].map(counts).eq(args.top_k)].copy()
    calendar_dates = calendar.set_index("session_id")["time"]
    selected = selected.rename(columns={"time": "signal_date"})
    selected["execution_date"] = selected["entry_session_id"].map(calendar_dates)
    selected["exit_date"] = selected["exit_session_id"].map(calendar_dates)
    selected["split"] = [split_for_period(s, e) for s, e in zip(selected["signal_date"], selected["exit_date"])]
    return selected.sort_values(["signal_date", "liquidity_proxy_60d"], ascending=[True, False]).reset_index(drop=True)


def build_sequences_and_targets(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    universe: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[pd.DataFrame] = []
    target_rows: list[pd.DataFrame] = []
    selected_by_symbol = {symbol: rows for symbol, rows in universe.groupby("symbol", sort=False)}
    for symbol, raw in tqdm(panel.groupby("symbol", sort=True), desc="Building features and targets"):
        selections = selected_by_symbol.get(symbol)
        if selections is None:
            continue
        prepared = prepare_symbol(raw, calendar, args)
        prepared["session_id"] = calendar["session_id"].to_numpy()
        prepared = prepared.set_index("session_id", drop=False)
        for selection in selections.itertuples(index=False):
            window = prepared.loc[selection.session_id - args.lookback + 1 : selection.session_id]
            if len(window) != args.lookback or not window[FEATURE_COLUMNS].notna().all().all():
                raise RuntimeError(f"{symbol} has an incomplete feature window at {selection.signal_date}")
            sequence = window.loc[:, ["time", "symbol", *FEATURE_COLUMNS]].copy()
            sequence = sequence.rename(columns={"time": "feature_date"})
            sequence["signal_date"] = selection.signal_date
            sequence["sequence_step"] = np.arange(args.lookback, dtype=np.int16)
            sequence["split"] = selection.split
            feature_rows.append(sequence.loc[:, ["signal_date", "feature_date", "sequence_step", "symbol", "split", *FEATURE_COLUMNS]])

            entry_in_cache = selection.entry_session_id in prepared.index
            exit_in_cache = selection.exit_session_id in prepared.index
            entry_close = prepared.at[selection.entry_session_id, "close"] if entry_in_cache else np.nan
            exit_close = prepared.at[selection.exit_session_id, "close"] if exit_in_cache else np.nan
            target_available = bool(pd.notna(entry_close) and entry_close > 0 and pd.notna(exit_close) and exit_close > 0)
            if not entry_in_cache:
                target_status = "entry_outside_cached_calendar"
            elif not exit_in_cache:
                target_status = "exit_outside_cached_calendar"
            elif not (pd.notna(entry_close) and entry_close > 0):
                target_status = "missing_entry_close"
            elif not (pd.notna(exit_close) and exit_close > 0):
                target_status = "missing_exit_close"
            else:
                target_status = "available"
            target_rows.append(
                pd.DataFrame(
                    {
                        "signal_date": [selection.signal_date],
                        "execution_date": [selection.execution_date],
                        "exit_date": [selection.exit_date],
                        "symbol": [symbol],
                        "split": [selection.split],
                        "entry_close": [entry_close],
                        "exit_close": [exit_close],
                        "exact_price_target_available": [target_available],
                        "target_status": [target_status],
                        "raw_return_5d": [exit_close / entry_close - 1.0 if target_available else np.nan],
                    }
                )
            )
    return pd.concat(feature_rows, ignore_index=True), pd.concat(target_rows, ignore_index=True)


def apply_risk_free(targets: pd.DataFrame, calendar: pd.DataFrame, path: Path | None) -> tuple[pd.DataFrame, dict[str, object]]:
    targets["risk_free_return_5d"] = np.nan
    targets["excess_return_5d"] = np.nan
    targets["final_target_available"] = False
    if path is None:
        return targets, {"available": False, "reason": "No risk-free file supplied; raw returns are diagnostics only."}
    if not path.exists():
        raise FileNotFoundError(f"Risk-free file does not exist: {path}")
    rf = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"date", "rf_daily"}
    if not required.issubset(rf.columns):
        raise ValueError(f"Risk-free file needs columns {sorted(required)}")
    rf = rf.loc[:, ["date", "rf_daily"]].copy()
    rf["date"] = normalise_time(rf["date"])
    rf["rf_daily"] = pd.to_numeric(rf["rf_daily"], errors="coerce")
    calendar_rf = calendar[["time", "session_id"]].merge(rf, left_on="time", right_on="date", how="left")
    # Fill only for cumulative arithmetic; `rf_observations` below prevents a
    # missing RF session from silently becoming a zero return.
    calendar_rf["growth"] = 1.0 + calendar_rf["rf_daily"].fillna(0.0)
    calendar_rf["cum_growth"] = calendar_rf["growth"].cumprod()
    calendar_rf["rf_observations"] = calendar_rf["rf_daily"].notna().astype("int16").cumsum()
    cumulative = calendar_rf.set_index("session_id")["cum_growth"]
    observations = calendar_rf.set_index("session_id")["rf_observations"]
    entry_ids = targets["execution_date"].map(calendar.set_index("time")["session_id"])
    exit_ids = targets["exit_date"].map(calendar.set_index("time")["session_id"])
    targets["risk_free_return_5d"] = [
        cumulative.loc[exit_id] / cumulative.loc[entry_id] - 1.0
        if (
            target_available
            and pd.notna(entry_id)
            and pd.notna(exit_id)
            and pd.notna(cumulative.loc[exit_id])
            and pd.notna(cumulative.loc[entry_id])
            and observations.loc[exit_id] - observations.loc[entry_id] == exit_id - entry_id
        )
        else np.nan
        for target_available, entry_id, exit_id in zip(targets["exact_price_target_available"], entry_ids, exit_ids)
    ]
    targets["excess_return_5d"] = targets["raw_return_5d"] - targets["risk_free_return_5d"]
    targets["final_target_available"] = targets["excess_return_5d"].notna()
    return targets, {"available": True, "path": str(path), "definition": "compound daily simple rf returns from execution to exit"}


def corporate_action_flags(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    flags: list[pd.DataFrame] = []
    session = calendar.set_index("time")["session_id"]
    for symbol, raw in panel.groupby("symbol", sort=True):
        item = raw[["time", "symbol", "close", "volume"]].sort_values("time").copy()
        item["session_id"] = item["time"].map(session)
        item["previous_session_id"] = item["session_id"].shift(1)
        item["one_session_return"] = item["close"].pct_change(fill_method=None)
        item["volume_ratio"] = item["volume"] / item["volume"].shift(1)
        suspicious = item.loc[
            item["session_id"].sub(item["previous_session_id"]).eq(1)
            & item["one_session_return"].abs().ge(0.15),
            ["time", "symbol", "close", "volume", "one_session_return", "volume_ratio"],
        ].copy()
        if not suspicious.empty:
            suspicious["flag_reason"] = "absolute one-market-session return >= 15%"
            suspicious["review_status"] = "requires corporate-action / source-adjustment review"
            flags.append(suspicious)
    columns = ["time", "symbol", "close", "volume", "one_session_return", "volume_ratio", "flag_reason", "review_status"]
    return pd.concat(flags, ignore_index=True).sort_values(["time", "symbol"]) if flags else pd.DataFrame(columns=columns)


def main() -> None:
    args = parse_args()
    if args.lookback <= 0 or args.top_k <= 0 or args.execution_lag < 1 or args.holding_sessions <= 0:
        raise ValueError("top-k/lookback/holding-sessions must be positive and execution-lag must be at least one")
    processed = args.data_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    print("Dataset V2: build calendar-aware signals, delayed-execution targets, and feature sequences.")
    panel = read_panel(args.data_root)
    universe_reference = read_universe(args.data_root, panel["symbol"])
    exchange = universe_reference.set_index("symbol")["exchange"]
    calendar = make_calendar(panel)
    candidates = candidate_table(panel, calendar, exchange, args)
    universe = select_universe(candidates, calendar, args)
    features, targets = build_sequences_and_targets(panel, calendar, universe, args)
    targets, rf_report = apply_risk_free(targets, calendar, args.risk_free_file)
    flags = corporate_action_flags(panel, calendar)

    calendar.to_parquet(processed / "market_calendar.parquet", index=False)
    universe.to_parquet(processed / "universe_weekly.parquet", index=False)
    features.to_parquet(processed / "features.parquet", index=False)
    targets.to_parquet(processed / "targets.parquet", index=False)
    flags.to_csv(processed / "corporate_action_flags.csv", index=False)

    split_counts = targets.groupby("split").size().sort_index().to_dict()
    available_split_counts = (
        targets.loc[targets["exact_price_target_available"]]
        .groupby("split")
        .size()
        .sort_index()
        .to_dict()
    )
    # Group only by fields that always exist. `exit_date` may be NaT for a
    # frozen end-of-cache universe date and must remain visible in the manifest.
    split_manifest = (
        universe.groupby(["signal_date", "split"], as_index=False)
        .agg(
            execution_date=("execution_date", "first"),
            exit_date=("exit_date", "first"),
            selected_stocks=("symbol", "nunique"),
        )
        .sort_values("signal_date")
    )
    split_manifest.to_csv(processed / "split_manifest.csv", index=False)
    split_date_counts = split_manifest.groupby("split").size().sort_index().to_dict()
    report = {
        "generated_at_utc": utc_now(),
        "status": "MODEL_DEVELOPMENT_ONLY",
        "raw_source": "Vnstock V1 current HOSE/HNX reference snapshot and daily OHLCV cache",
        "calendar": {
            "definition": "union of all observed raw OHLCV dates across the cached universe",
            "sessions": int(len(calendar)),
            "first": str(calendar["time"].min().date()),
            "last": str(calendar["time"].max().date()),
        },
        "sample_definition": {
            "signal": "last observed market session of W-FRI",
            "lookback_sessions": args.lookback,
            "selection": f"Top {args.top_k} by trailing {args.liquidity_window}-session median adjusted-close * volume proxy using only information available at signal t",
            "execution": f"close of market session t+{args.execution_lag}",
            "exit": f"close of market session t+{args.execution_lag + args.holding_sessions}",
            "raw_label": "exit_close / entry_close - 1",
        },
        "outputs": {
            "weekly_forecast_dates": int(universe["signal_date"].nunique()),
            "frozen_universe_stock_week_rows": int(len(targets)),
            "exact_price_target_available_rows": int(targets["exact_price_target_available"].sum()),
            "target_missing_rows": int((~targets["exact_price_target_available"]).sum()),
            "target_status_counts": {k: int(v) for k, v in targets["target_status"].value_counts().sort_index().items()},
            "feature_rows": int(len(features)),
            "feature_columns": FEATURE_COLUMNS,
            "split_stock_week_rows": {k: int(v) for k, v in split_counts.items()},
            "split_exact_price_target_rows": {k: int(v) for k, v in available_split_counts.items()},
            "split_forecast_dates": {k: int(v) for k, v in split_date_counts.items()},
            "corporate_action_review_flags": int(len(flags)),
        },
        "risk_free": rf_report,
        "freeze_blockers": [
            "No point-in-time historical membership or delisted-stock source is present; survivorship bias remains.",
            "No risk-free daily series was supplied, so excess_return_5d is intentionally unavailable unless risk_free_file is provided.",
            "Adjusted OHLCV convention is documented by Vnstock, but this one-time snapshot cannot verify historical re-adjustments without an overlap snapshot or independent corporate-action source.",
            "close * volume is retained only as an adjusted-price liquidity proxy; it is not asserted to be unadjusted turnover.",
        ],
    }
    (processed / "dataset_v2_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
