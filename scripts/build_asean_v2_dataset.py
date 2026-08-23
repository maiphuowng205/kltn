"""Build the protocol-V2 ASEAN modelling tables from the frozen V1 inputs.

V2 is deliberately written to a new root.  It never overwrites V1.  Its main
changes are: (1) EOD-safe execution/label timing, (2) purged split metadata,
(3) pure point-in-time Top-100 membership with variable N, and (4) an
investability flag that requires both feature and covariance history.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_HISTORY = 60


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "artifacts" / "asean_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "asean_v2")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--risk-min-history", type=int, default=126, choices=(90, 126, 180, 252))
    parser.add_argument("--risk-max-window", type=int, default=252)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, names: list[str]) -> None:
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"V1 input is missing required columns: {missing}")


def future_after_execution(panel: pd.DataFrame) -> pd.DataFrame:
    """Create close(t+1)-executed, close(t+2:t+6)-realized labels per stock."""
    parts: list[pd.DataFrame] = []
    for (country, ric), group in panel.groupby(["country", "ric"], sort=False):
        g = group.sort_values("session_id").copy()
        exact = pd.Series(True, index=g.index)
        stock, rf = pd.Series(1.0, index=g.index), pd.Series(1.0, index=g.index)
        for step in range(2, 7):
            exact &= g["session_id"].shift(-step).eq(g["session_id"] + step)
            stock *= 1.0 + g["return"].shift(-step)
            rf *= 1.0 + g["rf_daily"].shift(-step)
        g["execution_date_v2"] = g["date"].shift(-1).where(
            g["session_id"].shift(-1).eq(g["session_id"] + 1)
        )
        g["label_start_date_v2"] = g["date"].shift(-2).where(
            g["session_id"].shift(-2).eq(g["session_id"] + 2)
        )
        g["label_end_date_v2"] = g["date"].shift(-6).where(
            g["session_id"].shift(-6).eq(g["session_id"] + 6)
        )
        g["target_excess_return_5d_bps_v2"] = np.where(exact, (stock - rf) * 10000.0, np.nan)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_market_regime_and_relative_target(panel: pd.DataFrame) -> pd.DataFrame:
    daily = panel.groupby(["country", "date"], as_index=False).agg(
        market_return_1d=("return", "mean"),
        cross_sectional_dispersion=("return", "std"),
        market_breadth=("return", lambda x: float((x > 0).mean())),
        median_stock_vol_20d=("vol_20d", "median"),
        median_spread_bps=("quoted_spread_bps", "median"),
        median_amihud=("amihud", "median"),
    ).sort_values(["country", "date"])
    pieces = []
    for country, group in daily.groupby("country", sort=False):
        g = group.sort_values("date").copy()
        for window in (5, 20, 60):
            g[f"market_return_{window}d"] = (1.0 + g["market_return_1d"]).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        for window in (20, 60):
            g[f"market_vol_{window}d"] = g["market_return_1d"].rolling(window, min_periods=window).std(ddof=0)
        index_level = (1.0 + g["market_return_1d"].fillna(0.0)).cumprod()
        g["market_distance_200d"] = index_level / index_level.rolling(200, min_periods=200).mean() - 1.0
        pieces.append(g)
    regime = pd.concat(pieces, ignore_index=True)
    output = panel.merge(regime, on=["country", "date"], how="left", validate="many_to_one")
    cs_mean = output.groupby(["country", "date"])["target_excess_return_5d_bps_v2"].transform("mean")
    output["target_cs_excess_return_5d_bps_v2"] = output["target_excess_return_5d_bps_v2"] - cs_mean
    return output


def assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["split_v2"] = np.select(
        [out.date.between("2019-01-01", "2022-12-31"), out.date.between("2023-01-01", "2025-12-31")],
        ["train", "development"], default="warmup_or_future_holdout",
    )
    # A row is only eligible if its full future-label window remains within its
    # assigned fold.  This is the purge that prevents labels from crossing a
    # train/development boundary.
    out["purged_from_split"] = (
        out.split_v2.eq("train") & out.label_end_date_v2.ge(pd.Timestamp("2023-01-01"))
    )
    out["model_eligible_v2"] = out.split_v2.isin(["train", "development"]) & ~out.purged_from_split
    return out


def build_country(original: pd.DataFrame, a: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build one market at a time to stay inside ordinary Colab RAM."""
    require_columns(original, ["country", "ric", "date", "session_id", "return", "rf_daily", "market_cap_rank", "return_60d", "vol_60d"])
    original["date"] = pd.to_datetime(original["date"]).dt.normalize()
    panel = assign_splits(add_market_regime_and_relative_target(future_after_execution(original)))
    feature_ok = panel[["return_60d", "vol_60d", "log_dollar_volume", "log_market_cap"]].notna().all(axis=1)
    history_ok = []
    for _, group in panel.groupby("ric", sort=False):
        contiguous = group.sort_values("session_id")["session_id"].diff().eq(1).astype("int16")
        history_ok.append(contiguous.rolling(a.risk_min_history, min_periods=a.risk_min_history).sum().eq(a.risk_min_history))
    panel["risk_history_eligible_v2"] = pd.concat(history_ok).sort_index().to_numpy(dtype=bool)
    panel["feature_eligible_v2"] = feature_ok
    panel["investable_v2"] = feature_ok & panel["risk_history_eligible_v2"]
    signal = panel.loc[panel["is_weekly_signal"].eq(True) & panel["market_cap_rank"].le(a.top_n)].copy()
    universe = signal.loc[signal["investable_v2"]].copy()
    universe["n_assets_v2"] = universe.groupby("date")["ric"].transform("nunique")
    universe["target_cs_excess_return_5d_bps_v2"] = universe["target_excess_return_5d_bps_v2"] - universe.groupby("date")["target_excess_return_5d_bps_v2"].transform("mean")
    universe["target_available_v2"] = universe["target_cs_excess_return_5d_bps_v2"].notna()
    country = str(original.country.iloc[0])
    coverage={"country":country,"weekly_dates":int(universe.date.nunique()),"mean_assets":float(universe.n_assets_v2.mean()),"min_assets":int(universe.n_assets_v2.min()),"target_coverage":float(universe.target_available_v2.mean())}
    return panel,universe,coverage


def main() -> None:
    a = args()
    if a.risk_max_window < a.risk_min_history:
        raise ValueError("risk-max-window must be at least risk-min-history")
    source, output = a.source_root, a.output_root
    source_panel = source / "curated" / "daily_panel"
    if not source_panel.exists(): source_panel = source / "curated" / "daily_panel.parquet"
    if not source_panel.exists(): raise FileNotFoundError(source_panel)
    columns = [
        "country", "date", "ric", "market_cap_rank", "market_cap_usd", "split_v2", "model_eligible_v2",
        "feature_eligible_v2", "risk_history_eligible_v2", "investable_v2", "n_assets_v2",
        "execution_date_v2", "label_start_date_v2", "label_end_date_v2", "target_excess_return_5d_bps_v2",
        "target_cs_excess_return_5d_bps_v2", "target_available_v2",
    ]
    for directory in (output / "curated" / "daily_panel_v2", output / "curated" / "universe_weekly_v2", output / "model_ready" / "weekly_features_targets_v2", output / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    coverage_rows=[]
    for country in ("Indonesia", "Malaysia", "Philippines", "Singapore", "Thailand"):
        # V2 is a pure Top-100 protocol. Reading the V1 Top-300+ pool only
        # inflates memory and cannot change membership because no replacement
        # from rank 101 onward is permitted.
        original = pd.read_parquet(source_panel, filters=[["country", "=", country], ["market_cap_rank", "<=", a.top_n]])
        if original.empty: raise RuntimeError(f"No input rows for {country}")
        panel,universe,coverage=build_country(original,a); key=country.lower()
        (output / "curated" / "daily_panel_v2" / f"country={key}").mkdir(exist_ok=True)
        (output / "curated" / "universe_weekly_v2" / f"country={key}").mkdir(exist_ok=True)
        (output / "model_ready" / "weekly_features_targets_v2" / f"country={key}").mkdir(exist_ok=True)
        panel.to_parquet(output / "curated" / "daily_panel_v2" / f"country={key}" / "part.parquet", index=False)
        universe[columns].to_parquet(output / "curated" / "universe_weekly_v2" / f"country={key}" / "part.parquet", index=False)
        universe.to_parquet(output / "model_ready" / "weekly_features_targets_v2" / f"country={key}" / "part.parquet", index=False)
        coverage_rows.append(coverage)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output / "reports" / "coverage_v2.csv", index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ASEAN_V2_DATASET_BUILT",
        "source": str(source), "top_n": a.top_n,
        "universe_rule": "pure lagged market-cap Top-N; non-investable constituents are omitted and never replaced by ranks above N",
        "timing": "signal at close t; execute at close t+1; target/P&L starts close t+2 and spans five country sessions through t+6",
        "target": "cross-sectional demeaned five-session excess return after execution",
        "risk_history": {"minimum_sessions": a.risk_min_history, "maximum_covariance_window": a.risk_max_window},
        "purge": "training observations whose label end reaches 2023-01-01 or later are excluded",
        "test_status": "No 2026 holdout is present in this source. 2024-2025 are development evidence only.",
        "coverage": coverage.to_dict(orient="records"),
    }
    (output / "reports" / "dataset_report_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
