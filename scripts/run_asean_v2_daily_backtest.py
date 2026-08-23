"""Run the V2.1 daily portfolio engine and emit an auditable ledger.

The engine is deliberately continuous: a missing valuation is carried at zero
for that position and logged, rather than dropping the whole portfolio date.
Every rebalance writes weights, trades, costs, risk coverage and solver/fallback
records so turnover and cost can be independently recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.asean_v2 import (  # noqa: E402
    cost_aware_mvo_vector_cost,
    ledoit_covariance_min_history,
    summarize_daily_portfolio,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "artifacts" / "asean_v2")
    p.add_argument("--prediction-file", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--risk-min-history", type=int, default=126, choices=(90, 126, 180, 252))
    p.add_argument("--risk-aversion", type=float, required=True)
    p.add_argument("--turnover-cap", type=float, default=0.40)
    p.add_argument("--alpha-mode", choices=("calibrated", "zero"), default="calibrated", help="Use calibrated forecast alpha or a zero-alpha risk-only benchmark.")
    p.add_argument(
        "--cost-scenario",
        choices=("C0", "C1", "C2"),
        default="C2",
        help="C0=10bps; C1=lagged country median half-spread; C2=lagged stock-specific half-spread.",
    )
    return p.parse_args()


def normalise_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def write_table(frame: pd.DataFrame, run_dir: Path, stem: str) -> None:
    """Write both machine-efficient parquet and human-auditable CSV."""
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame({"_empty": pd.Series(dtype="object")})
    frame.to_parquet(run_dir / f"{stem}.parquet", index=False)
    frame.to_csv(run_dir / f"{stem}.csv", index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main() -> None:
    a = parse_args()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    panel_path = a.data_root / "curated" / "daily_panel_v2"
    weekly_path = a.data_root / "model_ready" / "weekly_features_targets_v2"
    if not panel_path.exists():
        panel_path = panel_path.with_suffix(".parquet")
    if not weekly_path.exists():
        weekly_path = weekly_path.with_suffix(".parquet")
    panel = normalise_dates(pd.read_parquet(panel_path))
    weekly = normalise_dates(pd.read_parquet(weekly_path))

    blob = np.load(a.prediction_file, allow_pickle=False)
    pred_dates = pd.to_datetime(blob["dates"]).normalize()
    pred_countries = blob["countries"].astype(str)
    pred_rics = blob["rics"].astype(str)
    alpha = blob["calibrated_alpha_decimal"].astype(float)
    asset_mask = blob["asset_mask"].astype(bool)
    if a.alpha_mode == "zero":
        alpha = np.zeros_like(alpha, dtype=float)

    schedule: dict[tuple[str, pd.Timestamp], dict[str, object]] = {}
    skipped_missing_execution = 0
    skipped_empty_signal = 0
    for i, (country, date) in enumerate(zip(pred_countries, pred_dates)):
        signal = weekly.loc[
            weekly.country.eq(country) & weekly.date.eq(date)
        ].sort_values("market_cap_rank")
        if signal.empty:
            continue
        execution_raw = pd.to_datetime(signal.execution_date_v2.iloc[0], errors="coerce")
        # The final signal date has no following session.  It is not a valid
        # close(t+1) trade and must not be normalized or silently treated as a
        # real execution date.
        if pd.isna(execution_raw):
            skipped_missing_execution += 1
            continue
        execution = pd.Timestamp(execution_raw).normalize()
        entries = [
            (ric, float(value))
            for ric, value, ok in zip(pred_rics[i], alpha[i], asset_mask[i])
            if bool(ok) and ric
        ]
        if not entries:
            skipped_empty_signal += 1
            continue
        schedule[(country, execution)] = {"signal_date": date, "entries": entries}

    daily_records: list[dict[str, object]] = []
    rebalance_records: list[dict[str, object]] = []
    missing_records: list[dict[str, object]] = []
    weight_records: list[dict[str, object]] = []
    trade_records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []
    risk_records: list[dict[str, object]] = []
    solver_records: list[dict[str, object]] = []
    fallback_records: list[dict[str, object]] = []
    turnover_records: list[dict[str, object]] = []

    for country, full_country_panel in panel.groupby("country", sort=True):
        full_country_panel = full_country_panel.sort_values(["date", "ric"])
        holdings: dict[str, float] = {}
        last_cost: dict[str, float] = {}
        execution_dates = [
            execution for scheduled_country, execution in schedule if scheduled_country == country
        ]
        if not execution_dates:
            continue
        # Keep the complete past panel for covariance/history.  Only the
        # evaluation loop is restricted to the first execution date.
        evaluation_panel = full_country_panel.loc[
            full_country_panel.date.ge(min(execution_dates))
        ]
        for date, day in evaluation_panel.groupby("date", sort=True):
            returns = day.set_index("ric")["return"].to_dict()
            missing_assets = [
                ric
                for ric in holdings
                if ric not in returns or not np.isfinite(returns.get(ric, np.nan))
            ]
            pnl = sum(
                weight * float(returns.get(ric, 0.0))
                for ric, weight in holdings.items()
                if ric not in missing_assets
            )
            for ric in missing_assets:
                missing_records.append(
                    {
                        "country": country,
                        "date": date,
                        "ric": ric,
                        "reason": "missing_daily_valuation_carried_zero",
                    }
                )

            if holdings:
                post = {
                    ric: weight
                    * (1 + float(returns.get(ric, 0.0) if np.isfinite(returns.get(ric, np.nan)) else 0.0))
                    for ric, weight in holdings.items()
                }
                total = sum(post.values())
                if total > 0:
                    holdings = {ric: value / total for ric, value in post.items()}

            cost = 0.0
            turnover = 0.0
            status = "no_rebalance"
            risk_fallback = None
            task = schedule.get((country, pd.Timestamp(date).normalize()))
            if task:
                signal_date = pd.Timestamp(task["signal_date"]).normalize()
                rics = [ric for ric, _ in task["entries"]]
                score = np.asarray([value for _, value in task["entries"]], dtype=float)
                history = full_country_panel.loc[full_country_panel.date.lt(signal_date)]
                country_half = (
                    float(history["quoted_spread_bps"].median() / 20000.0)
                    if len(history) and history["quoted_spread_bps"].notna().any()
                    else 0.001
                )
                if a.cost_scenario == "C0":
                    costs = np.full(len(rics), 0.001)
                elif a.cost_scenario == "C1":
                    costs = np.full(len(rics), max(country_half, 0.0))
                else:
                    last_quote = (
                        history.dropna(subset=["quoted_spread_bps"])
                        .drop_duplicates("ric", keep="last")
                        .set_index("ric")["quoted_spread_bps"]
                        .to_dict()
                    )
                    costs = np.asarray(
                        [
                            max(float(last_quote.get(ric, country_half)) / 20000.0, 0.0)
                            for ric in rics
                        ]
                    )

                covariance, valid, risk = ledoit_covariance_min_history(
                    full_country_panel, signal_date, rics, a.risk_min_history
                )
                exited = {ric: weight for ric, weight in holdings.items() if ric not in set(rics)}
                exited_turnover = float(sum(exited.values()))
                exited_cost = float(
                    sum(weight * last_cost.get(ric, 0.001) for ric, weight in exited.items())
                )
                w_pre = np.asarray([holdings.get(ric, 0.0) for ric in rics], dtype=float)
                max_weight = max(0.05, 1 / max(len(rics), 1))
                rebalance_type = "initial_deployment" if not holdings else "rebalance"
                # Initial deployment is not a rebalance: there is no prior
                # portfolio to constrain.  From the second trade onward the
                # pre-registered turnover cap is applied normally.
                optimizer_turnover_cap = 1.0 if rebalance_type == "initial_deployment" else a.turnover_cap
                target, info = cost_aware_mvo_vector_cost(
                    score,
                    covariance,
                    w_pre,
                    valid,
                    costs,
                    a.risk_aversion,
                    max_weight,
                    optimizer_turnover_cap,
                    exited_turnover,
                    exited_cost,
                )
                if info.get("fallback") and valid.any():
                    target = np.where(valid, 1 / valid.sum(), 0.0)

                current_set = set(rics)
                previous_set = set(holdings)
                continuing = previous_set & current_set
                new_names = current_set - previous_set
                forced_exit_turnover = exited_turnover
                forced_entry_turnover = float(
                    sum(target[index] for index, ric in enumerate(rics) if ric in new_names)
                )
                continuing_turnover = float(
                    sum(
                        abs(float(target[index]) - holdings.get(ric, 0.0))
                        for index, ric in enumerate(rics)
                        if ric in continuing
                    )
                )
                turnover = continuing_turnover + forced_entry_turnover + forced_exit_turnover
                trade_cost = float(costs @ np.abs(target - w_pre))
                cost = trade_cost + exited_cost

                union_rics = sorted(previous_set | current_set)
                for ric in union_rics:
                    current_index = rics.index(ric) if ric in current_set else None
                    pre_weight = float(holdings.get(ric, 0.0))
                    target_weight = float(target[current_index]) if current_index is not None else 0.0
                    rate = float(costs[current_index]) if current_index is not None else float(last_cost.get(ric, 0.001))
                    delta = target_weight - pre_weight
                    is_entry = ric in new_names and target_weight > 0
                    is_exit = ric in exited
                    abs_trade = abs(delta)
                    asset_cost = abs_trade * rate
                    weight_records.append(
                        {
                            "country": country,
                            "signal_date": signal_date,
                            "execution_date": date,
                            "ric": ric,
                            "pre_weight": pre_weight,
                            "target_weight": target_weight,
                            "post_trade_weight": target_weight,
                        }
                    )
                    trade_records.append(
                        {
                            "country": country,
                            "signal_date": signal_date,
                            "execution_date": date,
                            "ric": ric,
                            "weight_delta": delta,
                            "absolute_trade": abs_trade,
                            "is_entry": is_entry,
                            "is_exit": is_exit,
                            "trade_component": "forced_exit" if is_exit else ("forced_entry" if is_entry else "continuing_name"),
                        }
                    )
                    cost_records.append(
                        {
                            "country": country,
                            "signal_date": signal_date,
                            "execution_date": date,
                            "ric": ric,
                            "cost_rate_decimal": rate,
                            "cost_rate_bps": rate * 10000.0,
                            "absolute_trade": abs_trade,
                            "transaction_cost_decimal": asset_cost,
                        }
                    )

                holdings = {
                    ric: float(weight)
                    for ric, weight in zip(rics, target)
                    if float(weight) > 0
                }
                last_cost = {
                    ric: float(rate)
                    for ric, rate in zip(rics, costs)
                    if ric in holdings
                }
                status = info.get("status")
                risk_fallback = risk.get("fallback")
                rebalance_record = {
                    "country": country,
                    "signal_date": signal_date,
                    "execution_date": date,
                    "assets": len(rics),
                    "valid_risk_assets": risk.get("valid_assets"),
                    "risk_fallback": risk_fallback,
                    "turnover": turnover,
                    "cost": cost,
                    "solver": info.get("solver"),
                    "solver_status": status,
                    "solver_fallback": info.get("fallback"),
                    "rebalance_type": rebalance_type,
                }
                rebalance_records.append(rebalance_record)
                risk_records.append(
                    {
                        **rebalance_record,
                        "risk_min_history": a.risk_min_history,
                        "risk_window_rows": risk.get("window_rows"),
                    }
                )
                solver_records.append(
                    {
                        "country": country,
                        "signal_date": signal_date,
                        "execution_date": date,
                        "solver": info.get("solver"),
                        "status": status,
                        "fallback": info.get("fallback"),
                        "objective": info.get("objective"),
                        "rebalance_type": rebalance_type,
                    }
                )
                turnover_records.append(
                    {
                        "country": country,
                        "signal_date": signal_date,
                        "execution_date": date,
                        "continuing_name_turnover": continuing_turnover,
                        "forced_entry_turnover": forced_entry_turnover,
                        "forced_exit_turnover": forced_exit_turnover,
                        "total_l1_turnover": turnover,
                        "reported_turnover": turnover,
                        "turnover_cap": a.turnover_cap,
                        "rebalance_type": rebalance_type,
                        "turnover_cap_applied": rebalance_type != "initial_deployment",
                        "cap_utilization": turnover / a.turnover_cap if rebalance_type != "initial_deployment" and a.turnover_cap else np.nan,
                    }
                )
                if risk_fallback:
                    fallback_records.append(
                        {
                            "country": country,
                            "signal_date": signal_date,
                            "execution_date": date,
                            "fallback_type": "risk",
                            "reason": risk_fallback,
                        }
                    )
                if info.get("fallback"):
                    fallback_records.append(
                        {
                            "country": country,
                            "signal_date": signal_date,
                            "execution_date": date,
                            "fallback_type": "solver",
                            "reason": info.get("fallback"),
                        }
                    )

            daily_records.append(
                {
                    "country": country,
                    "date": date,
                    "gross_return": pnl,
                    "cost": cost,
                    "net_return": pnl - cost,
                    "turnover": turnover,
                    "positions": len(holdings),
                    "missing_valuations": len(missing_assets),
                    "rebalance_status": status,
                    "risk_fallback": risk_fallback,
                }
            )

    daily = pd.DataFrame(daily_records)
    rebalances = pd.DataFrame(rebalance_records)
    missing = pd.DataFrame(missing_records, columns=["country", "date", "ric", "reason"])
    weights = pd.DataFrame(weight_records)
    trades = pd.DataFrame(trade_records)
    costs = pd.DataFrame(cost_records)
    risk = pd.DataFrame(risk_records)
    solver = pd.DataFrame(solver_records)
    fallbacks = pd.DataFrame(fallback_records)
    turnover = pd.DataFrame(turnover_records)

    summary, reliability = summarize_daily_portfolio(daily, rebalances)
    write_table(daily, a.run_dir, "daily_portfolio_returns")
    write_table(rebalances, a.run_dir, "rebalance_log")
    write_table(missing, a.run_dir, "missing_valuation_events")
    write_table(weights, a.run_dir, "portfolio_weights")
    write_table(trades, a.run_dir, "portfolio_trades")
    write_table(costs, a.run_dir, "portfolio_costs")
    write_table(risk, a.run_dir, "risk_coverage")
    write_table(solver, a.run_dir, "solver_log")
    write_table(fallbacks, a.run_dir, "fallback_log")
    write_table(turnover, a.run_dir, "turnover_decomposition")
    summary.to_csv(a.run_dir / "portfolio_metrics_summary.csv", index=False)
    reliability.to_csv(a.run_dir / "reliability_metrics.csv", index=False)

    reports = a.data_root / "reports"
    for source_name, output_name in (
        ("split_audit.csv", "split_audit.csv"),
        ("coverage_v2.csv", "universe_coverage.csv"),
    ):
        source = reports / source_name
        if source.exists():
            (a.run_dir / output_name).write_bytes(source.read_bytes())
    calibration = a.prediction_file.parent / "calibration_summary.csv"
    if calibration.exists():
        (a.run_dir / "calibration_summary.csv").write_bytes(calibration.read_bytes())

    protocol = {
        "timing": "signal close t; execution close t+1; P&L starts t+1 close-to-close",
        "cost_scenario": a.cost_scenario,
        "alpha_mode": a.alpha_mode,
        "cost_definition": "C0=10bps; C1=lagged country median half-spread; C2=lagged stock-specific half-spread",
        "turnover_definition": "full L1 over the union of prior and current holdings; continuing + forced entry + forced exit",
        "evaluation": "no date dropped for a missing asset valuation",
        "risk_min_history": a.risk_min_history,
        "risk_aversion": a.risk_aversion,
        "skipped_prediction_rows_missing_execution_date": skipped_missing_execution,
        "skipped_prediction_rows_empty_signal": skipped_empty_signal,
        "history_source": "full country panel before signal date",
    }
    (a.run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (a.run_dir / "protocol_lock.yaml").write_text(
        "".join(f"{key}: {json.dumps(value)}\n" for key, value in protocol.items()),
        encoding="utf-8",
    )

    files = sorted(path.name for path in a.run_dir.iterdir() if path.is_file())
    manifest = {
        "status": "ASEAN_V21_PORTFOLIO_RUN",
        "git_commit": git_commit(),
        "prediction_file": str(a.prediction_file),
        "prediction_sha256": sha256_file(a.prediction_file),
        "data_root": str(a.data_root),
        "cost_scenario": a.cost_scenario,
        "alpha_mode": a.alpha_mode,
        "risk_aversion": a.risk_aversion,
        "files": files,
        "counts": {
            "daily_rows": int(len(daily)),
            "rebalance_rows": int(len(rebalances)),
            "trade_rows": int(len(trades)),
            "missing_valuation_rows": int(len(missing)),
            "fallback_rows": int(len(fallbacks)),
        },
    }
    (a.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
