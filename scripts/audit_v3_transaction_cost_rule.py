"""Audit whether V3 contains observed bid/ask or tick data for cost validation.

V3 intentionally uses a fixed one-way cost assumption.  This script does not
turn proxies into spreads: it reports whether a licensed observed quote source
is actually present and records the exact limitation when it is not.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/lseg_v3")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/v3_extension_p2/cost_validation")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    path = args.data_root / "curated" / "daily_panel.parquet"
    panel = pd.read_parquet(
        path,
        columns=[
            "ric",
            "date",
            "bid",
            "ask",
            "mid",
            "quoted_spread",
            "cost_one_way",
            "cost_is_imputed",
        ],
    )
    quote_mask = panel[["bid", "ask"]].notna().all(axis=1)
    spread_mask = panel["quoted_spread"].notna()
    imputed = panel["cost_is_imputed"].fillna(0).astype(bool)
    fixed_costs = sorted(float(x) for x in panel["cost_one_way"].dropna().unique())
    observed = panel.loc[quote_mask].copy()
    if not observed.empty:
        observed["observed_half_spread"] = ((observed["ask"] - observed["bid"]) / (observed["ask"] + observed["bid"]))
        observed["observed_half_spread"] = observed["observed_half_spread"].where(observed["observed_half_spread"] >= 0)
        observed_summary = {
            "count": int(len(observed)),
            "median_half_spread": float(observed["observed_half_spread"].median()),
            "p95_half_spread": float(observed["observed_half_spread"].quantile(0.95)),
            "date_min": str(pd.to_datetime(observed["date"]).min().date()),
            "date_max": str(pd.to_datetime(observed["date"]).max().date()),
        }
    else:
        observed_summary = {"count": 0, "median_half_spread": None, "p95_half_spread": None, "date_min": None, "date_max": None}

    status = "PASS" if int(quote_mask.sum()) > 0 and int(imputed.sum()) < len(panel) else "BLOCKED_NO_OBSERVED_QUOTES"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "validation_possible": bool(status == "PASS"),
        "data_file": str(path),
        "rows": int(len(panel)),
        "unique_rics": int(panel["ric"].nunique()),
        "date_min": str(pd.to_datetime(panel["date"]).min().date()),
        "date_max": str(pd.to_datetime(panel["date"]).max().date()),
        "bid_ask_complete_rows": int(quote_mask.sum()),
        "quoted_spread_non_null_rows": int(spread_mask.sum()),
        "cost_is_imputed_rows": int(imputed.sum()),
        "cost_is_imputed_share": float(imputed.mean()),
        "fixed_cost_one_way_values": fixed_costs,
        "observed_quote_summary": observed_summary,
        "proxies_not_substituted_for_quotes": ["high_low_proxy", "amihud"],
        "limitation": (
            "The frozen V3 panel has no observed bid/ask or tick observations; "
            "cost_one_way=0.001 is imputed for every row. Historical execution-cost "
            "validation therefore remains pending until a licensed quote source is added "
            "in a new dataset freeze."
        ),
    }
    (args.run_dir / "transaction_cost_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "metric": "rows",
                "value": len(panel),
            },
            {"metric": "bid_ask_complete_rows", "value": int(quote_mask.sum())},
            {"metric": "quoted_spread_non_null_rows", "value": int(spread_mask.sum())},
            {"metric": "cost_is_imputed_rows", "value": int(imputed.sum())},
            {"metric": "cost_is_imputed_share", "value": float(imputed.mean())},
        ]
    ).to_csv(args.run_dir / "transaction_cost_validation_summary.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
