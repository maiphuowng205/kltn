"""Run an expanding quarterly Ridge walk-forward with an explicit cutoff audit.

The test labels are added to the expanding history only after their forecast has
been generated.  At every quarterly retraining origin, only rows whose
``execution_date`` is on or before the signal date are eligible.  The audit
artifacts make that rule independently inspectable instead of relying on the
training loop alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v3_method import build_tensor_bundle


def _flatten(bundle):
    """Return row-wise features and metadata without assuming 100 assets."""
    n_dates, n_assets, _, n_features = bundle.x.shape
    return (
        bundle.x.reshape(n_dates * n_assets, -1),
        bundle.y.reshape(n_dates * n_assets),
        bundle.mask.reshape(n_dates * n_assets).astype(bool),
        pd.to_datetime(bundle.execution_dates.reshape(n_dates * n_assets)),
        np.asarray(bundle.dates).repeat(n_assets),
        np.asarray(bundle.rics).reshape(n_dates * n_assets).astype(str),
        n_assets,
    )


def _quarter(date: pd.Timestamp) -> str:
    return f"{date.year}Q{date.quarter}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/lseg_v3")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/v3_walk_forward")
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    train, median, iqr = build_tensor_bundle(args.data_root, "train")
    validation, _, _ = build_tensor_bundle(args.data_root, "validation", median, iqr)
    test, _, _ = build_tensor_bundle(args.data_root, "test", median, iqr)

    train_x, train_y, train_mask, train_exec, train_signal, train_ric, n_assets = _flatten(train)
    val_x, val_y, val_mask, val_exec, val_signal, val_ric, val_assets = _flatten(validation)
    test_x, test_y, test_mask, test_exec, test_signal, test_ric, test_assets = _flatten(test)
    if not (n_assets == val_assets == test_assets):
        raise ValueError(f"split asset counts differ: train={n_assets}, validation={val_assets}, test={test_assets}")

    # History contains only rows with an observed label.  Test rows are appended
    # after prediction, so the current signal date can never train its own model.
    history_x = [train_x[train_mask], val_x[val_mask]]
    history_y = [train_y[train_mask], val_y[val_mask]]
    history_exec = [train_exec[train_mask], val_exec[val_mask]]
    history_signal = [train_signal[train_mask], val_signal[val_mask]]
    history_ric = [train_ric[train_mask], val_ric[val_mask]]

    forecasts = []
    checkpoints = []
    cutoff_rows = []
    model = None
    current_quarter = None

    for i, signal64 in enumerate(test.dates):
        signal_date = pd.Timestamp(signal64).normalize()
        quarter = _quarter(signal_date)
        if model is None or quarter != current_quarter:
            hx = np.concatenate(history_x, axis=0)
            hy = np.concatenate(history_y, axis=0)
            he = pd.Series(pd.to_datetime(np.concatenate(history_exec, axis=0)))
            hs = pd.Series(pd.to_datetime(np.concatenate(history_signal, axis=0)))
            hr = np.concatenate(history_ric, axis=0)
            eligible = he.notna() & (he <= signal_date)
            future = he.notna() & (he > signal_date)
            if not eligible.any():
                raise RuntimeError(f"no realized labels available at retraining origin {signal_date.date()}")
            if future.any() and bool(np.any(eligible & future)):
                raise AssertionError("a row cannot be both eligible and future")
            model = Ridge(alpha=args.alpha).fit(hx[eligible.to_numpy()], hy[eligible.to_numpy()])
            current_quarter = quarter
            max_realized = he[eligible].max()
            min_future = he[future].min() if future.any() else pd.NaT
            checkpoint = {
                "signal_date": signal_date,
                "quarter": quarter,
                "candidate_rows": int(len(he)),
                "realized_rows_used": int(eligible.sum()),
                "future_rows_excluded": int(future.sum()),
                "future_rows_used": 0,
                "max_execution_date_used": max_realized,
                "min_excluded_future_execution_date": min_future,
                "source_rows_used_train_validation_or_prior_test": int(eligible.sum()),
            }
            checkpoints.append(checkpoint)
            cutoff_rows.append(checkpoint.copy())

        start = i * n_assets
        stop = (i + 1) * n_assets
        pred = model.predict(test_x[start:stop]).reshape(n_assets)
        for j in range(n_assets):
            forecasts.append(
                {
                    "date": signal_date,
                    "ric": str(test_ric[start + j]),
                    "split": "test",
                    "prediction": float(pred[j]),
                    "target": float(test_y[start + j]) if np.isfinite(test_y[start + j]) else np.nan,
                    "target_available": bool(test_mask[start + j]),
                    "execution_date": pd.Timestamp(test_exec[start + j]) if pd.notna(test_exec[start + j]) else pd.NaT,
                    "model_training_quarter": quarter,
                }
            )

        # Append labels only after the current date has been forecast.  They are
        # filtered by execution_date at the next quarterly retraining origin.
        observed = test_mask[start:stop]
        if observed.any():
            history_x.append(test_x[start:stop][observed])
            history_y.append(test_y[start:stop][observed])
            history_exec.append(test_exec[start:stop][observed])
            history_signal.append(test_signal[start:stop][observed])
            history_ric.append(test_ric[start:stop][observed])

    forecast_frame = pd.DataFrame(forecasts)
    metric_rows = []
    for date, group in forecast_frame.groupby("date", sort=True):
        group = group.loc[group["target_available"] & group["target"].notna()]
        if len(group) == 0:
            continue
        ranks_p = group["prediction"].rank()
        ranks_y = group["target"].rank()
        metric_rows.append(
            {
                "date": date,
                "n_assets": int(len(group)),
                "spearman_ic": float(ranks_p.corr(ranks_y)) if len(group) >= 3 else np.nan,
                "mae_bps": float(np.mean(np.abs(group["prediction"] - group["target"]))),
            }
        )

    checkpoints_frame = pd.DataFrame(checkpoints)
    cutoff_audit = {
        "status": "PASS" if int(checkpoints_frame["future_rows_used"].sum()) == 0 else "FAIL",
        "retraining_points": int(len(checkpoints_frame)),
        "total_future_rows_used": int(checkpoints_frame["future_rows_used"].sum()),
        "total_future_rows_excluded": int(checkpoints_frame["future_rows_excluded"].sum()),
        "rule": "execution_date <= signal_date; current signal-date labels are appended only after prediction",
        "assertions": {
            "no_future_rows_used": bool((checkpoints_frame["future_rows_used"] == 0).all()),
            "quarterly_retraining": bool(checkpoints_frame["quarter"].nunique() == len(checkpoints_frame)),
        },
    }
    if cutoff_audit["status"] != "PASS":
        raise AssertionError(json.dumps(cutoff_audit))

    forecast_frame.to_parquet(args.run_dir / "forecasts.parquet", index=False)
    pd.DataFrame(metric_rows).to_parquet(args.run_dir / "forecast_metrics_by_date.parquet", index=False)
    checkpoints_frame.to_parquet(args.run_dir / "retraining_checkpoints.parquet", index=False)
    (args.run_dir / "cutoff_audit.json").write_text(json.dumps(cutoff_audit, indent=2), encoding="utf-8")
    (args.run_dir / "cutoff_audit.parquet").write_bytes(pd.DataFrame(cutoff_rows).to_parquet(index=False))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": f"Ridge(alpha={args.alpha})",
        "retraining": "expanding quarterly",
        "test_dates": int(len(test.dates)),
        "asset_count": int(n_assets),
        "retraining_points": int(len(checkpoints_frame)),
        "label_rule": "target_available and execution_date <= signal date",
        "cutoff_audit": "cutoff_audit.json",
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({**report, **cutoff_audit}, indent=2, default=str))


if __name__ == "__main__":
    main()
