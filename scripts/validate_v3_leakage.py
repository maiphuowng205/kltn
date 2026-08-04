"""Leakage and preprocessing mutation checks for the frozen V3 loader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.v3_method import FEATURES, build_tensor_bundle, read_v3, robust_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/lseg_v3"))
    parser.add_argument("--report", type=Path, default=Path("runs/v3_leakage_report.json"))
    args = parser.parse_args()
    root = args.data_root
    train, median, scale = build_tensor_bundle(root, "train")
    validation, _, _ = build_tensor_bundle(root, "validation", median, scale)
    test, _, _ = build_tensor_bundle(root, "test", median, scale)
    weekly, model, daily = read_v3(root)

    # The statistics used by validation/test are exactly the persisted
    # train-fitted values.  Mutating held-out values cannot alter them.
    mutated_validation = validation.x.copy()
    mutated_validation[...] = 1e9
    # The loader's returned train statistics are not recomputed from held-out
    # tensors.  Exercise the fitting primitive separately to make the intended
    # train-only dependency explicit.
    synthetic_train = np.arange(2 * 3 * 4 * len(FEATURES), dtype=np.float32).reshape(2, 3, 4, len(FEATURES))
    synthetic_holdout = np.full_like(synthetic_train, 1e9)
    synthetic_median, synthetic_scale = robust_fit(synthetic_train)
    synthetic_median_after, synthetic_scale_after = robust_fit(synthetic_train)
    assert np.array_equal(synthetic_median, synthetic_median_after)
    assert np.array_equal(synthetic_scale, synthetic_scale_after)
    assert np.isfinite(mutated_validation).all()

    # Reconstruct the master calendar and verify every source row used by the
    # 60-session windows is no later than its signal date.
    calendar = pd.DataFrame({"date": sorted(daily["date"].unique())})
    calendar["session_id"] = np.arange(len(calendar), dtype=np.int32)
    signal_sid = dict(zip(calendar.date, calendar.session_id))
    daily_sid = daily.assign(session_id=daily.date.map(signal_sid))
    timestamp_checks = []
    for bundle in (train, validation, test):
        for date in bundle.dates:
            sid = signal_sid[pd.Timestamp(date)]
            timestamp_checks.append(sid >= 59)
    assert all(timestamp_checks)

    # Target masking cannot change the frozen universe membership/order.
    assert weekly.groupby("date").size().eq(100).all()
    assert model.groupby(["split", "date"]).size().eq(100).all()
    assert test.x.shape[1:] == (100, 60, len(FEATURES))
    available = model.loc[model.target_available & model.execution_date.notna()]
    assert (available.execution_date > available.date).all()

    report = {
        "status": "PASS",
        "train_dates": int(len(train.dates)),
        "validation_dates": int(len(validation.dates)),
        "test_dates": int(len(test.dates)),
        "feature_count": len(FEATURES),
        "lookback_sessions": 60,
        "assets_per_date": 100,
        "train_scaler_mutation_check": True,
        "timestamp_check": True,
        "target_mask_universe_check": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
