"""Read-only contract and kernel checks for the V3 PTCST method.

This is intentionally a small executable validator so the same checks can be
called from Colab without requiring a test runner or notebook-local logic.
"""
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

from src.v3_method import (
    FEATURES,
    build_tensor_bundle,
    cost_aware_mvo,
    ledoit_covariance,
    read_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/lseg_v3"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_root
    weekly, model, daily = read_v3(root)

    train, median, scale = build_tensor_bundle(root, "train")
    validation, _, _ = build_tensor_bundle(root, "validation", median, scale)
    test, _, _ = build_tensor_bundle(root, "test", median, scale)
    bundles = {"train": train, "validation": validation, "test": test}
    expected_dates = weekly.groupby("split")["date"].nunique().to_dict()
    report: dict[str, object] = {"data_root": str(root), "features": FEATURES, "splits": {}}

    for split, bundle in bundles.items():
        assert bundle.x.ndim == 4 and bundle.x.shape[1:] == (100, 60, len(FEATURES))
        assert len(bundle.dates) == expected_dates[split]
        assert np.isfinite(bundle.x).all()
        assert all(len(set(rics)) == 100 for rics in bundle.rics)
        report["splits"][split] = {
            "dates": int(len(bundle.dates)),
            "shape": list(bundle.x.shape),
            "target_available_fraction": float(bundle.mask.mean()),
        }

    # Available labels have an execution date strictly after the signal date;
    # missing labels remain masked and are never dropped from the cross-section.
    available = model.loc[model["target_available"] & model["execution_date"].notna()]
    assert (available["execution_date"] > available["date"]).all()
    assert model.groupby(["split", "date"]).size().eq(100).all()

    signal_date = pd.Timestamp(test.dates[0])
    rics = [str(x) for x in test.rics[0]]
    covariance, valid, risk = ledoit_covariance(daily, signal_date, rics)
    assert np.allclose(covariance, covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-9
    assert risk.get("window_rows", 0) <= 252

    # Feasible pre-trade weights and optimizer outputs are deterministic and
    # satisfy the locked constraints within numerical tolerance.
    w_pre = np.zeros(100)
    w_pre[valid] = 1.0 / max(int(valid.sum()), 1)
    mu = np.linspace(-0.001, 0.001, 100)
    w_one, info_one = cost_aware_mvo(mu, covariance, w_pre, valid)
    w_two, info_two = cost_aware_mvo(mu, covariance, w_pre, valid)
    assert np.allclose(w_one, w_two, atol=2e-6)
    assert abs(w_one.sum() - 1) <= 2e-6
    assert w_one.min() >= -2e-6 and w_one.max() <= 0.05 + 2e-6
    assert np.abs(w_one - w_pre).sum() <= 0.40 + 2e-6
    assert info_one.get("status") == info_two.get("status")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
