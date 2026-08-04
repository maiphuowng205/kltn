"""Run the common V3 forecast baselines with date-level evaluation.

Models are fit only on train rows with an available target.  Validation and
test predictions share the exact tensor/imputation/scaling contract used by
the PTCST method.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v3_method import build_tensor_bundle


def row_features(bundle, kind: str) -> np.ndarray:
    x = bundle.x
    if kind == "ridge":
        return x.reshape(len(x) * x.shape[1], -1)
    if kind == "xgb":
        # Last-step signal plus sequence summaries, as locked in the plan.
        last = x[:, :, -1, :]
        mean = x.mean(axis=2)
        std = x.std(axis=2)
        minimum = x.min(axis=2)
        maximum = x.max(axis=2)
        return np.concatenate([last, mean, std, minimum, maximum], axis=-1).reshape(len(x) * x.shape[1], -1)
    raise ValueError(kind)


def expanding_historical_mean(train, validation, test) -> dict[str, np.ndarray]:
    history: dict[str, list[float]] = {}
    all_values: list[float] = []
    for i in range(len(train.dates)):
        for j, ric in enumerate(train.rics[i]):
            if train.mask[i, j]:
                value = float(train.y[i, j]); history.setdefault(str(ric), []).append(value); all_values.append(value)
    global_mean = float(np.mean(all_values)) if all_values else 0.0
    predictions = {}
    for name, bundle in (("validation", validation), ("test", test)):
        out = np.zeros_like(bundle.y, dtype=np.float32)
        for i in range(len(bundle.dates)):
            for j, ric_value in enumerate(bundle.rics[i]):
                ric = str(ric_value); values = history.get(ric, [])
                out[i, j] = float(np.mean(values) if values else global_mean)
            # A prior five-session label is observable before the next weekly
            # signal, so update only after producing this date's forecast.
            for j, ric_value in enumerate(bundle.rics[i]):
                if bundle.mask[i, j]:
                    history.setdefault(str(ric_value), []).append(float(bundle.y[i, j]))
        predictions[name] = out
    return predictions


def make_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (model, split, date), group in rows.groupby(["model", "split", "date"], sort=True):
        group = group.loc[group.target_available & np.isfinite(group.prediction) & np.isfinite(group.target)]
        n = len(group)
        if n < 3:
            output.append({"model": model, "split": split, "date": date, "n_assets": n})
            continue
        p = group.prediction.to_numpy(float); y = group.target.to_numpy(float)
        p_rank = pd.Series(p).rank(); y_rank = pd.Series(y).rank()
        k = max(1, int(np.floor(n * 0.2)))
        order = np.argsort(p)
        output.append({
            "model": model, "split": split, "date": date, "n_assets": n,
            "spearman_ic": float(p_rank.corr(y_rank)),
            "pearson_ic": float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else np.nan,
            "mae_bps": float(np.mean(np.abs(p - y))),
            "rmse_bps": float(np.sqrt(np.mean((p - y) ** 2))),
            "directional_accuracy": float(np.mean(np.sign(p) == np.sign(y))),
            "top_minus_bottom_bps": float(np.mean(y[order[-k:]]) - np.mean(y[order[:k]])),
        })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/lseg_v3")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/v3_forecast_baselines")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    train, median, scale = build_tensor_bundle(args.data_root, "train")
    validation, _, _ = build_tensor_bundle(args.data_root, "validation", median, scale)
    test, _, _ = build_tensor_bundle(args.data_root, "test", median, scale)
    bundles = {"validation": validation, "test": test}
    forecasts: dict[str, dict[str, np.ndarray]] = {
        "zero": {name: np.zeros_like(bundle.y, dtype=np.float32) for name, bundle in bundles.items()},
        "historical_mean": expanding_historical_mean(train, validation, test),
    }

    ridge = Ridge(alpha=args.ridge_alpha)
    x_train = row_features(train, "ridge"); train_mask = train.mask.reshape(-1)
    ridge.fit(x_train[train_mask], train.y.reshape(-1)[train_mask])
    forecasts["ridge"] = {name: ridge.predict(row_features(bundle, "ridge")).reshape(bundle.y.shape).astype(np.float32) for name, bundle in bundles.items()}

    xgb_info = {"skipped": bool(args.skip_xgb)}
    if not args.skip_xgb:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("XGBoost is required; install requirements-colab.txt or pass --skip-xgb") from exc
        xgb = XGBRegressor(
            n_estimators=args.xgb_estimators, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="reg:squarederror", random_state=args.seed, n_jobs=-1,
        )
        x_train = row_features(train, "xgb")
        xgb.fit(x_train[train_mask], train.y.reshape(-1)[train_mask])
        forecasts["xgb"] = {name: xgb.predict(row_features(bundle, "xgb")).reshape(bundle.y.shape).astype(np.float32) for name, bundle in bundles.items()}

    rows = []
    for model_name, split_predictions in forecasts.items():
        for split, bundle in bundles.items():
            pred = split_predictions[split]
            for i, date in enumerate(bundle.dates):
                for j, ric in enumerate(bundle.rics[i]):
                    rows.append({"date": pd.Timestamp(date), "ric": str(ric), "split": split, "model": model_name, "prediction": float(pred[i, j]), "target": float(bundle.y[i, j]), "target_available": bool(bundle.mask[i, j])})
    forecast_df = pd.DataFrame(rows)
    metrics_by_date = make_metrics(forecast_df)
    summary = metrics_by_date.groupby(["model", "split"], as_index=False).agg({"n_assets": "mean", "spearman_ic": "mean", "pearson_ic": "mean", "mae_bps": "mean", "rmse_bps": "mean", "directional_accuracy": "mean", "top_minus_bottom_bps": "mean"})
    forecast_df.to_parquet(args.run_dir / "forecasts.parquet", index=False)
    metrics_by_date.to_parquet(args.run_dir / "forecast_metrics_by_date.parquet", index=False)
    summary.to_parquet(args.run_dir / "forecast_metrics_summary.parquet", index=False)
    (args.run_dir / "preprocessing.json").write_text(json.dumps({"features": int(train.x.shape[-1]), "lookback": int(train.x.shape[-2]), "median": median.tolist(), "iqr": scale.tolist()}, indent=2), encoding="utf-8")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "models": sorted(forecasts), "ridge_alpha": args.ridge_alpha, "xgb": xgb_info, "train_dates": len(train.dates), "validation_dates": len(validation.dates), "test_dates": len(test.dates)}
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
