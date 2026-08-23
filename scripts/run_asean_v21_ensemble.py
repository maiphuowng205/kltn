"""Build the V2.1 rank-normalized seed ensemble with an alignment audit."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri


def rank_normalize(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=float)
    for i in range(scores.shape[0]):
        valid = np.flatnonzero(mask[i])
        n = len(valid)
        if n < 2:
            continue
        ranks = (np.arange(n) + 0.5) / n
        ranked = np.empty(n, dtype=float)
        ranked[np.argsort(np.argsort(scores[i, valid], kind="mergesort"), kind="mergesort")] = ranks
        out[i, valid] = ndtri(np.clip(ranked, 0.001, 0.999))
    return out


def corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    return float(pd.Series(a[valid]).rank().corr(pd.Series(b[valid]).rank()))


def load_seeds(root: Path, split: str, seeds: list[int]):
    blobs = []
    for seed in seeds:
        with np.load(root / f"seed_{seed}" / f"{split}_predictions.npz", allow_pickle=False) as archive:
            blobs.append({key: archive[key] for key in archive.files})
    first = blobs[0]
    for seed, blob in zip(seeds[1:], blobs[1:]):
        for key in ("dates", "countries", "rics"):
            if not np.array_equal(first[key], blob[key]):
                raise ValueError(f"Seed {seed} {key} grid differs for {split}")
    scores = np.stack([blob["raw_score"].astype(float) for blob in blobs], axis=-1)
    masks = np.stack([blob["asset_mask"].astype(bool) for blob in blobs], axis=-1)
    common = masks.all(axis=-1)
    normalized = np.stack(
        [rank_normalize(scores[..., index], masks[..., index]) for index in range(len(seeds))],
        axis=-1,
    )
    # Average only available seed scores.  The portfolio still uses the
    # intersection mask, but a missing seed must not contribute an artificial
    # zero that changes the ranking of an otherwise valid asset.
    ensemble = np.divide(
        normalized.sum(axis=-1),
        masks.sum(axis=-1),
        out=np.zeros(normalized.shape[:2], dtype=float),
        where=masks.sum(axis=-1) > 0,
    )
    audit = ensemble_alignment_audit(first, scores, masks, ensemble, common, seeds)
    return first, ensemble, common, audit


def ensemble_alignment_audit(
    bundle,
    scores: np.ndarray,
    masks: np.ndarray,
    ensemble: np.ndarray,
    common: np.ndarray,
    seeds: list[int],
) -> pd.DataFrame:
    target = bundle["target_bps"].astype(float)
    target_mask = bundle["target_mask"].astype(bool)
    rows = []
    for i, (country, date) in enumerate(zip(bundle["countries"].astype(str), pd.to_datetime(bundle["dates"]))):
        common_target = common[i] & target_mask[i]
        row = {
            "country": country,
            "date": date,
            "common_assets": int(common[i].sum()),
            "available_seed_min": int(masks[i].sum(axis=1).min()),
            "available_seed_max": int(masks[i].sum(axis=1).max()),
            "ensemble_ic": corr(ensemble[i], target[i], common_target),
        }
        for index, seed in enumerate(seeds):
            row[f"seed_{seed}_ic"] = corr(scores[i, :, index], target[i], common_target)
        for left in range(len(seeds)):
            for right in range(left + 1, len(seeds)):
                pair_mask = masks[i, :, left] & masks[i, :, right]
                row[f"seed_{seeds[left]}_vs_seed_{seeds[right]}_rank_corr"] = corr(
                    scores[i, :, left], scores[i, :, right], pair_mask
                )
        rows.append(row)
    return pd.DataFrame(rows)


def country_calibration(validation, val_score, val_mask):
    rows = []
    betas = {}
    for country in sorted(set(validation["countries"].astype(str))):
        z_values = []
        y_values = []
        for i, name in enumerate(validation["countries"].astype(str)):
            if name != country:
                continue
            mask = val_mask[i]
            p = val_score[i, mask]
            y = validation["target_bps"][i, mask]
            if len(p) < 3 or p.std() <= 1e-12:
                continue
            z = (p - p.mean()) / p.std()
            z_values.extend(z)
            y_values.extend(y / 10000.0)
        z = np.asarray(z_values)
        y = np.asarray(y_values)
        beta = float((z @ y) / (z @ z)) if len(z) and z @ z > 0 else 0.0
        usable = bool(beta > 0)
        betas[country] = max(beta, 0.0)
        rows.append(
            {
                "country": country,
                "beta_decimal_per_z": beta,
                "beta_positive": max(beta, 0.0),
                "usable_positive_signal": usable,
                "validation_observations": int(len(z)),
            }
        )
    return betas, pd.DataFrame(rows)


def apply_country_calibration(bundle, score, mask, betas):
    alpha = np.zeros_like(score, dtype=float)
    for i, country in enumerate(bundle["countries"].astype(str)):
        valid = mask[i]
        p = score[i, valid]
        if valid.sum() >= 2 and p.std() > 1e-12:
            alpha[i, valid] = betas.get(country, 0.0) * (p - p.mean()) / p.std()
    return alpha


def deciles(bundle, score, mask):
    rows = []
    for i, (country, date) in enumerate(zip(bundle["countries"].astype(str), pd.to_datetime(bundle["dates"]))):
        valid = mask[i]
        n = int(valid.sum())
        if n < 10:
            continue
        order = np.flatnonzero(valid)[np.argsort(score[i, valid], kind="mergesort")]
        for decile, indices in enumerate(np.array_split(order, 10), 1):
            rows.append(
                {
                    "country": country,
                    "date": date,
                    "decile": decile,
                    "n_assets": len(indices),
                    "mean_target_bps": float(np.nanmean(bundle["target_bps"][i, indices])),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table, pd.DataFrame()
    summary = []
    for country, group in table.groupby("country"):
        pivot = group.groupby("decile").mean(numeric_only=True).sort_index()
        d1 = float(pivot.loc[1, "mean_target_bps"])
        d10 = float(pivot.loc[10, "mean_target_bps"])
        observed = pivot["mean_target_bps"].to_numpy(float)
        expected = np.arange(1, len(observed) + 1, dtype=float)
        monotonicity = float(np.corrcoef(observed, expected)[0, 1]) if len(observed) > 1 and np.std(observed) > 0 else np.nan
        summary.append({"country": country, "d10_minus_d1_bps": d10 - d1, "decile_monotonicity": monotonicity})
    return table, pd.DataFrame(summary)


def save_bundle(path, bundle, score, mask, alpha=None):
    values = {
        "dates": bundle["dates"],
        "countries": bundle["countries"],
        "rics": bundle["rics"],
        "raw_score": score,
        "target_bps": bundle["target_bps"],
        "target_mask": bundle["target_mask"],
        "asset_mask": mask,
    }
    if alpha is not None:
        values["calibrated_alpha_decimal"] = alpha
    np.savez_compressed(path, **values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,19,31,43,59")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val, val_score, val_mask, val_audit = load_seeds(args.run_root, "validation", seeds)
    dev, dev_score, dev_mask, dev_audit = load_seeds(args.run_root, "development", seeds)
    val_target_mask = val["target_mask"].astype(bool) & val_mask
    dev_target_mask = dev["target_mask"].astype(bool) & dev_mask
    betas, calibration = country_calibration(val, val_score, val_target_mask)
    dev_alpha = apply_country_calibration(dev, dev_score, dev_target_mask, betas)
    val_alpha = apply_country_calibration(val, val_score, val_target_mask, betas)

    save_bundle(args.output_dir / "validation_ensemble.npz", val, val_score, val_mask, val_alpha)
    save_bundle(args.output_dir / "development_ensemble.npz", dev, dev_score, dev_mask, dev_alpha)
    val_decile, val_decile_summary = deciles(val, val_score, val_target_mask)
    dev_decile, dev_decile_summary = deciles(dev, dev_score, dev_target_mask)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    val_audit.to_csv(args.output_dir / "validation_ensemble_audit.csv", index=False)
    dev_audit.to_csv(args.output_dir / "development_ensemble_audit.csv", index=False)
    val_audit.groupby("country", as_index=False).mean(numeric_only=True).to_csv(args.output_dir / "validation_ensemble_audit_summary.csv", index=False)
    dev_audit.groupby("country", as_index=False).mean(numeric_only=True).to_csv(args.output_dir / "development_ensemble_audit_summary.csv", index=False)
    val_decile.to_parquet(args.output_dir / "validation_decile_returns.parquet", index=False)
    dev_decile.to_parquet(args.output_dir / "development_decile_returns.parquet", index=False)
    val_decile_summary.to_csv(args.output_dir / "validation_decile_summary.csv", index=False)
    dev_decile_summary.to_csv(args.output_dir / "development_decile_summary.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "rank_normalization": "within country/date/seed; normal-score inverse CDF",
        "aggregation": "mean over available seed scores; portfolio asset mask is seed intersection",
        "calibration": "country-specific positive slope fit on validation only; beta<=0 mapped to unusable/no-alpha",
        "alignment_audit": ["dates", "countries", "rics", "per-date seed rank correlations", "seed and ensemble IC"],
        "status": "development_outputs_only",
    }
    (args.output_dir / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(calibration.to_string(index=False))
    print(dev_decile_summary.to_string(index=False))


if __name__ == "__main__":
    main()
