"""Run the pre-registered final ensemble audit without retraining.

The audit consumes the saved five-seed forecast NPZ files, rebuilds all
ensemble candidates on the common per-date universe, selects one candidate on
validation only, refits country calibration on validation, and compares the
frozen calibrated ensemble with a zero-alpha risk-only benchmark through the
same daily portfolio engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_asean_v21_ensemble import rank_normalize  # noqa: E402
from src.asean_v2 import _rank_date_metrics  # noqa: E402


SEEDS_DEFAULT = (7, 19, 31, 43, 59)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True, help="Saved pooled_ptcst/seed_* directory")
    p.add_argument("--original-run-root", type=Path, default=None, help="Optional independent copy for input reconciliation")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--risk-aversion", type=float, default=50.0)
    p.add_argument("--turnover-cap", type=float, default=0.40)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS_DEFAULT)))
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--bootstrap-block", type=int, default=5)
    return p.parse_args()


def load_seed_blobs(root: Path, split: str, seeds: list[int]) -> list[np.lib.npyio.NpzFile]:
    blobs = []
    for seed in seeds:
        path = root / f"seed_{seed}" / f"{split}_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        blobs.append(np.load(path, allow_pickle=False))
    first = blobs[0]
    for seed, blob in zip(seeds[1:], blobs[1:]):
        for key in ("dates", "countries", "rics", "target_bps", "target_mask"):
            if not np.array_equal(first[key], blob[key]):
                raise ValueError(f"{split} seed grid mismatch in {key}: seed {seed}")
    return blobs


def common_seed_arrays(blobs: list[np.lib.npyio.NpzFile]):
    raw = np.stack([blob["raw_score"].astype(float) for blob in blobs], axis=-1)
    masks = np.stack([blob["asset_mask"].astype(bool) for blob in blobs], axis=-1)
    common = masks.all(axis=-1)
    return raw, masks, common


def rank_matrix(raw: np.ndarray, masks: np.ndarray, common: np.ndarray) -> np.ndarray:
    """Recompute every seed rank on exactly the common universe."""
    out = []
    for index in range(raw.shape[-1]):
        out.append(rank_normalize(raw[..., index], common))
    return np.stack(out, axis=-1)


def spearman(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    return float(pd.Series(a[valid]).rank().corr(pd.Series(b[valid]).rank()))


def universe_audit(blobs: list[np.lib.npyio.NpzFile], seeds: list[int]) -> pd.DataFrame:
    masks = np.stack([blob["asset_mask"].astype(bool) for blob in blobs], axis=-1)
    first = blobs[0]
    rows = []
    for i, (country, date) in enumerate(zip(first["countries"].astype(str), pd.to_datetime(first["dates"]))):
        counts = masks[i].sum(axis=0)
        union = masks[i].any(axis=-1)
        intersection = masks[i].all(axis=-1)
        row = {"country": country, "date": date, "n_union": int(union.sum()), "n_intersection": int(intersection.sum()), "intersection_ratio": float(intersection.sum() / union.sum()) if union.sum() else np.nan}
        row.update({f"n_seed{seed}": int(count) for seed, count in zip(seeds, counts)})
        rows.append(row)
    return pd.DataFrame(rows)


def reconcile_inputs(original_root: Path, input_root: Path, split: str, seeds: list[int]) -> pd.DataFrame:
    old = load_seed_blobs(original_root, split, seeds)
    new = load_seed_blobs(input_root, split, seeds)
    rows = []
    for seed, old_blob, new_blob in zip(seeds, old, new):
        grid_equal = all(np.array_equal(old_blob[key], new_blob[key]) for key in ("dates", "countries", "rics"))
        if old_blob["raw_score"].shape != new_blob["raw_score"].shape:
            raise ValueError(f"Input shape mismatch for seed {seed}, split {split}")
        for i, (country, date) in enumerate(zip(old_blob["countries"].astype(str), pd.to_datetime(old_blob["dates"]))):
            for j, ric in enumerate(old_blob["rics"][i].astype(str)):
                old_asset = bool(old_blob["asset_mask"][i, j])
                new_asset = bool(new_blob["asset_mask"][i, j])
                old_score = float(old_blob["raw_score"][i, j] * 10000.0) if old_asset else np.nan
                new_score = float(new_blob["raw_score"][i, j] * 10000.0) if new_asset else np.nan
                old_target = float(old_blob["target_bps"][i, j]) if old_blob["target_mask"][i, j] else np.nan
                new_target = float(new_blob["target_bps"][i, j]) if new_blob["target_mask"][i, j] else np.nan
                score_match = (np.isnan(old_score) and np.isnan(new_score)) or (np.isfinite(old_score) and np.isfinite(new_score) and abs(old_score - new_score) < 1e-8)
                target_match = (np.isnan(old_target) and np.isnan(new_target)) or (np.isfinite(old_target) and np.isfinite(new_target) and abs(old_target - new_target) < 1e-8)
                rows.append({"country": country, "date": date, "RIC": ric, "seed": seed, "split": split, "prediction_original_bps": old_score, "prediction_rc3_input_bps": new_score, "abs_difference": abs(old_score - new_score) if np.isfinite(old_score) and np.isfinite(new_score) else np.nan, "target_original_bps": old_target, "target_rc3_bps": new_target, "universe_original": old_asset, "universe_rc3": new_asset, "match_flag": bool(grid_equal and old_asset == new_asset and score_match and target_match)})
    return pd.DataFrame(rows)


def per_model_metrics(bundle, prediction: np.ndarray, mask: np.ndarray, model: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(bundle["dates"])
    countries = bundle["countries"].astype(str)
    targets = bundle["target_bps"].astype(float)
    target_mask = bundle["target_mask"].astype(bool)
    daily = []
    for i, (country, date) in enumerate(zip(countries, dates)):
        valid = mask[i] & target_mask[i] & np.isfinite(prediction[i]) & np.isfinite(targets[i])
        row = _rank_date_metrics(prediction[i], targets[i], valid)
        row.update({"country": country, "date": date, "model": model, "split": split})
        daily.append(row)
    daily_frame = pd.DataFrame(daily)
    summary_rows = []
    for country, group in daily_frame.groupby("country", sort=True):
        rank = group.loc[group.rank_defined]
        ic = rank.spearman_ic.to_numpy(float)
        tb = rank.top_minus_bottom_bps.to_numpy(float)
        mean_ic = float(np.nanmean(ic)) if len(ic) else np.nan
        sd_ic = float(np.nanstd(ic, ddof=1)) if len(ic) > 1 else np.nan
        summary_rows.append({"country": country, "model": model, "split": split, "n_dates": int(len(group)), "mean_ic": mean_ic, "median_ic": float(np.nanmedian(ic)) if len(ic) else np.nan, "icir": float(mean_ic / sd_ic) if np.isfinite(sd_ic) and sd_ic > 0 else np.nan, "hit_rate": float(np.mean(ic > 0)) if len(ic) else np.nan, "top_bottom": float(np.nanmean(tb)) if len(tb) else np.nan, "pearson_ic": float(group.pearson_ic.mean()) if len(group) else np.nan, "mae_bps": float(group.mae_bps.mean()) if len(group) else np.nan, "rmse_bps": float(np.sqrt(np.nanmean(group.rmse_bps.to_numpy(float) ** 2))) if len(group) else np.nan, "dispersion_bps": float(group.cs_prediction_std_bps.median()) if len(group) else np.nan})
    return daily_frame, pd.DataFrame(summary_rows)


def candidate_arrays(rank_scores: np.ndarray, val_bundle, val_common: np.ndarray, seeds: list[int]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    candidates = {
        "E1_mean_rank": rank_scores.mean(axis=-1),
        "E2_median_rank": np.median(rank_scores, axis=-1),
    }
    seed_metrics = []
    for index, seed in enumerate(seeds):
        _, summary = per_model_metrics(val_bundle, rank_scores[..., index], val_common, f"seed{seed}", "validation")
        seed_metrics.append({"seed": seed, "mean_ic": float(summary.mean_ic.mean()), "top_bottom": float(summary.top_bottom.mean())})
    seed_table = pd.DataFrame(seed_metrics)
    ic_weights = np.maximum(seed_table.mean_ic.to_numpy(float), 0.0)
    tb_weights = np.maximum(seed_table.top_bottom.to_numpy(float), 0.0)
    ic_weights = ic_weights / ic_weights.sum() if ic_weights.sum() > 0 else np.full(len(seeds), 1 / len(seeds))
    tb_weights = tb_weights / tb_weights.sum() if tb_weights.sum() > 0 else np.full(len(seeds), 1 / len(seeds))
    candidates["E3_ic_weighted"] = np.tensordot(rank_scores, ic_weights, axes=([-1], [0]))
    candidates["E4_tb_weighted"] = np.tensordot(rank_scores, tb_weights, axes=([-1], [0]))
    return candidates, {"seed_metrics": seed_table, "ic_weights": dict(zip(seeds, ic_weights)), "tb_weights": dict(zip(seeds, tb_weights))}


def fit_country_betas(bundle, score: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    betas = {}
    for country in sorted(set(bundle["countries"].astype(str))):
        z_values, y_values = [], []
        for i, name in enumerate(bundle["countries"].astype(str)):
            if name != country:
                continue
            valid = mask[i]
            p = score[i, valid]
            y = bundle["target_bps"][i, valid]
            if valid.sum() < 3 or p.std() <= 1e-12:
                continue
            z_values.extend(((p - p.mean()) / p.std()).tolist())
            y_values.extend((y / 10000.0).tolist())
        z, y = np.asarray(z_values), np.asarray(y_values)
        beta = float((z @ y) / (z @ z)) if len(z) and z @ z > 0 else 0.0
        betas[country] = max(beta, 0.0)
        rows.append({"country": country, "calibration_beta_raw": beta, "calibration_beta_positive": max(beta, 0.0), "usable_positive_signal": bool(beta > 0), "observations": int(len(z))})
    return betas, pd.DataFrame(rows)


def apply_betas(bundle, score: np.ndarray, mask: np.ndarray, betas: dict[str, float]) -> np.ndarray:
    alpha = np.zeros_like(score, dtype=float)
    for i, country in enumerate(bundle["countries"].astype(str)):
        valid = mask[i]
        p = score[i, valid]
        if valid.sum() >= 2 and p.std() > 1e-12:
            alpha[i, valid] = betas.get(country, 0.0) * (p - p.mean()) / p.std()
    return alpha


def save_prediction_bundle(path: Path, bundle, score: np.ndarray, mask: np.ndarray, alpha: np.ndarray) -> None:
    np.savez_compressed(path, dates=bundle["dates"], countries=bundle["countries"], rics=bundle["rics"], raw_score=score, calibrated_alpha_decimal=alpha, target_bps=bundle["target_bps"], target_mask=bundle["target_mask"], asset_mask=mask)


def block_bootstrap(values: np.ndarray, draws: int, block: int, seed: int = 7) -> tuple[float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    blocks = [x[start : start + block] for start in range(0, len(x), block)]
    means = []
    for _ in range(draws):
        sample = []
        while len(sample) < len(x):
            sample.extend(blocks[int(rng.integers(0, len(blocks)))])
        means.append(float(np.mean(sample[: len(x)])))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def block_bootstrap_delta_sharpe(ptcst: np.ndarray, risk: np.ndarray, draws: int, block: int, seed: int = 19) -> tuple[float, float]:
    x = np.asarray(ptcst, float)
    y = np.asarray(risk, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    starts = list(range(0, len(x), block))
    values = []
    for _ in range(draws):
        indices = []
        while len(indices) < len(x):
            start = starts[int(rng.integers(0, len(starts)))]
            indices.extend(range(start, min(start + block, len(x))))
        idx = np.asarray(indices[: len(x)])
        xs, ys = x[idx], y[idx]
        x_sharpe = float(np.mean(xs) / np.std(xs, ddof=1) * np.sqrt(252)) if np.std(xs, ddof=1) > 0 else np.nan
        y_sharpe = float(np.mean(ys) / np.std(ys, ddof=1) * np.sqrt(252)) if np.std(ys, ddof=1) > 0 else np.nan
        if np.isfinite(x_sharpe) and np.isfinite(y_sharpe):
            values.append(x_sharpe - y_sharpe)
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if values else (np.nan, np.nan)


def run_backtest(args, prediction: Path, output: Path, alpha_mode: str, scenario: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "run_asean_v2_daily_backtest.py"), "--data-root", str(args.data_root), "--prediction-file", str(prediction), "--run-dir", str(output), "--risk-aversion", str(args.risk_aversion), "--turnover-cap", str(args.turnover_cap), "--cost-scenario", scenario, "--alpha-mode", alpha_mode]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Backtest failed ({alpha_mode}, {scenario}):\n{result.stdout}\n{result.stderr}")


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_root = args.original_run_root or args.run_root

    # 1–3: provenance and common-universe audits.
    reconciliation = pd.concat([reconcile_inputs(original_root, args.run_root, split, seeds) for split in ("validation", "development")], ignore_index=True)
    reconciliation.to_csv(args.output_dir / "ensemble_input_reconciliation.csv", index=False)
    val_blobs = load_seed_blobs(args.run_root, "validation", seeds)
    dev_blobs = load_seed_blobs(args.run_root, "development", seeds)
    universe = pd.concat([universe_audit(val_blobs, seeds).assign(split="validation"), universe_audit(dev_blobs, seeds).assign(split="development")], ignore_index=True)
    universe.to_csv(args.output_dir / "ensemble_universe_audit.csv", index=False)

    val_raw, val_masks, val_common = common_seed_arrays(val_blobs)
    dev_raw, dev_masks, dev_common = common_seed_arrays(dev_blobs)
    val_rank = rank_matrix(val_raw, val_masks, val_common)
    dev_rank = rank_matrix(dev_raw, dev_masks, dev_common)
    val_bundle, dev_bundle = val_blobs[0], dev_blobs[0]

    # 4–6: common-universe single-seed metrics, pairwise agreement and consensus.
    metric_frames = []
    for index, seed in enumerate(seeds):
        _, summary = per_model_metrics(val_bundle, val_rank[..., index], val_common, f"seed{seed}", "validation")
        metric_frames.append(summary)
        _, summary = per_model_metrics(dev_bundle, dev_rank[..., index], dev_common, f"seed{seed}", "development")
        metric_frames.append(summary)
    pair_rows = []
    consensus_rows = []
    for i, (country, date) in enumerate(zip(val_bundle["countries"].astype(str), pd.to_datetime(val_bundle["dates"]))):
        for left, right in combinations(range(len(seeds)), 2):
            pair_rows.append({"split": "validation", "country": country, "date": date, "seed_left": seeds[left], "seed_right": seeds[right], "rank_corr": spearman(val_raw[i, :, left], val_raw[i, :, right], val_common[i])})
        mean_rank = val_rank[i].mean(axis=-1)
        std_rank = val_rank[i].std(axis=-1)
        consensus = np.divide(mean_rank, std_rank + 1e-8)
        for j, ric in enumerate(val_bundle["rics"][i].astype(str)):
            if val_common[i, j]:
                consensus_rows.append({"split": "validation", "country": country, "date": date, "RIC": ric, "consensus_score": float(consensus[j]), "mean_rank": float(mean_rank[j]), "sd_rank": float(std_rank[j]), "target_bps": float(val_bundle["target_bps"][i, j]) if val_bundle["target_mask"][i, j] else np.nan})
    for i, (country, date) in enumerate(zip(dev_bundle["countries"].astype(str), pd.to_datetime(dev_bundle["dates"]))):
        for left, right in combinations(range(len(seeds)), 2):
            pair_rows.append({"split": "development", "country": country, "date": date, "seed_left": seeds[left], "seed_right": seeds[right], "rank_corr": spearman(dev_raw[i, :, left], dev_raw[i, :, right], dev_common[i])})
        mean_rank = dev_rank[i].mean(axis=-1)
        std_rank = dev_rank[i].std(axis=-1)
        consensus = np.divide(mean_rank, std_rank + 1e-8)
        for j, ric in enumerate(dev_bundle["rics"][i].astype(str)):
            if dev_common[i, j]:
                consensus_rows.append({"split": "development", "country": country, "date": date, "RIC": ric, "consensus_score": float(consensus[j]), "mean_rank": float(mean_rank[j]), "sd_rank": float(std_rank[j]), "target_bps": float(dev_bundle["target_bps"][i, j]) if dev_bundle["target_mask"][i, j] else np.nan})
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(args.output_dir / "seed_pairwise_rank_corr.csv", index=False)
    pd.DataFrame(consensus_rows).to_csv(args.output_dir / "seed_consensus_diagnostics.csv", index=False)
    pd.concat(metric_frames, ignore_index=True).to_csv(args.output_dir / "single_seed_recomputed_metrics.csv", index=False)

    # 7–9: E1–E4 validation-only selection and post-selection calibration.
    candidates, weighting = candidate_arrays(val_rank, val_bundle, val_common, seeds)
    candidate_metrics = []
    candidate_daily = {}
    for name, score in candidates.items():
        _, summary = per_model_metrics(val_bundle, score, val_common, name, "validation")
        candidate_metrics.append(summary)
        candidate_daily[name] = summary
        summary.to_csv(args.output_dir / f"ensemble_{name}_metrics.csv", index=False)
    candidate_table = pd.concat(candidate_metrics, ignore_index=True)
    candidate_table.to_csv(args.output_dir / "ensemble_validation_candidates.csv", index=False)
    candidate_equal = candidate_table.groupby("model", as_index=False).mean(numeric_only=True)
    baseline = candidate_equal.loc[candidate_equal.model.eq("E1_mean_rank")].iloc[0]
    eligible = candidate_equal.loc[(candidate_equal.mean_ic >= baseline.mean_ic + 0.005) & (candidate_equal.top_bottom > 0) & (candidate_equal.hit_rate > 0.5)].copy()
    max_weights = {"E1_mean_rank": 0.2, "E2_median_rank": 0.2, "E3_ic_weighted": max(weighting["ic_weights"].values()), "E4_tb_weighted": max(weighting["tb_weights"].values())}
    eligible = eligible.loc[eligible.model.map(max_weights).le(0.5)]
    if eligible.empty:
        selected = "E1_mean_rank"
    else:
        eligible = eligible.sort_values(["mean_ic", "icir", "top_bottom"], ascending=False)
        selected = str(eligible.iloc[0].model)
    selection = candidate_equal.copy()
    selection["max_seed_weight"] = selection.model.map(max_weights)
    selection["selected"] = selection.model.eq(selected)
    selection["selection_rule"] = "validation mean IC; >=0.005 over E1, TB>0, hit>50%, max weight<=0.50; otherwise E1"
    selection.to_csv(args.output_dir / "ensemble_validation_selection.csv", index=False)

    val_selected = candidates[selected]
    dev_candidates = {
        "E1_mean_rank": dev_rank.mean(axis=-1),
        "E2_median_rank": np.median(dev_rank, axis=-1),
        "E3_ic_weighted": np.tensordot(dev_rank, np.asarray(list(weighting["ic_weights"].values())), axes=([-1], [0])),
        "E4_tb_weighted": np.tensordot(dev_rank, np.asarray(list(weighting["tb_weights"].values())), axes=([-1], [0])),
    }
    for name, score in dev_candidates.items():
        _, development_summary = per_model_metrics(dev_bundle, score, dev_common, name, "development")
        development_summary.to_csv(args.output_dir / f"ensemble_{name}_development_metrics.csv", index=False)
    dev_selected = dev_candidates[selected]
    val_target_mask = val_bundle["target_mask"].astype(bool) & val_common
    dev_target_mask = dev_bundle["target_mask"].astype(bool) & dev_common
    betas, calibration = fit_country_betas(val_bundle, val_selected, val_target_mask)
    val_alpha = apply_betas(val_bundle, val_selected, val_target_mask, betas)
    dev_alpha = apply_betas(dev_bundle, dev_selected, dev_target_mask, betas)
    calibration.to_csv(args.output_dir / "ensemble_calibration_summary.csv", index=False)
    selected_validation = args.output_dir / "selected_validation_ensemble.npz"
    selected_development = args.output_dir / "selected_development_ensemble.npz"
    save_prediction_bundle(selected_validation, val_bundle, val_selected, val_common, val_alpha)
    save_prediction_bundle(selected_development, dev_bundle, dev_selected, dev_common, dev_alpha)
    decile_rows = []
    for i, (country, date) in enumerate(zip(val_bundle["countries"].astype(str), pd.to_datetime(val_bundle["dates"]))):
        valid = val_target_mask[i]
        if valid.sum() < 10:
            continue
        order = np.flatnonzero(valid)[np.argsort(val_selected[i, valid], kind="mergesort")]
        for decile, indices in enumerate(np.array_split(order, 10), 1):
            decile_rows.append({"country": country, "date": date, "decile": decile, "n_assets": len(indices), "mean_target_bps": float(np.mean(val_bundle["target_bps"][i, indices]))})
    pd.DataFrame(decile_rows).to_csv(args.output_dir / "ensemble_decile_validation.csv", index=False)
    final_config = {"ensemble_method": selected, "seed_list": seeds, "normalization_method": "rank-normalized on common per-date universe", "universe_alignment_rule": "seed intersection", "validation_period": "V2 validation split", "weights": weighting, "calibration_beta": betas, "calibration_rule": "country positive beta clipped at zero; fit validation only", "lambda": args.risk_aversion, "turnover_cap": args.turnover_cap, "cost_scenarios": ["C0", "C1", "C2"]}
    (args.output_dir / "ensemble_final_config.yaml").write_text("".join(f"{key}: {json.dumps(value)}\n" for key, value in final_config.items()), encoding="utf-8")

    # 10–12: same corrected portfolio engine, PTCST versus risk-only.
    portfolio_root = args.output_dir / "portfolio"
    ptcst_root = portfolio_root / "ptcst"
    risk_root = portfolio_root / "risk_only"
    portfolio_rows = []
    for scenario in ("C0", "C1", "C2"):
        ptcst_run = ptcst_root / scenario
        risk_run = risk_root / scenario
        run_backtest(args, selected_development, ptcst_run, "calibrated", scenario)
        run_backtest(args, selected_development, risk_run, "zero", scenario)
        ptcst = pd.read_csv(ptcst_run / "portfolio_metrics_summary.csv").assign(strategy="PTCST", cost_scenario=scenario)
        risk = pd.read_csv(risk_run / "portfolio_metrics_summary.csv").assign(strategy="risk_only", cost_scenario=scenario)
        portfolio_rows.extend([ptcst, risk])
    portfolio_summary = pd.concat(portfolio_rows, ignore_index=True)
    portfolio_summary.to_csv(args.output_dir / "portfolio_summary.csv", index=False)
    portfolio_summary.loc[portfolio_summary.strategy.eq("PTCST")].to_csv(args.output_dir / "ensemble_portfolio_summary.csv", index=False)
    portfolio_summary.loc[portfolio_summary.strategy.eq("risk_only")].to_csv(args.output_dir / "risk_only_summary.csv", index=False)
    incremental = []
    bootstrap = []
    for scenario in ("C0", "C1", "C2"):
        ptcst = pd.read_csv(ptcst_root / scenario / "daily_portfolio_returns.csv")
        risk = pd.read_csv(risk_root / scenario / "daily_portfolio_returns.csv")
        merged = ptcst.merge(risk, on=["country", "date"], suffixes=("_ptcst", "_risk"))
        for country, group in merged.groupby("country"):
            d = group.net_return_ptcst.to_numpy(float) - group.net_return_risk.to_numpy(float)
            p_sharpe = float(portfolio_summary.loc[(portfolio_summary.strategy == "PTCST") & (portfolio_summary.cost_scenario == scenario) & (portfolio_summary.country == country), "annualized_net_sharpe"].iloc[0])
            r_sharpe = float(portfolio_summary.loc[(portfolio_summary.strategy == "risk_only") & (portfolio_summary.cost_scenario == scenario) & (portfolio_summary.country == country), "annualized_net_sharpe"].iloc[0])
            p_turnover = float(portfolio_summary.loc[(portfolio_summary.strategy == "PTCST") & (portfolio_summary.cost_scenario == scenario) & (portfolio_summary.country == country), "mean_turnover_per_rebalance"].iloc[0])
            r_turnover = float(portfolio_summary.loc[(portfolio_summary.strategy == "risk_only") & (portfolio_summary.cost_scenario == scenario) & (portfolio_summary.country == country), "mean_turnover_per_rebalance"].iloc[0])
            low, high = block_bootstrap(d, args.bootstrap_draws, args.bootstrap_block)
            sharpe_low, sharpe_high = block_bootstrap_delta_sharpe(group.net_return_ptcst.to_numpy(float), group.net_return_risk.to_numpy(float), args.bootstrap_draws, args.bootstrap_block)
            incremental.append({"country": country, "cost_scenario": scenario, "ptcst_sharpe": p_sharpe, "risk_only_sharpe": r_sharpe, "delta_sharpe": p_sharpe - r_sharpe, "mean_delta_return": float(np.mean(d)), "delta_turnover": p_turnover - r_turnover})
            bootstrap.append({"country": country, "cost_scenario": scenario, "mean_delta_return": float(np.mean(d)), "mean_delta_return_ci_low": low, "mean_delta_return_ci_high": high, "delta_sharpe": p_sharpe - r_sharpe, "delta_sharpe_ci_low": sharpe_low, "delta_sharpe_ci_high": sharpe_high, "block_size": args.bootstrap_block, "draws": args.bootstrap_draws})
    pd.DataFrame(incremental).to_csv(args.output_dir / "incremental_value_summary.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(args.output_dir / "incremental_bootstrap_ci.csv", index=False)
    manifest = {"status": "V2.1_RC3_FINAL_AUDIT", "selected_ensemble": selected, "selected_development_sha256": sha256_file(selected_development), "reconciliation_all_match": bool(reconciliation.match_flag.all()), "max_reconciliation_abs_difference": float(reconciliation.abs_difference.max()) if reconciliation.abs_difference.notna().any() else 0.0, "portfolio_engine": "run_asean_v2_daily_backtest.py", "risk_aversion": args.risk_aversion, "seeds": seeds, "decision": "requires human review of ensemble reconciliation, incremental value and Philippines turnover before final freeze"}
    (args.output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
