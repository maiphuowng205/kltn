"""Train protocol-V2 pooled ASEAN PTCST with a hybrid ranking objective."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.asean_v2 import (ASEANPTCST, build_bundle, country_ids, daily_ic, fit_score_calibrator, hybrid_loss, save_json, seed_everything, apply_score_calibrator)


def predict(model, bundle, device):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(bundle.x).to(device), torch.from_numpy(bundle.asset_mask).to(device), torch.from_numpy(country_ids(bundle.countries)).to(device)).cpu().numpy()


def train_one(train, validation, output: Path, seed: int, epochs: int, device: str) -> dict:
    seed_everything(seed); output.mkdir(parents=True, exist_ok=True)
    model=ASEANPTCST(train.x.shape[-1]).to(device); optimiser=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    x=torch.from_numpy(train.x); y=torch.from_numpy(train.y); m=torch.from_numpy(train.target_mask & train.asset_mask); a=torch.from_numpy(train.asset_mask); c=torch.from_numpy(country_ids(train.countries))
    best=(-np.inf, 0); history=[]
    for epoch in range(1, epochs+1):
        model.train(); order=np.random.permutation(len(x)); losses=[]
        for start in range(0, len(order), 16):
            index=torch.as_tensor(order[start:start+16]); p=model(x[index].to(device), a[index].to(device), c[index].to(device)); loss, details=hybrid_loss(p, y[index].to(device), m[index].to(device)); optimiser.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step(); losses.append(details)
        val_prediction=predict(model, validation, device); ic=daily_ic(val_prediction, validation.y, validation.target_mask & validation.asset_mask)
        record={"epoch": epoch, "validation_spearman_ic": ic, "huber_loss": float(np.mean([z['huber'] for z in losses])), "ranking_loss": float(np.mean([z['rank'] for z in losses]))}; history.append(record)
        if np.isfinite(ic) and ic > best[0]:
            best=(float(ic), epoch); torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch, "validation_ic": float(ic)}, output / "best.pt")
        if epoch-best[1] >= 12: break
    save_json(output / "training_history.json", history); save_json(output / "metrics.json", {"seed":seed, "best_validation_spearman_ic":best[0], "best_epoch":best[1], "epochs_completed":len(history), "loss":"0.25 Huber + 0.75 pairwise logistic ranking"})
    model.load_state_dict(torch.load(output / "best.pt", map_location=device)["model"]); return {"model":model, "metrics":{"seed":seed,"best_validation_spearman_ic":best[0],"best_epoch":best[1]}}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, default=ROOT / "artifacts" / "asean_v2"); p.add_argument("--run-root", type=Path, default=ROOT / "runs" / "asean_v2_pooled_ptcst"); p.add_argument("--epochs", type=int, default=100); p.add_argument("--seeds", default="7,19,31,43,59"); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a=p.parse_args()
    train, med, scale=build_bundle(a.data_root, "2019-01-01", "2022-12-31")
    validation, _, _=build_bundle(a.data_root, "2023-01-01", "2023-12-31", med, scale)
    development, _, _=build_bundle(a.data_root, "2024-01-01", "2025-12-31", med, scale)
    a.run_root.mkdir(parents=True, exist_ok=True); save_json(a.run_root / "protocol.json", {"status":"development only", "timing":"t signal; t+1 close execution; t+2:t+6 target", "target":"cross-sectional excess return", "universe":"pure variable-N Top-100", "risk_min_history":126, "seeds":a.seeds})
    outcomes=[]
    for seed in [int(v) for v in a.seeds.split(",") if v.strip()]:
        result=train_one(train, validation, a.run_root / f"seed_{seed}", seed, a.epochs, a.device); model=result["model"]
        val_pred=predict(model, validation, a.device); dev_pred=predict(model, development, a.device)
        calibrator=fit_score_calibrator(val_pred, validation.y, validation.target_mask & validation.asset_mask); alpha=apply_score_calibrator(dev_pred, development.asset_mask, calibrator)
        np.savez_compressed(a.run_root / f"seed_{seed}" / "development_predictions.npz", dates=development.dates, countries=development.countries, rics=development.rics, raw_score=dev_pred, calibrated_alpha_decimal=alpha, target_bps=development.y, target_mask=development.target_mask, asset_mask=development.asset_mask)
        save_json(a.run_root / f"seed_{seed}" / "calibration.json", calibrator)
        outcomes.append({**result["metrics"], "development_spearman_ic":daily_ic(dev_pred, development.y, development.target_mask & development.asset_mask), "calibration_beta_decimal_per_z":calibrator["beta_decimal_per_z"]})
    save_json(a.run_root / "seed_summary.json", outcomes); print(outcomes)


if __name__ == "__main__": main()
