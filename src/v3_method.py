"""Core PTCST-CA-MVO method for the frozen Vietnam V3 dataset.

The module is deliberately independent of the notebook runtime.  It builds a
date-batched tensor from the frozen daily panel, trains the proposed
patch-temporal/cross-sectional Transformer, estimates past-only covariance and
solves the cost-aware long-only MVO problem.
"""
from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from torch import nn


FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d", "return_60d",
    "vol_5d", "vol_20d", "vol_60d", "log_volume", "log_dollar_volume",
    "log_price", "log_market_cap", "high_low_proxy", "amihud",
    "day_of_week", "is_month_end", "is_quarter_end",
]


@dataclass(frozen=True)
class TensorBundle:
    dates: np.ndarray
    rics: np.ndarray
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    execution_dates: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_v3(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_root)
    weekly = pd.read_parquet(root / "curated" / "universe_weekly.parquet")
    model = pd.read_parquet(root / "model_ready" / "weekly_features_targets.parquet")
    daily = pd.read_parquet(root / "curated" / "daily_panel.parquet")
    weekly["date"] = pd.to_datetime(weekly["date"]).dt.normalize()
    model["date"] = pd.to_datetime(model["date"]).dt.normalize()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return weekly, model, daily


def robust_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(x, axis=(0, 1, 2))
    q75 = np.nanpercentile(x, 75, axis=(0, 1, 2))
    q25 = np.nanpercentile(x, 25, axis=(0, 1, 2))
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    median[~np.isfinite(median)] = 0.0
    return median.astype(np.float32), scale.astype(np.float32)


def robust_apply(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # Impute with train-fitted medians before scaling.  Using zero in raw
    # space would leak an arbitrary level into missing observations whenever
    # the train median is not zero.
    filled = np.where(np.isfinite(x), x, median.reshape((1, 1, 1, -1)))
    z = (filled - median.reshape((1, 1, 1, -1))) / scale.reshape((1, 1, 1, -1))
    return np.clip(z, -10.0, 10.0).astype(np.float32)


def build_tensor_bundle(data_root: Path, split: str, median: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple[TensorBundle, np.ndarray, np.ndarray]:
    weekly, model, daily = read_v3(data_root)
    calendar = pd.DataFrame({"date": sorted(daily["date"].unique())})
    calendar["session_id"] = np.arange(len(calendar), dtype=np.int32)
    sid = dict(zip(calendar["date"], calendar["session_id"]))
    daily = daily.loc[daily["date"].isin(sid)].copy()
    daily["session_id"] = daily["date"].map(sid)
    daily = daily.sort_values(["ric", "session_id"]).drop_duplicates(["ric", "session_id"], keep="last")
    by_ric = {ric: frame.set_index("session_id") for ric, frame in daily.groupby("ric", sort=False)}
    model_key = model.set_index(["date", "ric"])
    selected = weekly.sort_values(["date", "market_cap_rank"])
    selected = selected.groupby("date", sort=True, group_keys=False).head(100)
    selected_dates = [d for d, g in selected.groupby("date", sort=True) if str(g["split"].iloc[0]) == split]
    selected_dates = sorted(selected_dates)
    xs, ys, masks, rics_out, executions, dates_out = [], [], [], [], [], []
    for date in selected_dates:
        group = selected.loc[selected["date"].eq(date)].sort_values("market_cap_rank")
        if len(group) != 100:
            continue
        end = sid.get(date)
        if end is None or end < 59:
            continue
        date_x, date_y, date_mask, date_exec = [], [], [], []
        ok = True
        for ric in group["ric"]:
            frame = by_ric.get(ric)
            if frame is None:
                # Keep the date/asset in the fixed 100-name cross-section;
                # train-fitted imputation will provide a neutral input for a
                # newly listed or otherwise unavailable history window.
                values = np.full((60, len(FEATURES)), np.nan, dtype=np.float32)
            else:
                window = frame.reindex(range(end - 59, end + 1))
                values = window.reindex(columns=FEATURES).to_numpy(dtype=np.float32)
            try:
                row = model_key.loc[(date, ric)]
            except KeyError:
                # The frozen model-ready file is expected to be complete, but
                # do not silently discard an otherwise valid 100-asset date if
                # a row is absent.  Mark its target unavailable instead.
                row = None
            date_x.append(values)
            if row is None:
                date_y.append(0.0)
                date_mask.append(False)
                date_exec.append(date)
            else:
                date_y.append(float(row["target_excess_return_5d_bps"]) if pd.notna(row["target_excess_return_5d_bps"]) else 0.0)
                date_mask.append(bool(row["target_available"]))
                date_exec.append(row["execution_date"] if pd.notna(row["execution_date"]) else date)
        if ok and len(date_x) == 100:
            xs.append(np.asarray(date_x, dtype=np.float32)); ys.append(date_y); masks.append(date_mask)
            rics_out.append(group["ric"].astype(str).to_numpy()); executions.append(date_exec); dates_out.append(date)
    if not xs:
        raise RuntimeError(f"No complete tensor dates for split={split}")
    raw = np.asarray(xs, dtype=np.float32)
    if median is None or scale is None:
        median, scale = robust_fit(raw)
    normalized = robust_apply(raw, median, scale)
    bundle = TensorBundle(np.asarray(dates_out, dtype="datetime64[ns]"), np.asarray(rics_out), normalized, np.asarray(ys, dtype=np.float32), np.asarray(masks, dtype=bool), np.asarray(executions, dtype="datetime64[ns]"))
    return bundle, median, scale


class TemporalTransformer(nn.Module):
    """Vanilla temporal Transformer baseline without patching or cross-attention."""

    def __init__(self, n_features: int = 17, d_model: int = 64, layers: int = 2, heads: int = 4, ffn_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, 60, d_model)); nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, length, _ = x.shape
        z = self.projection(x) + self.position[:, :length]
        z = self.encoder(z.reshape(b * n, length, -1))[:, -1]
        return self.head(z).reshape(b, n)


class PatchTST(nn.Module):
    """PatchTST-style temporal-only encoder; cross-sectional attention is absent."""

    def __init__(self, n_features: int = 17, patch_length: int = 5, d_model: int = 64, layers: int = 2, heads: int = 4, ffn_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        if 60 % patch_length: raise ValueError("lookback must be divisible by patch_length")
        self.patch_length = patch_length; self.patch_count = 60 // patch_length
        self.projection = nn.Linear(n_features * patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model)); nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _, f = x.shape
        patches = x.reshape(b, n, self.patch_count, self.patch_length * f)
        z = self.projection(patches) + self.position[:, :self.patch_count]
        z = self.encoder(z.reshape(b * n, self.patch_count, -1))[:, -1]
        return self.head(z).reshape(b, n)


class PTCST(nn.Module):
    """Patch temporal encoder followed by cross-sectional self-attention."""

    def __init__(self, n_features: int = 17, patch_length: int = 5, d_model: int = 64, temporal_layers: int = 2, cross_layers: int = 1, heads: int = 4, ffn_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        if 60 % patch_length:
            raise ValueError("lookback must be divisible by patch_length")
        self.patch_length = patch_length
        self.patch_count = 60 // patch_length
        self.projection = nn.Linear(n_features * patch_length, d_model)
        self.temporal_position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        nn.init.normal_(self.temporal_position, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(d_model, heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(temporal_layer, temporal_layers)
        self.cross_position = nn.Parameter(torch.zeros(1, 100, d_model))
        nn.init.normal_(self.cross_position, std=0.02)
        cross_layer = nn.TransformerEncoderLayer(d_model, heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.cross = nn.TransformerEncoder(cross_layer, cross_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, length, f = x.shape
        patches = x.reshape(b, n, self.patch_count, self.patch_length * f)
        z = self.projection(patches) + self.temporal_position[:, : self.patch_count]
        z = self.temporal(z.reshape(b * n, self.patch_count, -1))[:, -1].reshape(b, n, -1)
        z = self.cross(z + self.cross_position[:, :n])
        return self.head(z).squeeze(-1)


def spearman_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for p, y, m in zip(pred, target, mask):
        if m.sum() < 3: continue
        values.append(pd.Series(p[m]).rank().corr(pd.Series(y[m]).rank()))
    return float(np.nanmean(values)) if values else float("nan")


def make_forecast_model(model_type: str, n_features: int) -> nn.Module:
    key = model_type.lower().replace("-", "").replace("_", "")
    if key in ("temporal", "temporaltransformer", "vanillatransformer"):
        return TemporalTransformer(n_features=n_features)
    if key in ("patchtst",):
        return PatchTST(n_features=n_features)
    if key in ("ptcst", "proposed"):
        return PTCST(n_features=n_features)
    raise ValueError(f"unknown model_type={model_type}")


def train_ptcst(train: TensorBundle, validation: TensorBundle, output_dir: Path, epochs: int = 100, seed: int = 7, batch_dates: int = 16, device: str | None = None, early_stopping_patience: int = 10, model_type: str = "PTCST", checkpoint_sync_dir: Path | None = None, resume_checkpoint: Path | None = None) -> dict[str, object]:
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_forecast_model(model_type, train.x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0, reduction="none")
    tx, ty, tm = torch.from_numpy(train.x), torch.from_numpy(train.y), torch.from_numpy(train.mask)
    vx, vy, vm = torch.from_numpy(validation.x), torch.from_numpy(validation.y), torch.from_numpy(validation.mask)
    best_ic, best_epoch = -np.inf, 0
    history = []
    stale_epochs = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        history = list(state.get("history", []))
        best_ic = float(state.get("best_ic", -np.inf))
        best_epoch = int(state.get("best_epoch", 0))
        stale_epochs = int(state.get("stale_epochs", 0))
        start_epoch = int(state.get("epoch", 0)) + 1
    if checkpoint_sync_dir is not None:
        checkpoint_sync_dir = Path(checkpoint_sync_dir)
        checkpoint_sync_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs + 1):
        should_stop = False
        model.train(); order = np.random.permutation(len(tx)); train_loss = []
        for start in range(0, len(order), batch_dates):
            idx = torch.as_tensor(order[start:start + batch_dates])
            pred = model(tx[idx].to(device))
            loss = loss_fn(pred, ty[idx].to(device)).masked_select(tm[idx].to(device)).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); train_loss.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): val_pred = model(vx.to(device)).cpu().numpy()
        ic = spearman_ic(val_pred, validation.y, validation.mask)
        record = {"epoch": epoch, "train_loss": float(np.mean(train_loss)), "validation_spearman_ic": ic}; history.append(record)
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "model_type": model_type, "seed": seed, "epoch": epoch, "validation_spearman_ic": ic}, output_dir / "best.pt")
        else:
            stale_epochs += 1
            if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                should_stop = True
        last_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_type": model_type,
            "seed": seed,
            "epoch": epoch,
            "best_ic": float(best_ic),
            "best_epoch": int(best_epoch),
            "stale_epochs": int(stale_epochs),
            "history": history,
        }
        last_path = output_dir / "last.pt"
        torch.save(last_state, last_path)
        if checkpoint_sync_dir is not None:
            shutil.copy2(last_path, checkpoint_sync_dir / "last.pt")
            best_path = output_dir / "best.pt"
            if best_path.exists():
                shutil.copy2(best_path, checkpoint_sync_dir / "best.pt")
        if should_stop:
            break
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model_type": model_type, "best_epoch": best_epoch, "best_validation_spearman_ic": float(best_ic), "epochs_completed": len(history), "early_stopping_patience": early_stopping_patience, "device": device, "seed": seed}


def predict_ptcst(bundle: TensorBundle, checkpoint: Path, device: str | None = None, model_type: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    model_type = model_type or state.get("model_type", "PTCST")
    model = make_forecast_model(model_type, bundle.x.shape[-1]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    with torch.no_grad(): return model(torch.from_numpy(bundle.x).to(device)).cpu().numpy()


def ledoit_covariance(daily: pd.DataFrame, date: pd.Timestamp, rics: list[str], window: int = 252) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    frame = daily.copy(); frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    calendar = np.sort(frame["date"].unique()); positions = {d: i for i, d in enumerate(calendar)}
    end = positions.get(pd.Timestamp(date).to_datetime64())
    if end is None: return np.eye(len(rics)) * 1e-4, np.zeros(len(rics), dtype=bool), {"fallback": "missing_signal_date"}
    dates = calendar[max(0, end - window + 1): end + 1]
    pivot = frame.loc[frame["ric"].isin(rics) & frame["date"].isin(dates)].pivot_table(index="date", columns="ric", values="return", aggfunc="last").reindex(index=dates, columns=rics)
    valid = pivot.notna().all(axis=0).to_numpy()
    covariance = np.eye(len(rics), dtype=np.float64) * 1e-4
    if valid.sum() >= 20:
        fit = LedoitWolf().fit(pivot.loc[:, valid].to_numpy())
        covariance[np.ix_(valid, valid)] = fit.covariance_
        status = {"fallback": None, "valid_assets": int(valid.sum()), "window_rows": int(len(pivot))}
    else:
        status = {"fallback": "fewer_than_20_complete_assets", "valid_assets": int(valid.sum()), "window_rows": int(len(pivot))}
    covariance = (covariance + covariance.T) / 2
    return covariance, valid, status


def cost_aware_mvo(mu: np.ndarray, covariance: np.ndarray, w_pre: np.ndarray, valid: np.ndarray, risk_aversion: float = 10.0, cost: float = 0.001, max_weight: float = 0.05, turnover_cap: float = 0.40, turnover_fixed: float = 0.0) -> tuple[np.ndarray, dict[str, object]]:
    n = len(mu); w = cp.Variable(n)
    variable_turnover = cp.sum(cp.abs(w - w_pre))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight, variable_turnover + float(turnover_fixed) <= turnover_cap]
    fixed = np.flatnonzero(~valid)
    if len(fixed): constraints.append(w[fixed] == w_pre[fixed])
    sigma = np.asarray(covariance, dtype=float) + np.eye(n) * 1e-8
    objective = cp.Maximize(mu @ w - risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(sigma)) - cost * (variable_turnover + float(turnover_fixed)))
    problem = cp.Problem(objective, constraints); status = None
    for solver in ("CLARABEL", "OSQP"):
        try:
            problem.solve(solver=solver, verbose=False)
            status = solver
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None: break
        except Exception:
            continue
    if w.value is None or problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return w_pre.copy(), {"solver": status, "status": str(problem.status), "fallback": "w_pre", "turnover_fixed": float(turnover_fixed)}
    result = np.asarray(w.value).reshape(-1); result[result < 0] = 0; result /= result.sum()
    return result, {"solver": status, "status": str(problem.status), "fallback": None, "objective": float(problem.value), "turnover_fixed": float(turnover_fixed)}
