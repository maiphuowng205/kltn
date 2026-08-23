"""Variable-universe, pooled ASEAN forecasting primitives for protocol V2."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cvxpy as cp
from torch import nn


BASE_FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d", "return_60d",
    "vol_5d", "vol_20d", "vol_60d", "log_volume", "log_dollar_volume",
    "log_price", "log_market_cap", "high_low_proxy", "amihud", "quoted_spread_bps",
    "day_of_week", "is_month_end", "is_quarter_end",
]
REGIME_FEATURES = [
    "market_return_5d", "market_return_20d", "market_return_60d", "market_vol_20d",
    "market_vol_60d", "cross_sectional_dispersion", "market_breadth",
    "median_stock_vol_20d", "market_distance_200d",
]
FEATURES_V2 = BASE_FEATURES + REGIME_FEATURES
COUNTRIES = ("Indonesia", "Malaysia", "Philippines", "Singapore", "Thailand")


@dataclass(frozen=True)
class VariableBundle:
    dates: np.ndarray
    countries: np.ndarray
    rics: np.ndarray
    x: np.ndarray
    y: np.ndarray
    target_mask: np.ndarray
    asset_mask: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def robust_fit(x: np.ndarray, asset_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expanded = np.broadcast_to(asset_mask[:, :, None, None], x.shape)
    values = np.where(expanded, x, np.nan)
    median = np.nanmedian(values, axis=(0, 1, 2))
    scale = np.nanpercentile(values, 75, axis=(0, 1, 2)) - np.nanpercentile(values, 25, axis=(0, 1, 2))
    median[~np.isfinite(median)] = 0.0; scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return median.astype("float32"), scale.astype("float32")


def robust_apply(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, median.reshape(1, 1, 1, -1))
    return np.clip((filled - median.reshape(1, 1, 1, -1)) / scale.reshape(1, 1, 1, -1), -10, 10).astype("float32")


def build_bundle(root: Path, start: str, end: str, median: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple[VariableBundle, np.ndarray, np.ndarray]:
    root = Path(root)
    panel_path=root / "curated" / "daily_panel_v2"
    weekly_path=root / "model_ready" / "weekly_features_targets_v2"
    if not panel_path.exists(): panel_path=panel_path.with_suffix(".parquet")
    if not weekly_path.exists(): weekly_path=weekly_path.with_suffix(".parquet")
    panel = pd.read_parquet(panel_path)
    weekly = pd.read_parquet(weekly_path)
    for frame in (panel, weekly): frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    weekly = weekly.loc[weekly.model_eligible_v2 & weekly.date.between(start, end)].copy()
    if weekly.empty: raise RuntimeError(f"No V2 dates for {start} to {end}")
    panel = panel.sort_values(["country", "ric", "session_id"]).drop_duplicates(["country", "ric", "session_id"], keep="last")
    by_asset = {(country, ric): g.set_index("session_id") for (country, ric), g in panel.groupby(["country", "ric"], sort=False)}
    max_assets, lookback = 100, 60
    x_rows=[]; y_rows=[]; target_rows=[]; asset_rows=[]; ric_rows=[]; dates=[]; countries=[]
    for (country, date), group in weekly.sort_values(["country", "date", "market_cap_rank"]).groupby(["country", "date"], sort=True):
        group = group.head(max_assets); xs=np.zeros((max_assets, lookback, len(FEATURES_V2)), dtype="float32")
        ys=np.zeros(max_assets, dtype="float32"); tm=np.zeros(max_assets, dtype=bool); am=np.zeros(max_assets, dtype=bool); rics=np.full(max_assets, "", dtype="U64")
        for j, row in enumerate(group.itertuples(index=False)):
            frame = by_asset.get((country, row.ric))
            if frame is None: continue
            window = frame.reindex(range(int(row.session_id) - lookback + 1, int(row.session_id) + 1))
            if len(window) != lookback: continue
            xs[j] = window.reindex(columns=FEATURES_V2).to_numpy(dtype="float32")
            ys[j] = float(row.target_cs_excess_return_5d_bps_v2) if pd.notna(row.target_cs_excess_return_5d_bps_v2) else 0.0
            tm[j] = bool(row.target_available_v2); am[j] = True; rics[j] = str(row.ric)
        if am.sum() >= 3:
            x_rows.append(xs); y_rows.append(ys); target_rows.append(tm); asset_rows.append(am); ric_rows.append(rics); dates.append(date); countries.append(country)
    if not x_rows: raise RuntimeError("No variable-N tensors could be built")
    raw=np.asarray(x_rows); assets=np.asarray(asset_rows)
    if median is None or scale is None: median, scale = robust_fit(raw, assets)
    return VariableBundle(np.asarray(dates, dtype="datetime64[ns]"), np.asarray(countries), np.asarray(ric_rows), robust_apply(raw, median, scale), np.asarray(y_rows), np.asarray(target_rows), assets), median, scale


class ASEANPTCST(nn.Module):
    """Shared temporal encoder, within-market cross-sectional attention and country heads."""
    def __init__(self, n_features: int, countries: int = len(COUNTRIES), d_model: int = 64, heads: int = 4, dropout: float = .10):
        super().__init__()
        self.patch_count, self.patch_length = 12, 5
        self.projection = nn.Linear(n_features * self.patch_length, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.patch_count, d_model)); nn.init.normal_(self.temporal_pos, std=.02)
        temporal = nn.TransformerEncoderLayer(d_model, heads, 128, dropout, batch_first=True, norm_first=True)
        cross = nn.TransformerEncoderLayer(d_model, heads, 128, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(temporal, 2); self.cross = nn.TransformerEncoder(cross, 1)
        self.cross_pos = nn.Parameter(torch.zeros(1, 100, d_model)); nn.init.normal_(self.cross_pos, std=.02)
        self.country_embedding = nn.Embedding(countries, d_model)
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(countries)])

    def forward(self, x: torch.Tensor, asset_mask: torch.Tensor, country_ids: torch.Tensor) -> torch.Tensor:
        b, n, _, f = x.shape
        z = x.reshape(b, n, self.patch_count, self.patch_length * f)
        z = self.projection(z) + self.temporal_pos
        z = self.temporal(z.reshape(b*n, self.patch_count, -1))[:, -1].reshape(b, n, -1)
        z = z + self.cross_pos[:, :n] + self.country_embedding(country_ids).unsqueeze(1)
        z = self.cross(z, src_key_padding_mask=~asset_mask)
        output = torch.zeros((b, n), device=x.device)
        for cid, head in enumerate(self.heads):
            rows = country_ids.eq(cid)
            if rows.any(): output[rows] = head(z[rows]).squeeze(-1)
        return output


def hybrid_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, huber_weight: float = .25, rank_weight: float = .75) -> tuple[torch.Tensor, dict[str, float]]:
    valid = mask.bool()
    point = nn.functional.huber_loss(pred[valid], (target[valid] / 100.0), delta=1.0) if valid.any() else pred.sum() * 0
    rank_terms=[]
    for p, y, m in zip(pred, target / 100.0, valid):
        p, y = p[m], y[m]
        if len(p) < 2: continue
        diff_y = y[:, None] - y[None, :]; upper = torch.triu(torch.ones_like(diff_y, dtype=torch.bool), diagonal=1) & diff_y.ne(0)
        if upper.any(): rank_terms.append(nn.functional.softplus(-torch.sign(diff_y[upper]) * (p[:, None] - p[None, :])[upper]).mean())
    ranking = torch.stack(rank_terms).mean() if rank_terms else point * 0
    loss = huber_weight * point + rank_weight * ranking
    return loss, {"huber": float(point.detach()), "rank": float(ranking.detach())}


def country_ids(countries: np.ndarray) -> np.ndarray:
    lookup={name: i for i, name in enumerate(COUNTRIES)}
    return np.asarray([lookup[str(name)] for name in countries], dtype="int64")


def daily_ic(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    values=[]
    for p, y, m in zip(prediction, target, mask):
        if m.sum() >= 3: values.append(pd.Series(p[m]).rank().corr(pd.Series(y[m]).rank()))
    return float(np.nanmean(values)) if values else float("nan")


def _rank_date_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Metrics for one country/date. Constant forecasts have no rank signal."""
    p=np.asarray(prediction, float)[mask]; y=np.asarray(target, float)[mask]
    if len(p) < 3: return {"n_assets":int(len(p)),"spearman_ic":np.nan,"pearson_ic":np.nan,"top_minus_bottom_bps":np.nan,"mae_bps":np.nan,"rmse_bps":np.nan,"directional_accuracy":np.nan,"cs_prediction_std_bps":np.nan,"rank_defined":False}
    constant=bool(np.nanstd(p) <= 1e-12)
    order=np.argsort(p); k=max(1,int(np.floor(len(p)*.2)))
    return {"n_assets":int(len(p)),"spearman_ic":np.nan if constant else float(pd.Series(p).rank().corr(pd.Series(y).rank())),"pearson_ic":np.nan if constant else float(np.corrcoef(p,y)[0,1]),"top_minus_bottom_bps":np.nan if constant else float(y[order[-k:]].mean()-y[order[:k]].mean()),"mae_bps":float(np.abs(p-y).mean()),"rmse_bps":float(np.sqrt(np.mean((p-y)**2))),"directional_accuracy":np.nan if constant else float(np.mean(np.sign(p)==np.sign(y))),"cs_prediction_std_bps":float(np.std(p,ddof=0)),"rank_defined":not constant}


def bootstrap_time_ci(values: np.ndarray, seed: int = 7, draws: int = 2_000) -> tuple[float, float]:
    """Non-parametric bootstrap over rebalance dates, never individual stocks."""
    x=np.asarray(values,float); x=x[np.isfinite(x)]
    if len(x)<2: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); sample=rng.choice(x,size=(draws,len(x)),replace=True).mean(axis=1)
    return float(np.quantile(sample,.025)),float(np.quantile(sample,.975))


def summarize_forecasts(rows: pd.DataFrame, bootstrap_seed: int = 7) -> pd.DataFrame:
    """Produce the locked, country/model-level V2 forecast table."""
    needed={"country","date","model","prediction_bps","target_bps","target_available"}
    if missing:=needed.difference(rows): raise ValueError(f"Missing forecast columns: {sorted(missing)}")
    per_date=[]
    for (country,model,date),group in rows.groupby(["country","model","date"],sort=True):
        valid=group.target_available.to_numpy(bool) & np.isfinite(group.prediction_bps) & np.isfinite(group.target_bps)
        result={"country":country,"model":model,"date":pd.Timestamp(date)}; result.update(_rank_date_metrics(group.prediction_bps.to_numpy(),group.target_bps.to_numpy(),valid)); per_date.append(result)
    daily=pd.DataFrame(per_date); summaries=[]
    for (country,model),group in daily.groupby(["country","model"],sort=True):
        rank=group.loc[group.rank_defined]
        original=rows.loc[(rows.country.eq(country))&(rows.model.eq(model))&rows.target_available].copy()
        p=original.prediction_bps.to_numpy(float); y=original.target_bps.to_numpy(float); valid=np.isfinite(p)&np.isfinite(y)
        dispersion_ratio=float(np.std(p[valid],ddof=0)/np.std(y[valid],ddof=0)) if valid.sum()>1 and np.std(y[valid])>0 else np.nan
        slope=float(np.cov(p[valid],y[valid],ddof=0)[0,1]/np.var(p[valid])) if valid.sum()>1 and np.var(p[valid])>1e-12 else np.nan
        mean_ic=float(rank.spearman_ic.mean()) if len(rank) else np.nan; sd_ic=float(rank.spearman_ic.std(ddof=1)) if len(rank)>1 else np.nan
        lo,hi=bootstrap_time_ci(rank.spearman_ic.to_numpy(),bootstrap_seed)
        summaries.append({"country":country,"model":model,"forecast_dates":int(len(group)),"rank_defined_dates":int(len(rank)),"mean_spearman_ic":mean_ic,"mean_ic_ci_low":lo,"mean_ic_ci_high":hi,"median_spearman_ic":float(rank.spearman_ic.median()) if len(rank) else np.nan,"ic_hit_rate":float((rank.spearman_ic>0).mean()) if len(rank) else np.nan,"icir":float(mean_ic/sd_ic) if np.isfinite(sd_ic) and sd_ic>0 else np.nan,"top_minus_bottom_5d_bps":float(rank.top_minus_bottom_bps.mean()) if len(rank) else np.nan,"pearson_ic":float(rank.pearson_ic.mean()) if len(rank) else np.nan,"mae_bps":float(group.mae_bps.mean()),"rmse_bps":float(np.sqrt(np.nanmean(group.rmse_bps.to_numpy()**2))),"directional_accuracy":float(rank.directional_accuracy.mean()) if len(rank) else np.nan,"median_cs_forecast_dispersion_bps":float(group.cs_prediction_std_bps.median()),"dispersion_ratio":dispersion_ratio,"calibration_slope":slope})
    return daily, pd.DataFrame(summaries)


def summarize_daily_portfolio(daily: pd.DataFrame, rebalances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locked daily portfolio and reliability tables; daily annualization is 252."""
    performance=[]; reliability=[]
    for country,group in daily.groupby("country",sort=True):
        r=group.net_return.to_numpy(float); r=r[np.isfinite(r)]; gross=group.gross_return.to_numpy(float); cost=group.cost.to_numpy(float)
        wealth=np.cumprod(1+r) if len(r) else np.array([]); drawdown=wealth/np.maximum.accumulate(wealth)-1 if len(wealth) else np.array([])
        annual_return=float(np.mean(r)*252) if len(r) else np.nan; annual_vol=float(np.std(r,ddof=1)*np.sqrt(252)) if len(r)>1 else np.nan
        sharpe=float(annual_return/annual_vol) if np.isfinite(annual_vol) and annual_vol>0 else np.nan
        rb=rebalances.loc[rebalances.country.eq(country)] if len(rebalances) else rebalances
        mean_turnover=float(rb.turnover.mean()) if len(rb) else np.nan; annual_turnover=float(rb.turnover.sum()/(len(r)/252)) if len(r) else np.nan
        cumulative_cost_drag=float(np.nansum(gross)-np.nansum(r)); efficiency=float(annual_return/annual_turnover) if np.isfinite(annual_turnover) and annual_turnover>0 else np.nan
        held_asset_days=int(group.positions.fillna(0).sum()) if "positions" in group else 0
        missing_events=int(group.missing_valuations.fillna(0).sum()) if "missing_valuations" in group else 0
        missing_rate=float(missing_events/held_asset_days) if held_asset_days else np.nan
        performance.append({"country":country,"evaluation_days":int(len(r)),"annualized_net_excess_return":annual_return,"annualized_volatility":annual_vol,"annualized_net_sharpe":sharpe,"maximum_drawdown":float(drawdown.min()) if len(drawdown) else np.nan,"mean_turnover_per_rebalance":mean_turnover,"annualized_turnover":annual_turnover,"cumulative_cost_drag":cumulative_cost_drag,"cost_drag":cumulative_cost_drag,"return_per_unit_turnover":efficiency})
        reliability.append({"country":country,"scheduled_rebalances":int(len(rb)),"evaluation_coverage":float(len(r)/len(group)) if len(group) else np.nan,"mean_eligible_n":float(rb.assets.mean()) if len(rb) else np.nan,"mean_valid_risk_n":float(rb.valid_risk_assets.mean()) if len(rb) else np.nan,"covariance_fallback_rate":float(rb.risk_fallback.notna().mean()) if len(rb) else np.nan,"solver_fallback_rate":float(rb.solver_fallback.notna().mean()) if len(rb) and "solver_fallback" in rb else np.nan,"missing_valuation_events":missing_events,"held_asset_days":held_asset_days,"missing_valuation_event_rate_per_asset_day":missing_rate,"missing_events_per_evaluation_day":float(missing_events/max(len(group),1))})
    return pd.DataFrame(performance),pd.DataFrame(reliability)


def fit_score_calibrator(prediction: np.ndarray, target_bps: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    zs=[]; ys=[]
    for p, y, m in zip(prediction, target_bps, mask):
        if m.sum() < 3: continue
        score=p[m]; sd=score.std()
        if sd <= 1e-12: continue
        zs.extend(((score-score.mean())/sd).tolist()); ys.extend((y[m]/10000.0).tolist())
    z=np.asarray(zs); y=np.asarray(ys)
    beta=float((z@y)/(z@z)) if len(z) and (z@z)>0 else 0.0
    return {"method": "validation_cross_section_zscore_ols", "beta_decimal_per_z": beta, "observations": int(len(z))}


def apply_score_calibrator(prediction: np.ndarray, asset_mask: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    alpha=np.zeros_like(prediction, dtype="float64")
    for i, (p, m) in enumerate(zip(prediction, asset_mask)):
        if m.sum() < 2: continue
        score=p[m]; sd=score.std()
        if sd > 1e-12: alpha[i, m] = calibrator["beta_decimal_per_z"] * (score-score.mean())/sd
    return alpha


def ledoit_covariance_min_history(daily: pd.DataFrame, signal_date: pd.Timestamp, rics: list[str], minimum_history: int = 126) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Past-only Ledoit-Wolf covariance on a shared, complete minimum window."""
    frame = daily.copy(); frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    calendar = pd.DatetimeIndex(sorted(frame.date.unique()))
    try: end = calendar.get_loc(pd.Timestamp(signal_date).normalize())
    except KeyError: return np.eye(len(rics))*1e-4, np.zeros(len(rics), bool), {"fallback":"missing_signal_date"}
    dates=calendar[max(0, end-minimum_history+1):end+1]
    pivot=frame.loc[frame.ric.isin(rics) & frame.date.isin(dates)].pivot_table(index="date", columns="ric", values="return", aggfunc="last").reindex(index=dates, columns=rics)
    valid=pivot.notna().all(axis=0).to_numpy()
    covariance=np.eye(len(rics))*1e-4
    if valid.sum() >= 20:
        from sklearn.covariance import LedoitWolf
        covariance[np.ix_(valid,valid)] = LedoitWolf().fit(pivot.loc[:,valid].to_numpy()).covariance_
        status={"fallback":None,"valid_assets":int(valid.sum()),"window_rows":len(dates)}
    else: status={"fallback":"fewer_than_20_complete_assets","valid_assets":int(valid.sum()),"window_rows":len(dates)}
    return (covariance+covariance.T)/2, valid, status


def cost_aware_mvo_vector_cost(alpha: np.ndarray, covariance: np.ndarray, w_pre: np.ndarray, valid: np.ndarray, costs: np.ndarray, risk_aversion: float = 10.0, max_weight: float = .05, turnover_cap: float = .40, exited_turnover: float = 0., exited_cost: float = 0.) -> tuple[np.ndarray, dict[str, object]]:
    """Long-only MVO with stock-specific one-way costs in decimal-return units."""
    n=len(alpha); w=cp.Variable(n); turn=cp.abs(w-w_pre); costs=np.asarray(costs, float)
    constraints=[cp.sum(w)==1,w>=0,w<=max_weight,cp.sum(turn)+float(exited_turnover)<=turnover_cap]
    invalid=np.flatnonzero(~np.asarray(valid,bool))
    if len(invalid): constraints.append(w[invalid]==w_pre[invalid])
    sigma=np.asarray(covariance,float)+np.eye(n)*1e-8
    objective=cp.Maximize(np.asarray(alpha,float)@w-risk_aversion/2*cp.quad_form(w,cp.psd_wrap(sigma))-costs@turn-float(exited_cost))
    problem=cp.Problem(objective,constraints); solver=None
    for candidate in ("CLARABEL","OSQP"):
        try:
            problem.solve(solver=candidate, verbose=False); solver=candidate
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None: break
        except Exception: continue
    if w.value is None or problem.status not in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE):
        return w_pre.copy(), {"solver":solver,"status":str(problem.status),"fallback":"w_pre"}
    result=np.maximum(np.asarray(w.value).reshape(-1),0); result/=result.sum()
    return result, {"solver":solver,"status":str(problem.status),"fallback":None,"objective":float(problem.value)}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
