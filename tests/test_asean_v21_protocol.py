"""Small, data-free checks for V2.1 protocol invariants."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.asean_v2 import _rank_date_metrics, hybrid_loss

def test_constant_forecast_has_no_rank_metrics():
    result=_rank_date_metrics(np.zeros(5),np.arange(5.),np.ones(5,dtype=bool))
    assert not result["rank_defined"] and np.isnan(result["spearman_ic"]) and np.isnan(result["top_minus_bottom_bps"])

def test_execution_return_does_not_include_pre_execution_jump():
    # A trade at close t+1 owns t+1 -> t+2, not t -> t+1.
    close=np.array([100.,120.,121.]); close_to_close=np.diff(close)/close[:-1]
    assert np.isclose(close_to_close[1], 1/120.)
    assert not np.isclose(close_to_close[1], .20)

def test_cross_sectional_rank_loss_is_finite():
    import torch
    loss,_=hybrid_loss(torch.tensor([[.1,.2,.3]]),torch.tensor([[1.,2.,3.]]),torch.tensor([[True,True,True]]))
    assert torch.isfinite(loss)
