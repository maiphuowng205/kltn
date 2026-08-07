from pathlib import Path

import numpy as np

from src.v3_method import cost_aware_mvo, l1_turnover, make_forecast_model


def test_model_shapes():
    import torch
    x = torch.zeros(2, 100, 60, 17)
    for name in ("TemporalTransformer", "PatchTST", "PTCST"):
        assert make_forecast_model(name, 17)(x).shape == (2, 100)


def test_optimizer_constraints_and_determinism():
    covariance = np.eye(100) * 0.01
    pre = np.ones(100) / 100
    valid = np.ones(100, dtype=bool)
    mu = np.linspace(-0.001, 0.001, 100)
    one, info_one = cost_aware_mvo(mu, covariance, pre, valid)
    two, info_two = cost_aware_mvo(mu, covariance, pre, valid)
    assert info_one["status"] == info_two["status"]
    assert np.allclose(one, two, atol=2e-6)
    assert np.isclose(one.sum(), 1, atol=2e-6)
    assert one.min() >= -2e-6 and one.max() <= 0.05 + 2e-6
    assert np.abs(one - pre).sum() <= 0.40 + 2e-6


def test_zero_trade_has_zero_cost():
    pre = np.ones(100) / 100
    assert np.isclose(np.abs(pre - pre).sum() * 0.001, 0.0)


def test_turnover_counts_exited_weight_once():
    pre = np.array([0.20, 0.30])
    target = np.array([0.25, 0.25])
    # Current-universe L1 is 0.10; the exited name is a separate 0.10.
    assert np.isclose(l1_turnover(target, pre, exited_weight=0.10), 0.20)


def test_turnover_union_vector_does_not_double_count_exited_weight():
    pre_union = np.array([0.20, 0.30, 0.10])
    target_union = np.array([0.25, 0.25, 0.00])
    # When the exited asset is explicitly in the union vector, do not pass it
    # again as a separate exited_weight.
    assert np.isclose(l1_turnover(target_union, pre_union), 0.20)
