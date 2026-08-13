#!/usr/bin/env python3
"""Dependency-light release smoke test; requires project runtime dependencies only."""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import algorithms._GITBO as gitbo_module
from algorithms.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from fas_wca.evaluators import LineProcessEvaluator
from fas_wca.sigma_ball import cube_to_sigma_ball
from fas_wca.sram import SramReadDelayProblem


class _FakeBarDistribution:
    def variance(self, logits):
        return torch.full_like(logits, 4.0)


class _FakeAcquisitionBarDistribution:
    def ei(self, logits, best_f):
        return torch.ones_like(logits[..., 0]) * best_f

    def ucb(self, logits, best_f, rest_prob):
        del best_f, rest_prob
        return torch.full_like(logits[..., 0], 2.0)


class _QueryOnlyRegressor:
    def __init__(self, **_kwargs):
        pass

    def forward(self, X, _Y, single_eval_pos):
        return {"standard": X[single_eval_pos:, :, 0]}

    def predict_mean(self, logits):
        return logits

    def predict_variance(self, logits):
        return torch.ones_like(logits)

    def predict_ei(self, logits, _best_f):
        return logits.square()

    def predict_ucb(self, logits, _best_f, _percentile):
        return logits + 1.0


class _MockSramEvaluator:
    def evaluate(self, vector):
        return 2.9e-10 + np.linalg.norm(vector) * 1e-13


def check_i004() -> None:
    wrapper = VanillaDirectTabPFNRegressor.__new__(VanillaDirectTabPFNRegressor)
    wrapper.bardist_ = _FakeBarDistribution()
    wrapper.y_std = torch.tensor([[[3.0]]])
    result = wrapper.predict_variance(torch.zeros(2, 1))
    torch.testing.assert_close(result, torch.full((2, 1), 36.0))


def check_acquisition_units() -> None:
    wrapper = VanillaDirectTabPFNRegressor.__new__(VanillaDirectTabPFNRegressor)
    wrapper.bardist_ = _FakeAcquisitionBarDistribution()
    wrapper.y_mean = torch.tensor([[[10.0]]])
    wrapper.y_std = torch.tensor([[[2.0]]])
    logits = torch.zeros(3, 1, 4)
    torch.testing.assert_close(
        wrapper.predict_ei(logits, torch.tensor(14.0)), torch.full((3, 1), 4.0)
    )
    torch.testing.assert_close(
        wrapper.predict_ucb(logits, torch.tensor(14.0), 0.05),
        torch.full((3, 1), 14.0),
    )


def check_i001() -> None:
    original_regressor = gitbo_module.VanillaDirectTabPFNRegressor
    cuda_functions = {
        name: getattr(torch.cuda, name)
        for name in ("reset_peak_memory_stats", "empty_cache", "max_memory_allocated")
    }
    original_autocast = torch.amp.autocast
    try:
        gitbo_module.VanillaDirectTabPFNRegressor = _QueryOnlyRegressor
        torch.cuda.reset_peak_memory_stats = lambda: None
        torch.cuda.empty_cache = lambda: None
        torch.cuda.max_memory_allocated = lambda: 0
        torch.amp.autocast = lambda **_kwargs: contextlib.nullcontext()
        n_train, n_pending, dimension = 3, 7, 2
        acquisition, constraints, gradient = gitbo_module.compute_acquisition_values(
            "SamplingUCB",
            dimension,
            None,
            n_pending,
            1,
            None,
            torch.rand(n_train, dimension),
            torch.rand(n_train, 1),
            None,
            torch.rand(n_pending, 1, dimension),
            None,
            "cpu",
            GPU_DEVICE="cpu",
            tkwargs={"device": torch.device("cpu"), "dtype": torch.float32},
        )
        assert constraints is None
        assert acquisition.shape == (n_pending, 1)
        assert gradient.shape == (n_pending, 1, dimension)
        assert torch.all(torch.linalg.vector_norm(gradient, dim=-1) > 0)

        for acquisition_name in (
            "EI",
            "Quantile_UCB_95",
            "Quantile_UCB_975",
            "Quantile_UCB_995",
        ):
            acquisition, constraints, gradient = gitbo_module.compute_acquisition_values(
                acquisition_name,
                dimension,
                None,
                n_pending,
                1,
                None,
                torch.rand(n_train, dimension),
                torch.rand(n_train, 1),
                None,
                torch.rand(n_pending, 1, dimension),
                None,
                "cpu",
                GPU_DEVICE="cpu",
                tkwargs={"device": torch.device("cpu"), "dtype": torch.float32},
            )
            assert constraints is None
            assert acquisition.shape == (n_pending, 1)
            assert gradient.shape == (n_pending, 1, dimension)
    finally:
        gitbo_module.VanillaDirectTabPFNRegressor = original_regressor
        for name, function in cuda_functions.items():
            setattr(torch.cuda, name, function)
        torch.amp.autocast = original_autocast


def check_sigma_ball_and_sram_adapter() -> None:
    rng = np.random.default_rng(7)
    mapped = cube_to_sigma_ball(rng.random((2000, 12)), 4.0)
    assert np.all(np.linalg.norm(mapped, axis=1) <= 4.0 + 1e-12)
    problem = SramReadDelayProblem(_MockSramEvaluator(), dimension=4, radius=2.0)
    _, values = problem.evaluate(torch.tensor([[0.6, 0.4, 0.7, 0.3]]))
    assert values.shape == (1, 1) and values.item() >= 290.0


def check_line_evaluator() -> None:
    worker = Path(__file__).parent / "fixtures" / "line_worker.py"
    with tempfile.TemporaryDirectory(prefix="fas_wca_worker_") as directory:
        with LineProcessEvaluator([sys.executable, str(worker)], directory) as evaluator:
            assert evaluator.evaluate(np.array([1.25, 2.75])) == 4.0
            assert evaluator.evaluate(np.array([-1.0, 0.5])) == -0.5


def main() -> int:
    checks = (
        check_i004,
        check_acquisition_units,
        check_i001,
        check_sigma_ball_and_sram_adapter,
        check_line_evaluator,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"ALL PASS ({len(checks)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
