import contextlib

import torch

import algorithms._GITBO as gitbo_module
from algorithms.tabpfn_wrapper import VanillaDirectTabPFNRegressor


class _FakeBarDistribution:
    def variance(self, logits):
        return torch.full_like(logits, 4.0)


class _FakeAcquisitionBarDistribution:
    def ei(self, logits, best_f):
        # Makes both the incumbent conversion and output scale observable.
        return torch.ones_like(logits[..., 0]) * best_f

    def ucb(self, logits, best_f, rest_prob):
        del best_f, rest_prob
        return torch.full_like(logits[..., 0], 2.0)


def test_i004_variance_is_returned_in_physical_units():
    wrapper = VanillaDirectTabPFNRegressor.__new__(VanillaDirectTabPFNRegressor)
    wrapper.bardist_ = _FakeBarDistribution()
    wrapper.y_std = torch.tensor([[[3.0]]])
    result = wrapper.predict_variance(torch.zeros(2, 1))
    torch.testing.assert_close(result, torch.full((2, 1), 36.0))


def test_ei_and_quantile_ucb_are_returned_in_physical_units():
    wrapper = VanillaDirectTabPFNRegressor.__new__(VanillaDirectTabPFNRegressor)
    wrapper.bardist_ = _FakeAcquisitionBarDistribution()
    wrapper.y_mean = torch.tensor([[[10.0]]])
    wrapper.y_std = torch.tensor([[[2.0]]])
    logits = torch.zeros(3, 1, 4)
    # standardized best=(14-10)/2=2; standardized EI=2; physical EI=4
    torch.testing.assert_close(
        wrapper.predict_ei(logits, torch.tensor(14.0)), torch.full((3, 1), 4.0)
    )
    # standardized quantile=2; physical quantile=2*2+10=14
    torch.testing.assert_close(
        wrapper.predict_ucb(logits, torch.tensor(14.0), 0.05),
        torch.full((3, 1), 14.0),
    )


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


def test_i001_keeps_every_query_aligned_and_every_gradient_nonzero(monkeypatch):
    monkeypatch.setattr(gitbo_module, "VanillaDirectTabPFNRegressor", _QueryOnlyRegressor)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.amp, "autocast", lambda **_kwargs: contextlib.nullcontext())

    n_train, n_pending, dimension = 3, 7, 2
    trained_X = torch.rand(n_train, dimension)
    trained_Y = torch.rand(n_train, 1)
    pending = torch.rand(n_pending, 1, dimension)
    acquisition, constraints, gradient = gitbo_module.compute_acquisition_values(
        "SamplingUCB",
        dimension,
        None,
        n_pending,
        1,
        None,
        trained_X,
        trained_Y,
        None,
        pending,
        None,
        "cpu",
        GPU_DEVICE="cpu",
        tkwargs={"device": torch.device("cpu"), "dtype": torch.float32},
    )
    assert constraints is None
    assert acquisition.shape == (n_pending, 1)
    assert gradient.shape == (n_pending, 1, dimension)
    assert torch.all(torch.linalg.vector_norm(gradient, dim=-1) > 0)


def test_other_acquisitions_keep_every_query_aligned(monkeypatch):
    monkeypatch.setattr(gitbo_module, "VanillaDirectTabPFNRegressor", _QueryOnlyRegressor)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.amp, "autocast", lambda **_kwargs: contextlib.nullcontext())

    n_train, n_pending, dimension = 3, 7, 2
    for acquisition_name in ("EI", "Quantile_UCB_95", "Quantile_UCB_975", "Quantile_UCB_995"):
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
