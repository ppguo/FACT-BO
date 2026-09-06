"""Independent numerical and decision-chain regression; no model download."""
import itertools

import numpy as np
import pytest
import torch

from algorithms.tabpfn_wrapper import VanillaDirectTabPFNRegressor
from fas_wca.acquisition import compute_spec_acquisition, dynamic_target
from tabpfn.model.bar_distribution import BarDistribution, FullSupportBarDistribution
from ei_oracle import expected_improvement


TARGETS = [-20., -4., -3., -1.0001, -1., -.9999, -.2, .5,
           1.9999, 2., 2.0001, 5., 12., 30.]
SCALES = [(0., 1.), (10., 2.), (-3., .01), (0., 1e-9), (0., 1e3)]


def wrapper_for(dist, mean=0., std=1.):
    wrapper = VanillaDirectTabPFNRegressor.__new__(VanillaDirectTabPFNRegressor)
    wrapper.bardist_ = dist
    wrapper.y_mean = torch.tensor([[[mean]]])
    wrapper.y_std = torch.tensor([[[std]]])
    return wrapper


@pytest.mark.parametrize('full,z,scale', list(itertools.product((False, True), TARGETS, SCALES)))
def test_density_quadrature(full, z, scale):
    borders = torch.tensor([-3., -1., .5, 2., 5.])
    dist = (FullSupportBarDistribution if full else BarDistribution)(borders)
    rng = np.random.default_rng(20260905)
    logits = torch.tensor(np.concatenate([np.eye(4)*80, rng.normal(0, 2, (24, 4))]), dtype=torch.float32)[:, None]
    mean, std = scale
    w = wrapper_for(dist, mean, std)
    target = mean + std*z
    actual_z = float(((torch.tensor(target)-w.y_mean)/w.y_std).item())
    actual = w.predict_ei(logits, target).numpy()/float(w.y_std.item())
    expected, _ = expected_improvement(logits.numpy(), borders.numpy(), actual_z, full)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    # Require the selected reference score to be optimal within score precision.
    flat = expected.ravel()
    assert flat.max()-flat[actual.argmax()] <= 2e-6+2e-4*abs(flat.max())
    assert (actual >= 0).all()


@pytest.mark.parametrize('seed', range(5))
def test_random_borders_broadcast_and_logits_gradient(seed):
    rng = np.random.default_rng(seed)
    borders = torch.tensor(np.r_[0., np.cumsum(rng.uniform(.1, 3, 3+seed))], dtype=torch.float64)
    dist = FullSupportBarDistribution(borders)
    logits = torch.tensor(rng.normal(size=(3, 2, 3+seed)), dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[-1., 2.], [3., 1.], [12., -2.]], dtype=torch.float64)
    actual = dist.ei(logits, targets)
    for i, j in itertools.product(range(3), range(2)):
        ref, _ = expected_improvement(logits[i,j].detach().numpy(), borders.numpy(), float(targets[i,j]))
        np.testing.assert_allclose(float(actual[i,j]), ref, atol=1e-10, rtol=1e-9)
    assert torch.autograd.gradcheck(lambda x: dist.ei(x, targets), (logits,))


@pytest.mark.parametrize('direction', [-1, 1])
def test_target_before_equal_after(direction):
    theta = direction*2.
    for offset in [-.5, 0., .5]:
        assert dynamic_target(theta, theta+offset) == (max(theta, theta+offset), offset > 0)


@pytest.mark.parametrize('std', [0., -1., float('nan'), float('inf')])
def test_invalid_scale_rejected(std):
    w = wrapper_for(FullSupportBarDistribution(torch.tensor([-3., -1., 2., 5.])), std=std)
    with pytest.raises(ValueError, match='std'):
        w.predict_ei(torch.zeros(2, 1, 3), 1.)


class AnalyticRegressor(VanillaDirectTabPFNRegressor):
    scale = 1.
    def __init__(self, **kwargs):
        self.bardist_ = FullSupportBarDistribution(torch.tensor([-3., -1., .5, 2., 5.]))
        self.y_mean = torch.zeros(1, 1, 1)
        self.y_std = torch.full((1, 1, 1), self.scale)
    def forward(self, x, y, n):
        q = x[n:]
        return {'standard': torch.stack([q[...,0], q[...,1], -q[...,0], q[...,0]*q[...,1]], -1)}


def compute(points, grad=True, theta=.25):
    return compute_spec_acquisition(trained_x=torch.zeros(3, 2), trained_y=torch.zeros(3, 1),
        x_pen=points, theta=theta, device='cpu',
        tkwargs={'device':torch.device('cpu'),'dtype':torch.float32}, need_gradients=grad)


def test_full_chain_tiny_scale_permutation_and_gradient(monkeypatch):
    import algorithms.tabpfn_wrapper as module
    monkeypatch.setattr(module, 'VanillaDirectTabPFNRegressor', AnalyticRegressor)
    x = torch.tensor([[[.1,.2]], [[.4,.7]], [[.8,.3]]])
    original = compute(x)
    without = compute(x, False)
    torch.testing.assert_close(original.values, without.values, rtol=0, atol=0)
    assert without.gradients is None
    perm = torch.tensor([2,0,1])
    torch.testing.assert_close(compute(x[perm]).values, original.values[perm])
    # Positive physical scores below the old 1e-14 fallback cutoff remain valid.
    monkeypatch.setattr(AnalyticRegressor, 'scale', 1e-16)
    tiny = compute(x, theta=.25e-16)
    torch.testing.assert_close(tiny.values/1e-16, original.values, rtol=1e-5, atol=1e-6)
    assert int(tiny.values.argmax()) == int(original.values.argmax())
    assert not tiny.telemetry['fallback']
    monkeypatch.setattr(AnalyticRegressor, 'scale', 1.)
    # Independently difference the analytic surrogate's physical mean.
    model = AnalyticRegressor()
    for d in range(2):
        hi, lo = x.clone(), x.clone()
        hi[...,d] += 1e-3; lo[...,d] -= 1e-3
        mean = lambda q: model.predict_mean(model.forward(q, None, 0)['standard'])
        fd = (mean(hi)-mean(lo))/(2e-3)
        torch.testing.assert_close(original.gradients[...,d], fd, atol=3e-4, rtol=2e-3)


def test_public_acquisition_dispatch_and_invalid_simulator(monkeypatch):
    import algorithms._GITBO as loop
    import algorithms.tabpfn_wrapper as module
    from fas_wca.sram import SramReadDelayProblem
    monkeypatch.setattr(module, 'VanillaDirectTabPFNRegressor', AnalyticRegressor)
    x = torch.tensor([[[.1,.2]], [[.4,.7]], [[.8,.3]]])
    telemetry = []
    values, constraints, grad = loop.compute_acquisition_values(
        'ST-EFMI', 2, None, 3, 1, None, torch.zeros(3,2), torch.zeros(3,1), None,
        x, None, 'cpu', GPU_DEVICE='cpu', tkwargs={'device':torch.device('cpu'),'dtype':torch.float32},
        threshold_y=.25, acquisition_telemetry=telemetry)
    ref = compute(x)
    torch.testing.assert_close(values, ref.values)
    torch.testing.assert_close(grad, ref.gradients)
    assert constraints is None and telemetry[0]['gradient_source']=='posterior_mean'
    class Failed:
        def evaluate(self, vector): return float('nan')
    with pytest.raises(RuntimeError, match='simulator'):
        SramReadDelayProblem(Failed(), dimension=2).evaluate(torch.tensor([[.2,.8]]))


@pytest.mark.parametrize('subspace', [False, True])
def test_public_optimizer_closes_and_saves_observed_targets(monkeypatch, tmp_path, subspace):
    import algorithms._GITBO as loop
    import algorithms.tabpfn_wrapper as module
    monkeypatch.setattr(module, 'VanillaDirectTabPFNRegressor', AnalyticRegressor)
    torch.save(torch.tensor([[.1,.2],[.4,.7],[.8,.3]]),tmp_path/'_trial_0.pt')
    class Objective:
        dim=2
        def evaluate(self, x): return None, x.sum(-1,keepdim=True)
    points, history = loop.GITBO(Objective(),0,N_iterations=3,Acquisition='ST-EFMI',
        threshold_y=1.2,INITIAL_DIR=str(tmp_path),SAVE_DIR=str(tmp_path/'output'),
        N_PENDING=8,DEVICE='cpu',GPU_DEVICE='cpu',GI_SUBSPACE=subspace,rank_r=1)
    assert tuple(points.shape)==(6,2) and tuple(history.shape)==(3,)
    saved=torch.load(next((tmp_path/'output').rglob('*.pt')),weights_only=False)
    torch.testing.assert_close(saved['trained_Y'],points.sum(-1,keepdim=True))
    assert len(saved['acquisition_telemetry'])==3
    for i,row in enumerate(saved['acquisition_telemetry']):
        assert row['target_y']==max(1.2,float(saved['trained_Y'][:3+i].max()))
        assert row['gradient_computed']==subspace and not row['fallback']


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-float('inf')])
def test_nonfinite_logits_and_thresholds_rejected(bad):
    dist=FullSupportBarDistribution(torch.tensor([-3.,-1.,2.,5.]))
    with pytest.raises(ValueError,match='logits'):
        dist.ei(torch.tensor([[0.,bad,1.]]),0.)
    with pytest.raises(ValueError,match='finite'):
        dynamic_target(bad,0.)
