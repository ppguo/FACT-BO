"""Specification-triggered expected failure-margin improvement (ST-EFMI)."""

from dataclasses import dataclass
import math

import torch


def dynamic_target(theta: float, incumbent_y: float) -> tuple[float, bool]:
    """Responses are oriented so larger is worse; equality is not a failure."""
    if not math.isfinite(theta) or not math.isfinite(incumbent_y):
        raise ValueError("Threshold and incumbent must be finite")
    return max(float(theta), float(incumbent_y)), incumbent_y > theta


@dataclass
class AcquisitionResult:
    values: torch.Tensor
    gradients: torch.Tensor | None
    telemetry: dict


def compute_spec_acquisition(*, trained_x, trained_y, x_pen, theta, device,
                             tkwargs, need_gradients=True):
    """Compute E[(Y-max(theta,Y_best))+] and posterior-mean gradients.

    This is the corrected experimental forward/gradient path. Invalid scores
    or degenerate requested gradients stop the run instead of switching the
    acquisition policy. Small positive EI values are retained in physical units.
    """
    from algorithms.tabpfn_wrapper import VanillaDirectTabPFNRegressor

    target, found = dynamic_target(theta, float(trained_y.max().detach().cpu()))
    n_pending, n_candidates, dimension = x_pen.shape
    regressor = VanillaDirectTabPFNRegressor(device=device)
    n_train = trained_x.shape[0]
    x_train = trained_x.unsqueeze(1).expand(-1, n_candidates, -1).detach()
    x_cand = x_pen.clone().requires_grad_(need_gradients)
    x_concat = torch.cat([x_train, x_cand], dim=0).to(device)
    y_pad = torch.zeros(n_pending, 1, **tkwargs)
    y_full = torch.cat([trained_y, y_pad], dim=0).unsqueeze(1)
    y_full = y_full.expand(-1, n_candidates, -1)
    with torch.set_grad_enabled(need_gradients):
        logits = regressor.forward(x_concat, y_full, n_train)["standard"]
        mu = regressor.predict_mean(logits)
        values = regressor.predict_ei(logits, target)
        expected = (n_pending, n_candidates)
        if tuple(mu.shape) != expected or tuple(values.shape) != expected:
            raise AssertionError("Posterior/acquisition rows do not match candidates")
        if not torch.isfinite(values).all() or (values < 0).any():
            raise FloatingPointError("ST-EFMI values must be finite and nonnegative")
        if not (values > 0).any():
            raise FloatingPointError("All ST-EFMI values are zero")
        gradients = None
        if need_gradients:
            gradients, = torch.autograd.grad(mu.sum(), x_cand)
            gradients = gradients.reshape(n_pending, n_candidates, dimension)
            if not torch.isfinite(gradients).all() or not (gradients != 0).any():
                raise FloatingPointError("Posterior-mean gradient is nonfinite or zero")
    telemetry = {
        "target_y": target,
        "incumbent_y_before": float(trained_y.max().detach().cpu()),
        "failure_found_before": found,
        "acquisition": "ST-EFMI",
        "gradient_source": "posterior_mean" if need_gradients else "not_requested_no_subspace",
        "gradient_requested": bool(need_gradients),
        "gradient_computed": gradients is not None,
        "fallback": False,
        "fallback_reasons": [],
        "acquisition_min": float(values.detach().min().cpu()),
        "acquisition_max": float(values.detach().max().cpu()),
    }
    return AcquisitionResult(values.detach().to(**tkwargs), gradients, telemetry)
