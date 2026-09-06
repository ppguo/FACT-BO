"""Independent CPU quadrature of the density, never calls TabPFN EI/PI/CDF."""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri, softmax


def component_integrals(borders, target, full_support=True):
    b = np.asarray(borders, dtype=np.float64)
    if not np.all(np.diff(b) > 0):
        raise ValueError("Strictly increasing borders required by this audit")
    result, errors = [], []
    for i, (left, right) in enumerate(zip(b[:-1], b[1:])):
        if full_support and i in (0, len(b) - 2):
            scale = (right - left) / ndtri(0.75)
            # Integrate in units of the half-normal scale to avoid narrow-tail loss.
            if i == 0:
                limit = max((right - target) / scale, 0.0)
                value, error = quad(
                    lambda z: (right - target - scale*z)*math.sqrt(2/math.pi)*math.exp(-z*z/2),
                    0, min(limit, 40), epsabs=1e-12, epsrel=1e-11)
            else:
                limit = max((target - left) / scale, 0.0)
                value, error = quad(
                    lambda z: (left - target + scale*z)*math.sqrt(2/math.pi)*math.exp(-z*z/2),
                    limit, np.inf, epsabs=1e-12, epsrel=1e-11)
        else:
            lo = max(left, target)
            value, error = (0.0, 0.0) if lo >= right else quad(
                lambda y: (y-target)/(right-left), lo, right,
                epsabs=1e-12, epsrel=1e-11)
        result.append(value)
        errors.append(error)
    return np.asarray(result), np.asarray(errors)


def expected_improvement(logits, borders, target, full_support=True):
    components, errors = component_integrals(borders, target, full_support)
    p = softmax(np.asarray(logits, dtype=np.float64), axis=-1)
    return p @ components, p @ errors


def compare(actual, expected):
    a, e = np.asarray(actual), np.asarray(expected)
    error = np.abs(a-e)
    tol = 1e-6 + 1e-4*np.abs(e)
    finite = np.isfinite(a).all() and np.isfinite(e).all()
    near = np.abs(e) < 1e-6
    return {
        "pass": bool(finite and np.all(error <= tol)),
        "max_abs_error": float(error.max()),
        "max_relative_error_nonzero": float((error[~near]/np.abs(e[~near])).max()) if (~near).any() else None,
        "max_abs_error_near_zero": float(error[near].max()) if near.any() else None,
        "failed_values": int((~np.isfinite(a) | (error > tol)).sum()),
    }


def ranking_check(actual, expected):
    a, e = np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    selected, reference = int(a.argmax()), int(e.argmax())
    gap = float(e[reference]-e[selected])
    # A different index is harmless only if the score gap fits both error budgets.
    tol = 2e-6 + 1e-4*(abs(float(e[reference]))+abs(float(e[selected])))
    return {"selected_index": selected, "reference_index": reference,
            "exact_argmax_match": selected == reference, "reference_regret": gap,
            "argmax_pass": bool(gap <= tol)}
