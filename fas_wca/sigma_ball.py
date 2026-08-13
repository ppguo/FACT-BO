"""Deterministic maps from a unit cube to a process sigma-ball."""

from __future__ import annotations

import numpy as np
from scipy.special import gammainc, ndtri


def cube_to_sigma_ball(
    unit_points: np.ndarray,
    radius: float,
    *,
    fill: str = "solid",
    clip: float = 1e-6,
) -> np.ndarray:
    """Map points in ``[0, 1]^D`` to a radius-``K`` Gaussian sigma-ball.

    The inverse-normal transform supplies a uniform direction. For ``solid``
    sampling, the chi-square CDF is converted to the volume-uniform radial law
    ``r = K * U**(1/D)``. ``surface`` places every non-degenerate point at
    radius ``K``.

    Parameters
    ----------
    unit_points:
        One point of shape ``(D,)`` or a batch of shape ``(N, D)``.
    radius:
        Positive sigma-ball radius ``K``.
    fill:
        ``"solid"`` or ``"surface"``.
    clip:
        Numerical clipping applied before the inverse-normal transform.
    """

    points = np.asarray(unit_points, dtype=float)
    squeeze = points.ndim == 1
    if squeeze:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] == 0:
        raise ValueError("unit_points must have shape (D,) or (N, D), with D > 0")
    if not np.all(np.isfinite(points)) or np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("unit_points must be finite and lie in [0, 1]")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if fill not in {"solid", "surface"}:
        raise ValueError("fill must be 'solid' or 'surface'")
    if not 0.0 < clip < 0.5:
        raise ValueError("clip must lie in (0, 0.5)")

    dimension = points.shape[1]
    gaussian = ndtri(np.clip(points, clip, 1.0 - clip))
    norms = np.linalg.norm(gaussian, axis=1)
    result = np.zeros_like(gaussian)
    nonzero = norms > 0.0
    directions = np.zeros_like(gaussian)
    directions[nonzero] = gaussian[nonzero] / norms[nonzero, None]

    if fill == "solid":
        radial_uniform = gammainc(dimension / 2.0, norms * norms / 2.0)
        radii = radius * np.power(radial_uniform, 1.0 / dimension)
    else:
        radii = np.full(points.shape[0], radius, dtype=float)
    result[nonzero] = directions[nonzero] * radii[nonzero, None]
    return result[0] if squeeze else result
