"""OpenYield-backed 4x2 SRAM read-delay objective."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from .sigma_ball import cube_to_sigma_ball


class VectorEvaluator(Protocol):
    """Minimal simulator contract used by :class:`SramReadDelayProblem`."""

    def evaluate(self, vector: np.ndarray) -> float: ...


class SramReadDelayProblem:
    """Maximise 4x2 SRAM read delay over a 144-D process sigma-ball.

    The evaluator returns seconds. The BO-facing objective returns picoseconds
    and follows GIT-BO's maximisation convention.
    """

    metric = "read_delay"
    direction = 1

    def __init__(
        self,
        evaluator: VectorEvaluator,
        *,
        dimension: int = 144,
        radius: float = 16.0,
        fill: str = "solid",
        failure_value_ps: float = 280.0,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.evaluator = evaluator
        self.dim = dimension
        self.radius = radius
        self.fill = fill
        self.failure_value_ps = failure_value_ps

    def evaluate(self, unit_points: torch.Tensor, to_verify: bool = True):
        """Return ``(None, Y)`` for GIT-BO, with ``Y`` in picoseconds."""

        if unit_points.ndim != 2 or unit_points.shape[1] != self.dim:
            raise ValueError(f"expected points with shape (N, {self.dim})")
        process_vectors = cube_to_sigma_ball(
            unit_points.detach().cpu().numpy(), self.radius, fill=self.fill
        )
        values = torch.empty((unit_points.shape[0], 1), dtype=torch.float32)
        for index, vector in enumerate(process_vectors):
            delay_seconds = self.evaluator.evaluate(vector)
            delay_ps = delay_seconds * 1e12
            if not np.isfinite(delay_ps) or delay_ps <= 0.0:
                delay_ps = self.failure_value_ps
            values[index, 0] = float(delay_ps)
        return None, values.to(unit_points.device)
