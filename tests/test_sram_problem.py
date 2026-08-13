import numpy as np
import torch

from fas_wca.sram import SramReadDelayProblem


class _MockEvaluator:
    def __init__(self):
        self.vectors = []

    def evaluate(self, vector):
        self.vectors.append(np.asarray(vector))
        return 2.9e-10 + np.linalg.norm(vector) * 1e-13


def test_sram_adapter_maps_units_and_shape():
    evaluator = _MockEvaluator()
    problem = SramReadDelayProblem(evaluator, dimension=4, radius=2.0)
    _, values = problem.evaluate(torch.tensor([[0.6, 0.4, 0.7, 0.3]], dtype=torch.float32))
    assert values.shape == (1, 1)
    assert values.item() >= 290.0
    assert np.linalg.norm(evaluator.vectors[0]) <= 2.0 + 1e-12


def test_sram_adapter_penalizes_nonconvergence():
    class FailedEvaluator:
        def evaluate(self, _vector):
            return float("nan")

    problem = SramReadDelayProblem(FailedEvaluator(), dimension=2, failure_value_ps=280.0)
    _, values = problem.evaluate(torch.tensor([[0.25, 0.75]], dtype=torch.float32))
    assert values.item() == 280.0
