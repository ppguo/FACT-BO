import sys
from pathlib import Path

import numpy as np

from fas_wca.evaluators import LineProcessEvaluator


def test_persistent_line_evaluator_round_trip(tmp_path):
    worker = Path(__file__).parent / "fixtures" / "line_worker.py"
    with LineProcessEvaluator([sys.executable, str(worker)], tmp_path) as evaluator:
        assert evaluator.evaluate(np.array([1.25, 2.75])) == 4.0
        assert evaluator.evaluate(np.array([-1.0, 0.5])) == -0.5
