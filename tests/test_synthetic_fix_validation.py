import numpy as np

from examples.synthetic_fixes.validate_fixes import validate_i001, validate_i004


def test_synthetic_i001_selects_the_prediction_aligned_candidate():
    result = validate_i001(np.linspace(0.0, 1.0, 101), n_train=20)
    assert result.legacy_score_source_index == 80
    assert result.legacy_selected_index == 60
    assert result.fixed_selected_index == 80
    assert result.objective_loss > 99.0


def test_synthetic_i004_restores_exploration_in_physical_units():
    result, _mean, _legacy_scores, _fixed_scores = validate_i004(
        np.linspace(0.0, 1.0, 101),
        y_std=25.0,
        sample_count=512,
        seed=2026,
    )
    assert result.legacy_selected_x == 0.25
    assert result.fixed_selected_x == 0.80
    assert result.fixed_physical_variance_at_global_peak == 156.25
    assert result.objective_gain > 9.9
