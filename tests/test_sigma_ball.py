import numpy as np
import pytest

from fas_wca.sigma_ball import cube_to_sigma_ball


def test_solid_points_stay_inside_ball():
    rng = np.random.default_rng(7)
    points = rng.random((2000, 12))
    mapped = cube_to_sigma_ball(points, 4.0, fill="solid")
    assert mapped.shape == points.shape
    assert np.all(np.linalg.norm(mapped, axis=1) <= 4.0 + 1e-12)


def test_surface_points_have_requested_radius():
    points = np.full((3, 5), 0.7)
    mapped = cube_to_sigma_ball(points, 3.0, fill="surface")
    np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 3.0)


def test_invalid_cube_point_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cube_to_sigma_ball(np.array([0.5, 1.1]), 2.0)
