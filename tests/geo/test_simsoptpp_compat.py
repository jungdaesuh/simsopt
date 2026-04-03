import numpy as np
import pytest

sopp = pytest.importorskip("simsoptpp")
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier


def _make_clamped_tensor_surface(theta):
    surface = SurfaceXYZTensorFourier(
        nfp=2,
        stellsym=True,
        mpol=3,
        ntor=2,
        clamped_dims=[True, False, True],
        quadpoints_phi=np.array([0.23]),
        quadpoints_theta=np.array([theta]),
    )
    dofs = surface.get_dofs().copy()
    surface.x = dofs + 0.01 * np.linspace(0.1, 1.0, len(dofs))
    return surface


def test_mwpgp_algorithm_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        sopp.MwPGP_algorithm(
            np.zeros((1, 3)),
            np.zeros(1),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 3)),
            np.ones(1),
            1.0,
            max_iter=1,
        )


def test_surface_xyztensorfourier_theta_third_derivative_matches_finite_difference():
    theta0 = 0.37
    eps = 1.0e-6

    surface = _make_clamped_tensor_surface(theta0)
    plus_surface = _make_clamped_tensor_surface(theta0 + eps)
    minus_surface = _make_clamped_tensor_surface(theta0 - eps)

    analytical = np.zeros((1, 3))
    surface.gammadash2dash2dash2_lin(
        analytical,
        np.array([0.23]),
        np.array([theta0]),
    )

    finite_difference = (
        plus_surface.gammadash2dash2()[0, 0, :]
        - minus_surface.gammadash2dash2()[0, 0, :]
    ) / (2 * eps)

    np.testing.assert_allclose(analytical[0], finite_difference, rtol=1.0e-6, atol=1.0e-6)
