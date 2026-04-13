import numpy as np
import pytest

sopp = pytest.importorskip("simsoptpp")
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
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


def _make_zero_tensor_surface(
    phi,
    theta,
    *,
    nfp=2,
    stellsym=False,
    mpol=1,
    ntor=1,
    clamped_dims=(False, False, False),
):
    surface = SurfaceXYZTensorFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=mpol,
        ntor=ntor,
        clamped_dims=list(clamped_dims),
        quadpoints_phi=np.array([phi]),
        quadpoints_theta=np.array([theta]),
    )
    surface.xcs[:] = 0.0
    surface.ycs[:] = 0.0
    surface.zcs[:] = 0.0
    return surface


def _make_theta_clamped_single_mode_surface(theta):
    surface = _make_zero_tensor_surface(
        0.23,
        theta,
        clamped_dims=(True, False, False),
    )
    surface.xcs[0, 0] = 1.0
    return surface


def _make_phi_clamped_single_mode_surface(phi, *, nfp):
    surface = _make_zero_tensor_surface(
        phi,
        0.37,
        nfp=nfp,
        clamped_dims=(False, False, True),
    )
    surface.zcs[0, 0] = 1.0
    return surface


def _make_xyzfourier_surface(theta):
    surface = SurfaceXYZFourier(
        nfp=2,
        stellsym=False,
        mpol=3,
        ntor=2,
        quadpoints_phi=np.array([0.23]),
        quadpoints_theta=np.array([theta]),
    )
    surface.xc[:] = 0.0
    surface.xs[:] = 0.0
    surface.yc[:] = 0.0
    surface.ys[:] = 0.0
    surface.zc[:] = 0.0
    surface.zs[:] = 0.0
    # Exercise the zs sine-coefficient path used by the mixed third derivative.
    surface.zs[2, 1] = 0.37
    surface.zs[1, 3] = -0.11
    return surface


def _theta_finite_difference(make_surface, theta0, derivative_getter, *, eps=1.0e-6):
    plus_surface = make_surface(theta0 + eps)
    minus_surface = make_surface(theta0 - eps)
    return (derivative_getter(plus_surface) - derivative_getter(minus_surface)) / (2 * eps)


def _call_single_point_lin(surface, method_name, theta0):
    analytical = np.zeros((1, 3))
    getattr(surface, method_name)(
        analytical,
        np.array([0.23]),
        np.array([theta0]),
    )
    return analytical[0]


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


def test_surface_xyztensorfourier_theta_second_derivative_matches_finite_difference():
    theta0 = 0.37

    surface = _make_theta_clamped_single_mode_surface(theta0)
    finite_difference = _theta_finite_difference(
        _make_theta_clamped_single_mode_surface,
        theta0,
        lambda s: s.gammadash2()[0, 0, :],
    )

    np.testing.assert_allclose(
        surface.gammadash2dash2()[0, 0, :],
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_surface_xyztensorfourier_theta_third_derivative_matches_finite_difference():
    theta0 = 0.37

    surface = _make_clamped_tensor_surface(theta0)
    finite_difference = _theta_finite_difference(
        _make_clamped_tensor_surface,
        theta0,
        lambda s: s.gammadash2dash2()[0, 0, :],
    )

    np.testing.assert_allclose(
        _call_single_point_lin(surface, "gammadash2dash2dash2_lin", theta0),
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_surface_xyztensorfourier_phi_phi_theta_third_derivative_matches_finite_difference():
    theta0 = 0.37

    surface = _make_clamped_tensor_surface(theta0)
    finite_difference = _theta_finite_difference(
        _make_clamped_tensor_surface,
        theta0,
        lambda s: s.gammadash1dash1()[0, 0, :],
    )

    np.testing.assert_allclose(
        _call_single_point_lin(surface, "gammadash1dash1dash2_lin", theta0),
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_surface_xyztensorfourier_phi_theta_theta_third_derivative_matches_finite_difference():
    theta0 = 0.37

    surface = _make_clamped_tensor_surface(theta0)
    finite_difference = _theta_finite_difference(
        _make_clamped_tensor_surface,
        theta0,
        lambda s: s.gammadash1dash2()[0, 0, :],
    )

    np.testing.assert_allclose(
        _call_single_point_lin(surface, "gammadash1dash2dash2_lin", theta0),
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


@pytest.mark.parametrize("nfp", [1, 3])
def test_surface_xyztensorfourier_phi_second_derivative_matches_finite_difference_for_odd_nfp(nfp):
    phi0 = 0.23
    eps = 1.0e-6

    surface = _make_phi_clamped_single_mode_surface(phi0, nfp=nfp)
    plus_surface = _make_phi_clamped_single_mode_surface(phi0 + eps, nfp=nfp)
    minus_surface = _make_phi_clamped_single_mode_surface(phi0 - eps, nfp=nfp)

    finite_difference = (
        plus_surface.gammadash1()[0, 0, :]
        - minus_surface.gammadash1()[0, 0, :]
    ) / (2 * eps)

    np.testing.assert_allclose(
        surface.gammadash1dash1()[0, 0, :],
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_surface_xyzfourier_mixed_third_derivative_matches_finite_difference():
    theta0 = 0.37

    surface = _make_xyzfourier_surface(theta0)
    finite_difference = _theta_finite_difference(
        _make_xyzfourier_surface,
        theta0,
        lambda s: s.gammadash1dash2()[0, 0, :],
    )

    np.testing.assert_allclose(
        _call_single_point_lin(surface, "gammadash1dash2dash2_lin", theta0),
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
