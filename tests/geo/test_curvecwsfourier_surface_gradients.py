import numpy as np

from simsopt.geo import CurveCWSFourierCPP, SurfaceRZFourier
from simsopt.geo.curvecwsfourier import surfrz_gamma_lin_jax


def _surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier(nfp=3, stellsym=True, mpol=2, ntor=1)
    dofs = surface.get_dofs().copy()
    dofs += np.linspace(-0.03, 0.04, dofs.size)
    surface.local_full_x = dofs
    return surface


def _curve(surface: SurfaceRZFourier) -> CurveCWSFourierCPP:
    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, 17, endpoint=False),
        order=2,
        surf=surface,
        G=1,
        H=2,
    )
    curve.local_full_x = curve.get_dofs() + np.array(
        [0.13, -0.02, 0.04, 0.01, -0.03, 0.22, 0.05, -0.06, 0.02, -0.01]
    )
    return curve


def test_surfrz_jax_gamma_matches_cpp_arbitrary_points_stellsym() -> None:
    surface = _surface()
    rng = np.random.default_rng(1729)
    phi = rng.uniform(-0.25, 1.25, size=23)
    theta = rng.uniform(-0.25, 1.25, size=23)

    expected = np.zeros((phi.size, 3))
    surface.gamma_lin(expected, phi, theta)
    observed = np.asarray(
        surfrz_gamma_lin_jax(
            phi,
            theta,
            surface.mpol,
            surface.ntor,
            surface.get_dofs(),
            surface.nfp,
            surface.stellsym,
        )
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_curvecwsfouriercpp_surface_parent_enters_gradient_lineage() -> None:
    surface = _surface()
    curve = _curve(surface)

    assert surface in curve.parents
    assert curve.full_dof_size == curve.local_full_dof_size + surface.local_full_dof_size
    assert curve.dof_size == curve.local_dof_size + surface.local_dof_size


def _surface_directional_error(curve, surface, output_fn, vjp_fn) -> float:
    rng = np.random.default_rng(20260614)
    weights = rng.standard_normal(size=output_fn().shape)
    direction = rng.standard_normal(size=surface.get_dofs().shape)
    surface_dofs = surface.get_dofs().copy()
    analytic = np.asarray(vjp_fn(weights)(surface))
    analytic_direction = float(np.dot(analytic, direction))
    eps = 1.0e-6

    surface.local_full_x = surface_dofs + eps * direction
    plus = float(np.sum(output_fn() * weights))
    surface.local_full_x = surface_dofs - eps * direction
    minus = float(np.sum(output_fn() * weights))
    surface.local_full_x = surface_dofs

    finite_difference = (plus - minus) / (2.0 * eps)
    return abs(finite_difference - analytic_direction)


def test_curvecwsfouriercpp_surface_vjps_match_centered_finite_difference() -> None:
    surface = _surface()
    curve = _curve(surface)

    checks = [
        (curve.gamma, curve.dgamma_by_dcoeff_vjp, 1e-8),
        (curve.gammadash, curve.dgammadash_by_dcoeff_vjp, 1e-7),
        (curve.gammadashdash, curve.dgammadashdash_by_dcoeff_vjp, 1e-5),
        (curve.zfactor, curve.dzfactor_by_dcoeff_vjp, 1e-6),
        (curve.rfactor, curve.drfactor_by_dcoeff_vjp, 1e-6),
    ]
    for output_fn, vjp_fn, tolerance in checks:
        assert _surface_directional_error(curve, surface, output_fn, vjp_fn) < tolerance
