"""Tests for the Stage-2 non-diagonal Sobolev metric preconditioner."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.single_stage_geometry import (  # noqa: E402
    CurveSobolevBlock,
    Stage2PenaltyPreconditioner,
    build_curve_block_cholesky,
    build_curve_sobolev_metric,
    build_stage2_penalty_preconditioner,
    normalize_objective,
    run_scaled_winding_minimize,
)
from banana_opt.stage2_geometry import WINDING_DOF_CORRIDOR_SCALE_MAP  # noqa: E402
from simsopt.geo import CurveCWSFourierCPP, SurfaceRZFourier  # noqa: E402


def _surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier(nfp=3, stellsym=True, mpol=2, ntor=1)
    dofs = surface.get_dofs().copy()
    dofs += np.linspace(-0.03, 0.04, dofs.size)
    surface.local_full_x = dofs
    return surface


def _curve(order: int = 2, num_quadpoints: int = 17) -> CurveCWSFourierCPP:
    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, num_quadpoints, endpoint=False),
        order=order,
        surf=_surface(),
        G=1,
        H=2,
    )
    curve.local_full_x = curve.get_dofs() + np.linspace(0.01, 0.10, curve.num_dofs())
    return curve


def _quadratic(curvature, target):
    curvature = np.asarray(curvature, dtype=float)
    target = np.asarray(target, dtype=float)

    def fun(x):
        delta = np.asarray(x, dtype=float) - target
        return 0.5 * float(np.sum(curvature * delta * delta)), curvature * delta

    return fun


def _central_fd(fun, u):
    fd = np.zeros_like(u)
    eps = 1.0e-6
    for index in range(u.size):
        plus = u.copy()
        minus = u.copy()
        plus[index] += eps
        minus[index] -= eps
        fd[index] = (fun(plus)[0] - fun(minus)[0]) / (2.0 * eps)
    return fd


def test_curvecwsfouriercpp_accessor_order_matches_metric_block_order():
    curve = _curve(order=2, num_quadpoints=11)

    assert list(curve.local_dof_names) == [
        "phic(0)",
        "phic(1)",
        "phic(2)",
        "phis(1)",
        "phis(2)",
        "thetac(0)",
        "thetac(1)",
        "thetac(2)",
        "thetas(1)",
        "thetas(2)",
    ]
    assert list(curve.local_dof_names) == curve._make_names()
    assert curve.dgammadash_by_dcoeff().shape == (11, 3, curve.num_dofs())
    assert curve.dgammadashdash_by_dcoeff().shape == (11, 3, curve.num_dofs())


def test_curve_sobolev_metric_is_spd_and_cholesky_round_trips():
    curve = _curve(order=2, num_quadpoints=17)
    metric = build_curve_sobolev_metric(curve, alpha=2.0, h2_beta=0.25)
    cholesky_factor = build_curve_block_cholesky(metric)

    np.testing.assert_allclose(metric, metric.T, rtol=1.0e-12, atol=1.0e-12)
    assert float(np.min(np.linalg.eigvalsh(metric))) > 0.0
    np.testing.assert_allclose(
        cholesky_factor @ cholesky_factor.T,
        metric,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(
        build_curve_sobolev_metric(curve, alpha=0.0),
        np.eye(curve.num_dofs()),
    )


def test_operator_transformed_gradient_matches_central_fd():
    cholesky_factor = np.array([[2.0, 0.0], [0.5, 1.5]])
    preconditioner = Stage2PenaltyPreconditioner(
        diagonal_scale=np.array([1.0, 1.0, 1.0, 0.2]),
        curve_blocks=(
            CurveSobolevBlock(
                indices=np.array([1, 2]),
                cholesky_factor=cholesky_factor,
                metric_trace_mean=1.0,
            ),
        ),
        metric_kind="h1",
        alpha=1.0,
    )
    x0 = np.array([0.4, 0.2, -0.1, 0.8])
    fun = _quadratic(
        curvature=np.array([3.0, 5.0, 7.0, 11.0]),
        target=np.array([0.1, -0.3, 0.4, 0.7]),
    )
    captured = {}

    def capturing_minimize(scaled_fun, u0, **kwargs):
        captured["scaled_fun"] = scaled_fun
        return SimpleNamespace(
            x=np.asarray(u0, dtype=float),
            fun=0.0,
            nit=0,
            success=True,
            message="stub",
            status=0,
        )

    run_scaled_winding_minimize(
        capturing_minimize,
        fun,
        x0,
        scale=preconditioner,
        bounds=[(-np.inf, np.inf)] * x0.size,
    )
    scaled_fun = captured["scaled_fun"]
    u0 = preconditioner.to_u(x0)
    analytic = scaled_fun(u0)[1]
    np.testing.assert_allclose(analytic, _central_fd(scaled_fun, u0), rtol=1.0e-4)

    physical_grad = fun(x0)[1]
    expected_curve_grad = solve_triangular(
        cholesky_factor,
        physical_grad[[1, 2]],
        lower=True,
    )
    np.testing.assert_allclose(analytic[[1, 2]], expected_curve_grad, rtol=1.0e-12)


def test_metric_off_is_byte_identical_to_plain_minimize():
    x0 = np.array([0.3, 0.1, 0.903, 0.142])
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.903, 0.993), (0.130, 0.20)]
    fun = _quadratic(
        curvature=np.array([50.0, 20.0, 3.0, 2.0]),
        target=np.array([0.4, 0.2, 0.95, 0.16]),
    )
    preconditioner = Stage2PenaltyPreconditioner.from_scale(np.ones(x0.size))

    plain = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    transformed = run_scaled_winding_minimize(
        minimize,
        fun,
        x0,
        scale=preconditioner,
        bounds=bounds,
    )
    np.testing.assert_array_equal(transformed.x, plain.x)
    assert transformed.fun == plain.fun
    assert transformed.nit == plain.nit


def test_metric_operator_preserves_live_finite_bounds():
    curve = _curve(order=2, num_quadpoints=13)
    dof_names = (
        ["Current1:x0"]
        + [f"{curve.name}:{name}" for name in curve.local_dof_names]
        + ["SurfaceRZFourier1:rc(0,0)"]
    )
    bounds = (
        [(-5.0, 5.0)]
        + [(-np.inf, np.inf)] * curve.num_dofs()
        + [(0.903, 0.993)]
    )

    preconditioner = build_stage2_penalty_preconditioner(
        dof_names,
        (curve,),
        alpha=1.5,
        winding_dof_scale_map=WINDING_DOF_CORRIDOR_SCALE_MAP,
        bounds=bounds,
        metric_kind="h1",
    )
    curve_indices = preconditioner.curve_blocks[0].indices
    assert np.all(np.isneginf([bounds[index][0] for index in curve_indices]))
    assert np.all(np.isposinf([bounds[index][1] for index in curve_indices]))

    transformed_bounds = preconditioner.transform_bounds(bounds)
    assert transformed_bounds[0] == bounds[0]
    expected_rc_lower = bounds[-1][0] / WINDING_DOF_CORRIDOR_SCALE_MAP["rc(0,0)"]
    expected_rc_upper = bounds[-1][1] / WINDING_DOF_CORRIDOR_SCALE_MAP["rc(0,0)"]
    np.testing.assert_allclose(transformed_bounds[-1], (expected_rc_lower, expected_rc_upper))


def test_normalize_objective_scales_value_and_gradient_by_frozen_j_ref():
    curvature = np.array([1.0, 5.0])
    fun = _quadratic(curvature=curvature, target=np.zeros(2))
    j_ref = 10.0
    normalized = normalize_objective(fun, j_ref)
    x = np.array([2.0, -3.0])

    raw_j, raw_grad = fun(x)
    norm_j, norm_grad = normalized(x)
    assert norm_j == raw_j / j_ref
    np.testing.assert_allclose(norm_grad, raw_grad / j_ref)
    assert np.linalg.cond(np.diag(curvature / j_ref)) == np.linalg.cond(np.diag(curvature))
