import numpy as np

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
import simsopt.geo.accessibility as accessibility_module
from simsopt.geo.accessibility import (
    CurveInPortPenalty,
    DirectedFacingPort,
    PortSize,
    ProjectedCurveConvexity,
    ProjectedCurveCurveDistance,
    ProjectedEnclosedArea,
)
from simsopt.geo.curve import CurveCWSFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

_FD_GRADIENT_TOLS = parity_ladder_tolerances("fd-gradient")


def _make_surface():
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=1,
        nphi=16,
        ntheta=16,
        mpol=1,
        ntor=1,
    )
    surface.set("rc(0,0)", 1.0)
    surface.set("rc(1,0)", 0.2)
    surface.set("zs(1,0)", 0.2)
    return surface


def _make_cws_curve(phase=0.0):
    curve = CurveCWSFourier(16, 2, _make_surface())
    dofs = np.asarray(curve.x).copy()
    dofs[:4] += np.array([phase, 0.05, -0.02, 0.03])
    curve.x = dofs
    return curve


def _make_xyz_curve(dy=0.0, dz=0.0):
    quadpoints = np.linspace(0, 1, 24, endpoint=False)
    curve = CurveXYZFourier(quadpoints, order=2)
    dofs = np.zeros(curve.dof_size)
    dofs[1] = 1.0
    dofs[7] = 1.0
    dofs[11] = 0.2
    dofs[2] = dy
    dofs[4] = dz
    curve.x = dofs
    return curve


def _make_port_size_objective():
    port = _make_cws_curve(0.0)
    curves = [
        _make_xyz_curve(dy=0.4),
        _make_xyz_curve(dy=-0.3, dz=0.1),
        _make_xyz_curve(dy=0.25, dz=-0.1),
    ]
    objective = PortSize(
        port,
        curves,
        curve_port_distance_threshold=0.1,
        direction="vertical",
        solver="explicit",
    )
    return objective, curves


def _legacy_accessibility_jit_attrs(objective):
    return [
        name
        for name in vars(objective)
        if name == "J_jax"
        or name == "hessian"
        or name.startswith("thisgrad")
        or name.startswith("ddJd")
    ]


def _clear_caches(*functions):
    for fn in functions:
        fn.clear_cache()


def _assert_no_legacy_jit_attrs(objective):
    assert _legacy_accessibility_jit_attrs(objective) == []


def _assert_finite_value_and_gradient(objective, *, label):
    value = float(objective.J())
    gradient = np.asarray(objective.dJ(), dtype=float)
    assert np.isfinite(value), f"{label} value is not finite"
    assert gradient.shape == np.asarray(objective.x, dtype=float).shape
    assert np.all(np.isfinite(gradient)), f"{label} gradient is not finite"


def _assert_finite_symmetric_hessian(objective, *, label):
    _assert_finite_value_and_gradient(objective, label=label)
    hessian = np.asarray(objective.ddJ_ddport(), dtype=float)
    dof_size = np.asarray(objective.x, dtype=float).size
    assert hessian.shape == (dof_size, dof_size)
    assert np.all(np.isfinite(hessian)), f"{label} Hessian is not finite"
    np.testing.assert_allclose(
        hessian,
        hessian.T,
        rtol=1e-10,
        atol=1e-12,
        err_msg=label,
    )


def _assert_directional_fd_matches_dJ(objective, *, label):
    x0 = np.asarray(objective.x, dtype=float).copy()
    direction = np.linspace(-1.0, 1.0, x0.size, dtype=float)
    direction /= np.linalg.norm(direction)

    grad = np.asarray(objective.dJ(), dtype=float)
    analytic_directional = float(np.dot(grad, direction))
    eps = 1e-6
    objective.x = x0 + eps * direction
    value_plus = float(objective.J())
    objective.x = x0 - eps * direction
    value_minus = float(objective.J())
    objective.x = x0

    fd_directional = (value_plus - value_minus) / (2.0 * eps)
    abs_error = abs(analytic_directional - fd_directional)
    floor = float(_FD_GRADIENT_TOLS["directional_derivative_floor"])
    if abs(fd_directional) <= floor:
        np.testing.assert_allclose(
            analytic_directional,
            fd_directional,
            rtol=0.0,
            atol=float(_FD_GRADIENT_TOLS["directional_fd_atol"]),
            err_msg=label,
        )
        return

    assert abs_error / abs(fd_directional) < float(
        _FD_GRADIENT_TOLS["directional_fd_rtol"]
    ), label


def test_projected_enclosed_area_reuses_shared_jit_kernels():
    _clear_caches(
        accessibility_module._projected_enclosed_area_zphi_grad,
        accessibility_module._projected_enclosed_area_zphi_hessian,
    )
    curve1 = _make_cws_curve(0.0)
    objective1 = ProjectedEnclosedArea(curve1, projection="zphi")
    _assert_finite_symmetric_hessian(
        objective1,
        label="ProjectedEnclosedArea must expose finite symmetric ddJ",
    )

    assert accessibility_module._projected_enclosed_area_zphi_grad._cache_size() == 1
    assert accessibility_module._projected_enclosed_area_zphi_hessian._cache_size() == 1
    _assert_no_legacy_jit_attrs(objective1)

    curve2 = _make_cws_curve(0.1)
    objective2 = ProjectedEnclosedArea(curve2, projection="zphi")
    float(objective2.J())
    np.asarray(objective2.dJ(), dtype=float)
    objective2.ddJ_ddport()

    assert accessibility_module._projected_enclosed_area_zphi_grad._cache_size() == 1
    assert accessibility_module._projected_enclosed_area_zphi_hessian._cache_size() == 1
    _assert_no_legacy_jit_attrs(objective2)


def test_directed_facing_port_reuses_shared_jit_kernels():
    _clear_caches(
        accessibility_module._upward_facing_grad,
        accessibility_module._upward_facing_hessian,
    )
    curve1 = _make_cws_curve(0.0)
    objective1 = DirectedFacingPort(curve1, projection="xy")
    _assert_finite_symmetric_hessian(
        objective1,
        label="DirectedFacingPort must expose finite symmetric ddJ",
    )

    assert accessibility_module._upward_facing_grad._cache_size() == 1
    assert accessibility_module._upward_facing_hessian._cache_size() == 1
    _assert_no_legacy_jit_attrs(objective1)

    curve2 = _make_cws_curve(0.1)
    objective2 = DirectedFacingPort(curve2, projection="xy")
    float(objective2.J())
    np.asarray(objective2.dJ(), dtype=float)
    objective2.ddJ_ddport()

    assert accessibility_module._upward_facing_grad._cache_size() == 1
    assert accessibility_module._upward_facing_hessian._cache_size() == 1
    _assert_no_legacy_jit_attrs(objective2)


def test_curve_in_port_penalty_reuses_shared_jit_kernels():
    _clear_caches(
        accessibility_module._curve_in_port_penalty_xy_values,
        accessibility_module._curve_in_port_penalty_xy_grads,
    )
    port1 = _make_xyz_curve(dy=0.2, dz=0.1)
    curves1 = [_make_xyz_curve(dy=-0.5, dz=-0.2), _make_xyz_curve(dy=-0.7, dz=0.3)]
    objective1 = CurveInPortPenalty(port1, curves1, threshold=0.5, projection="xy")
    float(objective1.J())
    np.asarray(objective1.dJ(), dtype=float)
    _assert_directional_fd_matches_dJ(
        objective1,
        label="CurveInPortPenalty dJ must match directional FD",
    )

    assert accessibility_module._curve_in_port_penalty_xy_values._cache_size() == 1
    assert accessibility_module._curve_in_port_penalty_xy_grads._cache_size() == 1
    _assert_no_legacy_jit_attrs(objective1)

    port2 = _make_xyz_curve(dy=0.5, dz=0.1)
    curves2 = [_make_xyz_curve(dy=1.0, dz=-0.2), _make_xyz_curve(dy=0.8, dz=0.3)]
    objective2 = CurveInPortPenalty(port2, curves2, threshold=0.5, projection="xy")
    float(objective2.J())
    np.asarray(objective2.dJ(), dtype=float)

    assert accessibility_module._curve_in_port_penalty_xy_values._cache_size() == 1
    assert accessibility_module._curve_in_port_penalty_xy_grads._cache_size() == 1
    _assert_no_legacy_jit_attrs(objective2)


def test_projected_curve_curve_distance_reuses_shared_jit_kernels():
    _clear_caches(
        accessibility_module._projected_cc_distance_xy_values,
        accessibility_module._projected_cc_distance_xy_grads,
        accessibility_module._projected_cc_distance_xy_hessian,
        accessibility_module._projected_cc_distance_xy_hessians,
    )
    port1 = _make_cws_curve(0.0)
    base1 = _make_xyz_curve(dy=0.4)
    base1b = _make_xyz_curve(dy=-0.3, dz=0.1)
    objective1 = ProjectedCurveCurveDistance(
        [base1, base1b],
        port1,
        minimum_distance=0.1,
        projection="xy",
    )
    float(objective1.J())
    np.asarray(objective1.dJ(), dtype=float)
    objective1.ddJ_ddport()
    objective1.ddJ_dportdcoil(base1)
    _assert_directional_fd_matches_dJ(
        objective1,
        label="ProjectedCurveCurveDistance dJ must match directional FD",
    )
    hessian_cache_size = (
        accessibility_module._projected_cc_distance_xy_hessian._cache_size()
    )
    batch_hessian_cache_size = (
        accessibility_module._projected_cc_distance_xy_hessians._cache_size()
    )

    assert accessibility_module._projected_cc_distance_xy_values._cache_size() == 1
    assert accessibility_module._projected_cc_distance_xy_grads._cache_size() == 1
    assert hessian_cache_size > 0
    assert batch_hessian_cache_size > 0
    _assert_no_legacy_jit_attrs(objective1)

    port2 = _make_cws_curve(0.1)
    base2 = _make_xyz_curve(dy=0.3)
    base2b = _make_xyz_curve(dy=-0.2, dz=0.15)
    objective2 = ProjectedCurveCurveDistance(
        [base2, base2b],
        port2,
        minimum_distance=0.2,
        projection="xy",
    )
    float(objective2.J())
    np.asarray(objective2.dJ(), dtype=float)
    objective2.ddJ_ddport()
    objective2.ddJ_dportdcoil(base2)

    assert accessibility_module._projected_cc_distance_xy_values._cache_size() == 1
    assert accessibility_module._projected_cc_distance_xy_grads._cache_size() == 1
    assert (
        accessibility_module._projected_cc_distance_xy_hessian._cache_size()
        == hessian_cache_size
    )
    assert (
        accessibility_module._projected_cc_distance_xy_hessians._cache_size()
        == batch_hessian_cache_size
    )
    _assert_no_legacy_jit_attrs(objective2)


def test_projected_curve_convexity_reuses_shared_jit_kernels():
    _clear_caches(
        accessibility_module._projected_curve_convexity_zphi_value,
        accessibility_module._projected_curve_convexity_zphi_grad,
    )
    curve1 = _make_xyz_curve()
    objective1 = ProjectedCurveConvexity(curve1, projection="zphi")
    _assert_finite_value_and_gradient(
        objective1,
        label="ProjectedCurveConvexity must expose finite J/dJ",
    )

    assert accessibility_module._projected_curve_convexity_zphi_value._cache_size() == 1
    assert accessibility_module._projected_curve_convexity_zphi_grad._cache_size() == 1
    _assert_no_legacy_jit_attrs(objective1)

    curve2 = _make_xyz_curve(dy=0.1)
    objective2 = ProjectedCurveConvexity(curve2, projection="zphi")
    float(objective2.J())
    np.asarray(objective2.dJ(), dtype=float)

    assert accessibility_module._projected_curve_convexity_zphi_value._cache_size() == 1
    assert accessibility_module._projected_curve_convexity_zphi_grad._cache_size() == 1
    _assert_no_legacy_jit_attrs(objective2)


def test_port_size_refreshes_cached_port_solve_on_parent_curve_mutation():
    objective, curves = _make_port_size_objective()

    objective.dJ()
    initial_port_solve = objective._port_area_solve.copy()

    assert objective.need_to_run_code is False
    assert objective._port_area_solve is not None

    curve_dofs = np.asarray(curves[0].x).copy()
    curve_dofs[2] += 0.02
    curves[0].x = curve_dofs

    assert objective.need_to_run_code is True
    assert objective._port_area_solve is None

    objective.dJ()

    assert objective.need_to_run_code is False
    assert objective._port_area_solve is not None
    assert not np.allclose(objective._port_area_solve, initial_port_solve)
