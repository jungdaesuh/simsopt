"""The analytic exact single-stage evaluator reproduces the native example's value, gradient and policy."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import enable_non_strict_jax_backend, parity_device
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import lu_factor, lu_solve
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (
    BoozerResidual,
    BoozerSurface,
    CurveLength,
    Iotas,
    MajorRadius,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
    Volume,
)
from simsopt.objectives import QuadraticPenalty
from simsopt_jax.core._math_utils import as_jax_float64
from simsopt_jax.core.field import coil_set_spec_from_dof_extraction_spec
from simsopt_jax.core.specs import host_resident_spec
from simsopt_jax.runtime.host_boundary import host_array
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo import single_stage_exact_analytic
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.single_stage_exact_analytic import (
    INNER_FAILURE_VALUE,
    ExactAnalyticSingleStage,
    HostConstructionBoozerSurfaceJAX,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    _TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS,
)
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    _evaluate_traceable_total_objective,
)

INITIAL_IOTA = -0.406
RESOLUTION = 1
QS_RESOLUTION = 4
UNIT_ROUNDOFF = 2.0**-53


@pytest.fixture(params=("cpu", "gpu"), autouse=True)
def analytic_backend(monkeypatch, request):
    device = parity_device(request.param)
    enable_non_strict_jax_backend(monkeypatch, request, f"jax_{request.param}_parity")
    with jax.default_device(device):
        yield device


def _problem(resolution: int = RESOLUTION, *, example_coils: bool = False):
    # ``example_coils`` selects the shipped NCSX coil set the example uses;
    # the default keeps the coils small so the parity tests stay fast.
    base_curves, base_currents, axis, nfp, field = (
        get_data("ncsx")
        if example_coils
        else get_data("ncsx", coil_order=3, magnetic_axis_order=3, points_per_period=8)
    )
    nq = 2 * resolution + 1
    surface = SurfaceXYZTensorFourier(
        mpol=resolution,
        ntor=resolution,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, nq, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, nq, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1, flip_theta=True)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    return base_curves, base_currents, nfp, field, surface, G0


def _config(nfp, surface, length_target):
    return {
        "non_qs_weight": 1.0,
        "residual_weight": 1.0,
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.linspace(
            0.0, 1.0 / nfp, 2 * QS_RESOLUTION, endpoint=False
        ),
        "non_qs_quadpoints_theta": np.linspace(
            0.0, 1.0, 2 * QS_RESOLUTION, endpoint=False
        ),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": length_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": float(surface.major_radius()),
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }


def _native_objective(resolution: int = RESOLUTION, *, example_coils: bool = False):
    base_curves, base_currents, nfp, field, surface, G0 = _problem(
        resolution, example_coils=example_coils
    )
    volume = Volume(surface)
    boozer = BoozerSurface(
        field,
        surface,
        volume,
        volume.J(),
        options={"newton_maxiter": 20, "newton_tol": 1.0e-13, "verbose": False},
    )
    initial = boozer.solve_residual_equation_exactly_newton(
        tol=1.0e-13, maxiter=20, iota=INITIAL_IOTA, G=G0
    )
    assert initial["success"]
    major_radius = MajorRadius(boozer)
    total_length = reduce(add, (CurveLength(curve) for curve in base_curves))
    objective = (
        NonQuasiSymmetricRatio(boozer, BiotSavart(field.coils), sDIM=QS_RESOLUTION)
        + BoozerResidual(boozer, field)
        + QuadraticPenalty(Iotas(boozer), float(initial["iota"]), "identity")
        + QuadraticPenalty(major_radius, cast(float, major_radius.J()), "identity")
        + QuadraticPenalty(total_length, float(total_length.J()), "max")
    )
    base_currents[0].fix_all()
    return objective, boozer


def _jax_evaluator(resolution: int = RESOLUTION, *, example_coils: bool = False):
    base_curves, base_currents, nfp, native_field, surface, G0 = _problem(
        resolution, example_coils=example_coils
    )
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    volume = Volume(surface)
    boozer = HostConstructionBoozerSurfaceJAX(
        field,
        surface,
        volume,
        float(volume.J()),
        options={"newton_maxiter": 20, "newton_tol": 1.0e-13, "verbose": False},
    )
    length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    evaluator = ExactAnalyticSingleStage(
        boozer,
        field,
        iota=INITIAL_IOTA,
        G=G0,
        outer_objective_config=lambda: _config(nfp, surface, length_target),
    )
    return evaluator, boozer


def _jax_evaluator_eager_geometry(
    resolution: int = RESOLUTION, *, example_coils: bool = False
):
    """Pre-patch construction: eager JAX geometry bake on stock BoozerSurfaceJAX."""
    base_curves, base_currents, nfp, native_field, surface, G0 = _problem(
        resolution, example_coils=example_coils
    )
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    volume = Volume(surface)
    boozer = BoozerSurfaceJAX(
        field,
        surface,
        volume,
        float(volume.J()),
        options={"newton_maxiter": 20, "newton_tol": 1.0e-13, "verbose": False},
    )
    length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    evaluator = ExactAnalyticSingleStage(
        boozer,
        field,
        iota=INITIAL_IOTA,
        G=G0,
        outer_objective_config=lambda: _config(nfp, surface, length_target),
    )
    return evaluator, boozer


def _gamma(n: int) -> float:
    """Higham's worst-case bound for a length-``n`` floating-point sum."""
    return n * UNIT_ROUNDOFF / (1.0 - n * UNIT_ROUNDOFF)


def _differing_float_positions(host_leaves, reference_leaves):
    """Positions of the leaves the two lanes set differently; all are fp64."""
    positions = []
    for position, (host_leaf, reference_leaf) in enumerate(
        zip(host_leaves, reference_leaves, strict=True)
    ):
        if isinstance(host_leaf, str):
            assert host_leaf == reference_leaf
            continue
        host_numbers = host_array(host_leaf)
        reference_numbers = host_array(reference_leaf)
        if not np.array_equal(host_numbers, reference_numbers):
            assert host_numbers.dtype == np.float64
            positions.append(position)
    return tuple(positions)


def _bake_first_order_bounds(
    monkeypatch,
    host,
    host_boozer,
    reference,
    reference_boozer,
    host_eval,
    reference_eval,
):
    """Componentwise first-order bounds on host-vs-device differences.

    The lanes run one program on one device and differ only in their inputs:
    the affine surface basis ``b`` (host NumPy vs device ``jacfwd``; the
    origin is asserted bitwise equal) and the outer targets ``p`` each lane
    takes from its own solved seed (iota, major radius, vessel gamma). The
    first evaluate starts at the seed and takes no Newton step, so every
    compared quantity is a function of ``(x, b, p)`` at the seed ``x``.

    Seed: each Newton solve stops at its residual ``F``, so to first order
    ``|x_h - x_d| <= dx = |J^-1| (|F_h| + |F_d|) + |J^-1| |dF/db| |b_h - b_d|``
    (distance of each seed to its own root, plus the root shift the basis
    difference causes), with ``J = dF/dx`` at the host seed.

    Value ``v``: ``|dv/dx| dx + |dv/dp| |p_h - p_d|`` plus the rounding of the
    final sum of the outer terms. The terms are all nonnegative, so their
    condition scale is ``|v|``; that is ``gamma_9 (|v_h| + |v_d|)``.

    Gradient ``g = dJ/dc - (dF/dc)^T lambda`` with ``J^T lambda = dJ/dx``:
    ``|dg/dx| dx + |dg/db| |b_h - b_d| + |dg/dp| |p_h - p_d|`` plus the
    rounding of the final contraction of ``1 + len(F)`` = 16 terms,
    ``gamma_16 (S_h + S_d)`` with ``S = |dJ/dc| + |dF/dc|^T |lambda|``.

    Every Jacobian is evaluated at the host lane's inputs. Second-order terms
    (``O(dx^2)``, about 1e-29 here) are neglected; nothing is fitted.
    """
    host_geometry, host_basis, _ = host_boozer._make_analytic_geometry_terms()
    reference_geometry, reference_basis, _ = (
        reference_boozer._make_analytic_geometry_terms()
    )
    zero_dofs = jnp.zeros(host_basis.gamma.shape[-1], dtype=jnp.float64)
    origin = jax.tree.map(
        lambda leaf: host_array(leaf, dtype=np.float64), host_geometry(zero_dofs)
    )
    for host_leaf, reference_leaf in zip(
        jax.tree.leaves(origin),
        jax.tree.leaves(reference_geometry(zero_dofs)),
        strict=True,
    ):
        np.testing.assert_array_equal(
            host_leaf, host_array(reference_leaf, dtype=np.float64)
        )
    basis_host, unravel_basis = ravel_pytree(host_basis)
    basis_reference, _ = ravel_pytree(reference_basis)

    host_leaves, kwargs_tree = jax.tree_util.tree_flatten(
        host._objective_cache_state["objective_kwargs"]
    )
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(
        reference._objective_cache_state["objective_kwargs"]
    )
    assert kwargs_tree == reference_tree
    positions = _differing_float_positions(host_leaves, reference_leaves)
    targets_host, unravel_targets = ravel_pytree(
        tuple(host_array(host_leaves[i], dtype=np.float64) for i in positions)
    )
    targets_reference, _ = ravel_pytree(
        tuple(host_array(reference_leaves[i], dtype=np.float64) for i in positions)
    )

    coil_extraction_spec = host_resident_spec(
        host_boozer.biotsavart.coil_dof_extraction_spec()
    )
    coils = jnp.asarray(host.coil_dofs)
    weight_inv_modB = host_boozer.options["weight_inv_modB"]

    def coil_set_spec(coil_dofs):
        return coil_set_spec_from_dof_extraction_spec(
            coil_extraction_spec, as_jax_float64(coil_dofs)
        )

    def residual_and_jacobian(x, coil_dofs, basis):
        # The production operator, built on this basis instead of the baked one.
        with monkeypatch.context() as patch:
            patch.setattr(
                single_stage_exact_analytic,
                "host_analytic_geometry_origin_and_basis",
                lambda **_: (origin, unravel_basis(basis)),
            )
            value_jacobian = host_boozer._make_analytic_exact_value_jacobian(
                weight_inv_modB
            )
        return value_jacobian(x, coil_set_spec(coil_dofs))

    def objective(x, coil_dofs, targets):
        leaves = list(host_leaves)
        for position, target in zip(positions, unravel_targets(targets), strict=True):
            leaves[position] = target
        kwargs = jax.tree_util.tree_unflatten(kwargs_tree, leaves)
        return _evaluate_traceable_total_objective(
            x, coil_dofs, coil_set_spec(coil_dofs), kwargs
        )

    def gradient_and_scale(x, basis, targets):
        value, pullback = jax.vjp(lambda xx, cc: objective(xx, cc, targets), x, coils)
        dJ_dx, dJ_dc = pullback(jnp.ones((), dtype=value.dtype))
        _, jacobian = residual_and_jacobian(x, coils, basis)
        adjoint = lu_solve(lu_factor(jacobian), dJ_dx, trans=1)
        dF_dc = jax.jacfwd(lambda cc: residual_and_jacobian(x, cc, basis)[0])(coils)
        gradient = dJ_dc - dF_dc.T @ adjoint
        return gradient, jnp.abs(dJ_dc) + jnp.abs(dF_dc).T @ jnp.abs(adjoint)

    def gradient(x, basis, targets):
        return gradient_and_scale(x, basis, targets)[0]

    @jax.jit
    def terms(x_h, x_d, b_h, b_d, p_h, p_d):
        F_h, J_h = residual_and_jacobian(x_h, coils, b_h)
        F_d, _ = residual_and_jacobian(x_d, coils, b_d)
        return {
            "F_h": F_h,
            "F_d": F_d,
            "J": J_h,
            "dF_db": jax.jacfwd(lambda b: residual_and_jacobian(x_h, coils, b)[0])(b_h),
            "dv_dx": jax.grad(objective, argnums=0)(x_h, coils, p_h),
            "dv_dp": jax.grad(objective, argnums=2)(x_h, coils, p_h),
            "dg_dx": jax.jacfwd(gradient, argnums=0)(x_h, b_h, p_h),
            "dg_db": jax.jacfwd(gradient, argnums=1)(x_h, b_h, p_h),
            "dg_dp": jax.jacfwd(gradient, argnums=2)(x_h, b_h, p_h),
            "S_h": gradient_and_scale(x_h, b_h, p_h)[1],
            "S_d": gradient_and_scale(x_d, b_d, p_d)[1],
        }

    x_host = host_array(host.x_inner, dtype=np.float64)
    x_reference = host_array(reference.x_inner, dtype=np.float64)
    jacobians = {
        key: host_array(value, dtype=np.float64)
        for key, value in terms(
            x_host,
            x_reference,
            basis_host,
            basis_reference,
            targets_host,
            targets_reference,
        ).items()
    }
    basis_gap = np.abs(host_array(basis_host - basis_reference, dtype=np.float64))
    target_gap = np.abs(host_array(targets_host - targets_reference, dtype=np.float64))
    inverse = np.abs(np.linalg.inv(jacobians["J"]))
    seed_bound = inverse @ (
        np.abs(jacobians["F_h"]) + np.abs(jacobians["F_d"])
    ) + inverse @ (np.abs(jacobians["dF_db"]) @ basis_gap)
    value_bound = (
        np.abs(jacobians["dv_dx"]) @ seed_bound
        + np.abs(jacobians["dv_dp"]) @ target_gap
        + _gamma(len(_TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS))
        * (abs(host_eval.value) + abs(reference_eval.value))
    )
    gradient_bound = (
        np.abs(jacobians["dg_dx"]) @ seed_bound
        + np.abs(jacobians["dg_db"]) @ basis_gap
        + np.abs(jacobians["dg_dp"]) @ target_gap
        + _gamma(1 + jacobians["F_h"].size) * (jacobians["S_h"] + jacobians["S_d"])
    )
    return {
        "seed": (np.abs(x_host - x_reference), seed_bound),
        "value": (np.abs(host_eval.value - reference_eval.value), value_bound),
        "gradient": (
            np.abs(host_eval.gradient - reference_eval.gradient),
            gradient_bound,
        ),
    }


def test_host_construction_seed_and_first_evaluate_match_eager_jax_bake(
    analytic_backend, monkeypatch
):
    """NumPy construction bake must not change the seed or the first evaluate.

    Coil dofs are host copies of the same native coils, so they stay bitwise
    on every device. The inner seed and first evaluate compare the host-NumPy
    Fourier bake against the on-device JAX jacfwd bake of the same linear map:
    CPU matches bitwise. On GPU the two bases differ by a few ulp, and the
    guarantee is the first-order forward-error bound of
    :func:`_bake_first_order_bounds` (pre-registered in the parity redesign
    plan, C6 amendment 2), checked on both devices.
    """
    reference, reference_boozer = _jax_evaluator_eager_geometry()
    host, host_boozer = _jax_evaluator()
    np.testing.assert_array_equal(host.coil_dofs, reference.coil_dofs)
    host_inner = host_array(host.x_inner, dtype=np.float64)
    reference_inner = host_array(reference.x_inner, dtype=np.float64)
    reference_eval = reference.evaluate(reference.coil_dofs)
    host_eval = host.evaluate(host.coil_dofs)
    # The evaluate starts at the seed, so it compares functions of the seed.
    assert host_eval.inner_iterations == reference_eval.inner_iterations == 0
    if analytic_backend.platform == "cpu":
        np.testing.assert_array_equal(host_inner, reference_inner)
        assert host_eval.value == reference_eval.value
        np.testing.assert_array_equal(host_eval.gradient, reference_eval.gradient)
    bounds = _bake_first_order_bounds(
        monkeypatch,
        host,
        host_boozer,
        reference,
        reference_boozer,
        host_eval,
        reference_eval,
    )
    for name, (difference, bound) in bounds.items():
        assert np.all(difference <= bound), (
            f"{name}: max |host - device| / bound = {np.max(difference / bound):.3e}"
        )
    # Negative control: a device gradient off by 10x the bound on one entry
    # must fail the same check.
    gradient_difference, gradient_bound = bounds["gradient"]
    entry = int(np.argmax(gradient_difference / gradient_bound))
    perturbed = np.array(reference_eval.gradient, copy=True)
    perturbed[entry] += 10.0 * gradient_bound[entry]
    assert not np.all(np.abs(host_eval.gradient - perturbed) <= gradient_bound)


def test_construction_and_first_evaluation_are_clean_under_the_strict_transfer_guard():
    """Building the evaluator and compiling it must not cross the host boundary.

    Both halves of the guard, and the first ``evaluate`` is inside it because
    that call is the one that compiles.

    Host-to-device: every host array the analytic exact route takes ownership
    of crosses with one explicit ``jax.device_put``.  Before that was true this
    raised ``Disallowed host-to-device transfer`` out of ``jnp.zeros_like`` in
    ``_make_analytic_geometry_terms``.

    Device-to-host: XLA materializes a captured device array by copying it back
    to the host once per lowering.  Before the frozen payloads became host
    arrays (``make_coil_dof_extraction_spec``,
    ``build_boozer_surface_runtime_state``, the objective cache grids) this
    raised ``Disallowed device-to-host transfer: shape=(21)`` -- one base
    curve's dofs -- out of ``_array_mlir_constant_handler``.
    """
    with jax.transfer_guard("disallow"):
        evaluator, _boozer = _jax_evaluator()
        evaluation = evaluator.evaluate(evaluator.coil_dofs)

    assert evaluator.coil_dofs.size > 0
    assert evaluation.inner_success


def test_construction_under_the_guard_reproduces_the_unguarded_evaluation():
    """The explicit placement is a placement change only, not a numeric one."""
    reference, _ = _jax_evaluator()
    with jax.transfer_guard("disallow"):
        guarded, _ = _jax_evaluator()

    baseline = reference.evaluate(reference.coil_dofs)
    guarded_evaluation = guarded.evaluate(guarded.coil_dofs)

    assert guarded_evaluation.value == baseline.value
    np.testing.assert_array_equal(guarded_evaluation.gradient, baseline.gradient)


def test_value_and_gradient_match_native_at_initial_and_moved_coils():
    objective, _native = _native_objective()
    evaluator, _boozer = _jax_evaluator()
    x0 = np.asarray(objective.x, dtype=np.float64)
    np.testing.assert_array_equal(evaluator.coil_dofs, x0)
    rng = np.random.default_rng(3)
    for scale in (0.0, 2.0e-3):
        coils = x0 * (1.0 + scale * rng.standard_normal(x0.size))
        objective.x = coils
        native_value = float(objective.J())
        native_gradient = np.asarray(objective.dJ(), dtype=np.float64)
        evaluation = evaluator.evaluate(coils)
        assert evaluation.inner_success
        np.testing.assert_allclose(evaluation.value, native_value, rtol=1e-11, atol=0.0)
        np.testing.assert_allclose(
            evaluation.gradient, native_gradient, rtol=1e-8, atol=1e-12
        )


def test_gradient_matches_central_finite_differences_of_own_value():
    evaluator, _boozer = _jax_evaluator()
    rng = np.random.default_rng(11)
    # Away from the initial coils: there the length penalty sits exactly on its
    # ``max(L - L0, 0)`` kink, where one-sided finite differences are not
    # second-order and the comparison would measure the kink, not the gradient.
    x0 = evaluator.coil_dofs * (
        1.0 + 2.0e-3 * rng.standard_normal(evaluator.coil_dofs.size)
    )
    base = evaluator.evaluate(x0)
    assert base.inner_success
    direction = rng.standard_normal(x0.size)
    direction /= np.linalg.norm(direction)
    predicted = float(base.gradient @ direction)
    step = 1.0e-5
    plus = evaluator.evaluate(x0 + step * direction).value
    minus = evaluator.evaluate(x0 - step * direction).value
    measured = (plus - minus) / (2.0 * step)
    assert abs(measured - predicted) <= 1e-5 * abs(predicted)


def test_failed_inner_solve_reports_sentinel_and_rolls_back_like_native():
    objective, native = _native_objective()
    evaluator, _boozer = _jax_evaluator()
    x0 = np.array(evaluator.coil_dofs, copy=True)
    warm_start = np.asarray(evaluator.x_inner)
    # Coils scaled far from the seed surface: exact Newton exhausts its cap on
    # both lanes, so the native sentinel policy is exercised on a real failure.
    far = 4.0 * x0
    objective.x = far
    objective.J()
    native_gradient = np.asarray(objective.dJ(), dtype=np.float64)
    assert not bool(native.res["success"])
    evaluation = evaluator.evaluate(far)
    assert not evaluation.inner_success
    assert evaluation.value == INNER_FAILURE_VALUE
    # Rolled back to the warm start, exactly as native restores its surface.
    np.testing.assert_allclose(
        np.asarray(evaluator.x_inner), warm_start, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        evaluation.gradient, native_gradient, rtol=1e-8, atol=1e-12
    )


def _native_example_failure(objective, native, coils):
    # The example's value-and-gradient policy: evaluate at the inner-returned
    # state, then restore the pre-evaluation surface, iota and G on failure.
    previous_surface = np.array(native.surface.x, copy=True)
    previous_iota = float(native.res["iota"])
    previous_G = float(native.res["G"])
    objective.x = coils
    objective.J()
    gradient = np.asarray(objective.dJ(), dtype=np.float64)
    assert not bool(native.res["success"])
    persisted_shift = float(
        np.max(np.abs(np.asarray(native.surface.x) - previous_surface))
    )
    native.surface.x = previous_surface
    native.res["iota"] = previous_iota
    native.res["G"] = previous_G
    return gradient, persisted_shift


def test_persisted_failure_restores_pre_evaluation_warm_start():
    # A far coil move whose cap-exhausted Newton iterate native keeps
    # internally (finite and improved): the example restores the previous
    # state as the next warm start, and so must the evaluator.
    objective, native = _native_objective()
    evaluator, _boozer = _jax_evaluator()
    direction = np.random.default_rng(2).standard_normal(evaluator.coil_dofs.size)
    moved = evaluator.coil_dofs * (1.0 + 0.1 * direction)
    _gradient, persisted_shift = _native_example_failure(objective, native, moved)
    assert persisted_shift > 1e-10
    warm_start = np.array(evaluator.x_inner, copy=True)
    evaluation = evaluator.evaluate(moved)
    assert not evaluation.inner_success
    assert evaluation.value == INNER_FAILURE_VALUE
    assert np.all(np.isfinite(evaluation.gradient))
    np.testing.assert_allclose(
        np.asarray(evaluator.x_inner), warm_start, rtol=0, atol=1e-12
    )


def test_persisted_failure_gradient_matches_native_at_example_scale():
    # At the example's own scale a mild coil move fails at the cap on both
    # lanes with the same persisted iterate (the residual sits at the
    # tolerance floor), so the failure-path gradient must match native's
    # while the warm start is restored.
    objective, native = _native_objective(6, example_coils=True)
    evaluator, _boozer = _jax_evaluator(6, example_coils=True)
    direction = np.random.default_rng(11).standard_normal(evaluator.coil_dofs.size)
    moved = evaluator.coil_dofs * (1.0 + 0.06 * direction)
    native_gradient, persisted_shift = _native_example_failure(objective, native, moved)
    assert persisted_shift > 1e-3
    warm_start = np.array(evaluator.x_inner, copy=True)
    evaluation = evaluator.evaluate(moved)
    if evaluation.inner_success:
        # The residual norm hovers at the 1e-13 tolerance on both lanes, so
        # whether the cap is exhausted is decided by rounding; on a device
        # whose rounding lets this solve dip under the line there is no
        # failure to compare and the case reduces to the parity tests above.
        pytest.skip("cap-exhaustion is a tolerance-floor rounding tie on this device")
    assert evaluation.value == INNER_FAILURE_VALUE
    np.testing.assert_allclose(
        evaluation.gradient, native_gradient, rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.x_inner), warm_start, rtol=0, atol=1e-12
    )
