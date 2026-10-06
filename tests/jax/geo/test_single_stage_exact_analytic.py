"""The analytic exact single-stage evaluator reproduces the native example's value, gradient and policy."""

from __future__ import annotations

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    enable_non_strict_jax_backend,
    parity_device,
)

import math
from dataclasses import replace
from functools import reduce
from operator import add
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
from simsopt_jax.core._device_scalars import two_pi
from simsopt_jax.core._math_utils import as_jax_float64
from simsopt_jax.core.field import coil_set_spec_from_dof_extraction_spec
from simsopt_jax.core.specs import host_resident_spec
from simsopt_jax.runtime.host_boundary import host_array
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo import single_stage_exact_analytic
from simsopt_jax_adapters.geo.boozer_surface import (
    BoozerSurfaceJAX,
    _BoozerPenaltyGeometry,
)
from simsopt_jax_adapters.geo.single_stage_exact_analytic import (
    INNER_FAILURE_VALUE,
    ExactAnalyticSingleStage,
    HostConstructionBoozerSurfaceJAX,
)
from simsopt_jax_adapters.geo.single_stage_host_construction import (
    _TWO_PI,
    _host_eval_hat,
    _host_identity_xyzc,
    _host_move_dof_axis,
    _host_phi_basis,
    _host_theta_basis,
)
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    _evaluate_traceable_total_objective,
)

INITIAL_IOTA = -0.406
RESOLUTION = 1
QS_RESOLUTION = 4
UNIT_ROUNDOFF = 2.0**-53


@pytest.fixture(params=("cpu", "gpu"), autouse=True)
def analytic_backend(jax_runtime_guard, monkeypatch, request):
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
    """Higham's ``gamma_n``: bound on the relative error of ``n`` roundings."""
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


# Host-vs-device construction bake (parity redesign PLAN.md, amendment 5 part 2,
# section K). Every analytic-geometry basis entry is a sum of at most two terms,
# each a product of rounded factors of ``alpha = fl(k fl(tau q))`` (tau =
# fl(2 acos(-1)), k an integer mode number) with at most five roundings (the
# xphi in-plane term ``dv w c``: k tau, times sin, times w, times the rotation,
# final add) and three sin/cos factors. A sin/cos within E ulp of exact has
# relative error at most 2 E u, i.e. ceil(2 E) unit roundings, so each lane's
# entry is within ``gamma_K S_b`` of the exact entry, K = 5 + 3 ceil(2 E).
_BAKE_ROUNDINGS = 5
_BAKE_TRIG_FACTORS = 3
# Largest sin/cos error in ulp of the exact value, per lane. Host NumPy 2.4.6
# runs glibc 2.43 here (bitwise equal to ``math`` at every bake argument,
# measured 0.486 ulp; glibc documents about 0.55 ulp). XLA CPU is bitwise equal
# to glibc. XLA GPU calls libdevice, documented within 2 ulp of the correctly
# rounded result (CUDA Programming Guide), hence 2.5 ulp of exact.
_HOST_TRIG_ULP = 1.0
_DEVICE_TRIG_ULP = {"cpu": 1.0, "gpu": 2.5}
# Pre-registered regime ceiling on cond_2 of the seed Jacobian (observed 53).
_SEED_JACOBIAN_COND_CEILING = 1.0e3


def _assert_bitwise(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(host_array(actual, dtype=np.float64)).view(np.uint64),
        np.asarray(host_array(expected, dtype=np.float64)).view(np.uint64),
    )


def _bake_term_scale(boozer) -> _BoozerPenaltyGeometry:
    """Per-entry sum of |terms| of the analytic-geometry basis (its ``S_b``).

    The same one-hot Fourier evaluation as the host bake, with every factor
    replaced by its absolute value and every difference by a sum.
    """
    args = boozer._traceable_surface_runtime_args()
    mpol, ntor = int(args["mpol"]), int(args["ntor"])
    phi = np.asarray(args["quadpoints_phi"], dtype=np.float64)
    w_theta, dw_theta = _host_theta_basis(
        np.asarray(args["quadpoints_theta"], dtype=np.float64), mpol
    )
    v_phi, dv_phi = _host_phi_basis(phi, ntor, int(args["nfp"]))
    w, dw, v, dv = (np.abs(basis) for basis in (w_theta, dw_theta, v_phi, dv_phi))
    xc, yc, zc = _host_identity_xyzc(
        n_dofs=int(np.asarray(boozer.surface.get_dofs()).size),
        mpol=mpol,
        ntor=ntor,
        stellsym=bool(args["stellsym"]),
        scatter_indices=args["scatter_indices"],
    )
    angle = _TWO_PI * phi
    cosine = np.abs(np.cos(angle))[None, :, None]
    sine = np.abs(np.sin(angle))[None, :, None]

    def rotated(radial, toroidal, vertical):
        return _host_move_dof_axis(
            np.stack(
                [
                    radial * cosine + toroidal * sine,
                    radial * sine + toroidal * cosine,
                    vertical,
                ],
                axis=-1,
            )
        )

    def hats(phi_basis, theta_basis):
        return tuple(
            _host_eval_hat(phi_basis, theta_basis, coefficients)
            for coefficients in (xc, yc, zc)
        )

    x_hat, y_hat, z_hat = hats(v, w)
    dx_dphi, dy_dphi, dz_dphi = hats(dv, w)
    dx_dtheta, dy_dtheta, dz_dtheta = hats(v, dw)
    return _BoozerPenaltyGeometry(
        gamma=rotated(x_hat, y_hat, z_hat),
        xphi=rotated(dx_dphi + _TWO_PI * y_hat, dy_dphi + _TWO_PI * x_hat, dz_dphi),
        xtheta=rotated(dx_dtheta, dy_dtheta, dz_dtheta),
    )


def _basis_gap_and_bound(host_basis, reference_basis, term_scale, device_trig_ulp):
    """K1: ``|b_h - b_d| <= (gamma_Kh + gamma_Kd) S_b / (1 - gamma_Kh)``."""

    def roundings(trig_ulp):
        return _BAKE_ROUNDINGS + _BAKE_TRIG_FACTORS * math.ceil(2.0 * trig_ulp)

    host_roundings = roundings(_HOST_TRIG_ULP)
    factor = (_gamma(host_roundings) + _gamma(roundings(device_trig_ulp))) / (
        1.0 - _gamma(host_roundings)
    )
    gap = np.concatenate(
        [
            np.abs(
                host_array(host_leaf, dtype=np.float64)
                - host_array(reference_leaf, dtype=np.float64)
            ).ravel()
            for host_leaf, reference_leaf in zip(
                jax.tree.leaves(host_basis),
                jax.tree.leaves(reference_basis),
                strict=True,
            )
        ]
    )
    bound = factor * np.concatenate(
        [leaf.ravel() for leaf in jax.tree.leaves(term_scale)]
    )
    return gap, bound


def _shared_seed_host_lane(seed):
    """Host-construction lane built from a solved seed through the production API.

    The label target is the volume of the unsolved surface, taken before the
    surface moves to the seed, exactly as the reference lane took it.
    """
    base_curves, base_currents, nfp, native_field, surface, _ = _problem()
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    volume = Volume(surface)
    target = float(volume.J())
    surface.set_dofs(seed[:-2])
    boozer = HostConstructionBoozerSurfaceJAX(
        field,
        surface,
        volume,
        target,
        options={"newton_maxiter": 20, "newton_tol": 1.0e-13, "verbose": False},
    )
    length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    evaluator = ExactAnalyticSingleStage(
        boozer,
        field,
        iota=float(seed[-2]),
        G=float(seed[-1]),
        outer_objective_config=lambda: _config(nfp, surface, length_target),
    )
    return evaluator, boozer


def _objective_leaves(evaluator):
    return jax.tree_util.tree_leaves(
        evaluator._objective_cache_state["objective_kwargs"]
    )


def _seed_jacobian_and_gradient_scale(evaluator, boozer, coil_dofs):
    """Seed Jacobian ``J = dF/dx`` and the gradient's summation scale.

    ``S = |dJ/dc| + |dF/dc|^T |lambda|`` per entry, with ``J^T lambda = dJ/dx``.
    """
    extraction_spec = host_resident_spec(boozer.biotsavart.coil_dof_extraction_spec())
    value_jacobian = boozer._make_analytic_exact_value_jacobian(
        boozer.options["weight_inv_modB"]
    )
    objective_kwargs = evaluator._objective_cache_state["objective_kwargs"]

    def coil_set_spec(coils):
        return coil_set_spec_from_dof_extraction_spec(
            extraction_spec, as_jax_float64(coils)
        )

    @jax.jit
    def jacobian_and_scale(x, coils):
        value, pullback = jax.vjp(
            lambda xx, cc: _evaluate_traceable_total_objective(
                xx, cc, coil_set_spec(cc), objective_kwargs
            ),
            x,
            coils,
        )
        dJ_dx, dJ_dc = pullback(jnp.ones((), dtype=value.dtype))
        _, jacobian = value_jacobian(x, coil_set_spec(coils))
        adjoint = lu_solve(lu_factor(jacobian), dJ_dx, trans=1)
        dF_dc = jax.jacfwd(lambda cc: value_jacobian(x, coil_set_spec(cc))[0])(coils)
        return jacobian, jnp.abs(dJ_dc) + jnp.abs(dF_dc).T @ jnp.abs(adjoint)

    jacobian, scale = jacobian_and_scale(evaluator.x_inner, jnp.asarray(coil_dofs))
    return host_array(jacobian, dtype=np.float64), host_array(scale, dtype=np.float64)


def test_host_bake_basis_is_within_the_trig_library_error_of_the_device_bake(
    analytic_backend, monkeypatch
):
    """K1: the NumPy basis is within the sin/cos library error of the JAX basis.

    Both bakes form the same trig arguments with the same IEEE multiplications
    (tau is asserted bitwise), so each basis entry differs only through the two
    sin/cos libraries and the rounding of at most five operations per term; the
    per-entry bound uses fixed library constants and no measured gap.
    Negative control: a 1e-8 relative error injected into one basis entry fails.
    """
    reference, reference_boozer = _jax_evaluator_eager_geometry()
    host, host_boozer = _shared_seed_host_lane(
        host_array(reference.x_inner, dtype=np.float64)
    )
    device_two_pi = two_pi(jnp.zeros((), dtype=jnp.float64))
    _assert_bitwise(device_two_pi, _TWO_PI)
    host_geometry, host_basis, _ = host_boozer._make_analytic_geometry_terms()
    reference_geometry, reference_basis, _ = (
        reference_boozer._make_analytic_geometry_terms()
    )
    zero_dofs = jnp.zeros(host_basis.gamma.shape[-1], dtype=jnp.float64)
    for host_leaf, reference_leaf in zip(
        jax.tree.leaves(host_geometry(zero_dofs)),
        jax.tree.leaves(reference_geometry(zero_dofs)),
        strict=True,
    ):
        np.testing.assert_array_equal(
            host_array(host_leaf, dtype=np.float64),
            host_array(reference_leaf, dtype=np.float64),
        )
    term_scale = _bake_term_scale(host_boozer)
    device_trig_ulp = _DEVICE_TRIG_ULP[analytic_backend.platform]
    gap, bound = _basis_gap_and_bound(
        host_basis, reference_basis, term_scale, device_trig_ulp
    )
    assert np.all(gap <= bound), (
        f"basis: max |b_h - b_d| / bound = {np.max(gap / np.where(bound > 0, bound, 1.0)):.3e}"
    )
    print(
        f"\nK1 {analytic_backend.platform}: entries {gap.size}, differing "
        f"{int(np.count_nonzero(gap))}, max |b_h - b_d| / bound "
        f"{np.max(gap[bound > 0] / bound[bound > 0]):.4f}"
    )

    real_bake = single_stage_exact_analytic.host_analytic_geometry_origin_and_basis

    def injected_bake(**kwargs):
        origin, basis = real_bake(**kwargs)
        xphi = np.array(basis.xphi, dtype=np.float64, copy=True)
        entry = np.unravel_index(int(np.argmax(np.abs(xphi))), xphi.shape)
        xphi[entry] *= 1.0 + 1.0e-8
        return origin, replace(basis, xphi=xphi)

    with monkeypatch.context() as patch:
        patch.setattr(
            single_stage_exact_analytic,
            "host_analytic_geometry_origin_and_basis",
            injected_bake,
        )
        injected, injected_boozer = _shared_seed_host_lane(
            host_array(reference.x_inner, dtype=np.float64)
        )
        _, injected_basis, _ = injected_boozer._make_analytic_geometry_terms()
    injected_gap, injected_bound = _basis_gap_and_bound(
        injected_basis, reference_basis, term_scale, device_trig_ulp
    )
    assert not np.all(injected_gap <= injected_bound)
    print(
        f"K7(i) {analytic_backend.platform}: injected max |b_h - b_d| / bound "
        f"{np.max(injected_gap[injected_bound > 0] / injected_bound[injected_bound > 0]):.3e}, "
        f"initial Newton steps {injected.initial_inner_iterations} "
        f"(uninjected {host.initial_inner_iterations})"
    )


def test_host_construction_seed_and_first_evaluate_match_eager_jax_bake(
    analytic_backend, monkeypatch
):
    """NumPy construction bake must not change the seed or the first evaluate.

    Coil dofs are host copies of the same native coils, so they stay bitwise
    on every device. CPU: the independently constructed host lane matches the
    eager JAX bake bitwise. Both devices (amendment 5 part 2, section K):

    - K2: the host lane built from the reference lane's solved seed takes no
      Newton step and keeps the seed and every outer target bitwise.
    - K3: its first value is bitwise the reference's; no operation on the
      value path reads the basis once x, coils and targets are shared.
    - K5: the same construction with the reference's own basis substituted
      reproduces the reference bitwise, so the host lane is the reference
      program with only the basis swapped (whose error K1 bounds). No
      operation-count bound on the gradient gap is derivable (K4), so that gap
      is reported in units of ``u S`` together with ``cond_2(J)``.
    - K6: ``cond_2(J) <= 1e3`` for both lanes at the seed.
    - K7(ii): a fault injected into the host lane's own evaluation path (a
      1e-12 relative error in one entry of its adjoint solve) leaves the value
      bitwise and fails the K5 gradient comparison (PLAN.md amendment 8, B).
    """
    reference, reference_boozer = _jax_evaluator_eager_geometry()
    host, _ = _jax_evaluator()
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

    reference_targets = _objective_leaves(reference)

    def assert_shared_seed(lane, lane_boozer):
        assert lane_boozer.targetlabel == reference_boozer.targetlabel
        assert lane.initial_inner_iterations == 0
        _assert_bitwise(lane.x_inner, reference_inner)
        assert (
            _differing_float_positions(_objective_leaves(lane), reference_targets) == ()
        )
        evaluation = lane.evaluate(lane.coil_dofs)
        assert evaluation.inner_iterations == 0
        _assert_bitwise(lane.x_inner, reference_inner)
        return evaluation

    shared, shared_boozer = _shared_seed_host_lane(reference_inner)
    shared_eval = assert_shared_seed(shared, shared_boozer)
    _assert_bitwise(shared_eval.value, reference_eval.value)

    real_bake = single_stage_exact_analytic.host_analytic_geometry_origin_and_basis
    _, reference_basis, _ = reference_boozer._make_analytic_geometry_terms()

    def reference_bake(**kwargs):
        origin, _ = real_bake(**kwargs)
        return origin, jax.tree.map(
            lambda leaf: np.array(host_array(leaf, dtype=np.float64), copy=True),
            reference_basis,
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            single_stage_exact_analytic,
            "host_analytic_geometry_origin_and_basis",
            reference_bake,
        )
        substituted, substituted_boozer = _shared_seed_host_lane(reference_inner)
        substituted_eval = assert_shared_seed(substituted, substituted_boozer)
    _assert_bitwise(substituted_eval.value, reference_eval.value)
    _assert_bitwise(substituted_eval.gradient, reference_eval.gradient)

    host_jacobian, gradient_scale = _seed_jacobian_and_gradient_scale(
        shared, shared_boozer, shared.coil_dofs
    )
    reference_jacobian, _ = _seed_jacobian_and_gradient_scale(
        reference, reference_boozer, reference.coil_dofs
    )
    conditions = [
        float(np.linalg.cond(jacobian))
        for jacobian in (host_jacobian, reference_jacobian)
    ]
    assert max(conditions) <= _SEED_JACOBIAN_COND_CEILING, conditions
    gradient_gap = np.abs(shared_eval.gradient - reference_eval.gradient)
    gap_in_scale = gradient_gap / (UNIT_ROUNDOFF * gradient_scale)
    print(
        f"\nK4 report {analytic_backend.platform}: max |g_h - g_d| / (u S) "
        f"{np.max(gap_in_scale):.2f} (entry {int(np.argmax(gap_in_scale))}), "
        f"max relative {np.max(gradient_gap / np.abs(reference_eval.gradient)):.3e}, "
        f"cond_2(J) host {conditions[0]:.2f} reference {conditions[1]:.2f}"
    )

    real_lu_solve = single_stage_exact_analytic.lu_solve

    def faulty_lu_solve(factors, rhs, trans=0):
        return real_lu_solve(factors, rhs, trans=trans).at[0].multiply(1.0 + 1.0e-12)

    with monkeypatch.context() as patch:
        patch.setattr(
            single_stage_exact_analytic,
            "host_analytic_geometry_origin_and_basis",
            reference_bake,
        )
        patch.setattr(single_stage_exact_analytic, "lu_solve", faulty_lu_solve)
        faulty, faulty_boozer = _shared_seed_host_lane(reference_inner)
        faulty_eval = assert_shared_seed(faulty, faulty_boozer)
    _assert_bitwise(faulty_eval.value, reference_eval.value)
    assert not np.array_equal(
        np.asarray(host_array(faulty_eval.gradient, dtype=np.float64)).view(np.uint64),
        np.asarray(host_array(reference_eval.gradient, dtype=np.float64)).view(
            np.uint64
        ),
    )


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


def test_later_evaluations_reuse_the_first_compiled_kernel():
    """The initial warm start is committed like the kernel's own outputs.

    ``jit`` keys committed and uncommitted arguments separately; every later
    warm start is a committed kernel output, so an uncommitted initial one
    would compile a second executable.
    """
    evaluator, _boozer = _jax_evaluator()
    x0 = np.asarray(evaluator.coil_dofs, dtype=np.float64)

    assert evaluator.evaluate(x0).inner_success
    assert evaluator.evaluate(x0 * (1.0 + 1.0e-4)).inner_success

    assert evaluator._evaluate_kernel._cache_size() == 1


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
