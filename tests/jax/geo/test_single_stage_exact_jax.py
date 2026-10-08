"""Fused exact boozerQA against native Optimizable composition and re-solves.

The state/coil layout is native's NCSX example; native values and Derivatives
are independent anchors. Default 1e-13/40 Newton settings are never relaxed.
Finite differences use their truncation tolerance, not a relaxed parity gate.
"""

from jax_test_support import (
    assert_matches_native,
    fixture_jax_runtime_guard,  # noqa: F401
    fixture_parity_lane,  # noqa: F401
    jax_compilations,
    parity_default_device,
    parity_rng,
)

from copy import copy
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import cast

import numpy as np
from numpy.typing import DTypeLike
import jax
import pytest
from scipy.optimize import minimize
from scipy.linalg import lu

from simsopt.configs import get_data
from simsopt._core.derivative import Derivative
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, CurveLength, Iotas, MajorRadius, NonQuasiSymmetricRatio, SurfaceXYZTensorFourier, Volume, boozer_surface_residual
from simsopt.objectives import QuadraticPenalty
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.single_stage_exact import ExactSingleStageState, _absolute, exact_single_stage_evaluate
from simsopt_jax.core.surface_fourier_series import surface_spec_with_dofs
from simsopt_jax.core.surface_geometry import surface_gamma, surface_gammadash1, surface_gammadash2, surface_volume
from simsopt_jax.runtime.host_boundary import disallow_host_transfers, host_array, host_value
from simsopt_jax_adapters.field import JaxBiotSavart
from simsopt_jax_adapters.geo.boozer_surface import JaxBoozerSurface
from simsopt_jax_adapters.geo.single_stage_exact import JaxExactSingleStage
from simsopt_jax_adapters.geo.surface_objectives import JaxNonQuasiSymmetricRatio


def _problem(*, dropin=False, quasi_poloidal=False, active_penalties=False):
    curves, currents, axis, nfp, bs = get_data("ncsx")
    surface = SurfaceXYZTensorFourier(
        mpol=6, ntor=6, stellsym=True, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, 13, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 13, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1, flip_theta=True)
    volume = Volume(surface)
    field = JaxBiotSavart(bs.coils)
    boozer = (
        JaxBoozerSurface(field, surface, volume, volume.J(), options={"verbose": False})
        if dropin else BoozerSurface(bs, surface, volume, volume.J(), options={"verbose": False})
    )
    G = 4 * np.pi * 1e-7 * nfp * sum(abs(c.get_value()) for c in currents)
    solved = boozer.solve_residual_equation_exactly_newton(iota=-0.406, G=G, tol=1e-13, maxiter=20)
    assert solved["success"], "native example initialization must converge"
    radius = MajorRadius(boozer)
    lengths = [CurveLength(curve) for curve in curves]
    total_length = lengths[0] + lengths[1] + lengths[2]
    iota_target = solved["iota"] + (0.002 if active_penalties else 0)
    radius_target = float(np.asarray(radius.J())) + (0.003 if active_penalties else 0)
    length_target = float(total_length.J()) * (0.999 if active_penalties else 1)
    ratio = (
        JaxNonQuasiSymmetricRatio(boozer, JaxBiotSavart(bs.coils), quasi_poloidal=quasi_poloidal)
        if dropin else NonQuasiSymmetricRatio(boozer, BiotSavart(bs.coils), quasi_poloidal=quasi_poloidal)
    )
    objective = (
        ratio + QuadraticPenalty(Iotas(boozer), iota_target)
        + QuadraticPenalty(radius, radius_target) + QuadraticPenalty(total_length, length_target, "max")
    )
    currents[0].fix_all()
    evaluator = JaxExactSingleStage.from_boozer_surface(
        boozer, field, curves, iota_target=iota_target,
        major_radius_target=radius_target, length_target=length_target,
        quasi_poloidal=quasi_poloidal,
    )
    np.testing.assert_array_equal(host_array(evaluator.initial_parameters), objective.x)
    return evaluator, boozer, objective, currents


def _radius_reduction_bound(surface):
    """First-order float64 error in R=|V|/(2*pi*A) from two reductions.

    A sum of N terms has error <= gamma_N*sum(abs(terms)), with
    gamma_N=N*u/(1-N*u), u=eps/2. Each implementation sums volume and
    section area, so |delta R| <= 2*gamma_N*|R|*(kappa_V+kappa_A).
    The 2 counts native and JAX, not an empirical tolerance multiplier.
    Absolute integrands retain the conditioning of cancellation in A/V.
    This reduction-only envelope remains an additional tight fixture gate.
    _assert_radius_arithmetic_bounds separately certifies the other stages;
    this expression alone is not the full native/fused error budget.
    """
    gamma, phi, theta = surface.gamma(), surface.gammadash1(), surface.gammadash2()
    x, y = gamma[..., 0], gamma[..., 1]
    section = (
        theta[..., 2] * (x * phi[..., 1] - y * phi[..., 0])
        - phi[..., 2] * (x * theta[..., 1] - y * theta[..., 0])
    ) / np.sqrt(x * x + y * y)
    volume = np.sum(gamma * np.cross(phi, theta), axis=-1) / 3
    conditioning = (
        np.mean(np.abs(volume)) / abs(np.mean(volume))
        + np.mean(np.abs(section)) / abs(np.mean(section))
    )
    unit_roundoff = np.finfo(np.float64).eps / 2
    gamma_n = section.size * unit_roundoff / (1 - section.size * unit_roundoff)
    return 2 * gamma_n * abs(surface.major_radius()) * conditioning


def _roundoff_gamma(operations, dtype: DTypeLike = np.float64):
    u = np.finfo(dtype).eps / 2
    return operations * u / (1 - operations * u)


def _radius_integrands(geometry):
    """Extended-precision closed-form integrands and absolute product sums."""
    xyz, phi, theta = (np.asarray(array, dtype=np.longdouble) for array in geometry)
    x, y = xyz[..., 0], xyz[..., 1]
    r = np.sqrt(x * x + y * y)
    p = x * phi[..., 1] - y * phi[..., 0]
    t = x * theta[..., 1] - y * theta[..., 0]
    section = (theta[..., 2] * p - phi[..., 2] * t) / r
    section_scale = (
        abs(theta[..., 2]) * (abs(x * phi[..., 1]) + abs(y * phi[..., 0]))
        + abs(phi[..., 2]) * (abs(x * theta[..., 1]) + abs(y * theta[..., 0]))
    ) / r
    volume = np.sum(xyz * np.cross(phi, theta), axis=-1) / 3
    volume_scale = np.sum(abs(xyz) * (
        abs(phi[..., [1, 2, 0]] * theta[..., [2, 0, 1]])
        + abs(phi[..., [2, 0, 1]] * theta[..., [1, 2, 0]])
    ), axis=-1) / 3
    return volume, section, volume_scale, section_scale


def _assert_radius_arithmetic_bounds(surface, spec, fused_radius):
    """Certify each omitted stage without changing the existing scalar gate.

    Fourier/trig/rotation error is measured at the geometry arrays, then
    propagated via their closed-form integrands (not inferred from delta R).
    Long-double reference arithmetic has its own gamma_(N+16) budget.
    Volume's longest product/add/divide path has seven roundings, including
    native's rounded 1/3. Closed section has four numerator roundings, two
    in x*x+y*y, sqrt and division: ignoring sqrt's contraction gives gamma_8.
    Native det/inv error is certified from its actual map
    defects below, independently of LAPACK's implementation; the following
    products/sum have four roundings. Each mean adds gamma_N times the
    absolute integrands INCLUDING their local errors.

    Reconstruction A -> sqrt(A/pi) -> square -> R has eight relative error
    factors (sqrt appears twice; pi**2 is rounded), so gamma_8*|R| applies.
    For primitive errors eV,eS, the exact quotient perturbation bound is
    (eV+|V/S|*eS)/(|S|-eS). The same identity propagates geometry changes.
    These bounds certify the full discrepancy; the original reduction-only
    and unexplained-gradient gates below are additionally retained verbatim.
    This certificate applies to the regular, non-overflowing NCSX fixture.
    """
    native_geometry = (surface.gamma(), surface.gammadash1(), surface.gammadash2())
    fused_geometry = tuple(host_array(quantity(spec)) for quantity in (surface_gamma, surface_gammadash1, surface_gammadash2))
    for actual, expected in zip(fused_geometry, native_geometry):
        assert_matches_native(actual, expected, "radius Fourier geometry")
    native = _radius_integrands(native_geometry)
    fused = _radius_integrands(fused_geometry)
    n = native[0].size
    reduction_gamma = _roundoff_gamma(n)
    reference_gamma = _roundoff_gamma(n + 16, np.longdouble)

    # Map defects include forming its entries, det/inv, and radial sqrt.
    # For exact a=p/r^2,b=t/r^2 and computed d=det(J),h=inv(J),
    # |r*d*(zp*h01+zt*h11)-r*(a*zt-b*zp)| is bounded by
    # r*(|d-a|*(|zp*h01|+|zt*h11|)
    #    +|a|*(|zp|*|h01+b/a|+|zt|*|h11-1|)).
    xyz64, phi64, theta64 = native_geometry
    x64, y64 = xyz64[..., 0], xyz64[..., 1]
    r2_64 = x64 * x64 + y64 * y64
    mapping = np.zeros(x64.shape + (2, 2))
    mapping[..., 0, 0] = (x64 * phi64[..., 1] - y64 * phi64[..., 0]) / r2_64
    mapping[..., 0, 1] = (x64 * theta64[..., 1] - y64 * theta64[..., 0]) / r2_64
    mapping[..., 1, 1] = 1
    determinant = np.asarray(np.linalg.det(mapping), dtype=np.longdouble)
    inverse = np.asarray(np.linalg.inv(mapping), dtype=np.longdouble)
    xyz, phi, theta = (np.asarray(array, dtype=np.longdouble) for array in native_geometry)
    x, y = xyz[..., 0], xyz[..., 1]
    r2 = x * x + y * y
    r = np.sqrt(r2)
    a = (x * phi[..., 1] - y * phi[..., 0]) / r2
    b = (x * theta[..., 1] - y * theta[..., 0]) / r2
    map_scale = abs(phi[..., 2] * inverse[..., 0, 1]) + abs(theta[..., 2] * inverse[..., 1, 1])
    map_error = r * (
        abs(determinant - a) * map_scale
        + abs(a) * (abs(phi[..., 2]) * abs(inverse[..., 0, 1] + b / a)
                    + abs(theta[..., 2]) * abs(inverse[..., 1, 1] - 1))
    )
    rounded_r = np.asarray(np.sqrt(r2_64), dtype=np.longdouble)
    map_error += abs(rounded_r - r) * abs(determinant) * map_scale
    native_section_error = map_error + _roundoff_gamma(4) * rounded_r * abs(determinant) * map_scale

    primitive_errors = []
    mean_errors = []
    radii = []
    for integrands, section_error in (
        (native, native_section_error), (fused, _roundoff_gamma(8) * fused[3]),
    ):
        volume, section, volume_scale, section_scale = integrands
        v_error = _roundoff_gamma(7) * volume_scale
        s_error = section_error
        e_v = np.mean(v_error + reduction_gamma * (abs(volume) + v_error))
        e_s = np.mean(s_error + reduction_gamma * (abs(section) + s_error))
        e_v += reference_gamma * np.mean(volume_scale)
        e_s += reference_gamma * np.mean(section_scale)
        v, s = abs(np.mean(volume)), abs(np.mean(section))
        assert s > e_s, "radius arithmetic certificate requires a nonzero section mean"
        radius = v / s
        radii.append(radius)
        mean_errors.append((e_v, e_s))
        primitive_errors.append((e_v + radius * e_s) / (s - e_s))

    # Separately verify native primitive and reconstruction stages, so a
    # budget cannot silently compensate for a formula error in another stage.
    v_n, s_n = abs(np.mean(native[0])), abs(np.mean(native[1]))
    native_v = np.longdouble(abs(surface.volume()))
    native_a = np.longdouble(surface.mean_cross_sectional_area())
    reconstructed = native_v / (2 * np.longdouble(np.pi) * native_a)
    e_v_native, e_s_native = mean_errors[0]
    area_error = (e_s_native + _roundoff_gamma(1) * (s_n + e_s_native)) / (2 * np.longdouble(np.pi))
    assert abs(native_v - v_n) <= e_v_native, "native volume arithmetic exceeds its operation-count bound"
    assert abs(native_a - s_n / (2 * np.longdouble(np.pi))) <= area_error, "native det/inv section arithmetic exceeds its defect bound"
    assert abs(np.longdouble(surface.major_radius()) - reconstructed) <= _roundoff_gamma(8) * reconstructed
    assert abs(native_v / (2 * np.longdouble(np.pi) * native_a) - radii[0]) <= primitive_errors[0] + _roundoff_gamma(1) * radii[0]
    fused_volume = abs(np.longdouble(host_value(surface_volume(spec))))
    assert abs(fused_volume - abs(np.mean(fused[0]))) <= mean_errors[1][0]
    e_v_geometry = np.mean(abs(fused[0] - native[0])) + reference_gamma * np.mean(native[2] + fused[2])
    e_s_geometry = np.mean(abs(fused[1] - native[1])) + reference_gamma * np.mean(native[3] + fused[3])
    assert s_n > e_s_geometry, "geometry certificate requires a nonzero section mean"
    geometry_error = (e_v_geometry + (v_n / s_n) * e_s_geometry) / (s_n - e_s_geometry)
    assert abs(radii[1] - radii[0]) <= geometry_error, "Fourier geometry propagation exceeds its quotient bound"
    reconstruction_errors = [_roundoff_gamma(8) * (radius + error) for radius, error in zip(radii, primitive_errors)]
    assert abs(np.longdouble(fused_radius) - radii[1]) <= primitive_errors[1] + reconstruction_errors[1]
    complete_bound = geometry_error + sum(primitive_errors) + sum(reconstruction_errors)
    assert abs(np.longdouble(fused_radius) - np.longdouble(surface.major_radius())) <= complete_bound


@pytest.mark.parametrize("dropin", [False, True], ids=["native", "dropin"])
@pytest.mark.parametrize("quasi_poloidal,active_penalties", [(False, False), (False, True), (True, True)], ids=["QA", "QA-active", "QP-active"])
def test_value_gradient_and_solved_state_match_composition(dropin, quasi_poloidal, active_penalties):
    """Independent solves agree, then native objective math at the same
    inner point isolates fused composition from the 1e-13 residual floor."""
    evaluator, boozer, objective, _ = _problem(
        dropin=dropin, quasi_poloidal=quasi_poloidal, active_penalties=active_penalties,
    )
    state = evaluator.initial_state
    parameters = host_array(evaluator.initial_parameters)
    direction = parity_rng(13).uniform(size=parameters.size)
    for scale in (0, 1e-4, -1e-4):
        x = parameters + scale * direction
        result = evaluator.evaluate(x, state)
        objective.x = x
        expected_value = objective.J()
        assert boozer.res["success"] and result.success
        assert_matches_native(result.value, expected_value, "fused objective")
        native_inner = np.concatenate((boozer.surface.get_dofs(), [boozer.res["iota"], boozer.res["G"]]))
        assert_matches_native(result.state.x, native_inner, "fused warm-start solution")
        radius = next(obj for obj in objective.ancestors if isinstance(obj, MajorRadius))
        native_observables = np.array([boozer.res["iota"], radius.J()])
        # Independently recompute the analytic native Jacobian and label row
        # at the fused iterate, without another Newton step or JAX matrices.
        solved = host_array(result.solved_x)
        boozer.surface.set_dofs(solved[:-2])
        boozer.res["iota"], boozer.res["G"] = solved[-2:]
        _, jacobian = cast(tuple[np.ndarray, np.ndarray], boozer_surface_residual(
            boozer.surface, solved[-2], solved[-1], BiotSavart(boozer.biotsavart.coils), derivatives=1,
        ))
        tail = np.zeros((1, solved.size))
        tail[0, :-2] = boozer.surface.dvolume_by_dcoeff()
        rows = host_array(evaluator.problem.residual_rows)
        system_jacobian = np.concatenate((jacobian[rows], tail))
        boozer.res["PLU"] = lu(system_jacobian)
        boozer.need_to_run_code = False
        for ancestor in objective.ancestors:
            if isinstance(ancestor, (MajorRadius, NonQuasiSymmetricRatio, JaxNonQuasiSymmetricRatio, Iotas)):
                ancestor.recompute_bell()
        assert_matches_native(result.value, objective.J(), "shared-inner fused objective")
        expected_gradient = objective.dJ()
        if active_penalties:
            assert_matches_native(result.gradient, expected_gradient, "shared-inner fused free gradient")
        else:
            # Newton guarantees ||F||_2 <= tau, not relative accuracy of g.
            # For q=iota/R, A^T lambda_q=q_x gives delta q=lambda_q^T F
            # to first order: two solves allow (tau_f+tau_n)||lambda_q||_2.
            # A quadratic penalty has g_q=(q-target)*dq/dcoils, so scalar
            # uncertainty is amplified by dq/dcoils even at a zero penalty.
            # After projecting both onto the SAME inner point, this Newton
            # uncertainty cancels. What remains is float64 scalar arithmetic
            # error: certify R's stages, keep its tight reduction check, and L's
            # four-ulp gate. Subtract ONLY the measured scalar propagation;
            # the unchanged 1e-12 gate must still hold for unexplained error.
            radius_roundoff = _radius_reduction_bound(boozer.surface)
            radius_spec = surface_spec_with_dofs(evaluator.problem.boozer.surface, result.solved_x[:-2])
            _assert_radius_arithmetic_bounds(boozer.surface, radius_spec, result.terms[2])
            observable_partials = np.zeros((solved.size, 2))
            observable_partials[-2, 0] = 1
            observable_partials[:-2, 1] = boozer.surface.dmajor_radius_by_dcoeff()
            adjoints = np.linalg.solve(system_jacobian.T, observable_partials)
            tau_sum = float(host_value(evaluator.problem.tol)) + boozer.options["newton_tol"]
            solve_uncertainty = tau_sum * np.linalg.norm(adjoints, axis=0)
            assert np.all(
                np.abs(result.terms[1:3] - native_observables)
                <= solve_uncertainty + np.array([0, radius_roundoff])
            ), "iota/radius difference exceeds the stopping-tolerance/adjoint budget"
            radius_difference = result.terms[2] - radius.J()
            assert abs(radius_difference) <= radius_roundoff, "shared-inner radius exceeds its reduction envelope"
            radius_gradient = cast(Callable[..., Derivative], radius.dJ)(partials=True)(objective)
            lengths = [obj for obj in objective.ancestors if isinstance(obj, CurveLength)]
            native_length = sum(obj.J() for obj in lengths)
            np.testing.assert_array_max_ulp(result.terms[3], native_length, maxulp=4)
            length_gradient = np.zeros_like(expected_gradient)
            for length in lengths:
                length_gradient += cast(Callable[..., Derivative], length.dJ)(partials=True)(objective)
            target = float(host_value(evaluator.problem.targets[2]))
            multiplier_roundoff = max(result.terms[3] - target, 0) - max(native_length - target, 0)
            corrected = result.gradient - multiplier_roundoff * length_gradient - radius_difference * radius_gradient
            assert_matches_native(corrected, expected_gradient, "gradient after bounded scalar round-off propagation")
        state = result.state


def test_gradient_matches_finite_differences_through_exact_solves():
    evaluator, _, _, _ = _problem(active_penalties=True)
    parameters = host_array(evaluator.initial_parameters)
    direction = parity_rng(11).standard_normal(parameters.size) * np.maximum(np.abs(parameters), 1)
    initial = evaluator.evaluate(parameters, evaluator.initial_state)
    step = 1e-6
    plus = evaluator.evaluate(parameters + step * direction, initial.state)
    minus = evaluator.evaluate(parameters - step * direction, initial.state)
    assert plus.success and minus.success, "finite-difference exact solves must converge"
    plus2 = evaluator.evaluate(parameters + 2 * step * direction, initial.state)
    minus2 = evaluator.evaluate(parameters - 2 * step * direction, initial.state)
    assert plus2.success and minus2.success
    central = (8 * (plus.value - minus.value) - (plus2.value - minus2.value)) / (12 * step)
    adjoint = initial.gradient @ direction
    assert abs(adjoint - central) <= 1e-7 * abs(central), f"adjoint {adjoint} != difference {central}"


def test_failed_newton_returns_failed_gradient_and_restores_only_state():
    evaluator, boozer, objective, _ = _problem(active_penalties=True)
    parameters = host_array(evaluator.initial_parameters)
    saved = host_array(evaluator.initial_state.x)
    direction = parity_rng(3).uniform(size=parameters.size)
    problem = replace(evaluator.problem, maxiter=explicit_device_array(1, dtype=np.float64, reference=evaluator.problem.maxiter))
    evaluator = replace(evaluator, problem=problem)
    x = parameters + 1e-4 * direction
    result = evaluator.evaluate(x, evaluator.initial_state)
    boozer.options["newton_maxiter"] = 1
    objective.x = x
    objective.J()
    expected_gradient = objective.dJ()
    assert not boozer.res["success"] and not result.success, "the real one-step cap must fail"
    assert result.value == 1e3
    assert_matches_native(result.gradient, expected_gradient, "gradient at failed iterate")
    failed_inner = np.concatenate((boozer.surface.get_dofs(), [boozer.res["iota"], boozer.res["G"]]))
    assert_matches_native(result.solved_x, failed_inner, "returned failed iterate")
    np.testing.assert_array_equal(host_array(result.state.x), saved, err_msg="warm start must be restored")
    assert not np.array_equal(host_array(result.solved_x), saved), "failed iterate must remain separately visible"


def test_state_branches_and_copies_are_independent_and_immutable():
    evaluator, _, objective, currents = _problem()
    parameters = host_array(evaluator.initial_parameters)
    seed = evaluator.initial_state
    saved = host_array(seed.x)
    cloned = copy(evaluator)
    direction = parity_rng(7).uniform(size=parameters.size)
    first = evaluator.evaluate(parameters + 1e-4 * direction, seed)
    cloned.evaluate(parameters - 1e-4 * direction, seed)
    objective.x = parameters - 1e-4 * direction
    currents[0].local_full_x = 1.01 * np.asarray(currents[0].local_full_x, dtype=np.float64)
    repeated = cloned.evaluate(parameters + 1e-4 * direction, seed)
    np.testing.assert_array_equal(repeated.gradient, first.gradient)
    np.testing.assert_array_equal(host_array(seed.x), saved)
    with pytest.raises(FrozenInstanceError):
        setattr(seed, "x", first.state.x)
    with pytest.raises(FrozenInstanceError):
        setattr(evaluator, "initial_state", first.state)


def test_new_values_and_numeric_settings_do_not_recompile():
    evaluator, _, _, _ = _problem()
    parameters = host_array(evaluator.initial_parameters)
    initial = evaluator.evaluate(parameters, evaluator.initial_state)
    direction = parity_rng(5).uniform(size=parameters.size)
    targets = evaluator.problem.targets + explicit_device_array([0.001, 0.002, -0.01], dtype=np.float64, reference=evaluator.problem.targets)
    problem = replace(
        evaluator.problem, targets=targets,
        tol=explicit_device_array(9e-14, dtype=np.float64, reference=evaluator.problem.tol),
        maxiter=explicit_device_array(41, dtype=np.float64, reference=evaluator.problem.maxiter),
    )
    changed = replace(evaluator, problem=problem)
    with jax_compilations() as compilations:
        next_result = changed.evaluate(parameters + 1e-4 * direction, initial.state)
        changed.evaluate(parameters - 1e-4 * direction, next_result.state)
    assert not compilations, "numeric settings, new free DOFs and returned state must reuse the executable"
    assert next_result.value != initial.value, "replacement targets must take effect"


def test_evaluation_makes_no_implicit_transfers(parity_lane):
    with parity_default_device(parity_lane), disallow_host_transfers():
        evaluator, _, objective, _ = _problem(active_penalties=True)
        result = evaluator.evaluate(host_array(evaluator.initial_parameters), evaluator.initial_state)
        next_result = evaluator.evaluate(evaluator.initial_parameters, result.state)
    assert {device.platform for device in next_result.state.x.devices()} == {parity_lane}
    assert_matches_native(result.value, objective.J(), "transfer-guard objective")
    assert_matches_native(result.gradient, objective.dJ(), "transfer-guard gradient")


def test_short_host_lbfgsb_matches_native_composition():
    """Three scaled L-BFGS-B iterations with all quadratic terms active.

    Native and fused optimizers use the same coordinates, targets and stops;
    endpoint equality at the default targets' round-off hinges is not claimed.
    """
    runs = []
    for fused in (False, True):
        evaluator, _, objective, _ = _problem(active_penalties=True)
        state = evaluator.initial_state
        initial_parameters = host_array(evaluator.initial_parameters)
        evaluations = []

        def fun(parameters):
            nonlocal state
            if fused:
                evaluation = evaluator.evaluate(parameters, state)
                state = evaluation.state
                value, gradient = evaluation.value, evaluation.gradient
            else:
                objective.x = parameters
                value, gradient = objective.J(), objective.dJ()
            evaluations.append((value, gradient))
            return value, gradient

        def scaled_fun(parameters):
            value, gradient = fun(initial_parameters + 1e-3 * parameters)
            return value, 1e-3 * gradient

        result = minimize(scaled_fun, np.zeros_like(initial_parameters), jac=True, method="L-BFGS-B", options={"maxiter": 3}, tol=1e-15)
        runs.append((result, evaluations))
    (native, native_evals), (fused_result, fused_evals) = runs
    assert native.nit == fused_result.nit == 3 and native.nfev == fused_result.nfev
    for index, (actual, expected) in enumerate(zip(fused_evals, native_evals)):
        assert_matches_native(actual[0], expected[0], f"L-BFGS-B evaluation {index} value", 1e-10)
        assert_matches_native(actual[1], expected[1], f"L-BFGS-B evaluation {index} gradient", 1e-10)
    assert_matches_native(fused_result.x, native.x, "L-BFGS-B iterate", 1e-10)


def test_native_bfgs_trials_match_composition_at_the_same_coil_state():
    """Native BFGS options with independent native anchors at each trial.

    Compare gradients at the SAME coil point: separate BFGS runs amplify
    round-off into different trial points, whose gradients need not agree.
    This exercises the shipped unscaled optimizer without relaxing a gate.
    """
    evaluator, boozer, objective, _ = _problem()
    state = evaluator.initial_state
    evaluations = []

    def fun(parameters):
        nonlocal state
        result = evaluator.evaluate(parameters, state)
        state = result.state
        objective.x = parameters
        expected_value, expected_gradient = objective.J(), objective.dJ()
        assert result.success and boozer.res["success"], "BFGS trial exact solves must succeed"
        assert_matches_native(result.value, expected_value, "same-coil BFGS value", 1e-10)
        assert_matches_native(result.gradient, expected_gradient, "same-coil BFGS gradient", 1e-10)
        evaluations.append(result.value)
        return result.value, result.gradient

    result = minimize(fun, host_array(evaluator.initial_parameters), jac=True, method="BFGS", options={"maxiter": 3}, tol=1e-15)
    assert result.nit == 3 and result.nfev == len(evaluations)


def test_zero_field_keeps_native_nonfinite_adjoint_error():
    evaluator, boozer, _, _ = _problem()
    parameters = host_array(evaluator.initial_parameters)
    # The free current entries can be identified independently from the
    # native graph; the first current is fixed and changed via a new snapshot.
    for coil in boozer.biotsavart.coils:
        coil.current.local_full_x = np.zeros_like(coil.current.local_full_x)
    field = JaxBiotSavart(boozer.biotsavart.coils)
    evaluator = replace(evaluator, problem=replace(evaluator.problem, extraction=field.coil_dof_extraction_spec()))
    parameters = np.asarray(field.x, dtype=np.float64)
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        evaluator.evaluate(parameters, evaluator.initial_state)


@pytest.mark.parametrize("component", [Iotas, MajorRadius], ids=["iota", "radius"])
@pytest.mark.parametrize("target", [-np.inf, np.inf], ids=["negative-infinity", "positive-infinity"])
def test_infinite_penalty_target_keeps_native_value_and_gradient(component, target):
    """Native solves finite component adjoints before multiplying by infinity."""
    evaluator, _, objective, _ = _problem(active_penalties=True)
    penalty = next(
        obj for obj in objective.ancestors
        if isinstance(obj, QuadraticPenalty) and isinstance(obj.obj, component)
    )
    penalty.cons = target
    targets = host_array(evaluator.problem.targets)
    targets[0 if component is Iotas else 1] = target
    evaluator = replace(evaluator, problem=replace(
        evaluator.problem,
        targets=explicit_device_array(targets, dtype=np.float64, reference=evaluator.problem.targets),
    ))
    with np.errstate(invalid="ignore"):
        expected_value = objective.J()
        expected_gradient = np.asarray(objective.dJ(), dtype=np.float64)
    result = evaluator.evaluate(evaluator.initial_parameters, evaluator.initial_state)
    assert result.success and result.value == expected_value == np.inf
    assert np.any(np.isinf(expected_gradient)), "native penalty must exercise infinite gradient entries"
    np.testing.assert_array_equal(np.isnan(result.gradient), np.isnan(expected_gradient))
    np.testing.assert_array_equal(np.isposinf(result.gradient), np.isposinf(expected_gradient))
    np.testing.assert_array_equal(np.isneginf(result.gradient), np.isneginf(expected_gradient))
    finite = np.isfinite(expected_gradient)
    np.testing.assert_array_equal(result.gradient[finite], expected_gradient[finite])


def test_device_result_exposes_exactly_singular_newton():
    evaluator, _, _, _ = _problem()
    # A collapsed surface produces a finite rank-deficient Newton matrix.
    seed = host_array(evaluator.initial_state.x)
    seed[:-2] = 0
    state = ExactSingleStageState(explicit_device_array(seed, dtype=np.float64, reference=evaluator.initial_state.x))
    result = exact_single_stage_evaluate(evaluator.problem, evaluator.initial_parameters, state)
    assert bool(host_value(result.singular)), "collapsed surface must expose the singular solve"
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        evaluator.evaluate(evaluator.initial_parameters, state)


def test_invalid_free_layout_and_failed_seed_are_rejected():
    evaluator, boozer, _, _ = _problem()
    with pytest.raises(ValueError, match="expected coil DOFs of shape"):
        evaluator.evaluate(evaluator.initial_parameters[:-1], evaluator.initial_state)
    boozer.res["success"] = False
    with pytest.raises(ValueError, match="initial exact BoozerSurface solve must have succeeded"):
        JaxExactSingleStage.from_boozer_surface(
            boozer, JaxBiotSavart(boozer.biotsavart.coils), [],
            iota_target=-0.406, major_radius_target=1.0, length_target=0.0,
        )


def test_radius_absolute_value_uses_native_sign_derivative_at_zero():
    # Native dmajor_radius_by_dcoeff uses np.sign(volume), including sign(0).
    values = explicit_device_array([-np.inf, -2.0, -0.0, 0.0, 2.0, np.inf, np.nan], dtype=np.float64)
    magnitude, derivative = jax.jit(jax.vmap(jax.value_and_grad(_absolute)))(values)
    np.testing.assert_array_equal(host_array(magnitude), np.abs(host_array(values)))
    np.testing.assert_array_equal(host_array(derivative), np.sign(host_array(values)))
