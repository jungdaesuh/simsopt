"""Phase-0 tests for the post-hoc banana geometry-cleanup optimizer.

These pin the *contract* of :mod:`banana_opt.geometry_cleanup`, not its source:

  * the ported JAX curvature kappa(s) matches the production C++
    ``CurveCWSFourierCPP.kappa()`` on the same points (the port is faithful, not a
    re-derivation that drifted);
  * the penalty objective's analytic JAX gradient matches a centered
    finite-difference of its value (the stiff softmax-curvature gradient is exact);
  * ``Solver.adam_baseline`` lowers the peak curvature toward the cap while the
    Tikhonov term keeps the displacement small (a reshape, not an unwind);
  * every not-yet-implemented method member fails loudly, naming its phase --
    the typed surface never silently no-ops.

Hermetic: a small synthetic ``CurveCWSFourierCPP`` on a perturbed torus (the same
fixture style as ``tests/geo/test_curvecwsfourier_surface_gradients.py``); no
campaign artifacts, no certified-coil load.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import jax

# The curvature certificate is fail-open in float32; production and the optimizer
# both run double precision. Enable it before importing the module under test.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simsopt.geo import CurveCWSFourierCPP, SurfaceRZFourier  # noqa: E402

from banana_opt.geometry_cleanup import (  # noqa: E402
    AdamSchedule,
    ConstrainedSolve,
    CwsCurveGeometry,
    Formulation,
    ObjectiveWeights,
    OuterConstraint,
    Preconditioner,
    Solver,
    _lse_max,
    _penalty_value_and_grad,
    _trf_residual_factory,
    clean_geometry,
)


def _surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier(nfp=3, stellsym=True, mpol=2, ntor=1)
    dofs = surface.get_dofs().copy()
    dofs += np.linspace(-0.03, 0.04, dofs.size)
    surface.local_full_x = dofs
    return surface


def _curve(n_quadpoints: int = 64) -> CurveCWSFourierCPP:
    surface = _surface()
    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, n_quadpoints, endpoint=False),
        order=2,
        surf=surface,
        G=1,
        H=2,
    )
    curve.local_full_x = curve.get_dofs() + np.array(
        [0.13, -0.02, 0.04, 0.01, -0.03, 0.22, 0.05, -0.06, 0.02, -0.01]
    )
    return curve


def test_jax_curvature_matches_cpp_curve_kappa() -> None:
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    cdofs = jnp.asarray(curve.get_dofs())
    qpts = jnp.asarray(curve.quadpoints)

    expected = np.asarray(curve.kappa())
    observed = np.asarray(geom.curvature(cdofs, qpts))

    np.testing.assert_allclose(observed, expected, rtol=1e-7, atol=1e-9)


def test_penalty_objective_gradient_matches_finite_difference() -> None:
    # Evaluate OFF gamma0 so the displacement gradient is live (it is identically zero
    # at cdofs0==gamma0, which blinds an FD check to the Tikhonov term). This pins
    # FD-vs-analytic self-consistency of the FULL objective gradient -- i.e. autodiff
    # plumbing. It does NOT catch a wrong objective FORM: a dropped/sign-flipped/
    # unreferenced displacement term keeps fd==analytic (both sides autodiff the same
    # objective). Form is pinned by test_penalty_displacement_penalizes_off_reference_motion
    # and the TRF identity test; the disp_grad>1e-6 guard below stops silent re-blinding.
    curve = _curve(n_quadpoints=48)
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 48, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    # cap ~ the curve's own peak keeps (kappa/cap)^p well conditioned around 1.
    cap = float(jnp.max(geom.curvature(cdofs0, qpts)))

    rng = np.random.default_rng(20260623)
    cdofs_eval = cdofs0 + jnp.asarray(0.02 * rng.standard_normal(cdofs0.shape))

    value_and_grad = _penalty_value_and_grad(geom, gamma0, qpts, cap, softmax_p=20.0, lam=1.0e3)
    _, grad = value_and_grad(cdofs_eval)

    direction = jnp.asarray(rng.standard_normal(cdofs0.shape))
    eps = 1.0e-6
    plus = float(value_and_grad(cdofs_eval + eps * direction)[0])
    minus = float(value_and_grad(cdofs_eval - eps * direction)[0])
    finite_difference = (plus - minus) / (2.0 * eps)
    analytic = float(jnp.dot(grad, direction))

    assert finite_difference == pytest.approx(analytic, rel=1e-5, abs=1e-7)
    # Guard against silently re-blinding the test: the eval point must genuinely
    # exercise the displacement gradient (nonzero), else only softmax is covered.
    disp_grad = jax.grad(
        lambda cd: jnp.mean(jnp.sum((geom.gamma(cd, qpts) - gamma0) ** 2, axis=1))
    )(cdofs_eval)
    assert float(jnp.linalg.norm(disp_grad)) > 1e-6


def test_penalty_displacement_penalizes_off_reference_motion() -> None:
    # Pin the Tikhonov term's EXACT form and POSITIVE sign by isolating it. The
    # objective is softmax + lam*mean|gamma-gamma0|^2, so differencing two lambdas at
    # the same off-reference point cancels softmax and leaves (lam_b-lam_a)*mean|...|^2.
    # That isolate must be positive and equal the displacement exactly: a dropped term
    # reads 0, a sign flip reads negative, a missing -gamma0 reference reads the wrong
    # magnitude. This is the field-preservation contract no other test constrains.
    curve = _curve(n_quadpoints=48)
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 48, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    cap = float(jnp.max(geom.curvature(cdofs0, qpts)))

    rng = np.random.default_rng(11)
    cdofs1 = cdofs0 + jnp.asarray(0.02 * rng.standard_normal(cdofs0.shape))

    vg_a = _penalty_value_and_grad(geom, gamma0, qpts, cap, softmax_p=20.0, lam=1.0e6)
    vg_b = _penalty_value_and_grad(geom, gamma0, qpts, cap, softmax_p=20.0, lam=2.0e6)
    isolated_displacement = float(vg_b(cdofs1)[0]) - float(vg_a(cdofs1)[0])
    expected = 1.0e6 * float(jnp.mean(jnp.sum((geom.gamma(cdofs1, qpts) - gamma0) ** 2, axis=1)))

    assert isolated_displacement == pytest.approx(expected, rel=1e-6)
    assert isolated_displacement > 0.0  # the term PENALIZES moving off gamma0 (sign-flip -> negative)


def test_adam_baseline_reduces_peak_curvature_holding_displacement() -> None:
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))
    # Place the cap below the seed peak so the softmax term has a live gradient.
    cap = 0.6 * initial_peak

    result = clean_geometry(
        curve,
        curvature_cap_inv_m=cap,
        schedule=AdamSchedule(lambda_ladder=(1.0e3,), steps=300),
        n_quadpoints=64,
    )

    assert result.initial_max_curvature == pytest.approx(initial_peak, rel=1e-9)
    assert result.final_max_curvature < 0.99 * result.initial_max_curvature
    # Tikhonov keeps it a local reshape (~12 mm here), not the degenerate unwind: a
    # dropped/sign-flipped/unreferenced displacement term drives it to >=0.28 m. 0.05 m
    # separates the real reshape from every such mutant.
    assert result.max_displacement_m < 0.05
    assert result.cleaned_dofs.shape == curve.get_dofs().shape
    assert result.solver is Solver.adam_baseline


def test_lse_max_brackets_true_peak_and_converges() -> None:
    # The smooth max must sit at/above the true peak (fail-closed for a kappa<=cap
    # reading) and tighten toward it as the temperature rises.
    vals = jnp.asarray([1.0, 5.0, 4.2, 4.9])
    true_max = 5.0
    loose = float(_lse_max(vals, temperature=5.0))
    tight = float(_lse_max(vals, temperature=200.0))
    assert loose >= true_max - 1e-9
    assert tight >= true_max - 1e-9
    assert (loose - true_max) > (tight - true_max)  # higher temperature -> closer to peak
    assert tight == pytest.approx(true_max, abs=1e-2)


def test_scipy_trf_reduces_peak_curvature_holding_displacement() -> None:
    # Phase 1: TRF (adaptive LM damping) reaches the curvature reduction without the
    # L-BFGS-B stall, on the same penalty objective as adam_baseline.
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))
    cap = 0.6 * initial_peak

    result = clean_geometry(
        curve, curvature_cap_inv_m=cap, solver=Solver.scipy_trf,
        constrained=ConstrainedSolve(trf_lambda_ladder=(1.0e4, 1.0e3), max_iter=80),
        n_quadpoints=64,
    )

    assert result.solver is Solver.scipy_trf
    assert result.final_max_curvature < 0.99 * result.initial_max_curvature
    # A bounded reshape, not the degenerate unwind (same Tikhonov ladder as adam).
    assert result.max_displacement_m < 0.05


def test_trf_residual_squared_norm_equals_penalty_objective() -> None:
    # scipy_trf must minimize EXACTLY the adam penalty objective: ||r||^2 ==
    # softmax(kappa/cap) + lam*mean|gamma-gamma0|^2. This couples the TRF path to the
    # same field-preserving objective and pins the displacement residual's scale, sign,
    # and -gamma0 reference -- any of which breaking the identity would be caught here.
    curve = _curve(n_quadpoints=48)
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 48, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    cap = float(jnp.max(geom.curvature(cdofs0, qpts)))
    lam = 1.0e3

    rng = np.random.default_rng(3)
    cdofs1 = cdofs0 + jnp.asarray(0.02 * rng.standard_normal(cdofs0.shape))

    residual_fn, _ = _trf_residual_factory(geom, gamma0, qpts, cap, softmax_p=20.0, lam=lam)
    residual_sq_norm = float(jnp.sum(residual_fn(cdofs1) ** 2))
    objective = float(_penalty_value_and_grad(geom, gamma0, qpts, cap, softmax_p=20.0, lam=lam)(cdofs1)[0])

    assert residual_sq_norm == pytest.approx(objective, rel=1e-9)


def test_slsqp_minimizes_curvature_within_field_budget() -> None:
    # Phase 1 epsilon-constraint: SLSQP lowers the peak curvature while honoring the
    # hard RMS-displacement budget (the field-preservation bound least_squares cannot
    # enforce). The budget is binding, so the optimizer spends it but does not exceed it.
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))
    budget = 0.1

    result = clean_geometry(
        curve, curvature_cap_inv_m=initial_peak, solver=Solver.slsqp,
        formulation=Formulation.epsilon_constraint,
        constrained=ConstrainedSolve(displacement_budget_m=budget, max_iter=150),
        n_quadpoints=64,
    )

    assert result.solver is Solver.slsqp
    assert result.final_max_curvature < result.initial_max_curvature  # peak reduced
    assert result.rms_displacement_m <= budget * 1.05  # hard field budget respected (to SLSQP tol)
    assert result.rms_displacement_m > 1.0e-4  # the optimizer actually moved the curve


def test_slsqp_scaling_holds_budget_on_a_stiff_curve() -> None:
    # Regression for the SLSQP variable-scaling fix. The benign order-2 fixture above
    # cannot exercise it -- unscaled SLSQP converges there too -- so this uses a stiff
    # order-4 CWS curve whose Fourier modes span orders of magnitude in curvature
    # sensitivity. There, an UNSCALED SLSQP overshoots into a ~1.2 m unwind that violates
    # the budget ~120x; the gradient-scaled path (the fix) binds the budget instead.
    # Pinned seed => deterministic separation. Deleting the scaling block in _run_slsqp
    # makes this FAIL (rms ~1.2 m >> budget) -- the hermetic guard the benign curve lacks.
    surface = SurfaceRZFourier(nfp=3, stellsym=True, mpol=3, ntor=2)
    rng = np.random.default_rng(0)
    sdofs = surface.get_dofs().copy()
    sdofs += rng.uniform(-0.05, 0.07, sdofs.size)
    surface.local_full_x = sdofs
    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, 128, endpoint=False), order=4, surf=surface, G=1, H=2)
    curve.local_full_x = curve.get_dofs() + rng.standard_normal(curve.get_dofs().size) * 0.3

    budget = 0.01
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 128, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))

    result = clean_geometry(
        curve, curvature_cap_inv_m=initial_peak, solver=Solver.slsqp,
        formulation=Formulation.epsilon_constraint,
        constrained=ConstrainedSolve(displacement_budget_m=budget, max_iter=200),
        n_quadpoints=128,
    )

    assert result.solver is Solver.slsqp
    # Load-bearing: the hard field budget holds even on the stiff problem (unscaled
    # SLSQP overshoots to ~1.2 m here; the scaling is what keeps it feasible).
    assert result.rms_displacement_m <= budget * 1.05
    # And the optimizer genuinely spent the budget (binding constraint), not a no-op.
    assert result.rms_displacement_m > budget * 0.5


def test_method_pairings_and_unlanded_phases_raise() -> None:
    curve = _curve()
    cap = 50.0

    # scipy_trf is penalty-only (least_squares has no nonlinear constraints).
    with pytest.raises(ValueError, match="penalty-only|least_squares"):
        clean_geometry(curve, curvature_cap_inv_m=cap, solver=Solver.scipy_trf,
                       formulation=Formulation.epsilon_constraint)
    # slsqp is the epsilon-constraint path; pairing it with penalty is rejected.
    with pytest.raises(ValueError, match="epsilon_constraint"):
        clean_geometry(curve, curvature_cap_inv_m=cap, solver=Solver.slsqp,
                       formulation=Formulation.penalty)
    # adam_baseline only supports the penalty formulation.
    with pytest.raises(ValueError, match="penalty"):
        clean_geometry(curve, curvature_cap_inv_m=cap, formulation=Formulation.epsilon_constraint)
    # sobolev_h2 (Phase 2) is landed but is scipy_trf-ONLY (Adam absorbs a diagonal
    # metric; SLSQP has no x_scale) -> pairing it with the default adam solver raises.
    with pytest.raises(ValueError, match="sobolev_h2|scipy_trf"):
        clean_geometry(curve, curvature_cap_inv_m=cap, preconditioner=Preconditioner.sobolev_h2)
    # bo_confinement (Phase 5) is the last unlanded member; it fails loudly, naming its phase.
    with pytest.raises(NotImplementedError, match="Phase 5"):
        clean_geometry(curve, curvature_cap_inv_m=cap, outer=OuterConstraint.bo_confinement)


def test_fairing_terms_enter_objective_with_correct_sign_and_magnitude() -> None:
    # Pin the Phase-2 fairing terms by isolation (same technique as the displacement
    # test): differencing two weights at a fixed point cancels softmax+Tikhonov and
    # leaves exactly the fairing term. A dropped term reads 0; a wrong form reads the
    # wrong magnitude. Both must be positive (they PENALIZE curvature / its variation).
    curve = _curve(n_quadpoints=48)
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 48, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    cap = float(jnp.max(geom.curvature(cdofs0, qpts)))
    rng = np.random.default_rng(7)
    cd = cdofs0 + jnp.asarray(0.02 * rng.standard_normal(cdofs0.shape))

    base = _penalty_value_and_grad(geom, gamma0, qpts, cap, 20.0, 1.0e3)
    # integral kappa^2
    with_ik = _penalty_value_and_grad(geom, gamma0, qpts, cap, 20.0, 1.0e3, integral_kappa_sq=2.5)
    iso_ik = float(with_ik(cd)[0]) - float(base(cd)[0])
    exp_ik = 2.5 * float(jnp.mean((geom.curvature(cd, qpts) / cap) ** 2))
    assert iso_ik == pytest.approx(exp_ik, rel=1e-6)
    assert iso_ik > 0.0
    # min-variation (arc-length curvature derivative)
    with_mv = _penalty_value_and_grad(geom, gamma0, qpts, cap, 20.0, 1.0e3, min_variation=0.3)
    iso_mv = float(with_mv(cd)[0]) - float(base(cd)[0])
    dkdq = jax.jvp(lambda q: geom.curvature(cd, q), (qpts,), (jnp.ones_like(qpts),))[1]
    speed = jnp.linalg.norm(jax.jvp(lambda q: geom.gamma(cd, q), (qpts,), (jnp.ones_like(qpts),))[1], axis=1)
    exp_mv = 0.3 * float(jnp.mean((dkdq / speed) ** 2))
    assert iso_mv == pytest.approx(exp_mv, rel=1e-6)
    assert iso_mv > 0.0


def test_sobolev_x_scale_downweights_higher_modes() -> None:
    from banana_opt.geometry_cleanup import _sobolev_x_scale
    xs = _sobolev_x_scale((0, 1, 1, 2, 2), alpha=1.0)
    assert xs[0] == pytest.approx(1.0)           # mode 0 unscaled
    assert xs[3] == pytest.approx(1.0 / 5.0)     # mode 2 -> 1/(1+4)
    assert xs[0] > xs[1] > xs[3]                 # strictly decreasing in mode (smooths high freq)


def test_scipy_trf_with_sobolev_preconditioner_reduces_curvature() -> None:
    # Phase 2: the sobolev_h2 metric is a valid scipy_trf x_scale and still reduces the
    # peak (the explicit-metric alternative to x_scale='jac').
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))
    result = clean_geometry(
        curve, curvature_cap_inv_m=0.6 * initial_peak, solver=Solver.scipy_trf,
        preconditioner=Preconditioner.sobolev_h2,
        constrained=ConstrainedSolve(trf_lambda_ladder=(1.0e4, 1.0e3), max_iter=80),
        n_quadpoints=64,
    )
    assert result.solver is Solver.scipy_trf
    assert result.final_max_curvature < 0.99 * result.initial_max_curvature
    # Parity with the sibling smoke tests (adam/plain-trf at <0.05): pin reshape-not-unwind
    # (a degenerate >=0.28 m unwind slips past <1.0). The real sobolev run is ~12 mm.
    assert result.max_displacement_m < 0.05


def test_fairing_weights_with_non_adam_solver_raise() -> None:
    # The fairing terms enter only the adam penalty objective; scipy_trf's least-squares
    # residual and slsqp's LSE objective do not include them, so nonzero fairing weights
    # with those solvers would silently no-op. clean_geometry must raise instead (mirrors
    # the sobolev_h2 guard). The default (zero) weights stay allowed on every solver.
    curve = _curve()
    cap = 50.0
    with pytest.raises(ValueError, match="fairing"):
        clean_geometry(curve, curvature_cap_inv_m=cap, solver=Solver.scipy_trf,
                       weights=ObjectiveWeights(integral_kappa_sq=1.0))
    with pytest.raises(ValueError, match="fairing"):
        clean_geometry(curve, curvature_cap_inv_m=cap, solver=Solver.slsqp,
                       formulation=Formulation.epsilon_constraint,
                       weights=ObjectiveWeights(min_variation=1.0))


def test_mgda_reduces_peak_curvature_within_field_budget() -> None:
    # Phase 3: MGDA on the two CURVATURE objectives [peak, bulk] lowers the peak via the
    # min-norm common-descent direction, while the displacement-budget projection holds
    # the field. (Pairing curvature with displacement instead stalls at the field-optimal
    # seed, where the displacement gradient is zero -- this design avoids that.)
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))
    budget = 0.05
    result = clean_geometry(
        curve, curvature_cap_inv_m=0.6 * initial_peak, solver=Solver.mgda,
        schedule=AdamSchedule(lr=1.0e-3, steps=400),
        constrained=ConstrainedSolve(displacement_budget_m=budget), n_quadpoints=64)
    assert result.solver is Solver.mgda
    assert result.final_max_curvature < 0.99 * result.initial_max_curvature  # peak reduced
    assert result.rms_displacement_m <= budget * 1.05  # field-budget projection holds


def test_pareto_sweep_frontier_is_monotone_in_budget() -> None:
    # Phase 3: the buildability frontier is monotone -- a larger field budget enlarges
    # the feasible set, so the achievable peak curvature can only fall (or hold). Each
    # point must also respect its own RMS-displacement budget.
    from banana_opt.geometry_cleanup import pareto_sweep
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))
    budgets = (0.02, 0.08, 0.2)
    front = pareto_sweep(
        curve, curvature_cap_inv_m=initial_peak, displacement_budgets=budgets,
        constrained=ConstrainedSolve(max_iter=120), n_quadpoints=64)

    assert [p.displacement_budget_m for p in front] == list(budgets)
    for p in front:
        assert p.rms_displacement_m <= p.displacement_budget_m * 1.05  # budget respected
    # the largest budget reaches the lowest peak (monotone non-increasing; SLSQP slack)
    assert front[-1].final_max_curvature <= front[0].final_max_curvature + 1.0e-3
