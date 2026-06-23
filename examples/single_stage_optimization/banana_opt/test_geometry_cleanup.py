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
    KeepoutSpec,
    ObjectiveWeights,
    OuterConstraint,
    Preconditioner,
    Solver,
    _expected_improvement,
    _gp_posterior,
    _lse_max,
    _penalty_objective,
    _penalty_value_and_grad,
    _run_chomp,
    _trf_residual_factory,
    bo_confinement_sweep,
    clean_geometry,
)


def _cylinder_sdf(r_keepout: float):
    """Signed distance to the inboard column cylinder (axis = z): positive = clear.

    The physical banana keepout obstacle (and a clean JAX-traceable analytic SDF for the
    Phase-4 tests): ``sdf(p) = sqrt(px^2 + py^2) - r_keepout``.
    """
    return lambda pts: jnp.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) - r_keepout


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
    # bo_confinement is realized by bo_confinement_sweep (a multi-eval BO loop), NOT a single
    # clean_geometry solve -- clean_geometry redirects rather than running it.
    with pytest.raises(ValueError, match="bo_confinement_sweep"):
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


def test_mgda_exact_projection_holds_budget_on_a_stiff_curve() -> None:
    # Regression for the MGDA exact-bisection field projection. On a stiff curve the OLD
    # linear-scaling projection drifted to a metre-scale unwind (gamma-displacement is
    # nonlinear in the dof-step); the exact bisection holds the budget by construction.
    # NB on stiff curves MGDA may NOT reduce curvature (documented limitation), so this
    # pins ONLY the field-budget contract -- the property the fail-open fix guarantees.
    # Reverting to the linear scaling makes this FAIL (rms >> budget).
    surface = SurfaceRZFourier(nfp=3, stellsym=True, mpol=3, ntor=2)
    rng = np.random.default_rng(0)
    sdofs = surface.get_dofs().copy()
    sdofs += rng.uniform(-0.05, 0.07, sdofs.size)
    surface.local_full_x = sdofs
    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, 128, endpoint=False), order=4, surf=surface, G=1, H=2)
    curve.local_full_x = curve.get_dofs() + rng.standard_normal(curve.get_dofs().size) * 0.3

    budget = 0.012
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 128, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))

    # A LOW cap (0.6x) + large lr makes the curvature gradient steep, so each MGDA step is
    # large and the displacement is strongly nonlinear in the step -- the regime where the
    # old linear-scaling projection overshoots the budget. The exact bisection still holds.
    result = clean_geometry(
        curve, curvature_cap_inv_m=0.6 * initial_peak, solver=Solver.mgda,
        schedule=AdamSchedule(lr=0.1, steps=300),
        constrained=ConstrainedSolve(displacement_budget_m=budget), n_quadpoints=128)

    assert result.solver is Solver.mgda
    # Load-bearing: the field budget is held by the exact bisection projection; reverting
    # to the linear scaling overshoots it here. This is the fail-open fix's contract.
    assert result.rms_displacement_m <= budget * 1.05


def test_pareto_sweep_endpoints_improve_and_each_respects_budget() -> None:
    # Phase 3: END-TO-END, a larger field budget enlarges the feasible set, so the peak
    # curvature at the LARGEST budget is <= that at the smallest (asserted on endpoints).
    # The INTERIOR is NOT guaranteed monotone -- SLSQP finds different local minima per
    # budget -- so only the endpoint trend + each point's own budget-respect are checked.
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
        assert p.feasible  # all three solves converge within budget on the benign fixture
    # endpoint trend: the largest budget reaches a peak <= the smallest's (SLSQP slack).
    assert front[-1].final_max_curvature <= front[0].final_max_curvature + 1.0e-3


def test_sobolev_preconditioner_is_actually_applied(monkeypatch) -> None:
    # Wiring guard: _run_scipy_trf must hand the SOBOLEV x_scale ARRAY to least_squares,
    # not silently fall back to 'jac'. Spy on least_squares to capture the x_scale it
    # receives (the spy still runs the real solve). A mutant that drops the
    # `if preconditioner is sobolev_h2` branch passes 'jac' (a str) -> fails the assertion.
    import scipy.optimize as _sopt
    from banana_opt import geometry_cleanup as _gc
    captured = {}
    real_ls = _sopt.least_squares

    def spy(*args, **kwargs):
        captured["x_scale"] = kwargs.get("x_scale")
        return real_ls(*args, **kwargs)

    monkeypatch.setattr(_sopt, "least_squares", spy)
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))
    clean_geometry(curve, curvature_cap_inv_m=0.6 * initial_peak, solver=Solver.scipy_trf,
                   preconditioner=Preconditioner.sobolev_h2,
                   constrained=ConstrainedSolve(trf_lambda_ladder=(1.0e4,), max_iter=4),
                   n_quadpoints=64)
    xs = captured["x_scale"]
    assert not isinstance(xs, str)  # NOT the 'jac' fallback
    np.testing.assert_allclose(np.asarray(xs), _gc._sobolev_x_scale(geom.dof_modes, 1.0))


def test_float64_guard_raises_when_x64_disabled() -> None:
    # The curvature certificate is fail-open in float32 (a coil OVER the cap can read UNDER
    # it), so clean_geometry must refuse to run without x64. Toggle x64 off (restored in
    # finally; the guard raises BEFORE any computation, so no float32 contamination) and
    # assert it fires. A mutant deleting the guard runs in float32 and would NOT raise.
    import jax as _jax
    curve = _curve()
    _jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64|x64"):
            clean_geometry(curve, curvature_cap_inv_m=50.0)
    finally:
        _jax.config.update("jax_enable_x64", True)


def test_scipy_trf_rejects_empty_lambda_ladder() -> None:
    # An empty trf_lambda_ladder previously returned result=None -> AttributeError on
    # result.success. It is invalid config; clean_geometry must reject it loudly (parity
    # with the other fail-loud guards), not crash opaquely.
    curve = _curve()
    with pytest.raises(ValueError, match="trf_lambda_ladder"):
        clean_geometry(curve, curvature_cap_inv_m=50.0, solver=Solver.scipy_trf,
                       constrained=ConstrainedSolve(trf_lambda_ladder=()))


def test_keepout_hinge_enters_objective_with_correct_form() -> None:
    # Phase 4: pin the TrajOpt keepout term by ISOLATION (same technique as the fairing
    # test): differencing keepout-on vs keepout-off at a fixed point cancels softmax+Tikhonov
    # and leaves exactly keepout_hinge*mean relu(margin - sdf)^2. A dropped term reads 0; a
    # wrong form (linear hinge, missing square, wrong margin/sign) reads the wrong magnitude.
    curve = _curve(n_quadpoints=48)
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 48, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    cap = float(jnp.max(geom.curvature(cdofs0, qpts)))
    cyl_r0 = jnp.sqrt(gamma0[:, 0] ** 2 + gamma0[:, 1] ** 2)
    spec = KeepoutSpec(_cylinder_sdf(float(jnp.min(cyl_r0))), safety_margin_m=0.05)

    rng = np.random.default_rng(11)
    cd = cdofs0 + jnp.asarray(0.02 * rng.standard_normal(cdofs0.shape))
    base = _penalty_objective(geom, gamma0, qpts, cap, 20.0, 1.0e3)
    with_ko = _penalty_objective(
        geom, gamma0, qpts, cap, 20.0, 1.0e3, keepout_hinge=4.0, keepout=spec)
    iso = float(with_ko(cd)) - float(base(cd))
    pen = jnp.maximum(0.0, spec.safety_margin_m - spec.sdf_fn(geom.gamma(cd, qpts)))
    exp = 4.0 * float(jnp.mean(pen ** 2))

    assert exp > 0.0  # the obstacle is live at this point (the test exercises the hinge)
    assert iso == pytest.approx(exp, rel=1e-6)
    assert iso > 0.0  # the hinge PENALIZES penetration (positive cost), never rewards it


def test_keepout_hinge_gradient_is_differentiable_and_enters_the_gradient() -> None:
    # Two distinct guarantees. FD-vs-analytic ALONE is insufficient for the keepout term: both
    # sides autodiff the SAME objective, so it cannot catch a dropped/sign-flipped keepout term
    # -- only autodiff PLUMBING. (Form/sign are pinned by the isolation, push-off, and trf
    # tests.) So this test additionally checks the term actually ENTERS the gradient:
    #  (a) the softmax+keepout objective (lam=0) is cleanly differentiable -- analytic grad
    #      matches a centered FD directional derivative (obstacle OUTSIDE every point => relu in
    #      its linear region at all samples, no kink-crossing => smooth FD, no NaN);
    #  (b) the keepout term CONTRIBUTES to the gradient (grad != keepout-off grad) -- this is
    #      what catches a silently-dropped keepout gradient, which (a) provably cannot.
    curve = _curve(n_quadpoints=48)
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 48, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    cap = float(jnp.max(geom.curvature(cdofs0, qpts)))
    cyl_r0 = jnp.sqrt(gamma0[:, 0] ** 2 + gamma0[:, 1] ** 2)
    spec = KeepoutSpec(_cylinder_sdf(float(jnp.max(cyl_r0)) + 0.01), safety_margin_m=0.05)

    vg = _penalty_value_and_grad(geom, gamma0, qpts, cap, 20.0, 0.0, keepout_hinge=4.0, keepout=spec)
    grad = np.asarray(vg(cdofs0)[1])
    rng = np.random.default_rng(5)
    d = rng.standard_normal(cdofs0.shape)
    d = d / np.linalg.norm(d)
    eps = 1.0e-6
    val = lambda x: float(vg(jnp.asarray(x))[0])  # noqa: E731
    fd = (val(np.asarray(cdofs0) + eps * d) - val(np.asarray(cdofs0) - eps * d)) / (2.0 * eps)
    # (a) autodiff plumbing / differentiability at the (linear-region) hinge
    assert fd == pytest.approx(float(grad @ d), rel=1e-5, abs=1e-7)
    # (b) the keepout term is really in the gradient: dropping it changes grad measurably
    grad_off = np.asarray(_penalty_value_and_grad(geom, gamma0, qpts, cap, 20.0, 0.0)(cdofs0)[1])
    assert np.linalg.norm(grad - grad_off) > 1.0e-3


def test_keepout_hinge_is_inert_when_curve_is_clear() -> None:
    # Sign + no-op guard: with the obstacle far inside (every point clear by ~1 m), the
    # squared hinge and its gradient are identically zero, so a keepout-ON adam solve must
    # reproduce the keepout-OFF solve bit-for-bit. A wrong-sign hinge (penalizing clearance)
    # or an always-on bug would move the curve -> the cleaned dofs would diverge.
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    cyl_r0 = jnp.sqrt(gamma0[:, 0] ** 2 + gamma0[:, 1] ** 2)
    initial_peak = float(jnp.max(geom.curvature(cdofs0, qpts)))
    spec = KeepoutSpec(_cylinder_sdf(float(jnp.min(cyl_r0)) - 1.0), safety_margin_m=0.05)

    sched = AdamSchedule(lambda_ladder=(1.0e3,), steps=200)
    off = clean_geometry(curve, curvature_cap_inv_m=0.6 * initial_peak, schedule=sched, n_quadpoints=64)
    on = clean_geometry(
        curve, curvature_cap_inv_m=0.6 * initial_peak, schedule=sched,
        weights=ObjectiveWeights(keepout_hinge=1.0e3), keepout=spec, n_quadpoints=64)

    np.testing.assert_allclose(on.cleaned_dofs, off.cleaned_dofs, rtol=1e-9, atol=1e-12)


def test_keepout_hinge_pushes_curve_off_obstacle() -> None:
    # Observable behavior: with the curvature term parked (cap high) and a strong keepout
    # hinge against an obstacle the seed penetrates, adam reshapes the ON-SURFACE route to
    # increase its closest approach to the column -- the curve clears, bounded by Tikhonov.
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    cyl_r0 = jnp.sqrt(gamma0[:, 0] ** 2 + gamma0[:, 1] ** 2)
    initial_peak = float(jnp.max(geom.curvature(cdofs0, qpts)))
    r_keepout = float(jnp.min(cyl_r0))  # obstacle surface AT the seed's closest approach
    margin = 0.05  # a modest clearance; fully clearing needs ~margin of outward motion
    spec = KeepoutSpec(_cylinder_sdf(r_keepout), safety_margin_m=margin)

    result = clean_geometry(
        curve, curvature_cap_inv_m=5.0 * initial_peak,  # park the curvature term
        schedule=AdamSchedule(lambda_ladder=(1.0e2,), lr=1.0e-3, steps=500),
        weights=ObjectiveWeights(keepout_hinge=1.0e3), keepout=spec, n_quadpoints=64)

    gamma1 = geom.gamma(jnp.asarray(result.cleaned_dofs), qpts)
    final_min_cyl = float(jnp.min(jnp.sqrt(gamma1[:, 0] ** 2 + gamma1[:, 1] ** 2)))
    assert final_min_cyl > r_keepout + 0.005  # closest approach increased (curve cleared)
    assert result.max_displacement_m < 0.15    # bounded reshape (~margin), not a meters-scale unwind


def test_trf_residual_with_keepout_equals_penalty_objective() -> None:
    # Extends the ||r||^2 == objective identity to the Phase-4 keepout residual: the squared
    # hinge appends sqrt(keepout_hinge/n)*relu(margin - sdf) entries. A wrong scale (missing
    # /n, missing sqrt, wrong weight) breaks the identity. Obstacle outside all points so the
    # hinge is active at every sample -- a strong check of the per-sample residual.
    curve = _curve(n_quadpoints=48)
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 48, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)
    cap = float(jnp.max(geom.curvature(cdofs0, qpts)))
    cyl_r0 = jnp.sqrt(gamma0[:, 0] ** 2 + gamma0[:, 1] ** 2)
    spec = KeepoutSpec(_cylinder_sdf(float(jnp.max(cyl_r0)) + 0.01), safety_margin_m=0.05)
    lam = 1.0e3

    rng = np.random.default_rng(13)
    cd = cdofs0 + jnp.asarray(0.02 * rng.standard_normal(cdofs0.shape))
    residual_fn, _ = _trf_residual_factory(
        geom, gamma0, qpts, cap, 20.0, lam, keepout_hinge=4.0, keepout=spec)
    rsq = float(jnp.sum(residual_fn(cd) ** 2))
    obj = float(_penalty_value_and_grad(
        geom, gamma0, qpts, cap, 20.0, lam, keepout_hinge=4.0, keepout=spec)(cd)[0])

    assert rsq == pytest.approx(obj, rel=1e-9)


def test_chomp_reduces_peak_curvature_holding_displacement() -> None:
    # Phase 4: CHOMP covariant descent (Sobolev-preconditioned GD + Armijo line search)
    # lowers the peak on the same penalty objective as adam, with the Tikhonov term holding
    # the displacement small (a reshape, not an unwind). preconditioner stays none -- chomp
    # builds its covariant metric intrinsically from ConstrainedSolve.sobolev_alpha.
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))

    result = clean_geometry(
        curve, curvature_cap_inv_m=0.6 * initial_peak, solver=Solver.chomp,
        schedule=AdamSchedule(lambda_ladder=(1.0e3,), lr=1.0e-2, steps=300), n_quadpoints=64)

    assert result.solver is Solver.chomp
    assert result.final_max_curvature < 0.99 * result.initial_max_curvature  # peak reduced
    assert result.max_displacement_m < 0.05  # bounded reshape (same Tikhonov role as adam)
    assert result.solver_success is None      # chomp runs its full schedule (no scipy cert flag)


def test_keepout_and_chomp_pairing_guards() -> None:
    # The Phase-4 surface never silently no-ops: keepout_hinge needs an obstacle AND a solver
    # whose objective includes the term; chomp builds its own covariant metric, so the
    # separable sobolev preconditioner is rejected (use ConstrainedSolve.sobolev_alpha).
    curve = _curve()
    cap = 50.0
    spec = KeepoutSpec(_cylinder_sdf(0.5), safety_margin_m=0.05)
    # keepout_hinge with no obstacle spec
    with pytest.raises(ValueError, match="KeepoutSpec"):
        clean_geometry(curve, curvature_cap_inv_m=cap, weights=ObjectiveWeights(keepout_hinge=1.0))
    # keepout_hinge with a solver whose objective excludes it (slsqp's LSE peak)
    with pytest.raises(ValueError, match="keepout"):
        clean_geometry(curve, curvature_cap_inv_m=cap, solver=Solver.slsqp,
                       formulation=Formulation.epsilon_constraint,
                       weights=ObjectiveWeights(keepout_hinge=1.0), keepout=spec)
    # chomp's covariant metric is intrinsic -> the separable sobolev preconditioner raises
    with pytest.raises(ValueError, match="sobolev_h2|chomp"):
        clean_geometry(curve, curvature_cap_inv_m=cap, solver=Solver.chomp,
                       preconditioner=Preconditioner.sobolev_h2)


def test_run_chomp_applies_covariant_metric_and_holds_finitely_on_nonfinite_gradient() -> None:
    # Two unit guarantees of the CHOMP runner, isolated from the geometry (the behavioral
    # chomp test sees neither):
    #  (1) COVARIANT step: the update is -eta*(precond*grad), NOT -eta*grad, so a downweighted
    #      high mode moves proportionally less -- this pins the defining Sobolev metric (a
    #      plain-GD mutant that drops `precond` fails the second assert).
    #  (2) FAIL-CLOSED: a non-finite gradient yields no Armijo decrease, so the step is HELD and
    #      the result stays FINITE (== seed) -- the docstring's "stalls rather than diverges". A
    #      `cdofs - 0*direction` hold would poison it to NaN (0*inf); this asserts it does not.
    precond = jnp.array([1.0, 0.25])  # mode-0 unscaled; a stiff high mode downweighted 4x
    g = jnp.array([2.0, 2.0])
    # linear objective f(x)=g.x => grad == g (constant); value_fn is the SAME f, so Armijo
    # accepts eta=lr0 on the first try (a linear objective always decreases along -direction).
    vg = lambda x: (jnp.sum(g * x), g)  # noqa: E731
    vf = lambda x: jnp.sum(g * x)       # noqa: E731
    out = np.asarray(_run_chomp(vg, vf, jnp.zeros(2), precond, lr0=1.0, steps=1))
    np.testing.assert_allclose(out, -1.0 * np.asarray(precond) * np.asarray(g))  # covariant step
    assert not np.allclose(out, -1.0 * np.asarray(g))  # and NOT the identity-metric step

    # (2) non-finite gradient -> held at the seed, finite (catches the 0*NaN poisoning mutant)
    seed = jnp.array([0.5, -0.5])
    vg_nan = lambda x: (jnp.asarray(0.0), jnp.array([jnp.nan, jnp.inf]))  # noqa: E731
    vf_nan = lambda x: jnp.asarray(jnp.nan)                              # noqa: E731
    held = np.asarray(_run_chomp(vg_nan, vf_nan, seed, precond, lr0=1.0, steps=5))
    assert np.all(np.isfinite(held))
    np.testing.assert_array_equal(held, np.asarray(seed))


def test_gp_posterior_interpolates_training_points() -> None:
    # The hand-rolled GP must interpolate (mean == y at observed x, ~zero variance there) and be
    # MORE uncertain between observations -- the regressor the constrained-EI acquisition needs.
    x = np.array([0.0, 0.3, 0.6, 1.0])
    y = np.array([1.0, -0.5, 0.2, 0.8])
    mean_at_train, std_at_train = _gp_posterior(x, y, x, lengthscale=0.3, noise_var=1e-8)
    np.testing.assert_allclose(mean_at_train, y, atol=1e-3)   # interpolates the observations
    assert np.all(std_at_train < 1e-2)                        # ~zero posterior variance at observations
    _, std_mid = _gp_posterior(x, y, np.array([0.15]), lengthscale=0.3, noise_var=1e-8)
    assert std_mid[0] > float(std_at_train.max())             # more uncertain between observed points


def test_expected_improvement_is_nonneg_and_rewards_lower_mean() -> None:
    # EI (for MINIMIZATION) must be >= 0, increase as the predicted mean drops below the incumbent,
    # and vanish at a confidently non-improving point. This is what makes the BO chase lower kappa.
    best = 0.0
    ei = _expected_improvement(np.array([-1.0, 0.0, 2.0]), np.array([0.5, 0.5, 0.5]), best)
    assert np.all(ei >= 0.0)
    assert ei[0] > ei[1] > ei[2]                              # lower mean (more improvement) -> higher EI
    ei_confident = _expected_improvement(np.array([1.0]), np.array([1e-13]), best)
    assert ei_confident[0] == pytest.approx(0.0, abs=1e-6)    # mean above best, no uncertainty -> no EI


def test_bo_confinement_sweep_navigates_to_the_confinement_cliff() -> None:
    # Phase 5 end-to-end: the constrained BO must NAVIGATE to the confinement cliff, not merely
    # filter feasibility. On this curve slsqp is stable for budgets <= 0.03 with curvature
    # decreasing monotonically, and the synthetic confinement (survival = 50 - 1000*rms) is
    # feasible only for rms <= 0.02. So the constrained optimum is the INTERIOR budget ~0.02 (the
    # largest feasible), which is NOT one of the n_init=2 endpoints [0.003, 0.03] -- the cEI
    # acquisition has to find it. Mutation-verified: a constant, EI-only, or P(feasible)-only
    # acquisition all cluster at the conservative end (~0.003) and FAIL `best_budget > 0.012`; only
    # the true EI(curvature)*P(feasible) reaches the cliff (~0.0199).
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    gamma0 = geom.gamma(jnp.asarray(curve.get_dofs()), qpts)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))

    def confinement_fn(dofs):
        g = geom.gamma(jnp.asarray(dofs), qpts)
        rms = float(jnp.sqrt(jnp.mean(jnp.sum((g - gamma0) ** 2, axis=1))))
        return 50.0 - 1000.0 * rms  # synthetic cliff: feasible (>= 30) iff rms <= 0.02

    threshold = 30.0
    result = bo_confinement_sweep(
        curve, curvature_cap_inv_m=initial_peak, confinement_fn=confinement_fn,
        budget_bounds=(0.003, 0.03), survival_threshold=threshold,
        n_init=2, n_iter=6, constrained=ConstrainedSolve(max_iter=120), n_quadpoints=64)

    assert len(result.trials) == 2 + 6
    assert all(0.003 <= t.budget_m <= 0.03 for t in result.trials)
    assert result.best_budget_m is not None
    assert result.best_survival >= threshold                 # confinement constraint respected
    # NAVIGATED to the cliff interior (a blind acquisition would sit at the conservative ~0.003):
    assert 0.012 < result.best_budget_m <= 0.021
    # and that field budget bought real curvature margin vs the most-conservative budget:
    smallest = min(result.trials, key=lambda t: t.budget_m)
    assert result.best_final_max_curvature < smallest.final_max_curvature - 0.05
    assert result.best_final_max_curvature < initial_peak


def test_bo_confinement_sweep_reports_none_when_no_budget_is_feasible() -> None:
    # When the threshold is unreachable (survival maxes at 50, threshold 100), no budget is
    # feasible: best_* are None but the full trial history is still returned (so the caller sees
    # the attempted budgets + survivals to decide next).
    curve = _curve()
    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, 64, endpoint=False)
    gamma0 = geom.gamma(jnp.asarray(curve.get_dofs()), qpts)
    initial_peak = float(jnp.max(geom.curvature(jnp.asarray(curve.get_dofs()), qpts)))

    def confinement_fn(dofs):
        g = geom.gamma(jnp.asarray(dofs), qpts)
        rms = float(jnp.sqrt(jnp.mean(jnp.sum((g - gamma0) ** 2, axis=1))))
        return 50.0 - 1000.0 * rms

    result = bo_confinement_sweep(
        curve, curvature_cap_inv_m=initial_peak, confinement_fn=confinement_fn,
        budget_bounds=(0.003, 0.03), survival_threshold=100.0,  # unreachable (max survival is 50)
        n_init=2, n_iter=3, constrained=ConstrainedSolve(max_iter=120), n_quadpoints=64)

    assert result.best_budget_m is None
    assert result.best_final_max_curvature is None
    assert result.best_survival is None
    assert len(result.trials) == 2 + 3
    assert all(not t.confinement_ok for t in result.trials)  # nothing cleared the (impossible) bar
