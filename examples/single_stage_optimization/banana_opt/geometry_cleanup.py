"""Post-hoc geometry cleanup for an already-certified banana CWS coil.

Improve a certified coil's *buildability* margins (peak curvature first; later
keepout, curve-surface clearance, torsion, length) by reshaping the coil's
``CurveCWSFourierCPP`` centerline ON its winding surface while holding the field
-- the certified confinement is preserved, not re-optimized. The reshape lives
entirely in the curve's CWS dofs (the surface/field is frozen), so every step
stays on-surface by construction.

This is the committed, version-controlled generalization of the one-off
``check19``/``check20`` reshape (softmax-curvature + Tikhonov displacement
penalty + hand-rolled Adam over a descending lambda-ladder). It exposes a small
*typed, composable* method-selection surface -- ``Solver``/``Formulation``/
``Preconditioner``/``OuterConstraint`` -- so a researcher can compare optimization
families on the same problem. That surface is externally-owned behavior config
(the researcher is the decision owner comparing methods), not an undecided
internal knob: Phase 0 wires exactly one default, ``Solver.adam_baseline``, and
every not-yet-implemented member fails loudly with the phase that lands it.

Phase 0: the ``adam_baseline`` solver, the penalty formulation, and the JAX
curvature geometry -- a numerically faithful port of ``check19`` (no import of the
gitignored campaign script; validated to reproduce its result). Phase 1 ADDS the
principled constrained core alongside it: ``Solver.scipy_trf`` (penalty form solved
by trust-region least-squares = adaptive Levenberg-Marquardt damping, which cures
the L-BFGS-B stall / GD divergence) and ``Solver.slsqp`` with
``Formulation.epsilon_constraint`` (minimize peak curvature subject to a hard
field-preservation displacement budget -- the request-#2 "field budget" lever).
Phase 2 ADDS fairing terms (``ObjectiveWeights.integral_kappa_sq``/``min_variation``)
and the ``Preconditioner.sobolev_h2`` (first-order/H^1) mode metric; Phase 3 ADDS
``Solver.mgda`` (experimental) and ``pareto_sweep``. Phase 4 ADDS a TrajOpt-inspired SDF
keepout hinge (``ObjectiveWeights.keepout_hinge`` + a caller-supplied ``KeepoutSpec``) and
``Solver.chomp`` (CHOMP covariant descent). Phase 5 ADDS ``bo_confinement_sweep`` -- confinement
as a first-class black-box constraint via a hand-rolled GP Bayesian optimization (no BoTorch).
Nothing here replaces the baseline.

Output hygiene (measuring the reshaped field, writing provenance instead of a
fabricated ``results.json``) is owned by :mod:`banana_opt.reshape_run_dir`; this
module returns the cleaned dofs + an opt-grid convergence summary and leaves the
authoritative high-resolution verdict to that module.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import scipy.optimize
import scipy.special

import jax
import jax.numpy as jnp

from simsopt.geo.curvecwsfourier import gamma_curve_on_surfrz_surface


class Solver(Enum):
    """Which optimization engine reshapes the curve dofs.

    ``adam_baseline`` (Phase 0): softmax-curvature + Tikhonov penalty minimized by
    a hand-rolled Adam warm-started down a lambda-ladder (the check19 recipe).
    The remaining members are the roadmap; each raises until its phase lands it.
    """

    adam_baseline = "adam_baseline"
    scipy_trf = "scipy_trf"  # Phase 1: Levenberg-Marquardt damping (penalty form)
    slsqp = "slsqp"  # Phase 1: the hard epsilon-constraint path
    mgda = "mgda"  # Phase 3: gradient-native multi-objective [peak,bulk] curvature; EXPERIMENTAL
    # (holds the field budget exactly, but does not reduce curvature on stiff coils -- see _run_mgda)
    chomp = "chomp"  # Phase 4: CHOMP covariant gradient descent (penalty form), the Sobolev
    # smoothness metric as an intrinsic preconditioner + Armijo line search for a descent guarantee


class Formulation(Enum):
    """How the buildability caps enter the problem.

    ``penalty`` (Phase 0): the displacement cap is a soft Tikhonov term; the
    curvature cap is a softmax surrogate. ``epsilon_constraint`` (Phase 1): the
    physical caps become hard nonlinear inequalities (kappa(s) <= cap for all s).
    """

    penalty = "penalty"
    epsilon_constraint = "epsilon_constraint"


class Preconditioner(Enum):
    """Metric used to precondition the stiff Fourier modes.

    ``none`` (Phase 0): identity (raw Adam). ``sobolev_h2`` (Phase 2): a first-order
    (H^1) Sobolev metric ``1/(1+alpha*k^2)`` on the Fourier coefficients to escape
    Adam/GD plateaus (the ``_h2`` suffix is legacy; the realized symbol is H^1, ``1+k^2``).
    """

    none = "none"
    sobolev_h2 = "sobolev_h2"


class OuterConstraint(Enum):
    """How confinement (the Poincare survival count) is kept during cleanup.

    ``tikhonov_proxy`` (Phase 0): confinement is preserved *implicitly* -- the
    displacement penalty keeps gamma near gamma0 so the field barely moves; the
    caller re-traces Poincare after to confirm. ``bo_confinement`` (Phase 5):
    confinement becomes a first-class expensive black-box constraint -- realized by
    :func:`bo_confinement_sweep` (a hand-rolled GP Bayesian-optimization loop, NO BoTorch),
    NOT a single ``clean_geometry`` solve (so ``clean_geometry(outer=bo_confinement)`` redirects).
    """

    tikhonov_proxy = "tikhonov_proxy"
    bo_confinement = "bo_confinement"


@dataclass(frozen=True)
class ObjectiveWeights:
    """Exponents/weights of the geometry-cleanup objective terms.

    ``curvature_softmax_p`` is the L^p exponent of the smooth max approximating
    max_s kappa(s); check19 used 20 (higher tracks the peak more tightly but
    stiffens the gradient). The softmax owns the kappa_MAX wall (the buildability
    cap is on the peak). The Phase-2 fairing terms add AVERAGE smoothness on top,
    default 0 so ``adam_baseline`` is byte-unchanged:
      * ``integral_kappa_sq`` weights mean (kappa/cap)^2 -- the elastica/bending
        energy (Horn 1983): pulls down bulk curvature, not just the peak.
      * ``min_variation`` weights mean (dkappa/ds)^2 -- minimum-variation fairing
        (Moreton-Sequin): penalizes curvature wiggle for a smoother route. It is a
        dimensional weight (dkappa/ds has units 1/m^2); the researcher tunes it.
      * ``keepout_hinge`` weights mean relu(margin - sdf)^2 -- a TrajOpt-INSPIRED SDF keepout
        hinge (cf. Schulman 2014, whose published collision cost is the LINEAR hinge; this
        SQUARES it for a C^1 penalty, matching the repo's squared house keepout convention in
        ``hardware_keepout.py``). It pushes the curve off an obstacle when clearance drops below
        ``margin``, needs a ``KeepoutSpec`` (the obstacle SDF + margin) passed to
        ``clean_geometry``, and is honored by ``adam_baseline``/``chomp`` (penalty objective)
        and ``scipy_trf`` (least-squares residual).
    All are externally-owned research config (off by default, on to trade a little field
    budget for a smoother or better-cleared coil); the softmax is what guarantees kappa<=cap.
    """

    curvature_softmax_p: float = 20.0
    integral_kappa_sq: float = 0.0
    min_variation: float = 0.0
    keepout_hinge: float = 0.0


@dataclass(frozen=True)
class KeepoutSpec:
    """A keepout obstacle for the Phase-4 TrajOpt-inspired SDF-hinge cost.

    ``sdf_fn`` maps centerline points ``(N,3)`` -> signed distances ``(N,)`` in metres,
    POSITIVE = clear of the obstacle. It is the CALLER's frozen geometry knowledge (the
    cleanup module owns the descent, not the obstacle). It must be JAX-traceable (built from
    ``jax.numpy``) so the hinge gradient flows back to the curve dofs. ``safety_margin_m`` is
    the clearance the hinge enforces: it penalizes ``sdf < safety_margin_m``.

    NB the PRODUCTION banana keepout is NOT a simple analytic shape: it is the SWEPT Type-KK
    channel envelope vs a point-cloud/SDF of discrete hardware (shells, sensors, solenoid,
    REMC, limiter, quartz spools) -- see ``hardware_keepout.py``. A real cleanup must feed a
    JAX-traceable adapter of THAT swept-envelope SDF and account for the ~24.5 mm channel
    half-diagonal (the hinge measures the bare centerline ``gamma``). A closed-form SDF (a
    cylinder/sphere) is only an ILLUSTRATIVE obstacle for unit-testing the machinery.
    """

    sdf_fn: Callable[[jnp.ndarray], jnp.ndarray]
    safety_margin_m: float


@dataclass(frozen=True)
class AdamSchedule:
    """Hand-rolled Adam hyperparameters + the descending Tikhonov lambda-ladder.

    Each rung minimizes ``softmax(kappa/cap) + lambda*mean|gamma-gamma0|^2`` and
    warm-starts the next (smaller) lambda from its result; large lambda keeps the
    optimum near gamma0 (small, field-preserving displacement). Defaults reproduce
    the check19/check20 ``kappa ~ 37.2`` reshape. ``steps`` is per rung.
    """

    lambda_ladder: tuple[float, ...] = (3e6, 1e6, 3e5, 1e5, 3e4, 1e4)
    lr: float = 1e-4
    steps: int = 2000
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass(frozen=True)
class ConstrainedSolve:
    """Settings for the Phase-1 scipy constrained / least-squares solvers.

    ``displacement_budget_m`` is the field-preservation ε-bound: the SLSQP
    ε-constraint path minimizes peak curvature subject to
    ``sqrt(mean|gamma-gamma0|^2) <= displacement_budget_m`` (the "trust radius"
    that keeps the held field fixed -- this, not the LM damping, is what preserves
    confinement; the two roles are deliberately distinct). It is the request-#2
    "field budget" knob and is externally-owned research config. ``lse_temperature``
    is the log-sum-exp smooth-max sharpness on kappa(s) (higher -> closer to the
    true peak, stiffer gradient). ``trf_lambda_ladder`` is the penalty-form
    continuation for ``scipy_trf`` (same role as the Adam ladder). The remaining
    fields pass through to scipy.
    """

    displacement_budget_m: float = 0.012
    lse_temperature: float = 80.0
    trf_lambda_ladder: tuple[float, ...] = (3e6, 1e6, 3e5, 1e5, 3e4, 1e4)
    curvature_softmax_p: float = 20.0
    sobolev_alpha: float = 1.0  # H^1 Sobolev metric strength (1/(1+alpha*k^2)); used by BOTH
    # Preconditioner.sobolev_h2 (scipy_trf x_scale) AND Solver.chomp (its intrinsic covariant metric)
    max_iter: int = 200
    # 1e-8 is a sound solver tolerance; 1e-10 is needlessly tight. The curvature verdict
    # is re-measured at 4096 pts downstream, so the solver's own tol never gates the
    # certificate. NB on the stiff real coil the constrained solvers typically exhaust
    # max_iter at a good feasible point (budget binds, under cap) rather than certifying
    # to tol -- see CleanupResult.solver_success for how to read that.
    tol: float = 1e-8


@dataclass(frozen=True)
class CwsCurveGeometry:
    """JAX-differentiable map: curve dofs -> 3D centerline on a frozen CWS surface.

    Carries the curve ``order``, the secular winding terms ``G``/``H`` (which
    ``CurveCWSFourierCPP.get_dofs()`` does NOT round-trip and so must be held
    explicitly), and the FROZEN surface dofs + topology (mpol/ntor/nfp/stellsym).
    The surface is the held field's winding torus; cleanup moves only ``cdofs``.
    """

    order: int
    secular_g: int
    secular_h: int
    surf_dofs: jnp.ndarray
    mpol: int
    ntor: int
    nfp: int
    stellsym: bool
    dof_modes: tuple[int, ...]  # Fourier mode |k| per curve dof (aligned to get_dofs()), for the Sobolev metric

    @classmethod
    def from_curve(cls, curve) -> "CwsCurveGeometry":
        """Extract the frozen geometry from a live ``CurveCWSFourierCPP``."""
        surf = curve.surf
        # dof names are 'phic(k)'/'phis(k)'/'thetac(k)'/'thetas(k)'; the integer is the
        # Fourier mode. local_full_dof_names is ALL dof names, matching get_dofs()'s full
        # mode vector (np.concatenate(self.modes)); they stay aligned even if a curve dof is
        # later fixed -- get_dofs() returns all modes regardless of free status -- so
        # local_dof_names (free-only) would be the misaligned choice here.
        dof_modes = tuple(int(n[n.index("(") + 1:n.index(")")]) for n in curve.local_full_dof_names)
        return cls(
            order=curve.order,
            secular_g=curve.G,
            secular_h=curve.H,
            surf_dofs=jnp.asarray(surf.get_dofs()),
            mpol=surf.mpol,
            ntor=surf.ntor,
            nfp=surf.nfp,
            stellsym=surf.stellsym,
            dof_modes=dof_modes,
        )

    def gamma(self, cdofs: jnp.ndarray, qpts: jnp.ndarray) -> jnp.ndarray:
        """3D positions (N,3) at curve parameters ``qpts`` in [0,1)."""
        return gamma_curve_on_surfrz_surface(
            cdofs, qpts, self.order, self.secular_g, self.secular_h,
            self.surf_dofs, self.mpol, self.ntor, self.nfp, self.stellsym,
        )

    def curvature(self, cdofs: jnp.ndarray, qpts: jnp.ndarray) -> jnp.ndarray:
        """Scalar curvature kappa(s) = |gamma' x gamma''| / |gamma'|^3 at ``qpts``.

        Derivatives are taken w.r.t. the curve parameter via nested JVPs; gamma is
        applied elementwise across ``qpts`` so the ones-tangent JVP yields the
        per-point tangent. kappa is reparametrization-invariant, so the [0,1)
        parameter scaling cancels.
        """
        def gammadash(q: jnp.ndarray) -> jnp.ndarray:
            return jax.jvp(lambda x: self.gamma(cdofs, x), (q,), (jnp.ones_like(q),))[1]

        tangent = gammadash(qpts)
        accel = jax.jvp(gammadash, (qpts,), (jnp.ones_like(qpts),))[1]
        cross = jnp.cross(tangent, accel)
        return jnp.linalg.norm(cross, axis=1) / (jnp.linalg.norm(tangent, axis=1) ** 3)


@dataclass(frozen=True)
class CleanupResult:
    """Outcome of a geometry-cleanup solve, measured on the optimization grid.

    ``cleaned_dofs`` is the reshaped curve dofs (same layout as
    ``curve.get_dofs()``; rebuild via ``set_dofs``). The curvature/displacement
    figures are at ``n_quadpoints`` -- a convergence summary matching the solver's
    own grid (= check19's reported numbers), NOT the certified verdict. The
    authoritative high-resolution kappa and the held-field Poincare recheck are
    produced downstream by :mod:`banana_opt.reshape_run_dir`.

    There is deliberately NO ``under_cap`` boolean. The optimizer only sees the
    ``n_quadpoints`` samples and drives THAT sampled peak under cap, while the sharp
    U-turn between samples can stay over cap -- so a ``final_max_curvature <= cap``
    read here is non-conservative and can fail OPEN (reproduced: opt-grid PASS while
    the 4096-point measurement was 48x over cap). The buildability verdict is ONLY
    ``reshape_run_dir``'s 4096-point ``_max_kappa_highres``; a caller that wants the
    opt-grid value compares ``final_max_curvature`` explicitly, eyes open.
    """

    cleaned_dofs: np.ndarray
    initial_max_curvature: float
    final_max_curvature: float
    max_displacement_m: float
    rms_displacement_m: float  # sqrt(mean|gamma-gamma0|^2); the field-budget metric slsqp constrains
    curvature_cap_inv_m: float
    n_quadpoints: int
    solver: Solver
    solver_success: bool | None = None  # scipy's strict convergence-certification flag
    # (None for adam, which always runs its full schedule). On the stiff real coil the
    # scipy solvers often exhaust max_iter at a GOOD feasible point (budget binds, under
    # cap), so False means "not certified to tol", NOT "unusable" -- judge quality by the
    # curvature/displacement metrics above and the 4096-pt reshape_run_dir verdict.


def _penalty_objective(geom: CwsCurveGeometry, gamma0: jnp.ndarray,
                       qpts: jnp.ndarray, cap: float, softmax_p: float, lam: float,
                       integral_kappa_sq: float = 0.0, min_variation: float = 0.0,
                       keepout_hinge: float = 0.0, keepout: "KeepoutSpec | None" = None):
    """The penalty objective closure (un-jitted) -- the SSOT penalty form.

    ``softmax(kappa/cap) + lam*mean|gamma-gamma0|^2`` (+ optional Phase-2 fairing
    ``integral_kappa_sq*mean(kappa/cap)^2 + min_variation*mean(dkappa/ds)^2``; + optional
    Phase-4 keepout ``keepout_hinge*mean relu(margin - sdf(gamma))^2``). Weights are Python
    floats baked at trace time, so each ``if`` guard is compile-time -- zero cost (and exact
    byte-equality with the baseline) when off. ``gamma0`` is the original centerline on the
    same ``qpts``. Returned un-jitted so a caller can build value_and_grad (the descent) and
    a value-only jit (chomp's line search) from the SAME objective without duplicating it.
    """
    def objective(cdofs: jnp.ndarray) -> jnp.ndarray:
        kappa = geom.curvature(cdofs, qpts)
        gamma = geom.gamma(cdofs, qpts)
        softmax_kappa = jnp.mean((kappa / cap) ** softmax_p) ** (1.0 / softmax_p)
        displacement = jnp.mean(jnp.sum((gamma - gamma0) ** 2, axis=1))
        value = softmax_kappa + lam * displacement
        if integral_kappa_sq:
            value = value + integral_kappa_sq * jnp.mean((kappa / cap) ** 2)
        if min_variation:
            # dkappa/ds = (dkappa/dq) / |gamma'|  (arc-length derivative; q in [0,1))
            dkappa_dq = jax.jvp(lambda q: geom.curvature(cdofs, q), (qpts,), (jnp.ones_like(qpts),))[1]
            speed = jnp.linalg.norm(
                jax.jvp(lambda q: geom.gamma(cdofs, q), (qpts,), (jnp.ones_like(qpts),))[1], axis=1)
            value = value + min_variation * jnp.mean((dkappa_dq / speed) ** 2)
        if keepout_hinge:
            # TrajOpt hinge: penalize where the signed clearance drops below the margin.
            penetration = jnp.maximum(0.0, keepout.safety_margin_m - keepout.sdf_fn(gamma))
            value = value + keepout_hinge * jnp.mean(penetration ** 2)
        return value

    return objective


def _penalty_value_and_grad(geom: CwsCurveGeometry, gamma0: jnp.ndarray,
                            qpts: jnp.ndarray, cap: float, softmax_p: float,
                            lam: float, integral_kappa_sq: float = 0.0,
                            min_variation: float = 0.0, keepout_hinge: float = 0.0,
                            keepout: "KeepoutSpec | None" = None):
    """Jitted ``value_and_grad`` of :func:`_penalty_objective` (one closure per lambda rung)."""
    return jax.jit(jax.value_and_grad(_penalty_objective(
        geom, gamma0, qpts, cap, softmax_p, lam,
        integral_kappa_sq, min_variation, keepout_hinge, keepout)))


def _run_adam(value_and_grad, cdofs0: jnp.ndarray, schedule: AdamSchedule) -> jnp.ndarray:
    """Hand-rolled Adam for one lambda rung (check19's recipe, bias-corrected)."""
    cdofs = cdofs0
    first_moment = jnp.zeros_like(cdofs)
    second_moment = jnp.zeros_like(cdofs)
    b1, b2, eps = schedule.beta1, schedule.beta2, schedule.eps
    for step in range(1, schedule.steps + 1):
        _, grad = value_and_grad(cdofs)
        first_moment = b1 * first_moment + (1 - b1) * grad
        second_moment = b2 * second_moment + (1 - b2) * grad * grad
        m_hat = first_moment / (1 - b1 ** step)
        v_hat = second_moment / (1 - b2 ** step)
        cdofs = cdofs - schedule.lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return cdofs


def _lse_max(values: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """Numerically-stable log-sum-exp smooth max of ``values``.

    Differentiable surrogate for the non-smooth peak curvature used by the
    constrained solvers. Equals ``max`` plus a bias ``<= log(N)/temperature`` (i.e.
    it sits slightly ABOVE the true peak -- conservative/fail-closed for a
    ``kappa <= cap`` reading) and converges to the exact peak as temperature grows.
    The authoritative peak remains the downstream high-resolution kappa in
    :mod:`banana_opt.reshape_run_dir`, never this in-loop proxy.
    """
    m = jax.lax.stop_gradient(jnp.max(values))
    return m + jnp.log(jnp.sum(jnp.exp(temperature * (values - m)))) / temperature


def _trf_residual_factory(geom: CwsCurveGeometry, gamma0: jnp.ndarray, qpts: jnp.ndarray,
                          cap: float, softmax_p: float, lam: float,
                          keepout_hinge: float = 0.0, keepout: "KeepoutSpec | None" = None):
    """Jitted residual ``r`` (and Jacobian) with ``||r||^2 == softmax(kappa/cap) + lam*mean|gamma-gamma0|^2``
    (+ optional Phase-4 keepout ``keepout_hinge*mean relu(margin - sdf)^2``).

    The penalty form for ``Solver.scipy_trf``: exactly the objective ``adam_baseline``
    minimizes, re-expressed as a least-squares residual so ``least_squares``' TRF
    (trust-region reflective = adaptive Levenberg-Marquardt) damping controls the
    step. One scalar curvature residual + 3N displacement residuals scaled so the
    summed squares reproduce the objective exactly (scipy minimizes 0.5||r||^2 -- the
    constant half does not move the minimizer). ``lam`` baked per rung (the ladder).
    The squared keepout hinge is itself a sum of squares, so it appends N residual entries
    ``sqrt(keepout_hinge/n)*relu(margin - sdf(gamma))`` that keep the ||r||^2 == objective
    identity; compile-time off when ``keepout_hinge == 0`` (baseline residual byte-unchanged).
    NB this keepout residual is C^0 (kinked) at the active-set boundary ``sdf == margin``, so
    TRF's Gauss-Newton model is only locally valid across it (as is the pre-existing
    ``sqrt(softmax)`` at 0); TRF tolerates this, but the first-order ``adam_baseline``/``chomp``
    solvers are preferable for keepout-dominated solves.
    """
    n = qpts.shape[0]

    def residual(cdofs: jnp.ndarray) -> jnp.ndarray:
        gamma = geom.gamma(cdofs, qpts)
        softmax = jnp.mean((geom.curvature(cdofs, qpts) / cap) ** softmax_p) ** (1.0 / softmax_p)
        disp = (gamma - gamma0).reshape(-1)
        parts = [jnp.sqrt(softmax)[None], jnp.sqrt(lam / n) * disp]
        if keepout_hinge:
            penetration = jnp.maximum(0.0, keepout.safety_margin_m - keepout.sdf_fn(gamma))
            parts.append(jnp.sqrt(keepout_hinge / n) * penetration)
        return jnp.concatenate(parts)

    return jax.jit(residual), jax.jit(jax.jacfwd(residual))


def _sobolev_x_scale(dof_modes: tuple[int, ...], alpha: float) -> np.ndarray:
    """``least_squares`` x_scale implementing a first-order (H^1) Sobolev metric on the Fourier modes.

    x_scale[i] = 1/(1 + alpha*k_i^2): the H^1/velocity metric symbol is ``1+k^2`` (it penalizes
    ``int|gamma'|^2``), so higher modes get a SMALLER characteristic scale and the trust region
    steps them less -> a smooth (Sobolev-gradient) descent that damps high-frequency wiggle. (NB
    the ``sobolev_h2``/``_h2`` names are legacy; the realized metric is H^1, symbol ``1+k^2`` -- a
    true H^2 metric would be ``1/(1+alpha*k^4)``; the H^1 form is the validated one.) This is the
    explicit-metric alternative to the data-driven ``x_scale='jac'``. A diagonal metric like this
    is *absorbed* by Adam's per-coordinate scale-invariance, which is why ``sobolev_h2`` is a
    scipy_trf-only preconditioner (``Solver.chomp`` instead applies it as an intrinsic covariant
    GD metric) -- see the clean_geometry guard.
    """
    k = np.asarray(dof_modes, dtype=float)
    return 1.0 / (1.0 + alpha * k ** 2)


def _run_scipy_trf(geom: CwsCurveGeometry, gamma0: jnp.ndarray, qpts: jnp.ndarray,
                   cap: float, cdofs0: jnp.ndarray, settings: ConstrainedSolve,
                   preconditioner: "Preconditioner | None" = None,
                   keepout_hinge: float = 0.0, keepout: "KeepoutSpec | None" = None):
    """Penalty least-squares (TRF) warm-started down the lambda ladder.

    Returns ``(cleaned_dofs, last_scipy_result)``. TRF's adaptive LM damping is the
    principled step-control that cures the L-BFGS-B stall and GD divergence; the
    displacement residual (NOT the damping) is what holds the field. ``preconditioner``
    selects the variable scaling: ``none`` -> ``x_scale='jac'`` (data-driven, validated
    on the real coil); ``sobolev_h2`` -> the explicit Sobolev mode metric.
    """
    if not settings.trf_lambda_ladder:
        raise ValueError(
            "scipy_trf requires a non-empty ConstrainedSolve.trf_lambda_ladder "
            "(an empty ladder leaves the result undefined).")
    if preconditioner is Preconditioner.sobolev_h2:
        # Jacobian scaling matters: the Fourier modes differ in curvature sensitivity by
        # orders of magnitude; the explicit Sobolev metric down-scales high modes.
        x_scale = _sobolev_x_scale(geom.dof_modes, settings.sobolev_alpha)
    else:
        x_scale = "jac"  # data-driven scaling (validated under cap at ~4 mm on the real coil)
    cdofs = np.asarray(cdofs0)
    result = None
    for lam in settings.trf_lambda_ladder:
        res_fn, jac_fn = _trf_residual_factory(
            geom, gamma0, qpts, cap, settings.curvature_softmax_p, lam,
            keepout_hinge, keepout)
        result = scipy.optimize.least_squares(
            lambda x: np.asarray(res_fn(jnp.asarray(x))), cdofs,
            jac=lambda x: np.asarray(jac_fn(jnp.asarray(x))),
            method="trf", max_nfev=settings.max_iter, x_scale=x_scale,
            ftol=settings.tol, gtol=settings.tol, xtol=settings.tol)
        cdofs = result.x
    return jnp.asarray(cdofs), result


def _run_slsqp(geom: CwsCurveGeometry, gamma0: jnp.ndarray, qpts: jnp.ndarray,
               cap: float, cdofs0: jnp.ndarray, settings: ConstrainedSolve):
    """epsilon-constraint SLSQP: minimize the LSE peak curvature subject to a hard
    RMS-displacement budget (the field-preservation epsilon-bound).

    SLSQP -- unlike ``least_squares`` -- enforces the nonlinear inequality, so the
    field budget is a true hard cap, not a penalty. Minimizing the peak drives the
    curvature margin. Returns ``(cleaned_dofs, scipy_result)``.

    SLSQP has no ``x_scale``, and the Fourier modes differ in curvature sensitivity by
    orders of magnitude, so a cold UNSCALED run overshoots the stiff directions into a
    field-destroying step (validated: a metre-scale unwind on the certified coil). We
    therefore optimize in scaled variables ``z = x / scale`` with
    ``scale_i = 1/|d objective/d x_i|`` at the seed, floored by a RELATIVE
    ``seed_grad.max()*1e-3`` that caps the max/min scale ratio near 1e3 so an
    insensitive dof cannot get a runaway scale -- the SLSQP analogue of trf's
    ``x_scale='jac'``. The chain rule carries the scale into both gradients
    (``d/dz = scale * d/dx``).
    """
    budget_sq = settings.displacement_budget_m ** 2

    def objective(cdofs: jnp.ndarray) -> jnp.ndarray:
        return _lse_max(geom.curvature(cdofs, qpts), settings.lse_temperature) / cap

    def mean_sq_disp(cdofs: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.sum((geom.gamma(cdofs, qpts) - gamma0) ** 2, axis=1))

    obj_vg = jax.jit(jax.value_and_grad(objective))
    con_vg = jax.jit(jax.value_and_grad(mean_sq_disp))

    # Per-dof scale 1/|grad_i| from the seed objective gradient, floored by a RELATIVE
    # seed_grad.max()*1e-3 that caps the max/min scale ratio near 1e3 (a dof the peak
    # curvature barely depends on cannot get a runaway scale); the 1e-12 only guards an
    # all-zero gradient. x = scale * z.
    seed_grad = np.abs(np.asarray(obj_vg(cdofs0)[1]))
    scale = 1.0 / (seed_grad + seed_grad.max() * 1.0e-3 + 1.0e-12)
    z0 = np.asarray(cdofs0) / scale

    constraints = [{
        "type": "ineq",  # feasible when >= 0, i.e. mean|gamma-gamma0|^2 <= budget^2
        "fun": lambda z: float(budget_sq - con_vg(jnp.asarray(z * scale))[0]),
        "jac": lambda z: -np.asarray(con_vg(jnp.asarray(z * scale))[1]) * scale,
    }]
    result = scipy.optimize.minimize(
        lambda z: float(obj_vg(jnp.asarray(z * scale))[0]), z0,
        jac=lambda z: np.asarray(obj_vg(jnp.asarray(z * scale))[1]) * scale,
        method="SLSQP", constraints=constraints,
        options={"maxiter": settings.max_iter, "ftol": settings.tol})
    return jnp.asarray(result.x * scale), result


def _run_mgda(geom: CwsCurveGeometry, gamma0: jnp.ndarray, qpts: jnp.ndarray,
              cap: float, softmax_p: float, cdofs0: jnp.ndarray, lr: float, steps: int,
              budget: float) -> jnp.ndarray:
    """Multiple-Gradient Descent (Desideri 2012) on the two CURVATURE objectives
    [peak softmax(kappa/cap), bulk mean(kappa/cap)^2], field preserved by projection.

    Each step moves along the MIN-NORM convex combination of the two gradients -- the
    common-descent direction that lowers BOTH peak and bulk curvature (closed form
    alpha = clip(<g2, g2-g1>/|g1-g2|^2, 0, 1)) -- then PROJECTS back onto the field
    budget (EXACT bisection on the step scale so rms|gamma-gamma0| <= budget). Both objectives
    are curvature (nonzero gradient at the high-kappa seed), so the common direction is
    nonzero and progress is real. NB pairing curvature with the DISPLACEMENT objective
    instead is degenerate: displacement is at its global min (0) at the field-optimal
    seed, so its gradient vanishes there, the min-norm direction collapses to 0, and
    MGDA stalls -- hence the field is a projection (a constraint), not an MGDA objective.

    LIMITATION (real-coil, validated 2026-06-23): the EXACT projection holds the budget,
    but on the STIFF certified coil this min-norm-descent + radial-projection scheme does
    NOT reduce curvature (kappa rises 56->86+ at every lr tried) -- the penalty (adam /
    scipy_trf) and hard-constraint (slsqp) solvers do. MGDA is an OPTIONAL research lever
    for the peak-vs-bulk-curvature tradeoff on well-conditioned problems; a caller must
    check final_max_curvature < initial_max_curvature before trusting its result (the
    result always holds the field budget, but may be WORSE than the seed on stiff coils).
    """
    def f_peak(cd: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((geom.curvature(cd, qpts) / cap) ** softmax_p) ** (1.0 / softmax_p)

    def f_bulk(cd: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((geom.curvature(cd, qpts) / cap) ** 2)

    grad_peak = jax.jit(jax.grad(f_peak))
    grad_bulk = jax.jit(jax.grad(f_bulk))

    @jax.jit
    def project(cdofs: jnp.ndarray) -> jnp.ndarray:
        # EXACT projection onto rms|gamma-gamma0| <= budget along the step cdofs0->cdofs:
        # rms is monotone in the step scale t in [0,1], so a fixed bisection finds the
        # largest feasible t (fail-CLOSED: returns lo, the last t known <= budget). This
        # replaces a linear-scaling approximation that drifts to a metre-scale unwind on
        # the stiff real coil because the gamma-displacement is NONLINEAR in the dof-step.
        delta = cdofs - cdofs0

        def rms_at(t: jnp.ndarray) -> jnp.ndarray:
            g = geom.gamma(cdofs0 + t * delta, qpts)
            return jnp.sqrt(jnp.mean(jnp.sum((g - gamma0) ** 2, axis=1)))

        def bisect(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            over = rms_at(mid) > budget
            return (jnp.where(over, lo, mid), jnp.where(over, mid, hi))

        lo, _hi = jax.lax.fori_loop(0, 30, bisect, (0.0, 1.0))
        return jnp.where(rms_at(1.0) > budget, cdofs0 + lo * delta, cdofs)

    cdofs = cdofs0
    for _ in range(steps):
        g1 = grad_peak(cdofs)
        g2 = grad_bulk(cdofs)
        diff = g1 - g2
        alpha = jnp.clip(jnp.dot(g2, g2 - g1) / (jnp.dot(diff, diff) + 1e-30), 0.0, 1.0)
        cdofs = cdofs - lr * (alpha * g1 + (1.0 - alpha) * g2)
        cdofs = project(cdofs)  # hold the field budget EXACTLY each step
    return cdofs


def _run_chomp(value_and_grad, value_fn, cdofs0: jnp.ndarray, precond: np.ndarray,
               lr0: float, steps: int) -> jnp.ndarray:
    """One lambda rung of CHOMP covariant descent (Zucker 2013), Armijo-line-searched.

    Each step moves ``cdofs -= eta * (precond * grad)``: ``precond = M^-1`` (the diagonal
    first-order Sobolev/H^1 metric ``1/(1+alpha*k^2)``, symbol ``1+k^2``) makes the step
    COVARIANT, downscaling the stiff high Fourier modes -- the CHOMP idea. (Canonical CHOMP
    uses the H^2 acceleration metric ``~k^4``; this uses the gentler, validated H^1 variant.)
    Plain GD with this metric (unlike Adam, which is per-coordinate scale-invariant) does NOT
    absorb it, so it actually shapes the step. ``eta`` backtracks from ``lr0`` by halves until
    Armijo sufficient-decrease (c1=1e-4); if none is found in 30 halvings the step is HELD --
    cdofs is left unchanged (NOT ``cdofs - 0*direction``, which would be ``0*NaN = NaN`` if a
    non-finite gradient were supplied), so chomp stalls FINITELY rather than diverging on an
    ill-conditioned objective. This monotone-descent guarantee is what bare GD lacks on the
    stiff softmax-curvature problem. ``value_fn`` is the jitted objective value (no gradient)
    used only by the line search.
    """
    cdofs = cdofs0
    precond = jnp.asarray(precond)
    for _ in range(steps):
        value, grad = value_and_grad(cdofs)
        direction = precond * grad
        slope = jnp.dot(grad, direction)  # g^T M^-1 g >= 0: the Armijo descent magnitude
        eta = lr0
        for _bt in range(30):
            trial = cdofs - eta * direction
            if value_fn(trial) <= value - 1.0e-4 * eta * slope:
                cdofs = trial  # accept the first step meeting Armijo sufficient-decrease
                break
            eta = 0.5 * eta
        # no break -> no sufficient-decrease step in 30 halvings: HOLD cdofs unchanged (fail-
        # closed). Leaving cdofs as-is (vs cdofs - 0*direction) keeps it finite even when grad
        # is non-finite, so chomp stalls rather than poisoning the whole curve to NaN.
    return cdofs


# Each solver owns exactly one formulation. scipy_trf is penalty-ONLY by a hard scipy
# limitation: least_squares supports variable bounds but NOT nonlinear constraints, so
# the epsilon-constraint caps are reachable only via slsqp's SLSQP backend.
_REQUIRED_FORMULATION = {
    Solver.adam_baseline: Formulation.penalty,
    Solver.scipy_trf: Formulation.penalty,
    Solver.slsqp: Formulation.epsilon_constraint,
    Solver.mgda: Formulation.penalty,  # MGDA over the two CURVATURE penalty terms [peak softmax,
    # bulk mean-sq]; the field budget is held by a hard projection, NOT one of the objectives
    Solver.chomp: Formulation.penalty,  # CHOMP covariant descent on the same penalty objective
}


def clean_geometry(curve, *, curvature_cap_inv_m: float,
                   solver: Solver = Solver.adam_baseline,
                   formulation: Formulation = Formulation.penalty,
                   preconditioner: Preconditioner = Preconditioner.none,
                   outer: OuterConstraint = OuterConstraint.tikhonov_proxy,
                   weights: ObjectiveWeights = ObjectiveWeights(),
                   schedule: AdamSchedule = AdamSchedule(),
                   constrained: ConstrainedSolve = ConstrainedSolve(),
                   keepout: "KeepoutSpec | None" = None,
                   n_quadpoints: int = 512) -> CleanupResult:
    """Reshape ``curve``'s CWS dofs to lower peak curvature while holding the field.

    ``curve`` is a live ``CurveCWSFourierCPP`` (the certified banana master);
    ``curvature_cap_inv_m`` is the run's finite-build cap (never hardcoded -- a
    float32 or wrong-cap measurement fails open past the wall). Returns the cleaned
    dofs + an opt-grid convergence summary; the caller rebuilds the curve and runs
    the held-field Poincare recheck. ``schedule`` configures ``adam_baseline``/
    ``scipy_trf`` (penalty form); ``constrained`` configures ``scipy_trf``/``slsqp``
    (the field-budget epsilon-bound); ``weights`` fairing terms are honored only by
    ``adam_baseline``/``chomp`` (nonzero fairing with another solver raises);
    ``weights.keepout_hinge`` (the Phase-4 TrajOpt SDF hinge) needs a ``KeepoutSpec``
    ``keepout`` and is honored by ``adam_baseline``/``chomp``/``scipy_trf``. Members whose
    phase has not landed raise.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "geometry cleanup requires JAX float64 (jax.config.update('jax_enable_x64', "
            "True)); float32 curvature can read under the finite-build cap when it is "
            "over -- a fail-open the certification cannot tolerate.")
    if formulation is not _REQUIRED_FORMULATION[solver]:
        raise ValueError(
            f"Solver.{solver.value} requires Formulation.{_REQUIRED_FORMULATION[solver].value}, "
            f"got Formulation.{formulation.value}. scipy_trf is penalty-only -- "
            f"scipy.optimize.least_squares has no nonlinear-constraints argument -- so the "
            f"epsilon-constraint caps are enforced only by Solver.slsqp.")
    if preconditioner is Preconditioner.sobolev_h2 and solver is not Solver.scipy_trf:
        raise ValueError(
            "Preconditioner.sobolev_h2 applies only to Solver.scipy_trf (it sets "
            "least_squares' x_scale to an H^2 mode metric). Adam's per-coordinate "
            "scale-invariance absorbs a diagonal preconditioner, and SLSQP has no "
            "variable scaling -- so sobolev_h2 with those solvers would silently no-op. "
            "Solver.chomp applies the Sobolev covariant metric INTRINSICALLY (it IS the "
            "CHOMP step) -- use Preconditioner.none with chomp and set the strength via "
            "ConstrainedSolve.sobolev_alpha.")
    if (weights.integral_kappa_sq or weights.min_variation) and solver not in (
            Solver.adam_baseline, Solver.chomp):
        raise ValueError(
            "ObjectiveWeights fairing terms (integral_kappa_sq / min_variation) are honored "
            "only by the penalty-objective solvers Solver.adam_baseline / Solver.chomp: "
            "scipy_trf's least-squares residual and slsqp's LSE objective do not include them. "
            "Pairing nonzero fairing weights with another solver would silently no-op -- thread "
            "them into those backends to lift this restriction.")
    if weights.keepout_hinge and keepout is None:
        raise ValueError(
            "ObjectiveWeights.keepout_hinge > 0 needs a KeepoutSpec (the obstacle SDF + margin) "
            "passed as clean_geometry(..., keepout=...); without it the hinge has no obstacle "
            "and would silently no-op.")
    if weights.keepout_hinge and solver not in (
            Solver.adam_baseline, Solver.chomp, Solver.scipy_trf):
        raise ValueError(
            "ObjectiveWeights.keepout_hinge (the TrajOpt SDF hinge) is honored only by "
            "Solver.adam_baseline / Solver.chomp (penalty objective) and Solver.scipy_trf "
            "(least-squares residual); slsqp's LSE objective and mgda's curvature-only "
            "objectives do not include it, so it would silently no-op there.")
    if outer is not OuterConstraint.tikhonov_proxy:
        raise ValueError(
            f"OuterConstraint.{outer.value} is not a single-solve mode of clean_geometry. "
            f"Confinement as a first-class black-box constraint (Phase 5) is realized by "
            f"bo_confinement_sweep(), which wraps clean_geometry in a GP Bayesian-optimization "
            f"loop over the displacement budget with a caller-supplied confinement evaluator. "
            f"clean_geometry itself implements only OuterConstraint.tikhonov_proxy (implicit "
            f"field preservation via the displacement penalty).")

    geom = CwsCurveGeometry.from_curve(curve)
    qpts = jnp.linspace(0.0, 1.0, n_quadpoints, endpoint=False)
    cdofs0 = jnp.asarray(curve.get_dofs())
    gamma0 = geom.gamma(cdofs0, qpts)

    max_kappa = jax.jit(lambda cd: jnp.max(geom.curvature(cd, qpts)))
    max_disp = jax.jit(lambda cd: jnp.max(jnp.linalg.norm(geom.gamma(cd, qpts) - gamma0, axis=1)))
    rms_disp = jax.jit(lambda cd: jnp.sqrt(jnp.mean(jnp.sum((geom.gamma(cd, qpts) - gamma0) ** 2, axis=1))))
    initial_max_curvature = float(max_kappa(cdofs0))

    solver_success: bool | None = None
    if solver is Solver.adam_baseline:
        cdofs = cdofs0
        for lam in schedule.lambda_ladder:
            value_and_grad = _penalty_value_and_grad(
                geom, gamma0, qpts, curvature_cap_inv_m, weights.curvature_softmax_p, lam,
                weights.integral_kappa_sq, weights.min_variation, weights.keepout_hinge, keepout)
            cdofs = _run_adam(value_and_grad, cdofs, schedule)
    elif solver is Solver.scipy_trf:
        cdofs, result = _run_scipy_trf(
            geom, gamma0, qpts, curvature_cap_inv_m, cdofs0, constrained, preconditioner,
            weights.keepout_hinge, keepout)
        solver_success = bool(result.success)
    elif solver is Solver.slsqp:
        cdofs, result = _run_slsqp(geom, gamma0, qpts, curvature_cap_inv_m, cdofs0, constrained)
        solver_success = bool(result.success)
    elif solver is Solver.chomp:  # Phase 4: CHOMP covariant descent on the SAME penalty
        # objective as adam (incl. fairing + keepout), preconditioned by the intrinsic Sobolev
        # metric + Armijo line search. Same lambda ladder; the field is preserved implicitly by
        # the Tikhonov displacement term, like adam_baseline.
        precond = _sobolev_x_scale(geom.dof_modes, constrained.sobolev_alpha)
        cdofs = cdofs0
        for lam in schedule.lambda_ladder:
            objective = _penalty_objective(
                geom, gamma0, qpts, curvature_cap_inv_m, weights.curvature_softmax_p, lam,
                weights.integral_kappa_sq, weights.min_variation, weights.keepout_hinge, keepout)
            cdofs = _run_chomp(jax.jit(jax.value_and_grad(objective)), jax.jit(objective),
                               cdofs, precond, schedule.lr, schedule.steps)
    else:  # Solver.mgda -- gradient-native multi-objective over [peak, bulk] curvature,
        # field held by the displacement-budget projection. Reuses the schedule's lr/steps
        # (not its lambda ladder) and the constrained field budget.
        cdofs = _run_mgda(geom, gamma0, qpts, curvature_cap_inv_m, weights.curvature_softmax_p,
                          cdofs0, schedule.lr, schedule.steps, constrained.displacement_budget_m)

    return CleanupResult(
        cleaned_dofs=np.asarray(cdofs),
        initial_max_curvature=initial_max_curvature,
        final_max_curvature=float(max_kappa(cdofs)),
        max_displacement_m=float(max_disp(cdofs)),
        rms_displacement_m=float(rms_disp(cdofs)),
        curvature_cap_inv_m=float(curvature_cap_inv_m),
        n_quadpoints=n_quadpoints,
        solver=solver,
        solver_success=solver_success,
    )


@dataclass(frozen=True)
class ParetoPoint:
    """One point on the buildability frontier: the lowest peak curvature reachable at a
    given field-preservation budget (the request-#2 'how smooth within the field budget').

    ``feasible`` is the gate the caller MUST filter on: True iff the slsqp solve HELD its
    RMS budget (rms <= budget*1.05). It is deliberately NOT gated on ``solver_success`` --
    the constrained solver is iteration-limited at a GOOD feasible point on the stiff real
    coil (``solver_success`` is False even there), so requiring it would mark EVERY real-coil
    point infeasible. A genuinely diverged solve unwinds to metre-scale rms (a misleadingly
    low ``final_max_curvature`` off the field) and is excluded by the budget check alone --
    drop those, never read their curvature as achievable within the budget."""

    displacement_budget_m: float
    final_max_curvature: float
    rms_displacement_m: float
    max_displacement_m: float
    solver_success: bool | None
    feasible: bool


def pareto_sweep(curve, *, curvature_cap_inv_m: float,
                 displacement_budgets: tuple[float, ...],
                 constrained: ConstrainedSolve = ConstrainedSolve(),
                 n_quadpoints: int = 512) -> list[ParetoPoint]:
    """Trace the curvature-vs-field-budget Pareto frontier (Phase 3, epsilon-constraint).

    For each budget, run the slsqp solve (minimize peak curvature s.t. RMS displacement
    <= budget) and record the achieved curvature + displacement. The frontier shows how
    much curvature margin each unit of field budget buys; the caller picks the knee (the
    lowest kappa whose RE-TRACED Poincare still holds >=30/50 -- confinement is NOT
    checked here, only the geometry trade, so the caller must verify the chosen point).
    Filter to ``p.feasible`` FIRST: a large budget can make SLSQP diverge (a field-
    destroying unwind whose spuriously low kappa is not reachable within the budget) --
    that is not a frontier point. Each budget is a fresh solve from the seed, so the
    curve is never mutated. Returns one ParetoPoint per budget, in input order.
    """
    points: list[ParetoPoint] = []
    for budget in displacement_budgets:
        r = clean_geometry(
            curve, curvature_cap_inv_m=curvature_cap_inv_m, solver=Solver.slsqp,
            formulation=Formulation.epsilon_constraint,
            constrained=replace(constrained, displacement_budget_m=budget),
            n_quadpoints=n_quadpoints)
        points.append(ParetoPoint(
            displacement_budget_m=budget,
            final_max_curvature=r.final_max_curvature,
            rms_displacement_m=r.rms_displacement_m,
            max_displacement_m=r.max_displacement_m,
            solver_success=r.solver_success,
            # Feasible == held the field budget. NOT gated on solver_success: the stiff coil's
            # constrained solve is iteration-limited at a good feasible point (success always
            # False), so gating on it would mark every real-coil frontier point infeasible. A
            # diverged unwind has rms >> budget and is excluded by the rms check alone.
            feasible=r.rms_displacement_m <= budget * 1.05))
    return points


def _gp_posterior(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
                  lengthscale: float, noise_var: float) -> tuple[np.ndarray, np.ndarray]:
    """1-D Gaussian-process regression posterior (mean, std) at ``x_eval``.

    RBF kernel ``k(a,b)=exp(-0.5*(a-b)^2/lengthscale^2)`` with UNIT signal variance, so the
    caller standardizes ``y_train`` (zero mean / unit std) and normalizes ``x`` to ~[0,1]
    first. ``noise_var`` is the observation jitter (the inner solve is deterministic, so this
    is tiny -- it only conditions the Cholesky). Returns posterior mean and standard deviation
    (>= 0) at each ``x_eval``. Hand-rolled (no GPyTorch/BoTorch dependency): a 1-D BO over the
    displacement budget needs only this closed-form regressor.
    """
    a = np.asarray(x_train, dtype=float).reshape(-1, 1)
    b = np.asarray(x_eval, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float)
    kernel = lambda u, v: np.exp(-0.5 * (u - v.T) ** 2 / lengthscale ** 2)  # noqa: E731
    gram = kernel(a, a) + noise_var * np.eye(a.shape[0])
    cross = kernel(b, a)
    chol = np.linalg.cholesky(gram)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    mean = cross @ alpha
    v = np.linalg.solve(chol, cross.T)
    var = np.maximum(1.0 - np.sum(v ** 2, axis=0), 1.0e-12)  # prior var 1 (unit signal variance)
    return mean, np.sqrt(var)


def _expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    """Expected improvement for MINIMIZATION over the incumbent ``best`` (standardized units).

    ``EI = (best-mu)*Phi(z) + sigma*phi(z)``, ``z=(best-mu)/sigma`` (Jones 1998): >= 0, rising
    as the posterior mean drops below ``best`` or the posterior std grows, and ~0 at a confidently
    non-improving point. ``std`` is floored to avoid division by zero at observed points.
    """
    std = np.maximum(std, 1.0e-12)
    z = (best - mean) / std
    return (best - mean) * scipy.special.ndtr(z) + std * np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class BoTrial:
    """One black-box evaluation in :func:`bo_confinement_sweep`: a budget -> (curvature, confinement).

    ``solve_feasible`` is the inner slsqp's own feasibility (held its RMS budget, did not unwind --
    the ParetoPoint.feasible logic); ``confinement_ok`` is ``survival >= survival_threshold``. A
    point counts as a candidate ONLY if both hold (a diverged solve is not a real coil even if its
    confinement reading is high)."""

    budget_m: float
    final_max_curvature: float
    rms_displacement_m: float
    survival: float
    solve_feasible: bool
    confinement_ok: bool


@dataclass(frozen=True)
class BoConfinementResult:
    """Outcome of :func:`bo_confinement_sweep`: the lowest-curvature budget whose RE-TRACED
    confinement survival cleared the threshold (and whose inner solve was feasible), plus the full
    trial history.

    ``best_*`` are None when NO evaluated budget was feasible -- raise the budget range / lower the
    threshold, or accept that confinement caps the achievable curvature. The caller rebuilds the
    coil at ``best_budget_m`` (clean_geometry slsqp) for the certified artifact; this sweep only
    SELECTS the budget, it never mutates the curve."""

    best_budget_m: float | None
    best_final_max_curvature: float | None
    best_survival: float | None
    trials: tuple[BoTrial, ...]


def bo_confinement_sweep(curve, *, curvature_cap_inv_m: float,
                         confinement_fn: Callable[[np.ndarray], float],
                         budget_bounds: tuple[float, float],
                         survival_threshold: float,
                         n_init: int = 3, n_iter: int = 7,
                         constrained: ConstrainedSolve = ConstrainedSolve(),
                         n_quadpoints: int = 512) -> BoConfinementResult:
    """Phase 5: confinement as a first-class black-box constraint via hand-rolled GP-BO.

    Minimizes peak curvature over the displacement budget subject to the EXPENSIVE black-box
    constraint ``confinement_fn(cleaned_dofs) >= survival_threshold`` (the default-mode Poincare
    survival count -- the module cannot trace it, so the caller supplies the evaluator, the same
    boundary as ``KeepoutSpec.sdf_fn``). This is the request-#2 "push to the field budget" lever:
    as the budget grows, slsqp drives curvature down but the field moves more and confinement
    eventually falls off a cliff; BO spends its few expensive evaluations near that cliff rather
    than on a dense grid (``pareto_sweep`` is the confinement-BLIND dense grid; this is the
    confinement-AWARE sparse search). Conservative mode (request #1, recheck) = a narrow
    ``budget_bounds`` + small ``n_iter``.

    Each evaluation runs ``clean_geometry(Solver.slsqp, budget)`` then ``confinement_fn`` on the
    cleaned dofs. Acquisition = constrained EI = ``EI(curvature) * P(survival >= threshold)`` from
    two 1-D GPs (curvature, survival) over the standardized budget (Gardner 2014); before any
    feasible point is seen it falls back to pure feasibility search ``P(feasible)``. The grid node(s)
    nearest an already-evaluated budget are masked from the acquisition so an expensive evaluation is
    not wasted re-probing a budget already tried. Returns the lowest-curvature FEASIBLE budget + the
    full history; ``best_*`` are
    None if nothing cleared the threshold. NO BoTorch/PyTorch dependency -- the GP/EI are the
    closed forms above (a 1-D BO needs nothing heavier).
    """
    lo, hi = budget_bounds
    if not hi > lo:
        raise ValueError(f"budget_bounds must be (lo, hi) with hi > lo; got {budget_bounds!r}.")
    if n_init < 2:
        raise ValueError(f"n_init must be >= 2 to seed the GP; got {n_init}.")

    def evaluate(budget: float) -> BoTrial:
        r = clean_geometry(
            curve, curvature_cap_inv_m=curvature_cap_inv_m, solver=Solver.slsqp,
            formulation=Formulation.epsilon_constraint,
            constrained=replace(constrained, displacement_budget_m=budget),
            n_quadpoints=n_quadpoints)
        survival = float(confinement_fn(r.cleaned_dofs))
        # Feasible == held the field budget. NOT gated on solver_success: the constrained
        # solvers are iteration-limited at a GOOD feasible point on the stiff real coil
        # (solver_success is always False there), so gating on it would reject every
        # real-coil solve and bo_confinement_sweep could never find a budget. A genuinely
        # diverged solve has rms >> budget and is caught by the rms check alone.
        solve_feasible = r.rms_displacement_m <= budget * 1.05
        return BoTrial(
            budget_m=float(budget), final_max_curvature=r.final_max_curvature,
            rms_displacement_m=r.rms_displacement_m, survival=survival,
            solve_feasible=solve_feasible, confinement_ok=(survival >= survival_threshold))

    trials = [evaluate(float(b)) for b in np.linspace(lo, hi, n_init)]

    grid = np.linspace(lo, hi, 256)
    x_grid = (grid - lo) / (hi - lo)
    for _ in range(n_iter):
        b_obs = np.array([t.budget_m for t in trials])
        x_obs = (b_obs - lo) / (hi - lo)
        kappa = np.array([t.final_max_curvature for t in trials])
        surv = np.array([t.survival for t in trials])
        candidate = np.array([t.confinement_ok and t.solve_feasible for t in trials])

        # standardize each objective (flat-data guard: std=1 so the GP reduces to its mean)
        k_mean, k_std = float(kappa.mean()), float(kappa.std()) or 1.0
        s_mean, s_std = float(surv.mean()), float(surv.std()) or 1.0
        gp_k_mean, gp_k_std = _gp_posterior(x_obs, (kappa - k_mean) / k_std, x_grid, 0.3, 1.0e-6)
        gp_s_mean, gp_s_std = _gp_posterior(x_obs, (surv - s_mean) / s_std, x_grid, 0.3, 1.0e-6)

        # P(survival >= threshold) from the survival GP (threshold mapped to standardized units)
        thr_std = (survival_threshold - s_mean) / s_std
        p_feasible = scipy.special.ndtr((gp_s_mean - thr_std) / np.maximum(gp_s_std, 1.0e-12))

        if candidate.any():
            best_k_std = (float(kappa[candidate].min()) - k_mean) / k_std  # incumbent = best feasible kappa
            acquisition = _expected_improvement(gp_k_mean, gp_k_std, best_k_std) * p_feasible
        else:
            acquisition = p_feasible  # no feasible point yet -> search for feasibility first

        # mask the grid node(s) nearest each already-evaluated budget (radius = one grid spacing
        # (hi-lo)/255 > half-spacing, so both nodes bracketing a between-nodes budget are masked)
        # -- the deterministic inner solve at a near-duplicate budget would just waste an
        # expensive evaluation for no new information.
        for b in b_obs:
            acquisition = np.where(np.abs(grid - b) < (hi - lo) / 255.0, -np.inf, acquisition)
        trials.append(evaluate(float(grid[int(np.argmax(acquisition))])))

    candidates = [t for t in trials if t.confinement_ok and t.solve_feasible]
    best = min(candidates, key=lambda t: t.final_max_curvature) if candidates else None
    return BoConfinementResult(
        best_budget_m=best.budget_m if best else None,
        best_final_max_curvature=best.final_max_curvature if best else None,
        best_survival=best.survival if best else None,
        trials=tuple(trials))
