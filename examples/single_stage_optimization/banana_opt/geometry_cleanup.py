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
Later phases ADD fairing/min-variation terms, a Sobolev preconditioner, a Pareto
sweep, and a black-box confinement constraint. Nothing here replaces the baseline.

Output hygiene (measuring the reshaped field, writing provenance instead of a
fabricated ``results.json``) is owned by :mod:`banana_opt.reshape_run_dir`; this
module returns the cleaned dofs + an opt-grid convergence summary and leaves the
authoritative high-resolution verdict to that module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import scipy.optimize

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
    mgda = "mgda"  # Phase 3: gradient-native multi-objective local steps


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

    ``none`` (Phase 0): identity (raw Adam). ``sobolev_h2`` (Phase 2): an H^2
    Sobolev metric on the Fourier coefficients to escape Adam/GD plateaus.
    """

    none = "none"
    sobolev_h2 = "sobolev_h2"


class OuterConstraint(Enum):
    """How confinement (the Poincare survival count) is kept during cleanup.

    ``tikhonov_proxy`` (Phase 0): confinement is preserved *implicitly* -- the
    displacement penalty keeps gamma near gamma0 so the field barely moves; the
    caller re-traces Poincare after to confirm. ``bo_confinement`` (Phase 5):
    confinement becomes a first-class expensive black-box constraint (BoTorch).
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
    Both are externally-owned research config (off by default, on to trade a little
    field budget for a smoother coil); the softmax is what guarantees kappa<=cap.
    """

    curvature_softmax_p: float = 20.0
    integral_kappa_sq: float = 0.0
    min_variation: float = 0.0


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
    sobolev_alpha: float = 1.0  # H^2 metric strength for Preconditioner.sobolev_h2 (scipy_trf x_scale)
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


def _penalty_value_and_grad(geom: CwsCurveGeometry, gamma0: jnp.ndarray,
                            qpts: jnp.ndarray, cap: float, softmax_p: float,
                            lam: float, integral_kappa_sq: float = 0.0,
                            min_variation: float = 0.0):
    """Build ``value_and_grad`` of the penalty objective.

    ``softmax(kappa/cap) + lam*mean|gamma-gamma0|^2`` (+ optional Phase-2 fairing:
    ``integral_kappa_sq*mean(kappa/cap)^2 + min_variation*mean(dkappa/ds)^2``). A
    factory: one jitted closure per lambda rung (lambda baked into the graph). The
    fairing weights are Python floats baked at trace time, so their ``if`` guards
    are compile-time -- zero cost (and exact byte-equality with the baseline) when
    off. ``gamma0`` is the original centerline on the same ``qpts``.
    """
    def objective(cdofs: jnp.ndarray) -> jnp.ndarray:
        kappa = geom.curvature(cdofs, qpts)
        softmax_kappa = jnp.mean((kappa / cap) ** softmax_p) ** (1.0 / softmax_p)
        displacement = jnp.mean(jnp.sum((geom.gamma(cdofs, qpts) - gamma0) ** 2, axis=1))
        value = softmax_kappa + lam * displacement
        if integral_kappa_sq:
            value = value + integral_kappa_sq * jnp.mean((kappa / cap) ** 2)
        if min_variation:
            # dkappa/ds = (dkappa/dq) / |gamma'|  (arc-length derivative; q in [0,1))
            dkappa_dq = jax.jvp(lambda q: geom.curvature(cdofs, q), (qpts,), (jnp.ones_like(qpts),))[1]
            speed = jnp.linalg.norm(
                jax.jvp(lambda q: geom.gamma(cdofs, q), (qpts,), (jnp.ones_like(qpts),))[1], axis=1)
            value = value + min_variation * jnp.mean((dkappa_dq / speed) ** 2)
        return value

    return jax.jit(jax.value_and_grad(objective))


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
                          cap: float, softmax_p: float, lam: float):
    """Jitted residual ``r`` (and Jacobian) with ``||r||^2 == softmax(kappa/cap) + lam*mean|gamma-gamma0|^2``.

    The penalty form for ``Solver.scipy_trf``: exactly the objective ``adam_baseline``
    minimizes, re-expressed as a least-squares residual so ``least_squares``' TRF
    (trust-region reflective = adaptive Levenberg-Marquardt) damping controls the
    step. One scalar curvature residual + 3N displacement residuals scaled so the
    summed squares reproduce the objective exactly (scipy minimizes 0.5||r||^2 -- the
    constant half does not move the minimizer). ``lam`` baked per rung (the ladder).
    """
    n = qpts.shape[0]

    def residual(cdofs: jnp.ndarray) -> jnp.ndarray:
        softmax = jnp.mean((geom.curvature(cdofs, qpts) / cap) ** softmax_p) ** (1.0 / softmax_p)
        disp = (geom.gamma(cdofs, qpts) - gamma0).reshape(-1)
        return jnp.concatenate([jnp.sqrt(softmax)[None], jnp.sqrt(lam / n) * disp])

    return jax.jit(residual), jax.jit(jax.jacfwd(residual))


def _sobolev_x_scale(dof_modes: tuple[int, ...], alpha: float) -> np.ndarray:
    """``least_squares`` x_scale implementing an H^2 Sobolev metric on the Fourier modes.

    x_scale[i] = 1/(1 + alpha*k_i^2): higher modes get a SMALLER characteristic scale,
    so the trust region steps them less -> a smooth (Sobolev-gradient) descent that
    damps high-frequency wiggle. This is the explicit-metric alternative to the
    data-driven ``x_scale='jac'``. (A diagonal metric like this is *absorbed* by Adam's
    per-coordinate scale-invariance, which is why ``sobolev_h2`` is a scipy_trf-only
    preconditioner -- see the clean_geometry guard.)
    """
    k = np.asarray(dof_modes, dtype=float)
    return 1.0 / (1.0 + alpha * k ** 2)


def _run_scipy_trf(geom: CwsCurveGeometry, gamma0: jnp.ndarray, qpts: jnp.ndarray,
                   cap: float, cdofs0: jnp.ndarray, settings: ConstrainedSolve,
                   preconditioner: "Preconditioner" = None):
    """Penalty least-squares (TRF) warm-started down the lambda ladder.

    Returns ``(cleaned_dofs, last_scipy_result)``. TRF's adaptive LM damping is the
    principled step-control that cures the L-BFGS-B stall and GD divergence; the
    displacement residual (NOT the damping) is what holds the field. ``preconditioner``
    selects the variable scaling: ``none`` -> ``x_scale='jac'`` (data-driven, validated
    on the real coil); ``sobolev_h2`` -> the explicit H^2 mode metric.
    """
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
            geom, gamma0, qpts, cap, settings.curvature_softmax_p, lam)
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
    budget (scale the dof-displacement so rms|gamma-gamma0| <= budget). Both objectives
    are curvature (nonzero gradient at the high-kappa seed), so the common direction is
    nonzero and progress is real. NB pairing curvature with the DISPLACEMENT objective
    instead is degenerate: displacement is at its global min (0) at the field-optimal
    seed, so its gradient vanishes there, the min-norm direction collapses to 0, and
    MGDA stalls -- hence the field is a projection (a constraint), not an MGDA objective.
    """
    def f_peak(cd: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((geom.curvature(cd, qpts) / cap) ** softmax_p) ** (1.0 / softmax_p)

    def f_bulk(cd: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((geom.curvature(cd, qpts) / cap) ** 2)

    grad_peak = jax.jit(jax.grad(f_peak))
    grad_bulk = jax.jit(jax.grad(f_bulk))
    rms_disp = jax.jit(lambda cd: jnp.sqrt(jnp.mean(jnp.sum((geom.gamma(cd, qpts) - gamma0) ** 2, axis=1))))
    cdofs = cdofs0
    for _ in range(steps):
        g1 = grad_peak(cdofs)
        g2 = grad_bulk(cdofs)
        diff = g1 - g2
        denom = jnp.dot(diff, diff)
        alpha = jnp.clip(jnp.dot(g2, g2 - g1) / (denom + 1e-30), 0.0, 1.0)
        cdofs = cdofs - lr * (alpha * g1 + (1.0 - alpha) * g2)
        # Project onto the field budget: in the small-step ~linear regime, scaling the
        # dof-delta by budget/rms scales the gamma-displacement back under the budget.
        r = rms_disp(cdofs)
        cdofs = jnp.where(r > budget, cdofs0 + (cdofs - cdofs0) * (budget / (r + 1e-30)), cdofs)
    return cdofs


# Each solver owns exactly one formulation. scipy_trf is penalty-ONLY by a hard scipy
# limitation: least_squares supports variable bounds but NOT nonlinear constraints, so
# the epsilon-constraint caps are reachable only via slsqp's SLSQP backend.
_REQUIRED_FORMULATION = {
    Solver.adam_baseline: Formulation.penalty,
    Solver.scipy_trf: Formulation.penalty,
    Solver.slsqp: Formulation.epsilon_constraint,
    Solver.mgda: Formulation.penalty,  # multi-objective over the penalty terms (curvature vs displacement)
}


def clean_geometry(curve, *, curvature_cap_inv_m: float,
                   solver: Solver = Solver.adam_baseline,
                   formulation: Formulation = Formulation.penalty,
                   preconditioner: Preconditioner = Preconditioner.none,
                   outer: OuterConstraint = OuterConstraint.tikhonov_proxy,
                   weights: ObjectiveWeights = ObjectiveWeights(),
                   schedule: AdamSchedule = AdamSchedule(),
                   constrained: ConstrainedSolve = ConstrainedSolve(),
                   n_quadpoints: int = 512) -> CleanupResult:
    """Reshape ``curve``'s CWS dofs to lower peak curvature while holding the field.

    ``curve`` is a live ``CurveCWSFourierCPP`` (the certified banana master);
    ``curvature_cap_inv_m`` is the run's finite-build cap (never hardcoded -- a
    float32 or wrong-cap measurement fails open past the wall). Returns the cleaned
    dofs + an opt-grid convergence summary; the caller rebuilds the curve and runs
    the held-field Poincare recheck. ``schedule`` configures ``adam_baseline``/
    ``scipy_trf`` (penalty form); ``constrained`` configures ``scipy_trf``/``slsqp``
    (the field-budget epsilon-bound); ``weights`` fairing terms are honored only by
    ``adam_baseline`` (nonzero fairing with another solver raises). Members whose phase
    has not landed raise.
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
            "variable scaling -- so sobolev_h2 with those solvers would silently no-op.")
    if (weights.integral_kappa_sq or weights.min_variation) and solver is not Solver.adam_baseline:
        raise ValueError(
            "ObjectiveWeights fairing terms (integral_kappa_sq / min_variation) are honored "
            "only by Solver.adam_baseline: they enter the penalty objective, which scipy_trf's "
            "least-squares residual and slsqp's LSE objective do not include. Pairing nonzero "
            "fairing weights with another solver would silently no-op -- thread them into those "
            "backends to lift this restriction.")
    if outer is not OuterConstraint.tikhonov_proxy:
        raise NotImplementedError(
            f"OuterConstraint.{outer.value} lands in Phase 5; Phases 0-1 implement only "
            f"OuterConstraint.tikhonov_proxy (implicit field preservation).")

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
                weights.integral_kappa_sq, weights.min_variation)
            cdofs = _run_adam(value_and_grad, cdofs, schedule)
    elif solver is Solver.scipy_trf:
        cdofs, result = _run_scipy_trf(
            geom, gamma0, qpts, curvature_cap_inv_m, cdofs0, constrained, preconditioner)
        solver_success = bool(result.success)
    elif solver is Solver.slsqp:
        cdofs, result = _run_slsqp(geom, gamma0, qpts, curvature_cap_inv_m, cdofs0, constrained)
        solver_success = bool(result.success)
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
    given field-preservation budget (the request-#2 'how smooth within the field budget')."""

    displacement_budget_m: float
    final_max_curvature: float
    rms_displacement_m: float
    max_displacement_m: float
    solver_success: bool | None


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
    Each budget is a fresh solve from the seed, so the curve is never mutated. Returns
    one ParetoPoint per budget, in input order.
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
            solver_success=r.solver_success))
    return points
