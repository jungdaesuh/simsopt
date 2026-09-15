"""Canonical nested Boozer-LS bars: reconstruct physics vs banana timing.

Gate 0 of the reduced nested-LS JAX GPU track. These knobs are not the
flat-675 fused L-BFGS-B configuration and do not inherit its 7.70× claim.

Physics certification uses reconstruct-only LS Newton
(``constraint_weight=1``, free ``G``, ``weight_inv_modB``, ``stab=1e-4``,
``tol=1e-13`` on ``||∇J_LS||_2``, Volume label). Banana ``run_code`` is
BFGS then Newton with ``stab=0``, ``newton_tol=1e-11``, and
``newton_maxiter=40``. The two bars are not interchangeable. Timing
claims against production nested SIMSOPT use banana ``run_code`` only
after the physics bar's branch matches.
"""

from __future__ import annotations

from typing import Final, Generic, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray

NESTED_LS_CONSTRAINT_WEIGHT: Final[float] = 1.0
NESTED_LS_WEIGHT_INV_MODB: Final[bool] = True
NESTED_LS_OPTIMIZE_G: Final[bool] = True
NESTED_LS_LABEL: Final[str] = "Volume"

# Reconstruct / physics bar (C++ LS Newton judge).
NESTED_LS_NEWTON_STAB: Final[float] = 1.0e-4
NESTED_LS_NEWTON_TOL: Final[float] = 1.0e-13
NESTED_LS_NEWTON_MAXITER: Final[int] = 10
NESTED_LS_REDUCTION_MODE: Final[str] = "cpu_ordered"

# Three-valued inner exit (Phase 3 of
# ``docs/nested_ls_upgrade_implementation_plan.md``).
#
# The inner Schur-Newton walk has always published one bit, ``success``, set
# by ``reduced_gradient_l2 <= NESTED_LS_NEWTON_TOL``. That bit collapses two
# materially different outcomes: a walk that stopped five orders of magnitude
# short of tolerance reads the same as one that produced nothing usable. The
# distinction is not hypothetical. In the B37 v2 diagnostic the three late
# rejections stop at ``||g||_2`` of 1.89e-2 (eval 39), 9.93e-4 (eval 43) and
# 4.99e-3 (eval 53), having SPENT 9, 10 and 10 of
# ``NESTED_LS_NEWTON_MAXITER = 10`` Newton iterations -- evals 43 and 53 on
# budget exhaustion, eval 39 on the Armijo/quality bail at its tenth step.
#
# Read the published ledgers carefully here: ``NestedLsInnerSolveFailed``
# rendered that count under the label "iterations left" while passing
# ``iteration_count``, which ``nested_ls_reduced.py`` defines as iterations
# COMPLETED. Every rejection string recorded before that label was corrected
# reads inverted, and taking it at face value inverts which lever these
# failures call for: they are a budget/step-size problem, not an early bail.
#
# ``NESTED_LS_NEWTON_COARSE_TOL`` is DESC's production inner setting. A
# persisted walk landing in ``(NESTED_LS_NEWTON_TOL, COARSE_TOL]`` is
# ``coarse_converged``. That is TYPED EVIDENCE ONLY: ``success`` stays true
# for ``converged`` alone, so every existing consumer treats a coarse exit
# exactly as it treated a failure before this constant existed. Phase 4 must
# measure the IFT adjoint's gradient error as a function of inner residual
# norm before any consumer is allowed to read ``coarse_converged`` as
# usable; until then the status is recorded and not acted on.
NESTED_LS_NEWTON_COARSE_TOL: Final[float] = 1.0e-8

NESTED_LS_NEWTON_EXIT_CONVERGED: Final[str] = "converged"
NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED: Final[str] = "coarse_converged"
NESTED_LS_NEWTON_EXIT_FAILED: Final[str] = "failed"
NESTED_LS_NEWTON_EXIT_STATUSES: Final[tuple[str, str, str]] = (
    NESTED_LS_NEWTON_EXIT_CONVERGED,
    NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED,
    NESTED_LS_NEWTON_EXIT_FAILED,
)

# Frozen inner-lane policy (Phase 5 of the upgrade plan).
#
# ``OuterOptimizerPolicy`` pins the OUTER scipy knobs -- ftol, gtol, maxls,
# maxiter, maxcor, the rejection scale. It says nothing about the inner
# lane, and that was a real hole: a run with sub-stepping or the predictor
# enabled published a policy BYTE-IDENTICAL to a run without them. Two
# different optimizations, one record shape, and no way for a consumer to
# tell them apart -- which is the precondition for comparing numbers that
# must not be compared.
#
# Inner records name solver families, sequences, and per-stage options rather
# than a lane-agnostic policy. The JAX and native children use materially
# different inner solvers, so a shared record that omits that distinction is
# false provenance.
NESTED_LS_JAX_INNER_POLICY_NAME: Final[str] = "reduced_schur_newton_v1"
NESTED_LS_NATIVE_INNER_POLICY_NAME: Final[str] = "banana_bfgs_then_newton_v1"
NESTED_LS_JAX_INNER_STAB: Final[float] = 0.0

# Coarse diagnostic threshold (Phase 4 of the upgrade plan).
#
# This threshold classifies a completed solve for diagnostic evidence only;
# no outer consumer is authorized to use it in place of the tight Newton
# tolerance. A local diagnostic sampled eight achieved residuals from 4.3e-2
# to 1.7e-13 at one anchor and one trial displacement by capping Newton
# iterations. That artifact is not retained at HEAD, so these observations
# explain the diagnostic threshold but cannot certify a consumer policy.
#
# The measurements are informative but do not license a coarse use:
#
# 1. kappa_2(H_ss) = 2.055e5 at stab = 0, and the IFT adjoint DOES NOT
#    amplify by it. Measured relative gradient error is 0.30x to 3.94x the
#    achieved residual across eleven decades -- order unity.
# 2. Neither curve is monotone in the residual (rho = 4.3e-2 gives a
#    smaller error than rho = 2.0e-2, in both quantities), so this single
#    state cannot establish an envelope.
#
# SINGLE STATE. One anchor, one displacement, one host. These bounds are
# the best available evidence and are not a proof of an envelope; a second
# state may move the observed behavior, so Phase 5 may record this threshold
# but must not promote it into a solver-use policy.

# Predictor trust region (Phase 2 of the upgrade plan).
#
# DESC ``tr_ratio`` semantics: the bound SCALES the step, it does not reject
# it. The reject-to-bare-anchor decision is a separate rule of ours
# (:func:`nested_ls_predictor_arm`), because DESC has no equivalent.
NESTED_LS_PREDICTOR_TRUST_REGION_RATIO: Final[float] = 0.1

#: Which start an outer evaluation used.
NESTED_LS_PREDICTOR_ARM_BARE: Final[str] = "bare_anchor"
NESTED_LS_PREDICTOR_ARM_PREDICTED: Final[str] = "predicted"

# Inner Δc sub-stepping (Phase 3 of the upgrade plan).
#
# The ladder of leg counts an inner solve may retry a failed displacement
# with, each rung restarting from the committed anchor. This is the rung the
# B37 v2 evidence actually points at, once its ledger strings are read the
# right way round: the three late failures spend 9, 10 and 10 of 10 Newton
# iterations at ``‖Δc‖`` of 7.3e-3, 3.2e-3 and 7.2e-3 from a committed
# anchor. Two of the three exhaust the budget on a walk that is still
# converging, which is a step-size problem — shorten the step and the same
# budget reaches the branch. It is NOT what retry-with-regularization
# addresses; that rung was withdrawn.
#
# The first rung is 1, i.e. the undivided step, so a lane with sub-stepping
# enabled takes exactly today's trajectory whenever today's trajectory
# works, and pays nothing for the option.
NESTED_LS_INNER_SUBSTEP_LEGS: Final[tuple[int, ...]] = (1, 2, 4, 8)

# Banana run_code / later timing bar. Newton does not pass stab, so it
# keeps the method default of 0. BFGS runs first.
NESTED_LS_BANANA_NEWTON_STAB: Final[float] = 0.0
NESTED_LS_BANANA_NEWTON_TOL: Final[float] = 1.0e-11
NESTED_LS_BANANA_NEWTON_MAXITER: Final[int] = 40
NESTED_LS_BANANA_BFGS_TOL: Final[float] = 1.0e-10
NESTED_LS_BANANA_USES_BFGS_THEN_NEWTON: Final[bool] = True

NESTED_LS_PHYSICS_BAR: Final[str] = "reconstruct_newton"
NESTED_LS_TIMING_BAR: Final[str] = "banana_run_code"

# Gate-6 claim protocol. Jax-free so GPU parents can import it without
# initializing a device. Pre-lever measured seconds stay in
# nested_ls_reduced_scale.
NESTED_LS_GATE6_IOTA_G_TOL: Final[float] = 1.0e-11
NESTED_LS_GATE6_CLAIM_REPEATS: Final[int] = 3
NESTED_LS_GATE6_AGGREGATION: Final[str] = "min"
NESTED_LS_GATE6_NATIVE_OMP_THREADS: Final[int] = 16

# Fresh-process child payload schemas. These live in the JAX-free contract
# module so producers, the claim parent, and the rejudge consumer share one
# source of truth without importing either process-level child module.
# v6 -> v7 / v5 -> v6: ``inner_policy`` now identifies each
# child's actual solver family and sequence. The prior block claimed the JAX
# reduced-Schur Newton settings for the native banana BFGS-then-Newton lane.
#
# The bump is not merely additive bookkeeping. Before the block existed a
# consumer could not distinguish a stock inner lane from a sub-stepped or
# predicted one, so it could compare numbers that must not be compared. An
# old consumer reading a new record would still see the fields it knows and
# would still draw that wrong conclusion -- so the version has to move, to
# stop it.
NESTED_LS_OUTER_JAX_CHILD_SCHEMA: Final[str] = "nested-ls-outer-jax-child.v7"
NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA: Final[str] = "nested-ls-outer-native-child.v6"
NESTED_LS_OUTER_REJUDGE_SCHEMA: Final[str] = "nested-ls-outer-rejudge.v1"
NESTED_LS_OUTER_FD0_SCHEMA: Final[str] = "nested-ls-outer-fd0.v3"

# Gate FD-0 of the eight-term outer acceptance gate
# (``docs/jax_nested_ls_outer_charter.md``). The outer variable is the
# coil block only; every coil unit direction is differenced centrally at
# ``eps`` and ``eps/2``, and both the tolerance and the step rule are
# frozen here so no probe carries a bare number.
#
# The step is deliberately on the truncation side of the round-off
# optimum, because the gate asks for two things at once: an error at or
# under ``NESTED_LS_OUTER_FD0_REL_TOL`` and an error that *falls* when
# the step halves. Writing the central-difference error as
# ``A*eps^2 + n/eps`` (truncation plus the inner solve's noise floor),
# halving only improves it while ``eps`` exceeds ~1.4x the optimum, so a
# step chosen at the optimum would fail the order clause. The archived
# unregularized Schur spectrum at the frozen endpoint
# (``sigma_min = 6.5e-3``) bounds the inner-solve noise near 1e-11, which
# puts that crossover near 3e-4 of the coil scale.
NESTED_LS_OUTER_FD0_DIRECTIONS: Final[int] = 11
NESTED_LS_OUTER_FD0_REL_TOL: Final[float] = 1.0e-5
NESTED_LS_OUTER_FD0_STEP_RELATIVE: Final[float] = 3.0e-4
NESTED_LS_OUTER_FD0_STEP_SCALE_FLOOR: Final[float] = 1.0e-1
NESTED_LS_OUTER_FD0_STEP_HALVING: Final[float] = 0.5
NESTED_LS_OUTER_FD0_STEP_RULE: Final[str] = (
    "eps_i = NESTED_LS_OUTER_FD0_STEP_RELATIVE * max(abs(coil_i), "
    "NESTED_LS_OUTER_FD0_STEP_SCALE_FLOOR); the halved step is "
    "NESTED_LS_OUTER_FD0_STEP_HALVING * eps_i"
)

# Design amendment 3: the fixed two-rung ladder was the
# defect. A step chosen from an absolute floor is a large *relative*
# perturbation on a small-|c| DOF, which leaves the quadratic FD regime;
# the first run failed exactly the floor-clamped directions and passed
# both derived-eps ones, every failure improving under halving — the
# truncation signature. The replacement keeps halving while the halved
# step improves the relative error, to a pre-declared depth.
NESTED_LS_OUTER_FD0_MAX_HALVINGS: Final[int] = 8

# How far under the 1e-5 band the measured noise must sit at the smallest
# step the descent may take. At eps_min the J scatter alone accounts for
# at most a NESTED_LS_OUTER_FD0_NOISE_SAFETY-th of the band.
NESTED_LS_OUTER_FD0_NOISE_SAFETY: Final[float] = 10.0

# Repeated base-point re-solves behind the measured scatter. Two extra
# evaluations beside the base one bound the pairwise spread of J at fixed
# coils; the floor is measured from that spread, never guessed.
NESTED_LS_OUTER_FD0_SCATTER_REPEATS: Final[int] = 2

# eps_min, the floor the descent refuses to cross. A central difference
# divides the J difference by the realized span 2*eps, so a J band of
# width delta_J puts up to delta_J / (2*eps) of absolute error on the
# directional derivative, i.e. a relative error of
# delta_J / (2 * eps * abs(g.d)). Requiring that to stay at or under a
# NOISE_SAFETY-th of the band REL_TOL and solving for eps gives the rule
# below. A rung under eps_min could be explained by scatter alone, so it
# would prove nothing about the gradient and the ladder will not take it.
NESTED_LS_OUTER_FD0_MIN_STEP_RULE: Final[str] = (
    "eps_min_i = NESTED_LS_OUTER_FD0_NOISE_SAFETY * delta_J / "
    "(2 * NESTED_LS_OUTER_FD0_REL_TOL * abs(g.d_i)), with delta_J the "
    "measured max pairwise |J - J'| over the base-point re-solves; "
    "infinite when g.d_i is zero"
)

# Design amendment 1: the implicit surface ``s*(c)`` is only
# locally defined, and the B3 shakedown measured a unit-scale coil step
# throwing the inner solve onto a different Boozer branch
# (iota 0.1409 -> -0.0024, J 0.0143 -> 10.43) with every inner solve
# converging, so convergence alone rejects nothing. An accepted evaluation
# must stay on the anchor's branch: an inner solve whose iota moves more
# than this guard from the last accepted anchor is a failed evaluation and
# takes the shared rejection sentinel, identically in both lanes.
NESTED_LS_OUTER_IOTA_BRANCH_GUARD: Final[float] = 0.05
# Frozen per-host native OMP sweep set for F3 B37 banana-class work, and
# the outer sweep's interleaved repeat count. Jax-free single source: the
# claim driver's clock-owning parent must not import a device-initializing
# module to learn the contract set.
F3_B37_BANANA_OMP_CONTRACT_THREADS: Final[tuple[int, ...]] = (
    4,
    8,
    12,
    14,
    16,
    20,
    24,
    32,
)
NESTED_LS_OUTER_OMP_SWEEP_REPEATS: Final[int] = 2


class NestedLsPhysicsNewtonKwargs(TypedDict):
    constraint_weight: float
    tol: float
    maxiter: int
    stab: float
    verbose: bool
    weight_inv_modB: bool


class NestedLsBananaRunCodeOptions(TypedDict):
    verbose: bool
    newton_tol: float
    newton_maxiter: int
    bfgs_tol: float
    weight_inv_modB: bool


def nested_ls_physics_newton_kwargs() -> NestedLsPhysicsNewtonKwargs:
    """Keyword arguments for reconstruct-bar LS Newton on either lane."""

    return NestedLsPhysicsNewtonKwargs(
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        tol=NESTED_LS_NEWTON_TOL,
        maxiter=NESTED_LS_NEWTON_MAXITER,
        stab=NESTED_LS_NEWTON_STAB,
        verbose=False,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )


def nested_ls_banana_run_code_options() -> NestedLsBananaRunCodeOptions:
    """``BoozerSurface`` / ``BoozerSurfaceJAX`` options for banana ``run_code``.

    Banana Newton does not receive ``stab``; the method default is 0.
    """

    return NestedLsBananaRunCodeOptions(
        verbose=False,
        newton_tol=NESTED_LS_BANANA_NEWTON_TOL,
        newton_maxiter=NESTED_LS_BANANA_NEWTON_MAXITER,
        bfgs_tol=NESTED_LS_BANANA_BFGS_TOL,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )


def nested_ls_outer_fd0_step(coil_value: float) -> float:
    """The frozen FD-0 central-difference step for one coil DOF.

    One rule, one implementation: the probe records what this returns
    per direction rather than restating the arithmetic. This is where
    the descent ladder starts, not where it must stop.
    """

    scale = max(abs(float(coil_value)), NESTED_LS_OUTER_FD0_STEP_SCALE_FLOOR)
    return NESTED_LS_OUTER_FD0_STEP_RELATIVE * scale


def nested_ls_outer_fd0_minimum_step(
    scatter: float,
    directional_derivative: float,
) -> float:
    """The smallest step the measured ``J`` scatter licenses for one direction.

    ``NESTED_LS_OUTER_FD0_MIN_STEP_RULE`` in one implementation. A
    direction whose predicted derivative is zero cannot be gated
    relatively at all, so its floor is infinite and the ladder refuses
    every rung rather than descending into noise.
    """

    magnitude = abs(float(directional_derivative))
    if magnitude == 0.0:
        return float("inf")
    return (
        NESTED_LS_OUTER_FD0_NOISE_SAFETY
        * abs(float(scatter))
        / (2.0 * NESTED_LS_OUTER_FD0_REL_TOL * magnitude)
    )


NESTED_LS_OUTER_MAX_RESTARTS: Final[int] = 8
NESTED_LS_OUTER_FTOL_STALL_MESSAGE: Final[str] = (
    "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
)

_CandidateT = TypeVar("_CandidateT")


def nested_ls_outer_parameter_bytes(parameters: NDArray[np.float64]) -> bytes:
    """Return the exact little-endian float64 bytes of one outer point."""

    canonical = np.ascontiguousarray(parameters, dtype=np.dtype("<f8")).reshape(-1)
    return canonical.tobytes(order="C")


class NestedLsOuterAcceptWithoutCandidate(RuntimeError):
    """scipy accepted outer parameters no candidate was ever staged for.

    Control flow only: raised by :meth:`NestedLsOuterCandidateStore.accept`
    and caught at each lane's accepted-step callback seam so the child can
    publish its ledger, its committed incumbent and its telemetry before
    exiting nonzero. It never means the run recovered.

    Typed at the raise site rather than retyped by each caller, because
    ``accept`` reaches a second ``RuntimeError`` through
    :attr:`NestedLsOuterCandidateStore.committed` — an ``except RuntimeError``
    at the callback would publish an unprimed-store bug under this one's
    name. One class here also keeps the two lanes from inventing two.
    """

    def __init__(self, parameters: NDArray[np.float64]) -> None:
        super().__init__(
            "accepted outer parameters match neither the incumbent nor "
            "a pending candidate"
        )
        self.parameters = np.array(parameters, dtype=np.float64, copy=True)


class NestedLsOuterCandidateStore(Generic[_CandidateT]):
    """Keep trial candidates pending until scipy accepts their exact parameters.

    The first record must be the declared start point and becomes the initial
    incumbent immediately because scipy never calls its callback for ``x0``.
    Later records stay pending until :meth:`accept` matches callback bytes.
    """

    __slots__ = (
        "_committed",
        "_committed_parameter_bytes",
        "_pending",
        "_start_parameter_bytes",
    )

    def __init__(self, start_parameters: NDArray[np.float64]) -> None:
        self._start_parameter_bytes = nested_ls_outer_parameter_bytes(start_parameters)
        self._committed: _CandidateT | None = None
        self._committed_parameter_bytes: bytes | None = None
        self._pending: dict[bytes, _CandidateT] = {}

    @property
    def is_primed(self) -> bool:
        return self._committed is not None

    @property
    def committed(self) -> _CandidateT:
        candidate = self._committed
        if candidate is None:
            raise RuntimeError("the outer candidate store has no committed start")
        return candidate

    def record(
        self,
        parameters: NDArray[np.float64],
        candidate: _CandidateT,
    ) -> bool:
        """Prime ``x0`` or stage one later candidate; return whether it primed."""

        parameter_bytes = nested_ls_outer_parameter_bytes(parameters)
        if self._committed is None:
            if parameter_bytes != self._start_parameter_bytes:
                raise RuntimeError(
                    "the first feasible outer evaluation does not match x0"
                )
            self._committed = candidate
            self._committed_parameter_bytes = parameter_bytes
            return True
        self._pending[parameter_bytes] = candidate
        return False

    def accept(self, parameters: NDArray[np.float64]) -> _CandidateT:
        """Commit the pending candidate whose exact bytes match scipy's callback.

        Raises :class:`NestedLsOuterAcceptWithoutCandidate` when the accepted
        bytes are neither the incumbent's nor any staged candidate's.
        """

        parameter_bytes = nested_ls_outer_parameter_bytes(parameters)
        candidate = self._pending.get(parameter_bytes)
        if candidate is None:
            if parameter_bytes != self._committed_parameter_bytes:
                raise NestedLsOuterAcceptWithoutCandidate(parameters)
            candidate = self.committed
        self._committed = candidate
        self._committed_parameter_bytes = parameter_bytes
        self._pending.clear()
        return candidate

    def committed_matches(self, parameters: NDArray[np.float64]) -> bool:
        """Whether ``parameters`` are the exact committed outer point."""

        return self._committed_parameter_bytes == nested_ls_outer_parameter_bytes(
            parameters
        )


def nested_ls_outer_rejection_barrier(
    *,
    anchor_value: float,
    anchor_parameters: NDArray[np.float64],
    trial_parameters: NDArray[np.float64],
    scale: float,
) -> tuple[float, NDArray[np.float64]]:
    """Return the containment barrier and its exact derivative at one trial."""

    displacement = np.asarray(trial_parameters, dtype=np.float64) - np.asarray(
        anchor_parameters, dtype=np.float64
    )
    gradient = float(scale) * displacement
    value = float(anchor_value) + 0.5 * float(scale) * float(
        np.dot(displacement, displacement)
    )
    return value, gradient


#: What ``value_is_valid`` means, in one place, for both lanes and both
#: levels at which it is published (per evaluation row, and per restart
#: attempt beside scipy's ``fun``).
#:
#: ``True``  — the number IS the eight-term outer objective, and the
#:             ``grad_*`` norms beside it are its coil gradient.
#: ``False`` — the number is the containment barrier
#:             ``J_a + 0.5*mu*||c - c_a||^2`` and its derivative
#:             ``mu*(c - c_a)``: a real number in a real ledger row, priced
#:             off the committed anchor, but not a measurement of the
#:             objective at those coils.
#:
#: It is a DIFFERENT question from ``inner_feasible``, which reports whether
#: the inner solve landed on the anchor's branch within budget. The two
#: coincide today but could diverge if a future, separately certified coarse
#: inner tolerance feeds a line-search trial value: that row would be
#: inner-feasible while its number is a coarse surrogate. A consumer that
#: aggregates values across rows without reading this bit averages
#: surrogates into a physics figure, which is the failure the bit prevents.
NESTED_LS_OUTER_VALUE_IS_VALID_MEANING: Final[str] = (
    "true when the published value is the eight-term outer objective; "
    "false when it is the containment barrier priced off the committed anchor"
)


def nested_ls_outer_attempt_fun_is_objective(
    *,
    reported_parameters: NDArray[np.float64],
    last_evaluated_parameters: NDArray[np.float64],
    last_evaluation_value_is_valid: bool,
) -> bool:
    """Whether one attempt's scipy ``result.fun`` IS the outer objective.

    scipy restores ``result.x`` and ``result.jac`` to the incumbent when a
    line search rejects every step, but it does **not** restore
    ``result.fun``: that field keeps the last rejected trial's containment
    barrier. Both lanes publish scipy's raw datum and this bit beside it, so
    a consumer can tell an objective from a surrogate.

    This is a **provenance** check, not a value comparison. ``result.fun`` is
    whatever the last evaluation of the attempt returned, so it is the
    objective exactly when that evaluation published the objective and stood
    at the point the attempt reports. Comparing ``result.fun`` against the
    incumbent's stored value instead would be a coincidence test that the
    barrier can pass: ``J_a + 0.5*mu*||d||^2`` rounds to bitwise ``J_a`` for
    small ``||d||`` (measured: anchor 0.328125, ``||d|| = 1e-9``, scale 1.0),
    and dcsrch contracts toward exactly that regime, so a rejected trial near
    convergence would be stamped valid.
    """

    return bool(last_evaluation_value_is_valid) and nested_ls_outer_parameter_bytes(
        reported_parameters
    ) == nested_ls_outer_parameter_bytes(last_evaluated_parameters)


def nested_ls_predictor_trust_region(
    *,
    delta_surface: NDArray[np.float64],
    anchor_surface_dofs: NDArray[np.float64],
    ratio: float = NESTED_LS_PREDICTOR_TRUST_REGION_RATIO,
) -> tuple[NDArray[np.float64], float, float, float, bool]:
    """DESC ``tr_ratio``: scale the step to the bound, never reject it.

    Returns ``(applied, raw_norm, applied_norm, cap, scaled)``. The bound is
    ``ratio * ||s_anchor||_2``, and a step exactly at the bound is left
    untouched -- strict ``>`` triggers scaling, so the boundary is inside
    the region.

    Scaling rather than rejecting is DESC's rule and is deliberately kept:
    a predicted step that is too long is still pointing somewhere useful,
    and clipping keeps its direction. Whether to fall back to the bare
    anchor is a separate question this function does not answer. A nonfinite
    tangent is not a direction, so this pure arithmetic helper replaces it
    with a zero increment; the production caller detects it first and returns
    the original bare-anchor array.
    """

    delta = np.asarray(delta_surface, dtype=np.float64)
    cap = float(ratio) * float(np.linalg.norm(anchor_surface_dofs))
    raw_norm = float(np.linalg.norm(delta))
    if not np.all(np.isfinite(delta)):
        return np.zeros_like(delta, dtype=np.float64), raw_norm, 0.0, cap, False
    if raw_norm > cap:
        applied = delta * (cap / raw_norm)
        return applied, raw_norm, float(np.linalg.norm(applied)), cap, True
    return np.array(delta, dtype=np.float64, copy=True), raw_norm, raw_norm, cap, False


def nested_ls_predictor_arm(
    *,
    bare_gradient_l2: float,
    predicted_gradient_l2: float,
) -> str:
    """Envelope-gradient fallback: bare anchor when the prediction is worse.

    Ours, not DESC's. A tie keeps the prediction, and that tie-break is
    load-bearing rather than arbitrary: at ``Δc = 0`` the predicted step is
    zero, the two starts are the same vector, the two envelope gradients are
    bitwise equal, and keeping the prediction makes a predictor-ON run
    reproduce a predictor-OFF run exactly at an unmoved point. Preferring
    the bare anchor on a tie would break that identity for no gain.
    """

    predicted = float(predicted_gradient_l2)
    if not np.isfinite(predicted) or predicted > float(bare_gradient_l2):
        return NESTED_LS_PREDICTOR_ARM_BARE
    return NESTED_LS_PREDICTOR_ARM_PREDICTED


def _nested_ls_inner_policy_record(
    *,
    policy: str,
    solver_family: str,
    solver_sequence: tuple[str, ...],
    stages: tuple[tuple[str, dict[str, float | int]], ...],
) -> dict[str, object]:
    """Build the common immutable-shaped record portion for an inner lane."""

    return {
        "policy": policy,
        "solver_family": solver_family,
        "solver_sequence": list(solver_sequence),
        "stages": {name: dict(options) for name, options in stages},
    }


def nested_ls_jax_inner_policy(
    *,
    ift_stab: float,
    inner_substep_legs: tuple[int, ...],
    inner_predictor: bool,
) -> dict[str, object]:
    """Publish the active reduced-Schur Newton lane and its predictor settings."""

    legs = tuple(int(leg) for leg in inner_substep_legs)
    stock = legs == (1,) and not bool(inner_predictor)
    return {
        **_nested_ls_inner_policy_record(
            policy=NESTED_LS_JAX_INNER_POLICY_NAME,
            solver_family="reduced_schur_newton",
            solver_sequence=("reduced_schur_newton",),
            stages=(
                (
                    "reduced_schur_newton",
                    {
                        "stab": float(ift_stab),
                        "tol": float(NESTED_LS_NEWTON_TOL),
                        "maxiter": int(NESTED_LS_NEWTON_MAXITER),
                    },
                ),
            ),
        ),
        "inner_substep_legs": list(legs),
        "inner_substep_enabled": legs != (1,),
        "inner_predictor_enabled": bool(inner_predictor),
        "predictor_trust_region_ratio": (
            float(NESTED_LS_PREDICTOR_TRUST_REGION_RATIO) if inner_predictor else None
        ),
        "coarse_tier_diagnostic_residual": float(NESTED_LS_NEWTON_COARSE_TOL),
        "coarse_tier_honoured": None,
        "trajectory_is_stock": stock,
    }


def nested_ls_native_inner_policy() -> dict[str, object]:
    """Publish the native banana ``run_code`` BFGS-then-Newton lane."""

    return _nested_ls_inner_policy_record(
        policy=NESTED_LS_NATIVE_INNER_POLICY_NAME,
        solver_family="banana_run_code",
        solver_sequence=("BFGS", "Newton"),
        stages=(
            ("BFGS", {"tol": float(NESTED_LS_BANANA_BFGS_TOL)}),
            (
                "Newton",
                {
                    "stab": float(NESTED_LS_BANANA_NEWTON_STAB),
                    "tol": float(NESTED_LS_BANANA_NEWTON_TOL),
                    "maxiter": int(NESTED_LS_BANANA_NEWTON_MAXITER),
                },
            ),
        ),
    )


def nested_ls_inner_substep_points(
    *,
    anchor_coil_dofs: NDArray[np.float64],
    trial_coil_dofs: NDArray[np.float64],
    legs: int,
) -> tuple[NDArray[np.float64], ...]:
    """The coil points one sub-stepped inner solve walks through.

    ``legs`` equal fractions of ``Δc = trial − anchor``, returned as the
    points to solve AT — so the tuple has ``legs`` entries and never
    includes the anchor itself, which is already solved.

    The last entry is the trial's own bytes, copied, never a reconstruction
    ``anchor + Δc``. This matters because the outer objective and its
    gradient are evaluated at the coils the CALLER named, so a solve that
    landed at a reconstructed neighbour would publish ``s*(c′)`` under the
    label ``s*(c)``. The intermediate points are scratch and may be anything
    on the segment; the endpoint may not.

    Be precise about why this is a copy rather than a computation: NOT
    because the reconstruction is observably wrong. Probing 120 000 random
    pairs across six magnitude regimes (anchor 1e-8 to 1e6, displacement
    1e-9 to 1e8) found ZERO cases where ``a + (t - a) != t`` in binary64 —
    the subtraction is exact in the Sterbenz range and rounds back
    elsewhere. The copy is here so that exactness is a property of this
    function rather than of a floating-point regularity that holds on every
    input anyone has tried. A guarantee and an empirical regularity are
    different things to build a certification on.

    ``legs = 1`` returns exactly ``(trial,)``, which is the undivided step.
    That is what makes the first rung of the ladder free: a lane with
    sub-stepping enabled reproduces the un-sub-stepped trajectory bitwise
    whenever the undivided step succeeds.
    """

    if int(legs) < 1:
        raise ValueError(f"legs must be at least 1; got {legs!r}.")
    anchor = np.asarray(anchor_coil_dofs, dtype=np.float64).reshape(-1)
    trial = np.asarray(trial_coil_dofs, dtype=np.float64).reshape(-1)
    if anchor.shape != trial.shape:
        raise ValueError(
            "anchor and trial coil blocks must have the same shape; got "
            f"{anchor.shape} and {trial.shape}."
        )
    delta = trial - anchor
    points = [
        anchor + delta * (float(index) / float(legs)) for index in range(1, int(legs))
    ]
    points.append(np.array(trial, dtype=np.float64, copy=True))
    return tuple(points)


def nested_ls_newton_exit_status(
    *,
    persisted: bool,
    finite_iterate: bool,
    reduced_gradient_l2: float,
    tol: float,
    coarse_tol: float = NESTED_LS_NEWTON_COARSE_TOL,
) -> str:
    """Classify one inner Schur-Newton walk as converged/coarse/failed.

    ``persisted`` gates everything. When the walk does not persist, the
    result carries the START surface and start gradient, so
    ``reduced_gradient_l2`` describes a point the result does not report;
    classifying it by that norm would label the caller's own input as a
    solve outcome. A non-persisted walk is ``failed``, full stop.

    The bands are ordered so ``converged`` is decided first. That keeps the
    classification correct even if a caller passes ``coarse_tol < tol``:
    the coarse band is then empty and the status degrades to the two-valued
    answer ``success`` already gives, rather than demoting a converged walk.
    """

    if not persisted or not finite_iterate:
        return NESTED_LS_NEWTON_EXIT_FAILED
    if float(reduced_gradient_l2) <= float(tol):
        return NESTED_LS_NEWTON_EXIT_CONVERGED
    if float(reduced_gradient_l2) <= float(coarse_tol):
        return NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED
    return NESTED_LS_NEWTON_EXIT_FAILED


def nested_ls_outer_ftol_zero_stop(*, ftol: float, message: str) -> bool:
    """Whether scipy claimed FTOL convergence while FTOL was disabled."""

    return float(ftol) == 0.0 and "RELATIVE REDUCTION OF F" in str(message).upper()


#: The only scipy L-BFGS-B stop codes whose endpoint is a completed
#: optimization: 0 (a convergence test fired) and 1 (a declared budget ran
#: out). Everything else means the run was cut short — 2 is an abandoned line
#: search or a halting callback, and 99 is ``minimize``'s rewrite of a
#: ``StopIteration`` raised from the callback
#: (``scipy/optimize/_minimize.py:823-826``, scipy 1.17.1). Membership is an
#: allow-list on purpose: a status this contract has never seen must fail
#: closed, not inherit "publishable" by not being the one code we thought to
#: exclude.
NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES: Final[frozenset[int]] = frozenset({0, 1})


def nested_ls_outer_endpoint_success(
    *,
    endpoint_matches: bool,
    ftol: float,
    status: int,
    message: str,
) -> bool:
    """Judge a child endpoint under the shared transaction and stop contract.

    Three independent conditions, all required. The optimizer's final iterate
    must be the committed transaction point (``endpoint_matches``); the stop
    code must be one the run could legitimately end on
    (:data:`NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES`); and the stop must not
    be the FTOL fiction that ``ftol=0`` makes impossible by construction.
    """

    return (
        bool(endpoint_matches)
        and int(status) in NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES
        and not nested_ls_outer_ftol_zero_stop(
            ftol=ftol,
            message=message,
        )
    )


#: The one published name for the one way an outer child can die at the
#: accepted-step callback seam: scipy announced parameters that are neither
#: the incumbent nor any staged candidate. It lives here, beside
#: ``nested_ls_outer_restart_reason``'s ``abnormal_line_search`` /
#: ``false_ftol_stall``, because this module is the JAX-free single source
#: both lanes import at module level — a fault vocabulary that lived in one
#: lane would make the other lane's copy a deferred import away from drift.
NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON: Final[str] = "accept_without_candidate"


def nested_ls_outer_restart_reason(
    *,
    ftol: float,
    status: int,
    message: str,
) -> str | None:
    """Classify one scipy L-BFGS-B stop as restartable, or None if terminal.

    Two stop classes may consume less than the iteration budget without the
    outer problem being finished, both produced by the shared rejection
    sentinel's interaction with dcsrch (measured at B37):

    - ``abnormal_line_search``: scipy status 2 with an ``ABNORMAL`` message —
      the line search abandoned outright.
    - ``false_ftol_stall``: scipy reports FTOL convergence while the recorded
      policy set ``ftol=0``. That policy permits no relative-reduction stop;
      convergence belongs to the separately declared projected-gradient gate.

    Both outer children call this one classifier so the lanes cannot drift.
    """

    if int(status) == 2 and str(message).startswith("ABNORMAL"):
        return "abnormal_line_search"
    if int(status) == 0 and nested_ls_outer_ftol_zero_stop(
        ftol=ftol,
        message=message,
    ):
        return "false_ftol_stall"
    return None


__all__ = [
    "NESTED_LS_BANANA_BFGS_TOL",
    "NESTED_LS_BANANA_NEWTON_MAXITER",
    "NESTED_LS_BANANA_NEWTON_STAB",
    "NESTED_LS_BANANA_NEWTON_TOL",
    "NESTED_LS_BANANA_USES_BFGS_THEN_NEWTON",
    "NESTED_LS_CONSTRAINT_WEIGHT",
    "NESTED_LS_GATE6_AGGREGATION",
    "NESTED_LS_GATE6_CLAIM_REPEATS",
    "NESTED_LS_GATE6_IOTA_G_TOL",
    "NESTED_LS_GATE6_NATIVE_OMP_THREADS",
    "NESTED_LS_INNER_SUBSTEP_LEGS",
    "NESTED_LS_JAX_INNER_POLICY_NAME",
    "NESTED_LS_JAX_INNER_STAB",
    "NESTED_LS_LABEL",
    "NESTED_LS_NEWTON_COARSE_TOL",
    "NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED",
    "NESTED_LS_NEWTON_EXIT_CONVERGED",
    "NESTED_LS_NEWTON_EXIT_FAILED",
    "NESTED_LS_NEWTON_EXIT_STATUSES",
    "NESTED_LS_NEWTON_MAXITER",
    "NESTED_LS_NEWTON_STAB",
    "NESTED_LS_NEWTON_TOL",
    "NESTED_LS_NATIVE_INNER_POLICY_NAME",
    "NESTED_LS_OPTIMIZE_G",
    "NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON",
    "NESTED_LS_OUTER_FD0_DIRECTIONS",
    "NESTED_LS_OUTER_FD0_MAX_HALVINGS",
    "NESTED_LS_OUTER_FD0_MIN_STEP_RULE",
    "NESTED_LS_OUTER_FD0_NOISE_SAFETY",
    "NESTED_LS_OUTER_FD0_REL_TOL",
    "NESTED_LS_OUTER_FD0_SCATTER_REPEATS",
    "NESTED_LS_OUTER_FD0_SCHEMA",
    "NESTED_LS_OUTER_FD0_STEP_HALVING",
    "NESTED_LS_OUTER_FD0_STEP_RELATIVE",
    "NESTED_LS_OUTER_FD0_STEP_RULE",
    "NESTED_LS_OUTER_FD0_STEP_SCALE_FLOOR",
    "F3_B37_BANANA_OMP_CONTRACT_THREADS",
    "NESTED_LS_OUTER_IOTA_BRANCH_GUARD",
    "NESTED_LS_OUTER_JAX_CHILD_SCHEMA",
    "NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA",
    "NESTED_LS_OUTER_REJUDGE_SCHEMA",
    "NESTED_LS_OUTER_VALUE_IS_VALID_MEANING",
    "NESTED_LS_OUTER_FTOL_STALL_MESSAGE",
    "NESTED_LS_OUTER_MAX_RESTARTS",
    "NESTED_LS_OUTER_OMP_SWEEP_REPEATS",
    "NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES",
    "NESTED_LS_PHYSICS_BAR",
    "NESTED_LS_PREDICTOR_ARM_BARE",
    "NESTED_LS_PREDICTOR_ARM_PREDICTED",
    "NESTED_LS_PREDICTOR_TRUST_REGION_RATIO",
    "NESTED_LS_REDUCTION_MODE",
    "NESTED_LS_TIMING_BAR",
    "NESTED_LS_WEIGHT_INV_MODB",
    "NestedLsBananaRunCodeOptions",
    "NestedLsOuterAcceptWithoutCandidate",
    "NestedLsOuterCandidateStore",
    "NestedLsPhysicsNewtonKwargs",
    "nested_ls_banana_run_code_options",
    "nested_ls_inner_substep_points",
    "nested_ls_jax_inner_policy",
    "nested_ls_native_inner_policy",
    "nested_ls_newton_exit_status",
    "nested_ls_outer_attempt_fun_is_objective",
    "nested_ls_outer_endpoint_success",
    "nested_ls_outer_ftol_zero_stop",
    "nested_ls_outer_fd0_minimum_step",
    "nested_ls_outer_parameter_bytes",
    "nested_ls_outer_rejection_barrier",
    "nested_ls_outer_restart_reason",
    "nested_ls_outer_fd0_step",
    "nested_ls_physics_newton_kwargs",
    "nested_ls_predictor_arm",
    "nested_ls_predictor_trust_region",
]
