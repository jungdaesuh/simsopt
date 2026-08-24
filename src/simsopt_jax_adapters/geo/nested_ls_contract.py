"""Canonical nested Boozer-LS bars: reconstruct physics vs banana timing.

Gate 0 of the reduced nested-LS JAX GPU track. These knobs are not F3's
flat-675 fused L-BFGS-B campaign and do not inherit its 7.70× claim.

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

# Reconstruct / physics bar (C++ LS Newton judge, 2026-08-20).
NESTED_LS_NEWTON_STAB: Final[float] = 1.0e-4
NESTED_LS_NEWTON_TOL: Final[float] = 1.0e-13
NESTED_LS_NEWTON_MAXITER: Final[int] = 10
NESTED_LS_REDUCTION_MODE: Final[str] = "cpu_ordered"

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
NESTED_LS_OUTER_JAX_CHILD_SCHEMA: Final[str] = "nested-ls-outer-jax-child.v5"
NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA: Final[str] = "nested-ls-outer-native-child.v4"
NESTED_LS_OUTER_REJUDGE_SCHEMA: Final[str] = "nested-ls-outer-rejudge.v1"

# Gate FD-0 of the eight-term outer charter
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

# Charter Amendment 3 (2026-08-23): the fixed two-rung ladder was the
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

# Charter Amendment 1 (2026-08-22): the implicit surface ``s*(c)`` is only
# locally defined, and the B3 shakedown measured a unit-scale coil step
# throwing the inner solve onto a different Boozer branch
# (iota 0.1409 -> -0.0024, J 0.0143 -> 10.43) with every inner solve
# converging, so convergence alone rejects nothing. An accepted evaluation
# must stay on the anchor's branch: an inner solve whose iota moves more
# than this guard from the last accepted anchor is a failed evaluation and
# takes the sealed rejection sentinel, identically in both lanes.
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
#: coincide today and diverge at Phase 4 of
#: ``docs/nested_ls_upgrade_implementation_plan.md``, where a licensed coarse
#: inner tolerance may feed a line-search trial value: that row is
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
    outer problem being finished, both produced by the sealed rejection
    sentinel's interaction with dcsrch (measured at B37, 2026-08-24):

    - ``abnormal_line_search``: scipy status 2 with an ``ABNORMAL`` message —
      the line search abandoned outright.
    - ``false_ftol_stall``: scipy reports FTOL convergence while the sealed
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
    "NESTED_LS_LABEL",
    "NESTED_LS_NEWTON_MAXITER",
    "NESTED_LS_NEWTON_STAB",
    "NESTED_LS_NEWTON_TOL",
    "NESTED_LS_OPTIMIZE_G",
    "NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON",
    "NESTED_LS_OUTER_FD0_DIRECTIONS",
    "NESTED_LS_OUTER_FD0_MAX_HALVINGS",
    "NESTED_LS_OUTER_FD0_MIN_STEP_RULE",
    "NESTED_LS_OUTER_FD0_NOISE_SAFETY",
    "NESTED_LS_OUTER_FD0_REL_TOL",
    "NESTED_LS_OUTER_FD0_SCATTER_REPEATS",
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
    "NESTED_LS_REDUCTION_MODE",
    "NESTED_LS_TIMING_BAR",
    "NESTED_LS_WEIGHT_INV_MODB",
    "NestedLsBananaRunCodeOptions",
    "NestedLsOuterAcceptWithoutCandidate",
    "NestedLsOuterCandidateStore",
    "NestedLsPhysicsNewtonKwargs",
    "nested_ls_banana_run_code_options",
    "nested_ls_outer_attempt_fun_is_objective",
    "nested_ls_outer_endpoint_success",
    "nested_ls_outer_ftol_zero_stop",
    "nested_ls_outer_fd0_minimum_step",
    "nested_ls_outer_parameter_bytes",
    "nested_ls_outer_rejection_barrier",
    "nested_ls_outer_restart_reason",
    "nested_ls_outer_fd0_step",
    "nested_ls_physics_newton_kwargs",
]
