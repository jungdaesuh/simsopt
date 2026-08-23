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

from typing import Final, TypedDict

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
    "NESTED_LS_OUTER_OMP_SWEEP_REPEATS",
    "NESTED_LS_PHYSICS_BAR",
    "NESTED_LS_REDUCTION_MODE",
    "NESTED_LS_TIMING_BAR",
    "NESTED_LS_WEIGHT_INV_MODB",
    "NestedLsBananaRunCodeOptions",
    "NestedLsPhysicsNewtonKwargs",
    "nested_ls_banana_run_code_options",
    "nested_ls_outer_fd0_minimum_step",
    "nested_ls_outer_fd0_step",
    "nested_ls_physics_newton_kwargs",
]
