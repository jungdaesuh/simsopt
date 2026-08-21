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


__all__ = [
    "NESTED_LS_BANANA_BFGS_TOL",
    "NESTED_LS_BANANA_NEWTON_MAXITER",
    "NESTED_LS_BANANA_NEWTON_STAB",
    "NESTED_LS_BANANA_NEWTON_TOL",
    "NESTED_LS_BANANA_USES_BFGS_THEN_NEWTON",
    "NESTED_LS_CONSTRAINT_WEIGHT",
    "NESTED_LS_LABEL",
    "NESTED_LS_NEWTON_MAXITER",
    "NESTED_LS_NEWTON_STAB",
    "NESTED_LS_NEWTON_TOL",
    "NESTED_LS_OPTIMIZE_G",
    "NESTED_LS_PHYSICS_BAR",
    "NESTED_LS_REDUCTION_MODE",
    "NESTED_LS_TIMING_BAR",
    "NESTED_LS_WEIGHT_INV_MODB",
    "NestedLsBananaRunCodeOptions",
    "NestedLsPhysicsNewtonKwargs",
    "nested_ls_banana_run_code_options",
    "nested_ls_physics_newton_kwargs",
]
