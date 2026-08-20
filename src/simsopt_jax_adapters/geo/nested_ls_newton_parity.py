"""Compare native C++ Boozer LS Newton to ``BoozerSurfaceJAX`` on one state.

The locked execution is JAX (cpu-ordered ``J_LS`` / ``∇J_LS``, SciPy Newton)
versus C++. It is the nested-LS operator, not a fused GPU campaign.

This is the nested-LS inner problem, not flat-675 QR-in-``J`` and not BoozerExact.
The decision vector is ``[surface_dofs, iota, G]``. Stationarity is
``||∇J_LS||_2``, not ``||r||``. Native Newton's returned ``residual`` is that
gradient; JAX's ``residual`` is the long LS vector and must not be compared
to it.

Knobs match the 2026-08-20 reconstruct judge (constraint weight 1, free ``G``,
``weight_inv_modB``, ``stab=1e-4``, ``tol=1e-13``), not banana ``run_code``
defaults (no ``stab``, looser ``tol``, BFGS then Newton).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from simsopt.geo import BoozerSurface
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

NESTED_LS_CONSTRAINT_WEIGHT: Final[float] = 1.0
NESTED_LS_WEIGHT_INV_MODB: Final[bool] = True
NESTED_LS_NEWTON_STAB: Final[float] = 1.0e-4
NESTED_LS_NEWTON_TOL: Final[float] = 1.0e-13
NESTED_LS_NEWTON_MAXITER: Final[int] = 10
NESTED_LS_REDUCTION_MODE: Final[str] = "cpu_ordered"


@dataclass(frozen=True, slots=True)
class NestedLsPenaltyEvaluation:
    """Scalar ``J_LS`` and its gradient on one packed state."""

    objective: float
    gradient: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NestedLsPenaltyPair:
    """Native and JAX evaluations of the same packed LS state."""

    native: NestedLsPenaltyEvaluation
    jax: NestedLsPenaltyEvaluation


@dataclass(frozen=True, slots=True)
class NestedLsNewtonLane:
    """One LS Newton polish on frozen coils."""

    success: bool
    iteration_count: int
    iota: float
    G: float
    surface_dofs: NDArray[np.float64]
    gradient: NDArray[np.float64]
    objective: float
    coil_delta_inf: float


@dataclass(frozen=True, slots=True)
class NestedLsNewtonPair:
    """Native and JAX Newton polishes from the same packed start."""

    native: NestedLsNewtonLane
    jax: NestedLsNewtonLane


def pack_nested_ls_decision(
    surface_dofs: object,
    iota: float,
    G: float,
) -> NDArray[np.float64]:
    """Pack ``[surface_dofs, iota, G]`` as float64."""

    surface = np.asarray(surface_dofs, dtype=np.float64).reshape(-1)
    return np.concatenate((surface, [float(iota), float(G)]))


def nested_ls_gradient_from_newton_result(
    result: Mapping[str, object],
) -> NDArray[np.float64]:
    """Return ``∇J_LS`` from ``jacobian``.

    Native Newton also aliases this vector as ``residual``. JAX Newton stores
    the long LS residual in ``residual`` and the gradient in ``jacobian``.
    """

    return np.array(result["jacobian"], dtype=np.float64, copy=True).reshape(-1)


def jax_newton_residual_is_long_vector(
    result: Mapping[str, object],
    *,
    decision_size: int,
) -> bool:
    """True when JAX ``residual`` is the long LS vector, not ``∇J_LS``."""

    residual = np.asarray(result["residual"], dtype=np.float64).reshape(-1)
    jacobian = nested_ls_gradient_from_newton_result(result)
    return residual.size != decision_size and jacobian.size == decision_size


def evaluate_nested_ls_penalty_pair(
    *,
    native_boozer: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
    decision: object,
) -> NestedLsPenaltyPair:
    """Evaluate ``J_LS`` and ``∇J_LS`` on both lanes at one packed state."""

    packed = np.asarray(decision, dtype=np.float64).reshape(-1)
    native_objective, native_gradient = (
        native_boozer.boozer_penalty_constraints_vectorized(
            packed,
            derivatives=1,
            constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
            optimize_G=True,
            weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
        )
    )
    return NestedLsPenaltyPair(
        native=NestedLsPenaltyEvaluation(
            objective=float(native_objective),
            gradient=np.array(native_gradient, dtype=np.float64, copy=True).reshape(-1),
        ),
        jax=_jax_cpu_ordered_value_and_grad(jax_boozer, packed),
    )


def _coil_coordinates(biotsavart) -> NDArray[np.float64]:
    return np.array(biotsavart.x, dtype=np.float64, copy=True)


def _jax_cpu_ordered_value_and_grad(
    jax_boozer: BoozerSurfaceJAX,
    packed: NDArray[np.float64],
) -> NestedLsPenaltyEvaluation:
    jax_objective_fn = jax_boozer._make_penalty_objective_with(
        True,
        NESTED_LS_WEIGHT_INV_MODB,
        NESTED_LS_CONSTRAINT_WEIGHT,
        boozer_reduction_mode=NESTED_LS_REDUCTION_MODE,
        decision_split_mode="reverse",
    )
    jax_value, jax_gradient = jax.value_and_grad(jax_objective_fn)(
        jnp.asarray(packed, dtype=jnp.float64)
    )
    return NestedLsPenaltyEvaluation(
        objective=float(jax.device_get(jax_value)),
        gradient=np.array(
            jax.device_get(jax_gradient), dtype=np.float64, copy=True
        ).reshape(-1),
    )


def _run_native_newton(
    native_boozer: BoozerSurface,
    *,
    iota: float,
    G: float,
) -> NestedLsNewtonLane:
    coil_before = _coil_coordinates(native_boozer.biotsavart)
    native_boozer.need_to_run_code = True
    result = native_boozer.minimize_boozer_penalty_constraints_newton(
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        iota=float(iota),
        G=float(G),
        tol=NESTED_LS_NEWTON_TOL,
        maxiter=NESTED_LS_NEWTON_MAXITER,
        stab=NESTED_LS_NEWTON_STAB,
        verbose=False,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    iota_out = float(result["iota"])
    g_out = float(result["G"])
    surface_dofs = np.array(
        native_boozer.surface.get_dofs(), dtype=np.float64, copy=True
    )
    packed = pack_nested_ls_decision(surface_dofs, iota_out, g_out)
    objective, gradient = native_boozer.boozer_penalty_constraints_vectorized(
        packed,
        derivatives=1,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        optimize_G=True,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    coil_after = _coil_coordinates(native_boozer.biotsavart)
    return NestedLsNewtonLane(
        success=bool(result["success"]),
        iteration_count=int(result["iter"]),
        iota=iota_out,
        G=g_out,
        surface_dofs=surface_dofs,
        gradient=np.array(gradient, dtype=np.float64, copy=True).reshape(-1),
        objective=float(objective),
        coil_delta_inf=float(np.linalg.norm(coil_after - coil_before, ord=np.inf)),
    )


def _run_jax_newton(
    jax_boozer: BoozerSurfaceJAX,
    *,
    iota: float,
    G: float,
) -> NestedLsNewtonLane:
    coil_before = _coil_coordinates(jax_boozer.biotsavart)
    jax_boozer.need_to_run_code = True
    result = jax_boozer.minimize_boozer_penalty_constraints_newton(
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        iota=float(iota),
        G=float(G),
        tol=NESTED_LS_NEWTON_TOL,
        maxiter=NESTED_LS_NEWTON_MAXITER,
        stab=NESTED_LS_NEWTON_STAB,
        verbose=False,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    iota_out = float(result["iota"])
    g_out = float(result["G"])
    surface_dofs = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    packed = pack_nested_ls_decision(surface_dofs, iota_out, g_out)
    evaluation = _jax_cpu_ordered_value_and_grad(jax_boozer, packed)
    coil_after = _coil_coordinates(jax_boozer.biotsavart)
    return NestedLsNewtonLane(
        success=bool(result["success"]),
        iteration_count=int(result["iter"]),
        iota=iota_out,
        G=g_out,
        surface_dofs=surface_dofs,
        gradient=evaluation.gradient,
        objective=evaluation.objective,
        coil_delta_inf=float(np.linalg.norm(coil_after - coil_before, ord=np.inf)),
    )


def run_nested_ls_newton_pair(
    *,
    native_boozer: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
    iota: float,
    G: float,
) -> NestedLsNewtonPair:
    """Polish both lanes from the same ``(ι, G)`` with reconstruct Newton knobs."""

    return NestedLsNewtonPair(
        native=_run_native_newton(native_boozer, iota=iota, G=G),
        jax=_run_jax_newton(jax_boozer, iota=iota, G=G),
    )


def assert_nested_ls_penalty_pair(pair: NestedLsPenaltyPair) -> None:
    """Fail unless JAX ``J_LS`` and ``∇J_LS`` match native at the packed state."""

    value_tol = parity_ladder_tolerances("direct_kernel")
    grad_tol = parity_ladder_tolerances("ls_wrapper_gradient")
    if pair.native.gradient.shape != pair.jax.gradient.shape:
        raise AssertionError(
            "nested-LS gradient shapes differ: "
            f"native {pair.native.gradient.shape} vs JAX {pair.jax.gradient.shape}."
        )
    np.testing.assert_allclose(
        pair.jax.objective,
        pair.native.objective,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="nested-LS J_LS mismatch",
    )
    np.testing.assert_allclose(
        pair.jax.gradient,
        pair.native.gradient,
        rtol=float(grad_tol["rtol"]),
        atol=float(grad_tol["atol"]),
        err_msg="nested-LS ∇J_LS mismatch",
    )


def assert_nested_ls_newton_pair(
    pair: NestedLsNewtonPair,
    *,
    require_success: bool = True,
) -> None:
    """Fail unless JAX Newton matches native on the reconstruct stationarity objects."""

    if pair.native.coil_delta_inf != 0.0 or pair.jax.coil_delta_inf != 0.0:
        raise AssertionError(
            "nested-LS Newton moved coil DOFs: "
            f"native {pair.native.coil_delta_inf} JAX {pair.jax.coil_delta_inf}."
        )
    if pair.native.success != pair.jax.success:
        raise AssertionError(
            "nested-LS Newton success mismatch: "
            f"native {pair.native.success} JAX {pair.jax.success}."
        )
    if require_success and pair.native.success is not True:
        raise AssertionError(
            "nested-LS Newton did not reach ||∇J_LS||_2 ≤ "
            f"{NESTED_LS_NEWTON_TOL}: native success={pair.native.success}."
        )
    value_tol = parity_ladder_tolerances("direct_kernel")
    grad_tol = parity_ladder_tolerances("ls_wrapper_gradient")
    np.testing.assert_allclose(
        pair.jax.iota,
        pair.native.iota,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="nested-LS Newton iota mismatch",
    )
    np.testing.assert_allclose(
        pair.jax.G,
        pair.native.G,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="nested-LS Newton G mismatch",
    )
    np.testing.assert_allclose(
        pair.jax.surface_dofs,
        pair.native.surface_dofs,
        rtol=float(grad_tol["rtol"]),
        atol=float(grad_tol["atol"]),
        err_msg="nested-LS Newton surface-dof mismatch",
    )
    np.testing.assert_allclose(
        pair.jax.objective,
        pair.native.objective,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="nested-LS Newton J_LS mismatch",
    )
    np.testing.assert_allclose(
        pair.jax.gradient,
        pair.native.gradient,
        rtol=float(grad_tol["rtol"]),
        atol=float(grad_tol["atol"]),
        err_msg="nested-LS Newton ∇J_LS mismatch",
    )
