"""Pure JAX QFM objective and solver helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._math_utils import as_jax_float64, as_runtime_float64
from .field import grouped_biot_savart_A_from_spec, grouped_biot_savart_B_from_spec
from .specs import surface_spec_kind
from .surface_fourier import (
    surface_xyz_fourier_gamma_from_spec,
    surface_xyz_fourier_gammadash1_from_spec,
    surface_xyz_fourier_gammadash2_from_spec,
    surface_xyz_fourier_volume_from_spec,
    surface_xyz_tensor_fourier_gamma_from_spec,
    surface_xyz_tensor_fourier_gammadash1_from_spec,
    surface_xyz_tensor_fourier_gammadash2_from_spec,
    surface_xyz_tensor_fourier_volume_from_spec,
)
from .surface_rzfourier import (
    surface_rz_fourier_gamma_from_dofs,
    surface_rz_fourier_gammadash1_from_dofs,
    surface_rz_fourier_gammadash2_from_dofs,
    surface_rz_fourier_volume_from_dofs,
)

__all__ = [
    "QfmAugmentedLagrangianInfo",
    "QfmPenaltySolveInfo",
    "qfm_augmented_lagrangian_solve_jax",
    "qfm_label_jax_from_dofs",
    "qfm_penalty_jax_from_dofs",
    "qfm_penalty_solve_jax",
    "qfm_penalty_value_and_grad_jax_from_dofs",
    "qfm_residual_jax_from_dofs",
]


class QfmPenaltySolveInfo(NamedTuple):
    """Device-resident metadata for a QFM penalty solve."""

    success: jax.Array
    status: jax.Array
    fun: jax.Array
    gradient: jax.Array
    nit: jax.Array
    nfev: jax.Array
    njev: jax.Array
    label_value: jax.Array
    label_residual: jax.Array
    qfm_value: jax.Array
    penalty_value: jax.Array


class QfmAugmentedLagrangianInfo(NamedTuple):
    """Device-resident metadata for a QFM augmented-Lagrangian solve."""

    success: jax.Array
    status: jax.Array
    fun: jax.Array
    gradient: jax.Array
    nit: jax.Array
    nfev: jax.Array
    njev: jax.Array
    label_value: jax.Array
    label_residual: jax.Array
    qfm_value: jax.Array
    augmented_value: jax.Array
    multiplier: jax.Array
    penalty_weight: jax.Array


class _BFGSResult(NamedTuple):
    x: jax.Array
    success: jax.Array
    status: jax.Array
    fun: jax.Array
    jac: jax.Array
    nit: jax.Array
    nfev: jax.Array
    njev: jax.Array


class _BFGSState(NamedTuple):
    x: jax.Array
    fun: jax.Array
    grad: jax.Array
    hess_inv: jax.Array
    nit: jax.Array
    nfev: jax.Array
    njev: jax.Array
    status: jax.Array


class _QfmMetrics(NamedTuple):
    qfm_value: jax.Array
    label_value: jax.Array
    label_residual: jax.Array


_BFGS_STATUS_RUNNING = 1
_BFGS_STATUS_SUCCESS = 0
_BFGS_STATUS_MAXITER = 2
_BFGS_STATUS_LINE_SEARCH_FAILED = 3
_BFGS_STATUS_NOT_DESCENT = 4
_BFGS_LINE_SEARCH_MAXITER = 20


def _surface_spec_with_dofs(spec, dofs: object):
    return replace(spec, dofs=as_jax_float64(dofs))


def _surface_gamma_tangents_from_dofs(spec, dofs: object):
    dofs = as_jax_float64(dofs)
    kind = surface_spec_kind(spec)
    if kind == "rz_fourier":
        return (
            surface_rz_fourier_gamma_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash1_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash2_from_dofs(spec, dofs),
        )
    spec_with_dofs = _surface_spec_with_dofs(spec, dofs)
    if kind == "xyz_fourier":
        return (
            surface_xyz_fourier_gamma_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash1_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash2_from_spec(spec_with_dofs),
        )
    if kind == "xyz_tensor_fourier":
        return (
            surface_xyz_tensor_fourier_gamma_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash1_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash2_from_spec(spec_with_dofs),
        )
    raise TypeError(f"Unsupported surface spec kind {kind!r}.")


def _surface_volume_from_dofs(spec, dofs: object):
    dofs = as_jax_float64(dofs)
    kind = surface_spec_kind(spec)
    if kind == "rz_fourier":
        return surface_rz_fourier_volume_from_dofs(spec, dofs)
    spec_with_dofs = _surface_spec_with_dofs(spec, dofs)
    if kind == "xyz_fourier":
        return surface_xyz_fourier_volume_from_spec(spec_with_dofs)
    if kind == "xyz_tensor_fourier":
        return surface_xyz_tensor_fourier_volume_from_spec(spec_with_dofs)
    raise TypeError(f"Unsupported surface spec kind {kind!r}.")


def _surface_normal_from_tangents(gammadash1: object, gammadash2: object):
    return jnp.cross(gammadash1, gammadash2)


def _surface_norm(normal: object):
    normal_arr: jax.Array = as_jax_float64(normal)
    return jnp.sqrt(jnp.sum(normal_arr * normal_arr, axis=-1))


def _surface_area_from_dofs(spec, dofs: object):
    _gamma, gammadash1, gammadash2 = _surface_gamma_tangents_from_dofs(spec, dofs)
    normal = _surface_normal_from_tangents(gammadash1, gammadash2)
    return jnp.mean(_surface_norm(normal))


def qfm_residual_jax_from_dofs(spec, dofs: object, coil_set_spec: object):
    """Return the fixed-coil QFM residual for explicit surface DOFs."""
    gamma, gammadash1, gammadash2 = _surface_gamma_tangents_from_dofs(spec, dofs)
    normal = _surface_normal_from_tangents(gammadash1, gammadash2)
    norm_normal = _surface_norm(normal)
    unitnormal = normal / norm_normal[:, :, None]
    nphi, ntheta = gamma.shape[:2]
    B = grouped_biot_savart_B_from_spec(gamma.reshape(-1, 3), coil_set_spec).reshape(
        nphi,
        ntheta,
        3,
    )
    B_normal = jnp.sum(B * unitnormal, axis=2)
    B_norm_squared = jnp.sum(B * B, axis=2)
    return jnp.sum(B_normal * B_normal * norm_normal) / jnp.sum(
        B_norm_squared * norm_normal
    )


def qfm_label_jax_from_dofs(
    spec,
    dofs: object,
    coil_set_spec: object,
    *,
    label: str,
    toroidal_flux_idx: int = 0,
):
    """Return the QFM constraint label for explicit surface DOFs."""
    if label == "area":
        return _surface_area_from_dofs(spec, dofs)
    if label == "volume":
        return _surface_volume_from_dofs(spec, dofs)
    if label == "toroidal_flux":
        gamma, _gammadash1, gammadash2 = _surface_gamma_tangents_from_dofs(
            spec,
            dofs,
        )
        A = grouped_biot_savart_A_from_spec(
            gamma[int(toroidal_flux_idx)],
            coil_set_spec,
        )
        return jnp.sum(A * gammadash2[int(toroidal_flux_idx)]) / gamma.shape[1]
    raise ValueError(f"Unknown QFM label: {label!r}.")


def _qfm_metrics(
    spec,
    dofs: object,
    coil_set_spec: object,
    *,
    label: str,
    targetlabel: object,
    toroidal_flux_idx: int,
) -> _QfmMetrics:
    qfm_value = qfm_residual_jax_from_dofs(spec, dofs, coil_set_spec)
    label_value = qfm_label_jax_from_dofs(
        spec,
        dofs,
        coil_set_spec,
        label=label,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    label_residual = label_value - as_runtime_float64(
        targetlabel,
        reference=qfm_value,
    )
    return _QfmMetrics(
        qfm_value=qfm_value,
        label_value=label_value,
        label_residual=label_residual,
    )


def _qfm_penalty_from_metrics(
    metrics: _QfmMetrics,
    *,
    constraint_weight: object,
) -> jax.Array:
    weight = as_runtime_float64(
        constraint_weight,
        reference=metrics.qfm_value,
    )
    return metrics.qfm_value + 0.5 * weight * metrics.label_residual**2


def _qfm_augmented_from_metrics(
    metrics: _QfmMetrics,
    *,
    multiplier: object,
    penalty_weight: object,
) -> jax.Array:
    multiplier_value = as_runtime_float64(multiplier, reference=metrics.qfm_value)
    penalty_value = as_runtime_float64(penalty_weight, reference=metrics.qfm_value)
    return (
        metrics.qfm_value
        + multiplier_value * metrics.label_residual
        + 0.5 * penalty_value * metrics.label_residual**2
    )


def qfm_penalty_jax_from_dofs(
    spec,
    dofs: object,
    coil_set_spec: object,
    *,
    label: str,
    targetlabel: object,
    constraint_weight: object = 1.0,
    toroidal_flux_idx: int = 0,
):
    """Return ``QFM + 0.5 * weight * (label - target)^2`` in pure JAX."""
    metrics = _qfm_metrics(
        spec,
        dofs,
        coil_set_spec,
        label=label,
        targetlabel=targetlabel,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    return _qfm_penalty_from_metrics(
        metrics,
        constraint_weight=constraint_weight,
    )


def qfm_penalty_value_and_grad_jax_from_dofs(
    spec,
    dofs: object,
    coil_set_spec: object,
    *,
    label: str,
    targetlabel: object,
    constraint_weight: object = 1.0,
    toroidal_flux_idx: int = 0,
):
    """Return QFM penalty value and gradient with respect to surface DOFs."""
    return jax.value_and_grad(
        lambda surface_dofs: qfm_penalty_jax_from_dofs(
            spec,
            surface_dofs,
            coil_set_spec,
            label=label,
            targetlabel=targetlabel,
            constraint_weight=constraint_weight,
            toroidal_flux_idx=toroidal_flux_idx,
        )
    )(as_jax_float64(dofs))


def _qfm_augmented_lagrangian_jax_from_dofs(
    spec,
    dofs: object,
    coil_set_spec: object,
    *,
    label: str,
    targetlabel: object,
    multiplier: object,
    penalty_weight: object,
    toroidal_flux_idx: int = 0,
):
    metrics = _qfm_metrics(
        spec,
        dofs,
        coil_set_spec,
        label=label,
        targetlabel=targetlabel,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    return _qfm_augmented_from_metrics(
        metrics,
        multiplier=multiplier,
        penalty_weight=penalty_weight,
    )


def _identity_like_vector(vector: jax.Array) -> jax.Array:
    indices = jnp.arange(vector.shape[0])
    return jnp.asarray(indices[:, None] == indices[None, :], dtype=vector.dtype)


def _bfgs_line_search(value_and_grad, x, fun, grad, direction):
    grad_dot_direction = jnp.dot(grad, direction)
    c1 = jnp.asarray(1.0e-4, dtype=x.dtype)
    decay = jnp.asarray(0.5, dtype=x.dtype)

    def body(carry, _):
        best_x, best_fun, best_grad, accepted, n_evals, alpha = carry

        def evaluate(_):
            trial_x = x + alpha * direction
            trial_fun, trial_grad = value_and_grad(trial_x)
            finite = jnp.isfinite(trial_fun) & jnp.all(jnp.isfinite(trial_grad))
            decreases = trial_fun <= fun + c1 * alpha * grad_dot_direction
            accepts_trial = finite & decreases
            return (
                jnp.where(accepts_trial, trial_x, best_x),
                jnp.where(accepts_trial, trial_fun, best_fun),
                jnp.where(accepts_trial, trial_grad, best_grad),
                accepted | accepts_trial,
                n_evals + jnp.asarray(1, dtype=n_evals.dtype),
                alpha * decay,
            )

        return jax.lax.cond(
            accepted,
            lambda _: carry,
            evaluate,
            operand=None,
        ), None

    init = (
        x,
        fun,
        grad,
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=x.dtype),
    )
    final, _ = jax.lax.scan(body, init, xs=None, length=_BFGS_LINE_SEARCH_MAXITER)
    best_x, best_fun, best_grad, accepted, n_evals, _alpha = final
    return best_x, best_fun, best_grad, accepted, n_evals


def _bfgs_minimize(objective, init_dofs: object, *, max_iter: int, tol: float):
    max_iter_value = int(max_iter)
    if max_iter_value < 0:
        raise ValueError("max_iter must be non-negative.")
    tol_value = float(tol)
    value_and_grad = jax.value_and_grad(objective)

    @jax.jit
    def run(x_init):
        fun0, grad0 = value_and_grad(x_init)
        state0 = _BFGSState(
            x=x_init,
            fun=fun0,
            grad=grad0,
            hess_inv=_identity_like_vector(x_init),
            nit=jnp.asarray(0, dtype=jnp.int32),
            nfev=jnp.asarray(1, dtype=jnp.int32),
            njev=jnp.asarray(1, dtype=jnp.int32),
            status=jnp.asarray(_BFGS_STATUS_RUNNING, dtype=jnp.int32),
        )

        def scan_body(state, _):
            grad_norm = jnp.max(jnp.abs(state.grad))
            converged = grad_norm <= tol_value
            active = (state.status == _BFGS_STATUS_RUNNING) & ~converged

            def step(active_state):
                direction = -active_state.hess_inv @ active_state.grad
                grad_dot_direction = jnp.dot(active_state.grad, direction)
                descent = grad_dot_direction < 0.0
                finite_direction = jnp.all(jnp.isfinite(direction)) & jnp.isfinite(
                    grad_dot_direction
                )

                def not_descent(failed_state):
                    return failed_state._replace(
                        status=jnp.asarray(
                            _BFGS_STATUS_NOT_DESCENT,
                            dtype=failed_state.status.dtype,
                        )
                    )

                def descent_step(descent_state):
                    next_x, next_fun, next_grad, accepted, n_evals = _bfgs_line_search(
                        value_and_grad,
                        descent_state.x,
                        descent_state.fun,
                        descent_state.grad,
                        direction,
                    )
                    s = next_x - descent_state.x
                    y = next_grad - descent_state.grad
                    ys = jnp.dot(y, s)
                    update_floor = jnp.sqrt(jnp.finfo(next_x.dtype).eps)
                    update_hessian = accepted & (ys > update_floor)
                    safe_ys = jnp.where(
                        update_hessian, ys, jnp.asarray(1.0, next_x.dtype)
                    )
                    rho = 1.0 / safe_ys
                    identity = _identity_like_vector(next_x)
                    v = identity - rho * jnp.outer(s, y)
                    hess_inv_candidate = (
                        v @ descent_state.hess_inv @ v.T + rho * jnp.outer(s, s)
                    )
                    hess_inv = jnp.where(
                        update_hessian,
                        hess_inv_candidate,
                        descent_state.hess_inv,
                    )
                    return _BFGSState(
                        x=jnp.where(accepted, next_x, descent_state.x),
                        fun=jnp.where(accepted, next_fun, descent_state.fun),
                        grad=jnp.where(accepted, next_grad, descent_state.grad),
                        hess_inv=hess_inv,
                        nit=descent_state.nit + jnp.asarray(1, descent_state.nit.dtype),
                        nfev=descent_state.nfev + n_evals,
                        njev=descent_state.njev + n_evals,
                        status=jnp.where(
                            accepted,
                            descent_state.status,
                            jnp.asarray(
                                _BFGS_STATUS_LINE_SEARCH_FAILED,
                                dtype=descent_state.status.dtype,
                            ),
                        ),
                    )

                return jax.lax.cond(
                    descent & finite_direction,
                    descent_step,
                    not_descent,
                    active_state,
                )

            return jax.lax.cond(
                active, step, lambda inactive_state: inactive_state, state
            ), None

        final_state, _ = jax.lax.scan(
            scan_body,
            state0,
            xs=None,
            length=max_iter_value,
        )
        final_converged = jnp.max(jnp.abs(final_state.grad)) <= tol_value
        status = jnp.where(
            final_converged,
            jnp.asarray(_BFGS_STATUS_SUCCESS, dtype=final_state.status.dtype),
            jnp.where(
                final_state.status == _BFGS_STATUS_RUNNING,
                jnp.asarray(_BFGS_STATUS_MAXITER, dtype=final_state.status.dtype),
                final_state.status,
            ),
        )
        return _BFGSResult(
            x=final_state.x,
            success=status == _BFGS_STATUS_SUCCESS,
            status=status,
            fun=final_state.fun,
            jac=final_state.grad,
            nit=final_state.nit,
            nfev=final_state.nfev,
            njev=final_state.njev,
        )

    return run(as_jax_float64(init_dofs))


def _scalar_from_vector(reference: jax.Array, value: float) -> jax.Array:
    @jax.jit
    def build(reference_vector):
        zero = jnp.sum(reference_vector) - jnp.sum(reference_vector)
        return zero + float(value)

    return build(reference)


def _require_bfgs_optimizer(optimizer: str) -> None:
    if optimizer != "bfgs":
        raise NotImplementedError(
            "QFM JAX solves currently support optimizer='bfgs' only; optional "
            "Optimistix QFM methods are not wired in this milestone."
        )


def _penalty_info(
    spec,
    dofs: object,
    coil_set_spec: object,
    result,
    *,
    label: str,
    targetlabel: object,
    constraint_weight: object,
    toroidal_flux_idx: int,
) -> QfmPenaltySolveInfo:
    @jax.jit
    def compute(dofs_value, success, status, fun, gradient, nit, nfev, njev):
        metrics = _qfm_metrics(
            spec,
            dofs_value,
            coil_set_spec,
            label=label,
            targetlabel=targetlabel,
            toroidal_flux_idx=toroidal_flux_idx,
        )
        penalty_value = _qfm_penalty_from_metrics(
            metrics,
            constraint_weight=constraint_weight,
        )
        return QfmPenaltySolveInfo(
            success=success,
            status=status,
            fun=fun,
            gradient=gradient,
            nit=nit,
            nfev=nfev,
            njev=njev,
            label_value=metrics.label_value,
            label_residual=metrics.label_residual,
            qfm_value=metrics.qfm_value,
            penalty_value=penalty_value,
        )

    return compute(
        as_jax_float64(dofs),
        result.success,
        result.status,
        result.fun,
        result.jac,
        result.nit,
        result.nfev,
        result.njev,
    )


def qfm_penalty_solve_jax(
    spec,
    coil_set_spec: object,
    label: str,
    targetlabel: object,
    constraint_weight: object,
    init_dofs: object,
    *,
    max_iter: int,
    tol: float,
    optimizer: str = "bfgs",
    toroidal_flux_idx: int = 0,
):
    """Minimize the QFM penalty objective without mutating a surface object."""
    _require_bfgs_optimizer(optimizer)

    def objective(surface_dofs):
        return qfm_penalty_jax_from_dofs(
            spec,
            surface_dofs,
            coil_set_spec,
            label=label,
            targetlabel=targetlabel,
            constraint_weight=constraint_weight,
            toroidal_flux_idx=toroidal_flux_idx,
        )

    result = _bfgs_minimize(objective, init_dofs, max_iter=max_iter, tol=tol)
    final_dofs = as_jax_float64(result.x)
    return (
        final_dofs,
        _penalty_info(
            spec,
            final_dofs,
            coil_set_spec,
            result,
            label=label,
            targetlabel=targetlabel,
            constraint_weight=constraint_weight,
            toroidal_flux_idx=toroidal_flux_idx,
        ),
    )


def _augmented_info(
    spec,
    dofs: object,
    coil_set_spec: object,
    result,
    *,
    label: str,
    targetlabel: object,
    multiplier: object,
    penalty_weight: object,
    toroidal_flux_idx: int,
    tol: float,
) -> QfmAugmentedLagrangianInfo:
    @jax.jit
    def compute(dofs_value, success, status, fun, nit, nfev, njev):
        metrics = _qfm_metrics(
            spec,
            dofs_value,
            coil_set_spec,
            label=label,
            targetlabel=targetlabel,
            toroidal_flux_idx=toroidal_flux_idx,
        )
        augmented_value = _qfm_augmented_from_metrics(
            metrics,
            multiplier=multiplier,
            penalty_weight=penalty_weight,
        )
        qfm_gradient = jax.grad(
            lambda surface_dofs: qfm_residual_jax_from_dofs(
                spec,
                surface_dofs,
                coil_set_spec,
            )
        )(dofs_value)
        accepted = success & (jnp.abs(metrics.label_residual) <= float(tol))
        return QfmAugmentedLagrangianInfo(
            success=accepted,
            status=status,
            fun=fun,
            gradient=qfm_gradient,
            nit=nit,
            nfev=nfev,
            njev=njev,
            label_value=metrics.label_value,
            label_residual=metrics.label_residual,
            qfm_value=metrics.qfm_value,
            augmented_value=augmented_value,
            multiplier=as_runtime_float64(multiplier, reference=metrics.qfm_value),
            penalty_weight=as_runtime_float64(
                penalty_weight,
                reference=metrics.qfm_value,
            ),
        )

    return compute(
        as_jax_float64(dofs),
        result.success,
        result.status,
        result.fun,
        result.nit,
        result.nfev,
        result.njev,
    )


def qfm_augmented_lagrangian_solve_jax(
    spec,
    coil_set_spec: object,
    label: str,
    targetlabel: object,
    init_dofs: object,
    *,
    max_outer: int,
    inner_max_iter: int,
    tol: float,
    optimizer: str = "bfgs",
    initial_penalty_weight: float = 10.0,
    penalty_growth: float = 10.0,
    max_penalty_weight: float = 1.0e8,
    toroidal_flux_idx: int = 0,
):
    """Run a pure-JAX augmented-Lagrangian QFM solve."""
    _require_bfgs_optimizer(optimizer)
    dofs = as_jax_float64(init_dofs)
    max_outer_value = int(max_outer)
    inner_max_iter_value = int(inner_max_iter)
    if max_outer_value < 1:
        raise ValueError("max_outer must be positive.")
    if inner_max_iter_value < 1:
        raise ValueError("inner_max_iter must be positive.")
    multiplier = _scalar_from_vector(dofs, 0.0)
    penalty_weight = _scalar_from_vector(dofs, initial_penalty_weight)

    @jax.jit
    def update_multipliers(dofs_value, multiplier_value, penalty_weight_value):
        metrics = _qfm_metrics(
            spec,
            dofs_value,
            coil_set_spec,
            label=label,
            targetlabel=targetlabel,
            toroidal_flux_idx=toroidal_flux_idx,
        )
        next_multiplier = (
            multiplier_value + penalty_weight_value * metrics.label_residual
        )
        next_penalty_weight = jnp.minimum(
            penalty_weight_value * float(penalty_growth),
            as_runtime_float64(max_penalty_weight, reference=metrics.qfm_value),
        )
        return next_multiplier, next_penalty_weight

    result_multiplier = multiplier
    result_penalty_weight = penalty_weight
    for outer_index in range(max_outer_value):
        result_multiplier = multiplier
        result_penalty_weight = penalty_weight

        def objective(surface_dofs):
            return _qfm_augmented_lagrangian_jax_from_dofs(
                spec,
                surface_dofs,
                coil_set_spec,
                label=label,
                targetlabel=targetlabel,
                multiplier=multiplier,
                penalty_weight=penalty_weight,
                toroidal_flux_idx=toroidal_flux_idx,
            )

        result = _bfgs_minimize(
            objective,
            dofs,
            max_iter=inner_max_iter_value,
            tol=tol,
        )
        dofs = as_jax_float64(result.x)
        if outer_index + 1 < max_outer_value:
            multiplier, penalty_weight = update_multipliers(
                dofs,
                multiplier,
                penalty_weight,
            )

    return (
        dofs,
        _augmented_info(
            spec,
            dofs,
            coil_set_spec,
            result,
            label=label,
            targetlabel=targetlabel,
            multiplier=result_multiplier,
            penalty_weight=result_penalty_weight,
            toroidal_flux_idx=toroidal_flux_idx,
            tol=tol,
        ),
    )
