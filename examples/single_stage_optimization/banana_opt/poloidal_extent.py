from __future__ import annotations

import numpy as np
from jax import grad
import jax.numpy as jnp

from banana_opt.smoothing import smoothmax_selected
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit


@jit
def _poloidal_extent_pure(
    gamma,
    gammadash,
    R_winding,
    Z_winding,
    theta_target,
    p,
):
    R = jnp.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]
    theta_in = jnp.arctan2(Z - Z_winding, -(R - R_winding))
    arc_length = jnp.linalg.norm(gammadash, axis=-1)
    excess = jnp.maximum(jnp.abs(theta_in) - theta_target, 0.0)
    return (1.0 / p) * jnp.mean(excess**p * arc_length)


def inboard_poloidal_angles(gamma, R_winding, Z_winding=0.0):
    points = np.asarray(gamma, dtype=float)
    R = np.linalg.norm(points[:, :2], axis=-1)
    Z = points[:, 2]
    return np.arctan2(Z - float(Z_winding), -(R - float(R_winding)))


def max_poloidal_extent_rad(curve, R_winding, Z_winding=0.0) -> float:
    return float(
        np.max(np.abs(inboard_poloidal_angles(curve.gamma(), R_winding, Z_winding)))
    )


def poloidal_extent_signed_constraint(
    curve,
    R_winding,
    theta_threshold,
    *,
    Z_winding=0.0,
):
    theta_threshold_value = float(theta_threshold)
    signed_value = (
        max_poloidal_extent_rad(curve, R_winding, Z_winding) - theta_threshold_value
    )
    return signed_value, max(0.0, signed_value)


def poloidal_extent_rad_from_objective(poloidal_extent_obj) -> float:
    return max_poloidal_extent_rad(
        poloidal_extent_obj.curve,
        poloidal_extent_obj.R_winding,
        poloidal_extent_obj.Z_winding,
    )


def smooth_max_poloidal_extent_signed_constraint(
    curve,
    R_winding,
    theta_threshold,
    temperature,
    objective_optimizable,
    *,
    Z_winding=0.0,
    include_hard_signal=False,
):
    gamma = np.asarray(curve.gamma(), dtype=float)
    theta = inboard_poloidal_angles(gamma, R_winding, Z_winding)
    abs_theta = np.abs(theta)
    hard_max = float(np.max(abs_theta))
    active_mask = abs_theta >= (hard_max - 4.0 * float(temperature))
    smooth_max, active_weights = smoothmax_selected(
        abs_theta[active_mask],
        temperature,
        float(np.finfo(float).eps),
    )

    R = np.linalg.norm(gamma[:, :2], axis=-1)
    a = gamma[:, 2] - float(Z_winding)
    b = float(R_winding) - R
    denom = np.maximum(a * a + b * b, float(np.finfo(float).eps))
    safe_R = np.maximum(R, float(np.finfo(float).eps))

    theta_grad = np.zeros_like(gamma)
    theta_grad[:, 0] = a * gamma[:, 0] / (safe_R * denom)
    theta_grad[:, 1] = a * gamma[:, 1] / (safe_R * denom)
    theta_grad[:, 2] = b / denom

    point_weights = np.zeros_like(abs_theta)
    point_weights[active_mask] = active_weights * np.sign(theta[active_mask])
    point_gradient = point_weights[:, None] * theta_grad
    grad_value = np.asarray(
        curve.dgamma_by_dcoeff_vjp(point_gradient)(objective_optimizable),
        dtype=float,
    )
    theta_threshold_value = float(theta_threshold)
    surrogate_signed_value = float(smooth_max - theta_threshold_value)
    surrogate_violation = max(0.0, surrogate_signed_value)
    if include_hard_signal:
        hard_signed_value = hard_max - theta_threshold_value
        hard_violation = max(0.0, hard_signed_value)
        return (
            surrogate_signed_value,
            grad_value,
            surrogate_violation,
            hard_signed_value,
            hard_violation,
        )
    return surrogate_signed_value, grad_value, surrogate_violation


def smooth_max_poloidal_extent_signed_constraint_with_hard_signal(
    curve,
    R_winding,
    theta_threshold,
    temperature,
    objective_optimizable,
    *,
    Z_winding=0.0,
):
    return smooth_max_poloidal_extent_signed_constraint(
        curve,
        R_winding,
        theta_threshold,
        temperature,
        objective_optimizable,
        Z_winding=Z_winding,
        include_hard_signal=True,
    )


class PoloidalExtent(Optimizable):
    def __init__(self, curve, R_winding, theta_target, p=4, Z_winding=0.0):
        self.curve = curve
        self.R_winding = float(R_winding)
        self.Z_winding = float(Z_winding)
        self.theta_target = float(theta_target)
        self.p = int(p)
        super().__init__(depends_on=[curve])
        self.J_jax = jit(
            lambda g, gd: _poloidal_extent_pure(
                g,
                gd,
                self.R_winding,
                self.Z_winding,
                self.theta_target,
                self.p,
            )
        )
        self.dJ_dgamma = jit(lambda g, gd: grad(self.J_jax, argnums=0)(g, gd))
        self.dJ_dgammadash = jit(lambda g, gd: grad(self.J_jax, argnums=1)(g, gd))

    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        gamma = self.curve.gamma()
        gammadash = self.curve.gammadash()
        return self.curve.dgamma_by_dcoeff_vjp(
            np.asarray(self.dJ_dgamma(gamma, gammadash))
        ) + self.curve.dgammadash_by_dcoeff_vjp(
            np.asarray(self.dJ_dgammadash(gamma, gammadash))
        )

    return_fn_map = {"J": J, "dJ": dJ}
