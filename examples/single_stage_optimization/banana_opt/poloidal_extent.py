from __future__ import annotations

import numpy as np
from jax import grad
import jax.numpy as jnp

from banana_opt.hardware_keepout import live_winding_r0
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


def _iter_poloidal_extent_objectives(poloidal_extent_obj):
    if all(
        hasattr(poloidal_extent_obj, attr)
        for attr in ("curve", "R_winding", "Z_winding")
    ):
        yield poloidal_extent_obj
        return
    if hasattr(poloidal_extent_obj, "opt"):
        yield from _iter_poloidal_extent_objectives(poloidal_extent_obj.opt)
        return
    if hasattr(poloidal_extent_obj, "opts"):
        for opt in poloidal_extent_obj.opts:
            yield from _iter_poloidal_extent_objectives(opt)
        return
    raise AttributeError(
        "Expected a PoloidalExtent objective or algebraic objective containing one."
    )


def poloidal_extent_rad_from_objective(poloidal_extent_obj) -> float:
    return max(
        max_poloidal_extent_rad(
            term.curve,
            # B1.3: read the LIVE winding radius so the poloidal-extent verdict
            # tracks the moving frame under --winding-surface-free-r0, matching
            # the term's live J(). Use the module helper (not term.live_R_winding())
            # so it honors the iterator's curve/R_winding contract for any object.
            live_winding_r0([term.curve], term.R_winding),
            term.Z_winding,
        )
        for term in _iter_poloidal_extent_objectives(poloidal_extent_obj)
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
        # Fallback winding major radius for non-CWS lineages; with a CWS curve
        # the live value (surf.get_rc(0,0), read each evaluation) is what J
        # scores against, so a freed/moved winding surface is measured in its
        # true poloidal frame (B1.3 value-live).
        self.R_winding = float(R_winding)
        self.Z_winding = float(Z_winding)
        self.theta_target = float(theta_target)
        self.p = int(p)
        super().__init__(depends_on=[curve])
        # R_winding enters as a CALL-TIME argument sourced live in J/dJ; it stays
        # in the JAX-traced path (smooth arctan2 of R - R_winding), preserving
        # differentiability. B1.3: value-live; frame-orientation gradient
        # deferred (the curve VJP carries the centerline-translation gradient).
        self.J_jax = jit(
            lambda g, gd, r_winding: _poloidal_extent_pure(
                g,
                gd,
                r_winding,
                self.Z_winding,
                self.theta_target,
                self.p,
            )
        )
        self.dJ_dprimal = jit(
            lambda g, gd, r_winding: grad(self.J_jax, argnums=(0, 1))(g, gd, r_winding)
        )

    def live_R_winding(self):
        """Live winding major radius the poloidal frame is measured about, m."""
        return live_winding_r0([self.curve], self.R_winding)

    def J(self):
        return float(
            self.J_jax(
                self.curve.gamma(), self.curve.gammadash(), self.live_R_winding()
            )
        )

    @derivative_dec
    def dJ(self):
        gamma = self.curve.gamma()
        gammadash = self.curve.gammadash()
        r_winding = self.live_R_winding()
        dJ_dgamma, dJ_dgammadash = self.dJ_dprimal(gamma, gammadash, r_winding)
        return self.curve.dgamma_by_dcoeff_vjp(
            np.asarray(dJ_dgamma)
        ) + self.curve.dgammadash_by_dcoeff_vjp(
            np.asarray(dJ_dgammadash)
        )

    return_fn_map = {"J": J, "dJ": dJ}


# Smooth-max temperature (rad) for the poloidal-floor objective. The floor scores
# the PEAK inboard poloidal extent against a target, so it needs a differentiable
# maximum; this temperature sets how tightly the softmax-weighted mean tracks the
# hard max. Module-owned (not a caller knob): small enough that the smooth max sits
# within a few percent of the hard max at the sharp ~1 rad banana-tip peak, large
# enough that the gradient is shared across the near-peak points (not just the single
# argmax) for stable descent.
POLOIDAL_FLOOR_SMOOTH_MAX_TEMPERATURE_RAD = 0.05


@jit
def _poloidal_floor_pure(
    gamma,
    R_winding,
    Z_winding,
    theta_floor,
    temperature,
    p,
):
    R = jnp.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]
    theta_in = jnp.arctan2(Z - Z_winding, -(R - R_winding))
    abs_theta = jnp.abs(theta_in)
    # Softmax-weighted mean (Boltzmann soft-max): a differentiable, RESOLUTION-STABLE
    # estimate of max|theta_in|, tight (within ~T of the hard max for the sharp banana
    # tip the floor targets) and exact for a constant profile. The prior
    # T*(logsumexp(x/T) - log N) log-MEAN-exp under-read a peak by up to ~T*log(N) -- a
    # resolution-DEPENDENT bias (worst-case ~0.42 rad at T=0.05, N=4096; ~0.27 rad realized
    # on a sharp tip, and growing with the quadpoint count) that silently raised the
    # effective floor. The max-shift cancels in the ratio
    # (gradient unaffected); it only guards exp overflow. Under-estimating is the
    # conservative direction for a one-sided floor (the CAP path over-estimates via the
    # upper-bound LSE for the symmetric reason).
    weights = jnp.exp((abs_theta - jnp.max(abs_theta)) / temperature)
    smooth_max = jnp.sum(weights * abs_theta) / jnp.sum(weights)
    deficit = jnp.maximum(theta_floor - smooth_max, 0.0)
    return (1.0 / p) * deficit**p


class PoloidalExtentFloor(Optimizable):
    """Soft lower bound on the peak inboard poloidal extent of a banana coil.

    J = (1/p) * max(theta_floor - smoothmax|theta_in|, 0)^p, in rad^p. Zero once
    the furthest-inboard point of the coil reaches theta_floor; otherwise positive
    and decreasing in extent, so descent spreads the banana-tip U-turn further
    poloidally (distributing the turn over a longer arc, relaxing the tip
    curvature) instead of turning around within a short poloidal arc. The
    complement of PoloidalExtent, which caps the peak from above; here the peak is
    floored from below. theta is measured inboard about the live winding major
    radius (value-live under a freed winding R0), matching PoloidalExtent's frame.
    """

    def __init__(
        self,
        curve,
        R_winding,
        theta_floor,
        p=2,
        Z_winding=0.0,
        temperature=POLOIDAL_FLOOR_SMOOTH_MAX_TEMPERATURE_RAD,
    ):
        self.curve = curve
        self.R_winding = float(R_winding)
        self.Z_winding = float(Z_winding)
        self.theta_floor = float(theta_floor)
        self.p = int(p)
        self.temperature = float(temperature)
        super().__init__(depends_on=[curve])
        # theta_in depends on gamma only (no arc-length weighting on a peak term),
        # so the descent carries a single gamma VJP. R_winding enters live in J/dJ.
        self.J_jax = jit(
            lambda g, r_winding: _poloidal_floor_pure(
                g,
                r_winding,
                self.Z_winding,
                self.theta_floor,
                self.temperature,
                self.p,
            )
        )
        self.dJ_dgamma = jit(
            lambda g, r_winding: grad(self.J_jax, argnums=0)(g, r_winding)
        )

    def live_R_winding(self):
        """Live winding major radius the poloidal frame is measured about, m."""
        return live_winding_r0([self.curve], self.R_winding)

    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.live_R_winding()))

    @derivative_dec
    def dJ(self):
        gamma = self.curve.gamma()
        r_winding = self.live_R_winding()
        return self.curve.dgamma_by_dcoeff_vjp(
            np.asarray(self.dJ_dgamma(gamma, r_winding))
        )

    return_fn_map = {"J": J, "dJ": dJ}
