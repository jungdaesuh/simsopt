import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit


@jit
def _circular_vessel_clearance_pure(
        gamma, vessel_major_radius, vessel_minor_radius, minimum_clearance, p):
    cylindrical_r = jnp.linalg.norm(gamma[..., :2], axis=-1)
    cross_section_r = jnp.sqrt(
        (cylindrical_r - vessel_major_radius) ** 2 + gamma[..., 2] ** 2
    )
    clearance = vessel_minor_radius - cross_section_r
    excess = jnp.maximum(minimum_clearance - clearance, 0.0)
    return (1.0 / p) * jnp.mean(excess ** p)


class CircularVesselClearance(Optimizable):
    r"""
    Lp penalty for a surface inside a concentric circular vessel.

    The HBT_BANANA lane models the vessel cross-section as
    ``(R - R_vessel)^2 + Z^2 <= a_vessel^2``. This objective penalizes
    quadpoints whose clearance to that vessel falls below ``minimum_clearance``.
    """

    def __init__(
            self, surface, vessel_major_radius, vessel_minor_radius,
            minimum_clearance, p=4):
        self.surface = surface
        self.vessel_major_radius = vessel_major_radius
        self.vessel_minor_radius = vessel_minor_radius
        self.minimum_clearance = minimum_clearance
        self.p = p
        super().__init__(depends_on=[surface])
        self.J_jax = jit(lambda g: _circular_vessel_clearance_pure(
            g, vessel_major_radius, vessel_minor_radius, minimum_clearance, p))
        self.dJ_dgamma = jit(lambda g: grad(self.J_jax)(g))

    def J(self):
        return float(self.J_jax(self.surface.gamma()))

    @derivative_dec
    def dJ(self):
        return self.surface.dgamma_by_dcoeff_vjp(
            np.asarray(self.dJ_dgamma(self.surface.gamma()))
        )

    return_fn_map = {'J': J, 'dJ': dJ}
