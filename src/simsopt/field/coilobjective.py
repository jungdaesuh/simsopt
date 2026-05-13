import jax
import jax.numpy as jnp
import numpy as np

from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
from .._core.jax_host_boundary import host_array, strict_scalar_grad

__all__ = ["CurrentPenalty"]


def current_penalty_pure(I, threshold):
    abs_current = jnp.abs(I)
    excess = abs_current - threshold
    # Keep zero on-device for strict transfer_guard while preserving
    # inf-current penalties.
    zero_source = jnp.minimum(abs_current, threshold)
    zero = zero_source - zero_source
    positive_excess = jnp.maximum(excess, zero)
    return positive_excess * positive_excess


def _host_current_gradient_block(gradient):
    return host_array(gradient, dtype=np.float64).reshape((1,))


class CurrentPenalty(Optimizable):
    """
    A :obj:`CurrentPenalty` can be used to penalize
    large currents in coils.
    """

    def __init__(self, current, threshold=0):
        self.current = current
        self.threshold = threshold
        super().__init__(depends_on=[current])
        self.J_jax = lambda I: current_penalty_pure(I, self.threshold)
        self.this_grad = lambda I: strict_scalar_grad(self.J_jax, I)

    def J(self):
        with jax.transfer_guard("allow"):
            return self.J_jax(self.current.get_value())

    @derivative_dec
    def dJ(self):
        with jax.transfer_guard("allow"):
            grad0 = _host_current_gradient_block(
                self.this_grad(self.current.get_value())
            )
        return self.current.vjp(grad0)
