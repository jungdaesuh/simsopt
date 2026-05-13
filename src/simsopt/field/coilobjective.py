import jax
import jax.numpy as jnp
import numpy as np

from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
from .._core.jax_host_boundary import host_array, strict_scalar_grad

__all__ = ["CurrentPenalty"]


def current_penalty_pure(I, threshold):
    # The on-device zero is built from ``threshold - threshold`` (the
    # callers always pass a finite threshold) so the strict
    # ``transfer_guard("disallow")`` path never sees a Python literal:
    # both ``jnp.maximum(excess, 0.0)`` and ``jnp.zeros_like(excess)``
    # trip the guard because they materialise a host 0 on-device. The
    # previous ``min(|I|, t) - min(|I|, t)`` trick avoided the guard
    # but threaded 0 * inf through autodiff at I=±inf, returning NaN
    # gradients that broke L-BFGS line-search backtracks after
    # overshoots. ``jnp.maximum(excess, threshold - threshold)``
    # autodifferentiates to ``sign(I)`` at the boundary, so the
    # squared form has the correct ±inf gradient at I=±inf.
    excess = jnp.abs(I) - threshold
    on_device_zero = threshold - threshold
    positive_excess = jnp.maximum(excess, on_device_zero)
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
