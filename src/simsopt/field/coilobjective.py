import jax
import jax.numpy as jnp
import numpy as np

from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
from .._core.jax_host_boundary import host_array, strict_scalar_grad
from ..jax_core._math_utils import as_jax_float64

__all__ = ["CurrentPenalty"]


def _sum_to_primal_shape(value, primal):
    target_shape = jnp.shape(primal)
    value_shape = jnp.shape(value)
    if value_shape == target_shape:
        return value
    if target_shape == ():
        return jnp.sum(value)

    leading_axes = tuple(range(len(value_shape) - len(target_shape)))
    broadcast_axes = tuple(
        len(value_shape) - len(target_shape) + index
        for index, (value_size, target_size) in enumerate(
            zip(value_shape[-len(target_shape) :], target_shape)
        )
        if target_size == 1 and value_size != 1
    )
    axes = leading_axes + broadcast_axes
    if axes:
        value = jnp.sum(value, axis=axes, keepdims=True)
    return jnp.reshape(value, target_shape)


@jax.custom_vjp
def current_penalty_pure(I, threshold):
    # Both inputs must be on-device JAX arrays. ``threshold`` is staged
    # to device at ``CurrentPenalty`` construction, so ``threshold -
    # threshold`` is the on-device zero used to clamp ``max(excess, 0)``
    # without ever materialising a Python literal at trace time.
    # `jnp.maximum(excess, 0.0)` would trip ``transfer_guard("disallow")``
    # via the implicit `lax.full_like(..., 0, ...)` boundary; the prior
    # `min(|I|, t) - min(|I|, t)` self-cancel trick threaded 0 * inf
    # through autodiff and returned NaN gradients at I=±inf. The form
    # below autodifferentiates to ``sign(I)`` at the boundary so the
    # squared envelope keeps the correct ±inf gradient.
    excess = jnp.abs(I) - threshold
    on_device_zero = threshold - threshold
    positive_excess = jnp.maximum(excess, on_device_zero)
    return positive_excess * positive_excess


def _current_penalty_fwd(I, threshold):
    excess = jnp.abs(I) - threshold
    on_device_zero = threshold - threshold
    positive_excess = jnp.maximum(excess, on_device_zero)
    active = (excess > on_device_zero).astype(positive_excess.dtype)
    return positive_excess * positive_excess, (
        I,
        threshold,
        positive_excess,
        active,
        on_device_zero,
    )


def _current_penalty_bwd(residuals, cotangent):
    I, threshold, positive_excess, active, on_device_zero = residuals
    two_positive_excess = positive_excess + positive_excess
    current_grad = cotangent * two_positive_excess * jnp.sign(I) * active
    threshold_grad = cotangent * (on_device_zero - two_positive_excess) * active
    return current_grad, _sum_to_primal_shape(threshold_grad, threshold)


current_penalty_pure.defvjp(_current_penalty_fwd, _current_penalty_bwd)


def _host_current_gradient_block(gradient):
    return host_array(gradient, dtype=np.float64).reshape((1,))


class CurrentPenalty(Optimizable):
    """
    A :obj:`CurrentPenalty` can be used to penalize
    large currents in coils.
    """

    def __init__(self, current, threshold=0):
        self.current = current
        # Stage threshold to device once at construction. The pure
        # kernel's ``threshold - threshold`` then resolves to an
        # on-device zero under ``transfer_guard("disallow")`` without
        # needing an outer ``transfer_guard("allow")`` scope.
        self.threshold = as_jax_float64(threshold)
        super().__init__(depends_on=[current])
        self.J_jax = lambda I: current_penalty_pure(I, self.threshold)
        self.this_grad = lambda I: strict_scalar_grad(self.J_jax, I)

    def J(self):
        return self.J_jax(as_jax_float64(self.current.get_value()))

    @derivative_dec
    def dJ(self):
        grad0 = _host_current_gradient_block(
            self.this_grad(as_jax_float64(self.current.get_value()))
        )
        return self.current.vjp(grad0)
