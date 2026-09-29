"""First-order rounding bounds that hold for ANY implementation of one formula.

A same-state replay compares two implementations of one function -- here a C++
and NumPy native lane against a JAX lane -- that evaluate the same formula but
sum in different orders, associate products differently and form derivatives
differently (closed-form Jacobians and hand-written adjoints on one side,
reverse-mode autodiff on the other).  Neither lane is a reference for the other,
so the only defensible tolerance is a bound on how far EACH may round away from
the exact value of the formula; their difference is then at most twice that.

:class:`Bounded` carries a float64 value together with the bookkeeping of such a
bound, in units of the unit roundoff ``u = 2**-53`` and to first order in ``u``:

``|v_hat - v| <= e u``
    Wilkinson's running error bound.  Every sum is charged its worst order:
    ``(n - 1) sum |x_i|`` for ``n`` summands, which bounds every partial sum any
    order can form, so the charge holds for every order at once.  Products are
    charged ``|product|`` per rounding, which is independent of association.

``|d_hat - d| <= (Ed + (paths - 1) D) u``
    for the forward-mode derivative ``d = dv/dx`` of every component.  By the
    chain rule ``d`` is a sum over computational-graph PATHS of products of
    local partial derivatives, whatever mode or factorization evaluates it.
    ``D`` is the sum over paths of ``|path product|``; ``paths`` counts them.
    Any evaluation of the path sum -- forward, reverse, or a closed form that
    merely factorizes it -- passes each path product through at most
    ``paths - 1`` additions (each addition merges the path's group with at least
    one other path), which is the ``(paths - 1) D`` charge.  ``Ed`` collects the
    rest: one rounding per multiplication by a local partial, plus the error of
    each computed local partial times the path sum reaching it.

Transcendental functions are not modelled here: callers that need them build
the value with its own derived error and inject it with :func:`element`.  Every
constant a caller passes as exact must be the same float64 in every lane;
:func:`scale` takes a relative error for constants the lanes may round
differently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: float64 unit roundoff ``u``.
UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0

#: A replay compares two independently rounded implementations.
COMPARED_IMPLEMENTATIONS = 2.0


@dataclass(frozen=True)
class Bounded:
    """A float64 quantity with first-order rounding bookkeeping (units of ``u``).

    ``v`` and ``e`` have the quantity's shape; ``d``, ``D``, ``Ed`` and ``paths``
    append one trailing axis of the derivative components.
    """

    v: np.ndarray
    e: np.ndarray
    d: np.ndarray
    D: np.ndarray
    Ed: np.ndarray
    paths: np.ndarray

    @property
    def components(self) -> int:
        return int(self.d.shape[-1])

    def __getitem__(self, key) -> Bounded:
        key = key if isinstance(key, tuple) else (key,)
        # The derivative arrays carry one more (trailing) axis than the value.
        derivative_key = (
            key + (slice(None),) if any(k is Ellipsis for k in key) else key
        )
        return Bounded(
            self.v[key],
            self.e[key],
            self.d[derivative_key],
            self.D[derivative_key],
            self.Ed[derivative_key],
            self.paths[derivative_key],
        )

    def value_bound(self) -> np.ndarray:
        """Largest ``|v_hat - v|`` of one implementation."""
        return self.e * UNIT_ROUNDOFF

    def derivative_bound(self) -> np.ndarray:
        """Largest ``|d_hat - d|`` of one implementation, per component."""
        return (self.Ed + np.maximum(self.paths - 1.0, 0.0) * self.D) * UNIT_ROUNDOFF


def _f(array: object) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def element(
    value: object,
    error: object,
    derivative: object,
    derivative_abs: object,
    derivative_error: object,
) -> Bounded:
    """Inject a quantity whose rounding was derived outside this module.

    ``error`` bounds the value's rounding and ``derivative_error`` that of each
    derivative component (both in units of ``u``); ``derivative_abs`` is the
    sum over paths of ``|path product|`` of the injected derivative.  The
    injected derivative counts as one path: its internal additions are already
    inside ``derivative_error``.
    """
    derivative = _f(derivative)
    return Bounded(
        _f(value),
        _f(error),
        derivative,
        _f(derivative_abs),
        _f(derivative_error),
        (_f(derivative_abs) != 0.0).astype(np.float64),
    )


def seed(values: object) -> Bounded:
    """Exact independent variables: the derivative components are these."""
    values = _f(values)
    size = values.size
    eye = np.eye(size).reshape(values.shape + (size,))
    return Bounded(
        values, np.zeros_like(values), eye, eye.copy(), np.zeros_like(eye), eye.copy()
    )


def constant(values: object, components: int, error: object = 0.0) -> Bounded:
    """A quantity independent of the variables, with a known rounding bound."""
    values = _f(values)
    zeros = np.zeros(values.shape + (components,))
    return Bounded(
        values,
        np.broadcast_to(_f(error), values.shape).copy(),
        zeros,
        zeros,
        zeros,
        zeros,
    )


def _broadcast(a: Bounded, shape: tuple[int, ...]) -> Bounded:
    full = shape + (a.components,)
    return Bounded(
        np.broadcast_to(a.v, shape),
        np.broadcast_to(a.e, shape),
        np.broadcast_to(a.d, full),
        np.broadcast_to(a.D, full),
        np.broadcast_to(a.Ed, full),
        np.broadcast_to(a.paths, full),
    )


def _aligned(a: Bounded, b: Bounded) -> tuple[Bounded, Bounded]:
    shape = np.broadcast_shapes(a.v.shape, b.v.shape)
    return _broadcast(a, shape), _broadcast(b, shape)


def _x(values: np.ndarray) -> np.ndarray:
    """Value-shaped array broadcast against the derivative axis."""
    return values[..., None]


def add(a: Bounded, b: Bounded) -> Bounded:
    a, b = _aligned(a, b)
    v = a.v + b.v
    return Bounded(
        v, a.e + b.e + np.abs(v), a.d + b.d, a.D + b.D, a.Ed + b.Ed, a.paths + b.paths
    )


def neg(a: Bounded) -> Bounded:
    return Bounded(-a.v, a.e, -a.d, a.D, a.Ed, a.paths)


def sub(a: Bounded, b: Bounded) -> Bounded:
    return add(a, neg(b))


def mul(a: Bounded, b: Bounded) -> Bounded:
    """``a b``: each path through ``a`` is multiplied by the local partial ``b``."""
    a, b = _aligned(a, b)
    v = a.v * b.v
    abs_a, abs_b = np.abs(_x(a.v)), np.abs(_x(b.v))
    through_b = abs_a * b.D
    through_a = abs_b * a.D
    return Bounded(
        v,
        a.e * np.abs(b.v) + np.abs(a.v) * b.e + np.abs(v),
        _x(a.v) * b.d + _x(b.v) * a.d,
        through_b + through_a,
        abs_a * b.Ed
        + _x(a.e) * b.D
        + abs_b * a.Ed
        + _x(b.e) * a.D
        + through_b
        + through_a,
        a.paths + b.paths,
    )


def square(a: Bounded) -> Bounded:
    return mul(a, a)


#: Roundings allowed for forming the local partial of a division or a square
#: root (``1/b`` or ``-q/b``; ``1/(2 sqrt(b))``): the lanes form these
#: differently (``q/b`` or ``a/(b b)``; ``0.5/sqrt(b)`` or ``-0.5/(b sqrt(b))``
#: in a custom JVP), and two roundings bound every such form.
LOCAL_PARTIAL_ROUNDINGS = 2.0


def div(a: Bounded, b: Bounded) -> Bounded:
    """``a / b`` with ``b`` bounded away from zero by its own rounding."""
    a, b = _aligned(a, b)
    q = a.v / b.v
    abs_b = np.abs(b.v)
    e = a.e / abs_b + np.abs(q) * b.e / abs_b + np.abs(q)
    inv = 1.0 / abs_b
    inv_error = b.e / abs_b**2 + LOCAL_PARTIAL_ROUNDINGS * inv
    slope = np.abs(q) / abs_b
    slope_error = (
        e / abs_b + np.abs(q) * b.e / abs_b**2 + LOCAL_PARTIAL_ROUNDINGS * slope
    )
    through_a = _x(inv) * a.D
    through_b = _x(slope) * b.D
    return Bounded(
        q,
        e,
        a.d / _x(b.v) - _x(q / b.v) * b.d,
        through_a + through_b,
        _x(inv) * a.Ed
        + _x(inv_error) * a.D
        + _x(slope) * b.Ed
        + _x(slope_error) * b.D
        + through_a
        + through_b,
        a.paths + b.paths,
    )


def sqrt(b: Bounded) -> Bounded:
    """``sqrt(b)`` for ``b > 0``."""
    root = np.sqrt(b.v)
    e = b.e / (2.0 * root) + root
    slope = 0.5 / root
    slope_error = e / (2.0 * root**2) + LOCAL_PARTIAL_ROUNDINGS * slope
    through = _x(slope) * b.D
    return Bounded(
        root,
        e,
        _x(slope) * b.d,
        through,
        _x(slope) * b.Ed + _x(slope_error) * b.D + through,
        b.paths,
    )


def scale(a: Bounded, factor: float, relative_error: float = 0.0) -> Bounded:
    """``factor * a`` where each lane holds ``factor`` to ``relative_error`` u."""
    v = factor * a.v
    magnitude = abs(factor)
    return Bounded(
        v,
        magnitude * a.e + (1.0 + relative_error) * np.abs(v),
        factor * a.d,
        magnitude * a.D,
        magnitude * a.Ed + (1.0 + relative_error) * magnitude * a.D,
        a.paths,
    )


def extra_roundings(a: Bounded, roundings: float) -> Bounded:
    """Charge ``roundings`` more relative roundings to a value and its partials.

    For operations a lane forms with more roundings than the modelled form, such
    as ``x ** 2.0`` through ``pow`` against ``x * x``.
    """
    return Bounded(
        a.v,
        a.e + roundings * np.abs(a.v),
        a.d,
        a.D,
        a.Ed + roundings * a.D,
        a.paths,
    )


def total(a: Bounded, axis: int | tuple[int, ...]) -> Bounded:
    """Sum over value axes ``axis``, charged for its worst order."""
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(ax % a.v.ndim for ax in axes)
    count = int(np.prod([a.v.shape[ax] for ax in axes]))
    v = np.sum(a.v, axis=axes)
    return Bounded(
        v,
        np.sum(a.e, axis=axes) + (count - 1) * np.sum(np.abs(a.v), axis=axes),
        np.sum(a.d, axis=axes),
        np.sum(a.D, axis=axes),
        np.sum(a.Ed, axis=axes),
        np.sum(a.paths, axis=axes),
    )


def mean(a: Bounded, axis: int | tuple[int, ...]) -> Bounded:
    """``sum / n``, with ``1/n`` allowed one rounding (``sum * (1/n)``)."""
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    count = int(np.prod([a.v.shape[ax % a.v.ndim] for ax in axes]))
    return scale(total(a, axes), 1.0 / count, relative_error=1.0)


def positive_part(a: Bounded) -> Bounded:
    """``max(a, 0)``, which is exact; refuses a state whose sign rounding decides.

    Where ``a`` exceeds its own rounding bound both lanes take the active branch
    and where ``-a`` does both return an exact zero with a zero derivative.  A
    component inside the band would let the lanes branch differently, which no
    rounding bound covers, so it is an error rather than a tolerance.
    """
    band = a.e * UNIT_ROUNDOFF
    active = a.v > band
    inactive = a.v < -band
    if not np.all(active | inactive):
        raise ValueError("a max(x, 0) argument lies inside its own rounding band")
    keep = _x(active)
    return Bounded(
        np.where(active, a.v, 0.0),
        np.where(active, a.e, 0.0),
        np.where(keep, a.d, 0.0),
        np.where(keep, a.D, 0.0),
        np.where(keep, a.Ed, 0.0),
        np.where(keep, a.paths, 0.0),
    )


def envelope(a: Bounded, b: Bounded) -> Bounded:
    """One quantity formed two ways (one per lane): the larger bookkeeping.

    The values agree to first order; ``a``'s are kept.
    """
    a, b = _aligned(a, b)
    return Bounded(
        a.v,
        np.maximum(a.e, b.e),
        a.d,
        np.maximum(a.D, b.D),
        np.maximum(a.Ed, b.Ed),
        np.maximum(a.paths, b.paths),
    )


def stack(parts: Sequence[Bounded], axis: int = 0) -> Bounded:
    ndim = parts[0].v.ndim + 1
    axis = axis % ndim
    return Bounded(
        np.stack([p.v for p in parts], axis=axis),
        np.stack([p.e for p in parts], axis=axis),
        np.stack([p.d for p in parts], axis=axis),
        np.stack([p.D for p in parts], axis=axis),
        np.stack([p.Ed for p in parts], axis=axis),
        np.stack([p.paths for p in parts], axis=axis),
    )


def dot(a: Bounded, b: Bounded) -> Bounded:
    """Contraction over the last value axis."""
    return total(mul(a, b), -1)


def norm(a: Bounded) -> Bounded:
    """Euclidean norm over the last value axis, ``sqrt(sum(a * a))``."""
    return sqrt(dot(a, a))


def cross(a: Bounded, b: Bounded) -> Bounded:
    """Cross product over the last value axis (length three)."""
    return stack(
        [
            sub(mul(a[..., 1], b[..., 2]), mul(a[..., 2], b[..., 1])),
            sub(mul(a[..., 2], b[..., 0]), mul(a[..., 0], b[..., 2])),
            sub(mul(a[..., 0], b[..., 1]), mul(a[..., 1], b[..., 0])),
        ],
        axis=-1,
    )


def matmul_constant(a: Bounded, matrix: np.ndarray) -> Bounded:
    """``a @ matrix`` over the last value axis, ``matrix`` exact and shared."""
    columns = [
        total(
            stack(
                [
                    scale(a[..., row], float(matrix[row, col]))
                    for row in range(matrix.shape[0])
                ],
                axis=-1,
            ),
            -1,
        )
        for col in range(matrix.shape[1])
    ]
    return stack(columns, axis=-1)


def compose(outer: Bounded, inner: Bounded) -> Bounded:
    """``outer`` as a function of ``inner``: chain the bookkeeping through it.

    ``outer`` (value axes ``(..., A)``) was traced with the ``K`` entries of
    ``inner``'s last value axis as its derivative components, seeded with their
    values and value errors, a unit derivative and no derivative error; leading
    value axes are shared batch axes.  Every path from a variable into ``outer``
    passes through exactly one entry of ``inner`` and every recurrence above is
    linear in ``(d, D, Ed, paths)``, so the result equals the direct trace --
    except after :func:`envelope`, whose componentwise maximum it can only exceed.
    """
    return Bounded(
        outer.v,
        outer.e,
        np.einsum("...ak,...kc->...ac", outer.d, inner.d),
        np.einsum("...ak,...kc->...ac", outer.D, inner.D),
        np.einsum("...ak,...kc->...ac", outer.Ed, inner.D)
        + np.einsum("...ak,...kc->...ac", outer.D, inner.Ed),
        np.einsum("...ak,...kc->...ac", outer.paths, inner.paths),
    )


def cross_implementation_bounds(quantity: Bounded) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on the difference of two implementations: value, derivative."""
    return (
        COMPARED_IMPLEMENTATIONS * quantity.value_bound(),
        COMPARED_IMPLEMENTATIONS * quantity.derivative_bound(),
    )
