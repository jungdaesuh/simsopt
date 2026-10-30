"""Device-neutral ``jax.jit`` that owns its NumPy arguments before dispatch.

simsopt overwrites cached NumPy arrays (``gamma()``, field caches, evaluation
points) in place when DOFs or points change. ``jax.jit`` dispatches
asynchronously, and on CPU it may read an aligned NumPy argument's memory
directly, or copy an unaligned one in the background, after the call has
returned. A cache overwritten in that window silently changes the result.
:func:`jit` copies every NumPy array leaf of the dynamic arguments
synchronously, through ``snapshot_host_tree`` (the host-ownership owner),
before calling the compiled function.

Placement is left to JAX: no device is pinned, so NumPy and uncommitted
inputs land on JAX's default device (the one ``set_backend`` selects) and
committed inputs keep their device, exactly as with ``jax.jit``.
"""

from __future__ import annotations

import inspect
import operator
import types
from collections.abc import Callable, Iterable
from functools import update_wrapper
from itertools import repeat
from typing import overload

import jax
import numpy as np

from simsopt_jax.runtime.host_boundary import snapshot_host_tree

__all__ = ["CopyingJit", "jit"]

# Argument types that never hold a NumPy array; JAX arrays include tracers.
_NUMPY_FREE_TYPES = (jax.Array, bool, int, float, complex, str, type(None))
_NO_POSITIONS: frozenset[int] = frozenset()


def _index_tuple(argnums) -> tuple[int, ...]:
    """``static_argnums`` as a tuple of ints, accepting one index or several."""
    try:
        return (operator.index(argnums),)
    except TypeError:
        return tuple(map(operator.index, argnums))


def _str_tuple(argnames: str | Iterable[str]) -> tuple[str, ...]:
    return (argnames,) if isinstance(argnames, str) else tuple(argnames)


def _static_selectors(
    fun: Callable, options: dict
) -> tuple[frozenset[int], frozenset[str]]:
    """The positional indices and keyword names ``jax.jit`` treats as static.

    As ``jax.jit`` does, a selector given alone implies the other through the
    positional-or-keyword parameters of ``fun``'s signature; both given are
    taken as they are. Negative indices stay negative and are resolved
    against the number of positional arguments of each call.
    """
    argnums = options.get("static_argnums")
    argnames = options.get("static_argnames")
    nums = () if argnums is None else _index_tuple(argnums)
    names = () if argnames is None else _str_tuple(argnames)
    if (argnums is None) != (argnames is None):
        try:
            parameters = inspect.signature(fun).parameters
        except (TypeError, ValueError):
            parameters = {}
        positional = [
            name
            for name, parameter in parameters.items()
            if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        if argnames is None:
            names = tuple(name for i, name in enumerate(positional) if i in nums)
        else:
            nums = tuple(i for i, name in enumerate(positional) if name in names)
    return frozenset(nums), frozenset(names)


def _holds_numpy(value) -> bool:
    """Whether ``value`` is, or is a pytree with, a NumPy array leaf."""
    if isinstance(value, _NUMPY_FREE_TYPES):
        return False
    if isinstance(value, np.ndarray):
        return True
    return any(map(isinstance, jax.tree.leaves(value), repeat(np.ndarray)))


class CopyingJit:
    """A function compiled with ``jax.jit`` whose NumPy arguments are copied per call.

    Calls copy the NumPy array leaves of dynamic arguments and pass every
    other value through by identity: JAX arrays, tracers, Python scalars and
    static arguments, so the compilation cache keys are those of
    ``jax.jit``. Calls without NumPy leaves skip the copy. Attributes of the
    compiled function (``lower``, ``trace``, ``eval_shape``,
    ``clear_cache``, ...) are forwarded unchanged, and the wrapper binds as a
    method like the compiled function does.

    Not protected: NumPy arrays that ``fun`` captures from an enclosing scope
    and JAX arrays that already alias NumPy memory.
    """

    def __init__(self, fun: Callable, **options) -> None:
        """Wrap ``jax.jit(fun, **options)``, which compiles at the first call.

        Args:
            fun (callable): Function to compile.
            **options: Keyword arguments of :func:`jax.jit`, for example
                ``static_argnums``, ``static_argnames`` or ``donate_argnums``.
        """
        # First, so the metadata copied from ``fun`` cannot replace the state below.
        update_wrapper(self, fun)
        self._compiled = jax.jit(fun, **options)  # noqa: TID251 - the one raw jit; this wrapper owns NumPy inputs.
        self._static_argnums, self._static_argnames = _static_selectors(fun, options)

    def __call__(self, *args, **kwargs):
        """Copy NumPy leaves of dynamic arguments, then call the compiled function.

        Args:
            *args: Positional arguments of ``fun``; arrays of any shape.
            **kwargs: Keyword arguments of ``fun``; arrays of any shape.

        Returns:
            pytree: The compiled function's outputs, placed by JAX.
        """
        static = (
            {i + len(args) if i < 0 else i for i in self._static_argnums}
            if self._static_argnums
            else _NO_POSITIONS
        )
        dynamic = (
            (value for i, value in enumerate(args) if i not in static) if static else args
        )
        if any(map(_holds_numpy, dynamic)):
            args = tuple(
                value if i in static else snapshot_host_tree(value)
                for i, value in enumerate(args)
            )
        names = self._static_argnames
        if kwargs and any(
            _holds_numpy(value) for name, value in kwargs.items() if name not in names
        ):
            kwargs = {
                name: value if name in names else snapshot_host_tree(value)
                for name, value in kwargs.items()
            }
        return self._compiled(*args, **kwargs)

    def __get__(self, instance, owner=None):
        """Bind to ``instance`` as a method, as ``jax.jit`` functions do."""
        return self if instance is None else types.MethodType(self, instance)

    def __getattr__(self, name: str):
        """Forward attributes of the compiled function, such as ``lower``."""
        # ``_compiled`` reaches here only on an instance made without
        # ``__init__`` (``copy.copy``); raising avoids infinite recursion.
        if name == "_compiled":
            raise AttributeError(name)
        return getattr(self._compiled, name)


@overload
def jit(fun: Callable, /, **options) -> CopyingJit: ...


@overload
def jit(fun: None = None, /, **options) -> Callable[[Callable], CopyingJit]: ...


def jit(
    fun: Callable | None = None, /, **options
) -> CopyingJit | Callable[[Callable], CopyingJit]:
    """``jax.jit`` that copies NumPy arguments before asynchronous dispatch.

    Use it wherever ``jax.jit`` would be used: as ``@jit``,
    ``@partial(jit, static_argnames=...)``, ``@jit(static_argnames=...)`` or
    ``jit(f, static_argnums=...)``. See :class:`CopyingJit` for what is copied.

    Args:
        fun (callable or None): Function to compile; None returns a decorator
            that applies ``options``.
        **options: Keyword arguments of :func:`jax.jit`.

    Returns:
        CopyingJit or callable: The compiled wrapper, or a decorator producing
            one when ``fun`` is None.
    """
    if fun is None:
        return lambda f: CopyingJit(f, **options)
    return CopyingJit(fun, **options)
