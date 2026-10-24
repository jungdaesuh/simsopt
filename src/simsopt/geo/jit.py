import inspect
import operator
import os
from collections import OrderedDict, defaultdict
from functools import wraps

import jax
import numpy as np
from jax import jit as jaxjit

from .config import parameters

if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    jax.config.update("jax_platform_name", "cpu")


def native_jax_device():
    """Device on which native simsopt runs the functions compiled by :func:`jit`.

    Returns:
        jax.Device object: First CPU device when CPU is enabled, otherwise the
            first default device (for example with ``JAX_PLATFORMS=cuda``).
    """
    platforms = jax.config.jax_platforms
    if platforms and "cpu" not in platforms.split(","):
        return jax.devices()[0]
    return jax.devices("cpu")[0]


def _owned(value):
    # Native caches can overwrite NumPy views while JAX still reads them.
    return value.copy(order="K") if isinstance(value, np.ndarray) else value


def _owned_plain_leaf(value):
    if type(value) in (dict, defaultdict, OrderedDict):
        # These built-in containers have known copy/setitem contracts.
        result = value.copy()
        for key, item in value.items():
            result[key] = _snapshot_plain(item)
        return result
    return _owned(value)


def _snapshot_plain(value):
    # Treat dicts as leaves so JAX does not sort their keys in plain Python calls.
    return jax.tree_util.tree_map(
        _owned_plain_leaf,
        value,
        is_leaf=lambda item: type(item) in (dict, defaultdict, OrderedDict),
    )


def _static_arguments(fun, options):
    """Resolve JAX's static selectors without changing the actual call layout."""
    argnums = options.get("static_argnums")
    argnames = options.get("static_argnames")
    compile_options = dict(options)
    if argnums is not None:
        # Match JAX's scalar __index__ protocol before considering an iterable.
        try:
            argnums = (operator.index(argnums),)
        except TypeError:
            argnums = tuple(map(operator.index, argnums))
        compile_options["static_argnums"] = argnums
    if argnames is not None:
        argnames = (argnames,) if isinstance(argnames, str) else tuple(argnames)
        compile_options["static_argnames"] = argnames

    if (argnums is None) != (argnames is None):
        # As in JAX, one selector implies the other through the signature.
        try:
            signature = inspect.signature(fun)
        except (TypeError, ValueError):
            signature = None
        signature_parameters = (
            () if signature is None else tuple(signature.parameters.items())
        )
        if signature is not None and argnames is not None:
            argnums = tuple(
                i
                for i, (name, param) in enumerate(signature_parameters)
                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and name in argnames
            )
        elif signature is not None and argnums is not None:
            argnames = tuple(
                name
                for i, (name, param) in enumerate(signature_parameters)
                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and i in argnums
            )
    return frozenset(argnums or ()), frozenset(argnames or ()), compile_options


def jit(fun, **args):
    """Compile ``fun`` with :func:`jax.jit` for native simsopt.

    Dynamic NumPy pytree leaves are private copies, preserving ndarray subclasses.
    Static compiled arguments retain their identity and compilation cache keys.
    Compiled calls use ``jax.default_device(native_jax_device())`` for uncommitted
    inputs; committed JAX inputs keep their placement. With jit disabled, all
    NumPy leaves are copied and ``fun`` runs in the caller's device context.
    Plain dict, defaultdict and OrderedDict inputs preserve their key order;
    other registered pytrees use JAX traversal. Opaque unregistered container
    leaves retain their identity and contents. Captured arrays and JAX inputs
    already aliasing external mutable storage are outside this ownership boundary.
    Disabling jit still returns a wrapper, with
    ``fun`` available as ``__wrapped__`` and its metadata retained.

    Args:
        fun (callable): Function to compile.
        **args: Keyword arguments forwarded to :func:`jax.jit`.

    Returns:
        callable: The input-owning wrapper, compiled when jit is enabled.
    """
    use_jit = parameters["jit"]
    if use_jit:
        static_argnums, static_argnames, compile_options = _static_arguments(fun, args)
        run = jaxjit(fun, **compile_options)
    else:
        static_argnums, static_argnames = (), ()
        run = fun

    @wraps(fun)
    def on_native_device(*fun_args, **fun_kwargs):
        if not use_jit:
            return run(
                *tuple(_snapshot_plain(value) for value in fun_args),
                **{name: _snapshot_plain(value) for name, value in fun_kwargs.items()},
            )
        fun_args = tuple(
            value
            if i in static_argnums or i - len(fun_args) in static_argnums
            else jax.tree_util.tree_map(_owned, value)
            for i, value in enumerate(fun_args)
        )
        fun_kwargs = {
            name: value
            if name in static_argnames
            else jax.tree_util.tree_map(_owned, value)
            for name, value in fun_kwargs.items()
        }
        with jax.default_device(native_jax_device()):
            return run(*fun_args, **fun_kwargs)

    return on_native_device
