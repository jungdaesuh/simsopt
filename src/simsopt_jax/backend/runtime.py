"""Process configuration of the optional JAX backend, and its host boundary.

Call ``set_backend("jax", device="cpu")`` or ``set_backend("jax", device="gpu")``
before constructing ``JaxBiotSavart``; ``set_backend("native")`` restores the
default. Importing the adapter does not call it.

The JAX backend computes in float64 (``jax_enable_x64``, which importing
``simsopt_jax`` enables and ``set_backend`` enables again) with matrix
products at the highest precision, on the first device of the selected
platform; it does not distribute work across devices. Before JAX starts its
backends, ``set_backend`` exports, for this process and its children:

* ``JAX_PLATFORMS``: the selected platform. CUDA keeps the CPU next to it,
  where native simsopt runs the kernels ``simsopt.geo.jit.jit`` compiles,
  unless the caller's ``JAX_PLATFORMS`` lists cuda without cpu.
* ``XLA_FLAGS`` defaults, each unless the caller set that flag already:
  ``--xla_cpu_opt_preset=FAST_COMPILE`` on the CPU; on the GPU
  ``--xla_gpu_experimental_enable_fusion_autotuner=false`` and
  ``--xla_gpu_autotune_level=0``, which fix the kernels a fresh compile picks,
  and ``--xla_gpu_exclude_nondeterministic_ops=true``, which makes XLA use
  deterministic implementations (of scatter-add, for example), so repeated GPU
  runs give the same bits.

XLA reads both only when JAX starts. A later ``set_backend`` exports neither
and warns when GPU defaults are missing; with several platforms running it
makes the selected one JAX's default device. A platform JAX cannot start, or
did not start, raises ``RuntimeError``.

Every other setting is JAX's or XLA's own environment variable, for example
``JAX_DEBUG_NANS``, ``JAX_DISABLE_JIT``, ``JAX_TRANSFER_GUARD``,
``JAX_COMPILATION_CACHE_DIR``, ``XLA_PYTHON_CLIENT_PREALLOCATE``,
``XLA_PYTHON_CLIENT_MEM_FRACTION``, ``XLA_PYTHON_CLIENT_ALLOCATOR`` and
``TF_GPU_ALLOCATOR`` (set the XLA ones before JAX starts).

The host boundary lives here too: ``host_value``, ``host_array`` and
``host_tree`` copy device values to the host explicitly,
``snapshot_host_tree`` gives NumPy inputs a private copy before asynchronous
JAX dispatch reads them, and ``disallow_host_transfers`` and
``allow_host_transfers`` scope JAX's transfer guard.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, TypeVar

import jax
import jax.extend.backend
from jax import core as jax_core
import numpy as np

from simsopt_jax._requirements import enable_float64

__all__ = [
    "BackendConfig",
    "allow_host_transfers",
    "block_until_ready",
    "disallow_host_transfers",
    "get_backend_config",
    "get_runtime_jax_device",
    "host_array",
    "host_tree",
    "host_tree_after_ready",
    "host_value",
    "register_backend_cache_clear",
    "set_backend",
    "snapshot_host_tree",
]

JaxDevice = Literal["cpu", "gpu"]

# Per device: its ``JAX_PLATFORMS`` entry, the names JAX's backends report for
# it, and the XLA flags ``set_backend`` adds before JAX starts.
_JAX_PLATFORM: dict[JaxDevice, str] = {"cpu": "cpu", "gpu": "cuda"}
_BACKEND_NAMES: dict[JaxDevice, frozenset[str]] = {
    "cpu": frozenset({"cpu"}),
    "gpu": frozenset({"cuda", "gpu"}),
}
_XLA_FLAG_DEFAULTS: dict[JaxDevice, tuple[str, ...]] = {
    "cpu": ("--xla_cpu_opt_preset=FAST_COMPILE",),
    "gpu": (
        "--xla_gpu_experimental_enable_fusion_autotuner=false",
        "--xla_gpu_autotune_level=0",
        "--xla_gpu_exclude_nondeterministic_ops=true",
    ),
}
_DEVICE_OF_PLATFORM: dict[str, JaxDevice] = {"cpu": "cpu", "cuda": "gpu"}


@dataclass(frozen=True)
class BackendConfig:
    """The backend :func:`set_backend` installed.

    Args:
        backend (str): native or jax.
        device (str or None): cpu or gpu for the JAX backend; None for native.
    """

    backend: Literal["native", "jax"]
    device: JaxDevice | None = None


_lock = threading.RLock()
_config = BackendConfig("native")
_cache_clear_callbacks: dict[tuple[str, str], Callable[[], None]] = {}


def get_backend_config() -> BackendConfig:
    """Return the configuration :func:`set_backend` installed last.

    Returns:
        BackendConfig object: The native configuration until the first call.
    """
    return _config


def register_backend_cache_clear(callback: Callable[[], None]) -> None:
    """Run ``callback`` whenever :func:`set_backend` installs a configuration.

    Args:
        callback (Callable[[], None]): Cache-clearing function, keyed by module and
            qualified name; registration replaces a callback with the same key.
    """
    with _lock:
        _cache_clear_callbacks[(callback.__module__, callback.__qualname__)] = callback


def get_runtime_jax_device():
    """Return the device on which the JAX backend places values nobody placed.

    The JAX backend's device is its platform's first device. Under the native
    backend it is the first device of the first platform this process's JAX was
    configured with (``jax.config.jax_platforms``); before JAX has a
    configuration (not imported yet, or its first import still running in
    another thread), ``JAX_PLATFORMS`` is what JAX will read, so a native
    process that never touched JAX does not import it here.

    Returns:
        jax.Device or None: The device, or None under the native backend when
            no platform is configured.
    """
    device = _config.device
    if device is None:
        jax_config = getattr(sys.modules.get("jax"), "config", None)
        platforms = (
            os.environ.get("JAX_PLATFORMS")
            if jax_config is None
            else jax_config.jax_platforms
        )
        first = next(
            (part.strip().lower() for part in (platforms or "").split(",") if part.strip()),
            None,
        )
        device = _DEVICE_OF_PLATFORM.get(first or "")
        if device is None:
            return None
    return jax.local_devices(backend=device)[0]


def _jax_backends_initialized() -> bool:
    """Whether JAX started its backends, checked without starting them."""
    xla_bridge = sys.modules.get("jax._src.xla_bridge")
    return xla_bridge is not None and bool(xla_bridge.backends_are_initialized())


def _missing_xla_flags(device: JaxDevice) -> tuple[str, ...]:
    """The XLA flag defaults of ``device`` whose flag ``XLA_FLAGS`` does not set."""
    present = {token.split("=", 1)[0] for token in os.environ.get("XLA_FLAGS", "").split()}
    return tuple(
        flag for flag in _XLA_FLAG_DEFAULTS[device] if flag.split("=", 1)[0] not in present
    )


def _export_xla_flag_defaults(device: JaxDevice) -> None:
    """Append ``device``'s missing XLA flag defaults to ``XLA_FLAGS``.

    XLA reads them only when JAX starts its backends; the test support calls
    this before it starts them.
    """
    missing = _missing_xla_flags(device)
    if missing:
        os.environ["XLA_FLAGS"] = " ".join(
            filter(None, (os.environ.get("XLA_FLAGS", "").strip(), *missing))
        )


def _startup_platforms(device: JaxDevice) -> str:
    """The ``jax_platforms`` value that starts ``device``'s platform as JAX's default."""
    if device == "cpu":
        return "cpu"
    requested = {part.strip().lower() for part in os.environ.get("JAX_PLATFORMS", "").split(",")}
    return "cuda" if "cuda" in requested and "cpu" not in requested else "cuda,cpu"


def _default_device(device: JaxDevice, platform: str):
    """The ``jax_default_device`` value that makes ``platform`` JAX's default.

    ``jax_platforms`` only decides which platforms JAX starts, so with several
    running the selected one becomes the default through
    ``jax_default_device``: a selection that already runs on it is kept, a
    stale one is cleared (None) when JAX's own default backend is the
    platform, and otherwise the platform's first device is selected.
    """
    names = _BACKEND_NAMES[device]
    selected = jax.config.jax_default_device
    default_backend = jax.default_backend()
    if selected is None:
        effective = default_backend
    else:
        effective = selected if isinstance(selected, str) else selected.platform
    if effective in names:
        return selected
    if default_backend in names:
        return None
    return jax.devices(platform)[0]


def _start_jax(device: JaxDevice) -> None:
    """Start JAX on ``device``'s platform, or make it the default of the running ones.

    Raises:
        RuntimeError: JAX cannot start the platform, or did not start it.
    """
    if not _jax_backends_initialized():
        platforms = _startup_platforms(device)
        os.environ["JAX_PLATFORMS"] = platforms
        _export_xla_flag_defaults(device)
        jax.config.update("jax_platforms", platforms)
    elif device == "gpu" and (missing := _missing_xla_flags(device)):
        warnings.warn(
            "JAX started its backends before set_backend('jax', device='gpu'), so "
            f"XLA_FLAGS cannot add {' '.join(missing)} to this process, and fresh "
            "GPU compiles may differ between runs. Call set_backend, or export "
            "these flags, before touching JAX devices.",
            RuntimeWarning,
            stacklevel=3,
        )
    # Starts the backends if nothing did; JAX raises RuntimeError for a listed
    # platform it cannot start.
    running = tuple(jax.extend.backend.backends())
    platform = next((name for name in running if name in _BACKEND_NAMES[device]), None)
    if platform is None:
        raise RuntimeError(
            f"set_backend('jax', device={device!r}) needs JAX platform "
            f"{_JAX_PLATFORM[device]!r}, but JAX started only {sorted(running)} in this "
            "process. Call set_backend before touching JAX devices."
        )
    # JAX picks its default backend by the deprecated ``jax_platform_name``
    # before ``jax_platforms``; ``simsopt.geo`` sets it to ``"cpu"`` when no JAX
    # platform variable is set. Cleared, the selected platform is the default.
    jax.config.update("jax_platform_name", "")
    default_device = _default_device(device, platform)
    enable_float64()
    jax.config.update("jax_default_matmul_precision", "highest")
    if default_device is not jax.config.jax_default_device:
        jax.config.update("jax_default_device", default_device)


def set_backend(
    backend: Literal["native", "jax"], *, device: JaxDevice | None = None
) -> BackendConfig:
    """Select the native backend, or the JAX backend on ``device``, for this process.

    The JAX backend starts JAX on the device's platform as the module
    docstring describes, or, in a process whose JAX already started several
    platforms, makes it JAX's default device. Every call runs the callbacks
    registered with :func:`register_backend_cache_clear`. A call that raises
    keeps the previous configuration.

    Args:
        backend (str): native or jax.
        device (str or None): cpu or gpu with jax; omitted with native.

    Returns:
        BackendConfig object: The installed configuration, as
            :func:`get_backend_config` returns it.

    Raises:
        ValueError: Any other combination of backend and device.
        RuntimeError: JAX cannot start the device's platform, or did not start it.
    """
    global _config
    if backend == "native" and device is None:
        config = BackendConfig("native")
    elif backend == "jax" and device in ("cpu", "gpu"):
        config = BackendConfig("jax", device)
    else:
        raise ValueError(
            f"set_backend({backend!r}, device={device!r}) is not valid; use "
            "set_backend('native') or set_backend('jax', device='cpu' or 'gpu')."
        )
    with _lock:
        if config.device is not None:
            _start_jax(config.device)
        _config = config
        for callback in tuple(_cache_clear_callbacks.values()):
            callback()
    return config


# ── Host boundary ─────────────────────────────────────────────────────

_TreeT = TypeVar("_TreeT")


@contextmanager
def disallow_host_transfers() -> Iterator[None]:
    """Refuse IMPLICIT host-to-device transfers for the duration of the block.

    That refusal holds on every backend.  Explicit ``device_get`` passes, and so
    does an explicit ``device_put`` of host values, within one backend, or from
    the CPU to an accelerator.  JAX checks a ``device_put`` of an accelerator
    array to the CPU as an implicit device-to-host copy and refuses it; the
    placement helpers in ``simsopt_jax.backend.dtypes`` gather such arrays with
    ``device_get`` first, so their explicit placements pass.  On CPU --
    verified on jax 0.12.0.1, where no copy actually happens -- implicit
    device-to-host (``np.sum(x)``, ``x.tolist()``) passes too.

    Returns:
        contextlib.AbstractContextManager[None] object: Scoped transfer guard
            that restores the surrounding setting on exit.
    """

    with jax.transfer_guard("disallow"):
        yield


@contextmanager
def allow_host_transfers() -> Iterator[None]:
    """Permit implicit transfers inside one explicit host-driven boundary.

    For host callers that hand device arrays to compiled code while an outer
    strict guard may be active: lowering a program that closes over device
    arrays reads them back, which the strict guard would refuse. The counterpart
    of :func:`disallow_host_transfers`.

    Returns:
        contextlib.AbstractContextManager[None] object: Scoped transfer guard
            that restores the surrounding setting on exit.
    """

    with jax.transfer_guard("allow"):
        yield


def host_value(value: _TreeT) -> _TreeT:
    """Materialize a JAX value or pytree while preserving its Python structure.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.

    Returns:
        pytree object: Same structure with JAX leaves materialized as host
            values of matching shapes and dtypes.
    """
    return jax.device_get(value)


def host_array(
    value: object,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> np.ndarray:
    """Materialize ``value`` to a writeable NumPy array at an explicit D2H boundary.

    Args:
        value (array-like): Value of any shape to materialize on the host.
        dtype (dtype-like or None): Host conversion dtype; None preserves the
            materialized dtype.

    Returns:
        numpy.ndarray: Writable host array with the input shape; already
            writable NumPy storage may be reused.
    """
    array = np.asarray(host_value(value))
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return array


def block_until_ready(value: _TreeT) -> _TreeT:
    """Wait for every JAX leaf and return the same pytree structure and values.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.

    Returns:
        pytree object: Same structure and values after all asynchronous JAX
            leaves are ready.
    """
    return jax.block_until_ready(value)


def snapshot_host_tree(value: _TreeT, *, dtype=None) -> _TreeT:
    """Own NumPy leaves synchronously before asynchronous JAX consumption.

    Device arrays and tracers pass through unchanged. A private contiguous
    NumPy copy prevents both CPU buffer aliasing and delayed transfer reads.
    The copy keeps a NumPy subclass, so JAX still sees, for example, a masked
    array's mask and rejects it as it rejects the original.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.
        dtype (dtype-like or None): Optional dtype for host NumPy leaves and host
            scalars; None preserves dtypes. Device arrays and tracers are not coerced by
            snapshotting.

    Returns:
        pytree object: Same structure with private C-contiguous copies of
            NumPy array leaves, preserving each leaf shape. Device arrays and
            tracers remain unchanged.
    """
    if isinstance(value, (jax.Array, jax_core.Tracer)):
        return value

    def _hostify_leaf(leaf):
        if isinstance(leaf, jax_core.Tracer):
            return leaf
        if isinstance(leaf, np.ndarray):
            leaf_dtype = leaf.dtype if dtype is None else dtype
            return np.array(leaf, dtype=leaf_dtype, copy=True, order="C", subok=True)
        if dtype is not None and (isinstance(leaf, np.generic) or np.isscalar(leaf)):
            return np.asarray(leaf, dtype=dtype)
        return leaf

    return jax.tree.map(_hostify_leaf, value)


def host_tree(value, *, dtype=None):
    """Materialize a pytree on the host and snapshot its NumPy storage.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.
        dtype (dtype-like or None): Conversion dtype for materialized NumPy leaves
            and host scalars; default None preserves dtypes. Device arrays are
            materialized before this conversion; each leaf keeps its shape.

    Returns:
        pytree: Host leaves with the same shape per leaf and privately owned
            NumPy array buffers.
    """
    return snapshot_host_tree(host_value(value), dtype=dtype)


def host_tree_after_ready(value, *, dtype=None):
    """Wait for a pytree, then materialize its array leaves on the host.

    Args:
        value (pytree): JAX or host leaves of arbitrary shape, retaining their Python
            container structure.
        dtype (dtype-like or None): Conversion dtype for materialized NumPy leaves
            and host scalars; default None preserves dtypes. Device arrays are
            materialized before this conversion; each leaf keeps its shape.

    Returns:
        pytree object: Ready host leaves with the same shape per leaf and
            privately owned NumPy array buffers.
    """
    return host_tree(block_until_ready(value), dtype=dtype)
