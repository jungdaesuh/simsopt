"""ABI-compatibility helpers for simsoptpp Boozer residual entrypoints.

simsoptpp exposes two supported signatures for each Boozer residual
entrypoint (``boozer_residual``, ``boozer_residual_ds``,
``boozer_residual_ds2``, ``boozer_dresidual_dc``):

* ``with_I`` — the current ABI inserts an explicit net toroidal plasma
  current ``I`` immediately after ``G``. All simsopt Boozer-surface consumers
  compute the residual against a vacuum (coil-only) BiotSavart field, so the
  net toroidal plasma current through the Boozer surface is zero by
  construction and we pass ``I=0.0``.
* ``alpha_only`` — the legacy ABI without the explicit ``I`` argument.

The first call per entrypoint probes the ``with_I`` signature; subsequent
calls reuse the cached decision. The cache is keyed per entrypoint so
independent wrappers do not interfere with one another.
"""

from __future__ import annotations

from typing import Callable, Optional


KEY_BOOZER_RESIDUAL = "boozer_residual"
KEY_BOOZER_RESIDUAL_DS = "boozer_residual_ds"
KEY_BOOZER_RESIDUAL_DS2 = "boozer_residual_ds2"
KEY_BOOZER_DRESIDUAL_DC = "boozer_dresidual_dc"

_CALL_MODES: dict[str, str] = {}


def _call_with_abi_fallback(
    key: str,
    func: Callable,
    with_I_args: tuple,
    alpha_only_args: tuple,
):
    """Dispatch ``func`` using the cached (or probed) simsoptpp ABI signature.

    Parameters
    ----------
    key:
        Stable identifier for the entrypoint (e.g. ``"boozer_residual"``).
    func:
        The simsoptpp C++ entrypoint to call.
    with_I_args:
        Positional arguments for the current ABI (includes ``I=0.0``).
    alpha_only_args:
        Positional arguments for the legacy ABI (no explicit ``I``).
    """
    mode = _CALL_MODES.get(key)
    if mode == "with_I":
        return func(*with_I_args)
    if mode == "alpha_only":
        return func(*alpha_only_args)
    try:
        value = func(*with_I_args)
    except TypeError as exc:
        if "incompatible function arguments" not in str(exc):
            raise
        _CALL_MODES[key] = "alpha_only"
        return func(*alpha_only_args)
    _CALL_MODES[key] = "with_I"
    return value


def _reset_call_modes() -> None:
    """Test helper: clear cached ABI-probe decisions."""
    _CALL_MODES.clear()


def _get_call_mode(key: str) -> Optional[str]:
    """Test helper: read the cached ABI-probe decision for ``key``."""
    return _CALL_MODES.get(key)
