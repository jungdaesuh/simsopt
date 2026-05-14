"""Pure-JAX kernels for the analytic Landreman-Sengupta Boozer field.

The CPU oracle is :class:`simsopt.field.boozermagneticfield.BoozerAnalytic`,
which evaluates ``modB``, ``G``, ``I``, ``iota``, ``psip``, ``K`` and their
derivatives via closed-form analytic expressions (no spline/Fourier
machinery).  The kernels here mirror those expressions exactly on a
frozen pytree state, so a wrapper class can route the public API through
JAX without inheriting from ``sopp.BoozerMagneticField``.

Math reference (CPU oracle lines under ``boozermagneticfield.py``):

* ``modB`` at 236-243
* ``G(s) = G0 + s*G1`` at 220-223
* ``I(s) = I0 + s*I1`` at 228-231
* ``iota(s) = iota0`` at 214-215
* ``psip(s) = psi0*s*iota0`` at 209-212
* ``K(s, theta, zeta) = K1*r*sin(theta - N*zeta)`` at 274-281
* derivative kernels at 245-299

All scalar parameters are stored as float64 ``jax.Array`` scalars so the
frozen state is a clean pytree leaf set (no Python ints, no broadcasts to
heterogeneous dtypes).

Note: every kernel returns a ``(num_points,)`` array.  The wrapper class
in :mod:`simsopt.field.boozermagneticfield_jax` reshapes to ``(num_points, 1)``
to match the upstream ``_*_impl`` shape contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._math_utils import as_jax_float64 as _as_jax_float64

__all__ = [
    "BoozerAnalyticFrozenState",
    "_eval_G",
    "_eval_I",
    "_eval_K",
    "_eval_dGds",
    "_eval_dIds",
    "_eval_dKdtheta",
    "_eval_dKdzeta",
    "_eval_diotads",
    "_eval_dmodBds",
    "_eval_dmodBdtheta",
    "_eval_dmodBdzeta",
    "_eval_iota",
    "_eval_modB",
    "_eval_psip",
    "freeze_boozer_analytic_state",
]


# ----------------------------------------------------------------------
# Frozen state pytree
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BoozerAnalyticFrozenState:
    """Immutable scalar payload for ``BoozerAnalytic``-style evaluation.

    Every field is a float64 ``jax.Array`` scalar (shape ``()``).  ``N`` is
    semantically an integer helicity, but stored as float64 so the pytree
    has uniform leaf dtype and broadcasts cleanly inside ``cos``/``sin``
    arguments.  The wrapper class exposes a Python ``int`` accessor.
    """

    etabar: jax.Array
    B0: jax.Array
    Bbar: jax.Array
    N: jax.Array
    G0: jax.Array
    I0: jax.Array
    G1: jax.Array
    I1: jax.Array
    K1: jax.Array
    iota0: jax.Array
    psi0: jax.Array


jax.tree_util.register_dataclass(
    BoozerAnalyticFrozenState,
    data_fields=[
        "etabar",
        "B0",
        "Bbar",
        "N",
        "G0",
        "I0",
        "G1",
        "I1",
        "K1",
        "iota0",
        "psi0",
    ],
    meta_fields=[],
)


def freeze_boozer_analytic_state(
    etabar,
    B0,
    N,
    G0,
    psi0,
    iota0,
    Bbar=1.0,
    I0=0.0,
    G1=0.0,
    I1=0.0,
    K1=0.0,
) -> BoozerAnalyticFrozenState:
    """Build a frozen state pytree from the CPU constructor argument list.

    Positional argument order matches
    :class:`simsopt.field.boozermagneticfield.BoozerAnalytic.__init__`
    exactly: ``(etabar, B0, N, G0, psi0, iota0, Bbar=1.0, I0=0.0,
    G1=0.0, I1=0.0, K1=0.0)``.
    """
    return BoozerAnalyticFrozenState(
        etabar=_as_jax_float64(etabar),
        B0=_as_jax_float64(B0),
        Bbar=_as_jax_float64(Bbar),
        N=_as_jax_float64(N),
        G0=_as_jax_float64(G0),
        I0=_as_jax_float64(I0),
        G1=_as_jax_float64(G1),
        I1=_as_jax_float64(I1),
        K1=_as_jax_float64(K1),
        iota0=_as_jax_float64(iota0),
        psi0=_as_jax_float64(psi0),
    )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _split_points(points: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    # ``jnp.unstack`` lowers to JAX-native primitives without integer-indexed
    # ``dynamic_slice`` calls, which under ``JAX_TRANSFER_GUARD=disallow`` would
    # otherwise need to push the Python literal index to the device.
    columns = jnp.unstack(points, axis=1)
    return columns[0], columns[1], columns[2]


def _r_value(s: jax.Array, psi0: jax.Array, Bbar: jax.Array) -> jax.Array:
    """Return ``r(s) = sqrt(|2 * s * psi0 / Bbar|)``.

    Matches the CPU formula ``r = sqrt(np.abs(2*psi/self.Bbar))`` with
    ``psi = s * psi0``. Avoid Python literal scalars so the kernel
    stays clean under ``JAX_TRANSFER_GUARD=disallow``.
    """
    psi = s * psi0
    return jnp.sqrt(jnp.abs((psi + psi) / Bbar))


def _angle(
    state: BoozerAnalyticFrozenState, thetas: jax.Array, zetas: jax.Array
) -> jax.Array:
    """Return ``θ − N*ζ`` with broadcasting compatible with CPU NumPy."""
    return thetas - state.N * zetas


# ----------------------------------------------------------------------
# Kernels — scalar profiles
# ----------------------------------------------------------------------


def _eval_psip(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, _thetas, _zetas = _split_points(points)
    return state.psi0 * s * state.iota0


def _eval_iota(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, _thetas, _zetas = _split_points(points)
    # ``state.iota0 + (s - s)`` is identically ``iota0`` per sample but
    # stays on-device under ``transfer_guard("disallow")`` (no constant
    # broadcast that triggers a host-to-device transfer).
    return state.iota0 + (s - s)


def _eval_diotads(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, _thetas, _zetas = _split_points(points)
    return s - s


def _eval_G(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, _thetas, _zetas = _split_points(points)
    return state.G0 + s * state.G1


def _eval_dGds(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, _thetas, _zetas = _split_points(points)
    return state.G1 + (s - s)


def _eval_I(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, _thetas, _zetas = _split_points(points)
    return state.I0 + s * state.I1


def _eval_dIds(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, _thetas, _zetas = _split_points(points)
    return state.I1 + (s - s)


# ----------------------------------------------------------------------
# Kernels — modB and derivatives
# ----------------------------------------------------------------------


def _eval_modB(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, thetas, zetas = _split_points(points)
    r = _r_value(s, state.psi0, state.Bbar)
    arg = _angle(state, thetas, zetas)
    return state.B0 + state.B0 * state.etabar * r * jnp.cos(arg)


def _eval_dmodBds(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, thetas, zetas = _split_points(points)
    psi = s * state.psi0
    r = _r_value(s, state.psi0, state.Bbar)
    # CPU: drdpsi = 0.5*r/psi; drds = drdpsi*psi0
    # Use r/(psi+psi) instead of (0.5*r)/psi to avoid Python float literals
    # crossing the JAX boundary under ``transfer_guard("disallow")``.
    drds = r * state.psi0 / (psi + psi)
    arg = _angle(state, thetas, zetas)
    return state.B0 * state.etabar * drds * jnp.cos(arg)


def _eval_dmodBdtheta(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, thetas, zetas = _split_points(points)
    r = _r_value(s, state.psi0, state.Bbar)
    arg = _angle(state, thetas, zetas)
    return -state.B0 * state.etabar * r * jnp.sin(arg)


def _eval_dmodBdzeta(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, thetas, zetas = _split_points(points)
    r = _r_value(s, state.psi0, state.Bbar)
    arg = _angle(state, thetas, zetas)
    return state.N * state.B0 * state.etabar * r * jnp.sin(arg)


# ----------------------------------------------------------------------
# Kernels — K and derivatives
# ----------------------------------------------------------------------


def _eval_K(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, thetas, zetas = _split_points(points)
    r = _r_value(s, state.psi0, state.Bbar)
    arg = _angle(state, thetas, zetas)
    return state.K1 * r * jnp.sin(arg)


def _eval_dKdtheta(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, thetas, zetas = _split_points(points)
    r = _r_value(s, state.psi0, state.Bbar)
    arg = _angle(state, thetas, zetas)
    return state.K1 * r * jnp.cos(arg)


def _eval_dKdzeta(state: BoozerAnalyticFrozenState, points: jax.Array) -> jax.Array:
    s, thetas, zetas = _split_points(points)
    r = _r_value(s, state.psi0, state.Bbar)
    arg = _angle(state, thetas, zetas)
    return -state.N * state.K1 * r * jnp.cos(arg)
