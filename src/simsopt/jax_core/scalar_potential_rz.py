"""JAX kernels for ``ScalarPotentialRZMagneticField`` expressions."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import sympy as sp

from ._math_utils import as_jax_float64 as _as_jax_float64
from ._sympy_to_jax import lower_sympy_expression, lower_sympy_expressions


ScalarPotentialKernel = Callable[[object], jax.Array]

_R_SYMBOL, _Z_SYMBOL, _PHI_SYMBOL = sp.symbols("R Z phi")
_TINY_SYMPY_TERM = sp.Float("1e-30") * _PHI_SYMBOL * _R_SYMBOL * _Z_SYMBOL


def scalar_potential_rz_kernels(
    phi_expr: sp.Expr,
) -> tuple[
    ScalarPotentialKernel,
    ScalarPotentialKernel,
]:
    """Build JIT-compiled ``B`` and ``dB/dX`` kernels for a scalar potential."""

    lower_sympy_expression(phi_expr)
    phi_R = phi_expr.diff(_R_SYMBOL)
    phi_Phi_over_R = phi_expr.diff(_PHI_SYMBOL) / _R_SYMBOL
    phi_Z = phi_expr.diff(_Z_SYMBOL)
    b_eval = lower_sympy_expressions(
        (
            phi_R + _TINY_SYMPY_TERM,
            phi_Phi_over_R + _TINY_SYMPY_TERM,
            phi_Z + _TINY_SYMPY_TERM,
        )
    )
    dB_eval = lower_sympy_expressions(
        (
            _TINY_SYMPY_TERM
            + sp.cos(_PHI_SYMBOL) * phi_R.diff(_R_SYMBOL)
            - (sp.sin(_PHI_SYMBOL) / _R_SYMBOL) * phi_R.diff(_PHI_SYMBOL),
            _TINY_SYMPY_TERM
            + sp.cos(_PHI_SYMBOL) * phi_Phi_over_R.diff(_R_SYMBOL)
            - (sp.sin(_PHI_SYMBOL) / _R_SYMBOL) * phi_Phi_over_R.diff(_PHI_SYMBOL),
            _TINY_SYMPY_TERM
            + sp.cos(_PHI_SYMBOL) * phi_Z.diff(_R_SYMBOL)
            - (sp.sin(_PHI_SYMBOL) / _R_SYMBOL) * phi_Z.diff(_PHI_SYMBOL),
            _TINY_SYMPY_TERM
            + sp.sin(_PHI_SYMBOL) * phi_R.diff(_R_SYMBOL)
            + (sp.cos(_PHI_SYMBOL) / _R_SYMBOL) * phi_R.diff(_PHI_SYMBOL),
            _TINY_SYMPY_TERM
            + sp.sin(_PHI_SYMBOL) * phi_Phi_over_R.diff(_R_SYMBOL)
            + (sp.cos(_PHI_SYMBOL) / _R_SYMBOL) * phi_Phi_over_R.diff(_PHI_SYMBOL),
            _TINY_SYMPY_TERM
            + sp.sin(_PHI_SYMBOL) * phi_Z.diff(_R_SYMBOL)
            + (sp.cos(_PHI_SYMBOL) / _R_SYMBOL) * phi_Z.diff(_PHI_SYMBOL),
            _TINY_SYMPY_TERM + phi_R.diff(_Z_SYMBOL),
            _TINY_SYMPY_TERM + phi_Phi_over_R.diff(_Z_SYMBOL),
            _TINY_SYMPY_TERM + phi_Z.diff(_Z_SYMBOL),
        )
    )

    @jax.jit
    def B(points: object) -> jax.Array:
        points_arr = _points(points)
        r, phi, z = _cartesian_to_cylindrical(points_arr)
        Br, Bphi, Bz = b_eval(r, z, phi)
        return _cylindrical_field_to_cartesian(Br, Bphi, Bz, phi)

    @jax.jit
    def dB_by_dX(points: object) -> jax.Array:
        points_arr = _points(points)
        r, phi, z = _cartesian_to_cylindrical(points_arr)
        Br, Bphi, _ = b_eval(r, z, phi)
        values = dB_eval(r, z, phi)
        dB_cyl = jnp.stack(values, axis=1).reshape((points_arr.shape[0], 3, 3))
        return _cylindrical_dB_to_cartesian(points_arr, r, phi, Br, Bphi, dB_cyl)

    return B, dB_by_dX


def _points(points: object) -> jax.Array:
    points_arr = _as_jax_float64(points)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3); got {tuple(points_arr.shape)!r}."
        )
    return points_arr


def _cartesian_to_cylindrical(
    points: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = points[:, 0]
    y = points[:, 1]
    r = jnp.sqrt(x * x + y * y)
    phi = jnp.arctan2(y, x)
    z = points[:, 2]
    return r, phi, z


def _cylindrical_field_to_cartesian(
    Br: jax.Array, Bphi: jax.Array, Bz: jax.Array, phi: jax.Array
) -> jax.Array:
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    return jnp.stack(
        (
            Br * cos_phi - Bphi * sin_phi,
            Br * sin_phi + Bphi * cos_phi,
            Bz,
        ),
        axis=1,
    )


def _cylindrical_dB_to_cartesian(
    points: jax.Array,
    r: jax.Array,
    phi: jax.Array,
    Br: jax.Array,
    Bphi: jax.Array,
    dB_cyl: jax.Array,
) -> jax.Array:
    x = points[:, 0]
    y = points[:, 1]
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    r3 = r * r * r
    dcosphidx = -x * x / r3 + 1.0 / r
    dsinphidx = -x * y / r3
    dcosphidy = -x * y / r3
    dsinphidy = -y * y / r3 + 1.0 / r

    dBrdx = dB_cyl[:, 0, 0]
    dBrdy = dB_cyl[:, 1, 0]
    dBrdz = dB_cyl[:, 2, 0]
    dBphidx = dB_cyl[:, 0, 1]
    dBphidy = dB_cyl[:, 1, 1]
    dBphidz = dB_cyl[:, 2, 1]

    dBdx = jnp.stack(
        (
            dBrdx * cos_phi + Br * dcosphidx - dBphidx * sin_phi - Bphi * dsinphidx,
            dBrdx * sin_phi + Br * dsinphidx + dBphidx * cos_phi + Bphi * dcosphidx,
            dB_cyl[:, 0, 2],
        ),
        axis=1,
    )
    dBdy = jnp.stack(
        (
            dBrdy * cos_phi + Br * dcosphidy - dBphidy * sin_phi - Bphi * dsinphidy,
            dBrdy * sin_phi + Br * dsinphidy + dBphidy * cos_phi + Bphi * dcosphidy,
            dB_cyl[:, 1, 2],
        ),
        axis=1,
    )
    dBdz = jnp.stack(
        (
            dBrdz * cos_phi - dBphidz * sin_phi,
            dBrdz * sin_phi + dBphidz * cos_phi,
            dB_cyl[:, 2, 2],
        ),
        axis=1,
    )
    return jnp.stack((dBdx, dBdy, dBdz), axis=1)
