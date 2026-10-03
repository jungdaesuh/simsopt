"""Pure JAX SSOT kernels for finite-build multifilament construction.

Wave R4 item 20: provide the kernel-level filament geometry builder used
by ``simsopt.geo.finitebuild.create_multifilament_grid``. The host adapter
still owns the public ``CurveFilament`` Optimizable graph; this module
exposes the underlying pure functions so JAX-native consumers can build
filament gammas + gammadashs directly from immutable specs without
constructing ``Optimizable`` objects.

The filament construction follows Singh et al. (2020),
``doi:10.1017/S0022377820000756``: each filament is the parent curve plus
``dn * normal + db * binormal`` (and its parameter derivative for
``gammadash``). The normal/binormal vectors come from the rotated
centroid or Frenet frame, depending on ``frame_kind``.

Conventions
-----------
* All geometry tensors use ``(N, 3)`` layout where ``N`` is the
  quadrature-point count of the base curve.
* ``dn`` / ``db`` are scalar shifts in normal / binormal directions.
* ``filament_offsets`` is a sequence of ``(dn, db)`` tuples — each
  entry produces one filament. The grid layout owned by the host helper
  ``create_multifilament_grid`` is reduced to this offsets sequence at
  the SSOT boundary.
"""

from __future__ import annotations

from typing import Iterable

import jax
import numpy as np

from ._math_utils import as_jax_float64 as _as_jax_float64
from .curve_geometry import (
    _curve_geometry_with_third_derivative_from_dofs,
    _mapped_input_dofs,
    _rotation_alpha_and_dash_from_dofs,
    curve_geometry_from_dofs,
)
from .framedcurve import (
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotated_frenet_frame,
    rotated_frenet_frame_dash,
)
from .specs import CurveFilamentSpec


__all__ = [
    "build_filament_gamma_and_dash",
    "build_filament_gammas",
    "compute_filament_offsets",
]


def compute_filament_offsets(
    *,
    numfilaments_n: int,
    numfilaments_b: int,
    gapsize_n: float,
    gapsize_b: float,
) -> tuple[tuple[float, float], ...]:
    """Return the ``(dn, db)`` offsets for the multifilament grid.

    Mirrors the geometry of
    ``simsopt.geo.finitebuild.create_multifilament_grid`` so the JAX SSOT
    layer can be parity-checked against the host adapter without
    reconstructing ``CurveFilament`` objects.
    """
    if int(numfilaments_n) % 2 == 1:
        shifts_n = np.arange(int(numfilaments_n)) - int(numfilaments_n) // 2
    else:
        shifts_n = np.arange(int(numfilaments_n)) - int(numfilaments_n) / 2.0 + 0.5
    shifts_n = shifts_n.astype(np.float64) * float(gapsize_n)
    if int(numfilaments_b) % 2 == 1:
        shifts_b = np.arange(int(numfilaments_b)) - int(numfilaments_b) // 2
    else:
        shifts_b = np.arange(int(numfilaments_b)) - int(numfilaments_b) / 2.0 + 0.5
    shifts_b = shifts_b.astype(np.float64) * float(gapsize_b)

    return tuple(
        (float(shifts_n[i]), float(shifts_b[j]))
        for i in range(int(numfilaments_n))
        for j in range(int(numfilaments_b))
    )


def _frame_geometry_from_spec(spec: CurveFilamentSpec, dofs: jax.Array):
    """Return the frame components and base geometry needed for one filament."""
    base_dofs = _mapped_input_dofs(spec.base_curve_map, dofs)
    alpha, alphadash = _rotation_alpha_and_dash_from_dofs(
        spec.rotation,
        spec.rotation_map,
        dofs,
    )
    if spec.frame_kind == "frenet":
        gamma, gammadash, gammadashdash, gammadashdashdash = (
            _curve_geometry_with_third_derivative_from_dofs(spec.base_curve, base_dofs)
        )
        _t, normal, binormal = rotated_frenet_frame(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
        )
        _td, normal_dash, binormal_dash = rotated_frenet_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            gammadashdashdash,
            alpha,
            alphadash,
        )
    else:
        gamma, gammadash, gammadashdash = curve_geometry_from_dofs(
            spec.base_curve, base_dofs
        )
        _t, normal, binormal = rotated_centroid_frame(
            gamma,
            gammadash,
            alpha,
        )
        _td, normal_dash, binormal_dash = rotated_centroid_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
            alphadash,
        )
    return gamma, gammadash, normal, binormal, normal_dash, binormal_dash


def build_filament_gamma_and_dash(
    spec: CurveFilamentSpec,
    *,
    dofs: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(gamma, gammadash)`` for one filament defined by ``spec``.

    The filament is ``gamma_base + dn * normal + db * binormal``; its
    parameter derivative uses ``normal_dash`` / ``binormal_dash`` from
    the rotated frame. This is the SSOT version of
    ``CurveFilament._gamma_jax_from_full_dofs`` /
    ``CurveFilament._gammadash_jax_from_full_dofs`` without the
    surrounding ``Optimizable`` graph.
    """
    owner_dofs = spec.dofs if dofs is None else _as_jax_float64(dofs)
    (
        gamma,
        gammadash,
        normal,
        binormal,
        normal_dash,
        binormal_dash,
    ) = _frame_geometry_from_spec(spec, owner_dofs)
    dn_jax = _as_jax_float64(spec.dn)
    db_jax = _as_jax_float64(spec.db)
    filament_gamma = gamma + dn_jax * normal + db_jax * binormal
    filament_gammadash = gammadash + dn_jax * normal_dash + db_jax * binormal_dash
    return filament_gamma, filament_gammadash


def build_filament_gammas(
    spec: CurveFilamentSpec,
    filament_offsets: Iterable[tuple[float, float]],
    *,
    dofs: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    """Return ``(gamma, gammadash)`` per filament in ``filament_offsets``.

    Reuses one frame-geometry build per call: the base curve geometry,
    rotated frame, and frame derivative are evaluated once and then
    shifted by each ``(dn, db)`` offset. This makes the SSOT layer
    naturally cheaper than building one Optimizable per filament.
    """
    owner_dofs = spec.dofs if dofs is None else _as_jax_float64(dofs)
    (
        gamma,
        gammadash,
        normal,
        binormal,
        normal_dash,
        binormal_dash,
    ) = _frame_geometry_from_spec(spec, owner_dofs)
    filaments: list[tuple[jax.Array, jax.Array]] = []
    for dn, db in filament_offsets:
        dn_jax = _as_jax_float64(dn)
        db_jax = _as_jax_float64(db)
        filament_gamma = gamma + dn_jax * normal + db_jax * binormal
        filament_gammadash = gammadash + dn_jax * normal_dash + db_jax * binormal_dash
        filaments.append((filament_gamma, filament_gammadash))
    return tuple(filaments)
