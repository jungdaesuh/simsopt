"""JAX-backed cylindrical wrapper around the rectangular grid interpolant.

This module implements the wrapper-level contract demanded by
``InterpolatedField`` (item 15-sub) on top of the immutable
rectangular-grid kernel exposed by
:mod:`simsopt.jax_core.regular_grid_interp` (item 13).

The C++ ``InterpolatedField`` class
(``src/simsoptpp/magneticfield_interpolated.h``):

1. Samples a source :class:`MagneticField` on a cylindrical grid in
   :math:`(r, \\phi, z)` at construction time.
2. At every evaluation, folds the query point through ``nfp``
   rotational symmetry (``phi`` modulo :math:`2\\pi/n_{fp}`) and through
   stellarator symmetry (``z<0`` reflects across the midplane with a
   compensating ``B_r`` sign flip when ``stellsym=True``).
3. Evaluates the rectangular interpolant on the folded cylindrical
   coordinates and unfolds the cylindrical output before rotating it
   back to Cartesian using the *original* (unfolded) ``phi``.

This module exposes the same machinery as a pure JAX pipeline. Two
``RegularGridInterpolant3DSpec`` instances are constructed per source
field — one for the cylindrical ``B_cyl`` triple and one for the
cylindrical ``\\nabla|B|`` triple — and the symmetry-fold / unfold layer
is implemented entirely in JAX so the evaluation hot path is JIT-friendly
and never dispatches back to Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from .regular_grid_interp import (
    RegularGridInterpolant3DDeviceSpec,
    RegularGridInterpolant3DSpec,
    _evaluate_batch_jit,
    build_regular_grid_interpolant_3d_device_spec,
)


# ── Spec ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InterpolatedFieldSpec:
    """Immutable spec for the cylindrical-wrapper JAX interpolant.

    Fields:

    - ``nfp``: number of field periods.
    - ``stellsym``: whether stellarator symmetry is exploited.
    - ``B_spec``: rectangular-grid interpolant for the cylindrical
      ``(B_r, B_\\phi, B_z)`` triple on the :math:`(r, \\phi, z)`
      reduced-symmetry mesh.
    - ``GradAbsB_spec``: same, for the cylindrical
      :math:`\\nabla |B|` triple.
    - ``_device_B``, ``_device_GradAbsB``:
      :class:`RegularGridInterpolant3DDeviceSpec` bundles staged once at
      construction time so the JAX evaluation path stays clean under
      :func:`jax.transfer_guard("disallow")`.
    """

    nfp: int
    stellsym: bool
    B_spec: RegularGridInterpolant3DSpec
    GradAbsB_spec: RegularGridInterpolant3DSpec
    _device_B: RegularGridInterpolant3DDeviceSpec
    _device_GradAbsB: RegularGridInterpolant3DDeviceSpec


def make_interpolated_field_spec(
    *,
    nfp: int,
    stellsym: bool,
    B_spec: RegularGridInterpolant3DSpec,
    GradAbsB_spec: RegularGridInterpolant3DSpec,
) -> InterpolatedFieldSpec:
    """Construct an :class:`InterpolatedFieldSpec` with pre-staged device specs."""

    return InterpolatedFieldSpec(
        nfp=int(nfp),
        stellsym=bool(stellsym),
        B_spec=B_spec,
        GradAbsB_spec=GradAbsB_spec,
        _device_B=build_regular_grid_interpolant_3d_device_spec(B_spec),
        _device_GradAbsB=build_regular_grid_interpolant_3d_device_spec(GradAbsB_spec),
    )


# ── Coordinate conversions ───────────────────────────────────────────


def _cart_to_cyl(points_cart: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert Cartesian ``(x, y, z)`` points to cylindrical ``(r, phi, z)``.

    The C++ implementation uses ``simsopt_cyl_from_cart`` (see
    ``src/simsoptpp/magneticfield.h``) which calls ``std::atan2``. JAX's
    ``jnp.arctan2`` matches ``std::atan2`` on the principal branch.

    Returns three flat ``(N,)`` arrays.
    """

    x = points_cart[:, 0]
    y = points_cart[:, 1]
    z = points_cart[:, 2]
    r = jnp.sqrt(x * x + y * y)
    phi = jnp.arctan2(y, x)
    return r, phi, z


# ── Symmetry folding ─────────────────────────────────────────────────


def _fold_phi_nfp(phi: jax.Array, nfp: int) -> jax.Array:
    """Fold ``phi`` into ``[0, 2*pi/nfp)``.

    Matches the C++ ``exploit_symmetries_points`` machinery, which first
    maps a possibly-negative ``phi`` into ``[0, 2*pi)`` and then takes
    the remainder modulo ``2*pi/nfp``. The C++ folds ``int(phi/period)``
    which is floor division for non-negative ``phi`` — we use
    ``jnp.mod`` which is equivalent and additionally handles the
    negative case introduced by stellarator reflection.
    """

    period = 2.0 * jnp.pi / float(nfp)
    # Map (-pi, pi] -> [0, 2*pi) first so the modulo step is the same
    # as the C++ integer-division step on non-negative phi.
    phi_wrapped = jnp.where(phi < 0.0, phi + 2.0 * jnp.pi, phi)
    return jnp.mod(phi_wrapped, period)


def _fold_symmetry(
    r: jax.Array,
    phi: jax.Array,
    z: jax.Array,
    *,
    nfp: int,
    stellsym: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply the ``nfp`` and stellarator symmetry folds.

    Returns ``(r_fold, phi_fold, z_fold, sign_br)`` where ``sign_br`` is
    ``-1`` for points that were reflected across the midplane (under
    stellarator symmetry) and ``+1`` otherwise.

    Matches ``exploit_symmetries_points`` in
    ``src/simsoptpp/magneticfield_interpolated.h``. Note in particular
    that the stellsym branch first negates ``z`` and applies
    ``phi -> 2*pi - phi`` before the ``nfp`` modulo reduction — this
    ordering is significant because the modulo step uses the
    post-reflection ``phi``.
    """

    reflect = jnp.asarray(stellsym, dtype=jnp.bool_) & (z < 0.0)
    z_pre = jnp.where(reflect, -z, z)
    phi_pre = jnp.where(reflect, 2.0 * jnp.pi - phi, phi)
    sign_br = jnp.where(reflect, -1.0, 1.0)

    phi_folded = _fold_phi_nfp(phi_pre, nfp)
    return r, phi_folded, z_pre, sign_br


def _unfold_B_cyl(B_cyl_fold: jax.Array, sign_br: jax.Array) -> jax.Array:
    """Undo the stellsym ``B_r`` sign flip on cylindrical ``B``.

    Matches ``apply_symmetries_to_B_cyl`` in
    ``src/simsoptpp/magneticfield_interpolated.h``.
    """

    B_r = B_cyl_fold[:, 0] * sign_br
    B_phi = B_cyl_fold[:, 1]
    B_z = B_cyl_fold[:, 2]
    return jnp.stack([B_r, B_phi, B_z], axis=1)


def _unfold_GradAbsB_cyl(GradAbsB_cyl_fold: jax.Array, sign_br: jax.Array) -> jax.Array:
    """Undo the stellsym sign flip on cylindrical ``\\nabla|B|``.

    The C++ ``apply_symmetries_to_GradAbsB_cyl`` flips components 1
    and 2 — i.e. the ``\\phi`` and ``z`` derivatives of ``|B|`` — for
    reflected points. The ``r`` derivative is invariant.
    """

    # ``sign`` is +1 for non-reflected and -1 for reflected points.
    GradAbsB_r = GradAbsB_cyl_fold[:, 0]
    GradAbsB_phi = GradAbsB_cyl_fold[:, 1] * sign_br
    GradAbsB_z = GradAbsB_cyl_fold[:, 2] * sign_br
    return jnp.stack([GradAbsB_r, GradAbsB_phi, GradAbsB_z], axis=1)


# ── Cylindrical -> Cartesian for field vectors ───────────────────────


def _cyl_vector_to_cart(field_cyl: jax.Array, phi: jax.Array) -> jax.Array:
    """Rotate a cylindrical vector field to Cartesian using the original ``phi``.

    Matches the C++ ``_B_impl`` and ``_GradAbsB_impl`` in
    ``src/simsoptpp/magneticfield_interpolated.h``. The unfolded ``phi``
    (i.e. the original ``arctan2(y, x)`` of the query point) is used —
    not the folded ``phi`` that addressed the interpolant.
    """

    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    F_r = field_cyl[:, 0]
    F_phi = field_cyl[:, 1]
    F_z = field_cyl[:, 2]
    F_x = cos_phi * F_r - sin_phi * F_phi
    F_y = sin_phi * F_r + cos_phi * F_phi
    return jnp.stack([F_x, F_y, F_z], axis=1)


# ── Public field-evaluation entry points ─────────────────────────────


@partial(
    jax.jit,
    static_argnames=(
        "nfp",
        "stellsym",
        "unfold_kind",
        "degree",
        "value_size",
        "out_of_bounds_ok",
    ),
)
def _evaluate_cyl_field_jit(
    points_cart: jax.Array,
    initial_cyl: jax.Array,
    *,
    cell_table: jax.Array,
    cell_to_row: jax.Array,
    nodes: jax.Array,
    scalings: jax.Array,
    xmesh: jax.Array,
    ymesh: jax.Array,
    zmesh: jax.Array,
    xmin: jax.Array,
    xmax: jax.Array,
    ymin: jax.Array,
    ymax: jax.Array,
    zmin: jax.Array,
    zmax: jax.Array,
    hx: jax.Array,
    hy: jax.Array,
    hz: jax.Array,
    nx: jax.Array,
    ny: jax.Array,
    nz: jax.Array,
    sentinel_row: jax.Array,
    nfp: int,
    stellsym: bool,
    unfold_kind: int,
    degree: int,
    value_size: int,
    out_of_bounds_ok: bool,
) -> jax.Array:
    """Symmetry-fold + rectangular kernel + cylindrical cache preservation.

    The kernel is wrapped in a single ``jit`` boundary so the host-side
    helper can pre-stage all device arrays once per call. The
    ``unfold_kind`` selector is statically baked into the trace:

    - ``unfold_kind == 0``: ``B`` semantics. ``sign_br`` flips
      :math:`B_r` only.
    - ``unfold_kind == 1``: ``\\nabla|B|`` semantics. ``sign_br`` flips
      the ``\\phi`` and ``z`` components.
    """

    r, phi, z = _cart_to_cyl(points_cart)
    r_fold, phi_fold, z_fold, sign_br = _fold_symmetry(
        r, phi, z, nfp=nfp, stellsym=stellsym
    )
    query = jnp.stack([r_fold, phi_fold, z_fold], axis=1)
    # Native InterpolatedField applies symmetry signs after evaluate_batch,
    # so skipped/OOB rows preserve the old output buffer first and are then
    # transformed by the current query's symmetry flags.
    field_cyl_fold = _evaluate_batch_jit(
        query,
        initial_output=initial_cyl,
        cell_table=cell_table,
        cell_to_row=cell_to_row,
        nodes=nodes,
        scalings=scalings,
        xmesh=xmesh,
        ymesh=ymesh,
        zmesh=zmesh,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        hx=hx,
        hy=hy,
        hz=hz,
        nx=nx,
        ny=ny,
        nz=nz,
        sentinel_row=sentinel_row,
        degree=degree,
        value_size=value_size,
        out_of_bounds_ok=out_of_bounds_ok,
    )
    if unfold_kind == 0:
        return _unfold_B_cyl(field_cyl_fold, sign_br)
    return _unfold_GradAbsB_cyl(field_cyl_fold, sign_br)


@partial(
    jax.jit,
    static_argnames=(
        "nfp",
        "stellsym",
        "unfold_kind",
        "degree",
        "value_size",
        "out_of_bounds_ok",
    ),
)
def _evaluate_cart_field_jit(
    points_cart: jax.Array,
    initial_cyl: jax.Array,
    *,
    cell_table: jax.Array,
    cell_to_row: jax.Array,
    nodes: jax.Array,
    scalings: jax.Array,
    xmesh: jax.Array,
    ymesh: jax.Array,
    zmesh: jax.Array,
    xmin: jax.Array,
    xmax: jax.Array,
    ymin: jax.Array,
    ymax: jax.Array,
    zmin: jax.Array,
    zmax: jax.Array,
    hx: jax.Array,
    hy: jax.Array,
    hz: jax.Array,
    nx: jax.Array,
    ny: jax.Array,
    nz: jax.Array,
    sentinel_row: jax.Array,
    nfp: int,
    stellsym: bool,
    unfold_kind: int,
    degree: int,
    value_size: int,
    out_of_bounds_ok: bool,
) -> jax.Array:
    """Evaluate the cylindrical kernel and rotate the result to Cartesian."""

    _, phi, _ = _cart_to_cyl(points_cart)
    field_cyl = _evaluate_cyl_field_jit(
        points_cart,
        initial_cyl,
        cell_table=cell_table,
        cell_to_row=cell_to_row,
        nodes=nodes,
        scalings=scalings,
        xmesh=xmesh,
        ymesh=ymesh,
        zmesh=zmesh,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        hx=hx,
        hy=hy,
        hz=hz,
        nx=nx,
        ny=ny,
        nz=nz,
        sentinel_row=sentinel_row,
        nfp=nfp,
        stellsym=stellsym,
        unfold_kind=unfold_kind,
        degree=degree,
        value_size=value_size,
        out_of_bounds_ok=out_of_bounds_ok,
    )
    return _cyl_vector_to_cart(field_cyl, phi)


@partial(
    jax.jit,
    static_argnames=(
        "nfp",
        "stellsym",
        "unfold_kind",
        "degree",
        "value_size",
        "out_of_bounds_ok",
    ),
)
def _evaluate_cart_field_zero_jit(
    points_cart: jax.Array,
    *,
    cell_table: jax.Array,
    cell_to_row: jax.Array,
    nodes: jax.Array,
    scalings: jax.Array,
    xmesh: jax.Array,
    ymesh: jax.Array,
    zmesh: jax.Array,
    xmin: jax.Array,
    xmax: jax.Array,
    ymin: jax.Array,
    ymax: jax.Array,
    zmin: jax.Array,
    zmax: jax.Array,
    hx: jax.Array,
    hy: jax.Array,
    hz: jax.Array,
    nx: jax.Array,
    ny: jax.Array,
    nz: jax.Array,
    sentinel_row: jax.Array,
    nfp: int,
    stellsym: bool,
    unfold_kind: int,
    degree: int,
    value_size: int,
    out_of_bounds_ok: bool,
) -> jax.Array:
    return _evaluate_cart_field_jit(
        points_cart,
        jnp.zeros((points_cart.shape[0], value_size), dtype=jnp.float64),
        cell_table=cell_table,
        cell_to_row=cell_to_row,
        nodes=nodes,
        scalings=scalings,
        xmesh=xmesh,
        ymesh=ymesh,
        zmesh=zmesh,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        hx=hx,
        hy=hy,
        hz=hz,
        nx=nx,
        ny=ny,
        nz=nz,
        sentinel_row=sentinel_row,
        nfp=nfp,
        stellsym=stellsym,
        unfold_kind=unfold_kind,
        degree=degree,
        value_size=value_size,
        out_of_bounds_ok=out_of_bounds_ok,
    )


def _evaluate_cart_field_zero(
    points_cart: jax.Array,
    *,
    device_spec: RegularGridInterpolant3DDeviceSpec,
    nfp: int,
    stellsym: bool,
    unfold_kind: int,
) -> jax.Array:
    return _evaluate_cart_field_zero_jit(
        points_cart,
        cell_table=device_spec.cell_table,
        cell_to_row=device_spec.cell_to_row,
        nodes=device_spec.nodes,
        scalings=device_spec.scalings,
        xmesh=device_spec.xmesh,
        ymesh=device_spec.ymesh,
        zmesh=device_spec.zmesh,
        xmin=device_spec.xmin,
        xmax=device_spec.xmax,
        ymin=device_spec.ymin,
        ymax=device_spec.ymax,
        zmin=device_spec.zmin,
        zmax=device_spec.zmax,
        hx=device_spec.hx,
        hy=device_spec.hy,
        hz=device_spec.hz,
        nx=device_spec.nx,
        ny=device_spec.ny,
        nz=device_spec.nz,
        sentinel_row=device_spec.sentinel_row,
        nfp=int(nfp),
        stellsym=bool(stellsym),
        unfold_kind=int(unfold_kind),
        degree=device_spec.degree,
        value_size=device_spec.value_size,
        out_of_bounds_ok=device_spec.out_of_bounds_ok,
    )


def _evaluate_cyl_field(
    points_cart: jax.Array,
    *,
    initial_cyl: jax.Array,
    device_spec: RegularGridInterpolant3DDeviceSpec,
    nfp: int,
    stellsym: bool,
    unfold_kind: int,
) -> jax.Array:
    return _evaluate_cyl_field_jit(
        points_cart,
        initial_cyl,
        cell_table=device_spec.cell_table,
        cell_to_row=device_spec.cell_to_row,
        nodes=device_spec.nodes,
        scalings=device_spec.scalings,
        xmesh=device_spec.xmesh,
        ymesh=device_spec.ymesh,
        zmesh=device_spec.zmesh,
        xmin=device_spec.xmin,
        xmax=device_spec.xmax,
        ymin=device_spec.ymin,
        ymax=device_spec.ymax,
        zmin=device_spec.zmin,
        zmax=device_spec.zmax,
        hx=device_spec.hx,
        hy=device_spec.hy,
        hz=device_spec.hz,
        nx=device_spec.nx,
        ny=device_spec.ny,
        nz=device_spec.nz,
        sentinel_row=device_spec.sentinel_row,
        nfp=int(nfp),
        stellsym=bool(stellsym),
        unfold_kind=int(unfold_kind),
        degree=device_spec.degree,
        value_size=device_spec.value_size,
        out_of_bounds_ok=device_spec.out_of_bounds_ok,
    )


def interpolated_field_B(
    spec: InterpolatedFieldSpec, points_cart: jax.Array
) -> jax.Array:
    """Evaluate the interpolated cylindrical ``B`` field in Cartesian.

    Args:
        spec: :class:`InterpolatedFieldSpec` built from a source
            ``MagneticField`` via the public
            :class:`simsopt.field.interpolated_field_jax.InterpolatedFieldJAX`
            wrapper.
        points_cart: ``(N, 3)`` Cartesian query points.

    Returns:
        ``(N, 3)`` Cartesian ``B`` vectors.
    """

    return _evaluate_cart_field_zero(
        points_cart,
        device_spec=spec._device_B,
        nfp=int(spec.nfp),
        stellsym=bool(spec.stellsym),
        unfold_kind=0,
    )


def interpolated_field_GradAbsB(
    spec: InterpolatedFieldSpec, points_cart: jax.Array
) -> jax.Array:
    """Evaluate the interpolated cylindrical ``\\nabla |B|`` in Cartesian.

    Args:
        spec: :class:`InterpolatedFieldSpec`.
        points_cart: ``(N, 3)`` Cartesian query points.

    Returns:
        ``(N, 3)`` Cartesian ``\\nabla |B|`` vectors.
    """

    return _evaluate_cart_field_zero(
        points_cart,
        device_spec=spec._device_GradAbsB,
        nfp=int(spec.nfp),
        stellsym=bool(spec.stellsym),
        unfold_kind=1,
    )


def interpolated_field_B_cyl_with_initial(
    spec: InterpolatedFieldSpec, points_cart: jax.Array, initial_cyl: jax.Array
) -> jax.Array:
    """Evaluate cylindrical ``B`` while preserving supplied output rows.

    ``initial_cyl`` is the previous cylindrical cache contents for the
    same point-buffer shape. It matches the native
    ``InterpolatedField::_B_cyl_impl`` contract: skipped or out-of-domain
    rows are left untouched when ``extrapolate=True``.
    """

    return _evaluate_cyl_field(
        points_cart,
        initial_cyl=initial_cyl,
        device_spec=spec._device_B,
        nfp=int(spec.nfp),
        stellsym=bool(spec.stellsym),
        unfold_kind=0,
    )


def interpolated_field_GradAbsB_cyl_with_initial(
    spec: InterpolatedFieldSpec, points_cart: jax.Array, initial_cyl: jax.Array
) -> jax.Array:
    """Evaluate cylindrical ``\\nabla|B|`` with native cache semantics."""

    return _evaluate_cyl_field(
        points_cart,
        initial_cyl=initial_cyl,
        device_spec=spec._device_GradAbsB,
        nfp=int(spec.nfp),
        stellsym=bool(spec.stellsym),
        unfold_kind=1,
    )


__all__ = [
    "InterpolatedFieldSpec",
    "interpolated_field_B",
    "interpolated_field_B_cyl_with_initial",
    "interpolated_field_GradAbsB",
    "interpolated_field_GradAbsB_cyl_with_initial",
    "make_interpolated_field_spec",
]
