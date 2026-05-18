"""JAX port of ``regular_grid_interpolant_3d`` (Tier P1 item 13).

This module re-implements the C++ ``RegularGridInterpolant3D`` kernel
(see ``src/simsoptpp/regular_grid_interpolant_3d.h`` and
``src/simsoptpp/regular_grid_interpolant_3d_impl.h``) as a JAX-compatible
piecewise polynomial interpolant on a rectangular cuboid mesh in three
dimensions. The interpolant is vector-valued and uses tensor-product
1D Lagrange polynomials on each cell.

Two construction stages are exposed:

1. ``build_regular_grid_interpolant_3d`` — pure-Python NumPy build pass.
   Materialises the mesh, evaluates the user function on every retained
   degree-of-freedom, builds the per-cell local-value table, and assembles
   an immutable :class:`RegularGridInterpolant3DSpec`.
2. ``build_regular_grid_interpolant_3d_device_spec`` — stage the immutable
   spec arrays once for a JAX device hot path.
3. ``evaluate_batch`` — JAX-compiled evaluation. Given a packed
   ``xyz`` array of evaluation points, returns the interpolated values
   ``fxyz`` of shape ``(N, value_size)``. Skipped cells route to zero
   from the zero-initialised output buffer. Out-of-domain points either
   route to zero (``out_of_bounds_ok=True``) or surface ``NaN``
   (``out_of_bounds_ok=False``).

Autodiff through ``evaluate_batch`` differentiates this local interpolation
polynomial with respect to the query coordinates. It is not the physical
``GradAbsB`` table used by ``InterpolatedFieldJAX``; callers that need the
magnetic-field gradient should use the wrapper's ``GradAbsB`` path instead.

The public surface mirrors the C++ binding contract while leaving the JAX
evaluation loop fully traceable and ``jit``-compatible. The
``evaluate_batch_with_initial`` helper models the mutable C++
``evaluate_batch(xyz, fxyz)`` contract in pure-functional form: retained
cells overwrite the supplied output rows, while skipped or out-of-domain
rows preserve them when ``out_of_bounds_ok=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._math_utils import as_jax_array


# Epsilon used in the C++ kernel to softly clamp points that are within
# floating-point noise of the domain boundary. Matches
# ``regular_grid_interpolant_3d_impl.h:8`` (``_EPS_``).
_BOUNDARY_EPSILON = 1e-13
_CELL_INDEX_DTYPE = jnp.int64
_INT64_MIN_FLOAT = float(np.iinfo(np.int64).min)
_INT64_MAX_FLOAT = float(np.iinfo(np.int64).max)


def _cell_index_array(value: object) -> jax.Array:
    return jnp.asarray(value, dtype=_CELL_INDEX_DTYPE)


def _flat_cell_index(
    xidx: object,
    yidx: object,
    zidx: object,
    *,
    ny: object,
    nz: object,
) -> jax.Array:
    """Return a row-major flat cell index using 64-bit arithmetic."""

    return (
        (_cell_index_array(xidx) * _cell_index_array(ny) + _cell_index_array(yidx))
        * _cell_index_array(nz)
        + _cell_index_array(zidx)
    )


@dataclass(frozen=True)
class InterpolationRule:
    """Closed-form 1D Lagrange interpolation rule on ``[0, 1]``."""

    nodes: np.ndarray
    scalings: np.ndarray
    degree: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nodes",
            _readonly_array_copy(self.nodes, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "scalings",
            _readonly_array_copy(self.scalings, dtype=np.float64),
        )
        object.__setattr__(self, "degree", int(self.degree))


def _build_scalings(nodes: np.ndarray) -> np.ndarray:
    """Return the Lagrange basis denominators for the given node set."""
    nodes_array = np.asarray(nodes, dtype=np.float64)
    degree_plus_one = int(nodes_array.shape[0])
    scalings = np.ones(degree_plus_one, dtype=np.float64)
    for idx in range(degree_plus_one):
        for other in range(degree_plus_one):
            if other == idx:
                continue
            scalings[idx] *= 1.0 / (nodes_array[idx] - nodes_array[other])
    return scalings


def UniformInterpolationRule(degree: int) -> InterpolationRule:
    """Equispaced Lagrange nodes on ``[0, 1]``.

    Matches ``UniformInterpolationRule`` in
    ``regular_grid_interpolant_3d.h``.
    """
    degree_int = int(degree)
    if degree_int < 1:
        raise ValueError(f"degree must be >= 1, got {degree_int}")
    degree_inv = 1.0 / float(degree_int)
    nodes = np.array(
        [i * degree_inv for i in range(degree_int + 1)],
        dtype=np.float64,
    )
    return InterpolationRule(
        nodes=nodes,
        scalings=_build_scalings(nodes),
        degree=degree_int,
    )


def ChebyshevInterpolationRule(degree: int) -> InterpolationRule:
    """Chebyshev-Lobatto Lagrange nodes on ``[0, 1]``.

    Matches ``ChebyshevInterpolationRule`` in
    ``regular_grid_interpolant_3d.h``.
    """
    degree_int = int(degree)
    if degree_int < 1:
        raise ValueError(f"degree must be >= 1, got {degree_int}")
    degree_inv = 1.0 / float(degree_int)
    nodes = np.array(
        [-0.5 * np.cos(i * np.pi * degree_inv) + 0.5 for i in range(degree_int + 1)],
        dtype=np.float64,
    )
    return InterpolationRule(
        nodes=nodes,
        scalings=_build_scalings(nodes),
        degree=degree_int,
    )


@dataclass(frozen=True)
class RegularGridInterpolant3DSpec:
    """Immutable spec for evaluation on a built interpolant.

    Fields:

    - ``rule``: 1D Lagrange rule (nodes + scalings + degree).
    - ``nx``, ``ny``, ``nz``: cell counts per axis.
    - ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``: axis
      ranges.
    - ``hx``, ``hy``, ``hz``: cell sizes (``(xmax-xmin)/nx`` etc.).
    - ``xmesh``, ``ymesh``, ``zmesh``: ``nx+1`` / ``ny+1`` / ``nz+1`` mesh
      node positions used to recover the local fractional coordinate.
    - ``value_size``: output dimension of the interpolated function.
    - ``out_of_bounds_ok``: matches the C++ flag. ``True`` preserves
      the initial output row for out-of-domain queries; ``False`` routes
      them to ``NaN`` so the caller can detect the error post-hoc.
    - ``cell_to_row``: ``(nx*ny*nz,)`` int64 lookup that maps a flat
      cell index to its row in ``cell_table``. Skipped cells (and the
      explicit out-of-domain sentinel) point at the last row, which is
      forced to zero.
    - ``cell_table``: ``(cells_to_keep + 1, degree+1, degree+1, degree+1,
      value_size)`` float64 array of per-cell DOF values, padded with a
      zero sentinel row used for safe skipped / OOB gathers.
    """

    rule: InterpolationRule
    nx: int
    ny: int
    nz: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    hx: float
    hy: float
    hz: float
    xmesh: np.ndarray
    ymesh: np.ndarray
    zmesh: np.ndarray
    value_size: int
    out_of_bounds_ok: bool
    cell_to_row: np.ndarray
    cell_table: np.ndarray

    def __post_init__(self) -> None:
        rule = InterpolationRule(
            nodes=self.rule.nodes,
            scalings=self.rule.scalings,
            degree=int(self.rule.degree),
        )
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "nx", int(self.nx))
        object.__setattr__(self, "ny", int(self.ny))
        object.__setattr__(self, "nz", int(self.nz))
        object.__setattr__(self, "xmin", float(self.xmin))
        object.__setattr__(self, "xmax", float(self.xmax))
        object.__setattr__(self, "ymin", float(self.ymin))
        object.__setattr__(self, "ymax", float(self.ymax))
        object.__setattr__(self, "zmin", float(self.zmin))
        object.__setattr__(self, "zmax", float(self.zmax))
        object.__setattr__(self, "hx", float(self.hx))
        object.__setattr__(self, "hy", float(self.hy))
        object.__setattr__(self, "hz", float(self.hz))
        object.__setattr__(
            self,
            "xmesh",
            _readonly_array_copy(self.xmesh, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "ymesh",
            _readonly_array_copy(self.ymesh, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "zmesh",
            _readonly_array_copy(self.zmesh, dtype=np.float64),
        )
        object.__setattr__(self, "value_size", int(self.value_size))
        object.__setattr__(self, "out_of_bounds_ok", bool(self.out_of_bounds_ok))
        object.__setattr__(
            self,
            "cell_to_row",
            _readonly_array_copy(self.cell_to_row, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "cell_table",
            _readonly_array_copy(self.cell_table, dtype=np.float64),
        )


@dataclass(frozen=True)
class RegularGridInterpolant3DDeviceSpec:
    """Device-resident bundle of spec arrays plus static metadata.

    Built once per :class:`RegularGridInterpolant3DSpec` and reused by
    hot JAX callsites. Staging the host arrays at construction time keeps
    repeated evaluation clean under :func:`jax.transfer_guard("disallow")`
    and avoids per-batch host-to-device traffic.
    """

    cell_table: jax.Array
    cell_to_row: jax.Array
    nodes: jax.Array
    scalings: jax.Array
    xmesh: jax.Array
    ymesh: jax.Array
    zmesh: jax.Array
    xmin: jax.Array
    xmax: jax.Array
    ymin: jax.Array
    ymax: jax.Array
    zmin: jax.Array
    zmax: jax.Array
    hx: jax.Array
    hy: jax.Array
    hz: jax.Array
    nx: jax.Array
    ny: jax.Array
    nz: jax.Array
    sentinel_row: jax.Array
    degree: int
    value_size: int
    out_of_bounds_ok: bool


def build_regular_grid_interpolant_3d_device_spec(
    spec: RegularGridInterpolant3DSpec,
) -> RegularGridInterpolant3DDeviceSpec:
    """Stage every regular-grid spec field to device arrays once."""

    cell_table = _stage_float64(spec.cell_table)
    return RegularGridInterpolant3DDeviceSpec(
        cell_table=cell_table,
        cell_to_row=_stage_cell_index(spec.cell_to_row),
        nodes=_stage_float64(spec.rule.nodes),
        scalings=_stage_float64(spec.rule.scalings),
        xmesh=_stage_float64(spec.xmesh),
        ymesh=_stage_float64(spec.ymesh),
        zmesh=_stage_float64(spec.zmesh),
        xmin=_stage_float64(spec.xmin),
        xmax=_stage_float64(spec.xmax),
        ymin=_stage_float64(spec.ymin),
        ymax=_stage_float64(spec.ymax),
        zmin=_stage_float64(spec.zmin),
        zmax=_stage_float64(spec.zmax),
        hx=_stage_float64(spec.hx),
        hy=_stage_float64(spec.hy),
        hz=_stage_float64(spec.hz),
        nx=_stage_cell_index(spec.nx),
        ny=_stage_cell_index(spec.ny),
        nz=_stage_cell_index(spec.nz),
        sentinel_row=_stage_cell_index(cell_table.shape[0] - 1),
        degree=int(spec.rule.degree),
        value_size=int(spec.value_size),
        out_of_bounds_ok=bool(spec.out_of_bounds_ok),
    )


def _stage_float64(value: object) -> jax.Array:
    return as_jax_array(value, dtype=jnp.float64)


def _stage_cell_index(value: object) -> jax.Array:
    return as_jax_array(value, dtype=_CELL_INDEX_DTYPE)


def _readonly_array_copy(array: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _validate_range(label: str, axis_range: tuple) -> tuple[float, float, int]:
    if len(axis_range) != 3:
        raise ValueError(
            f"{label} range must be a 3-tuple (min, max, n_cells); got {axis_range!r}"
        )
    minimum = float(axis_range[0])
    maximum = float(axis_range[1])
    n_cells = int(axis_range[2])
    if not maximum > minimum:
        raise ValueError(f"{label} range max ({maximum}) must exceed min ({minimum})")
    if n_cells < 1:
        raise ValueError(f"{label} range cell count must be >= 1, got {n_cells}")
    return minimum, maximum, n_cells


def _default_skip(xs: np.ndarray, *_unused: np.ndarray) -> np.ndarray:
    return np.zeros(xs.shape, dtype=bool)


def _build_cell_table(
    *,
    rule: InterpolationRule,
    nx: int,
    ny: int,
    nz: int,
    xmesh: np.ndarray,
    ymesh: np.ndarray,
    zmesh: np.ndarray,
    hx: float,
    hy: float,
    hz: float,
    value_size: int,
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    skip_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``f`` on the retained DOFs and pack them per cell.

    Returns ``(cell_to_row, cell_table)``:

    - ``cell_to_row`` has shape ``(nx*ny*nz,)`` and maps flat 3D cell
      indices to a row in ``cell_table``. Skipped cells map to the
      sentinel row (``cells_to_keep``).
    - ``cell_table`` has shape ``(cells_to_keep + 1, degree+1, degree+1,
      degree+1, value_size)``; the final row is forced to zero.
    """
    degree = rule.degree
    nodes = np.asarray(rule.nodes, dtype=np.float64)

    # Per-axis DOF coordinates (size nx*degree+1 along x and analogous).
    xdof = np.zeros(nx * degree + 1, dtype=np.float64)
    ydof = np.zeros(ny * degree + 1, dtype=np.float64)
    zdof = np.zeros(nz * degree + 1, dtype=np.float64)
    for i in range(nx):
        for j in range(degree + 1):
            xdof[i * degree + j] = xmesh[i] + nodes[j] * hx
    for i in range(ny):
        for j in range(degree + 1):
            ydof[i * degree + j] = ymesh[i] + nodes[j] * hy
    for i in range(nz):
        for j in range(degree + 1):
            zdof[i * degree + j] = zmesh[i] + nodes[j] * hz

    # Build the tensor-product DOF grid and the kept-DOF mask.
    nx_dof = nx * degree + 1
    ny_dof = ny * degree + 1
    nz_dof = nz * degree + 1
    total_dofs = nx_dof * ny_dof * nz_dof

    keep_dof = np.zeros((nx_dof, ny_dof, nz_dof), dtype=bool)
    cell_keep_mask = np.logical_not(skip_cell).reshape(nx, ny, nz)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not cell_keep_mask[i, j, k]:
                    continue
                keep_dof[
                    i * degree : i * degree + degree + 1,
                    j * degree : j * degree + degree + 1,
                    k * degree : k * degree + degree + 1,
                ] = True

    keep_flat = keep_dof.reshape(total_dofs)
    keep_indices = np.flatnonzero(keep_flat)
    dofs_to_keep = int(keep_indices.shape[0])

    # Build the flat coordinate vectors of just the retained DOFs, then
    # evaluate ``f`` once on the full retained set (batched).
    if dofs_to_keep == 0:
        # No retained DOFs: every cell is skipped. Build the zero
        # sentinel cell only.
        cells_to_keep = int(np.logical_not(skip_cell).sum())
        assert cells_to_keep == 0, (
            "skip mask declared cells to keep but every DOF is skipped; "
            "this indicates an inconsistent skip function."
        )
        cell_table = np.zeros(
            (1, degree + 1, degree + 1, degree + 1, value_size),
            dtype=np.float64,
        )
        cell_to_row = np.full((nx * ny * nz,), 0, dtype=np.int64)
        return cell_to_row, cell_table

    xdoftensor_full = np.broadcast_to(
        xdof[:, None, None], (nx_dof, ny_dof, nz_dof)
    ).reshape(total_dofs)
    ydoftensor_full = np.broadcast_to(
        ydof[None, :, None], (nx_dof, ny_dof, nz_dof)
    ).reshape(total_dofs)
    zdoftensor_full = np.broadcast_to(
        zdof[None, None, :], (nx_dof, ny_dof, nz_dof)
    ).reshape(total_dofs)

    xs_kept = xdoftensor_full[keep_indices]
    ys_kept = ydoftensor_full[keep_indices]
    zs_kept = zdoftensor_full[keep_indices]
    # The user function in upstream returns a flattened (N*value_size,)
    # row-major buffer where rows iterate over points and columns
    # iterate over output components.
    fvals_flat = np.asarray(f(xs_kept, ys_kept, zs_kept), dtype=np.float64)
    if fvals_flat.size != dofs_to_keep * value_size:
        raise ValueError(
            "interpolated function returned "
            f"{fvals_flat.size} entries; expected "
            f"{dofs_to_keep * value_size} "
            f"(dofs_to_keep={dofs_to_keep}, value_size={value_size})"
        )
    fvals = fvals_flat.reshape(dofs_to_keep, value_size)

    # Scatter into the full grid for cheap per-cell slicing below. Slots
    # for skipped DOFs stay at zero (they will never be read because
    # cell_to_row routes skipped cells away).
    full_vals = np.zeros((nx_dof, ny_dof, nz_dof, value_size), dtype=np.float64)
    full_flat = full_vals.reshape(total_dofs, value_size)
    full_flat[keep_indices] = fvals

    # Build the per-cell table by row-major iteration over kept cells.
    cells_to_keep = int(np.logical_not(skip_cell).sum())
    cell_table = np.zeros(
        (cells_to_keep + 1, degree + 1, degree + 1, degree + 1, value_size),
        dtype=np.float64,
    )

    cell_to_row = np.full((nx * ny * nz,), cells_to_keep, dtype=np.int64)
    row_counter = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                flat_cell_index = (i * ny + j) * nz + k
                if not cell_keep_mask[i, j, k]:
                    continue
                cell_to_row[flat_cell_index] = row_counter
                cell_table[row_counter] = full_vals[
                    i * degree : i * degree + degree + 1,
                    j * degree : j * degree + degree + 1,
                    k * degree : k * degree + degree + 1,
                ]
                row_counter += 1
    assert row_counter == cells_to_keep

    return cell_to_row, cell_table


def build_regular_grid_interpolant_3d(
    *,
    rule: InterpolationRule,
    xrange: tuple[float, float, int],
    yrange: tuple[float, float, int],
    zrange: tuple[float, float, int],
    value_size: int,
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    out_of_bounds_ok: bool = False,
    skip: (Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None) = None,
) -> RegularGridInterpolant3DSpec:
    """Build an immutable interpolant spec from a user-supplied function.

    Args:
        rule: 1D Lagrange rule. Use :func:`UniformInterpolationRule` or
            :func:`ChebyshevInterpolationRule`.
        xrange / yrange / zrange: ``(min, max, n_cells)`` tuples.
        value_size: output dimension of ``f``.
        f: function ``(xs, ys, zs) -> flat array of shape ``(N*value_size,)``
           matching the C++ ``interpolate_batch`` callback contract.
        out_of_bounds_ok: ``True`` preserves the caller/output-buffer value
            for skipped or out-of-domain rows; the default zero-initialised
            :func:`evaluate_batch` output therefore returns zero. ``False``
            surfaces ``NaN`` so the caller can detect the error.
        skip: optional predicate. Cells whose 8 mesh corners all evaluate
            to ``True`` are skipped (matches upstream PR #227 contract).
    """
    xmin, xmax, nx = _validate_range("x", xrange)
    ymin, ymax, ny = _validate_range("y", yrange)
    zmin, zmax, nz = _validate_range("z", zrange)
    value_size_int = int(value_size)
    if value_size_int < 1:
        raise ValueError(f"value_size must be >= 1, got {value_size_int}")

    hx = (xmax - xmin) / nx
    hy = (ymax - ymin) / ny
    hz = (zmax - zmin) / nz

    xmesh = np.linspace(xmin, xmax, nx + 1, dtype=np.float64)
    ymesh = np.linspace(ymin, ymax, ny + 1, dtype=np.float64)
    zmesh = np.linspace(zmin, zmax, nz + 1, dtype=np.float64)

    nmesh_x = nx + 1
    nmesh_y = ny + 1
    nmesh_z = nz + 1
    skip_fn = skip if skip is not None else _default_skip

    xmesh_grid = np.broadcast_to(
        xmesh[:, None, None], (nmesh_x, nmesh_y, nmesh_z)
    ).reshape(-1)
    ymesh_grid = np.broadcast_to(
        ymesh[None, :, None], (nmesh_x, nmesh_y, nmesh_z)
    ).reshape(-1)
    zmesh_grid = np.broadcast_to(
        zmesh[None, None, :], (nmesh_x, nmesh_y, nmesh_z)
    ).reshape(-1)

    skip_mesh_flat = np.asarray(skip_fn(xmesh_grid, ymesh_grid, zmesh_grid), dtype=bool)
    if skip_mesh_flat.shape != (nmesh_x * nmesh_y * nmesh_z,):
        raise ValueError(
            "skip(x, y, z) must return one bool per input point; got "
            f"shape {skip_mesh_flat.shape} for {nmesh_x * nmesh_y * nmesh_z} inputs"
        )
    skip_mesh = skip_mesh_flat.reshape(nmesh_x, nmesh_y, nmesh_z)

    skip_cell_grid = (
        skip_mesh[:-1, :-1, :-1]
        & skip_mesh[:-1, :-1, 1:]
        & skip_mesh[:-1, 1:, :-1]
        & skip_mesh[:-1, 1:, 1:]
        & skip_mesh[1:, :-1, :-1]
        & skip_mesh[1:, :-1, 1:]
        & skip_mesh[1:, 1:, :-1]
        & skip_mesh[1:, 1:, 1:]
    )
    skip_cell = skip_cell_grid.reshape(-1)

    cell_to_row, cell_table = _build_cell_table(
        rule=rule,
        nx=nx,
        ny=ny,
        nz=nz,
        xmesh=xmesh,
        ymesh=ymesh,
        zmesh=zmesh,
        hx=hx,
        hy=hy,
        hz=hz,
        value_size=value_size_int,
        f=f,
        skip_cell=skip_cell,
    )

    return RegularGridInterpolant3DSpec(
        rule=rule,
        nx=nx,
        ny=ny,
        nz=nz,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        hx=hx,
        hy=hy,
        hz=hz,
        xmesh=xmesh,
        ymesh=ymesh,
        zmesh=zmesh,
        value_size=value_size_int,
        out_of_bounds_ok=bool(out_of_bounds_ok),
        cell_to_row=cell_to_row,
        cell_table=cell_table,
    )


def _basis_values(
    *,
    local_coord: jax.Array,
    nodes: jax.Array,
    scalings: jax.Array,
    degree: int,
) -> jax.Array:
    """Evaluate all ``degree+1`` Lagrange basis polynomials at ``local_coord``.

    Mirrors ``InterpolationRule::basis_fun`` in the C++ header.
    Returns a vector ``p`` of length ``degree+1`` with
    ``p[idx] = scalings[idx] * Π_{i != idx} (x - nodes[i])``.
    """
    # ``local_coord`` is a scalar; ``diffs`` has shape (degree+1,).
    diffs = local_coord - nodes
    eye = jnp.eye(degree + 1, dtype=jnp.float64)
    ones_row = jnp.ones((degree + 1,), dtype=jnp.float64)
    # ``factors[i, idx]`` is ``diffs[i]`` when ``i != idx`` and ``1.0``
    # otherwise, so the column-wise product gives the (degree+1)-vector
    # of basis values before scaling.
    factors = diffs[:, None] * (ones_row[:, None] - eye) + eye
    products = jnp.prod(factors, axis=0)
    return products * scalings


def _cpu_ordered_tensor_contract(pkx, pky, pkz, local_vals):
    """Contract one cell in C++ loop order: value channel outside, k fastest."""

    degree_plus_one = int(pkx.shape[0])
    zero = jnp.zeros((local_vals.shape[-1],), dtype=local_vals.dtype)

    def i_body(i, i_acc):
        def j_body(j, j_acc):
            xy_weight = pkx[i] * pky[j]

            def k_body(k, k_acc):
                return k_acc + xy_weight * pkz[k] * local_vals[i, j, k]

            return jax.lax.fori_loop(0, degree_plus_one, k_body, j_acc)

        return jax.lax.fori_loop(0, degree_plus_one, j_body, i_acc)

    return jax.lax.fori_loop(0, degree_plus_one, i_body, zero)


def _fused_tensor_contract(pkx, pky, pkz, local_vals):
    """Contract one cell with a fused tensor-product expression."""

    return jnp.einsum("i,j,k,ijkv->v", pkx, pky, pkz, local_vals)


@partial(
    jax.jit,
    static_argnames=("degree", "value_size", "out_of_bounds_ok", "strict_cell_order"),
)
def _evaluate_batch_jit(
    xyz: jax.Array,
    *,
    initial_output: jax.Array,
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
    degree: int,
    value_size: int,
    out_of_bounds_ok: bool,
    strict_cell_order: bool = True,
) -> jax.Array:
    """JIT-compiled per-sample evaluation kernel."""

    def evaluate_one(point: jax.Array, initial_value: jax.Array) -> jax.Array:
        x = point[0]
        y = point[1]
        z = point[2]

        # Soft boundary clamp matching ``evaluate_inplace`` in the C++
        # implementation. This handles points that are within
        # floating-point noise of the domain boundary.
        x_clamped = jnp.where(x >= xmax, x - _BOUNDARY_EPSILON, x)
        x_clamped = jnp.where(
            x_clamped <= xmin, x_clamped + _BOUNDARY_EPSILON, x_clamped
        )
        y_clamped = jnp.where(y >= ymax, y - _BOUNDARY_EPSILON, y)
        y_clamped = jnp.where(
            y_clamped <= ymin, y_clamped + _BOUNDARY_EPSILON, y_clamped
        )
        z_clamped = jnp.where(z >= zmax, z - _BOUNDARY_EPSILON, z)
        z_clamped = jnp.where(
            z_clamped <= zmin, z_clamped + _BOUNDARY_EPSILON, z_clamped
        )

        xscaled = nx * (x_clamped - xmin) / (xmax - xmin)
        yscaled = ny * (y_clamped - ymin) / (ymax - ymin)
        zscaled = nz * (z_clamped - zmin) / (zmax - zmin)
        x_indexable = (
            jnp.isfinite(xscaled)
            & (xscaled >= _INT64_MIN_FLOAT)
            & (xscaled < _INT64_MAX_FLOAT)
        )
        y_indexable = (
            jnp.isfinite(yscaled)
            & (yscaled >= _INT64_MIN_FLOAT)
            & (yscaled < _INT64_MAX_FLOAT)
        )
        z_indexable = (
            jnp.isfinite(zscaled)
            & (zscaled >= _INT64_MIN_FLOAT)
            & (zscaled < _INT64_MAX_FLOAT)
        )

        # Cell index as in ``locate_unsafe``. Truncation is intentionally
        # allowed before the bounds check so slightly negative coordinates
        # in ``(-1, 0)`` still map to cell 0, matching C++ ``int64_t(...)``.
        xidx_raw = jnp.trunc(jnp.where(x_indexable, xscaled, -1.0)).astype(
            _CELL_INDEX_DTYPE
        )
        yidx_raw = jnp.trunc(jnp.where(y_indexable, yscaled, -1.0)).astype(
            _CELL_INDEX_DTYPE
        )
        zidx_raw = jnp.trunc(jnp.where(z_indexable, zscaled, -1.0)).astype(
            _CELL_INDEX_DTYPE
        )

        in_bounds_x = x_indexable & (xidx_raw >= 0) & (xidx_raw < nx)
        in_bounds_y = y_indexable & (yidx_raw >= 0) & (yidx_raw < ny)
        in_bounds_z = z_indexable & (zidx_raw >= 0) & (zidx_raw < nz)
        in_bounds = in_bounds_x & in_bounds_y & in_bounds_z

        # Clamp cell indices to a valid range so the gather is safe;
        # the ``in_bounds`` flag controls whether the result is used.
        xidx = jnp.clip(xidx_raw, 0, nx - 1)
        yidx = jnp.clip(yidx_raw, 0, ny - 1)
        zidx = jnp.clip(zidx_raw, 0, nz - 1)

        flat_cell_idx = _flat_cell_index(xidx, yidx, zidx, ny=ny, nz=nz)
        candidate_row = cell_to_row[flat_cell_idx]
        is_kept_cell = candidate_row != sentinel_row
        row_idx = jnp.where(in_bounds, candidate_row, sentinel_row)
        # ``in_kept_cell`` distinguishes "in spatial bounds AND not a
        # skipped cell" from "in spatial bounds but skipped". The C++
        # binding raises in both not-in-bounds and skipped-cell cases
        # when ``out_of_bounds_ok=False``; JAX routes both through NaN.
        in_kept_cell = in_bounds & is_kept_cell

        xlocal = (x_clamped - xmesh[xidx]) / hx
        ylocal = (y_clamped - ymesh[yidx]) / hy
        zlocal = (z_clamped - zmesh[zidx]) / hz

        pkx = _basis_values(
            local_coord=xlocal, nodes=nodes, scalings=scalings, degree=degree
        )
        pky = _basis_values(
            local_coord=ylocal, nodes=nodes, scalings=scalings, degree=degree
        )
        pkz = _basis_values(
            local_coord=zlocal, nodes=nodes, scalings=scalings, degree=degree
        )

        local_vals = cell_table[row_idx]  # (degree+1, degree+1, degree+1, value_size)
        if strict_cell_order:
            result = _cpu_ordered_tensor_contract(pkx, pky, pkz, local_vals)
        else:
            result = _fused_tensor_contract(pkx, pky, pkz, local_vals)
        if out_of_bounds_ok:
            # Match the C++ mutable-output contract: skipped or out-of-domain
            # rows leave the caller-supplied output row unchanged.
            result = jnp.where(in_kept_cell, result, initial_value)
        else:
            # ``out_of_bounds_ok=False`` surfaces NaN to the host so the
            # caller can detect the error post-hoc. The C++ binding raises
            # in both the not-in-bounds and skipped-cell cases; raising
            # from inside a jitted kernel would abandon JIT entirely, so
            # we route both through NaN instead.
            result = jnp.where(in_kept_cell, result, jnp.nan)
        return result

    return jax.vmap(evaluate_one)(xyz, initial_output)


@partial(
    jax.jit,
    static_argnames=("degree", "value_size", "out_of_bounds_ok", "strict_cell_order"),
)
def _evaluate_batch_zero_jit(
    xyz: jax.Array,
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
    degree: int,
    value_size: int,
    out_of_bounds_ok: bool,
    strict_cell_order: bool = True,
) -> jax.Array:
    return _evaluate_batch_jit(
        xyz,
        initial_output=jnp.zeros((xyz.shape[0], value_size), dtype=jnp.float64),
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
        strict_cell_order=strict_cell_order,
    )


def _as_xyz_array(xyz: object) -> jax.Array:
    xyz_array = _stage_float64(xyz)
    if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3); got {xyz_array.shape}")
    return xyz_array


def _as_initial_output_array(
    initial_output: object,
    *,
    npoints: int,
    value_size: int,
) -> jax.Array:
    initial_array = _stage_float64(initial_output)
    expected_shape = (npoints, value_size)
    if initial_array.shape != expected_shape:
        raise ValueError(
            f"initial_output must have shape {expected_shape}; got {initial_array.shape}"
        )
    return initial_array


def evaluate_batch_device(
    device_spec: RegularGridInterpolant3DDeviceSpec,
    xyz: object,
    *,
    initial_output: object | None = None,
    strict_cell_order: bool = True,
) -> jax.Array:
    """Evaluate with a pre-staged device spec.

    ``initial_output`` is the pure-functional analogue of the C++
    caller-provided ``fxyz`` buffer. When omitted, it is a zero buffer,
    matching the common Python binding/oracle use.
    """

    xyz_array = _as_xyz_array(xyz)
    if initial_output is None:
        return _evaluate_batch_zero_jit(
            xyz_array,
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
            degree=int(device_spec.degree),
            value_size=int(device_spec.value_size),
            out_of_bounds_ok=bool(device_spec.out_of_bounds_ok),
            strict_cell_order=bool(strict_cell_order),
        )
    initial_array = _as_initial_output_array(
        initial_output,
        npoints=xyz_array.shape[0],
        value_size=int(device_spec.value_size),
    )
    return _evaluate_batch_jit(
        xyz_array,
        initial_output=initial_array,
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
        degree=int(device_spec.degree),
        value_size=int(device_spec.value_size),
        out_of_bounds_ok=bool(device_spec.out_of_bounds_ok),
        strict_cell_order=bool(strict_cell_order),
    )


def evaluate_batch(spec: RegularGridInterpolant3DSpec, xyz: object) -> jax.Array:
    """Evaluate the interpolant at every row of ``xyz``.

    ``jax.grad`` / ``jax.jacrev`` applied to this function differentiates
    the piecewise interpolation polynomial, not the physical ``GradAbsB``
    interpolant maintained by ``InterpolatedFieldJAX``.

    Args:
        spec: spec returned by :func:`build_regular_grid_interpolant_3d`.
        xyz: ``(N, 3)`` JAX or NumPy array of evaluation points.

    Returns:
        ``(N, value_size)`` JAX array of interpolated values. Skipped
        cells produce zero; out-of-domain points produce zero or ``NaN``
        depending on ``spec.out_of_bounds_ok``.
    """

    return evaluate_batch_device(
        build_regular_grid_interpolant_3d_device_spec(spec),
        xyz,
    )


def evaluate_batch_with_initial(
    spec: RegularGridInterpolant3DSpec,
    xyz: object,
    initial_output: object,
) -> jax.Array:
    """Evaluate while preserving C++ caller-buffer semantics for skipped rows."""

    return evaluate_batch_device(
        build_regular_grid_interpolant_3d_device_spec(spec),
        xyz,
        initial_output=initial_output,
    )


def estimate_error(
    spec: RegularGridInterpolant3DSpec,
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    samples: int,
    *,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimate ``(mean - std, mean + std)`` interpolation error.

    Mirrors ``RegularGridInterpolant3D::estimate_error`` in
    ``regular_grid_interpolant_3d_impl.h``. Samples are drawn uniformly
    from the cuboid domain.
    """
    rng = np.random.default_rng(int(seed))
    nsamples = int(samples)
    if nsamples < 2:
        raise ValueError(f"samples must be >= 2 to estimate variance, got {nsamples}")
    xs = spec.xmin + rng.uniform(0.0, 1.0, size=nsamples) * (spec.xmax - spec.xmin)
    ys = spec.ymin + rng.uniform(0.0, 1.0, size=nsamples) * (spec.ymax - spec.ymin)
    zs = spec.zmin + rng.uniform(0.0, 1.0, size=nsamples) * (spec.zmax - spec.zmin)

    xyz = np.stack([xs, ys, zs], axis=-1)
    fhxyz = np.asarray(evaluate_batch(spec, xyz))
    fxyz_flat = np.asarray(f(xs, ys, zs), dtype=np.float64)
    fxyz = fxyz_flat.reshape(nsamples, spec.value_size)
    diffs = np.linalg.norm(fxyz - fhxyz, axis=1)
    err = float(diffs.sum())
    errsq = float((diffs * diffs).sum())
    mean = err / nsamples
    var = (errsq - err * err / nsamples) / (nsamples - 1) / nsamples
    std = float(np.sqrt(max(var, 0.0)))
    return mean - std, mean + std


__all__ = [
    "ChebyshevInterpolationRule",
    "InterpolationRule",
    "RegularGridInterpolant3DDeviceSpec",
    "RegularGridInterpolant3DSpec",
    "UniformInterpolationRule",
    "build_regular_grid_interpolant_3d",
    "build_regular_grid_interpolant_3d_device_spec",
    "estimate_error",
    "evaluate_batch",
    "evaluate_batch_device",
    "evaluate_batch_with_initial",
]
