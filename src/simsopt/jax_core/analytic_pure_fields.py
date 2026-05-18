"""JAX port of analytic-pure fields from ``magneticfieldclasses.py``.

Tier P1 item 12 (partial). Covers ``ToroidalField``, ``PoloidalField``, and
``MirrorModel`` as immutable specs plus pure JAX kernels. ``CircularCoil`` is
explicitly deferred as item 12-sub because ``jax.scipy.special.ellipk`` /
``ellipe`` are not exposed by ``jaxlib`` 0.10.0; see
``.artifacts/jax_port_goal/blockers/12-circularcoil-debug.md``.

Conventions
-----------

* Inputs are cartesian ``points`` of shape ``(N, 3)`` in metres.
* ``B`` and ``A`` outputs use the same simsopt cartesian layout as the upstream
  CPU classes: ``B[p, l] = B_l(x_p)``, ``A[p, l] = A_l(x_p)``.
* First-derivative layouts mirror the upstream CPU classes **as they actually
  store the array**, not the abstract simsopt convention documented in
  ``CLAUDE.md``. The upstream Python analytic classes assemble derivatives
  via ``np.array([dB_by_dX1, ...]).T`` for ``ToroidalField`` /
  ``PoloidalField`` (axis 1 = ``B`` component, axis 2 = derivative
  direction), and via direct ``dB[:, j, l] = ...`` for ``MirrorModel``
  (axis 1 = derivative direction, axis 2 = component). These two layouts
  differ, but each JAX kernel exactly reproduces the corresponding CPU
  class output so the ``direct_kernel`` parity contract is exact.
* Second derivatives: ``d2B[p, j, k, l]`` matches the CPU
  ``_d2B_by_dXdX_impl`` literal output (see invariants doc for the known
  upstream typo in the ``ToroidalField`` second derivative).

The kernels intentionally mirror the upstream cartesian closed-form
expressions character-for-character rather than re-deriving them by autodiff
because:

* The CPU ``_d2B_by_dXdX_impl`` / ``_d2A_by_dXdX_impl`` expressions are the
  oracle for the parity gate. Reproducing them literally keeps the
  ``direct_kernel`` parity contract tight even when the upstream form differs
  from the analytic derivative of ``_dB_by_dX_impl``.
* The CPU ``_dB_by_dX_impl`` of ``PoloidalField`` stores entries in a layout
  that is the upstream simsopt-public contract (see the analytic reference
  block in ``tests/field/test_magneticfields.py::test_poloidal_field``).

Singular regimes:

* ``PoloidalField`` is singular at ``sqrt(x**2 + y**2) == R0`` (the magnetic
  axis). The CPU class returns NaN there; the JAX kernel matches.
* ``MirrorModel`` is singular at ``sqrt(x**2 + y**2) == 0``. The CPU class
  returns NaN/inf there; the JAX kernel matches.
* No defensive guards beyond what upstream provides.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._math_utils import as_jax_float64 as _as_jax_float64


__all__ = [
    "ToroidalFieldSpec",
    "PoloidalFieldSpec",
    "MirrorModelSpec",
    "toroidal_B",
    "toroidal_dB",
    "toroidal_d2B",
    "toroidal_A",
    "toroidal_dA",
    "poloidal_B",
    "poloidal_dB",
    "mirror_B",
    "mirror_dB",
]


@dataclass(frozen=True)
class ToroidalFieldSpec:
    """Immutable payload for a pure JAX ``ToroidalField`` evaluation.

    Parameters
    ----------
    R0
        Major radius normalising scale, ``B = B0 * R0 / R * e_phi``.
    B0
        Field modulus at ``R = R0``.
    """

    R0: float
    B0: float


jax.tree_util.register_dataclass(
    ToroidalFieldSpec,
    data_fields=[],
    meta_fields=["R0", "B0"],
)


@dataclass(frozen=True)
class PoloidalFieldSpec:
    """Immutable payload for a pure JAX ``PoloidalField`` evaluation.

    Parameters
    ----------
    R0
        Major radius of the magnetic axis.
    B0
        Field modulus at ``R = R0``.
    q
        Safety factor (pitch).
    """

    R0: float
    B0: float
    q: float


jax.tree_util.register_dataclass(
    PoloidalFieldSpec,
    data_fields=[],
    meta_fields=["R0", "B0", "q"],
)


@dataclass(frozen=True)
class MirrorModelSpec:
    """Immutable payload for a pure JAX ``MirrorModel`` (WHAM) evaluation.

    Parameters
    ----------
    B0
        Flux-function amplitude ``B`` (Tesla).
    gamma
        Lorentzian width (metres).
    Z_m
        Lorentzian centre offset on the Z-axis (metres).
    """

    B0: float
    gamma: float
    Z_m: float


jax.tree_util.register_dataclass(
    MirrorModelSpec,
    data_fields=[],
    meta_fields=["B0", "gamma", "Z_m"],
)


def _validate_points(points: jax.Array) -> jax.Array:
    points_arr = _as_jax_float64(points)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3); got {tuple(points_arr.shape)!r}."
        )
    return points_arr


# ── ToroidalField kernels ────────────────────────────────────────────


def _toroidal_B_pointwise(point: jax.Array, R0: jax.Array, B0: jax.Array) -> jax.Array:
    """B at one cartesian point. Mirrors ``ToroidalField._B_impl``."""
    x = point[0]
    y = point[1]
    z = point[2]
    R2 = x * x + y * y
    coeff = B0 * R0 / R2
    bx = -coeff * y
    by = coeff * x
    bz = jnp.zeros_like(z)
    return jnp.stack((bx, by, bz))


def _toroidal_dB_pointwise(point: jax.Array, R0: jax.Array, B0: jax.Array) -> jax.Array:
    """``dB[l, j]`` at one point. Mirrors ``_dB_by_dX_impl`` storage.

    The CPU class assembles ``dB`` via ``np.array([dB_by_dX1, dB_by_dX2,
    dB_by_dX3]).T``. The pre-transpose array has axis order
    ``(deriv, component, point)``; ``.T`` reverses all axes to
    ``(point, component, deriv)``. The returned per-point tensor therefore
    has axis 0 = ``B`` component, axis 1 = derivative direction.
    """
    x = point[0]
    y = point[1]
    R2 = x * x + y * y
    R4 = R2 * R2
    factor = B0 * R0 / R4
    zero = jnp.zeros_like(x)
    # Each slab is dB_by_dX<j> in CPU code: a length-3 column over components.
    deriv_x = jnp.stack((factor * 2.0 * x * y, factor * (y * y - x * x), zero))
    deriv_y = jnp.stack((factor * (y * y - x * x), -factor * 2.0 * x * y, zero))
    deriv_z = jnp.stack((zero, zero, zero))
    # Reverse axes (component, deriv) to match CPU `.T` layout.
    return jnp.stack((deriv_x, deriv_y, deriv_z), axis=0).T


def _toroidal_d2B_pointwise(
    point: jax.Array, R0: jax.Array, B0: jax.Array
) -> jax.Array:
    """``d2B[j, k, l] = ∂_j ∂_k B_l`` matching ``_d2B_by_dXdX_impl``.

    Replicates the upstream CPU arithmetic literally so the
    ``direct_kernel`` parity gate matches the CPU oracle to machine
    precision. The CPU expression differs from the analytic third-derivative
    of ``_B_impl`` (see ``.artifacts/jax_port_goal/plans/12-invariants.md``);
    this kernel preserves the upstream behaviour by construction.
    """
    x = point[0]
    y = point[1]
    R2 = x * x + y * y
    coeff = 2.0 * B0 * R0 / (R2 * R2 * R2)
    x2 = x * x
    y2 = y * y
    y3 = y2 * y
    x3 = x2 * x
    zero = jnp.zeros_like(x)

    # ddB[j, k, l] using upstream layout. The CPU code assembles a (3,3,3,N)
    # array and transposes to (N,3,3,3). The slab axis order in the source is
    # [j, k, l], which is what we build here.
    j0 = jnp.stack(
        (
            jnp.stack((3.0 * x2 + y3, x3 - 3.0 * x * y2, zero)),
            jnp.stack((x3 - 3.0 * x * y2, 3.0 * x2 * y - y3, zero)),
            jnp.stack((zero, zero, zero)),
        ),
        axis=0,
    )
    j1 = jnp.stack(
        (
            jnp.stack((x3 - 3.0 * x * y2, 3.0 * x2 * y - y3, zero)),
            jnp.stack((3.0 * x2 * y - y3, -x3 + 3.0 * x * y2, zero)),
            jnp.stack((zero, zero, zero)),
        ),
        axis=0,
    )
    j2 = jnp.stack(
        (
            jnp.stack((zero, zero, zero)),
            jnp.stack((zero, zero, zero)),
            jnp.stack((zero, zero, zero)),
        ),
        axis=0,
    )
    # Stacked along axis 0 the order is (j_outer, k_middle, comp). CPU stores
    # ``np.array([...]).T`` which reverses all axes, ending up as
    # ``ddB[p, comp, k, j]``. Reverse axes to match.
    stacked = jnp.stack((j0, j1, j2), axis=0)
    return coeff * jnp.transpose(stacked, (2, 1, 0))


def _toroidal_A_pointwise(point: jax.Array, R0: jax.Array, B0: jax.Array) -> jax.Array:
    """A at one cartesian point. Mirrors ``ToroidalField._A_impl``."""
    x = point[0]
    y = point[1]
    z = point[2]
    R2 = x * x + y * y
    scale = B0 * R0
    ax = scale * z * x / R2
    ay = scale * z * y / R2
    az = jnp.zeros_like(z)
    return jnp.stack((ax, ay, az))


def _toroidal_dA_pointwise(point: jax.Array, R0: jax.Array, B0: jax.Array) -> jax.Array:
    """``dA[l, j]`` per-point, matching ``_dA_by_dX_impl`` storage.

    Same axis convention as ``_toroidal_dB_pointwise``: axis 0 = ``A``
    component, axis 1 = derivative direction (CPU ``.T`` reverses an
    inner ``(j, comp, point)`` layout to ``(point, comp, j)``).
    """
    x = point[0]
    y = point[1]
    z = point[2]
    R2 = x * x + y * y
    coeff = B0 * R0 * z / (R2 * R2)
    zero = jnp.zeros_like(x)
    # CPU slab order: slab 0 = ∂_x A, slab 1 = ∂_y A, slab 2 = ∂_z A.
    deriv_x = jnp.stack((coeff * (-x * x + y * y), -2.0 * coeff * x * y, zero))
    deriv_y = jnp.stack((-2.0 * coeff * x * y, coeff * (x * x - y * y), zero))
    # ∂_z A_x = B0 R0 x / R2; ∂_z A_y = B0 R0 y / R2. CPU writes this as
    # ``coeff * x * (R2 / z) = (B0 R0 z / R4) * x * R2 / z = B0 R0 x / R2``.
    coeff_z = B0 * R0 / R2
    deriv_z = jnp.stack((coeff_z * x, coeff_z * y, zero))
    return jnp.stack((deriv_x, deriv_y, deriv_z), axis=0).T


_toroidal_B_vmap = jax.vmap(_toroidal_B_pointwise, in_axes=(0, None, None))
_toroidal_dB_vmap = jax.vmap(_toroidal_dB_pointwise, in_axes=(0, None, None))
_toroidal_d2B_vmap = jax.vmap(_toroidal_d2B_pointwise, in_axes=(0, None, None))
_toroidal_A_vmap = jax.vmap(_toroidal_A_pointwise, in_axes=(0, None, None))
_toroidal_dA_vmap = jax.vmap(_toroidal_dA_pointwise, in_axes=(0, None, None))


@jax.jit
def _toroidal_B_jit(R0: jax.Array, B0: jax.Array, points: jax.Array) -> jax.Array:
    return _toroidal_B_vmap(points, R0, B0)


@jax.jit
def _toroidal_dB_jit(R0: jax.Array, B0: jax.Array, points: jax.Array) -> jax.Array:
    return _toroidal_dB_vmap(points, R0, B0)


@jax.jit
def _toroidal_d2B_jit(R0: jax.Array, B0: jax.Array, points: jax.Array) -> jax.Array:
    return _toroidal_d2B_vmap(points, R0, B0)


@jax.jit
def _toroidal_A_jit(R0: jax.Array, B0: jax.Array, points: jax.Array) -> jax.Array:
    return _toroidal_A_vmap(points, R0, B0)


@jax.jit
def _toroidal_dA_jit(R0: jax.Array, B0: jax.Array, points: jax.Array) -> jax.Array:
    return _toroidal_dA_vmap(points, R0, B0)


def _toroidal_scalars(spec: ToroidalFieldSpec) -> tuple[jax.Array, jax.Array]:
    return (_as_jax_float64(spec.R0), _as_jax_float64(spec.B0))


def toroidal_B(spec: ToroidalFieldSpec, points: jax.Array) -> jax.Array:
    """``B(x)`` for a toroidal field at ``N`` cartesian points."""
    R0, B0 = _toroidal_scalars(spec)
    return _toroidal_B_jit(R0, B0, _validate_points(points))


def toroidal_dB(spec: ToroidalFieldSpec, points: jax.Array) -> jax.Array:
    """First spatial gradient of the toroidal field.

    The CPU oracle's ``np.array([dBdx, dBdy, dBdz]).T`` storage in
    ``ToroidalField._dB_by_dX_impl`` is preserved bit-for-bit so that
    ``direct_kernel`` same-state parity holds against the C++ reference.
    See the module-level docstring.

    Returns
    -------
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, l, j] = ∂_j B_l(x_p)`` (component-first; matches the
        simsoptpp C++ storage order). Axis 1 is the B-field component;
        axis 2 is the spatial derivative direction.
    """
    R0, B0 = _toroidal_scalars(spec)
    return _toroidal_dB_jit(R0, B0, _validate_points(points))


def toroidal_d2B(spec: ToroidalFieldSpec, points: jax.Array) -> jax.Array:
    """``d2B/dXdX[p, j, k, l]`` matching the CPU oracle layout."""
    R0, B0 = _toroidal_scalars(spec)
    return _toroidal_d2B_jit(R0, B0, _validate_points(points))


def toroidal_A(spec: ToroidalFieldSpec, points: jax.Array) -> jax.Array:
    """Vector potential ``A(x)`` for a toroidal field."""
    R0, B0 = _toroidal_scalars(spec)
    return _toroidal_A_jit(R0, B0, _validate_points(points))


def toroidal_dA(spec: ToroidalFieldSpec, points: jax.Array) -> jax.Array:
    """First spatial gradient of the toroidal vector potential.

    Returns
    -------
    dA : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dA[p, l, j] = ∂_j A_l(x_p)`` (component-first; matches the
        simsoptpp C++ storage order). Axis 1 is the A-field component;
        axis 2 is the spatial derivative direction. CPU-oracle layout;
        see ``toroidal_dB`` for the parity rationale.
    """
    R0, B0 = _toroidal_scalars(spec)
    return _toroidal_dA_jit(R0, B0, _validate_points(points))


# ── PoloidalField kernels ────────────────────────────────────────────


def _poloidal_B_pointwise(
    point: jax.Array, R0: jax.Array, B0: jax.Array, q: jax.Array
) -> jax.Array:
    """B at one point. Mirrors ``PoloidalField._B_impl``."""
    x = point[0]
    y = point[1]
    z = point[2]
    R_xy = jnp.sqrt(x * x + y * y)
    phi = jnp.atan2(y, x)
    theta = jnp.atan2(z, R_xy - R0)
    r = jnp.sqrt((R_xy - R0) ** 2 + z * z)
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    scale = B0 / (R0 * q)
    bx = -scale * sin_theta * r * cos_phi
    by = -scale * sin_theta * r * sin_phi
    bz = scale * cos_theta * r
    return jnp.stack((bx, by, bz))


def _poloidal_dB_pointwise(
    point: jax.Array, R0: jax.Array, B0: jax.Array, q: jax.Array
) -> jax.Array:
    """``dB[j, l] = (∂_j B_l)`` matching ``_dB_by_dX_impl`` layout."""
    x = point[0]
    y = point[1]
    z = point[2]
    R_xy = jnp.sqrt(x * x + y * y)
    phi = jnp.atan2(y, x)
    theta = jnp.atan2(z, R_xy - R0)
    r = jnp.sqrt((R_xy - R0) ** 2 + z * z)
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    R2 = x2 + y2
    R0_sq = R0 * R0
    denom_theta = R_xy * (R2 + z2 - 2.0 * R_xy * R0 + R0_sq)
    dtheta_dX1 = -(x * z) / denom_theta
    dtheta_dX2 = -(y * z) / denom_theta
    dtheta_dX3 = 1.0 / ((-R0 + R_xy) * (1.0 + z2 / (R0 - R_xy) ** 2))

    dphi_dX1 = -y / R2
    dphi_dX2 = x / R2
    dphi_dX3 = jnp.zeros_like(x)

    dthetauv_dX1 = jnp.stack(
        (
            -cos_theta * cos_phi * dtheta_dX1 + sin_theta * sin_phi * dphi_dX1,
            -cos_theta * sin_phi * dtheta_dX1 - sin_theta * cos_phi * dphi_dX1,
            -sin_theta * dtheta_dX1,
        )
    )
    dthetauv_dX2 = jnp.stack(
        (
            -cos_theta * cos_phi * dtheta_dX2 + sin_theta * sin_phi * dphi_dX2,
            -cos_theta * sin_phi * dtheta_dX2 - sin_theta * cos_phi * dphi_dX2,
            -sin_theta * dtheta_dX2,
        )
    )
    dthetauv_dX3 = jnp.stack(
        (
            -cos_theta * cos_phi * dtheta_dX3 + sin_theta * sin_phi * dphi_dX3,
            -cos_theta * sin_phi * dtheta_dX3 - sin_theta * cos_phi * dphi_dX3,
            -sin_theta * dtheta_dX3,
        )
    )

    term1_dX1 = dthetauv_dX1 * r
    term1_dX2 = dthetauv_dX2 * r
    term1_dX3 = dthetauv_dX3 * r

    theta_unit_vec = jnp.stack((-sin_theta * cos_phi, -sin_theta * sin_phi, cos_theta))

    R0_minus_R = R0 - R_xy
    denom_r = jnp.sqrt(R0_minus_R * R0_minus_R + z2)
    dr_dX1 = (x * (-R0 + R_xy)) / (R_xy * denom_r)
    dr_dX2 = (y * (-R0 + R_xy)) / (R_xy * denom_r)
    dr_dX3 = z / denom_r

    term2_dX1 = theta_unit_vec * dr_dX1
    term2_dX2 = theta_unit_vec * dr_dX2
    term2_dX3 = theta_unit_vec * dr_dX3

    scale = B0 / (R0 * q)
    # Each ``term*_dX<j>`` is length-3 across the ``B`` component axis;
    # stacking along axis 0 gives ``(j_deriv, l_comp)``. CPU stores
    # ``dB[p, l, j]`` via the ``np.array(...).T`` pattern, so transpose to
    # ``(l_comp, j_deriv)``.
    return (
        scale
        * jnp.stack(
            (term1_dX1 + term2_dX1, term1_dX2 + term2_dX2, term1_dX3 + term2_dX3),
            axis=0,
        ).T
    )


_poloidal_B_vmap = jax.vmap(_poloidal_B_pointwise, in_axes=(0, None, None, None))
_poloidal_dB_vmap = jax.vmap(_poloidal_dB_pointwise, in_axes=(0, None, None, None))


@jax.jit
def _poloidal_B_jit(
    R0: jax.Array, B0: jax.Array, q: jax.Array, points: jax.Array
) -> jax.Array:
    return _poloidal_B_vmap(points, R0, B0, q)


@jax.jit
def _poloidal_dB_jit(
    R0: jax.Array, B0: jax.Array, q: jax.Array, points: jax.Array
) -> jax.Array:
    return _poloidal_dB_vmap(points, R0, B0, q)


def _poloidal_scalars(
    spec: PoloidalFieldSpec,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        _as_jax_float64(spec.R0),
        _as_jax_float64(spec.B0),
        _as_jax_float64(spec.q),
    )


def poloidal_B(spec: PoloidalFieldSpec, points: jax.Array) -> jax.Array:
    """``B(x)`` for a poloidal field with safety factor ``q``."""
    R0, B0, q = _poloidal_scalars(spec)
    return _poloidal_B_jit(R0, B0, q, _validate_points(points))


def poloidal_dB(spec: PoloidalFieldSpec, points: jax.Array) -> jax.Array:
    """First spatial gradient of the poloidal field.

    Returns
    -------
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, l, j] = ∂_j B_l(x_p)`` (component-first; matches the
        simsoptpp C++ storage order). Axis 1 is the B-field component;
        axis 2 is the spatial derivative direction. CPU-oracle layout;
        see ``toroidal_dB`` for the parity rationale.
    """
    R0, B0, q = _poloidal_scalars(spec)
    return _poloidal_dB_jit(R0, B0, q, _validate_points(points))


# ── MirrorModel kernels ──────────────────────────────────────────────


def _mirror_psi(
    R: jax.Array,
    Z: jax.Array,
    spec_B0: jax.Array,
    spec_gamma: jax.Array,
    spec_Zm: jax.Array,
) -> jax.Array:
    factor1 = 1.0 + ((Z - spec_Zm) / spec_gamma) ** 2
    factor2 = 1.0 + ((Z + spec_Zm) / spec_gamma) ** 2
    return (R * R * spec_B0 / (2.0 * jnp.pi * spec_gamma)) * (
        1.0 / factor1 + 1.0 / factor2
    )


def _mirror_B_pointwise(
    point: jax.Array, B0_p: jax.Array, gamma_p: jax.Array, Zm_p: jax.Array
) -> jax.Array:
    """B at one point. Mirrors ``MirrorModel._B_impl``."""
    x = point[0]
    y = point[1]
    z = point[2]
    R_xy = jnp.sqrt(x * x + y * y)
    phi = jnp.atan2(y, x)
    factor1 = (1.0 + ((z - Zm_p) / gamma_p) ** 2) ** 2
    factor2 = (1.0 + ((z + Zm_p) / gamma_p) ** 2) ** 2
    Br = (R_xy * B0_p / (jnp.pi * gamma_p**3)) * (
        (z - Zm_p) / factor1 + (z + Zm_p) / factor2
    )
    Bz = _mirror_psi(R_xy, z, B0_p, gamma_p, Zm_p) * 2.0 / (R_xy * R_xy)
    bx = Br * jnp.cos(phi)
    by = Br * jnp.sin(phi)
    return jnp.stack((bx, by, Bz))


def _mirror_dB_pointwise(
    point: jax.Array, B0_p: jax.Array, gamma_p: jax.Array, Zm_p: jax.Array
) -> jax.Array:
    """``dB[j, l] = (∂_j B_l)`` matching ``_dB_by_dX_impl`` layout."""
    x = point[0]
    y = point[1]
    z = point[2]
    R_xy = jnp.sqrt(x * x + y * y)
    phi = jnp.atan2(y, x)

    factor1 = (1.0 + ((z - Zm_p) / gamma_p) ** 2) ** 2
    factor2 = (1.0 + ((z + Zm_p) / gamma_p) ** 2) ** 2
    Br = (R_xy * B0_p / (jnp.pi * gamma_p**3)) * (
        (z - Zm_p) / factor1 + (z + Zm_p) / factor2
    )
    dBrdr = (B0_p / (jnp.pi * gamma_p**3)) * (
        (z - Zm_p) / factor1 + (z + Zm_p) / factor2
    )
    dBzdz = -2.0 * dBrdr
    dBrdz = (B0_p * R_xy / (jnp.pi * gamma_p**3)) * (
        1.0 / factor1
        + 1.0 / factor2
        - 4.0
        * gamma_p**4
        * (
            (z - Zm_p) ** 2 / ((z - Zm_p) ** 2 + gamma_p**2) ** 3
            + (z + Zm_p) ** 2 / ((z + Zm_p) ** 2 + gamma_p**2) ** 3
        )
    )
    cosphi = jnp.cos(phi)
    sinphi = jnp.sin(phi)
    dcosphidx = -x * x / R_xy**3 + 1.0 / R_xy
    dsinphidx = -x * y / R_xy**3
    dcosphidy = -x * y / R_xy**3
    dsinphidy = -y * y / R_xy**3 + 1.0 / R_xy
    drdx = x / R_xy
    drdy = y / R_xy
    dBxdx = dBrdr * drdx * cosphi + Br * dcosphidx
    dBxdy = dBrdr * drdy * cosphi + Br * dcosphidy
    dBxdz = dBrdz * cosphi
    dBydx = dBrdr * drdx * sinphi + Br * dsinphidx
    dBydy = dBrdr * drdy * sinphi + Br * dsinphidy
    dBydz = dBrdz * sinphi

    # CPU layout (post-asignments in source):
    #   dB[:, 0, 0] = dBxdx, dB[:, 1, 0] = dBxdy, dB[:, 2, 0] = dBxdz,
    #   dB[:, 0, 1] = dBydx, dB[:, 1, 1] = dBydy, dB[:, 2, 1] = dBydz,
    #   dB[:, 0, 2] = 0, dB[:, 1, 2] = 0, dB[:, 2, 2] = dBzdz.
    zero = jnp.zeros_like(x)
    row_j0 = jnp.stack((dBxdx, dBydx, zero))
    row_j1 = jnp.stack((dBxdy, dBydy, zero))
    row_j2 = jnp.stack((dBxdz, dBydz, dBzdz))
    return jnp.stack((row_j0, row_j1, row_j2), axis=0)


_mirror_B_vmap = jax.vmap(_mirror_B_pointwise, in_axes=(0, None, None, None))
_mirror_dB_vmap = jax.vmap(_mirror_dB_pointwise, in_axes=(0, None, None, None))


@jax.jit
def _mirror_B_jit(
    B0: jax.Array, gamma: jax.Array, Zm: jax.Array, points: jax.Array
) -> jax.Array:
    return _mirror_B_vmap(points, B0, gamma, Zm)


@jax.jit
def _mirror_dB_jit(
    B0: jax.Array, gamma: jax.Array, Zm: jax.Array, points: jax.Array
) -> jax.Array:
    return _mirror_dB_vmap(points, B0, gamma, Zm)


def _mirror_scalars(spec: MirrorModelSpec) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        _as_jax_float64(spec.B0),
        _as_jax_float64(spec.gamma),
        _as_jax_float64(spec.Z_m),
    )


def mirror_B(spec: MirrorModelSpec, points: jax.Array) -> jax.Array:
    """``B(x)`` for the WHAM double-Lorentzian mirror model."""
    B0, gamma, Zm = _mirror_scalars(spec)
    return _mirror_B_jit(B0, gamma, Zm, _validate_points(points))


def mirror_dB(spec: MirrorModelSpec, points: jax.Array) -> jax.Array:
    """First spatial gradient of the WHAM double-Lorentzian mirror field.

    Returns
    -------
    dB : jax.Array
        Shape ``(n_points, 3, 3)``. Axis convention:
        ``dB[p, j, l] = ∂_j B_l(x_p)``. Axis 1 is the spatial derivative
        direction; axis 2 is the B-field component. This matches the
        CPU ``MirrorModel._dB_by_dX_impl`` row-major slab order.
    """
    B0, gamma, Zm = _mirror_scalars(spec)
    return _mirror_dB_jit(B0, gamma, Zm, _validate_points(points))
