"""JAX-backed public wrapper for :class:`simsopt.field.DipoleField`.

Item 26 of the Tier P2 wave: a drop-in :class:`MagneticField` subclass
that mirrors the upstream CPU ``DipoleField`` API while routing the
``B`` / ``A`` / ``dB`` / ``dA`` hot paths through the pure JAX kernels
introduced by item 24 in :mod:`simsopt.jax_core.dipole_field`.

Architecture invariants
-----------------------

The upstream CPU class :class:`simsopt.field.magneticfieldclasses.DipoleField`
remains the parity oracle. This wrapper does NOT modify it. The new
class is a parallel subclass of :class:`simsopt.field.MagneticField`
that participates in the ``Optimizable`` dependency graph through the
same base-class boilerplate.

JAX integration policy
----------------------

* The dipole grid (``dipole_grid``) and moment vectors (``m_vec``) are
  expanded into the full symmetric manifold once at construction time
  using pure NumPy. This mirrors :meth:`DipoleField._dipole_fields_from_symmetries`
  exactly. The expanded arrays are then staged to JAX float64 device
  arrays via :func:`simsopt.jax_core._math_utils.as_jax_float64`, which
  routes through :func:`jax.device_put` so subsequent kernel calls are
  clean under ``transfer_guard("disallow")``.
* All hot-path kernel calls go through the JIT boundaries in
  :mod:`simsopt.jax_core.dipole_field`. The compiled output is
  materialised back to NumPy at the ``_*_impl`` boundary because the
  upstream ``sopp.MagneticField`` cache buffer is a contiguous NumPy
  array, so a deliberate device-to-host copy is required there.
* The dipole locations are immutable after construction: callers that
  need to retune the dipole moments must build a fresh
  :class:`DipoleFieldJAX`, matching the CPU class semantics.
"""

from __future__ import annotations

import jax
import numpy as np

from .._core.json import GSONDecoder
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.dipole_field import (
    dipole_field_A,
    dipole_field_B,
    dipole_field_dA,
    dipole_field_dB,
)
from .magneticfield import MagneticField


__all__ = ["DipoleFieldJAX"]


def _points_device(points: np.ndarray) -> jax.Array:
    """Stage host points to a JAX float64 device array via ``device_put``.

    The CPU ``MagneticField`` cache hands us a contiguous NumPy array.
    Routing through :func:`as_jax_float64` uses :func:`jax.device_put`,
    which is explicit and allowed under ``transfer_guard("disallow")``.
    """

    return _as_jax_float64(points)


def _expand_symmetries(
    dipole_grid: np.ndarray,
    dipole_vectors: np.ndarray,
    stellsym: bool,
    nfp: int,
    coordinate_flag: str,
    m_maxima: np.ndarray | None,
    R0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy expansion of the half-period dipole manifold.

    Mirrors :meth:`simsopt.field.DipoleField._dipole_fields_from_symmetries`
    line by line. Returns ``(full_grid_xyz, full_m_vec, full_m_maxima)``,
    each as a contiguous NumPy float64 array.
    """

    ndipoles = dipole_grid.shape[0]
    if m_maxima is None:
        m_maxima_arr = np.max(np.linalg.norm(dipole_vectors, axis=-1)) * np.ones(
            ndipoles
        )
    else:
        m_maxima_arr = np.asarray(m_maxima, dtype=np.float64)

    if stellsym:
        stell_list = [1, -1]
        nsym = nfp * 2
    else:
        stell_list = [1]
        nsym = nfp

    m = dipole_vectors.reshape(ndipoles, 3)

    grid_x = np.zeros(ndipoles * nsym, dtype=np.float64)
    grid_y = np.zeros(ndipoles * nsym, dtype=np.float64)
    grid_z = np.zeros(ndipoles * nsym, dtype=np.float64)
    m_vec = np.zeros((ndipoles * nsym, 3), dtype=np.float64)
    m_max = np.zeros(ndipoles * nsym, dtype=np.float64)

    ox = dipole_grid[:, 0]
    oy = dipole_grid[:, 1]
    oz = dipole_grid[:, 2]

    mmx = m[:, 0]
    mmy = m[:, 1]
    mmz = m[:, 2]
    if coordinate_flag == "cylindrical":
        phi_dipole = np.arctan2(oy, ox)
        mmx_rot = mmx * np.cos(phi_dipole) - mmy * np.sin(phi_dipole)
        mmy_rot = mmx * np.sin(phi_dipole) + mmy * np.cos(phi_dipole)
        mmx = mmx_rot
        mmy = mmy_rot
    elif coordinate_flag == "toroidal":
        phi_dipole = np.arctan2(oy, ox)
        theta_dipole = np.arctan2(oz, np.sqrt(ox**2 + oy**2) - R0)
        mmx_rot = (
            mmx * np.cos(phi_dipole) * np.cos(theta_dipole)
            - mmy * np.sin(phi_dipole)
            - mmz * np.cos(phi_dipole) * np.sin(theta_dipole)
        )
        mmy_rot = (
            mmx * np.sin(phi_dipole) * np.cos(theta_dipole)
            + mmy * np.cos(phi_dipole)
            - mmz * np.sin(phi_dipole) * np.sin(theta_dipole)
        )
        mmz_rot = mmx * np.sin(theta_dipole) + mmz * np.cos(theta_dipole)
        mmx = mmx_rot
        mmy = mmy_rot
        mmz = mmz_rot

    index = 0
    n = ndipoles
    for stell in stell_list:
        for fp in range(nfp):
            phi0 = (2 * np.pi / nfp) * fp

            grid_x[index : index + n] = ox * np.cos(phi0) - oy * np.sin(phi0) * stell
            grid_y[index : index + n] = ox * np.sin(phi0) + oy * np.cos(phi0) * stell
            grid_z[index : index + n] = oz * stell

            m_vec[index : index + n, 0] = mmx * np.cos(phi0) * stell - mmy * np.sin(
                phi0
            )
            m_vec[index : index + n, 1] = mmx * np.sin(phi0) * stell + mmy * np.cos(
                phi0
            )
            m_vec[index : index + n, 2] = mmz

            m_max[index : index + n] = m_maxima_arr
            index += n

    contig = np.ascontiguousarray
    full_grid = contig(np.array([grid_x, grid_y, grid_z]).T)
    return contig(full_grid), contig(m_vec), contig(m_max)


class DipoleFieldJAX(MagneticField):
    r"""JAX-backed dipole field, drop-in for :class:`simsopt.field.DipoleField`.

    Computes the magnetic field induced by ``N`` magnetic dipoles via

    .. math::

        B(\mathbf{x}) = \frac{\mu_0}{4\pi}
            \sum_{i=1}^{N} \left(
                \frac{3 \mathbf{r}_i \cdot \mathbf{m}_i}{|\mathbf{r}_i|^5}
                \mathbf{r}_i
                - \frac{\mathbf{m}_i}{|\mathbf{r}_i|^3}
            \right)

    where :math:`\mathbf{r}_i = \mathbf{x} - \mathbf{x}^{dipole}_i`. The
    constructor argument layout and symmetry-expansion convention match
    :class:`simsopt.field.DipoleField` exactly.

    Args:
        dipole_grid: 2D numpy array, shape ``(ndipoles, 3)``. Dipole
            site coordinates for one half-period; the constructor
            expands this into the full toroidal/stellarator manifold.
        dipole_vectors: 2D numpy array, shape ``(ndipoles, 3)``. Dipole
            moment vectors for the same half-period sites.
        stellsym: Whether to apply stellarator symmetry expansion.
        nfp: Number of toroidal field periods.
        coordinate_flag: ``"cartesian"`` (default), ``"cylindrical"`` or
            ``"toroidal"``. Selects the local frame for ``dipole_vectors``.
        m_maxima: Optional per-dipole maximum moments, shape
            ``(ndipoles,)``. Defaults to the largest moment magnitude
            broadcast across all sites, matching the CPU class.
        R0: Major radius used by the ``"toroidal"`` coordinate flag.
    """

    def __init__(
        self,
        dipole_grid,
        dipole_vectors,
        stellsym: bool = True,
        nfp: int = 1,
        coordinate_flag: str = "cartesian",
        m_maxima=None,
        R0: float = 1.0,
    ):
        MagneticField.__init__(self)
        if coordinate_flag not in ("cartesian", "cylindrical", "toroidal"):
            raise ValueError(
                "coordinate_flag must be 'cartesian', 'cylindrical' or "
                f"'toroidal'; got {coordinate_flag!r}."
            )
        self.R0 = float(R0)
        self.stellsym = bool(stellsym)
        self.nfp = int(nfp)
        self.coordinate_flag = coordinate_flag

        dipole_grid_arr = np.ascontiguousarray(dipole_grid, dtype=np.float64)
        dipole_vectors_arr = np.ascontiguousarray(dipole_vectors, dtype=np.float64)
        if dipole_grid_arr.ndim != 2 or dipole_grid_arr.shape[1] != 3:
            raise ValueError(
                "dipole_grid must have shape (ndipoles, 3); got "
                f"{dipole_grid_arr.shape!r}."
            )
        if dipole_vectors_arr.shape != dipole_grid_arr.shape:
            raise ValueError(
                "dipole_vectors must have the same shape as dipole_grid; "
                f"got {dipole_vectors_arr.shape!r} and {dipole_grid_arr.shape!r}."
            )

        full_grid, m_vec, m_max = _expand_symmetries(
            dipole_grid_arr,
            dipole_vectors_arr,
            self.stellsym,
            self.nfp,
            self.coordinate_flag,
            m_maxima,
            self.R0,
        )
        self.dipole_grid = full_grid
        self.m_vec = m_vec
        self.dipole_vectors = m_vec
        self.m_maxima = m_max

        # Pre-stage the expanded dipole arrays to the device once. The
        # dipole layout is immutable after construction, so subsequent
        # kernel calls reuse these device buffers under
        # ``transfer_guard("disallow")``.
        self._dipole_points_device = _as_jax_float64(self.dipole_grid)
        self._dipole_moments_device = _as_jax_float64(self.m_vec)

    def set_points_cart(self, xyz):
        result = super().set_points_cart(xyz)
        self._points_device = _points_device(np.asarray(xyz, dtype=np.float64))
        return result

    def set_points_cyl(self, rphiz):
        result = super().set_points_cyl(rphiz)
        self._points_device = _points_device(
            np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        )
        return result

    def _B_impl(self, B):
        B[:] = np.asarray(
            dipole_field_B(
                self._points_device,
                self._dipole_points_device,
                self._dipole_moments_device,
            ),
            dtype=np.float64,
        )

    def _dB_by_dX_impl(self, dB):
        dB[:] = np.asarray(
            dipole_field_dB(
                self._points_device,
                self._dipole_points_device,
                self._dipole_moments_device,
            ),
            dtype=np.float64,
        )

    def _A_impl(self, A):
        A[:] = np.asarray(
            dipole_field_A(
                self._points_device,
                self._dipole_points_device,
                self._dipole_moments_device,
            ),
            dtype=np.float64,
        )

    def _dA_by_dX_impl(self, dA):
        dA[:] = np.asarray(
            dipole_field_dA(
                self._points_device,
                self._dipole_points_device,
                self._dipole_moments_device,
            ),
            dtype=np.float64,
        )

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        d["dipole_grid"] = self.dipole_grid
        d["m_vec"] = self.m_vec
        d["m_maxima"] = self.m_maxima
        d["stellsym"] = self.stellsym
        d["nfp"] = self.nfp
        d["coordinate_flag"] = self.coordinate_flag
        d["R0"] = self.R0
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        dipole_grid = decoder.process_decoded(
            d["dipole_grid"], serial_objs_dict, recon_objs
        )
        m_vec = decoder.process_decoded(d["m_vec"], serial_objs_dict, recon_objs)
        m_maxima = decoder.process_decoded(d["m_maxima"], serial_objs_dict, recon_objs)
        # The serialised arrays are already the full expanded manifold; the
        # caller will have applied symmetries before saving. Rebuild with
        # nfp=1, stellsym=False so the constructor does not re-expand the
        # already-expanded arrays.
        field = cls(
            dipole_grid=dipole_grid,
            dipole_vectors=m_vec,
            stellsym=False,
            nfp=1,
            coordinate_flag="cartesian",
            m_maxima=m_maxima,
            R0=d["R0"],
        )
        # Restore the original symmetry metadata so downstream introspection
        # observes the same flags as the source object.
        field.stellsym = bool(d["stellsym"])
        field.nfp = int(d["nfp"])
        field.coordinate_flag = str(d["coordinate_flag"])
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field
