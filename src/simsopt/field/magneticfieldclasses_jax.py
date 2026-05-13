"""JAX-backed public wrappers for analytic magnetic-field classes.

This module is a partial P2 item-15 implementation. It exposes JAX-backed
``MagneticField`` subclasses that mirror the public API of the upstream
CPU classes in :mod:`simsopt.field.magneticfieldclasses` while routing
the field-evaluation hot paths through the immutable JAX specs/kernels
introduced by items 11 and 12:

- :class:`ToroidalFieldJAX` -- wraps :func:`toroidal_B`, :func:`toroidal_dB`,
  :func:`toroidal_d2B`, :func:`toroidal_A`, :func:`toroidal_dA` from
  :mod:`simsopt.jax_core.analytic_pure_fields`.
- :class:`PoloidalFieldJAX` -- wraps :func:`poloidal_B` /
  :func:`poloidal_dB`.
- :class:`MirrorModelJAX` -- wraps :func:`mirror_B` / :func:`mirror_dB`.
- :class:`DommaschkJAX` -- wraps :func:`dommaschk_B` /
  :func:`dommaschk_dB` from :mod:`simsopt.jax_core.analytic_fields`,
  including the ``ToroidalField(1, 1)`` baseline that the CPU
  ``Dommaschk`` adds explicitly.
- :class:`ReimanJAX` -- wraps :func:`reiman_B` / :func:`reiman_dB`.

Architecture invariants observed
--------------------------------

The upstream CPU classes (``ToroidalField``, ``PoloidalField``,
``MirrorModel``, ``Dommaschk``, ``Reiman``, ``InterpolatedField``) in
:mod:`simsopt.field.magneticfieldclasses` are **not modified**. They
remain the parity oracle. The new classes here are parallel
subclasses of :class:`simsopt.field.magneticfield.MagneticField` that
participate in the ``Optimizable`` dependency graph through the same
base-class boilerplate the CPU classes use.

JAX integration policy
----------------------

* All hot-path kernel calls go through the pre-built ``jit`` boundaries
  in :mod:`simsopt.jax_core.analytic_pure_fields` and
  :mod:`simsopt.jax_core.analytic_fields`. Host-to-device staging
  happens through ``jnp.asarray(..., dtype=jnp.float64)`` at the
  ``set_points`` boundary so the compiled paths never trigger an
  implicit transfer.
* Compiled output is materialised back to NumPy at the
  ``MagneticField._*_impl`` boundary (the C++ cache buffer is a
  contiguous NumPy array, so a deliberate device-to-host copy is
  required at that boundary). Internal JAX gradients (autodiff,
  ``Derivative`` projection) do not flow through this CPU-facing path;
  they are owned by downstream Stage-2 / single-stage JAX objective
  paths.
* ``InterpolatedField`` is deliberately **not** ported here. Item 13
  already exposes a JAX kernel for a rectangular 3D Cartesian grid
  (:mod:`simsopt.jax_core.regular_grid_interp`), but the public
  ``InterpolatedField`` semantics are cylindrical
  ``(r, phi, z)``, fold over ``nfp`` and ``stellsym``, and consume a
  source ``MagneticField`` via its CPU evaluation path. Building a
  drop-in ``InterpolatedFieldJAX`` therefore requires a significantly
  larger surface (cylindrical wrap-around, period folding, per-field
  cache table assembly) than item 15's analytic-field wrappers. That
  port is recorded as item 15-sub
  (``.artifacts/jax_port_goal/blockers/15-interpolatedfield-debug.md``)
  with category ``architecture`` and a concrete resolution path.

``CircularCoil`` remains deferred at item 12-sub
(``.artifacts/jax_port_goal/blockers/12-circularcoil-debug.md``) because
the required complete-elliptic-integral primitives are not exposed by
the runtime ``jaxlib`` 0.10.0 build.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from .._core.json import GSONDecoder
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.analytic_fields import (
    DommaschkSpec,
    ReimanSpec,
    dommaschk_B,
    dommaschk_dB,
    reiman_B,
    reiman_dB,
)
from ..jax_core.analytic_pure_fields import (
    MirrorModelSpec,
    PoloidalFieldSpec,
    ToroidalFieldSpec,
    mirror_B,
    mirror_dB,
    poloidal_B,
    poloidal_dB,
    toroidal_A,
    toroidal_B,
    toroidal_d2B,
    toroidal_dA,
    toroidal_dB,
)
from .magneticfield import MagneticField


__all__ = [
    "ToroidalFieldJAX",
    "PoloidalFieldJAX",
    "MirrorModelJAX",
    "DommaschkJAX",
    "ReimanJAX",
]


def _points_device(points: np.ndarray) -> jnp.ndarray:
    """Stage host points to a JAX float64 device array via the strict-safe
    ``jax.device_put`` path.

    The CPU ``MagneticField`` cache hands us a contiguous NumPy array.
    Going through :func:`simsopt.jax_core._math_utils.as_jax_float64`
    routes the staging through :func:`jax.device_put`, which is
    explicit and allowed under ``transfer_guard("disallow")``. The
    result is reused for every kernel call until ``set_points``
    invalidates the cache.
    """

    return _as_jax_float64(points)


# ── ToroidalField wrapper ────────────────────────────────────────────


class ToroidalFieldJAX(MagneticField):
    """JAX-backed ``B = B0 * R0 / R * e_phi`` toroidal field.

    Drop-in replacement for :class:`simsopt.field.ToroidalField`. The
    CPU ``ToroidalField`` remains the parity oracle.
    """

    def __init__(self, R0, B0):
        MagneticField.__init__(self)
        self.R0 = float(R0)
        self.B0 = float(B0)
        self._spec = ToroidalFieldSpec(R0=self.R0, B0=self.B0)

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        B[:] = np.asarray(
            toroidal_B(self._spec, _points_device(points)), dtype=np.float64
        )

    def _dB_by_dX_impl(self, dB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        dB[:] = np.asarray(
            toroidal_dB(self._spec, _points_device(points)), dtype=np.float64
        )

    def _d2B_by_dXdX_impl(self, ddB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        ddB[:] = np.asarray(
            toroidal_d2B(self._spec, _points_device(points)), dtype=np.float64
        )

    def _A_impl(self, A):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        A[:] = np.asarray(
            toroidal_A(self._spec, _points_device(points)), dtype=np.float64
        )

    def _dA_by_dX_impl(self, dA):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        dA[:] = np.asarray(
            toroidal_dA(self._spec, _points_device(points)), dtype=np.float64
        )

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["R0"], d["B0"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


# ── PoloidalField wrapper ────────────────────────────────────────────


class PoloidalFieldJAX(MagneticField):
    """JAX-backed poloidal field, drop-in for :class:`simsopt.field.PoloidalField`."""

    def __init__(self, R0, B0, q):
        MagneticField.__init__(self)
        self.R0 = float(R0)
        self.B0 = float(B0)
        self.q = float(q)
        self._spec = PoloidalFieldSpec(R0=self.R0, B0=self.B0, q=self.q)

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        B[:] = np.asarray(
            poloidal_B(self._spec, _points_device(points)), dtype=np.float64
        )

    def _dB_by_dX_impl(self, dB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        dB[:] = np.asarray(
            poloidal_dB(self._spec, _points_device(points)), dtype=np.float64
        )

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["R0"], d["B0"], d["q"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


# ── MirrorModel wrapper ──────────────────────────────────────────────


class MirrorModelJAX(MagneticField):
    """JAX-backed WHAM double-Lorentzian mirror field, drop-in for
    :class:`simsopt.field.MirrorModel`.
    """

    def __init__(self, B0=6.51292, gamma=0.124904, Z_m=0.98):
        MagneticField.__init__(self)
        self.B0 = float(B0)
        self.gamma = float(gamma)
        self.Z_m = float(Z_m)
        self._spec = MirrorModelSpec(B0=self.B0, gamma=self.gamma, Z_m=self.Z_m)

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        B[:] = np.asarray(
            mirror_B(self._spec, _points_device(points)), dtype=np.float64
        )

    def _dB_by_dX_impl(self, dB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        dB[:] = np.asarray(
            mirror_dB(self._spec, _points_device(points)), dtype=np.float64
        )

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["B0"], d["gamma"], d["Z_m"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


# ── Dommaschk wrapper ────────────────────────────────────────────────


def _toroidal_baseline_B_dB(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian ``ToroidalField(R0=1, B0=1)`` baseline B and dB.

    The CPU :class:`simsopt.field.Dommaschk` class folds an explicit
    ``ToroidalField(1, 1)`` contribution into the returned ``B`` and
    ``dB`` (compare :class:`Dommaschk._B_impl` /
    :class:`Dommaschk._dB_by_dX_impl`). Reproducing that addition via
    the new JAX wrappers keeps :class:`DommaschkJAX` bit-identical to
    the CPU oracle.
    """

    baseline = ToroidalFieldJAX(R0=1.0, B0=1.0)
    baseline.set_points_cart(np.ascontiguousarray(points, dtype=np.float64))
    return np.asarray(baseline.B(), dtype=np.float64), np.asarray(
        baseline.dB_by_dX(), dtype=np.float64
    )


class DommaschkJAX(MagneticField):
    """JAX-backed Dommaschk vacuum field, drop-in for
    :class:`simsopt.field.Dommaschk`.

    The wrapper mirrors the CPU class's ``mn`` and ``coeffs`` public API
    and folds in the same ``ToroidalField(R0=1, B0=1)`` baseline that
    the CPU class adds to the raw mode contribution.
    """

    def __init__(self, mn=None, coeffs=None):
        MagneticField.__init__(self)
        if mn is None:
            mn = [[0, 0]]
        if coeffs is None:
            coeffs = [[0, 0]]
        mn_array = np.array(mn, dtype=np.int16)
        coeffs_array = np.asarray(coeffs, dtype=np.float64)
        if mn_array.ndim != 2 or mn_array.shape[1] != 2:
            raise ValueError(
                f"mn must have shape (K, 2); got {tuple(mn_array.shape)!r}."
            )
        if coeffs_array.ndim != 2 or coeffs_array.shape[1] != 2:
            raise ValueError(
                f"coeffs must have shape (K, 2); got {tuple(coeffs_array.shape)!r}."
            )
        if mn_array.shape[0] != coeffs_array.shape[0]:
            raise ValueError(
                f"mn and coeffs must agree on K; got {mn_array.shape[0]} mn rows "
                f"and {coeffs_array.shape[0]} coeff rows."
            )
        self.m = mn_array[:, 0]
        self.n = mn_array[:, 1]
        self.coeffs = coeffs
        self._spec = self._build_spec(mn_array, coeffs_array)

    @staticmethod
    def _build_spec(mn_array: np.ndarray, coeffs_array: np.ndarray) -> DommaschkSpec:
        # Pre-stage ``coeffs`` to a device array via the strict-safe
        # ``jax.device_put`` path so subsequent kernel calls are clean
        # under ``transfer_guard("disallow")``.
        return DommaschkSpec(
            m=tuple(int(v) for v in mn_array[:, 0]),
            n=tuple(int(v) for v in mn_array[:, 1]),
            coeffs=_as_jax_float64(coeffs_array),
        )

    @property
    def mn(self):
        return np.column_stack((self.m, self.n))

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        per_mode = np.asarray(
            dommaschk_B(self._spec, _points_device(points)), dtype=np.float64
        )
        baseline_B, _ = _toroidal_baseline_B_dB(points)
        B[:] = np.add.reduce(per_mode) + baseline_B

    def _dB_by_dX_impl(self, dB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        per_mode = np.asarray(
            dommaschk_dB(self._spec, _points_device(points)), dtype=np.float64
        )
        _, baseline_dB = _toroidal_baseline_B_dB(points)
        dB[:] = np.add.reduce(per_mode) + baseline_dB

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        mn = decoder.process_decoded(d["mn"], serial_objs_dict, recon_objs)
        field = cls(mn, d["coeffs"])
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


# ── Reiman wrapper ───────────────────────────────────────────────────


class ReimanJAX(MagneticField):
    """JAX-backed Reiman island-model field, drop-in for
    :class:`simsopt.field.Reiman`.
    """

    def __init__(self, iota0=0.15, iota1=0.38, k=None, epsilonk=None, m0=1):
        MagneticField.__init__(self)
        if k is None:
            k = [6]
        if epsilonk is None:
            epsilonk = [0.01]
        if len(k) != len(epsilonk):
            raise ValueError(
                f"k and epsilonk must have equal length; got {len(k)} and "
                f"{len(epsilonk)}."
            )
        self.iota0 = float(iota0)
        self.iota1 = float(iota1)
        self.k = list(k)
        self.epsilonk = list(epsilonk)
        self.m0 = int(m0)
        self._spec = self._build_spec()

    def _build_spec(self) -> ReimanSpec:
        # Pre-stage the scalar fields ``iota0`` / ``iota1`` to device
        # arrays via the strict-safe ``jax.device_put`` path so the
        # public ``reiman_B`` / ``reiman_dB`` calls do not trigger
        # implicit host transfers under ``transfer_guard("disallow")``.
        return ReimanSpec(
            iota0=_as_jax_float64(self.iota0),
            iota1=_as_jax_float64(self.iota1),
            k_theta=tuple(int(v) for v in self.k),
            epsilon=_as_jax_float64(self.epsilonk),
            m0_symmetry=int(self.m0),
        )

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        B[:] = np.asarray(
            reiman_B(self._spec, _points_device(points)), dtype=np.float64
        )

    def _dB_by_dX_impl(self, dB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        dB[:] = np.asarray(
            reiman_dB(self._spec, _points_device(points)), dtype=np.float64
        )

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["iota0"], d["iota1"], d["k"], d["epsilonk"], d["m0"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field
