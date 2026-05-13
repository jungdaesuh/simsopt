"""JAX-backed wrapper for :class:`simsopt.field.InterpolatedField`.

This class samples a source :class:`MagneticField` on a cylindrical
:math:`(r, \\phi, z)` grid at construction time and routes subsequent
:meth:`B` / :meth:`GradAbsB` calls through the JAX kernel pipeline in
:mod:`simsopt.jax_core.interpolated_field`. The CPU
``simsopt.field.InterpolatedField`` remains the parity oracle.

Layered design
--------------

- :mod:`simsopt.jax_core.regular_grid_interp` (item 13) is the
  rectangular-cuboid kernel layer.
- :mod:`simsopt.jax_core.interpolated_field` (this item) wraps the
  rectangular kernel with the cylindrical coordinate conversion and the
  ``nfp`` / ``stellsym`` folding rules from
  ``src/simsoptpp/magneticfield_interpolated.h``.
- This module attaches the JAX-backed pipeline to the public
  :class:`simsopt.field.MagneticField` cache contract.

The wrapper deliberately does NOT implement ``_dB_by_dX_impl``. The CPU
:class:`InterpolatedField` does not implement it either (it raises a
runtime error inside the C++ binding); the upstream class exposes
``_GradAbsB_impl`` instead, computed from a separately-interpolated
``\\nabla |B|`` table. The JAX wrapper preserves this semantic: a
caller that needs Cartesian Jacobians of ``B`` should evaluate the
underlying source field directly.

The interpolated tables are constructed eagerly in ``__init__``; the
sampling step is the documented CPU<->JAX boundary for this wrapper.
"""

from __future__ import annotations

import numpy as np

from .._core.json import GSONDecoder
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.interpolated_field import (
    interpolated_field_B,
    interpolated_field_GradAbsB,
    make_interpolated_field_spec,
)
from ..jax_core.regular_grid_interp import (
    UniformInterpolationRule,
    build_regular_grid_interpolant_3d,
)
from .magneticfield import MagneticField


__all__ = ["InterpolatedFieldJAX"]


def _points_device(points: np.ndarray):
    """Stage host points to a JAX float64 device array via the strict-safe
    ``jax.device_put`` path. Mirrors the helper in
    :mod:`simsopt.field.magneticfieldclasses_jax`.
    """

    return _as_jax_float64(points)


def _cyl_to_cart_points(r: np.ndarray, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Convert flat ``(N,)`` cylindrical coordinate arrays to a
    contiguous ``(N, 3)`` Cartesian array.

    Used at construction time to feed the source field's
    ``set_points_cart`` boundary while sampling the cylindrical mesh DOFs.
    The host-side conversion is exact for our purposes (no autodiff is
    required across this step).
    """

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float64)


def _build_skip_callback(skip_callable):
    """Adapt the upstream cylindrical-coordinate skip predicate to the
    rectangular-kernel callback contract.

    Upstream ``InterpolatedField`` calls ``skip(r, phi, z) -> [bool]``
    where the inputs are the cylindrical mesh-node coordinates. The
    rectangular-kernel builder calls ``skip(xs, ys, zs)`` where the
    arguments are flat arrays in the rectangular-axis names. We
    therefore forward the rectangular axes verbatim because the
    rectangular axes ARE :math:`(r, \\phi, z)` here.
    """

    if skip_callable is None:

        def _no_skip(xs, ys, zs):
            return np.zeros_like(xs, dtype=bool)

        return _no_skip

    def _forward(xs, ys, zs):
        return np.asarray(skip_callable(xs, ys, zs), dtype=bool)

    return _forward


def _build_sampler(source_field, value_kind: str):
    """Return a callback that the rectangular-kernel builder can invoke.

    ``value_kind`` is either ``"B"`` or ``"GradAbsB"``. The callback
    receives the flat ``(r, phi, z)`` coordinates of the cells' kept
    Lagrange DOFs, sets the source field's points via
    ``set_points_cyl``, fetches the corresponding cylindrical field
    tensor, and returns it as a flat ``(N*3,)`` row-major buffer matching
    the rectangular-kernel callback contract.
    """

    def _sample(rs, phis, zs):
        cyl_points = np.ascontiguousarray(
            np.stack([rs, phis, zs], axis=1), dtype=np.float64
        )
        old_points = source_field.get_points_cart()
        source_field.set_points_cyl(cyl_points)
        if value_kind == "B":
            field_cyl = np.asarray(source_field.B_cyl(), dtype=np.float64)
        else:
            field_cyl = np.asarray(source_field.GradAbsB_cyl(), dtype=np.float64)
        # Restore the source field's points so this construction step
        # does not leak state into the caller's cache.
        source_field.set_points_cart(np.ascontiguousarray(old_points))
        return field_cyl.reshape(-1)

    return _sample


def _checked_host_result(value, *, extrapolate: bool, quantity: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not extrapolate and np.isnan(result).any():
        raise RuntimeError(
            f"InterpolatedFieldJAX {quantity} query is outside the interpolation "
            "domain or inside a skipped cell while extrapolate=False."
        )
    return result


class InterpolatedFieldJAX(MagneticField):
    """JAX-backed drop-in for :class:`simsopt.field.InterpolatedField`.

    The constructor signature matches the CPU class. At ``__init__``, the
    source field is sampled on the cylindrical :math:`(r, \\phi, z)`
    mesh into two immutable rectangular-grid interpolant specs (one for
    cylindrical ``B``, one for cylindrical ``\\nabla |B|``). The
    sampling step is the CPU<->JAX boundary for the wrapper.
    Subsequent :meth:`B` / :meth:`GradAbsB` calls evaluate via the
    JIT-compiled pipeline in
    :mod:`simsopt.jax_core.interpolated_field`.

    The ``set_points`` cache invalidation semantics inherited from
    :class:`MagneticField` carry over unchanged.
    """

    def __init__(
        self,
        field,
        degree,
        rrange,
        phirange,
        zrange,
        extrapolate=True,
        nfp=1,
        stellsym=False,
        skip=None,
    ):
        MagneticField.__init__(self)
        self.__field = field
        self.degree = int(degree)
        self.rrange = tuple(rrange)
        self.phirange = tuple(phirange)
        self.zrange = tuple(zrange)
        self.extrapolate = bool(extrapolate)
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)
        self._skip_callable = skip

        rule = UniformInterpolationRule(self.degree)
        skip_cb = _build_skip_callback(skip)

        B_spec = build_regular_grid_interpolant_3d(
            rule=rule,
            xrange=self.rrange,
            yrange=self.phirange,
            zrange=self.zrange,
            value_size=3,
            f=_build_sampler(field, "B"),
            out_of_bounds_ok=self.extrapolate,
            skip=skip_cb,
        )
        GradAbsB_spec = build_regular_grid_interpolant_3d(
            rule=rule,
            xrange=self.rrange,
            yrange=self.phirange,
            zrange=self.zrange,
            value_size=3,
            f=_build_sampler(field, "GradAbsB"),
            out_of_bounds_ok=self.extrapolate,
            skip=skip_cb,
        )
        self._spec = make_interpolated_field_spec(
            nfp=self.nfp,
            stellsym=self.stellsym,
            B_spec=B_spec,
            GradAbsB_spec=GradAbsB_spec,
        )

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        B[:] = _checked_host_result(
            interpolated_field_B(self._spec, _points_device(points)),
            extrapolate=self.extrapolate,
            quantity="B",
        )

    def GradAbsB(self):
        """JAX-direct ``\\nabla |B|`` evaluation in Cartesian coordinates.

        The C++ ``MagneticField`` pybind11 trampoline (see
        ``src/simsoptpp/pymagneticfield.h``) forwards ``_B_impl``,
        ``_dB_by_dX_impl``, ``_d2B_by_dXdX_impl``, ``_A_impl``,
        ``_dA_by_dX_impl`` and ``_d2A_by_dXdX_impl`` to Python overrides
        but does NOT forward ``_GradAbsB_impl``. The CPU
        :class:`simsopt.field.InterpolatedField` works around this by
        being a C++ subclass that overrides
        ``InterpolatedField::_GradAbsB_impl`` directly. The JAX wrapper
        cannot use the same C++ hook, so it shadows the bound
        ``GradAbsB`` method at the Python level and routes the call
        through the JAX kernel pipeline. The cache
        invalidation behaviour of :meth:`set_points` is unchanged because
        this path does not touch the C++ ``data_GradAbsB`` slot.
        """

        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        return _checked_host_result(
            interpolated_field_GradAbsB(self._spec, _points_device(points)),
            extrapolate=self.extrapolate,
            quantity="GradAbsB",
        )

    def GradAbsB_cyl(self):
        """JAX-direct ``\\nabla |B|`` evaluation in cylindrical coordinates.

        Companion to :meth:`GradAbsB`; rotates the Cartesian result back
        through the standard cylindrical projection so callers that use
        ``GradAbsB_cyl`` see the same JAX-backed table.
        """

        cart = self.GradAbsB()
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        phi = np.arctan2(points[:, 1], points[:, 0])
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        out = np.empty_like(cart)
        out[:, 0] = cos_phi * cart[:, 0] + sin_phi * cart[:, 1]
        out[:, 1] = cos_phi * cart[:, 1] - sin_phi * cart[:, 0]
        out[:, 2] = cart[:, 2]
        return out

    def B_cyl(self):
        """JAX-direct ``B`` evaluation in cylindrical coordinates.

        Mirrors :meth:`GradAbsB_cyl`. Bypasses the C++ ``data_Bcyl``
        cache (which would otherwise be filled by the base-class
        ``_B_cyl_impl`` from the Cartesian ``B``) so the result comes
        directly from the JAX interpolant. The numerical result of the
        two paths is the same up to FP rotation roundoff, but the
        direct path keeps cylindrical output decoupled from any future
        change to the Cartesian cache.
        """

        cart = np.asarray(self.B(), dtype=np.float64)
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        phi = np.arctan2(points[:, 1], points[:, 0])
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        out = np.empty_like(cart)
        out[:, 0] = cos_phi * cart[:, 0] + sin_phi * cart[:, 1]
        out[:, 1] = cos_phi * cart[:, 1] - sin_phi * cart[:, 0]
        out[:, 2] = cart[:, 2]
        return out

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["field"] = self.__field
        d["degree"] = self.degree
        d["rrange"] = list(self.rrange)
        d["phirange"] = list(self.phirange)
        d["zrange"] = list(self.zrange)
        d["extrapolate"] = self.extrapolate
        d["nfp"] = self.nfp
        d["stellsym"] = self.stellsym
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        source = decoder.process_decoded(d["field"], serial_objs_dict, recon_objs)
        field = cls(
            source,
            d["degree"],
            tuple(d["rrange"]),
            tuple(d["phirange"]),
            tuple(d["zrange"]),
            extrapolate=d.get("extrapolate", True),
            nfp=d.get("nfp", 1),
            stellsym=d.get("stellsym", False),
        )
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field
