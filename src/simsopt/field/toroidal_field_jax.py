"""JAX-backed drop-in for :class:`simsopt.field.ToroidalField`."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .._core.json import GSONDecoder
from ..jax_core.analytic_pure_fields import (
    ToroidalFieldSpec,
    toroidal_A,
    toroidal_B,
    toroidal_d2B,
    toroidal_dA,
    toroidal_dB,
)
from ._jax_common import points_device as _points_device
from .magneticfield import MagneticField


__all__ = ["ToroidalFieldJAX"]


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

    def jax_B_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return toroidal_B(self._spec, points)[0]

    def jax_B_dB_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return toroidal_B(self._spec, points)[0], toroidal_dB(self._spec, points)[0]

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
