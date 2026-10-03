"""JAX-backed drop-in for :class:`simsopt.field.PoloidalField`."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from simsopt._core.json import GSONDecoder
from simsopt_jax.core.analytic_pure_fields import (
    PoloidalFieldSpec,
    poloidal_B,
    poloidal_dB,
)
from simsopt_jax.field.common import host_cache_array as _host_cache_array
from simsopt_jax.field.common import points_device as _points_device
from simsopt.field.magneticfield import MagneticField


__all__ = ["PoloidalFieldJAX"]


class PoloidalFieldJAX(MagneticField):
    """JAX-backed poloidal field, drop-in for :class:`simsopt.field.PoloidalField`."""

    _simsopt_jax_native_field = True

    def __init__(self, R0, B0, q):
        MagneticField.__init__(self)
        self.R0 = float(R0)
        self.B0 = float(B0)
        self.q = float(q)
        self._spec = PoloidalFieldSpec(R0=self.R0, B0=self.B0, q=self.q)

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        B[:] = _host_cache_array(
            poloidal_B(self._spec, _points_device(points)), dtype=np.float64
        )

    def jax_B_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return poloidal_B(self._spec, points)[0]

    def jax_B_dB_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return poloidal_B(self._spec, points)[0], poloidal_dB(self._spec, points)[0]

    def _dB_by_dX_impl(self, dB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        dB[:] = _host_cache_array(
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
