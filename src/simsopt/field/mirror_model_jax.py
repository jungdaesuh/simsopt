"""JAX-backed drop-in for :class:`simsopt.field.MirrorModel`."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .._core.json import GSONDecoder
from ..jax_core.analytic_pure_fields import (
    MirrorModelSpec,
    mirror_B,
    mirror_dB,
)
from ._jax_common import points_device as _points_device
from .magneticfield import MagneticField


__all__ = ["MirrorModelJAX"]


class MirrorModelJAX(MagneticField):
    """JAX-backed WHAM double-Lorentzian mirror field, drop-in for
    :class:`simsopt.field.MirrorModel`.
    """

    _simsopt_jax_native_field = True

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

    def jax_B_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return mirror_B(self._spec, points)[0]

    def jax_B_dB_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return mirror_B(self._spec, points)[0], mirror_dB(self._spec, points)[0]

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
