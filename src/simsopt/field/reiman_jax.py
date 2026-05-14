"""JAX-backed drop-in for :class:`simsopt.field.Reiman`."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .._core.json import GSONDecoder
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.analytic_fields import (
    ReimanSpec,
    reiman_B,
    reiman_dB,
)
from ._jax_common import points_device as _points_device
from .magneticfield import MagneticField


__all__ = ["ReimanJAX"]


class ReimanJAX(MagneticField):
    """JAX-backed Reiman island-model field, drop-in for
    :class:`simsopt.field.Reiman`.
    """

    _simsopt_jax_native_field = True

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

    def jax_B_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return reiman_B(self._spec, points)[0]

    def jax_B_dB_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return reiman_B(self._spec, points)[0], reiman_dB(self._spec, points)[0]

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
