"""JAX-backed drop-in for :class:`simsopt.field.CircularCoil`."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .._core.json import GSONDecoder
from ..jax_core.circular_coil import (
    CircularCoilSpec,
    circular_coil_B,
    circular_coil_dB,
)
from ._jax_common import points_device as _points_device
from .magneticfield import MagneticField


__all__ = ["CircularCoilJAX"]


class CircularCoilJAX(MagneticField):
    """JAX-backed circular-coil field, drop-in for
    :class:`simsopt.field.CircularCoil` B and dB evaluation.

    The CPU class remains the parity oracle. The JAX wrapper preserves
    the public constructor payload and current normalization while
    routing the field hot path through immutable
    :class:`~simsopt.jax_core.circular_coil.CircularCoilSpec` kernels
    (which call the Carlson elliptic helpers in
    :mod:`simsopt.jax_core._elliptic`).
    """

    _simsopt_jax_native_field = True

    def __init__(self, r0=0.1, center=(0, 0, 0), I=5e5 / np.pi, normal=(0, 0)):
        MagneticField.__init__(self)
        self.r0 = float(r0)
        self.Inorm = float(I) * 4e-7
        self.center = tuple(float(v) for v in center)
        self.normal = tuple(float(v) for v in normal)
        self._spec = self._build_spec()

    @property
    def I(self):  # noqa: E743 - mirrors upstream CircularCoil public attribute.
        return self.Inorm * 25e5

    def _build_spec(self) -> CircularCoilSpec:
        return CircularCoilSpec(
            r0=self.r0,
            center=self.center,
            Inorm=self.Inorm,
            normal=self.normal,
        )

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        B[:] = np.asarray(
            circular_coil_B(self._spec, _points_device(points)), dtype=np.float64
        )

    def jax_B_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return circular_coil_B(self._spec, points)[0]

    def jax_B_dB_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        return circular_coil_B(self._spec, points)[0], circular_coil_dB(
            self._spec, points
        )[0]

    def _dB_by_dX_impl(self, dB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        dB[:] = np.asarray(
            circular_coil_dB(self._spec, _points_device(points)), dtype=np.float64
        )

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["r0"], d["center"], d["I"], d["normal"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field
