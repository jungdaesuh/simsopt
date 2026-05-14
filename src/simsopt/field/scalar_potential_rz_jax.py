"""JAX-backed ``ScalarPotentialRZMagneticField`` wrapper."""

from __future__ import annotations

import numpy as np
from sympy.parsing.sympy_parser import parse_expr

from .._core.json import GSONDecoder
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.scalar_potential_rz import scalar_potential_rz_kernels
from .magneticfield import MagneticField


__all__ = ["ScalarPotentialRZMagneticFieldJAX"]


class ScalarPotentialRZMagneticFieldJAX(MagneticField):
    """JAX-backed scalar-potential field with static SymPy lowering.

    The constructor mirrors ``ScalarPotentialRZMagneticField(phi_str)`` but
    lowers the parsed SymPy expression once at construction time. Runtime
    ``B`` and ``dB_by_dX`` evaluation then stays in pure JAX kernels.
    """

    _simsopt_jax_native_field = True

    def __init__(self, phi_str: str):
        MagneticField.__init__(self)
        self.phi_str = str(phi_str)
        self.phi_parsed = parse_expr(self.phi_str)
        self._B_kernel, self._dB_kernel = scalar_potential_rz_kernels(self.phi_parsed)

    def _points_device(self):
        return _as_jax_float64(np.asarray(self.get_points_cart_ref(), dtype=np.float64))

    def _B_impl(self, B):
        B[:] = np.asarray(self._B_kernel(self._points_device()), dtype=np.float64)

    def _dB_by_dX_impl(self, dB):
        dB[:] = np.asarray(self._dB_kernel(self._points_device()), dtype=np.float64)

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        d["phi_str"] = self.phi_str
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["phi_str"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field
