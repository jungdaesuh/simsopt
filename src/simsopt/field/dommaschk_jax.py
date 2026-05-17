"""JAX-backed drop-in for :class:`simsopt.field.Dommaschk`."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .._core.json import GSONDecoder
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.analytic_fields import (
    DommaschkSpec,
    dommaschk_B,
    dommaschk_dB,
)
from ..jax_core.analytic_pure_fields import (
    ToroidalFieldSpec,
    toroidal_B,
    toroidal_dB,
)
from ._jax_common import points_device as _points_device
from .magneticfield import MagneticField
from .toroidal_field_jax import ToroidalFieldJAX


__all__ = ["DommaschkJAX"]

_DommaschkSpecKey = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[float, float], ...],
]


def _toroidal_baseline_B_dB(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian ``ToroidalField(R0=1, B0=1)`` baseline B and dB.

    The CPU :class:`simsopt.field.Dommaschk` class folds an explicit
    ``ToroidalField(1, 1)`` contribution into the returned ``B`` and
    ``dB``. Reproducing that addition via :class:`ToroidalFieldJAX`
    keeps :class:`DommaschkJAX` bit-identical to the CPU oracle.
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

    _simsopt_jax_native_field = True

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
        self._spec_key = self._spec_signature()
        self._spec = self._build_spec_from_key(self._spec_key)

    @staticmethod
    def _build_spec_from_key(key: _DommaschkSpecKey) -> DommaschkSpec:
        m_tuple, n_tuple, coeffs_tuple = key
        return DommaschkSpec(
            m=m_tuple,
            n=n_tuple,
            coeffs=_as_jax_float64(coeffs_tuple),
        )

    def _spec_signature(self) -> _DommaschkSpecKey:
        coeffs_array = np.asarray(self.coeffs, dtype=np.float64)
        return (
            tuple(int(v) for v in self.m),
            tuple(int(v) for v in self.n),
            tuple((float(row[0]), float(row[1])) for row in coeffs_array),
        )

    def _current_spec(self) -> DommaschkSpec:
        key = self._spec_signature()
        if key != self._spec_key:
            self._spec_key = key
            self._spec = self._build_spec_from_key(key)
        return self._spec

    @property
    def mn(self):
        return np.column_stack((self.m, self.n))

    def _B_impl(self, B):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        per_mode = np.asarray(
            dommaschk_B(self._current_spec(), _points_device(points)), dtype=np.float64
        )
        baseline_B, _ = _toroidal_baseline_B_dB(points)
        B[:] = np.add.reduce(per_mode) + baseline_B

    def jax_B_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        baseline_spec = ToroidalFieldSpec(R0=1.0, B0=1.0)
        return (
            jnp.sum(dommaschk_B(self._current_spec(), points), axis=0)[0]
            + toroidal_B(baseline_spec, points)[0]
        )

    def jax_B_dB_at(self, point):
        points = jnp.asarray(point, dtype=jnp.float64).reshape((1, 3))
        baseline_spec = ToroidalFieldSpec(R0=1.0, B0=1.0)
        spec = self._current_spec()
        B = (
            jnp.sum(dommaschk_B(spec, points), axis=0)[0]
            + toroidal_B(baseline_spec, points)[0]
        )
        dB = (
            jnp.sum(dommaschk_dB(spec, points), axis=0)[0]
            + toroidal_dB(baseline_spec, points)[0]
        )
        return B, dB

    def _dB_by_dX_impl(self, dB):
        points = np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        per_mode = np.asarray(
            dommaschk_dB(self._current_spec(), _points_device(points)), dtype=np.float64
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
