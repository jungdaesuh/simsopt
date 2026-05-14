"""JAX-backed public wrapper for :class:`simsopt.field.WireframeField`."""

from __future__ import annotations

import numpy as np

import jax

from ..jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
)
from ..jax_core.wireframe import (
    wireframe_B,
    wireframe_B_and_dB_by_dX,
    wireframe_segment_B_contributions,
)
from ..geo.surfacerzfourier import SurfaceRZFourier
from .magneticfield import MagneticField

__all__ = ["WireframeFieldJAX"]


def _snapshot_wireframe_arrays(wframe):
    nodes = np.array(np.stack(wframe.nodes), dtype=np.float64, order="C", copy=True)
    segments = np.array(wframe.segments, dtype=np.int32, order="C", copy=True)
    seg_signs = np.array(wframe.seg_signs, dtype=np.float64, order="C", copy=True)
    currents = np.array(wframe.currents, dtype=np.float64, order="C", copy=True)
    return nodes, segments, seg_signs, currents


@jax.jit
def _wireframe_segment_B_contributions_jit(points, nodes, segments, seg_signs):
    return wireframe_segment_B_contributions(points, nodes, segments, seg_signs)


class WireframeFieldJAX(MagneticField):
    """JAX-backed magnetic field for a fixed-current toroidal wireframe.

    This wrapper mirrors the public ``WireframeField`` field-evaluation
    surface while keeping the upstream C++ class as the parity oracle.
    The wireframe geometry and currents are snapshotted at construction,
    matching the CPU wrapper's construction-time handoff to
    ``simsoptpp.WireframeField``.
    """

    _simsopt_jax_native_field = True

    def __init__(self, wframe):
        MagneticField.__init__(self)
        self.wireframe = wframe
        self.nodes, self.segments, self.seg_signs, self.currents = (
            _snapshot_wireframe_arrays(wframe)
        )
        self._nodes_device = _as_jax_float64(self.nodes)
        self._segments_device = _as_jax_int32(self.segments)
        self._seg_signs_device = _as_jax_float64(self.seg_signs)
        self._currents_device = _as_jax_float64(self.currents)

    def set_points_cart(self, xyz):
        result = super().set_points_cart(xyz)
        self._points_device = _as_jax_float64(np.asarray(xyz, dtype=np.float64))
        return result

    def set_points_cyl(self, rphiz):
        result = super().set_points_cyl(rphiz)
        self._points_device = _as_jax_float64(
            np.asarray(self.get_points_cart_ref(), dtype=np.float64)
        )
        return result

    def _B_impl(self, B):
        B[:] = np.asarray(
            wireframe_B(
                self._points_device,
                self._nodes_device,
                self._segments_device,
                self._seg_signs_device,
                self._currents_device,
            ),
            dtype=np.float64,
        )

    def _dB_by_dX_impl(self, dB):
        _, dB_jax = wireframe_B_and_dB_by_dX(
            self._points_device,
            self._nodes_device,
            self._segments_device,
            self._seg_signs_device,
            self._currents_device,
        )
        dB[:] = np.asarray(dB_jax, dtype=np.float64)

    def dB_by_dsegmentcurrents(self, compute_derivatives):
        """Return unit-current segment field contributions.

        The CPU wrapper accepts ``compute_derivatives=1`` but still returns
        the cached ``B_i`` segment-field arrays. Second spatial derivatives
        are not implemented by the C++ oracle and are not claimed here.
        """

        assert compute_derivatives >= 0
        if compute_derivatives > 1:
            raise NotImplementedError(
                "Second spatial derivatives are not implemented for WireframeField."
            )
        contributions = np.asarray(
            _wireframe_segment_B_contributions_jit(
                self._points_device,
                self._nodes_device,
                self._segments_device,
                self._seg_signs_device,
            ),
            dtype=np.float64,
        )
        self._dB_by_dcoilcurrents = [
            np.ascontiguousarray(contributions[i])
            for i in range(self.wireframe.n_segments)
        ]
        return self._dB_by_dcoilcurrents

    def dBnormal_by_dsegmentcurrents_matrix(self, surface, area_weighted=False):
        """Build the normal-field derivative matrix on a plasma surface."""

        points = self.get_points_cart_ref()
        n_points = len(points)

        if not isinstance(surface, SurfaceRZFourier):
            raise ValueError("Surface must be a SurfaceRZFourier object")

        normal = surface.normal()
        absn = np.linalg.norm(normal, axis=2)
        unitn = normal * (1.0 / absn)[:, :, None]

        if area_weighted:
            fac = np.sqrt(absn / float(absn.size))
        else:
            fac = np.ones(absn.shape)

        matrix = np.ascontiguousarray(
            np.zeros((n_points, self.wireframe.n_segments), dtype=np.float64)
        )
        dB_dsc = self.dB_by_dsegmentcurrents(0)
        for i in range(self.wireframe.n_segments):
            dB_dsc_i = dB_dsc[i].reshape(normal.shape)
            matrix[:, i] = (fac * np.sum(dB_dsc_i * unitn, axis=2)).reshape((-1))
        return matrix
