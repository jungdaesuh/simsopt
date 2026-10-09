"""QFM host operands retain call-time values while CPU work is pending."""

from unittest_jax_support import JaxTestCase

from dataclasses import replace

import jax
import numpy as np

from core.test_buffer_ownership import _host_array, make_execution_gate
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import SurfaceRZFourier, ToroidalFlux, create_equally_spaced_curves
from simsopt_jax.core import qfm
from simsopt_jax.runtime.host_boundary import host_array
from simsopt_jax_adapters.field import JaxBiotSavart
from simsopt_jax_adapters.geo.qfm import JaxQfmSurface


class TestQfmBufferOwnership(JaxTestCase):
    def test_public_kernels_snapshot_host_operands_before_dispatch(self):
        """Every QFM kernel owns host point/target/weight buffers before caller mutation."""
        for name in (
            "qfm_residual", "qfm_residual_value_and_grad", "qfm_label",
            "qfm_label_constraint", "qfm_label_constraint_value_and_grad",
            "qfm_penalty_constraints", "qfm_penalty_constraints_value_and_grad",
        ):
            with self.subTest(kernel=name), self.case():
                self._case_snapshot(name)

    def _case_snapshot(self, name):
        curves = create_equally_spaced_curves(1, 1, False, order=1, numquadpoints=32)
        coils = [Coil(curves[0], Current(1e5))]
        field = JaxBiotSavart(coils)
        surface = SurfaceRZFourier.from_nphi_ntheta(mpol=1, ntor=1, nphi=5, ntheta=6)
        surface.set_rc(1, 1, 0.02)
        label = ToroidalFlux(surface, BiotSavart(coils))
        solver = JaxQfmSurface(field, surface, label, 0.1)
        spec = solver.qfm._spec()
        label_spec = solver._label_spec()
        points = _host_array(spec.points.shape)
        points[:] = host_array(spec.points)
        label_points = _host_array(host_array(label_spec.points).shape)
        label_points[:] = host_array(label_spec.points)
        targetlabel = _host_array((1,)).reshape(())
        targetlabel[()] = 0.1
        constraint_weight = _host_array((1,)).reshape(())
        constraint_weight[()] = 2.0
        gate = make_execution_gate()
        function = getattr(qfm, name)

        def evaluate(phi):
            pending_surface = replace(spec.surface, quadpoints_phi=phi)
            operands = replace(spec, surface=pending_surface, points=points)
            constraint = replace(label_spec, surface=pending_surface, points=label_points)
            if name.startswith("qfm_residual"):
                return function(operands)
            if name == "qfm_label":
                return function(constraint)
            if name.startswith("qfm_label_constraint"):
                return function(constraint, targetlabel)
            return function(operands, constraint, targetlabel, constraint_weight)

        expected = jax.tree.map(host_array, evaluate(spec.surface.quadpoints_phi))
        jax.block_until_ready(gate(spec.surface.quadpoints_phi))
        actual = evaluate(gate(spec.surface.quadpoints_phi))
        self.assertTrue(any(not leaf.is_ready() for leaf in jax.tree.leaves(actual)))
        points[:] = 9.0
        label_points[:] = 9.0
        targetlabel[()] = 9.0
        constraint_weight[()] = 9.0
        for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(host_array(value), reference, rtol=1e-12, atol=1e-12)
