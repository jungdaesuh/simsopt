"""Regression tests for the single-stage surface-stack spacing ALM constraint gradient.

Root cause (banana_opt/single_stage_constraints.py): the inter-surface
("surface-stack spacing") ALM constraint built its gradient with
``surface_dgamma_by_dcoeff_derivative``, i.e. a ``Derivative({surface: ...})`` that
lives on the FIXED surface Fourier dofs. The nested surfaces are not in the coil dof
graph -- they are an implicit function of the coil dofs through the per-iteration
BoozerSurface solve -- so projecting that derivative onto the coil-only objective is
identically zero. The constraint was therefore gradient-dead over the coil dofs and
could not actuate the coils (the same mechanism already fixed for the LCFS
MajorRadius/MinorRadius constraints).

The fix routes the per-surface point gradient through each surface's BoozerSurface
adjoint (reusing simsopt's ``forward_backward`` + the cached ``res['vjp']``/``res['PLU']``
exactly as ``simsopt.geo.surfaceobjectives.MajorRadius`` does) and sums across the
stack. These tests construct a real nested pair of solved BoozerSurfaces sharing the
same coils (no optimization) and assert observable behavior:

  * the OLD (``boozer_surfaces=None``) path is gradient-dead over the coil dofs,
  * the NEW (adjoint) path is nonzero and finite-difference-consistent through the
    BoozerSurface solve, and
  * the signed value / violation are byte-identical between the two paths.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier
from simsopt.geo.surfaceobjectives import Volume

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.single_stage_constraints import (  # noqa: E402
    smooth_min_surface_stack_signed_constraint,
    smooth_min_surface_stack_signed_constraint_with_hard_signal,
)

# Soft-min temperature; small relative to the surface gaps so the soft-min tracks the
# hard min closely while staying differentiable.
TEMPERATURE = 0.01
# Active-constraint slack: a threshold this far above the hard minimum surface gap so
# the spacing constraint binds (positive violation), the regime the adjoint must
# actuate. Mirrors the production wall where the gap can sit below SURFACE_GAP_THRESHOLD.
ACTIVE_SLACK_M = 0.02


def _build_nested_boozer_stack():
    """Two nested BoozerSurfaces (Volume label) sharing the same NCSX coils.

    ``bs`` (BiotSavart) is the coil optimizable -- the faithful analog of the
    single-stage objective JF whose leaves are the coils. The surface Fourier dofs are
    NOT among ``bs``'s dofs, which is exactly why the fixed-surface derivative is
    gradient-dead over the coils.
    """
    base_curves, base_currents, ma = get_ncsx_data()
    coils = coils_via_symmetries(base_curves, base_currents, 3, True)
    bs = BiotSavart(coils)
    current_sum = sum(abs(c.current.get_value()) for c in coils)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))

    mpol = ntor = 6
    nfp = 3
    phis = np.linspace(0, 1 / nfp, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0, 1, 2 * mpol + 1, endpoint=False)

    def solve_at(radius):
        surface = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )
        surface.fit_to_curve(ma, radius, flip_theta=True)
        label = Volume(surface)
        boozer_surface = BoozerSurface(
            bs, surface, label, label.J(), constraint_weight=None,
            options={"weight_inv_modB": False},
        )
        boozer_surface.run_code(-0.406, G=G0)
        return boozer_surface

    inner = solve_at(0.10)
    outer = solve_at(0.16)
    return bs, (inner, outer)


class SurfaceStackSpacingGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bs, cls.boozer_surfaces = _build_nested_boozer_stack()
        cls.surfaces = tuple(b.surface for b in cls.boozer_surfaces)
        # Hard minimum gap between the two surfaces, used to place an active threshold.
        inner_pts = cls.surfaces[0].gamma().reshape(-1, 3)
        outer_pts = cls.surfaces[1].gamma().reshape(-1, 3)
        distances, _ = cKDTree(outer_pts).query(inner_pts, k=1)
        cls.hard_min_gap = float(distances.min())
        cls.min_distance = cls.hard_min_gap + ACTIVE_SLACK_M

    def _old_path(self):
        return smooth_min_surface_stack_signed_constraint(
            self.surfaces,
            self.min_distance,
            TEMPERATURE,
            self.bs,
        )

    def _new_path(self):
        return smooth_min_surface_stack_signed_constraint(
            self.surfaces,
            self.min_distance,
            TEMPERATURE,
            self.bs,
            boozer_surfaces=self.boozer_surfaces,
        )

    def test_old_fixed_surface_gradient_is_dead_over_coils(self):
        """The OLD path produced a zero coil gradient -- this is the bug.

        Documents the broken baseline: the gradient that should actuate the coils is
        identically zero, so the spacing constraint could not move the coils.
        """
        _signed, grad_old, _viol = self._old_path()
        self.assertEqual(
            grad_old.shape[0],
            self.bs.dof_size,
            "old-path gradient must be sized over the coil dofs",
        )
        self.assertEqual(
            np.linalg.norm(grad_old),
            0.0,
            "OLD fixed-surface path is gradient-dead over the coil dofs "
            "(the root-cause bug); a nonzero norm here would mean the fix is moot",
        )

    def test_new_gradient_is_nonzero_and_fd_consistent(self):
        """The fixed gradient is live over the coils and matches a finite difference.

        Fails against the old gradient-dead behavior (norm 0.0). The directional finite
        difference of the signed constraint value -- re-solving BOTH BoozerSurfaces at
        each perturbed coil state -- proves the gradient flows through the implicit
        surface solve, not just the fixed surface dofs.
        """
        _signed, grad_new, _viol = self._new_path()
        self.assertEqual(grad_new.shape[0], self.bs.dof_size)
        grad_norm = np.linalg.norm(grad_new)
        self.assertGreater(
            grad_norm,
            1e-6,
            "Boozer-adjoint spacing gradient must be nonzero over the coil dofs",
        )

        bs = self.bs
        inner, outer = self.boozer_surfaces
        x0 = bs.x.copy()
        inner_iota, inner_G = inner.res["iota"], inner.res["G"]
        outer_iota, outer_G = outer.res["iota"], outer.res["G"]
        direction = grad_new / grad_norm
        eps = 1e-5

        def signed_value_at(x):
            bs.x = x
            inner.run_code(inner_iota, G=inner_G)
            outer.run_code(outer_iota, G=outer_G)
            signed, _grad, _viol = smooth_min_surface_stack_signed_constraint(
                tuple(b.surface for b in self.boozer_surfaces),
                self.min_distance,
                TEMPERATURE,
                bs,
            )
            return signed

        signed_plus = signed_value_at(x0 + eps * direction)
        signed_minus = signed_value_at(x0 - eps * direction)
        bs.x = x0  # restore the solved state for sibling tests
        inner.run_code(inner_iota, G=inner_G)
        outer.run_code(outer_iota, G=outer_G)

        directional_slope = (signed_plus - signed_minus) / (2.0 * eps)
        # The directional derivative equals grad . (grad/||grad||) = ||grad||.
        np.testing.assert_allclose(directional_slope, grad_norm, rtol=1e-3)

    def test_signed_value_and_violation_unchanged(self):
        """Feasibility reporting is identical; only the gradient changed.

        The fix must not move the constraint boundary: signed value and violation from
        the adjoint path must match the fixed-surface path exactly (both equal
        ``minimum_distance - smooth_min``).
        """
        signed_old, _grad_old, viol_old = self._old_path()
        signed_new, _grad_new, viol_new = self._new_path()
        self.assertEqual(
            signed_new,
            signed_old,
            "signed value (min_distance - smooth_min) must be byte-identical",
        )
        self.assertEqual(
            viol_new,
            viol_old,
            "violation must be byte-identical to the old path",
        )
        self.assertGreater(
            viol_new, 0.0, "threshold chosen so the spacing constraint is active"
        )

    def test_hard_signal_wrapper_routes_adjoint_gradient(self):
        """The hard-signal wrapper forwards ``boozer_surfaces`` and stays live.

        The hard-signal variant is what production wires for dual diagnostics; it must
        produce the same live coil gradient and the same signed value, plus the extra
        hard signed value / hard violation tuple entries.
        """
        result = smooth_min_surface_stack_signed_constraint_with_hard_signal(
            self.surfaces,
            self.min_distance,
            TEMPERATURE,
            self.bs,
            boozer_surfaces=self.boozer_surfaces,
        )
        signed_value, grad, violation, hard_signed_value, hard_violation = result
        signed_new, grad_new, viol_new = self._new_path()
        self.assertEqual(signed_value, signed_new)
        self.assertEqual(violation, viol_new)
        np.testing.assert_array_equal(grad, grad_new)
        self.assertGreater(np.linalg.norm(grad), 1e-6)
        # Hard signal equals min_distance - hard_min_gap and is active here.
        self.assertAlmostEqual(
            hard_signed_value, self.min_distance - self.hard_min_gap, places=9
        )
        self.assertGreater(hard_violation, 0.0)

    def test_length_mismatch_is_rejected(self):
        """A boozer_surfaces / surfaces length mismatch is a hard error, not silent.

        Misaligned stacks would silently mis-attribute the adjoint to the wrong
        surface; the constructor rejects it instead.
        """
        with self.assertRaises(ValueError):
            smooth_min_surface_stack_signed_constraint(
                self.surfaces,
                self.min_distance,
                TEMPERATURE,
                self.bs,
                boozer_surfaces=self.boozer_surfaces[:1],
            )


if __name__ == "__main__":
    unittest.main()
