"""Regression tests for the single-stage LCFS major-radius ALM constraint gradient.

Root cause (banana_opt/single_stage_objectives.py): the LCFS upper-bound constraint
on major radius was built with ``_surface_upper_bound_constraint``, which projects
``surface.dmajor_radius_by_dcoeff()`` (a derivative living on the FIXED LCFS Boozer
surface dofs) onto the coil objective. Because the LCFS surface dofs are not in the
coil dof graph, that projection is identically zero -> the ALM constraint could not
actuate the coils and major radius stayed stuck above the threshold. The fix routes
the gradient through simsopt's vetted ``MajorRadius`` Boozer adjoint via
``_boozer_adjoint_upper_bound_constraint``.

These tests construct a real solved BoozerSurface + coils (one Newton solve, no
optimization) and assert observable behavior: the new gradient is nonzero over the
coil dofs and points in the direction that REDUCES major radius, while the signed
value / violation are byte-identical to the old fixed-surface path.
"""

import importlib.util
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

from simsopt.geo.surfaceobjectives import MajorRadius

from .surface_test_helpers import get_boozer_surface

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.single_stage_objectives import (  # noqa: E402
    _boozer_adjoint_upper_bound_constraint,
    _surface_upper_bound_constraint,
)

EXAMPLE_MODULE_PATH = EXAMPLES_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"


# A threshold safely below the NCSX test surface major radius (~1.47 m) so the
# constraint is active (positive violation), mirroring the production wall where
# major radius sits above TARGET_LCFS_MAX_MAJOR_RADIUS_M.
ACTIVE_THRESHOLD_M = 0.903


class LCFSMajorRadiusGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One real Boozer solve shared across tests; `bs` (BiotSavart) is the coil
        # optimizable, the faithful analog of the single-stage objective JF whose
        # leaves are the coils. The LCFS surface dofs are NOT among bs's dofs.
        cls.bs, cls.boozer_surface = get_boozer_surface(
            label="Volume", boozer_type="exact"
        )
        cls.surface = cls.boozer_surface.surface

    def _old_path(self):
        surface = self.surface
        return _surface_upper_bound_constraint(
            surface,
            surface.major_radius(),
            surface.dmajor_radius_by_dcoeff(),
            ACTIVE_THRESHOLD_M,
            self.bs,
        )

    def _new_path(self):
        return _boozer_adjoint_upper_bound_constraint(
            MajorRadius(self.boozer_surface),
            ACTIVE_THRESHOLD_M,
            self.bs,
        )

    def test_old_fixed_surface_gradient_is_dead_over_coils(self):
        """The OLD path produced a zero coil gradient -- this is the bug.

        Documents the broken baseline the regression tests below would fail
        against: the gradient that should actuate the coils is identically zero.
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

    def test_new_gradient_is_nonzero_and_reduces_major_radius(self):
        """The fixed gradient actuates the coils in the direction that lowers R.

        Fails against the old gradient-dead behavior (norm 0.0, no descent
        direction). The finite-difference check proves the sign: stepping the
        coils along +grad raises major_radius, so an upper-bound penalty with
        that gradient pushes the coils to reduce it.
        """
        _signed, grad_new, _viol = self._new_path()

        self.assertEqual(grad_new.shape[0], self.bs.dof_size)
        grad_norm = np.linalg.norm(grad_new)
        self.assertGreater(
            grad_norm,
            1e-6,
            "Boozer-adjoint gradient must be nonzero over the coil dofs",
        )

        # Directional finite difference of major_radius along +grad_new.
        bs = self.bs
        booz = self.boozer_surface
        x0 = bs.x.copy()
        iota0 = booz.res["iota"]
        G0 = booz.res["G"]
        direction = grad_new / grad_norm
        eps = 1e-5

        def major_radius_at(x):
            bs.x = x
            booz.run_code(iota0, G=G0)
            return MajorRadius(booz).J()

        r_plus = major_radius_at(x0 + eps * direction)
        r_minus = major_radius_at(x0 - eps * direction)
        bs.x = x0  # restore
        booz.run_code(iota0, G=G0)

        directional_slope = (r_plus - r_minus) / (2.0 * eps)
        self.assertGreater(
            directional_slope,
            0.0,
            "stepping coils along +grad must INCREASE major_radius, so the "
            "upper-bound penalty gradient correctly drives R downward",
        )
        # The directional derivative equals the gradient norm (grad . grad/||grad||).
        np.testing.assert_allclose(directional_slope, grad_norm, rtol=2e-3)

    def test_signed_value_and_violation_unchanged(self):
        """Feasibility reporting is identical; only the gradient changed.

        The fix must not move the constraint boundary: signed value and violation
        from the new adjoint path must match the old fixed-surface path exactly
        (both equal major_radius - threshold).
        """
        signed_old, _grad_old, viol_old = self._old_path()
        signed_new, _grad_new, viol_new = self._new_path()

        self.assertEqual(
            signed_new,
            signed_old,
            "signed value (major_radius - threshold) must be byte-identical",
        )
        self.assertEqual(
            viol_new,
            viol_old,
            "violation must be byte-identical to the old path",
        )
        # And it must actually equal major_radius - threshold (active here).
        self.assertAlmostEqual(
            signed_new, self.surface.major_radius() - ACTIVE_THRESHOLD_M, places=12
        )
        self.assertGreater(viol_new, 0.0, "threshold chosen so constraint is active")

    def test_inactive_constraint_signed_value_matches(self):
        """With a threshold above major_radius the violation is zero on both paths.

        Guards the inactive branch: when the constraint is not binding, the signed
        value is negative and the reported violation is zero, identically for both
        the old and new gradient paths.
        """
        slack_threshold = self.surface.major_radius() + 1.0
        surface = self.surface
        signed_old, _g, viol_old = _surface_upper_bound_constraint(
            surface,
            surface.major_radius(),
            surface.dmajor_radius_by_dcoeff(),
            slack_threshold,
            self.bs,
        )
        signed_new, _g2, viol_new = _boozer_adjoint_upper_bound_constraint(
            MajorRadius(self.boozer_surface),
            slack_threshold,
            self.bs,
        )
        self.assertEqual(signed_new, signed_old)
        self.assertEqual(viol_old, 0.0)
        self.assertEqual(viol_new, 0.0)


class LCFSConstraintNameContractTests(unittest.TestCase):
    """The constraint-name contract must be untouched by the gradient fix.

    The fix only changed how ``hardware_constraints["lcfs_major_radius"]`` is
    computed inside ``evaluate_alm_objective`` (plus an opt-in objective kwarg);
    it must not add, remove, or reorder the registered ALM constraint names.
    """

    @classmethod
    def setUpClass(cls):
        # The example module registers Optimizable subclasses that resolve their
        # own module via sys.modules, so it must be registered under a stable name.
        module_name = "single_stage_banana_example"
        spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        original_sys_path = sys.path.copy()
        sys.path.insert(0, str(EXAMPLES_ROOT))
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_sys_path
        cls.module = module

    def test_default_weighted_sum_constraint_names_unchanged(self):
        names = self.module.single_stage_alm_constraint_names(
            alm_formulation="weighted_sum",
        )
        self.assertEqual(
            names,
            [
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "coil_length_min",
                "poloidal_extent",
                "width_min",
                "width_max",
                "self_intersect",
                "banana_current_upper_bound",
                "lcfs_major_radius",
                "lcfs_minor_radius",
            ],
        )
        # The fix must keep both LCFS constraints registered (the gradient route
        # changed, not the contract).
        self.assertIn("lcfs_major_radius", names)
        self.assertIn("lcfs_minor_radius", names)


if __name__ == "__main__":
    unittest.main()
