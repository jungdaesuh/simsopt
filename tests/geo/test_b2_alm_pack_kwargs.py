"""Behavioral regression tests for two B2 finite-build review findings.

F1 (blocker): the example forwards the four B2 pack-rotation kwargs
(``JPackRotationFold`` / ``PACK_ROTATION_FOLD_WEIGHT`` / ``JPackTwistStrain`` /
``PACK_TWIST_STRAIN_WEIGHT``) into the MODULE ``evaluate_alm_objective``
unconditionally. Before the fix the module signature did not accept them, so
*every* ``--constraint-method alm`` run -- including the default non-finite-build
ALM run, which passes them as None/0.0 -- crashed with ``TypeError: got an
unexpected keyword argument 'JPackRotationFold'``. These tests drive the real
module function (mirroring the lightweight harness in
``test_banana_objective_modules.py``) and assert observable behavior: no
TypeError, the fold/twist terms enter the returned objective value when their
weights are positive, and they do NOT (byte-identical total) when None/0.0.

F2: ``coil_order_upgrade._resolve_master_banana_seed`` must return the NET
banana current for a finite-build pack seed (the per-filament ``ScaledCurrent``'s
``current_to_scale``), and the unchanged current for a thin seed.
"""

import importlib.util
import inspect
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
SINGLE_STAGE_OBJECTIVES_PATH = (
    EXAMPLES_ROOT / "banana_opt" / "single_stage_objectives.py"
)
COIL_ORDER_UPGRADE_PATH = EXAMPLES_ROOT / "banana_opt" / "coil_order_upgrade.py"


def _load_module(module_path: Path, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    original_sys_path = sys.path.copy()
    sys.path.insert(0, str(EXAMPLES_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


import unittest  # noqa: E402

from simsopt.field import Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import CurveCWSFourierCPP  # noqa: E402

sys.path.insert(0, str(EXAMPLES_ROOT))
from banana_opt.reference_surfaces import (  # noqa: E402
    build_banana_reference_surfaces,
)
from banana_opt.stage2_geometry import (  # noqa: E402
    FiniteBuildSettings,
    build_finite_build_banana_coils,
)

del sys.path[0]

NFP = 5
NET_BANANA_CURRENT_A = -16000.0
BANANA_SURF_RADIUS_M = 0.21


class _FakeAlgebraicObjective:
    """Minimal Optimizable stand-in: scalar value + gradient under +/* algebra.

    Mirrors the fake in ``test_banana_objective_modules.py`` so the module
    objective builders (``A + w * B``) accumulate value and gradient exactly as
    they do for real terms, letting a unit-light call exercise the genuine
    ``evaluate_alm_objective`` -> ``evaluate_base_objective`` summation.
    """

    def __init__(self, value, gradient, projected_gradient=None):
        self._value = float(value)
        self._gradient = np.asarray(gradient, dtype=float)
        projected = gradient if projected_gradient is None else projected_gradient
        self._projected_gradient = np.asarray(projected, dtype=float)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if partials:
            return lambda _objective: self._projected_gradient.copy()
        return self._gradient.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return _FakeAlgebraicObjective(
            self._value + other._value,
            self._gradient + other._gradient,
            self._projected_gradient + other._projected_gradient,
        )

    __radd__ = __add__

    def __mul__(self, scalar):
        return _FakeAlgebraicObjective(
            self._value * scalar,
            self._gradient * scalar,
            self._projected_gradient * scalar,
        )

    __rmul__ = __mul__


def _zero_constraint_result_2d(*_args, **_kwargs):
    return -0.1, np.array([0.0, 0.0]), 0.0


class EvaluateAlmObjectivePackKwargsTest(unittest.TestCase):
    """F1: the module ``evaluate_alm_objective`` binds and applies the 4 B2 kwargs."""

    def setUp(self):
        self.module = _load_module(
            SINGLE_STAGE_OBJECTIVES_PATH, "b2_single_stage_objectives"
        )

    def _evaluate(self, **pack_kwargs):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        return self.module.evaluate_alm_objective(
            np.array([1.0]),
            [zero],
            [zero],
            RES_WEIGHT=0.0,
            Jiota=zero,
            IOTAS_WEIGHT=0.0,
            JVolume=None,
            VOLUME_WEIGHT=0.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=0.0,
            JCurveCurve=zero,
            JCurveSurface=zero,
            JCurvature=zero,
            multipliers=np.array([0.0, 0.0, 0.0]),
            penalty=1.0,
            objective_optimizable=SimpleNamespace(),
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=("coil_coil_spacing", "coil_surface_spacing", "max_curvature"),
            curve_curve_constraint_fn=_zero_constraint_result_2d,
            curve_surface_constraint_fn=_zero_constraint_result_2d,
            curvature_constraint_fn=_zero_constraint_result_2d,
            activity_tolerances_fn=lambda *_a, **_k: np.zeros(3),
            **pack_kwargs,
        )

    def test_signature_binds_the_four_b2_kwargs(self):
        # The original crash was a TypeError at bind time; assert the contract
        # directly so a future signature regression fails loudly here too.
        signature = inspect.signature(self.module.evaluate_alm_objective)
        signature.bind_partial(
            JPackRotationFold=None,
            PACK_ROTATION_FOLD_WEIGHT=0.0,
            JPackTwistStrain=None,
            PACK_TWIST_STRAIN_WEIGHT=0.0,
        )

    def test_default_alm_path_accepts_none_zero_kwargs_without_typeerror(self):
        # The regression: the DEFAULT non-finite-build ALM run passes the 4 kwargs
        # as None/0.0. This must run and add nothing to the objective.
        baseline = self._evaluate()
        with_none = self._evaluate(
            JPackRotationFold=None,
            PACK_ROTATION_FOLD_WEIGHT=0.0,
            JPackTwistStrain=None,
            PACK_TWIST_STRAIN_WEIGHT=0.0,
        )
        self.assertEqual(with_none["total"], baseline["total"])
        np.testing.assert_array_equal(with_none["grad"], baseline["grad"])

    def test_positive_weights_add_fold_and_twist_terms_to_objective(self):
        fold = _FakeAlgebraicObjective(2.0, [1.0, 0.0])
        twist = _FakeAlgebraicObjective(3.0, [0.0, 1.0])
        baseline = self._evaluate()
        steered = self._evaluate(
            JPackRotationFold=fold,
            PACK_ROTATION_FOLD_WEIGHT=5.0,
            JPackTwistStrain=twist,
            PACK_TWIST_STRAIN_WEIGHT=7.0,
        )
        # weighted_sum ALM total = base physics terms; fold/twist enter as
        # 5*2 + 7*3 = 31 over the baseline, and their gradients as 5*[1,0]+7*[0,1].
        self.assertAlmostEqual(
            steered["total"] - baseline["total"], 5.0 * 2.0 + 7.0 * 3.0
        )
        np.testing.assert_allclose(
            steered["grad"] - baseline["grad"], [5.0, 7.0]
        )

    def test_terms_absent_when_objects_supplied_but_weights_zero(self):
        # Objects present but zero-weight must not perturb the objective, so the
        # finite-build levers are opt-in (default OFF is byte-identical).
        fold = _FakeAlgebraicObjective(2.0, [1.0, 0.0])
        twist = _FakeAlgebraicObjective(3.0, [0.0, 1.0])
        baseline = self._evaluate()
        zero_weight = self._evaluate(
            JPackRotationFold=fold,
            PACK_ROTATION_FOLD_WEIGHT=0.0,
            JPackTwistStrain=twist,
            PACK_TWIST_STRAIN_WEIGHT=0.0,
        )
        self.assertEqual(zero_weight["total"], baseline["total"])
        np.testing.assert_array_equal(zero_weight["grad"], baseline["grad"])


class ResolveMasterBananaSeedNetCurrentTest(unittest.TestCase):
    """F2: ``_resolve_master_banana_seed`` recovers the NET current per lineage."""

    def setUp(self):
        self.module = _load_module(COIL_ORDER_UPGRADE_PATH, "b2_coil_order_upgrade")
        self.surf_coils = build_banana_reference_surfaces(
            NFP, BANANA_SURF_RADIUS_M
        ).coil_winding_surface
        self.master = CurveCWSFourierCPP(
            np.linspace(0.0, 1.0, 96, endpoint=False),
            order=4,
            surf=self.surf_coils,
        )
        self.master.set("phic(0)", 0.0)
        self.master.set("thetac(0)", 0.0)
        self.master.set("phic(1)", 0.05)
        self.master.set("thetas(1)", 0.5)

    def _settings(self, rotation_order=1):
        return FiniteBuildSettings(
            numfilaments_n=2,
            numfilaments_b=7,
            gapsize_n=0.0099,
            gapsize_b=0.0066,
            rotation_order=rotation_order,
            frame="surface_tangent",
        )

    def test_thin_seed_returns_unchanged_net_current(self):
        thin_coils = coils_via_symmetries(
            [self.master],
            [Current(NET_BANANA_CURRENT_A)],
            self.surf_coils.nfp,
            self.surf_coils.stellsym,
        )
        master_curve, current = self.module._resolve_master_banana_seed(thin_coils)
        # Thin lineage: coil.current IS the net -- returned identity, unchanged.
        self.assertIs(master_curve, self.master)
        self.assertIs(current, thin_coils[0].current)
        self.assertAlmostEqual(float(current.get_value()), NET_BANANA_CURRENT_A)

    def test_finite_build_pack_returns_net_not_filament_scale(self):
        pack_coils = list(
            build_finite_build_banana_coils(
                self.master,
                NET_BANANA_CURRENT_A,
                self._settings(),
                self.surf_coils,
            )
        )
        # Sanity: the per-filament current the OLD code returned is net/nfilaments.
        nfilaments = self._settings().nfilaments
        self.assertAlmostEqual(
            float(pack_coils[0].current.get_value()),
            NET_BANANA_CURRENT_A / nfilaments,
        )

        master_curve, current = self.module._resolve_master_banana_seed(pack_coils)
        # The recovered current must be the NET (net/nfilaments * nfilaments), i.e.
        # the shared ``current_to_scale`` optimizable -- NOT the per-filament scale.
        self.assertIsInstance(master_curve, CurveCWSFourierCPP)
        self.assertAlmostEqual(float(current.get_value()), NET_BANANA_CURRENT_A)
        self.assertIs(current, pack_coils[0].current.current_to_scale)


if __name__ == "__main__":
    unittest.main()
