"""Regression tests for the opt-in ``--free-tf-geometry`` capability.

``--free-tf-geometry`` unfreezes the TF coil curve geometry so the optimizer can
shape the TF coils to supply rotational transform (decoupling banana poloidal
extent from iota), while keeping the freed TF coils buildable via curvature,
length, and clearance penalties. These tests pin three contracts:

1. Default-off is byte-identical: ``build_total_objective`` with the TF terms
   absent and with them explicitly ``None`` produce the same ``J`` and ``dJ``.
2. The TF buildability terms are real, finite, and gradient-bearing on the freed
   TF dofs, and reuse the same SIMSOPT penalty helpers/thresholds as the banana
   coils (SSOT).
3. The objective bundle threads ``JTFCurvature``/``JTFCurveLength`` only when TF
   curves are supplied; the default frozen-TF call passes ``None`` for both.
"""

import importlib.util
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simsopt.geo import CurveLength, CurveXYZFourier, LpCurveCurvature
from simsopt.geo.curveobjectives import CurveCurveDistance
from simsopt.objectives import QuadraticPenalty

import pytest


_OBJECTIVES_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "single_stage_objectives.py"
)
_EXAMPLE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)
_BANANA_OPT_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)


def _load_module(module_path, name_prefix):
    # The banana_opt package must be importable for the example module's absolute
    # ``from banana_opt...`` imports to resolve.
    banana_opt_dir = str(_BANANA_OPT_DIR)
    if banana_opt_dir not in sys.path:
        sys.path.insert(0, banana_opt_dir)
    spec = importlib.util.spec_from_file_location(
        f"{name_prefix}_{uuid.uuid4().hex}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _objectives_module():
    return _load_module(_OBJECTIVES_MODULE_PATH, "single_stage_objectives")


def _example_module():
    return _load_module(_EXAMPLE_MODULE_PATH, "single_stage_banana_example")


def _tf_like_curve(*, x_scale=1.0, y_scale=0.9, z_scale=0.2, order=1):
    """A nondegenerate TF-like loop whose length and curvature are appreciable."""
    curve = CurveXYZFourier(64, order)
    curve.set("xc(0)", 1.0)
    curve.set("xc(1)", x_scale)
    curve.set("ys(1)", y_scale)
    curve.set("zs(1)", z_scale)
    return curve


def _scalar_terms_for_total_objective(reference_curve):
    """The non-TF terms ``build_total_objective`` needs, as simple real objectives."""
    other = CurveXYZFourier(64, 2)
    other.set("xc(0)", 0.6)
    return {
        "JnonQSRatio": QuadraticPenalty(CurveLength(reference_curve), 0.0, "max"),
        "JBoozerResidual": QuadraticPenalty(CurveLength(reference_curve), 0.0, "max"),
        "Jiota": QuadraticPenalty(CurveLength(reference_curve), 0.15, "max"),
        "JVolume": QuadraticPenalty(CurveLength(reference_curve), 0.0, "max"),
        "JCurveLength": QuadraticPenalty(CurveLength(reference_curve), 1.9, "max"),
        "JCurveCurve": CurveCurveDistance([reference_curve, other], 0.05),
        "JCurveSurface": CurveCurveDistance([reference_curve, other], 0.01),
        "JCurvature": LpCurveCurvature(reference_curve, 4, 100.0),
    }


def _build_total(module, terms, **extra):
    return module.build_total_objective(
        terms["JnonQSRatio"],
        1.0,
        terms["JBoozerResidual"],
        1.0,
        terms["Jiota"],
        1.0,
        terms["JVolume"],
        1.0,
        terms["JCurveLength"],
        1.0,
        terms["JCurveCurve"],
        1.0,
        terms["JCurveSurface"],
        1.0,
        terms["JCurvature"],
        **extra,
    )


class TestBuildTotalObjectiveTFTermsDefaultOff:
    """Absent TF terms == explicit ``None`` TF terms, bit-for-bit."""

    def test_absent_tf_terms_match_explicit_none_J_and_dJ(self):
        module = _objectives_module()
        banana = CurveXYZFourier(64, 2)
        banana.x = banana.x + 0.01 * np.arange(len(banana.x))
        terms = _scalar_terms_for_total_objective(banana)

        objective_absent = _build_total(module, terms)
        objective_explicit_none = _build_total(
            module, terms, JTFCurvature=None, JTFCurveLength=None
        )

        assert float(objective_absent.J()) == float(objective_explicit_none.J())
        np.testing.assert_array_equal(
            objective_absent.dJ(), objective_explicit_none.dJ()
        )


class TestBuildTotalObjectiveTFTermsContribute:
    """When supplied, the TF terms add their weighted value and gradient."""

    def test_active_tf_terms_add_weighted_value_and_flow_gradient(self):
        module = _objectives_module()
        banana = CurveXYZFourier(64, 2)
        banana.x = banana.x + 0.01 * np.arange(len(banana.x))
        terms = _scalar_terms_for_total_objective(banana)

        tf = _tf_like_curve()
        # Thresholds set below the seed values so both buildability terms are active.
        j_tf_curvature = LpCurveCurvature(tf, 4, 0.5)
        j_tf_length = QuadraticPenalty(CurveLength(tf), 1.0, "max")
        assert float(j_tf_curvature.J()) > 0.0
        assert float(j_tf_length.J()) > 0.0

        objective_off = _build_total(module, terms)
        objective_on = _build_total(
            module,
            terms,
            JTFCurvature=j_tf_curvature,
            JTFCurveLength=j_tf_length,
        )

        # CURVATURE_WEIGHT and LENGTH_WEIGHT are both 1.0 in this harness, so the
        # delta is exactly the sum of the two TF buildability term values.
        expected_delta = float(j_tf_curvature.J()) + float(j_tf_length.J())
        assert np.isclose(
            float(objective_on.J()) - float(objective_off.J()), expected_delta
        )
        # The TF terms depend on the TF curve, so the freed TF dofs receive gradient.
        assert float(np.linalg.norm(j_tf_curvature.dJ())) > 0.0
        assert float(np.linalg.norm(j_tf_length.dJ())) > 0.0


class TestFreeTFCurveLengthPenaltyGradient:
    """The TF length ceiling penalty gradient matches finite differences."""

    def test_tf_length_penalty_gradient_matches_finite_difference(self):
        tf = _tf_like_curve()
        # Ceiling below the seed length -> the "max" penalty is active.
        length_ceiling = float(CurveLength(tf).J()) - 1.0
        assert length_ceiling > 0.0
        penalty = QuadraticPenalty(CurveLength(tf), length_ceiling, "max")

        x0 = tf.x.copy()
        analytic = np.asarray(penalty.dJ(), dtype=float)

        def penalty_value(x_vec):
            tf.x = x_vec
            value = float(penalty.J())
            tf.x = x0
            return value

        step = 1e-7
        worst_relative_error = 0.0
        for index in range(len(x0)):
            x_plus = x0.copy()
            x_plus[index] += step
            x_minus = x0.copy()
            x_minus[index] -= step
            finite_difference = (penalty_value(x_plus) - penalty_value(x_minus)) / (
                2.0 * step
            )
            if abs(analytic[index]) > 1e-6 or abs(finite_difference) > 1e-6:
                relative_error = abs(finite_difference - analytic[index]) / (
                    abs(analytic[index]) + 1e-12
                )
                worst_relative_error = max(worst_relative_error, relative_error)
        assert worst_relative_error < 1e-6


class TestUnfreezingTFCurvesAddsDofs:
    """Unfreezing TF curve geometry frees their Fourier dofs; currents untouched."""

    def test_unfix_all_on_tf_curves_frees_curve_dofs_only(self):
        from simsopt.field import Coil, Current

        tf_curves = [_tf_like_curve(x_scale=0.9 + 0.01 * index) for index in range(20)]
        for curve in tf_curves:
            curve.fix_all()
        tf_currents = [Current(-80000.0) for _ in tf_curves]
        for current in tf_currents:
            current.fix_all()
        tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

        free_before = sum(coil.curve.dof_size for coil in tf_coils)
        assert free_before == 0

        for coil in tf_coils:
            coil.curve.unfix_all()

        free_after = sum(coil.curve.dof_size for coil in tf_coils)
        # 20 order-1 CurveXYZFourier curves: 3*(2*1+1) = 9 dofs each => +180.
        assert free_after - free_before == 20 * 9
        # TF currents stay frozen: the change unfreezes geometry only.
        assert all(coil.current.dof_size == 0 for coil in tf_coils)


class TestObjectiveBundleThreadsTFTerms:
    """The bundle builds + threads TF terms only when TF curves are supplied."""

    def _patched_bundle(self, stack, module):
        stack.enter_context(
            patch.object(
                module,
                "build_boozer_derived_objective_terms",
                return_value={
                    "surface_iota_terms": ["iota"],
                    "nonQSs": ["nonqs"],
                    "brs": ["br"],
                    "boozer_objective_biot_savarts": ["bs"],
                },
            )
        )
        for name in (
            "Volume",
            "build_single_stage_iota_objective",
            "build_single_stage_volume_objective",
            "average_surface_objectives",
            "QuadraticPenalty",
            "CurveCurveDistance",
            "CurveSurfaceDistance",
            "PoloidalExtent",
            "EllipseWidth",
            "MajorRadius",
            "MinorRadius",
            "CurveSelfIntersect",
        ):
            stack.enter_context(patch.object(module, name, return_value=object()))
        stack.enter_context(
            patch.object(
                module,
                "resolve_single_stage_goal_objective_terms",
                return_value={
                    "JnonQSRatioObjective": object(),
                    "effective_res_weight": 1.0,
                    "JBoozerResidualObjective": object(),
                    "effective_iotas_weight": 1.0,
                    "effective_volume_weight": 1.0,
                },
            )
        )

    def test_default_frozen_tf_passes_none_tf_terms(self):
        module = _example_module()
        banana_curve = SimpleNamespace(order=8)

        with ExitStack() as stack:
            self._patched_bundle(stack, module)
            stack.enter_context(
                patch.object(module, "CurveLength", return_value=object())
            )
            stack.enter_context(
                patch.object(module, "LpCurveCurvature", return_value=object())
            )
            build_total_mock = stack.enter_context(
                patch.object(module, "build_total_objective", return_value=object())
            )

            module.build_single_stage_objective_bundle(
                stage="full",
                surface_data=[
                    {"boozer_surface": SimpleNamespace(surface=object())}
                ],
                coils=["coil"],
                curves=["banana_curve"],
                banana_curves=[banana_curve],
                iota_target=0.2,
                RES_WEIGHT=1.0,
                IOTAS_WEIGHT=1.0,
                LENGTH_WEIGHT=1.0,
                CC_WEIGHT=1.0,
                CC_DIST=0.05,
                CS_WEIGHT=1.0,
                CS_DIST=0.015,
                CURVATURE_WEIGHT=1.0,
                CURVATURE_THRESHOLD=100.0,
            )

        assert build_total_mock.call_args.kwargs["JTFCurvature"] is None
        assert build_total_mock.call_args.kwargs["JTFCurveLength"] is None

    def test_free_tf_builds_tf_terms_with_shared_helpers_and_thresholds(self):
        module = _example_module()
        banana_curve = SimpleNamespace(order=8)
        tf_curves = [SimpleNamespace(name=f"tf{index}") for index in range(3)]
        recorded_curvature_calls = []
        recorded_length_curve_calls = []
        recorded_length_penalty_calls = []

        class _SummableTerm:
            # The bundle folds per-TF-curve terms with ``sum(terms[1:], terms[0])``;
            # this stand-in collapses that fold to a single tagged sentinel so we can
            # assert which fold (curvature vs length) reached build_total_objective.
            def __init__(self, tag):
                self.tag = tag

            def __add__(self, other):
                return self

            def __radd__(self, other):
                return self

        tf_curve_ids = {id(tf_curve) for tf_curve in tf_curves}
        tf_curvature_sentinel = _SummableTerm("tf_curvature")
        tf_length_penalty_sentinel = _SummableTerm("tf_length")

        class _LengthOf:
            # A CurveLength stand-in tagged with the curve it wraps, so the length
            # penalty recorder can tell banana-length from TF-length construction.
            def __init__(self, curve):
                self.curve = curve

        def fake_lp_curve_curvature(curve, p_norm, threshold):
            recorded_curvature_calls.append((curve, p_norm, threshold))
            return tf_curvature_sentinel

        def fake_curve_length(curve):
            recorded_length_curve_calls.append(curve)
            return _LengthOf(curve)

        def fake_quadratic_penalty(objective, threshold, mode):
            recorded_length_penalty_calls.append((objective, threshold, mode))
            wrapped_curve = getattr(objective, "curve", None)
            if wrapped_curve is not None and id(wrapped_curve) in tf_curve_ids:
                return tf_length_penalty_sentinel
            return object()

        with ExitStack() as stack:
            self._patched_bundle(stack, module)
            # Re-patch QuadraticPenalty/CurveLength with recorders (the generic
            # patch in _patched_bundle already covers QuadraticPenalty; override it
            # here so we can capture the TF-length construction specifically).
            stack.enter_context(
                patch.object(module, "QuadraticPenalty", fake_quadratic_penalty)
            )
            stack.enter_context(patch.object(module, "CurveLength", fake_curve_length))
            stack.enter_context(
                patch.object(module, "LpCurveCurvature", fake_lp_curve_curvature)
            )
            build_total_mock = stack.enter_context(
                patch.object(module, "build_total_objective", return_value=object())
            )

            module.build_single_stage_objective_bundle(
                stage="full",
                surface_data=[
                    {"boozer_surface": SimpleNamespace(surface=object())}
                ],
                coils=["coil"],
                curves=["banana_curve"] + tf_curves,
                banana_curves=[banana_curve],
                iota_target=0.2,
                RES_WEIGHT=1.0,
                IOTAS_WEIGHT=1.0,
                LENGTH_WEIGHT=1.0,
                CC_WEIGHT=1.0,
                CC_DIST=0.05,
                CS_WEIGHT=1.0,
                CS_DIST=0.015,
                CURVATURE_WEIGHT=1.0,
                CURVATURE_THRESHOLD=100.0,
                tf_curves=tf_curves,
                tf_length_target=2.5,
            )

        # The bundle passes non-None TF terms to the total objective.
        assert build_total_mock.call_args.kwargs["JTFCurvature"] is tf_curvature_sentinel
        assert (
            build_total_mock.call_args.kwargs["JTFCurveLength"]
            is tf_length_penalty_sentinel
        )
        # TF curvature reuses LpCurveCurvature on each TF curve at the shared cap.
        tf_curvature_calls = [
            call
            for call in recorded_curvature_calls
            if any(call[0] is tf_curve for tf_curve in tf_curves)
        ]
        curvature_curves = [call[0] for call in tf_curvature_calls]
        assert all(
            any(curve is tf_curve for curve in curvature_curves)
            for tf_curve in tf_curves
        )
        assert len(tf_curvature_calls) == len(tf_curves)
        assert all(call[1] == module.CURVATURE_P_NORM for call in tf_curvature_calls)
        assert all(call[2] == 100.0 for call in tf_curvature_calls)
        # TF length reuses CurveLength on each TF curve, capped at the TF ceiling.
        # (CurveLength is also called once for the banana coil length term, so the
        # TF curves are a subset of the recorded CurveLength calls, not all of them.)
        length_curves = list(recorded_length_curve_calls)
        assert all(
            any(curve is tf_curve for curve in length_curves) for tf_curve in tf_curves
        )
        tf_length_penalty_calls = [
            call
            for call in recorded_length_penalty_calls
            if id(getattr(call[0], "curve", None)) in tf_curve_ids
        ]
        assert len(tf_length_penalty_calls) == len(tf_curves)
        assert all(call[1] == 2.5 for call in tf_length_penalty_calls)
        assert all(call[2] == "max" for call in tf_length_penalty_calls)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
