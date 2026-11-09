"""JaxBiotSavart reuses its coil state per DOF state and never serves a stale one.

The adapter builds grouped coil geometry once per coil-DOF state and caches
field values per state and point set. Every mutation simsopt notifies through
its recompute mechanism (DOF setters, fix/unfix, resample, non-DOF parameters)
and every point or backend change must retire that state; native BiotSavart
is the oracle. The module also checks the host transfers of the owner
projection, ownership of caller buffers, copies, serialization, and the native
field arithmetic the adapter rejects.
"""

from unittest_jax_support import JaxTestCase, aligned_host_copy, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend import set_backend
    from simsopt_jax_adapters.field import biotsavart_backend as backend

import copy
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

import numpy as np

from simsopt import load
from simsopt._core.derivative import Derivative
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.field.magneticfield import MagneticFieldMultiply, MagneticFieldSum
from simsopt.geo import RotatedCurve, create_equally_spaced_curves
from simsopt.geo.curveperturbed import (
    CurvePerturbed,
    GaussianSampler,
    PerturbationSample,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import FramedCurveCentroid, FrameRotation


_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1], [0.9, 0.3, -0.05]])
_OTHER_POINTS = np.array([[1.2, 0.05, 0.1], [0.7, -0.3, 0.0]])
_COTANGENT = np.arange(9, dtype=float).reshape(3, 3) / 11


# Two points and a B cotangent at them, for the single-point-set checks.
_TWO_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]])
_TWO_POINT_COTANGENT = np.linspace(-0.7, 0.9, 6).reshape(2, 3)


def _base_curves(count=2):
    return create_equally_spaced_curves(
        count, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=32
    )


def _fields(coils, points=_POINTS):
    field, native = backend.JaxBiotSavart(coils), BiotSavart(coils)
    field.set_points(points)
    native.set_points(points)
    return field, native


def _assert_field_matches_native(field, native):
    """B, dB/dX and the B VJP equal native BiotSavart's at the live state."""
    np.testing.assert_allclose(
        np.asarray(field.B()), native.B(), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(field.dB_by_dX()), native.dB_by_dX(), rtol=1e-12, atol=1e-14
    )
    cotangent = _COTANGENT[: native.B().shape[0]]
    expected = cast(Derivative, native.B_vjp(cotangent))
    np.testing.assert_allclose(
        cast(np.ndarray, field.B_vjp(cotangent)(field)),
        cast(np.ndarray, expected(native)),
        rtol=1e-11,
        atol=1e-13,
    )


def _count_coil_state_builds(patches):
    return patches.enter_context(
        mock.patch.object(
            backend,
            "_jitted_coil_set_spec_from_extraction_spec",
            wraps=backend._jitted_coil_set_spec_from_extraction_spec,
        )
    )


def _coils(*, perturbed=False):
    curves = _base_curves()
    if perturbed:
        sampler = GaussianSampler(curves[0].quadpoints, 1e-3, 0.2, n_derivs=1)
        curves = [
            CurvePerturbed(
                curves[0],
                PerturbationSample(sampler, randomgen=np.random.default_rng(713)),
            )
        ]
    currents = [Current(1e5) for _curve in curves]
    return curves, coils_via_symmetries(curves, currents, 2, True)


class _CountedSample(np.ndarray):
    copies = []

    def tobytes(self, order="C"):
        self.copies.append(self.nbytes)
        return super().tobytes(order)

    def copy(self, order="C"):
        self.copies.append(self.nbytes)
        return super().copy(order)


class TestBiotSavartJaxStateCache(JaxTestCase):
    """Coil-state and field-output reuse, and invalidation on every notified change."""

    def test_evaluations_at_one_dof_state_build_coil_state_once(self):
        """Unchanged evaluations reuse coil state and field outputs, rebuilding once
        after a free-DOF change."""
        patches = self.patches
        curves = _base_curves()
        field, native = _fields(
            coils_via_symmetries(curves, [Current(1e5) for _ in curves], 2, True)
        )
        builds = _count_coil_state_builds(patches)
        B_evaluations = patches.enter_context(
            mock.patch.object(
                backend,
                "grouped_biot_savart_B_from_spec",
                wraps=backend.grouped_biot_savart_B_from_spec,
            )
        )

        first = field.B()
        for _ in range(3):
            self.assertTrue(field.B() is first, "field.B() is first")
        field.dB_by_dX()
        field.B_vjp(_COTANGENT)(field)
        field.dB_by_dcoilcurrents()
        field.coil_set_spec()
        self.assertTrue(
            (builds.call_count, B_evaluations.call_count) == (1, 1),
            "repeated evaluations at an unchanged coil-DOF state rebuilt the coil "
            f"spec {builds.call_count} times and B {B_evaluations.call_count} times",
        )

        field.x = np.asarray(field.x) + 1e-3
        _assert_field_matches_native(field, native)
        self.assertTrue(builds.call_count == 2, "builds.call_count == 2")

    def test_per_current_outputs_reuse_arrays_and_follow_every_state_change(self):
        """Per-current outputs return independent lists, reuse array identities and
        refresh after every tested state change."""
        for method_name, kernel in [
            ("dB_by_dcoilcurrents", backend.biot_savart_B),
            ("d2B_by_dXdcoilcurrents", backend.biot_savart_dB_by_dX),
            ("d3B_by_dXdXdcoilcurrents", backend.biot_savart_d2B_by_dXdX),
            ("dA_by_dcoilcurrents", backend.biot_savart_A),
            ("d2A_by_dXdcoilcurrents", backend.biot_savart_dA_by_dX),
            ("d3A_by_dXdXdcoilcurrents", backend.biot_savart_d2A_by_dXdX),
        ]:
            with self.subTest(method_name=method_name, kernel=kernel), self.case():
                self._case_per_current_outputs_reuse_arrays_and_follow_every_state_change(
                    method_name, kernel
                )

    def _case_per_current_outputs_reuse_arrays_and_follow_every_state_change(
        self, method_name, kernel
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        curves = _base_curves()
        currents = [Current(1e5), Current(-3e4)]
        field, _native = _fields(
            [Coil(curve, current) for curve, current in zip(curves, currents)]
        )
        evaluate = getattr(field, method_name)
        previous = evaluate()
        repeated = evaluate()
        self.assertTrue(repeated is not previous, "repeated is not previous")
        self.assertTrue(
            all(a is b for a, b in zip(previous, repeated, strict=True)),
            "all(a is b for a, b in zip(previous, repeated, strict=True))",
        )
        repeated.clear()
        self.assertTrue(
            len(evaluate()) == len(field.coils),
            "caller list edits changed cached outputs",
        )

        for mutation in (
            "free",
            "full",
            "parent",
            "current",
            "layout",
            "fixed",
            "points",
            "backend",
        ):
            if mutation == "free":
                field.x = np.asarray(field.x) * 1.01
            elif mutation == "full":
                field.full_x = np.asarray(field.full_x) * 0.99
            elif mutation == "parent":
                curves[1].x = curves[1].x + 2e-3
            elif mutation == "current":
                currents[0].x = np.array([2.5e5])
            elif mutation == "layout":
                curves[0].fix(0)
            elif mutation == "fixed":
                curves[0].set(0, curves[0].get(0) + 5e-3)
            elif mutation == "points":
                field.set_points(_OTHER_POINTS)
            else:
                set_backend("native")
            actual = evaluate()
            expected = backend._per_coil_unit_field(
                field.get_points_cart_ref(), field.coil_set_spec(), kernel
            )
            self.assertTrue(
                all(a is not b for a, b in zip(actual, previous, strict=True)),
                mutation,
            )
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(
                    np.asarray(a), np.asarray(b), err_msg=mutation
                )
            self.assertTrue(
                all(a is b for a, b in zip(actual, evaluate(), strict=True)),
                mutation,
            )
            previous = actual

    def test_point_changes_release_cached_outputs_and_preserve_coil_geometry(self):
        """Every point setter releases obsolete field arrays while retaining cached coil
        geometry."""
        for setter in ["cart", "jax", "cyl", "spec", "clear"]:
            with self.subTest(setter=setter), self.case():
                self._case_point_changes_release_cached_outputs_and_preserve_coil_geometry(
                    setter
                )

    def _case_point_changes_release_cached_outputs_and_preserve_coil_geometry(
        self, setter
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        field, _native = _fields(
            [Coil(curve, Current(1e5)) for curve in _base_curves()]
        )
        spec = field.coil_set_spec()
        field.B()
        field.dB_by_dX()
        field.B_and_dB()
        field.dB_by_dcoilcurrents()
        references = [
            weakref.ref(array)
            for array in jax.tree_util.tree_leaves(tuple(field._field_outputs.values()))
        ]

        if setter == "cart":
            field.set_points(_OTHER_POINTS)
        elif setter == "jax":
            field.set_points(jnp.asarray(_OTHER_POINTS))
        elif setter == "cyl":
            field.set_points_cyl(np.array([[1.0, 0.3, 0.1], [0.9, -0.2, -0.1]]))
        elif setter == "spec":
            field.set_points_from_spec(
                backend.make_field_eval_spec(jnp.asarray(_OTHER_POINTS))
            )
        else:
            field.clear_points()

        self.assertTrue(
            all(reference() is None for reference in references),
            "obsolete outputs remain retained",
        )
        self.assertTrue(field.coil_set_spec() is spec, "field.coil_set_spec() is spec")

    def test_field_and_parent_dof_changes_refresh_the_field(self):
        """Field and parent DOF setters refresh JAX field outputs to match native
        evaluation."""
        curves = _base_curves()
        field, native = _fields(
            coils_via_symmetries(curves, [Current(1e5) for _ in curves], 2, True)
        )
        _assert_field_matches_native(field, native)

        field.x = np.asarray(field.x) * 1.01
        _assert_field_matches_native(field, native)

        curves[1].x = curves[1].x + 2e-3
        _assert_field_matches_native(field, native)

        field.full_x = np.asarray(field.full_x) * 0.995
        _assert_field_matches_native(field, native)

    def test_current_change_refreshes_the_field(self):
        """Changing a source current refreshes JAX field outputs to match native
        evaluation."""
        curves = _base_curves()
        currents = [Current(1e5), Current(-3e4)]
        field, native = _fields(
            [Coil(curve, current) for curve, current in zip(curves, currents)]
        )
        _assert_field_matches_native(field, native)

        currents[0].x = np.array([2.5e5])
        _assert_field_matches_native(field, native)

    def test_fixed_dof_change_refreshes_the_field(self):
        """Fixing and then changing a captured DOF refreshes the field to match native
        evaluation."""
        curves = _base_curves()
        field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
        _assert_field_matches_native(field, native)

        curves[0].fix(0)
        _assert_field_matches_native(field, native)
        curves[0].set(0, curves[0].get(0) + 5e-3)
        _assert_field_matches_native(field, native)

    def test_set_points_refreshes_the_field(self):
        """Replacing evaluation points refreshes the JAX field to match the native field."""
        curves = _base_curves()
        field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
        _assert_field_matches_native(field, native)

        for points in (_OTHER_POINTS, _POINTS):
            field.set_points(points)
            native.set_points(points)
            _assert_field_matches_native(field, native)

    def test_shared_dofs_refresh_every_coil_that_reads_them(self):
        """Changing shared curve DOFs refreshes all affected coil contributions to match
        native evaluation."""
        curve = _base_curves(1)[0]
        shared = CurveXYZFourier(curve.quadpoints, curve.order, dofs=curve.dofs)
        field, native = _fields(
            [Coil(curve, Current(1e5)), Coil(shared, Current(-4e4))]
        )
        _assert_field_matches_native(field, native)

        shared.x = np.asarray(shared.x) + 3e-3
        np.testing.assert_array_equal(curve.x, shared.x)
        _assert_field_matches_native(field, native)

    def test_current_sum_term_change_refreshes_the_field(self):
        """Changing a term in a composed current refreshes field outputs to match native
        evaluation."""
        curves = _base_curves()
        first, second = Current(1e5), Current(2e4)
        field, native = _fields(
            [Coil(curves[0], first + second), Coil(curves[1], -2.5 * (first - second))]
        )
        _assert_field_matches_native(field, native)

        second.x = np.array([-7e4])
        _assert_field_matches_native(field, native)

    def test_resample_refreshes_the_field(self):
        """Notified and direct sample replacements refresh cached perturbed geometry and
        field values."""
        base = _base_curves(1)[0]
        sampler = GaussianSampler(base.quadpoints, 1e-3, 0.2, n_derivs=1)
        sample = PerturbationSample(sampler, randomgen=np.random.default_rng(2024))
        perturbed = CurvePerturbed(base, sample)
        coils = [Coil(perturbed, Current(1e5))]
        field, native = _fields(coils)
        _assert_field_matches_native(field, native)

        before = np.asarray(field.B()).copy()
        perturbed.resample()
        _assert_field_matches_native(field, native)
        self.assertTrue(
            not np.array_equal(np.asarray(field.B()), before),
            "not np.array_equal(np.asarray(field.B()), before)",
        )

        # A sample redrawn without notification: native caches stay stale, the
        # adapter detects the replaced sample on its next evaluation.
        before = np.asarray(field.B()).copy()
        sample.resample()
        perturbed.invalidate_cache()
        _assert_field_matches_native(field, _fields(coils)[1])
        self.assertTrue(
            not np.array_equal(np.asarray(field.B()), before),
            "not np.array_equal(np.asarray(field.B()), before)",
        )

    def test_notified_in_place_sample_edit_refreshes_the_field(self):
        """An in-place edit of the sample's arrays reaches the adapter's B, dB/dX and
        B VJP at the next notification, as it reaches native."""
        base = _base_curves(1)[0]
        sampler = GaussianSampler(base.quadpoints, 1e-3, 0.2, n_derivs=1)
        sample = PerturbationSample(sampler, randomgen=np.random.default_rng(2025))
        perturbed = CurvePerturbed(base, sample)
        field, native = _fields([Coil(perturbed, Current(1e5))])
        _assert_field_matches_native(field, native)

        before = np.asarray(field.B()).copy()
        sample[0][:, 0] += 0.05
        perturbed.set_recompute_flag()
        _assert_field_matches_native(field, native)
        self.assertTrue(
            not np.array_equal(np.asarray(field.B()), before),
            "not np.array_equal(np.asarray(field.B()), before)",
        )

    def test_backend_reconfiguration_rebuilds_the_coil_state(self):
        """Backend invalidation rebuilds cached coil state once while retaining native
        field agreement."""
        patches = self.patches
        curves = _base_curves()
        field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
        builds = _count_coil_state_builds(patches)
        field.B()
        field.B()
        self.assertTrue(builds.call_count == 1, "builds.call_count == 1")

        set_backend("native")
        _assert_field_matches_native(field, native)
        self.assertTrue(builds.call_count == 2, "builds.call_count == 2")

    def test_owner_updates_fingerprint_fixed_dofs_once(self):
        """Each owner update notifies the field; fixed DOFs are compared once, at the next read."""
        patches = self.patches
        curves = _base_curves(3)
        field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
        curves[0].fix(0)
        _assert_field_matches_native(field, native)
        reads = patches.enter_context(
            mock.patch.object(
                field,
                "_current_captured_coil_state_fingerprint",
                wraps=field._current_captured_coil_state_fingerprint,
            )
        )
        for curve in curves:
            curve.x = np.asarray(curve.x) + 1e-3
        field.B()
        field.B_vjp(_COTANGENT)(field)
        self.assertTrue(
            reads.call_count == 1,
            f"{reads.call_count} fixed-DOF fingerprints for one evaluation",
        )
        _assert_field_matches_native(field, native)

    def test_set_points_owns_a_jax_point_set_that_aliases_a_numpy_buffer(self):
        """Editing the caller's buffer after set_points moves neither the points nor B."""
        curves = _base_curves()
        field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
        buffer = aligned_host_copy(_POINTS)
        points = jax.device_put(buffer)  # noqa: TID251 test needs a JAX array aliasing a host buffer
        if next(iter(points.devices())).platform == "cpu":
            self.assertTrue(
                np.shares_memory(np.asarray(points), buffer),
                "precondition: CPU placement aliases",
            )
        field.set_points(points)
        field.B()

        buffer += 0.3

        np.testing.assert_array_equal(field.get_points_cart(), _POINTS)
        _assert_field_matches_native(field, native)

    def test_copies_follow_later_dof_changes(self):
        """A copy is a registered field over the coils: later DOF and layout changes reach it."""
        for make_copy in ["copy", "deepcopy_shared_coils"]:
            with self.subTest(make_copy=make_copy), self.case():
                self._case_copies_follow_later_dof_changes(make_copy)

    def _case_copies_follow_later_dof_changes(self, make_copy):
        """Run one row with case-local objects released before runtime cleanup."""
        curves = _base_curves()
        currents = [Current(1e5), Current(-3e4)]
        coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
        field, native = _fields(coils)
        field.B()
        field.B_vjp(_COTANGENT)(field)
        copied = (
            copy.copy(field)
            if make_copy == "copy"
            else copy.deepcopy(field, memo={id(field.coils): field.coils})
        )
        self.assertTrue(copied is not field, "copied is not field")
        self.assertTrue(
            all(
                mine is theirs
                for mine, theirs in zip(copied.coils, field.coils, strict=True)
            ),
            "all(mine is theirs for mine, theirs in zip(copied.coils, field.coils, strict=True))",
        )
        np.testing.assert_array_equal(copied.get_points_cart(), _POINTS)

        currents[0].x = np.array([2e5])
        curves[1].x = np.asarray(curves[1].x) + 2e-3
        _assert_field_matches_native(copied, native)
        _assert_field_matches_native(field, native)

        curves[0].fix(0)
        curves[0].set(0, curves[0].get(0) + 5e-3)
        self.assertTrue(
            copied.dof_size == field.dof_size,
            "copied.dof_size == field.dof_size",
        )
        _assert_field_matches_native(copied, native)

    def test_deepcopy_needs_deep_copyable_coils(self):
        """simsopt coils are not deep-copyable, so neither is a field over them (as natively)."""
        curves = _base_curves()
        field, _native = _fields([Coil(curve, Current(1e5)) for curve in curves])
        with self.assertRaises(TypeError):
            copy.deepcopy(field)

    def test_resampling_updates_captured_geometry_and_pullbacks(self):
        """Resampling perturbed curves refreshes field values and VJPs, including
        symmetry replicas."""
        for replicas in [False, True]:
            with self.subTest(replicas=replicas), self.case():
                self._case_resampling_updates_captured_geometry_and_pullbacks(replicas)

    def _case_resampling_updates_captured_geometry_and_pullbacks(self, replicas):
        """Run one row with case-local objects released before runtime cleanup."""
        curve = _base_curves()[0]
        sampler = GaussianSampler(curve.quadpoints, 1e-3, 0.2, n_derivs=1)
        sample = PerturbationSample(sampler, randomgen=np.random.default_rng(713))
        perturbed = CurvePerturbed(curve, sample)
        current = Current(1e5)
        coils = (
            coils_via_symmetries([perturbed], [current], 2, True)
            if replicas
            else [Coil(perturbed, current)]
        )
        native = BiotSavart(coils).set_points(_TWO_POINTS)
        adapter = backend.JaxBiotSavart(coils).set_points(_TWO_POINTS)
        before = np.asarray(adapter.B()).copy()
        perturbed.resample()
        native.set_points(_TWO_POINTS)
        adapter.set_points(_TWO_POINTS)
        expected = native.B()
        actual = np.asarray(adapter.B())
        self.assertTrue(
            np.max(np.abs(expected - before)) > 1e-6,
            "np.max(np.abs(expected - before)) > 1e-6",
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            cast(np.ndarray, adapter.B_vjp(_TWO_POINT_COTANGENT)(adapter)),
            cast(np.ndarray, cast(Derivative, native.B_vjp(_TWO_POINT_COTANGENT))(native)),
            rtol=1e-11,
            atol=1e-13,
        )

    def test_unchanged_perturbation_evaluations_do_not_copy_samples(self):
        """Repeated unchanged perturbed-field evaluations reuse results without copying
        samples or reading fingerprints."""
        patches = self.patches
        curves, coils = _coils(perturbed=True)
        curve = curves[0]
        curve.sample._sample = [
            sample.view(_CountedSample) for sample in curve.sample._sample
        ]
        field = backend.JaxBiotSavart(coils)
        field.set_points(_TWO_POINTS)
        expected = np.asarray(field.B()).copy()
        native = BiotSavart(coils)
        native.set_points(_TWO_POINTS)
        np.testing.assert_allclose(expected, native.B(), rtol=1e-12, atol=1e-14)
        spec = field.coil_dof_extraction_spec()
        fingerprint_reads = patches.enter_context(
            mock.patch.object(
                field,
                "_current_captured_coil_state_fingerprint",
                wraps=field._current_captured_coil_state_fingerprint,
            )
        )
        _CountedSample.copies.clear()
        for _ in range(10):
            np.testing.assert_allclose(np.asarray(field.B()), expected, rtol=0, atol=0)
        self.assertTrue(not _CountedSample.copies, "not _CountedSample.copies")
        self.assertTrue(not fingerprint_reads.called, "not fingerprint_reads.called")
        self.assertTrue(
            field.coil_dof_extraction_spec() is spec,
            "field.coil_dof_extraction_spec() is spec",
        )

    def test_notified_non_dof_parameter_changes_match_native(self):
        """A changed current scale, filament offset, frame-rotation scale or
        RotatedCurve rotation matrix reaches the adapter's B and B pullback as it
        reaches native, whether the changed object, the field, or the field's DOF
        setter sends the notification."""
        for parameter in (
            "current_scale", "filament_offset", "rotation_scale", "rotation_matrix",
        ):
            for notifier in ("changed_object", "field", "field_x_setter"):
                with self.subTest(parameter=parameter, notifier=notifier), self.case():
                    self._case_notified_non_dof_parameter_changes_match_native(
                        parameter, notifier
                    )

    def _case_notified_non_dof_parameter_changes_match_native(self, parameter, notifier):
        """Run one row with case-local objects released before runtime cleanup."""
        curves, _ = _coils()
        rotation = FrameRotation(curves[0].quadpoints, 1)
        rotation.x = np.array([0.2, -0.1, 0.05])
        filament = CurveFilament(FramedCurveCentroid(curves[0], rotation), 0.01, 0.02)
        current = ScaledCurrent(Current(1e5), 2.0)
        rotated = RotatedCurve(curves[1], 0.3, False)
        coils = [Coil(filament, current), Coil(rotated, Current(-4e4))]
        field = backend.JaxBiotSavart(coils).set_points(_TWO_POINTS)
        field.B()
        if parameter == "current_scale":
            current.scale = 3.0
            changed = current
        elif parameter == "filament_offset":
            filament.dn, filament.db = 0.03, -0.01
            changed = filament
        elif parameter == "rotation_scale":
            rotation.scale = 0.5
            changed = rotation
        else:
            # Native evaluation reads rotmat and its transpose.
            rotated.rotmat = RotatedCurve(curves[1], 0.7, False).rotmat
            rotated.rotmatT = rotated.rotmat.T.copy()
            changed = rotated
        if notifier == "changed_object":
            changed.set_recompute_flag()
        elif notifier == "field":
            field.set_recompute_flag()
        else:
            field.x = field.x
        native = BiotSavart(coils).set_points(_TWO_POINTS)
        np.testing.assert_allclose(
            np.asarray(field.B()), native.B(), rtol=1e-12, atol=1e-14
        )
        cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
        expected = cast(Derivative, native.B_vjp(cotangent))
        actual = field.B_vjp(cotangent)
        for owner in expected.data:
            np.testing.assert_allclose(
                actual.data[owner], expected.data[owner], rtol=1e-11, atol=1e-13
            )


class TestBiotSavartJaxCoilStateToken(JaxTestCase):
    """Traceable runtime cache invalidation is keyed to coil DOF state."""

    @staticmethod
    def _make_two_basic_coils():

        coils = []
        for current_amp in (1.0e6, -1.0e6):
            curve = CurveXYZFourier(quadpoints=16, order=1)
            curve.x = np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            coils.append(Coil(curve, Current(current_amp)))
        return coils

    @staticmethod
    def _coil_arrays_in_original_order(coil_set_spec):
        coil_count = sum(len(group.coil_indices) for group in coil_set_spec.groups)
        gammas = [None] * coil_count
        gammadashs = [None] * coil_count
        currents = [None] * coil_count
        for group in coil_set_spec.groups:
            for position, coil_index in enumerate(group.coil_indices):
                gammas[coil_index] = group.gammas[position]
                gammadashs[coil_index] = group.gammadashs[position]
                currents[coil_index] = group.currents[position]
        return gammas, gammadashs, currents

    def test_live_coil_set_spec_matches_explicit_dofs_and_native_coils(self):
        """The cached live spec is the explicit-DOF reconstruction of the native coils."""
        coils = self._make_two_basic_coils()
        bs_jax = backend.JaxBiotSavart(coils)

        live = self._coil_arrays_in_original_order(bs_jax.coil_set_spec())
        explicit = self._coil_arrays_in_original_order(
            bs_jax.coil_set_spec_from_dofs(bs_jax.x)
        )
        native = (
            [coil.curve.gamma() for coil in coils],
            [coil.curve.gammadash() for coil in coils],
            [coil.current.get_value() for coil in coils],
        )

        for live_entries, explicit_entries, native_entries in zip(
            live, explicit, native, strict=True
        ):
            for actual, from_dofs, expected in zip(
                live_entries, explicit_entries, native_entries, strict=True
            ):
                np.testing.assert_allclose(
                    cast(jax.Array, actual),
                    cast(jax.Array, from_dofs),
                    rtol=1e-14,
                    atol=1e-15,
                )
                np.testing.assert_allclose(
                    cast(jax.Array, actual),
                    expected,
                    rtol=1e-13,
                    atol=1e-14,
                )

    def test_biotsavart_extraction_spec_changes_only_for_captured_dof_contract(self):
        """Free-DOF changes reuse extraction specs; layout or captured fixed-DOF
        changes replace them."""
        coils = self._make_two_basic_coils()
        field = backend.JaxBiotSavart(list(coils))
        curve = coils[0].curve
        initial_spec = field.coil_dof_extraction_spec()
        initial_value = float(curve.local_full_x[0])

        curve.set(0, initial_value + 1.0e-3)
        self.assertTrue(
            field.coil_dof_extraction_spec() is initial_spec,
            "field.coil_dof_extraction_spec() is initial_spec",
        )

        curve.fix(0)
        fixed_spec = field.coil_dof_extraction_spec()
        self.assertTrue(
            fixed_spec is not initial_spec, "fixed_spec is not initial_spec"
        )

        curve.set(0, initial_value + 2.0e-3)
        self.assertTrue(
            field.coil_dof_extraction_spec() is not fixed_spec,
            "field.coil_dof_extraction_spec() is not fixed_spec",
        )


class TestBiotSavartJaxOwnerProjection(JaxTestCase):
    """The packed owner projection: host transfers, shared-DOF owners and fixed
    nested owners.
    """

    def test_warmed_full_owner_projection_host_boundaries(self):
        """Warmed owner projection matches native gradients with one free-vector
        placement and one packed host read."""
        patches = self.patches
        _, coils = _coils()
        field = backend.JaxBiotSavart(coils)
        field.set_points(_TWO_POINTS)
        cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
        native = BiotSavart(coils)
        native.set_points(_TWO_POINTS)
        native.B()
        expected = cast(Derivative, native.B_vjp(cotangent))
        pullback = field.B_pullback_native(cotangent)
        arrays, indices = pullback.d_coil_arrays, pullback.coil_indices
        field.coil_cotangents_to_derivative(arrays, indices)
        placements = []
        put = jax.device_put  # noqa: TID251 wraps the raw primitive to count placements

        def counted_put(value, *args, **kwargs):
            placements.extend(
                leaf.nbytes
                for leaf in jax.tree.leaves(value)
                if isinstance(leaf, np.ndarray) and leaf.ndim > 0
            )
            return put(value, *args, **kwargs)

        patches.enter_context(mock.patch.object(jax, "device_put", counted_put))
        materializations = patches.enter_context(
            mock.patch.object(backend, "host_array", wraps=backend.host_array)
        )
        metadata_placements = patches.enter_context(
            mock.patch.object(
                backend,
                "_place_array_tree_on_device",
                wraps=backend._place_array_tree_on_device,
            )
        )
        actual = field.coil_cotangents_to_derivative(arrays, indices)
        np.testing.assert_allclose(
            cast(np.ndarray, actual(field)),
            cast(np.ndarray, expected(native)),
            rtol=1e-11,
            atol=1e-13,
        )
        # One free-vector H2D boundary (32 doubles), one packed D2H.
        self.assertTrue(
            (len(placements), sum(placements), materializations.call_count)
            == (1, 256, 1),
            "(len(placements), sum(placements), materializations.call_count) == (1, 256, 1)",
        )
        self.assertTrue(
            not metadata_placements.called, "not metadata_placements.called"
        )

    def test_packed_projection_preserves_distinct_owners_with_shared_dofs(self):
        """Packed projection retains independent writable native partials for distinct
        owners sharing DOFs."""
        curves, _ = _coils()
        curve = curves[0]
        shared = CurveXYZFourier(curve.quadpoints, curve.order, dofs=curve.dofs)
        curve.fix(curve.local_dof_names[0])
        first, second = Current(1e5), Current(2e4)
        second.fix_all()
        coils = [Coil(curve, first + second), Coil(shared, -2.5 * (first - second))]
        field, native = backend.JaxBiotSavart(coils), BiotSavart(coils)
        for evaluator in (field, native):
            evaluator.set_points(_TWO_POINTS)
            evaluator.B()
        cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
        expected = cast(Derivative, native.B_vjp(cotangent))
        actual = field.B_vjp(cotangent)
        self.assertTrue(
            set(actual.data) == set(expected.data),
            "set(actual.data) == set(expected.data)",
        )
        for owner in expected.data:
            np.testing.assert_allclose(
                actual.data[owner], expected.data[owner], rtol=1e-11, atol=1e-13
            )
        self.assertTrue(
            not np.array_equal(actual.data[curve], actual.data[shared]),
            "not np.array_equal(actual.data[curve], actual.data[shared])",
        )
        self.assertTrue(
            cast(np.ndarray, actual.data[curve]).flags.writeable,
            "cast(np.ndarray, actual.data[curve]).flags.writeable",
        )
        self.assertTrue(
            not np.shares_memory(actual.data[curve], actual.data[shared]),
            "not np.shares_memory(actual.data[curve], actual.data[shared])",
        )

    def test_packed_projection_preserves_fixed_nested_curve_partials(self):
        """Packed projection preserves native partials for partially or fully fixed
        nested curve owners."""
        for fixed in ["partial", "full"]:
            with self.subTest(fixed=fixed), self.case():
                self._case_packed_projection_preserves_fixed_nested_curve_partials(
                    fixed
                )

    def _case_packed_projection_preserves_fixed_nested_curve_partials(self, fixed):
        """Run one row with case-local objects released before runtime cleanup."""
        curves, coils = _coils(perturbed=True)
        if fixed == "partial":
            curves[0].curve.fix(2)
        else:
            curves[0].curve.fix_all()
        field, native = backend.JaxBiotSavart(coils), BiotSavart(coils)
        for evaluator in (field, native):
            evaluator.set_points(_TWO_POINTS)
        np.testing.assert_allclose(
            np.asarray(field.B()), native.B(), rtol=1e-12, atol=1e-14
        )
        cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
        expected, actual = (
            cast(Derivative, native.B_vjp(cotangent)),
            field.B_vjp(cotangent),
        )
        self.assertTrue(
            set(actual.data) == set(expected.data),
            "set(actual.data) == set(expected.data)",
        )
        for owner in expected.data:
            np.testing.assert_allclose(
                actual.data[owner], expected.data[owner], rtol=1e-11, atol=1e-13
            )


class TestBiotSavartJaxPointsAndSerialization(JaxTestCase):
    """Evaluation points are owned snapshots, must be set before evaluation, and
    survive serialization.
    """

    def test_cartesian_points_are_owned_snapshots(self):
        """Mutating caller-owned Cartesian points leaves the adapter points and cached
        field equal to native snapshots."""
        points = aligned_host_copy(_TWO_POINTS)
        coils = [Coil(_base_curves()[0], Current(1e5))]
        native = BiotSavart(coils).set_points(points)
        adapter = backend.JaxBiotSavart(coils).set_points(points)
        adapter.B()
        points[:] += 0.3
        actual = adapter.get_points_cart()
        np.testing.assert_array_equal(actual, native.get_points_cart())
        np.testing.assert_allclose(adapter.B(), native.B(), rtol=1e-12, atol=1e-14)

    def test_cleared_points_must_be_set_before_evaluation(self):
        """After clear_points, evaluation raises until points are set again."""
        coils = [Coil(_base_curves()[0], Current(1e5))]
        adapter = backend.JaxBiotSavart(coils)
        adapter.clear_points()
        for quantity in ("B", "GradAbsB", "B_cyl", "get_points_cart"):
            with self.subTest(quantity=quantity), self.assertRaisesRegex(
                RuntimeError, "call set_points"
            ):
                getattr(adapter, quantity)()
        with self.assertRaisesRegex(RuntimeError, "call set_points"):
            adapter.B_vjp(np.zeros((1, 3)))
        native = BiotSavart(coils).set_points(_TWO_POINTS)
        np.testing.assert_allclose(
            adapter.set_points(_TWO_POINTS).B(), native.B(), rtol=1e-12, atol=1e-14
        )

    def test_serialization_preserves_points_and_field(self):
        """GSON reconstruction and JSON loading preserve evaluation points and field
        values in either coordinate convention."""
        for coordinates in ["cart", "cyl"]:
            with self.subTest(coordinates=coordinates), self.case() as patches:
                self._case_serialization_preserves_points_and_field(
                    coordinates, patches
                )

    def _case_serialization_preserves_points_and_field(self, coordinates, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        tmp_path = Path(patches.enter_context(TemporaryDirectory()))
        coils = [Coil(_base_curves()[0], Current(1e5))]
        native, adapter = BiotSavart(coils), backend.JaxBiotSavart(coils)
        for field in (native, adapter):
            getattr(field, "set_points_" + coordinates)(_TWO_POINTS)
        expected_points = native.get_points_cart()
        for name, field in (("native", native), ("adapter", adapter)):
            serialized = {}
            restored = type(field).from_dict(field.as_dict(serialized), serialized, {})
            filename = tmp_path / (name + ".json")
            field.save(filename, fmt="json")
            loaded = load(filename)
            for result in (restored, loaded):
                np.testing.assert_array_equal(result.get_points_cart(), expected_points)
                np.testing.assert_allclose(
                    result.B(), native.B(), rtol=1e-12, atol=1e-14
                )

    def test_payload_without_points_is_rejected_like_native(self):
        """A payload without a points key raises KeyError in both fields; None
        restores cleared points."""
        coils = [Coil(_base_curves()[0], Current(1e5))]
        for field in (BiotSavart(coils), backend.JaxBiotSavart(coils)):
            with self.subTest(field=type(field).__name__):
                serialized = {}
                payload = field.as_dict(serialized)
                del payload["points"]
                with self.assertRaises(KeyError):
                    type(field).from_dict(payload, serialized, {})
        adapter = backend.JaxBiotSavart(coils)
        adapter.clear_points()
        serialized = {}
        restored = backend.JaxBiotSavart.from_dict(adapter.as_dict(serialized), serialized, {})
        self.assertIsNone(restored.get_points_cart_ref())


class TestBiotSavartJaxContract(JaxTestCase):
    """Native field arithmetic rejects the JAX adapter."""

    def test_native_field_arithmetic_is_rejected(self):
        """Adapter field sums and scalar multiplication reject unsupported native
        arithmetic with TypeError."""
        for operation in ["left_scale", "right_scale", "add", "reverse_add"]:
            with self.subTest(operation=operation), self.case():
                self._case_native_field_arithmetic_is_rejected(operation)

    def _case_native_field_arithmetic_is_rejected(self, operation):
        """Run one row with case-local objects released before runtime cleanup."""
        field = backend.JaxBiotSavart([])
        with self.assertRaisesRegex(TypeError, "simsopt.field.BiotSavart"):
            if operation == "left_scale":
                _ = 2 * field
            elif operation == "right_scale":
                _ = field * 2
            elif operation == "add":
                _ = field + field
            else:
                _ = 0 + field

    def test_native_field_wrappers_reject_adapter_dependencies(self):
        """Native field sum and multiplication wrappers reject JAX adapter dependencies."""
        for operation in [
            "native_left",
            "native_right",
            "sum_wrapper",
            "scale_wrapper",
        ]:
            with self.subTest(operation=operation), self.case():
                self._case_native_field_wrappers_reject_adapter_dependencies(operation)

    def _case_native_field_wrappers_reject_adapter_dependencies(self, operation):
        """Run one row with case-local objects released before runtime cleanup."""
        field, native = backend.JaxBiotSavart([]), BiotSavart([])
        with self.assertRaisesRegex(TypeError, "simsopt.field.BiotSavart"):
            if operation == "native_left":
                _ = native + field
            elif operation == "native_right":
                _ = field + native
            elif operation == "sum_wrapper":
                MagneticFieldSum([field, native])
            else:
                MagneticFieldMultiply(2, field)
