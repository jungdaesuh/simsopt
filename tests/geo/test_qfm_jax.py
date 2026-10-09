"""Native 9e027eac3 QFM values, callbacks, optimizer and state contracts."""

from unittest_jax_support import JaxTestCase
from unittest import mock
import os

from unittest_jax_support import parity_default_device

from contextlib import ExitStack, contextmanager
from copy import copy
from typing import Iterator, cast

import jax
import numpy as np

from simsopt.configs import get_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import (
    RotatedCurve,
    SurfaceRZFourier,
    SurfaceXYZFourier,
    SurfaceXYZTensorFourier,
)
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import Area, ToroidalFlux, Volume
from simsopt_jax.backend import get_field_kernel_tuning, set_backend
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.qfm import qfm_label, qfm_residual
from simsopt_jax.runtime.host_boundary import disallow_host_transfers, host_array
from simsopt_jax_adapters.field import JaxBiotSavart
from simsopt_jax_adapters.geo.qfm import JaxQfmSurface

_SURFACES = [SurfaceRZFourier, SurfaceXYZFourier, SurfaceXYZTensorFourier]
_LABELS = [Volume, Area, ToroidalFlux]


def _pair(
    surface_class=SurfaceRZFourier,
    stellsym=True,
    label_class=Volume,
    cloned=False,
    break_field_symmetry=True,
):
    curves, currents, axis, nfp, field = get_data("ncsx")
    if not stellsym and break_field_symmetry:
        # The upstream QFM solve test breaks *field* symmetry when surface
        # symmetry is disabled. A perfectly symmetric field/start introduces
        # roundoff-sized asymmetric directions in the squared-equality solve.
        reflected = [RotatedCurve(curve, 0, True) for curve in curves]
        rng = np.random.default_rng(1)
        for curve in reflected:
            curve.rotmat += 0.001 * rng.uniform(-1, 1, curve.rotmat.shape)
            curve.rotmatT = curve.rotmat.T.copy()
        field = BiotSavart(
            coils_via_symmetries(
                curves + reflected,
                currents + [-current for current in currents],
                nfp,
                False,
            )
        )
    surfaces = [
        surface_class(
            mpol=1,
            ntor=1,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0, 1 / nfp, 7, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 8, endpoint=False),
        )
        for _ in range(2)
    ]
    for surface in surfaces:
        surface.fit_to_curve(axis, 0.2, flip_theta=True)
    native_field = BiotSavart(field.coils)
    jax_field = JaxBiotSavart(field.coils)
    options = {"nphi": 5, "ntheta": 9, "range": "field period"} if cloned else {}
    labels = []
    for surface in surfaces:
        if label_class is ToroidalFlux:
            labels.append(
                label_class(surface, BiotSavart(field.coils), idx=-1, **options)
            )
        else:
            labels.append(label_class(surface, **options))
    targetlabel = labels[0].J() * 1.03
    return (
        QfmSurface(native_field, surfaces[0], labels[0], targetlabel),
        JaxQfmSurface(jax_field, surfaces[1], labels[1], targetlabel),
    )


def _assert_pair(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@contextmanager
def _compilations() -> Iterator[list[str]]:
    events: list[str] = []

    def record(event: str, duration_secs: float, **kwargs: str | int):
        if "/compile" in event:
            events.append(event)

    with ExitStack() as stack:
        jax.monitoring.register_event_duration_secs_listener(record)
        stack.callback(jax.monitoring.unregister_event_duration_listener, record)
        yield events


def _evaluate(port):
    x = np.array(port.surface.x, copy=True)
    for derivatives in (0, 1):
        port.qfm_objective(x, derivatives)
        port.qfm_label_constraint(x, derivatives)
        port.qfm_penalty_constraints(x, derivatives, constraint_weight=2.0)


class TestQfmJax(JaxTestCase):
    def test_values_gradients_and_own_label_grid_match_native(self):
        """QFM values and full gradients match native on every supported surface and label grid."""
        for cloned in [False, True]:
            for label_class in _LABELS:
                for stellsym in [True, False]:
                    for surface_class in _SURFACES:
                        with self.subTest(
                            cloned=cloned,
                            label_class=label_class,
                            stellsym=stellsym,
                            surface_class=surface_class,
                        ), self.case() as patches:
                            self._case_values_gradients_and_own_label_grid_match_native(
                                cloned, label_class, stellsym, surface_class, patches
                            )

    def _case_values_gradients_and_own_label_grid_match_native(
        self, cloned, label_class, stellsym, surface_class, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair(surface_class, stellsym, label_class, cloned)
        x = np.array(native.surface.x, copy=True)
        x += np.random.default_rng(13).normal(0, 1e-4, x.shape)
        for name in (
            "qfm_objective",
            "qfm_label_constraint",
            "qfm_penalty_constraints",
        ):
            for derivatives in (0, 1):
                expected = getattr(native, name)(x, derivatives=derivatives)
                actual = getattr(port, name)(x, derivatives=derivatives)
                if derivatives:
                    for a, e in zip(cast(tuple, actual), expected, strict=True):
                        _assert_pair(a, e)
                else:
                    _assert_pair(actual, expected)
        _assert_pair(port.qfm.J(), native.qfm.J())
        _assert_pair(
            port.qfm.dJ_by_dsurfacecoefficients(),
            native.qfm.dJ_by_dsurfacecoefficients(),
        )
        np.testing.assert_array_equal(
            port.biotsavart.get_points_cart(), native.biotsavart.get_points_cart()
        )

    def test_gradients_are_central_differences_of_native_values(self):
        """All QFM gradients match centered differences of native values for both symmetries."""
        for stellsym in [True, False]:
            for label_class in _LABELS:
                with self.subTest(
                    stellsym=stellsym, label_class=label_class
                ), self.case() as patches:
                    self._case_gradients_are_central_differences_of_native_values(
                        stellsym, label_class, patches
                    )

    def _case_gradients_are_central_differences_of_native_values(
        self, stellsym, label_class, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair(stellsym=stellsym, label_class=label_class, cloned=True)
        x = np.array(native.surface.x, copy=True)
        direction = np.random.default_rng(11).normal(size=x.shape)
        direction /= np.linalg.norm(direction)
        for name in (
            "qfm_objective",
            "qfm_label_constraint",
            "qfm_penalty_constraints",
        ):
            _, gradient = getattr(port, name)(x, derivatives=1)
            step = 1e-6
            difference = (
                getattr(native, name)(x + step * direction)
                - getattr(native, name)(x - step * direction)
            ) / (2 * step)
            np.testing.assert_allclose(
                gradient @ direction, difference, rtol=1e-7, atol=1e-9
            )

    def test_solves_from_the_same_start_match_native(self):
        """Penalty and squared-equality solves match native state and optimizer results from identical starts."""
        for stellsym in [True, False]:
            for method in ["LBFGS", "SLSQP"]:
                with self.subTest(
                    stellsym=stellsym, method=method
                ), self.case() as patches:
                    self._case_solves_from_the_same_start_match_native(
                        stellsym, method, patches
                    )

    def _case_solves_from_the_same_start_match_native(self, stellsym, method, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair(stellsym=stellsym)
        if method == "SLSQP":
            # Upstream initializes exact minimization with a penalty solve. A cold
            # symmetric start in the larger nonsymmetric coefficient space is
            # ill-conditioned: a one-ulp *native* label perturbation changes the
            # final asymmetric coefficients by 1e-6 (validation probe/report).
            native.minimize_qfm(method="LBFGS", tol=1e-12, maxiter=500)
            port.surface.x = np.array(native.surface.x, copy=True)
        expected = native.minimize_qfm(method=method, tol=1e-13, maxiter=1000)
        actual = port.minimize_qfm(method=method, tol=1e-13, maxiter=1000)
        self.assertTrue(
            list(actual)
            == list(expected)
            == ["fun", "gradient", "iter", "info", "success", "s"],
            'list(actual) == list(expected) == ["fun", "gradient", "iter", "info", "success", "s"]',
        )
        self.assertTrue(
            actual["success"] == expected["success"],
            'actual["success"] == expected["success"]',
        )
        self.assertTrue(expected["success"], expected["info"].message)
        self.assertTrue(actual["s"] is port.surface, 'actual["s"] is port.surface')
        np.testing.assert_allclose(actual["s"].x, expected["s"].x, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            actual["fun"], expected["fun"], rtol=1e-8, atol=1e-12
        )
        self.assertTrue(
            list(actual["info"]) == list(expected["info"]),
            'list(actual["info"]) == list(expected["info"])',
        )

    def test_iteration_limit_and_squared_constraint_failure_match_native(self):
        """Capped solves and feasible squared-equality failures retain native status and final surface."""
        for method in ["LBFGS", "SLSQP"]:
            with self.subTest(method=method), self.case() as patches:
                self._case_iteration_limit_and_squared_constraint_failure_match_native(
                    method, patches
                )

    def _case_iteration_limit_and_squared_constraint_failure_match_native(
        self, method, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair()
        if method == "SLSQP":
            # This perturbation makes native and compiled Volume differ by ulps;
            # the singular squared equality must use the native scalar exactly.
            for solver in (native, port):
                start = np.asarray(solver.surface.x)
                solver.surface.x = start + np.random.default_rng(4).normal(
                    0, 1e-3, start.shape
                )
            native.targetlabel = native.label.J()
            port.targetlabel = port.label.J()
            self.assertTrue(
                host_array(qfm_label(port._label_spec())) != port.targetlabel,
                "host_array(qfm_label(port._label_spec())) != port.targetlabel",
            )
            start = np.array(native.surface.x, copy=True)
        expected = native.minimize_qfm(method=method, tol=1e-12, maxiter=1)
        actual = port.minimize_qfm(method=method, tol=1e-12, maxiter=1)
        self.assertTrue(
            actual["success"] == expected["success"] == False,
            'actual["success"] == expected["success"] == False',
        )
        self.assertTrue(
            actual["info"].status == expected["info"].status,
            'actual["info"].status == expected["info"].status',
        )
        np.testing.assert_allclose(
            actual["s"].x, expected["s"].x, rtol=1e-12, atol=1e-12
        )
        if method == "SLSQP":
            self.assertTrue(actual["info"].status == 6, 'actual["info"].status == 6')
            np.testing.assert_array_equal(actual["s"].x, start)

    def test_flux_gradient_uses_the_current_label_field_points(self):
        """Flux derivatives use the live independent field points after index copies or external edits."""
        for changed_state in ["copied_index", "external_points"]:
            for jax_label_field in [False, True]:
                with self.subTest(
                    changed_state=changed_state, jax_label_field=jax_label_field
                ), self.case() as patches:
                    self._case_flux_gradient_uses_the_current_label_field_points(
                        changed_state, jax_label_field, patches
                    )

    def _case_flux_gradient_uses_the_current_label_field_points(
        self, changed_state, jax_label_field, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair(label_class=ToroidalFlux)
        if jax_label_field:
            port.label = ToroidalFlux(
                port.surface, JaxBiotSavart(port.label.biotsavart.coils), idx=-1
            )
        # Set the surface before changing the independent field buffer: another
        # assignment would notify the original label and refresh its points.
        for solver in (native, port):
            solver.surface.x = np.array(solver.surface.x, copy=True)
            if changed_state == "copied_index":
                solver.label = copy(solver.label)
                solver.label.idx = 0
            else:
                # An independent label surface keeps QFM's surface assignment
                # from notifying the label and replacing the external points.
                label_surface = SurfaceRZFourier(
                    mpol=1,
                    ntor=1,
                    nfp=solver.surface.nfp,
                    quadpoints_phi=solver.surface.quadpoints_phi,
                    quadpoints_theta=solver.surface.quadpoints_theta,
                )
                label_surface.set_dofs(solver.surface.get_dofs())
                field_class = (
                    JaxBiotSavart
                    if isinstance(solver.label.biotsavart, JaxBiotSavart)
                    else BiotSavart
                )
                solver.label = ToroidalFlux(
                    label_surface, field_class(solver.label.biotsavart.coils), idx=-1
                )
                points = solver.label.surface.gamma()[solver.label.idx] + [
                    0.01,
                    0.02,
                    0.03,
                ]
                solver.label.biotsavart.set_points(points)
        for name in ("qfm_label_constraint", "qfm_penalty_constraints"):
            expected = getattr(native, name)(native.surface.x, derivatives=1)
            actual = getattr(port, name)(port.surface.x, derivatives=1)
            for a, e in zip(cast(tuple, actual), expected, strict=True):
                _assert_pair(a, e)

    def test_jax_flux_feasible_start_preserves_label_arithmetic_and_status(self):
        """A JAX flux label preserves native feasible-start arithmetic and SLSQP status."""
        native, port = _pair(label_class=ToroidalFlux)
        for current in {
            owner
            for coil in native.biotsavart.coils
            for owner in coil.current.unique_dof_lineage
        }:
            current.local_full_x *= 2
        for solver in (native, port):
            start = np.asarray(solver.surface.x)
            solver.surface.x = start + np.random.default_rng(1).normal(
                0, 1e-3, start.shape
            )
            solver.label = ToroidalFlux(
                solver.surface,
                JaxBiotSavart(solver.biotsavart.coils),
                idx=0,
                nphi=3,
                ntheta=7,
                range="field period",
            )
            solver.targetlabel = np.float64(host_array(solver.label.J()))
        self.assertTrue(
            native.targetlabel == port.targetlabel,
            "native.targetlabel == port.targetlabel",
        )
        start = np.array(native.surface.x, copy=True)
        # Warming preserves the same explicit-boundary contract under the guard.
        _evaluate(port)
        with disallow_host_transfers(), _compilations() as events:
            value, gradient = port.qfm_label_constraint(port.surface.x, derivatives=1)
            self.assertTrue(value == 0, "value == 0")
            np.testing.assert_array_equal(gradient, np.zeros_like(gradient))
            actual = port.minimize_qfm(method="SLSQP", tol=1e-12, maxiter=1)
        self.assertTrue(events == [], "events == []")
        expected = native.minimize_qfm(method="SLSQP", tol=1e-12, maxiter=1)
        self.assertTrue(
            actual["success"] == expected["success"] == False,
            'actual["success"] == expected["success"] == False',
        )
        self.assertTrue(
            actual["info"].status == expected["info"].status == 6,
            'actual["info"].status == expected["info"].status == 6',
        )
        np.testing.assert_array_equal(actual["s"].x, start)
        np.testing.assert_array_equal(expected["s"].x, start)

    def test_python_float_target_overflow_matches_native(self):
        """Python-float target overflow raises the same error before every native derivative path."""
        for label_class in [Volume, Area]:
            for method in ["qfm_label_constraint", "qfm_penalty_constraints"]:
                for derivatives in [0, 1]:
                    with self.subTest(
                        label_class=label_class, method=method, derivatives=derivatives
                    ), self.case() as patches:
                        self._case_python_float_target_overflow_matches_native(
                            label_class, method, derivatives, patches
                        )

    def _case_python_float_target_overflow_matches_native(
        self, label_class, method, derivatives, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair(label_class=label_class)
        for solver in (native, port):
            solver.targetlabel = 1e200
            with self.assertRaises(OverflowError):
                getattr(solver, method)(solver.surface.x, derivatives=derivatives)

    def test_numpy_float_target_keeps_native_nonfinite_results(self):
        """NumPy targets retain native nonfinite scalar and derivative results."""
        for method in ["qfm_label_constraint", "qfm_penalty_constraints"]:
            for derivatives in [0, 1]:
                with self.subTest(
                    method=method, derivatives=derivatives
                ), self.case() as patches:
                    self._case_numpy_float_target_keeps_native_nonfinite_results(
                        method, derivatives, patches
                    )

    def _case_numpy_float_target_keeps_native_nonfinite_results(
        self, method, derivatives, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair(label_class=Area)
        # Perturbed Area has nonzero gradient components. Volume is invariant to
        # theta-independent vertical displacements even after DOF perturbation.
        for solver in (native, port):
            start = np.asarray(solver.surface.x)
            solver.surface.x = start + np.random.default_rng(4).normal(
                0, 1e-3, start.shape
            )
        native.targetlabel = port.targetlabel = np.float64(1e200)
        with np.errstate(all="ignore"):
            expected = getattr(native, method)(
                native.surface.x, derivatives=derivatives
            )
            actual = getattr(port, method)(port.surface.x, derivatives=derivatives)
        if derivatives:
            for a, e in zip(cast(tuple, actual), expected, strict=True):
                _assert_pair(a, e)
        else:
            _assert_pair(actual, expected)

    def test_shared_flux_field_with_short_point_buffer_raises_native_value_error(self):
        """Shared flux fields with incompatible QFM points raise the native reshape error."""
        for method in ["J", "dJ_by_dsurfacecoefficients"]:
            with self.subTest(method=method), self.case() as patches:
                self._case_shared_flux_field_with_short_point_buffer_raises_native_value_error(
                    method, patches
                )

    def _case_shared_flux_field_with_short_point_buffer_raises_native_value_error(
        self, method, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair()
        labels = [
            ToroidalFlux(solver.surface, solver.biotsavart) for solver in (native, port)
        ]
        for solver, label in zip((native, port), labels, strict=True):
            label.invalidate_cache()
            with self.assertRaisesRegex(ValueError, "cannot reshape"):
                getattr(solver.qfm, method)()

    def test_independent_jax_flux_incompatible_points_raise_native_type_error(self):
        """Independent JAX flux point counts preserve the native broadcasting error."""
        for method in ["qfm_label_constraint", "qfm_penalty_constraints"]:
            for derivatives in [0, 1]:
                for point_count in [3, 9]:
                    with self.subTest(
                        method=method, derivatives=derivatives, point_count=point_count
                    ), self.case() as patches:
                        self._case_independent_jax_flux_incompatible_points_raise_native_type_error(
                            method, derivatives, point_count, patches
                        )

    def _case_independent_jax_flux_incompatible_points_raise_native_type_error(
        self, method, derivatives, point_count, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair()
        for solver in (native, port):
            # Separate surface and field prevent the QFM callback from refreshing
            # the label's external buffer before its JAX-backed J() multiply.
            label_surface = SurfaceRZFourier(
                mpol=1,
                ntor=1,
                nfp=solver.surface.nfp,
                quadpoints_phi=solver.surface.quadpoints_phi,
                quadpoints_theta=solver.surface.quadpoints_theta,
            )
            label_surface.set_dofs(solver.surface.get_dofs())
            label_field = JaxBiotSavart(solver.biotsavart.coils)
            solver.label = ToroidalFlux(label_surface, label_field, idx=0)
            label_field.set_points(
                np.resize(label_surface.gamma()[0], (point_count, 3))
            )
        with self.assertRaisesRegex(
            TypeError, "mul got incompatible shapes for broadcasting"
        ):
            getattr(native, method)(native.surface.x, derivatives=derivatives)
        with disallow_host_transfers():
            with self.assertRaisesRegex(
                TypeError, "mul got incompatible shapes for broadcasting"
            ):
                getattr(port, method)(port.surface.x, derivatives=derivatives)
        for solver in (native, port):
            self.assertTrue(
                solver.label.biotsavart.get_points_cart().shape == (point_count, 3),
                "solver.label.biotsavart.get_points_cart().shape == (point_count, 3)",
            )

    def test_new_values_and_full_solves_have_no_implicit_transfers_or_recompiles(self):
        """Changed inputs and capped solves reuse compiled programs under the explicit-transfer guard."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_new_values_and_full_solves_have_no_implicit_transfers_or_recompiles(
                    parity_lane, patches
                )

    def _case_new_values_and_full_solves_have_no_implicit_transfers_or_recompiles(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        with parity_default_device(parity_lane):
            native, port = _pair(label_class=ToroidalFlux)
            _evaluate(port)
            old_spec = port.qfm._spec()
            old_value = host_array(qfm_residual(old_spec))
            with disallow_host_transfers(), _compilations() as events:
                port.surface.x = np.asarray(port.surface.x) * 1.001
                native.surface.x = port.surface.x
                port.targetlabel = explicit_device_array(
                    port.targetlabel * 1.002,
                    dtype=np.float64,
                    reference=old_spec.points,
                )
                native.targetlabel = host_array(port.targetlabel)[()]
                port.label.idx = native.label.idx = 1
                port.label.biotsavart.coils[0].current.x *= 1.0001
                _evaluate(port)
                actual = port.qfm_penalty_constraints(
                    port.surface.x, 1, constraint_weight=3.0
                )
                port.minimize_qfm(method="LBFGS", maxiter=1)
                port.minimize_qfm(method="SLSQP", maxiter=1)
            self.assertTrue(events == [], "events == []")
            native.surface.x = port.surface.x
            _assert_pair(
                port.qfm_objective(port.surface.x),
                native.qfm_objective(native.surface.x),
            )
            self.assertTrue(np.isfinite(actual[0]), "np.isfinite(actual[0])")
            np.testing.assert_array_equal(host_array(qfm_residual(old_spec)), old_value)

    def test_public_label_target_and_copy_are_live(self):
        """Public label/target edits and shallow copies are read afresh by each evaluation."""
        native, port = _pair()
        copied = copy(port)
        for solver in (port, copied):
            solver.label = Area(solver.surface, nphi=4, ntheta=6)
            solver.targetlabel = 2.0
        native.label = Area(native.surface, nphi=4, ntheta=6)
        native.targetlabel = 2.0
        for solver in (port, copied):
            actual = solver.qfm_penalty_constraints(solver.surface.x, 1)
            expected = native.qfm_penalty_constraints(native.surface.x, 1)
            for a, e in zip(cast(tuple, actual), expected, strict=True):
                _assert_pair(a, e)

    def test_residual_field_replacement_is_live_on_a_shallow_copy(self):
        """Replacing a copied residual field changes values and gradients without stale snapshots."""
        native, port = _pair()
        copied = copy(port)
        _, _, _, _, replacement = get_data("ncsx")
        replacement.coils[0].current.x *= 1.2
        native.qfm.biotsavart = replacement
        port.qfm.biotsavart = JaxBiotSavart(replacement.coils)
        for solver in (native, port, copied):
            solver.qfm.invalidate_cache()
        _assert_pair(port.qfm.J(), native.qfm.J())
        _assert_pair(copied.qfm.J(), native.qfm.J())

    def test_fixed_dofs_keep_native_full_gradient_contract(self):
        """Fixed surface DOFs preserve the native full-coefficient gradient contract."""
        native, port = _pair()
        for solver in (native, port):
            solver.surface.fix(solver.surface.local_dof_names[0])
        x = np.array(native.surface.x, copy=True)
        for a, e in zip(
            port.qfm_objective(x, 1), native.qfm_objective(x, 1), strict=True
        ):
            _assert_pair(a, e)
        self.assertTrue(
            port.qfm.dJ_by_dsurfacecoefficients().size == port.surface.get_dofs().size,
            "port.qfm.dJ_by_dsurfacecoefficients().size == port.surface.get_dofs().size",
        )

    def test_external_field_points_follow_native_evaluation_boundary(self):
        """Externally set field points are used until native parent invalidation refreshes them."""
        native, port = _pair()
        points = native.surface.gamma().reshape(-1, 3) + [0.01, 0.02, 0.03]
        for solver in (native, port):
            solver.biotsavart.set_points(points)
        _assert_pair(port.qfm.J(), native.qfm.J())
        _assert_pair(
            port.qfm.dJ_by_dsurfacecoefficients(),
            native.qfm.dJ_by_dsurfacecoefficients(),
        )

    def test_jax_flux_label_uses_explicit_host_boundary(self):
        """JAX flux labels preserve native values under explicit host materialization."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_jax_flux_label_uses_explicit_host_boundary(
                    parity_lane, patches
                )

    def _case_jax_flux_label_uses_explicit_host_boundary(self, parity_lane, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        with parity_default_device(parity_lane):
            native, port = _pair(label_class=ToroidalFlux, cloned=True)
            port.label = ToroidalFlux(
                port.surface,
                JaxBiotSavart(port.label.biotsavart.coils),
                idx=-1,
                nphi=5,
                ntheta=9,
                range="field period",
            )
            _evaluate(port)
            with disallow_host_transfers():
                for derivatives in (0, 1):
                    expected = native.qfm_label_constraint(
                        native.surface.x, derivatives
                    )
                    actual = port.qfm_label_constraint(port.surface.x, derivatives)
                    if derivatives:
                        for a, e in zip(cast(tuple, actual), expected, strict=True):
                            _assert_pair(a, e)
                    else:
                        _assert_pair(actual, expected)

    def test_nonfinite_residuals_are_not_sanitized(self):
        """Zero fields/normals, NaNs and infinities retain native nonfinite residual behavior."""
        for condition in ["zero_field", "zero_normal", "nan", "inf"]:
            with self.subTest(condition=condition), self.case() as patches:
                self._case_nonfinite_residuals_are_not_sanitized(condition, patches)

    def _case_nonfinite_residuals_are_not_sanitized(self, condition, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair()
        if condition == "zero_field":
            for coil in native.biotsavart.coils:
                for owner in coil.current.unique_dof_lineage:
                    owner.local_full_x = np.zeros_like(owner.local_full_x)
        else:
            dofs = native.surface.get_dofs().copy()
            dofs[:] = {"zero_normal": 0.0, "nan": np.nan, "inf": np.inf}[condition]
            native.surface.set_dofs(dofs)
            port.surface.set_dofs(dofs)
        with np.errstate(all="ignore"):
            _assert_pair(port.qfm.J(), native.qfm.J())
            _assert_pair(
                port.qfm.dJ_by_dsurfacecoefficients(),
                native.qfm.dJ_by_dsurfacecoefficients(),
            )

    def test_nonfinite_numeric_operands_match_native(self):
        """Nonfinite target and penalty coefficients preserve native values and gradients."""
        for targetlabel, constraint_weight in [
            (np.inf, 0.0),
            (np.nan, 0.0),
            (0.0, np.inf),
            (0.0, np.nan),
        ]:
            with self.subTest(
                targetlabel=targetlabel, constraint_weight=constraint_weight
            ), self.case() as patches:
                self._case_nonfinite_numeric_operands_match_native(
                    targetlabel, constraint_weight, patches
                )

    def _case_nonfinite_numeric_operands_match_native(
        self, targetlabel, constraint_weight, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair()
        native.targetlabel = port.targetlabel = targetlabel
        with np.errstate(all="ignore"):
            for a, e in zip(
                port.qfm_penalty_constraints(port.surface.x, 1, constraint_weight),
                native.qfm_penalty_constraints(native.surface.x, 1, constraint_weight),
                strict=True,
            ):
                _assert_pair(a, e)

    def test_nonfinite_weight_at_exact_label_feasibility_keeps_native_nans(self):
        """A nonfinite coefficient times an exactly feasible label retains native NaNs."""
        for constraint_weight in [np.inf, np.nan]:
            with self.subTest(
                constraint_weight=constraint_weight
            ), self.case() as patches:
                self._case_nonfinite_weight_at_exact_label_feasibility_keeps_native_nans(
                    constraint_weight, patches
                )

    def _case_nonfinite_weight_at_exact_label_feasibility_keeps_native_nans(
        self, constraint_weight, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, port = _pair()
        native.targetlabel = port.targetlabel = native.label.J()
        with np.errstate(all="ignore"):
            for a, e in zip(
                port.qfm_penalty_constraints(port.surface.x, 1, constraint_weight),
                native.qfm_penalty_constraints(native.surface.x, 1, constraint_weight),
                strict=True,
            ):
                _assert_pair(a, e)

    def test_invalid_derivative_method_and_flux_index_match_native(self):
        """Invalid orders, method names and flux indices raise native exception classes."""
        native, port = _pair(label_class=ToroidalFlux)
        for solver in (native, port):
            with self.assertRaises(AssertionError):
                solver.qfm_objective(solver.surface.x, 2)
            with self.assertRaises(ValueError):
                solver.minimize_qfm(method="invalid")
            solver.label.idx = 1000
            with self.assertRaises(IndexError):
                solver.qfm_label_constraint(solver.surface.x)

    def test_backend_settings_rebuild_enclosing_qfm_programs(self):
        """Changing field tuning clears the enclosing compiled QFM programs."""
        patches = self.patches
        set_backend("jax", device="cpu", intent="fast")
        _, port = _pair()
        old_tuning = get_field_kernel_tuning()
        old = port.qfm_objective(port.surface.x, 1)
        patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_JAX_POINT_CHUNK_SIZE": "3"})
        )
        set_backend("jax", device="cpu", intent="fast")
        for a, e in zip(port.qfm_objective(port.surface.x, 1), old, strict=True):
            _assert_pair(a, e)
        self.assertTrue(
            get_field_kernel_tuning() != old_tuning,
            "get_field_kernel_tuning() != old_tuning",
        )

    def test_solver_exception_keeps_last_callback_surface(self):
        """A callback exception leaves the mutable surface at its last native callback iterate."""
        patches = self.patches
        native, port = _pair()
        starts = [np.array(solver.surface.x, copy=True) for solver in (native, port)]
        visited = []

        def fail(self, x, derivatives=0, constraint_weight=1):
            moved = x * 1.01
            self.qfm_objective(moved, derivatives=1)
            visited.append(moved.copy())
            raise RuntimeError("callback failed after an evaluation")

        patches.enter_context(
            mock.patch.object(QfmSurface, "qfm_penalty_constraints", fail)
        )
        patches.enter_context(
            mock.patch.object(JaxQfmSurface, "qfm_penalty_constraints", fail)
        )
        for solver, start in zip((native, port), starts, strict=True):
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                solver.minimize_qfm(method="LBFGS")
            np.testing.assert_array_equal(solver.surface.x, visited[-1])
            self.assertTrue(
                np.linalg.norm(np.asarray(solver.surface.x) - start) > 0,
                "np.linalg.norm(np.asarray(solver.surface.x) - start) > 0",
            )

    def test_gradients_taylor_against_native_values(self):
        """QFM, label and penalty gradients converge with the native forward Taylor stencils."""
        for label_class in _LABELS:
            for stellsym in (True, False):
                with self.subTest(
                    label_class=label_class, stellsym=stellsym
                ), self.case():
                    self._case_gradients_taylor(label_class, stellsym)

    def _case_gradients_taylor(self, label_class, stellsym):
        native, port = _pair(stellsym=stellsym, label_class=label_class, cloned=True)
        native.targetlabel = port.targetlabel = 0.1
        x = np.array(native.surface.x, copy=True)
        # Native QFM Taylor tests use this target and unnormalized seeded direction.
        direction = np.random.RandomState(1).uniform(size=x.shape) - 0.5
        for name in (
            "qfm_objective",
            "qfm_label_constraint",
            "qfm_penalty_constraints",
        ):
            options = {"constraint_weight": 11.1232} if name == "qfm_penalty_constraints" else {}
            _, gradient = getattr(port, name)(x, derivatives=1, **options)
            analytic = gradient @ direction
            base = getattr(native, name)(x, derivatives=0, **options)
            previous_error = 1e9
            powers = range(7, 17) if name == "qfm_label_constraint" else range(13, 20)
            for step in np.power(2.0, -np.asarray(list(powers))):
                difference = (
                    getattr(native, name)(x + step * direction, derivatives=0, **options) - base
                ) / step
                error = np.linalg.norm(difference - analytic) / np.linalg.norm(analytic)
                self.assertLess(error, 0.6 * previous_error, name)
                previous_error = error
            native.surface.x = x
            port.surface.x = x
