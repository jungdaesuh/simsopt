"""JAX ``JaxNonQuasiSymmetricRatio`` and the JAX boozerQA run against native.

The objective's ratio and partial derivatives match native
``NonQuasiSymmetricRatio``'s (``J``, ``dJ_by_dsurfacecoefficients``,
``B_vjp(dJ_by_dB)``); its value and coil gradient on solved BoozerExact and
BoozerLS surfaces match native's, with and without stellarator symmetry, for
both symmetry axes and Volume, Area and ToroidalFlux labels; the coil gradient
is the derivative through re-solves; upstream's ``boozerQA.py`` takes the same
first BFGS iterations in JAX as natively. Then: settings read at every
evaluation, no recompilation for new values, no implicit transfers, native's
non-finite values for a zero field, and unsupported inputs.

Problems are upstream's (``tests/geo/surface_test_helpers.get_boozer_surface``
and ``examples/2_Intermediate/boozerQA.py``): NCSX coils, BoozerExact
``mpol = ntor = 6`` on 13 x 13, BoozerLS ``mpol = ntor = 3`` on 20 x 20 with
weight 100, iota -0.406. Exact solves use ``newton_tol = 1e-10``, above the
residual's round-off floor (PR 8). Comparisons are against native objects on
native surfaces where PR 8's adjoint is native's, and against native
objectives on the same JAX surface where PR 8 deliberately differs (BoozerExact
without stellarator symmetry, where native raises; BoozerLS with a
ToroidalFlux label, where native's adjoint is not the derivative) or where the
native and JAX solves differ by more than round-off (BoozerLS without
stellarator symmetry, along a near-null direction of its DOFs).
Values and gradients agree to 1e-12 of the largest native entry (measured
worst 4e-14).
"""

from unittest_jax_support import JaxTestCase
from unittest import mock

from unittest_jax_support import (
    assert_matches_native,
    jax_compilations,
    parity_default_device,
    parity_rng,
)

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import numpy as np
from scipy.optimize import minimize

from simsopt._core.derivative import Derivative
from simsopt.configs import get_data
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil, Current
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.curveobjectives import CurveLength
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfaceobjectives import (
    Area,
    Iotas,
    MajorRadius,
    NonQuasiSymmetricRatio,
    ToroidalFlux,
    Volume,
)
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.objectives import QuadraticPenalty
from simsopt_jax.core.quasisymmetry import non_quasi_symmetric_ratio
from simsopt_jax.runtime.host_boundary import disallow_host_transfers, host_tree
from simsopt_jax_adapters.field import JaxBiotSavart
from simsopt_jax_adapters.geo.boozer_surface import JaxBoozerSurface
from simsopt_jax_adapters.geo.surface_objectives import JaxNonQuasiSymmetricRatio
from simsopt_jax_adapters.geo.surface_specs import surface_spec_from_surface

_IOTA = -0.406
_RTOL = 1e-12
_WEIGHT = 100.0


@dataclass(frozen=True)
class _Problem:
    coils: list
    currents: list
    boozer: BoozerSurface | JaxBoozerSurface


def _G0(nfp: int, base_currents) -> float:
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    return 2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi))


def _problem(
    jax_surface: bool,
    boozer_type: str = "exact",
    label: str = "Volume",
    *,
    stellsym: bool = True,
    solve: bool = True,
) -> _Problem:
    """Upstream's NCSX problem, solved from the axis fit, as a native
    ``BoozerSurface`` or a ``JaxBoozerSurface``."""
    _, base_currents, axis, nfp, bs = get_data("ncsx")
    mpol = 6 if boozer_type == "exact" else 3
    npoints = 13 if boozer_type == "exact" else 20
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=mpol,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, npoints, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, npoints, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1, flip_theta=True)
    labels = {
        "Volume": lambda: Volume(surface),
        "Area": lambda: Area(surface, nphi=31, ntheta=31),
        "ToroidalFlux": lambda: ToroidalFlux(
            surface, BiotSavart(bs.coils), nphi=51, ntheta=51
        ),
    }
    label_object = labels[label]()
    options = (
        {"verbose": False, "newton_tol": 1e-10}
        if boozer_type == "exact"
        else {"verbose": False}
    )
    weight = None if boozer_type == "exact" else _WEIGHT
    boozer = (
        JaxBoozerSurface(
            JaxBiotSavart(bs.coils),
            surface,
            label_object,
            label_object.J(),
            weight,
            options,
        )
        if jax_surface
        else BoozerSurface(bs, surface, label_object, label_object.J(), weight, options)
    )
    if solve:
        boozer.run_code(_IOTA, G=_G0(nfp, base_currents))
        assert boozer.res["success"], f"{boozer_type} {label} solve failed"
    return _Problem(bs.coils, base_currents, boozer)


def _native_objective(problem: _Problem, **kwargs) -> NonQuasiSymmetricRatio:
    return NonQuasiSymmetricRatio(problem.boozer, BiotSavart(problem.coils), **kwargs)


def _jax_objective(problem: _Problem, **kwargs) -> JaxNonQuasiSymmetricRatio:
    return JaxNonQuasiSymmetricRatio(
        problem.boozer, JaxBiotSavart(problem.coils), **kwargs
    )


# --- the ratio and its partial derivatives ----------------------------------------


# --- the objective on solved surfaces ---------------------------------------------

# name: (boozer type, label, stellsym, objective arguments)
_NATIVE_CASES = {
    "exact-volume": ("exact", "Volume", True, {}),
    "exact-volume-quasi-poloidal": ("exact", "Volume", True, {"quasi_poloidal": True}),
    "exact-area-own-grid": ("exact", "Area", True, {}),
    "exact-flux-own-grid": ("exact", "ToroidalFlux", True, {"sDIM": 12}),
    "ls-volume": ("ls", "Volume", True, {}),
}


# Native's objective on the same JAX surface is the reference where PR 8's adjoint deliberately
# differs from native's (BoozerExact without stellarator symmetry, a BoozerLS ToroidalFlux label)
# and where the solves differ beyond round-off (BoozerLS without stellarator symmetry: the native
# and JAX solves end on parametrizations of the surface 20 % apart in the DOFs along a near-null
# direction, with iota equal to 6e-14, so J on them differs by 4e-8).
_JAX_SURFACE_CASES = {
    "exact-flux-nonsym": ("exact", "ToroidalFlux", False, {}),
    "ls-flux": ("ls", "ToroidalFlux", True, {}),
    "ls-volume-nonsym-quasi-poloidal": (
        "ls",
        "Volume",
        False,
        {"quasi_poloidal": True},
    ),
}


def _boozer_qa(jax_lane: bool, *, newton_maxiter: int = 40):
    """Upstream's ``examples/2_Intermediate/boozerQA.py`` up to ``minimize``,
    with the JAX switch of ``boozerQA_jax.py``; returns ``fun``, ``x0`` and the
    evaluations ``fun`` records."""
    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    G0 = _G0(nfp, base_currents)
    s = SurfaceXYZTensorFourier(
        mpol=6,
        ntor=6,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, 13, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 13, endpoint=False),
    )
    s.fit_to_curve(ma, 0.1, flip_theta=True)
    vol = Volume(s)
    vol_target = vol.J()
    options = {"verbose": False, "newton_maxiter": newton_maxiter}
    if jax_lane:
        boozer_surface = JaxBoozerSurface(
            JaxBiotSavart(bs.coils), s, vol, vol_target, options=options
        )
    else:
        boozer_surface = BoozerSurface(bs, s, vol, vol_target, options=options)
    res = boozer_surface.solve_residual_equation_exactly_newton(
        tol=1e-13, maxiter=20, iota=_IOTA, G=G0
    )
    assert res["success"]
    mr = MajorRadius(boozer_surface)
    ls = [CurveLength(c) for c in base_curves]
    J_major_radius = QuadraticPenalty(mr, float(np.asarray(mr.J())), "identity")
    J_iotas = QuadraticPenalty(Iotas(boozer_surface), res["iota"], "identity")
    if jax_lane:
        J_nonQSRatio = JaxNonQuasiSymmetricRatio(
            boozer_surface, JaxBiotSavart(bs.coils)
        )
    else:
        J_nonQSRatio = NonQuasiSymmetricRatio(boozer_surface, BiotSavart(bs.coils))
    total_length = ls[0] + ls[1] + ls[2]
    Jls = QuadraticPenalty(total_length, float(np.asarray(total_length.J())), "max")
    JF = J_nonQSRatio + J_iotas + J_major_radius + Jls
    base_currents[0].fix_all()
    evaluations = []

    def fun(dofs):
        sdofs_prev = boozer_surface.surface.x
        iota_prev = boozer_surface.res["iota"]
        G_prev = boozer_surface.res["G"]
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        if not boozer_surface.res["success"]:
            J = 1e3
            boozer_surface.surface.x = sdofs_prev
            boozer_surface.res["iota"] = iota_prev
            boozer_surface.res["G"] = G_prev
        evaluations.append(
            (
                J,
                grad,
                boozer_surface.res["success"],
                boozer_surface.surface.get_dofs().copy(),
                boozer_surface.res["iota"],
                boozer_surface.res["G"],
            )
        )
        return J, grad

    return fun, np.asarray(JF.x, dtype=np.float64), evaluations


# --- settings, compilation, transfers and the boundary --------------------------


class TestSurfaceObjectivesJax(JaxTestCase):
    def test_ratio_and_partials_match_native(self):
        """The kernel at a native solved surface: native's ``J``,
        ``dJ_by_dsurfacecoefficients`` and direct coil derivative
        ``B_vjp(dJ_by_dB)``. (Native's BoozerExact adjoint raises without
        stellarator symmetry, so that surface is a BoozerLS one.)"""
        for case_id_0, quasi_poloidal in zip(["QA", "QP"], [False, True], strict=True):
            for case_id_1, (boozer_type, stellsym) in zip(
                ["exact-stellsym", "ls-nonsym"],
                [("exact", True), ("ls", False)],
                strict=True,
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    quasi_poloidal=quasi_poloidal,
                    case_id_1=case_id_1,
                    boozer_type=boozer_type,
                    stellsym=stellsym,
                ), self.case() as patches:
                    self._case_ratio_and_partials_match_native(
                        quasi_poloidal, boozer_type, stellsym, patches
                    )

    def _case_ratio_and_partials_match_native(
        self, quasi_poloidal, boozer_type, stellsym, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native = _problem(False, boozer_type, stellsym=stellsym)
        objective = _native_objective(native, quasi_poloidal=quasi_poloidal)
        value = objective.J()  # also leaves the field at the auxiliary surface's points
        dsurface = objective.dJ_by_dsurfacecoefficients()
        direct = objective.biotsavart.B_vjp(objective.dJ_by_dB().reshape((-1, 3)))

        field = JaxBiotSavart(native.coils)
        jax_value, jax_dsurface, jax_dcoils = non_quasi_symmetric_ratio(
            surface_spec_from_surface(objective.surface),
            field.coil_set_spec(),
            axis=objective.axis,
        )
        jax_direct = field.coil_cotangents_to_derivative(
            jax_dcoils.field_inputs(), jax_dcoils.coil_index_lists()
        )
        assert_matches_native(jax_value, value, "J")
        assert_matches_native(jax_dsurface, dsurface, "dJ/dsurface")
        assert_matches_native(
            jax_direct(field), direct(objective.biotsavart), "direct coil derivative"
        )

    def test_zero_field_gives_native_non_finite_values(self):
        """Zero currents: native's ``0/0`` ratio and non-finite partials."""
        native = _problem(False, solve=False)
        for current in native.currents:
            current.local_full_x = np.zeros_like(current.local_full_x)
        objective = _native_objective(native)
        objective.biotsavart.set_points(objective.surface.gamma().reshape((-1, 3)))
        with np.errstate(divide="ignore", invalid="ignore"):
            dsurface = objective.dJ_by_dsurfacecoefficients()
            direct = objective.biotsavart.B_vjp(objective.dJ_by_dB().reshape((-1, 3)))(
                objective.biotsavart
            )
        field = JaxBiotSavart(native.coils)
        value, jax_dsurface, jax_dcoils = non_quasi_symmetric_ratio(
            surface_spec_from_surface(objective.surface), field.coil_set_spec(), axis=0
        )
        jax_direct = cast(
            np.ndarray,
            field.coil_cotangents_to_derivative(
                jax_dcoils.field_inputs(), jax_dcoils.coil_index_lists()
            )(field),
        )
        self.assertTrue(np.isnan(host_tree(value)), "np.isnan(host_tree(value))")
        for name, actual, expected in (
            ("dJ/dsurface", jax_dsurface, dsurface),
            ("direct", jax_direct, direct),
        ):
            np.testing.assert_array_equal(
                np.isnan(host_tree(actual)),
                np.isnan(expected),
                err_msg=f"{name} NaN pattern",
            )
            self.assertTrue(
                not np.any(np.isfinite(expected)), f"native {name} has finite entries"
            )

    def test_degenerate_normals_give_native_non_finite_values(self):
        """A collapsed auxiliary surface keeps the native undefined ratio and
        the adjoint's rejection of non-finite partials."""
        problem = _problem(False)
        expected, objective = _native_objective(problem), _jax_objective(problem)
        problem.boozer.surface.x = np.zeros_like(problem.boozer.surface.x)
        # Native BoozerSurface does not depend on the surface: changing its
        # DOFs leaves the completed solve in force until a coil notification.
        self.assertTrue(
            not problem.boozer.need_to_run_code, "not problem.boozer.need_to_run_code"
        )
        expected.recompute_bell()
        objective.recompute_bell()
        with np.errstate(divide="ignore", invalid="ignore"):
            for candidate in (expected, objective):
                with self.assertRaisesRegex(
                    ValueError, "array must not contain infs or NaNs"
                ):
                    candidate.J()
                self.assertTrue(
                    np.isnan(np.asarray(candidate.J())),
                    "the cached ratio must remain undefined",
                )
                with self.assertRaisesRegex(
                    ValueError, "array must not contain infs or NaNs"
                ):
                    candidate.dJ()

    def test_objective_matches_native(self):
        """Value and coil gradient on a ``JaxBoozerSurface`` as native's on a native
        ``BoozerSurface`` solved from the same start, and on the native surface
        itself (the objective also works with native solves)."""
        for name in _NATIVE_CASES:
            with self.subTest(name=name), self.case() as patches:
                self._case_objective_matches_native(name, patches)

    def _case_objective_matches_native(self, name, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        boozer_type, label, stellsym, kwargs = _NATIVE_CASES[name]
        native, jax_problem = (
            _problem(lane, boozer_type, label, stellsym=stellsym)
            for lane in (False, True)
        )
        expected = _native_objective(native, **kwargs)
        for surface_name, problem in (
            ("JAX surface", jax_problem),
            ("native surface", native),
        ):
            objective = _jax_objective(problem, **kwargs)
            assert_matches_native(
                objective.J(), expected.J(), f"{name} J on the {surface_name}"
            )
            assert_matches_native(
                objective.dJ(),
                expected.dJ(),
                f"{name} coil gradient on the {surface_name}",
            )

    def test_objective_matches_native_on_the_jax_surface(self):
        """The non-QS objective agrees with native when both use a JAX surface solve."""
        for name in _JAX_SURFACE_CASES:
            with self.subTest(name=name), self.case() as patches:
                self._case_objective_matches_native_on_the_jax_surface(name, patches)

    def _case_objective_matches_native_on_the_jax_surface(self, name, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        boozer_type, label, stellsym, kwargs = _JAX_SURFACE_CASES[name]
        problem = _problem(True, boozer_type, label, stellsym=stellsym)
        expected, objective = (
            _native_objective(problem, **kwargs),
            _jax_objective(problem, **kwargs),
        )
        assert_matches_native(objective.J(), expected.J(), f"{name} J")
        assert_matches_native(objective.dJ(), expected.dJ(), f"{name} coil gradient")

    def test_coil_gradient_is_the_derivative_through_re_solves(self):
        """Central differences of ``J`` through re-solves (``run_code`` from the
        solution) along a random direction over every free coil DOF, each moved by
        a fraction of its own size."""
        for boozer_type, label in [("exact", "Volume"), ("ls", "ToroidalFlux")]:
            with self.subTest(
                boozer_type=boozer_type, label=label
            ), self.case() as patches:
                self._case_coil_gradient_is_the_derivative_through_re_solves(
                    boozer_type, label, patches
                )

    def _case_coil_gradient_is_the_derivative_through_re_solves(
        self, boozer_type, label, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        problem = _problem(True, boozer_type, label)
        objective = _jax_objective(problem)
        x0 = np.asarray(objective.x, dtype=np.float64)
        direction = parity_rng(11).standard_normal(x0.size) * np.maximum(
            np.abs(x0), 1.0
        )
        adjoint = np.asarray(objective.dJ()) @ direction
        step = 1e-6

        def value(sign: float) -> float:
            objective.x = x0 + sign * step * direction
            self.assertTrue(
                problem.boozer.need_to_run_code, "problem.boozer.need_to_run_code"
            )
            J = float(np.asarray(objective.J()))
            self.assertTrue(
                problem.boozer.res["success"], 'problem.boozer.res["success"]'
            )
            return J

        central = (value(1.0) - value(-1.0)) / (2 * step)
        # Truncation and the re-solves' tolerances bound the agreement (measured below 1e-8 relative).
        self.assertTrue(
            abs(adjoint - central) <= 1e-7 * abs(central),
            f"adjoint {adjoint} != difference {central}",
        )

    def test_boozer_qa_first_iterations_match_native(self):
        """Three BFGS iterations of upstream's boozerQA.py (native defaults:
        Boozer ``newton_tol`` 1e-13, cap 40): the same evaluations, each solve
        successful, values, gradients and surfaces to round-off. At 1e-13 the two
        lanes can take different Newton step counts (the residual's round-off
        floor), not different solutions."""
        runs = {}
        for lane in (False, True):
            fun, x0, evaluations = _boozer_qa(lane)
            result = minimize(
                fun, x0, jac=True, method="BFGS", options={"maxiter": 3}, tol=1e-15
            )
            runs[lane] = (result, evaluations)
        (native, native_evaluations), (jax_result, jax_evaluations) = (
            runs[False],
            runs[True],
        )
        self.assertTrue(
            jax_result.nit == native.nit == 3 and jax_result.nfev == native.nfev,
            "jax_result.nit == native.nit == 3 and jax_result.nfev == native.nfev",
        )
        for k, (jax_eval, native_eval) in enumerate(
            zip(jax_evaluations, native_evaluations)
        ):
            self.assertTrue(
                jax_eval[2] and native_eval[2], f"evaluation {k}: a Boozer solve failed"
            )
            assert_matches_native(
                jax_eval[0], native_eval[0], f"evaluation {k} J", 1e-10
            )
            assert_matches_native(
                jax_eval[1], native_eval[1], f"evaluation {k} gradient", 1e-10
            )
            assert_matches_native(
                jax_eval[3], native_eval[3], f"evaluation {k} surface", 1e-10
            )
        assert_matches_native(jax_result.x, native.x, "BFGS iterate", 1e-10)

    def test_boozer_qa_failed_solve_restores_the_previous_surface_as_natively(self):
        """A one-step outer solve cap triggers upstream's actual failure path:
        return 1e3 with the computed gradient, restoring surface, iota and G."""
        runs = []
        for lane in (False, True):
            fun, x0, evaluations = _boozer_qa(lane, newton_maxiter=1)
            fun(x0)
            initial = evaluations[-1]
            self.assertTrue(
                initial[2], "the initial state must have a successful solve"
            )
            direction = parity_rng(3).uniform(size=x0.size)
            value, gradient = fun(x0 + 1e-4 * direction)
            failed = evaluations[-1]
            self.assertTrue(
                not failed[2], "the one-step cap must trigger the real restore path"
            )
            self.assertTrue(
                value == 1e3 and np.asarray(gradient).shape == x0.shape,
                "value == 1e3 and np.asarray(gradient).shape == x0.shape",
            )
            np.testing.assert_array_equal(
                failed[3], initial[3], err_msg="previous surface was not restored"
            )
            self.assertTrue(
                failed[4:] == initial[4:], "previous iota and G were not restored"
            )
            runs.append(failed)
        assert_matches_native(
            runs[1][1], runs[0][1], "gradient returned on failure", 1e-10
        )

    def test_settings_are_read_at_every_evaluation(self):
        """The axis, auxiliary surface and field in force at an evaluation are
        used, as natively (both cache their value until a recompute notification)."""
        native, jax_problem = _problem(False), _problem(True)
        objectives = (_native_objective(native), _jax_objective(jax_problem))
        for objective, problem in zip(objectives, (native, jax_problem)):
            objective.J()
            objective.axis = 1
            surface = objective.surface
            objective.surface = SurfaceXYZTensorFourier(
                mpol=surface.mpol,
                ntor=surface.ntor,
                stellsym=surface.stellsym,
                nfp=surface.nfp,
                quadpoints_phi=np.linspace(0, 1 / surface.nfp, 18, endpoint=False),
                quadpoints_theta=np.linspace(0, 1, 22, endpoint=False),
                dofs=surface.dofs,
            )
            field_class = (
                JaxBiotSavart
                if isinstance(objective, JaxNonQuasiSymmetricRatio)
                else BiotSavart
            )
            objective.biotsavart = field_class(problem.coils[::2])
            objective.recompute_bell()
        assert_matches_native(
            objectives[1].J(), objectives[0].J(), "J with new settings"
        )
        assert_matches_native(
            objectives[1].dJ(), objectives[0].dJ(), "coil gradient with new settings"
        )

    def test_adjoint_failure_keeps_the_computed_value_as_natively(self):
        """Native caches the ratio before looking up the adjoint factors. A
        failed gradient evaluation still leaves that value available to J()."""
        patches = self.patches
        problem = _problem(False)
        expected = _native_objective(problem).J()
        objectives = (_native_objective(problem), _jax_objective(problem))
        patches.enter_context(mock.patch.dict(problem.boozer.res))
        del problem.boozer.res["PLU"]
        for objective in objectives:
            with self.assertRaisesRegex(KeyError, "PLU"):
                objective.J()
            assert_matches_native(
                objective.J(), expected, "cached J after adjoint failure"
            )
            with self.assertRaisesRegex(KeyError, "PLU"):
                objective.dJ()

    def test_shared_surface_dofs_and_fixed_coil_partials_match_native(self):
        """The auxiliary grid shares the solved DOFs, and fixing a coil DOF
        changes the free gradient without dropping its partial derivative."""
        problem = _problem(True)
        expected, objective = _native_objective(problem), _jax_objective(problem)
        self.assertTrue(
            objective.surface.dofs is problem.boozer.surface.dofs,
            "objective.surface.dofs is problem.boozer.surface.dofs",
        )
        objective.dJ()
        problem.currents[0].fix_all()
        problem.currents[1].local_full_x = 1.001 * problem.currents[1].local_full_x
        self.assertTrue(
            problem.boozer.need_to_run_code, "problem.boozer.need_to_run_code"
        )
        self.assertTrue(
            objective._J is None and objective._dJ is None,
            "objective._J is None and objective._dJ is None",
        )
        assert_matches_native(
            objective.J(), expected.J(), "J after a shared coil change"
        )
        self.assertTrue(problem.boozer.res["success"], 'problem.boozer.res["success"]')
        # The native decorator accepts ``partials``; its undecorated signature
        # does not express that keyword to the type checker.
        actual_partials = cast(Callable[..., Derivative], objective.dJ)(partials=True)
        native_partials = cast(Callable[..., Derivative], expected.dJ)(partials=True)
        self.assertTrue(
            set(actual_partials.data) == set(native_partials.data),
            "partial owners differ",
        )
        for owner in native_partials.data:
            assert_matches_native(
                actual_partials.data[owner],
                native_partials.data[owner],
                f"partials for {owner}",
            )
        assert_matches_native(
            objective.dJ(), expected.dJ(), "free gradient after fixing a current"
        )
        np.testing.assert_array_equal(
            objective.surface.get_dofs(), problem.boozer.surface.get_dofs()
        )

    def test_coils_only_in_the_objective_field_are_parents(self):
        """A coil that ``bs`` holds and the Boozer surface's field does not is
        part of ``x``, invalidates ``J`` when it changes, and carries the direct
        derivative in ``dJ`` (the surface does not depend on it, so no re-solve)."""
        problem = _problem(True)
        ring = CurveXYZFourier(64, 1)
        ring_dofs = {"xc(1)": 2.5, "ys(1)": 2.5, "zc(0)": 0.4}
        ring.x = np.array([ring_dofs.get(name, 0.0) for name in ring.local_dof_names])
        extra = Coil(ring, Current(2.0e4))
        objective = JaxNonQuasiSymmetricRatio(
            problem.boozer, JaxBiotSavart([*problem.coils, extra])
        )
        extra_names = set(extra.dof_names)
        self.assertTrue(
            extra_names <= set(objective.dof_names),
            "extra_names <= set(objective.dof_names)",
        )
        J0 = float(np.asarray(objective.J()))
        gradient = dict(
            zip(objective.dof_names, np.asarray(objective.dJ()), strict=True)
        )
        x0 = np.asarray(extra.x, dtype=np.float64)
        direction = parity_rng(12).standard_normal(x0.size) * np.maximum(
            np.abs(x0), 1.0
        )
        adjoint = sum(
            gradient[name] * d
            for name, d in zip(extra.dof_names, direction, strict=True)
        )
        step = 1e-6

        def value(sign: float) -> float:
            extra.x = x0 + sign * step * direction
            self.assertTrue(objective._J is None, "objective._J is None")
            self.assertTrue(
                not problem.boozer.need_to_run_code,
                "not problem.boozer.need_to_run_code",
            )
            return float(np.asarray(objective.J()))

        central = (value(1.0) - value(-1.0)) / (2 * step)
        self.assertTrue(central != 0.0, "central != 0.0")
        self.assertTrue(
            abs(adjoint - central) <= 1e-7 * abs(central),
            f"adjoint {adjoint} != difference {central}",
        )
        extra.x = x0
        self.assertTrue(
            float(np.asarray(objective.J())) == J0,
            "float(np.asarray(objective.J())) == J0",
        )

    def test_evaluation_preserves_the_fields_existing_points(self):
        """The documented difference from native: an objective does not change
        its JAX field's evaluation points, including after a re-solve."""
        problem = _problem(True)
        objective = _jax_objective(problem)
        points = np.array([[1.2, 0.1, 0.0], [1.3, 0.0, 0.2]])
        objective.biotsavart.set_points(points)
        expected = _native_objective(problem)
        assert_matches_native(
            objective.J(), expected.J(), "J with existing field points"
        )
        assert_matches_native(
            objective.dJ(), expected.dJ(), "gradient with existing field points"
        )
        np.testing.assert_array_equal(objective.biotsavart.get_points_cart(), points)
        problem.currents[0].local_full_x = 1.001 * problem.currents[0].local_full_x
        objective.dJ()
        np.testing.assert_array_equal(objective.biotsavart.get_points_cart(), points)

    def test_new_values_reuse_the_compiled_programs(self):
        """New coils (a re-solve) and a new surface reuse the ratio, solve and
        adjoint programs."""
        problem = _problem(True)
        objective = _jax_objective(problem)
        objective.dJ()
        problem.currents[0].local_full_x = 1.002 * problem.currents[0].local_full_x
        with jax_compilations() as compilations:
            objective.J()
            objective.dJ()
        self.assertTrue(
            compilations == [], "new values retraced or recompiled a program"
        )
        self.assertTrue(problem.boozer.res["success"], 'problem.boozer.res["success"]')

    def test_evaluation_makes_no_implicit_transfers(self):
        """Evaluation uses explicit host/device transfers under the runtime transfer guard."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_evaluation_makes_no_implicit_transfers(parity_lane, patches)

    def _case_evaluation_makes_no_implicit_transfers(self, parity_lane, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        native = _problem(False)
        with parity_default_device(parity_lane):
            jax_problem = _problem(True)
            objective = _jax_objective(jax_problem)
            with disallow_host_transfers():
                value, gradient = objective.J(), objective.dJ()
                jax_problem.currents[0].local_full_x = (
                    1.001 * jax_problem.currents[0].local_full_x
                )
                objective.J()
                objective.dJ()
        coils = objective.biotsavart.coil_set_spec()
        self.assertTrue(
            {
                device.platform
                for leaf in jax.tree.leaves(coils)
                for device in leaf.devices()
            }
            == {parity_lane},
            "{device.platform for leaf in jax.tree.leaves(coils) for device in leaf.devices()} == {parity_lane}",
        )
        expected = _native_objective(native)
        assert_matches_native(value, expected.J(), f"J on {parity_lane}", 1e-10)
        assert_matches_native(
            gradient, expected.dJ(), f"coil gradient on {parity_lane}", 1e-10
        )

    def test_boundary_refuses_unsupported_inputs(self):
        """Ratio construction rejects unsupported surfaces and fields."""
        _, _, axis, nfp, bs = get_data("ncsx")
        xyz = SurfaceXYZFourier(
            mpol=2,
            ntor=2,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1 / nfp, 6, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 7, endpoint=False),
        )
        xyz.fit_to_curve(axis, 0.1, flip_theta=True)
        # As natively, only SurfaceXYZTensorFourier.
        with self.assertRaises(AssertionError):
            JaxNonQuasiSymmetricRatio(
                JaxBoozerSurface(JaxBiotSavart(bs.coils), xyz, Volume(xyz), 1.0, 1.0),
                JaxBiotSavart(bs.coils),
            )
        with self.assertRaisesRegex(TypeError, "JaxBiotSavart"):
            JaxNonQuasiSymmetricRatio(_problem(False, solve=False).boozer, bs)

    def test_coil_gradient_taylor(self):
        """Non-QS coil gradients converge through exact and flux-label penalty re-solves."""
        for boozer_type, label in (("exact", "Volume"), ("ls", "ToroidalFlux")):
            with self.subTest(boozer_type=boozer_type, label=label), self.case():
                self._case_coil_gradient_taylor(boozer_type, label)

    def _case_coil_gradient_taylor(self, boozer_type, label):
        """Run one row with case-local objects released before runtime cleanup."""
        problem = _problem(True, boozer_type, label)
        objective = _jax_objective(problem)
        x0 = np.asarray(objective.x, dtype=np.float64)
        direction = parity_rng(11).standard_normal(x0.size) * np.maximum(
            np.abs(x0), 1.0
        )
        adjoint = np.asarray(objective.dJ()) @ direction
        step = 1e-6
        solution = problem.boozer.surface.get_dofs().copy()
        solved_iota, solved_G = problem.boozer.res["iota"], problem.boozer.res["G"]

        def value(sign: float) -> float:
            problem.boozer.surface.set_dofs(solution)
            problem.boozer.res["iota"], problem.boozer.res["G"] = solved_iota, solved_G
            objective.x = x0 + sign * step * direction
            self.assertTrue(
                problem.boozer.need_to_run_code, "problem.boozer.need_to_run_code"
            )
            J = float(np.asarray(objective.J()))
            self.assertTrue(
                problem.boozer.res["success"], 'problem.boozer.res["success"]'
            )
            return J

        central = (value(1.0) - value(-1.0)) / (2 * step)
        # Truncation and the re-solves' tolerances bound the agreement (measured below 1e-8 relative).
        self.assertTrue(
            abs(adjoint - central) <= 1e-7 * abs(central),
            f"adjoint {adjoint} != difference {central}",
        )

        previous_error = 1e9
        for step in np.power(2.0, -np.arange(13, 19)):
            error = abs((value(1.0) - value(-1.0)) / (2 * step) - adjoint)
            self.assertLess(error, max(1e-9, 0.35 * previous_error))
            previous_error = error
        objective.x = x0
        problem.boozer.surface.set_dofs(solution)

    def test_objective_only_coil_gradient_taylor(self):
        """A shrinking-step Taylor check for a coil that ``bs`` holds and the Boozer surface's field does not is
        part of ``x``, invalidates ``J`` when it changes, and carries the direct
        derivative in ``dJ`` (the surface does not depend on it, so no re-solve)."""
        problem = _problem(True)
        ring = CurveXYZFourier(64, 1)
        ring_dofs = {"xc(1)": 2.5, "ys(1)": 2.5, "zc(0)": 0.4}
        ring.x = np.array([ring_dofs.get(name, 0.0) for name in ring.local_dof_names])
        extra = Coil(ring, Current(2.0e4))
        objective = JaxNonQuasiSymmetricRatio(
            problem.boozer, JaxBiotSavart([*problem.coils, extra])
        )
        extra_names = set(extra.dof_names)
        self.assertTrue(
            extra_names <= set(objective.dof_names),
            "extra_names <= set(objective.dof_names)",
        )
        J0 = float(np.asarray(objective.J()))
        gradient = dict(
            zip(objective.dof_names, np.asarray(objective.dJ()), strict=True)
        )
        x0 = np.asarray(extra.x, dtype=np.float64)
        direction = parity_rng(12).standard_normal(x0.size) * np.maximum(
            np.abs(x0), 1.0
        )
        adjoint = sum(
            gradient[name] * d
            for name, d in zip(extra.dof_names, direction, strict=True)
        )
        step = 1e-6

        def value(sign: float) -> float:
            extra.x = x0 + sign * step * direction
            self.assertTrue(objective._J is None, "objective._J is None")
            self.assertTrue(
                not problem.boozer.need_to_run_code,
                "not problem.boozer.need_to_run_code",
            )
            return float(np.asarray(objective.J()))

        central = (value(1.0) - value(-1.0)) / (2 * step)
        self.assertTrue(central != 0.0, "central != 0.0")
        self.assertTrue(
            abs(adjoint - central) <= 1e-7 * abs(central),
            f"adjoint {adjoint} != difference {central}",
        )
        previous_error = 1e9
        for step in np.power(2.0, -np.arange(13, 19)):
            error = abs((value(1.0) - value(-1.0)) / (2 * step) - adjoint)
            self.assertLess(error, max(1e-9, 0.35 * previous_error))
            previous_error = error
        extra.x = x0
        self.assertTrue(
            float(np.asarray(objective.J())) == J0,
            "float(np.asarray(objective.J())) == J0",
        )
