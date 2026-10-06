"""The official BoozerQA problem runs upstream's analytic exact Newton route.

Upstream's ``examples/2_Intermediate/boozerQA.py`` re-solves the Boozer surface
with ``solve_residual_equation_exactly_newton`` on every objective evaluation
(dense analytic Jacobian, direct solve, undamped step) and returns ``J = 1e3``
with the previous surface restored when that solve fails.  The checks here hold
:class:`~simsopt_jax_adapters.geo.boozer_qa_problem.BoozerQAProblem` against two
references: the traceable-session route (the JAX route the production
compute-graph optimizers still run) and the native library's own objective,
which is upstream's.

Both references are built live on the reduced NCSX construction (coil and axis
order 3, eight points per period, ``mpol = ntor = 2``).  The two agreement tests
run the official 1e-13 Newton tolerance on it, which the exact Newton solve
reaches from the seed and from every moved state, so the reduced scale still
exercises the tolerance the official run uses.
"""

from __future__ import annotations

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    enable_non_strict_jax_backend,
)

from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np
import pytest
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (
    BoozerResidual,
    BoozerSurface,
    CurveLength,
    Iotas,
    MajorRadius,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
    Volume,
)
from simsopt.objectives import QuadraticPenalty
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_QA_INITIAL_IOTA,
    OFFICIAL_QA_NEWTON_MAXITER,
    OFFICIAL_QA_NEWTON_TOLERANCE,
    OFFICIAL_QA_NON_QS_RESOLUTION,
    OFFICIAL_QA_RESIDUAL_WEIGHT,
    OFFICIAL_QA_SURFACE_DISTANCE,
    boozer_qa_outer_objective_config,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_qa_problem import BoozerQAProblem
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.single_stage_exact_analytic import (
    INNER_FAILURE_VALUE,
    HostConstructionBoozerSurfaceJAX,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    make_traceable_objective_session,
)

#: Agreement required at the same coil state: the objective to 1e-13
#: relative, the gradient to 1e-11 relative to its maximum component.  Measured
#: at the official scale on these four states: value 1.1e-14 to 4.3e-14 against
#: the session route and 2.0e-14 to 2.5e-14 against native; gradient 5.4e-13 to
#: 1.5e-12 against the session route and 5.5e-13 to 3.8e-12 against native.
#: The bars sit 2.3x (value) and 2.6x (gradient) above the worst measurement of
#: two DIFFERENT exact-Newton algorithms in fp64.
VALUE_RELATIVE_TOLERANCE = 1.0e-13
GRADIENT_RELATIVE_TOLERANCE = 1.0e-11
#: The solved Boozer state (surface dofs, iota, G) agrees to 1e-13 ABSOLUTE,
#: the requirement's own figure for the inner solution.
SOLVED_STATE_ABSOLUTE_TOLERANCE = 1.0e-13
#: The targets frozen at the seed solution (iota target, seed volume) agree to
#: 1e-13 RELATIVE: the same inner solution read through the same labels.
FROZEN_TARGET_RELATIVE_TOLERANCE = 1.0e-13
#: Coil perturbations of the seed state, small enough that the exact Newton
#: solve converges at the official 1e-13 tolerance from its warm start and large
#: enough that the objective and its gradient both move.
_MOVED_STATE_SCALES = (2.0e-4, 5.0e-4, 1.0e-3)
_MOVED_STATE_SEED = 20260920
#: The reduced NCSX construction and the tolerance the reduced-scale failure
#: and transfer-guard checks run.
_REDUCED_COILS = {"coil_order": 3, "magnetic_axis_order": 3, "points_per_period": 8}
_REDUCED_RESOLUTION = 2
_REDUCED_NEWTON_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class _Construction:
    """One fresh NCSX coil set and its fitted volume-labelled seed surface."""

    base_curves: list
    native_field: BiotSavart
    surface: SurfaceXYZTensorFourier
    nfp: int
    initial_G: float


def _construction() -> _Construction:
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx", **_REDUCED_COILS
    )
    base_currents[0].fix_all()
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    initial_G = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    surface = SurfaceXYZTensorFourier(
        mpol=_REDUCED_RESOLUTION,
        ntor=_REDUCED_RESOLUTION,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0, 1.0 / nfp, 2 * _REDUCED_RESOLUTION + 1, endpoint=False
        ),
        quadpoints_theta=np.linspace(
            0.0, 1.0, 2 * _REDUCED_RESOLUTION + 1, endpoint=False
        ),
    )
    surface.fit_to_curve(magnetic_axis, OFFICIAL_QA_SURFACE_DISTANCE, flip_theta=True)
    return _Construction(base_curves, native_field, surface, int(nfp), float(initial_G))


def _boozer_options(tolerance: float) -> dict[str, object]:
    return {
        "newton_maxiter": OFFICIAL_QA_NEWTON_MAXITER,
        "newton_tol": tolerance,
        "verbose": False,
    }


def _states(initial_parameters: np.ndarray) -> tuple[np.ndarray, ...]:
    """The start state followed by three moved coil states."""
    generator = np.random.default_rng(_MOVED_STATE_SEED)
    moved = tuple(
        initial_parameters
        * (1.0 + scale * generator.standard_normal(initial_parameters.size))
        for scale in _MOVED_STATE_SCALES
    )
    return (np.array(initial_parameters, copy=True), *moved)


def _official_problem(tolerance: float) -> BoozerQAProblem:
    """The problem the shipped mirror builds, on a fresh construction."""
    built = _construction()
    return BoozerQAProblem(
        base_curves=built.base_curves,
        native_field=built.native_field,
        surface=built.surface,
        nfp=built.nfp,
        initial_G=built.initial_G,
        initial_iota=OFFICIAL_QA_INITIAL_IOTA,
        boozer_options=_boozer_options(tolerance),
        non_qs_resolution=OFFICIAL_QA_NON_QS_RESOLUTION,
        residual_weight=OFFICIAL_QA_RESIDUAL_WEIGHT,
    )


@dataclass(frozen=True)
class _NativeReference:
    """Upstream's objective over the native exact Newton solve, with its policy."""

    solver: BoozerSurface
    objective: object
    initial_parameters: np.ndarray
    initial_solution_success: bool
    initial_iota: float
    initial_volume: float

    def value_and_grad(self, parameters: np.ndarray) -> tuple[float, np.ndarray]:
        """Upstream boozerQA.py's ``fun``: ``J = 1e3`` and a restore on failure."""
        previous_surface = np.asarray(self.solver.surface.x, dtype=np.float64)
        previous_iota = float(self.solver.res["iota"])
        previous_G = float(self.solver.res["G"])
        self.objective.x = parameters
        value = float(self.objective.J())
        gradient = np.asarray(self.objective.dJ(), dtype=np.float64)
        if not bool(self.solver.res["success"]):
            value = 1.0e3
            self.solver.surface.x = previous_surface
            self.solver.res["iota"] = previous_iota
            self.solver.res["G"] = previous_G
        return value, gradient


def _native_reference(tolerance: float) -> _NativeReference:
    """Upstream's boozerQA.py objective with the native library, on a fresh construction."""
    built = _construction()
    volume = Volume(built.surface)
    solver = BoozerSurface(
        built.native_field,
        built.surface,
        volume,
        float(volume.J()),
        options=_boozer_options(tolerance),
    )
    initial_solution = solver.solve_residual_equation_exactly_newton(
        tol=tolerance,
        maxiter=OFFICIAL_QA_NEWTON_MAXITER,
        iota=OFFICIAL_QA_INITIAL_IOTA,
        G=built.initial_G,
    )
    iota_target = float(initial_solution["iota"])
    initial_volume = float(volume.J())
    major_radius = MajorRadius(solver)
    total_length = sum(CurveLength(curve) for curve in built.base_curves)
    objective = (
        NonQuasiSymmetricRatio(
            solver,
            BiotSavart(built.native_field.coils),
            sDIM=OFFICIAL_QA_NON_QS_RESOLUTION,
        )
        + OFFICIAL_QA_RESIDUAL_WEIGHT * BoozerResidual(solver, built.native_field)
        + QuadraticPenalty(Iotas(solver), iota_target, "identity")
        + QuadraticPenalty(major_radius, float(major_radius.J()), "identity")
        + QuadraticPenalty(total_length, float(total_length.J()), "max")
    )
    return _NativeReference(
        solver=solver,
        objective=objective,
        initial_parameters=np.asarray(objective.x, dtype=np.float64),
        initial_solution_success=bool(initial_solution["success"]),
        initial_iota=iota_target,
        initial_volume=initial_volume,
    )


@dataclass(frozen=True)
class _SessionRoute:
    """The traceable-session route and the targets it froze at the seed solve."""

    value_and_grad: Callable[[np.ndarray], tuple[object, object]]
    iota_target: float
    initial_volume: float


def _session_route(tolerance: float) -> _SessionRoute:
    """The traceable-session route on a fresh construction, seeded by its own solve."""
    built = _construction()
    field = BiotSavartJAX(built.native_field.coils)
    surface = built.surface
    volume = Volume(surface)
    solver = BoozerSurfaceJAX(
        field,
        surface,
        volume,
        float(volume.J()),
        options=_boozer_options(tolerance),
    )
    initial_solution = solver.run_code_traceable(
        field.coil_set_spec(),
        jax.device_put(np.asarray(surface.get_dofs(), dtype=np.float64)),
        jax.device_put(np.asarray(OFFICIAL_QA_INITIAL_IOTA, dtype=np.float64)),
        jax.device_put(np.asarray(built.initial_G, dtype=np.float64)),
    )
    solver.install_traceable_solved_runtime_state(initial_solution)
    iota_target = float(np.asarray(jax.device_get(initial_solution["iota"])))
    initial_volume = float(volume.J())
    configuration = boozer_qa_outer_objective_config(
        nfp=built.nfp,
        non_qs_resolution=OFFICIAL_QA_NON_QS_RESOLUTION,
        length_target=float(sum(CurveLength(curve).J() for curve in built.base_curves)),
        major_radius_target=float(surface.major_radius()),
        vessel_gamma=surface.gamma(),
        residual_weight=OFFICIAL_QA_RESIDUAL_WEIGHT,
    )
    session = make_traceable_objective_session(
        solver,
        field,
        iota_target,
        outer_objective_config=configuration,
    )
    controller = session.accepted_incumbent_host_value_and_grad()
    return _SessionRoute(controller.value_and_grad, iota_target, initial_volume)


def _solved_state(problem: BoozerQAProblem) -> np.ndarray:
    return np.asarray(problem._evaluator.x_inner, dtype=np.float64)


def _assert_value_and_gradient_agree(
    label: str,
    index: int,
    value: float,
    gradient: np.ndarray,
    reference_value: float,
    reference_gradient: np.ndarray,
) -> None:
    assert abs(value - reference_value) <= VALUE_RELATIVE_TOLERANCE * abs(
        reference_value
    ), f"{label} value at state {index}: {value!r} vs {reference_value!r}"
    gradient_scale = float(np.max(np.abs(reference_gradient)))
    assert gradient_scale > 0.0
    difference = float(np.max(np.abs(gradient - reference_gradient)))
    assert difference <= GRADIENT_RELATIVE_TOLERANCE * gradient_scale, (
        f"{label} gradient at state {index}: max difference {difference:.3e} "
        f"against ||g||inf {gradient_scale:.3e}"
    )


def test_official_route_matches_the_session_route_at_the_official_tolerance(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Against the route the JAX optimizers ran before, at the official tolerance.

    The same frozen targets, and the same objective and gradient at the start
    state and three moved states.  The solved Boozer state itself is pinned
    against the native library in the next test: the session controller exposes
    only its ACCEPTED incumbent, not the candidate solve of each evaluation.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    problem = _official_problem(OFFICIAL_QA_NEWTON_TOLERANCE)
    states = _states(problem.initial_coil_dofs)

    session = _session_route(OFFICIAL_QA_NEWTON_TOLERANCE)
    assert problem.iota_target == pytest.approx(
        session.iota_target, rel=FROZEN_TARGET_RELATIVE_TOLERANCE, abs=0.0
    )
    assert problem.initial_volume == pytest.approx(
        session.initial_volume, rel=FROZEN_TARGET_RELATIVE_TOLERANCE, abs=0.0
    )

    for index, state in enumerate(states):
        reference_value, reference_gradient = session.value_and_grad(state)
        value, gradient = problem.value_and_gradient(state)
        _assert_value_and_gradient_agree(
            "session",
            index,
            value,
            gradient,
            float(reference_value),
            np.asarray(reference_gradient, dtype=np.float64),
        )


def test_official_route_matches_the_native_library_at_the_official_tolerance(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """The JAX problem runs the native library's algorithm.

    The native reference is upstream's objective over
    ``solve_residual_equation_exactly_newton`` with upstream's failed-solve
    policy.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    problem = _official_problem(OFFICIAL_QA_NEWTON_TOLERANCE)
    native = _native_reference(OFFICIAL_QA_NEWTON_TOLERANCE)
    np.testing.assert_array_equal(problem.initial_coil_dofs, native.initial_parameters)
    states = _states(problem.initial_coil_dofs)

    assert problem.initial_inner_success is native.initial_solution_success is True
    assert problem.iota_target == pytest.approx(
        native.initial_iota, rel=FROZEN_TARGET_RELATIVE_TOLERANCE, abs=0.0
    )
    assert problem.initial_volume == pytest.approx(
        native.initial_volume, rel=FROZEN_TARGET_RELATIVE_TOLERANCE, abs=0.0
    )

    for index, state in enumerate(states):
        reference_value, reference_gradient = native.value_and_grad(state)
        value, gradient = problem.value_and_gradient(state)
        assert bool(native.solver.res["success"])
        native_solved = np.concatenate(
            (
                np.asarray(native.solver.surface.x, dtype=np.float64),
                np.asarray(
                    (native.solver.res["iota"], native.solver.res["G"]),
                    dtype=np.float64,
                ),
            )
        )
        np.testing.assert_allclose(
            _solved_state(problem),
            native_solved,
            rtol=0.0,
            atol=SOLVED_STATE_ABSOLUTE_TOLERANCE,
        )
        _assert_value_and_gradient_agree(
            "native",
            index,
            value,
            gradient,
            float(reference_value),
            np.asarray(reference_gradient, dtype=np.float64),
        )


def test_failed_inner_solve_reports_the_sentinel_and_restores_the_warm_start(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Upstream's failed-evaluation policy, against the native library's own.

    Upstream returns ``J = 1e3`` and restores the surface, ``iota`` and ``G``
    that the evaluation started from, while the gradient it reports is the one
    taken at the iterate the failed solve returned, before that restore
    (upstream boozerQA.py:99-109).  The native reference implements exactly
    that policy, so the failure decision, the sentinel value and the restored
    warm start are pinned against it exactly.

    The failure-path GRADIENT is deliberately not pinned to native's: a
    cap-exhausted Newton run has no converged state, and the two
    implementations return different final iterates, so there is no shared
    reference to compare at.  What is pinned is that a failed evaluation
    reports upstream's sentinel with a finite derivative rather than a NaN, and
    that the run recovers from it.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    problem = _official_problem(_REDUCED_NEWTON_TOLERANCE)
    native = _native_reference(_REDUCED_NEWTON_TOLERANCE)
    warm_start = np.array(_solved_state(problem), copy=True)

    # Coils scaled far from the seed surface: the exact Newton solve exhausts
    # its iteration cap in both implementations, so the failure policy is
    # exercised on a real failure rather than a simulated one.
    far = 4.0 * problem.initial_coil_dofs
    native_value, native_gradient = native.value_and_grad(far)
    assert not bool(native.solver.res["success"])
    assert native_value == INNER_FAILURE_VALUE

    # Both public entry points carry upstream's policy: the one the outer
    # optimizer calls and the one the endpoint report calls.
    value, gradient = problem.value_and_gradient(far)
    assert value == INNER_FAILURE_VALUE
    # Restored bitwise: the failed iterate never becomes the next warm start.
    np.testing.assert_array_equal(_solved_state(problem), warm_start)
    # A derivative at the returned iterate, not a NaN: upstream reports
    # ``JF.dJ()`` on the failed evaluation too.
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(np.asarray(native_gradient, dtype=np.float64)))

    endpoint = problem.endpoint(far)
    assert endpoint.inner_success is False
    assert endpoint.value == INNER_FAILURE_VALUE
    np.testing.assert_array_equal(_solved_state(problem), warm_start)
    # The reported physics is the restored warm start's.
    assert np.isfinite(endpoint.iota)
    assert np.isfinite(endpoint.volume)
    # The next evaluation still starts from the restored state, so a failure
    # cannot poison the run.
    recovered_value, recovered_gradient = problem.value_and_gradient(
        problem.initial_coil_dofs
    )
    assert recovered_value != INNER_FAILURE_VALUE
    assert np.all(np.isfinite(recovered_gradient))


def test_official_construction_and_evaluation_are_clean_under_the_transfer_guard(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """The strict lane's guard sees no implicit host boundary crossing.

    The strict GPU lane runs under ``JAX_TRANSFER_GUARD=disallow``
    (``examples/jax/_lane_environment.py``).  Construction, the first
    value-and-gradient -- the call that compiles -- and the endpoint report are
    all inside the guard here.  Only the host-to-device half can be exercised on
    CPU; the device-to-host half needs a real device.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")

    with jax.transfer_guard("disallow"):
        problem = _official_problem(_REDUCED_NEWTON_TOLERANCE)
        value, gradient = problem.value_and_gradient(problem.initial_coil_dofs)
        endpoint = problem.endpoint(problem.initial_coil_dofs)

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert endpoint.inner_success
    assert np.isfinite(endpoint.iota)


def test_official_compiled_program_captures_host_geometry_and_coil_template(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """The captured constants are host arrays.

    The two payloads the compiled evaluate program closes over and that dominate
    its constants are the affine surface basis -- shape
    ``(nphi, ntheta, 3, n_dofs)`` per geometry term -- and the frozen coil
    reconstruction template.  A device-resident basis is produced by an eager
    device ``jacfwd`` that the strict lane's guard refuses, and XLA then copies
    it back to the host once per lowering.  The adapter this problem uses bakes
    both with NumPy.

    The argument side is the opposite claim: the warm-start state the program
    takes as an argument is a device array, not a host one.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    problem = _official_problem(_REDUCED_NEWTON_TOLERANCE)
    assert type(problem._boozer_surface) is HostConstructionBoozerSurfaceJAX

    _geometry_from_dofs, basis, _label_value = (
        problem._boozer_surface._make_analytic_geometry_terms()
    )
    basis_leaves = jax.tree.leaves(basis)
    assert basis_leaves
    assert [leaf for leaf in basis_leaves if isinstance(leaf, jax.Array)] == []

    coil_template = problem._evaluator._objective_cache_state["objective_kwargs"][
        "coil_dof_extraction_spec"
    ]
    template_leaves = jax.tree.leaves(coil_template)
    assert template_leaves
    assert [leaf for leaf in template_leaves if isinstance(leaf, jax.Array)] == []

    assert isinstance(problem._evaluator.x_inner, jax.Array)
