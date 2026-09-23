"""The official BoozerQA lanes run upstream's analytic exact Newton route.

Upstream's ``examples/2_Intermediate/boozerQA.py`` re-solves the Boozer surface
with ``solve_residual_equation_exactly_newton`` on every objective evaluation
(dense analytic Jacobian, direct solve, undamped step) and returns ``J = 1e3``
with the previous surface restored when that solve fails.  The checks here hold
:class:`~simsopt_jax_adapters.geo.boozer_qa_problem.BoozerQAProblem` against both
references at the shipped scale: the traceable-session route the JAX lanes used
before (still the production route of the sealed compute-graph campaigns) and the
native lane's own objective, which is upstream's.

The shipped-scale comparisons are deliberately expensive: the session route costs
about 13 s per value-and-gradient at ``native_default`` (255 unknowns, 461 coil
dofs), so this file runs for about a minute on the development host.  A reduced
scale would not exercise
the 1e-13 Newton tolerance the official run uses.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest
from conftest import enable_non_strict_jax_backend
from examples.jax.parity.cases.native_boozerqa import (
    BOOZER_QA_SPEC,
    _prepare_jax_variant_runtime,
    _prepare_native_variant_runtime,
    build_variant_problem,
    create_variant_input,
)
from examples.jax.parity.input_bundle import load_input_bundle
from simsopt_jax.examples import ExecutionScale
from simsopt_jax_adapters.geo.boozer_qa_problem import BoozerQAProblem
from simsopt_jax_adapters.geo.single_stage_exact_analytic import (
    INNER_FAILURE_VALUE,
    HostConstructionBoozerSurfaceJAX,
)

#: Agreement required at the same coil state: the bars of requirement R2 as
#: AMENDED on 2026-09-20 (campaign record ``boozerqa-perf/REQUIREMENTS.md``,
#: section 8): the objective to 1e-13 relative, the gradient to 1e-11 relative
#: to its maximum component.  Measured on these four states at
#: ``native_default``: value 1.1e-14 to 4.3e-14 against the session route and
#: 2.0e-14 to 2.5e-14 against native; gradient 5.4e-13 to 1.5e-12 against the
#: session route and 5.5e-13 to 3.8e-12 against native.  The bars sit 2.3x
#: (value) and 2.6x (gradient) above the worst measurement of two DIFFERENT
#: exact-Newton algorithms in fp64; the requirement's first figures (1e-14 and
#: 1e-12) were written before the measurement and would reject that difference.
VALUE_RELATIVE_TOLERANCE = 1.0e-13
GRADIENT_RELATIVE_TOLERANCE = 1.0e-11
#: The solved Boozer state (surface dofs, iota, G) agrees to 1e-13 ABSOLUTE,
#: the requirement's own figure for the inner solution.
SOLVED_STATE_ABSOLUTE_TOLERANCE = 1.0e-13
#: The targets frozen at the seed solution (iota target, seed volume) agree to
#: 1e-13 RELATIVE: the same inner solution read through the same labels.  Both
#: targets are below one in magnitude here (iota -0.406, volume -0.290), so the
#: relative reading is the stricter one of the requirement's 1e-13.
FROZEN_TARGET_RELATIVE_TOLERANCE = 1.0e-13
#: Coil perturbations of the seed state, small enough that the exact Newton
#: solve converges at the official 1e-13 tolerance from its warm start and large
#: enough that the objective and its gradient both move.
_MOVED_STATE_SCALES = (2.0e-4, 5.0e-4, 1.0e-3)
_MOVED_STATE_SEED = 20260920


def _states(initial_parameters: np.ndarray) -> tuple[np.ndarray, ...]:
    """The case's start state followed by three moved coil states."""
    generator = np.random.default_rng(_MOVED_STATE_SEED)
    moved = tuple(
        initial_parameters
        * (1.0 + scale * generator.standard_normal(initial_parameters.size))
        for scale in _MOVED_STATE_SCALES
    )
    return (np.array(initial_parameters, copy=True), *moved)


def _bundle(root: Path, scale: ExecutionScale):
    bundle = create_variant_input(root, scale, BOOZER_QA_SPEC)
    _stored, arrays = load_input_bundle(root, bundle)
    return bundle, arrays


def _official_problem(bundle, arrays: dict[str, np.ndarray]) -> BoozerQAProblem:
    """The problem the official JAX lanes build, from the frozen input bundle."""
    configuration = bundle.configuration
    (
        base_curves,
        _base_currents,
        _magnetic_axis,
        nfp,
        native_field,
        surface,
        initial_G,
    ) = build_variant_problem(configuration, bundle.scale)
    surface.set_dofs(arrays["surface_dofs"])
    return BoozerQAProblem(
        base_curves=base_curves,
        native_field=native_field,
        surface=surface,
        nfp=nfp,
        initial_G=initial_G,
        initial_iota=float(configuration["initial_iota"]),
        boozer_options={
            "newton_maxiter": int(configuration["inner_maxiter"]),
            "newton_tol": float(configuration["inner_tolerance"]),
            "verbose": False,
        },
        non_qs_resolution=int(configuration["non_qs_sdim"]),
        residual_weight=float(configuration["residual_weight"]),
    )


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


def test_official_route_matches_the_session_route_at_the_shipped_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """R2 against the route the JAX lanes ran before, at the official scale.

    The session route is the one the sealed compute-graph campaigns still
    execute, so this is the compatibility claim: the same frozen targets, and
    the same objective and gradient at the case's start state and three moved
    states.  The solved Boozer state itself is pinned against the native lane
    in the next test: the session controller exposes only its ACCEPTED
    incumbent, not the candidate solve of each evaluation.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    bundle, arrays = _bundle(tmp_path / "inputs", "native_default")
    problem = _official_problem(bundle, arrays)
    states = _states(problem.initial_coil_dofs)

    prepared = _prepare_jax_variant_runtime(bundle, arrays, BOOZER_QA_SPEC, None)
    session = prepared.fresh_incumbent_controller()
    assert problem.iota_target == pytest.approx(
        prepared.iota_target, rel=FROZEN_TARGET_RELATIVE_TOLERANCE, abs=0.0
    )
    assert problem.initial_volume == pytest.approx(
        prepared.initial_volume, rel=FROZEN_TARGET_RELATIVE_TOLERANCE, abs=0.0
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


def test_official_route_matches_the_native_lane_at_the_shipped_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """R1: the JAX lanes run the native lane's algorithm, at the official scale.

    The native runtime is the case's own native-cpu evaluator, which is
    upstream's objective over ``solve_residual_equation_exactly_newton`` with
    upstream's failed-solve policy.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    bundle, arrays = _bundle(tmp_path / "inputs", "native_default")
    problem = _official_problem(bundle, arrays)
    native = _prepare_native_variant_runtime(bundle, arrays, BOOZER_QA_SPEC)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Upstream's failed-evaluation policy, against the native lane's own.

    Upstream returns ``J = 1e3`` and restores the surface, ``iota`` and ``G``
    that the evaluation started from, while the gradient it reports is the one
    taken at the iterate the failed solve returned, before that restore
    (upstream boozerQA.py:99-109).  The native runtime implements exactly that
    policy, so the failure decision, the sentinel value and the restored warm
    start are pinned against it exactly.

    The failure-path GRADIENT is deliberately not pinned to native's: a
    cap-exhausted Newton run has no converged state, and the two lanes return
    different final iterates, so there is no shared reference to compare at.
    Measured on this state: the two gradients agree to 9.5e-7 relative in norm
    and to 5.0e-5 componentwise against the maximum component.  What is pinned
    is that a failed evaluation reports upstream's sentinel with a finite
    derivative rather than a NaN, and that the run recovers from it.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    bundle, arrays = _bundle(tmp_path / "inputs", "bounded")
    problem = _official_problem(bundle, arrays)
    native = _prepare_native_variant_runtime(bundle, arrays, BOOZER_QA_SPEC)
    warm_start = np.array(_solved_state(problem), copy=True)

    # Coils scaled far from the seed surface: the exact Newton solve exhausts
    # its iteration cap on both lanes, so the failure policy is exercised on a
    # real failure rather than a simulated one.
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
    # The reported physics is the restored warm start's, which is what the
    # published ``final:*`` observables then describe.
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """N3: the strict lane's guard sees no implicit host boundary crossing.

    The strict GPU lane runs under ``JAX_TRANSFER_GUARD=disallow``
    (``examples/jax/_lane_environment.py:61-75``).  Construction, the first
    value-and-gradient -- the call that compiles -- and the endpoint report are
    all inside the guard here.  Only the host-to-device half can be exercised on
    CPU; the device-to-host half needs a real device.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    bundle, arrays = _bundle(tmp_path / "inputs", "bounded")

    with jax.transfer_guard("disallow"):
        problem = _official_problem(bundle, arrays)
        value, gradient = problem.value_and_gradient(problem.initial_coil_dofs)
        endpoint = problem.endpoint(problem.initial_coil_dofs)

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert endpoint.inner_success
    assert np.isfinite(endpoint.iota)


def test_official_compiled_program_captures_host_geometry_and_coil_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """N3's other half: the captured constants are host arrays.

    The two payloads the compiled evaluate program closes over and that dominate
    its constants are the affine surface basis -- shape
    ``(nphi, ntheta, 3, n_dofs)`` per geometry term -- and the frozen coil
    reconstruction template.  A device-resident basis is the wave-6 defect class:
    it is produced by an eager device ``jacfwd`` that the strict lane's guard
    refuses, and XLA then copies it back to the host once per lowering.  The
    adapter this problem uses bakes both with NumPy.

    The argument side is the opposite claim: the warm-start state the program
    takes as an argument is a device array, not a host one.
    """
    enable_non_strict_jax_backend(monkeypatch, request, "jax_cpu_parity")
    bundle, arrays = _bundle(tmp_path / "inputs", "bounded")
    problem = _official_problem(bundle, arrays)
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
