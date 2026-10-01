"""Matched native/JAX parity for the exact ``1_Simple/qfm.py`` mirror.

Every official fact asserted here is read from the tracked official-reference
fixture ``examples/jax/parity/official_reference`` (upstream
``9e027eac38028d57aa23777be52a781aa860e347``): never pasted as a literal and
never read from the git-ignored campaign capture tree.

Upstream's ``1_Simple/qfm.py`` has no reduced CI configuration, so only the
``native_default`` scale has an official run. Assertions about a ``bounded``
run are therefore limited to what a bounded run can prove about itself.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
import os
import subprocess
import sys
from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from examples.jax._lane_environment import build_execution_environment
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_qfm import (
    NATIVE_DEFAULT_LABEL_RESIDUAL_GROSS_FAILURE_CEILING,
    OFFICIAL_EXACT_LABEL_RESIDUALS,
    QFM_EXACT_METHOD,
    QFM_PENALTY_METHOD,
    QfmProviderCall,
    official_qfm_stopping_reasons,
    qfm_scientific_predicate,
    qfm_stage_stopping_reason,
)
from examples.jax.parity.input_bundle import load_input_bundle
from examples.jax.parity.official_reference import load_official_reference
from simsopt.configs.zoo import get_data
from simsopt_contracts.optimization_endpoint import normalized_terminal_status
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.qfm_host_scipy import (
    QFM_SURFACE_RESOLUTION,
    build_qfm_host_kernels,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.isolated_kernel import pythonpath_with_loaded_kernel

# venv site-packages/tests shadows the repo tests package, so the helper is
# imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from mirror_example_constants import (  # noqa: E402
    mirror_example_call_bindings,
    mirror_example_constants,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "jax" / "1_Simple" / "qfm.py"
OFFICIAL = load_official_reference("native-qfm")
# The official shipped-scale start state and the official end of the whole run.
OFFICIAL_INITIAL_QFM_VALUE = float(OFFICIAL.scalar("initial:qfm_value"))
OFFICIAL_INITIAL_QFM_GRADIENT_INF_NORM = float(
    np.max(np.abs(OFFICIAL.array("initial:qfm_gradient")))
)
OFFICIAL_VOLUME_TARGET = float(OFFICIAL.scalar("volume:target"))
OFFICIAL_FINAL_QFM_VALUE = float(OFFICIAL.scalar("area:exact:qfm_value"))
OFFICIAL_PARAMETER_COUNT = int(OFFICIAL.array("initial:parameters").size)
# Upstream's own resolution, transcribed from ``examples/1_Simple/qfm.py``
# (``mpol = ntor = 5``; ``np.linspace(..., 25, endpoint=False)`` for both
# angles). The script is pinned by ``OFFICIAL.official_script_sha256``, so
# these two are checked against upstream rather than against the branch table
# they are compared with below.
OFFICIAL_SURFACE_ORDER = 5
OFFICIAL_QUADRATURE_SIZE = 25


def _initial_jax_state(scale: ExecutionScale, root: Path):
    """The JAX lane's own physics at the start state of one scale.

    This is the lane's code path -- ``create_input`` then
    ``build_qfm_host_kernels`` over the bundle's arrays -- stopped before the
    six SciPy calls, so a shipped-scale comparison costs one compile instead of
    the whole official solve.
    """
    case = get_case("native-qfm")
    bundle = case.create_input(root, scale)
    _, arrays = load_input_bundle(root, bundle)
    _, _, _, _, biotsavart = get_data("ncsx")
    field = BiotSavartJAX(biotsavart.coils)
    coil_set_spec = field.coil_set_spec_from_dofs(
        explicit_device_array(
            field.x, dtype=np.float64, device=get_runtime_jax_device()
        )
    )
    kernels = build_qfm_host_kernels(
        initial_parameters=arrays["initial_parameters"],
        quadpoints_phi=arrays["quadrature_phi"],
        quadpoints_theta=arrays["quadrature_theta"],
        coil_set_spec=coil_set_spec,
        mpol=int(bundle.configuration["mpol"]),
        ntor=int(bundle.configuration["ntor"]),
        nfp=int(bundle.configuration["nfp"]),
        stellsym=bool(bundle.configuration["stellsym"]),
    )
    qfm_value, qfm_gradient = kernels.host_qfm()(arrays["initial_parameters"])
    volume_value, _ = kernels.host_label("volume")(arrays["initial_parameters"])
    return bundle, arrays, qfm_value, qfm_gradient, volume_value


def test_official_qfm_sequence_matches_native_and_jax_host_scipy_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-qfm")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        native = case.execute("native-cpu", bundle, arrays)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    # Not ``native.success == jax.success``: that predicate passes when both
    # lanes fail, so each lane is required to reach the official outcome.
    # Classification, not counters and not SciPy's message text. ``(nit, nfev,
    # njev)`` equality across a C++ lane and a JAX lane pins a trajectory
    # identity, not an invariant: at shipped scale four starts 4.4e-16 apart
    # gave 921, 984, 997 and 1000 L-BFGS-B iterations
    # (A/fix-wave-1/qfm/IMPLEMENTATION.md, task 4). And a verbatim message is a
    # SciPy version detail: ``pyproject.toml`` allows ``scipy>=1.13``, whose
    # L-BFGS-B spells the FACTR*EPSMCH task with underscores. What both lanes
    # must reproduce is the fold of the official run's own six stopping
    # reasons, which is scale- and version-independent.
    official_terminal = normalized_terminal_status(
        scientific_predicate=True,
        stage_stopping_reasons=official_qfm_stopping_reasons(),
    )
    assert official_terminal.normalized_status == native.normalized_status
    assert official_terminal.normalized_status == jax.normalized_status
    assert official_terminal.success is native.success is jax.success is True
    # Published, not gated.
    print(
        "native-qfm bounded counters (nit, nfev, njev): "
        f"native={(native.nit, native.nfev, native.njev)} "
        f"jax={(jax.nit, jax.nfev, jax.njev)}"
    )
    assert native.scale == jax.scale == "bounded"
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "initial:parameters",
        "initial:qfm_value",
        "initial:qfm_gradient",
        "volume:target",
        "volume:initial:label_value",
        "volume:initial:label_gradient",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-9,
            atol=1.0e-11,
        )

    for stage in ("volume", "toroidal_flux", "area"):
        for phase in ("initial", "penalty", "exact"):
            np.testing.assert_allclose(
                jax.values[f"{stage}:{phase}:qfm_value"],
                native.values[f"{stage}:{phase}:qfm_value"],
                rtol=5.0e-5,
                atol=1.0e-7,
            )
        np.testing.assert_allclose(
            jax.values[f"{stage}:exact:parameters"],
            native.values[f"{stage}:exact:parameters"],
            rtol=2.0e-3,
            atol=2.0e-4,
        )
        # What a BOUNDED run can prove about its own exact stage: the endpoint
        # is finite and the stage moved. No residual ceiling here -- upstream's
        # script has no reduced configuration, so this resolution has no
        # official run and no official residual; the ceiling derived from the
        # shipped run lives on the ``native_default`` contract only (see
        # ``test_native_default_predicate_rejects_a_gross_feasibility_failure``).
        for observation in (native, jax):
            residual = observation.values[f"{stage}:exact:label_residual_abs"]
            assert np.all(np.isfinite(residual))
            assert np.any(
                observation.values[f"{stage}:exact:parameters"]
                != observation.values[f"{stage}:initial:parameters"]
            )
        # Both lanes hold their own label target, so the two lanes are compared
        # to each other rather than to a gate upstream never states.
        np.testing.assert_allclose(
            jax.values[f"{stage}:exact:label_residual_abs"],
            native.values[f"{stage}:exact:label_residual_abs"],
            rtol=1.0e-3,
            atol=1.0e-12,
        )

    for stage in ("toroidal_flux", "area"):
        np.testing.assert_allclose(
            jax.values[f"{stage}:volume_persistence_objective"],
            native.values[f"{stage}:volume_persistence_objective"],
            rtol=2.0e-2,
            atol=1.0e-5,
        )

    assert jax.driver == "scipy_lbfgsb_slsqp_qfm_sequence"
    assert native.driver == "simsopt_lbfgsb_then_slsqp_qfm_sequence"


def test_native_default_qfm_uses_official_resolution_and_solver_tolerances(
    tmp_path: Path,
) -> None:
    """The shipped example uses order five and 25 nodes, unlike its smoke case."""
    case = get_case("native-qfm")
    bundle = case.create_input(tmp_path / "shipped", "native_default")
    _, arrays = load_input_bundle(tmp_path / "shipped", bundle)
    assert (bundle.configuration["mpol"], bundle.configuration["ntor"]) == (
        OFFICIAL_SURFACE_ORDER,
        OFFICIAL_SURFACE_ORDER,
    )
    assert arrays["quadrature_phi"].shape == (OFFICIAL_QUADRATURE_SIZE,)
    assert arrays["quadrature_theta"].shape == (OFFICIAL_QUADRATURE_SIZE,)
    np.testing.assert_allclose(
        arrays["quadrature_phi"],
        np.linspace(
            0.0,
            1.0 / bundle.configuration["nfp"],
            OFFICIAL_QUADRATURE_SIZE,
            endpoint=False,
        ),
        rtol=0,
        atol=0,
    )
    # The resolution reaches the surface the official capture recorded.
    assert arrays["initial_parameters"].size == OFFICIAL_PARAMETER_COUNT
    assert bundle.configuration["tolerance"] == 1.0e-12
    assert bundle.configuration["max_steps"] == 1000
    bounded = case.create_input(tmp_path / "bounded", "bounded")
    _, bounded_arrays = load_input_bundle(tmp_path / "bounded", bounded)
    assert bounded_arrays["quadrature_phi"].shape == (6,)
    assert arrays["initial_parameters"].size > bounded_arrays["initial_parameters"].size


def test_native_default_jax_start_state_matches_the_official_capture(
    tmp_path: Path,
) -> None:
    """At shipped scale the JAX lane's own physics reproduces the official start.

    The six shipped-scale SciPy calls are a knife edge (wave 1 measured 921 to
    1000 L-BFGS-B iterations from starts 4.4e-16 apart), so the end point is
    reported, not gated; the start state is not, and it is compared with the
    official capture rather than with the other lane.
    """
    _, arrays, qfm_value, qfm_gradient, volume_value = _initial_jax_state(
        "native_default", tmp_path / "shipped"
    )

    assert arrays["initial_parameters"].size == OFFICIAL_PARAMETER_COUNT
    np.testing.assert_allclose(
        qfm_value, OFFICIAL_INITIAL_QFM_VALUE, rtol=1.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        volume_value, OFFICIAL_VOLUME_TARGET, rtol=1.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        float(np.max(np.abs(qfm_gradient))),
        OFFICIAL_INITIAL_QFM_GRADIENT_INF_NORM,
        rtol=1.0e-12,
        atol=0.0,
    )
    # The official run reduces this value to 2.057923850370213e-07; the start
    # is four orders above the end, which is what the lanes' decrease
    # predicate is measured against.
    assert qfm_value > OFFICIAL_FINAL_QFM_VALUE


@pytest.fixture(scope="module")
def bounded_mirror_observables() -> dict:
    """The shipped script's own published observables at bounded scale.

    One subprocess run, shared by the tests below: the script is executed in
    the parity lane's environment exactly as a user would run it.
    """
    _, environment = build_execution_environment(
        "cpu",
        "parity",
        os.environ,
        repo_root=REPO_ROOT,
    )
    environment["PYTHONPATH"] = pythonpath_with_loaded_kernel(
        *environment["PYTHONPATH"].split(os.pathsep)
    )
    completed = subprocess.run(
        (sys.executable, "-S", str(EXAMPLE), "--smoke", "--json"),
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)["observables"]


def test_the_mirror_and_its_case_classify_a_run_the_same_way(
    bounded_mirror_observables: dict,
    tmp_path: Path,
) -> None:
    """Script and parity case cannot label the same run differently.

    The mirror used to keep its own success rule (raw ``success`` flags) while
    the case classified every call through the contract's status tables, so a
    budget-stopped stage was "not successful" to the script and
    ``budget_exhausted`` to the case. Both now go through the SAME owner
    (``simsopt_contracts.optimization_endpoint.scipy_minimize_stopping_reason``),
    and this asserts it on a real run: the reasons the script published are the
    reasons the case derives from the script's own provider fields.

    The script's endpoints are finite because its ``solver_success`` is true
    and finiteness on the full arrays is one of that predicate's conjuncts.
    """
    observables = bounded_mirror_observables
    assert observables["solver_success"] is True
    case = get_case("native-qfm")
    bundle = case.create_input(tmp_path / "inputs", "bounded")
    max_steps = bundle.configuration["max_steps"]
    assert isinstance(max_steps, int)

    for stage in observables["stages"]:
        for phase, method in (
            ("penalty", QFM_PENALTY_METHOD),
            ("exact", QFM_EXACT_METHOD),
        ):
            published = stage[f"{phase}_stopping_reason"]
            derived = qfm_stage_stopping_reason(
                QfmProviderCall(
                    method=method,
                    provider_success=bool(stage[f"{phase}_success"]),
                    provider_status=int(stage[f"{phase}_status"]),
                    iterations=int(stage[f"{phase}_nit"]),
                    max_iterations=max_steps,
                    endpoint_finite=True,
                )
            )
            assert published == derived, (stage["label"], phase)
        assert stage["penalty_stopping_reason"] == "converged"
        assert stage["exact_stopping_reason"] == "converged"


def test_the_mirror_and_the_case_run_the_same_methods_and_budgets(
    tmp_path: Path,
) -> None:
    """The two owners of the method names and the budgets are bound together.

    The mirror is a standalone script: it cannot import the parity case, so it
    carries its own constants, and nothing but this test stops them drifting
    from the case's. The values are read out of the shipped bytes with ``ast``
    (the example directories are not importable and dynamic loading is
    forbidden here).
    """
    constants = mirror_example_constants(EXAMPLE)
    case = get_case("native-qfm")

    assert constants["PENALTY_METHOD"] == QFM_PENALTY_METHOD
    for scale, constant in (
        ("bounded", "BOUNDED_STEPS"),
        ("native_default", "NATIVE_DEFAULT_STEPS"),
    ):
        bundle = case.create_input(tmp_path / scale, scale)
        assert constants[constant] == bundle.configuration["max_steps"], scale
    # And the script's own budget really is the constant.
    main_bindings = mirror_example_call_bindings(EXAMPLE, "main", "run_example")
    assert main_bindings["bounded_steps"] == {"BOUNDED_STEPS"}
    assert main_bindings["native_default_steps"] == {"NATIVE_DEFAULT_STEPS"}
    # Upstream's own budget, from the tracked record of its six calls.
    assert constants["NATIVE_DEFAULT_STEPS"] == int(
        OFFICIAL.provider_calls[0].options["maxiter"]
    )


def test_public_qfm_scale_policy_reaches_the_surface_it_builds(
    bounded_mirror_observables: dict,
    tmp_path: Path,
) -> None:
    """The shipped example's scale policy reaches ``mpol``/``ntor`` and the grid.

    ``QFM_SURFACE_RESOLUTION`` is the one copy of the contract. This runs the
    shipped script and requires its published start state to equal the state
    built independently here from that table: the parameter count can only
    match if ``order`` reached ``mpol``/``ntor``, and the QFM value can only
    match if ``quadrature_size`` reached both ``linspace`` calls.
    """
    bundle, arrays, qfm_value, _, _ = _initial_jax_state("bounded", tmp_path / "input")
    resolution = QFM_SURFACE_RESOLUTION["bounded"]
    assert (bundle.configuration["mpol"], bundle.configuration["ntor"]) == (
        resolution.order,
        resolution.order,
    )
    assert arrays["quadrature_phi"].shape == (resolution.quadrature_size,)

    observables = bounded_mirror_observables

    assert len(observables["initial_parameters"]) == arrays["initial_parameters"].size
    np.testing.assert_allclose(
        observables["initial_parameters"],
        arrays["initial_parameters"],
        rtol=0.0,
        atol=0.0,
    )
    # The two paths reach the same surface by different construction routes
    # (``surface.get_dofs()`` in the script, the bundle's array here), so they
    # agree to fp64 rounding, 6.3e-16 relative as measured. A wrong
    # ``quadrature_size`` moves this value by orders of magnitude: the bounded
    # 6x6 grid gives 1.661e-02 against the shipped 25x25 grid's 1.563e-02.
    np.testing.assert_allclose(
        observables["initial_residuals"][0], qfm_value, rtol=1.0e-12, atol=0.0
    )


def _official_exact_values() -> dict[str, np.ndarray]:
    """The official run's own published values, as a lane would publish them.

    Only the keys the scientific predicate reads; the predicate's finiteness
    conjunct is checked on whatever it is given, so this is the official
    endpoint rather than a hand-made one.
    """
    values = {
        "initial:parameters": OFFICIAL.array("initial:parameters"),
        "initial:qfm_value": np.asarray(OFFICIAL.scalar("initial:qfm_value")),
        "area:exact:parameters": OFFICIAL.array("area:exact:parameters"),
        "area:exact:qfm_value": np.asarray(OFFICIAL.scalar("area:exact:qfm_value")),
    }
    for stage in ("volume", "toroidal_flux", "area"):
        values[f"{stage}:exact:label_residual_abs"] = np.asarray(
            OFFICIAL.scalar(f"{stage}:exact:label_residual_abs")
        )
    return values


def test_the_gross_failure_ceiling_admits_the_official_run_by_an_order() -> None:
    """The ``native_default`` ceiling is derived from upstream's own residuals.

    It is ten times the largest residual the official run leaves, so the
    official run itself sits an order of magnitude below it while the branch's
    demoted ``1e-8`` gate would reject upstream.
    """
    assert set(OFFICIAL_EXACT_LABEL_RESIDUALS) == {"volume", "toroidal_flux", "area"}
    for label, residual in OFFICIAL_EXACT_LABEL_RESIDUALS.items():
        assert residual == float(
            OFFICIAL.scalar(f"{label}:exact:label_residual_abs")
        ), label
        assert residual < NATIVE_DEFAULT_LABEL_RESIDUAL_GROSS_FAILURE_CEILING
        assert residual > 1.0e-8
    assert NATIVE_DEFAULT_LABEL_RESIDUAL_GROSS_FAILURE_CEILING == 10.0 * max(
        OFFICIAL_EXACT_LABEL_RESIDUALS.values()
    )


def test_native_default_predicate_rejects_a_gross_feasibility_failure() -> None:
    """The ceiling gates the scale it is derived from, and can fail there.

    No CPU test runs the six shipped-scale SciPy calls (1000 maxiter, and wave
    1 measured that solve as a knife edge), so the gate is exercised at the
    predicate rather than through a lane: the official endpoint is admitted,
    and the same endpoint with one stage pushed to the ceiling is not. The
    bounded scale does not consult it at all.
    """
    predicate = qfm_scientific_predicate
    official_values = _official_exact_values()

    assert predicate(official_values, "native_default") is True
    assert predicate(official_values, "bounded") is True

    infeasible = dict(official_values)
    infeasible["volume:exact:label_residual_abs"] = np.asarray(
        NATIVE_DEFAULT_LABEL_RESIDUAL_GROSS_FAILURE_CEILING
    )
    assert predicate(infeasible, "native_default") is False
    # The bounded scale has no official residual, so it must not be gated by
    # one: the same values are still admitted there.
    assert predicate(infeasible, "bounded") is True

    stalled = dict(official_values)
    stalled["area:exact:parameters"] = official_values["initial:parameters"]
    assert predicate(stalled, "native_default") is False
    assert predicate(stalled, "bounded") is False


def test_a_budget_stopped_qfm_call_is_classified_apart_from_a_failure() -> None:
    """The three labels a QFM lane can now carry are really distinguishable.

    Before this wave both lanes folded everything into ``converged``/
    ``failed``; the fold has to separate a provider failure, a budget stop and
    a scientific-predicate failure.
    """
    official_calls = OFFICIAL.provider_calls
    penalty = official_calls[0]
    exact = official_calls[1]
    assert penalty.method == QFM_PENALTY_METHOD
    assert exact.method == QFM_EXACT_METHOD
    budget = int(penalty.options["maxiter"])

    at_budget = QfmProviderCall(
        method=QFM_PENALTY_METHOD,
        provider_success=False,
        provider_status=1,
        iterations=budget,
        max_iterations=budget,
        endpoint_finite=True,
    )
    assert qfm_stage_stopping_reason(at_budget) == "iteration-limit"
    slsqp_at_budget = QfmProviderCall(
        method=QFM_EXACT_METHOD,
        provider_success=False,
        provider_status=9,
        iterations=budget,
        max_iterations=budget,
        endpoint_finite=True,
    )
    assert qfm_stage_stopping_reason(slsqp_at_budget) == "iteration-limit"
    slsqp_failed = QfmProviderCall(
        method=QFM_EXACT_METHOD,
        provider_success=False,
        provider_status=8,
        iterations=1,
        max_iterations=budget,
        endpoint_finite=True,
    )
    # Named precisely, not lumped into ``failed``: mode 8 is SLSQP's positive
    # directional derivative in the line search, which the contract's
    # ``scipy-slsqp`` table distinguishes from a subproblem failure.
    assert qfm_stage_stopping_reason(slsqp_failed) == "line-search-failed"
    assert (
        normalized_terminal_status(
            scientific_predicate=True,
            stage_stopping_reasons=official_qfm_stopping_reasons()[:5]
            + ("line-search-failed",),
        ).normalized_status
        == "failed"
    )
    subproblem_failure = QfmProviderCall(
        method=QFM_EXACT_METHOD,
        provider_success=False,
        provider_status=5,
        iterations=1,
        max_iterations=budget,
        endpoint_finite=True,
    )
    assert qfm_stage_stopping_reason(subproblem_failure) == "failed"

    official_reasons = official_qfm_stopping_reasons()
    assert official_reasons == ("converged",) * 6
    assert (
        normalized_terminal_status(
            scientific_predicate=True,
            stage_stopping_reasons=(*official_reasons[:1], "iteration-limit")
            + official_reasons[2:],
        ).normalized_status
        == "budget_exhausted"
    )
    assert (
        normalized_terminal_status(
            scientific_predicate=False,
            stage_stopping_reasons=official_reasons,
        ).normalized_status
        == "failed"
    )
