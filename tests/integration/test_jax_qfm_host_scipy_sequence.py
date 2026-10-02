"""The ``1_Simple/qfm.py`` mirror's six-call QFM sequence against the native library.

The native side is ``QfmSurface``'s L-BFGS-B penalty and SLSQP exact solves for
the volume, toroidal-flux and area labels in turn, as upstream's
``examples/1_Simple/qfm.py`` runs them; the JAX side is
``solve_qfm_host_scipy_sequence`` over ``build_qfm_host_kernels``, which the
shipped mirror runs.  Both start from the same fitted NCSX surface; every input
is built live.  Upstream's script has no reduced configuration, so the bounded
scale (``QFM_SURFACE_RESOLUTION["bounded"]``) has no official run and its
assertions are limited to what the two implementations prove about each other.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
import json
import os
import subprocess
import sys
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from examples.jax._lane_environment import build_execution_environment
from simsopt.configs.zoo import get_data
from simsopt.field import BiotSavart
from simsopt.geo import Area, QfmSurface, SurfaceRZFourier, ToroidalFlux, Volume
from simsopt_contracts.optimization_endpoint import (
    StoppingReason,
    normalized_terminal_status,
    scipy_minimize_stopping_reason,
)
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.qfm_host_scipy import (
    QFM_EXACT_METHOD,
    QFM_SURFACE_RESOLUTION,
    build_qfm_host_kernels,
    solve_qfm_host_scipy_sequence,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.isolated_kernel import pythonpath_with_loaded_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "jax" / "1_Simple" / "qfm.py"
#: Upstream's own resolution, transcribed from ``examples/1_Simple/qfm.py``
#: (``mpol = ntor = 5``; ``np.linspace(..., 25, endpoint=False)`` for both
#: angles).
OFFICIAL_SURFACE_ORDER = 5
OFFICIAL_QUADRATURE_SIZE = 25
#: Upstream's own solver settings (``tol=1e-12``, ``constraint_weight=1``) and
#: its first SciPy method; the exact stage's method is the solver module's.
TOLERANCE = 1.0e-12
CONSTRAINT_WEIGHT = 1.0
PENALTY_METHOD = "L-BFGS-B"
#: Upstream's ``maxiter`` for every one of its six calls.
OFFICIAL_MAX_STEPS = 1000
_LABELS = ("volume", "toroidal_flux", "area")


def _script_constants() -> dict[str, object]:
    """Every ``NAME = <literal>`` at the shipped script's module scope, read with ``ast``.

    The numbered example directory is not an importable package and dynamic
    loading is forbidden here.
    """
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"), filename=str(EXAMPLE))
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }


def _bounded_steps() -> int:
    steps = _script_constants()["BOUNDED_STEPS"]
    assert isinstance(steps, int)
    return steps


def _fitted_problem(scale: ExecutionScale) -> tuple[BiotSavart, SurfaceRZFourier]:
    """NCSX coils and upstream's fitted QFM start surface at one resolution."""
    _, _, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    resolution = QFM_SURFACE_RESOLUTION[scale]
    surface = SurfaceRZFourier(
        mpol=resolution.order,
        ntor=resolution.order,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0, 1.0 / nfp, resolution.quadrature_size, endpoint=False
        ),
        quadpoints_theta=np.linspace(
            0.0, 1.0, resolution.quadrature_size, endpoint=False
        ),
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    return biotsavart, surface


def _jax_kernels(biotsavart: BiotSavart, surface: SurfaceRZFourier):
    """The mirror's kernels, built exactly as the shipped script builds them."""
    field = BiotSavartJAX(biotsavart.coils)
    return build_qfm_host_kernels(
        initial_parameters=np.asarray(surface.x, dtype=np.float64),
        quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64),
        coil_set_spec=field.coil_set_spec_from_dofs(
            explicit_device_array(
                field.x, dtype=np.float64, device=get_runtime_jax_device()
            )
        ),
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )


@dataclass(frozen=True)
class _ProviderCall:
    method: str
    success: bool
    status: int
    iterations: int


def _stopping_reasons(
    calls: list[_ProviderCall], values: dict[str, np.ndarray], max_steps: int
) -> tuple[StoppingReason, ...]:
    """Each call through the contract's owner, with the finiteness of its endpoint."""
    phases = [f"{label}:{phase}" for label in _LABELS for phase in ("penalty", "exact")]
    return tuple(
        scipy_minimize_stopping_reason(
            method=call.method,
            provider_success=call.success,
            provider_status=call.status,
            iterations=call.iterations,
            max_iterations=max_steps,
            endpoint_finite=bool(
                np.all(np.isfinite(values[f"{phase}:parameters"]))
                and np.all(np.isfinite(values[f"{phase}:qfm_value"]))
            ),
        )
        for call, phase in zip(calls, phases, strict=True)
    )


def _scientific_predicate(values: dict[str, np.ndarray]) -> bool:
    """Finite everywhere, the QFM value decreased, and the run moved off its start."""
    finite = all(bool(np.all(np.isfinite(value))) for value in values.values())
    decreased = float(values["area:exact:qfm_value"]) < float(
        values["initial:qfm_value"]
    )
    moved = bool(
        np.any(values["area:exact:parameters"] != values["initial:parameters"])
    )
    return finite and decreased and moved


def _native_state(qfm_surface, label, target: float) -> dict[str, np.ndarray]:
    parameters = np.array(qfm_surface.surface.x, dtype=np.float64, copy=True)
    qfm_value, qfm_gradient = qfm_surface.qfm_objective(parameters, derivatives=1)
    label_value = float(label.J())
    return {
        "parameters": parameters,
        "qfm_value": np.asarray(qfm_value, dtype=np.float64),
        "qfm_gradient": np.array(qfm_gradient, dtype=np.float64, copy=True),
        "label_value": np.asarray(label_value, dtype=np.float64),
        "label_gradient": np.array(
            label.dJ_by_dsurfacecoefficients(), dtype=np.float64, copy=True
        ),
        "label_residual_abs": np.asarray(abs(label_value - target), dtype=np.float64),
    }


def _prefixed(stage: str, phase: str, state: dict[str, np.ndarray]):
    return {f"{stage}:{phase}:{name}": value for name, value in state.items()}


def _native_sequence(max_steps: int):
    """Upstream's six QFM calls with the native library, on a fresh construction."""
    biotsavart, surface = _fitted_problem("bounded")
    toroidal_field = BiotSavart(biotsavart.coils)
    initial_volume = float(Volume(surface).J())
    values: dict[str, np.ndarray] = {}
    calls: list[_ProviderCall] = []
    for stage in _LABELS:
        if stage == "volume":
            label = Volume(surface)
        elif stage == "toroidal_flux":
            label = ToroidalFlux(surface, toroidal_field)
        else:
            label = Area(surface)
        target = float(label.J())
        qfm_surface = QfmSurface(biotsavart, surface, label, target)
        initial = _native_state(qfm_surface, label, target)
        if stage == "volume":
            values["initial:parameters"] = initial["parameters"]
            values["initial:qfm_value"] = initial["qfm_value"]
            values["initial:qfm_gradient"] = initial["qfm_gradient"]
        values[f"{stage}:target"] = np.asarray(target, dtype=np.float64)
        values.update(_prefixed(stage, "initial", initial))
        penalty = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=TOLERANCE, maxiter=max_steps, constraint_weight=CONSTRAINT_WEIGHT
        )
        values.update(
            _prefixed(stage, "penalty", _native_state(qfm_surface, label, target))
        )
        exact = qfm_surface.minimize_qfm_exact_constraints_SLSQP(
            tol=TOLERANCE, maxiter=max_steps
        )
        values.update(
            _prefixed(stage, "exact", _native_state(qfm_surface, label, target))
        )
        volume_residual = float(Volume(surface).J()) - initial_volume
        values[f"{stage}:volume_persistence_objective"] = np.asarray(
            0.5 * volume_residual * volume_residual, dtype=np.float64
        )
        calls.extend(
            (
                _ProviderCall(
                    PENALTY_METHOD,
                    bool(penalty["success"]),
                    int(penalty["info"].status),
                    int(penalty["iter"]),
                ),
                _ProviderCall(
                    QFM_EXACT_METHOD,
                    bool(exact["success"]),
                    int(exact["info"].status),
                    int(exact["iter"]),
                ),
            )
        )
    return values, _stopping_reasons(calls, values, max_steps)


def _jax_state(state) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(state, name), dtype=np.float64)
        for name in (
            "parameters",
            "qfm_value",
            "qfm_gradient",
            "label_value",
            "label_gradient",
            "label_residual_abs",
        )
    }


def _jax_sequence(max_steps: int):
    """The mirror's six calls, on a fresh construction."""
    biotsavart, surface = _fitted_problem("bounded")
    initial_parameters = np.asarray(surface.x, dtype=np.float64)
    result = solve_qfm_host_scipy_sequence(
        initial_parameters,
        kernels=_jax_kernels(biotsavart, surface),
        max_steps=max_steps,
        tolerance=TOLERANCE,
        constraint_weight=CONSTRAINT_WEIGHT,
    )
    values: dict[str, np.ndarray] = {
        "initial:parameters": np.asarray(
            result.volume.initial.parameters, dtype=np.float64
        ),
        "initial:qfm_value": np.asarray(
            result.volume.initial.qfm_value, dtype=np.float64
        ),
        "initial:qfm_gradient": np.asarray(
            result.volume.initial.qfm_gradient, dtype=np.float64
        ),
    }
    calls: list[_ProviderCall] = []
    for stage_name in _LABELS:
        stage = getattr(result, stage_name)
        values[f"{stage_name}:target"] = np.asarray(stage.target, dtype=np.float64)
        values.update(_prefixed(stage_name, "initial", _jax_state(stage.initial)))
        values.update(_prefixed(stage_name, "penalty", _jax_state(stage.penalty)))
        values.update(_prefixed(stage_name, "exact", _jax_state(stage.exact)))
        values[f"{stage_name}:volume_persistence_objective"] = np.asarray(
            stage.volume_persistence_objective, dtype=np.float64
        )
        calls.extend(
            (
                _ProviderCall(
                    PENALTY_METHOD,
                    bool(stage.penalty_optimizer.success),
                    int(stage.penalty_optimizer.status),
                    int(stage.penalty_optimizer.nit),
                ),
                _ProviderCall(
                    QFM_EXACT_METHOD,
                    bool(stage.exact_optimizer.success),
                    int(stage.exact_optimizer.status),
                    int(stage.exact_optimizer.nit),
                ),
            )
        )
    return values, _stopping_reasons(calls, values, max_steps)


def test_qfm_sequence_matches_native_and_jax_host_scipy_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_steps = _bounded_steps()
    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        native, native_reasons = _native_sequence(max_steps)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax, jax_reasons = _jax_sequence(max_steps)

    # Each implementation must reach upstream's outcome -- all six calls
    # converged, folded with the scientific predicate -- rather than merely
    # agree with the other: two failures would agree too.  Classification,
    # not counters and not SciPy's message text: ``(nit, nfev, njev)`` across
    # a C++ and a JAX implementation pin a trajectory identity, not an
    # invariant, and a verbatim message is a SciPy version detail.
    official_terminal = normalized_terminal_status(
        scientific_predicate=True,
        stage_stopping_reasons=("converged",) * 6,
    )
    for values, reasons in ((native, native_reasons), (jax, jax_reasons)):
        terminal = normalized_terminal_status(
            scientific_predicate=_scientific_predicate(values),
            stage_stopping_reasons=reasons,
        )
        assert terminal.normalized_status == official_terminal.normalized_status
        assert terminal.success is official_terminal.success is True
    assert set(native) == set(jax)

    for observable in (
        "initial:parameters",
        "initial:qfm_value",
        "initial:qfm_gradient",
        "volume:target",
        "volume:initial:label_value",
        "volume:initial:label_gradient",
    ):
        np.testing.assert_allclose(
            jax[observable],
            native[observable],
            rtol=1.0e-9,
            atol=1.0e-11,
        )

    for stage in _LABELS:
        for phase in ("initial", "penalty", "exact"):
            np.testing.assert_allclose(
                jax[f"{stage}:{phase}:qfm_value"],
                native[f"{stage}:{phase}:qfm_value"],
                rtol=5.0e-5,
                atol=1.0e-7,
            )
        np.testing.assert_allclose(
            jax[f"{stage}:exact:parameters"],
            native[f"{stage}:exact:parameters"],
            rtol=2.0e-3,
            atol=2.0e-4,
        )
        # What a BOUNDED run can prove about its own exact stage: the endpoint
        # is finite and the stage moved.  No residual ceiling: this resolution
        # has no official run and no official residual.
        for values in (native, jax):
            residual = values[f"{stage}:exact:label_residual_abs"]
            assert np.all(np.isfinite(residual))
            assert np.any(
                values[f"{stage}:exact:parameters"]
                != values[f"{stage}:initial:parameters"]
            )
        # Both hold their own label target, so they are compared with each
        # other rather than with a gate upstream never states.
        np.testing.assert_allclose(
            jax[f"{stage}:exact:label_residual_abs"],
            native[f"{stage}:exact:label_residual_abs"],
            rtol=1.0e-3,
            atol=1.0e-12,
        )

    for stage in ("toroidal_flux", "area"):
        np.testing.assert_allclose(
            jax[f"{stage}:volume_persistence_objective"],
            native[f"{stage}:volume_persistence_objective"],
            rtol=2.0e-2,
            atol=1.0e-5,
        )


def test_native_default_qfm_uses_the_official_resolution() -> None:
    """The shipped scale uses upstream's order five and 25 nodes, unlike the smoke scale."""
    shipped = QFM_SURFACE_RESOLUTION["native_default"]
    bounded = QFM_SURFACE_RESOLUTION["bounded"]
    assert (shipped.order, shipped.quadrature_size) == (
        OFFICIAL_SURFACE_ORDER,
        OFFICIAL_QUADRATURE_SIZE,
    )
    _, surface = _fitted_problem("native_default")
    assert (surface.mpol, surface.ntor) == (OFFICIAL_SURFACE_ORDER,) * 2
    np.testing.assert_allclose(
        surface.quadpoints_phi,
        np.linspace(0.0, 1.0 / surface.nfp, OFFICIAL_QUADRATURE_SIZE, endpoint=False),
        rtol=0,
        atol=0,
    )
    assert np.asarray(surface.quadpoints_theta).shape == (OFFICIAL_QUADRATURE_SIZE,)
    _, bounded_surface = _fitted_problem("bounded")
    assert bounded.quadrature_size == 6
    assert np.asarray(bounded_surface.quadpoints_phi).shape == (6,)
    assert np.asarray(surface.x).size > np.asarray(bounded_surface.x).size


def test_native_default_jax_start_state_matches_the_native_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the shipped scale the mirror's own physics reproduces the start state.

    The reference is the native library's ``QfmSurface.qfm_objective`` and
    ``Volume`` at the same fitted surface, upstream's own functions.  The six
    shipped-scale SciPy calls are a knife edge, so the end point is not
    compared; the start state is.
    """
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    biotsavart, surface = _fitted_problem("native_default")
    parameters = np.asarray(surface.x, dtype=np.float64)
    volume = Volume(surface)
    native_volume = float(volume.J())
    native_value, native_gradient = QfmSurface(
        biotsavart, surface, volume, native_volume
    ).qfm_objective(parameters, derivatives=1)

    kernels = _jax_kernels(*_fitted_problem("native_default"))
    qfm_value, qfm_gradient = kernels.host_qfm()(parameters)
    volume_value, _ = kernels.host_label("volume")(parameters)

    np.testing.assert_allclose(qfm_value, float(native_value), rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(volume_value, native_volume, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(
        float(np.max(np.abs(qfm_gradient))),
        float(np.max(np.abs(np.asarray(native_gradient)))),
        rtol=1.0e-12,
        atol=0.0,
    )


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


def test_the_mirror_classifies_its_calls_through_the_contract(
    bounded_mirror_observables: dict,
) -> None:
    """The script's stopping reasons are the contract owner's, on a real run.

    The mirror used to keep its own success rule (raw ``success`` flags); it now
    classifies every call through
    ``simsopt_contracts.optimization_endpoint.scipy_minimize_stopping_reason``,
    and this asserts it: the reasons the script published are the reasons the
    contract derives from the script's own provider fields at its own budget.

    The script's endpoints are finite because its ``solver_success`` is true
    and finiteness on the full arrays is one of that predicate's conjuncts.
    """
    observables = bounded_mirror_observables
    assert observables["solver_success"] is True
    max_steps = _bounded_steps()

    for stage in observables["stages"]:
        for phase, method in (
            ("penalty", PENALTY_METHOD),
            ("exact", QFM_EXACT_METHOD),
        ):
            published = stage[f"{phase}_stopping_reason"]
            derived = scipy_minimize_stopping_reason(
                method=method,
                provider_success=bool(stage[f"{phase}_success"]),
                provider_status=int(stage[f"{phase}_status"]),
                iterations=int(stage[f"{phase}_nit"]),
                max_iterations=max_steps,
                endpoint_finite=True,
            )
            assert published == derived, (stage["label"], phase)
        assert stage["penalty_stopping_reason"] == "converged"
        assert stage["exact_stopping_reason"] == "converged"


def test_public_qfm_scale_policy_reaches_the_surface_it_builds(
    bounded_mirror_observables: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped example's scale policy reaches ``mpol``/``ntor`` and the grid.

    ``QFM_SURFACE_RESOLUTION`` is the one copy of the contract.  This runs the
    shipped script and requires its published start state to equal the state
    built independently here from that table: the parameter count can only
    match if ``order`` reached ``mpol``/``ntor``, and the QFM value can only
    match if ``quadrature_size`` reached both ``linspace`` calls.
    """
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    biotsavart, surface = _fitted_problem("bounded")
    resolution = QFM_SURFACE_RESOLUTION["bounded"]
    assert (surface.mpol, surface.ntor) == (resolution.order, resolution.order)
    assert np.asarray(surface.quadpoints_phi).shape == (resolution.quadrature_size,)
    parameters = np.asarray(surface.x, dtype=np.float64)
    qfm_value, _ = _jax_kernels(biotsavart, surface).host_qfm()(parameters)

    observables = bounded_mirror_observables

    assert len(observables["initial_parameters"]) == parameters.size
    np.testing.assert_allclose(
        observables["initial_parameters"],
        parameters,
        rtol=0.0,
        atol=0.0,
    )
    # A wrong ``quadrature_size`` moves this value by orders of magnitude: the
    # bounded 6x6 grid gives 1.661e-02 against the shipped 25x25 grid's
    # 1.563e-02.
    np.testing.assert_allclose(
        observables["initial_residuals"][0], qfm_value, rtol=1.0e-12, atol=0.0
    )


def test_a_budget_stopped_qfm_call_is_classified_apart_from_a_failure() -> None:
    """A provider failure, a budget stop and a predicate failure are distinguishable.

    The fold has to separate the three, not reduce everything to
    ``converged``/``failed``.
    """
    budget = OFFICIAL_MAX_STEPS

    def reason(
        method: str, status: int, iterations: int, success: bool = False
    ) -> StoppingReason:
        return scipy_minimize_stopping_reason(
            method=method,
            provider_success=success,
            provider_status=status,
            iterations=iterations,
            max_iterations=budget,
            endpoint_finite=True,
        )

    assert reason(PENALTY_METHOD, 1, budget) == "iteration-limit"
    assert reason(QFM_EXACT_METHOD, 9, budget) == "iteration-limit"
    # Named precisely, not lumped into ``failed``: mode 8 is SLSQP's positive
    # directional derivative in the line search, which the contract's
    # ``scipy-slsqp`` table distinguishes from a subproblem failure.
    assert reason(QFM_EXACT_METHOD, 8, 1) == "line-search-failed"
    converged = ("converged",) * 6
    assert (
        normalized_terminal_status(
            scientific_predicate=True,
            stage_stopping_reasons=converged[:5] + ("line-search-failed",),
        ).normalized_status
        == "failed"
    )
    assert reason(QFM_EXACT_METHOD, 5, 1) == "failed"
    assert (
        normalized_terminal_status(
            scientific_predicate=True,
            stage_stopping_reasons=(*converged[:1], "iteration-limit") + converged[2:],
        ).normalized_status
        == "budget_exhausted"
    )
    assert (
        normalized_terminal_status(
            scientific_predicate=False,
            stage_stopping_reasons=converged,
        ).normalized_status
        == "failed"
    )
