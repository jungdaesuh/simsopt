"""Exact parity for the modular and sector-saddle GSCO mirrors."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_wireframe_gsco_multistep import (
    _configuration as multistep_configuration,
)
from examples.jax.parity.cases.native_wireframe_gsco_multistep import (
    _observation as multistep_observation,
)
from examples.jax.parity.input_bundle import InputBundle, load_input_bundle
from simsopt_jax.core.wireframe_workflow import (
    WIREFRAME_GSCO_MULTISTEP_STAGE_CAPACITY,
)


@pytest.mark.parametrize(
    "case_id",
    (
        "native-wireframe-gsco-modular",
        "native-wireframe-gsco-sector-saddle",
        "native-wireframe-gsco-multistep",
    ),
)
def test_exact_wireframe_gsco_matches_native_and_jax_cpu(
    case_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case(case_id)
    input_root = tmp_path / case_id
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = case.execute("native-cpu", bundle, arrays)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.normalized_status == jax.normalized_status
    assert native.raw_status == jax.raw_status
    assert native.success == jax.success
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in native.values:
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    if case_id == "native-wireframe-gsco-multistep":
        stages = int(native.values["final:nonfinal_steps"]) + int(
            native.values["final:adjustment_run"]
        )
        budget = int(bundle.configuration["max_iterations_per_step"])
        assert stages > 0
        assert int(jax.values["final:allocated_iteration_budget"]) == stages * budget
        assert len(jax.values["history:stage_normal_objective"]) == stages
        # One iteration count and one allocated budget per stage that ran, as
        # one array each in the history phase, in the objective series' order.
        iterations = jax.values["history:stage_iterations"]
        allocated = jax.values["history:stage_allocated_iteration_budget"]
        assert iterations.shape == (stages,)
        np.testing.assert_array_equal(allocated, np.full(stages, budget))
        assert bool(np.all((iterations >= 0) & (iterations <= budget)))
        assert native.nit is None and native.nfev is None
        assert jax.nit is None and jax.nfev is None
    else:
        # At the bounded scale the reduced 40-iteration cap is reached, which is
        # upstream's stop_last_iter (wireframe_optimization.cpp:281) and a
        # budget stop, not convergence. The official native_default and CI runs
        # stop earlier on "minimum objective reached"; the bounded scale carries
        # a work-budget contract for exactly this.
        assert int(jax.values["final:iterations"]) == int(
            bundle.configuration["max_iterations"]
        )
        assert native.raw_status == "gsco_maximum_iteration_reached"
        assert native.normalized_status == "budget_exhausted"
        assert native.success is False
    # Reported diagnostic, never a success gate.
    assert float(jax.values["final:normal_objective"]) < float(
        jax.values["initial:normal_objective"]
    )


def test_multistep_outer_guard_cannot_report_success() -> None:
    bundle = InputBundle(
        schema_version=2,
        case_id="native-wireframe-gsco-multistep",
        scale="native_default",
        random_seed=0,
        configuration=multistep_configuration("native_default"),
        configuration_fingerprint="configuration",
        arrays={},
        input_fingerprint="input",
    )
    values = {
        "initial:normal_objective": np.asarray(2.0),
        "final:normal_objective": np.asarray(1.0),
        "final:currents": np.asarray([1.0]),
        "final:constraints_satisfied": np.asarray(True),
        "final:adjustment_run": np.asarray(False),
        "final:allocated_iteration_budget": np.asarray(2_500),
        "history:stage_normal_objective": np.asarray([1.0]),
        "history:stage_iterations": np.asarray([2_167], dtype=np.int64),
        "history:stage_allocated_iteration_budget": np.asarray([2_500], dtype=np.int64),
    }
    bounded_bundle = InputBundle(
        schema_version=2,
        case_id="native-wireframe-gsco-multistep",
        scale="bounded",
        random_seed=0,
        configuration=multistep_configuration("bounded"),
        configuration_fingerprint="configuration",
        arrays={},
        input_fingerprint="input",
    )

    exhausted = multistep_observation(
        "native-cpu",
        bundle,
        {},
        backend_mode="native_cpu",
        platform="cpu",
        driver="simsopt_cpp_multistep_gsco",
        values=values,
    )
    assert exhausted.success is False
    assert exhausted.normalized_status == "failed"
    # No outer guard is configured at this scale, so the only other stop is the
    # declared stage capacity, and it is named as such.
    assert exhausted.raw_status == (
        "stage_history_capacity_exhausted_without_final_adjustment"
    )

    bounded_exhausted = multistep_observation(
        "native-cpu",
        bounded_bundle,
        {},
        backend_mode="native_cpu",
        platform="cpu",
        driver="simsopt_cpp_multistep_gsco",
        values=values,
    )
    assert bounded_exhausted.success is False
    assert bounded_exhausted.raw_status == (
        "outer_step_guard_exhausted_without_final_adjustment"
    )

    values["final:adjustment_run"] = np.asarray(True)
    completed = multistep_observation(
        "native-cpu",
        bundle,
        {},
        backend_mode="native_cpu",
        platform="cpu",
        driver="simsopt_cpp_multistep_gsco",
        values=values,
    )
    assert completed.success is True
    assert completed.raw_status == "stable_current_final_adjustment_complete"

    values["final:constraints_satisfied"] = np.asarray(False)
    rejected = multistep_observation(
        "native-cpu",
        bundle,
        {},
        backend_mode="native_cpu",
        platform="cpu",
        driver="simsopt_cpp_multistep_gsco",
        values=values,
    )
    assert rejected.success is False
    assert rejected.raw_status == "failed_endpoint_after_final_adjustment"


def _multistep_label_values(
    stage_iterations: list[int],
    stage_budget: list[int],
) -> dict[str, np.ndarray]:
    return {
        "initial:normal_objective": np.asarray(2.0),
        "final:normal_objective": np.asarray(1.0),
        "final:currents": np.asarray([1.0]),
        "final:constraints_satisfied": np.asarray(True),
        "final:adjustment_run": np.asarray(True),
        "final:allocated_iteration_budget": np.asarray(sum(stage_budget)),
        "history:stage_normal_objective": np.asarray([1.0] * len(stage_iterations)),
        "history:stage_iterations": np.asarray(stage_iterations, dtype=np.int64),
        "history:stage_allocated_iteration_budget": np.asarray(
            stage_budget, dtype=np.int64
        ),
    }


def _multistep_native_default_bundle() -> InputBundle:
    return InputBundle(
        schema_version=2,
        case_id="native-wireframe-gsco-multistep",
        scale="native_default",
        random_seed=0,
        configuration=multistep_configuration("native_default"),
        configuration_fingerprint="configuration",
        arrays={},
        input_fingerprint="input",
    )


def _multistep_label(values: dict[str, np.ndarray]):
    return multistep_observation(
        "native-cpu",
        _multistep_native_default_bundle(),
        {},
        backend_mode="native_cpu",
        platform="cpu",
        driver="simsopt_cpp_multistep_gsco",
        values=values,
    )


def test_multistep_stage_at_its_budget_is_never_converged() -> None:
    """A staged GSCO call that reached ``max_iter`` is upstream's stop_last_iter.

    The official capture's seven stages stop at 2167/763/539/379/155/2/136 of a
    2500 budget, so the official run stays ``converged``; a stage that used its
    whole budget is a budget stop and the lane reports it as one, even though
    the endpoint objective decreased (which is what the removed predicate
    checked).
    """

    official_like = _multistep_label(
        _multistep_label_values(
            [2_167, 763, 539, 379, 155, 2, 136],
            [2_500] * 7,
        )
    )
    assert official_like.normalized_status == "converged"
    assert official_like.success is True
    assert official_like.raw_status == "stable_current_final_adjustment_complete"

    exhausted = _multistep_label(
        _multistep_label_values(
            [2_167, 2_500, 539, 379, 155, 2, 136],
            [2_500] * 7,
        )
    )
    assert exhausted.normalized_status == "budget_exhausted"
    assert exhausted.success is False
    assert exhausted.raw_status == "gsco_multistep_stage_budget_reached"
    # The scientific predicate the label no longer uses still holds here: the
    # old code reported this run as converged.
    assert float(exhausted.values["final:normal_objective"]) < float(
        exhausted.values["initial:normal_objective"]
    )


def test_multistep_first_stage_without_an_accepted_update_fails() -> None:
    """Zero accepted loops in the first stage moved nothing at all."""

    degenerate = _multistep_label(_multistep_label_values([0, 763, 539], [2_500] * 3))
    assert degenerate.normalized_status == "failed"
    assert degenerate.success is False
    assert degenerate.raw_status == "gsco_multistep_first_stage_accepted_no_update"
    assert float(degenerate.values["final:normal_objective"]) < float(
        degenerate.values["initial:normal_objective"]
    )


def test_multistep_uses_natural_termination_only_for_native_default() -> None:
    native_configuration = multistep_configuration("native_default")
    bounded_configuration = multistep_configuration("bounded")

    assert native_configuration["max_outer_steps"] is None
    assert bounded_configuration["max_outer_steps"] == 4
    assert (
        native_configuration["stage_history_capacity"]
        == bounded_configuration["stage_history_capacity"]
        == WIREFRAME_GSCO_MULTISTEP_STAGE_CAPACITY
    )


def test_public_multistep_example_discloses_completion_and_unknown_iterations() -> None:
    repository = Path(__file__).resolve().parents[2]
    example = repository / "examples/jax/3_Advanced/wireframe_gsco_multistep.py"
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [sys.executable, str(example), "--smoke", "--json"],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    published = json.loads(completed.stdout.splitlines()[-1])
    observables = published["observables"]
    assert published["status"] == "ok"
    assert observables["final_adjustment_run"] is True
    assert observables["constraints_satisfied"] is True
    # The example derives its completion from the same per-stage arrays as the
    # parity case, never from "final normal error below initial", which it now
    # publishes as a diagnostic only.
    assert observables["terminal_status"] == "converged"
    assert observables["terminal_reason"] == "stable_current_final_adjustment_complete"
    assert observables["solver_success"] is True
    assert isinstance(observables["normal_error_decreased"], bool)
    assert all(
        iterations < budget
        for iterations, budget in zip(
            observables["stage_iterations"],
            observables["stage_allocated_iteration_budget"],
            strict=True,
        )
    )
    assert observables["iterations"] is None
    assert observables["allocated_iteration_budget"] == 120
    assert len(observables["stage_iterations"]) == 3
    assert observables["stage_allocated_iteration_budget"] == [40, 40, 40]
    assert all(0 <= value <= 40 for value in observables["stage_iterations"])
    assert len(observables["stage_objectives"]) == 3
