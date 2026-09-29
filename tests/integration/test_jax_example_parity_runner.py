from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import examples.jax.run_parity as parity_cli
import numpy as np
import pytest
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.outer_optimizer_policy import (
    OuterOptimizerPolicyError,
    parse_outer_optimizer_policy,
    policy_owns_parity_case,
)
from examples.jax.parity._manifest import ComparisonRoute
from examples.jax.parity.arbiter import (
    ArbitrationError,
    LaneObservation,
    LaneOutcomeRejection,
    arbitrate,
)
from examples.jax.parity.artifacts import (
    canonical_json_bytes,
    write_array,
    write_bytes_exclusive,
)
from examples.jax.parity.audit import audit_published_run
from examples.jax.parity.cases import (
    get_case,
    implemented_case_ids,
)
from examples.jax.parity.contracts import QualityBand
from examples.jax.parity.official_quality_bands import OFFICIAL_BAND_CASE_IDS
from examples.jax.parity.official_reference import load_official_sensitivity
from examples.jax.parity.input_bundle import create_input_bundle, read_input_bundle
from examples.jax.parity.provenance import (
    REQUIRED_PROVENANCE_SOURCE_PATHS,
    DeviceMetadata,
    ExecutedSource,
    LaneProvenance,
    collect_explicit_sources,
    collect_repository_state,
    generated_version_matches_checkout,
)
from examples.jax.parity.publication import begin_run, publish_run
from examples.jax.parity.receipts import write_lane_observation
from examples.jax.parity.runner import (
    ChildExecution,
    ChildProcessResult,
    RunnerError,
    build_child_command,
    execute_case_lanes,
    execute_child_process,
)
from examples.jax.parity.work_budget import WorkBudgetContract
from simsopt.single_stage_boozer_vacuum import JAX_FAST_DRIVER_ID
from simsopt_jax.config import ExecutionIntent
from simsopt_jax.examples import ExecutionScale


def _routes() -> tuple[ComparisonRoute, ...]:
    return tuple(
        ComparisonRoute(
            phase="initial",
            observable="objective_sum_squares",
            lane_pair=lane_pair,
            applicable=True,
            comparator="allclose",
            tolerance_bucket=(
                "gpu_runtime" if lane_pair == "jax-cpu:jax-gpu" else "native_workflow"
            ),
        )
        for lane_pair in (
            "native-cpu:jax-cpu",
            "native-cpu:jax-gpu",
            "jax-cpu:jax-gpu",
        )
    )


def _provenance(backend_mode: str) -> LaneProvenance:
    is_jax = backend_mode != "native_cpu"
    platform = "gpu" if backend_mode.startswith("jax_gpu_") else "cpu"
    transfer_guard = "log" if backend_mode == "jax_gpu_fast" else "disallow"
    policy = {"SIMSOPT_BACKEND_MODE": backend_mode}
    if platform == "gpu":
        policy.update(
            {
                "SIMSOPT_JAX_TRANSFER_GUARD": transfer_guard,
                "JAX_TRANSFER_GUARD": transfer_guard,
            }
        )
    return LaneProvenance(
        repository_commit="d" * 40,
        repository_dirty=False,
        tracked_diff_sha256="e" * 64,
        untracked_files=(),
        executed_sources=(
            ExecutedSource("examples/jax/run_parity.py", "f" * 64, "a" * 40),
        ),
        python_version="3.11.0",
        jax_version="0.10.0" if is_jax else None,
        simsopt_version="1.10.0",
        simsopt_version_commit="g" + "d" * 9,
        simsopt_version_checkout_compatible=True,
        lane_environment_policy=policy,
        jax_effective_transfer_guards=(
            {
                "device_to_device": transfer_guard,
                "device_to_host": transfer_guard,
                "host_to_device": transfer_guard,
            }
            if platform == "gpu"
            else {}
        ),
        devices=(DeviceMetadata(0, platform, "test", 0),) if is_jax else (),
        host_peak_rss_bytes=1024,
        host_peak_rss_method="test fixture",
        device_memory_peak_bytes=None,
        device_memory_status="unavailable: test fixture",
        memory_measurement_scope=(
            "combined import, compile/warmup, and one bounded execution"
        ),
        steady_state_memory_measured=False,
        measurement_synchronization=(
            "jax.block_until_ready over published observation values"
            if is_jax
            else "native synchronous execution"
        ),
        simsoptpp_path=None,
        simsoptpp_sha256=None,
        simsoptpp_version=None,
        simsoptpp_build_commit=None,
        simsoptpp_checkout_compatible=None,
        authoritative=True,
    )


def _observations() -> dict[str, LaneObservation]:
    return {
        "native-cpu": LaneObservation(
            lane="native-cpu",
            backend_mode="native_cpu",
            platform="cpu",
            precision="fp64",
            scale="bounded",
            input_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
            effective_construction_fingerprint="c" * 64,
            driver="scipy_least_squares",
            normalized_status="converged",
            raw_status="1",
            success=True,
            nit=None,
            nfev=3,
            njev=3,
            completed_workflow_stages=("construct", "evaluate"),
            provenance=_provenance("native_cpu"),
            values={"initial:objective_sum_squares": np.array(1.0)},
        ),
        "jax-cpu": LaneObservation(
            lane="jax-cpu",
            backend_mode="jax_cpu_parity",
            platform="cpu",
            precision="fp64",
            scale="bounded",
            input_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
            effective_construction_fingerprint="c" * 64,
            driver="simsopt_lm_qr",
            normalized_status="converged",
            raw_status="0",
            success=True,
            nit=2,
            nfev=3,
            njev=3,
            completed_workflow_stages=("construct", "evaluate"),
            provenance=_provenance("jax_cpu_parity"),
            values={"initial:objective_sum_squares": np.array(1.0)},
        ),
        "jax-gpu": LaneObservation(
            lane="jax-gpu",
            backend_mode="jax_gpu_parity",
            platform="gpu",
            precision="fp64",
            scale="bounded",
            input_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
            effective_construction_fingerprint="c" * 64,
            driver="simsopt_lm_qr",
            normalized_status="converged",
            raw_status="0",
            success=True,
            nit=2,
            nfev=3,
            njev=3,
            completed_workflow_stages=("construct", "evaluate"),
            provenance=_provenance("jax_gpu_parity"),
            values={"initial:objective_sum_squares": np.array(1.0)},
        ),
    }


def _fast_observations() -> dict[str, LaneObservation]:
    observations = _observations()
    for lane, backend_mode in (
        ("jax-cpu", "jax_cpu_fast"),
        ("jax-gpu", "jax_gpu_fast"),
    ):
        observations[lane] = dataclasses.replace(
            observations[lane],
            backend_mode=backend_mode,
            driver=JAX_FAST_DRIVER_ID,
            provenance=_provenance(backend_mode),
        )
    return observations


def test_native_workflow_tolerance_is_centrally_owned_and_adversarial() -> None:
    tolerance = parity_ladder_tolerances("native_workflow")

    assert tolerance["requires_native_workflow_oracle"] is True
    assert tolerance["requires_direct_cpp_oracle"] is False
    assert float(tolerance["same_state_value_rtol"]) < 1.0e-8
    assert float(tolerance["terminal_relative_reduction"]) == 1.0e-12
    assert float(tolerance["terminal_constraint_norm_atol"]) == 1.0e-10
    assert float(tolerance["terminal_orthonormality_atol"]) == 1.0e-12


@pytest.mark.parametrize(
    ("bucket", "rtol", "atol"),
    (
        ("mirror_boozer_value", 1.0e-3, 1.0e-8),
        ("mirror_boozer_parameters", 0.0, 2.0e-3),
        ("mirror_optimization_5e2", 5.0e-2, 1.0e-9),
        ("mirror_optimization_3e2", 3.0e-2, 1.0e-9),
        ("mirror_optimization_2e2", 2.0e-2, 1.0e-9),
        ("mirror_optimization_5e3", 5.0e-3, 1.0e-10),
        ("mirror_optimization_1e1", 1.0e-1, 1.0e-9),
        ("mirror_pmqa_final", 5.0e-4, 0.0),
        ("mirror_qfm_value", 5.0e-5, 1.0e-7),
        ("mirror_qfm_parameters", 2.0e-3, 2.0e-4),
        ("mirror_qfm_persistence", 2.0e-2, 1.0e-5),
        ("mirror_surface_invariant", 1.0e-8, 1.0e-10),
        ("mirror_trace_ncsx_time", 0.0, 2.0e-2),
        ("mirror_trace_ncsx_state", 0.0, 3.0e-2),
        ("mirror_trace_qa_time", 0.0, 6.0e-3),
        ("mirror_trace_qa_state", 0.0, 2.0e-3),
        ("mirror_trace_qa_poincare", 0.0, 7.0e-3),
        ("mirror_trace_particle_time", 0.0, 2.0e-6),
        ("mirror_trace_particle_state", 0.0, 2.0e-3),
    ),
)
def test_mirror_parity_tolerances_preserve_source_owned_thresholds(
    bucket: str,
    rtol: float,
    atol: float,
) -> None:
    tolerance = parity_ladder_tolerances(bucket)

    assert tolerance["rtol"] == rtol
    assert tolerance["atol"] == atol


def test_generated_version_source_must_name_the_clean_checkout() -> None:
    repository_commit = "123456789abcdef" + "0" * 25

    assert generated_version_matches_checkout(repository_commit, "g123456789")
    assert not generated_version_matches_checkout(repository_commit, "gabcdef123")
    assert not generated_version_matches_checkout(repository_commit, None)


def test_arbiter_requires_direct_all_pairs_and_passes_matching_receipts() -> None:
    result = arbitrate(_routes(), _observations())

    assert result.verdict == "pass"
    assert len(result.comparisons) == 3
    assert all(comparison.passed for comparison in result.comparisons)


def test_arbiter_accepts_truthful_fast_receipts_only_with_fast_intent() -> None:
    result = arbitrate(_routes(), _fast_observations(), execution_intent="fast")

    assert result.verdict == "pass"


def test_arbiter_default_parity_intent_rejects_fast_receipts() -> None:
    with pytest.raises(ArbitrationError, match="backend_mode must be jax_cpu_parity"):
        arbitrate(_routes(), _fast_observations())


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    (
        ("backend", "backend_mode must be jax_gpu_fast"),
        ("provenance", "provenance backend policy mismatch"),
        ("guard_policy", "effective transfer guards must log"),
        ("effective_guard", "effective transfer guards must log"),
    ),
)
def test_fast_intent_rejects_runtime_provenance_mismatches(
    mutation: str, expected_message: str
) -> None:
    observations = _fast_observations()
    gpu = observations["jax-gpu"]
    provenance = gpu.provenance
    assert provenance is not None
    if mutation == "backend":
        observations["jax-gpu"] = dataclasses.replace(
            gpu, backend_mode="jax_gpu_parity"
        )
    elif mutation == "provenance":
        observations["jax-gpu"] = dataclasses.replace(
            gpu,
            provenance=dataclasses.replace(
                provenance,
                lane_environment_policy={
                    **provenance.lane_environment_policy,
                    "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
                },
            ),
        )
    elif mutation == "guard_policy":
        observations["jax-gpu"] = dataclasses.replace(
            gpu,
            provenance=dataclasses.replace(
                provenance,
                lane_environment_policy={
                    **provenance.lane_environment_policy,
                    "JAX_TRANSFER_GUARD": "disallow",
                },
            ),
        )
    elif mutation == "effective_guard":
        observations["jax-gpu"] = dataclasses.replace(
            gpu,
            provenance=dataclasses.replace(
                provenance,
                jax_effective_transfer_guards={
                    direction: "disallow"
                    for direction in provenance.jax_effective_transfer_guards
                },
            ),
        )
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ArbitrationError, match=expected_message):
        arbitrate(_routes(), observations, execution_intent="fast")


def test_arbiter_rejects_invalid_execution_intent() -> None:
    with pytest.raises(ArbitrationError, match="invalid execution intent"):
        arbitrate(
            _routes(),
            _observations(),
            execution_intent=cast(ExecutionIntent, "benchmark"),
        )


def _lane_outcome_observations(mutation: str) -> dict[str, LaneObservation]:
    """Put one lane into a terminal state the arbiter must reject as a result."""
    observations = _observations()
    if mutation == "budget_exhausted_reports_success":
        observations["jax-cpu"] = dataclasses.replace(
            observations["jax-cpu"],
            normalized_status="budget_exhausted",
            success=True,
        )
    elif mutation == "failed_reports_success":
        observations["jax-cpu"] = dataclasses.replace(
            observations["jax-cpu"], normalized_status="failed", success=True
        )
    elif mutation == "no_scientific_success":
        observations["jax-cpu"] = dataclasses.replace(
            observations["jax-cpu"], normalized_status="failed", success=False
        )
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")
    return observations


def _integrity_arbitration_inputs(
    mutation: str,
) -> tuple[tuple[ComparisonRoute, ...], dict[str, LaneObservation]]:
    """Break one harness/contract invariant that says nothing about an outcome."""
    routes = _routes()
    observations = _observations()
    jax_cpu = observations["jax-cpu"]
    provenance = jax_cpu.provenance
    assert provenance is not None
    if mutation == "backend_mode":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu, backend_mode="jax_gpu_parity"
        )
    elif mutation == "platform":
        observations["jax-cpu"] = dataclasses.replace(jax_cpu, platform="gpu")
    elif mutation == "precision":
        observations["jax-cpu"] = dataclasses.replace(jax_cpu, precision="fp32")
    elif mutation == "provenance_policy":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu,
            provenance=dataclasses.replace(
                provenance,
                lane_environment_policy={"SIMSOPT_BACKEND_MODE": "native_cpu"},
            ),
        )
    elif mutation == "device_provenance":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu,
            provenance=dataclasses.replace(
                provenance, devices=(DeviceMetadata(0, "tpu", "test", 0),)
            ),
        )
    elif mutation == "forbidden_driver":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu, driver="scipy_least_squares_trf"
        )
    elif mutation == "workflow_stages":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu, completed_workflow_stages=("construct",)
        )
    elif mutation == "fingerprint":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu, input_fingerprint="z" * 64
        )
    elif mutation == "duplicate_sources":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu,
            provenance=dataclasses.replace(
                provenance,
                executed_sources=(
                    ExecutedSource("examples/jax/run_parity.py", "f" * 64, "a" * 40),
                    ExecutedSource("examples/jax/run_parity.py", "0" * 64, "a" * 40),
                ),
            ),
        )
    elif mutation == "repository_provenance":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu,
            provenance=dataclasses.replace(provenance, repository_commit="0" * 40),
        )
    elif mutation == "executed_source":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu,
            provenance=dataclasses.replace(
                provenance,
                executed_sources=(
                    ExecutedSource("examples/jax/run_parity.py", "0" * 64, "a" * 40),
                ),
            ),
        )
    elif mutation == "route_matrix":
        observations = {
            lane: dataclasses.replace(
                observation,
                values={
                    **observation.values,
                    "final:residual": np.zeros(2, dtype=np.float64),
                },
                applicability={},
            )
            for lane, observation in observations.items()
        }
    elif mutation == "no_applicable_routes":
        routes = tuple(dataclasses.replace(route, applicable=False) for route in routes)
    elif mutation == "non_finite_observable":
        observations["jax-cpu"] = dataclasses.replace(
            jax_cpu,
            values={"initial:objective_sum_squares": np.array(np.nan)},
            applicability={},
        )
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")
    return routes, observations


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    (
        (
            "budget_exhausted_reports_success",
            "jax-cpu budget_exhausted cannot report success",
        ),
        ("failed_reports_success", "jax-cpu provider reported failure"),
        ("no_scientific_success", "jax-cpu did not report scientific success"),
    ),
)
def test_only_lane_outcome_gates_raise_a_recordable_rejection(
    mutation: str, expected_message: str
) -> None:
    """These three gates report a lane's OWN result, so a run may record them."""
    with pytest.raises(LaneOutcomeRejection, match=expected_message):
        arbitrate(_routes(), _lane_outcome_observations(mutation))


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    (
        ("backend_mode", "backend_mode must be jax_cpu_parity"),
        ("platform", "platform must be cpu"),
        ("precision", "precision must be fp64"),
        ("provenance_policy", "provenance backend policy mismatch"),
        ("device_provenance", "device provenance mismatch"),
        ("forbidden_driver", "forbidden parity driver"),
        ("workflow_stages", "workflow stage mismatch"),
        ("fingerprint", "input fingerprint mismatch"),
        ("duplicate_sources", "duplicate executed source paths"),
        ("repository_provenance", "repository provenance mismatch"),
        ("executed_source", "executed source mismatch"),
        ("route_matrix", "complete direct lane-pair matrix"),
        ("no_applicable_routes", "no applicable comparison routes"),
        ("non_finite_observable", "non-finite required observable"),
    ),
)
def test_integrity_violations_are_never_a_lane_outcome_rejection(
    mutation: str, expected_message: str
) -> None:
    """A harness/contract violation is not a scientific result of any lane.

    ``run_parity`` records a ``LaneOutcomeRejection`` as a failed case and
    writes a summary; every error below must instead abort the whole run, so
    none of them may be an instance of that class.
    """
    routes, observations = _integrity_arbitration_inputs(mutation)

    with pytest.raises(ArbitrationError, match=expected_message) as raised:
        arbitrate(routes, observations)
    assert not isinstance(raised.value, LaneOutcomeRejection)


def test_arbiter_rejects_applicable_observable_without_routes() -> None:
    observations = {
        lane: dataclasses.replace(
            observation,
            values={
                **observation.values,
                "final:residual_jacobian": np.eye(1, dtype=np.float64),
            },
            applicability={},
        )
        for lane, observation in _observations().items()
    }

    with pytest.raises(ArbitrationError, match="complete direct lane-pair matrix"):
        arbitrate(_routes(), observations)


def test_arbiter_accepts_explicit_noncertifying_diagnostic_routes() -> None:
    observations = {
        lane: dataclasses.replace(
            observation,
            values={
                **observation.values,
                "final:diagnostic_trace": np.arange(3, dtype=np.float64),
            },
            applicability={},
        )
        for lane, observation in _observations().items()
    }
    diagnostic_routes = tuple(
        ComparisonRoute(
            phase="final",
            observable="diagnostic_trace",
            lane_pair=lane_pair,
            applicable=False,
            comparator="allclose",
            tolerance_bucket="native_workflow",
        )
        for lane_pair in (
            "native-cpu:jax-cpu",
            "native-cpu:jax-gpu",
            "jax-cpu:jax-gpu",
        )
    )

    result = arbitrate((*_routes(), *diagnostic_routes), observations)

    assert result.verdict == "pass"
    assert len(result.comparisons) == 3


def test_arbiter_accepts_a_source_owned_not_worse_objective() -> None:
    observations = {
        lane: dataclasses.replace(
            observation,
            values={
                "initial:objective_sum_squares": np.asarray(
                    1.0 if lane == "native-cpu" else 0.5,
                    dtype=np.float64,
                )
            },
            applicability={},
        )
        for lane, observation in _observations().items()
        if lane in {"native-cpu", "jax-cpu"}
    }
    route = ComparisonRoute(
        phase="initial",
        observable="objective_sum_squares",
        lane_pair="native-cpu:jax-cpu",
        applicable=True,
        comparator="not_worse",
        tolerance_bucket="mirror_optimization_3e2",
    )

    result = arbitrate(
        (route,),
        observations,
        required_lanes=frozenset({"native-cpu", "jax-cpu"}),
    )

    assert result.verdict == "pass"


def test_lane_receipts_expose_explicit_observable_applicability() -> None:
    observation = _observations()["native-cpu"]

    assert observation.applicability == {
        "initial:objective_sum_squares": True,
        "optimizer_outcome": True,
    }


def test_memory_receipt_distinguishes_compile_and_steady_state_scope() -> None:
    provenance = _provenance("jax_gpu_parity")

    assert provenance.memory_measurement_scope == (
        "combined import, compile/warmup, and one bounded execution"
    )
    assert provenance.steady_state_memory_measured is False
    assert provenance.measurement_synchronization.startswith("jax.block_until_ready")


def test_gpu_receipt_records_effective_jax_transfer_guards() -> None:
    provenance = _provenance("jax_gpu_parity")

    assert provenance.jax_effective_transfer_guards == {
        "device_to_device": "disallow",
        "device_to_host": "disallow",
        "host_to_device": "disallow",
    }


def test_arbiter_accepts_unrelated_dirty_worktree_drift() -> None:
    observations = _observations()
    jax_cpu_provenance = observations["jax-cpu"].provenance
    assert jax_cpu_provenance is not None
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"],
        provenance=dataclasses.replace(
            jax_cpu_provenance,
            repository_dirty=True,
            tracked_diff_sha256="1" * 64,
            untracked_files=("unrelated.txt",),
            authoritative=False,
        ),
    )

    result = arbitrate(_routes(), observations)

    assert result.verdict == "pass"


def test_arbiter_rejects_shared_executed_source_hash_mismatch() -> None:
    observations = _observations()
    jax_cpu_provenance = observations["jax-cpu"].provenance
    assert jax_cpu_provenance is not None
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"],
        provenance=dataclasses.replace(
            jax_cpu_provenance,
            executed_sources=(
                ExecutedSource("examples/jax/run_parity.py", "1" * 64, "a" * 40),
            ),
        ),
    )

    with pytest.raises(ArbitrationError, match="executed source mismatch"):
        arbitrate(_routes(), observations)


def test_arbiter_rejects_workflow_stage_mismatch() -> None:
    observations = _observations()

    with pytest.raises(ArbitrationError, match="workflow stage mismatch"):
        arbitrate(
            _routes(),
            observations,
            expected_workflow_stages=("construct", "evaluate", "solve"),
        )


def test_self_reported_scipy_driver_does_not_waive_scipy_policy() -> None:
    stages = (
        "construct_ncsx_coils_and_volume_labelled_surface",
        "solve_initial_boozer_surface",
        "assemble_nonqs_residual_iota_radius_and_length_objective",
        "evaluate_initial_objective_and_gradient",
        "optimize_coils_and_currents_with_bfgs",
        "record_final_objective_gradient_and_implicit_physics_state",
    )
    observations = _observations()
    observations = {
        lane: dataclasses.replace(
            observation,
            completed_workflow_stages=stages,
            driver=(
                "simsopt_jax_scipy_bfgs_outer_driver"
                if lane.startswith("jax-")
                else observation.driver
            ),
        )
        for lane, observation in observations.items()
    }

    with pytest.raises(ArbitrationError, match="forbidden parity driver"):
        arbitrate(_routes(), observations, expected_workflow_stages=stages)
    with pytest.raises(ArbitrationError, match="forbidden parity driver"):
        arbitrate(
            _routes(),
            observations,
            expected_workflow_stages=stages[:-1] + ("different_final_stage",),
        )


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("missing_lane", "missing required lane"),
        ("wrong_gpu", "jax-gpu platform must be gpu"),
        ("input_mismatch", "input fingerprint mismatch"),
        ("effective_mismatch", "effective construction fingerprint mismatch"),
        ("nonfinite", "non-finite"),
        ("wrong_float_dtype", "must be FP64"),
        ("missing_direct_pair", "complete direct lane-pair matrix"),
        ("hidden_host_solver", "forbidden parity driver"),
        ("scientific_failure", "scientific success"),
    ],
)
def test_arbiter_fails_closed_on_invalid_receipts(
    mutation: str, expected_message: str
) -> None:
    observations = _observations()
    routes = _routes()
    if mutation == "missing_lane":
        observations.pop("jax-gpu")
    elif mutation == "wrong_gpu":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"], platform="cpu"
        )
    elif mutation == "input_mismatch":
        observations["jax-cpu"] = dataclasses.replace(
            observations["jax-cpu"], input_fingerprint="d" * 64
        )
    elif mutation == "effective_mismatch":
        observations["jax-cpu"] = dataclasses.replace(
            observations["jax-cpu"], effective_construction_fingerprint="d" * 64
        )
    elif mutation == "nonfinite":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"],
            values={"initial:objective_sum_squares": np.array(np.nan)},
        )
    elif mutation == "wrong_float_dtype":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"],
            values={"initial:objective_sum_squares": np.array(1.0, dtype=np.float32)},
        )
    elif mutation == "missing_direct_pair":
        routes = tuple(
            route for route in routes if route.lane_pair != "native-cpu:jax-gpu"
        )
    elif mutation == "hidden_host_solver":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"], driver="scipy_host_callback"
        )
    elif mutation == "scientific_failure":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"], success=False, normalized_status="failed"
        )
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ArbitrationError, match=expected_message):
        arbitrate(routes, observations)


def test_direct_native_gpu_gate_catches_transitive_tolerance_drift() -> None:
    tolerance = parity_ladder_tolerances("native_workflow")
    absolute_tolerance = float(tolerance["same_state_value_atol"])
    observations = _observations()
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"],
        values={"initial:objective_sum_squares": np.array(0.75 * absolute_tolerance)},
    )
    observations["jax-gpu"] = dataclasses.replace(
        observations["jax-gpu"],
        values={"initial:objective_sum_squares": np.array(1.5 * absolute_tolerance)},
    )
    observations["native-cpu"] = dataclasses.replace(
        observations["native-cpu"],
        values={"initial:objective_sum_squares": np.array(0.0)},
    )

    result = arbitrate(_routes(), observations)

    comparisons = {
        comparison.lane_pair: comparison for comparison in result.comparisons
    }
    assert comparisons["native-cpu:jax-cpu"].passed
    assert comparisons["jax-cpu:jax-gpu"].passed
    assert not comparisons["native-cpu:jax-gpu"].passed
    assert result.verdict == "fail"


# ---------------------------------------------------------- quality-band gate
#
# Rule 3 of the 2026-08-15 native_default certification-gate ruling. The lane
# numbers below are the measured 2026-08-14 three-lane native_default run of
# the boozer-vacuum mirror (durable archive
# ~/simsopt-campaigns/ndparity-boozer-vacuum-20260814/): every lane
# budget_exhausted at the matched 1000-iteration budget, final objectives
# 4.3972e-08 / 4.5074e-08 / 4.5614e-08, forks of 2.5e-2 and 3.7e-2 against the
# mirror_single_stage_final_value equality bucket's rtol 2e-8.
_ARCHIVED_FINAL_OBJECTIVE = {
    "native-cpu": 4.3972015892540963e-08,
    "jax-cpu": 4.5074090114235766e-08,
    "jax-gpu": 4.561437157279435e-08,
}
_ARCHIVED_MATCHED_BUDGET = 1000
_ARCHIVED_QUALITY_BAND = QualityBand(
    observable="final:objective",
    max_value=1.0e-07,
    derivation="test fixture mirroring the 2026-08-14 native_default archive",
)


def _final_objective_routes() -> tuple[ComparisonRoute, ...]:
    return tuple(
        ComparisonRoute(
            phase="final",
            observable="objective",
            lane_pair=lane_pair,
            applicable=True,
            comparator="allclose",
            tolerance_bucket="mirror_single_stage_final_value",
        )
        for lane_pair in (
            "native-cpu:jax-cpu",
            "native-cpu:jax-gpu",
            "jax-cpu:jax-gpu",
        )
    )


def _native_default_observations(
    *,
    final_objective: dict[str, float] | None = None,
    budgets: dict[str, int] | None = None,
    statuses: dict[str, str] | None = None,
    successes: dict[str, bool] | None = None,
) -> dict[str, LaneObservation]:
    objectives = final_objective or _ARCHIVED_FINAL_OBJECTIVE
    nits = budgets or dict.fromkeys(_ARCHIVED_FINAL_OBJECTIVE, _ARCHIVED_MATCHED_BUDGET)
    normalized = statuses or dict.fromkeys(
        _ARCHIVED_FINAL_OBJECTIVE, "budget_exhausted"
    )
    reported = successes or dict.fromkeys(_ARCHIVED_FINAL_OBJECTIVE, False)
    return {
        lane: dataclasses.replace(
            observation,
            scale="native_default",
            normalized_status=normalized[lane],
            raw_status="stopping_reason=iteration-limit",
            success=reported[lane],
            nit=nits[lane],
            values={
                "initial:objective_sum_squares": np.asarray(1.0, dtype=np.float64),
                "final:objective": np.asarray([objectives[lane]], dtype=np.float64),
            },
            applicability={},
        )
        for lane, observation in _observations().items()
    }


def test_quality_band_certifies_matched_budget_exhausted_lanes() -> None:
    result = arbitrate(
        (*_routes(), *_final_objective_routes()),
        _native_default_observations(),
        quality_band=_ARCHIVED_QUALITY_BAND,
    )

    assert result.verdict == "quality-band"
    assert result.verdict != "pass"
    assert [item.lane for item in result.quality_band_results] == [
        "jax-cpu",
        "jax-gpu",
        "native-cpu",
    ]
    assert all(item.passed for item in result.quality_band_results)
    assert all(
        item.observable == "final:objective" and item.max_value == 1.0e-07
        for item in result.quality_band_results
    )
    final_objective_comparisons = [
        comparison
        for comparison in result.comparisons
        if comparison.phase == "final" and comparison.observable == "objective"
    ]
    assert len(final_objective_comparisons) == 3
    assert not any(comparison.passed for comparison in final_objective_comparisons)
    assert all(
        comparison.diagnostic.startswith("informational (quality-band, non-certifying)")
        for comparison in result.comparisons
    )


def test_quality_band_rejects_an_endpoint_outside_the_band() -> None:
    result = arbitrate(
        (*_routes(), *_final_objective_routes()),
        _native_default_observations(
            final_objective={**_ARCHIVED_FINAL_OBJECTIVE, "jax-gpu": 4.6e-07}
        ),
        quality_band=_ARCHIVED_QUALITY_BAND,
    )

    assert result.verdict == "fail"
    outside = {item.lane: item.passed for item in result.quality_band_results}
    assert outside == {"jax-cpu": True, "jax-gpu": False, "native-cpu": True}


def test_quality_band_requires_one_identical_matched_budget() -> None:
    with pytest.raises(ArbitrationError, match="matched\n?\\s?iteration budget"):
        arbitrate(
            (*_routes(), *_final_objective_routes()),
            _native_default_observations(
                budgets={
                    "native-cpu": _ARCHIVED_MATCHED_BUDGET,
                    "jax-cpu": _ARCHIVED_MATCHED_BUDGET,
                    "jax-gpu": _ARCHIVED_MATCHED_BUDGET - 1,
                }
            ),
            quality_band=_ARCHIVED_QUALITY_BAND,
        )


def test_quality_band_fails_closed_when_a_lane_converged_early() -> None:
    with pytest.raises(ArbitrationError, match="scientific success"):
        arbitrate(
            (*_routes(), *_final_objective_routes()),
            _native_default_observations(
                statuses={
                    "native-cpu": "converged",
                    "jax-cpu": "budget_exhausted",
                    "jax-gpu": "budget_exhausted",
                },
                successes={
                    "native-cpu": True,
                    "jax-cpu": False,
                    "jax-gpu": False,
                },
            ),
            quality_band=_ARCHIVED_QUALITY_BAND,
        )


def test_quality_band_never_certifies_converged_lanes_as_equivalence() -> None:
    result = arbitrate(
        (*_routes(), *_final_objective_routes()),
        _native_default_observations(
            statuses=dict.fromkeys(_ARCHIVED_FINAL_OBJECTIVE, "converged"),
            successes=dict.fromkeys(_ARCHIVED_FINAL_OBJECTIVE, True),
        ),
        quality_band=_ARCHIVED_QUALITY_BAND,
    )

    assert result.verdict == "quality-band"


def test_quality_band_is_refused_outside_native_default() -> None:
    observations = {
        lane: dataclasses.replace(
            observation,
            values={
                "initial:objective_sum_squares": np.asarray(1.0, dtype=np.float64),
                "final:objective": np.asarray(
                    [_ARCHIVED_FINAL_OBJECTIVE[lane]], dtype=np.float64
                ),
            },
            applicability={},
        )
        for lane, observation in _observations().items()
    }

    with pytest.raises(ArbitrationError, match="requires the native_default scale"):
        arbitrate(
            (*_routes(), *_final_objective_routes()),
            observations,
            quality_band=_ARCHIVED_QUALITY_BAND,
        )


def test_budget_exhausted_stays_fatal_without_a_declared_band() -> None:
    with pytest.raises(ArbitrationError, match="scientific success"):
        arbitrate(
            (*_routes(), *_final_objective_routes()),
            _native_default_observations(),
        )


def _work_budget_observations(scale: str) -> dict[str, LaneObservation]:
    return {
        lane: dataclasses.replace(
            observation,
            scale=cast(ExecutionScale, scale),
            normalized_status="budget_exhausted",
            raw_status="stopping_reason=iteration-limit",
            success=False,
            nit={"native-cpu": 8, "jax-cpu": 7, "jax-gpu": 6}[lane],
        )
        for lane, observation in _observations().items()
    }


def test_work_budget_contract_is_case_owned_and_scoped() -> None:
    declared = {
        case_id
        for case_id in implemented_case_ids()
        if get_case(case_id).work_budget_contract is not None
    }
    # The GSCO pair is bounded-only on purpose: upstream GSCO does accept an
    # explicit max_iter cap (wireframe_optimization.cpp:281), but the official
    # native_default and CI runs both stop earlier on "minimum objective
    # reached", so only the reduced bounded cap is ever reached.
    fixed_budget_scales = {
        "native-stage-two-optimization",
        "native-stage-two-optimization-planar-coils",
        "native-stage-two-optimization-stochastic",
        "native-strain-optimization",
    }
    # Finite build is bounded-only for a different reason: at native_default its verdict is the official endpoint
    # quality band, which already admits upstream's iteration-limit outcome and excludes a work budget there.
    # BoozerQA joined that group on 2026-09-21 (user decision C18: a post-hoc band at native_default, where
    # upstream's BFGS hits its 1000-iteration cap in every one of its own one-ulp runs).
    # Coil forces joined on 2026-09-21 (user decision C14, accepted 2026-09-20 23:48 EDT): at native_default its
    # verdict is the band plus one admitted provider termination upstream's own sample k = 5 produces.
    bounded_only = {
        "native-boozerqa",
        "native-coil-forces",
        "native-stage-two-optimization-finitebuild",
        "native-wireframe-gsco-modular",
        "native-wireframe-gsco-sector-saddle",
    }
    # Minimal stage two declares no work budget: upstream's native_default run ends at its L-BFGS-B iteration limit
    # (status 1, nit 300), which the official endpoint quality band admits, and the reduced scale converges.
    native_default_only: set[str] = set()
    assert declared == fixed_budget_scales | bounded_only | native_default_only
    expected_scales = {
        **{case_id: ("bounded", "native_default") for case_id in fixed_budget_scales},
        **{case_id: ("bounded",) for case_id in bounded_only},
        **{case_id: ("native_default",) for case_id in native_default_only},
    }
    for case_id in declared:
        contract = get_case(case_id).work_budget_contract
        assert contract is not None
        assert contract.scales == expected_scales[case_id]
        assert "Upstream" in contract.derivation
    assert get_case(_BAND_CASE_ID).work_budget_contract is None


def test_native_default_quality_band_conflicts_with_work_budget_contract() -> None:
    with pytest.raises(ValueError, match="cannot combine"):
        dataclasses.replace(
            get_case(_BAND_CASE_ID),
            work_budget_contract=WorkBudgetContract(
                scales=("native_default",),
                derivation="Upstream fixed optimizer budget",
            ),
        )


@pytest.mark.parametrize("scale", ("bounded", "native_default"))
def test_work_budget_admits_unequal_actual_counts_but_comparisons_decide(
    scale: str,
) -> None:
    contract = get_case("native-stage-two-optimization").work_budget_contract
    assert contract is not None
    observations = _work_budget_observations(scale)
    result = arbitrate(_routes(), observations, work_budget_contract=contract)
    assert result.verdict == "pass"
    assert result.work_budget_admitted is True
    assert result.quality_band_results == ()
    assert not any("informational" in item.diagnostic for item in result.comparisons)
    assert {item.nit for item in observations.values()} == {6, 7, 8}

    observations["jax-gpu"] = dataclasses.replace(
        observations["jax-gpu"],
        values={"initial:objective_sum_squares": np.asarray(2.0)},
    )
    failed = arbitrate(_routes(), observations, work_budget_contract=contract)
    assert failed.verdict == "fail"
    assert failed.work_budget_admitted is True
    assert not all(item.passed for item in failed.comparisons)


@pytest.mark.parametrize(
    ("scales", "observed_scale"),
    (
        (("native_default",), "bounded"),
        (("bounded",), "native_default"),
        (("bounded", "native_default"), "unexpected"),
    ),
)
def test_work_budget_rejects_undeclared_or_unknown_scale(
    scales: tuple[str, ...], observed_scale: str
) -> None:
    contract = WorkBudgetContract(
        scales=cast(tuple[ExecutionScale, ...], scales),
        derivation="Upstream fixed optimizer budget",
    )
    with pytest.raises(ArbitrationError, match="scientific success"):
        arbitrate(
            _routes(),
            _work_budget_observations(observed_scale),
            work_budget_contract=contract,
        )


@pytest.mark.parametrize(
    ("status", "success", "expected"),
    (
        ("converged", True, "scientific success"),
        ("failed", False, "scientific success"),
        ("failed", True, "provider reported failure"),
        ("unknown", False, "invalid normalized status"),
        ("budget_exhausted", True, "budget_exhausted cannot report success"),
    ),
)
def test_work_budget_rejects_mixed_failure_unknown_and_forged_success(
    status: str, success: bool, expected: str
) -> None:
    contract = get_case("native-stage-two-optimization").work_budget_contract
    assert contract is not None
    observations = _work_budget_observations("bounded")
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"], normalized_status=status, success=success
    )
    with pytest.raises(ArbitrationError, match=expected):
        arbitrate(_routes(), observations, work_budget_contract=contract)


@pytest.mark.parametrize(
    "fingerprint",
    (
        "input_fingerprint",
        "configuration_fingerprint",
        "effective_construction_fingerprint",
    ),
)
def test_work_budget_requires_matching_configured_inputs(fingerprint: str) -> None:
    contract = get_case("native-stage-two-optimization").work_budget_contract
    assert contract is not None
    observations = _work_budget_observations("bounded")
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"], **{fingerprint: "d" * 64}
    )
    with pytest.raises(ArbitrationError, match="fingerprint mismatch"):
        arbitrate(_routes(), observations, work_budget_contract=contract)


@pytest.mark.parametrize(
    ("scales", "derivation"),
    (
        ((), "policy"),
        (("bounded", "bounded"), "policy"),
        (("other",), "policy"),
        (("bounded",), ""),
    ),
)
def test_work_budget_declaration_rejects_invalid_scales_or_rationale(
    scales: tuple[str, ...], derivation: str
) -> None:
    with pytest.raises(ValueError, match="work-budget"):
        WorkBudgetContract(
            scales=cast(tuple[ExecutionScale, ...], scales),
            derivation=derivation,
        )


@pytest.mark.parametrize("mutation", ("unpublished", "disclaimed"))
def test_quality_band_refuses_an_observable_no_lane_certifies(mutation: str) -> None:
    observations = {
        lane: dataclasses.replace(
            observation,
            values=(
                {"initial:objective_sum_squares": np.asarray(1.0, dtype=np.float64)}
                if mutation == "unpublished"
                else observation.values
            ),
            applicability=(
                {} if mutation == "unpublished" else {"final:objective": False}
            ),
        )
        for lane, observation in _native_default_observations().items()
    }

    with pytest.raises(
        ArbitrationError, match="quality-band observable is not applicable"
    ):
        arbitrate(_routes(), observations, quality_band=_ARCHIVED_QUALITY_BAND)


@pytest.mark.parametrize(
    ("status", "expected_message"),
    (
        (np.asarray([0], dtype=np.int64), "quality-band observable must be FP64"),
        (
            np.asarray([], dtype=np.float64),
            "non-finite quality-band observable",
        ),
    ),
)
def test_quality_band_refuses_an_unmeasurable_observable(
    status: np.ndarray, expected_message: str
) -> None:
    status_routes = tuple(
        ComparisonRoute(
            phase="final",
            observable="outer_solver_status",
            lane_pair=lane_pair,
            applicable=True,
            comparator="exact",
            tolerance_bucket="native_workflow",
        )
        for lane_pair in (
            "native-cpu:jax-cpu",
            "native-cpu:jax-gpu",
            "jax-cpu:jax-gpu",
        )
    )
    observations = {
        lane: dataclasses.replace(
            observation,
            values={
                "initial:objective_sum_squares": np.asarray(1.0, dtype=np.float64),
                "final:outer_solver_status": status,
            },
            applicability={},
        )
        for lane, observation in _native_default_observations().items()
    }

    with pytest.raises(ArbitrationError, match=expected_message):
        arbitrate(
            (*_routes(), *status_routes),
            observations,
            quality_band=QualityBand(
                observable="final:outer_solver_status",
                max_value=1.0,
                derivation="test fixture",
            ),
        )


def test_quality_band_declaration_is_opt_in_per_case() -> None:
    # Opt-in stays per case: only the official mirrors whose band is derived from
    # upstream's own end-point scatter (official_quality_bands, rule v2 of
    # 2026-09-20) declare one.
    assert {
        case_id
        for case_id in implemented_case_ids()
        if get_case(case_id).native_default_quality_band is not None
    } == set(OFFICIAL_BAND_CASE_IDS)
    # Re-derived here from the tracked upstream record, never by calling the
    # function that built the entry: comparing an entry with a second call of
    # its own factory cannot fail. S is upstream's end values under one-ulp
    # start perturbations and the pre-registered ceiling is
    # max(S) * (1 + (max(S) - min(S)) / min(S)).
    for case_id in OFFICIAL_BAND_CASE_IDS:
        official = get_case(case_id).native_default_quality_band
        sensitivity = load_official_sensitivity(case_id)
        samples = sensitivity.end_values
        low, high = min(samples), max(samples)

        assert official is not None
        assert official.observable == sensitivity.lane_observable
        assert official.max_value == high * (1.0 + (high - low) / low)


@pytest.mark.parametrize(
    ("observable", "max_value", "derivation"),
    (
        ("objective", 1.0e-07, "measured"),
        ("final:objective", float("inf"), "measured"),
        ("final:objective", 1.0e-07, ""),
    ),
)
def test_quality_band_declaration_rejects_an_unusable_floor(
    observable: str, max_value: float, derivation: str
) -> None:
    with pytest.raises(ValueError, match="quality band"):
        QualityBand(observable=observable, max_value=max_value, derivation=derivation)


# ------------------------------------------- quality-band published-run audit
#
# audit_published_run's quality-band branches are only reachable from a
# published native_default run, which costs hours to produce for real. These
# tests synthesize one for an official banded case -- real input bundle, real
# lane receipts, real publication marker -- and drive the production auditor
# over it. The per-lane end points are the archived fixture values above: they
# sit below this case's band and fork beyond its equality bucket.
_BAND_CASE_ID = "native-stage-two-optimization-minimal"


def _band_case_receipt_values(
    routes: tuple[ComparisonRoute, ...], lane: str, *, fork: bool
) -> dict[str, np.ndarray]:
    """Publish exactly the observables the case's route matrix names.

    Every observable is lane-identical except ``final:objective``, which under
    ``fork`` carries the archived per-lane endpoint so the equality bucket
    genuinely fails while the declared quality band still holds.
    """
    values: dict[str, np.ndarray] = {}
    for route in routes:
        key = f"{route.phase}:{route.observable}"
        if key in values:
            continue
        if key == "final:objective":
            values[key] = np.asarray(
                [
                    _ARCHIVED_FINAL_OBJECTIVE[lane]
                    if fork
                    else _ARCHIVED_FINAL_OBJECTIVE["native-cpu"]
                ],
                dtype=np.float64,
            )
        elif key == "final:outer_solver_status":
            values[key] = np.asarray(0, dtype=np.int64)
        elif route.observable.endswith(("success", "stationary", "satisfied")):
            values[key] = np.asarray(True, dtype=np.bool_)
        else:
            values[key] = np.asarray([1.0], dtype=np.float64)
    return values


def _publish_quality_band_run(
    artifact_root: Path,
    *,
    verdict: str = "quality-band",
    fork: bool = True,
    tamper_band: bool = False,
    cost_tier_override: str | None = None,
    terminal_contract_override: str | None = None,
    receipt_mutation: str | None = None,
    used_legacy_manifest_adapter: bool = False,
) -> tuple[Path, dict[str, object]]:
    """Publish one synthetic native_default quality-band run for the auditor."""
    repo_root = Path(__file__).resolve().parents[2]
    contract_pair = load_runtime_contract_pair(
        repo_root / "examples/jax/manifest.json",
        repo_root / "examples/jax/parity_manifest.json",
        repo_root=repo_root,
    )
    relationship = next(
        item
        for item in contract_pair.parity.all_relationships
        if item.case_id == _BAND_CASE_ID
    ).resolve_scale("native_default")
    outer_optimizer_policy = next(
        example.outer_optimizer_policy
        for example in contract_pair.examples
        if example.id == relationship.jax_example_id
    )
    assert outer_optimizer_policy is not None
    repository_state = collect_repository_state(repo_root)
    explicit_sources = collect_explicit_sources(
        repo_root, ("examples/jax/parity/publication.py",)
    )
    lanes = ("native-cpu", "jax-cpu", "jax-gpu")
    paths = begin_run(artifact_root, "20260816T000000Z-0badc0de")
    bundle = create_input_bundle(
        paths.partial / _BAND_CASE_ID / "inputs",
        case_id=_BAND_CASE_ID,
        random_seed=1,
        arrays={"seed": np.zeros(1, dtype=np.float64)},
        configuration={"outer_maxiter": _ARCHIVED_MATCHED_BUDGET},
        scale="native_default",
    )
    observations: dict[str, LaneObservation] = {}
    for lane, template in _observations().items():
        provenance = template.provenance
        assert provenance is not None
        observation = dataclasses.replace(
            template,
            scale="native_default",
            input_fingerprint=bundle.input_fingerprint,
            configuration_fingerprint=bundle.configuration_fingerprint,
            driver=(
                "scipy_lbfgsb"
                if lane == "native-cpu"
                else outer_optimizer_policy.expected_driver
            ),
            normalized_status="budget_exhausted",
            raw_status="stopping_reason=iteration-limit",
            success=False,
            nit=_ARCHIVED_MATCHED_BUDGET,
            completed_workflow_stages=relationship.workflow_stages,
            provenance=dataclasses.replace(
                provenance,
                repository_commit=repository_state.repository_commit,
                repository_dirty=repository_state.repository_dirty,
                tracked_diff_sha256=repository_state.tracked_diff_sha256,
                untracked_files=repository_state.untracked_files,
                executed_sources=explicit_sources,
                authoritative=False,
            ),
            values=_band_case_receipt_values(
                relationship.comparison_routes, lane, fork=fork
            ),
            applicability={},
        )
        published_observation = observation
        if lane == "jax-gpu":
            if receipt_mutation == "lane":
                published_observation = dataclasses.replace(
                    observation, lane="native-cpu"
                )
            elif receipt_mutation == "scale":
                published_observation = dataclasses.replace(
                    observation, scale="bounded"
                )
            elif receipt_mutation == "workflow":
                published_observation = dataclasses.replace(
                    observation, completed_workflow_stages=("unrelated",)
                )
            elif receipt_mutation == "outcome":
                published_observation = dataclasses.replace(
                    observation, normalized_status="failed", success=False
                )
            elif receipt_mutation == "objective":
                published_observation = dataclasses.replace(
                    observation,
                    values={
                        **observation.values,
                        "final:objective": np.asarray([1.0], dtype=np.float64),
                    },
                )
        write_lane_observation(
            paths.partial / _BAND_CASE_ID / lane, published_observation
        )
        observations[lane] = observation
    arbitration = arbitrate(
        relationship.comparison_routes,
        observations,
        required_lanes=frozenset(lanes),
        expected_workflow_stages=relationship.workflow_stages,
        case_id=_BAND_CASE_ID,
        example_id=relationship.jax_example_id,
        outer_optimizer_policy=outer_optimizer_policy,
        quality_band=get_case(_BAND_CASE_ID).native_default_quality_band,
    )
    quality_band_payload = [
        {
            "lane": band_result.lane,
            "observable": band_result.observable,
            "max_value": band_result.max_value,
            "observed_value": (0.0 if tamper_band else band_result.observed_value),
            "passed": band_result.passed,
        }
        for band_result in arbitration.quality_band_results
    ]
    case_record: dict[str, object] = {
        "case_id": _BAND_CASE_ID,
        "jax_example_id": relationship.jax_example_id,
        "native_source": relationship.native_source,
        "classification": relationship.classification,
        "classification_reason": relationship.classification_reason,
        "scale_tier": "native_default",
        "oracle_kind": relationship.oracle_kind,
        "cost_tier": cost_tier_override or relationship.cost_tier,
        "omitted_scientific_stages": list(relationship.omitted_scientific_stages),
        "excluded_teaching_stages": list(relationship.excluded_teaching_stages),
        "authoritative": False,
        "repository_changed_during_run": False,
        "input_fingerprint": bundle.input_fingerprint,
        "configuration_fingerprint": bundle.configuration_fingerprint,
        "completed_workflow_stages": list(relationship.workflow_stages),
        "verdict": verdict,
        "comparisons": [
            {
                "phase": comparison.phase,
                "observable": comparison.observable,
                "lane_pair": comparison.lane_pair,
                "passed": comparison.passed,
                "tolerance_bucket": comparison.tolerance_bucket,
                "diagnostic": comparison.diagnostic,
            }
            for comparison in arbitration.comparisons
        ],
        "quality_band": quality_band_payload,
        "executions": [
            {
                "lane": lane,
                "command": ["python", "-S", "-m", "examples.jax.parity.child"],
                "stdout": "",
                "stderr": "",
                "returncode": 0,
                "elapsed_seconds": 1.0,
                "parent_peak_rss_bytes": 1024,
                "result_directory": f"{_BAND_CASE_ID}/{lane}",
            }
            for lane in lanes
        ],
    }
    if terminal_contract_override is not None:
        case_record["terminal_contract"] = terminal_contract_override
    summary: dict[str, object] = {
        "schema_version": 2,
        "manifest_schema_version": contract_pair.version_pair[0],
        "parity_manifest_schema_version": contract_pair.version_pair[1],
        "used_legacy_manifest_adapter": used_legacy_manifest_adapter,
        "run_id": paths.run_id,
        "lanes": list(lanes),
        "scale": "native_default",
        "smoke": False,
        "authoritative": False,
        "repository_commit": repository_state.repository_commit,
        "repository_dirty": repository_state.repository_dirty,
        "repository_changed_during_run": False,
        "tracked_diff_sha256": repository_state.tracked_diff_sha256,
        "untracked_files": list(repository_state.untracked_files),
        "explicit_sources": [dataclasses.asdict(source) for source in explicit_sources],
        "verdict": verdict,
        "cases": [case_record],
    }
    write_bytes_exclusive(paths.partial, "summary.json", canonical_json_bytes(summary))
    return publish_run(paths), summary


def test_audit_accepts_a_published_quality_band_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    published, _summary = _publish_quality_band_run(tmp_path)

    result = audit_published_run(published, repo_root=repo_root)

    assert result.verdict == "quality-band"
    assert result.case_count == 1
    assert result.lane_receipt_count == 3
    assert result.authoritative is False
    assert result.comparison_count > 0


def test_audit_keeps_failed_equality_comparisons_noncertifying(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    published, summary = _publish_quality_band_run(tmp_path)
    cases = summary["cases"]
    assert isinstance(cases, list)
    comparisons = cases[0]["comparisons"]
    assert isinstance(comparisons, list)
    failed = [
        comparison
        for comparison in comparisons
        if not comparison["passed"]
        and comparison["phase"] == "final"
        and comparison["observable"] == "objective"
    ]

    result = audit_published_run(published, repo_root=repo_root)

    assert len(failed) == 3
    assert all(
        comparison["diagnostic"].startswith("informational (quality-band")
        for comparison in comparisons
    )
    assert result.verdict == "quality-band"


def test_audit_rejects_a_tampered_cost_tier(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    published, _summary = _publish_quality_band_run(
        tmp_path, cost_tier_override="smoke"
    )
    with pytest.raises(ValueError, match="summary cost tier does not match contract"):
        audit_published_run(published, repo_root=repo_root)


def test_audit_rejects_a_forged_work_budget_marker(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    published, _summary = _publish_quality_band_run(
        tmp_path, terminal_contract_override="work-budget"
    )
    with pytest.raises(ValueError, match="stored work-budget admission differs"):
        audit_published_run(published, repo_root=repo_root)


def test_audit_rejects_a_tampered_quality_band_payload(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    published, _summary = _publish_quality_band_run(tmp_path, tamper_band=True)

    with pytest.raises(ValueError, match="stored quality band differs"):
        audit_published_run(published, repo_root=repo_root)


def test_audit_rejects_a_summary_claiming_the_legacy_manifest_adapter(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    published, _summary = _publish_quality_band_run(
        tmp_path, used_legacy_manifest_adapter=True
    )

    with pytest.raises(ValueError, match="legacy manifest adapter mismatch"):
        audit_published_run(published, repo_root=repo_root)


def test_audit_refuses_equivalence_labelling_for_a_banded_case(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    published, _summary = _publish_quality_band_run(
        tmp_path, verdict="pass", fork=False
    )

    with pytest.raises(ValueError, match="recomputed comparison verdict is not pass"):
        audit_published_run(published, repo_root=repo_root)


def test_child_command_is_exact_and_bounded(tmp_path: Path) -> None:
    command = build_child_command(
        python_executable="/venv/bin/python",
        case_id="quadratic",
        lane="jax-gpu",
        input_bundle_path=tmp_path / "input_bundle.json",
        result_directory=tmp_path / "result",
        scale="bounded",
    )
    assert command == (
        "/venv/bin/python",
        "-S",
        "-m",
        "examples.jax.parity.child",
        "--case",
        "quadratic",
        "--lane",
        "jax-gpu",
        "--input-bundle",
        str(tmp_path / "input_bundle.json"),
        "--result-directory",
        str(tmp_path / "result"),
        "--scale",
        "bounded",
    )


def test_parent_observes_child_peak_rss(tmp_path: Path) -> None:
    completed: ChildProcessResult = execute_child_process(
        (
            sys.executable,
            "-c",
            "payload = bytearray(32_000_000); print(len(payload))",
        ),
        tmp_path,
        {},
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "32000000"
    assert completed.parent_peak_rss_bytes is not None
    assert completed.parent_peak_rss_bytes >= 32_000_000


def test_runner_executes_isolated_lanes_and_loads_hash_bound_receipts(
    tmp_path: Path,
) -> None:
    observations = _observations()
    seen: list[tuple[tuple[str, ...], str]] = []

    def executor(
        command: tuple[str, ...], cwd: Path, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        lane = command[command.index("--lane") + 1]
        result_directory = Path(command[command.index("--result-directory") + 1])
        write_lane_observation(result_directory, observations[lane])
        seen.append((command, environment["SIMSOPT_BACKEND_MODE"]))
        return subprocess.CompletedProcess(command, 0, f"{lane} stdout", "")

    executions, loaded = execute_case_lanes(
        case_id="quadratic",
        lanes=("native-cpu", "jax-cpu", "jax-gpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=Path(__file__).resolve().parents[2],
        base_environment={"PRESERVED": "yes"},
        python_executable=sys.executable,
        scale="bounded",
        executor=executor,
    )

    assert tuple(item.lane for item in executions) == (
        "native-cpu",
        "jax-cpu",
        "jax-gpu",
    )
    assert tuple(item.returncode for item in executions) == (0, 0, 0)
    assert loaded == observations
    assert [backend for _, backend in seen] == [
        "native_cpu",
        "jax_cpu_parity",
        "jax_gpu_parity",
    ]


@pytest.mark.parametrize("failure", ["nonzero", "missing_receipt", "wrong_lane"])
def test_runner_fails_closed_on_child_failure(tmp_path: Path, failure: str) -> None:
    observation = _observations()["native-cpu"]

    def executor(
        command: tuple[str, ...], cwd: Path, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        del cwd, environment
        result_directory = Path(command[command.index("--result-directory") + 1])
        if failure != "missing_receipt":
            receipt = (
                dataclasses.replace(observation, lane="jax-cpu")
                if failure == "wrong_lane"
                else observation
            )
            write_lane_observation(result_directory, receipt)
        return subprocess.CompletedProcess(
            command,
            7 if failure == "nonzero" else 0,
            "child stdout",
            "child stderr",
        )

    with pytest.raises(RunnerError, match="native-cpu"):
        execute_case_lanes(
            case_id="quadratic",
            lanes=("native-cpu",),
            input_bundle_path=tmp_path / "input_bundle.json",
            run_directory=tmp_path / "run.partial",
            repo_root=Path(__file__).resolve().parents[2],
            base_environment={},
            python_executable=sys.executable,
            scale="bounded",
            executor=executor,
        )


def test_run_parity_cli_refuses_to_build_input_bundles_without_float64(
    tmp_path: Path,
) -> None:
    """``create_input`` runs in the runner process, so x64 must hold there.

    ``examples/jax/_lane_environment.py`` pins ``JAX_ENABLE_X64=1`` for the lane
    subprocesses only. A runner launched without it built the
    ``native-permanent-magnet-qa`` bundle from a float32-quantized TF-coil
    pre-optimization, and all three lanes then agreed on the wrong problem
    (``.artifacts/official-mirror-closure-20260919/investigations/pm-qa-coils/REPORT.md``).
    The runner now refuses before it touches the artifact root.
    """

    repo_root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "JAX_ENABLE_X64": "0"}
    environment.pop("SIMSOPT_PRECISION", None)
    completed = subprocess.run(
        (
            sys.executable,
            str(repo_root / "examples" / "jax" / "run_parity.py"),
            "--case",
            "native-just-a-quadratic",
            "--lanes",
            "native-cpu,jax-cpu",
            "--scale",
            "bounded",
            "--artifact-root",
            str(tmp_path),
        ),
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires JAX float64 in the runner process" in completed.stderr
    assert "JAX_ENABLE_X64=1" in completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_run_parity_cli_publishes_complete_wave_a_cpu_artifact(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    initial_repository_state = collect_repository_state(repo_root)
    completed = subprocess.run(
        (
            sys.executable,
            str(repo_root / "examples" / "jax" / "run_parity.py"),
            "--case",
            "native-just-a-quadratic",
            "--lanes",
            "native-cpu,jax-cpu",
            "--scale",
            "bounded",
            "--artifact-root",
            str(tmp_path),
        ),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    published = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(published) == 1
    assert not published[0].name.endswith(".partial")
    summary = json.loads((published[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "pass"
    assert summary["manifest_schema_version"] == 3
    assert summary["parity_manifest_schema_version"] == 2
    assert summary["used_legacy_manifest_adapter"] is False
    assert summary["scale"] == "bounded"
    assert summary["lanes"] == ["native-cpu", "jax-cpu"]
    assert isinstance(summary["authoritative"], bool)
    if initial_repository_state.repository_dirty:
        assert summary["authoritative"] is False
    # A clean checkout with a verified native build may establish provenance.
    # Independently replay that claim before exercising receipt tampering below.
    audited = audit_published_run(
        published[0],
        repo_root=repo_root,
        require_authoritative=summary["authoritative"],
    )
    assert audited.verdict == "pass"
    assert audited.authoritative is summary["authoritative"]
    assert len(summary["repository_commit"]) == 40
    assert summary["repository_dirty"] is initial_repository_state.repository_dirty
    assert len(summary["tracked_diff_sha256"]) == 64
    assert summary["untracked_files"] == sorted(summary["untracked_files"])
    assert len(summary["cases"]) == 1
    assert summary["cases"][0]["jax_example_id"] == "native-just-a-quadratic"
    assert summary["cases"][0]["native_source"] == "1_Simple/just_a_quadratic.py"
    assert summary["cases"][0]["classification"] == "full"
    assert summary["cases"][0]["scale_tier"] == "bounded"
    assert summary["cases"][0]["oracle_kind"] == "native_source_owned_simsopt"
    assert len(summary["cases"][0]["comparisons"]) == 10
    executions = {
        execution["lane"]: execution for execution in summary["cases"][0]["executions"]
    }
    for lane in ("native-cpu", "jax-cpu"):
        assert executions[lane]["parent_peak_rss_bytes"] > 0
        receipt = json.loads(
            (
                published[0] / "native-just-a-quadratic" / lane / "lane_result.json"
            ).read_text(encoding="utf-8")
        )
        provenance = receipt["provenance"]
        assert provenance["repository_commit"] == summary["repository_commit"]
        assert (
            provenance["repository_dirty"] is initial_repository_state.repository_dirty
        )
        assert provenance["executed_sources"]
        assert provenance["python_version"]
        assert provenance["lane_environment_policy"]["SIMSOPT_BACKEND_MODE"]
        assert provenance["host_peak_rss_method"].startswith("child getrusage")

    for mutation in ("input-json", "input-sidecar"):
        copied_run = tmp_path / mutation / published[0].name
        shutil.copytree(published[0], copied_run)
        input_root = copied_run / "native-just-a-quadratic" / "inputs"
        bundle_path = input_root / "input_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if mutation == "input-json":
            bundle["configuration"]["rtol"] = 1.0e-4
            bundle_path.write_bytes(canonical_json_bytes(bundle))
        else:
            targets = bundle["arrays"]["targets"]
            (input_root / targets["path"]).unlink()
            write_array(
                input_root,
                targets["path"],
                np.asarray([9.0, 8.0, 7.0], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="fingerprint|SHA-256"):
            audit_published_run(copied_run, repo_root=repo_root)

    jax_receipt_path = (
        published[0] / "native-just-a-quadratic" / "jax-cpu" / "lane_result.json"
    )
    jax_receipt = json.loads(jax_receipt_path.read_text(encoding="utf-8"))
    changed_value_path = (
        jax_receipt_path.parent
        / jax_receipt["values"]["initial:objective_sum_squares"]["path"]
    )
    changed_value_path.unlink()
    changed_reference = write_array(
        jax_receipt_path.parent,
        jax_receipt["values"]["initial:objective_sum_squares"]["path"],
        np.asarray(123.0, dtype=np.float64),
    )
    jax_receipt["values"]["initial:objective_sum_squares"] = dataclasses.asdict(
        changed_reference
    )
    jax_receipt_path.write_bytes(canonical_json_bytes(jax_receipt))

    with pytest.raises(ValueError, match="recomputed comparison verdict"):
        audit_published_run(published[0], repo_root=repo_root)


def test_run_parity_cli_resolves_relative_artifact_root(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        (
            sys.executable,
            str(repo_root / "examples" / "jax" / "run_parity.py"),
            "--case",
            "native-just-a-quadratic",
            "--lanes",
            "native-cpu,jax-cpu",
            "--scale",
            "bounded",
            "--artifact-root",
            "parity-artifacts",
        ),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    published = tuple((tmp_path / "parity-artifacts").glob("*/summary.json"))
    assert len(published) == 1


def test_run_parity_cli_rejects_scale_unsupported_by_relationship(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    contract_pair = load_runtime_contract_pair(
        repo_root / "examples/jax/manifest.json",
        repo_root / "examples/jax/parity_manifest.json",
        repo_root=repo_root,
    )
    bounded_only_relationships = tuple(
        dataclasses.replace(relationship, scale_contracts=())
        for relationship in contract_pair.parity.relationships
        if relationship.case_id == "native-just-a-quadratic"
    )
    assert len(bounded_only_relationships) == 1
    unsupported_relationship = bounded_only_relationships[0]
    assert unsupported_relationship.supported_scales == ("bounded",)
    relationships = tuple(
        unsupported_relationship
        if relationship.case_id == unsupported_relationship.case_id
        else relationship
        for relationship in contract_pair.parity.relationships
    )
    monkeypatch.setattr(
        parity_cli,
        "load_runtime_contract_pair",
        lambda *_args, **_kwargs: dataclasses.replace(
            contract_pair,
            parity=dataclasses.replace(
                contract_pair.parity, relationships=relationships
            ),
        ),
    )
    result = parity_cli.main(
        [
            "--case",
            "native-just-a-quadratic",
            "--lanes",
            "native-cpu,jax-cpu",
            "--scale",
            "native_default",
            "--artifact-root",
            str(tmp_path),
        ]
    )

    assert result == 1
    stderr = capsys.readouterr().err
    assert "declares scale_tier 'bounded'" in stderr
    assert "requested scale 'native_default' is unsupported" in stderr
    published = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and not path.name.endswith(".partial")
    ]
    assert published == []


@pytest.mark.parametrize(
    "mutation", ("lane", "scale", "workflow", "outcome", "objective")
)
def test_canonical_audit_rejects_lane_evidence_that_disagrees_with_quality_summary(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    published, _summary = _publish_quality_band_run(tmp_path, receipt_mutation=mutation)
    with pytest.raises(ValueError):
        audit_published_run(published, repo_root=repo_root)


@pytest.mark.parametrize(
    "driver",
    ("jax_optax_lbfgs", "optimistix_bfgs", "host_callback_bfgs", "scipy_lbfgsb"),
)
def test_a_jax_lane_without_a_policy_rejects_every_forbidden_driver_family(
    driver: str,
) -> None:
    # Optax is gone from the code, but a receipt that names an Optax driver
    # (a recorded one, or a forged one) must still fail arbitration.
    observations = {
        lane: dataclasses.replace(observation, driver=driver)
        if lane.startswith("jax-")
        else observation
        for lane, observation in _observations().items()
    }

    with pytest.raises(ArbitrationError, match="forbidden parity driver"):
        arbitrate(_routes(), observations)


@pytest.mark.parametrize(
    "mutation",
    (
        "none",
        "absent",
        "wrong_case",
        "wrong_example",
        "borrowed_policy",
        "forged_policy",
        "wrong_driver",
        "optax",
        "optimistix",
        "host_callback",
    ),
)
def test_approved_scipy_policy_is_bound_to_real_case_example_and_driver(
    mutation: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pair = load_runtime_contract_pair(
        repo_root / "examples/jax/manifest.json",
        repo_root / "examples/jax/parity_manifest.json",
        repo_root=repo_root,
    )
    example = next(example for example in pair.examples if example.id == _BAND_CASE_ID)
    policy = example.outer_optimizer_policy
    assert policy is not None
    case_id = _BAND_CASE_ID
    example_id = example.id
    driver = policy.expected_driver
    if mutation == "absent":
        policy = None
    elif mutation == "wrong_case":
        case_id = "native-just-a-quadratic"
    elif mutation == "wrong_example":
        example_id = "native-just-a-quadratic"
    elif mutation == "borrowed_policy":
        policy = next(
            example.outer_optimizer_policy
            for example in pair.examples
            if example.id == "native-qfm"
        )
        example_id = "native-qfm"
    elif mutation == "forged_policy":
        policy = dataclasses.replace(policy, expected_driver="scipy_arbitrary")
        driver = "scipy_arbitrary"
    elif mutation == "wrong_driver":
        driver = "simsopt_lm_qr"
    elif mutation in {"optax", "optimistix", "host_callback"}:
        driver = mutation + "_scipy"
    observations = {
        lane: dataclasses.replace(observation, driver=driver)
        if lane.startswith("jax-")
        else observation
        for lane, observation in _observations().items()
    }
    if mutation == "none":
        assert (
            arbitrate(
                _routes(),
                observations,
                case_id=case_id,
                example_id=example_id,
                outer_optimizer_policy=policy,
            ).verdict
            == "pass"
        )
    else:
        with pytest.raises(
            ArbitrationError, match="(outer optimizer policy|forbidden parity driver)"
        ):
            arbitrate(
                _routes(),
                observations,
                case_id=case_id,
                example_id=example_id,
                outer_optimizer_policy=policy,
            )


@pytest.mark.parametrize(
    ("example_id", "expected_policy_id"),
    (
        ("native-stage-two-optimization", "scipy-lbfgsb-over-jax-standard-stage-two"),
        (
            "native-stage-two-optimization-planar-coils",
            "scipy-lbfgsb-over-jax-planar-stage-two",
        ),
        (
            "native-stage-two-optimization-stochastic",
            "scipy-lbfgsb-over-jax-stochastic-stage-two",
        ),
        ("native-coil-forces", "scipy-lbfgsb-over-jax-coil-forces"),
    ),
)
def test_shipped_scipy_policy_rejects_wrong_owner_and_driver(
    example_id: str, expected_policy_id: str
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pair = load_runtime_contract_pair(
        repo_root / "examples/jax/manifest.json",
        repo_root / "examples/jax/parity_manifest.json",
        repo_root=repo_root,
    )
    example = next(item for item in pair.examples if item.id == example_id)
    policy = example.outer_optimizer_policy
    assert policy is not None
    assert policy.policy_id == expected_policy_id
    assert policy.expected_driver == "scipy_lbfgsb"
    manifest_example = next(
        item
        for item in json.loads(
            (repo_root / "examples/jax/manifest.json").read_text(encoding="utf-8")
        )["jax_examples"]
        if item["id"] == example_id
    )
    assert any(
        "SciPy L-BFGS-B" in boundary and "host" in boundary
        for boundary in manifest_example["host_boundaries"]
    )
    assert policy_owns_parity_case(policy, case_id=example_id, example_id=example_id)
    assert not policy_owns_parity_case(
        policy, case_id="native-just-a-quadratic", example_id=example_id
    )
    assert not policy_owns_parity_case(
        policy, case_id=example_id, example_id="native-just-a-quadratic"
    )
    with pytest.raises(OuterOptimizerPolicyError, match="different example"):
        parse_outer_optimizer_policy(
            policy.policy_id,
            example_id=example_id,
            example_path="1_Simple/just_a_quadratic.py",
            ready=True,
        )
    observations = {
        lane: dataclasses.replace(observation, driver="scipy_lbfgsb")
        if lane.startswith("jax-")
        else observation
        for lane, observation in _observations().items()
    }
    assert (
        arbitrate(
            _routes(),
            observations,
            case_id=example_id,
            example_id=example_id,
            outer_optimizer_policy=policy,
        ).verdict
        == "pass"
    )
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"], driver="scipy_lbfgsb_other"
    )
    with pytest.raises(ArbitrationError, match="driver differs"):
        arbitrate(
            _routes(),
            observations,
            case_id=example_id,
            example_id=example_id,
            outer_optimizer_policy=policy,
        )


def test_finitebuild_clearance_has_complete_direct_lane_pair_routes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pair = load_runtime_contract_pair(
        repo_root / "examples/jax/manifest.json",
        repo_root / "examples/jax/parity_manifest.json",
        repo_root=repo_root,
    )
    relationship = next(
        item
        for item in pair.parity.relationships
        if item.case_id == "native-stage-two-optimization-finitebuild"
    )
    expected_buckets = {
        "native-cpu:jax-cpu": "native_workflow",
        "native-cpu:jax-gpu": "native_workflow",
        "jax-cpu:jax-gpu": "gpu_runtime",
    }
    for phase in ("initial", "final"):
        routes = tuple(
            route
            for route in relationship.comparison_routes
            if route.phase == phase and route.observable == "minimum_clearance"
        )
        assert len(routes) == 3
        assert {
            route.lane_pair: route.tolerance_bucket for route in routes
        } == expected_buckets
        assert all(
            route.applicable and route.comparator == "allclose" for route in routes
        )


#: An observable no relationship declares a comparison route for.
_UNROUTED_OBSERVABLE = "final:unrouted_observable"


def _inject_completed_lane_receipts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rejected_lane: str | None = "jax-cpu",
    corrupt_source: bool = False,
    integrity_break: str | None = None,
) -> None:
    """Serve hand-built lane receipts in place of real child executions.

    ``rejected_lane`` makes one lane report an honest failure (a lane outcome).
    ``integrity_break`` instead violates a harness/contract invariant, which is
    never a lane's result and must abort the run.
    """

    def execute_case_lanes(
        *,
        case_id: str,
        lanes: tuple[str, ...],
        input_bundle_path: Path,
        run_directory: Path,
        repo_root: Path,
        base_environment: dict[str, str],
        python_executable: str,
        scale: str,
    ):
        del base_environment, python_executable
        bundle, _arrays = read_input_bundle(input_bundle_path.parent)
        repository_state = collect_repository_state(repo_root)
        sources = collect_explicit_sources(repo_root, REQUIRED_PROVENANCE_SOURCE_PATHS)
        if corrupt_source:
            sources = tuple(
                ExecutedSource(source.path, "0" * 64, source.git_blob_id)
                for source in sources
            )
        pair = load_runtime_contract_pair(
            repo_root / "examples/jax/manifest.json",
            repo_root / "examples/jax/parity_manifest.json",
            repo_root=repo_root,
        )
        relationship = next(
            item for item in pair.parity.all_relationships if item.case_id == case_id
        )
        example = next(
            item for item in pair.examples if item.id == relationship.jax_example_id
        )
        # The JAX lanes must carry the driver the case's outer-optimizer policy
        # declares, or the arbiter rejects on the driver gate (arbiter.py:153-160)
        # before it can ever reach the lane-success gate this fixture exercises.
        policy = example.outer_optimizer_policy
        templates = _observations()
        observations: dict[str, LaneObservation] = {}
        executions = []
        for lane in lanes:
            template = templates[lane]
            success = lane != rejected_lane
            provenance = template.provenance
            assert provenance is not None
            driver = (
                policy.expected_driver
                if policy is not None and lane.startswith("jax-")
                else template.driver
            )
            observation = dataclasses.replace(
                template,
                scale=scale,
                driver=driver,
                input_fingerprint=bundle.input_fingerprint,
                configuration_fingerprint=bundle.configuration_fingerprint,
                success=success,
                normalized_status="converged" if success else "failed",
                raw_status="1" if success else "area=False;flux=True",
                completed_workflow_stages=relationship.workflow_stages,
                provenance=dataclasses.replace(
                    provenance,
                    repository_commit=repository_state.repository_commit,
                    repository_dirty=repository_state.repository_dirty,
                    tracked_diff_sha256=repository_state.tracked_diff_sha256,
                    untracked_files=repository_state.untracked_files,
                    executed_sources=sources,
                    authoritative=False,
                ),
            )
            if integrity_break == "backend_mode" and lane == "jax-cpu":
                observation = dataclasses.replace(
                    observation, backend_mode="jax_gpu_parity"
                )
            elif integrity_break == "fingerprint" and lane == "jax-cpu":
                observation = dataclasses.replace(
                    observation, input_fingerprint="z" * 64
                )
            elif integrity_break == "route_matrix":
                # Every lane publishes an observable the manifest declares no
                # route for, so the applicable keys and the route matrix
                # disagree. Breaking one lane only would trip the applicability
                # gate instead, which is a different violation.
                observation = dataclasses.replace(
                    observation,
                    values={
                        **observation.values,
                        _UNROUTED_OBSERVABLE: np.asarray([1.0], dtype=np.float64),
                    },
                )
            result_directory = run_directory / case_id / lane
            result_directory.mkdir(parents=True, exist_ok=True)
            write_lane_observation(result_directory, observation)
            observations[lane] = observation
            executions.append(
                ChildExecution(
                    lane=lane,
                    command=("python", "-m", "examples.jax.parity.child", lane),
                    stdout="",
                    stderr="",
                    returncode=0,
                    elapsed_seconds=0.01,
                    result_directory=result_directory,
                    parent_peak_rss_bytes=1,
                )
            )
        return tuple(executions), observations

    monkeypatch.setattr(parity_cli, "execute_case_lanes", execute_case_lanes)


def test_run_parity_rejected_lane_writes_fail_summary_and_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _inject_completed_lane_receipts(monkeypatch)
    result = parity_cli.main(
        [
            "--case",
            "native-just-a-quadratic",
            "--lanes",
            "native-cpu,jax-cpu",
            "--scale",
            "bounded",
            "--artifact-root",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert result == 1
    published = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and not path.name.endswith(".partial")
    ]
    partials = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.endswith(".partial")
    ]
    assert published == []
    assert len(partials) == 1
    summary = json.loads((partials[0] / "summary.json").read_text(encoding="utf-8"))
    failure = json.loads((partials[0] / "FAILURE.json").read_text(encoding="utf-8"))
    case = summary["cases"][0]
    assert summary["verdict"] == "fail"
    assert case["verdict"] == "fail"
    assert case["comparisons"] == []
    assert "scientific success" in case["arbitration_rejection"]
    assert "jax-cpu" in case["arbitration_rejection"]
    assert len(case["executions"]) == 2
    assert case["completed_workflow_stages"]
    # The stage record is observed per lane on every path, so a disagreement is
    # written down instead of collapsing to "no stages completed".
    assert case["lane_completed_workflow_stages"] == {
        "jax-cpu": case["completed_workflow_stages"],
        "native-cpu": case["completed_workflow_stages"],
    }
    assert failure["status"] == "failed"
    assert "PUBLISHED" not in captured.out


@pytest.mark.parametrize(
    ("integrity_break", "expected_reason"),
    (
        ("backend_mode", "jax-cpu backend_mode must be jax_cpu_parity"),
        ("fingerprint", "input fingerprint"),
        (
            "route_matrix",
            "applicable observables require a complete direct lane-pair matrix",
        ),
    ),
)
def test_run_parity_integrity_violation_aborts_without_a_case_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    integrity_break: str,
    expected_reason: str,
) -> None:
    """Only a lane outcome may be recorded as a scientific case failure.

    A backend/fingerprint/route-matrix violation is a harness or contract bug.
    Recording one as ``verdict: fail`` with an ``arbitration_rejection`` would
    publish it into the index as a parity attempt, so the run must abort and
    write no ``summary.json`` at all. Each parameter injects its own violation
    and asserts the gate that rejected it, so no parameter can pass on some
    other gate the hand-built receipts happen to trip.
    """
    _inject_completed_lane_receipts(
        monkeypatch,
        rejected_lane=None,
        integrity_break=integrity_break,
    )
    result = parity_cli.main(
        [
            "--case",
            "native-just-a-quadratic",
            "--lanes",
            "native-cpu,jax-cpu",
            "--scale",
            "bounded",
            "--artifact-root",
            str(tmp_path),
        ]
    )
    assert result == 1
    published = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and not path.name.endswith(".partial")
    ]
    partials = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.endswith(".partial")
    ]
    assert published == []
    assert len(partials) == 1
    assert not (partials[0] / "summary.json").is_file()
    failure = json.loads((partials[0] / "FAILURE.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert expected_reason in failure["reason"]


def test_run_parity_integrity_failure_after_rejection_does_not_write_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inject_completed_lane_receipts(monkeypatch, corrupt_source=True)
    result = parity_cli.main(
        [
            "--case",
            "native-just-a-quadratic",
            "--lanes",
            "native-cpu,jax-cpu",
            "--scale",
            "bounded",
            "--artifact-root",
            str(tmp_path),
        ]
    )
    assert result == 1
    partials = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.endswith(".partial")
    ]
    published = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and not path.name.endswith(".partial")
    ]
    assert published == []
    assert len(partials) == 1
    assert (partials[0] / "FAILURE.json").is_file()
    assert not (partials[0] / "summary.json").is_file()
