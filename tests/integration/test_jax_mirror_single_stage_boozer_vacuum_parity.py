"""Parity contract for the VMEC-free Boozer single-stage mirror."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.arbiter import SHIPPED_SINGLE_STAGE_SCIPY_DRIVER_ID
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_boozerqa import (
    variant_lane_observation,
    variant_scale_configuration,
)
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import InputBundle, load_input_bundle
from examples.jax.parity.measurement import MeasurementExecution
from simsopt.single_stage_boozer_vacuum import JAX_OPTAX_DRIVER_ID, NATIVE_ITERATIONS
from simsopt_contracts.optimization_endpoint import certify_optimization_endpoint
from simsopt_jax_adapters.geo.single_stage_boozer_vacuum_problem import (
    BOUNDED_SCALE,
    NATIVE_SCALE,
    NCSX_INITIAL_IOTA,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "native-single-stage-boozer-vacuum-optimization"


def test_single_stage_boozer_vacuum_declares_approved_case_specific_driver_policy() -> (
    None
):
    case = get_case(CASE_ID)
    runtime = load_runtime_contract_pair(
        REPO_ROOT / "examples" / "jax" / "manifest.json",
        REPO_ROOT / "examples" / "jax" / "parity_manifest.json",
        repo_root=REPO_ROOT,
    )
    relationship = next(
        item
        for item in runtime.parity.all_relationships
        if item.jax_example_id == CASE_ID
    )

    assert relationship in runtime.parity.experimental_relationships
    assert case.case_id == CASE_ID
    assert relationship.case_id == CASE_ID
    assert relationship.classification == "full"
    assert relationship.scale_tier == "native_default"
    assert relationship.blocker is None
    assert not relationship.omitted_scientific_stages
    assert relationship.workflow_stages
    assert relationship.comparison_routes
    example = next(example for example in runtime.examples if example.id == CASE_ID)
    assert example.outer_optimizer_policy is not None
    assert example.outer_optimizer_policy.case_id == CASE_ID
    assert (
        example.outer_optimizer_policy.expected_driver
        == SHIPPED_SINGLE_STAGE_SCIPY_DRIVER_ID
    )


def test_single_stage_boozer_vacuum_routes_native_default_without_solving() -> None:
    configuration = variant_scale_configuration("native_default", SPEC)

    assert configuration["outer_maxiter"] == NATIVE_ITERATIONS
    assert configuration["mpol"] == 6
    assert configuration["ntor"] == 6


def test_parity_harness_and_shipped_mirror_size_the_same_problem() -> None:
    """The harness case and shipped example configure the same problem."""
    for scale in ("native_default", "bounded"):
        shipped = NATIVE_SCALE if scale == "native_default" else BOUNDED_SCALE
        configuration = variant_scale_configuration(scale, SPEC)

        assert configuration["mpol"] == shipped.surface_resolution, scale
        assert configuration["ntor"] == shipped.surface_resolution, scale
        assert configuration["non_qs_sdim"] == shipped.non_qs_resolution, scale
        assert configuration["initial_iota"] == NCSX_INITIAL_IOTA, scale
        reduced = {
            "coil_order": configuration["reduced_coil_order"],
            "magnetic_axis_order": configuration["reduced_axis_order"],
            "points_per_period": configuration["reduced_points_per_period"],
        }
        assert (
            dict(shipped.ncsx_options) == reduced
            if scale == "bounded"
            else dict(shipped.ncsx_options) == {}
        ), scale


def test_single_stage_parity_preserves_public_solver_boundaries() -> None:
    source = (
        REPO_ROOT / "examples" / "jax" / "parity" / "cases" / "native_boozerqa.py"
    ).read_text(encoding="utf-8")

    assert '"newton_maxiter":' in source
    assert '"newton_tol":' in source
    assert '"verbose": False' in source
    assert "make_traceable_objective_session" in source
    assert "accepted_incumbent_host_value_and_grad" in source
    assert "evaluate_candidate_from_anchor" in source
    assert source.count("callback=accept_optimizer_trial") == 2
    assert "final_eval_value_and_grad_host=evaluate_optimizer_final" in source
    assert 'gradient_source="candidate"' not in source
    assert "forward_success=evidence.forward_success" in source
    assert "primal_success=evidence.primal_success" in source
    assert "actual_adjoint_success=evidence.actual_adjoint_success" in source
    assert "eligible=evidence.eligible" in source
    assert "scalar_example_driver()" in source
    assert "minimize_lbfgs_host_core" in source
    assert "minimize_bfgs_host_core" in source
    assert "OUTER_GRADIENT_TOLERANCE" in source
    assert "lbfgs_status_is_success" in source
    assert '"host-lbfgsb"' in source
    assert '"host-bfgs"' in source
    assert "final_objective, final_gradient = value_and_grad(final_parameters)" not in (
        source
    )
    assert "objective.x = final_parameters" in source


def test_endpoint_certificate_cannot_mask_failed_scientific_gate() -> None:
    certificate = certify_optimization_endpoint(
        status_convention="scipy-bfgs",
        provider_success=True,
        provider_status=0,
        iterations=1,
        max_iterations=2,
        initial_gradient_inf_norm=1.0,
        final_gradient_inf_norm=0.0,
        parameters_finite=True,
        observables_finite=True,
        inner_success=True,
    )
    bundle = InputBundle(
        schema_version=2,
        case_id=CASE_ID,
        scale="bounded",
        random_seed=1,
        configuration={},
        configuration_fingerprint="configuration",
        arrays={},
        input_fingerprint="input",
    )
    values = {
        "construction:surface_dofs": np.asarray([], dtype=np.float64),
        "construction:coil_dofs": np.asarray([], dtype=np.float64),
        "initial:gradient": np.asarray([1.0], dtype=np.float64),
        "final:gradient": np.asarray([0.0], dtype=np.float64),
        "final:parameters": np.asarray([0.0], dtype=np.float64),
        "initial:objective": np.asarray(1.0, dtype=np.float64),
        "final:objective": np.asarray(2.0, dtype=np.float64),
        "final:inner_solver_success": np.asarray(True, dtype=np.bool_),
        "final:outer_solver_success": np.asarray(True, dtype=np.bool_),
    }

    observation = variant_lane_observation(
        "native-cpu",
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver="test-driver",
        workflow_stages=("synthetic-scientific-gate",),
        solver_counts=(1, 1, 1),
        endpoint_certificate=certificate,
    )

    assert certificate.success is True
    assert observation.success is False
    assert observation.normalized_status == "failed"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.single_stage
@pytest.mark.native_cpu_reference
def test_single_stage_boozer_vacuum_case_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case(CASE_ID)
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = case.execute("native-cpu", bundle, arrays)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.success is False
    assert jax.success is False
    for observation in (native, jax):
        assert observation.nit is not None and observation.nit > 0
        assert observation.nfev is not None and observation.nfev > 0
        assert observation.njev is not None and observation.njev > 0
        assert observation.normalized_status == "budget_exhausted"
        assert "certificate=False" in observation.raw_status
        assert "stopping_reason=iteration-limit" in observation.raw_status
        assert bool(observation.values["final:endpoint_certificate_success"]) is False
        assert int(observation.values["final:outer_solver_status"]) == 1
    assert native.driver == "simsopt_scipy_bfgs_with_boozer_newton"
    assert jax.driver == SHIPPED_SINGLE_STAGE_SCIPY_DRIVER_ID
    assert (jax.nit, jax.nfev, jax.njev) == (native.nit, native.nfev, native.njev)
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    for observable in (
        "construction:surface_dofs",
        "construction:coil_dofs",
        "initial:parameters",
    ):
        np.testing.assert_array_equal(
            jax.values[observable],
            native.values[observable],
        )
    np.testing.assert_allclose(
        jax.values["initial:objective"],
        native.values["initial:objective"],
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        jax.values["initial:gradient"],
        native.values["initial:gradient"],
        rtol=2.0e-9,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jax.values["final:parameters"],
        native.values["final:parameters"],
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    for observable in (
        "final:objective",
        "final:iota",
        "final:volume",
        "final:non_qs_ratio",
        "final:boozer_residual",
        "final:boozer_residual_rms",
        "final:major_radius_penalty",
        "final:length_penalty",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=2.0e-8,
            atol=2.0e-12,
        )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.single_stage
def test_single_stage_bounded_optax_measurement_executes_and_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case(CASE_ID)
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    trajectory_path = tmp_path / "optax-trajectory.jsonl"
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_fast")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")

    assert case.measurement_execute is not None
    plain = case.measurement_execute(
        "jax-cpu",
        bundle,
        arrays,
        MeasurementExecution(optimizer_backend="optax-lbfgs"),
    )
    observation = case.measurement_execute(
        "jax-cpu",
        bundle,
        arrays,
        MeasurementExecution(
            trajectory_path=trajectory_path,
            optimizer_backend="optax-lbfgs",
        ),
    )

    records = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    assert records
    assert len(records) == observation.nit
    assert [record["iteration"] for record in records] == list(
        range(1, len(records) + 1)
    )
    assert observation.driver == JAX_OPTAX_DRIVER_ID
    assert observation.backend_mode == "jax_cpu_fast"
    assert observation.precision == "fp64"
    np.testing.assert_array_equal(
        observation.values["final:parameters"],
        plain.values["final:parameters"],
    )
    np.testing.assert_array_equal(
        observation.values["final:objective"],
        plain.values["final:objective"],
    )
    assert observation.nit == plain.nit
    assert observation.nfev == plain.nfev
    assert observation.njev == plain.njev
    assert bool(observation.values["final:inner_solver_success"])
    assert all(
        np.all(np.isfinite(value))
        for key, value in observation.values.items()
        if key.startswith("final:") and value.dtype.kind in {"f", "c"}
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.single_stage
@pytest.mark.native_cpu_reference
def test_single_stage_bounded_native_recording_is_sequence_neutral(
    tmp_path: Path,
) -> None:
    case = get_case(CASE_ID)
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    trajectory_path = tmp_path / "native-trajectory.jsonl"

    plain = case.execute("native-cpu", bundle, arrays)
    assert case.measurement_execute is not None
    recorded = case.measurement_execute(
        "native-cpu",
        bundle,
        arrays,
        MeasurementExecution(trajectory_path=trajectory_path),
    )

    np.testing.assert_array_equal(
        recorded.values["final:parameters"],
        plain.values["final:parameters"],
    )
    np.testing.assert_array_equal(
        recorded.values["final:objective"],
        plain.values["final:objective"],
    )
    assert recorded.nit == plain.nit
    assert recorded.nfev == plain.nfev
    assert recorded.njev == plain.njev
    records = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    assert len(records) == recorded.nit


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.single_stage
def test_single_stage_bounded_fast_recording_is_sequence_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case(CASE_ID)
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    trajectory_path = tmp_path / "fast-trajectory.jsonl"
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_fast")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")

    assert case.measurement_execute is not None
    plain = case.measurement_execute(
        "jax-cpu",
        bundle,
        arrays,
        MeasurementExecution(
            optimization_timing_path=tmp_path / "fast-plain-timing.json"
        ),
    )
    recorded = case.measurement_execute(
        "jax-cpu",
        bundle,
        arrays,
        MeasurementExecution(trajectory_path=trajectory_path),
    )

    np.testing.assert_array_equal(
        recorded.values["final:parameters"],
        plain.values["final:parameters"],
    )
    np.testing.assert_array_equal(
        recorded.values["final:objective"],
        plain.values["final:objective"],
    )
    assert recorded.nit == plain.nit
    assert recorded.nfev == plain.nfev
    assert recorded.njev == plain.njev
    assert recorded.driver == plain.driver
    records = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    assert len(records) == recorded.nit
