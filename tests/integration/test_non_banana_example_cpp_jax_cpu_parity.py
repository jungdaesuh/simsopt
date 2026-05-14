"""CI-safe tests for the non-banana example CPU C++/JAX parity harness.

Covers:
  * Fixture registry completeness (all 11 plan IDs are present).
  * Phase 0 baseline gates (JAX x64 + no GPU devices required).
  * Phase 1 ``minimal_stage2_flux_length_gap`` value/gradient parity and
    its unsupported-component classification.
  * Phase 2 ``cws_saved_local_flux_nfp{2,3}`` local-flux fixtures.
  * Phase 3/4 full and planar Stage-II partial native-subproblem parity.
  * Support-gate fixture (``position_orientation_flux_support_gate``)
    correctly reports verdict='unsupported' without claiming a pass.

Tests require simsoptpp for the CPU oracle.  Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 conda run -n jax-0.9.2 \\
        python -m pytest tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

pytest.importorskip(
    "simsoptpp",
    reason="Non-banana parity tests require simsoptpp for the CPU oracle.",
)

from benchmarks import non_banana_example_cpp_jax_cpu_parity as harness  # noqa: E402
from benchmarks import non_banana_example_parity_fixtures as fixtures  # noqa: E402
from benchmarks.validation_ladder_contract import (  # noqa: E402
    PARITY_LADDER_TOLERANCES,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_env(*, jax_platforms: str, jax_enable_x64: str) -> dict[str, str]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = jax_platforms
    env["JAX_ENABLE_X64"] = jax_enable_x64
    return env


# ---------------------------------------------------------------------------
# Registry coverage.


EXPECTED_FIXTURE_IDS = {
    "minimal_stage2_flux_length_gap",
    "cws_saved_local_flux_nfp2",
    "cws_saved_local_flux_nfp3",
    "full_stage2_composite",
    "planar_stage2_composite",
    "position_orientation_flux_support_gate",
    "boozer_surface_basic",
    "boozer_qa_wrappers",
    "finite_beta_target_flux",
    "finitebuild_multifilament_support_gate",
    "qfm_surface",
}


def test_fixture_registry_covers_plan_ids():
    actual_ids = set(fixtures.fixture_ids())
    assert actual_ids == EXPECTED_FIXTURE_IDS, (
        f"Unexpected fixture IDs: {actual_ids ^ EXPECTED_FIXTURE_IDS}"
    )


def test_phase0_gates_require_x64_and_cpu():
    metadata = harness.build_run_metadata(git_sha_override=None)
    assert metadata["jax_enable_x64"] is True
    assert metadata["jax_platform"] == "cpu"
    for device in metadata["jax_devices"]:
        assert device["platform"] == "cpu", f"Expected CPU-only devices; saw {device!r}"


def test_harness_forces_cpu_platform_before_jax_backend_init():
    code = """
import json
import os

from benchmarks import non_banana_example_cpp_jax_cpu_parity as harness

metadata = harness.build_run_metadata(git_sha_override="test-suite-fixed-sha")
print(json.dumps({
    "env_jax_platforms": os.environ["JAX_PLATFORMS"],
    "env_jax_enable_x64": os.environ["JAX_ENABLE_X64"],
    "metadata_jax_enable_x64": metadata["jax_enable_x64"],
    "device_platforms": [device["platform"] for device in metadata["jax_devices"]],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_subprocess_env(jax_platforms="cuda,cpu", jax_enable_x64="0"),
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout.strip().splitlines()[-1])
    assert observed["env_jax_platforms"] == "cpu"
    assert observed["env_jax_enable_x64"] == "1"
    assert observed["metadata_jax_enable_x64"] is True
    assert observed["device_platforms"]
    assert set(observed["device_platforms"]) == {"cpu"}


def test_cli_minimal_fixture_uses_current_source_tree(tmp_path):
    output_path = tmp_path / "minimal.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks/non_banana_example_cpp_jax_cpu_parity.py"),
            "--fixtures",
            "minimal_stage2_flux_length_gap",
            "--git-sha",
            "test-suite-fixed-sha",
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(jax_platforms="cpu", jax_enable_x64="1"),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Wrote parity artifact:" in completed.stdout

    payload = json.loads(output_path.read_text())
    entry = _select_fixture(payload, "minimal_stage2_flux_length_gap")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial"


# ---------------------------------------------------------------------------
# Phase 1 — P0 minimal Stage-II fixture parity.


@pytest.fixture(scope="module")
def minimal_fixture_payload():
    payload = harness.run_fixtures(
        ["minimal_stage2_flux_length_gap"],
        git_sha_override="test-suite-fixed-sha",
    )
    return payload


def _select_fixture(payload, fixture_id):
    for entry in payload["fixtures"]:
        if entry["fixture_id"] == fixture_id:
            return entry
    raise KeyError(f"Fixture {fixture_id!r} not present in payload")


def test_minimal_stage2_passes_native_supported_parity(minimal_fixture_payload):
    entry = _select_fixture(minimal_fixture_payload, "minimal_stage2_flux_length_gap")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] in ("pass", "partial"), entry
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"Native-supported comparisons failed: {failing}"


def test_minimal_stage2_lists_unsupported_length_penalty(minimal_fixture_payload):
    entry = _select_fixture(minimal_fixture_payload, "minimal_stage2_flux_length_gap")
    assert (
        "QuadraticPenalty_over_sum_CurveLength_max" in entry["unsupported_components"]
    )
    # Partial verdict is the contractual outcome until exact native support
    # for QuadraticPenalty(sum(CurveLength), 'max') lands.
    assert entry["verdict"] == "partial", entry["verdict"]


def test_minimal_stage2_dof_basis_matches(minimal_fixture_payload):
    entry = _select_fixture(minimal_fixture_payload, "minimal_stage2_flux_length_gap")
    cpu_hash = entry["dof_contract"]["active_dof_hash_cpu"]
    jax_hash = entry["dof_contract"]["active_dof_hash_jax"]
    assert cpu_hash == jax_hash, (
        "CPU and JAX lanes must build from identical active-DOF vectors."
    )
    # The harness must report basis alignment explicitly: positional
    # gradient comparison is only valid when the structural names match.
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True
    cpu_struct = entry["dof_contract"]["active_dof_structural_names_cpu"]
    jax_struct = entry["dof_contract"]["active_dof_structural_names_jax"]
    assert cpu_struct == jax_struct, (cpu_struct[:3], jax_struct[:3])
    # Native curve spec hashes must be non-empty for the JAX lane.
    spec_hashes = entry["native_spec_contract"]["native_curve_spec_hashes"]
    assert spec_hashes, "JAX lane must have non-empty native curve spec hashes"


def test_minimal_stage2_perturbation_diagnostic_is_live(minimal_fixture_payload):
    """The seed=1 Taylor diagnostic must run real CPU/JAX evaluations."""
    entry = _select_fixture(minimal_fixture_payload, "minimal_stage2_flux_length_gap")
    perturbation = entry["perturbation_diagnostics"]
    assert perturbation is not None, "perturbation diagnostic must be present"
    assert perturbation["seed"] == 1
    assert perturbation["direction_hash"] is not None
    samples = perturbation["samples"]
    assert len(samples) >= 5, samples
    # Cross-lane central differences must agree to within a tight tolerance
    # at every recorded eps (the SquaredFlux/SquaredFluxJAX subproblem is
    # the native-supported portion and shares the same DOF basis).
    for sample in samples:
        assert sample["abs_diff"] < 1e-6, sample
    grad_jax_dir = perturbation["directional_derivative_grad_jax"]
    grad_cpu_dir = perturbation["directional_derivative_grad_cpu"]
    assert np.isfinite(grad_jax_dir) and np.isfinite(grad_cpu_dir)
    assert abs(grad_jax_dir - grad_cpu_dir) < 1e-6, (grad_jax_dir, grad_cpu_dir)


# ---------------------------------------------------------------------------
# Phase 2 — P1 CWS saved local-flux fixture parity.


def _cws_artifacts_present(fixture_id: str) -> bool:
    record = fixtures.get_fixture(fixture_id)
    inputs = record.spec.inputs
    case_dir = REPO_ROOT / "examples" / "3_Advanced" / inputs["case_dir"]
    coils_path = case_dir / "coils" / inputs["coils_file"]
    vmec_path = case_dir / inputs["vmec_input"]
    return coils_path.exists() and vmec_path.exists()


_CWS_CURVECWSFOURIER_XFAIL_REASON = (
    "Upstream simsopt.load() cannot currently reconstruct CurveCWSFourier "
    "saved artifacts, so the parity harness reports verdict='unsupported' "
    "with 'CurveCWSFourier' in entry['error']. Flip this xfail (delete the "
    "marker) once the upstream JSON deserializer learns the CurveCWSFourier "
    "schema and the fixture starts returning verdict in ('pass', 'partial')."
)


@pytest.mark.parametrize(
    "fixture_id",
    [
        pytest.param(
            "cws_saved_local_flux_nfp2",
            marks=pytest.mark.xfail(
                strict=True,
                reason=_CWS_CURVECWSFOURIER_XFAIL_REASON,
            ),
        ),
        pytest.param(
            "cws_saved_local_flux_nfp3",
            marks=pytest.mark.xfail(
                strict=True,
                reason=_CWS_CURVECWSFOURIER_XFAIL_REASON,
            ),
        ),
    ],
)
def test_cws_saved_local_flux_parity(fixture_id):
    if not _cws_artifacts_present(fixture_id):
        pytest.skip(f"{fixture_id} requires saved CWS artifacts in examples/3_Advanced")
    payload = harness.run_fixtures([fixture_id])
    entry = _select_fixture(payload, fixture_id)
    # Strict xfail trigger: today the harness reports verdict='unsupported'
    # with 'CurveCWSFourier' in entry['error'] because upstream simsopt.load()
    # cannot reconstruct the saved CurveCWSFourier artifacts. When the
    # upstream deserializer fix lands, this assertion will pass, the rest of
    # the test body will run, and strict-xfail will flag the parameter as
    # XPASS -> FAIL so the marker must be removed.
    assert not (
        entry["verdict"] == "unsupported"
        and entry["error"]
        and "CurveCWSFourier" in entry["error"]
    ), (
        "upstream CurveCWSFourier deserializer appears fixed: "
        f"verdict={entry['verdict']!r}, error={entry['error']!r}. "
        "Remove the xfail marker and re-enable the parity assertions below."
    )
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] in ("pass", "partial"), entry
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"CWS parity comparisons failed: {failing}"


# ---------------------------------------------------------------------------
# Support-gate / unsupported classification paths.


@pytest.mark.parametrize(
    "fixture_id",
    [
        "position_orientation_flux_support_gate",
        "finitebuild_multifilament_support_gate",
        "finite_beta_target_flux",
        "qfm_surface",
    ],
)
def test_deferred_fixtures_report_unsupported(fixture_id):
    payload = harness.run_fixtures([fixture_id])
    entry = _select_fixture(payload, fixture_id)
    assert entry["verdict"] == "unsupported", (
        f"Deferred fixture {fixture_id!r} must report verdict='unsupported'; "
        f"got {entry['verdict']!r}"
    )
    assert entry["error"] is not None
    assert entry["comparisons"]["cpu_cpp_vs_jax_cpu"] == []


# ---------------------------------------------------------------------------
# Phase 6 — P2 Boozer surface fixed-state residual + label fixture parity.


def test_boozer_surface_basic_passes_residual_and_label_parity():
    """Fixed-state Boozer parity must pass for residual + labels.

    The fixture rebuilds the NCSX initial state on both CPU and JAX lanes
    (independent coil trees) and compares the pre-solve Boozer residual
    vector plus Area, Volume, and ToroidalFlux labels. The verdict must
    be 'pass' (no unsupported components) and every recorded comparison
    must individually pass.
    """
    payload = harness.run_fixtures(["boozer_surface_basic"])
    entry = _select_fixture(payload, "boozer_surface_basic")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"Boozer fixed-state comparisons failed: {failing}"

    # Sanity-check that the expected quantities were exercised.
    recorded_quantities = {
        cmp["quantity"] for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    }
    expected_quantities = {
        "surface_gamma",
        "surface_unit_normal",
        "field_B",
        "boozer_residual",
        "area",
        "volume",
        "toroidal_flux",
    }
    assert expected_quantities <= recorded_quantities, (
        f"Missing expected comparisons: {expected_quantities - recorded_quantities}"
    )

    # Label and residual values must agree at the direct_kernel bucket.
    for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]:
        if cmp["quantity"] in {
            "boozer_residual",
            "area",
            "volume",
            "toroidal_flux",
        }:
            assert cmp["tolerance_bucket"] == "direct_kernel", cmp
            expected_tolerance = PARITY_LADDER_TOLERANCES["direct_kernel"]
            assert cmp["tolerance_rtol"] == expected_tolerance["rtol"], cmp
            assert cmp["tolerance_atol"] == expected_tolerance["atol"], cmp


# ---------------------------------------------------------------------------
# Phase 6 — P2 BoozerQA wrappers fixed-solved-state parity.


def test_boozer_qa_wrappers_passes_native_supported_parity():
    """Solved-state QA wrapper parity must pass for Iotas/MajorRadius/NQS.

    The fixture solves the NCSX Boozer surface on the CPU side via
    ``BoozerSurface.solve_residual_equation_exactly_newton``, then
    evaluates the upstream QA wrappers (Iotas, MajorRadius,
    NonQuasiSymmetricRatio) on both lanes. The JAX side recomputes the
    wrappers as pure JAX functions
    (``surface_major_radius_jax_from_dofs``, ``_qs_ratio_pure``) at the
    byte-equal solved surface DOFs and using a fresh ``BiotSavartJAX``
    coil_set_spec.

    The CPU-only length penalty ``sum(CurveLength)`` is listed as
    unsupported, so the contractual verdict is 'partial'. Every recorded
    native comparison must individually pass at the direct_kernel bucket.
    """
    payload = harness.run_fixtures(["boozer_qa_wrappers"])
    entry = _select_fixture(payload, "boozer_qa_wrappers")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry
    assert "sum_CurveLength" in entry["unsupported_components"]

    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"BoozerQA wrapper comparisons failed: {failing}"

    # Sanity-check that the expected quantities were exercised.
    recorded_quantities = {
        cmp["quantity"] for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    }
    expected_quantities = {
        "surface_gamma",
        "surface_unit_normal",
        "field_B",
        "iota",
        "major_radius",
        "nq_symmetric_ratio",
    }
    assert expected_quantities <= recorded_quantities, (
        f"Missing expected comparisons: {expected_quantities - recorded_quantities}"
    )

    # The wrapper scalars must use the direct_kernel bucket: the JAX side
    # is a pure-JAX recomputation at the same surface DOFs, with no LS
    # solver / adjoint involved in this fixture's parity claim.
    expected_tolerance = PARITY_LADDER_TOLERANCES["direct_kernel"]
    for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]:
        if cmp["quantity"] in {"iota", "major_radius", "nq_symmetric_ratio"}:
            assert cmp["tolerance_bucket"] == "direct_kernel", cmp
            assert cmp["tolerance_rtol"] == expected_tolerance["rtol"], cmp
            assert cmp["tolerance_atol"] == expected_tolerance["atol"], cmp

    # CPU lane must record the CPU-only length penalty for traceability,
    # even though it is excluded from cross-lane comparison.
    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    assert "sum_CurveLength" in cpu_components
    assert "iota" in cpu_components
    assert "major_radius" in cpu_components
    assert "nq_symmetric_ratio" in cpu_components

    # Native curve spec hashes must be present on the JAX lane.
    spec_hashes = entry["native_spec_contract"]["native_curve_spec_hashes"]
    assert spec_hashes, "JAX lane must have non-empty native curve spec hashes"


# ---------------------------------------------------------------------------
# Phase 3 — P1 full Stage-II composite fixture parity (partial verdict).


def test_full_stage2_composite_partial_pass_with_cpu_only_components():
    """SquaredFlux portion must pass; CPU-only components must be listed.

    Also enforces the plan §"Math, Physics, And Computation Gates"
    requirement that composite objectives record BOTH raw and weighted
    component values plus the composite total.
    """
    payload = harness.run_fixtures(["full_stage2_composite"])
    entry = _select_fixture(payload, "full_stage2_composite")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry["verdict"]
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"Native-supported comparisons failed: {failing}"

    expected_unsupported = {
        "sum_CurveLength",
        "CurveCurveDistance",
        "CurveSurfaceDistance",
        "sum_LpCurveCurvature",
        "sum_QuadraticPenalty_MeanSquaredCurvature_max",
    }
    assert set(entry["unsupported_components"]) == expected_unsupported, entry[
        "unsupported_components"
    ]
    # Composite total is recorded in the CPU lane components for traceability
    # but is not compared against the JAX subtotal (different physics).
    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    assert "JF_total_cpu" in cpu_components
    assert "SquaredFlux" in cpu_components

    # Plan §"Math, Physics, And Computation Gates": composite objectives
    # must record raw component values (before weights) AND weighted
    # component values for every unsupported component.
    for component in (
        "sum_CurveLength",
        "CurveCurveDistance",
        "CurveSurfaceDistance",
        "sum_LpCurveCurvature",
        "sum_QuadraticPenalty_MeanSquaredCurvature_max",
    ):
        assert f"{component}_raw" in cpu_components, (
            f"Missing _raw entry for {component}: {sorted(cpu_components)}"
        )
        assert f"{component}_weighted" in cpu_components, (
            f"Missing _weighted entry for {component}: {sorted(cpu_components)}"
        )


# ---------------------------------------------------------------------------
# Phase 4 — P2 planar Stage-II composite fixture parity (partial verdict).


def test_planar_stage2_composite_partial_pass_with_link_number_unsupported():
    payload = harness.run_fixtures(["planar_stage2_composite"])
    entry = _select_fixture(payload, "planar_stage2_composite")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry["verdict"]
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"Planar native-supported comparisons failed: {failing}"

    # LinkingNumber must be listed as unsupported (the planar fixture is the
    # only one that adds it).
    assert "LinkingNumber" in entry["unsupported_components"]
    # Native spec hashes must be present (CurvePlanarFourier exposes to_spec).
    assert entry["native_spec_contract"]["native_curve_spec_hashes"]

    # Plan §"Math, Physics, And Computation Gates": raw + weighted entries
    # for every unsupported component. LinkingNumber enters with weight 1,
    # so _raw and _weighted are equal but both must be recorded.
    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    for component in (
        "QuadraticPenalty_over_sum_CurveLength_identity",
        "CurveCurveDistance",
        "CurveSurfaceDistance",
        "sum_LpCurveCurvature",
        "sum_QuadraticPenalty_MeanSquaredCurvature_identity",
        "LinkingNumber",
    ):
        assert f"{component}_raw" in cpu_components, (
            f"Missing _raw entry for {component}: {sorted(cpu_components)}"
        )
        assert f"{component}_weighted" in cpu_components, (
            f"Missing _weighted entry for {component}: {sorted(cpu_components)}"
        )


# ---------------------------------------------------------------------------
# Schema sanity.


def test_payload_schema_versioned(minimal_fixture_payload):
    assert fixtures.SCHEMA_VERSION == "1.0"
    assert minimal_fixture_payload["schema_version"] == "1.0"
    assert minimal_fixture_payload["harness"] == "non_banana_example_cpp_jax_cpu_parity"
    metadata = minimal_fixture_payload["metadata"]
    assert metadata["jax_platform"] == "cpu"
    assert metadata["jax_enable_x64"] is True


def test_no_gpu_required_anywhere(minimal_fixture_payload):
    metadata = minimal_fixture_payload["metadata"]
    for device in metadata["jax_devices"]:
        assert device["platform"] == "cpu"
