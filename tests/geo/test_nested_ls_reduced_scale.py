"""Gate 1 start: reduced nested-LS at archived 255x64 F3 geometry.

Skipped unless the host-local genuine-675 bundle is present. Marked slow
because the LS residual is 3*255*64+2 = 48962 rows. Not an F3 timing claim.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_NEWTON_STAB,
    NESTED_LS_NEWTON_TOL,
    nested_ls_physics_newton_kwargs,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES,
    NestedLsB37TimingBlocked,
    nested_ls_reduced_closures,
    pack_surface_and_y,
    require_full_y_rank,
    run_reduced_nested_ls_newton,
    run_reduced_nested_ls_schur_newton,
    schur_dense_operator_bytes,
    solve_projected_y,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    ARCHIVED_START_QR_G,
    ARCHIVED_START_QR_IOTA,
    DEFAULT_F3_B37_GPU_LANE,
    DEFAULT_F3_B37_NATIVE_LANE,
    F3_B37_BANANA_OMP_CONTRACT_THREADS,
    F3_B37_BANANA_OMP_GAP_THREADS,
    F3_B37_BANANA_OMP_MIN_BRACKET_THREADS,
    F3_B37_BANANA_OMP_THREADS,
    F3_B37_CHUNK_WIDTHS,
    F3_B37_DENSE_LU_ENDPOINT_G,
    F3_B37_DENSE_LU_ENDPOINT_GRAD_L2,
    F3_B37_DENSE_LU_ENDPOINT_IOTA,
    F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256,
    F3_B37_IFT_STAB,
    F3_B37_STEP6_CAP_2048,
    F3_B37_STEP6_CAP_2048_START_MAXITER,
    F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO,
    F3_B37_STEP6_WALL_SECONDS_2048,
    NESTED_LS_GATE6_AGGREGATION,
    NESTED_LS_GATE6_CLAIM_REPEATS,
    NESTED_LS_GATE6_IOTA_G_TOL,
    NESTED_LS_GATE6_NATIVE_OMP_THREADS,
    NESTED_LS_GATE6_PRE_LEVER_CLOCK,
    NESTED_LS_GATE6_PRE_LEVER_JAX_WALK_SECONDS,
    NESTED_LS_GATE6_PRE_LEVER_NATIVE_OMP16_SECONDS,
    NESTED_LS_GATE6_PRE_LEVER_NATIVE_SECONDS,
    NestedLsCountedMatvec,
    NestedLsSchurNewtonWalkProbe,
    archived_f3_b37_lanes_available,
    archived_flat675_bundle_available,
    dump_strict_json,
    evaluate_f3_b37_banana_omp_sweep,
    evaluate_f3_b37_bounded_probe,
    evaluate_f3_b37_chunk_banana_probe,
    evaluate_f3_b37_endpoint_adjoint_probe,
    evaluate_f3_b37_flat_native_probe,
    evaluate_f3_b37_nested_timing,
    evaluate_f3_b37_schur_newton_step,
    evaluate_f3_b37_schur_newton_walk,
    evaluate_f3_b37_step6_architecture_probe,
    evaluate_f3_b37_volume_outer_probe,
    float64_ulps,
    gmres_doubling_cycle_budget,
    kib_to_gib,
    last_step_meets_forcing,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
    nested_ls_omp_threads_pinned,
    nested_ls_receipt_provenance,
    predict_start_at_cap_wall_seconds,
    replace_native_solver_options,
    sha256_file,
    unpreconditioned_gmres_is_insufficient,
)

_EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "receipts" / "evidence"
_F3_B37_BOUNDED_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gate1_f3_b37_bounded_20260820.json"
)
_F3_B37_SCHUR_ONE_STEP_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gate1_f3_b37_schur_one_step_20260820.json"
)
_F3_B37_GPU_WALK_EVIDENCE = _EVIDENCE_DIR / "nested_ls_reduced_gpu_walk_20260821.json"
_F3_B37_GPU_STEP2_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_step2_forcing_20260821.json"
)
_F3_B37_GPU_WALK_CAP64_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_walk_20260821.cap64.incomplete.json"
)
_F3_B37_GPU_STEP4_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_step4_forcing_20260821.json"
)
_F3_B37_GPU_STEP4_INCOMPLETE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_step4_forcing_20260821.incomplete.json"
)
_GPU_STEP4_PUBLICATION = (
    "GPU step-4 forcing probe. Not a walk, not a timing claim, and not F3 7.70x."
)
_F3_B37_GPU_WALK_CAP512_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_walk_20260821.cap512.incomplete.json"
)
_F3_B37_GPU_STEP6_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_step6_forcing_20260821.json"
)
_F3_B37_GPU_STEP6_ARCH_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_step6_architecture_20260821.json"
)
_GPU_STEP6_PUBLICATION = (
    "GPU step-6 forcing probe. Not a walk, not a timing claim, and not F3 7.70x."
)
_GPU_STEP6_ARCH_PUBLICATION = (
    "GPU step-6 solver-architecture canary. Not a walk, not cap-2048, "
    "not a timing claim, and not F3 7.70x."
)
_F3_B37_GPU_WALK_DENSE_LU_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_walk_20260821.dense_lu.json"
)
_GPU_WALK_DENSE_LU_PUBLICATION = (
    "GPU dense-LU Schur walk canary. Opt-in linear_solver=dense_lu, "
    "not a default switch. Not a timing claim and not F3 7.70x."
)
_F3_B37_GPU_ENDPOINT_ADJOINT_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_endpoint_adjoint_20260821.json"
)
_GPU_ENDPOINT_ADJOINT_PUBLICATION = (
    "GPU unregularized IFT adjoint canary at the dense-LU walk endpoint. "
    "Opt-in past the 1 MiB stored-matrix cap. Not a walk, not cap-2048, "
    "not a default switch, not a timing claim, and not F3 7.70x."
)
_F3_B37_GPU_CHUNK_BANANA_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_chunk_banana_20260821.json"
)
_GPU_CHUNK_BANANA_PUBLICATION = (
    "GPU native banana run_code bar plus dense-assemble chunk sweep. "
    "Operators are not comparable. Not a nested speed claim and not F3 7.70x."
)
_F3_B37_GPU_VOLUME_OUTER_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_volume_outer_20260821.json"
)
_GPU_VOLUME_OUTER_PUBLICATION = (
    "GPU F3-B37 Volume directional-gradient canary at the dense-LU endpoint. "
    "Unregularized IFT, one coil-direction FD. Not a B3 outer optimization, "
    "not an outer optimizer loop, not a nested speed claim, and not F3 7.70x."
)
_F3_B37_GPU_BANANA_OMP_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_banana_omp_20260821.json"
)
_GPU_BANANA_OMP_PUBLICATION = (
    "OMP-pinned interleaved native banana run_code sweep. "
    "Not a nested speed claim and not F3 7.70x."
)
_F3_B37_GPU_BANANA_OMP_GAP_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_banana_omp_gap_20260821.json"
)
_GPU_BANANA_OMP_GAP_PUBLICATION = (
    "OMP-pinned banana gap fill at 20 and 24 threads. Fills the "
    "16-to-32 hole on a 32-core box. Not a nested speed claim."
)
_F3_B37_GPU_BANANA_OMP_MIN_BRACKET_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_banana_omp_min_bracket_20260821.json"
)
_GPU_BANANA_OMP_MIN_BRACKET_PUBLICATION = (
    "OMP-pinned banana min-bracket fill at 12 and 14 threads. Brackets "
    "the OMP=16 native peak on a 32-core box. Records process wall and "
    "inner solver time. Not a nested speed claim."
)
_F3_B37_GPU_SHAMANSKII_ATTR_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_shamanskii_attr_20260822.json"
)
_GPU_SHAMANSKII_ATTR_PUBLICATION = (
    "Shamanskii and compile-cache attribution: cache-only, lag-only, and both. "
    "Not a nested speed claim and not F3 7.70x."
)
_F3_B37_GPU_JAX_FLOOR_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_jax_floor_20260822.json"
)
_F3_B37_GPU_GATE6_EVIDENCE = _EVIDENCE_DIR / "nested_ls_reduced_gpu_gate6_20260822.json"
_GPU_GATE6_PUBLICATION = (
    "Gate-6 process-wall vs process-wall claim run. Native banana at "
    "best-of-contract OMP=16, JAX Shamanskii with persistent compile "
    "cache. Not F3 7.70x."
)
_A100_BANANA_OMP_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_a100_banana_omp_20260822.json"
)
_A100_GATE6_EVIDENCE = _EVIDENCE_DIR / "nested_ls_reduced_gpu_gate6_20260822.a100.json"
_F3_B37_GPU_CHUNK_WARM_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gpu_chunk_warm_20260821.json"
)
_GPU_CHUNK_WARM_PUBLICATION = (
    "Warm in-process dense-assemble repeats at the LU endpoint. "
    "Cold first-touch discarded. Not a default switch and not a nested speed claim."
)
_GPU_WALK_PUBLICATION = (
    "GPU forcing-certified Schur walk. Not a timing claim and not F3 7.70x."
)
_GPU_STEP2_PUBLICATION = (
    "GPU step-2 forcing probe. Not a walk, not a timing claim, and not F3 7.70x."
)
_GPU_WALK_GMRES_MATVECS_NOTE = (
    "JAX incremental GMRES does not report operator applications; "
    "gmres_matvecs stays 0 as unavailable telemetry, not zero work. "
    "Default maxiter=1 with restart=8 is one restart cycle of up to "
    "eight Krylov iterations."
)
_RECONSTRUCT_IOTA = 0.14085710955307942
_SCHUR_PUBLICATION = (
    "The Schur operator and one accepted inexact CPU correction are "
    "validated. Independent C++ rejudging confirms the reconstruct "
    "branch. Scientific feasibility passes; receipt provenance and "
    "GPU-performance qualification remain open."
)
_PAIR2_L1_SHA256 = "bde32ab9987d4f2116cf7c7410753c83a9d74ca7836031c25db0e12603155d64"
_NATIVE_BIOT_SHA256 = "0415ae937c78b9f2d68e8463a9176e8f330a9aa172eece160341afccdc29429d"
_SCIENTIFIC_COMMIT = "063b4fe83cb46a5537908a88e233c33030f6f107"
_PUBLICATION = (
    "The bounded F3 B37 feasibility probe is complete: QR elimination, an "
    "off-manifold reduced gradient, a finite synchronized HVP, and the native "
    "reconstruction reference were produced. AD-through-QR GMRES did not "
    "produce a Newton step within the attempted bound. No JAX nested-LS walk, "
    "endpoint parity, or speed claim exists yet."
)
_REQUIRES_BUNDLE = pytest.mark.skipif(
    not archived_flat675_bundle_available(),
    reason="the frozen genuine-675 input bundle is host-local",
)
_REQUIRES_F3_B37 = pytest.mark.skipif(
    not archived_f3_b37_lanes_available(),
    reason="the host-local F3 B37 pair2-l1 lane JSON is missing",
)

pytestmark = [pytest.mark.boozer]


@_REQUIRES_BUNDLE
@pytest.mark.slow
def test_reduced_y_star_matches_archived_start_qr():
    native, jax_boozer, _target = load_archived_nested_ls_pair()
    assert int(np.asarray(native.surface.quadpoints_phi).size) == 255
    assert int(np.asarray(native.surface.quadpoints_theta).size) == 64
    assert int(np.asarray(native.surface.get_dofs()).size) == 661
    residual_fn, _objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    expected_residual_rows = 3 * 255 * 64 + 2
    wrong_probe = np.zeros(2, dtype=np.float64)
    residual = np.asarray(residual_fn(pack_surface_and_y(surface, wrong_probe)))
    assert residual.shape == (expected_residual_rows,)
    solution = solve_projected_y(residual_fn, surface, wrong_probe)
    require_full_y_rank(solution)
    assert tuple(int(v) for v in solution.design_matrix.shape) == (
        expected_residual_rows,
        2,
    )
    y_star = np.asarray(solution.solution, dtype=np.float64)
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        y_star[0],
        ARCHIVED_START_QR_IOTA,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="reduced y* iota missed the archived start QR certificate",
    )
    np.testing.assert_allclose(
        y_star[1],
        ARCHIVED_START_QR_G,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="reduced y* G missed the archived start QR certificate",
    )


@_REQUIRES_BUNDLE
@pytest.mark.slow
def test_reduced_newton_is_a_noop_at_archived_start():
    native, jax_boozer, _target = load_archived_nested_ls_pair()
    native.need_to_run_code = True
    native_result = native.minimize_boozer_penalty_constraints_newton(
        iota=ARCHIVED_START_QR_IOTA,
        G=ARCHIVED_START_QR_G,
        **nested_ls_physics_newton_kwargs(),
    )
    assert bool(native_result["success"]) is True
    assert int(native_result["iter"]) == 0
    start_dofs = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    reduced = run_reduced_nested_ls_newton(
        jax_boozer,
        iota=ARCHIVED_START_QR_IOTA,
        G=ARCHIVED_START_QR_G,
    )
    assert reduced.success is True
    assert reduced.persisted is True
    assert reduced.coil_delta_inf == 0.0
    assert reduced.iteration_count == 0
    assert reduced.reduced_gradient.shape == start_dofs.shape
    np.testing.assert_allclose(
        np.linalg.norm(reduced.reduced_gradient),
        0.0,
        atol=NESTED_LS_NEWTON_TOL,
    )
    np.testing.assert_array_equal(reduced.surface_dofs, start_dofs)
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        reduced.iota,
        float(native_result["iota"]),
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        reduced.G,
        float(native_result["G"]),
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )


def _assert_grid(native) -> None:
    assert int(np.asarray(native.surface.quadpoints_phi).size) == 255
    assert int(np.asarray(native.surface.quadpoints_theta).size) == 64
    assert int(np.asarray(native.surface.get_dofs()).size) == 661


def test_scale_tests_do_not_write_tracked_evidence():
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    write_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"write_text", "write_bytes", "dump"}
    ]
    assert write_calls == []
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
        for node in ast.walk(tree)
    )


def test_f3_b37_bounded_evidence_is_strict_authored_json():
    raw = _F3_B37_BOUNDED_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "nan" not in raw.lower()
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gate1-f3-b37-bounded.v2"
    assert payload["publication"] == _PUBLICATION
    assert payload["driver"] == (
        "simsopt_jax_adapters.geo.nested_ls_reduced_scale.evaluate_f3_b37_bounded_probe"
    )
    assert payload["written_by_pytest"] is False
    probe = payload["probe"]
    assert probe["native_rejudge_iota"] is None
    assert probe["native_rejudge_g"] is None
    assert probe["one_step_attempted"] is False
    assert probe["full_walk_attempted"] is False
    memory = payload["memory"]
    assert memory["rss_kib_after_hvp"] == 1970732
    assert memory["rss_after_hvp_gib"] == pytest.approx(1.879, abs=5.0e-4)
    assert memory["rss_after_hvp_kind"] == "VmRSS_current"
    assert memory["peak_rss_kib_after_hvp"] == 32321968
    assert memory["peak_rss_kind"] == "ru_maxrss_process_lifetime"
    ulp = payload["y_star_versus_lane_inner"]
    assert ulp["iota_ulp"] == pytest.approx(19.0, abs=0.5)
    assert ulp["g_ulp"] == pytest.approx(7.0, abs=0.5)
    assert "8-ULP slogan" in ulp["note"]
    runtime = payload["runtime"]
    assert runtime["jax_default_backend"] == "cpu"
    assert runtime["input_lane_is_not_hvp_hardware"] is True
    assert payload["ad_through_qr_gmres"]["auditable"] is False
    assert kib_to_gib(1970732) == pytest.approx(1.879, abs=5.0e-4)
    assert float64_ulps(
        probe["y_star_iota"],
        ulp["lane_inner_iota"],
    ) == pytest.approx(19.0, abs=0.5)


def test_receipt_provenance_is_strict_json():
    payload = nested_ls_receipt_provenance()
    dump_strict_json(payload)
    assert payload["git_commit"]
    source = payload["source_sha256"]
    assert isinstance(source, dict)
    assert "nested_ls_reduced.py" in source


def test_f3_b37_schur_one_step_evidence_is_strict_authored_json():
    raw = _F3_B37_SCHUR_ONE_STEP_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gate1-f3-b37-schur-one-step.v2"
    assert payload["written_by_pytest"] is False
    assert payload["full_walk_attempted"] is False
    assert payload["publication"] == _SCHUR_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["newton_quality_linear_solve"] is False
    assert boundary["device_resident_krylov"] is False
    assert boundary["endpoint_parity"] is False
    assert boundary["f3_sealed"] is True
    assert boundary["full_walk_attempted"] is False
    assert boundary["provenance_gpu_performance_open"] is True
    assert payload["driver"] == (
        "simsopt_jax_adapters.geo.nested_ls_reduced_scale."
        "evaluate_f3_b37_schur_newton_step"
    )
    assert payload["scientific_run"]["git_commit"] == _SCIENTIFIC_COMMIT
    assert payload["scientific_run"]["git_dirty"] is False
    assert payload["inputs"]["pair2-l1_lane.json"] == _PAIR2_L1_SHA256
    assert payload["inputs"]["native_biot_savart.json"] == _NATIVE_BIOT_SHA256
    sources = payload["scientific_run"]["source_sha256_at_scientific_run"]
    assert sources["nested_ls_reduced.py"] == (
        "132da0338415aa0d907a2096460464039f38708a0dd473a70bbca53ceb448159"
    )
    assert payload["runtime"]["jax_version"] == "0.10.0"
    assert payload["runtime"]["scipy_version"] == "1.17.1"
    assert payload["runtime"]["simsoptpp_sha256"] == (
        "41b2ca791a720f325ffa9b382b31d29bade73f6516693805d41adc0de6f6ed4b"
    )
    schur = payload["independent_replay"]["schur_vs_ad_through_qr"]
    assert schur["rel_l2"] == pytest.approx(8.2859e-16, rel=0, abs=1.0e-20)
    assert schur["max_abs"] == pytest.approx(1.4210854715202004e-13, rel=0, abs=1.0e-20)
    branch = payload["independent_replay"]["cpp_rejudge_branch"]
    assert branch["grad_l2"] == pytest.approx(2.239e-14, rel=0, abs=1.0e-17)
    assert branch["surface_inf_vs_reconstruct"] == pytest.approx(
        4.157e-12, rel=0, abs=1.0e-15
    )
    assert branch["iota"] == pytest.approx(0.1408571095660965, rel=0, abs=1.0e-16)
    sweep = payload["krylov_restart_sweep"]["rows"]
    assert sweep == [
        {"matvecs": 9, "residual_l2": 0.00379309, "restart": 8},
        {"matvecs": 17, "residual_l2": 0.00177416, "restart": 16},
        {"matvecs": 33, "residual_l2": 0.000834828, "restart": 32},
    ]
    assert payload["gmres"]["info"] == 1
    assert payload["gmres"]["restart"] == 8
    assert payload["gmres"]["maxiter"] == 1
    assert payload["krylov_backend"] == "scipy.sparse.linalg.gmres"
    assert payload["hvp_transport"] == "jax.device_get"
    assert payload["probe"]["step_accepted"] is True
    assert payload["probe"]["step_iter"] == 1
    assert payload["probe"]["step_success"] is False
    assert payload["probe"]["native_rejudge_success"] is True
    assert payload["probe"]["native_rejudge_iter"] == 10
    np.testing.assert_allclose(
        payload["probe"]["native_rejudge_iota"],
        _RECONSTRUCT_IOTA,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert payload["runtime"]["jax_default_backend"] == "cpu"
    assert payload["execution_log"] is None


@pytest.mark.skipif(
    not _F3_B37_GPU_WALK_EVIDENCE.is_file(),
    reason="authored GPU walk JSON not yet produced",
)
def test_authored_gpu_walk_json_is_strict_and_claim_grade():
    raw = _F3_B37_GPU_WALK_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-walk.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_WALK_PUBLICATION
    assert payload["driver"] == (
        "simsopt_jax_adapters.geo.nested_ls_reduced_scale."
        "evaluate_f3_b37_schur_newton_walk"
    )
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["f3_sealed"] is True
    assert boundary["full_walk_attempted"] is True
    assert boundary["ten_step_walk"] is True
    assert boundary["device_resident_krylov"] is True
    assert boundary["jax_success_1e13"] is True
    assert boundary["endpoint_parity"] is True
    assert boundary["newton_quality_linear_solve"] is True
    probe = payload["probe"]
    assert probe["success"] is True
    assert float(probe["grad_l2"]) <= NESTED_LS_NEWTON_TOL
    assert int(probe["native_rejudge_iter"]) == 0
    assert float(probe["rejudge_vs_jax_surface_inf"]) == 0.0
    assert float(probe["rejudge_vs_jax_iota"]) == pytest.approx(0.0, abs=1.0e-15)
    assert float(probe["rejudge_vs_jax_g"]) == pytest.approx(0.0, abs=1.0e-15)
    assert float(probe["coil_delta_inf"]) == 0.0
    np.testing.assert_allclose(
        probe["native_rejudge_iota"],
        probe["jax_iota"],
        rtol=0.0,
        atol=1.0e-15,
        err_msg="authored C++ rejudge moved iota away from the JAX endpoint",
    )
    np.testing.assert_allclose(
        probe["native_rejudge_g"],
        probe["jax_g"],
        rtol=0.0,
        atol=1.0e-15,
        err_msg="authored C++ rejudge moved G away from the JAX endpoint",
    )
    assert all(bool(step["step_accepted"]) for step in probe["steps"])
    assert all(
        float(step["gmres_forcing_eta"]) <= float(step["gmres_rtol"])
        for step in probe["steps"]
        if bool(step["step_accepted"])
    )
    runtime = probe["runtime"]
    assert runtime["jax_default_backend"] == "gpu"
    provenance = probe["provenance"]
    repo = Path(__file__).resolve().parents[2]
    sources = provenance["source_sha256"]
    assert sources["nested_ls_reduced.py"] == sha256_file(
        repo / "src/simsopt_jax_adapters/geo/nested_ls_reduced.py"
    )
    assert sources["nested_ls_reduced_scale.py"] == sha256_file(
        repo / "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py"
    )
    assert sources["nested_ls_contract.py"] == sha256_file(
        repo / "src/simsopt_jax_adapters/geo/nested_ls_contract.py"
    )
    assert payload["krylov_backend"] == (
        "jax.scipy.sparse.linalg.gmres incremental via _run_operator_gmres"
    )
    assert "speed" not in payload["publication"].lower() or "not a timing" in (
        payload["publication"].lower()
    )
    assert payload["gmres_matvecs_note"] == _GPU_WALK_GMRES_MATVECS_NOTE


@pytest.mark.skipif(
    not _F3_B37_GPU_STEP2_EVIDENCE.is_file(),
    reason="authored GPU step-2 forcing JSON not yet produced",
)
def test_authored_gpu_step2_forcing_json_is_strict_and_not_a_walk():
    raw = _F3_B37_GPU_STEP2_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-step2-forcing.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_STEP2_PUBLICATION
    assert payload["driver"] == (
        "simsopt_jax_adapters.geo.nested_ls_reduced_scale."
        "evaluate_f3_b37_step2_forcing_probe"
    )
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["f3_sealed"] is True
    assert boundary["full_walk_attempted"] is False
    assert boundary["ten_step_walk"] is False
    probe = payload["probe"]
    assert probe["step1_accepted"] is True
    assert float(probe["step2_eta_requested"]) < 0.24
    assert int(probe["gmres_restart"]) == 8
    assert float(probe["coil_delta_inf"]) == 0.0
    assert probe["rows"]
    assert all(int(row["gmres_restart"]) == 8 for row in probe["rows"])
    assert all(not bool(row["meets_forcing"]) for row in probe["rows"])
    follow_up = payload["follow_up_cap64"]
    assert follow_up["preconditioner"] == "none"
    assert int(follow_up["gmres_restart"]) == 8
    assert int(follow_up["gmres_maxiter_used"]) == 64
    assert float(follow_up["eta_achieved"]) <= float(probe["step2_eta_requested"])
    assert follow_up["meets_forcing"] is True
    runtime = probe["runtime"]
    assert runtime["jax_default_backend"] == "gpu"
    assert payload["gmres_matvecs_note"] == _GPU_WALK_GMRES_MATVECS_NOTE


def test_cap64_incomplete_walk_json_points_at_its_own_log():
    raw = _F3_B37_GPU_WALK_CAP64_EVIDENCE.read_text(encoding="utf-8")
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["execution_log"] == (
        "docs/receipts/evidence/nested_ls_reduced_gpu_walk_20260821.cap64.log"
    )
    probe = payload["probe"]
    assert probe["jax_surface_sha256"] == (
        "286e3dabf3c9113d25aa691e1abe36a109057738e67536d9e46e6d56faa17e24"
    )
    step4 = probe["steps"][3]
    assert float(step4["gmres_rtol"]) == pytest.approx(
        0.04071795165373735, rel=0.0, abs=0.0
    )
    assert float(step4["gmres_forcing_eta"]) == pytest.approx(
        0.1203881060498997, rel=0.0, abs=0.0
    )
    assert float(step4["gmres_forcing_eta"]) != pytest.approx(
        float(step4["gmres_rtol"]), rel=0.0, abs=1.0e-6
    )
    assert payload["claim_boundary"]["nested_speed_claim"] is False
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


def test_cap512_walk_last_step_forcing_is_false():
    payload = json.loads(_F3_B37_GPU_WALK_CAP512_EVIDENCE.read_text(encoding="utf-8"))
    dump_strict_json(payload)
    steps = payload["probe"]["steps"]
    last = steps[-1]
    assert last["step_accepted"] is False
    assert float(last["gmres_forcing_eta"]) == pytest.approx(
        0.09404256261986046, rel=0.0, abs=0.0
    )
    assert float(last["gmres_rtol"]) == pytest.approx(
        0.027034810094191494, rel=0.0, abs=0.0
    )
    assert last_step_meets_forcing(steps) is False
    accepted = [step for step in steps if bool(step["step_accepted"])]
    assert all(
        float(step["gmres_forcing_eta"]) <= float(step["gmres_rtol"])
        for step in accepted
    )


def test_step4_forcing_json_is_dirty_source_replay_not_promotion():
    payload = json.loads(_F3_B37_GPU_STEP4_EVIDENCE.read_text(encoding="utf-8"))
    dump_strict_json(payload)
    assert payload["probe"]["provenance"]["git_dirty"] is True
    assert payload["claim_boundary"]["nested_speed_claim"] is False
    assert payload["claim_boundary"]["full_walk_attempted"] is False
    assert payload["probe"]["meets_forcing"] is True


def test_step4_incomplete_json_fail_closed_on_surface_sha_mismatch():
    raw = _F3_B37_GPU_STEP4_INCOMPLETE.read_text(encoding="utf-8")
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-step4-forcing.v1"
    assert payload["written_by_pytest"] is False
    assert payload["claim_boundary"]["nested_speed_claim"] is False
    assert payload["claim_boundary"]["full_walk_attempted"] is False
    probe = payload["probe"]
    assert probe["surface_sha_match"] is False
    assert probe["fail_closed_reason"] == "surface_sha_mismatch"
    assert probe["rows"] == []
    assert probe["expected_surface_sha256"] == (
        "286e3dabf3c9113d25aa691e1abe36a109057738e67536d9e46e6d56faa17e24"
    )
    assert float(probe["cap64_eta_achieved"]) == pytest.approx(
        0.1203881060498997, rel=0.0, abs=0.0
    )
    assert float(probe["eta_requested"]) != pytest.approx(
        float(probe["cap64_eta_achieved"]), rel=0.0, abs=1.0e-6
    )


@pytest.mark.skipif(
    not _F3_B37_GPU_STEP4_EVIDENCE.is_file(),
    reason="authored GPU step-4 forcing JSON not yet produced",
)
def test_authored_gpu_step4_forcing_json_is_strict_and_not_a_walk():
    raw = _F3_B37_GPU_STEP4_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-step4-forcing.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_STEP4_PUBLICATION
    assert payload["driver"] == (
        "simsopt_jax_adapters.geo.nested_ls_reduced_scale."
        "evaluate_f3_b37_step4_forcing_probe"
    )
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["f3_sealed"] is True
    assert boundary["full_walk_attempted"] is False
    assert boundary["ten_step_walk"] is False
    probe = payload["probe"]
    assert probe["reload_sha_match"] is True
    assert probe["surface_sha256"] == probe["reloaded_surface_sha256"]
    assert probe["historical_surface_sha256"] == (
        "286e3dabf3c9113d25aa691e1abe36a109057738e67536d9e46e6d56faa17e24"
    )
    assert len(probe["jax_surface_dofs"]) == 661
    assert float(probe["historical_eta_requested"]) == pytest.approx(
        0.04071795165373735, rel=0.0, abs=0.0
    )
    assert float(probe["cap64_eta_achieved"]) == pytest.approx(
        0.1203881060498997, rel=0.0, abs=0.0
    )
    assert float(probe["eta_requested"]) != pytest.approx(
        float(probe["cap64_eta_achieved"]), rel=0.0, abs=1.0e-6
    )
    assert int(probe["gmres_restart"]) == 8
    assert float(probe["coil_delta_inf"]) == 0.0
    assert probe["meets_forcing"] is True
    assert probe["rows"]
    assert all(row.get("preconditioner") == "none" for row in probe["rows"])
    assert all(int(row["gmres_restart"]) == 8 for row in probe["rows"])
    assert any(bool(row["meets_forcing"]) for row in probe["rows"])
    runtime = probe["runtime"]
    assert runtime["jax_default_backend"] == "gpu"
    assert payload["gmres_matvecs_note"] == _GPU_WALK_GMRES_MATVECS_NOTE


@pytest.mark.skipif(
    not _F3_B37_GPU_STEP6_EVIDENCE.is_file(),
    reason="authored GPU step-6 forcing JSON not yet produced",
)
def test_authored_gpu_step6_forcing_json_is_strict_and_not_a_walk():
    raw = _F3_B37_GPU_STEP6_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-step6-forcing.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_STEP6_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["full_walk_attempted"] is False
    probe = payload["probe"]
    assert probe["reload_sha_match"] is True
    assert probe["surface_sha256"] == (
        "a0493560d7ebe3455b68bb834830ad59fb1fb510f79a447c61267547cdc0effe"
    )
    assert probe["surface_sha256"] == probe["reloaded_surface_sha256"]
    assert float(probe["iota"]) == pytest.approx(0.1484103489869863, rel=0.0, abs=0.0)
    assert float(probe["G"]) == pytest.approx(2.0106193052280394, rel=0.0, abs=0.0)
    assert float(probe["eta_requested"]) == pytest.approx(
        0.027034810094191494, rel=0.0, abs=0.0
    )
    assert float(probe["cap512_eta_achieved"]) == pytest.approx(
        0.09404256261986046, rel=0.0, abs=0.0
    )
    assert float(probe["eta_requested"]) != pytest.approx(
        float(probe["cap512_eta_achieved"]), rel=0.0, abs=1.0e-6
    )
    assert int(probe["gmres_restart"]) == 8
    assert all(row.get("preconditioner") == "none" for row in probe["rows"])
    assert payload["gmres_matvecs_note"] == _GPU_WALK_GMRES_MATVECS_NOTE
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


def test_step6_forcing_json_is_dirty_source_replay_not_promotion():
    payload = json.loads(_F3_B37_GPU_STEP6_EVIDENCE.read_text(encoding="utf-8"))
    dump_strict_json(payload)
    assert payload["probe"]["provenance"]["git_dirty"] is True
    assert payload["probe"]["provenance"]["source_sha256"][
        "nested_ls_reduced_scale.py"
    ].startswith("1f4d66ac")
    assert payload["claim_boundary"]["nested_speed_claim"] is False
    assert payload["claim_boundary"]["full_walk_attempted"] is False
    assert payload["probe"]["unpreconditioned_gmres_insufficient"] is False
    assert payload["probe"]["fail_closed_reason"] == "wall_time_cap"


def test_unpreconditioned_gmres_insufficient_semantics():
    ratio = F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO
    assert unpreconditioned_gmres_is_insufficient("stagnation") is True
    assert (
        unpreconditioned_gmres_is_insufficient(
            "eta_unmet",
            residual_ratio=0.34,
            material_residual_ratio=ratio,
        )
        is False
    )
    assert (
        unpreconditioned_gmres_is_insufficient(
            "eta_unmet",
            residual_ratio=0.95,
            material_residual_ratio=ratio,
        )
        is True
    )
    assert (
        unpreconditioned_gmres_is_insufficient(
            "eta_unmet",
            residual_ratio=None,
            material_residual_ratio=ratio,
        )
        is True
    )
    assert unpreconditioned_gmres_is_insufficient("wall_time_cap") is False
    assert unpreconditioned_gmres_is_insufficient("surface_sha_mismatch") is False


def test_step6_2048_leg_starts_at_cap_and_predictor_drops_double_pay():
    assert F3_B37_STEP6_CAP_2048_START_MAXITER == F3_B37_STEP6_CAP_2048 == 2048
    assert gmres_doubling_cycle_budget(1024, 2048) == 1024 + 2048
    assert gmres_doubling_cycle_budget(2048, 2048) == 2048
    assert gmres_doubling_cycle_budget(512, 1024) == 512 + 1024
    predicted = predict_start_at_cap_wall_seconds(
        628.6738862220664,
        previous_start_maxiter=512,
        previous_cap=1024,
        next_cap=2048,
    )
    assert predicted == pytest.approx(
        628.6738862220664 * (2048.0 / 1536.0), rel=0.0, abs=1.0e-12
    )
    assert predicted <= F3_B37_STEP6_WALL_SECONDS_2048
    source = Path(
        evaluate_f3_b37_step6_architecture_probe.__code__.co_filename
    ).read_text(encoding="utf-8")
    assert "maxiter=int(F3_B37_STEP6_CAP_2048_START_MAXITER)" in source
    assert "maxiter_cap=int(F3_B37_STEP6_CAP_2048)" in source
    assert "maxiter=1024,\n                    maxiter_cap=2048" not in source


def test_step6_architecture_probe_is_not_a_walk_or_cap2048():
    source_path = Path(evaluate_f3_b37_step6_architecture_probe.__code__.co_filename)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    fn_node = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "evaluate_f3_b37_step6_architecture_probe"
        ):
            fn_node = node
            break
    assert fn_node is not None
    text = ast.get_source_segment(source_path.read_text(encoding="utf-8"), fn_node)
    assert text is not None
    assert "evaluate_f3_b37_schur_newton_walk" not in text
    assert "F3_B37_STEP6_CAP_2048" not in text
    assert "maxiter_cap=2048" not in text


def test_schur_newton_walk_defaults_to_gmres():
    parameter = inspect.signature(evaluate_f3_b37_schur_newton_walk).parameters[
        "linear_solver"
    ]
    assert parameter.default == "gmres"
    assert (
        inspect.signature(run_reduced_nested_ls_schur_newton)
        .parameters["linear_solver"]
        .default
        == "gmres"
    )
    source_path = Path(evaluate_f3_b37_schur_newton_walk.__code__.co_filename)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    fn_node = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "evaluate_f3_b37_schur_newton_walk"
        ):
            fn_node = node
            break
    assert fn_node is not None
    text = ast.get_source_segment(source, fn_node)
    assert text is not None
    assert "linear_solver=str(linear_solver)" in text
    assert 'linear_solver="dense_lu"' not in text
    assert "linear_solver='dense_lu'" not in text
    assert 'linear_solver="shamanskii"' not in text
    assert "linear_solver='shamanskii'" not in text


def test_counted_matvec_increments_through_jax_gmres():
    from jax.scipy.sparse.linalg import gmres as jax_gmres

    diagonal = jnp.arange(1.0, 9.0, dtype=jnp.float64)

    def matvec(tangent: jax.Array) -> jax.Array:
        return diagonal * tangent

    counter = NestedLsCountedMatvec(matvec)
    solution, _info = jax_gmres(
        counter,
        jnp.ones((8,), dtype=jnp.float64),
        None,
        tol=1.0e-16,
        atol=0.0,
        restart=4,
        maxiter=2,
        solve_method="incremental",
    )
    jax.block_until_ready(solution)
    assert counter.count >= 8


@pytest.mark.skipif(
    not _F3_B37_GPU_STEP6_ARCH_EVIDENCE.is_file(),
    reason="authored GPU step-6 architecture JSON not yet produced",
)
def test_authored_gpu_step6_architecture_json_is_strict_and_not_a_walk():
    raw = _F3_B37_GPU_STEP6_ARCH_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-step6-architecture.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_STEP6_ARCH_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["newton_quality_linear_solve"] is True
    assert boundary["explicit_inverse_m_production"] is False
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["full_walk_attempted"] is False
    assert boundary["cap_2048_attempted"] is False
    probe = payload["probe"]
    assert probe["reload_sha_match"] is True
    assert probe["surface_sha256"] == (
        "a0493560d7ebe3455b68bb834830ad59fb1fb510f79a447c61267547cdc0effe"
    )
    assert int(probe["hvp_budget"]) == 128
    assert any(row.get("row") == "dense_lu" for row in probe["rows"])
    assert probe["dense_meets_forcing"] is True
    dense_row = next(row for row in probe["rows"] if row.get("row") == "dense_lu")
    assert dense_row["eta_achieved"] is not None
    assert float(dense_row["eta_achieved"]) <= float(probe["eta_requested"])
    assert dense_row.get("eta_achieved_dense_materialization") is not None
    dense_chunk_batch_size = probe.get("dense_chunk_batch_size")
    if dense_chunk_batch_size is not None:
        assert isinstance(dense_chunk_batch_size, int)
        assert dense_chunk_batch_size >= 1
    option_b = next(
        (row for row in probe["rows"] if row.get("row") == "option_b_dense_inverse_m"),
        None,
    )
    if option_b is not None and "shared_dense_assembly" in option_b:
        assert option_b["shared_dense_assembly"] is True
        assert option_b["excludes_assembly_seconds"] is True
        assert option_b["excludes_inversion_seconds"] is True
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _F3_B37_GPU_WALK_DENSE_LU_EVIDENCE.is_file(),
    reason="authored GPU dense-LU walk JSON not yet produced",
)
def test_authored_gpu_dense_lu_walk_json_is_strict_and_not_a_speed_claim():
    raw = _F3_B37_GPU_WALK_DENSE_LU_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-walk.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_WALK_DENSE_LU_PUBLICATION
    assert payload["linear_solver"] == "dense_lu"
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["explicit_inverse_m_production"] is False
    assert boundary["cap_2048_attempted"] is False
    assert boundary["full_walk_attempted"] is True
    assert boundary["jax_success_1e13"] is True
    probe = payload["probe"]
    assert probe["linear_solver"] == "dense_lu"
    assert probe["success"] is True
    assert probe["provenance"]["git_dirty"] is False
    assert float(probe["grad_l2"]) <= NESTED_LS_NEWTON_TOL
    assert int(probe["native_rejudge_iter"]) == 0
    assert float(probe["rejudge_vs_jax_iota"]) == 0.0
    assert float(probe["rejudge_vs_jax_g"]) == 0.0
    assert float(probe["rejudge_vs_jax_surface_inf"]) == 0.0
    assert float(probe["coil_delta_inf"]) == 0.0
    assert last_step_meets_forcing(probe["steps"]) is True
    assert all(bool(step["step_accepted"]) for step in probe["steps"])
    assert all(
        float(step["gmres_forcing_eta"]) <= float(step["gmres_rtol"])
        for step in probe["steps"]
    )
    assert probe["runtime"]["jax_default_backend"] == "gpu"
    assert int(payload["dense_chunk_batch_size"]) >= 1
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


def test_endpoint_adjoint_probe_is_not_a_walk_or_cap2048_or_newton_stab_ift():
    assert F3_B37_IFT_STAB == 0.0
    assert F3_B37_IFT_STAB != NESTED_LS_NEWTON_STAB
    assert (
        schur_dense_operator_bytes(661) > NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES
    )
    assert F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256 == (
        "e25ca0f2fedf25cf411f7b7ad7192860c813ad5fd37bdb5471355834d42ede6c"
    )
    assert F3_B37_DENSE_LU_ENDPOINT_IOTA == pytest.approx(
        0.14085710957665173, rel=0.0, abs=0.0
    )
    assert F3_B37_DENSE_LU_ENDPOINT_G == pytest.approx(
        2.0106193053897154, rel=0.0, abs=0.0
    )
    assert F3_B37_DENSE_LU_ENDPOINT_GRAD_L2 == pytest.approx(
        2.404212353322172e-14, rel=0.0, abs=0.0
    )
    source_path = Path(evaluate_f3_b37_endpoint_adjoint_probe.__code__.co_filename)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    fn_node = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "evaluate_f3_b37_endpoint_adjoint_probe"
        ):
            fn_node = node
            break
    assert fn_node is not None
    text = ast.get_source_segment(source, fn_node)
    assert text is not None
    assert "evaluate_f3_b37_schur_newton_walk" not in text
    assert "F3_B37_STEP6_CAP_2048" not in text
    assert "maxiter_cap=2048" not in text

    def _called(node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _is_float_name(expr: ast.AST, name: str) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "float"
            and len(expr.args) == 1
            and isinstance(expr.args[0], ast.Name)
            and expr.args[0].id == name
        )

    def _keyword(call: ast.Call, name: str) -> ast.AST | None:
        for keyword in call.keywords:
            if keyword.arg == name:
                return keyword.value
        return None

    materialize_calls = [
        node
        for node in ast.walk(fn_node)
        if isinstance(node, ast.Call)
        and _called(node) == "materialize_stabilized_schur_dense"
    ]
    assert len(materialize_calls) == 1
    materialize = materialize_calls[0]
    assert len(materialize.args) >= 2
    assert _is_float_name(materialize.args[1], "F3_B37_IFT_STAB")
    cap = _keyword(materialize, "max_dense_linearization_bytes")
    assert isinstance(cap, ast.Constant) and cap.value is None

    adjoint_calls = [
        node
        for node in ast.walk(fn_node)
        if isinstance(node, ast.Call)
        and _called(node) == "implicit_adjoint_coil_gradient"
    ]
    assert len(adjoint_calls) == 1
    adjoint = adjoint_calls[0]
    adjoint_stab = _keyword(adjoint, "stab")
    assert adjoint_stab is not None and _is_float_name(adjoint_stab, "F3_B37_IFT_STAB")
    adjoint_solver = _keyword(adjoint, "linear_solver")
    assert isinstance(adjoint_solver, ast.Constant) and adjoint_solver.value == (
        "dense_lu"
    )
    adjoint_cap = _keyword(adjoint, "max_dense_linearization_bytes")
    assert isinstance(adjoint_cap, ast.Constant) and adjoint_cap.value is None

    newton_calls = [
        node
        for node in ast.walk(fn_node)
        if isinstance(node, ast.Call)
        and _called(node) == "run_reduced_nested_ls_schur_newton"
    ]
    assert len(newton_calls) == 2
    for newton in newton_calls:
        newton_stab = _keyword(newton, "stab")
        assert newton_stab is not None and _is_float_name(
            newton_stab, "F3_B37_IFT_STAB"
        )
        newton_solver = _keyword(newton, "linear_solver")
        assert isinstance(newton_solver, ast.Constant) and newton_solver.value == (
            "dense_lu"
        )

    walk_default = inspect.signature(evaluate_f3_b37_schur_newton_walk).parameters[
        "linear_solver"
    ]
    assert walk_default.default == "gmres"
    newton_default = inspect.signature(run_reduced_nested_ls_schur_newton).parameters[
        "linear_solver"
    ]
    assert newton_default.default == "gmres"


@pytest.mark.skipif(
    not _F3_B37_GPU_ENDPOINT_ADJOINT_EVIDENCE.is_file(),
    reason="authored GPU endpoint adjoint JSON not yet produced",
)
def test_authored_gpu_endpoint_adjoint_json_is_strict_and_not_a_walk():
    raw = _F3_B37_GPU_ENDPOINT_ADJOINT_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-endpoint-adjoint.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_ENDPOINT_ADJOINT_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["explicit_inverse_m_production"] is False
    assert boundary["full_walk_attempted"] is False
    assert boundary["cap_2048_attempted"] is False
    assert boundary["moving_coil_b3_outer"] is False
    assert boundary["ift_used_newton_stab"] is False
    probe = payload["probe"]
    assert probe["reload_sha_match"] is True
    assert probe["surface_sha256"] == F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256
    assert probe["ift_stab"] == 0.0
    assert probe["ift_used_newton_stab"] is False
    assert probe["default_cap_refuses_661"] is True
    assert int(probe["default_adjoint_cap_bytes"]) == 1_048_576
    assert int(probe["dense_bytes"]) == schur_dense_operator_bytes(661)
    assert probe["fail_closed_reason"] is None
    assert probe["unregularized_positive_definite"] is True
    assert probe["adjoint_live_eta"] is not None
    assert float(probe["adjoint_live_eta"]) <= 1.0e-10
    assert probe["vjp_match"] is True
    assert probe["fd_match"] is True
    assert float(probe["coil_delta_inf"]) == 0.0
    assert float(probe["coil_delta_inf_after"]) == 0.0
    assert probe["runtime"]["jax_default_backend"] == "gpu"
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


def test_chunk_banana_and_volume_outer_are_not_walks_or_default_switches():
    assert F3_B37_CHUNK_WIDTHS == (8, 16, 32, 64)
    assert (
        inspect.signature(run_reduced_nested_ls_schur_newton)
        .parameters["linear_solver"]
        .default
        == "gmres"
    )
    assert (
        inspect.signature(evaluate_f3_b37_schur_newton_walk)
        .parameters["linear_solver"]
        .default
        == "gmres"
    )
    assert evaluate_f3_b37_volume_outer_probe.__name__ == (
        "evaluate_f3_b37_volume_outer_probe"
    )
    source_path = Path(evaluate_f3_b37_chunk_banana_probe.__code__.co_filename)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    def _fn(name: str) -> ast.FunctionDef:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(name)

    def _called(node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    chunk_fn = _fn("evaluate_f3_b37_chunk_banana_probe")
    chunk_text = ast.get_source_segment(source, chunk_fn)
    assert chunk_text is not None
    assert "evaluate_f3_b37_schur_newton_walk" not in chunk_text
    assert "F3_B37_STEP6_CAP_2048" not in chunk_text
    assert any(
        isinstance(node, ast.Call)
        and _called(node) == "materialize_stabilized_schur_dense"
        and any(kw.arg == "chunk_batch_size" for kw in node.keywords)
        for node in ast.walk(chunk_fn)
    )
    volume_fn = _fn("evaluate_f3_b37_volume_outer_probe")
    volume_text = ast.get_source_segment(source, volume_fn)
    assert volume_text is not None
    assert "evaluate_f3_b37_schur_newton_walk" not in volume_text
    newton_calls = [
        node
        for node in ast.walk(volume_fn)
        if isinstance(node, ast.Call)
        and _called(node) == "run_reduced_nested_ls_schur_newton"
    ]
    assert len(newton_calls) == 2
    for call in newton_calls:
        stab = next(kw.value for kw in call.keywords if kw.arg == "stab")
        solver = next(kw.value for kw in call.keywords if kw.arg == "linear_solver")
        assert isinstance(stab, ast.Call)
        assert isinstance(stab.func, ast.Name) and stab.func.id == "float"
        assert isinstance(stab.args[0], ast.Name)
        assert stab.args[0].id == "F3_B37_IFT_STAB"
        assert isinstance(solver, ast.Constant) and solver.value == "dense_lu"


@pytest.mark.skipif(
    not _F3_B37_GPU_CHUNK_BANANA_EVIDENCE.is_file(),
    reason="authored GPU chunk/banana JSON not yet produced",
)
def test_authored_gpu_chunk_banana_json_is_strict_and_not_a_speed_claim():
    raw = _F3_B37_GPU_CHUNK_BANANA_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-chunk-banana.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_CHUNK_BANANA_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["comparable_operators"] is False
    assert boundary["cap_2048_attempted"] is False
    probe = payload["probe"]
    assert probe["banana_success"] is True
    assert float(probe["banana_coil_delta_inf"]) == 0.0
    widths = [int(row["chunk_batch_size"]) for row in probe["rows"]]
    assert widths == list(F3_B37_CHUNK_WIDTHS)
    assert probe["fail_closed_reason"] is None
    assert probe.get("banana_omp_pinned") is not True
    assert probe["runtime"]["jax_default_backend"] == "gpu"
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _F3_B37_GPU_VOLUME_OUTER_EVIDENCE.is_file(),
    reason="authored GPU Volume outer JSON not yet produced",
)
def test_authored_gpu_volume_outer_json_is_strict_and_not_a_walk():
    raw = _F3_B37_GPU_VOLUME_OUTER_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-volume-outer.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_VOLUME_OUTER_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["moving_coil_b3_outer"] is False
    assert boundary["outer_optimizer_loop"] is False
    assert boundary.get("volume_directional_gradient") is True
    assert boundary["cap_2048_attempted"] is False
    probe = payload["probe"]
    assert probe["surface_sha256"] == F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256
    assert probe["fail_closed_reason"] is None
    assert probe["vjp_match"] is True
    assert probe["fd_match"] is True
    assert float(probe["coil_delta_inf_after"]) == 0.0
    assert probe["runtime"]["jax_default_backend"] == "gpu"
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


def test_omp_pin_requires_positive_integer_env():
    assert nested_ls_omp_threads_pinned({"OMP_NUM_THREADS": None}) is False
    assert nested_ls_omp_threads_pinned({"OMP_NUM_THREADS": ""}) is False
    assert nested_ls_omp_threads_pinned({"OMP_NUM_THREADS": "unset"}) is False
    assert nested_ls_omp_threads_pinned({"OMP_NUM_THREADS": "8"}) is True
    assert F3_B37_BANANA_OMP_THREADS == (4, 8, 16, 32)
    assert F3_B37_BANANA_OMP_GAP_THREADS == (20, 24)
    assert F3_B37_BANANA_OMP_MIN_BRACKET_THREADS == (12, 14)
    assert F3_B37_BANANA_OMP_CONTRACT_THREADS == (4, 8, 12, 14, 16, 20, 24, 32)
    assert F3_B37_BANANA_OMP_CONTRACT_THREADS == tuple(
        sorted(
            set(F3_B37_BANANA_OMP_THREADS)
            | set(F3_B37_BANANA_OMP_GAP_THREADS)
            | set(F3_B37_BANANA_OMP_MIN_BRACKET_THREADS)
        )
    )
    assert NESTED_LS_GATE6_IOTA_G_TOL == 1.0e-11
    assert NESTED_LS_GATE6_CLAIM_REPEATS == 3
    assert NESTED_LS_GATE6_AGGREGATION == "min"
    assert NESTED_LS_GATE6_PRE_LEVER_CLOCK == "inner_solver"
    assert NESTED_LS_GATE6_PRE_LEVER_JAX_WALK_SECONDS == 153.06041832105257
    assert NESTED_LS_GATE6_PRE_LEVER_NATIVE_OMP16_SECONDS == 116.0183689862024
    assert (
        NESTED_LS_GATE6_PRE_LEVER_NATIVE_SECONDS
        <= NESTED_LS_GATE6_PRE_LEVER_NATIVE_OMP16_SECONDS
    )
    assert schur_dense_operator_bytes(661) == 3_495_368
    assert (
        schur_dense_operator_bytes(661) > NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES
    )
    newton_cap = inspect.signature(run_reduced_nested_ls_schur_newton).parameters[
        "max_dense_linearization_bytes"
    ]
    assert newton_cap.default is None
    if _F3_B37_GPU_WALK_DENSE_LU_EVIDENCE.is_file():
        walk_seconds = float(
            json.loads(_F3_B37_GPU_WALK_DENSE_LU_EVIDENCE.read_text(encoding="utf-8"))[
                "probe"
            ]["walk_seconds"]
        )
        assert walk_seconds == NESTED_LS_GATE6_PRE_LEVER_JAX_WALK_SECONDS
    if _F3_B37_GPU_BANANA_OMP_EVIDENCE.is_file():
        omp_rows = json.loads(
            _F3_B37_GPU_BANANA_OMP_EVIDENCE.read_text(encoding="utf-8")
        )["probe"]["rows"]
        best16 = min(
            float(row["seconds"])
            for row in omp_rows
            if int(row["omp_num_threads"]) == 16
        )
        assert best16 == NESTED_LS_GATE6_PRE_LEVER_NATIVE_OMP16_SECONDS
    source = Path(evaluate_f3_b37_banana_omp_sweep.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    assert "OMP_NUM_THREADS" in source
    assert "process_wall_seconds" in source
    assert "inner_solver_seconds" in source
    driver = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "nested_ls_f3_b37_gpu_canaries.py"
    )
    text = driver.read_text(encoding="utf-8")
    assert "docs/receipts/evidence" in text
    assert "/tmp/" not in text
    attr = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "nested_ls_shamanskii_attribution.py"
    )
    attr_text = attr.read_text(encoding="utf-8")
    assert "success={row['success']!r}" in attr_text
    assert "REPEATS = NESTED_LS_GATE6_CLAIM_REPEATS" in attr_text
    assert NESTED_LS_GATE6_CLAIM_REPEATS == 3
    assert NESTED_LS_GATE6_NATIVE_OMP_THREADS == 16
    _repo = Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from benchmarks.nested_ls_shamanskii_attribution import (
        REPEATS as ATTR_REPEATS,
    )
    from benchmarks.nested_ls_shamanskii_attribution import (
        jax_claim_wall_seconds,
        parse_args,
    )

    assert ATTR_REPEATS == NESTED_LS_GATE6_CLAIM_REPEATS
    assert parse_args(["--lane", "lag_only"]).lane == "lag_only"
    assert jax_claim_wall_seconds(
        {
            "process_wall_seconds": 10.0,
            "reconstruct_seconds": 3.0,
            "native_rejudge_seconds": 2.0,
        }
    ) == pytest.approx(5.0)
    assert '"--lane"' in attr_text
    assert "lane per process" in attr_text
    assert "jax_floor_seconds" in attr_text
    assert "jax_process_wall_seconds" in attr_text
    assert "reconstruct_seconds" in attr_text
    assert "jax_claim_wall_seconds" in attr_text
    assert "--allow-dirty" in attr_text
    assert "git_implementation_dirty" in attr_text
    assert "docs/receipts/evidence/" in attr_text
    child = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "nested_ls_shamanskii_child.py"
    )
    child_text = child.read_text(encoding="utf-8")
    assert "_T0 = time.perf_counter()" in child_text
    assert 'linear_solver == "floor"' in child_text
    assert "jax_floor_seconds" in child_text
    assert "process_elapsed_seconds" in child_text
    assert "reconstruct_seconds" in NestedLsSchurNewtonWalkProbe.__dataclass_fields__
    walk_source = Path(
        evaluate_f3_b37_schur_newton_walk.__code__.co_filename
    ).read_text(encoding="utf-8")
    assert "reconstruct_seconds=float(reconstruct_seconds)" in walk_source
    gate6 = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "nested_ls_gate6_claim.py"
    )
    gate6_text = gate6.read_text(encoding="utf-8")
    assert "nested_speed_claim" in gate6_text
    assert "jax_claim_wall_seconds" in gate6_text
    assert "parent_wait_minus_reconstruct_rejudge" in gate6_text
    assert "OMP_NUM_THREADS" in gate6_text
    assert "clean tree" in gate6_text
    assert "shamanskii" in gate6_text
    assert "observed_omp_num_threads" in gate6_text


@pytest.mark.skipif(
    not _F3_B37_GPU_BANANA_OMP_EVIDENCE.is_file(),
    reason="authored OMP banana sweep JSON not yet produced",
)
def test_authored_gpu_banana_omp_json_is_pinned_and_not_a_speed_claim():
    raw = _F3_B37_GPU_BANANA_OMP_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-banana-omp.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_BANANA_OMP_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["omp_pinned"] is True
    assert boundary["interleaved_repeats"] is True
    probe = payload["probe"]
    assert probe["fail_closed_reason"] is None
    assert probe["any_unpinned"] is False
    assert {int(row["omp_num_threads"]) for row in probe["rows"]} == set(
        F3_B37_BANANA_OMP_THREADS
    )
    assert all(bool(row["omp_pinned"]) for row in probe["rows"])
    assert all(float(row["coil_delta_inf"]) == 0.0 for row in probe["rows"])
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _F3_B37_GPU_BANANA_OMP_GAP_EVIDENCE.is_file(),
    reason="authored OMP banana 20/24 gap JSON not yet produced",
)
def test_authored_gpu_banana_omp_gap_json_fills_16_to_32():
    raw = _F3_B37_GPU_BANANA_OMP_GAP_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-banana-omp-gap.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_BANANA_OMP_GAP_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["omp_pinned"] is True
    assert boundary["gap_fill_20_24"] is True
    assert boundary["interleaved_repeats"] is True
    probe = payload["probe"]
    assert probe["fail_closed_reason"] is None
    assert probe["repeats"] == 2
    assert payload["threads"] == list(F3_B37_BANANA_OMP_GAP_THREADS)
    assert {int(row["omp_num_threads"]) for row in probe["rows"]} == set(
        F3_B37_BANANA_OMP_GAP_THREADS
    )
    assert all(bool(row["omp_pinned"]) for row in probe["rows"])
    assert all(bool(row["success"]) for row in probe["rows"])
    assert all(float(row["coil_delta_inf"]) == 0.0 for row in probe["rows"])
    assert all(
        abs(float(row["iota"]) - F3_B37_DENSE_LU_ENDPOINT_IOTA)
        <= NESTED_LS_GATE6_IOTA_G_TOL
        for row in probe["rows"]
    )
    assert all(
        abs(float(row["G"]) - F3_B37_DENSE_LU_ENDPOINT_G) <= NESTED_LS_GATE6_IOTA_G_TOL
        for row in probe["rows"]
    )
    gap_best = min(float(row["seconds"]) for row in probe["rows"])
    assert gap_best > NESTED_LS_GATE6_PRE_LEVER_NATIVE_OMP16_SECONDS
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _F3_B37_GPU_BANANA_OMP_MIN_BRACKET_EVIDENCE.is_file(),
    reason="authored OMP banana 12/14 min-bracket JSON not yet produced",
)
def test_authored_gpu_banana_omp_min_bracket_json_brackets_16():
    raw = _F3_B37_GPU_BANANA_OMP_MIN_BRACKET_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-banana-omp-min-bracket.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_BANANA_OMP_MIN_BRACKET_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["omp_pinned"] is True
    assert boundary["gap_fill_12_14"] is True
    assert boundary["inner_and_process_wall"] is True
    assert boundary["interleaved_repeats"] is True
    probe = payload["probe"]
    assert probe["fail_closed_reason"] is None
    assert probe["repeats"] == 2
    assert payload["threads"] == list(F3_B37_BANANA_OMP_MIN_BRACKET_THREADS)
    assert {int(row["omp_num_threads"]) for row in probe["rows"]} == set(
        F3_B37_BANANA_OMP_MIN_BRACKET_THREADS
    )
    assert all(bool(row["omp_pinned"]) for row in probe["rows"])
    assert all(bool(row["success"]) for row in probe["rows"])
    assert all(float(row["coil_delta_inf"]) == 0.0 for row in probe["rows"])
    assert all("process_wall_seconds" in row for row in probe["rows"])
    assert all("inner_solver_seconds" in row for row in probe["rows"])
    assert all("omp_proc_bind" in row for row in probe["rows"])
    assert all("omp_places" in row for row in probe["rows"])
    assert all(
        float(row["inner_solver_seconds"]) == float(row["seconds"])
        for row in probe["rows"]
    )
    assert all(
        float(row["process_wall_seconds"]) >= float(row["inner_solver_seconds"])
        for row in probe["rows"]
    )
    assert all(
        abs(float(row["iota"]) - F3_B37_DENSE_LU_ENDPOINT_IOTA)
        <= NESTED_LS_GATE6_IOTA_G_TOL
        for row in probe["rows"]
    )
    assert all(
        abs(float(row["G"]) - F3_B37_DENSE_LU_ENDPOINT_G) <= NESTED_LS_GATE6_IOTA_G_TOL
        for row in probe["rows"]
    )
    bracket_best = min(float(row["inner_solver_seconds"]) for row in probe["rows"])
    assert NESTED_LS_GATE6_PRE_LEVER_NATIVE_SECONDS == min(
        NESTED_LS_GATE6_PRE_LEVER_NATIVE_OMP16_SECONDS, bracket_best
    )
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _F3_B37_GPU_CHUNK_WARM_EVIDENCE.is_file(),
    reason="authored warm chunk JSON not yet produced",
)
def test_authored_gpu_chunk_warm_json_is_not_a_default_switch():
    raw = _F3_B37_GPU_CHUNK_WARM_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-chunk-warm.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_CHUNK_WARM_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["production_chunk_default_unchanged"] is True
    assert boundary["warm_repeated"] is True
    probe = payload["probe"]
    assert probe["fail_closed_reason"] is None
    assert all(bool(row["discarded_warmup"]) for row in probe["rows"])
    assert [int(row["chunk_batch_size"]) for row in probe["rows"]] == list(
        F3_B37_CHUNK_WIDTHS
    )
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _F3_B37_GPU_SHAMANSKII_ATTR_EVIDENCE.is_file(),
    reason="authored Shamanskii attribution JSON not yet produced",
)
def test_authored_gpu_shamanskii_attr_json_is_not_a_speed_claim():
    raw = _F3_B37_GPU_SHAMANSKII_ATTR_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-shamanskii-attr.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_SHAMANSKII_ATTR_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["shamanskii_attribution"] is True
    assert boundary["inner_and_process_wall"] is True
    lanes = {
        str(row["lane"])
        for row in payload["rows"]
        if row["role"] in {"prime", "measure"}
    }
    assert lanes == {"cache_only", "lag_only", "both"}
    measures = [row for row in payload["rows"] if row["role"] == "measure"]
    assert measures
    assert {str(row["lane"]) for row in measures} == {"cache_only", "lag_only", "both"}
    measure_counts = {
        lane: sum(1 for row in measures if row["lane"] == lane)
        for lane in ("cache_only", "lag_only", "both")
    }
    assert all(count >= 3 for count in measure_counts.values())
    assert all(bool(row["success"]) for row in measures)
    assert all(int(row["native_rejudge_iter"]) == 0 for row in measures)
    assert all(float(row["coil_delta_inf"]) == 0.0 for row in measures)
    assert all("process_wall_seconds" in row for row in measures)
    assert all("process_elapsed_seconds" in row for row in measures)
    assert all("jax_process_wall_seconds" in row for row in measures)
    assert all("jax_floor_seconds" in row for row in measures)
    assert all("reconstruct_seconds" in row for row in measures)
    assert all("shamanskii_refine_passes" in row for row in measures)
    assert all(
        float(row["process_wall_seconds"]) >= float(row["walk_seconds"])
        for row in measures
    )
    assert all(
        float(row["jax_process_wall_seconds"]) >= float(row["walk_seconds"])
        for row in measures
    )
    assert all(
        float(row["jax_process_wall_seconds"])
        == pytest.approx(
            float(row["process_elapsed_seconds"])
            - float(row["reconstruct_seconds"])
            - float(row["native_rejudge_seconds"]),
            abs=1.0e-6,
        )
        for row in measures
    )
    assert all(
        float(row["jax_floor_seconds"])
        == pytest.approx(
            float(row["jax_process_wall_seconds"]) - float(row["walk_seconds"]),
            abs=1.0e-6,
        )
        for row in measures
    )
    assert payload["claim_boundary"]["one_lane_per_process"] is True
    assert payload["claim_boundary"]["repeats"] == NESTED_LS_GATE6_CLAIM_REPEATS
    assert (
        payload["claim_boundary"]["jax_claim_clock"]
        == "parent_wait_minus_reconstruct_rejudge"
    )
    cache_only = [row for row in measures if row["lane"] == "cache_only"]
    lag_only = [row for row in measures if row["lane"] == "lag_only"]
    both = [row for row in measures if row["lane"] == "both"]
    assert cache_only and lag_only and both
    primes = [row for row in payload["rows"] if row["role"] == "prime"]
    assert {str(row["lane"]) for row in primes} == {"cache_only", "both"}
    assert all(row["linear_solver"] == "dense_lu" for row in cache_only)
    assert all(row["linear_solver"] == "shamanskii" for row in lag_only)
    assert all(row["linear_solver"] == "shamanskii" for row in both)
    assert all(row["disable_cache"] is False for row in cache_only)
    assert all(row["disable_cache"] is True for row in lag_only)
    assert all(row["disable_cache"] is False for row in both)
    assert all(row["cache_dir"] is not None for row in cache_only)
    assert all(row["cache_dir"] is None for row in lag_only)
    assert all(row["cache_dir"] is not None for row in both)
    assert all(row["shamanskii_reused_steps"] == [] for row in cache_only)
    assert all(
        row["shamanskii_reused_steps"] or row["shamanskii_reassembled_steps"]
        for row in lag_only
    )
    assert all(
        row["shamanskii_reused_steps"] or row["shamanskii_reassembled_steps"]
        for row in both
    )
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _F3_B37_GPU_JAX_FLOOR_EVIDENCE.is_file(),
    reason="authored JAX process-wall floor JSON not yet produced",
)
def test_authored_gpu_jax_floor_json_is_not_a_speed_claim():
    raw = _F3_B37_GPU_JAX_FLOOR_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["written_by_pytest"] is False
    floors = [row for row in payload["rows"] if row["role"] == "floor"]
    assert {str(row["lane"]) for row in floors} == {
        "floor_cache_on",
        "floor_cache_off",
    }
    assert all(row["linear_solver"] == "floor" for row in floors)
    assert all(bool(row["success"]) for row in floors)
    assert all(float(row["walk_seconds"]) == 0.0 for row in floors)
    assert all(float(row["reconstruct_seconds"]) == 0.0 for row in floors)
    assert all("jax_floor_seconds" in row for row in floors)
    assert all(float(row["jax_floor_seconds"]) > 0.0 for row in floors)
    assert all(
        float(row["jax_floor_seconds"])
        == pytest.approx(float(row["process_elapsed_seconds"]), abs=1.0e-6)
        for row in floors
    )
    cache_on = [row for row in floors if row["lane"] == "floor_cache_on"]
    cache_off = [row for row in floors if row["lane"] == "floor_cache_off"]
    assert cache_on and cache_off
    assert all(row["disable_cache"] is False for row in cache_on)
    assert all(row["disable_cache"] is True for row in cache_off)
    assert payload["claim_boundary"]["nested_speed_claim"] is False
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _F3_B37_GPU_GATE6_EVIDENCE.is_file(),
    reason="authored Gate-6 claim JSON not yet produced",
)
def test_authored_gpu_gate6_json_matches_frozen_contract():
    raw = _F3_B37_GPU_GATE6_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-gate6.v1"
    assert payload["written_by_pytest"] is False
    assert payload["publication"] == _GPU_GATE6_PUBLICATION
    boundary = payload["claim_boundary"]
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["comparable_operators"] is False
    assert boundary["aggregation"] == NESTED_LS_GATE6_AGGREGATION
    assert boundary["repeats"] == NESTED_LS_GATE6_CLAIM_REPEATS
    assert boundary["native_omp_num_threads"] == NESTED_LS_GATE6_NATIVE_OMP_THREADS
    assert boundary["jax_linear_solver"] == "shamanskii"
    assert boundary["jax_persistent_cache"] is True
    assert boundary["interleaved_repeats"] is True
    assert boundary["jax_claim_clock"] == "parent_wait_minus_reconstruct_rejudge"
    assert boundary["native_claim_clock"] == "parent_wait"
    pairs = payload["pairs"]
    assert len(pairs) >= NESTED_LS_GATE6_CLAIM_REPEATS
    assert {int(pair["repeat"]) for pair in pairs} >= set(
        range(NESTED_LS_GATE6_CLAIM_REPEATS)
    )
    assert all(pair["physics_ok"] for pair in pairs)
    assert payload["fail_closed_reason"] is None
    jax_claim_walls: list[float] = []
    native_claim_walls: list[float] = []
    for pair in pairs:
        native = pair["native"]
        jax_row = pair["jax"]
        assert (
            int(native["observed_omp_num_threads"])
            == NESTED_LS_GATE6_NATIVE_OMP_THREADS
        )
        assert bool(native["omp_pinned"]) is True
        assert float(native["coil_delta_inf"]) == 0.0
        assert float(jax_row["coil_delta_inf"]) == 0.0
        assert int(jax_row["native_rejudge_iter"]) == 0
        assert float(jax_row["grad_l2"]) <= NESTED_LS_NEWTON_TOL
        assert (
            abs(float(jax_row["iota"]) - float(native["iota"]))
            <= NESTED_LS_GATE6_IOTA_G_TOL
        )
        assert (
            abs(float(jax_row["G"]) - float(native["G"])) <= NESTED_LS_GATE6_IOTA_G_TOL
        )
        native_claim = float(native["claim_wall_seconds"])
        jax_claim = float(jax_row["claim_wall_seconds"])
        native_claim_walls.append(native_claim)
        jax_claim_walls.append(jax_claim)
        assert native_claim == pytest.approx(
            float(native["process_wall_seconds"]), abs=1.0e-9
        )
        assert jax_claim == pytest.approx(
            float(jax_row["process_wall_seconds"])
            - float(jax_row["reconstruct_seconds"])
            - float(jax_row["native_rejudge_seconds"]),
            abs=1.0e-6,
        )
    assert payload["native_min_process_wall_seconds"] == min(native_claim_walls)
    assert payload["jax_min_process_wall_seconds"] == min(jax_claim_walls)
    expected_claim = payload["fail_closed_reason"] is None and min(
        jax_claim_walls
    ) < min(native_claim_walls)
    assert boundary["nested_speed_claim"] is expected_claim
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _A100_BANANA_OMP_EVIDENCE.is_file(),
    reason="authored A100 banana OMP JSON not yet produced",
)
def test_authored_a100_banana_omp_json_is_host_best_of_contract():
    raw = _A100_BANANA_OMP_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-a100-banana-omp.v1"
    assert payload["written_by_pytest"] is False
    boundary = payload["claim_boundary"]
    assert boundary["nested_speed_claim"] is False
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["host"] == "landau"
    assert boundary["omp_pinned"] is True
    assert payload["threads"] == list(F3_B37_BANANA_OMP_CONTRACT_THREADS)
    rows = payload["probe"]["rows"]
    assert {int(row["omp_num_threads"]) for row in rows} == set(
        F3_B37_BANANA_OMP_CONTRACT_THREADS
    )
    assert all(bool(row["omp_pinned"]) for row in rows)
    assert all(bool(row["success"]) for row in rows)
    best = int(payload["best_omp_num_threads"])
    assert best in F3_B37_BANANA_OMP_CONTRACT_THREADS
    best_inner = min(float(row["inner_solver_seconds"]) for row in rows)
    assert float(payload["best_inner_solver_seconds"]) == pytest.approx(best_inner)
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@pytest.mark.skipif(
    not _A100_GATE6_EVIDENCE.is_file(),
    reason="authored A100 Gate-6 JSON not yet produced",
)
def test_authored_a100_gate6_json_uses_host_best_omp():
    raw = _A100_GATE6_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gpu-gate6.v1"
    assert payload["written_by_pytest"] is False
    boundary = payload["claim_boundary"]
    assert boundary["inherits_f3_7_70x"] is False
    assert boundary["tag"] == "a100"
    assert boundary["repeats"] == NESTED_LS_GATE6_CLAIM_REPEATS
    omp = int(boundary["native_omp_num_threads"])
    if _A100_BANANA_OMP_EVIDENCE.is_file():
        omp_payload = json.loads(_A100_BANANA_OMP_EVIDENCE.read_text(encoding="utf-8"))
        assert omp == int(omp_payload["best_omp_num_threads"])
    pairs = payload["pairs"]
    assert len(pairs) >= NESTED_LS_GATE6_CLAIM_REPEATS
    assert all(pair["physics_ok"] for pair in pairs)
    assert payload["fail_closed_reason"] is None
    assert not _F3_B37_GPU_WALK_EVIDENCE.is_file()


@_REQUIRES_BUNDLE
@_REQUIRES_F3_B37
@pytest.mark.slow
@pytest.mark.skipif(
    jax.default_backend() == "gpu",
    reason="CPU frozen bounded packet; GPU one-step is a separate node",
)
def test_f3_gpu_b37_bounded_hvp_and_native_reference():
    coils, surface, meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    _assert_grid(native)
    probe = evaluate_f3_b37_bounded_probe(
        native,
        jax_boozer,
        one_newton_step=False,
        compare_schur_hvp=True,
    )
    dump_strict_json({"probe": probe.as_payload()})
    frozen = json.loads(_F3_B37_BOUNDED_EVIDENCE.read_text(encoding="utf-8"))
    frozen_probe = frozen["probe"]
    value_tol = parity_ladder_tolerances("direct_kernel")
    assert probe.residual_rows == 3 * 255 * 64 + 2
    assert probe.y_rank == 2
    assert probe.reduced_grad_finite is True
    assert probe.reduced_grad_l2 > 1.0e-6
    assert probe.hvp_finite is True
    assert probe.hvp_seconds > 0.0
    assert probe.one_step_attempted is False
    assert probe.full_walk_attempted is False
    assert probe.native_ref_success is True
    assert probe.native_ref_iter >= 1
    assert abs(probe.native_ref_delta_iota) > 1.0e-3
    assert probe.native_ref_coil_delta_inf == 0.0
    assert probe.native_rejudge_iota is None
    assert probe.native_rejudge_g is None
    assert probe.schur_hvp_finite is True
    assert probe.schur_vs_ad_rel_l2 is not None
    assert probe.schur_vs_ad_max_abs is not None
    assert probe.schur_vs_ad_rel_l2 < 1.0e-12
    assert probe.schur_vs_ad_max_abs < 1.0e-12
    assert jax.default_backend() == "cpu"
    np.testing.assert_allclose(
        probe.y_star_iota,
        frozen_probe["y_star_iota"],
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        probe.native_ref_delta_iota,
        frozen_probe["native_ref_delta_iota"],
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    lane_inner = meta["inner_state"]
    assert lane_inner is not None
    assert float64_ulps(probe.y_star_iota, float(lane_inner[0])) == pytest.approx(
        19.0, abs=0.5
    )
    assert float64_ulps(probe.y_star_g, float(lane_inner[1])) == pytest.approx(
        7.0, abs=0.5
    )


@_REQUIRES_BUNDLE
@_REQUIRES_F3_B37
@pytest.mark.slow
@pytest.mark.skipif(
    jax.default_backend() == "gpu",
    reason="CPU SciPy-packet live rejudge; GPU one-step is a separate node",
)
def test_f3_b37_one_schur_newton_step_and_cpp_rejudge():
    coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    _assert_grid(native)
    probe = evaluate_f3_b37_schur_newton_step(native, jax_boozer)
    dump_strict_json(probe.as_payload())
    frozen = json.loads(_F3_B37_SCHUR_ONE_STEP_EVIDENCE.read_text(encoding="utf-8"))
    frozen_probe = frozen["probe"]
    replay = frozen["independent_replay"]
    assert probe.gmres_info in (0, -1)
    assert probe.gmres_restart == 8
    assert probe.gmres_maxiter >= 1
    assert probe.gmres_maxiter <= 8
    if probe.step_accepted:
        assert probe.gmres_forcing_eta <= probe.gmres_rtol
    assert probe.step_coil_delta_inf == 0.0
    assert probe.native_rejudge_coil_delta_inf == 0.0
    assert probe.runtime["jax_default_backend"] == "cpu"
    assert probe.step_accepted is True
    assert probe.step_iter == 1
    assert probe.step_success is False
    assert probe.native_rejudge_iter == 10
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        probe.y_star_iota,
        frozen_probe["y_star_iota"],
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        probe.native_rejudge_iota,
        frozen_probe["native_rejudge_iota"],
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="C++ rejudge of the Schur step left the reconstruct iota branch",
    )
    np.testing.assert_allclose(
        probe.native_rejudge_iota,
        _RECONSTRUCT_IOTA,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    assert probe.native_rejudge_success is True
    assert probe.native_rejudge_grad_l2 < 1.0e-13
    assert probe.rejudge_vs_reconstruct_surface_inf is not None
    assert probe.rejudge_vs_reconstruct_surface_inf < 1.0e-9
    np.testing.assert_allclose(
        probe.rejudge_vs_reconstruct_surface_inf,
        replay["cpp_rejudge_branch"]["surface_inf_vs_reconstruct"],
        rtol=1.0e-2,
        atol=1.0e-11,
    )
    assert probe.schur_vs_ad_rel_l2 is not None
    assert probe.schur_vs_ad_rel_l2 < 1.0e-12
    assert probe.schur_vs_ad_max_abs is not None
    assert probe.schur_vs_ad_max_abs < 1.0e-12
    assert probe.provenance["input_sha256"]["pair2-l1_lane.json"] == _PAIR2_L1_SHA256
    assert (
        probe.provenance["input_sha256"]["native_biot_savart.json"]
        == _NATIVE_BIOT_SHA256
    )


def test_b37_nested_timing_refuses_before_b3():
    with pytest.raises(NestedLsB37TimingBlocked, match="B3 banana run_code"):
        evaluate_f3_b37_nested_timing(
            native=None,  # type: ignore[arg-type]
            jax_boozer=None,  # type: ignore[arg-type]
            b3_matched=False,
        )


def test_replace_native_solver_options_restores_original_identity():
    reconstruct = {
        "newton_tol": 1.0e-13,
        "newton_maxiter": 10,
        "bfgs_tol": 1.0e-10,
    }

    class _Native:
        def __init__(self):
            self.options = reconstruct

    native = _Native()
    original = replace_native_solver_options(native, {"newton_tol": 1.0e-11})
    assert original is reconstruct
    assert native.options is not reconstruct
    assert native.options["newton_tol"] == 1.0e-11
    assert native.options["newton_maxiter"] == 10
    native.options = original
    assert native.options is reconstruct
    assert native.options["newton_tol"] == 1.0e-13


@_REQUIRES_BUNDLE
@_REQUIRES_F3_B37
@pytest.mark.slow
@pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="F3 B37 GPU one-step requires jax.default_backend() == 'gpu'",
)
def test_f3_b37_gpu_one_schur_newton_step_and_cpp_rejudge():
    coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    _assert_grid(native)
    probe = evaluate_f3_b37_schur_newton_step(native, jax_boozer)
    dump_strict_json(probe.as_payload())
    assert probe.runtime["jax_default_backend"] == "gpu"
    assert probe.gmres_info in (0, -1)
    assert probe.step_coil_delta_inf == 0.0
    assert probe.native_rejudge_coil_delta_inf == 0.0
    assert probe.step_accepted is True
    assert probe.gmres_forcing_eta <= probe.gmres_rtol
    assert probe.step_iter == 1
    assert probe.native_rejudge_success is True
    assert probe.native_rejudge_iter == 10
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        probe.native_rejudge_iota,
        _RECONSTRUCT_IOTA,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="GPU Schur step left the reconstruct iota branch",
    )
    assert probe.native_rejudge_grad_l2 < 1.0e-13
    assert probe.rejudge_vs_reconstruct_surface_inf is not None
    assert probe.rejudge_vs_reconstruct_surface_inf < 1.0e-9
    assert probe.schur_vs_ad_rel_l2 is not None
    assert probe.schur_vs_ad_rel_l2 < 1.0e-12


@_REQUIRES_BUNDLE
@_REQUIRES_F3_B37
@pytest.mark.slow
def test_f3_b37_flat_native_probe_is_off_manifold():
    coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_NATIVE_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    _assert_grid(native)
    probe = evaluate_f3_b37_flat_native_probe(native, jax_boozer)
    dump_strict_json(probe.as_payload())
    assert probe.residual_rows == 3 * 255 * 64 + 2
    assert probe.y_rank == 2
    assert probe.reduced_grad_l2 > 1.0e-6
    assert probe.native_ref_success is True
    assert probe.native_ref_coil_delta_inf == 0.0
    np.testing.assert_allclose(
        probe.native_ref_iota,
        _RECONSTRUCT_IOTA,
        rtol=1.0e-8,
        atol=1.0e-10,
        err_msg="flat-native B37 C++ reconstruct left the reconstruct iota branch",
    )


@_REQUIRES_BUNDLE
@_REQUIRES_F3_B37
@pytest.mark.slow
@pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="F3 B37 ten-step walk requires jax.default_backend() == 'gpu'",
)
def test_f3_b37_schur_newton_walk_and_cpp_rejudge():
    coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    _assert_grid(native)
    probe = evaluate_f3_b37_schur_newton_walk(native, jax_boozer, maxiter=10)
    dump_strict_json(probe.as_payload())
    assert probe.runtime["jax_default_backend"] == "gpu"
    assert probe.coil_delta_inf == 0.0
    assert probe.native_rejudge_coil_delta_inf == 0.0
    assert probe.success is True
    assert probe.grad_l2 <= NESTED_LS_NEWTON_TOL
    assert probe.steps
    assert all(bool(step["step_accepted"]) for step in probe.steps)
    assert all(
        float(step["gmres_forcing_eta"]) <= float(step["gmres_rtol"])
        for step in probe.steps
        if bool(step["step_accepted"])
    )
    assert probe.native_rejudge_success is True
    assert probe.native_rejudge_iter == 0
    assert probe.rejudge_vs_jax_surface_inf == 0.0
    assert probe.rejudge_vs_jax_iota == pytest.approx(0.0, abs=1.0e-15)
    assert probe.rejudge_vs_jax_g == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(
        probe.native_rejudge_iota,
        probe.jax_iota,
        rtol=0.0,
        atol=1.0e-15,
        err_msg="C++ rejudge moved iota away from the JAX endpoint",
    )
    np.testing.assert_allclose(
        probe.native_rejudge_g,
        probe.jax_g,
        rtol=0.0,
        atol=1.0e-15,
        err_msg="C++ rejudge moved G away from the JAX endpoint",
    )
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        probe.native_rejudge_iota,
        probe.reconstruct_ref_iota,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="ten-step walk left the reconstruct iota branch",
    )
    np.testing.assert_allclose(
        probe.native_rejudge_iota,
        _RECONSTRUCT_IOTA,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    assert probe.rejudge_vs_reconstruct_surface_inf < 1.0e-9
