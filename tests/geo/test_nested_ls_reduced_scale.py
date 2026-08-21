"""Gate 1 start: reduced nested-LS at archived 255x64 F3 geometry.

Skipped unless the host-local genuine-675 bundle is present. Marked slow
because the LS residual is 3*255*64+2 = 48962 rows. Not an F3 timing claim.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_NEWTON_TOL,
    nested_ls_physics_newton_kwargs,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    NestedLsB37TimingBlocked,
    nested_ls_reduced_closures,
    pack_surface_and_y,
    require_full_y_rank,
    run_reduced_nested_ls_newton,
    solve_projected_y,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    ARCHIVED_START_QR_G,
    ARCHIVED_START_QR_IOTA,
    DEFAULT_F3_B37_GPU_LANE,
    DEFAULT_F3_B37_NATIVE_LANE,
    archived_f3_b37_lanes_available,
    archived_flat675_bundle_available,
    dump_strict_json,
    evaluate_f3_b37_bounded_probe,
    evaluate_f3_b37_flat_native_probe,
    evaluate_f3_b37_nested_timing,
    evaluate_f3_b37_schur_newton_step,
    evaluate_f3_b37_schur_newton_walk,
    float64_ulps,
    kib_to_gib,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
    nested_ls_receipt_provenance,
    replace_native_solver_options,
    sha256_file,
)

_EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "receipts" / "evidence"
_F3_B37_BOUNDED_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gate1_f3_b37_bounded_20260820.json"
)
_F3_B37_SCHUR_ONE_STEP_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gate1_f3_b37_schur_one_step_20260820.json"
)
_F3_B37_GPU_WALK_EVIDENCE = _EVIDENCE_DIR / "nested_ls_reduced_gpu_walk_20260821.json"
_GPU_WALK_PUBLICATION = (
    "GPU forcing-certified Schur walk. Not a timing claim and not F3 7.70x."
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
    assert float(probe["coil_delta_inf"]) == 0.0
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
    assert payload["gmres_matvecs_note"] == (
        "JAX incremental GMRES does not report operator applications; "
        "gmres_matvecs stays 0."
    )


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
