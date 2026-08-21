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
    archived_f3_b37_lanes_available,
    archived_flat675_bundle_available,
    dump_strict_json,
    evaluate_f3_b37_bounded_probe,
    evaluate_f3_b37_schur_newton_step,
    float64_ulps,
    kib_to_gib,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
)

_EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "receipts" / "evidence"
_F3_B37_BOUNDED_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gate1_f3_b37_bounded_20260820.json"
)
_F3_B37_SCHUR_ONE_STEP_EVIDENCE = (
    _EVIDENCE_DIR / "nested_ls_reduced_gate1_f3_b37_schur_one_step_20260820.json"
)
_RECONSTRUCT_IOTA = 0.14085710955307942
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


def test_f3_b37_schur_one_step_evidence_is_strict_authored_json():
    raw = _F3_B37_SCHUR_ONE_STEP_EVIDENCE.read_text(encoding="utf-8")
    assert "NaN" not in raw
    payload = json.loads(raw)
    dump_strict_json(payload)
    assert payload["schema"] == "nested-ls-reduced-gate1-f3-b37-schur-one-step.v1"
    assert payload["written_by_pytest"] is False
    assert payload["full_walk_attempted"] is False
    assert payload["claim_boundary"]["nested_speed_claim"] is False
    assert payload["claim_boundary"]["newton_quality_linear_solve"] is False
    assert payload["driver"] == (
        "simsopt_jax_adapters.geo.nested_ls_reduced_scale."
        "evaluate_f3_b37_schur_newton_step"
    )
    assert payload["gmres"]["info"] == 1
    assert payload["gmres"]["restart"] == 8
    assert payload["gmres"]["maxiter"] == 1
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
    assert "fully solved Newton step" in payload["publication"]


@_REQUIRES_BUNDLE
@_REQUIRES_F3_B37
@pytest.mark.slow
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
    derivative_tol = parity_ladder_tolerances("derivative_heavy")
    assert probe.schur_vs_ad_rel_l2 <= float(derivative_tol["second_derivative_rtol"])
    assert probe.schur_vs_ad_max_abs <= 1.0e-3
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
def test_f3_b37_one_schur_newton_step_and_cpp_rejudge():
    coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    _assert_grid(native)
    probe = evaluate_f3_b37_schur_newton_step(native, jax_boozer)
    dump_strict_json(probe.as_payload())
    assert probe.gmres_matvecs >= 1
    assert probe.gmres_matvecs <= probe.gmres_restart * probe.gmres_maxiter + 2
    assert probe.gmres_restart == 8
    assert probe.gmres_maxiter == 1
    assert probe.step_coil_delta_inf == 0.0
    assert probe.native_rejudge_coil_delta_inf == 0.0
    assert probe.runtime["jax_default_backend"] == "cpu"
    assert probe.step_accepted is True
    assert probe.step_iter == 1
    assert probe.step_grad_l2 < probe.reduced_grad_l2_before
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        probe.native_rejudge_iota,
        _RECONSTRUCT_IOTA,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="C++ rejudge of the Schur step left the reconstruct iota branch",
    )
    assert probe.native_rejudge_success is True
