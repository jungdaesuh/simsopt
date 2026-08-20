"""Gate 1 start: reduced nested-LS at archived 255x64 F3 geometry.

Skipped unless the host-local genuine-675 bundle is present. Marked slow
because the LS residual is 3*255*64+2 = 48962 rows. Not an F3 timing claim.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    evaluate_f3_b37_bounded_probe,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
)

_F3_B37_BOUNDED_EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "receipts"
    / "evidence"
    / "nested_ls_reduced_gate1_f3_b37_bounded_20260820.json"
)

pytestmark = [
    pytest.mark.boozer,
    pytest.mark.slow,
    pytest.mark.skipif(
        not archived_flat675_bundle_available(),
        reason="the frozen genuine-675 input bundle is host-local",
    ),
]


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


@pytest.mark.skipif(
    not archived_f3_b37_lanes_available(),
    reason="the host-local F3 B37 pair2-l1 lane JSON is missing",
)
def test_f3_gpu_b37_bounded_hvp_and_native_reference():
    coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    _assert_grid(native)
    probe = evaluate_f3_b37_bounded_probe(native, jax_boozer, one_newton_step=False)
    _F3_B37_BOUNDED_EVIDENCE.write_text(
        json.dumps(
            {
                "schema": "nested-ls-reduced-gate1-f3-b37-bounded.v1",
                "one_newton_step": False,
                "full_walk_attempted": probe.full_walk_attempted,
                "probe": probe.as_payload(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
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
