"""Gate 1 start: reduced nested-LS at archived 255x64 F3 geometry.

Skipped unless the host-local genuine-675 bundle is present. Marked slow
because the LS residual is 3*255*64+2 = 48962 rows. Not an F3 timing claim.
"""

from __future__ import annotations

import ast
import inspect
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
    last_step_meets_forcing,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
    nested_ls_omp_threads_pinned,
    nested_ls_receipt_provenance,
    predict_start_at_cap_wall_seconds,
    replace_native_solver_options,
    unpreconditioned_gmres_is_insufficient,
)

_RECONSTRUCT_IOTA = 0.14085710955307942
_PAIR2_L1_SHA256 = "bde32ab9987d4f2116cf7c7410753c83a9d74ca7836031c25db0e12603155d64"
_NATIVE_BIOT_SHA256 = "0415ae937c78b9f2d68e8463a9176e8f330a9aa172eece160341afccdc29429d"
# Frozen F3 B37 probe numbers used as live-test oracles.
_F3_B37_Y_STAR_IOTA = 0.15164961478467412
_F3_B37_NATIVE_REF_DELTA_IOTA = -0.010792505231594696
_F3_B37_SCHUR_REJUDGE_IOTA = 0.1408571095660965
_F3_B37_SCHUR_REJUDGE_SURFACE_INF = 4.157e-12

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


def test_last_step_meets_forcing_uses_the_recorded_last_step():
    """Rejected last steps count; accepted-only filtering would hide a miss."""
    accepted = {
        "step_accepted": True,
        "gmres_forcing_eta": 0.05,
        "gmres_rtol": 0.10,
    }
    rejected_miss = {
        "step_accepted": False,
        "gmres_forcing_eta": 0.09,
        "gmres_rtol": 0.03,
    }
    assert last_step_meets_forcing([]) is False
    assert last_step_meets_forcing([accepted]) is True
    assert last_step_meets_forcing([accepted, rejected_miss]) is False
    assert last_step_meets_forcing([rejected_miss]) is False


def test_receipt_provenance_is_strict_json():
    payload = nested_ls_receipt_provenance()
    dump_strict_json(payload)
    assert payload["git_commit"]
    source = payload["source_sha256"]
    assert isinstance(source, dict)
    assert "nested_ls_reduced.py" in source


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
        _F3_B37_Y_STAR_IOTA,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        probe.native_ref_delta_iota,
        _F3_B37_NATIVE_REF_DELTA_IOTA,
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
        _F3_B37_Y_STAR_IOTA,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        probe.native_rejudge_iota,
        _F3_B37_SCHUR_REJUDGE_IOTA,
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
        _F3_B37_SCHUR_REJUDGE_SURFACE_INF,
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
