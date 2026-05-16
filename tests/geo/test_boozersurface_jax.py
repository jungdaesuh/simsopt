"""
Tests for the JAX Boozer surface solver (Milestone 4).

Validates:
1. Stellsym DOF scatter/gather round-trip.
2. Volume computation against analytical formula.
3. Composed penalty objective value and gradient.
4. BFGS convergence on a synthetic problem.
5. Newton polish convergence.
6. Exact Newton path convergence.
7. Vector potential A correctness.
"""

import inspect
import logging
import sys
import types
from contextlib import contextmanager
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt.geo.optimizer_jax_reference as _opt_ref
import simsopt.jax_core.biotsavart as _biotsavart_jax_core
import scipy.linalg
from jax.flatten_util import ravel_pytree
from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from conftest import (
    assert_array_on_device,
    assert_arrays_on_device,
    enable_non_strict_jax_backend,
    enable_strict_jax_backend,
    parity_device,
)
from simsopt.field.coil import Coil, Current
from simsopt.geo.boozersurface import BoozerSurface as LegacyBoozerSurface
from simsopt.geo._boozersurface_current_guard import (
    guard_none_G_coil_gradient_callback,
)
from simsopt.jax_core import (
    GroupedCoilSetSpec,
    grouped_coil_set_spec_from_lists,
    grouped_field_data_from_spec,
    make_current_value_spec,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.objectives.utilities import forward_backward

from .boozersurface_jax_test_helpers import (
    BoozerSurfaceJAX,
    _MockBiotSavart,
    _MockCoil,
    _MockSurface,
    _PlumbingVolumeLabel,
    _boozer_exact_coil_vjp,
    _bsj,
    _build_penalty_problem,
    _build_upstream_boozer_exact_constraints_case,
    _ensure_solved_jax,
    _build_upstream_boozer_immutable_inputs,
    _build_upstream_boozer_penalty_case,
    _evaluate_upstream_boozer_exact_constraints_case,
    _evaluate_upstream_boozer_penalty_case,
    _evaluate_upstream_boozer_penalty_hessian_case,
    _build_upstream_exact_surface_case,
    _extract_upstream_jax_penalty_inputs,
    _make_circular_coil,
    _make_mixed_quad_mock_coils,
    _make_mock_boozer_surface,
    _make_mock_coils,
    _make_simple_torus_coeffs,
    _opt,
    _patch_newton_polish_runner,
    _simple_torus_geometry_values,
    _successful_minimize_result,
    _successful_newton_polish_result,
    biot_savart_A,
    biot_savart_B,
    biot_savart_dA_by_dX,
    compute_G_from_currents,
    dofs_to_xyzc,
    jax_minimize,
    jax_least_squares,
    newton_exact,
    newton_polish,
    require_target_backend_x64,
    resolve_least_squares_optimizer_method,
    resolve_optimizer_backend_method,
    stellsym_scatter_indices,
    UPSTREAM_BOOZER_OPTIMIZE_G,
    UPSTREAM_BOOZER_STELLSYM,
    UPSTREAM_BOOZER_SURFACE_TYPES,
)


_TORUS_GEOMETRY_RTOL = 1e-13
_ROSENBROCK_SOLUTION_ATOL = 1e-8
_STOKES_FLUX_RTOL = 1e-5
_STOKES_FLUX_ATOL = 5e-7
_STOKES_DISK_NR = 96
_STOKES_DISK_NTHETA = 192

_PUBLIC_LBFGS_RESULT_SCHEMA = _bsj._BOOZER_RESULT_SCHEMAS["lbfgs"]
_PUBLIC_LS_MANUAL_RESULT_SCHEMA = _bsj._BOOZER_RESULT_SCHEMAS["ls_manual"]
_PUBLIC_LS_LM_RESULT_SCHEMA = _bsj._BOOZER_RESULT_SCHEMAS["ls_lm"]
_PUBLIC_NEWTON_RESULT_SCHEMA = _bsj._BOOZER_RESULT_SCHEMAS["newton"]
_PUBLIC_EXACT_RESULT_SCHEMA = _bsj._BOOZER_RESULT_SCHEMAS["exact"]
_PUBLIC_EXACT_CONSTRAINTS_RESULT_SCHEMA = _bsj._BOOZER_RESULT_SCHEMAS[
    "exact_constraints"
]
_TRACEABLE_EXACT_RESULT_SCHEMA = _bsj._BOOZER_RESULT_SCHEMAS["traceable_exact"]
_TRACEABLE_LS_RESULT_SCHEMA = _bsj._BOOZER_RESULT_SCHEMAS["traceable_ls"]
_SOLVED_RUNTIME_STATE_FIELDS = frozenset({"sdofs", "iota", "G", "weight_inv_modB"})
_ADJOINT_RUNTIME_STATE_FIELDS = frozenset(
    {
        "solved_state",
        "linearization_kind",
        "decision_size",
        "dtype",
        "apply_forward",
        "apply_transpose",
        "solve_forward",
        "solve_transpose",
        "solve_forward_with_status",
        "solve_transpose_with_status",
        "stream_group_vjps",
        "linear_solve_backend",
        "dense_linear_solve_factors_available",
        "linear_solve_factors",
    }
)


def _assert_result_schema(result, schema):
    keys = set(result.keys())
    assert schema.required_keys <= keys
    assert not schema.forbidden_keys & keys


def _runtime_sdofs_for(booz):
    return jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)


def test_public_solver_result_schema_registry_is_mode_aware():
    entrypoint_cases = {
        "lbfgs": "minimize_boozer_penalty_constraints_LBFGS",
        "ls_manual": "minimize_boozer_penalty_constraints_ls",
        "ls_lm": "minimize_boozer_penalty_constraints_ls",
        "newton": "minimize_boozer_penalty_constraints_newton",
        "exact": "solve_residual_equation_exactly_newton",
        "exact_constraints": "minimize_boozer_exact_constraints_newton",
        "traceable": "run_code_traceable",
    }
    schemas = _bsj._BOOZER_RESULT_SCHEMAS

    assert entrypoint_cases.keys() <= schemas.keys()
    for schema_name, entrypoint_name in entrypoint_cases.items():
        assert hasattr(BoozerSurfaceJAX, entrypoint_name)
        schema = schemas[schema_name]
        assert schema.required_keys
        assert not schema.required_keys & schema.forbidden_keys

    assert "info" in schemas["ls_lm"].required_keys
    assert "info" in schemas["ls_manual"].forbidden_keys
    assert "PLU" in schemas["exact"].required_keys
    assert "PLU" in schemas["traceable"].forbidden_keys
    assert "plu" in schemas["traceable"].required_keys
    assert "plu" in schemas["exact"].forbidden_keys
    assert "linearization_kind" in schemas["newton"].required_keys
    assert "linearization_kind" in schemas["lbfgs"].forbidden_keys
    for schema_name in ("lbfgs", "ls_manual", "ls_lm", "newton", "traceable_ls"):
        assert set(_bsj.SOLVE_QUALITY_LS_FIELDS) <= schemas[schema_name].required_keys
    for schema_name in ("exact", "exact_constraints", "traceable_exact"):
        assert (
            set(_bsj.SOLVE_QUALITY_EXACT_FIELDS) <= schemas[schema_name].required_keys
        )


def _assert_runtime_state_schema(runtime_state, required_fields):
    assert required_fields <= set(runtime_state.__dataclass_fields__)


def _assert_operator_adjoint_state(adjoint_state, *, dense_factors_available):
    assert adjoint_state.linear_solve_backend == "operator"
    assert adjoint_state.linear_solve_factors is None
    assert adjoint_state.plu is None
    assert adjoint_state.dense_linear_solve_factors_available is dense_factors_available


def _assert_dense_plu_adjoint_state(adjoint_state):
    assert adjoint_state.linear_solve_backend == "dense-plu"
    assert adjoint_state.linear_solve_factors is not None
    assert adjoint_state.plu is adjoint_state.linear_solve_factors
    assert adjoint_state.dense_linear_solve_factors_available is True


def _collect_exact_well_conditioned_runtime_metadata(device):
    metadata = {
        "jax_version": str(jax.__version__),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "selected_device": str(device),
        "platform": str(device.platform),
        "device_kind": str(device.device_kind),
        "platform_version": str(
            getattr(getattr(device, "client", None), "platform_version", "")
        ),
    }
    assert metadata["jax_version"]
    assert metadata["jax_enable_x64"] is True
    assert metadata["selected_device"]
    assert metadata["platform"] in {"cpu", "gpu"}
    assert metadata["device_kind"]
    if metadata["platform"] == "gpu":
        assert metadata["platform_version"]
    return metadata


def _build_exact_well_conditioned_operator_fixture(monkeypatch, *, device):
    metadata = _collect_exact_well_conditioned_runtime_metadata(device)
    with jax.default_device(device):
        booz = _make_mock_boozer_surface_exact(options={"newton_tol": 1e-12})
        booz.need_to_run_code = False
        solved_x = booz._pack_decision_vector(0.3, 0.05)
        n = int(solved_x.size)

        diagonal = np.linspace(2.0, 2.5, n)
        A_np = np.diag(diagonal)
        A_np += 0.01 * np.tril(np.ones((n, n)), k=-1)
        A_np += 0.005 * np.triu(np.ones((n, n)), k=1)
        A = jnp.asarray(A_np, dtype=jnp.float64)
        rhs_np = np.linspace(-0.4, 0.6, n)
        matrix_rhs_np = np.column_stack(
            (
                rhs_np,
                np.sin(np.linspace(0.0, np.pi, n)),
                np.cos(np.linspace(0.0, np.pi, n)),
            )
        )
        rhs = jnp.asarray(rhs_np, dtype=jnp.float64)
        matrix_rhs = jnp.asarray(matrix_rhs_np, dtype=jnp.float64)
        P_lu, L_lu, U_lu = scipy.linalg.lu(A_np)
        gradient_projection = np.vstack(
            (
                np.ones(n),
                np.linspace(-1.0, 1.0, n),
                np.cos(np.linspace(0.0, np.pi, n)),
            )
        )

        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "exact_jacobian",
            "PLU": tuple(
                jnp.asarray(piece, dtype=jnp.float64) for piece in (P_lu, L_lu, U_lu)
            ),
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

    def matrix_residual(_self, _mask):
        return lambda x: A @ x

    monkeypatch.setattr(
        _bsj.BoozerSurfaceJAX,
        "_compute_stellsym_mask_indices",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        _bsj.BoozerSurfaceJAX,
        "_make_exact_residual",
        matrix_residual,
    )

    return {
        "A": A,
        "A_np": A_np,
        "P_lu": P_lu,
        "L_lu": L_lu,
        "U_lu": U_lu,
        "booz": booz,
        "device": device,
        "gradient_projection": gradient_projection,
        "metadata": metadata,
        "matrix_rhs": matrix_rhs,
        "matrix_rhs_np": matrix_rhs_np,
        "rhs": rhs,
        "rhs_np": rhs_np,
    }


def _solve_exact_well_conditioned_operator_case(monkeypatch, *, device):
    case = _build_exact_well_conditioned_operator_fixture(
        monkeypatch,
        device=device,
    )
    with jax.default_device(device):
        adjoint_state = case["booz"].get_adjoint_runtime_state()
        operator_adj, success = adjoint_state.solve_transpose_with_status(case["rhs"])
        matrix_operator_adj, matrix_success = adjoint_state.solve_transpose_with_status(
            case["matrix_rhs"]
        )
        matrix_applied = adjoint_state.apply_transpose(matrix_operator_adj)
        jax_dense_adj = jnp.linalg.solve(case["A"].T, case["rhs"])
        matrix_jax_dense_adj = jnp.linalg.solve(case["A"].T, case["matrix_rhs"])
        operator_adj_np = np.asarray(jax.device_get(operator_adj), dtype=float)
        matrix_operator_adj_np = np.asarray(
            jax.device_get(matrix_operator_adj),
            dtype=float,
        )
        jax_dense_adj_np = np.asarray(jax.device_get(jax_dense_adj), dtype=float)
        matrix_jax_dense_adj_np = np.asarray(
            jax.device_get(matrix_jax_dense_adj),
            dtype=float,
        )
        matrix_applied_np = np.asarray(jax.device_get(matrix_applied), dtype=float)
        assert_array_on_device(case["rhs"], device)
        assert_array_on_device(case["matrix_rhs"], device)
        assert_array_on_device(operator_adj, device)
        assert_array_on_device(matrix_operator_adj, device)
        assert_array_on_device(matrix_applied, device)
        assert_array_on_device(jax_dense_adj, device)
        assert_array_on_device(matrix_jax_dense_adj, device)

    plu_adj_np = forward_backward(
        case["P_lu"],
        case["L_lu"],
        case["U_lu"],
        case["rhs_np"],
    )
    matrix_plu_adj_np = forward_backward(
        case["P_lu"],
        case["L_lu"],
        case["U_lu"],
        case["matrix_rhs_np"],
    )
    residual_rel = np.linalg.norm(case["A_np"].T @ operator_adj_np - case["rhs_np"]) / (
        1.0 + np.linalg.norm(case["rhs_np"])
    )
    matrix_residual_rel = np.linalg.norm(
        case["A_np"].T @ matrix_operator_adj_np - case["matrix_rhs_np"]
    ) / (1.0 + np.linalg.norm(case["matrix_rhs_np"]))
    matrix_apply_residual_rel = np.linalg.norm(
        matrix_applied_np - case["matrix_rhs_np"]
    ) / (1.0 + np.linalg.norm(case["matrix_rhs_np"]))
    operator_gradient = case["gradient_projection"] @ operator_adj_np
    dense_gradient = case["gradient_projection"] @ jax_dense_adj_np
    plu_gradient = case["gradient_projection"] @ plu_adj_np
    return {
        **case,
        "adjoint_state": adjoint_state,
        "dense_gradient": dense_gradient,
        "jax_dense_adj_np": jax_dense_adj_np,
        "matrix_jax_dense_adj_np": matrix_jax_dense_adj_np,
        "matrix_apply_residual_rel": matrix_apply_residual_rel,
        "matrix_operator_adj_np": matrix_operator_adj_np,
        "matrix_plu_adj_np": matrix_plu_adj_np,
        "matrix_residual_rel": matrix_residual_rel,
        "matrix_success": bool(np.asarray(matrix_success)),
        "operator_adj_np": operator_adj_np,
        "operator_gradient": operator_gradient,
        "plu_adj_np": plu_adj_np,
        "plu_gradient": plu_gradient,
        "residual_rel": residual_rel,
        "success": bool(np.asarray(success)),
    }


def _assert_exact_well_conditioned_operator_case(case, exact_lane):
    assert case["metadata"]["platform"] == case["device"].platform
    _assert_operator_adjoint_state(
        case["adjoint_state"],
        dense_factors_available=True,
    )
    assert case["success"] is True
    assert case["matrix_success"] is True
    np.testing.assert_allclose(
        case["operator_adj_np"],
        case["jax_dense_adj_np"],
        rtol=exact_lane["adjoint_rtol"],
        atol=exact_lane["adjoint_atol"],
    )
    np.testing.assert_allclose(
        case["operator_adj_np"],
        case["plu_adj_np"],
        rtol=exact_lane["adjoint_rtol"],
        atol=exact_lane["adjoint_atol"],
    )
    np.testing.assert_allclose(
        case["matrix_operator_adj_np"],
        case["matrix_jax_dense_adj_np"],
        rtol=exact_lane["adjoint_rtol"],
        atol=exact_lane["adjoint_atol"],
    )
    np.testing.assert_allclose(
        case["matrix_operator_adj_np"],
        case["matrix_plu_adj_np"],
        rtol=exact_lane["adjoint_rtol"],
        atol=exact_lane["adjoint_atol"],
    )
    assert case["residual_rel"] <= exact_lane["residual_rel_tol"]
    assert case["matrix_residual_rel"] <= exact_lane["residual_rel_tol"]
    assert case["matrix_apply_residual_rel"] <= exact_lane["residual_rel_tol"]
    np.testing.assert_allclose(
        case["operator_gradient"],
        case["dense_gradient"],
        rtol=exact_lane["gradient_rtol"],
        atol=exact_lane["gradient_atol"],
    )
    np.testing.assert_allclose(
        case["operator_gradient"],
        case["plu_gradient"],
        rtol=exact_lane["gradient_rtol"],
        atol=exact_lane["gradient_atol"],
    )


# ---------------------------------------------------------------------------
# Action-level adjoint parity for ill-conditioned exact lane
# ---------------------------------------------------------------------------
#
# The ``exact_ill_conditioned_adjoint`` lane in
# ``benchmarks/validation_ladder_contract.py`` correctly disables raw vector
# parity (``vector_parity_required=False``) because near-singular Jacobians
# admit infinitely many adjoint vectors that all satisfy the residual gate.
# But that leaves a real coverage gap: a regression that returns total
# garbage in the well-conditioned subspace would only be caught if the
# residual check itself blew up. Action-level (range-space) parity is well
# defined even when raw vectors are not — it asserts that two adjoint
# solvers agree on the components that *are* uniquely defined by ``A``,
# while still tolerating disagreement in the genuinely ambiguous null
# direction.


def _build_exact_ill_conditioned_operator_fixture(
    monkeypatch,
    *,
    device,
    cond_target=1e10,
):
    """Synthesize an ill-conditioned exact-Jacobian fixture.

    Mirrors ``_build_exact_well_conditioned_operator_fixture`` but the
    diagonal carries a near-singular smallest entry so ``cond(A) ≈
    cond_target``. Used to exercise the ``exact_ill_conditioned_adjoint``
    lane where vector parity is unstable but action-level parity (on the
    well-conditioned subspace) remains strict.
    """
    metadata = _collect_exact_well_conditioned_runtime_metadata(device)
    with jax.default_device(device):
        booz = _make_mock_boozer_surface_exact(options={"newton_tol": 1e-12})
        booz.need_to_run_code = False
        solved_x = booz._pack_decision_vector(0.3, 0.05)
        n = int(solved_x.size)

        # Make the diagonal explicitly ill-conditioned: σ_max ≈ 2.5,
        # σ_min ≈ 2.5 / cond_target. Off-diagonal couplings are kept tiny
        # so the SVD spectrum stays close to the diagonal.
        diagonal = np.linspace(2.0, 2.5, n)
        diagonal[-1] = diagonal[0] / cond_target  # tiny last singular value
        A_np = np.diag(diagonal)
        A_np += 1e-6 * np.tril(np.ones((n, n)), k=-1)
        A_np += 5e-7 * np.triu(np.ones((n, n)), k=1)
        A = jnp.asarray(A_np, dtype=jnp.float64)

        # SVD bookkeeping: ``A = U Σ Vᵀ``. The transposed adjoint system
        # is ``Aᵀ λ = rhs``, so existence requires ``rhs ∈ range(Aᵀ) =
        # span(V_well)`` (V columns where σ is well-conditioned). The
        # adjoint vector ``λ`` lives in the *primal* coordinate system
        # and is unique modulo ``null(Aᵀ) = span(U_null)``; well-defined
        # components live in ``span(U_well)``. So we project the RHS
        # through V_well and the adjoint through U_well — these are
        # *different* subspaces in general (they coincide only for
        # symmetric A, which the synthetic fixture happens to satisfy
        # but the helper must not assume).
        U_full, sigma_full, Vt_full = np.linalg.svd(A_np)
        sigma_max = float(sigma_full[0])
        well_mask = sigma_full > sigma_max * 1e-8
        U_well = U_full[:, well_mask]
        V_well = Vt_full[well_mask, :].T
        rhs_full = np.linspace(-0.4, 0.6, n)
        rhs_np = V_well @ (V_well.T @ rhs_full)
        rhs = jnp.asarray(rhs_np, dtype=jnp.float64)

        P_lu, L_lu, U_lu = scipy.linalg.lu(A_np)

        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "exact_jacobian",
            "PLU": tuple(
                jnp.asarray(piece, dtype=jnp.float64) for piece in (P_lu, L_lu, U_lu)
            ),
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

    def matrix_residual(_self, _mask):
        return lambda x: A @ x

    monkeypatch.setattr(
        _bsj.BoozerSurfaceJAX,
        "_compute_stellsym_mask_indices",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        _bsj.BoozerSurfaceJAX,
        "_make_exact_residual",
        matrix_residual,
    )

    return {
        "A": A,
        "A_np": A_np,
        "P_lu": P_lu,
        "L_lu": L_lu,
        "U_lu": U_lu,
        "U_full": U_full,
        "U_well": U_well,
        "V_well": V_well,
        "sigma_full": sigma_full,
        "sigma_max": sigma_max,
        "well_mask": well_mask,
        "booz": booz,
        "device": device,
        "metadata": metadata,
        "rhs": rhs,
        "rhs_np": rhs_np,
    }


def _solve_exact_ill_conditioned_operator_case(
    monkeypatch,
    *,
    device,
    cond_target=1e10,
):
    case = _build_exact_ill_conditioned_operator_fixture(
        monkeypatch,
        device=device,
        cond_target=cond_target,
    )
    with jax.default_device(device):
        adjoint_state = case["booz"].get_adjoint_runtime_state()
        operator_adj, success = adjoint_state.solve_transpose_with_status(case["rhs"])
        operator_adj_np = np.asarray(jax.device_get(operator_adj), dtype=np.float64)
        applied_np = np.asarray(
            jax.device_get(adjoint_state.apply_transpose(operator_adj)),
            dtype=np.float64,
        )

    plu_adj_np = forward_backward(
        case["P_lu"],
        case["L_lu"],
        case["U_lu"],
        case["rhs_np"],
    )
    residual_rel = np.linalg.norm(case["A_np"].T @ operator_adj_np - case["rhs_np"]) / (
        1.0 + np.linalg.norm(case["rhs_np"])
    )
    apply_residual_rel = np.linalg.norm(applied_np - case["rhs_np"]) / (
        1.0 + np.linalg.norm(case["rhs_np"])
    )
    return {
        **case,
        "adjoint_state": adjoint_state,
        "operator_adj_np": operator_adj_np,
        "plu_adj_np": plu_adj_np,
        "residual_rel": residual_rel,
        "apply_residual_rel": apply_residual_rel,
        "success": bool(np.asarray(success)),
    }


def _assert_exact_ill_conditioned_operator_action_parity(case, exact_lane):
    """Action-level (range-space) parity in the well-conditioned subspace.

    Vector parity ``‖λ_a − λ_b‖`` is unstable when σ_min(A) ≈ 0 because
    multiple adjoints satisfy ``A^T λ ≈ rhs`` to the residual budget.
    Project both adjoints onto ``U_well`` (the columns of ``U`` whose
    singular values exceed ``σ_max * 1e-8``) and assert agreement
    *there* at one order looser than the well-conditioned lane —
    ``adjoint_rtol = 1e-6`` per plan §W3.2 — plus a directional check on
    a fixed deterministic basis.
    """
    # The exact-lane residual gate must hold regardless of conditioning.
    assert case["residual_rel"] <= exact_lane["residual_rel_tol"], (
        f"Adjoint residual {case['residual_rel']:.3e} exceeds "
        f"{exact_lane['residual_rel_tol']:.0e}"
    )
    assert case["apply_residual_rel"] <= exact_lane["residual_rel_tol"]

    # Action-level (well-conditioned-subspace) parity threshold from the
    # parity-ladder lane SSOT.
    action_rtol = float(exact_lane["action_level_rtol"])
    P = case["U_well"] @ case["U_well"].T
    operator_proj = P @ case["operator_adj_np"]
    plu_proj = P @ case["plu_adj_np"]
    proj_norm = np.linalg.norm(operator_proj)
    proj_diff = np.linalg.norm(operator_proj - plu_proj)
    assert proj_diff <= action_rtol * (1.0 + proj_norm), (
        f"Range-space adjoint parity violated: ‖P(λ_op − λ_plu)‖={proj_diff:.3e}, "
        f"‖P λ_op‖={proj_norm:.3e}, rtol={action_rtol:.0e}"
    )

    # Directional check on a deterministic basis. ``v_k @ U_well @ U_well.T @ λ``
    # measures the well-conditioned-subspace component of λ along v_k; both
    # solvers must agree there even when raw ``v_k @ λ`` would diverge.
    rng = np.random.default_rng(seed=20260515)
    n = case["operator_adj_np"].shape[0]
    directions = rng.standard_normal((5, n))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    op_dirs = directions @ operator_proj
    plu_dirs = directions @ plu_proj
    np.testing.assert_allclose(
        op_dirs,
        plu_dirs,
        rtol=action_rtol,
        atol=action_rtol * float(np.max(np.abs(operator_proj))),
        err_msg=(
            "Directional projection of operator vs PLU adjoint disagrees: "
            f"op={op_dirs}, plu={plu_dirs}"
        ),
    )


def _disk_flux_through_circle_z0(*, radius, nr, ntheta, gammas, gammadashs, currents):
    rs = (np.arange(nr) + 0.5) * (radius / nr)
    thetas = (np.arange(ntheta) + 0.5) * (2.0 * np.pi / ntheta)
    rr, tt = np.meshgrid(rs, thetas, indexing="ij")
    points = np.stack(
        [rr * np.cos(tt), rr * np.sin(tt), np.zeros_like(rr)],
        axis=-1,
    ).reshape(-1, 3)
    B = np.asarray(
        biot_savart_B(jnp.array(points), gammas, gammadashs, currents)
    ).reshape(nr, ntheta, 3)
    area_element = (radius / nr) * (2.0 * np.pi / ntheta) * rr
    return float(np.sum(B[..., 2] * area_element))


def _emit_newton_progress(progress_callback):
    progress_callback(1, 0.25, 1.0e-2)
    progress_callback(2, 0.05, 1.0e-4)


def _stage_payload(observed, label):
    return next(
        payload for current_label, payload in observed if current_label == label
    )


def _assert_solver_completion_payload(payload):
    assert payload["objective"] == pytest.approx(0.0)
    assert payload["grad_inf"] == pytest.approx(0.0)


def _patch_matrix_free_exact_linear_solver(monkeypatch, *, A, expected_device=None):
    dense_calls = []

    def fake_jvp_fn(_residual_fn):
        def apply(_x, v):
            _maybe_assert_arrays_on_device(expected_device, v)
            return _matrix_constant(A, v) @ v

        return apply

    def fake_gmres(_jvp_fn, _x, rhs, *, tol):
        del _jvp_fn, tol
        _maybe_assert_arrays_on_device(expected_device, _x, rhs)
        A_runtime = _matrix_constant(A, rhs)
        dx = jnp.linalg.solve(A_runtime, rhs)
        return dx, rhs - A_runtime @ dx, None

    def fake_materialize(_jvp_fn, _x):
        del _jvp_fn
        A_runtime = _matrix_constant(A, _x)
        dense_calls.append(True)
        _maybe_assert_arrays_on_device(expected_device, _x, A_runtime)
        return A_runtime

    monkeypatch.setattr(_opt, "_jacobian_vector_product_fn", fake_jvp_fn)
    monkeypatch.setattr(_opt, "_gmres_solve_exact_newton_system", fake_gmres)
    monkeypatch.setattr(_opt, "_materialize_dense_jacobian", fake_materialize)
    return dense_calls


def _patch_matrix_free_lm_solver(monkeypatch, *, A, expected_device=None):
    dense_calls = []
    gmres_calls = []

    def fake_gmres(_flat_residual_fn, _x, grad, _pullback, *, damping, tol):
        del _flat_residual_fn, _pullback, tol
        gmres_calls.append(True)
        _maybe_assert_arrays_on_device(expected_device, _x, grad)
        A_runtime = _matrix_constant(A, grad)
        hessian = A_runtime.T @ A_runtime + damping * _explicit_eye(
            A_runtime.shape[1],
            dtype=A.dtype,
            device=expected_device,
        )
        step = jnp.linalg.solve(hessian, grad)
        return step, grad - hessian @ step, None

    def fake_materialize(flat_residual_fn, x):
        dense_calls.append(True)
        _maybe_assert_arrays_on_device(expected_device, x)
        residual = flat_residual_fn(x)
        jacobian = _matrix_constant(A, x)
        grad, hessian = _opt._least_squares_linearization_from_jacobian(
            residual,
            jacobian,
        )
        _maybe_assert_arrays_on_device(
            expected_device,
            residual,
            jacobian,
            grad,
            hessian,
        )
        return residual, jacobian, grad, hessian

    monkeypatch.setattr(_opt, "_gmres_solve_least_squares_system", fake_gmres)
    monkeypatch.setattr(
        _opt,
        "_materialize_dense_least_squares_linearization",
        fake_materialize,
    )
    return dense_calls, gmres_calls


def _assert_linear_lm_result(result, *, A, b):
    np.testing.assert_allclose(
        result["x"],
        np.linalg.solve(np.asarray(A), np.asarray(b)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["residual_jacobian"],
        np.asarray(A),
        atol=1e-12,
    )


def _make_structured_quadratic_problem():
    target_surface = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    target_iota = jnp.asarray(0.25, dtype=jnp.float64)

    def objective_fn(state):
        surface = jnp.asarray(state["surface"], dtype=jnp.float64)
        iota = jnp.asarray(state["iota"], dtype=jnp.float64)
        return 0.5 * (
            jnp.sum((surface - target_surface) ** 2) + (iota - target_iota) ** 2
        )

    x0 = {
        "surface": jnp.asarray([5.0, 3.0], dtype=jnp.float64),
        "iota": jnp.asarray(0.0, dtype=jnp.float64),
    }
    return objective_fn, x0, np.asarray(target_surface), float(target_iota)


def _assert_lu_is_not_called(message):
    raise AssertionError(message)


def _maybe_assert_arrays_on_device(device, *arrays):
    if device is None:
        return
    concrete_arrays = tuple(
        array
        for array in arrays
        if isinstance(array, jax.Array) and not isinstance(array, jax.core.Tracer)
    )
    if concrete_arrays:
        assert_arrays_on_device(device, *concrete_arrays)


def _explicit_eye(size, *, dtype, device=None):
    return jax.device_put(np.eye(int(size), dtype=np.dtype(dtype)), device=device)


def _explicit_scalar(value, *, dtype, device=None):
    return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)), device=device)


def _matrix_constant(matrix, reference):
    return jnp.asarray(
        np.asarray(matrix, dtype=np.dtype(reference.dtype)), dtype=reference.dtype
    )


def test_materialize_dense_linear_operator_matches_linear_map():
    A = jnp.asarray(
        [
            [2.0, -1.0, 0.5],
            [0.0, 3.0, 1.0],
            [1.5, -2.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    x = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)

    dense = _opt._materialize_dense_linear_operator(
        lambda _x, v: A @ v,
        x,
    )

    np.testing.assert_allclose(np.asarray(dense), np.asarray(A), atol=1.0e-12)


def _build_gpu_traceable_linear_problem(booz, gpu, *, step_scale):
    coil_set_spec = booz.coil_set_spec
    surface_dofs = np.asarray(booz.surface.get_dofs(), dtype=np.float64)
    sdofs = jax.device_put(
        surface_dofs,
        device=gpu,
    )
    iota = jax.device_put(np.asarray(0.3, dtype=np.float64), device=gpu)
    G = jax.device_put(np.asarray(0.05, dtype=np.float64), device=gpu)
    x0 = booz._pack_decision_vector(iota, G, sdofs=sdofs)
    x_target = np.concatenate(
        (surface_dofs, np.asarray([0.3, 0.05], dtype=np.float64))
    ) + np.linspace(
        step_scale,
        step_scale * x0.shape[0],
        x0.shape[0],
        dtype=np.float64,
    )
    A = np.eye(int(x0.shape[0]), dtype=np.float64)
    return coil_set_spec, sdofs, iota, G, x_target, A


def _assert_traceable_gpu_result(
    result,
    expected_x,
    gpu,
    *,
    jacobian_shape=None,
    hessian_shape=None,
):
    np.testing.assert_allclose(
        np.asarray(jax.device_get(result["x"])),
        np.asarray(jax.device_get(expected_x)),
        atol=1e-12,
    )
    assert_array_on_device(result["x"], gpu)

    if jacobian_shape is not None:
        np.testing.assert_allclose(
            np.asarray(jax.device_get(result["jacobian"])),
            np.eye(jacobian_shape[0]),
            atol=1e-12,
        )
        assert_array_on_device(result["jacobian"], gpu)

    if hessian_shape is not None:
        assert_array_on_device(result["hessian"], gpu)


def _patch_counting_scipy_minimize(monkeypatch):
    state = {"jit_call_count": 0}
    original_jit = _opt.jax.jit

    def counting_jit(*args, **kwargs):
        state["jit_call_count"] += 1
        return original_jit(*args, **kwargs)

    def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
        del jac, method, options, callback
        value, grad = fun(np.asarray(x0))
        return types.SimpleNamespace(
            x=np.asarray(x0),
            jac=np.asarray(grad),
            fun=float(value),
            nit=0,
            nfev=1,
            njev=1,
            success=True,
            status=0,
        )

    monkeypatch.setattr(_opt.jax, "jit", counting_jit)
    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
    return state


_enable_strict_jax_backend = partial(enable_strict_jax_backend, mode="jax_gpu_parity")
_enable_non_strict_jax_backend = partial(
    enable_non_strict_jax_backend,
    mode="jax_gpu_parity",
)
_enable_fast_strict_jax_backend = partial(
    enable_strict_jax_backend,
    mode="jax_gpu_fast",
)
_enable_fast_non_strict_jax_backend = partial(
    enable_non_strict_jax_backend,
    mode="jax_gpu_fast",
)
_ALL_JAX_BACKEND_MODES = (
    "jax_cpu_parity",
    "jax_gpu_parity",
    "jax_gpu_fast",
    "jax_metal_smoke",
)
_NON_ONDEVICE_LS_BACKENDS = ("scipy",)
_NON_TARGET_MINIMIZE_METHODS = ("adam", "bfgs", "lbfgs")


_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN = (
    r"BoozerSurfaceJAX requires a biotsavart object that provides "
    r"coil_set_spec\(\) for explicit immutable grouped-coil state"
)


def _target_lane_rejection_pattern(
    component: str, method: str, backend_mode: str
) -> str:
    return (
        rf"{component}.*method='{method}'.*{backend_mode}.*requires an "
        r"ondevice optimizer method"
    )


_LEGACY_CURVE_X = np.array([1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _make_legacy_coils_list_biotsavart(coils):
    class _LegacyCoilsListBiotSavart(_bsj.Optimizable):
        def __init__(self, legacy_coils):
            super().__init__(x0=np.asarray([]))
            self._coils = legacy_coils

    return _LegacyCoilsListBiotSavart(coils)


def _make_legacy_spec_capable_coils():
    curve = CurveXYZFourier(16, 1)
    curve.x = _LEGACY_CURVE_X.copy()
    return [Coil(curve, Current(1.23))]


def _make_curve_current_spec_only_legacy_coils():
    live_curve = CurveXYZFourier(16, 1)
    live_curve.x = _LEGACY_CURVE_X.copy()
    curve_spec = live_curve.to_spec()
    current_spec = make_current_value_spec(1.23)

    class _CurveSpecOnly:
        def to_spec(self):
            return curve_spec

        def gamma(self):
            raise AssertionError("gamma() should not be read")

        def gammadash(self):
            raise AssertionError("gammadash() should not be read")

    class _CurrentSpecOnly:
        def to_spec(self):
            return current_spec

        def get_value(self):
            raise AssertionError("get_value() should not be read")

    class _CurveCurrentSpecOnlyCoil:
        def __init__(self, legacy_curve, legacy_current):
            self.curve = legacy_curve
            self.current = legacy_current

    return [_CurveCurrentSpecOnlyCoil(_CurveSpecOnly(), _CurrentSpecOnly())]


def _make_mock_boozer_surface_with_free_currents():
    class _MutableCurrent:
        def __init__(self, value):
            self._value = value
            self.dofs = self

        def get_value(self):
            return self._value

        def all_fixed(self):
            return False

    booz = _make_mock_boozer_surface()
    booz.biotsavart.coils[0].current = _MutableCurrent(
        booz.biotsavart.coils[0].current.get_value()
    )
    return booz


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStellsymScatterIndices:
    """Test the stellsym DOF packing/unpacking."""

    def test_nonsym_uses_identity(self):
        """Non-stellsym doesn't need scatter — DOFs map directly."""
        mpol, ntor = 2, 3
        n = (2 * mpol + 1) * (2 * ntor + 1)
        # For non-stellsym, surface_gamma_from_dofs uses direct reshape.
        # Verify stellsym indices are a strict subset.
        indices = stellsym_scatter_indices(mpol, ntor)
        assert len(indices) < 3 * n

    def test_stellsym_count(self):
        """Stellsym reduces DOF count."""
        mpol, ntor = 2, 3
        n_full = (2 * mpol + 1) * (2 * ntor + 1)
        indices = stellsym_scatter_indices(mpol, ntor)
        assert len(indices) < 3 * n_full
        # x: cos-cos + sin-sin = (mpol+1)*(ntor+1) + mpol*ntor
        # y,z: cos-sin + sin-cos = (mpol+1)*ntor + mpol*(ntor+1) each
        n_x = (mpol + 1) * (ntor + 1) + mpol * ntor
        n_yz = (mpol + 1) * ntor + mpol * (ntor + 1)
        expected = n_x + 2 * n_yz
        assert len(indices) == expected

    def test_round_trip(self):
        """Scatter then gather recovers original DOFs."""
        mpol, ntor = 1, 1
        indices = jnp.array(stellsym_scatter_indices(mpol, ntor))
        ndofs = len(indices)
        dofs = jnp.arange(ndofs, dtype=jnp.float64) + 1.0

        xc, yc, zc = dofs_to_xyzc(dofs, indices, mpol, ntor)
        flat = jnp.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
        recovered = flat[indices]
        np.testing.assert_allclose(recovered, dofs)

    def test_stellsym_zeros_correct_quadrants(self):
        """Stellsym zeroes out the correct coefficient quadrants."""
        mpol, ntor = 2, 2
        indices = jnp.array(stellsym_scatter_indices(mpol, ntor))
        ndofs = len(indices)
        dofs = jnp.ones(ndofs, dtype=jnp.float64)

        xc, yc, zc = dofs_to_xyzc(dofs, indices, mpol, ntor)

        # x: cs and sc quadrants should be zero
        # cs: rows 0..mpol, cols ntor+1..2*ntor
        assert float(jnp.sum(jnp.abs(xc[: mpol + 1, ntor + 1 :]))) == 0.0
        # sc: rows mpol+1..2*mpol, cols 0..ntor
        assert float(jnp.sum(jnp.abs(xc[mpol + 1 :, : ntor + 1]))) == 0.0

        # y/z: cc and ss quadrants should be zero
        assert float(jnp.sum(jnp.abs(yc[: mpol + 1, : ntor + 1]))) == 0.0
        assert float(jnp.sum(jnp.abs(yc[mpol + 1 :, ntor + 1 :]))) == 0.0
        assert float(jnp.sum(jnp.abs(zc[: mpol + 1, : ntor + 1]))) == 0.0
        assert float(jnp.sum(jnp.abs(zc[mpol + 1 :, ntor + 1 :]))) == 0.0


class TestSurfaceVolume:
    """Test the JAX volume computation."""

    def test_simple_torus_volume(self):
        """Volume of a simple torus: V = 2π² R r²."""
        geometry = _simple_torus_geometry_values(
            R0=1.0,
            r=0.1,
            mpol=1,
            ntor=1,
            nfp=1,
            nphi=32,
            ntheta=32,
        )
        np.testing.assert_allclose(
            geometry["volume"],
            geometry["expected_volume"],
            rtol=_TORUS_GEOMETRY_RTOL,
        )


class TestVectorPotentialA:
    """Test the Biot-Savart vector potential."""

    def test_stokes_theorem(self):
        """∫ B·n dA ≈ ∮ A·dl on a small disk for a single coil."""
        gammas, gammadashs, currents = _make_circular_coil(R=1.0, current=1e5)

        # Evaluate A on a circle at r=0.5 in the z=0 plane
        npts = 64
        r_test = 0.5
        theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        pts = np.stack(
            [r_test * np.cos(theta), r_test * np.sin(theta), np.zeros(npts)], axis=-1
        )
        tangent = np.stack(
            [-r_test * np.sin(theta), r_test * np.cos(theta), np.zeros(npts)], axis=-1
        )

        A = biot_savart_A(jnp.array(pts), gammas, gammadashs, currents)
        # Line integral: ∮ A · dl ≈ (2π/N) Σ A · tangent
        flux_A = float(jnp.sum(A * jnp.array(tangent))) * (2 * np.pi / npts)
        flux_B = _disk_flux_through_circle_z0(
            radius=r_test,
            nr=_STOKES_DISK_NR,
            ntheta=_STOKES_DISK_NTHETA,
            gammas=gammas,
            gammadashs=gammadashs,
            currents=currents,
        )
        np.testing.assert_allclose(
            flux_A,
            flux_B,
            rtol=_STOKES_FLUX_RTOL,
            atol=_STOKES_FLUX_ATOL,
        )

    def test_A_divergence_free_proxy(self):
        """dA/dX trace should be approximately zero (Coulomb gauge)."""
        gammas, gammadashs, currents = _make_circular_coil(R=1.0, current=1e5)
        pts = jnp.array([[0.5, 0.0, 0.1]])

        dA_dX = biot_savart_dA_by_dX(pts, gammas, gammadashs, currents)
        # div A = trace of dA/dX (Coulomb gauge: ∇·A = 0)
        div_A = float(jnp.trace(dA_dX[0]))
        np.testing.assert_allclose(div_A, 0.0, atol=1e-12)


class TestLabelConstraints:
    """Test volume and toroidal flux computations."""

    def test_compute_G(self):
        """G = μ₀ Σ|I_k|."""
        currents = jnp.array([1e5, -2e5])
        G = float(compute_G_from_currents(currents))
        mu0 = 4 * np.pi * 1e-7
        expected = mu0 * (1e5 + 2e5)
        np.testing.assert_allclose(G, expected, rtol=1e-14)


class TestComposedPenaltyObjective:
    """Test the full composed penalty objective function."""

    def _setup(self, nphi=8, ntheta=8, mpol=1, ntor=1, nfp=1):
        return _build_penalty_problem(
            nphi=nphi,
            ntheta=ntheta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
        )

    def test_penalty_returns_scalar(self):
        """Penalty objective returns a scalar."""
        d = self._setup()
        val = d["objective"](d["x"])
        assert val.shape == ()
        assert float(val) >= 0.0

    def test_penalty_gradient_fd(self):
        """Gradient of penalty objective matches centred finite differences."""
        d = self._setup()
        obj = d["objective"]

        grad_fn = jax.grad(obj)
        grad_jax = grad_fn(d["x"])

        # Finite differences on a few components (full FD is expensive)
        eps = 1e-6
        for idx in [0, len(d["x"]) // 2, -2, -1]:
            x_p = d["x"].at[idx].add(eps)
            x_m = d["x"].at[idx].add(-eps)
            fd = (float(obj(x_p)) - float(obj(x_m))) / (2 * eps)
            np.testing.assert_allclose(
                float(grad_jax[idx]),
                fd,
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"Gradient mismatch at index {idx}",
            )


class TestOptimizerAdapter:
    """Test the JAX optimizer adapter."""

    def test_bfgs_rosenbrock(self):
        """BFGS minimizes the Rosenbrock function."""

        def rosenbrock(x):
            return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

        x0 = jnp.array([-1.0, 1.0])
        result = jax_minimize(rosenbrock, x0, method="bfgs", tol=1e-8, maxiter=500)
        assert result.success
        np.testing.assert_allclose(
            result.x,
            jnp.array([1.0, 1.0]),
            atol=_ROSENBROCK_SOLUTION_ATOL,
        )

    def test_scipy_lbfgs_preserves_supported_options(self, monkeypatch):
        """SciPy L-BFGS-B must receive its valid tuning knobs."""
        captured = {}

        def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
            del jac
            captured["method"] = method
            captured["options"] = dict(options)
            captured["callback"] = callback
            fun(x0)
            return types.SimpleNamespace(
                x=np.asarray(x0),
                jac=np.asarray(x0),
                fun=0.0,
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
        jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -2.0]),
            method="lbfgs",
            tol=1e-8,
            maxiter=7,
            options={"maxcor": 33, "ftol": 1e-12, "maxfun": 55, "maxls": 66},
        )

        assert captured["method"] == "L-BFGS-B"
        assert captured["options"]["maxcor"] == 33
        assert captured["options"]["ftol"] == 1e-12
        assert captured["options"]["maxfun"] == 55
        assert captured["options"]["maxls"] == 66
        assert captured["callback"] is None  # no callback in this call

    def test_scipy_bfgs_strips_limited_memory_and_callback_options(self, monkeypatch):
        """CPU-parity BFGS exposes the exact SciPy call contract."""
        captured = {}
        callback_marker = object()
        progress_marker = object()
        failure_marker = object()

        def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
            captured["method"] = method
            captured["options"] = dict(options)
            captured["callback"] = callback
            value, gradient = fun(x0)
            fun(np.asarray(x0) + 0.5)
            captured["value_type"] = type(value)
            captured["gradient_type"] = type(gradient)
            captured["gradient_dtype"] = gradient.dtype
            assert jac is True
            return types.SimpleNamespace(
                x=np.asarray(x0),
                jac=np.asarray(gradient),
                fun=float(value),
                nit=3,
                nfev=5,
                njev=5,
                success=True,
                status=0,
                message="Optimization terminated successfully.",
            )

        monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
        result = _opt_ref._scipy_minimize(
            lambda x: jnp.sum((x - 1.0) ** 2),
            jnp.array([0.0, 2.0], dtype=jnp.float64),
            method="bfgs",
            tol=1e-8,
            maxiter=7,
            options={
                "callback": callback_marker,
                "progress_callback": progress_marker,
                "failure_callback": failure_marker,
                "maxcor": 33,
                "ftol": 1e-12,
                "maxfun": 55,
                "maxls": 66,
                "record_scipy_callback_trace": True,
            },
        )

        assert captured["method"] == "BFGS"
        assert captured["options"] == {"maxiter": 7, "gtol": 1e-8}
        assert captured["callback"] is not None
        assert captured["value_type"] is np.float64
        assert captured["gradient_type"] is np.ndarray
        assert captured["gradient_dtype"] == np.dtype(jnp.float64)
        assert result.success is True
        assert result.status == 0
        assert result.message == "Optimization terminated successfully."
        assert result.nit == 3
        assert result.nfev == 5
        assert result.njev == 5
        assert result.scipy_call_contract == {
            "semantic_method": "bfgs",
            "scipy_method": "BFGS",
            "scipy_options": {"maxiter": 7, "gtol": 1e-8},
            "callback": "callable",
            "success": True,
            "status": 0,
            "message": "Optimization terminated successfully.",
            "nit": 3,
            "nfev": 5,
            "njev": 5,
        }
        np.testing.assert_allclose(
            result.scipy_initial_call["decision_vector"],
            np.asarray([0.0, 2.0], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        assert result.scipy_initial_call["fun"] == np.float64(2.0)
        np.testing.assert_allclose(
            result.scipy_initial_call["gradient"],
            np.asarray([-2.0, 2.0], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        assert len(result.scipy_callback_trace) == 2
        np.testing.assert_allclose(
            result.scipy_callback_trace[0]["decision_vector"],
            np.asarray([0.0, 2.0], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        assert result.scipy_callback_trace[0]["fun"] == np.float64(2.0)
        np.testing.assert_allclose(
            result.scipy_callback_trace[0]["gradient"],
            np.asarray([-2.0, 2.0], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result.scipy_callback_trace[1]["decision_vector"],
            np.asarray([0.5, 2.5], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        ("adapter_name", "objective_fn"),
        [
            ("_scipy_minimize", lambda x: jnp.sum((x - 1.0) ** 2)),
            (
                "_scipy_minimize_value_and_grad",
                lambda x: (jnp.sum((x - 1.0) ** 2), 2.0 * (x - 1.0)),
            ),
        ],
    )
    def test_reference_scipy_adapters_materialize_host_contract(
        self, monkeypatch, adapter_name, objective_fn
    ):
        """The reference SciPy adapter is the intentional NumPy host boundary."""
        captured = {}

        def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
            captured["x0_type"] = type(x0)
            captured["x0_dtype"] = x0.dtype
            captured["jac"] = jac
            captured["method"] = method
            captured["options"] = dict(options)
            captured["callback"] = callback
            value, grad = fun(x0)
            captured["value_type"] = type(value)
            captured["grad_type"] = type(grad)
            captured["grad_dtype"] = grad.dtype
            return types.SimpleNamespace(
                x=np.asarray(x0),
                jac=np.asarray(grad),
                fun=float(value),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
        x0 = jnp.array([0.0, 2.0], dtype=jnp.float64)
        adapter = getattr(_opt_ref, adapter_name)

        result = adapter(
            objective_fn,
            x0,
            method="lbfgs",
            tol=1e-8,
            maxiter=3,
            options={"maxcor": 7},
        )

        assert captured["x0_type"] is np.ndarray
        assert captured["x0_dtype"] == np.dtype(jnp.float64)
        assert captured["jac"] is True
        assert captured["method"] == "L-BFGS-B"
        assert captured["options"] == {"maxiter": 3, "gtol": 1e-8, "maxcor": 7}
        assert captured["callback"] is None
        assert captured["value_type"] is np.float64
        assert captured["grad_type"] is np.ndarray
        assert captured["grad_dtype"] == np.dtype(jnp.float64)
        np.testing.assert_allclose(result.x, np.asarray(x0))
        np.testing.assert_allclose(result.jac, np.asarray([-2.0, 2.0]))

    def test_reference_scipy_adapter_rejects_non_scalar_objective(self, monkeypatch):
        def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
            del jac, method, options, callback
            fun(x0)

        monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
        x0 = jnp.array([0.0, 2.0], dtype=jnp.float64)

        with pytest.raises(ValueError, match="scalar shape"):
            _opt_ref._scipy_minimize_value_and_grad(
                lambda x: (jnp.ones((1,), dtype=x.dtype), jnp.ones_like(x)),
                x0,
                method="bfgs",
                tol=1e-8,
                maxiter=3,
                options={},
            )

    @pytest.mark.parametrize(
        "method",
        ["lbfgs-scipy-jax", "lbfgs-scipy-jax-fullgraph"],
    )
    def test_target_scipy_jax_uses_scipy_control_with_jax_value_grad(
        self, monkeypatch, method
    ):
        """The parity target lane keeps SciPy control and JAX value/grad evals."""
        captured = {}

        def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
            captured["x0_type"] = type(x0)
            captured["method"] = method
            captured["options"] = dict(options)
            captured["callback"] = callback
            value, grad = fun(x0)
            captured["value_type"] = type(value)
            captured["grad_type"] = type(grad)
            captured["grad_dtype"] = grad.dtype
            return types.SimpleNamespace(
                x=np.asarray(x0),
                jac=np.asarray(grad),
                fun=float(value),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        objective_inputs = []

        def value_and_grad(x):
            objective_inputs.append(type(x))
            return jnp.sum((x - 1.0) ** 2), 2.0 * (x - 1.0)

        monkeypatch.setattr(_opt, "_x64_enabled", lambda: True)
        monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
        x0 = jnp.array([0.0, 2.0], dtype=jnp.float64)

        result = _opt.target_minimize(
            value_and_grad,
            x0,
            method=method,
            tol=1e-8,
            maxiter=3,
            options={"maxcor": 7, "ftol": 1e-12, "maxls": 9},
            value_and_grad=True,
        )

        assert captured["x0_type"] is np.ndarray
        assert captured["method"] == "L-BFGS-B"
        assert captured["options"] == {
            "maxiter": 3,
            "gtol": 1e-8,
            "maxcor": 7,
            "ftol": 1e-12,
            "maxls": 9,
        }
        assert captured["callback"] is None
        assert captured["value_type"] is np.float64
        assert captured["grad_type"] is np.ndarray
        assert captured["grad_dtype"] == np.dtype(jnp.float64)
        assert objective_inputs
        assert all(issubclass(input_type, jax.Array) for input_type in objective_inputs)
        np.testing.assert_allclose(result.x, np.asarray(x0))
        np.testing.assert_allclose(result.jac, np.asarray([-2.0, 2.0]))

    @pytest.mark.parametrize(
        "method",
        ["lbfgs-scipy-jax", "lbfgs-scipy-jax-fullgraph"],
    )
    def test_target_scipy_jax_requires_explicit_value_grad(self, monkeypatch, method):
        monkeypatch.setattr(_opt, "_x64_enabled", lambda: True)
        with pytest.raises(RuntimeError, match="requires value_and_grad=True"):
            _opt.target_minimize(
                lambda x: jnp.sum(x**2),
                jnp.array([1.0], dtype=jnp.float64),
                method=method,
            )

    def test_target_scipy_jax_fullstate_method_is_unsupported(self):
        with pytest.raises(ValueError, match="only supports target-lane methods"):
            _opt.target_minimize(
                lambda x: (jnp.sum(x**2), 2.0 * x),
                jnp.array([1.0], dtype=jnp.float64),
                method="lbfgs-scipy-jax-fullstate",
                value_and_grad=True,
            )

    def test_reference_lbfgs_trace_uses_host_core_without_scipy(self, monkeypatch):
        """CPU/C++ trace lane must use the shared host L-BFGS core, not SciPy."""

        def forbidden_scipy_minimize(*_args, **_kwargs):
            raise AssertionError("lbfgs-trace must not enter scipy_minimize().")

        monkeypatch.setattr(_opt_ref, "scipy_minimize", forbidden_scipy_minimize)
        observed_argument_types = []

        def explicit_quad(x):
            observed_argument_types.append(type(x))
            x = np.asarray(x, dtype=np.float64)
            return float(0.5 * np.dot(x, x)), np.asarray(x, dtype=np.float64)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        result = _opt.reference_minimize(
            explicit_quad,
            x0,
            method="lbfgs-trace",
            tol=1e-10,
            maxiter=3,
            options={"ftol": 0.0, "initial_step_size": 1.0},
            value_and_grad=True,
            initial_value_and_grad=explicit_quad(np.asarray(x0)),
        )

        assert result.success is True
        assert int(result.status) == 0
        np.testing.assert_allclose(np.asarray(result.x), np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.jac), np.zeros(2), atol=1e-12)
        assert observed_argument_types
        assert all(arg_type is np.ndarray for arg_type in observed_argument_types)
        assert len(result.optimizer_state_trace) == 1
        trace = result.optimizer_state_trace[0]
        assert trace["iteration"] == 1
        assert trace["line_search_status"] == 0
        assert trace["accepted"] is True
        assert trace["wolfe_satisfied"] is True

    def test_scipy_minimize_does_not_cache_unmarked_objective(self, monkeypatch):
        """Generic optimizer callables should keep the historical fresh-jit semantics."""
        state = _patch_counting_scipy_minimize(monkeypatch)

        def quad(x):
            return jnp.sum((x - 1.0) ** 2)

        x0 = jnp.array([0.0, 2.0], dtype=jnp.float64)
        _opt_ref._scipy_minimize(
            quad,
            x0,
            method="lbfgs",
            tol=1e-8,
            maxiter=1,
            options={},
        )
        _opt_ref._scipy_minimize(
            quad,
            x0,
            method="lbfgs",
            tol=1e-8,
            maxiter=1,
            options={},
        )

        assert state["jit_call_count"] == 2

    def test_newton_polish_quadratic(self):
        """Newton polish converges in 1 iteration for a quadratic."""
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        b = jnp.array([1.0, 2.0])

        def obj(x):
            return 0.5 * x @ A @ x - b @ x

        x0 = jnp.zeros(2)
        result = newton_polish(obj, x0, maxiter=5, tol=1e-14)
        x_exact = jnp.linalg.solve(A, b)
        np.testing.assert_allclose(result["x"], x_exact, atol=1e-12)
        assert result["success"]

    def test_newton_polish_refines_nontrivial_gmres_residual(self, monkeypatch):
        """Iterative refinement should run when GMRES leaves a small residual."""

        def obj(x):
            return 0.5 * x[0] ** 2

        calls = []

        def fake_hvp_fn(_objective_fn):
            return lambda _x, v: v

        def fake_gmres(_hvp_fn, _x, rhs, *, stab, tol):
            calls.append(np.asarray(rhs, dtype=float).copy())
            if len(calls) == 1:
                return jnp.array([0.75]), jnp.array([1e-6]), None
            return jnp.array([0.25]), jnp.array([0.0]), None

        monkeypatch.setattr(_opt, "_hessian_vector_product_fn", fake_hvp_fn)
        monkeypatch.setattr(_opt, "_gmres_solve_newton_system", fake_gmres)

        result = newton_polish(obj, jnp.array([1.0]), maxiter=1, tol=1e-12)

        assert len(calls) == 2
        np.testing.assert_allclose(result["x"], np.array([0.0]), atol=1e-12)

    def test_newton_polish_dense_steps_use_materialized_solve(self, monkeypatch):
        """CPU-parity Newton polish uses dense steps when explicitly requested.

        Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §9
        Phase 1, the *final* reported Hessian is always mirror-upper
        symmetrized regardless of ``dense_newton_steps``. The per-iter dense
        Newton step keeps ``symmetrize=False`` so its solve bytes stay
        consistent with the C++ oracle's pre-symmetrization Hessian.
        """
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]], dtype=jnp.float64)
        b = jnp.array([1.0, 2.0], dtype=jnp.float64)
        symmetrize_requests = []

        def obj(x):
            return 0.5 * x @ A @ x - b @ x

        def forbid_gmres(*_args, **_kwargs):
            raise AssertionError("dense Newton steps should not call GMRES")

        original_materialize = _opt._materialize_dense_hessian

        def record_materialize(hvp_fn, x, *, symmetrize=True):
            symmetrize_requests.append(bool(symmetrize))
            return original_materialize(hvp_fn, x, symmetrize=symmetrize)

        monkeypatch.setattr(_opt, "_gmres_solve_newton_system", forbid_gmres)
        monkeypatch.setattr(_opt, "_materialize_dense_hessian", record_materialize)

        result = newton_polish(
            obj,
            jnp.zeros(2, dtype=jnp.float64),
            maxiter=1,
            tol=1e-14,
            dense_newton_steps=True,
        )

        np.testing.assert_allclose(
            result["x"],
            np.linalg.solve(np.asarray(A), np.asarray(b)),
            atol=1e-12,
        )
        assert symmetrize_requests == [False, True]
        assert result["dense_newton_steps_materialized"] is True

    def test_materialize_dense_hessian_is_bit_symmetric_under_mirror_upper(self):
        """Mirror-upper symmetrization yields a bit-symmetric Hessian.

        Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §5.1
        and §9 Phase 1, ``_materialize_dense_hessian`` must produce ``H ==
        H.T`` element-wise (no FP rounding) so the LU input is bit-symmetric
        and the upper triangle of the raw HVP-built Hessian is preserved.
        """
        rng = np.random.default_rng(0)
        A = rng.standard_normal(size=(7, 7))
        A = A + A.T
        # Inject FP rounding asymmetry so the symmetrization is observable.
        A[0, 1] += 1e-15

        def hvp_fn(_x, v):
            return jnp.asarray(A) @ v

        x = jnp.zeros(7, dtype=jnp.float64)
        H_symmetric = _opt._materialize_dense_hessian(hvp_fn, x, symmetrize=True)
        H_raw = _opt._materialize_dense_hessian(hvp_fn, x, symmetrize=False)

        sym_diff = float(jnp.linalg.norm(H_symmetric - H_symmetric.T, ord="fro"))
        upper_diff = float(
            jnp.linalg.norm(jnp.triu(H_symmetric) - jnp.triu(H_raw), ord="fro")
        )
        assert sym_diff == 0.0
        assert upper_diff == 0.0

    def test_solve_quality_helpers_handle_edge_inputs(self):
        """Solve-quality helpers propagate non-finite norms instead of masking.

        Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §3.1,
        a NaN-laden Hessian is a real failure signal — the helper must NOT
        silently substitute ``None``. Zero-norm Hessians stay ``None``
        (degenerate, not "field unavailable" — but consistent with the spec
        formula that has no defined value at zero norm).
        """
        from simsopt.geo.boozersurface_jax import (
            EXACT_FACTORIZATION_BACKEND,
            _ls_factorization_backend,
            _ls_hessian_symmetry_rel,
            _none_solve_quality_fields,
            SOLVE_QUALITY_EXACT_FIELDS,
            SOLVE_QUALITY_LS_FIELDS,
        )

        assert _ls_hessian_symmetry_rel(None) is None
        assert _ls_hessian_symmetry_rel(jnp.zeros((3, 3))) is None
        assert np.isnan(_ls_hessian_symmetry_rel(jnp.full((3, 3), jnp.nan)))
        assert np.isinf(_ls_hessian_symmetry_rel(jnp.full((3, 3), jnp.inf)))
        symmetric = jnp.asarray([[2.0, 0.5], [0.5, 3.0]])
        assert _ls_hessian_symmetry_rel(symmetric) == 0.0

        assert _ls_factorization_backend(None, optimizer_backend="scipy") is None
        H = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
        assert (
            _ls_factorization_backend(H, optimizer_backend="scipy") == "lapack-dgetrf"
        )
        assert _ls_factorization_backend(H, optimizer_backend="ondevice") == (
            "cusolver-getrf-ffi"
            if str(H.device.platform).lower() in {"gpu", "cuda"}
            else "lapack-dgetrf"
        )
        assert EXACT_FACTORIZATION_BACKEND == "operator-gmres"

        ls_placeholders = _none_solve_quality_fields(SOLVE_QUALITY_LS_FIELDS)
        exact_placeholders = _none_solve_quality_fields(SOLVE_QUALITY_EXACT_FIELDS)
        assert set(ls_placeholders) == set(SOLVE_QUALITY_LS_FIELDS)
        assert set(exact_placeholders) == set(SOLVE_QUALITY_EXACT_FIELDS)
        assert all(value is None for value in ls_placeholders.values())
        assert all(value is None for value in exact_placeholders.values())

    def test_solve_quality_field_sets_match_contract(self):
        """The frozen field tuples must match the contract §3.1 / §3.2 spec."""
        from simsopt.geo.boozersurface_jax import (
            SOLVE_QUALITY_EXACT_FIELDS,
            SOLVE_QUALITY_LS_FIELDS,
        )

        assert SOLVE_QUALITY_LS_FIELDS == (
            "ls_hessian_symmetry_rel",
            "ls_hessian_action_max_rel",
            "ls_newton_linear_residual_rel",
            "ls_newton_step_abs_diff_rel",
            "ls_factorization_backend",
            "ls_condition_estimate",
        )
        assert SOLVE_QUALITY_EXACT_FIELDS == (
            "exact_jacobian_action_max_rel",
            "exact_newton_linear_residual_rel",
            "exact_refinement_correction_rel",
            "exact_adjoint_solve_residual_rel",
            "exact_factorization_backend",
            "exact_condition_estimate",
        )

    def test_least_squares_normal_system_fails_closed_on_nonfinite_operator_solve(
        self, monkeypatch
    ):
        """LS adjoint solves should not cascade into a dense normal fallback."""
        x = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
        rhs = jnp.asarray([2.0, -3.0], dtype=jnp.float64)
        operator_calls = []
        dense_calls = []

        monkeypatch.setattr(
            _opt,
            "_least_squares_normal_operator",
            lambda _residual_fn, _x: {
                "flat_residual_fn": "flat-residual-marker",
                "matvec": lambda vec: vec,
                "transpose_matvec": lambda vec: vec,
            },
        )

        def fake_operator_solve(_matvec, solve_rhs, *, tol):
            operator_calls.append((np.asarray(solve_rhs, dtype=float), float(tol)))
            return jnp.full_like(solve_rhs, jnp.nan), False

        def fake_materialize(flat_residual_fn, materialize_x):
            dense_calls.append(
                (flat_residual_fn, np.asarray(materialize_x, dtype=float))
            )
            return None, None, None, jnp.eye(2, dtype=jnp.float64)

        monkeypatch.setattr(
            _opt,
            "_solve_square_array_system_operator_only",
            fake_operator_solve,
        )
        monkeypatch.setattr(
            _opt,
            "_materialize_dense_least_squares_linearization",
            fake_materialize,
        )

        solved, success = _opt._solve_least_squares_normal_system_with_status(
            lambda trial_x: trial_x,
            x,
            rhs,
            tol=1.0e-10,
        )

        assert len(operator_calls) == 1
        assert dense_calls == []
        assert bool(np.asarray(success)) is False
        assert not np.any(np.isfinite(np.asarray(solved)))

    @staticmethod
    def _patch_newton_polish_linear_step(monkeypatch, step, residual):
        step = jnp.asarray(step, dtype=jnp.float64)
        residual = jnp.asarray(residual, dtype=jnp.float64)

        def fake_hvp_fn(_objective_fn):
            return lambda _x, v: v

        def fake_gmres(_hvp_fn, _x, _rhs, *, stab, tol):
            del stab, tol
            return step, residual, None

        monkeypatch.setattr(_opt, "_hessian_vector_product_fn", fake_hvp_fn)
        monkeypatch.setattr(_opt, "_gmres_solve_newton_system", fake_gmres)

    def test_newton_polish_backtracks_finite_norm_increasing_operator_steps(
        self, monkeypatch
    ):
        """Newton polish backtracks finite operator steps that worsen gradient norm."""

        def obj(x):
            return 0.5 * x[0] ** 2

        self._patch_newton_polish_linear_step(monkeypatch, [10.0], [0.0])

        result = newton_polish(obj, jnp.array([1.0]), maxiter=1, tol=1e-12)

        np.testing.assert_allclose(result["x"], np.array([-0.25]), atol=1e-12)
        np.testing.assert_allclose(result["grad"], np.array([-0.25]), atol=1e-12)
        assert result["nit"] == 1
        assert not result["success"]

    def test_newton_polish_backtracks_nonfinite_value_candidate(self, monkeypatch):
        """Newton polish must reject candidates with non-finite objective values."""

        def obj(x):
            return jnp.where(x[0] < 0.9, jnp.inf, 0.5 * x[0] ** 2)

        self._patch_newton_polish_linear_step(monkeypatch, [0.2], [0.0])

        result = newton_polish(
            obj,
            jnp.array([1.0], dtype=jnp.float64),
            maxiter=1,
            tol=1e-12,
            materialize_hessian=False,
        )

        np.testing.assert_allclose(result["x"], np.array([0.9]), atol=1e-12)
        assert np.isfinite(float(result["fun"]))
        np.testing.assert_allclose(result["grad"], np.array([0.9]), atol=1e-12)
        assert result["nit"] == 1
        assert not result["success"]

    def test_newton_polish_rejects_nonfinite_gradient_norm_candidate(self, monkeypatch):
        """Finite gradient entries are not enough when their norm overflows."""

        def obj(x):
            return 1e308 * x[0] + 1e308 * x[1]

        self._patch_newton_polish_linear_step(
            monkeypatch,
            [0.0, 0.0],
            [0.0, 0.0],
        )

        result = newton_polish(
            obj,
            jnp.zeros(2, dtype=jnp.float64),
            maxiter=1,
            tol=1e-12,
            materialize_hessian=False,
        )

        assert result["nit"] == 0
        assert not result["success"]

    def test_newton_polish_nonfinite_operator_step_fails_without_dense_fallback(
        self,
        monkeypatch,
    ):
        """Newton polish must not materialize a dense Hessian rescue in the loop."""

        def obj(x):
            return 0.5 * x[0] ** 2

        def fake_hvp_fn(_objective_fn):
            return lambda _x, v: v

        def fake_gmres(_hvp_fn, _x, rhs, *, stab, tol):
            del rhs, stab, tol
            return jnp.asarray([jnp.nan]), jnp.asarray([jnp.nan]), None

        def fake_materialize_dense_hessian(_hvp_fn, _x):
            raise AssertionError("newton_polish must stay operator-only in the loop")

        monkeypatch.setattr(_opt, "_hessian_vector_product_fn", fake_hvp_fn)
        monkeypatch.setattr(_opt, "_gmres_solve_newton_system", fake_gmres)
        monkeypatch.setattr(
            _opt,
            "_materialize_dense_hessian",
            fake_materialize_dense_hessian,
        )

        result = newton_polish(
            obj,
            jnp.asarray([1.0]),
            maxiter=5,
            tol=1e-12,
            materialize_hessian=False,
        )

        np.testing.assert_allclose(result["x"], np.asarray([1.0]), atol=1e-12)
        np.testing.assert_allclose(result["grad"], np.asarray([1.0]), atol=1e-12)
        assert result["nit"] == 0
        assert not result["success"]
        assert result["hessian"] is None
        assert result["hessian_materialized"] is False

    def test_newton_exact_linear_system(self):
        """Newton exact solver finds root of a linear system in 1 step."""
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        b = jnp.array([5.0, 7.0])

        def residual(x):
            return A @ x - b

        x0 = jnp.zeros(2)
        result = newton_exact(residual, x0, maxiter=5, tol=1e-14)
        x_exact = jnp.linalg.solve(A, b)
        np.testing.assert_allclose(result["x"], x_exact, atol=1e-12)
        np.testing.assert_allclose(result["jacobian"], np.asarray(A), atol=1e-12)
        assert result["success"]

    def test_newton_exact_materializes_dense_jacobian_once_at_final_iterate(
        self, monkeypatch
    ):
        """Exact Newton keeps the loop matrix-free and rebuilds ``J`` once at the end."""
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        b = jnp.array([5.0, 7.0])
        dense_calls = _patch_matrix_free_exact_linear_solver(
            monkeypatch,
            A=A,
        )
        x_exact = np.linalg.solve(np.asarray(A), np.asarray(b))

        def residual(x):
            return A @ x - b

        result = newton_exact(residual, jnp.zeros(2), maxiter=5, tol=1e-14)

        assert len(dense_calls) == 1
        np.testing.assert_allclose(result["x"], x_exact)
        np.testing.assert_allclose(result["jacobian"], np.asarray(A), atol=1e-12)
        assert result["success"]

    def test_newton_exact_traceable_is_operator_only(self, monkeypatch):
        """Traceable exact Newton must keep finalization strictly operator-only."""
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        b = jnp.array([5.0, 7.0])
        _patch_matrix_free_exact_linear_solver(monkeypatch, A=A)
        x_exact = np.linalg.solve(np.asarray(A), np.asarray(b))
        materialize_calls = []

        def fake_materialize(_jvp_fn, _x):
            materialize_calls.append(True)
            raise AssertionError("traceable exact Newton must not materialize dense J")

        monkeypatch.setattr(_opt, "_materialize_dense_jacobian", fake_materialize)

        def residual(x):
            return A @ x - b

        result = _opt.newton_exact_traceable(
            residual,
            jnp.zeros(2),
            maxiter=5,
            tol=1e-14,
        )

        assert not materialize_calls
        np.testing.assert_allclose(result["x"], x_exact)
        assert result["jacobian"] is None
        assert result["jacobian_materialized"] is False
        assert result["failure_category"] is None
        assert result["failure_stage"] is None
        assert result["message"] is None
        assert bool(result["success"])

    def test_newton_exact_skips_dense_jacobian_when_ceiling_is_exceeded(
        self, monkeypatch
    ):
        """Exact Newton must fail predictably before dense finalization would exceed the ceiling."""
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        b = jnp.array([5.0, 7.0])
        materialize_calls = []

        def fake_materialize(_jvp_fn, _x):
            materialize_calls.append(True)
            raise AssertionError("dense Jacobian materialization should be skipped")

        monkeypatch.setattr(_opt, "_materialize_dense_jacobian", fake_materialize)

        def residual(x):
            return A @ x - b

        result = newton_exact(
            residual,
            jnp.zeros(2),
            maxiter=5,
            tol=1e-14,
            max_dense_jacobian_bytes=8,
        )

        assert not materialize_calls
        assert result["jacobian"] is None
        assert result["jacobian_materialized"] is False
        assert result["failure_category"] == "scaling_limit"
        assert result["failure_stage"] == "dense_jacobian_finalization"
        assert result["dense_jacobian_shape"] == (2, 2)
        assert result["dense_jacobian_bytes"] == 32
        assert result["max_dense_jacobian_bytes"] == 8
        assert "max_dense_jacobian_bytes=8" in result["message"]
        assert result["success"]

    def test_newton_exact_traceable_operator_only_path_remains_jittable(self):
        """Traceable exact Newton must remain composable under an enclosing jax.jit."""
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        b = jnp.array([5.0, 7.0])

        def residual(x):
            return A @ x - b

        @jax.jit
        def solve_success(x0):
            return _opt.newton_exact_traceable(
                residual,
                x0,
                maxiter=5,
                tol=1e-14,
            )["success"]

        assert bool(solve_success(jnp.zeros(2)))

    def test_dense_matrix_condition_estimate_matches_diagonal_condition(self):
        """Phase 5 Hager-Higham estimate is JAX-native for dense operators."""
        matrix = jnp.diag(jnp.asarray([0.25, 1.0, 4.0], dtype=jnp.float64))

        estimate = _opt._dense_matrix_condition_estimate(matrix)

        assert float(np.asarray(estimate)) == pytest.approx(16.0)

    def test_hager_higham_inverse_estimate_keeps_best_iteration(self):
        """Hager-Higham estimate must not regress to the final probe only."""

        def solve(x):
            return jnp.asarray([100.0 * x[1], x[0]], dtype=x.dtype)

        def transpose_solve(_signs):
            return jnp.asarray([2.0, 1.0], dtype=jnp.float64)

        estimate = _opt._hager_higham_inverse_1_norm_estimate(
            solve,
            transpose_solve,
            size=2,
            dtype=jnp.float64,
            iterations=3,
        )

        assert float(np.asarray(estimate)) == pytest.approx(50.5)

    def test_dense_matrix_forward_error_gate_rejects_bad_solution(self):
        """Phase 5 Skeel/FERR gate rejects large forward-error risk."""
        matrix = jnp.diag(jnp.asarray([1.0e-8, 1.0], dtype=jnp.float64))
        rhs = jnp.asarray([1.0e-8, 1.0], dtype=jnp.float64)
        good_solution = jnp.linalg.solve(matrix, rhs)
        bad_solution = jnp.asarray([0.0, 1.0], dtype=jnp.float64)

        assert bool(
            np.asarray(
                _opt._dense_matrix_solve_forward_error_success(
                    matrix,
                    good_solution,
                    rhs,
                    tol=1.0e-10,
                )
            )
        )
        assert not bool(
            np.asarray(
                _opt._dense_matrix_solve_forward_error_success(
                    matrix,
                    bad_solution,
                    rhs,
                    tol=1.0e-10,
                )
            )
        )

    def test_eisenstat_walker_tolerance_preserves_strict_newton_cap(self):
        """Phase 5 forcing must not loosen the established Newton solve cap."""
        norm = jnp.asarray(8.0, dtype=jnp.float64)

        linear_tol = _opt._eisenstat_walker_choice2_tolerance(
            norm,
            norm,
            tol=1.0e-14,
        )

        assert float(np.asarray(linear_tol)) == pytest.approx(1.0e-14)

    def test_newton_exact_traceable_backtracks_oversized_newton_step(self):
        """Phase 5 exact traceable Newton backtracks residual-increasing steps."""

        def residual(x):
            return jnp.asarray([x[0] ** 2 - 1.0], dtype=x.dtype)

        x0 = jnp.asarray([0.1], dtype=jnp.float64)
        initial_norm = float(np.linalg.norm(np.asarray(residual(x0))))

        result = _opt.newton_exact_traceable(
            residual,
            x0,
            maxiter=1,
            tol=1e-14,
        )

        final_norm = float(np.linalg.norm(np.asarray(result["residual"])))
        assert final_norm < initial_norm
        assert float(np.asarray(result["x"][0])) < 5.0
        assert int(np.asarray(result["nit"])) == 1

    def test_newton_polish_backtracks_gradient_increasing_step(self):
        """Host Newton polish must reject a full Newton step that worsens gradient."""

        def objective(x):
            return x[0] ** 4 + x[0]

        x0 = jnp.asarray([0.1], dtype=jnp.float64)
        initial_grad_norm = float(jnp.linalg.norm(jax.grad(objective)(x0)))

        result = newton_polish(
            objective,
            x0,
            maxiter=1,
            tol=1e-14,
            materialize_hessian=False,
        )

        final_grad_norm = float(jnp.linalg.norm(result["grad"]))
        assert final_grad_norm <= initial_grad_norm
        assert float(result["x"][0]) > -1.0
        assert int(result["nit"]) == 1


class TestNewtonPolishBoozer:
    """Test Newton polish after BFGS on the Boozer penalty objective."""

    def test_newton_polish_reduces_gradient(self):
        """Newton polish reduces gradient norm below BFGS."""
        case = _build_penalty_problem()
        obj = case["objective"]

        # BFGS first
        bfgs_result = jax_minimize(obj, case["x"], method="bfgs", tol=1e-8, maxiter=200)
        bfgs_grad_norm = float(jnp.linalg.norm(jax.grad(obj)(bfgs_result.x)))

        # Newton polish
        newton_result = newton_polish(obj, bfgs_result.x, maxiter=20, tol=1e-12)
        newton_grad_norm = float(jnp.linalg.norm(newton_result["grad"]))

        assert newton_grad_norm <= bfgs_grad_norm + 1e-15, (
            f"Newton polish did not improve: BFGS grad={bfgs_grad_norm:.3e}, "
            f"Newton grad={newton_grad_norm:.3e}"
        )


class TestOptimizeGFalse:
    """Test the optimize_G=False code path."""

    def test_penalty_with_fixed_G(self):
        """Penalty objective works with G computed from currents."""
        case = _build_penalty_problem(optimize_G=False)
        obj = case["objective"]

        val = float(obj(case["x"]))
        assert val >= 0.0
        # Gradient should have len = len(sdofs) + 1 (iota only)
        grad = jax.grad(obj)(case["x"])
        assert grad.shape == case["x"].shape


class TestToroidalFluxLabel:
    """Test the toroidal flux label constraint path."""

    def test_penalty_with_toroidal_flux(self):
        """Penalty objective works with label_type='toroidal_flux'."""
        case = _build_penalty_problem(label_type="toroidal_flux", targetlabel=0.01)
        val = case["objective"](case["x"])
        assert val.shape == ()
        assert float(val) >= 0.0

        # Gradient should be computable
        grad = jax.grad(case["objective"])(case["x"])
        assert grad.shape == case["x"].shape


class TestLBFGSMethod:
    """Test L-BFGS-B method through the adapter."""

    def test_lbfgs_reduces_objective(self):
        """L-BFGS-B reduces the Boozer penalty objective."""
        case = _build_penalty_problem()
        val_init = float(case["objective"](case["x"]))
        result = jax_minimize(
            case["objective"], case["x"], method="lbfgs", tol=1e-10, maxiter=200
        )
        assert float(result.fun) < val_init


class TestSurfaceArea:
    """Test the JAX area computation."""

    def test_simple_torus_area(self):
        """Area of a simple torus: A = 4pi^2 R r."""
        geometry = _simple_torus_geometry_values(
            R0=1.0,
            r=0.1,
            mpol=1,
            ntor=1,
            nfp=1,
            nphi=32,
            ntheta=32,
        )
        np.testing.assert_allclose(
            geometry["area"],
            geometry["expected_area"],
            rtol=_TORUS_GEOMETRY_RTOL,
        )


class TestAreaLabelPath:
    """Test the Area label constraint through the penalty objective."""

    def test_penalty_with_area_label(self):
        """Penalty objective works with label_type='area'."""
        case = _build_penalty_problem(label_type="area")
        val = case["objective"](case["x"])
        assert val.shape == ()
        assert float(val) >= 0.0

        # Gradient computable
        grad = jax.grad(case["objective"])(case["x"])
        assert grad.shape == case["x"].shape


# ---------------------------------------------------------------------------
# P2 #4: BoozerSurfaceJAX adapter class instantiation tests
# ---------------------------------------------------------------------------

_fake_exact_surface_module = types.ModuleType("simsopt.geo.surfacexyztensorfourier")
_fake_exact_surface_module.SurfaceXYZTensorFourier = _MockSurface


@contextmanager
def _patched_exact_surface_module():
    module_name = "simsopt.geo.surfacexyztensorfourier"
    original_module = sys.modules.get(module_name)
    original_surface_type = _bsj.SurfaceXYZTensorFourier
    _bsj.SurfaceXYZTensorFourier = _MockSurface
    sys.modules[module_name] = _fake_exact_surface_module
    try:
        yield
    finally:
        _bsj.SurfaceXYZTensorFourier = original_surface_type
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


def _make_mock_boozer_surface_mixed_quad(nphi=8, ntheta=8, mpol=1, ntor=1, nfp=1):
    """BoozerSurfaceJAX with mixed-quadrature coils (no simsoptpp needed)."""
    R0, r = 1.0, 0.1
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])

    bs = _MockBiotSavart(_make_mixed_quad_mock_coils())
    surf = _MockSurface(sdofs, mpol, ntor, nfp, False, qphi, qtheta)
    label = _PlumbingVolumeLabel(surf)
    target = 2.0 * np.pi**2 * R0 * r**2

    return BoozerSurfaceJAX(bs, surf, label, target, constraint_weight=1.0)


class _MockToroidalFluxLabel:
    def __init__(self, surface):
        self.surface = surface

    def J(self):
        return 0.0


def _make_mock_toroidal_flux_boozer_surface(nphi=8, ntheta=8, mpol=1, ntor=1, nfp=1):
    R0, r = 1.0, 0.1
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])

    bs = _MockBiotSavart(_make_mock_coils())
    surf = _MockSurface(sdofs, mpol, ntor, nfp, False, qphi, qtheta)
    return BoozerSurfaceJAX(
        bs,
        surf,
        _MockToroidalFluxLabel(surf),
        0.0,
        constraint_weight=1.0,
    )


def _make_spec_only_biotsavart(coils):
    class _SpecOnlyBiotSavart(_MockBiotSavart):
        def __init__(self, grouped_coils):
            super().__init__(grouped_coils)
            self._coil_spec = grouped_coil_set_spec_from_lists(
                [coil.curve.gamma() for coil in grouped_coils],
                [coil.curve.gammadash() for coil in grouped_coils],
                [coil.current.get_value() for coil in grouped_coils],
            )
            del self._coils

        def coil_set_spec(self):
            return self._coil_spec

    return _SpecOnlyBiotSavart(coils)


def _make_grouped_extractor_only_biotsavart(coils):
    class _GroupedExtractorOnlyBiotSavart(_MockBiotSavart):
        coil_set_spec = None

        def __init__(self, grouped_coils):
            super().__init__(grouped_coils)
            self._coil_spec = grouped_coil_set_spec_from_lists(
                [coil.curve.gamma() for coil in grouped_coils],
                [coil.curve.gammadash() for coil in grouped_coils],
                [coil.current.get_value() for coil in grouped_coils],
            )
            del self._coils

        def _extract_coil_data_grouped(self):
            return grouped_field_data_from_spec(self._coil_spec)

    return _GroupedExtractorOnlyBiotSavart(coils)


def _make_basic_mock_surface_and_label():
    surface = _MockSurface(
        np.zeros(27),
        1,
        1,
        1,
        False,
        np.linspace(0.0, 1.0, 3, endpoint=False),
        np.linspace(0.0, 1.0, 3, endpoint=False),
    )
    return surface, _PlumbingVolumeLabel(surface)


class TestBoozerSurfaceJAXClass:
    """Test the adapter class instantiation and run_code orchestration."""

    def test_instantiation(self):
        """BoozerSurfaceJAX can be instantiated with mock objects."""
        booz = _make_mock_boozer_surface()
        assert booz.boozer_type == "ls"
        assert booz.label_type == "volume"
        assert booz.need_to_run_code is True

    def test_instantiation_accepts_spec_only_biotsavart(self):
        """The grouped-coil spec path must not require a legacy ``_coils`` list."""

        coils = _make_mixed_quad_mock_coils()
        bs = _make_spec_only_biotsavart(coils)
        surf, label = _make_basic_mock_surface_and_label()
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
        )

        assert isinstance(booz.coil_set_spec, GroupedCoilSetSpec)
        np.testing.assert_allclose(
            np.asarray(booz.coil_currents),
            np.asarray([coil.current.get_value() for coil in coils]),
        )

    def test_instantiation_caches_surface_runtime_state_and_dofs(self):
        """The JAX solver keeps an explicit cached copy of the surface DOFs."""
        booz = _make_mock_boozer_surface()
        expected_dofs = np.asarray(booz.surface.get_dofs(), dtype=np.float64)

        def _unexpected_get_dofs():
            raise AssertionError("live surface get_dofs() should not be queried")

        booz.surface.get_dofs = _unexpected_get_dofs

        np.testing.assert_allclose(
            np.asarray(booz._get_cached_surface_dofs(), dtype=np.float64),
            expected_dofs,
        )
        assert booz.surface_runtime_state.mpol == booz.surface.mpol
        assert booz.surface_runtime_state.ntor == booz.surface.ntor

    @pytest.mark.parametrize(
        ("optimizer_backend", "expected_algorithm"),
        [
            ("scipy", "quasi-newton"),
            ("ondevice", "quasi-newton"),
        ],
    )
    def test_instantiation_defaults_least_squares_algorithm_from_backend(
        self,
        optimizer_backend,
        expected_algorithm,
    ):
        bs = _MockBiotSavart(_make_mock_coils())
        surf, label = _make_basic_mock_surface_and_label()

        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={"optimizer_backend": optimizer_backend},
        )

        assert booz.options["least_squares_algorithm"] == expected_algorithm

    @pytest.mark.parametrize(
        ("optimizer_backend", "expected_materialize"),
        [
            ("scipy", True),
            ("ondevice", True),
        ],
    )
    def test_instantiation_defaults_materialize_dense_linearization_from_backend(
        self,
        optimizer_backend,
        expected_materialize,
    ):
        """Dense finalization is enabled by default and capped by matrix bytes."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf, label = _make_basic_mock_surface_and_label()

        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={"optimizer_backend": optimizer_backend},
        )

        assert booz.options["materialize_dense_linearization"] is expected_materialize

    def test_backend_mutation_refreshes_implicit_dense_linearization_default(self):
        """Backend mutation keeps the implicit byte-capped dense default."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf, label = _make_basic_mock_surface_and_label()
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
        )

        assert booz.options["optimizer_backend"] == "scipy"
        assert booz.options["materialize_dense_linearization"] is True

        booz.options["optimizer_backend"] = "ondevice"

        assert booz.options["materialize_dense_linearization"] is True

        booz.options["optimizer_backend"] = "scipy"

        assert booz.options["materialize_dense_linearization"] is True

    def test_backend_mutation_preserves_explicit_dense_linearization_request(self):
        """Dense artifacts remain available when materialization is explicit."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf, label = _make_basic_mock_surface_and_label()
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={"materialize_dense_linearization": True},
        )

        booz.options["optimizer_backend"] = "ondevice"

        assert booz.options["materialize_dense_linearization"] is True

    @pytest.mark.parametrize(
        ("backend_mode", "expected_optimizer_backend", "expected_algorithm"),
        [
            (None, "scipy", "quasi-newton"),
            ("jax_cpu_parity", "ondevice", "quasi-newton"),
        ],
    )
    def test_instantiation_defaults_optimizer_backend_from_runtime_contract(
        self,
        monkeypatch,
        request,
        backend_mode,
        expected_optimizer_backend,
        expected_algorithm,
    ):
        if backend_mode is not None:
            enable_non_strict_jax_backend(monkeypatch, request, mode=backend_mode)

        bs = _MockBiotSavart(_make_mock_coils())
        surf, label = _make_basic_mock_surface_and_label()

        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
        )

        assert booz.options["optimizer_backend"] == expected_optimizer_backend
        assert booz.options["least_squares_algorithm"] == expected_algorithm

    def test_instantiation_applies_jax_default_before_private_ls_option_validation(
        self,
        monkeypatch,
        request,
    ):
        enable_non_strict_jax_backend(monkeypatch, request, mode="jax_cpu_parity")

        bs = _MockBiotSavart(_make_mock_coils())
        surf, label = _make_basic_mock_surface_and_label()

        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={"force_ondevice_limited_memory": True},
        )

        assert booz.options["optimizer_backend"] == "ondevice"
        assert booz.options["force_ondevice_limited_memory"] is True

    def test_instantiation_rejects_grouped_extractor_only_adapter(self):
        """BoozerSurfaceJAX now requires explicit ``coil_set_spec()`` state."""
        coils = _make_mixed_quad_mock_coils()
        bs = _make_grouped_extractor_only_biotsavart(coils)
        surf, label = _make_basic_mock_surface_and_label()

        with pytest.raises(AttributeError, match=_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
            )

    def test_instantiation_rejects_hidden_coils_list_adapter(self):
        """Raw ``_coils`` compatibility extraction is no longer supported."""
        bs = _make_legacy_coils_list_biotsavart(_make_legacy_spec_capable_coils())
        surf, label = _make_basic_mock_surface_and_label()

        with pytest.raises(AttributeError, match=_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
            )

    def test_constructor_rejects_spoof_surface_names(self):
        spoof_surface = type(
            "SurfaceXYZFourier",
            (),
            {
                "mpol": 1,
                "ntor": 1,
                "nfp": 1,
                "stellsym": False,
                "quadpoints_phi": np.linspace(0.0, 1.0, 3, endpoint=False),
                "quadpoints_theta": np.linspace(0.0, 1.0, 3, endpoint=False),
            },
        )()
        bs = _MockBiotSavart(_make_mock_coils())

        with pytest.raises(
            TypeError, match="Unsupported BoozerSurfaceJAX surface type"
        ):
            BoozerSurfaceJAX(
                bs,
                spoof_surface,
                _PlumbingVolumeLabel(spoof_surface),
                1.0,
                constraint_weight=1.0,
            )

    def test_constructor_rejects_unknown_explicit_surface_kind(self):
        surf, label = _make_basic_mock_surface_and_label()
        surf.jax_surface_kind = "tensor"
        bs = _MockBiotSavart(_make_mock_coils())

        with pytest.raises(ValueError, match="Unsupported BoozerSurfaceJAX"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
            )

    def test_surface_geometry_rejects_unknown_surface_kind(self):
        with pytest.raises(ValueError, match="Unsupported Boozer JAX surface_kind"):
            _bsj._surface_geometry_from_dofs(
                jnp.zeros(27, dtype=jnp.float64),
                jnp.linspace(0.0, 1.0, 3, endpoint=False),
                jnp.linspace(0.0, 1.0, 3, endpoint=False),
                1,
                1,
                1,
                False,
                None,
                surface_kind="tensor",
            )

    def test_spec_only_biotsavart_supports_G_none_ls_path(self):
        """Spec-driven grouped-field state should work without a legacy ``_coils`` list."""
        coils = _make_mixed_quad_mock_coils()
        bs = _make_spec_only_biotsavart(coils)
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
        )

        result = booz.run_code(iota=0.2, G=None)

        assert result is not None
        assert result["type"] == "ls"

    def test_run_code_rejects_G_none_with_free_currents(self):
        booz = _make_mock_boozer_surface_with_free_currents()

        with pytest.raises(ValueError, match="fixed coil currents when G=None"):
            booz.run_code(iota=0.2, G=None)

    def test_none_G_coil_gradient_callback_rejects_free_currents(self):
        callback = lambda *_args, **_kwargs: None
        booz = _make_mock_boozer_surface_with_free_currents()
        guarded = guard_none_G_coil_gradient_callback(
            callback,
            biotsavart=booz.biotsavart,
            component="BoozerSurfaceJAX",
            coil_attrs=("coils",),
            G_provided=False,
        )

        with pytest.raises(ValueError, match="fixed coil currents when G=None"):
            guarded(None)

    def test_none_G_coil_gradient_callback_allows_explicit_G(self):
        callback = lambda *_args, **_kwargs: None
        booz = _make_mock_boozer_surface_with_free_currents()
        guarded = guard_none_G_coil_gradient_callback(
            callback,
            biotsavart=booz.biotsavart,
            component="BoozerSurfaceJAX",
            coil_attrs=("coils",),
            G_provided=True,
        )

        assert guarded is callback

    def test_reference_ls_reuses_cached_scipy_value_and_grad_transform(
        self,
        monkeypatch,
    ):
        """Repeated SciPy reference solves should not rebuild the JIT transform."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "scipy"
        booz.options["least_squares_algorithm"] = "quasi-newton"
        booz.options["limited_memory"] = True
        booz.options["verbose"] = False

        state = _patch_counting_scipy_minimize(monkeypatch)

        booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=-0.3,
            G=None,
            maxiter=1,
            verbose=False,
        )
        first_call_jit_count = state["jit_call_count"]
        assert first_call_jit_count > 0

        booz.recompute_bell()
        booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=-0.3,
            G=None,
            maxiter=1,
            verbose=False,
        )

        assert state["jit_call_count"] == first_call_jit_count

    def test_lbfgs_public_api_uses_options_default_when_limited_memory_omitted(
        self,
        monkeypatch,
    ):
        """Omitted limited_memory should preserve the configured JAX default."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "scipy"
        booz.options["limited_memory"] = False
        captured_methods = []

        def fake_reference_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad=False,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options, progress_callback
            assert value_and_grad is True
            captured_methods.append(method)
            return _successful_minimize_result(x0)

        monkeypatch.setattr(_bsj, "reference_minimize", fake_reference_minimize)

        res = booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=0.3,
            G=0.05,
            verbose=False,
        )

        _assert_result_schema(res, _PUBLIC_LBFGS_RESULT_SCHEMA)
        assert captured_methods == ["bfgs"]
        assert res["optimizer_method"] == "bfgs"

        booz.recompute_bell()
        res = booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=0.3,
            G=0.05,
            verbose=False,
            limited_memory=True,
        )

        _assert_result_schema(res, _PUBLIC_LBFGS_RESULT_SCHEMA)
        assert captured_methods == ["bfgs", "lbfgs"]
        assert res["optimizer_method"] == "lbfgs"

    @pytest.mark.parametrize(
        ("limited_memory", "expected_method"),
        [(False, "bfgs"), (True, "lbfgs")],
    )
    def test_host_scipy_quasi_newton_uses_cpu_ordered_value_grad_without_trace(
        self,
        monkeypatch,
        limited_memory,
        expected_method,
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "scipy"
        booz.options["record_scipy_callback_trace"] = False
        calls = []

        def cpu_ordered_value_and_grad(*args, **kwargs):
            calls.append((args, kwargs))
            return lambda x: (jnp.asarray(0.0, dtype=x.dtype), jnp.zeros_like(x))

        def default_objective(*_args, **_kwargs):
            raise AssertionError("host SciPy quasi-Newton must use value_and_grad")

        def fake_reference_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad=False,
            progress_callback=None,
        ):
            del tol, maxiter, options, progress_callback
            assert method == expected_method
            assert value_and_grad is True
            value, grad = fun(x0)
            assert float(np.asarray(value)) == 0.0
            np.testing.assert_allclose(np.asarray(grad), np.zeros_like(np.asarray(x0)))
            return _successful_minimize_result(x0)

        monkeypatch.setattr(
            booz,
            "_make_penalty_value_and_grad_cpu_ordered_with",
            cpu_ordered_value_and_grad,
        )
        monkeypatch.setattr(booz, "_make_penalty_objective_with", default_objective)
        monkeypatch.setattr(_bsj, "reference_minimize", fake_reference_minimize)

        res = booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=0.3,
            G=0.05,
            verbose=False,
            limited_memory=limited_memory,
        )

        assert len(calls) == 1
        assert res["optimizer_method"] == expected_method

    def test_public_solver_signatures_do_not_expose_vectorize(self):
        for method_name in (
            "minimize_boozer_penalty_constraints_LBFGS",
            "minimize_boozer_penalty_constraints_newton",
            "minimize_boozer_exact_constraints_newton",
        ):
            params = inspect.signature(
                getattr(BoozerSurfaceJAX, method_name)
            ).parameters
            assert "vectorize" not in params

    @pytest.mark.parametrize(
        ("explicit_materialize", "expected_materialize"),
        [
            (None, True),
            (True, True),
        ],
    )
    def test_public_ls_api_routes_ondevice_lm(
        self,
        monkeypatch,
        explicit_materialize,
        expected_materialize,
    ):
        """The restored public LS method should route to the ondevice LM lane."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        if explicit_materialize is not None:
            booz.options["materialize_dense_linearization"] = explicit_materialize
        captured = {}

        def fake_target_least_squares(
            residual_fn,
            x0,
            *,
            method,
            tol,
            maxiter,
            options=None,
            callback=None,
            progress_callback=None,
        ):
            del residual_fn, tol, maxiter, callback, progress_callback
            captured["method"] = method
            captured["options"] = dict(options or {})
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                residual=jnp.zeros_like(flat_x0),
                jac=jnp.zeros_like(flat_x0),
                residual_jacobian=jnp.eye(flat_x0.size, dtype=flat_x0.dtype),
                success=True,
            )

        monkeypatch.setattr(_bsj, "target_least_squares", fake_target_least_squares)

        res = booz.minimize_boozer_penalty_constraints_ls(
            iota=0.3,
            G=0.05,
            method="lm",
        )

        _assert_result_schema(res, _PUBLIC_LS_LM_RESULT_SCHEMA)
        assert captured["method"] == "lm-ondevice"
        assert (
            captured["options"]["materialize_dense_linearization"]
            is expected_materialize
        )
        assert (
            captured["options"]["max_dense_linearization_bytes"]
            == booz.options["max_dense_linearization_bytes"]
        )
        assert res["optimizer_method"] == "lm-ondevice"
        assert res["success"] is True
        assert booz.need_to_run_code is False

    def test_public_ls_api_accepts_weight_inv_modB_override(self, monkeypatch):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "scipy"
        captured = {}

        def fake_make_penalty_residual_with(
            optimize_G,
            weight_inv_modB,
            constraint_weight=None,
            coil_set_spec=None,
            coil_arrays=None,
            *,
            hostify_inputs=True,
        ):
            del (
                optimize_G,
                constraint_weight,
                coil_set_spec,
                coil_arrays,
                hostify_inputs,
            )
            captured["weight_inv_modB"] = weight_inv_modB
            return lambda x: jnp.zeros_like(x)

        def fake_reference_least_squares(
            residual_fn,
            x0,
            *,
            method,
            tol,
            maxiter,
            options=None,
            callback=None,
            progress_callback=None,
        ):
            del residual_fn, method, tol, maxiter, options, callback, progress_callback
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                residual=jnp.zeros_like(flat_x0),
                jac=jnp.zeros_like(flat_x0),
                residual_jacobian=jnp.eye(flat_x0.size, dtype=flat_x0.dtype),
                success=True,
            )

        monkeypatch.setattr(
            booz, "_make_penalty_residual_with", fake_make_penalty_residual_with
        )
        monkeypatch.setattr(
            _bsj, "reference_least_squares", fake_reference_least_squares
        )

        res = booz.minimize_boozer_penalty_constraints_ls(
            iota=0.3,
            G=0.05,
            method="lm",
            weight_inv_modB=False,
        )

        assert captured["weight_inv_modB"] is False
        assert res["weight_inv_modB"] is False

    def test_public_ls_api_rejects_invalid_backend_after_options_mutation(self):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "bogus"

        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            booz.minimize_boozer_penalty_constraints_ls(
                iota=0.3,
                G=0.05,
                method="lm",
            )

    def test_public_manual_ls_api_supports_baseline_demo_sequence(self, monkeypatch):
        """The restored public LS API should support the old demo call pattern."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "scipy"
        lbfgs_target = booz._pack_decision_vector(0.25, 0.04) - 0.05

        def fake_reference_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad=False,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options, progress_callback
            assert value_and_grad is True
            flat_target, _ = ravel_pytree(lbfgs_target)
            return types.SimpleNamespace(
                x=lbfgs_target,
                fun=0.0,
                jac=jnp.zeros_like(flat_target),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        monkeypatch.setattr(_bsj, "reference_minimize", fake_reference_minimize)

        res_lbfgs = booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=0.3,
            G=0.05,
            verbose=False,
            limited_memory=True,
        )
        assert res_lbfgs["optimizer_method"] == "lbfgs"

        booz.recompute_bell()
        manual_target = (
            booz._pack_decision_vector(res_lbfgs["iota"], res_lbfgs["G"]) - 0.1
        )
        monkeypatch.setattr(
            booz,
            "_make_penalty_residual_with",
            lambda *args, **kwargs: lambda x: x - manual_target,
        )

        res_manual = booz.minimize_boozer_penalty_constraints_ls(
            iota=res_lbfgs["iota"],
            G=res_lbfgs["G"],
            method="manual",
            maxiter=40,
            tol=1e-8,
        )

        _assert_result_schema(res_manual, _PUBLIC_LS_MANUAL_RESULT_SCHEMA)
        assert res_manual["optimizer_method"] == "manual"
        assert res_manual["success"] is True
        np.testing.assert_allclose(
            np.asarray(res_manual["residual"]),
            np.zeros_like(np.asarray(manual_target)),
            atol=1e-6,
        )

    def test_public_manual_ls_api_increases_damping_after_worsening_trial(self):
        """The manual LS compatibility loop must not shrink damping on rejected steps."""
        booz = _make_mock_boozer_surface()
        result = booz._run_manual_penalty_least_squares(
            lambda x: jnp.asarray([x[0] ** 2 - 1.0], dtype=x.dtype),
            jnp.asarray([0.1], dtype=jnp.float64),
            tol=1e-10,
            maxiter=80,
        )

        assert result["success"] is True
        assert result["nit"] > 1
        assert abs(abs(float(np.asarray(result["x"])[0])) - 1.0) < 1e-6

    def test_public_manual_ls_api_matches_legacy_manual_linear_contract(self):
        """JAX manual LS must match the legacy damped Gauss-Newton linear contract."""

        class _ManualSurface:
            def __init__(self):
                self._dofs = np.asarray([0.35, -0.25], dtype=float)

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                self._dofs = np.asarray(dofs, dtype=float).copy()

        surface = _ManualSurface()
        cpu_booz = LegacyBoozerSurface.__new__(LegacyBoozerSurface)
        cpu_booz.need_to_run_code = True
        cpu_booz.surface = surface
        x0 = np.concatenate([surface.get_dofs(), np.asarray([0.3, 0.05])])
        matrix = np.asarray(
            [
                [1.8, 0.2, -0.1, 0.05],
                [0.1, 1.6, 0.25, -0.2],
                [-0.05, 0.15, 1.7, 0.1],
                [0.2, -0.1, 0.05, 1.5],
            ],
            dtype=float,
        )
        target = np.asarray([0.1, -0.2, 0.3, 0.05], dtype=float)
        rhs = matrix @ target

        def residual_and_jacobian(x, constraint_weight, optimize_G, weight_inv_modB):
            del constraint_weight, weight_inv_modB
            assert optimize_G is True
            return matrix @ np.asarray(x, dtype=float) - rhs, matrix

        cpu_booz._get_residual_vector_and_jacobian = residual_and_jacobian
        cpu_result = LegacyBoozerSurface.minimize_boozer_penalty_constraints_ls(
            cpu_booz,
            tol=1e-12,
            maxiter=80,
            constraint_weight=1.0,
            iota=0.3,
            G=0.05,
            method="manual",
        )

        jax_booz = _make_mock_boozer_surface()
        matrix_jax = jnp.asarray(matrix, dtype=jnp.float64)
        rhs_jax = jnp.asarray(rhs, dtype=jnp.float64)
        jax_result = jax_booz._run_manual_penalty_least_squares(
            lambda x: matrix_jax @ x - rhs_jax,
            jnp.asarray(x0, dtype=jnp.float64),
            tol=1e-12,
            maxiter=80,
        )

        assert bool(cpu_result["success"]) is True
        assert jax_result["success"] is True
        np.testing.assert_allclose(
            np.asarray(jax_result["x"]),
            np.concatenate(
                [
                    np.asarray(cpu_result["s"].get_dofs(), dtype=float),
                    np.asarray([cpu_result["iota"], cpu_result["G"]], dtype=float),
                ]
            ),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jax_result["residual"]),
            np.asarray(cpu_result["residual"], dtype=float),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jax_result["gradient"]),
            np.asarray(cpu_result["gradient"], dtype=float),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("stellsym", [True, False])
    @pytest.mark.parametrize("optimize_G", [True, False])
    def test_public_exact_constraints_newton_restores_cpu_api(
        self,
        monkeypatch,
        stellsym,
        optimize_G,
    ):
        """The restored exact-constraints method should expose the CPU-shaped API."""
        booz = _make_mock_boozer_surface_exact(stellsym=stellsym)
        initial_G = 0.05 if optimize_G else None
        x0 = booz._pack_decision_vector(0.3, initial_G)
        xl0 = jnp.concatenate([x0, jnp.array([0.0, 0.0], dtype=x0.dtype)])
        shift = jnp.linspace(0.01, 0.01 * xl0.size, xl0.size, dtype=xl0.dtype)
        if stellsym:
            shift = shift.at[-1].set(0.0)
        target = xl0 - shift

        monkeypatch.setattr(
            booz,
            "_make_exact_constraints_residual_with",
            lambda *args, **kwargs: lambda xl: xl - target,
        )

        res = booz.minimize_boozer_exact_constraints_newton(
            iota=0.3,
            G=initial_G,
            maxiter=5,
            tol=1e-10,
        )

        _assert_result_schema(res, _PUBLIC_EXACT_CONSTRAINTS_RESULT_SCHEMA)
        assert res["success"] is True
        assert "jacobian" in res
        assert "residual" in res
        if optimize_G:
            assert isinstance(res["G"], float)
        else:
            assert res["G"] is None
        if stellsym:
            assert res["lm"] == pytest.approx(float(np.asarray(target[-2])))
        else:
            np.testing.assert_allclose(
                np.asarray(res["lm"]),
                np.asarray(target[-2:]),
                atol=1e-12,
            )

    def test_public_exact_constraints_newton_nonstellsym_stays_native_without_root(
        self,
        monkeypatch,
    ):
        booz = _make_mock_boozer_surface_exact(stellsym=False)
        initial_G = 0.05
        x0 = booz._pack_decision_vector(0.3, initial_G)
        xl0 = jnp.concatenate([x0, jnp.array([0.0, 0.0], dtype=x0.dtype)])
        target = xl0 - 0.02

        monkeypatch.setattr(
            booz,
            "_make_exact_constraints_residual_with",
            lambda *args, **kwargs: lambda xl: xl - target,
        )
        monkeypatch.setattr(
            _bsj,
            "root",
            lambda *_args, **_kwargs: pytest.fail(
                "nonstellsym exact Newton must stay on the native JAX solve path"
            ),
            raising=False,
        )

        res = booz.minimize_boozer_exact_constraints_newton(
            iota=0.3,
            G=initial_G,
            maxiter=5,
            tol=1e-10,
        )

        assert res["success"] is True
        np.testing.assert_allclose(np.asarray(res["residual"]), 0.0, atol=1e-12)

    def test_public_exact_constraints_newton_nonstellsym_uses_full_jacobian_solve(
        self,
        monkeypatch,
    ):
        booz = _make_mock_boozer_surface_exact(stellsym=False)
        initial_G = 0.05
        x0 = booz._pack_decision_vector(0.3, initial_G)
        xl0 = jnp.concatenate([x0, jnp.array([0.0, 0.0], dtype=x0.dtype)])
        target = xl0 - jnp.linspace(0.01, 0.01 * xl0.size, xl0.size, dtype=xl0.dtype)
        solve_calls = []
        original_solve = _bsj.jnp.linalg.solve

        monkeypatch.setattr(
            booz,
            "_make_exact_constraints_residual_with",
            lambda *args, **kwargs: lambda xl: xl - target,
        )

        def recording_solve(matrix, rhs):
            solve_calls.append((matrix.shape, rhs.shape))
            return original_solve(matrix, rhs)

        monkeypatch.setattr(_bsj.jnp.linalg, "solve", recording_solve)

        res = booz.minimize_boozer_exact_constraints_newton(
            iota=0.3,
            G=initial_G,
            maxiter=5,
            tol=1e-10,
        )

        assert res["success"] is True
        np.testing.assert_allclose(np.asarray(res["residual"]), 0.0, atol=1e-12)
        assert solve_calls == [((xl0.size, xl0.size), (xl0.size,))]

    def test_public_newton_api_routes_without_legacy_vectorize_kwarg(self, monkeypatch):
        booz = _make_mock_boozer_surface()
        target = booz._pack_decision_vector(0.3, 0.05) - 0.01
        captured = {}

        def fake_run_newton_polish_for_method(
            method,
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            materialize_hessian=True,
            max_dense_hessian_bytes=None,
            progress_callback=None,
            objective_args=(),
        ):
            del (
                obj_fn,
                maxiter,
                tol,
                stab,
                materialize_hessian,
                max_dense_hessian_bytes,
                progress_callback,
                objective_args,
            )
            captured["method"] = method
            np.testing.assert_allclose(
                np.asarray(x0),
                np.asarray(booz._pack_decision_vector(0.3, 0.05)),
            )
            return {
                "x": target,
                "fun": 0.0,
                "grad": jnp.zeros_like(target),
                "hessian": jnp.eye(target.size, dtype=target.dtype),
                "nit": 2,
                "success": True,
            }

        monkeypatch.setattr(
            booz,
            "_run_newton_polish_for_method",
            fake_run_newton_polish_for_method,
        )

        res = booz.minimize_boozer_penalty_constraints_newton(
            iota=0.3,
            G=0.05,
            verbose=False,
        )

        _assert_result_schema(res, _PUBLIC_NEWTON_RESULT_SCHEMA)
        assert captured["method"] == "bfgs"
        assert res["success"] is True

    def test_stale_bfgs_method_rejected(self):
        """The removed bfgs_method option must fail fast."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        with pytest.raises(ValueError, match="bfgs_method.*removed"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
                options={"bfgs_method": "bfgs"},
            )

    def test_unknown_option_rejected(self):
        """Unknown constructor options must fail fast instead of being ignored."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        with pytest.raises(ValueError, match="Unknown BoozerSurfaceJAX option"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
                options={"optimizer_backend_typo": "ondevice"},
            )

    def test_parity_mode_rejects_damped_boozer_linearization(self, monkeypatch):
        monkeypatch.setattr(_bsj, "is_parity_mode", lambda: True)

        with pytest.raises(ValueError, match="parity mode requires newton_stab=0.0"):
            _bsj._normalize_solver_options({"newton_stab": 1.0e-3}, "ls")

    def test_private_options_rejected_with_scipy_backend(self):
        """Private optimizer options must be rejected when backend is scipy."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        with pytest.raises(ValueError, match="require optimizer_backend"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
                options={"line_search_maxiter": 11},
            )

    def test_scipy_limited_memory_options_are_accepted(self):
        """SciPy limited-memory solves must keep their public L-BFGS tuning knobs."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={
                "limited_memory": True,
                "maxcor": 12,
                "ftol": 1e-12,
                "maxfun": 99,
                "maxls": 13,
            },
        )

        assert booz.options["limited_memory"] is True
        assert booz.options["maxcor"] == 12
        assert booz.options["ftol"] == pytest.approx(1e-12)
        assert booz.options["maxfun"] == 99
        assert booz.options["maxls"] == 13

    def test_removed_hybrid_backend_is_rejected(self):
        """The removed hybrid backend must no longer be accepted at construction."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
                options={
                    "optimizer_backend": "hybrid",
                },
            )

    def test_optimizer_tuning_options_are_accepted(self):
        """Private optimizer tuning knobs accepted with non-scipy backend."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={
                "optimizer_backend": "ondevice",
                "line_search_maxiter": 11,
                "maxcor": 12,
                "ftol": 1e-12,
                "maxfun": 99,
                "maxgrad": 101,
                "maxls": 13,
            },
        )

        assert booz.options["line_search_maxiter"] == 11
        assert booz.options["maxcor"] == 12
        assert booz.options["ftol"] == pytest.approx(1e-12)
        assert booz.options["maxfun"] == 99
        assert booz.options["maxgrad"] == 101
        assert booz.options["maxls"] == 13

    @pytest.mark.parametrize(
        ("optimizer_backend", "limited_memory", "expected_method"),
        [
            ("scipy", False, "bfgs"),
            ("scipy", True, "lbfgs"),
            ("ondevice", False, "bfgs-ondevice"),
            ("ondevice", True, "lbfgs-ondevice"),
            ("scipy-jax", False, "lbfgs-scipy-jax"),
            ("scipy-jax", True, "lbfgs-scipy-jax"),
        ],
    )
    def test_resolve_ls_optimizer_method_contract(
        self, optimizer_backend, limited_memory, expected_method
    ):
        """LS backend contract must route to the expected optimizer method."""
        assert (
            resolve_optimizer_backend_method(
                optimizer_backend,
                limited_memory=limited_memory,
            )
            == expected_method
        )

    def test_resolve_ls_optimizer_method_rejects_invalid_backend(self):
        """Invalid backend names must fail instead of silently falling through."""
        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            resolve_optimizer_backend_method("bogus", limited_memory=False)

    @pytest.mark.parametrize(
        (
            "optimizer_backend",
            "limited_memory",
            "least_squares_algorithm",
            "expected_method",
        ),
        [
            ("scipy", False, "quasi-newton", "bfgs"),
            ("scipy", True, "quasi-newton", "lbfgs"),
            ("ondevice", False, "quasi-newton", "bfgs-ondevice"),
            ("ondevice", False, "lm", "lm-ondevice"),
            ("scipy-jax", False, "quasi-newton", "lbfgs-scipy-jax"),
            ("scipy", False, "lm", "lm"),
        ],
    )
    def test_resolve_least_squares_optimizer_method_contract(
        self,
        optimizer_backend,
        limited_memory,
        least_squares_algorithm,
        expected_method,
    ):
        assert (
            resolve_least_squares_optimizer_method(
                optimizer_backend,
                limited_memory=limited_memory,
                least_squares_algorithm=least_squares_algorithm,
            )
            == expected_method
        )

    def test_resolve_least_squares_optimizer_method_rejects_invalid_backend(self):
        with pytest.raises(
            ValueError,
            match="optimizer_backend must be one of",
        ):
            resolve_least_squares_optimizer_method(
                "bogus",
                limited_memory=False,
                least_squares_algorithm="lm",
            )

    def test_resolve_least_squares_optimizer_method_rejects_limited_memory_lm(self):
        with pytest.raises(
            ValueError,
            match="least_squares_algorithm='lm'.*limited_memory=True",
        ):
            resolve_least_squares_optimizer_method(
                "ondevice",
                limited_memory=True,
                least_squares_algorithm="lm",
            )

    @pytest.mark.parametrize("optimizer_backend", ["ondevice", "scipy-jax"])
    def test_require_target_backend_x64_rejects_disabled_float64(
        self, monkeypatch, optimizer_backend
    ):
        """Target-lane backends must fail fast when x64 is disabled."""
        monkeypatch.setattr(_opt, "_x64_enabled", lambda: False)

        with pytest.raises(
            RuntimeError,
            match=rf"optimizer_backend='{optimizer_backend}'.*requires jax_enable_x64=True",
        ):
            require_target_backend_x64(optimizer_backend)

    def test_newton_polish_returns_stabilized_hessian_when_requested(self):
        """Returned Hessian must match the stabilized linear system."""
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        b = jnp.array([1.0, 2.0])
        stab = 0.25

        def obj(x):
            return 0.5 * x @ A @ x - b @ x

        result = newton_polish(obj, jnp.zeros(2), maxiter=5, tol=1e-14, stab=stab)
        np.testing.assert_allclose(
            result["hessian"],
            np.asarray(A + stab * jnp.eye(2)),
            atol=1e-12,
        )

    def test_newton_polish_dense_hessian_matches_jacfwd_grad_candidate(self):
        """Benchmark candidates must preserve the current dense Hessian value."""

        def obj(x):
            return (
                0.5 * jnp.dot(x, jnp.array([2.0, 3.0, 4.0]) * x)
                + 0.1 * x[0] * x[1]
                + 1.0e-3 * jnp.sum(jnp.sin(x) ** 2)
            )

        x = jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float64)
        hvp_fn = _opt._hessian_vector_product_fn(obj)

        current = _opt._materialize_dense_hessian(hvp_fn, x)
        candidate = jax.jacfwd(jax.grad(obj))(x)

        np.testing.assert_allclose(
            np.asarray(current),
            np.asarray(candidate),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_newton_polish_dense_hessian_symmetrizes_numerical_asymmetry(self):
        """Dense-compatible Hessian artifacts mirror the upper triangle.

        Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §5.1,
        symmetrization preserves the upper triangle of the raw HVP-built
        Hessian and copies it into the lower triangle so the LU input is
        bit-symmetric without averaging.
        """
        operator = jnp.asarray(
            [
                [2.0, 0.5 + 1.0e-8, -0.25],
                [0.5 - 2.0e-8, 3.0, 0.75 + 3.0e-8],
                [-0.25, 0.75 - 1.0e-8, 4.0],
            ],
            dtype=jnp.float64,
        )

        def hvp_fn(_x, v):
            return operator @ v

        dense = _opt._materialize_dense_hessian(
            hvp_fn,
            jnp.zeros(3, dtype=jnp.float64),
        )
        expected = jnp.triu(operator) + jnp.triu(operator, 1).T

        np.testing.assert_array_equal(np.asarray(dense), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(dense), np.asarray(dense.T))

    def test_recompute_bell(self):
        """recompute_bell sets the dirty flag."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.recompute_bell()
        assert booz.need_to_run_code is True

    def test_pack_unpack_roundtrip(self):
        """_pack and _unpack are inverses."""
        booz = _make_mock_boozer_surface()
        x = booz._pack_decision_vector(0.3, 1.5)
        sdofs, iota, G = booz._unpack_decision_vector(x, optimize_G=True)
        np.testing.assert_allclose(iota, 0.3)
        np.testing.assert_allclose(G, 1.5)

    def test_run_code_ls_converges(self):
        """run_code() LS path converges on the mock problem."""
        booz = _make_mock_boozer_surface()
        res = booz.run_code(iota=0.3, G=0.05)
        assert res is not None
        _assert_result_schema(res, _PUBLIC_NEWTON_RESULT_SCHEMA)
        assert res["type"] == "ls"
        assert "residual" in res
        assert "jacobian" in res
        assert "hessian" in res
        assert "PLU" in res
        assert "vjp" in res

    @pytest.mark.parametrize(
        ("optimizer_backend", "limited_memory", "expected_method"),
        [
            ("scipy", False, "bfgs"),
            ("scipy", True, "lbfgs"),
            ("ondevice", False, "bfgs-ondevice"),
            ("ondevice", True, "lbfgs-ondevice"),
        ],
    )
    def test_run_code_routes_backend_contract_to_expected_method(
        self, monkeypatch, optimizer_backend, limited_memory, expected_method
    ):
        """run_code() must honor the documented backend contract."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = optimizer_backend
        booz.options["limited_memory"] = limited_memory
        booz.options["materialize_dense_linearization"] = (
            optimizer_backend != "ondevice"
        )

        captured = {}

        def fake_minimize_runner(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad=False,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options, progress_callback
            captured["method"] = method
            assert value_and_grad is (method in {"bfgs", "lbfgs"})
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                fun=0.0,
                jac=jnp.zeros_like(flat_x0),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            materialize_hessian=True,
            max_dense_hessian_bytes=None,
            progress_callback=None,
            objective_args=(),
        ):
            del (
                maxiter,
                tol,
                stab,
                max_dense_hessian_bytes,
                progress_callback,
                objective_args,
            )
            if not materialize_hessian:
                return {
                    "x": x0,
                    "fun": jnp.asarray(0.0),
                    "grad": jnp.zeros_like(x0),
                    "hessian": None,
                    "nit": 0,
                    "success": True,
                    "hessian_materialized": False,
                }
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "reference_minimize", fake_minimize_runner)
        monkeypatch.setattr(_bsj, "target_minimize", fake_minimize_runner)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert captured["method"] == expected_method
        assert res["success"] is True
        assert res["adjoint_linear_solve_available"] is True
        if optimizer_backend == "ondevice":
            assert res["PLU"] is None
            adjoint_state = booz.get_adjoint_runtime_state()
            assert adjoint_state.plu is None
            assert callable(adjoint_state.solve_forward)
            assert callable(adjoint_state.solve_transpose)
            assert callable(adjoint_state.solve_forward_with_status)
            assert callable(adjoint_state.solve_transpose_with_status)
        else:
            assert isinstance(res["PLU"], tuple)
            assert len(res["PLU"]) == 3
            assert all(piece is not None for piece in res["PLU"])
        assert callable(res["vjp"])
        assert "iota" in res
        assert booz.need_to_run_code is False

    def test_scipy_bfgs_pre_newton_contract_uses_cpu_call_shape(self, monkeypatch):
        """The host-SciPy pre-Newton lane uses CPU BFGS method/options/layout."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "scipy"
        booz.options["limited_memory"] = False
        booz.options["maxcor"] = 33
        booz.options["ftol"] = 1e-12
        booz.options["maxfun"] = 55
        booz.options["maxls"] = 66
        booz.options["record_scipy_callback_trace"] = True
        captured = {}

        def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
            value, gradient = fun(x0)
            fun(np.asarray(x0) + 0.25)
            captured["x0"] = np.asarray(x0)
            captured["jac"] = jac
            captured["method"] = method
            captured["options"] = dict(options)
            captured["callback"] = callback
            captured["value_type"] = type(value)
            captured["gradient_type"] = type(gradient)
            captured["gradient_dtype"] = gradient.dtype
            captured["value"] = value
            captured["gradient"] = np.asarray(gradient)
            return types.SimpleNamespace(
                x=np.asarray(x0),
                jac=np.asarray(gradient),
                fun=float(value),
                nit=4,
                nfev=6,
                njev=6,
                success=True,
                status=0,
                message="Optimization terminated successfully.",
            )

        monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
        result = booz.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-8,
            maxiter=7,
            constraint_weight=1.0,
            iota=0.3,
            G=0.05,
            limited_memory=False,
            weight_inv_modB=True,
        )
        expected_x0 = np.concatenate(
            [
                np.asarray(booz.surface.get_dofs(), dtype=np.float64),
                np.asarray([0.3, 0.05], dtype=np.float64),
            ]
        )

        assert captured["jac"] is True
        assert captured["method"] == "BFGS"
        assert captured["options"] == {"maxiter": 7, "gtol": 1e-8}
        assert captured["callback"] is None
        assert captured["x0"].dtype == np.dtype(np.float64)
        assert captured["x0"].tobytes() == expected_x0.tobytes()
        assert captured["value_type"] is np.float64
        assert captured["gradient_type"] is np.ndarray
        assert captured["gradient_dtype"] == np.dtype(np.float64)
        assert result["optimizer_method"] == "bfgs"
        assert result["info"].status == 0
        assert result["info"].message == "Optimization terminated successfully."
        assert result["info"].nit == 4
        assert result["info"].nfev == 6
        assert result["info"].njev == 6
        assert result["scipy_call_contract"] == {
            "semantic_method": "bfgs",
            "scipy_method": "BFGS",
            "scipy_options": {"maxiter": 7, "gtol": 1e-8},
            "callback": None,
            "success": True,
            "status": 0,
            "message": "Optimization terminated successfully.",
            "nit": 4,
            "nfev": 6,
            "njev": 6,
        }
        np.testing.assert_allclose(
            result["scipy_initial_call"]["decision_vector"],
            expected_x0,
            atol=0.0,
            rtol=0.0,
        )
        assert result["scipy_initial_call"]["fun"] == captured["value"]
        np.testing.assert_allclose(
            result["scipy_initial_call"]["gradient"],
            captured["gradient"],
            atol=0.0,
            rtol=0.0,
        )
        assert len(result["scipy_callback_trace"]) == 2
        np.testing.assert_allclose(
            result["scipy_callback_trace"][0]["decision_vector"],
            expected_x0,
            atol=0.0,
            rtol=0.0,
        )
        assert result["scipy_callback_trace"][0]["fun"] == captured["value"]
        np.testing.assert_allclose(
            result["scipy_callback_trace"][0]["gradient"],
            captured["gradient"],
            atol=0.0,
            rtol=0.0,
        )

    def test_run_code_ondevice_default_requests_byte_capped_dense_hessian(
        self,
        monkeypatch,
    ):
        """The target lane requests dense finalization; byte cap gates scale."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf, label = _make_basic_mock_surface_and_label()
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={"optimizer_backend": "ondevice"},
        )
        captured = {}

        def fake_target_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad=False,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options, progress_callback
            assert value_and_grad is False
            return _successful_minimize_result(x0)

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            materialize_hessian=True,
            max_dense_hessian_bytes=None,
            progress_callback=None,
            objective_args=(),
        ):
            del (
                maxiter,
                tol,
                stab,
                max_dense_hessian_bytes,
                progress_callback,
                objective_args,
            )
            captured["materialize_hessian"] = materialize_hessian
            return {
                "x": x0,
                "fun": jnp.asarray(0.0),
                "grad": jnp.zeros_like(x0),
                "hessian": None,
                "nit": 0,
                "success": True,
                "hessian_materialized": False,
            }

        monkeypatch.setattr(_bsj, "target_minimize", fake_target_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert captured["materialize_hessian"] is True
        assert res["hessian"] is None
        assert res["PLU"] is None
        assert res["hessian_materialized"] is False

    def test_run_code_rejects_removed_hybrid_backend_after_options_mutation(self):
        """Mutated option dicts must not revive the removed hybrid backend."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "hybrid"

        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            booz.run_code(iota=0.3, G=0.05)

    def test_run_code_rejects_invalid_backend_after_options_mutation(self):
        """Mutable option dicts must not permit silent fallback to ondevice."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "bogus"

        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            booz.run_code(iota=0.3, G=0.05)

    @pytest.mark.parametrize(
        ("explicit_materialize", "expected_materialize"),
        [
            (None, True),
            (True, True),
        ],
    )
    def test_run_code_routes_lm_least_squares_contract(
        self,
        monkeypatch,
        explicit_materialize,
        expected_materialize,
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"
        if explicit_materialize is not None:
            booz.options["materialize_dense_linearization"] = explicit_materialize

        captured = {}

        def fake_target_least_squares(
            residual_fn,
            x0,
            *,
            method,
            tol,
            maxiter,
            options=None,
            callback=None,
            progress_callback=None,
        ):
            del residual_fn, tol, maxiter, callback, progress_callback
            captured["method"] = method
            captured["options"] = dict(options or {})
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                fun=0.0,
                jac=jnp.zeros_like(flat_x0),
                residual=jnp.zeros_like(flat_x0),
                residual_jacobian=jnp.eye(flat_x0.size, dtype=flat_x0.dtype),
                hessian=jnp.eye(flat_x0.size, dtype=flat_x0.dtype),
                damping=jnp.asarray(1.0e-3, dtype=flat_x0.dtype),
                nit=0,
                nfev=1,
                njev=1,
                status=0,
                success=True,
            )

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "target_least_squares", fake_target_least_squares)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert captured["method"] == "lm-ondevice"
        assert (
            captured["options"]["materialize_dense_linearization"]
            is expected_materialize
        )
        assert (
            captured["options"]["max_dense_linearization_bytes"]
            == booz.options["max_dense_linearization_bytes"]
        )
        assert res["optimizer_method"] == "lm-ondevice"
        assert res["success"] is True

    def test_run_code_emits_actual_first_stage_method_for_lm(self, monkeypatch):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"

        observed = []

        def record_stage(label, **payload):
            observed.append((label, payload))

        booz.options["stage_callback"] = record_stage

        def fake_target_least_squares(
            residual_fn,
            x0,
            *,
            method,
            tol,
            maxiter,
            options=None,
            callback=None,
            progress_callback=None,
        ):
            del residual_fn, tol, maxiter, options, callback, progress_callback
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                fun=0.0,
                jac=jnp.zeros_like(flat_x0),
                residual=jnp.zeros_like(flat_x0),
                residual_jacobian=jnp.eye(flat_x0.size, dtype=flat_x0.dtype),
                hessian=jnp.eye(flat_x0.size, dtype=flat_x0.dtype),
                damping=jnp.asarray(1.0e-3, dtype=flat_x0.dtype),
                nit=0,
                nfev=1,
                njev=1,
                status=0,
                success=True,
            )

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "target_least_squares", fake_target_least_squares)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        booz.run_code(iota=0.3, G=0.05)

        before_payload = _stage_payload(observed, "before_boozer_lbfgs")
        assert before_payload["method"] == "lm-ondevice"

    def test_penalty_residual_closure_hostifies_surface_metadata(self):
        booz = _make_mock_boozer_surface(stellsym=True, mpol=2, ntor=2)
        residual_fn = booz._make_penalty_residual_with(
            True,
            booz.options["weight_inv_modB"],
            1.0,
        )
        closure_nonlocals = inspect.getclosurevars(
            inspect.unwrap(residual_fn)
        ).nonlocals

        assert "self" not in closure_nonlocals
        assert not any(
            isinstance(leaf, jax.Array)
            for value in closure_nonlocals.values()
            for leaf in jax.tree_util.tree_leaves(value)
        )

    def test_run_code_uses_quasi_newton_for_fixed_G_ondevice_lm_option(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"

        captured = {}

        def forbidden_target_least_squares(*args, **kwargs):
            raise AssertionError("fixed-G ondevice path should not enter lm-ondevice")

        def fake_target_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options, progress_callback
            captured["method"] = method
            return _successful_minimize_result(x0)

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(
            _bsj, "target_least_squares", forbidden_target_least_squares
        )
        monkeypatch.setattr(_bsj, "target_minimize", fake_target_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=None)

        assert captured["method"] == "bfgs-ondevice"
        assert res["optimizer_method"] == "bfgs-ondevice"
        assert res["success"] is True

    @pytest.mark.parametrize("backend_mode", _ALL_JAX_BACKEND_MODES)
    @pytest.mark.parametrize("optimizer_backend", ["scipy"])
    def test_run_code_rejects_non_ondevice_ls_lane_in_any_jax_backend_mode(
        self,
        monkeypatch,
        request,
        backend_mode,
        optimizer_backend,
    ):
        """Any JAX backend mode must keep Boozer LS on the ondevice lane."""
        booz = _make_mock_boozer_surface()
        enable_non_strict_jax_backend(monkeypatch, request, mode=backend_mode)
        booz.options["optimizer_backend"] = optimizer_backend

        with pytest.raises(
            RuntimeError,
            match=rf"optimizer_backend='{optimizer_backend}'.*{backend_mode}.*requires optimizer_backend='ondevice'",
        ):
            booz.run_code(iota=0.3, G=0.05)

    @pytest.mark.parametrize("backend_mode", _ALL_JAX_BACKEND_MODES)
    @pytest.mark.parametrize("optimizer_backend", _NON_ONDEVICE_LS_BACKENDS)
    def test_resolve_optimizer_method_rejects_non_ondevice_ls_lane_in_any_jax_backend_mode(
        self,
        monkeypatch,
        request,
        backend_mode,
        optimizer_backend,
    ):
        booz = _make_mock_boozer_surface()
        enable_non_strict_jax_backend(monkeypatch, request, mode=backend_mode)
        booz.options["optimizer_backend"] = optimizer_backend

        with pytest.raises(
            RuntimeError,
            match=rf"optimizer_backend='{optimizer_backend}'.*{backend_mode}.*requires optimizer_backend='ondevice'",
        ):
            booz._resolve_optimizer_method()

    @pytest.mark.parametrize("backend_mode", _ALL_JAX_BACKEND_MODES)
    @pytest.mark.parametrize("method", _NON_TARGET_MINIMIZE_METHODS)
    def test_jax_minimize_rejects_fallback_methods_in_any_jax_backend_mode(
        self,
        monkeypatch,
        request,
        backend_mode,
        method,
    ):
        """Any JAX backend mode must keep direct minimization on the ondevice lane."""
        enable_non_strict_jax_backend(monkeypatch, request, mode=backend_mode)
        with pytest.raises(
            RuntimeError,
            match=_target_lane_rejection_pattern(
                r"optimizer_jax\.jax_minimize", method, backend_mode
            ),
        ):
            jax_minimize(
                lambda x: jnp.sum(x**2),
                jnp.array([1.0]),
                method=method,
                value_and_grad=False,
                options={"step_size": 0.1} if method == "adam" else None,
            )

    def test_jax_minimize_rejects_removed_hybrid_before_private_package_load(
        self,
        monkeypatch,
        request,
    ):
        _enable_non_strict_jax_backend(monkeypatch, request)

        def _fail_private_package_load():
            raise AssertionError(
                "JAX-backend contract rejection must happen before "
                "private optimizer package loading."
            )

        monkeypatch.setattr(_opt, "_load_private_pkg", _fail_private_package_load)

        with pytest.raises(ValueError, match="Unknown method 'bfgs-hybrid'"):
            jax_minimize(
                lambda x: jnp.sum(x**2),
                jnp.array([1.0], dtype=jnp.float64),
                method="bfgs-hybrid",
            )

    @pytest.mark.parametrize("backend_mode", _ALL_JAX_BACKEND_MODES)
    @pytest.mark.parametrize("method", ("adam", "bfgs", "lbfgs"))
    def test_jax_minimize_rejects_reference_methods_in_jax_backend_mode(
        self,
        monkeypatch,
        request,
        backend_mode,
        method,
    ):
        enable_non_strict_jax_backend(monkeypatch, request, mode=backend_mode)
        with pytest.raises(
            RuntimeError,
            match=_target_lane_rejection_pattern(
                r"optimizer_jax\.jax_minimize", method, backend_mode
            ),
        ):
            jax_minimize(
                lambda x: 0.5 * jnp.dot(x, x),
                jnp.asarray([5.0, 3.0], dtype=jnp.float64),
                method=method,
                maxiter=5,
                tol=1e-8,
                options={"step_size": 0.1} if method == "adam" else None,
            )

    @pytest.mark.parametrize("backend_mode", _ALL_JAX_BACKEND_MODES)
    def test_jax_least_squares_reference_lm_rejects_in_jax_backend_mode(
        self,
        monkeypatch,
        request,
        backend_mode,
    ):
        enable_non_strict_jax_backend(monkeypatch, request, mode=backend_mode)
        with pytest.raises(
            RuntimeError,
            match=_target_lane_rejection_pattern(
                r"optimizer_jax\.jax_least_squares", "lm", backend_mode
            ),
        ):
            jax_least_squares(
                lambda x: x - jnp.asarray([2.0, -1.0], dtype=jnp.float64),
                jnp.asarray([5.0, 3.0], dtype=jnp.float64),
                method="lm",
                maxiter=25,
                tol=1e-12,
            )

    def test_jax_least_squares_solves_simple_structured_problem(self):
        def residual_fn(state):
            return jnp.asarray(
                [
                    state["surface"][0] - 2.0,
                    state["surface"][1] + 1.0,
                    state["iota"] - 0.25,
                ],
                dtype=jnp.float64,
            )

        x0 = {
            "surface": jnp.asarray([5.0, 3.0], dtype=jnp.float64),
            "iota": jnp.asarray(0.0, dtype=jnp.float64),
        }

        result = jax_least_squares(residual_fn, x0, method="lm", maxiter=25, tol=1e-12)

        assert result.success is True
        np.testing.assert_allclose(result.x["surface"], np.asarray([2.0, -1.0]))
        np.testing.assert_allclose(result.x["iota"], 0.25)
        np.testing.assert_allclose(result.jac["surface"], np.zeros(2), atol=1e-10)
        np.testing.assert_allclose(result.jac["iota"], 0.0, atol=1e-10)

    def test_jax_minimize_adam_solves_simple_structured_problem(self):
        objective_fn, x0, target_surface, target_iota = (
            _make_structured_quadratic_problem()
        )

        result = jax_minimize(
            objective_fn,
            x0,
            method="adam",
            maxiter=800,
            tol=1e-8,
            options={"step_size": 0.1},
        )

        assert result.success is True
        np.testing.assert_allclose(result.x["surface"], target_surface, atol=1e-4)
        np.testing.assert_allclose(result.x["iota"], target_iota, atol=1e-4)

    def test_jax_minimize_adam_supports_explicit_value_and_grad(self):
        target = np.asarray([2.0, -1.0], dtype=float)

        def objective_value_and_grad(x):
            x = np.asarray(x, dtype=float)
            diff = x - target
            return 0.5 * float(np.dot(diff, diff)), diff

        result = jax_minimize(
            objective_value_and_grad,
            np.asarray([5.0, 3.0], dtype=float),
            method="adam",
            maxiter=800,
            tol=1e-8,
            value_and_grad=True,
            options={"step_size": 0.1},
        )

        assert result.success is True
        np.testing.assert_allclose(result.x, target, atol=1e-4)

    def test_jax_minimize_adam_ondevice_solves_simple_structured_problem(self):
        objective_fn, x0, target_surface, target_iota = (
            _make_structured_quadratic_problem()
        )

        result = jax_minimize(
            objective_fn,
            x0,
            method="adam-ondevice",
            maxiter=800,
            tol=1e-8,
            options={"step_size": 0.1},
        )

        assert result.success is True
        np.testing.assert_allclose(result.x["surface"], target_surface, atol=1e-4)
        np.testing.assert_allclose(result.x["iota"], target_iota, atol=1e-4)

    def test_reference_minimize_supports_structured_explicit_value_and_grad(self):
        target_surface = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
        target_iota = jnp.asarray(0.25, dtype=jnp.float64)
        x0 = {
            "surface": jnp.asarray([5.0, 3.0], dtype=jnp.float64),
            "iota": jnp.asarray(0.0, dtype=jnp.float64),
        }
        observed = []

        def objective_value_and_grad(state):
            surface_diff = state["surface"] - target_surface
            iota_diff = state["iota"] - target_iota
            value = 0.5 * (jnp.dot(surface_diff, surface_diff) + jnp.square(iota_diff))
            grad = {
                "surface": surface_diff,
                "iota": iota_diff,
            }
            return value, grad

        def callback(state):
            observed.append(state)

        result = _opt.reference_minimize(
            objective_value_and_grad,
            x0,
            method="bfgs",
            maxiter=100,
            tol=1e-10,
            value_and_grad=True,
            callback=callback,
        )

        assert result.success is True
        assert observed
        assert set(result.x) == {"surface", "iota"}
        np.testing.assert_allclose(result.x["surface"], np.asarray(target_surface))
        np.testing.assert_allclose(result.x["iota"], float(target_iota))
        np.testing.assert_allclose(result.jac["surface"], np.zeros(2), atol=1e-8)
        np.testing.assert_allclose(result.jac["iota"], 0.0, atol=1e-8)

    def test_jax_least_squares_pytree_hot_path_skips_flattening_adapter(
        self,
        monkeypatch,
    ):
        def residual_fn(state):
            return jnp.asarray(
                [
                    state["surface"][0] - 2.0,
                    state["surface"][1] + 1.0,
                    state["iota"] - 0.25,
                ],
                dtype=jnp.float64,
            )

        x0 = {
            "surface": jnp.asarray([5.0, 3.0], dtype=jnp.float64),
            "iota": jnp.asarray(0.0, dtype=jnp.float64),
        }

        monkeypatch.setattr(
            _opt,
            "_prepare_optimizer_callable_inputs",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "LM least-squares hot path should not route through the "
                    "flat-vector pytree adapter."
                )
            ),
        )

        result = jax_least_squares(residual_fn, x0, method="lm", maxiter=25, tol=1e-12)

        assert result.success is True
        np.testing.assert_allclose(result.x["surface"], np.asarray([2.0, -1.0]))
        np.testing.assert_allclose(result.x["iota"], 0.25)

    def test_levenberg_marquardt_materializes_dense_linearization_once_at_final_iterate(
        self,
        monkeypatch,
    ):
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]], dtype=jnp.float64)
        b = jnp.array([5.0, 7.0], dtype=jnp.float64)
        dense_calls, gmres_calls = _patch_matrix_free_lm_solver(
            monkeypatch,
            A=A,
        )

        def residual(x):
            return A @ x - b

        result = _opt.levenberg_marquardt(residual, jnp.zeros(2), maxiter=25, tol=1e-14)

        assert gmres_calls
        assert len(dense_calls) == 1
        _assert_linear_lm_result(result, A=A, b=b)
        assert result["success"]

    def test_levenberg_marquardt_traceable_materializes_dense_linearization_once_at_final_iterate(
        self,
        monkeypatch,
    ):
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]], dtype=jnp.float64)
        b = jnp.array([5.0, 7.0], dtype=jnp.float64)
        dense_calls, gmres_calls = _patch_matrix_free_lm_solver(monkeypatch, A=A)

        def residual(x):
            return A @ x - b

        result = _opt.levenberg_marquardt_traceable(
            residual,
            jnp.zeros(2),
            maxiter=25,
            tol=1e-14,
        )

        assert gmres_calls
        assert len(dense_calls) == 1
        _assert_linear_lm_result(result, A=A, b=b)
        assert bool(result["success"])

    def test_jax_minimize_allows_explicit_value_grad_ondevice_in_strict_mode(
        self,
        monkeypatch,
        request,
    ):
        """Strict JAX mode must allow the JAX-native explicit value/grad path."""

        target = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
        x0 = jnp.asarray([5.0, 3.0], dtype=jnp.float64)

        def objective_value_and_grad(x):
            diff = jnp.asarray(x, dtype=jnp.float64) - target
            return 0.5 * jnp.dot(diff, diff), diff

        _enable_strict_jax_backend(monkeypatch, request)
        result = jax_minimize(
            objective_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            value_and_grad=True,
        )

        assert result.success is True
        assert result.nit > 0
        np.testing.assert_allclose(result.x, np.asarray(target), atol=1e-10)

    @pytest.mark.parametrize("optimizer_backend", ["ondevice"])
    def test_run_code_rejects_target_backend_without_x64(
        self, monkeypatch, optimizer_backend
    ):
        """run_code() must fail at the public seam before target-lane execution without x64."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = optimizer_backend
        monkeypatch.setattr(_opt, "_x64_enabled", lambda: False)

        with pytest.raises(
            RuntimeError,
            match=rf"optimizer_backend='{optimizer_backend}'.*requires jax_enable_x64=True",
        ):
            booz.run_code(iota=0.3, G=0.05)

    def test_run_code_ls_converges_with_stellsym_surface(self):
        """LS solve must also converge when the surface uses stellsym DOFs."""
        booz = _make_mock_boozer_surface(stellsym=True)
        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["type"] == "ls"
        assert res["success"] is True
        assert callable(res["vjp"])

    def test_run_code_idempotent(self):
        """Second run_code() call returns None (not dirty)."""
        booz = _make_mock_boozer_surface()
        booz.run_code(iota=0.3, G=0.05)
        assert booz.run_code(iota=0.3, G=0.05) is None

    def test_same_input_same_result_without_res_identity(self):
        """Same explicit runtime inputs reproduce values without sharing res identity."""
        left = _make_mock_boozer_surface()
        right = _make_mock_boozer_surface()

        left_res = left.run_code(iota=0.3, G=0.05)
        right_res = right.run_code(iota=0.3, G=0.05)

        assert left_res is not right_res
        assert left_res["type"] == right_res["type"] == "ls"
        assert left_res["success"] is right_res["success"]
        assert left_res["weight_inv_modB"] == right_res["weight_inv_modB"]
        np.testing.assert_allclose(left_res["iota"], right_res["iota"], atol=1e-14)
        np.testing.assert_allclose(left_res["G"], right_res["G"], atol=1e-14)
        np.testing.assert_allclose(left_res["fun"], right_res["fun"], atol=1e-14)
        np.testing.assert_allclose(
            left_res["final_gradient_inf_norm"],
            right_res["final_gradient_inf_norm"],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.asarray(left_res["residual"]),
            np.asarray(right_res["residual"]),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            left.surface.get_dofs(),
            right.surface.get_dofs(),
            atol=1e-14,
        )

    def test_run_code_sdofs_matches_implicit_path(self):
        """run_code(sdofs=surface_dofs) must produce the same result as run_code()."""
        booz_ref = _make_mock_boozer_surface()
        sdofs_orig = booz_ref.surface.get_dofs().copy()
        res_ref = booz_ref.run_code(iota=0.3, G=0.05)

        booz_sdofs = _make_mock_boozer_surface()
        res_sdofs = booz_sdofs.run_code(iota=0.3, G=0.05, sdofs=sdofs_orig)

        assert res_sdofs["success"] == res_ref["success"]
        np.testing.assert_allclose(res_sdofs["iota"], res_ref["iota"], atol=1e-14)
        np.testing.assert_allclose(res_sdofs["fun"], res_ref["fun"], atol=1e-14)
        np.testing.assert_allclose(
            booz_sdofs.surface.get_dofs(), booz_ref.surface.get_dofs(), atol=1e-14
        )

    def test_run_code_sdofs_overrides_stale_surface(self):
        """run_code(sdofs=...) must use explicit DOFs, not stale self.surface."""
        booz = _make_mock_boozer_surface()
        sdofs_good = booz.surface.get_dofs().copy()

        # Solve once to get reference result
        res_ref = booz.run_code(iota=0.3, G=0.05)
        surface_after_ref = booz.surface.get_dofs().copy()

        # Perturb surface to garbage, mark dirty, re-solve with explicit sdofs
        booz.surface.set_dofs(sdofs_good * 0.0 + 999.0)
        booz.need_to_run_code = True
        res_sdofs = booz.run_code(iota=0.3, G=0.05, sdofs=sdofs_good)

        # Must reproduce the reference solution from explicit DOFs.
        assert res_sdofs["success"] is res_ref["success"]
        np.testing.assert_allclose(res_sdofs["iota"], res_ref["iota"], atol=1e-12)
        np.testing.assert_allclose(res_sdofs["fun"], res_ref["fun"], atol=1e-12)
        # Surface must hold solved DOFs, not the garbage or the warm-start
        np.testing.assert_allclose(
            booz.surface.get_dofs(), surface_after_ref, atol=1e-12
        )

    def test_run_code_sdofs_syncs_surface_on_exact_failure(self):
        """On exact-path failure, self.surface must hold warm-start sdofs.

        The exact-path failure (NaN iterates) returns before calling
        ``_set_surface_dofs``.  The pre-sync in ``run_code`` must leave
        ``self.surface`` in the warm-start state, not whatever garbage
        was there before.
        """
        booz = _make_mock_boozer_surface_exact()
        sdofs_good = booz.surface.get_dofs().copy()

        # Corrupt surface state
        booz.surface.set_dofs(sdofs_good * 0.0 + 999.0)
        booz.need_to_run_code = True

        # Force exact Newton to fail → failure path skips _set_surface_dofs
        with _patched_exact_surface_module():
            with _patched_exact_newton_result(success=False, step=jnp.nan, nit=0):
                res = booz.run_code(iota=0.3, G=0.05, sdofs=sdofs_good)

        assert res["success"] is False
        # Surface must hold the warm-start DOFs, not the garbage
        np.testing.assert_allclose(booz.surface.get_dofs(), sdofs_good, atol=1e-14)

    def test_run_code_sdofs_syncs_surface_on_ls_newton_failure(self, monkeypatch):
        """On LS Newton-polish failure with sdofs, surface holds LBFGS output.

        In the LS path, LBFGS always calls ``_set_surface_dofs`` before
        Newton runs, so the pre-sync is overwritten by LBFGS.  On Newton
        NaN failure, surface retains the LBFGS result — NOT the warm-start
        sdofs and NOT the pre-corruption garbage.
        """
        booz = _make_mock_boozer_surface()
        sdofs_good = booz.surface.get_dofs().copy()

        # Solve once to capture the LBFGS-only surface output
        booz_ref = _make_mock_boozer_surface()

        def nan_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab, progress_callback
            return {
                "x": x0,
                "fun": jnp.asarray(jnp.nan),
                "grad": jnp.full_like(x0, jnp.nan),
                "hessian": jnp.full(
                    (x0.shape[0], x0.shape[0]), jnp.nan, dtype=x0.dtype
                ),
                "nit": 0,
                "success": False,
            }

        _patch_newton_polish_runner(monkeypatch, nan_newton_polish)
        booz_ref.run_code(iota=0.3, G=0.05)
        lbfgs_surface = booz_ref.surface.get_dofs().copy()

        # Corrupt surface, re-solve with explicit sdofs
        booz.surface.set_dofs(sdofs_good * 0.0 + 999.0)
        booz.need_to_run_code = True
        res = booz.run_code(iota=0.3, G=0.05, sdofs=sdofs_good)

        assert res["success"] is False
        # Surface must hold LBFGS output (not garbage, not warm-start)
        np.testing.assert_allclose(booz.surface.get_dofs(), lbfgs_surface, atol=1e-12)

    def test_run_code_invalid_newton_iterate_aborts_adjoint_state(self, monkeypatch):
        """Finite iterates with invalid Newton derivatives must not build adjoint metadata."""
        booz = _make_mock_boozer_surface()

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return {
                "x": x0,
                "fun": jnp.asarray(jnp.nan),
                "grad": jnp.full_like(x0, jnp.nan),
                "hessian": jnp.full(
                    (x0.shape[0], x0.shape[0]), jnp.nan, dtype=x0.dtype
                ),
                "nit": 0,
                "success": False,
            }

        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)
        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["success"] is False
        assert res["PLU"] is None
        assert res["vjp"] is None
        assert booz.need_to_run_code is False
        assert np.all(np.isfinite(booz.surface.get_dofs()))

    def test_run_code_finite_unsuccessful_newton_keeps_adjoint_state(self, monkeypatch):
        """Finite maxiter-exhausted Newton exits must still keep dense metadata/VJPs."""
        booz = _make_mock_boozer_surface()

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            n = x0.shape[0]
            return {
                "x": x0,
                "fun": jnp.asarray(0.0),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(n, dtype=x0.dtype),
                "nit": 3,
                "success": False,
            }

        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)
        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["success"] is False
        assert res["PLU"] is not None
        assert callable(res["vjp"])

    def test_get_adjoint_runtime_state_prefers_operator_callbacks_for_ls(
        self, monkeypatch
    ):
        """LS runtime adjoints must stay operator-backed even when PLU exists."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.need_to_run_code = False
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "hessian",
            "PLU": tuple(jnp.eye(booz.x.size, dtype=jnp.float64) for _ in range(3)),
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        recorded = {}

        def fake_solve_hessian_system(objective_fn, x, rhs, *, stab, tol):
            del objective_fn
            recorded["x_shape"] = tuple(np.asarray(x).shape)
            recorded["rhs"] = np.asarray(rhs)
            recorded["stab"] = stab
            recorded["tol"] = tol
            return rhs

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_solve_hessian_system",
            fake_solve_hessian_system,
        )
        adjoint_state = booz.get_adjoint_runtime_state()
        solved = adjoint_state.solve_transpose(
            jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        )

        _assert_operator_adjoint_state(
            adjoint_state,
            dense_factors_available=True,
        )
        np.testing.assert_allclose(np.asarray(solved), np.asarray([1.0, -2.0]))
        np.testing.assert_allclose(recorded["rhs"], np.asarray([1.0, -2.0]))
        assert recorded["x_shape"][0] == booz._pack_decision_vector(0.3, 0.05).size

    def test_get_adjoint_runtime_state_uses_dense_plu_for_scipy_hessian(self):
        """Host-dispatched CPU parity uses the same dense adjoint solve as CPU."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "scipy"
        booz.need_to_run_code = False
        P = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float64)
        L = jnp.asarray([[1.0, 0.0], [0.25, 1.0]], dtype=jnp.float64)
        U = jnp.asarray([[4.0, 1.0], [0.0, 2.5]], dtype=jnp.float64)
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "hessian",
            "PLU": (P, L, U),
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        adjoint_state = booz.get_adjoint_runtime_state()
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        solved, success = adjoint_state.solve_transpose_with_status(rhs)
        H = np.asarray(P) @ np.asarray(L) @ np.asarray(U)

        _assert_dense_plu_adjoint_state(adjoint_state)
        np.testing.assert_allclose(H.T @ np.asarray(solved), np.asarray(rhs))
        assert bool(np.asarray(success)) is True

    def test_get_adjoint_runtime_state_uses_shared_lu_piv_for_ondevice_hessian(self):
        """JAX on-device factor-once metadata must match the packed-LU branch."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.need_to_run_code = False
        n = booz._pack_decision_vector(0.3, 0.05).size
        H = jnp.diag(jnp.linspace(1.0, 2.0, n, dtype=jnp.float64))
        lu_piv = _opt._factor_dense_hessian(H, optimizer_backend="ondevice")
        plu = _opt._plu_from_lu_piv(lu_piv)
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "hessian",
            "hessian": H,
            "PLU": plu,
            "LU_PIV": lu_piv,
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        adjoint_state = booz.get_adjoint_runtime_state()
        rhs = jnp.arange(1, n + 1, dtype=jnp.float64)
        solved = adjoint_state.solve_forward(rhs)
        expected = _opt._lu_solve_dense_hessian(
            lu_piv,
            rhs,
            transpose=False,
        )

        assert adjoint_state.linear_solve_backend == "dense-plu-shared"
        assert adjoint_state.linear_solve_factors is not None
        assert adjoint_state.dense_linear_solve_factors_available is True
        np.testing.assert_array_equal(np.asarray(solved), np.asarray(expected))

    def test_get_adjoint_runtime_state_rejects_unknown_kind_even_when_plu_exists(
        self,
    ):
        """Dense factors must not provide an implicit fallback JAX runtime branch."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        identity = jnp.eye(rhs.size, dtype=jnp.float64)
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "unknown_kind_forcing_unsupported_branch",
            "PLU": (identity, identity, identity),
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        with pytest.raises(RuntimeError, match="Unsupported BoozerSurfaceJAX"):
            booz.get_adjoint_runtime_state()

    def test_get_adjoint_runtime_state_hessian_uses_configured_stab_once(
        self, monkeypatch
    ):
        """Hessian runtime solves must not mutate into a hidden retry ridge."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.options["newton_stab"] = 1.0e-4
        booz.options["newton_tol"] = 1e-8
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "hessian",
            "PLU": None,
            "dense_linear_solve_factors_available": False,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        recorded_stabs = []

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_hessian_vector_product_fn",
            lambda _objective_fn: lambda _x, vec: vec,
        )

        def fake_solve_hessian_system_with_status(
            _objective_fn,
            _x,
            rhs,
            *,
            stab,
            tol,
        ):
            del tol
            stab_value = float(np.asarray(stab))
            recorded_stabs.append(stab_value)
            return rhs / (1.0 + stab_value), True

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_solve_hessian_system_with_status",
            fake_solve_hessian_system_with_status,
        )

        adjoint_state = booz.get_adjoint_runtime_state()
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        solved, success = adjoint_state.solve_transpose_with_status(rhs)

        assert bool(np.asarray(success)) is True
        np.testing.assert_allclose(recorded_stabs, np.asarray([1.0e-4]))
        np.testing.assert_allclose(
            np.asarray(adjoint_state.apply_transpose(solved)),
            np.asarray(rhs),
            rtol=0.0,
            atol=1e-12,
        )

    def test_get_adjoint_runtime_state_hessian_apply_uses_column_batched_rhs(
        self, monkeypatch
    ):
        """Hessian runtime application must match matrix-RHS solve semantics."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.options["newton_stab"] = 0.25
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "hessian",
            "PLU": None,
            "dense_linear_solve_factors_available": False,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        def vector_only_hvp(_x, vec):
            assert vec.ndim == 1
            return jnp.asarray([2.0, -3.0], dtype=vec.dtype) * vec

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_hessian_vector_product_fn",
            lambda _objective_fn: vector_only_hvp,
        )

        adjoint_state = booz.get_adjoint_runtime_state()
        rhs = jnp.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=jnp.float64)

        np.testing.assert_allclose(
            np.asarray(adjoint_state.apply_transpose(rhs)),
            np.asarray([[2.25, 4.5], [2.75, -1.375]]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_get_adjoint_runtime_state_status_uses_explicit_host_bool_boundary(
        self, monkeypatch
    ):
        """Status conversion must not coerce device bools through module np.asarray."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.options["newton_stab"] = 0.0
        booz.options["newton_tol"] = 1e-8
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "hessian",
            "PLU": None,
            "dense_linear_solve_factors_available": False,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_hessian_vector_product_fn",
            lambda _objective_fn: lambda _x, vec: vec,
        )

        def fake_solve_hessian_system_with_status(
            _objective_fn,
            _x,
            rhs,
            *,
            stab,
            tol,
        ):
            del stab, tol
            return rhs, jnp.asarray(True)

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_solve_hessian_system_with_status",
            fake_solve_hessian_system_with_status,
        )

        original_asarray = _bsj.np.asarray

        def reject_jax_array_asarray(value, *args, **kwargs):
            if isinstance(value, jax.Array):
                raise AssertionError("unexpected implicit device bool materialization")
            return original_asarray(value, *args, **kwargs)

        monkeypatch.setattr(_bsj.np, "asarray", reject_jax_array_asarray)

        adjoint_state = booz.get_adjoint_runtime_state()
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        solved, success = adjoint_state.solve_transpose_with_status(rhs)

        assert bool(np.asarray(success)) is True
        np.testing.assert_allclose(
            original_asarray(adjoint_state.apply_transpose(solved)),
            original_asarray(rhs),
            rtol=0.0,
            atol=1e-12,
        )

    def test_get_adjoint_runtime_state_ls_normal_uses_host_tolerance_boundary(
        self, monkeypatch
    ):
        """LS-normal runtime solves must not pass device scalars into eager helpers."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.options["newton_tol"] = 1e-8
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "least_squares_normal",
            "PLU": tuple(jnp.eye(booz.x.size, dtype=jnp.float64) for _ in range(3)),
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_least_squares_normal_operator",
            lambda _residual_fn, _x: {
                "matvec": lambda vec: vec,
                "transpose_matvec": lambda vec: vec,
            },
        )

        def fake_solve_least_squares_normal_system_with_status(
            _residual_fn,
            _x,
            rhs,
            *,
            tol,
        ):
            tol_value = float(np.asarray(tol))
            assert tol_value == pytest.approx(booz._linear_solve_tolerance())
            return rhs, jnp.asarray(True)

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_solve_least_squares_normal_system_with_status",
            fake_solve_least_squares_normal_system_with_status,
        )

        original_asarray = _bsj.np.asarray

        def reject_jax_array_asarray(value, *args, **kwargs):
            if isinstance(value, jax.Array):
                raise AssertionError(
                    "unexpected implicit device scalar materialization"
                )
            return original_asarray(value, *args, **kwargs)

        monkeypatch.setattr(_bsj.np, "asarray", reject_jax_array_asarray)

        adjoint_state = booz.get_adjoint_runtime_state()
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        solved, success = adjoint_state.solve_transpose_with_status(rhs)

        assert bool(np.asarray(success)) is True
        np.testing.assert_allclose(original_asarray(solved), original_asarray(rhs))
        np.testing.assert_allclose(
            original_asarray(adjoint_state.apply_transpose(solved)),
            original_asarray(rhs),
            rtol=0.0,
            atol=1e-12,
        )

    def test_get_adjoint_runtime_state_exact_jacobian_uses_host_tolerance_boundary(
        self, monkeypatch
    ):
        """Exact-Jacobian runtime solves use the well-conditioned adjoint lane."""
        exact_lane = parity_ladder_tolerances("exact-well-conditioned-adjoint")
        booz = _make_mock_boozer_surface_exact()
        booz.need_to_run_code = False
        booz.options["newton_tol"] = 1e-12
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "exact_jacobian",
            "PLU": tuple(jnp.eye(booz.x.size, dtype=jnp.float64) for _ in range(3)),
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }

        monkeypatch.setattr(
            _bsj.BoozerSurfaceJAX,
            "_compute_stellsym_mask_indices",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            _bsj.BoozerSurfaceJAX,
            "_make_exact_residual",
            lambda self, _mask: lambda _x: _x,
        )
        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_jacobian_linear_operator",
            lambda _residual_fn, _x: {
                "matvec": lambda vec: vec,
                "transpose_matvec": lambda vec: vec,
            },
        )

        def fake_solve_jacobian_system_with_status(
            _residual_fn,
            _x,
            rhs,
            *,
            transpose,
            tol,
        ):
            del transpose
            tol_value = float(np.asarray(tol))
            assert tol_value == pytest.approx(booz._linear_solve_tolerance())
            return rhs, jnp.asarray(True)

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_solve_jacobian_system_with_status",
            fake_solve_jacobian_system_with_status,
        )

        original_asarray = _bsj.np.asarray

        def reject_jax_array_asarray(value, *args, **kwargs):
            if isinstance(value, jax.Array):
                raise AssertionError(
                    "unexpected implicit device scalar materialization"
                )
            return original_asarray(value, *args, **kwargs)

        monkeypatch.setattr(_bsj.np, "asarray", reject_jax_array_asarray)

        adjoint_state = booz.get_adjoint_runtime_state()
        rhs = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
        solved, success = adjoint_state.solve_transpose_with_status(rhs)

        _assert_operator_adjoint_state(
            adjoint_state,
            dense_factors_available=True,
        )
        assert bool(np.asarray(success)) is True
        np.testing.assert_allclose(
            original_asarray(solved),
            original_asarray(rhs),
            rtol=exact_lane["adjoint_rtol"],
            atol=exact_lane["adjoint_atol"],
        )
        applied = original_asarray(adjoint_state.apply_transpose(solved))
        rhs_host = original_asarray(rhs)
        residual = applied - rhs_host
        residual_rel = np.linalg.norm(residual) / (1.0 + np.linalg.norm(rhs_host))
        assert residual_rel <= exact_lane["residual_rel_tol"]
        np.testing.assert_allclose(
            residual,
            np.zeros_like(residual),
            rtol=0.0,
            atol=exact_lane["residual_rel_tol"],
        )

    @pytest.mark.parametrize(
        ("mode", "lane"),
        (("jax_cpu_parity", "cpu"), ("jax_gpu_parity", "gpu")),
        ids=("cpu_parity", "gpu_parity"),
    )
    def test_exact_well_conditioned_operator_adjoint_matches_dense_reference_and_plu(
        self,
        monkeypatch,
        request,
        mode,
        lane,
    ):
        """Well-conditioned exact operator GMRES matches dense JAX and PLU adjoints."""
        exact_lane = parity_ladder_tolerances("exact-well-conditioned-adjoint")
        enable_strict_jax_backend(monkeypatch, request, mode=mode)
        case = _solve_exact_well_conditioned_operator_case(
            monkeypatch,
            device=parity_device(lane),
        )
        _assert_exact_well_conditioned_operator_case(case, exact_lane)

    def test_exact_well_conditioned_operator_adjoint_cpu_gpu_same_state_parity(
        self,
        monkeypatch,
        request,
    ):
        """CPU and GPU exact adjoints should agree on the same well-conditioned state."""
        exact_lane = parity_ladder_tolerances("exact-well-conditioned-adjoint")
        enable_strict_jax_backend(monkeypatch, request, mode="jax_gpu_parity")
        cpu_case = _solve_exact_well_conditioned_operator_case(
            monkeypatch,
            device=parity_device("cpu"),
        )
        gpu_case = _solve_exact_well_conditioned_operator_case(
            monkeypatch,
            device=parity_device("gpu"),
        )

        _assert_exact_well_conditioned_operator_case(cpu_case, exact_lane)
        _assert_exact_well_conditioned_operator_case(gpu_case, exact_lane)
        np.testing.assert_allclose(
            cpu_case["operator_adj_np"],
            gpu_case["operator_adj_np"],
            rtol=exact_lane["adjoint_rtol"],
            atol=exact_lane["adjoint_atol"],
        )
        np.testing.assert_allclose(
            cpu_case["operator_gradient"],
            gpu_case["operator_gradient"],
            rtol=exact_lane["gradient_rtol"],
            atol=exact_lane["gradient_atol"],
        )

    def test_exact_ill_conditioned_operator_adjoint_action_level_parity(
        self,
        monkeypatch,
        request,
    ):
        """W3.2 (C3): action-level (range-space) adjoint parity on the
        ill-conditioned exact-Jacobian lane.

        The ``exact_ill_conditioned_adjoint`` lane intentionally disables
        raw-vector parity (``vector_parity_required=False``) because
        near-singular Jacobians admit infinitely many adjoint vectors
        that all satisfy the residual gate. This test fills the
        remaining coverage gap by asserting that the operator-backed
        adjoint and the dense PLU adjoint agree on the well-conditioned
        subspace ``U_well`` (singular values above ``σ_max * 1e-8``)
        even when raw-vector ``‖λ_op − λ_plu‖`` is dominated by the
        near-null direction. See plan §W3.2 (C3).
        """
        exact_lane = parity_ladder_tolerances("exact-ill-conditioned-adjoint")
        enable_strict_jax_backend(monkeypatch, request, mode="jax_cpu_parity")
        case = _solve_exact_ill_conditioned_operator_case(
            monkeypatch,
            device=parity_device("cpu"),
            cond_target=1e10,
        )
        assert case["success"] is True or exact_lane["operator_failure_allowed"]
        _assert_exact_ill_conditioned_operator_action_parity(case, exact_lane)

    def test_exact_adjoint_dense_metadata_does_not_change_operator_runtime(
        self,
        monkeypatch,
    ):
        booz = _make_mock_boozer_surface_exact()
        booz.need_to_run_code = False
        identity = jnp.eye(2, dtype=jnp.float64)
        booz.res = {
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "iota": jnp.asarray(0.3, dtype=jnp.float64),
            "G": jnp.asarray(0.05, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "exact_jacobian",
            "PLU": (identity, identity, identity),
            "dense_linear_solve_factors_available": True,
            "vjp_groups": lambda *_args, **_kwargs: iter(()),
        }
        operator_calls = []

        monkeypatch.setattr(
            _bsj.BoozerSurfaceJAX,
            "_compute_stellsym_mask_indices",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            _bsj.BoozerSurfaceJAX,
            "_make_exact_residual",
            lambda _self, _mask: lambda x: x,
        )

        def fake_operator_solve(_residual_fn, _x, rhs, *, transpose, tol):
            operator_calls.append((bool(transpose), float(tol)))
            return rhs + 1.0, jnp.asarray(True)

        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_solve_jacobian_system_with_status",
            fake_operator_solve,
        )

        adjoint_state = booz.get_adjoint_runtime_state()
        rhs = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
        solved, success = adjoint_state.solve_transpose_with_status(rhs)

        assert adjoint_state.linear_solve_backend == "operator"
        assert adjoint_state.dense_linear_solve_factors_available is True
        assert len(operator_calls) == 1
        assert operator_calls[0][0] is True
        assert operator_calls[0][1] == pytest.approx(booz._linear_solve_tolerance())
        assert bool(np.asarray(success)) is True
        np.testing.assert_allclose(np.asarray(solved), np.asarray(rhs + 1.0))

    def test_run_code_emits_newton_progress_updates(self, monkeypatch):
        """run_code() should surface Newton start/progress/completion through stage_callback."""
        booz = _make_mock_boozer_surface()

        observed = []

        def record_stage(label, **payload):
            observed.append((label, payload))

        booz.options["stage_callback"] = record_stage

        def fake_reference_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad=False,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options, progress_callback
            assert value_and_grad is True
            return _successful_minimize_result(x0)

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, objective_args
            assert progress_callback is not None
            _emit_newton_progress(progress_callback)
            return _successful_newton_polish_result(x0, nit=2)

        monkeypatch.setattr(_bsj, "reference_minimize", fake_reference_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        labels = [label for label, _payload in observed]
        progress_events = [
            payload for label, payload in observed if label == "boozer_newton_progress"
        ]
        after_lbfgs_payload = _stage_payload(observed, "after_boozer_lbfgs")
        before_newton_payload = _stage_payload(observed, "before_boozer_newton")
        after_newton_payload = _stage_payload(observed, "after_boozer_newton")
        assert res is not None
        assert res["success"] is True
        assert "after_boozer_lbfgs" in labels
        assert "before_boozer_newton" in labels
        assert "after_boozer_newton" in labels
        _assert_solver_completion_payload(after_lbfgs_payload)
        assert before_newton_payload["method"] == "newton-polish"
        assert before_newton_payload["ls_method"] == "bfgs"
        _assert_solver_completion_payload(after_newton_payload)
        assert np.isfinite(after_newton_payload["residual_inf"])
        assert [int(payload["iteration"]) for payload in progress_events] == [1, 2]
        assert all("grad_norm" in payload for payload in progress_events)

    def test_run_code_ondevice_skips_newton_progress_callback(self, monkeypatch):
        """The ondevice Newton polish lane must stay callback-free inside the traced loop."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"

        observed = []
        captured = {}

        def record_stage(label, **payload):
            observed.append((label, payload))

        booz.options["stage_callback"] = record_stage

        def fake_target_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options
            assert method == "bfgs-ondevice"
            assert progress_callback is not None
            return _successful_minimize_result(x0)

        def fake_newton_polish_traceable(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            materialize_hessian=True,
            max_dense_hessian_bytes=None,
            progress_callback=None,
            args=(),
        ):
            del maxiter, tol, stab, materialize_hessian, max_dense_hessian_bytes, args
            captured["progress_callback"] = progress_callback
            return _successful_newton_polish_result(x0, nit=2)

        monkeypatch.setattr(_bsj, "target_minimize", fake_target_minimize)
        monkeypatch.setattr(
            _bsj,
            "newton_polish_traceable",
            fake_newton_polish_traceable,
        )

        res = booz.run_code(iota=0.3, G=0.05)

        labels = [label for label, _payload in observed]
        progress_events = [
            payload for label, payload in observed if label == "boozer_newton_progress"
        ]
        after_lbfgs_payload = _stage_payload(observed, "after_boozer_lbfgs")
        before_newton_payload = _stage_payload(observed, "before_boozer_newton")
        after_newton_payload = _stage_payload(observed, "after_boozer_newton")
        assert res is not None
        assert res["success"] is True
        assert captured["progress_callback"] is None
        assert "after_boozer_lbfgs" in labels
        assert "before_boozer_newton" in labels
        assert "after_boozer_newton" in labels
        _assert_solver_completion_payload(after_lbfgs_payload)
        assert before_newton_payload["method"] == "newton-polish"
        assert before_newton_payload["ls_method"] == "bfgs-ondevice"
        _assert_solver_completion_payload(after_newton_payload)
        assert np.isfinite(after_newton_payload["residual_inf"])
        assert progress_events == []

    def test_run_code_passes_newton_stab(self, monkeypatch):
        """run_code() must forward newton_stab into the Newton polish call."""
        booz = _make_mock_boozer_surface()
        booz.options["newton_stab"] = 0.125

        captured = {}

        def fake_reference_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad=False,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options, progress_callback
            assert value_and_grad is True
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                fun=0.0,
                jac=jnp.zeros_like(flat_x0),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, progress_callback, objective_args
            captured["stab"] = stab
            n = x0.shape[0]
            return {
                "x": x0,
                "fun": jnp.asarray(0.0),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(n, dtype=x0.dtype),
                "nit": 0,
                "success": True,
            }

        monkeypatch.setattr(_bsj, "reference_minimize", fake_reference_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert res["success"] is True
        assert captured["stab"] == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# P2 #4b: BoozerSurfaceJAX exact-path tests
# ---------------------------------------------------------------------------


def _make_mock_boozer_surface_exact(
    mpol=1,
    ntor=1,
    nfp=1,
    stellsym=False,
    options=None,
    constraint_weight=None,
):
    """Build a BoozerSurfaceJAX in exact (Newton) mode -- constraint_weight=None.

    The exact Newton path requires a SQUARE system: n_eq == n_dof.
    For non-stellsym: n_eq = 3*nphi*ntheta + 2, n_dof = 3*(2m+1)*(2n+1) + 2.
    Square when nphi*ntheta = (2m+1)*(2n+1).  For mpol=ntor=1: 3x3 grid.
    """
    R0, r = 1.0, 0.1
    nphi = 2 * mpol + 1
    ntheta = 2 * ntor + 1

    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    full_sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
    assert full_sdofs.shape[0] == 3 * nphi * ntheta

    if stellsym:
        scatter = np.asarray(stellsym_scatter_indices(mpol, ntor), dtype=np.int32)
        sdofs = full_sdofs[scatter]
    else:
        sdofs = full_sdofs

    bs = _MockBiotSavart(_make_mock_coils())
    surf = _MockSurface(sdofs, mpol, ntor, nfp, stellsym, qphi, qtheta)
    surf.jax_surface_kind = "xyztensorfourier"
    label = _PlumbingVolumeLabel(surf)
    target = 2.0 * np.pi**2 * R0 * r**2

    return BoozerSurfaceJAX(
        bs,
        surf,
        label,
        target,
        constraint_weight=constraint_weight,
        options=options,
    )


def _run_mock_exact_boozer(booz, iota=0.3, G=0.05):
    with _patched_exact_surface_module():
        return booz.run_code(iota=iota, G=G)


def _dense_jacobian_scaling_limit_result(
    x0,
    *,
    max_dense_jacobian_bytes,
    message="dense Jacobian skipped",
    residual=None,
):
    n = x0.shape[0]
    if residual is None:
        residual = jnp.zeros(n, dtype=x0.dtype)
    report = _opt._exact_newton_dense_jacobian_report(
        n,
        n,
        x0.dtype,
        max_dense_jacobian_bytes,
    )
    return {
        "x": x0,
        "residual": residual,
        "jacobian": None,
        "nit": 0,
        "success": True,
        "message": message,
        "failure_category": "scaling_limit",
        "failure_stage": "dense_jacobian_finalization",
        "jacobian_materialized": False,
        **report,
    }


def _operator_only_exact_newton_result(
    x0,
    *,
    residual=None,
):
    n = x0.shape[0]
    if residual is None:
        residual = jnp.zeros(n, dtype=x0.dtype)
    return {
        "x": x0,
        "residual": residual,
        "jacobian": None,
        "nit": 0,
        "success": True,
        "message": None,
        "failure_category": None,
        "failure_stage": None,
        "jacobian_materialized": False,
    }


def _successful_exact_newton_result(
    x0,
    *,
    step=0.0,
    nit=0,
    max_dense_jacobian_bytes=None,
):
    n = x0.shape[0]
    return {
        "x": x0 + step,
        "residual": jnp.zeros(n, dtype=x0.dtype),
        "jacobian": jnp.eye(n, dtype=x0.dtype),
        "nit": nit,
        "success": True,
        "message": None,
        "jacobian_materialized": True,
        "max_dense_jacobian_bytes": max_dense_jacobian_bytes,
    }


@contextmanager
def _patched_exact_newton_result(*, success, step=0.1, nit=3):
    original_newton_exact = _bsj.newton_exact

    def fake_newton_exact(
        _residual_fn, x0, *, maxiter, tol, max_dense_jacobian_bytes=None
    ):
        del maxiter, tol, max_dense_jacobian_bytes
        result = _successful_exact_newton_result(x0, step=step, nit=nit)
        result["success"] = success
        return result

    _bsj.newton_exact = fake_newton_exact
    try:
        yield
    finally:
        _bsj.newton_exact = original_newton_exact


def _run_mock_exact_boozer_success(booz, iota=0.3, G=0.05):
    with _patched_exact_newton_result(success=True):
        return _run_mock_exact_boozer(booz, iota=iota, G=G)


class TestBoozerSurfaceJAXExactPath:
    """Test the exact (Newton) path of BoozerSurfaceJAX.

    Validates:
    - Exact-type instantiation and boozer_type.
    - run_code() exact-path convergence.
    - Result dict contract parity with CPU BoozerSurface.
    - Mask is boolean (not integer indices).
    - Residual is raw unmasked (full grid size).
    """

    def test_exact_instantiation(self):
        """constraint_weight=None yields boozer_type='exact'."""
        booz = _make_mock_boozer_surface_exact()
        assert booz.boozer_type == "exact"
        assert booz.constraint_weight is None

    def test_exact_mask_indices_use_cached_runtime_state(self):
        """Exact-path mask construction should not call the live surface getter."""
        booz = _make_mock_boozer_surface_exact(stellsym=True)
        expected = np.asarray(booz._compute_stellsym_mask_indices(), dtype=np.int32)
        booz._exact_mask_indices = None

        def _unexpected_get_mask():
            raise AssertionError(
                "live surface get_stellsym_mask() should not be queried"
            )

        booz.surface.get_stellsym_mask = _unexpected_get_mask

        np.testing.assert_array_equal(
            np.asarray(booz._compute_stellsym_mask_indices(), dtype=np.int32),
            expected,
        )

    def test_run_code_exact_converges(self):
        """run_code() exact path runs and returns a result dict."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        assert res is not None
        _assert_result_schema(res, _PUBLIC_EXACT_RESULT_SCHEMA)
        assert res["type"] == "exact"
        assert booz.need_to_run_code is False

    def test_exact_result_dict_keys(self):
        """Exact-path result dict has all CPU-contract keys."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        expected_keys = {
            "residual",
            "fun",
            "jacobian",
            "iter",
            "success",
            "G",
            "s",
            "iota",
            "PLU",
            "mask",
            "type",
            "vjp",
            "jacobian_materialized",
            "linear_solve_backend",
            "dense_linear_solve_factors_available",
            "exact_factorization_backend",
        }
        assert expected_keys <= set(res.keys())
        _assert_result_schema(res, _PUBLIC_EXACT_RESULT_SCHEMA)
        assert res["jacobian_materialized"] is True
        assert isinstance(res["PLU"], tuple)
        assert len(res["PLU"]) == 3
        assert all(piece is not None for piece in res["PLU"])
        assert res["linear_solve_backend"] == "operator"
        assert res["dense_linear_solve_factors_available"] is True
        assert res["exact_factorization_backend"] == _bsj.EXACT_FACTORIZATION_BACKEND
        assert res["vjp"] is _boozer_exact_coil_vjp
        assert callable(res["vjp"])

    def test_resolved_coil_set_spec_uses_explicit_coil_arrays(self):
        """coil_arrays-only calls must build an explicit grouped spec."""
        booz = _make_mock_boozer_surface_exact()
        override_coil_arrays = tuple(
            (
                gammas,
                gammadashs,
                currents + 0.25,
            )
            for gammas, gammadashs, currents in booz._coil_arrays
        )
        resolved = _bsj._resolved_coil_set_spec(
            booz.coil_set_spec,
            coil_arrays=override_coil_arrays,
        )
        assert resolved is not booz.coil_set_spec
        resolved_groups = grouped_field_data_from_spec(resolved)
        for resolved_group, override_group in zip(
            resolved_groups,
            override_coil_arrays,
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(resolved_group[2]),
                np.asarray(override_group[2]),
            )

    def test_exact_fun_tracks_exact_system_residual(self):
        """Exact-path fun must reflect the actual Newton system residual."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = booz._make_exact_residual(mask_indices)
        x_final = booz._pack_decision_vector(res["iota"], res["G"])
        expected_fun = float(0.5 * jnp.mean(jnp.square(res_fn(x_final))))
        assert res["fun"] == pytest.approx(expected_fun)

    def test_ls_surface_exact_newton_has_default_dense_jacobian_ceiling(
        self, monkeypatch
    ):
        """LS-constructed surfaces must still provide the exact-solve memory cap."""
        booz = _make_mock_boozer_surface_exact(constraint_weight=1.0)
        captured = {}
        original_newton_exact = _bsj.newton_exact

        def fake_newton_exact(
            _residual_fn, x0, *, maxiter, tol, max_dense_jacobian_bytes=None
        ):
            del maxiter, tol
            captured["max_dense_jacobian_bytes"] = max_dense_jacobian_bytes
            return _successful_exact_newton_result(
                x0,
                max_dense_jacobian_bytes=max_dense_jacobian_bytes,
            )

        monkeypatch.setattr(_bsj, "newton_exact", fake_newton_exact)
        try:
            with _patched_exact_surface_module():
                res = booz.solve_residual_equation_exactly_newton(iota=0.3, G=0.05)
        finally:
            monkeypatch.setattr(_bsj, "newton_exact", original_newton_exact)

        assert (
            captured["max_dense_jacobian_bytes"]
            == _bsj._DEFAULT_MAX_DENSE_JACOBIAN_BYTES
        )
        assert res["max_dense_jacobian_bytes"] == _bsj._DEFAULT_MAX_DENSE_JACOBIAN_BYTES

    def test_run_code_exact_reports_scaling_limit_failure_without_fake_success(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface_exact(options={"max_dense_jacobian_bytes": 77})
        captured = {}
        original_newton_exact = _bsj.newton_exact

        def fake_newton_exact(
            _residual_fn, x0, *, maxiter, tol, max_dense_jacobian_bytes=None
        ):
            del maxiter, tol
            captured["max_dense_jacobian_bytes"] = max_dense_jacobian_bytes
            return _dense_jacobian_scaling_limit_result(
                x0,
                max_dense_jacobian_bytes=max_dense_jacobian_bytes,
            )

        monkeypatch.setattr(_bsj, "newton_exact", fake_newton_exact)
        try:
            res = _run_mock_exact_boozer(booz)
        finally:
            monkeypatch.setattr(_bsj, "newton_exact", original_newton_exact)

        assert captured["max_dense_jacobian_bytes"] == 77
        assert res["jacobian"] is None
        assert res["PLU"] is None
        assert res["success"] is False
        assert res["primal_success"] is False
        assert res["adjoint_linear_solve_available"] is False
        assert res["failure_category"] == "scaling_limit"
        assert res["failure_stage"] == "dense_jacobian_finalization"
        assert res["exact_factorization_backend"] == _bsj.EXACT_FACTORIZATION_BACKEND
        assert res["jacobian_materialized"] is False
        assert res["dense_jacobian_shape"] is not None
        assert res["dense_jacobian_bytes"] is not None
        assert res["max_dense_jacobian_bytes"] == 77
        assert res["message"] == "dense Jacobian skipped"
        _assert_result_schema(res, _PUBLIC_EXACT_RESULT_SCHEMA)
        with pytest.raises(RuntimeError, match="no successful solve state"):
            booz.get_adjoint_runtime_state()

    def test_run_code_exact_reports_dense_jacobian_ceiling_when_verbose(
        self, monkeypatch, capsys
    ):
        booz = _make_mock_boozer_surface_exact(options={"max_dense_jacobian_bytes": 77})
        original_newton_exact = _bsj.newton_exact

        def fake_newton_exact(
            _residual_fn, x0, *, maxiter, tol, max_dense_jacobian_bytes=None
        ):
            del maxiter, tol
            return _dense_jacobian_scaling_limit_result(
                x0,
                max_dense_jacobian_bytes=max_dense_jacobian_bytes,
                message=(
                    "Exact Newton skipped dense Jacobian materialization because "
                    "the final Jacobian would exceed max_dense_jacobian_bytes=77."
                ),
            )

        monkeypatch.setattr(_bsj, "newton_exact", fake_newton_exact)
        try:
            _run_mock_exact_boozer(booz)
        finally:
            monkeypatch.setattr(_bsj, "newton_exact", original_newton_exact)

        captured = capsys.readouterr()
        assert "Exact Newton skipped dense Jacobian materialization" in captured.out
        assert "max_dense_jacobian_bytes=77" in captured.out

    def test_exact_residual_jits_with_integer_mask_indices(self):
        """The exact residual closure must trace with integer mask indices."""
        booz = _make_mock_boozer_surface_exact()
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = jax.jit(booz._make_exact_residual(mask_indices))
        x = booz._pack_decision_vector(0.3, 0.05)
        residual = res_fn(x)
        assert residual.shape == (mask_indices.shape[0] + 2,)
        assert jnp.all(jnp.isfinite(residual))

    def test_run_code_traceable_accepts_grouped_coil_spec_source(self, monkeypatch):
        """Traceable exact solves must accept ``GroupedCoilSetSpec`` directly."""
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        captured = {}

        def fake_newton_exact_traceable(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            args=(),
        ):
            del maxiter, tol
            residual = residual_fn(x0, *args)
            captured["residual"] = residual
            return {
                "x": x0,
                "residual": residual,
                "jacobian": None,
                "nit": 0,
                "success": True,
                "message": None,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["type"] == "exact"
        assert bool(result["success"])
        assert result["plu"] is None
        assert result["linear_solve_backend"] == "operator"
        assert result["dense_linear_solve_factors_available"] is False
        assert "residual" in captured
        assert jnp.all(jnp.isfinite(captured["residual"]))

    def test_run_code_traceable_exact_uses_operator_only_newton(self, monkeypatch):
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        captured = {}

        def fake_newton_exact_traceable(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            args=(),
        ):
            del maxiter, tol
            captured["called"] = True
            residual = residual_fn(x0, *args)
            return _operator_only_exact_newton_result(
                x0,
                residual=residual,
            )

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        _assert_result_schema(result, _TRACEABLE_EXACT_RESULT_SCHEMA)
        assert captured["called"] is True
        assert result["jacobian"] is None
        assert result["plu"] is None
        assert result["linear_solve_backend"] == "operator"
        assert result["dense_linear_solve_factors_available"] is False
        assert bool(result["success"]) is True
        assert bool(result["primal_success"]) is True
        assert bool(result["adjoint_linear_solve_available"]) is True
        assert result["failure_category"] is None
        assert result["failure_stage"] is None
        assert result["jacobian_materialized"] is False
        assert result["max_dense_jacobian_bytes"] is None
        assert result["message"] is None
        for field in _bsj.SOLVE_QUALITY_EXACT_FIELDS:
            assert field in result
        assert result["exact_factorization_backend"] == _bsj.EXACT_FACTORIZATION_BACKEND

    def test_run_code_traceable_exact_reuses_stable_residual_callable(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        captured_ids = []
        res_before = booz.res
        need_to_run_before = booz.need_to_run_code

        def fake_newton_exact_traceable(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            args=(),
        ):
            del maxiter, tol
            captured_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            return {
                "x": x0,
                "residual": residual,
                "jacobian": None,
                "nit": 0,
                "success": True,
                "message": None,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert first is not second
        assert len(captured_ids) == 2
        assert captured_ids[0] == captured_ids[1]
        for key in ("x", "sdofs", "iota", "G", "fun", "residual"):
            np.testing.assert_allclose(np.asarray(first[key]), np.asarray(second[key]))
        assert first["nit"] == second["nit"]
        assert first["type"] == second["type"] == "exact"
        assert first["weight_inv_modB"] == second["weight_inv_modB"]
        assert booz.res is res_before
        assert booz.need_to_run_code is need_to_run_before

    def test_run_code_traceable_exact_rebuilds_residual_callable_after_target_change(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        residual_ids = []
        residuals = []

        def fake_newton_exact_traceable(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            args=(),
        ):
            del maxiter, tol
            residual_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            residuals.append(np.asarray(residual))
            return {
                "x": x0,
                "residual": residual,
                "jacobian": None,
                "nit": 0,
                "success": True,
                "message": None,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        booz.targetlabel = float(booz.targetlabel + 0.5)
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert residual_ids[0] != residual_ids[1]
        assert not np.allclose(residuals[0], residuals[1])

    def test_run_code_traceable_exact_rebuilds_residual_callable_after_option_change(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        residual_ids = []

        def fake_newton_exact_traceable(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            args=(),
        ):
            del maxiter, tol
            residual_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            return {
                "x": x0,
                "residual": residual,
                "jacobian": None,
                "nit": 0,
                "success": True,
                "message": None,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        booz.options["weight_inv_modB"] = not booz.options["weight_inv_modB"]
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert residual_ids[0] != residual_ids[1]
        assert first["weight_inv_modB"] is not second["weight_inv_modB"]

    def test_run_code_traceable_exact_executes_inner_solve_on_gpu(
        self,
        monkeypatch,
        request,
    ):
        gpu = parity_device("gpu")
        _enable_fast_strict_jax_backend(monkeypatch, request)
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec, sdofs, iota, G, x_target, A = (
            _build_gpu_traceable_linear_problem(
                booz,
                gpu,
                step_scale=0.05,
            )
        )
        dense_calls = _patch_matrix_free_exact_linear_solver(
            monkeypatch,
            A=A,
            expected_device=gpu,
        )

        def fake_get_traceable_exact_residual(_weight_inv_modB):
            def residual(x, _coil_set_spec):
                return _matrix_constant(A, x) @ x - _matrix_constant(x_target, x)

            return residual

        monkeypatch.setattr(
            booz,
            "_get_traceable_exact_residual",
            fake_get_traceable_exact_residual,
        )

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert len(dense_calls) == 1
        assert result["type"] == "exact"
        assert bool(np.asarray(jax.device_get(result["success"])))
        _assert_traceable_gpu_result(
            result,
            x_target,
            gpu,
            jacobian_shape=A.shape,
        )

    def test_run_code_traceable_ls_reuses_newton_fun_and_grad(self, monkeypatch):
        """LS traceable path must reuse Newton outputs instead of re-differentiating."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        expected_fun = jnp.asarray(3.5, dtype=jnp.float64)
        expected_grad = jnp.arange(
            sdofs.size + 2,
            dtype=jnp.float64,
        )

        def fake_minimize(_fun, x0, **_kwargs):
            return types.SimpleNamespace(x_k=x0)

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del obj_fn, maxiter, tol, stab, progress_callback, objective_args
            return {
                "x": x0,
                "fun": expected_fun,
                "grad": expected_grad,
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "nit": 2,
                "success": True,
            }

        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", fake_minimize)
        monkeypatch.setattr(
            _bsj.jax,
            "value_and_grad",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("run_code_traceable() should reuse Newton fun/grad")
            ),
        )
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["type"] == "ls"
        _assert_result_schema(result, _TRACEABLE_LS_RESULT_SCHEMA)
        # ``ls_condition_estimate`` is populated eagerly from the materialized
        # Hessian on the traceable LS path (Phase 5), so it is no longer ``None``
        # even with the fake newton-polish runner here. The κ_1 of the identity
        # Hessian is 1.0 by definition. Other LS quality fields remain ``None``
        # placeholders until the parity arbiter populates them.
        condition_estimate = result["ls_condition_estimate"]
        assert condition_estimate == pytest.approx(1.0)
        for field in _bsj.SOLVE_QUALITY_LS_FIELDS:
            if field == "ls_condition_estimate":
                continue
            assert result[field] is None
        assert bool(result["success"])
        np.testing.assert_allclose(np.asarray(result["fun"]), np.asarray(expected_fun))
        np.testing.assert_allclose(
            np.asarray(result["grad"]),
            np.asarray(expected_grad),
        )

    @pytest.mark.parametrize(
        ("explicit_materialize", "expected_materialize"),
        [
            (None, True),
            (True, True),
        ],
    )
    def test_run_code_traceable_ls_routes_lm_ondevice(
        self,
        monkeypatch,
        explicit_materialize,
        expected_materialize,
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"
        if explicit_materialize is not None:
            booz.options["materialize_dense_linearization"] = explicit_materialize
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        captured = {}

        def forbidden_private_minimize(*_args, **_kwargs):
            raise AssertionError("LM route should not enter private BFGS/L-BFGS")

        def fake_lm(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            materialize_dense_linearization=True,
            max_dense_linearization_bytes=None,
            callback=None,
            progress_callback=None,
            args=(),
        ):
            del maxiter, tol, callback, progress_callback
            captured["materialize_dense_linearization"] = (
                materialize_dense_linearization
            )
            captured["max_dense_linearization_bytes"] = max_dense_linearization_bytes
            captured["residual"] = residual_fn(x0, *args)
            return {
                "x": x0,
                "residual": captured["residual"],
                "residual_jacobian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "fun": jnp.asarray(1.25, dtype=x0.dtype),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "damping": jnp.asarray(1.0e-3, dtype=x0.dtype),
                "nit": jnp.asarray(1, dtype=jnp.int32),
                "status": jnp.asarray(0, dtype=jnp.int32),
                "success": jnp.asarray(True),
            }

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del obj_fn, maxiter, tol, stab, progress_callback, objective_args
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_opt, "_minimize_bfgs_private", forbidden_private_minimize)
        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", forbidden_private_minimize)
        monkeypatch.setattr(_bsj, "levenberg_marquardt_traceable", fake_lm)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        _assert_result_schema(result, _TRACEABLE_LS_RESULT_SCHEMA)
        assert result["optimizer_method"] == "lm-ondevice"
        assert bool(result["success"])
        assert captured["materialize_dense_linearization"] is expected_materialize
        assert (
            captured["max_dense_linearization_bytes"]
            == booz.options["max_dense_linearization_bytes"]
        )
        assert jnp.all(jnp.isfinite(captured["residual"]))

    def test_run_code_traceable_lm_ondevice_executes_inner_solve_on_gpu(
        self,
        monkeypatch,
        request,
    ):
        gpu = parity_device("gpu")
        _enable_fast_strict_jax_backend(monkeypatch, request)
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"
        coil_set_spec, sdofs, iota, G, x_target, A = (
            _build_gpu_traceable_linear_problem(
                booz,
                gpu,
                step_scale=0.02,
            )
        )
        dense_calls, gmres_calls = _patch_matrix_free_lm_solver(
            monkeypatch,
            A=A,
            expected_device=gpu,
        )

        def fake_get_traceable_penalty_residual(_optimize_G, _weight_inv_modB):
            def residual(x, _coil_set_spec):
                return _matrix_constant(A, x) @ x - _matrix_constant(x_target, x)

            return residual

        def fake_newton_polish(
            method,
            obj_fn,
            x_ls,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del method, obj_fn, maxiter, tol, stab, progress_callback, objective_args
            np.testing.assert_allclose(
                np.asarray(jax.device_get(x_ls)),
                np.asarray(x_target),
                atol=1e-12,
            )
            return {
                "x": x_ls,
                "fun": _explicit_scalar(0.0, dtype=x_ls.dtype, device=gpu),
                "grad": x_ls - x_ls,
                "hessian": _explicit_eye(
                    x_ls.shape[0],
                    dtype=x_ls.dtype,
                    device=gpu,
                ),
                "nit": _explicit_scalar(0, dtype=jnp.int32, device=gpu),
                "success": _explicit_scalar(True, dtype=np.bool_, device=gpu),
            }

        monkeypatch.setattr(
            booz,
            "_get_traceable_penalty_residual",
            fake_get_traceable_penalty_residual,
        )
        monkeypatch.setattr(
            booz,
            "_run_newton_polish_for_method",
            fake_newton_polish,
        )

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert gmres_calls
        assert len(dense_calls) == 1
        assert result["type"] == "ls"
        assert result["optimizer_method"] == "lm-ondevice"
        assert bool(np.asarray(jax.device_get(result["success"])))
        _assert_traceable_gpu_result(
            result,
            x_target,
            gpu,
            hessian_shape=A.shape,
        )

    def test_run_code_traceable_lm_reuses_stable_residual_and_objective_callables(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        residual_ids = []
        objective_ids = []
        res_before = booz.res
        need_to_run_before = booz.need_to_run_code

        def fake_lm(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            materialize_dense_linearization=True,
            max_dense_linearization_bytes=None,
            callback=None,
            progress_callback=None,
            args=(),
        ):
            del (
                maxiter,
                tol,
                materialize_dense_linearization,
                max_dense_linearization_bytes,
                callback,
                progress_callback,
            )
            residual_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            return {
                "x": x0,
                "residual": residual,
                "residual_jacobian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "fun": jnp.asarray(1.25, dtype=x0.dtype),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "damping": jnp.asarray(1.0e-3, dtype=x0.dtype),
                "nit": jnp.asarray(1, dtype=jnp.int32),
                "status": jnp.asarray(0, dtype=jnp.int32),
                "success": jnp.asarray(True),
            }

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback
            objective_ids.append(id(obj_fn))
            obj_fn(x0, *objective_args)
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_opt, "_minimize_bfgs_private", lambda *_a, **_k: None)
        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", lambda *_a, **_k: None)
        monkeypatch.setattr(_bsj, "levenberg_marquardt_traceable", fake_lm)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert first is not second
        assert residual_ids == [residual_ids[0], residual_ids[0]]
        assert objective_ids == [objective_ids[0], objective_ids[0]]
        for key in ("x", "sdofs", "iota", "G", "fun", "grad", "hessian"):
            np.testing.assert_allclose(np.asarray(first[key]), np.asarray(second[key]))
        assert first["nit"] == second["nit"]
        assert first["optimizer_method"] == second["optimizer_method"] == "lm-ondevice"
        assert first["type"] == second["type"] == "ls"
        assert first["weight_inv_modB"] == second["weight_inv_modB"]
        assert booz.res is res_before
        assert booz.need_to_run_code is need_to_run_before

    def test_run_code_traceable_lm_rebuilds_callables_after_target_change(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        residual_ids = []
        objective_ids = []
        residuals = []
        objective_values = []

        def fake_lm(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            materialize_dense_linearization=True,
            max_dense_linearization_bytes=None,
            callback=None,
            progress_callback=None,
            args=(),
        ):
            del (
                maxiter,
                tol,
                materialize_dense_linearization,
                max_dense_linearization_bytes,
                callback,
                progress_callback,
            )
            residual_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            residuals.append(np.asarray(residual))
            return {
                "x": x0,
                "residual": residual,
                "residual_jacobian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "fun": jnp.asarray(1.25, dtype=x0.dtype),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "damping": jnp.asarray(1.0e-3, dtype=x0.dtype),
                "nit": jnp.asarray(1, dtype=jnp.int32),
                "status": jnp.asarray(0, dtype=jnp.int32),
                "success": jnp.asarray(True),
            }

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback
            objective_ids.append(id(obj_fn))
            objective_values.append(np.asarray(obj_fn(x0, *objective_args)))
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_opt, "_minimize_bfgs_private", lambda *_a, **_k: None)
        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", lambda *_a, **_k: None)
        monkeypatch.setattr(_bsj, "levenberg_marquardt_traceable", fake_lm)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        booz.targetlabel = float(booz.targetlabel + 0.5)
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert residual_ids[0] != residual_ids[1]
        assert objective_ids[0] != objective_ids[1]
        assert not np.allclose(residuals[0], residuals[1])
        assert not np.allclose(objective_values[0], objective_values[1])

    def test_run_code_traceable_ls_skips_lu_for_nonfinite_newton_result(
        self, monkeypatch
    ):
        """LS traceable failures must return the dummy PLU payload."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)

        def fake_minimize(_fun, x0, **_kwargs):
            return types.SimpleNamespace(x_k=x0)

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del obj_fn, maxiter, tol, stab, progress_callback, objective_args
            return {
                "x": x0,
                "fun": jnp.asarray(jnp.nan, dtype=x0.dtype),
                "grad": jnp.full_like(x0, jnp.nan),
                "hessian": jnp.full(
                    (x0.shape[0], x0.shape[0]), jnp.nan, dtype=x0.dtype
                ),
                "nit": 0,
                "success": False,
            }

        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", fake_minimize)
        monkeypatch.setattr(
            _bsj.jax.scipy.linalg,
            "lu",
            lambda *_args, **_kwargs: _assert_lu_is_not_called(
                "non-finite traceable LS solves must skip LU"
            ),
        )
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["type"] == "ls"
        assert bool(result["success"]) is False
        assert result["plu"] is not None
        for factor in result["plu"]:
            assert np.all(np.isnan(np.asarray(factor))), (
                "Dummy PLU for failed solves must be NaN-filled to prevent "
                "silent zero-gradient propagation"
            )

    def test_run_code_functional_aliases_run_code_traceable_schema(self, monkeypatch):
        """run_code_functional() should forward to the runtime-native traceable schema."""
        booz = _make_mock_boozer_surface()
        coil_arrays = booz._coil_arrays
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        expected = {
            "x": booz._pack_decision_vector(iota, G, sdofs=sdofs),
            "sdofs": sdofs,
            "iota": iota,
            "G": G,
            "fun": jnp.asarray(1.25, dtype=jnp.float64),
            "grad": jnp.arange(sdofs.size + 2, dtype=jnp.float64),
            "hessian": jnp.eye(sdofs.size + 2, dtype=jnp.float64),
            "plu": tuple(jnp.eye(sdofs.size + 2, dtype=jnp.float64) for _ in range(3)),
            "nit": jnp.asarray(3, dtype=jnp.int32),
            "success": jnp.asarray(True),
            "primal_success": jnp.asarray(True),
            "adjoint_linear_solve_available": jnp.asarray(True),
            "linearization_kind": "hessian",
            "linear_solve_backend": "operator",
            "dense_linear_solve_factors_available": True,
            "optimizer_method": "lbfgs-ondevice",
            "type": "ls",
            "weight_inv_modB": booz.options["weight_inv_modB"],
            **_bsj._none_solve_quality_fields(_bsj.SOLVE_QUALITY_LS_FIELDS),
            "hessian_materialized": None,
            "dense_hessian_shape": None,
            "dense_hessian_bytes": None,
            "max_dense_hessian_bytes": None,
            "dense_newton_steps_materialized": None,
            "dense_newton_steps_message": None,
            "newton_iter": None,
            "final_gradient_norm": None,
            "final_gradient_inf_norm": None,
            "iterative_refinement_ran": None,
            "final_step_iterative_refinement_ran": None,
            "dense_refinement_ran": None,
            "final_step_dense_refinement_ran": None,
            "failure_category": None,
            "failure_stage": None,
            "message": None,
        }

        def fake_run_code_traceable(coil_source, sdofs_arg, iota_arg, G_arg):
            if coil_source is not coil_arrays:
                raise AssertionError("unexpected functional call")
            if not np.allclose(np.asarray(sdofs_arg), np.asarray(sdofs)):
                raise AssertionError("unexpected functional call")
            if not np.allclose(np.asarray(iota_arg), np.asarray(iota)):
                raise AssertionError("unexpected functional call")
            if not np.allclose(np.asarray(G_arg), np.asarray(G)):
                raise AssertionError("unexpected functional call")
            return expected

        monkeypatch.setattr(
            booz,
            "run_code_traceable",
            fake_run_code_traceable,
        )

        result = booz.run_code_functional(coil_arrays, sdofs, iota, G)

        assert result is expected
        _assert_result_schema(result, _TRACEABLE_LS_RESULT_SCHEMA)
        assert "PLU" not in result
        assert result["plu"] is not None
        np.testing.assert_allclose(np.asarray(result["sdofs"]), np.asarray(sdofs))

    def test_run_code_traceable_exact_skips_lu_for_nonfinite_newton_result(
        self, monkeypatch
    ):
        """Exact traceable failures must not build or carry PLU payloads."""
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)

        def fake_newton_exact_traceable(
            _residual_fn,
            x0,
            *,
            maxiter,
            tol,
            args=(),
        ):
            del maxiter, tol, args
            n = x0.shape[0]
            return {
                "x": x0,
                "residual": jnp.full((n - 1,), jnp.nan, dtype=x0.dtype),
                "jacobian": None,
                "nit": 0,
                "success": False,
                "message": None,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)
        monkeypatch.setattr(
            _bsj.jax.scipy.linalg,
            "lu",
            lambda *_args, **_kwargs: _assert_lu_is_not_called(
                "non-finite traceable exact solves must skip LU"
            ),
        )

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["type"] == "exact"
        assert bool(result["success"]) is False
        assert result["plu"] is None
        assert result["linear_solve_backend"] == "operator"
        assert result["dense_linear_solve_factors_available"] is False
        assert result["exact_factorization_backend"] == _bsj.EXACT_FACTORIZATION_BACKEND

    def test_exact_accepts_and_ignores_optimizer_backend_option(self):
        """Exact solves accept optimizer_backend but ignore it."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=None,
            options={"optimizer_backend": "ondevice"},
        )

        assert booz.boozer_type == "exact"
        assert "optimizer_backend" not in booz.options

    def test_exact_accepts_stage_callback_option(self):
        """Exact solves must accept stage_callback because init probes thread it in."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)

        def stage_callback(_label, **_payload):
            return None

        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=None,
            options={"stage_callback": stage_callback},
        )

        assert booz.boozer_type == "exact"
        assert booz.options["stage_callback"] is stage_callback

    def test_exact_rejects_invalid_optimizer_backend_value(self):
        """Exact solves still validate optimizer_backend values."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _PlumbingVolumeLabel(surf)
        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=None,
                options={"optimizer_backend": "bogus"},
            )

    def test_exact_mask_is_boolean(self):
        """CPU contract: mask is a boolean array, not integer indices."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        mask = res["mask"]
        assert mask.dtype == np.bool_, f"mask dtype should be bool, got {mask.dtype}"
        nphi = len(booz.quadpoints_phi)
        ntheta = len(booz.quadpoints_theta)
        assert mask.shape == (3 * nphi * ntheta,)

    def test_exact_residual_is_raw_unmasked(self):
        """CPU contract: residual is the full unmasked Boozer residual."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        nphi = len(booz.quadpoints_phi)
        ntheta = len(booz.quadpoints_theta)
        assert res["residual"].shape == (3 * nphi * ntheta,), (
            f"residual shape should be {(3 * nphi * ntheta,)}, "
            f"got {res['residual'].shape}"
        )

    def test_exact_mask_selects_from_residual(self):
        """mask can index into residual (CPU pattern: r[mask])."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        masked_r = res["residual"][res["mask"]]
        assert masked_r.ndim == 1
        assert len(masked_r) <= len(res["residual"])
        assert len(masked_r) == int(res["mask"].sum())

    def test_exact_idempotent(self):
        """Second run_code() returns None when not dirty."""
        booz = _make_mock_boozer_surface_exact()
        _run_mock_exact_boozer_success(booz)
        assert booz.run_code(iota=0.3, G=0.05) is None

    def test_exact_invalid_newton_iterate_aborts_adjoint_state(self):
        """Exact-path failures must not expose dense metadata/VJP placeholders."""
        booz = _make_mock_boozer_surface_exact()
        dofs_before = booz.surface.get_dofs()

        with _patched_exact_newton_result(success=False, step=jnp.nan, nit=0):
            res = _run_mock_exact_boozer(booz)

        assert res["success"] is False
        assert res["PLU"] is None
        assert res["vjp"] is None
        assert res["mask"] is None
        np.testing.assert_allclose(booz.surface.get_dofs(), dofs_before)

    def test_exact_unsuccessful_finite_newton_exit_aborts_adjoint_state(self):
        """Finite exact-Newton failures must not publish solved adjoint state."""
        booz = _make_mock_boozer_surface_exact()
        dofs_before = booz.surface.get_dofs()

        with _patched_exact_newton_result(success=False):
            res = _run_mock_exact_boozer(booz)

        assert res["success"] is False
        assert res["PLU"] is None
        assert res["vjp"] is None
        assert res["mask"] is None
        np.testing.assert_allclose(booz.surface.get_dofs(), dofs_before)

    def test_newton_residual_uses_call_weight_override(self, monkeypatch):
        """Newton residual reconstruction must respect the call-time weighting flag."""
        booz = _make_mock_boozer_surface()
        captured = {}

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return {
                "x": x0,
                "fun": jnp.asarray(0.0, dtype=x0.dtype),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "nit": 0,
                "success": True,
            }

        def fake_boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB):
            del G, iota, B
            captured["weight_inv_modB"] = bool(weight_inv_modB)
            nphi, ntheta = xphi.shape[:2]
            return jnp.zeros((3 * nphi * ntheta,), dtype=xtheta.dtype)

        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)
        monkeypatch.setattr(
            _bsj,
            "boozer_residual_vector",
            fake_boozer_residual_vector,
        )

        res = booz.minimize_boozer_penalty_constraints_newton(
            constraint_weight=booz.constraint_weight,
            iota=0.3,
            G=0.05,
            weight_inv_modB=False,
            verbose=False,
        )

        assert captured["weight_inv_modB"] is False
        assert res["weight_inv_modB"] is False


# ---------------------------------------------------------------------------
# P2 #5: VJP hook tests
# ---------------------------------------------------------------------------


def _mock_ls_group_vjp_case():
    booz = _make_mock_boozer_surface()
    res = booz.run_code(iota=0.3, G=0.05)
    lm = jnp.asarray(np.eye(res["jacobian"].shape[0])[0], dtype=jnp.float64)
    return booz, res, res["vjp_groups"], lm


class TestVJPHooks:
    """Test the VJP hooks stored in result dicts."""

    def test_boozer_coil_vjp_callback_signatures_are_stable(self):
        exact_signature = inspect.signature(_bsj._boozer_exact_coil_vjp)
        ls_signature = inspect.signature(_bsj._boozer_ls_coil_vjp)

        assert tuple(exact_signature.parameters) == ("lm", "booz_surf", "iota", "G")
        assert tuple(ls_signature.parameters) == (
            "lm",
            "booz_surf",
            "iota",
            "G",
            "weight_inv_modB",
        )
        assert ls_signature.parameters["weight_inv_modB"].default is True

    def test_ls_vjp_returns_correct_shapes(self):
        """LS VJP returns cotangent arrays with correct shapes."""
        booz = _make_mock_boozer_surface()
        res = booz.run_code(iota=0.3, G=0.05)
        vjp_fn = res["vjp"]
        iota_sol = res["iota"]
        G_sol = res["G"]

        # lm has same shape as the decision vector (gradient)
        lm = np.zeros_like(res["jacobian"])
        lm[0] = 1.0

        d_coil_arrays, coil_indices = vjp_fn(jnp.asarray(lm), booz, iota_sol, G_sol)
        # d_coil_arrays is a list of (d_g, d_gd, d_c) tuples, one per group
        assert len(d_coil_arrays) == len(booz.coil_groups)
        for (d_g, d_gd, d_c), (g, gd, c, _) in zip(d_coil_arrays, booz.coil_groups):
            assert d_g.shape == g.shape
            assert d_gd.shape == gd.shape
            assert d_c.shape == c.shape

    def test_ls_group_vjp_uses_grouped_spec_path(self, monkeypatch):
        """Streaming LS VJP should avoid rebuilding full grouped input arrays."""
        booz, res, vjp_groups_fn, lm = _mock_ls_group_vjp_case()

        def _forbid_legacy_group_array_path(*_args, **_kwargs):
            raise AssertionError(
                "LS grouped VJP should not rebuild full grouped input arrays"
            )

        monkeypatch.setattr(
            _bsj,
            "_replace_group_coil_array",
            _forbid_legacy_group_array_path,
        )
        monkeypatch.setattr(
            _bsj,
            "grouped_biot_savart_B_from_inputs",
            _forbid_legacy_group_array_path,
        )
        monkeypatch.setattr(
            _bsj,
            "grouped_biot_savart_A_from_inputs",
            _forbid_legacy_group_array_path,
        )

        grouped_entries = list(vjp_groups_fn(lm, booz, res["iota"], res["G"]))

        assert len(grouped_entries) == len(booz.coil_groups)

    def test_ls_group_vjp_does_not_route_through_full_grouped_vjp(self, monkeypatch):
        """Streaming LS VJP should not materialize the full grouped cotangent pytree."""
        booz, res, vjp_groups_fn, lm = _mock_ls_group_vjp_case()

        def _forbid_full_grouped_vjp(*_args, **_kwargs):
            raise AssertionError(
                "LS grouped VJP should stay on the per-group streaming path"
            )

        monkeypatch.setattr(_bsj, "_boozer_ls_coil_vjp", _forbid_full_grouped_vjp)

        grouped_entries = list(vjp_groups_fn(lm, booz, res["iota"], res["G"]))

        assert len(grouped_entries) == len(booz.coil_groups)

    def test_ls_reduced_directional_requires_spatial_field_derivatives(self):
        """Symmetric FD catches the missing dB/dX * dgamma LS adjoint term."""
        booz = _make_mock_boozer_surface()
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        x, optimize_G = _bsj._ls_decision_vector(booz, iota, G)

        direction = np.zeros(int(x.shape[0]), dtype=np.float64)
        direction[: min(5, direction.size - 2)] = np.linspace(
            0.2,
            1.0,
            min(5, direction.size - 2),
        )
        direction /= np.linalg.norm(direction)
        direction = jnp.asarray(direction, dtype=jnp.float64)

        snapshot = _bsj._build_ls_grouped_vjp_snapshot(
            booz,
            iota,
            G,
            solve_generation=booz._solver_generation,
            weight_inv_modB=True,
        )
        full_directional = _bsj._ls_directional_from_field_terms(
            snapshot,
            snapshot.field_terms,
            direction,
        )
        dropped_spatial = _bsj._BoozerLocalFieldTerms(
            B=snapshot.field_terms.B,
            dB_dX=jnp.zeros_like(snapshot.field_terms.dB_dX),
        )
        dropped_directional = _bsj._ls_directional_from_field_terms(
            snapshot,
            dropped_spatial,
            direction,
        )

        objective = _bsj._make_ls_penalty_objective(
            booz,
            booz._coil_arrays,
            optimize_G,
            True,
        )
        eps = 1.0e-5
        fd_directional = (
            objective(x + eps * direction) - objective(x - eps * direction)
        ) / (2.0 * eps)

        np.testing.assert_allclose(
            np.asarray(full_directional),
            np.asarray(fd_directional),
            rtol=1e-8,
            atol=1e-10,
        )
        assert abs(float(dropped_directional - fd_directional)) > 1e-7

    @pytest.mark.parametrize("stellsym", [False, True])
    def test_ls_group_vjp_snapshot_static_metadata_is_hashable(self, stellsym):
        """Registered callback snapshots must be valid JIT pytrees."""
        booz = _make_mock_boozer_surface(stellsym=stellsym)
        snapshot = _bsj._build_ls_grouped_vjp_snapshot(
            booz,
            jnp.asarray(0.3, dtype=jnp.float64),
            jnp.asarray(0.05, dtype=jnp.float64),
            solve_generation=booz._solver_generation,
            weight_inv_modB=True,
        )

        assert all(isinstance(indices, tuple) for indices in snapshot.coil_indices)

        @jax.jit
        def snapshot_scalar(snap):
            return snap.iota + snap.G

        np.testing.assert_allclose(
            np.asarray(snapshot_scalar(snapshot)),
            np.asarray(snapshot.iota + snapshot.G),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_ls_group_vjp_toroidal_flux_matches_full_vjp(self):
        """Streaming LS VJP must include A and dA/dX terms for toroidal flux."""
        booz = _make_mock_toroidal_flux_boozer_surface()
        booz.need_to_run_code = False
        booz._solver_generation = 1
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        x, _ = _bsj._ls_decision_vector(booz, iota, G)
        lm = jnp.linspace(0.25, 1.25, int(x.shape[0]), dtype=jnp.float64)

        full_d_coil_arrays, full_coil_indices = _bsj._boozer_ls_coil_vjp(
            lm,
            booz,
            iota,
            G,
        )
        streamed = list(
            _bsj._boozer_ls_coil_vjp_groups(
                lm,
                booz,
                iota,
                G,
            )
        )

        assert [indices for _, indices in streamed] == full_coil_indices
        for (streamed_arrays, _), full_arrays in zip(streamed, full_d_coil_arrays):
            for streamed_arr, full_arr in zip(streamed_arrays, full_arrays):
                np.testing.assert_allclose(
                    np.asarray(streamed_arr, dtype=float),
                    np.asarray(full_arr, dtype=float),
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_ls_group_vjp_repeated_call_reuses_field_kernel_cache(self):
        """Warm repeated grouped VJPs should not grow the field-kernel cache."""
        booz, res, vjp_groups_fn, lm = _mock_ls_group_vjp_case()
        kernel = _biotsavart_jax_core._get_kernel(
            _biotsavart_jax_core._Integrand.B,
            _biotsavart_jax_core._DiffMode.VALUE_AND_JACOBIAN,
        )

        list(vjp_groups_fn(lm, booz, res["iota"], res["G"]))
        cache_size_after_first_call = kernel._cache_size()
        list(vjp_groups_fn(lm, booz, res["iota"], res["G"]))

        assert kernel._cache_size() == cache_size_after_first_call

    def test_add_G_current_cotangent_matches_abs_vjp_at_zero_current(self):
        """Closed-form G cotangent must match JAX's abs subgradient convention."""
        currents = jnp.asarray([-2.0, 0.0, 3.0], dtype=jnp.float64)
        group_array = (
            jnp.zeros((1, 2, 3), dtype=jnp.float64),
            jnp.zeros((1, 2, 3), dtype=jnp.float64),
            currents,
        )
        d_group = (
            jnp.zeros_like(group_array[0]),
            jnp.zeros_like(group_array[1]),
            jnp.ones_like(currents),
        )
        bar_G = jnp.asarray(2.5, dtype=jnp.float64)
        _, current_pullback = jax.vjp(compute_G_from_currents, currents)

        result = _bsj._add_G_current_cotangent(
            d_group,
            group_array,
            bar_G,
            optimize_G=False,
        )

        np.testing.assert_allclose(
            np.asarray(result[2]),
            np.asarray(d_group[2] + current_pullback(bar_G)[0]),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_run_code_rejects_bad_group_vjp_signature(self, monkeypatch):
        """run_code() must fail fast when grouped VJP hooks have the wrong arity."""
        booz = _make_mock_boozer_surface()

        def bad_vjp_groups(lm):
            del lm
            return ()

        monkeypatch.setattr(
            _bsj,
            "_build_ls_group_vjp_callback",
            lambda *args, **kwargs: bad_vjp_groups,
        )

        with pytest.raises(TypeError, match="vjp_groups"):
            booz.run_code(iota=0.3, G=0.05)

    def test_ls_group_vjp_detects_stale_reuse_after_resolve(self):
        """Grouped VJP hooks must reject stale reuse after a new solve."""
        booz, first, old_vjp_groups, lm = _mock_ls_group_vjp_case()

        booz.need_to_run_code = True
        second = booz.run_code(iota=0.3, G=0.05)
        assert second["solve_generation"] == first["solve_generation"] + 1

        with pytest.raises(RuntimeError, match="stale"):
            list(old_vjp_groups(lm, booz, first["iota"], first["G"]))


# ---------------------------------------------------------------------------
# P2 #6: Negative tests
# ---------------------------------------------------------------------------


class TestNegativeCases:
    """Test error handling for unsupported inputs."""

    def test_extract_grouped_coil_set_spec_rejects_legacy_coils_fallback(
        self,
    ):
        bs = _make_legacy_coils_list_biotsavart(_make_legacy_spec_capable_coils())
        with pytest.raises(AttributeError, match=_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN):
            _bsj._extract_grouped_coil_set_spec(bs)

    def test_extract_grouped_coil_set_spec_rejects_hidden_coils_even_if_spec_capable(
        self,
    ):
        coils = _make_curve_current_spec_only_legacy_coils()
        bs = _make_legacy_coils_list_biotsavart(coils)
        with pytest.raises(AttributeError, match=_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN):
            _bsj._extract_grouped_coil_set_spec(bs)

    def test_unsupported_label_raises(self):
        """Constructor rejects unsupported label types."""

        class AspectRatioLabel:
            def J(self):
                return 0.0

        nphi, ntheta = 4, 4
        qphi = np.linspace(0, 1.0, nphi, endpoint=False)
        qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
        sdofs = np.zeros(3 * 9)

        bs = _MockBiotSavart([_MockCoil(np.zeros((32, 3)), np.zeros((32, 3)), 1e5)])
        surf = _MockSurface(sdofs, 1, 1, 1, False, qphi, qtheta)

        with pytest.raises(
            ValueError, match="Unsupported label type.*AspectRatioLabel"
        ):
            BoozerSurfaceJAX(bs, surf, AspectRatioLabel(), 1.0, constraint_weight=1.0)


# ---------------------------------------------------------------------------
# Issue-2 validation: nfp>1 volume and area correctness
# ---------------------------------------------------------------------------


class TestNfpVolumeArea:
    """Verify volume/area are correct for nfp>1 (one-period quadrature)."""

    @pytest.mark.parametrize("nfp", [1, 2, 3, 5])
    def test_volume_nfp(self, nfp):
        """Volume = 2π²Rr² regardless of nfp."""
        geometry = _simple_torus_geometry_values(
            R0=1.0,
            r=0.1,
            mpol=1,
            ntor=1,
            nfp=nfp,
            nphi=32,
            ntheta=32,
        )
        np.testing.assert_allclose(
            geometry["volume"],
            geometry["expected_volume"],
            rtol=_TORUS_GEOMETRY_RTOL,
        )

    @pytest.mark.parametrize("nfp", [1, 2, 3, 5])
    def test_area_nfp(self, nfp):
        """Area = 4π²Rr regardless of nfp."""
        geometry = _simple_torus_geometry_values(
            R0=1.0,
            r=0.1,
            mpol=1,
            ntor=1,
            nfp=nfp,
            nphi=32,
            ntheta=32,
        )
        np.testing.assert_allclose(
            geometry["area"],
            geometry["expected_area"],
            rtol=_TORUS_GEOMETRY_RTOL,
        )


# ---------------------------------------------------------------------------
# Issue-1 validation: _ensure_solved crash guard
# ---------------------------------------------------------------------------


class TestEnsureSolvedGuard:
    """Verify runtime guard behavior around cached Boozer solve state."""

    def test_dirty_unsolved_surface_without_cached_result_is_rejected(
        self, monkeypatch
    ):
        """Dirty surfaces without cached iota/G must fail before attempting re-solve."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = True
        booz.res = None

        called = False

        def fake_run_code(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("run_code should not be called without cached res")

        monkeypatch.setattr(booz, "run_code", fake_run_code)

        with pytest.raises(RuntimeError, match="has not been solved yet"):
            _ensure_solved_jax(booz)

        assert called is False

    def test_dirty_resolve_preserves_nondefault_backend_contract(self, monkeypatch):
        """Dirty on-device surfaces must re-solve from cached iota/G before use."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.res = {
            "iota": 0.3,
            "G": 0.05,
            "success": True,
            "PLU": (np.eye(1), np.eye(1), np.eye(1)),
            "vjp": lambda *_args, **_kwargs: None,
            "vjp_groups": lambda *_args, **_kwargs: None,
        }
        booz.need_to_run_code = True

        captured = {}

        def fake_run_code(iota, G=None):
            captured["iota"] = iota
            captured["G"] = G
            booz.res = {
                "iota": iota,
                "G": G,
                "success": True,
                "primal_success": True,
                "adjoint_linear_solve_available": True,
                "sdofs": _runtime_sdofs_for(booz),
                "weight_inv_modB": True,
                "linearization_kind": "hessian",
                "dense_linear_solve_factors_available": True,
                "PLU": (np.eye(1), np.eye(1), np.eye(1)),
                "vjp": lambda *_args, **_kwargs: None,
                "vjp_groups": lambda *_args, **_kwargs: None,
            }
            booz.need_to_run_code = False
            return booz.res

        monkeypatch.setattr(booz, "run_code", fake_run_code)

        _ensure_solved_jax(booz)

        assert captured == {"iota": 0.3, "G": 0.05}
        assert booz.need_to_run_code is False

    def test_finite_unsuccessful_state_with_adjoint_contract_is_rejected(self):
        """_ensure_solved must reject unsuccessful solves even with adjoint metadata."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.res = {
            "iota": 0.3,
            "G": 0.05,
            "success": False,
            "primal_success": False,
            "adjoint_linear_solve_available": False,
            "weight_inv_modB": True,
            "linearization_kind": "hessian",
            "dense_linear_solve_factors_available": True,
            "PLU": (np.eye(1), np.eye(1), np.eye(1)),
            "vjp": lambda *_args, **_kwargs: None,
            "vjp_groups": lambda *_args, **_kwargs: None,
        }

        with pytest.raises(RuntimeError, match="failed"):
            _ensure_solved_jax(booz)

    def test_ensure_solved_logs_success_and_norms(self, caplog):
        """_ensure_solved must log cached solve quality alongside success."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.res = {
            "iota": 0.3,
            "G": 0.05,
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "weight_inv_modB": True,
            "linearization_kind": "hessian",
            "dense_linear_solve_factors_available": True,
            "type": "ls",
            "gradient": np.asarray([3.0, -4.0]),
            "residual": np.asarray([1.5, -2.5]),
            "PLU": (np.eye(1), np.eye(1), np.eye(1)),
            "vjp": lambda *_args, **_kwargs: None,
            "vjp_groups": lambda *_args, **_kwargs: None,
        }

        with caplog.at_level(
            logging.DEBUG,
            logger="simsopt.geo.surfaceobjectives_jax",
        ):
            _ensure_solved_jax(booz)

        assert "success=True" in caplog.text
        assert "grad_inf=4.0" in caplog.text
        assert "residual_inf=2.5" in caplog.text

    def test_ensure_solved_exact_logs_residual_without_jacobian_as_grad(self, caplog):
        """Exact cached solves should not label Jacobian size as a gradient norm."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.res = {
            "iota": 0.3,
            "G": 0.05,
            "success": True,
            "primal_success": True,
            "adjoint_linear_solve_available": True,
            "sdofs": _runtime_sdofs_for(booz),
            "weight_inv_modB": True,
            "linearization_kind": "exact_jacobian",
            "dense_linear_solve_factors_available": True,
            "type": "exact",
            "jacobian": np.asarray([[3.0, -4.0], [1.0, 2.0]]),
            "residual": np.asarray([0.1, -0.2]),
            "PLU": (np.eye(1), np.eye(1), np.eye(1)),
            "vjp": lambda *_args, **_kwargs: None,
            "vjp_groups": lambda *_args, **_kwargs: None,
        }

        with caplog.at_level(
            logging.DEBUG,
            logger="simsopt.geo.surfaceobjectives_jax",
        ):
            _ensure_solved_jax(booz)

        assert "success=True" in caplog.text
        assert "grad_inf=None" in caplog.text
        assert "residual_inf=0.2" in caplog.text


class TestMixedQuadratureBoozer:
    """BoozerSurfaceJAX works when coils have different nquad counts."""

    def test_instantiation(self):
        """Mixed-quad coils don't crash _refresh_coil_data."""
        booz = _make_mock_boozer_surface_mixed_quad()
        assert len(booz.coil_groups) == 2  # two distinct nquad values
        assert isinstance(booz.coil_set_spec, GroupedCoilSetSpec)
        assert len(booz.coil_set_spec.groups) == 2

    def test_run_code_ls_converges(self):
        """LS solve converges with mixed-quadrature coils."""
        booz = _make_mock_boozer_surface_mixed_quad()
        res = booz.run_code(iota=0.3, G=0.05)
        assert res is not None
        assert res["type"] == "ls"
        assert res["success"]

    def test_penalty_matches_uniform(self):
        """Penalty value is close to uniform-quad reference.

        The mixed-quad setup uses 64+128 points while the uniform setup
        uses 64+64.  The B field differs slightly due to quadrature
        accuracy, but the solved penalty should stay close. The mock
        problem is not unique in ``iota``/``G``, so compare the scalar
        objective directly instead of the recovered parameters.
        """
        booz_mixed = _make_mock_boozer_surface_mixed_quad()
        booz_uniform = _make_mock_boozer_surface()

        res_mixed = booz_mixed.run_code(iota=0.3, G=0.05)
        res_uniform = booz_uniform.run_code(iota=0.3, G=0.05)
        penalty_fun_rel_tol = 2e-3
        penalty_fun_abs_tol = 2e-6

        assert res_mixed["success"]
        assert res_uniform["final_gradient_inf_norm"] < 2e-9

        # Observed mixed-vs-uniform gap on this mock torus is ~1.4e-3 relative.
        np.testing.assert_allclose(
            res_mixed["fun"],
            res_uniform["fun"],
            rtol=penalty_fun_rel_tol,
            atol=penalty_fun_abs_tol,
        )


# ---------------------------------------------------------------------------
# P3: Boozer exact-constraints Jacobian Taylor test
# ---------------------------------------------------------------------------


class TestBoozerExactConstraintsJacobianTaylor:
    """Taylor convergence test for the exact Boozer constraints Jacobian.

    Verifies that the Jacobian of ``_boozer_exact_residual`` (computed
    via ``jax.jacfwd``) is consistent with finite-difference
    directional derivatives.  For each epsilon the FD error must
    shrink by at least a factor of 0.55, confirming first-order
    convergence.

    Mirrors ``TestBoozerSurface.subtest_boozer_constrained_jacobian``
    from the upstream CPU test suite.
    """

    @staticmethod
    def _run_taylor_series(res_fn, x, epsilons, ratio_bound=0.55):
        """Run the multi-epsilon FD convergence loop.

        Returns the final error so callers can assert it shrank
        monotonically.
        """
        np.random.seed(1)
        r0 = res_fn(x)
        J = jax.jacfwd(res_fn)(x)
        h = jnp.array(np.random.uniform(size=x.shape) - 0.5)
        dr_exact = J @ h

        err_old = 1e9
        for eps in epsilons:
            r1 = res_fn(x + eps * h)
            dr_fd = (r1 - r0) / eps
            err = float(jnp.linalg.norm(dr_fd - dr_exact))
            assert err < err_old * ratio_bound, (
                f"FD error did not shrink: err={err:.3e}, "
                f"prev={err_old:.3e}, ratio={err / err_old:.3f}"
            )
            err_old = err
        return err_old

    def test_exact_jacobian_taylor_nonstellsym(self):
        """Taylor test for non-stellsym exact residual Jacobian."""
        booz = _make_mock_boozer_surface_exact()
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = booz._make_exact_residual(mask_indices)

        iota, G = 0.3, 0.05
        x = booz._pack_decision_vector(iota, G)

        epsilons = jnp.pow(2.0, -jnp.arange(7, 20, dtype=jnp.float64))
        self._run_taylor_series(res_fn, x, epsilons)

    def test_exact_jacobian_taylor_stellsym(self):
        """Taylor test for stellsym exact residual Jacobian."""
        booz = _make_mock_boozer_surface_exact(stellsym=True)
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = booz._make_exact_residual(mask_indices)

        iota, G = 0.3, 0.05
        x = booz._pack_decision_vector(iota, G)

        epsilons = jnp.pow(2.0, -jnp.arange(7, 20, dtype=jnp.float64))
        self._run_taylor_series(res_fn, x, epsilons)


# ---------------------------------------------------------------------------
# P18: Parametrized stellsym x optimize_G sweep for penalty gradient/BFGS
# ---------------------------------------------------------------------------

_STELLSYM_LIST = [True, False]
_OPTIMIZE_G_LIST = [True, False]


def _assert_gradient_taylor_convergence(obj, x, *, label="", ratio_bound=0.55):
    """Multi-epsilon forward-FD Taylor test for ``jax.grad(obj)``."""
    np.random.seed(1)
    f0 = float(obj(x))
    grad = jax.grad(obj)(x)
    h = jnp.array(np.random.uniform(size=x.shape) - 0.5)
    Jex = float(jnp.dot(grad, h))

    err_old = 1e9
    epsilons = jnp.pow(2.0, -jnp.arange(7, 20, dtype=jnp.float64))
    for eps in epsilons:
        f1 = float(obj(x + eps * h))
        Jfd = (f1 - f0) / float(eps)
        err = abs(Jfd - Jex) / max(abs(Jex), 1e-30)
        assert err < err_old * ratio_bound, (
            f"{label}FD ratio {err / err_old:.3f} >= {ratio_bound} "
            f"at eps={float(eps):.2e}"
        )
        err_old = err


def _assert_upstream_penalty_parity(parity):
    """Assert the documented LS-wrapper same-state CPU/JAX parity contract."""

    tolerances = parity_ladder_tolerances("ls-wrapper-gradient")
    np.testing.assert_allclose(
        parity["jax_value"],
        parity["cpu_value"],
        rtol=tolerances["rtol"],
        atol=tolerances["atol"],
        err_msg="CPU/JAX penalty value mismatch",
    )
    np.testing.assert_allclose(
        parity["jax_gradient"],
        parity["cpu_gradient"],
        rtol=tolerances["rtol"],
        atol=tolerances["atol"],
        err_msg="CPU/JAX penalty gradient mismatch",
    )


def _seeded_hessian_direction_pair(size: int):
    rng = np.random.default_rng(17)
    return rng.normal(size=size), rng.normal(size=size)


class TestParametrizedPenaltyGradientTaylor:
    """Multi-epsilon Taylor test for penalty gradient across the sweep."""

    @pytest.mark.parametrize("stellsym", _STELLSYM_LIST)
    @pytest.mark.parametrize("optimize_G", _OPTIMIZE_G_LIST)
    def test_gradient_taylor(self, stellsym, optimize_G):
        case = _build_penalty_problem(stellsym=stellsym, optimize_G=optimize_G)
        _assert_gradient_taylor_convergence(
            case["objective"],
            case["x"],
            label=f"stellsym={stellsym}, optimize_G={optimize_G}: ",
        )


class TestParametrizedBFGSConvergence:
    """BFGS convergence across the stellsym x optimize_G sweep."""

    @pytest.mark.parametrize("stellsym", _STELLSYM_LIST)
    @pytest.mark.parametrize("optimize_G", _OPTIMIZE_G_LIST)
    def test_bfgs_reduces_objective(self, stellsym, optimize_G):
        case = _build_penalty_problem(stellsym=stellsym, optimize_G=optimize_G)
        val_init = float(case["objective"](case["x"]))
        result = jax_minimize(
            case["objective"],
            case["x"],
            method="bfgs",
            tol=1e-10,
            maxiter=200,
        )
        val_final = float(result.fun)
        assert val_final < val_init, (
            f"stellsym={stellsym}, optimize_G={optimize_G}: "
            f"BFGS did not reduce objective {val_init:.6e} → {val_final:.6e}"
        )


# ---------------------------------------------------------------------------
# P19 + P21: Upstream surface factory parity (coil_arrays dispatch path)
# ---------------------------------------------------------------------------


class TestUpstreamSurfaceFactoryParity:
    """Validate the coil_arrays dispatch path of ``_boozer_penalty_objective``.

    Unlike ``TestUpstreamFactoryBoozerMatrix`` (which tests via the adapter's
    ``coil_set_spec`` path), this class passes raw ``coil_arrays`` tuples
    directly, exercising the ``grouped_*_from_inputs`` dispatch branch.
    """

    @staticmethod
    def _make_problem(stellsym, optimize_G):
        from simsopt.configs import get_data

        from .boozersurface_jax_test_helpers import (
            _UPSTREAM_BOOZER_CONSTRAINT_WEIGHT,
            _UPSTREAM_BOOZER_TF_TARGET,
            _boozer_penalty_objective,
        )
        from .surface_test_helpers import get_surface

        _, _, ma, nfp, bs = get_data("ncsx")
        s = get_surface("SurfaceXYZTensorFourier", stellsym, nfp=nfp)
        s.fit_to_curve(ma, 0.1)

        inputs = _extract_upstream_jax_penalty_inputs(
            s,
            bs,
            optimize_G=optimize_G,
        )

        def objective(xx):
            return _boozer_penalty_objective(
                xx,
                coil_arrays=inputs["coil_arrays"],
                quadpoints_phi=inputs["qphi"],
                quadpoints_theta=inputs["qtheta"],
                mpol=inputs["mpol"],
                ntor=inputs["ntor"],
                nfp=inputs["nfp"],
                stellsym=inputs["stellsym"],
                scatter_indices=inputs["scatter_indices"],
                surface_kind="generic",
                label_quadpoints_phi=inputs["qphi"],
                label_quadpoints_theta=inputs["qtheta"],
                label_mpol=inputs["mpol"],
                label_ntor=inputs["ntor"],
                label_nfp=inputs["nfp"],
                label_stellsym=inputs["stellsym"],
                label_scatter_indices=inputs["scatter_indices"],
                label_surface_kind="generic",
                targetlabel=_UPSTREAM_BOOZER_TF_TARGET,
                constraint_weight=_UPSTREAM_BOOZER_CONSTRAINT_WEIGHT,
                label_type="volume",
                phi_idx=0,
                optimize_G=optimize_G,
                weight_inv_modB=True,
            )

        return objective, inputs["x"]

    @pytest.mark.parametrize("stellsym", _STELLSYM_LIST)
    @pytest.mark.parametrize("optimize_G", _OPTIMIZE_G_LIST)
    def test_penalty_value_is_finite(self, stellsym, optimize_G):
        """Penalty objective returns a finite non-negative scalar."""
        obj, x = self._make_problem(stellsym, optimize_G)
        val = float(obj(x))
        assert np.isfinite(val), f"Non-finite penalty: {val}"
        assert val >= 0.0, f"Negative penalty: {val}"

    @pytest.mark.parametrize("stellsym", _STELLSYM_LIST)
    @pytest.mark.parametrize("optimize_G", _OPTIMIZE_G_LIST)
    def test_gradient_taylor_convergence(self, stellsym, optimize_G):
        """Multi-epsilon Taylor test via the coil_arrays dispatch path."""
        obj, x = self._make_problem(stellsym, optimize_G)
        _assert_gradient_taylor_convergence(
            obj,
            x,
            label=f"stellsym={stellsym}, optimize_G={optimize_G}: ",
        )


# ---------------------------------------------------------------------------
# P18 / P19 / P21: upstream factory matrix coverage (coil_set_spec path)
# ---------------------------------------------------------------------------


class TestUpstreamFactoryBoozerMatrix:
    """Factory-driven JAX sweep over the upstream Boozer surface matrix."""

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_penalty_case_uses_copied_matching_cpu_jax_fixtures(
        self,
        surfacetype,
        stellsym,
        optimize_G,
    ):
        """CPU/JAX parity fixtures must match by value, not mutable identity."""
        case = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            optimize_G,
        )

        cpu_surface = case.cpu_boozer.surface
        jax_surface = case.jax_boozer.surface
        cpu_label_surface = case.cpu_boozer.label.surface
        jax_label_surface = case.jax_boozer.label.surface
        assert cpu_surface is not jax_surface
        assert cpu_label_surface is not jax_label_surface
        np.testing.assert_allclose(cpu_surface.get_dofs(), jax_surface.get_dofs())
        np.testing.assert_allclose(
            cpu_surface.quadpoints_phi,
            jax_surface.quadpoints_phi,
        )
        np.testing.assert_allclose(
            cpu_surface.quadpoints_theta,
            jax_surface.quadpoints_theta,
        )
        assert cpu_surface.mpol == jax_surface.mpol
        assert cpu_surface.ntor == jax_surface.ntor
        assert cpu_surface.nfp == jax_surface.nfp
        assert cpu_surface.stellsym == jax_surface.stellsym
        np.testing.assert_allclose(
            cpu_label_surface.get_dofs(),
            jax_label_surface.get_dofs(),
        )
        np.testing.assert_allclose(
            cpu_label_surface.quadpoints_phi,
            jax_label_surface.quadpoints_phi,
        )
        np.testing.assert_allclose(
            cpu_label_surface.quadpoints_theta,
            jax_label_surface.quadpoints_theta,
        )
        assert case.cpu_boozer.targetlabel == case.jax_boozer.targetlabel
        assert case.cpu_boozer.label.nphi == case.jax_boozer.label.nphi
        assert case.cpu_boozer.label.ntheta == case.jax_boozer.label.ntheta
        assert case.jax_boozer.constraint_weight == case.constraint_weight
        assert (
            case.cpu_boozer.options["weight_inv_modB"]
            == case.jax_boozer.options["weight_inv_modB"]
        )

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_penalty_gradient_taylor_matrix(
        self,
        surfacetype,
        stellsym,
        optimize_G,
    ):
        """Mirror upstream's surfacetype x stellsym x optimize_G sweep."""
        case = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            optimize_G,
        )
        objective = case.jax_boozer._make_penalty_objective_with(
            case.optimize_G,
            case.jax_boozer.options["weight_inv_modB"],
            case.constraint_weight,
        )
        x = jnp.asarray(case.x, dtype=jnp.float64)
        _assert_gradient_taylor_convergence(objective, x)

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_penalty_value_and_gradient_cpu_parity_matrix(
        self,
        surfacetype,
        stellsym,
        optimize_G,
    ):
        """Factory cases must preserve the direct CPU/JAX LS-wrapper parity contract."""
        case = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            optimize_G,
        )
        parity = _evaluate_upstream_boozer_penalty_case(case)
        _assert_upstream_penalty_parity(parity)

    def test_penalty_raw_inner_callback_cpu_parity_fixed_state(self):
        """Raw CPU/JAX LS objective callback matches before SciPy BFGS sees it."""
        case = _build_upstream_boozer_penalty_case(
            UPSTREAM_BOOZER_SURFACE_TYPES[0],
            UPSTREAM_BOOZER_STELLSYM[0],
            UPSTREAM_BOOZER_OPTIMIZE_G[0],
        )
        cpu_x = np.asarray(case.x, dtype=np.float64)
        jax_x = jnp.asarray(case.x, dtype=jnp.float64)
        jax_host_x = np.asarray(jax.device_get(jax_x), dtype=np.float64)

        assert cpu_x.dtype == np.dtype(np.float64)
        assert jax_host_x.dtype == np.dtype(np.float64)
        assert jax_host_x.tobytes() == cpu_x.tobytes()

        cpu_value, cpu_gradient = case.cpu_boozer.boozer_penalty_constraints_vectorized(
            cpu_x,
            derivatives=1,
            constraint_weight=case.constraint_weight,
            optimize_G=case.optimize_G,
            weight_inv_modB=case.cpu_boozer.options["weight_inv_modB"],
        )
        jax_objective = case.jax_boozer._make_penalty_objective_with(
            case.optimize_G,
            case.jax_boozer.options["weight_inv_modB"],
            case.constraint_weight,
            boozer_reduction_mode="cpu_ordered",
        )
        jax_value, jax_gradient = jax.value_and_grad(jax_objective)(jax_x)

        value_tolerances = parity_ladder_tolerances("direct_kernel")
        gradient_tolerances = parity_ladder_tolerances("ls_wrapper_gradient")
        np.testing.assert_allclose(
            float(jax_value),
            float(cpu_value),
            rtol=value_tolerances["rtol"],
            atol=value_tolerances["atol"],
            err_msg="Raw CPU/JAX penalty value mismatch before SciPy BFGS",
        )
        np.testing.assert_allclose(
            np.asarray(jax_gradient, dtype=np.float64),
            np.asarray(cpu_gradient, dtype=np.float64),
            rtol=gradient_tolerances["rtol"],
            atol=gradient_tolerances["atol"],
            err_msg="Raw CPU/JAX penalty gradient mismatch before SciPy BFGS",
        )

    def test_penalty_cpu_ordered_value_and_grad_cpu_parity_fixed_state(self):
        """Host-SciPy BFGS value/gradient closure matches CPU LS callback."""
        case = _build_upstream_boozer_penalty_case(
            UPSTREAM_BOOZER_SURFACE_TYPES[0],
            UPSTREAM_BOOZER_STELLSYM[0],
            UPSTREAM_BOOZER_OPTIMIZE_G[0],
        )
        cpu_x = np.asarray(case.x, dtype=np.float64)
        jax_x = jnp.asarray(case.x, dtype=jnp.float64)

        cpu_value, cpu_gradient = case.cpu_boozer.boozer_penalty_constraints_vectorized(
            cpu_x,
            derivatives=1,
            constraint_weight=case.constraint_weight,
            optimize_G=case.optimize_G,
            weight_inv_modB=case.cpu_boozer.options["weight_inv_modB"],
        )
        value_and_grad = case.jax_boozer._make_penalty_value_and_grad_cpu_ordered_with(
            case.optimize_G,
            case.jax_boozer.options["weight_inv_modB"],
            case.constraint_weight,
        )
        jax_value, jax_gradient = value_and_grad(jax_x)
        jax_gradient = np.asarray(jax.device_get(jax_gradient), dtype=np.float64)

        assert jax_gradient.dtype == np.dtype(np.float64)
        assert jax_gradient.shape == cpu_gradient.shape
        value_tolerances = parity_ladder_tolerances("direct_kernel")
        gradient_tolerances = parity_ladder_tolerances("ls_wrapper_gradient")
        np.testing.assert_allclose(
            float(jax_value),
            float(cpu_value),
            rtol=value_tolerances["rtol"],
            atol=value_tolerances["atol"],
            err_msg="CPU-ordered value closure mismatch before SciPy BFGS",
        )
        np.testing.assert_allclose(
            jax_gradient,
            np.asarray(cpu_gradient, dtype=np.float64),
            rtol=gradient_tolerances["rtol"],
            atol=gradient_tolerances["atol"],
            err_msg="CPU-ordered gradient closure mismatch before SciPy BFGS",
        )

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_penalty_hessian_directional_cpu_parity_matrix(
        self,
        surfacetype,
        stellsym,
        optimize_G,
    ):
        """Hessian coverage uses same-state CPU Hessian vs JAX HVP directions."""
        case = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            optimize_G,
        )
        _, _, cpu_hessian = case.cpu_boozer.boozer_penalty_constraints_vectorized(
            case.x,
            derivatives=2,
            constraint_weight=case.constraint_weight,
            optimize_G=case.optimize_G,
        )
        h1, h2 = _seeded_hessian_direction_pair(case.x.shape[0])
        cpu_directional = float(h1 @ cpu_hessian @ h2)

        objective = case.jax_boozer._make_penalty_objective_with(
            case.optimize_G,
            case.jax_boozer.options["weight_inv_modB"],
            case.constraint_weight,
        )
        _, jax_hvp = jax.jvp(
            jax.grad(objective),
            (jnp.asarray(case.x, dtype=jnp.float64),),
            (jnp.asarray(h2, dtype=jnp.float64),),
        )
        jax_directional = float(jnp.dot(jnp.asarray(h1, dtype=jnp.float64), jax_hvp))
        tolerances = parity_ladder_tolerances("fd-gradient")

        np.testing.assert_allclose(
            jax_directional,
            cpu_directional,
            rtol=tolerances["directional_fd_rtol"],
            atol=tolerances["directional_fd_atol"],
            err_msg="CPU/JAX penalty Hessian directional mismatch",
        )

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_penalty_hessian_column_complete_cpu_parity_matrix(
        self,
        surfacetype,
        stellsym,
        optimize_G,
    ):
        """Column-complete CPU/JAX Hessian parity runs one HVP per decision variable.

        The basis sweep covers the upstream matrix and is expected to live on the
        auto-marked slow Boozer lane.
        """
        case = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            optimize_G,
        )
        parity = _evaluate_upstream_boozer_penalty_hessian_case(case)
        cpu_hessian = parity["cpu_hessian"]
        jax_hessian = parity["jax_hessian"]
        tolerances = parity_ladder_tolerances("direct-hessian-oracle")
        second_derivative_rtol = tolerances["second_derivative_rtol"]
        second_derivative_atol = tolerances["second_derivative_atol"]

        assert cpu_hessian.shape == jax_hessian.shape
        assert np.all(np.isfinite(cpu_hessian))
        assert np.all(np.isfinite(jax_hessian))
        np.testing.assert_allclose(
            cpu_hessian,
            cpu_hessian.T,
            rtol=1e-12,
            atol=1e-12,
            err_msg="CPU Hessian asymmetric; check CPU/C++ oracle regression",
        )
        np.testing.assert_allclose(
            jax_hessian,
            jax_hessian.T,
            rtol=second_derivative_rtol,
            atol=second_derivative_atol,
            err_msg="JAX Hessian asymmetric; check HVP column assembly",
        )
        np.testing.assert_allclose(
            jax_hessian,
            cpu_hessian,
            rtol=second_derivative_rtol,
            atol=second_derivative_atol,
            err_msg="CPU/JAX penalty Hessian column-complete mismatch",
        )

    def test_penalty_dense_newton_step_cpu_parity_fixed_state(self):
        """Dense Newton step parity uses the same fixed-state LS Hessian oracle."""
        case = _build_upstream_boozer_penalty_case(
            UPSTREAM_BOOZER_SURFACE_TYPES[0],
            UPSTREAM_BOOZER_STELLSYM[0],
            UPSTREAM_BOOZER_OPTIMIZE_G[0],
        )
        gradient_parity = _evaluate_upstream_boozer_penalty_case(case)
        hessian_parity = _evaluate_upstream_boozer_penalty_hessian_case(case)
        cpu_hessian = hessian_parity["cpu_hessian"]
        jax_hessian = hessian_parity["jax_hessian"]
        cpu_gradient = gradient_parity["cpu_gradient"]
        jax_gradient = gradient_parity["jax_gradient"]

        cpu_step = np.linalg.solve(cpu_hessian, cpu_gradient)
        jax_step = _opt._solve_dense_newton_step(
            jnp.asarray(jax_hessian, dtype=jnp.float64),
            jnp.asarray(jax_gradient, dtype=jnp.float64),
            refine=False,
        )
        cpu_refined_step = cpu_step + np.linalg.solve(
            cpu_hessian,
            cpu_gradient - cpu_hessian @ cpu_step,
        )
        jax_refined_step = _opt._solve_dense_newton_step(
            jnp.asarray(jax_hessian, dtype=jnp.float64),
            jnp.asarray(jax_gradient, dtype=jnp.float64),
            refine=True,
        )
        tolerances = parity_ladder_tolerances("direct-hessian-oracle")

        np.testing.assert_allclose(
            np.asarray(jax_step),
            cpu_step,
            rtol=tolerances["second_derivative_rtol"],
            atol=tolerances["second_derivative_atol"],
            err_msg="CPU/JAX dense Newton step mismatch",
        )
        np.testing.assert_allclose(
            np.asarray(jax_refined_step),
            cpu_refined_step,
            rtol=tolerances["second_derivative_rtol"],
            atol=tolerances["second_derivative_atol"],
            err_msg="CPU/JAX refined dense Newton step mismatch",
        )

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_exact_constraints_residual_and_jvp_cpu_parity_matrix(
        self,
        surfacetype,
        stellsym,
        optimize_G,
    ):
        """Exact KKT coverage uses real CPU residual/Jacobian vs JAX residual/JVP."""
        case = _build_upstream_boozer_exact_constraints_case(
            surfacetype,
            stellsym,
            optimize_G,
        )
        parity = _evaluate_upstream_boozer_exact_constraints_case(case)
        tolerances = parity_ladder_tolerances("derivative-heavy")

        np.testing.assert_allclose(
            parity["jax_residual"],
            parity["cpu_residual"],
            rtol=tolerances["first_derivative_rtol"],
            atol=tolerances["first_derivative_atol"],
            err_msg="CPU/JAX exact-constraints residual mismatch",
        )
        np.testing.assert_allclose(
            parity["jax_jvp"],
            parity["cpu_jvp"],
            rtol=tolerances["first_derivative_rtol"],
            atol=tolerances["first_derivative_atol"],
            err_msg="CPU/JAX exact-constraints Jacobian-vector mismatch",
        )
        np.testing.assert_allclose(
            parity["jax_jacobian"],
            parity["cpu_jacobian"],
            rtol=tolerances["first_derivative_rtol"],
            atol=tolerances["first_derivative_atol"],
            err_msg="CPU/JAX exact-constraints dense Jacobian mismatch",
        )

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    def test_penalty_case_packs_G_only_when_requested(self, surfacetype, stellsym):
        """The upstream factory should append only the optional G decision variable."""
        case_without_G = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            False,
        )
        case_with_G = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            True,
        )

        assert case_with_G.x.shape == (case_without_G.x.size + 1,)
        np.testing.assert_allclose(case_with_G.x[:-1], case_without_G.x)
        assert np.isfinite(case_with_G.x[-1])

    @pytest.mark.parametrize(
        ("surfacetype", "expected_kind"),
        [
            ("SurfaceRZFourier", "rzfourier"),
            ("SurfaceXYZFourier", "xyzfourier"),
            ("SurfaceXYZTensorFourier", "xyztensorfourier"),
        ],
    )
    def test_runtime_state_accepts_supported_surface_family_matrix(
        self,
        surfacetype,
        expected_kind,
    ):
        from .surface_test_helpers import get_surface

        surface = get_surface(surfacetype, stellsym=True)
        runtime_state = _bsj.build_boozer_surface_runtime_state(surface)

        assert runtime_state.surface_kind == expected_kind
        assert runtime_state.scatter_indices is not None
        assert runtime_state.mpol == surface.mpol
        assert runtime_state.ntor == surface.ntor

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_penalty_case_immutable_inputs_match_pair_state(
        self,
        surfacetype,
        stellsym,
        optimize_G,
    ):
        """The shared fixture exposes one immutable target-lane input snapshot."""
        case = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            optimize_G,
        )
        inputs = _build_upstream_boozer_immutable_inputs(case)
        sdofs, iota, G = case.jax_boozer._unpack_decision_vector(
            jnp.asarray(case.x),
            optimize_G=optimize_G,
        )

        assert inputs.coil_set_spec is case.jax_boozer.coil_set_spec
        np.testing.assert_allclose(inputs.sdofs, np.asarray(sdofs))
        assert inputs.iota == pytest.approx(float(iota))
        assert inputs.G == (None if G is None else pytest.approx(float(G)))
        assert inputs.options_snapshot == (
            ("constraint_weight", case.constraint_weight),
            ("optimize_G", optimize_G),
            ("weight_inv_modB", bool(case.jax_boozer.options["weight_inv_modB"])),
        )

    def test_exact_surface_factory_rejects_surface_xyzfourier(self):
        """The exact path is hard-gated to SurfaceXYZTensorFourier."""
        case = _build_upstream_exact_surface_case("SurfaceXYZFourier")

        with pytest.raises(
            RuntimeError,
            match="Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier",
        ):
            case.jax_boozer.solve_residual_equation_exactly_newton(
                iota=case.initial_iota,
                G=case.initial_G,
                maxiter=1,
            )

    def test_run_code_exact_accepts_explicit_xyztensor_surface_contract(self):
        """The exact path is keyed by the explicit tensor surface contract."""
        booz = _make_mock_boozer_surface_exact()

        with _patched_exact_newton_result(success=True):
            res = booz.run_code(iota=0.3, G=0.05)

        assert res["success"] is True
        assert res["type"] == "exact"

    def test_exact_surface_factory_tensor_residual_is_finite(self):
        """The exact factory data remains valid for SurfaceXYZTensorFourier."""
        case = _build_upstream_exact_surface_case("SurfaceXYZTensorFourier")
        mask_indices = case.jax_boozer._compute_stellsym_mask_indices()
        residual_fn = case.jax_boozer._make_exact_residual(mask_indices)
        x = case.jax_boozer._pack_decision_vector(case.initial_iota, case.initial_G)
        residual = residual_fn(x)

        assert residual.ndim == 1
        assert residual.size > 0
        assert np.all(np.isfinite(np.asarray(residual)))

    def test_exact_surface_scalar_residual_matches_legacy_cpu_state(self):
        """Mirror legacy test_residual with the exact NCSX surface state."""
        case = _build_upstream_exact_surface_case("SurfaceXYZTensorFourier")
        x = np.concatenate((case.jax_boozer.surface.get_dofs(), [case.initial_iota]))
        cpu_value = case.cpu_boozer.boozer_penalty_constraints_vectorized(
            x,
            derivatives=0,
            constraint_weight=1.0,
            optimize_G=False,
        )
        jax_objective = case.jax_boozer._make_penalty_objective_with(
            False,
            True,
            1.0,
        )
        jax_value = float(jax_objective(jnp.asarray(x)))

        assert cpu_value < 1e-6
        np.testing.assert_allclose(jax_value, cpu_value, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# CPU-vs-JAX parity regression tests (require simsoptpp)
# ---------------------------------------------------------------------------
try:
    from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

    _HAS_SURFACE_XYZ_TENSOR = True
except (ImportError, ModuleNotFoundError):
    _HAS_SURFACE_XYZ_TENSOR = False

_skip_no_simsoptpp = pytest.mark.skipif(
    not _HAS_SURFACE_XYZ_TENSOR,
    reason="SurfaceXYZTensorFourier requires simsoptpp",
)


@_skip_no_simsoptpp
class TestStellsymMaskCPUJAXParity:
    """Verify that the extracted JAX stellsym mask matches the CPU surface mask."""

    _GRID_CONFIGS = [
        # (description, phi_builder, theta_builder)
        (
            "full_phi_x_full_theta",
            lambda ntor, nfp: np.linspace(0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
            lambda mpol: np.linspace(0, 1.0, 2 * mpol + 1, endpoint=False),
        ),
        (
            "full_phi_x_half_theta",
            lambda ntor, nfp: np.linspace(0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
            lambda mpol: np.linspace(0, 0.5, mpol + 1, endpoint=False),
        ),
        (
            "half_phi_x_full_theta",
            lambda ntor, nfp: np.linspace(
                0, 1.0 / (2.0 * nfp), ntor + 1, endpoint=False
            ),
            lambda mpol: np.linspace(0, 1.0, 2 * mpol + 1, endpoint=False),
        ),
    ]

    @pytest.mark.parametrize(
        "mpol,ntor,nfp",
        [(2, 2, 2), (4, 3, 3), (6, 6, 3)],
        ids=["2x2_nfp2", "4x3_nfp3", "6x6_nfp3"],
    )
    @pytest.mark.parametrize(
        "grid_label,phi_fn,theta_fn",
        _GRID_CONFIGS,
        ids=[cfg[0] for cfg in _GRID_CONFIGS],
    )
    def test_mask_matches_cpu_surface(
        self, mpol, ntor, nfp, grid_label, phi_fn, theta_fn
    ):
        """surface_stellsym_mask_for_grid() matches SurfaceXYZTensorFourier.get_stellsym_mask()."""
        from simsopt.geo._surface_stellsym import surface_stellsym_mask_for_grid

        phis = phi_fn(ntor, nfp)
        thetas = theta_fn(mpol)

        s = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )

        cpu_mask = s.get_stellsym_mask()
        jax_mask = surface_stellsym_mask_for_grid(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=True,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )

        np.testing.assert_array_equal(
            jax_mask,
            cpu_mask,
            err_msg=f"Mask mismatch for {grid_label} mpol={mpol} ntor={ntor} nfp={nfp}",
        )

    @pytest.mark.parametrize(
        "mpol,ntor,nfp",
        [(2, 2, 2), (6, 6, 3)],
        ids=["2x2_nfp2", "6x6_nfp3"],
    )
    def test_mask_indices_match_cpu_surface(self, mpol, ntor, nfp):
        """compute_stellsym_mask_indices_for_grid() matches the old _compute_stellsym_mask_indices logic."""
        from simsopt.geo._surface_stellsym import (
            compute_stellsym_mask_indices_for_grid,
        )

        phis = np.linspace(0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
        thetas = np.linspace(0, 1.0, 2 * mpol + 1, endpoint=False)

        s = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )

        # Reproduce the old _compute_stellsym_mask_indices logic exactly
        m = s.get_stellsym_mask()
        mask_3d = np.repeat(m[..., None], 3, axis=2)
        mask_3d[0, 0, 0] = False
        expected_indices = np.flatnonzero(mask_3d)

        jax_indices = np.asarray(
            compute_stellsym_mask_indices_for_grid(
                mpol=mpol,
                ntor=ntor,
                nfp=nfp,
                stellsym=True,
                quadpoints_phi=phis,
                quadpoints_theta=thetas,
            ),
            dtype=np.int32,
        )

        np.testing.assert_array_equal(
            jax_indices,
            expected_indices.astype(np.int32),
            err_msg=f"Index mismatch for mpol={mpol} ntor={ntor} nfp={nfp}",
        )

    def test_mask_rejects_invalid_stellsym_grid_with_typed_error(self):
        from simsopt.geo._surface_stellsym import surface_stellsym_mask_for_grid

        with pytest.raises(ValueError, match="specific set of quadrature points"):
            surface_stellsym_mask_for_grid(
                mpol=2,
                ntor=2,
                nfp=2,
                stellsym=True,
                quadpoints_phi=np.asarray([0.0, 0.07, 0.21]),
                quadpoints_theta=np.asarray([0.0, 0.2, 0.4]),
            )


@_skip_no_simsoptpp
class TestBuildBoozerSurfaceRuntimeState:
    """End-to-end tests for build_boozer_surface_runtime_state → constructor."""

    def _make_real_surface(self, mpol=3, ntor=3, nfp=2, stellsym=True):
        phis = np.linspace(0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
        thetas = np.linspace(0, 1.0, 2 * mpol + 1, endpoint=False)
        return SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=stellsym,
            nfp=nfp,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )

    @staticmethod
    def _round_trip_pytree(value):
        leaves, treedef = jax.tree_util.tree_flatten(value)
        return jax.tree_util.tree_unflatten(treedef, leaves)

    @staticmethod
    def _assert_coil_set_specs_match(left, right):
        assert left.coil_index_lists() == right.coil_index_lists()
        for left_group, right_group in zip(left.groups, right.groups, strict=True):
            np.testing.assert_allclose(
                np.asarray(left_group.gammas),
                np.asarray(right_group.gammas),
            )
            np.testing.assert_allclose(
                np.asarray(left_group.gammadashs),
                np.asarray(right_group.gammadashs),
            )
            np.testing.assert_allclose(
                np.asarray(left_group.currents),
                np.asarray(right_group.currents),
            )

    def test_runtime_state_fields_match_surface(self):
        """build_boozer_surface_runtime_state captures correct surface metadata."""
        from simsopt.geo.boozersurface_jax import build_boozer_surface_runtime_state

        s = self._make_real_surface()
        rs = build_boozer_surface_runtime_state(s)

        assert rs.mpol == s.mpol
        assert rs.ntor == s.ntor
        assert rs.nfp == s.nfp
        assert rs.stellsym == s.stellsym
        np.testing.assert_allclose(np.asarray(rs.quadpoints_phi), s.quadpoints_phi)
        np.testing.assert_allclose(np.asarray(rs.quadpoints_theta), s.quadpoints_theta)
        assert rs.scatter_indices is not None  # stellsym=True

    def test_runtime_state_non_stellsym_has_no_scatter(self):
        """Non-stellsym surface produces scatter_indices=None."""
        from simsopt.geo.boozersurface_jax import build_boozer_surface_runtime_state

        s = self._make_real_surface(stellsym=False)
        rs = build_boozer_surface_runtime_state(s)

        assert rs.stellsym is False
        assert rs.scatter_indices is None

    def test_constructor_uses_prebuilt_runtime_state(self):
        """BoozerSurfaceJAX uses pre-built runtime state without re-querying surface."""
        from simsopt.geo.boozersurface_jax import build_boozer_surface_runtime_state

        s = self._make_real_surface(mpol=3, ntor=3, nfp=2)
        rs = build_boozer_surface_runtime_state(s)

        # Build minimal mock biotsavart and label for the constructor
        coils = _make_mock_coils()
        bs = _MockBiotSavart(coils)
        label = _PlumbingVolumeLabel(s)

        booz = BoozerSurfaceJAX(
            bs,
            s,
            label,
            targetlabel=0.1,
            constraint_weight=1.0,
            surface_runtime_state=rs,
        )

        assert booz.mpol == rs.mpol
        assert booz.ntor == rs.ntor
        assert booz.nfp == rs.nfp
        assert booz.stellsym == rs.stellsym
        assert booz.surface_runtime_state is rs
        np.testing.assert_allclose(
            np.asarray(booz.quadpoints_phi), np.asarray(rs.quadpoints_phi)
        )
        np.testing.assert_allclose(
            np.asarray(booz.quadpoints_theta), np.asarray(rs.quadpoints_theta)
        )

    def test_constructor_does_not_require_label_surface_identity(self):
        """Label geometry may come from a distinct structured surface object."""
        mpol, ntor, nfp = 1, 1, 1
        qphi = np.linspace(0.0, 1.0 / nfp, 8, endpoint=False)
        qtheta = np.linspace(0.0, 1.0, 8, endpoint=False)
        xc, yc, zc = _make_simple_torus_coeffs(mpol=mpol, ntor=ntor, nfp=nfp)
        sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
        surface = _MockSurface(sdofs, mpol, ntor, nfp, False, qphi, qtheta)
        label_surface = _MockSurface(sdofs.copy(), mpol, ntor, nfp, False, qphi, qtheta)

        booz = BoozerSurfaceJAX(
            _MockBiotSavart(_make_mock_coils()),
            surface,
            _PlumbingVolumeLabel(label_surface),
            targetlabel=2.0 * np.pi**2 * 1.0 * 0.1**2,
            constraint_weight=1.0,
        )

        assert booz.surface is surface
        assert booz.label.surface is label_surface
        assert booz.label.surface is not booz.surface
        assert booz.label_mpol == label_surface.mpol
        assert booz.label_ntor == label_surface.ntor
        assert booz._label_surface_geometry_kind == "generic"

    def test_runtime_state_round_trip_compares_values_not_identity(self):
        """Immutable runtime state replaces legacy object-identity serialization."""
        booz = _make_mock_boozer_surface(stellsym=True)
        res = booz.run_code(iota=0.3, G=0.05)
        assert res["success"] is True

        surface_state = booz.surface_runtime_state
        solved_state = booz.get_solved_runtime_state()
        coil_set_spec = booz.coil_set_spec

        surface_round_trip = self._round_trip_pytree(surface_state)
        solved_round_trip = self._round_trip_pytree(solved_state)
        coil_round_trip = self._round_trip_pytree(coil_set_spec)

        np.testing.assert_allclose(
            np.asarray(surface_round_trip.quadpoints_phi),
            np.asarray(surface_state.quadpoints_phi),
        )
        np.testing.assert_allclose(
            np.asarray(surface_round_trip.quadpoints_theta),
            np.asarray(surface_state.quadpoints_theta),
        )
        np.testing.assert_array_equal(
            np.asarray(surface_round_trip.scatter_indices),
            np.asarray(surface_state.scatter_indices),
        )
        assert surface_round_trip.mpol == surface_state.mpol
        assert surface_round_trip.ntor == surface_state.ntor
        assert surface_round_trip.nfp == surface_state.nfp
        assert surface_round_trip.stellsym == surface_state.stellsym
        assert surface_round_trip.surface_kind == surface_state.surface_kind

        np.testing.assert_allclose(
            np.asarray(solved_round_trip.sdofs),
            np.asarray(solved_state.sdofs),
        )
        np.testing.assert_allclose(
            np.asarray(solved_round_trip.iota),
            np.asarray(solved_state.iota),
        )
        np.testing.assert_allclose(
            np.asarray(solved_round_trip.G),
            np.asarray(solved_state.G),
        )
        assert solved_round_trip.weight_inv_modB == solved_state.weight_inv_modB

        assert booz.label_type == "volume"
        assert booz.targetlabel == pytest.approx(2.0 * np.pi**2 * 1.0 * 0.1**2)
        self._assert_coil_set_specs_match(coil_round_trip, coil_set_spec)

    def test_get_solved_runtime_state_uses_result_dofs(self):
        """Solved runtime summary must report solver-result DOFs, not live surface state."""
        s = self._make_real_surface(mpol=3, ntor=3, nfp=2)
        coils = _make_mock_coils()
        bs = _MockBiotSavart(coils)
        label = _PlumbingVolumeLabel(s)

        booz = BoozerSurfaceJAX(
            bs,
            s,
            label,
            targetlabel=0.1,
            constraint_weight=1.0,
        )
        booz.res = {
            "success": True,
            "primal_success": True,
            "sdofs": jnp.asarray([1.25, -0.5, 0.75], dtype=jnp.float64),
            "iota": jnp.asarray(0.23, dtype=jnp.float64),
            "G": jnp.asarray(1.7, dtype=jnp.float64),
            "weight_inv_modB": False,
        }
        booz.need_to_run_code = False
        booz._surface_dofs = jnp.asarray([9.0, -2.0, 3.5], dtype=jnp.float64)
        booz.surface.get_dofs = lambda: (_ for _ in ()).throw(
            AssertionError("live surface DOFs must not be reread")
        )

        solved_state = booz.get_solved_runtime_state()

        _assert_runtime_state_schema(solved_state, _SOLVED_RUNTIME_STATE_FIELDS)
        np.testing.assert_allclose(
            np.asarray(solved_state.sdofs),
            np.asarray(booz.res["sdofs"]),
        )
        np.testing.assert_allclose(np.asarray(solved_state.iota), 0.23)
        np.testing.assert_allclose(np.asarray(solved_state.G), 1.7)
        assert solved_state.weight_inv_modB is False

    def test_get_adjoint_runtime_state_exposes_runtime_callbacks_and_stream(
        self,
        monkeypatch,
    ):
        """Adjoint runtime summary must expose operator callbacks plus group VJPs."""
        s = self._make_real_surface(mpol=3, ntor=3, nfp=2)
        coils = _make_mock_coils()
        bs = _MockBiotSavart(coils)
        label = _PlumbingVolumeLabel(s)

        booz = BoozerSurfaceJAX(
            bs,
            s,
            label,
            targetlabel=0.1,
            constraint_weight=1.0,
        )
        expected_plu = (
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
        )
        recorded = {}

        def fake_vjp_groups(adjoint, passed_booz, iota, G):
            recorded["adjoint"] = np.asarray(adjoint)
            recorded["booz"] = passed_booz
            recorded["iota"] = np.asarray(iota)
            recorded["G"] = np.asarray(G)
            yield ("cotangent", (0, 1))

        booz.res = {
            "success": True,
            "primal_success": True,
            "sdofs": jnp.asarray([], dtype=jnp.float64),
            "iota": jnp.asarray(0.23, dtype=jnp.float64),
            "G": jnp.asarray(1.7, dtype=jnp.float64),
            "weight_inv_modB": True,
            "linearization_kind": "exact_jacobian",
            "adjoint_linear_solve_available": True,
            "dense_linear_solve_factors_available": True,
            "PLU": expected_plu,
            "vjp_groups": fake_vjp_groups,
        }
        booz.need_to_run_code = False
        booz._surface_dofs = jnp.asarray([], dtype=jnp.float64)
        monkeypatch.setattr(
            _bsj.BoozerSurfaceJAX,
            "_compute_stellsym_mask_indices",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            _bsj.BoozerSurfaceJAX,
            "_make_exact_residual",
            lambda self, _mask: lambda _x: _x,
        )
        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_jacobian_linear_operator",
            lambda _residual_fn, _x: {
                "matvec": lambda vec: vec,
                "transpose_matvec": lambda vec: vec,
            },
        )
        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_solve_jacobian_system",
            lambda _residual_fn, _x, rhs, *, transpose, tol: rhs,
        )
        monkeypatch.setattr(
            _bsj._optimizer_jax,
            "_solve_jacobian_system_with_status",
            lambda _residual_fn, _x, rhs, *, transpose, tol: (
                rhs,
                jnp.asarray(True),
            ),
        )

        adjoint_state = booz.get_adjoint_runtime_state()
        solved = adjoint_state.solve_transpose(
            jnp.asarray([2.0, -4.0], dtype=jnp.float64)
        )
        solved_with_status, solve_success = adjoint_state.solve_transpose_with_status(
            jnp.asarray([2.0, -4.0], dtype=jnp.float64)
        )
        streamed = list(
            adjoint_state.stream_group_vjps(jnp.asarray([5.0, -1.0], dtype=jnp.float64))
        )

        _assert_runtime_state_schema(
            adjoint_state,
            _ADJOINT_RUNTIME_STATE_FIELDS,
        )
        _assert_runtime_state_schema(
            adjoint_state.solved_state,
            _SOLVED_RUNTIME_STATE_FIELDS,
        )
        np.testing.assert_allclose(np.asarray(solved), np.asarray([2.0, -4.0]))
        np.testing.assert_allclose(
            np.asarray(solved_with_status),
            np.asarray([2.0, -4.0]),
        )
        assert bool(np.asarray(solve_success)) is True
        assert adjoint_state.decision_size == 2
        assert adjoint_state.dtype == jnp.float64
        _assert_operator_adjoint_state(
            adjoint_state,
            dense_factors_available=True,
        )
        np.testing.assert_allclose(
            np.asarray(
                adjoint_state.apply_transpose(
                    jnp.asarray([1.0, 3.0], dtype=jnp.float64)
                )
            ),
            np.asarray([1.0, 3.0]),
        )
        assert streamed == [("cotangent", (0, 1))]
        assert recorded["booz"] is booz
        np.testing.assert_allclose(recorded["adjoint"], np.asarray([5.0, -1.0]))
        np.testing.assert_allclose(recorded["iota"], 0.23)
        np.testing.assert_allclose(recorded["G"], 1.7)


# ---------------------------------------------------------------------------
# Coil-VJP CPU oracle parity
# ---------------------------------------------------------------------------


def _coil_dof_cotangent_l2_per_coil(deriv, coils):
    """Reduce a CPU ``Derivative`` to one ``ℓ₂(coil DOFs)`` value per coil.

    Reads per-Optimizable raw cotangent blocks directly from
    ``Derivative.data`` so that fully-fixed components (e.g. fixed
    ``Current``) contribute their stored cotangent rather than being
    discarded by the ``free_x`` filter. Each unique Optimizable
    contributes its squared cotangent norm at most once; this is
    important for symmetric fixtures (e.g. ``coils_via_symmetries``)
    where multiple ``Coil`` objects share the same ``Current`` instance
    and a naive per-coil sum would double-count the shared component.

    The per-coil ``ℓ₂`` norm is independent of the curve's internal DOF
    ordering and is a meaningful basis-free comparison surface for a
    JAX cotangent that has been routed through the same ``Coil.vjp``
    projector.
    """
    out = np.empty(len(coils), dtype=np.float64)
    seen: set[int] = set()
    for idx, coil in enumerate(coils):
        squared_norm = 0.0
        for component in (coil.curve, coil.current):
            component_id = id(component)
            if component_id in seen:
                continue
            seen.add(component_id)
            block = deriv.data.get(component)
            if block is None:
                continue
            block_np = np.asarray(block, dtype=np.float64)
            squared_norm += float(np.sum(block_np * block_np))
        out[idx] = float(np.sqrt(squared_norm))
    return out


def _project_jax_coil_cotangent_to_derivative(d_coil_arrays, coil_indices, coils):
    """Project per-quadrature JAX cotangents onto coil DOFs via ``Coil.vjp``.

    ``d_coil_arrays`` is the grouped ``[(d_g, d_gd, d_c), ...]`` pytree
    returned by both ``boozer_residual_coil_vjp`` and
    ``_boozer_ls_coil_vjp``; ``coil_indices`` maps each per-group slot back
    to the original coil-list index. This rebuilds a CPU ``Derivative`` so
    the JAX result lands in the same DOF basis as the CPU oracle and can be
    compared element-wise.
    """
    accumulated = None
    for group_idx, (d_g_group, d_gd_group, d_cur_group) in enumerate(d_coil_arrays):
        group_coil_indices = coil_indices[group_idx]
        d_g_group_np = np.asarray(d_g_group, dtype=np.float64)
        d_gd_group_np = np.asarray(d_gd_group, dtype=np.float64)
        d_cur_group_np = np.asarray(d_cur_group, dtype=np.float64)
        for slot, coil_idx in enumerate(group_coil_indices):
            coil = coils[int(coil_idx)]
            contribution = coil.vjp(
                d_g_group_np[slot],
                d_gd_group_np[slot],
                np.asarray([d_cur_group_np[slot]], dtype=np.float64),
            )
            accumulated = (
                contribution if accumulated is None else accumulated + contribution
            )
    return accumulated


@_skip_no_simsoptpp
class TestBoozerCoilVJPCpuOracle:
    """JAX coil VJPs match independent CPU-side oracles at derivative-heavy tols.

    There are two distinct JAX coil-VJP entry points and each must be
    compared to its own CPU oracle:

    - ``boozer_residual_coil_vjp`` differentiates the **Boozer residual**
      w.r.t. coils at fixed surface; the CPU oracle is
      ``boozer_surface_residual_dB`` followed by
      ``BiotSavart.B_vjp``.
    - ``_boozer_ls_coil_vjp`` differentiates the **LS penalty gradient**
      w.r.t. coils (residual + label + z); the CPU oracle is
      ``boozer_surface_dlsqgrad_dcoils_vjp``, which the JAX docstring
      explicitly cites as its replacement.

    Both tests evaluate at the same input state on a same-state fixture
    (no re-solve) and assert per-coil ``ℓ₂(coil DOFs)`` parity at the
    derivative-heavy lane (``rtol=1e-8, atol=1e-10``).
    """

    @staticmethod
    def _build_matched_fixture(*, weight_inv_modB):
        from simsopt.configs import get_data
        from simsopt.field.biotsavart import BiotSavart
        from simsopt.geo import Volume
        from simsopt.geo.boozer_residual_jax import boozer_residual_coil_vjp
        from simsopt.geo.boozersurface import BoozerSurface
        from simsopt.geo.surfaceobjectives import (
            boozer_surface_dlsqgrad_dcoils_vjp,
            boozer_surface_residual_dB,
        )

        from .surface_test_helpers import get_surface

        _, base_currents, ma, nfp, bs = get_data("ncsx")
        surface = get_surface("SurfaceXYZTensorFourier", True, nfp=nfp)
        surface.fit_to_curve(ma, 0.1)

        biotsavart = BiotSavart(bs.coils)
        biotsavart.set_points(surface.gamma().reshape((-1, 3)))

        # Same-state fixture: pin iota / G at the upstream Boozer initial
        # guess. We do not re-solve because both oracles only need
        # *consistent* inputs to be comparable.
        iota = -0.4
        current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
        G = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7) / (2 * np.pi)

        # JAX side: extract raw arrays.
        gamma_jax = jnp.asarray(surface.gamma(), dtype=jnp.float64)
        xphi_jax = jnp.asarray(surface.gammadash1(), dtype=jnp.float64)
        xtheta_jax = jnp.asarray(surface.gammadash2(), dtype=jnp.float64)
        gammas_stack = jnp.asarray(
            np.stack([c.curve.gamma() for c in bs.coils]), dtype=jnp.float64
        )
        gammadashs_stack = jnp.asarray(
            np.stack([c.curve.gammadash() for c in bs.coils]), dtype=jnp.float64
        )
        currents_stack = jnp.asarray(
            np.array([c.current.get_value() for c in bs.coils]), dtype=jnp.float64
        )
        coil_arrays = [(gammas_stack, gammadashs_stack, currents_stack)]
        coil_indices = [list(range(len(bs.coils)))]

        return {
            "surface": surface,
            "biotsavart": biotsavart,
            "coils": bs.coils,
            "iota": iota,
            "G": G,
            "weight_inv_modB": weight_inv_modB,
            "gamma_jax": gamma_jax,
            "xphi_jax": xphi_jax,
            "xtheta_jax": xtheta_jax,
            "coil_arrays": coil_arrays,
            "coil_indices": coil_indices,
            "BoozerSurface": BoozerSurface,
            "Volume": Volume,
            "boozer_surface_residual_dB": boozer_surface_residual_dB,
            "boozer_surface_dlsqgrad_dcoils_vjp": boozer_surface_dlsqgrad_dcoils_vjp,
            "boozer_residual_coil_vjp": boozer_residual_coil_vjp,
        }

    @pytest.mark.parametrize("weight_inv_modB", [False, True])
    def test_residual_coil_vjp_matches_cpu_oracle(self, weight_inv_modB):
        """Test A — ``boozer_residual_coil_vjp`` vs CPU residual + ``B_vjp``."""
        fixture = self._build_matched_fixture(weight_inv_modB=weight_inv_modB)
        nphi, ntheta = fixture["gamma_jax"].shape[:2]
        rng = np.random.default_rng(seed=20260515 + int(weight_inv_modB))
        adjoint = rng.standard_normal(nphi * ntheta * 3)
        adjoint /= np.linalg.norm(adjoint)

        # CPU oracle: residual + dr/dB followed by B_vjp.
        _r_cpu, drtil_dB = fixture["boozer_surface_residual_dB"](
            fixture["surface"],
            fixture["iota"],
            fixture["G"],
            fixture["biotsavart"],
            derivatives=0,
            weight_inv_modB=weight_inv_modB,
        )
        # ``adjoint^T @ drtil_dB`` reduces over residual rows. Each
        # quadrature point contributes 3 residual rows whose B-cotangents
        # add into a single per-point 3-vector.
        adjoint_per_point = adjoint.reshape(nphi * ntheta, 3)
        drtil_dB_per_point = drtil_dB.reshape(nphi * ntheta, 3, 3)
        cotangent_on_B = np.einsum(
            "pr,prk->pk",
            adjoint_per_point,
            drtil_dB_per_point,
        )
        cpu_deriv = fixture["biotsavart"].B_vjp(cotangent_on_B)
        cpu_per_coil = _coil_dof_cotangent_l2_per_coil(cpu_deriv, fixture["coils"])

        # JAX side.
        (d_coil_arrays,) = fixture["boozer_residual_coil_vjp"](
            jnp.asarray(adjoint, dtype=jnp.float64),
            gamma=fixture["gamma_jax"],
            xphi=fixture["xphi_jax"],
            xtheta=fixture["xtheta_jax"],
            coil_arrays=fixture["coil_arrays"],
            iota=fixture["iota"],
            G=fixture["G"],
            weight_inv_modB=weight_inv_modB,
        )
        jax_deriv = _project_jax_coil_cotangent_to_derivative(
            d_coil_arrays,
            fixture["coil_indices"],
            fixture["coils"],
        )
        jax_per_coil = _coil_dof_cotangent_l2_per_coil(jax_deriv, fixture["coils"])

        # Derivative-heavy lane tolerance from the parity ladder SSOT.
        tols = parity_ladder_tolerances("derivative-heavy")
        np.testing.assert_allclose(
            jax_per_coil,
            cpu_per_coil,
            rtol=float(tols["first_derivative_rtol"]),
            atol=float(tols["first_derivative_atol"]),
            err_msg=(
                f"Per-coil residual VJP norms disagree (weight_inv_modB="
                f"{weight_inv_modB}). cpu={cpu_per_coil}, jax={jax_per_coil}"
            ),
        )

    @pytest.mark.parametrize("weight_inv_modB", [True, False])
    def test_ls_coil_vjp_matches_cpu_oracle(self, weight_inv_modB):
        """Test B — ``_boozer_ls_coil_vjp`` vs ``boozer_surface_dlsqgrad_dcoils_vjp``.

        Both oracles compute ``λᵀ d/d_coils [grad_x(½‖r‖²)]`` for the
        Boozer-residual contribution to the LS penalty gradient. CPU's
        ``boozer_surface_dlsqgrad_dcoils_vjp`` evaluates this analytically;
        JAX's ``_boozer_ls_coil_vjp`` autodiffs the FULL LS penalty
        ``½‖r‖² + ½ w_c (label − target)² + ½ w_c z_axis²``. For Volume
        label + stellsym=True the label and z contributions to
        ``d/d_coils`` vanish analytically (Volume depends only on surface
        DOFs; ``z_axis = 0`` structurally under stellsym), so both
        backends compute the same scalar.

        Same-state setup: build the JAX side AT CPU's converged state by
        cloning ``surf_cpu`` post-solve. We do NOT run a separate JAX
        solve and then mutate ``surf_jax``: ``BoozerSurfaceJAX`` snapshots
        ``surface.get_dofs()`` into ``self._surface_dofs`` at construction
        and refreshes that cache only inside ``run_code()`` /
        ``_set_surface_dofs()``. Mutating ``surf_jax.x`` post-solve leaves
        the cache stale, so the VJP would silently evaluate at the
        prior solver state. Constructing JAX at the synced state lets
        ``BoozerSurfaceJAX.__init__`` populate the cache correctly.
        """
        from benchmarks.benchmark_problem import (
            build_ls_parity_problem,
            clone_tensor_surface,
        )
        from simsopt.field.biotsavart import BiotSavart
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
        from simsopt.geo import Volume
        from simsopt.geo.boozersurface import BoozerSurface
        from simsopt.geo.boozersurface_jax import (
            BoozerSurfaceJAX,
            _boozer_ls_coil_vjp,
        )
        from simsopt.geo.surfaceobjectives import boozer_surface_dlsqgrad_dcoils_vjp

        problem = build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)
        solver_options = {
            "verbose": False,
            "bfgs_maxiter": 1500,
            "bfgs_tol": 1e-8,
            "newton_maxiter": 40,
            "newton_tol": 1e-9,
            "weight_inv_modB": weight_inv_modB,
        }

        # CPU solve defines the reference converged state.
        surf_cpu = clone_tensor_surface(problem.surface)
        bs_cpu = BiotSavart(problem.coils)
        vol_cpu = Volume(surf_cpu)
        booz_cpu = BoozerSurface(
            bs_cpu,
            surf_cpu,
            vol_cpu,
            problem.vol_target,
            constraint_weight=1.0,
            options=dict(solver_options),
        )
        booz_cpu.run_code(problem.iota0, G=problem.G0)
        assert booz_cpu.res.get("success", False), "CPU LS solve did not converge"

        iota_at = float(booz_cpu.res["iota"])
        G_at = float(booz_cpu.res["G"]) if booz_cpu.res.get("G") is not None else None
        decision_size = booz_cpu.res["jacobian"].shape[0]

        # Build JAX side at the converged state. ``clone_tensor_surface``
        # copies ``surf_cpu`` (which is already at the converged state),
        # so ``BoozerSurfaceJAX.__init__`` snapshots the converged DOFs
        # into its internal cache. No JAX solve is needed and no manual
        # cache refresh is required.
        surf_jax = clone_tensor_surface(surf_cpu)
        bs_jax = BiotSavartJAX(problem.coils)
        vol_jax = Volume(surf_jax)
        booz_jax = BoozerSurfaceJAX(
            bs_jax,
            surf_jax,
            vol_jax,
            problem.vol_target,
            constraint_weight=1.0,
            options={**solver_options, "optimizer_backend": "scipy"},
        )

        rng = np.random.default_rng(seed=20260515 + int(weight_inv_modB) + 100)
        adjoint = rng.standard_normal(decision_size)
        adjoint /= np.linalg.norm(adjoint)

        cpu_deriv = boozer_surface_dlsqgrad_dcoils_vjp(
            adjoint,
            booz_cpu,
            iota_at,
            G_at,
            weight_inv_modB=weight_inv_modB,
        )
        cpu_per_coil = _coil_dof_cotangent_l2_per_coil(cpu_deriv, problem.coils)

        d_coil_arrays_jax, coil_indices_jax = _boozer_ls_coil_vjp(
            jnp.asarray(adjoint, dtype=jnp.float64),
            booz_jax,
            iota_at,
            G_at,
            weight_inv_modB=weight_inv_modB,
        )
        jax_deriv = _project_jax_coil_cotangent_to_derivative(
            d_coil_arrays_jax,
            coil_indices_jax,
            problem.coils,
        )
        jax_per_coil = _coil_dof_cotangent_l2_per_coil(jax_deriv, problem.coils)

        tols = parity_ladder_tolerances("derivative-heavy")
        np.testing.assert_allclose(
            jax_per_coil,
            cpu_per_coil,
            rtol=float(tols["first_derivative_rtol"]),
            atol=float(tols["first_derivative_atol"]),
            err_msg=(
                f"Per-coil LS VJP norms disagree (weight_inv_modB="
                f"{weight_inv_modB}). cpu={cpu_per_coil}, jax={jax_per_coil}"
            ),
        )
