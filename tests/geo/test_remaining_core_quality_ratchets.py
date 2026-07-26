"""Architecture ratchets for the remaining core JAX quality items."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACEABLE_OBJECTIVES = (
    REPO_ROOT / "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py"
)
SURFACE_OBJECTIVES = REPO_ROOT / "src/simsopt_jax_adapters/geo/surface_objectives.py"
OBJECTIVE_ADAPTERS = (SURFACE_OBJECTIVES, TRACEABLE_OBJECTIVES)
BOOZER_SURFACE = REPO_ROOT / "src/simsopt_jax_adapters/geo/boozer_surface.py"
PRIVATE_FACADE_FREE_ADAPTERS = (*OBJECTIVE_ADAPTERS, BOOZER_SURFACE)
OPTIMIZER_ROOT = REPO_ROOT / "src/simsopt_jax/geo/optimizers"
LINEAR_SOLVE = OPTIMIZER_ROOT / "linear_solve.py"
DENSE_IR = OPTIMIZER_ROOT / "dense_ir.py"
ADJOINT_LINEAR_SOLVE = OPTIMIZER_ROOT / "adjoint_linear_solve.py"
OPTIMIZER = OPTIMIZER_ROOT / "optimizer.py"
SHARED = OPTIMIZER_ROOT / "_shared.py"

LINEAR_SOLVE_MODULE = "simsopt_jax.geo.optimizers.linear_solve"
DENSE_IR_MODULE = "simsopt_jax.geo.optimizers.dense_ir"
ADJOINT_LINEAR_SOLVE_MODULE = "simsopt_jax.geo.optimizers.adjoint_linear_solve"
OPTIMIZER_MODULE = "simsopt_jax.geo.optimizers.optimizer"

_OPTIMIZER_COMPAT_REEXPORT_ALLOWLIST = {
    LINEAR_SOLVE_MODULE: frozenset(
        {
            "_CACHEABLE_LINEAR_OPERATOR_ATTR",
            "_CACHED_HVP_ATTR",
            "_CACHED_JVP_ATTR",
            "_DENSE_LINEAR_SOLVE_RESIDUAL_DIMENSION_FACTOR",
            "_DENSE_LINEAR_SOLVE_SMALL_SOLUTION_FACTOR",
            "_DENSE_OPERATOR_ACTIVATION_BYTES_PER_PARALLEL_COLUMN",
            "_DENSE_OPERATOR_CHUNK_BATCH_SIZE",
            "_DENSE_OPERATOR_CHUNK_BATCH_SIZE_ENV",
            "_DENSE_OPERATOR_CHUNK_BATCH_SIZE_FALLBACK",
            "_DENSE_OPERATOR_CHUNK_BATCH_SIZE_MAX",
            "_DENSE_OPERATOR_DEFAULT_BUDGET_BYTES",
            "_DENSE_OPERATOR_LEGACY_BYTES_PER_PARALLEL_COLUMN",
            "_EXACT_ADJOINT_DENSE_LU",
            "_FLOAT64_DENSE_MATRIX_MAX_CONDITION_ESTIMATE",
            "_HAGER_HIGHAM_CONDITION_ITERATIONS",
            "_JIT_LINEAR_OPERATOR_CACHE_LOCK",
            "_LINEAR_SOLVE_ITERATIONS_UNKNOWN",
            "_LinearSolveStatus",
            "_SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS",
            "_apply_column_batched_operator",
            "_cached_jit_linear_operator",
            "_combine_linear_solve_iteration_counts",
            "_complete_linear_solve_status",
            "_dense_linear_solve_status",
            "_dense_matrix_backward_error_success",
            "_dense_matrix_condition_estimate",
            "_dense_matrix_condition_estimate_numerically_safe",
            "_dense_matrix_nonsingular_threshold",
            "_dense_matrix_solve_forward_error_success",
            "_dense_matrix_solve_numerically_safe",
            "_dense_matrix_solve_small_solution_success",
            "_dense_operator_chunk_batch_size_from_budget",
            "_dense_square_operator_lu_materialization_allowed",
            "_dense_square_operator_materialization_allowed",
            "_dense_square_operator_matrix",
            "_dense_square_operator_matrix_bytes_allowed",
            "_dense_square_operator_matrix_dtype",
            "_device_int32",
            "_device_scalar",
            "_effective_dense_backward_error_tolerance",
            "_effective_linear_solve_tolerance",
            "_factor_dense_hessian",
            "_forward_error_bound",
            "_forward_error_success",
            "_forward_error_tolerance",
            "_gmres_iteration_limits",
            "_gmres_solve_array_system",
            "_hager_higham_inverse_1_norm_estimate",
            "_hessian_vector_product_fn",
            "_jacobian_linear_operator",
            "_jacobian_vector_product_fn",
            "_linear_solve_effective_tolerance_reached",
            "_linear_solve_finite",
            "_linear_solve_iteration_count",
            "_linear_solve_iterations_host_value",
            "_linear_solve_residual_scale",
            "_linear_solve_residual_tolerance",
            "_linear_solve_solution_or_nan",
            "_linear_solve_status",
            "_linear_solve_status_iterations",
            "_linear_solve_status_success",
            "_lu_solve_dense_hessian",
            "_materialize_dense_hessian",
            "_materialize_dense_hessian_host",
            "_materialize_dense_jacobian",
            "_materialize_dense_linear_operator",
            "_matrix_one_norm",
            "_place_like_concrete_array",
            "_place_like_concrete_scalar",
            "_plu_from_lu_piv",
            "_relative_residual_1_norm",
            "_relative_residual_norm",
            "_resolve_dense_operator_chunk_batch_size",
            "_run_operator_gmres",
            "_solve_dense_square_operator_least_squares_system_with_status",
            "_solve_dense_square_operator_lu_system_with_status",
            "_solve_jacobian_operator",
            "_solve_jacobian_operator_with_status",
            "_solve_jacobian_system_with_status",
            "_solve_square_array_system_operator_only",
            "_solve_square_vector_system_operator_only",
            "_solve_square_vector_system_operator_only_nonzero_rhs",
            "_symmetrize_dense_hessian",
            "_terminal_linear_solve_status",
        }
    ),
    DENSE_IR_MODULE: frozenset(
        {
            "DENSE_IR_HISTORY_MAX_CORRECTIONS",
            "_DENSE_IR_NEWTON_MATVEC_BUDGET",
            "_DENSE_IR_NEWTON_REFINEMENT_STEPS",
            "_DenseIrContractionTelemetry",
            "_DenseIrHistory",
            "_DenseIrRefinementState",
            "_MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND",
            "_MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT",
            "_MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA",
            "_MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT",
            "_MixedDenseIrFallbackTelemetry",
            "_MixedDenseIrResolvedPolicy",
            "_MixedDenseIrSolveStatus",
            "_MixedDenseIrTrustTelemetry",
            "_certify_mixed_dense_ir_factors_with_telemetry",
            "_fixed_dense_ir_history",
            "_inactive_dense_ir_history",
            "_inactive_mixed_dense_ir_fallback_telemetry",
            "_inactive_mixed_dense_ir_trust_telemetry",
            "_materialize_dense_ir_proposal_matrix",
            "_mixed_dense_ir_contraction_operator_norm_upper",
            "_mixed_dense_ir_correction_tail_relative_bound",
            "_run_dense_ir_refinement",
            "_solve_dense_ir_system_with_contraction_telemetry",
            "_solve_dense_ir_system_with_status",
            "_solve_fp64_dense_ir_rebuild_with_telemetry",
            "_solve_mixed_dense_ir_operator_with_status",
            "_solve_mixed_dense_ir_operator_with_telemetry",
            "resolve_mixed_dense_ir_policy",
        }
    ),
    ADJOINT_LINEAR_SOLVE_MODULE: frozenset(
        {
            "_ADJOINT_LINEAR_SOLVER",
            "_AdjointHessianLinearSolver",
            "_EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS",
            "_hessian_linear_operator",
            "_lineax_lsmr_solver",
            "_require_tree_first_leaf",
            "_solve_hessian_least_squares_system_with_status",
            "_solve_hessian_system",
            "_solve_hessian_system_with_status",
            "_solve_regularized_normal_system_lsmr_j_with_status",
            "_solve_symmetric_operator_cg_with_status",
            "adjoint_hessian_stabilization",
        }
    ),
}


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            definitions.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            definitions.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    return definitions


def _top_level_import_aliases(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                aliases.setdefault(alias.name, set()).add(bound_name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                imported_name = f"{node.module}.{alias.name}"
                bound_name = alias.asname or alias.name
                aliases.setdefault(imported_name, set()).add(bound_name)
    return aliases


def _top_level_import_aliases_by_module(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            bound_names = aliases.setdefault(node.module, set())
            bound_names.update(alias.asname or alias.name for alias in node.names)
    return aliases


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_objective_adapters_do_not_reach_into_optimizer_privates() -> None:
    """R07: adapters import acyclic owners instead of the optimizer facade."""
    private_reaches: list[str] = []
    optimizer_aliases: list[str] = []
    for path in PRIVATE_FACADE_FREE_ADAPTERS:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "_optimizer_jax._" in line:
                private_reaches.append(f"{path.name}:{lineno}")
            if "shared_linear_solve" in line:
                private_reaches.append(f"{path.name}:{lineno}:shared_linear_solve")

    for path in OBJECTIVE_ADAPTERS:
        aliases = _top_level_import_aliases(path)
        optimizer_aliases.extend(
            f"{path.name}:{alias}" for alias in aliases.get(OPTIMIZER_MODULE, set())
        )
        optimizer_aliases.extend(
            f"{path.name}:{alias}"
            for alias in aliases.get("simsopt_jax.geo.optimizers.optimizer", set())
        )

    assert private_reaches == []
    assert optimizer_aliases == []


def test_linear_solve_owners_contain_their_actual_definitions() -> None:
    """R07: owner modules are implementations, not pass-through facades."""
    expected_definitions = {
        LINEAR_SOLVE: {
            "_LinearSolveStatus",
            "_dense_square_operator_matrix",
            "_hessian_vector_product_fn",
            "_jacobian_vector_product_fn",
            "_linear_solve_status",
            "_solve_dense_square_operator_least_squares_system_with_status",
            "_solve_dense_square_operator_lu_system_with_status",
        },
        DENSE_IR: {
            "_MixedDenseIrSolveStatus",
            "_run_dense_ir_refinement",
            "_solve_mixed_dense_ir_operator_with_status",
        },
        ADJOINT_LINEAR_SOLVE: {
            "_ADJOINT_LINEAR_SOLVER",
            "_hessian_linear_operator",
            "_solve_hessian_least_squares_system_with_status",
        },
        SHARED: {
            "cast_floating_tree",
            "mark_cacheable_jit_value_and_grad",
        },
    }
    for path, expected in expected_definitions.items():
        assert expected <= _top_level_definitions(path), path

    optimizer_definitions = _top_level_definitions(OPTIMIZER)
    assert not optimizer_definitions.intersection(
        set().union(*expected_definitions.values())
    )


def test_optimizer_compat_reexports_are_explicitly_allowlisted() -> None:
    """R07: optimizer.py may only keep declared compatibility reexports."""
    actual_aliases = _top_level_import_aliases_by_module(OPTIMIZER)
    compat_aliases = {
        module: frozenset(actual_aliases.get(module, set()))
        for module in _OPTIMIZER_COMPAT_REEXPORT_ALLOWLIST
    }
    assert compat_aliases == _OPTIMIZER_COMPAT_REEXPORT_ALLOWLIST


def test_linear_solve_owner_graph_has_no_reverse_or_facade_edges() -> None:
    """R07: lower-level owners never import optimizer or compatibility facades."""
    forbidden_modules = {
        "simsopt_jax.geo.optimizers.optimizer",
        "simsopt_jax.geo.optimizers.private._dense_ir",
        "simsopt_jax.geo.optimizers.shared_linear_solve",
    }
    for path in (SHARED, LINEAR_SOLVE, DENSE_IR, ADJOINT_LINEAR_SOLVE):
        assert not (_imported_modules(path) & forbidden_modules), path
    assert not (OPTIMIZER_ROOT / "shared_linear_solve.py").exists()


def test_traceable_runtime_state_is_not_attached_to_boozer_objects() -> None:
    """R08: mutable traceable runtime state belongs to an explicit session."""
    forbidden = (
        "_traceable_runtime_entry_cache",
        "_traceable_solved_state_value_and_grad_entry_cache",
        "_traceable_penalty_objective_cache",
        "_traceable_penalty_residual_cache",
        "_traceable_exact_residual_cache",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (TRACEABLE_OBJECTIVES, BOOZER_SURFACE)
    )
    assert [name for name in forbidden if name in source] == []

    traceable_tree = ast.parse(
        TRACEABLE_OBJECTIVES.read_text(encoding="utf-8"),
        filename=str(TRACEABLE_OBJECTIVES),
    )
    assert [
        node.lineno
        for node in ast.walk(traceable_tree)
        if isinstance(node, ast.Nonlocal)
    ] == []


def test_objective_adapters_use_the_host_boundary_owner() -> None:
    """R13: objective adapters contain no direct device-to-host operations."""
    violations: list[str] = []
    for path in (TRACEABLE_OBJECTIVES, BOOZER_SURFACE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "jax"
                and node.attr == "device_get"
            ):
                violations.append(f"{path.name}:{node.lineno}")
    assert violations == []
