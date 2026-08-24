"""Full-census ratchet for the JAX host/device boundary owners."""

from __future__ import annotations

import ast
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from simsopt_jax.backend.dtypes import runtime_device_put_tree
from simsopt_jax.runtime.host_boundary import block_until_ready, host_tree_after_ready

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPO_ROOT / "src/simsopt_jax",
    REPO_ROOT / "src/simsopt_jax_adapters",
)

# Direct JAX boundary calls belong only in these implementation functions.
_ALLOWED_OWNER_CALLS = frozenset(
    {
        # H2D placement and D2H/readiness SSOT implementations.
        "src/simsopt_jax/backend/dtypes.py::_device_put::device_put",
        "src/simsopt_jax/backend/dtypes.py::runtime_device_put_tree::device_put",
        "src/simsopt_jax/runtime/host_boundary.py::block_until_ready::block_until_ready",
        "src/simsopt_jax/runtime/host_boundary.py::host_value::device_get",
        # Sharding owns explicit H2D/D2D placement scopes for mesh collectives.
        "src/simsopt_jax/core/sharding.py::_place_leading_axis_arrays::transfer_guard_device_to_device",
        "src/simsopt_jax/core/sharding.py::_place_leading_axis_arrays::transfer_guard_host_to_device",
        "src/simsopt_jax/core/sharding.py::replicate_tree_on_mesh::transfer_guard_device_to_device",
        "src/simsopt_jax/core/sharding.py::replicate_tree_on_mesh::transfer_guard_host_to_device",
        # JAX library calls that internally stage scalar loop/GMRES constants.
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_hager_higham_inverse_1_norm_estimate::transfer_guard_host_to_device",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_run_operator_gmres::transfer_guard_host_to_device",
        # Commit fed04522c added the counted incremental GMRES sibling; it scopes
        # the same allowance around one ``lax.custom_linear_solve`` call.
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_run_operator_gmres_counted_incremental::transfer_guard_host_to_device",
        "src/simsopt_jax/geo/optimizers/optimizer.py::_gmres_solve_least_squares_system::transfer_guard_host_to_device",
        "src/simsopt_jax/solve/dispatch.py::_run_optimistix_lm::transfer_guard_host_to_device",
        "src/simsopt_jax/solve/minimize_runtime.py::run_optimistix_minimize::transfer_guard_host_to_device",
        # This file-reporting bridge explicitly permits its owned D2H log writes.
        "src/simsopt_jax/solve/serial.py::_write_bounded_objective_log::transfer_guard",
        # Explicit host-extension and reporting bridge scopes.
        "src/simsopt_jax/geo/optimizers/reference.py::_scipy_host_array::transfer_guard_device_to_host",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_array_from_scipy_host::transfer_guard_host_to_device",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_scipy_host_extension_scope::transfer_guard_device_to_host",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_scipy_host_extension_scope::transfer_guard_host_to_device",
        "src/simsopt_jax_adapters/geo/boozer_surface.py::wrapped::transfer_guard_device_to_host",
        "src/simsopt_jax_adapters/geo/boozer_surface.py::wrapped::transfer_guard_host_to_device",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_baseline_reporting_metrics::transfer_guard_device_to_host",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::compute_baseline_value_and_grad::transfer_guard_device_to_host",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::compute_baseline_value_and_grad::transfer_guard_host_to_device",
        # Host-boundary call sites in the nested-LS solver and probe modules:
        # host-side norms, convergence receipts, and probe evaluation at the
        # solver boundary. Entered verbatim (2026-08-24) so the gate stays
        # exact; rerouting a call site through the runtime/host_boundary SSOT
        # helpers removes its entry from this list.
        "src/simsopt_jax_adapters/geo/nested_ls_newton_parity.py::_jax_cpu_ordered_value_and_grad::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::_envelope_value_and_grad::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::_host_vector::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::_require_full_phi_yy::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::factor_reduced_nested_ls_schur::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::factor_schur_fourier_block_preconditioner::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::require_full_y_rank::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::run_reduced_nested_ls_newton::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::run_reduced_nested_ls_schur_newton::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_counted_gmres_row::block_until_ready",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_counted_gmres_row::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_counted_gmres_row::transfer_guard_host_to_device",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_flat675_value_and_grad_at::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_live_unpreconditioned_eta::block_until_ready",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_live_unpreconditioned_eta::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_mixed_coil_correction_vjp::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_banana_probe::block_until_ready",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_banana_probe::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_warm_probe::block_until_ready",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_endpoint_adjoint_probe::block_until_ready",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_endpoint_adjoint_probe::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe::block_until_ready",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_volume_outer_probe::block_until_ready",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_volume_outer_probe::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::nested_ls_outer_value_and_grad::block_until_ready",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::nested_ls_outer_value_and_grad::device_get",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::prepare_f3_b37_outer_state::device_get",
    }
)
_BOUNDARY_CALL_NAMES = frozenset(
    {
        "block_until_ready",
        "device_get",
        "device_put",
        "transfer_guard",
        "transfer_guard_device_to_device",
        "transfer_guard_device_to_host",
        "transfer_guard_host_to_device",
    }
)


def _iter_python_files():
    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            yield path, relative


class _BoundaryCallCensus(ast.NodeVisitor):
    def __init__(self, relative_path: str, tree: ast.Module) -> None:
        self.relative_path = relative_path
        self.function_names: list[str] = []
        self.calls: list[str] = []
        self.jax_module_aliases = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "jax"
        }
        self.jax_symbol_aliases = {
            alias.asname or alias.name: alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "jax"
            for alias in node.names
            if alias.name in _BOUNDARY_CALL_NAMES
        }

    def _boundary_call_name(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return self.jax_symbol_aliases.get(node.func.id)
        if not isinstance(node.func, ast.Attribute):
            return None
        if node.func.attr not in _BOUNDARY_CALL_NAMES:
            return None
        if node.func.attr == "block_until_ready":
            return node.func.attr
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.jax_module_aliases
        ):
            return node.func.attr
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_names.append(node.name)
        self.generic_visit(node)
        self.function_names.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_names.append(node.name)
        self.generic_visit(node)
        self.function_names.pop()

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._boundary_call_name(node)
        if call_name is not None:
            function_name = (
                self.function_names[-1] if self.function_names else "<module>"
            )
            self.calls.append(f"{self.relative_path}::{function_name}::{call_name}")
        self.generic_visit(node)


def _direct_boundary_calls() -> set[str]:
    calls: set[str] = set()
    for path, relative in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        census = _BoundaryCallCensus(relative, tree)
        census.visit(tree)
        calls.update(census.calls)
    return calls


def test_only_boundary_owners_call_jax_transfer_and_readiness_primitives() -> None:
    actual = _direct_boundary_calls()
    unexpected = sorted(actual - _ALLOWED_OWNER_CALLS)
    stale = sorted(_ALLOWED_OWNER_CALLS - actual)

    assert not unexpected, (
        "Direct JAX boundary calls outside owners:\n  " + "\n  ".join(unexpected)
    )
    assert not stale, "Stale owner-call allowlist entries:\n  " + "\n  ".join(stale)


def test_runtime_device_put_tree_preserves_structure_and_exact_leaf_dtypes() -> None:
    value = {
        "float": np.asarray([1.0, 2.0], dtype=np.float32),
        "integer": (np.asarray(3, dtype=np.int16),),
    }

    placed = runtime_device_put_tree(value)

    assert placed.keys() == value.keys()
    assert placed["float"].dtype == jnp.float32
    assert placed["integer"][0].dtype == jnp.int16


def test_readiness_and_host_tree_preserve_pytree_structure() -> None:
    value = {
        "vector": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        "scalar": (jnp.asarray(3, dtype=jnp.int32),),
    }

    ready = block_until_ready(value)
    host = host_tree_after_ready(ready)

    assert host.keys() == value.keys()
    np.testing.assert_array_equal(host["vector"], np.asarray([1.0, 2.0]))
    assert host["scalar"][0].item() == 3
    assert host["vector"].flags.writeable
