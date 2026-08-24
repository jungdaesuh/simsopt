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
_JAX_MODULE_BINDING = "jax-module"
_NON_JAX_BINDING = "non-jax"

# Each allowlist item names one direct invocation, not merely an owning function.
# The source coordinate deliberately ratchets additions, removals, and duplicate
# primitive calls. A JAX import or direct alias must resolve lexically to count.
# This baseline contains 82 direct invocations, including all 55 nested-LS calls.
_ALLOWED_OWNER_CALLS = frozenset(
    {
        "src/simsopt_jax/backend/dtypes.py::_device_put::device_put::286:19",
        "src/simsopt_jax/backend/dtypes.py::_device_put::device_put::287:15",
        "src/simsopt_jax/backend/dtypes.py::_device_put::device_put::288:11",
        "src/simsopt_jax/backend/dtypes.py::runtime_device_put_tree::device_put::314:15",
        "src/simsopt_jax/backend/dtypes.py::runtime_device_put_tree::device_put::315:11",
        "src/simsopt_jax/core/sharding.py::_place_leading_axis_arrays::transfer_guard_device_to_device::255:13",
        "src/simsopt_jax/core/sharding.py::_place_leading_axis_arrays::transfer_guard_host_to_device::254:9",
        "src/simsopt_jax/core/sharding.py::replicate_tree_on_mesh::transfer_guard_device_to_device::249:13",
        "src/simsopt_jax/core/sharding.py::replicate_tree_on_mesh::transfer_guard_host_to_device::248:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_hager_higham_inverse_1_norm_estimate::transfer_guard_host_to_device::1337:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_run_operator_gmres::transfer_guard_host_to_device::685:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_run_operator_gmres_counted_incremental::transfer_guard_host_to_device::967:9",
        "src/simsopt_jax/geo/optimizers/optimizer.py::_gmres_solve_least_squares_system::transfer_guard_host_to_device::2728:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_scipy_host_array::transfer_guard_device_to_host::231:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_array_from_scipy_host::transfer_guard_host_to_device::248:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_scipy_host_extension_scope::transfer_guard_device_to_host::122:13",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_scipy_host_extension_scope::transfer_guard_host_to_device::121:9",
        "src/simsopt_jax/runtime/host_boundary.py::block_until_ready::block_until_ready::209:11",
        "src/simsopt_jax/runtime/host_boundary.py::host_value::device_get::174:11",
        "src/simsopt_jax/solve/dispatch.py::_run_optimistix_lm::transfer_guard_host_to_device::560:9",
        "src/simsopt_jax/solve/minimize_runtime.py::run_optimistix_minimize::transfer_guard_host_to_device::301:9",
        "src/simsopt_jax/solve/serial.py::_write_bounded_objective_log::transfer_guard::455:9",
        "src/simsopt_jax_adapters/geo/boozer_surface.py::_with_host_bridge_transfer_guard.wrapped::transfer_guard_device_to_host::198:13",
        "src/simsopt_jax_adapters/geo/boozer_surface.py::_with_host_bridge_transfer_guard.wrapped::transfer_guard_host_to_device::199:17",
        "src/simsopt_jax_adapters/geo/nested_ls_newton_parity.py::_jax_cpu_ordered_value_and_grad::device_get::160:24",
        "src/simsopt_jax_adapters/geo/nested_ls_newton_parity.py::_jax_cpu_ordered_value_and_grad::device_get::162:12",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::_envelope_value_and_grad::device_get::797:25",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::_host_vector::device_get::577:20",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::_require_full_phi_yy::device_get::384:24",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::factor_reduced_nested_ls_schur::device_get::436:30",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::factor_schur_fourier_block_preconditioner::device_get::928:45",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::require_full_y_rank::device_get::317:29",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::require_full_y_rank::device_get::318:26",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::require_full_y_rank::device_get::321:12",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::run_reduced_nested_ls_newton::device_get::758:37",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::run_reduced_nested_ls_newton::device_get::763:32",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::run_reduced_nested_ls_newton::device_get::768:37",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::run_reduced_nested_ls_newton::device_get::771:32",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::run_reduced_nested_ls_newton::device_get::779:39",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced.py::run_reduced_nested_ls_schur_newton::device_get::1356:28",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_flat675_value_and_grad_at::device_get::4759:14",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_flat675_value_and_grad_at::device_get::4760:19",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_live_unpreconditioned_eta::block_until_ready::2163:15",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_live_unpreconditioned_eta::device_get::2165:34",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_live_unpreconditioned_eta::device_get::2167:45",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::_mixed_coil_correction_vjp::device_get::4814:8",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_banana_probe::block_until_ready::3628:16",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_banana_probe::block_until_ready::3645:20",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_banana_probe::block_until_ready::3647:23",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_banana_probe::device_get::3636:30",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_banana_probe::device_get::3651:42",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_warm_probe::block_until_ready::4372:8",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_chunk_warm_probe::block_until_ready::4382:12",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_endpoint_adjoint_probe::block_until_ready::3004:34",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_endpoint_adjoint_probe::block_until_ready::3018:22",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_endpoint_adjoint_probe::block_until_ready::3026:29",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_endpoint_adjoint_probe::device_get::3013:31",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_endpoint_adjoint_probe::device_get::3022:36",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_endpoint_adjoint_probe::device_get::3029:31",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe._counted_gmres_row::block_until_ready::2469:16",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe._counted_gmres_row::device_get::2474:16",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe._counted_gmres_row::transfer_guard_host_to_device::2457:13",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe::block_until_ready::2505:34",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe::block_until_ready::2511:19",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe::block_until_ready::2518:25",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe::device_get::2520:38",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe::device_get::2564:34",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_step6_architecture_probe::device_get::2565:34",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_volume_outer_probe::block_until_ready::3855:12",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_volume_outer_probe::block_until_ready::3862:14",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_volume_outer_probe::block_until_ready::3863:15",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_volume_outer_probe::device_get::3865:34",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::evaluate_f3_b37_volume_outer_probe::device_get::3919:35",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::nested_ls_outer_value_and_grad::block_until_ready::4910:12",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::nested_ls_outer_value_and_grad::block_until_ready::4917:14",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::nested_ls_outer_value_and_grad::block_until_ready::4918:20",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::nested_ls_outer_value_and_grad::device_get::4921:34",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::nested_ls_outer_value_and_grad::device_get::4943:23",
        "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py::prepare_f3_b37_outer_state::device_get::4659:24",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_ensure_traceable_runtime_host_wrappers.compute_baseline_value_and_grad::transfer_guard_device_to_host::5012:17",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_ensure_traceable_runtime_host_wrappers.compute_baseline_value_and_grad::transfer_guard_host_to_device::5000:17",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_make_traceable_lazy_host_reporting_metrics._baseline_reporting_metrics::transfer_guard_device_to_host::4945:17",
    }
)


def _iter_python_files():
    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            yield path, relative


class _BoundaryCallCensus(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.scope_names: list[str] = []
        self.scope_kinds: list[str] = ["module"]
        self.bindings: list[dict[str, str]] = [{}]
        self.calls: list[str] = []

    def _enter_scope(self, name: str, kind: str) -> None:
        self.scope_names.append(name)
        self.scope_kinds.append(kind)
        self.bindings.append({})

    def _leave_scope(self) -> None:
        self.scope_names.pop()
        self.scope_kinds.pop()
        self.bindings.pop()

    def _scope_name(self) -> str:
        return ".".join(self.scope_names) or "<module>"

    def _lookup_binding(self, name: str) -> str | None:
        current_scope = len(self.bindings) - 1
        for index in range(current_scope, -1, -1):
            if index != current_scope and self.scope_kinds[index] == "class":
                continue
            binding = self.bindings[index].get(name)
            if binding is not None:
                return binding
        return None

    def _bind_name(self, name: str, binding: str | None) -> None:
        if binding is None:
            self.bindings[-1].pop(name, None)
        else:
            self.bindings[-1][name] = binding

    def _binding_from_expression(self, expression: ast.expr) -> str | None:
        if isinstance(expression, ast.Name):
            return self._lookup_binding(expression.id)
        if (
            isinstance(expression, ast.Attribute)
            and expression.attr in _BOUNDARY_CALL_NAMES
            and isinstance(expression.value, ast.Name)
            and self._lookup_binding(expression.value.id) == _JAX_MODULE_BINDING
        ):
            return expression.attr
        return None

    def _bind_assignment_target(self, target: ast.expr, binding: str | None) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, binding or _NON_JAX_BINDING)

    def _boundary_call_name(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            binding = self._lookup_binding(node.func.id)
            return binding if binding in _BOUNDARY_CALL_NAMES else None
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _BOUNDARY_CALL_NAMES
            and isinstance(node.func.value, ast.Name)
            and self._lookup_binding(node.func.value.id) == _JAX_MODULE_BINDING
        ):
            return node.func.attr
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            if alias.name == "jax" or (
                alias.name.startswith("jax.") and alias.asname is None
            ):
                self._bind_name(bound_name, _JAX_MODULE_BINDING)
            else:
                self._bind_name(bound_name, _NON_JAX_BINDING)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name
            if node.module == "jax" and alias.name in _BOUNDARY_CALL_NAMES:
                self._bind_name(bound_name, alias.name)
            elif alias.name in _BOUNDARY_CALL_NAMES:
                self._bind_name(bound_name, _NON_JAX_BINDING)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        binding = self._binding_from_expression(node.value)
        for target in node.targets:
            self._bind_assignment_target(target, binding)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._bind_assignment_target(
                node.target, self._binding_from_expression(node.value)
            )

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        self._bind_assignment_target(node.target, None)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._enter_scope(node.name, "class")
        for statement in node.body:
            self.visit(statement)
        self._leave_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self._enter_scope(node.name, "function")
        for statement in node.body:
            self.visit(statement)
        self._leave_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._boundary_call_name(node)
        if call_name is not None:
            self.calls.append(
                f"{self.relative_path}::{self._scope_name()}::{call_name}"
                f"::{node.lineno}:{node.col_offset}"
            )
        self.generic_visit(node)


def _direct_boundary_calls() -> set[str]:
    calls: set[str] = set()
    for path, relative in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        census = _BoundaryCallCensus(relative)
        census.visit(tree)
        calls.update(census.calls)
    return calls


def _census_source(source: str) -> list[str]:
    tree = ast.parse(source)
    census = _BoundaryCallCensus("src/example.py")
    census.visit(tree)
    return census.calls


def test_only_boundary_owners_call_jax_transfer_and_readiness_primitives() -> None:
    actual = _direct_boundary_calls()
    unexpected = sorted(actual - _ALLOWED_OWNER_CALLS)
    stale = sorted(_ALLOWED_OWNER_CALLS - actual)

    assert not unexpected, (
        "Direct JAX boundary calls outside owners:\n  " + "\n  ".join(unexpected)
    )
    assert not stale, "Stale owner-call allowlist entries:\n  " + "\n  ".join(stale)


def test_boundary_census_distinguishes_duplicate_invocations_in_one_function() -> None:
    calls = _census_source(
        "import jax\n\n"
        "def owner(left, right):\n"
        "    return jax.device_get(left), jax.device_get(right)\n"
    )

    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert all("::owner::device_get::" in call for call in calls)


def test_boundary_census_resolves_function_local_jax_import() -> None:
    calls = _census_source(
        "def owner(value):\n    import jax\n    return jax.device_get(value)\n"
    )

    assert len(calls) == 1
    assert "::owner::device_get::" in calls[0]


def test_boundary_census_resolves_from_and_assignment_aliases() -> None:
    calls = _census_source(
        "from jax import device_get as imported_get\n"
        "import jax\n"
        "direct_get = jax.device_get\n"
        "runtime = jax\n\n"
        "def owner(value):\n"
        "    from jax import device_get as local_get\n"
        "    return (\n"
        "        imported_get(value),\n"
        "        direct_get(value),\n"
        "        runtime.device_get(value),\n"
        "        local_get(value),\n"
        "    )\n"
    )

    assert len(calls) == 4
    assert all("::owner::device_get::" in call for call in calls)


def test_boundary_census_qualifies_same_named_methods_by_class() -> None:
    calls = _census_source(
        "import jax\n\n"
        "class First:\n"
        "    def owner(self, value):\n"
        "        return jax.device_get(value)\n\n"
        "class Second:\n"
        "    def owner(self, value):\n"
        "        return jax.device_get(value)\n"
    )

    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert any("::First.owner::device_get::" in call for call in calls)
    assert any("::Second.owner::device_get::" in call for call in calls)


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
