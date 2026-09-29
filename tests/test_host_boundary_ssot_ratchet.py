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
# This baseline contains 33 direct invocations (2026-09-28); record every later change
# as a dated note here. The admission notes before the 2026-09-28 carve-out (the
# baseline peaked at 89 before the removed research modules took 57 sites with them)
# are in git history. Since then:
# Re-pinned 2026-09-28, no new site: the manual-LS operand fix in
# ``boozer_surface.py`` moved its two host-bridge guard coordinates, and the
# frozen-grid host-boundary fix in ``surface_objectives_traceable.py`` moved its
# three baseline-wrapper coordinates; file, scope and primitive are unchanged.
# Baseline: 32.
# Admitted 2026-09-28, ONE new owner-internal site: the official tiny
# least-squares policy moved from ``examples/jax/`` (not swept) into
# ``src/simsopt_jax/examples/``, and its 18 direct transfers now route through
# owners instead of being admitted as call sites: placement uses
# ``dtypes.explicit_device_array``, reads use ``host_boundary.host_array``, and
# its permissive scopes use the new ``host_boundary.allow_host_transfers``,
# whose single internal ``transfer_guard`` is the one addition. The two later
# ``host_boundary.py`` entries are re-pins after that insertion. Baseline: 33.
_ALLOWED_OWNER_CALLS = frozenset(
    {
        "src/simsopt_jax/backend/dtypes.py::_device_put::device_put::308:19",
        "src/simsopt_jax/backend/dtypes.py::_device_put::device_put::309:15",
        "src/simsopt_jax/backend/dtypes.py::_device_put::device_put::310:11",
        "src/simsopt_jax/backend/dtypes.py::runtime_device_put_tree::device_put::336:15",
        "src/simsopt_jax/backend/dtypes.py::runtime_device_put_tree::device_put::337:11",
        "src/simsopt_jax/core/sharding.py::_place_leading_axis_arrays::transfer_guard_device_to_device::255:13",
        "src/simsopt_jax/core/sharding.py::_place_leading_axis_arrays::transfer_guard_host_to_device::254:9",
        "src/simsopt_jax/core/sharding.py::replicate_tree_on_mesh::transfer_guard_device_to_device::249:13",
        "src/simsopt_jax/core/sharding.py::replicate_tree_on_mesh::transfer_guard_host_to_device::248:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_hager_higham_inverse_1_norm_estimate::transfer_guard_host_to_device::1338:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_run_operator_gmres::transfer_guard_host_to_device::686:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_run_operator_gmres_counted_incremental::transfer_guard_host_to_device::968:9",
        "src/simsopt_jax/geo/optimizers/optimizer.py::_gmres_solve_least_squares_system::transfer_guard_host_to_device::2736:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_scipy_host_array::transfer_guard_device_to_host::231:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_array_from_scipy_host::transfer_guard_host_to_device::248:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_scipy_host_extension_scope::transfer_guard_device_to_host::122:13",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_scipy_host_extension_scope::transfer_guard_host_to_device::121:9",
        "src/simsopt_jax/runtime/host_boundary.py::allow_host_transfers::transfer_guard::148:9",
        "src/simsopt_jax/runtime/host_boundary.py::block_until_ready::block_until_ready::239:11",
        "src/simsopt_jax/runtime/host_boundary.py::disallow_host_transfers::transfer_guard::134:9",
        "src/simsopt_jax/runtime/host_boundary.py::host_value::device_get::204:11",
        "src/simsopt_jax/solve/dispatch.py::_run_optimistix_lm::transfer_guard_host_to_device::1039:9",
        "src/simsopt_jax/solve/dispatch.py::_run_scipy_minimize.value_and_gradient_at::device_get::768:15",
        "src/simsopt_jax/solve/dispatch.py::_run_scipy_minimize.value_and_gradient_at::device_get::768:38",
        "src/simsopt_jax/solve/dispatch.py::_run_scipy_minimize.value_and_gradient_at::device_put::766:54",
        "src/simsopt_jax/solve/dispatch.py::_run_scipy_minimize::device_get::819:25",
        "src/simsopt_jax/solve/minimize_runtime.py::run_optimistix_minimize::transfer_guard_host_to_device::301:9",
        "src/simsopt_jax/solve/serial.py::_write_bounded_objective_log::transfer_guard::455:9",
        "src/simsopt_jax_adapters/geo/boozer_surface.py::_with_host_bridge_transfer_guard.wrapped::transfer_guard_device_to_host::214:13",
        "src/simsopt_jax_adapters/geo/boozer_surface.py::_with_host_bridge_transfer_guard.wrapped::transfer_guard_host_to_device::215:17",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_ensure_traceable_runtime_host_wrappers.compute_baseline_value_and_grad::transfer_guard_device_to_host::5090:17",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_ensure_traceable_runtime_host_wrappers.compute_baseline_value_and_grad::transfer_guard_host_to_device::5078:17",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_make_traceable_lazy_host_reporting_metrics._baseline_reporting_metrics::transfer_guard_device_to_host::5023:17",
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


# ``host_boundary.allow_host_transfers`` lifts an outer strict guard for its whole
# block, so every caller is admitted explicitly: file -> number of call sites.
# Admitted 2026-09-28: the official tiny least-squares policy's five host-driven
# scopes (SciPy's residual callback, the endpoint Jacobian, three residual builders).
_ALLOWED_ALLOW_HOST_TRANSFERS_CALLS = {
    "src/simsopt_jax/examples/official_tiny_least_squares.py": 5,
}
_ALLOW_HOST_TRANSFERS_ROOTS = (*SOURCE_ROOTS, REPO_ROOT / "examples/jax")
_ALLOW_HOST_TRANSFERS_OWNER = "src/simsopt_jax/runtime/host_boundary.py"


def _allow_host_transfers_calls() -> dict[str, int]:
    counts: dict[str, int] = {}
    for source_root in _ALLOW_HOST_TRANSFERS_ROOTS:
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative == _ALLOW_HOST_TRANSFERS_OWNER:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        assert not (
                            alias.name == "allow_host_transfers" and alias.asname
                        ), f"{relative} imports allow_host_transfers under an alias"
                if isinstance(node, ast.Call) and (
                    (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "allow_host_transfers"
                    )
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "allow_host_transfers"
                    )
                ):
                    counts[relative] = counts.get(relative, 0) + 1
    return counts


def test_only_admitted_callers_lift_the_strict_transfer_guard() -> None:
    assert _allow_host_transfers_calls() == _ALLOWED_ALLOW_HOST_TRANSFERS_CALLS


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
