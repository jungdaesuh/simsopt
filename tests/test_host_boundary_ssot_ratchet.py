"""Full-census ratchet for the JAX host/device boundary owners.

The census is static analysis over the source text. It resolves the owner
module and ``allow_host_transfers`` through static imports, aliases, attribute
chains, ``getattr``/``hasattr`` with a constant name, ``sys.modules`` keys and
``importlib.import_module``/``__import__`` arguments. It cannot see a lookup
whose name is computed at run time on an object it does not recognise as the
owner module; those forms are outside what an AST can decide, and the
repository's rule against dynamic imports covers them.
"""

from __future__ import annotations

import ast
import hashlib
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
        "src/simsopt_jax/geo/optimizers/optimizer.py::_gmres_solve_least_squares_system::transfer_guard_host_to_device::2734:9",
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
# block, so every permitted region is admitted individually. The only admitted form
# is a ``with allow_host_transfers():`` item (or ``with host_boundary.allow_host_
# transfers():``). Each admitted site is keyed by its enclosing scope
# (``path::qualified.name``) and pinned by a fingerprint of the whole ``with``
# statement -- items and body, serialized by ``_stable_ast_dump`` -- together with
# its statement path inside that scope (``body[1]``, ``body[0].orelse[2]``, ...).
# Moving a region to another scope, moving it within its scope, or editing
# anything inside it therefore fails until the allowlist is updated on purpose.
#
# Escapes: a bare call, a decorator, a call inside a lambda or comprehension, a
# call used as a value, an aliased import, ``permit = allow_host_transfers``,
# passing the name as a value; the owner module object (bound by ``from
# simsopt_jax.runtime import host_boundary``, ``import simsopt_jax.runtime.
# host_boundary as hb``, the dotted import, or ``runtime.host_boundary`` after
# ``from simsopt_jax import runtime``) used as anything but the base of an
# attribute access by a literal, non-dunder name (``getattr(hb, name)``,
# ``vars(hb)``, ``hb.__dict__``, the module passed as a value); and these lookup
# forms: a ``getattr``/``hasattr`` name equal to ``"allow_host_transfers"`` or
# ``"host_boundary"``, a ``sys.modules[...]`` key or ``sys.modules.get(...)``
# argument, and an ``importlib.import_module(...)``/``__import__(...)`` argument
# that is not a constant or that names the owner module or its package. Strings
# elsewhere (docstrings, messages) are not lookups and are not escapes.
#
# Limit: this is static analysis. A lookup whose name is computed and whose target
# object is not recognisably the owner module -- ``getattr(obj, "allow_" +
# suffix)`` on an object reached through an unrecognised alias -- is not
# visible to it; the repository rule against dynamic imports and review cover
# that residue. Non-constant dynamic imports are escapes unless their scope is
# admitted in ``_ALLOWED_DYNAMIC_LOOKUP_SITES``.
#
# Regenerate a fingerprint with ``_allow_host_transfers_census(source, path)`` after
# reading the edited region.
_ALLOWED_ALLOW_HOST_TRANSFERS_SITES = {
    # Builds the curve spec and frozen DOF vector on device from host inputs.
    "src/simsopt_jax/examples/official_tiny_least_squares.py::curve_length_residual": (
        "a783309da54f",
    ),
    # Places the quadratic targets and weights on device.
    "src/simsopt_jax/examples/official_tiny_least_squares.py::quadratic_residual": (
        "3f827b920703",
    ),
    # SciPy's residual callback: places each host trial point, returns a host residual.
    "src/simsopt_jax/examples/official_tiny_least_squares.py::solve_jax_residual.host_residual": (
        "d9273088359e",
    ),
    # Builds the surface spec, frozen DOF vector and targets on device from host inputs.
    "src/simsopt_jax/examples/official_tiny_least_squares.py::surface_area_volume_residual": (
        "317b98758dbe",
    ),
    # Endpoint residual and exact Jacobian, read back to host float64.
    "src/simsopt_jax/examples/official_tiny_least_squares.py::value_and_jacobian": (
        "c97967c48d74",
    ),
}
# Scopes admitted to import a module by a non-constant name, with their count.
_ALLOWED_DYNAMIC_LOOKUP_SITES = {
    # Upstream serialization: imports a class's parent package to record its version.
    "src/simsopt/_core/json.py::GSONEncoder.default": 1,
    "src/simsopt/_core/json.py::GSONable.as_dict": 1,
    "src/simsopt/_core/json.py::SIMSON.as_dict": 1,
    # Upstream deserialization: imports the module a serialized document names.
    "src/simsopt/_core/json.py::GSONDecoder.process_decoded": 2,
}
_ALLOW_HOST_TRANSFERS_REQUIRED_ROOTS = (
    "src/simsopt",
    "src/simsopt_contracts",
    "src/simsopt_jax",
    "src/simsopt_jax_adapters",
    "examples",
)
_ALLOW_HOST_TRANSFERS_OPTIONAL_ROOTS = ("benchmarks",)
_ALLOW_HOST_TRANSFERS_OWNER = "src/simsopt_jax/runtime/host_boundary.py"
_ALLOW_HOST_TRANSFERS = "allow_host_transfers"
_OWNER_PACKAGE = "simsopt_jax.runtime"
_OWNER_MODULE_NAME = "host_boundary"
_OWNER_MODULE = f"{_OWNER_PACKAGE}.{_OWNER_MODULE_NAME}"
_LOOKUP_NAMES = frozenset({_ALLOW_HOST_TRANSFERS, _OWNER_MODULE_NAME})
_NAME_LOOKUP_BUILTINS = frozenset({"getattr", "hasattr"})
_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})
_SCOPE_NODES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_allow_host_transfers_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == _ALLOW_HOST_TRANSFERS)
        or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == _ALLOW_HOST_TRANSFERS
        )
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _statement_paths(tree: ast.Module) -> dict[int, str]:
    """Map each statement's id to its path of statement lists inside its scope."""
    paths: dict[int, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, child in enumerate(value):
                if not isinstance(child, ast.stmt):
                    continue
                path = f"{prefix}{field}[{index}]"
                paths[id(child)] = path
                walk(child, "" if isinstance(child, _SCOPE_NODES) else f"{path}.")

    walk(tree, "")
    return paths


def _stable_ast_dump(value: object) -> str:
    """Serialize an AST in ``_fields`` order without positions or empty fields.

    Empty-list and ``None`` fields are omitted, the canonical form Python 3.13's
    ``ast.dump`` uses, so a field a newer grammar adds empty (``type_params=[]``
    on functions since 3.12) leaves the serialization unchanged.
    """
    if isinstance(value, ast.AST):
        fields = ", ".join(
            f"{field}={_stable_ast_dump(getattr(value, field))}"
            for field in value._fields
            if getattr(value, field) not in ([], None)
        )
        return f"{type(value).__name__}({fields})"
    if isinstance(value, list):
        return "[" + ", ".join(_stable_ast_dump(item) for item in value) + "]"
    return repr(value)


def _with_fingerprint(node: ast.With | ast.AsyncWith, path: str) -> str:
    normalized = f"{path}\n{_stable_ast_dump(node)}"
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def _module_package(relative_path: str) -> tuple[str, ...]:
    parts = Path(relative_path).with_suffix("").parts
    return (parts[1:] if parts[:1] == ("src",) else parts)[:-1]


def _import_from_module(node: ast.ImportFrom, relative_path: str) -> str:
    if node.level == 0:
        return node.module or ""
    package = _module_package(relative_path)
    base = package[: len(package) - (node.level - 1)]
    return ".".join((*base, *((node.module,) if node.module else ())))


def _owner_bindings(
    tree: ast.Module, relative_path: str
) -> tuple[frozenset[str], frozenset[str]]:
    """Names bound to the owner module and to its package by the file's imports."""
    owner: set[str] = set()
    package: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None and alias.name == _OWNER_MODULE:
                    owner.add(alias.asname)
                elif alias.asname is not None and alias.name == _OWNER_PACKAGE:
                    package.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            module = _import_from_module(node, relative_path)
            for alias in node.names:
                bound = alias.asname or alias.name
                if module == _OWNER_PACKAGE and alias.name == _OWNER_MODULE_NAME:
                    owner.add(bound)
                elif f"{module}.{alias.name}" == _OWNER_PACKAGE:
                    package.add(bound)
    return frozenset(owner), frozenset(package)


def _literal_attribute_bases(tree: ast.Module) -> frozenset[int]:
    """Ids of expressions used as the base of a literal non-dunder attribute."""
    return frozenset(
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and not (node.attr.startswith("__") and node.attr.endswith("__"))
    )


def _names_owner_module(value: str) -> bool:
    return value in (_OWNER_MODULE, _OWNER_PACKAGE) or value.startswith(".")


def _first_argument(node: ast.Call, keyword: str) -> ast.expr | None:
    if node.args:
        return node.args[0]
    return next(
        (item.value for item in node.keywords if item.arg == keyword),
        None,
    )


class _AllowHostTransfersCensus(ast.NodeVisitor):
    def __init__(
        self,
        relative_path: str,
        statement_paths: dict[int, str],
        owner_bindings: tuple[frozenset[str], frozenset[str]],
        literal_attribute_bases: frozenset[int],
    ) -> None:
        self.relative_path = relative_path
        self.statement_paths = statement_paths
        self.owner_names, self.package_names = owner_bindings
        self.literal_attribute_bases = literal_attribute_bases
        self.scope_names: list[str] = []
        self.sites: dict[str, tuple[str, ...]] = {}
        self.escapes: list[str] = []
        self.dynamic_lookups: dict[str, int] = {}

    def _scope_key(self) -> str:
        scope = ".".join(self.scope_names) or "<module>"
        return f"{self.relative_path}::{scope}"

    def _escape(self, node: ast.AST, form: str) -> None:
        self.escapes.append(f"{self._scope_key()}::{node.lineno} {form}")

    def _dynamic_lookup(self) -> None:
        key = self._scope_key()
        self.dynamic_lookups[key] = self.dynamic_lookups.get(key, 0) + 1

    def _is_owner_module(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.owner_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr == _OWNER_MODULE_NAME
            and (
                (
                    isinstance(node.value, ast.Name)
                    and node.value.id in self.package_names
                )
                or _dotted_name(node) == _OWNER_MODULE
            )
        )

    def _check_module_key(self, key: ast.expr | None, node: ast.AST) -> None:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            self._dynamic_lookup()
        elif _names_owner_module(key.value):
            self._escape(node, "owner module lookup by name")

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scope_names.append(name)
        self.generic_visit(node)
        self.scope_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        admitted = False
        for item in node.items:
            if _is_allow_host_transfers_call(item.context_expr):
                admitted = True
                call = item.context_expr
                # The callee itself is the admitted reference; visit the rest.
                if isinstance(call.func, ast.Attribute):
                    self.visit(call.func.value)
                for argument in (*call.args, *call.keywords):
                    self.visit(argument)
            else:
                self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
        if admitted:
            key = self._scope_key()
            fingerprint = _with_fingerprint(node, self.statement_paths[id(node)])
            self.sites[key] = (*self.sites.get(key, ()), fingerprint)
        for statement in node.body:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == _ALLOW_HOST_TRANSFERS and alias.asname is not None:
                self._escape(node, "aliased import")

    def visit_Call(self, node: ast.Call) -> None:
        if _is_allow_host_transfers_call(node):
            self._escape(node, "call outside a with item")
            if isinstance(node.func, ast.Attribute):
                self.visit(node.func.value)
            for argument in (*node.args, *node.keywords):
                self.visit(argument)
            return
        callee = node.func
        if (
            isinstance(callee, ast.Name)
            and callee.id in _NAME_LOOKUP_BUILTINS
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in _LOOKUP_NAMES
        ):
            self._escape(node, f"{callee.id} of {node.args[1].value!r}")
        callee_name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr
            if isinstance(callee, ast.Attribute)
            else None
        )
        if callee_name in _DYNAMIC_IMPORTERS:
            self._check_module_key(_first_argument(node, "name"), node)
        if (
            isinstance(callee, ast.Attribute)
            and callee.attr == "get"
            and _dotted_name(callee.value) == "sys.modules"
        ):
            self._check_module_key(_first_argument(node, "key"), node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _dotted_name(node.value) == "sys.modules":
            self._check_module_key(node.slice, node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == _ALLOW_HOST_TRANSFERS:
            self._escape(node, "non-call reference")
        elif (
            self._is_owner_module(node) and id(node) not in self.literal_attribute_bases
        ):
            self._escape(node, "owner module reference")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == _ALLOW_HOST_TRANSFERS:
            self._escape(node, "non-call reference")
        elif (
            self._is_owner_module(node) and id(node) not in self.literal_attribute_bases
        ):
            self._escape(node, "owner module reference")
        self.generic_visit(node)


def _allow_host_transfers_census(
    source: str,
    relative_path: str,
    admitted_dynamic_lookups: dict[str, int] | None = None,
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    tree = ast.parse(source)
    census = _AllowHostTransfersCensus(
        relative_path,
        _statement_paths(tree),
        _owner_bindings(tree, relative_path),
        _literal_attribute_bases(tree),
    )
    census.visit(tree)
    admitted = {
        key: count
        for key, count in (admitted_dynamic_lookups or {}).items()
        if key.startswith(f"{relative_path}::")
    }
    dynamic_escapes = [
        f"{key} dynamic module lookup x{census.dynamic_lookups.get(key, 0)}"
        f" (admitted {admitted.get(key, 0)})"
        for key in sorted({*census.dynamic_lookups, *admitted})
        if census.dynamic_lookups.get(key, 0) != admitted.get(key, 0)
    ]
    return census.sites, [*census.escapes, *dynamic_escapes]


def _allow_host_transfers_roots(repo_root: Path) -> tuple[Path, ...]:
    required = tuple(repo_root / root for root in _ALLOW_HOST_TRANSFERS_REQUIRED_ROOTS)
    missing = [root for root in required if not root.is_dir()]
    assert not missing, f"allow_host_transfers census roots are missing: {missing}"
    optional = tuple(
        repo_root / root
        for root in _ALLOW_HOST_TRANSFERS_OPTIONAL_ROOTS
        if (repo_root / root).is_dir()
    )
    return (*required, *optional)


def _allow_host_transfers_sites(
    repo_root: Path,
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    sites: dict[str, tuple[str, ...]] = {}
    escapes: list[str] = []
    relatives = sorted(
        {
            path.relative_to(repo_root).as_posix()
            for root in _allow_host_transfers_roots(repo_root)
            for path in root.rglob("*.py")
        }
        - {_ALLOW_HOST_TRANSFERS_OWNER}
    )
    for relative in relatives:
        file_sites, file_escapes = _allow_host_transfers_census(
            (repo_root / relative).read_text(), relative, _ALLOWED_DYNAMIC_LOOKUP_SITES
        )
        sites.update(file_sites)
        escapes.extend(file_escapes)
    return sites, escapes


def test_only_admitted_regions_lift_the_strict_transfer_guard() -> None:
    sites, escapes = _allow_host_transfers_sites(REPO_ROOT)

    assert not escapes, "allow_host_transfers escapes the with-site census:\n  " + (
        "\n  ".join(escapes)
    )
    assert sites == _ALLOWED_ALLOW_HOST_TRANSFERS_SITES
    assert all(
        (REPO_ROOT / key.split("::", maxsplit=1)[0]).is_file()
        for key in _ALLOWED_DYNAMIC_LOOKUP_SITES
    )


_ADMITTED_SOURCE = (
    "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
    "def approved():\n"
    "    prepare()\n"
    "    with allow_host_transfers():\n"
    "        transfer()\n\n"
    "def other():\n"
    "    pass\n"
)


def test_allow_host_transfers_census_admits_only_with_items() -> None:
    sites, escapes = _allow_host_transfers_census(_ADMITTED_SOURCE, "src/example.py")
    attribute_sites, attribute_escapes = _allow_host_transfers_census(
        "from simsopt_jax.runtime import host_boundary\n\n"
        "def approved():\n"
        "    async def inner():\n"
        "        async with host_boundary.allow_host_transfers():\n"
        "            transfer()\n",
        "src/example.py",
    )

    assert list(sites) == ["src/example.py::approved"]
    assert len(sites["src/example.py::approved"]) == 1
    assert escapes == []
    assert list(attribute_sites) == ["src/example.py::approved.inner"]
    assert attribute_escapes == []
    module_imports = {
        "aliased module import": (
            "import simsopt_jax.runtime.host_boundary as hb\n\n"
            "def approved():\n"
            "    with hb.allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "dotted module import": (
            "import simsopt_jax.runtime.host_boundary\n\n"
            "def approved():\n"
            "    with simsopt_jax.runtime.host_boundary.allow_host_transfers():\n"
            "        transfer()\n"
        ),
    }
    for form, source in module_imports.items():
        module_sites, module_escapes = _allow_host_transfers_census(
            source, "src/example.py"
        )
        assert list(module_sites) == ["src/example.py::approved"], form
        assert module_escapes == [], form


def test_allow_host_transfers_census_refuses_a_moved_or_edited_region() -> None:
    admitted, _ = _allow_host_transfers_census(_ADMITTED_SOURCE, "src/example.py")
    mutations = {
        "moved to another scope": _ADMITTED_SOURCE.replace(
            "    prepare()\n    with allow_host_transfers():\n        transfer()\n\n"
            "def other():\n    pass\n",
            "    prepare()\n\n"
            "def other():\n    with allow_host_transfers():\n        transfer()\n",
        ),
        "moved within its scope": _ADMITTED_SOURCE.replace(
            "    prepare()\n    with allow_host_transfers():\n        transfer()\n",
            "    with allow_host_transfers():\n        transfer()\n    prepare()\n",
        ),
        "body edited": _ADMITTED_SOURCE.replace(
            "        transfer()\n", "        transfer()\n        transfer_more()\n"
        ),
        "second region added": _ADMITTED_SOURCE.replace(
            "        transfer()\n",
            "        transfer()\n    with allow_host_transfers():\n        more()\n",
        ),
    }
    for form, source in mutations.items():
        assert source != _ADMITTED_SOURCE, f"{form} mutation did not apply"
        mutated, escapes = _allow_host_transfers_census(source, "src/example.py")
        assert escapes == [], form
        assert mutated != admitted, f"{form} kept the admitted fingerprint"


def test_allow_host_transfers_census_refuses_every_other_reference() -> None:
    escaping_sources = {
        "aliased import": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers as permit\n\n"
            "def approved():\n"
            "    with permit():\n"
            "        pass\n"
        ),
        "name alias": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    permit = allow_host_transfers\n"
            "    with permit():\n"
            "        pass\n"
        ),
        "attribute alias": (
            "from simsopt_jax.runtime import host_boundary\n\n"
            "def approved():\n"
            "    permit = host_boundary.allow_host_transfers\n"
            "    with permit():\n"
            "        pass\n"
        ),
        "passed as a value": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved(run):\n"
            "    run(allow_host_transfers)\n"
        ),
        "bare call": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    allow_host_transfers()\n"
            "    transfer()\n"
        ),
        "call used as a value": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    permit = allow_host_transfers()\n"
            "    with permit:\n"
            "        transfer()\n"
        ),
        "decorator": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    @allow_host_transfers()\n"
            "    def inner():\n"
            "        transfer()\n"
            "    inner()\n"
        ),
        "lambda": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    permit = lambda: allow_host_transfers()\n"
            "    with permit():\n"
            "        transfer()\n"
        ),
        "comprehension": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    permits = [allow_host_transfers() for _ in range(1)]\n"
            "    with permits[0]:\n"
            "        transfer()\n"
        ),
        "conditional with item": (
            "from contextlib import nullcontext\n"
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved(lift):\n"
            "    with allow_host_transfers() if lift else nullcontext():\n"
            "        transfer()\n"
        ),
        "getattr with a string": (
            "from simsopt_jax.runtime import host_boundary\n\n"
            "def approved():\n"
            '    with getattr(host_boundary, "allow_host_transfers")():\n'
            "        transfer()\n"
        ),
        "getattr with a concatenated string": (
            "from simsopt_jax.runtime import host_boundary\n\n"
            "def approved():\n"
            '    with getattr(host_boundary, "allow_" + "host_transfers")():\n'
            "        transfer()\n"
        ),
        "getattr on an aliased module import": (
            "import simsopt_jax.runtime.host_boundary as hb\n\n"
            "def approved(name):\n"
            "    with getattr(hb, name)():\n"
            "        transfer()\n"
        ),
        "getattr on a dotted module import": (
            "import simsopt_jax.runtime.host_boundary\n\n"
            "def approved(name):\n"
            "    with getattr(simsopt_jax.runtime.host_boundary, name)():\n"
            "        transfer()\n"
        ),
        "vars of the module": (
            "from simsopt_jax.runtime import host_boundary as hb\n\n"
            "def approved(name):\n"
            "    with vars(hb)[name]():\n"
            "        transfer()\n"
        ),
        "module __dict__": (
            "from simsopt_jax.runtime import host_boundary\n\n"
            "def approved(name):\n"
            "    with host_boundary.__dict__[name]():\n"
            "        transfer()\n"
        ),
        "module passed as a value": (
            "from simsopt_jax.runtime import host_boundary\n\n"
            "def approved(run):\n"
            "    run(host_boundary)\n"
        ),
        "module reached through its package": (
            "from simsopt_jax import runtime\n\n"
            "def approved(run):\n"
            "    run(runtime.host_boundary)\n"
        ),
        "computed sys.modules key": (
            "import sys\n\n"
            "def approved(name):\n"
            "    with getattr(sys.modules[name], 'allow_' + 'host_transfers')():\n"
            "        transfer()\n"
        ),
        "computed sys.modules.get key": (
            "import sys\n\n"
            "def approved(name, attribute):\n"
            "    with getattr(sys.modules.get(name), attribute)():\n"
            "        transfer()\n"
        ),
        "computed import_module name": (
            "import importlib\n\n"
            "def approved(name, attribute):\n"
            "    with getattr(importlib.import_module(name), attribute)():\n"
            "        transfer()\n"
        ),
        "import_module of the owner": (
            "from importlib import import_module\n\n"
            "def approved(attribute):\n"
            '    module = import_module("simsopt_jax.runtime.host_boundary")\n'
            "    with getattr(module, attribute)():\n"
            "        transfer()\n"
        ),
        "hasattr probe of the permit": (
            "def approved(module):\n"
            '    return hasattr(module, "allow_host_transfers")\n'
        ),
        "relative import of the owner": (
            "from . import host_boundary\n\n"
            "def approved(run):\n"
            "    run(host_boundary)\n"
        ),
        "module named by a string": (
            "import sys\n\n"
            "def approved(name):\n"
            '    with getattr(sys.modules["simsopt_jax.runtime.host_boundary"], name)():\n'
            "        transfer()\n"
        ),
    }
    for form, source in escaping_sources.items():
        _, escapes = _allow_host_transfers_census(
            source, "src/simsopt_jax/runtime/example.py"
        )
        assert escapes, f"{form} escaped the allow_host_transfers census"


def test_allow_host_transfers_census_ignores_unrelated_names_and_plain_strings() -> (
    None
):
    benign_sources = {
        "unrelated host_boundary module": (
            "import other_package.host_boundary\n"
            "from other_package import host_boundary as other\n\n"
            "def unrelated(run):\n"
            "    run(other_package.host_boundary)\n"
            "    run(other)\n"
        ),
        "docstring equal to the module name": (
            'def unrelated():\n    """host_boundary"""\n    return "allow_host_transfers"\n'
        ),
        "constant sys.modules key": (
            "import sys\n\n"
            "def unrelated():\n"
            '    return sys.modules["jax"], sys.modules.get("jax")\n'
        ),
        "literal attribute of the owner module": (
            "from simsopt_jax.runtime import host_boundary\n\n"
            "def approved(value):\n"
            "    return host_boundary.host_array(value)\n"
        ),
    }
    for form, source in benign_sources.items():
        sites, escapes = _allow_host_transfers_census(source, "src/example.py")
        assert (sites, escapes) == ({}, []), f"{form} was flagged: {escapes}"


def test_allow_host_transfers_census_admits_only_listed_dynamic_lookups() -> None:
    source = (
        "import importlib\n\n"
        "def load(name):\n"
        "    return importlib.import_module(name)\n"
    )
    admitted = {"src/example.py::load": 1}

    _, unlisted = _allow_host_transfers_census(source, "src/example.py")
    _, listed = _allow_host_transfers_census(source, "src/example.py", admitted)
    _, stale = _allow_host_transfers_census(
        "def load(name):\n    return name\n", "src/example.py", admitted
    )

    assert unlisted == ["src/example.py::load dynamic module lookup x1 (admitted 0)"]
    assert listed == []
    assert stale == ["src/example.py::load dynamic module lookup x0 (admitted 1)"]


def test_allow_host_transfers_census_scans_every_production_root(
    tmp_path: Path,
) -> None:
    for root in _ALLOW_HOST_TRANSFERS_REQUIRED_ROOTS:
        (tmp_path / root).mkdir(parents=True)
    injected = {
        "src/simsopt/field/injected.py": _ADMITTED_SOURCE,
        "src/simsopt_contracts/injected.py": _ADMITTED_SOURCE,
        "examples/2_Intermediate/injected.py": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def run():\n"
            "    allow_host_transfers()\n"
        ),
        "benchmarks/injected.py": _ADMITTED_SOURCE,
        _ALLOW_HOST_TRANSFERS_OWNER: (
            "def allow_host_transfers():\n    allow_host_transfers()\n"
        ),
    }
    for relative, source in injected.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    sites, escapes = _allow_host_transfers_sites(tmp_path)

    assert set(sites) == {
        "benchmarks/injected.py::approved",
        "src/simsopt/field/injected.py::approved",
        "src/simsopt_contracts/injected.py::approved",
    }
    assert [escape.split("::")[0] for escape in escapes] == [
        "examples/2_Intermediate/injected.py"
    ]


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
