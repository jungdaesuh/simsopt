"""Full-census ratchet for the JAX host/device boundary owners.

The ``allow_host_transfers`` census is static analysis over the source of
``src/simsopt``, ``src/simsopt_contracts``, ``src/simsopt_jax``,
``src/simsopt_jax_adapters``, ``examples`` and ``benchmarks`` (when present),
the owner excluded. It admits only ``with allow_host_transfers():`` items, each
pinned by scope and by a fingerprint of the ``with`` statement's syntax tree
and statement path (comments and formatting do not change it). An admitted
item's callee must resolve to the owner in its lexical scope chain: the
function, enclosing functions, then the module (a class body only when the
site sits directly in it). ``global`` and ``nonlocal`` declarations in
functions choose the scope a name resolves in, and a function-level write to
a declared name counts as a binding of the scope it rebinds. The innermost
scope binding the name must bind it only by an import of the owner; a
parameter, assignment, loop, ``with``/``except``/``match`` target, ``:=``
target, other import or nested ``def``/``class`` of that name there is an
escape. The census also resolves the owner module through static imports,
aliases and attribute chains, and checks ``getattr``/``hasattr`` constant
names, ``sys.modules`` keys and ``import_module``/``__import__`` arguments;
non-constant dynamic imports must be pinned in
``_ALLOWED_DYNAMIC_LOOKUP_SITES``.

Two mechanisms are at work, and their limits differ. The visitor flags names
wherever they appear: every ``Name`` or attribute named
``allow_host_transfers`` except the callee of a ``with`` item (so a ``:=``
target, an assignment or a reference is flagged), and every use of a name
imported as the owner module other than as the base of a literal attribute.
Binding resolution decides whether an admitted callee, or the base of an
admitted attribute callee, is the owner.

Known limits that can let a change through unseen (false negatives):

* Binding resolution does not read the headers of nested definitions:
  decorators, default values and annotations are evaluated in the enclosing
  scope, but a ``:=`` there is not collected as a binding of it. A default
  such as ``def f(x=(runtime := substitute))`` rebinds an owner-package alias
  (``from simsopt_jax import runtime``) or the ``simsopt_jax`` root of a
  dotted import unseen; the visitor catches the same form only for the
  permit's name and for names imported as the owner module itself.
* Binding resolution collects ``global`` writes from functions only. A class
  body that declares a name ``global`` and rebinds it -- by an import, or by
  assignment to an owner-package alias or the ``simsopt_jax`` root -- changes
  the module binding an admitted site resolves to, unseen.
* A lookup whose name is computed at run time on an object the census does
  not recognise as the owner module (reached through a call's return value,
  say), code run from strings by ``exec``/``eval``/``compile``, and writes
  through ``globals()``/``locals()``/``vars()`` of a non-owner namespace
  (``globals()["allow_host_transfers"] = ...``). The repository's rule against
  dynamic imports and review cover these.
* A fingerprint pins the syntax tree of an admitted region, not the behaviour
  of the functions it calls.
* Only the roots above are scanned; ``tests/``, ``scripts/`` and ``docs/`` are
  not.

Known limits that flag benign code (false positives, fail-closed; the author
renames the code or updates an allowlist on purpose):

* Names, not bindings, identify the permit in the visitor: any ``Name`` or
  attribute named ``allow_host_transfers`` outside the owner, other than an
  admitted ``with`` callee, is flagged. (A ``def``/``class`` of that name that
  is never referenced is not flagged, since its name is not such a node.)
* The owner-module reference check collects bindings per file, so a local
  variable that shadows an owner import elsewhere in the file and is used as
  a value is flagged.
* ``getattr``/``hasattr`` with the constant name ``"host_boundary"`` or
  ``"allow_host_transfers"`` is flagged whatever the target object is.
* Any wildcard import in a file with admitted sites is flagged, whether or not
  it can rebind the permit.
* A ``nonlocal`` write in a nested function counts as a binding of every
  enclosing function, not only the nearest one that binds the name. So an
  admitted site in ``outer`` that uses the imported owner is rejected when an
  ``inner`` function rebinds, through ``nonlocal``, a same-named variable
  that belongs to an intermediate ``middle`` function.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import NamedTuple

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
# Removed 2026-09-29, no new site (Tier-2 port of banana-trim): removing the
# Optax and Optimistix backends took ``dispatch._run_optimistix_lm`` and
# ``minimize_runtime.run_optimistix_minimize`` with them, and keeping one
# Levenberg-Marquardt took ``optimizer._gmres_solve_least_squares_system``; the
# other coordinates that range moved are re-pins after deletions above them
# (file, scope and primitive unchanged). Then the lm-minpack traced-objective
# refusal in ``surface_objectives_traceable.py`` moved its three
# baseline-wrapper coordinates. Baseline: 30.
# Re-pinned 2026-09-29, no new site: carrying lm-minpack's Jacobian count in
# ``dispatch._public_result`` moved the four ``_run_scipy_minimize``
# coordinates below it. Baseline: 30.
# Removed 2026-09-30, no new site: the unplaced-value rule moved the runtime
# placement of ``dtypes._device_put`` and ``dtypes.runtime_device_put_tree``
# into their shared owner ``dtypes._unplaced_device_put`` (two sites there),
# leaving one explicit-placement site in each caller. Baseline: 29.
# Admitted 2026-09-30, ONE new owner-internal site: ``_unplaced_device_put``
# now places per leaf (a leaf left uncommitted on another device by an exited
# ``jax.default_device`` scope is moved to the runtime device), a third
# ``device_put`` in the same owner. ``dtypes.commit_in_place`` routes through
# ``_device_put`` and adds none. Baseline: 30.
_ALLOWED_OWNER_CALLS = frozenset(
    {
        "src/simsopt_jax/backend/dtypes.py::_device_put::device_put::369:11",
        "src/simsopt_jax/backend/dtypes.py::_unplaced_device_put::device_put::332:15",
        "src/simsopt_jax/backend/dtypes.py::_unplaced_device_put::device_put::337:15",
        "src/simsopt_jax/backend/dtypes.py::_unplaced_device_put::device_put::343:39",
        "src/simsopt_jax/backend/dtypes.py::runtime_device_put_tree::device_put::394:11",
        "src/simsopt_jax/core/sharding.py::_place_leading_axis_arrays::transfer_guard_device_to_device::255:13",
        "src/simsopt_jax/core/sharding.py::_place_leading_axis_arrays::transfer_guard_host_to_device::254:9",
        "src/simsopt_jax/core/sharding.py::replicate_tree_on_mesh::transfer_guard_device_to_device::249:13",
        "src/simsopt_jax/core/sharding.py::replicate_tree_on_mesh::transfer_guard_host_to_device::248:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_hager_higham_inverse_1_norm_estimate::transfer_guard_host_to_device::1333:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_run_operator_gmres::transfer_guard_host_to_device::681:9",
        "src/simsopt_jax/geo/optimizers/linear_solve.py::_run_operator_gmres_counted_incremental::transfer_guard_host_to_device::963:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_scipy_host_array::transfer_guard_device_to_host::230:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_array_from_scipy_host::transfer_guard_host_to_device::247:9",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_scipy_host_extension_scope::transfer_guard_device_to_host::121:13",
        "src/simsopt_jax/geo/optimizers/reference.py::_target_scipy_host_extension_scope::transfer_guard_host_to_device::120:9",
        "src/simsopt_jax/runtime/host_boundary.py::allow_host_transfers::transfer_guard::143:9",
        "src/simsopt_jax/runtime/host_boundary.py::block_until_ready::block_until_ready::234:11",
        "src/simsopt_jax/runtime/host_boundary.py::disallow_host_transfers::transfer_guard::129:9",
        "src/simsopt_jax/runtime/host_boundary.py::host_value::device_get::199:11",
        "src/simsopt_jax/solve/dispatch.py::_run_scipy_minimize.value_and_gradient_at::device_get::714:15",
        "src/simsopt_jax/solve/dispatch.py::_run_scipy_minimize.value_and_gradient_at::device_get::714:38",
        "src/simsopt_jax/solve/dispatch.py::_run_scipy_minimize.value_and_gradient_at::device_put::712:54",
        "src/simsopt_jax/solve/dispatch.py::_run_scipy_minimize::device_get::765:25",
        "src/simsopt_jax/solve/serial.py::_write_bounded_objective_log::transfer_guard::423:9",
        "src/simsopt_jax_adapters/geo/boozer_surface.py::_with_host_bridge_transfer_guard.wrapped::transfer_guard_device_to_host::199:13",
        "src/simsopt_jax_adapters/geo/boozer_surface.py::_with_host_bridge_transfer_guard.wrapped::transfer_guard_host_to_device::200:17",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_ensure_traceable_runtime_host_wrappers.compute_baseline_value_and_grad::transfer_guard_device_to_host::4764:17",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_ensure_traceable_runtime_host_wrappers.compute_baseline_value_and_grad::transfer_guard_host_to_device::4752:17",
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py::_make_traceable_lazy_host_reporting_metrics._baseline_reporting_metrics::transfer_guard_device_to_host::4697:17",
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
# Also escapes: an admitted site whose callee does not resolve to the owner in
# its lexical scope, and a wildcard import in a file with admitted sites.
# Non-constant dynamic imports are escapes unless pinned in
# ``_ALLOWED_DYNAMIC_LOOKUP_SITES``. The module docstring lists the known limits.
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
# Lookups admitted to import a module by a non-constant name, keyed by scope and
# pinned, in source order, by a fingerprint of the lookup expression together with
# the path of the statement holding it (``_lookup_fingerprint``).
_ALLOWED_DYNAMIC_LOOKUP_SITES = {
    # Upstream serialization: import_module(parent_module) records a package version.
    "src/simsopt/_core/json.py::GSONEncoder.default": ("68b26feea9a7",),
    "src/simsopt/_core/json.py::GSONable.as_dict": ("e61549e4653f",),
    "src/simsopt/_core/json.py::SIMSON.as_dict": ("6ad3293a1eef",),
    # Upstream deserialization: __import__(modname, ..., [objname or classname], 0)
    # loads the class a serialized document names.
    "src/simsopt/_core/json.py::GSONDecoder.process_decoded": (
        "b5bcb10c41a8",
        "f93617830ce9",
    ),
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
                if not isinstance(child, ast.AST):
                    continue
                path = f"{prefix}{field}[{index}]"
                if isinstance(child, ast.stmt):
                    paths[id(child)] = path
                # Non-statement containers (``except`` handlers, ``match``
                # cases) hold statement lists too; walk through them.
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


def _lookup_fingerprint(node: ast.expr, statement_path: str) -> str:
    return _with_fingerprint(node, statement_path)


def _enclosing_statement_paths(
    tree: ast.Module, statement_paths: dict[int, str]
) -> dict[int, str]:
    """Map each node's id to the path of the innermost statement holding it."""
    enclosing: dict[int, str] = {}
    # ast.walk is breadth-first, so an inner statement overwrites its parents.
    for statement in ast.walk(tree):
        if isinstance(statement, ast.stmt):
            path = statement_paths[id(statement)]
            enclosing.update((id(node), path) for node in ast.walk(statement))
    return enclosing


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


_PERMIT = "permit"
_OWNER_MODULE_BINDING = "owner-module"
_OWNER_PACKAGE_BINDING = "owner-package"
_SIMSOPT_JAX_ROOT_BINDING = "simsopt_jax-root"
_OTHER_BINDING = "other"
_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _import_bindings(
    node: ast.Import | ast.ImportFrom, relative_path: str
) -> tuple[tuple[str, str], ...]:
    """Return ``(bound name, kind)`` for each name an import statement binds."""
    if isinstance(node, ast.Import):
        return tuple(
            (
                alias.asname,
                _OWNER_MODULE_BINDING
                if alias.name == _OWNER_MODULE
                else _OWNER_PACKAGE_BINDING
                if alias.name == _OWNER_PACKAGE
                else _OTHER_BINDING,
            )
            if alias.asname is not None
            else (
                alias.name.split(".")[0],
                _SIMSOPT_JAX_ROOT_BINDING
                if alias.name.split(".")[0] == "simsopt_jax"
                else _OTHER_BINDING,
            )
            for alias in node.names
        )
    module = _import_from_module(node, relative_path)
    return tuple(
        (
            alias.asname or alias.name,
            _PERMIT
            if module == _OWNER_MODULE
            and alias.name == _ALLOW_HOST_TRANSFERS
            and alias.asname is None
            else _OWNER_MODULE_BINDING
            if module == _OWNER_PACKAGE and alias.name == _OWNER_MODULE_NAME
            else _OWNER_PACKAGE_BINDING
            if f"{module}.{alias.name}" == _OWNER_PACKAGE
            else _OTHER_BINDING,
        )
        for alias in node.names
    )


class _ScopeBindings(NamedTuple):
    """What one scope binds, and the names it declares ``global``/``nonlocal``."""

    bindings: dict[str, frozenset[str]]
    global_names: frozenset[str]
    nonlocal_names: frozenset[str]


def _direct_scope_bindings(scope: ast.AST, relative_path: str) -> _ScopeBindings:
    """Map each name ``scope``'s own code binds to the kinds of its bindings.

    Nested functions, classes, lambdas and comprehensions are their own scopes:
    only a nested ``def``/``class`` name and a comprehension's ``:=`` targets
    bind here. Names declared ``global``/``nonlocal`` are included; the caller
    moves them to the scope they rebind.
    """
    bindings: dict[str, set[str]] = {}
    global_names: set[str] = set()
    nonlocal_names: set[str] = set()

    def bind(name: str, kind: str = _OTHER_BINDING) -> None:
        bindings.setdefault(name, set()).add(kind)

    def collect(node: ast.AST) -> None:
        if isinstance(node, (*_FUNCTION_SCOPES, ast.ClassDef)):
            bind(node.name)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, _COMPREHENSIONS):
            for inner in ast.walk(node):
                if isinstance(inner, ast.NamedExpr):
                    bind(inner.target.id)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bind(node.id)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name is not None:
                bind(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            bind(node.rest)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for name, kind in _import_bindings(node, relative_path):
                bind(name, kind)
        elif isinstance(node, ast.Global):
            global_names.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            nonlocal_names.update(node.names)
        for child in ast.iter_child_nodes(node):
            collect(child)

    if isinstance(scope, _FUNCTION_SCOPES):
        arguments = scope.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *(item for item in (arguments.vararg, arguments.kwarg) if item),
        ):
            bind(argument.arg)
    for statement in scope.body:
        collect(statement)
    return _ScopeBindings(
        {name: frozenset(kinds) for name, kinds in bindings.items()},
        frozenset(global_names),
        frozenset(nonlocal_names),
    )


def _scope_bindings(scope: ast.AST, relative_path: str) -> _ScopeBindings:
    """Bindings that live in ``scope``, with ``global``/``nonlocal`` honoured.

    A name ``scope`` declares ``global`` or ``nonlocal`` is not bound here.
    Conversely, a nested function that declares a name ``global`` rebinds it in
    the module, and one that declares it ``nonlocal`` rebinds it in an
    enclosing function; those bindings are added to the module and to every
    enclosing function respectively, which can only make resolution stricter.
    """
    direct = _direct_scope_bindings(scope, relative_path)
    declared = direct.global_names | direct.nonlocal_names
    bindings = {
        name: set(kinds)
        for name, kinds in direct.bindings.items()
        if name not in declared
    }
    if isinstance(scope, (ast.Module, *_FUNCTION_SCOPES)):
        for inner in ast.walk(scope):
            if inner is scope or not isinstance(inner, _FUNCTION_SCOPES):
                continue
            inner_direct = _direct_scope_bindings(inner, relative_path)
            rebound = (
                inner_direct.global_names
                if isinstance(scope, ast.Module)
                else inner_direct.nonlocal_names
            )
            for name in rebound & inner_direct.bindings.keys():
                bindings.setdefault(name, set()).update(inner_direct.bindings[name])
    return _ScopeBindings(
        {name: frozenset(kinds) for name, kinds in bindings.items()},
        direct.global_names,
        direct.nonlocal_names,
    )


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
        enclosing_statement_paths: dict[int, str],
        owner_bindings: tuple[frozenset[str], frozenset[str]],
        literal_attribute_bases: frozenset[int],
        tree: ast.Module,
    ) -> None:
        self.relative_path = relative_path
        self.statement_paths = statement_paths
        self.enclosing_statement_paths = enclosing_statement_paths
        self.owner_names, self.package_names = owner_bindings
        self.tree = tree
        self.scope_nodes: list[ast.AST] = []
        self.scope_bindings: dict[int, _ScopeBindings] = {}
        self.literal_attribute_bases = literal_attribute_bases
        self.scope_names: list[str] = []
        self.sites: dict[str, tuple[str, ...]] = {}
        self.escapes: list[str] = []
        self.dynamic_lookups: dict[str, tuple[str, ...]] = {}

    def _scope_key(self) -> str:
        scope = ".".join(self.scope_names) or "<module>"
        return f"{self.relative_path}::{scope}"

    def _escape(self, node: ast.AST, form: str) -> None:
        self.escapes.append(f"{self._scope_key()}::{node.lineno} {form}")

    def _dynamic_lookup(self, node: ast.expr) -> None:
        key = self._scope_key()
        fingerprint = _lookup_fingerprint(
            node, self.enclosing_statement_paths.get(id(node), "<module>")
        )
        self.dynamic_lookups[key] = (*self.dynamic_lookups.get(key, ()), fingerprint)

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

    def _bindings(self, scope: ast.AST) -> _ScopeBindings:
        if id(scope) not in self.scope_bindings:
            self.scope_bindings[id(scope)] = _scope_bindings(scope, self.relative_path)
        return self.scope_bindings[id(scope)]

    def _lexical_binding(self, name: str) -> frozenset[str]:
        """Kinds of the binding ``name`` resolves to from the current scope."""
        module_kinds = self._bindings(self.tree).bindings.get(name, frozenset())
        for depth, scope in enumerate(reversed(self.scope_nodes)):
            # A class body is visible only to code directly inside it.
            if depth > 0 and isinstance(scope, ast.ClassDef):
                continue
            scope_bindings = self._bindings(scope)
            if name in scope_bindings.global_names:
                return module_kinds
            if name in scope_bindings.nonlocal_names:
                continue
            kinds = scope_bindings.bindings.get(name)
            if kinds:
                return kinds
        return module_kinds

    def _resolves_to_owner(self, callee: ast.expr) -> bool:
        if isinstance(callee, ast.Name):
            return self._lexical_binding(callee.id) == {_PERMIT}
        if not isinstance(callee, ast.Attribute):
            return False
        base = callee.value
        if isinstance(base, ast.Name):
            return self._lexical_binding(base.id) == {_OWNER_MODULE_BINDING}
        if _dotted_name(base) == _OWNER_MODULE:
            return self._lexical_binding("simsopt_jax") == {_SIMSOPT_JAX_ROOT_BINDING}
        return (
            isinstance(base, ast.Attribute)
            and base.attr == _OWNER_MODULE_NAME
            and isinstance(base.value, ast.Name)
            and self._lexical_binding(base.value.id) == {_OWNER_PACKAGE_BINDING}
        )

    def _check_module_key(self, key: ast.expr | None, node: ast.expr) -> None:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            self._dynamic_lookup(node)
        elif _names_owner_module(key.value):
            self._escape(node, "owner module lookup by name")

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scope_names.append(name)
        self.scope_nodes.append(node)
        self.generic_visit(node)
        self.scope_nodes.pop()
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
                if not self._resolves_to_owner(call.func):
                    self._escape(node, "admitted site does not resolve to the owner")
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
            if alias.name != _ALLOW_HOST_TRANSFERS:
                continue
            if alias.asname is not None:
                self._escape(node, "aliased import")
            if _import_from_module(node, self.relative_path) != _OWNER_MODULE:
                self._escape(node, "imported from a non-owner module")

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
    admitted_dynamic_lookups: dict[str, tuple[str, ...]] | None = None,
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    tree = ast.parse(source)
    statement_paths = _statement_paths(tree)
    census = _AllowHostTransfersCensus(
        relative_path,
        statement_paths,
        _enclosing_statement_paths(tree, statement_paths),
        _owner_bindings(tree, relative_path),
        _literal_attribute_bases(tree),
        tree,
    )
    census.visit(tree)
    if census.sites:
        census.escapes.extend(
            f"{relative_path}::<module>::{node.lineno} wildcard import in a file "
            "with admitted sites"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
        )
    admitted = {
        key: fingerprints
        for key, fingerprints in (admitted_dynamic_lookups or {}).items()
        if key.startswith(f"{relative_path}::")
    }
    dynamic_escapes = [
        f"{key} dynamic module lookups {census.dynamic_lookups.get(key, ())}"
        f" (admitted {admitted.get(key, ())})"
        for key in sorted({*census.dynamic_lookups, *admitted})
        if census.dynamic_lookups.get(key, ()) != admitted.get(key, ())
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
    module_imports["site inside an except handler and a match case"] = (
        "from simsopt_jax.runtime import host_boundary\n\n"
        "def approved(value):\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError:\n"
        "        with host_boundary.allow_host_transfers():\n"
        "            transfer()\n"
        "    match value:\n"
        "        case 1:\n"
        "            with host_boundary.allow_host_transfers():\n"
        "                transfer()\n"
    )
    for form, source in module_imports.items():
        module_sites, module_escapes = _allow_host_transfers_census(
            source, "src/example.py"
        )
        assert list(module_sites) == ["src/example.py::approved"], form
        assert module_escapes == [], form
    class_source = (
        "class Holder:\n"
        "    def approved(self):\n"
        "        from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
        "        with allow_host_transfers():\n"
        "            transfer()\n"
    )
    declared_sources = {
        "global": (
            "from simsopt_jax.runtime import host_boundary\n\n"
            "def outer():\n"
            "    host_boundary = None\n"
            "    def approved():\n"
            "        global host_boundary\n"
            "        with host_boundary.allow_host_transfers():\n"
            "            transfer()\n"
            "    return approved, host_boundary\n"
        ),
        "nonlocal": (
            "def outer():\n"
            "    from simsopt_jax.runtime import host_boundary\n"
            "    def approved():\n"
            "        nonlocal host_boundary\n"
            "        with host_boundary.allow_host_transfers():\n"
            "            transfer()\n"
            "    return approved\n"
        ),
    }
    for form, source in declared_sources.items():
        declared_sites, declared_escapes = _allow_host_transfers_census(
            source, "src/example.py"
        )
        assert list(declared_sites) == ["src/example.py::outer.approved"], form
        assert [
            escape
            for escape in declared_escapes
            if "admitted site does not resolve" in escape
        ] == [], form
    class_sites, class_escapes = _allow_host_transfers_census(
        class_source, "src/example.py"
    )
    assert list(class_sites) == ["src/example.py::Holder.approved"]
    assert class_escapes == []


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
        "admitted site imported from another module": _ADMITTED_SOURCE.replace(
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers",
            "from other_package.guards import allow_host_transfers",
        ),
        "admitted site through another module object": (
            "from other_package import guards\n\n"
            "def approved():\n"
            "    with guards.allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "admitted site shadowed by a parameter": (
            "from contextlib import nullcontext\n"
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved(allow_host_transfers=nullcontext):\n"
            "    with allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "admitted site shadowed by a keyword-only parameter": (
            "from contextlib import nullcontext\n"
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved(*, allow_host_transfers=nullcontext):\n"
            "    with allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "admitted site shadowed by a local assignment": (
            "from contextlib import nullcontext\n"
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    allow_host_transfers = nullcontext\n"
            "    with allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "admitted site shadowed in an enclosing function": (
            "from contextlib import nullcontext\n"
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def outer():\n"
            "    for allow_host_transfers in (nullcontext,):\n"
            "        pass\n"
            "    def approved():\n"
            "        with allow_host_transfers():\n"
            "            transfer()\n"
            "    return approved\n"
        ),
        "admitted site shadowed by a local import": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    from contextlib import nullcontext as allow_host_transfers\n"
            "    with allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "admitted site shadowed by a nested def": (
            "from contextlib import nullcontext\n"
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved():\n"
            "    def allow_host_transfers():\n"
            "        return nullcontext()\n"
            "    with allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "admitted site shadowed by a with target": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def approved(factory):\n"
            "    with factory() as allow_host_transfers:\n"
            "        pass\n"
            "    with allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "owner module shadowed by a parameter": (
            "from simsopt_jax.runtime import host_boundary\n\n"
            "def approved(host_boundary):\n"
            "    with host_boundary.allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "wildcard import beside an admitted site": (
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n"
            "from other_package import *\n\n"
            "def approved():\n"
            "    with allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "global rebinds the owner module name at module scope": (
            "from other_package import host_boundary\n\n"
            "def outer():\n"
            "    from simsopt_jax.runtime import host_boundary\n"
            "    def approved():\n"
            "        global host_boundary\n"
            "        with host_boundary.allow_host_transfers():\n"
            "            transfer()\n"
            "    return approved, host_boundary\n"
        ),
        "global in a nested function rebinds the module permit": (
            "from contextlib import nullcontext\n"
            "from simsopt_jax.runtime.host_boundary import allow_host_transfers\n\n"
            "def swap():\n"
            "    global allow_host_transfers\n"
            "    allow_host_transfers = nullcontext\n\n"
            "def approved():\n"
            "    with allow_host_transfers():\n"
            "        transfer()\n"
        ),
        "nonlocal rebinds the enclosing owner module name": (
            "def outer(other):\n"
            "    from simsopt_jax.runtime import host_boundary\n"
            "    def swap():\n"
            "        nonlocal host_boundary\n"
            "        host_boundary = other\n"
            "    def approved():\n"
            "        with host_boundary.allow_host_transfers():\n"
            "            transfer()\n"
            "    return swap, approved\n"
        ),
        "nonlocal resolves to the enclosing rebinding": (
            "import other_package as host_boundary\n\n"
            "def outer():\n"
            "    host_boundary = make()\n"
            "    def approved():\n"
            "        nonlocal host_boundary\n"
            "        with host_boundary.allow_host_transfers():\n"
            "            transfer()\n"
            "    return approved\n"
        ),
        "admitted site with no import": (
            "def approved():\n    with allow_host_transfers():\n        transfer()\n"
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


def test_allow_host_transfers_census_admits_only_pinned_dynamic_lookups() -> None:
    source = (
        "import importlib\n\n"
        "def load(name):\n"
        "    return importlib.import_module(name)\n"
    )
    admitted = {
        "src/example.py::load": (
            _lookup_fingerprint(
                ast.parse("importlib.import_module(name)", mode="eval").body, "body[0]"
            ),
        )
    }
    same_count_replacements = {
        "argument replaced": source.replace(
            "import_module(name)", "import_module(other)"
        ),
        "importer replaced": source.replace(
            "importlib.import_module(name)", "__import__(name)"
        ),
        "moved within its scope": source.replace(
            "    return importlib.import_module(name)\n",
            "    prepare()\n    return importlib.import_module(name)\n",
        ),
    }

    _, unlisted = _allow_host_transfers_census(source, "src/example.py")
    _, listed = _allow_host_transfers_census(source, "src/example.py", admitted)
    _, stale = _allow_host_transfers_census(
        "def load(name):\n    return name\n", "src/example.py", admitted
    )

    assert len(unlisted) == 1 and unlisted[0].startswith(
        "src/example.py::load dynamic module lookups ("
    )
    assert listed == []
    expected_stale = (
        "src/example.py::load dynamic module lookups () "
        f"(admitted {admitted['src/example.py::load']})"
    )
    assert stale == [expected_stale]
    for form, replaced in same_count_replacements.items():
        assert replaced != source, f"{form} mutation did not apply"
        _, escapes = _allow_host_transfers_census(replaced, "src/example.py", admitted)
        assert escapes, f"{form} kept the admitted dynamic lookup fingerprint"


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
