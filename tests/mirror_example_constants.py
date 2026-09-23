"""Read a shipped mirror example's module constants out of its own bytes.

The example scripts live in ``1_Simple`` / ``2_Intermediate`` directories, which
are not importable module paths, and this repository forbids dynamic module
loading (no ``importlib``, ``runpy``, ``exec`` or ``compile``). The parity
tests that bind a mirror's configuration to its parity case therefore read the
shipped source with ``ast``, the way ``tests/integration/test_jax_examples.py``
and ``tests/jax/examples/test_muse_post_workflow.py`` already inspect examples.

Reading the bytes is not the same as reading the BINDING. A test that asks
whether a constant's name appears somewhere in a function body passes for a
constant that is merely mentioned, is satisfied by any superstring
(``"HISTORY_COUNT" in source`` also matches ``BOUNDED_HISTORY_COUNT``), and
cannot see a value handed to the wrong keyword.
:func:`mirror_example_call_bindings` answers the question that matters instead:
which module constants does this call's keyword actually carry?
"""

from __future__ import annotations

import ast
from pathlib import Path


def mirror_example_constants(path: Path) -> dict[str, object]:
    """Every ``NAME = <literal>`` assignment at module scope, by name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }


def mirror_example_function_source(path: Path, name: str) -> str:
    """The named module-level function of the shipped script, unparsed."""
    return ast.unparse(_function(_module(path), name))


def mirror_example_call_bindings(
    path: Path,
    function: str,
    callee: str,
) -> dict[str, frozenset[str]]:
    """Which names each keyword of one call inside ``function`` carries.

    The single call to ``callee`` in ``function`` is located, and every keyword
    is reduced to the set of names its expression depends on, expanded through
    the function's own local assignments (``magnet_cap = NATIVE_MAGNET_CAP if
    native_scale else BOUNDED_MAGNET_CAP`` resolves to the two constants).  A
    name that is neither a local assignment target nor removed by that
    expansion -- a module constant, a parameter, an imported module -- is
    returned as itself.

    ``ValueError`` when ``function`` does not contain exactly one call to
    ``callee``: a binding read from an ambiguous match is not evidence.
    """
    function_node = _function(_module(path), function)
    calls = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call) and _callee_name(node) == callee
    ]
    if len(calls) != 1:
        raise ValueError(
            f"{path.name}:{function} contains {len(calls)} calls to {callee}, not one"
        )
    assignments = _local_assignments(function_node)
    return {
        keyword.arg: _expand(_names(keyword.value), assignments)
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _callee_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _names(node: ast.expr) -> frozenset[str]:
    return frozenset(
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    )


def _local_assignments(function_node: ast.FunctionDef) -> dict[str, frozenset[str]]:
    """Name -> names its value depends on, for single-target local assignments."""
    assignments: dict[str, frozenset[str]] = {}
    for node in ast.walk(function_node):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments[node.targets[0].id] = _names(node.value)
    return assignments


def _expand(
    names: frozenset[str],
    assignments: dict[str, frozenset[str]],
) -> frozenset[str]:
    """Replace every local assignment target by the names it was assigned."""
    resolved: set[str] = set()
    pending = list(names)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in assignments:
            pending.extend(assignments[name])
        else:
            resolved.add(name)
    return frozenset(resolved)
