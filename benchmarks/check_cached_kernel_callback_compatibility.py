"""AST gate for cached JAX kernels that must not use host callbacks.

Persistent JAX compilation cache entries are only reusable when the cached
kernel does not capture host/debug callback primitives.  This module keeps the
policy local: each target names one source function, and the checker resolves
callback aliases with Python's AST rather than fragile substring scans.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_CALLBACK_QUALNAMES = (
    "jax.debug.callback",
    "jax.debug.breakpoint",
    "jax.debug.print",
    "jax.pure_callback",
    "jax.experimental.io_callback",
    "jax.experimental.host_callback.call",
    "jax.experimental.host_callback.id_print",
    "jax.experimental.host_callback.id_tap",
)


@dataclass(frozen=True)
class CallbackFreeTarget:
    """Source function whose cached kernel must not reference host callbacks."""

    module_path: str
    symbol: str


@dataclass(frozen=True)
class _TargetFunction:
    module_path: Path
    node: ast.FunctionDef | ast.AsyncFunctionDef
    module_aliases: dict[str, str]
    module_star_imports: tuple[str, ...]
    module_functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]


CALLBACK_FREE_TARGETS = (
    CallbackFreeTarget(
        "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py",
        "_value_and_grad_for",
    ),
)


def _record_import_aliases(
    node: ast.Import | ast.ImportFrom,
    aliases: dict[str, str],
    star_imports: list[str],
) -> None:
    if isinstance(node, ast.Import):
        for name in node.names:
            if name.asname is None:
                root_name = name.name.split(".", maxsplit=1)[0]
                aliases[root_name] = root_name
            else:
                aliases[name.asname] = name.name
    elif node.module is not None:
        for name in node.names:
            if name.name == "*":
                star_imports.append(node.module)
            else:
                aliases[name.asname or name.name] = f"{node.module}.{name.name}"


def _import_aliases_in_scope(root: ast.AST) -> tuple[dict[str, str], tuple[str, ...]]:
    aliases: dict[str, str] = {}
    star_imports: list[str] = []

    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            _record_import_aliases(node, aliases, star_imports)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            _record_import_aliases(node, aliases, star_imports)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is root:
                self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is root:
                self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node is root:
                self.generic_visit(node)

    ImportVisitor().visit(root)
    return aliases, tuple(star_imports)


def _find_target_function(
    module_path: str | Path,
    symbol: str,
) -> _TargetFunction:
    path = Path(module_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    matches = [node for node in function_nodes if node.name == symbol]
    if len(matches) != 1:
        raise ValueError(f"{path}: found {len(matches)} functions named {symbol!r}")
    duplicate_names = {
        node.name
        for node in function_nodes
        if sum(1 for candidate in function_nodes if candidate.name == node.name) > 1
    }
    module_functions = {
        node.name: node for node in function_nodes if node.name not in duplicate_names
    }
    module_aliases, module_star_imports = _import_aliases_in_scope(tree)
    return _TargetFunction(
        module_path=path,
        node=matches[0],
        module_aliases=module_aliases,
        module_star_imports=module_star_imports,
        module_functions=module_functions,
    )


def _argument_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in args.posonlyargs}
    names.update(arg.arg for arg in args.args)
    names.update(arg.arg for arg in args.kwonlyargs)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()

    class BindingVisitor(ast.NodeVisitor):
        def visit_Name(self, child: ast.Name) -> None:
            if isinstance(child.ctx, ast.Store):
                names.add(child.id)

        def visit_ExceptHandler(self, child: ast.ExceptHandler) -> None:
            if child.name is not None:
                names.add(child.name)
            self.generic_visit(child)

        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            if child is node:
                self.generic_visit(child)
            else:
                names.add(child.name)

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            if child is node:
                self.generic_visit(child)
            else:
                names.add(child.name)

        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            if child is node:
                self.generic_visit(child)
            else:
                names.add(child.name)

    BindingVisitor().visit(node)
    return names


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _qualname_candidates(
    expression_name: str,
    *,
    aliases: dict[str, str],
    star_imports: tuple[str, ...],
    shadowed_names: set[str],
) -> tuple[str, ...]:
    prefix, dot, suffix = expression_name.partition(".")
    if prefix in aliases:
        resolved = aliases[prefix]
        if dot:
            resolved = f"{resolved}.{suffix}"
        return (resolved,)
    if dot:
        return (expression_name,)
    if prefix in shadowed_names:
        return ()
    return tuple(f"{module}.{prefix}" for module in star_imports) + (expression_name,)


def _forbidden_callbacks_in_function(function: _TargetFunction) -> tuple[str, ...]:
    found: set[str] = set()
    forbidden = set(FORBIDDEN_CALLBACK_QUALNAMES)
    visited_scope_ids: set[int] = set()

    def scan_scope(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
        parent_aliases: dict[str, str],
        parent_star_imports: tuple[str, ...],
    ) -> None:
        if id(node) in visited_scope_ids:
            return
        visited_scope_ids.add(id(node))
        local_aliases, local_star_imports = _import_aliases_in_scope(node)
        aliases = dict(parent_aliases)
        aliases.update(local_aliases)
        star_imports = parent_star_imports + local_star_imports
        local_imported_names = set(local_aliases)
        shadowed_names = (
            _argument_names(node.args) | _assigned_names(node)
        ) - local_imported_names
        for name in shadowed_names:
            aliases.pop(name, None)
        helper_calls: set[str] = set()

        class CallbackVisitor(ast.NodeVisitor):
            def visit_Name(self, child: ast.Name) -> None:
                self._record_expression(child)

            def visit_Attribute(self, child: ast.Attribute) -> None:
                self._record_expression(child)
                self.generic_visit(child)

            def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
                if child is node:
                    self.generic_visit(child)
                else:
                    scan_scope(child, aliases, star_imports)

            def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
                if child is node:
                    self.generic_visit(child)
                else:
                    scan_scope(child, aliases, star_imports)

            def visit_Lambda(self, child: ast.Lambda) -> None:
                if child is node:
                    self.generic_visit(child)
                else:
                    scan_scope(child, aliases, star_imports)

            def visit_Call(self, child: ast.Call) -> None:
                self._record_expression(child.func)
                self._record_helper_call(child.func)
                self.generic_visit(child)

            def _record_expression(self, child: ast.AST) -> None:
                expression_name = _dotted_name(child)
                if expression_name is not None:
                    for qualname in _qualname_candidates(
                        expression_name,
                        aliases=aliases,
                        star_imports=star_imports,
                        shadowed_names=shadowed_names,
                    ):
                        if qualname in forbidden:
                            found.add(qualname)

            def _record_helper_call(self, child: ast.AST) -> None:
                if not isinstance(child, ast.Name):
                    return
                if child.id in aliases or child.id in shadowed_names:
                    return
                if child.id in function.module_functions:
                    helper_calls.add(child.id)

        CallbackVisitor().visit(node)
        for helper_name in sorted(helper_calls):
            scan_scope(
                function.module_functions[helper_name],
                function.module_aliases,
                function.module_star_imports,
            )

    scan_scope(
        function.node,
        function.module_aliases,
        function.module_star_imports,
    )
    return tuple(
        qualname for qualname in FORBIDDEN_CALLBACK_QUALNAMES if qualname in found
    )


def _target_result(repo_root: Path, target: CallbackFreeTarget) -> dict[str, object]:
    module_path = repo_root / target.module_path
    target_function = _find_target_function(module_path, target.symbol)
    callbacks = _forbidden_callbacks_in_function(target_function)
    return {
        "module_path": target.module_path,
        "symbol": target.symbol,
        "forbidden_callbacks": list(callbacks),
        "passed": len(callbacks) == 0,
    }


def evaluate_cached_kernel_callback_compatibility(
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, object]:
    root = Path(repo_root)
    target_results = [_target_result(root, target) for target in CALLBACK_FREE_TARGETS]
    failures = [result for result in target_results if not bool(result["passed"])]
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "targets": target_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check cacheable JAX kernels for host/debug callback references.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan. Defaults to this checkout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_cached_kernel_callback_compatibility(args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
