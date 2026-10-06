"""Native example tiers and the example-implementation resolution rule."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

TIERS = frozenset(
    {"1_Simple", "2_Intermediate", "3_Advanced", "stellarator_benchmarks"}
)
EXAMPLE_IMPLEMENTATION_PACKAGE = "simsopt_jax_adapters.examples"


class ExampleImplementationError(ValueError):
    """An example script does not resolve to a readable set of sources."""


@dataclass(frozen=True)
class ExampleSource:
    """One file a gate must read to see an example's behaviour.

    ``module`` is the dotted name inside the implementation package, or ``None``
    for the example script itself.  ``imports`` is every module this source
    imports, as ABSOLUTE dotted names with relative levels already resolved, so
    a gate never re-walks the syntax tree with a different vocabulary.
    """

    path: Path
    module: str | None
    text: str
    imports: tuple[str, ...]


@dataclass(frozen=True)
class ExampleImplementation:
    """Every source one example script's behaviour can live in.

    ``sources[0]`` is the script itself; the rest are the modules it reaches
    transitively inside ``EXAMPLE_IMPLEMENTATION_PACKAGE``, sorted by dotted
    name.  A gate reads the whole tuple, never only the script.
    """

    script_path: Path
    sources: tuple[ExampleSource, ...]


def _defines_main(syntax_tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "main"
        for node in syntax_tree.body
    )


def _module_source_path(module: str, repo_root: Path) -> Path | None:
    """Return the file backing one dotted module under ``src``, if there is one.

    Resolution is by file existence in CPython's own precedence: a package
    ``<name>/__init__.py`` wins over a module ``<name>.py``, because that is the
    file the interpreter executes when both exist.  PEP-420 namespace
    subpackages are not part of this contract: a bare directory with neither
    file raises rather than being dropped, since dropping it would hide
    everything beneath it.  ``None`` means no module of that name exists, which
    is how a ``from <package> import <name>`` naming a plain symbol is known.
    """

    module_path = repo_root / "src" / Path(*module.split("."))
    for candidate in (module_path / "__init__.py", module_path.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    if module_path.is_dir():
        raise ExampleImplementationError(
            f"namespace package is not a readable implementation module: {module}"
        )
    return None


def _in_implementation_package(module: str) -> bool:
    return module == EXAMPLE_IMPLEMENTATION_PACKAGE or module.startswith(
        f"{EXAMPLE_IMPLEMENTATION_PACKAGE}."
    )


def _ancestor_packages(module: str) -> tuple[str, ...]:
    """Name every package from the implementation package root down to ``module``.

    CPython executes each ``__init__.py`` on the way in, so implementation code
    parked in one of them runs; every ancestor therefore belongs in the closure.
    """

    suffix = module[len(EXAMPLE_IMPLEMENTATION_PACKAGE) :].strip(".")
    parts = suffix.split(".") if suffix else []
    return tuple(
        ".".join([EXAMPLE_IMPLEMENTATION_PACKAGE, *parts[:depth]])
        for depth in range(len(parts))
    ) or (EXAMPLE_IMPLEMENTATION_PACKAGE,)


def _relative_base(module: str | None, level: int, repo_root: Path) -> str:
    """Resolve the package a relative import counts up from.

    ``level`` 1 is the importing module's own package, 2 its parent, and so on,
    exactly as CPython resolves them.  A module whose file is a package
    ``__init__.py`` IS its package, so level 1 is itself.
    """

    if module is None:
        raise ExampleImplementationError(
            "example script cannot use a relative import: it is not in a package"
        )
    source_path = _module_source_path(module, repo_root)
    package = (
        module
        if source_path is not None and source_path.name == "__init__.py"
        else module.rpartition(".")[0]
    )
    parts = package.split(".") if package else []
    if level > len(parts):
        raise ExampleImplementationError(
            f"relative import escapes the source tree: level {level} from {module}"
        )
    return ".".join(parts[: len(parts) - level + 1])


def _absolute_imports(
    syntax_tree: ast.Module, *, module: str | None, repo_root: Path
) -> set[str]:
    """Name every module one source imports, as an ABSOLUTE dotted name.

    Relative forms are resolved against ``module``'s own package first, so
    ``from .deep import f`` and ``from simsopt_jax_adapters.examples.deep import
    f`` produce the same name and every later filter sees one vocabulary.
    ``module`` is the importing module's dotted name, or ``None`` for an example
    script, which is not in a package at all.  Inside the implementation package
    a ``from <package> import <name>`` also names ``<package>.<name>`` whenever a
    module of that name exists.
    """

    names: set[str] = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            target = node.module
        else:
            base = _relative_base(module, node.level, repo_root)
            target = f"{base}.{node.module}" if node.module else base
        if target is None:
            continue
        names.add(target)
        if not _in_implementation_package(target):
            continue
        for alias in node.names:
            submodule = f"{target}.{alias.name}"
            if _module_source_path(submodule, repo_root) is not None:
                names.add(submodule)
    return names


def _implementation_closure(
    imports: set[str], repo_root: Path
) -> dict[str, ExampleSource]:
    """Close the implementation-package import graph reachable from one source.

    One filter decides membership -- whether the ABSOLUTE name lies inside the
    implementation package.  Anything outside it is a library dependency and is
    ignored, however it was spelled; anything inside it must have a source under
    ``src``, or resolution fails with the named error.
    """

    pending = sorted(name for name in imports if _in_implementation_package(name))
    closure: dict[str, ExampleSource] = {}
    while pending:
        module = pending.pop()
        if module in closure:
            continue
        path = _module_source_path(module, repo_root)
        if path is None:
            raise ExampleImplementationError(
                "example reaches an implementation module with no source under "
                f"src: {module}"
            )
        text = path.read_text(encoding="utf-8")
        module_imports = _absolute_imports(
            ast.parse(text), module=module, repo_root=repo_root
        )
        closure[module] = ExampleSource(
            path=path,
            module=module,
            text=text,
            imports=tuple(sorted(module_imports)),
        )
        pending.extend(
            name for name in module_imports if _in_implementation_package(name)
        )
        pending.extend(_ancestor_packages(module))
    return closure


def resolve_example_implementation(
    script_path: Path, *, repo_root: Path
) -> ExampleImplementation:
    """Resolve one example script to every source its behaviour can live in.

    ``EXAMPLE_IMPLEMENTATION_PACKAGE`` is the sanctioned home of example
    implementations, so the answer is the script body plus the transitive
    closure of the modules it reaches inside that package, including every
    ancestor ``__init__.py`` CPython would execute on the way in.  The entry
    point must exist somewhere in that tuple: a script that defines no ``main``
    and reaches no module that defines one cannot be read by rule and raises
    :class:`ExampleImplementationError`.
    """

    script_source = script_path.read_text(encoding="utf-8")
    syntax_tree = ast.parse(script_source)
    script_imports = _absolute_imports(syntax_tree, module=None, repo_root=repo_root)
    closure = _implementation_closure(script_imports, repo_root)
    if not _defines_main(syntax_tree) and not any(
        _defines_main(ast.parse(source.text)) for source in closure.values()
    ):
        raise ExampleImplementationError(
            "neither the example script nor any module it reaches defines main: "
            f"{script_path}"
        )
    return ExampleImplementation(
        script_path=script_path,
        sources=(
            ExampleSource(
                path=script_path,
                module=None,
                text=script_source,
                imports=tuple(sorted(script_imports)),
            ),
            *(closure[module] for module in sorted(closure)),
        ),
    )
