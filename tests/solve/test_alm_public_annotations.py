"""Every public annotation of ``simsopt.solve.alm``, its opt-in plugins and the
library examples resolves with ``typing.get_type_hints``.

The library examples are the ALM example code ``tests/solve`` imports. A caller
that introspects the public API (a type checker, a ``TypedDict`` consumer,
documentation tooling) needs every name in those annotations to exist at
runtime, on every supported Python.

The floor is the package's ``requires-python`` (3.8), so the ALM sources, the
signed geometry constraints, the ALM examples and these tests are also checked
statically for constructs newer than 3.8: a builtin generic such as
``tuple[int, ...]`` (a runtime ``TypeError`` before 3.9, and
``typing.get_type_hints`` evaluates string annotations), ``X | Y`` in an
annotation or an ``isinstance`` check, ``typing`` names added after 3.8 and
standard-library calls added after 3.8.
"""

import ast
import inspect
import sys
import typing
import unittest
from pathlib import Path

import simsopt.solve.alm as alm
from simsopt.solve.alm import checkpoint as alm_checkpoint
from simsopt.solve.alm import continuation as alm_continuation
from simsopt.solve.alm import history as alm_history
from simsopt.solve.alm import policy as alm_policy

LIBRARY_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "2_Intermediate"
if str(LIBRARY_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(LIBRARY_EXAMPLES_DIR))

import alm_composition_example  # noqa: E402
import alm_typed_evaluator_example  # noqa: E402
import boozerQA_alm  # noqa: E402  (module scope builds no problem)

EXAMPLE_MODULES = (
    alm_composition_example,
    alm_typed_evaluator_example,
    boozerQA_alm,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_FLOOR = (3, 8)
PYTHON_FLOOR_SOURCES = (
    *sorted((REPO_ROOT / "src" / "simsopt" / "solve" / "alm").glob("*.py")),
    REPO_ROOT / "src" / "simsopt" / "geo" / "signed_constraints.py",
    *(LIBRARY_EXAMPLES_DIR / name for name in (
        "stage_two_optimization_alm.py",
        "boozerQA_alm.py",
        "alm_composition_example.py",
        "alm_typed_evaluator_example.py",
    )),
    *sorted((REPO_ROOT / "tests" / "solve").glob("test_alm*.py")),
    REPO_ROOT / "tests" / "geo" / "test_signed_constraints.py",
)
# Subscripting these is a runtime TypeError before Python 3.9 (PEP 585).
_BUILTIN_GENERICS = frozenset({"dict", "frozenset", "list", "set", "tuple", "type"})
# ``typing`` names added after Python 3.8.
_POST_38_TYPING = frozenset({
    "Annotated", "Concatenate", "LiteralString", "Never", "NotRequired",
    "ParamSpec", "ParamSpecArgs", "ParamSpecKwargs", "Required", "Self",
    "TypeAlias", "TypeGuard", "TypeIs", "TypeVarTuple", "Unpack",
    "assert_never", "assert_type", "dataclass_transform", "is_typeddict",
    "override", "reveal_type",
})
# Methods added after Python 3.8, matched by attribute name.
_POST_38_METHODS = frozenset({
    "bit_count", "is_relative_to", "removeprefix", "removesuffix", "with_stem",
})
# ``module.function`` added after Python 3.8.
_POST_38_FUNCTIONS = frozenset({
    ("ast", "unparse"), ("functools", "cache"), ("math", "lcm"),
    ("math", "nextafter"), ("math", "ulp"), ("itertools", "pairwise"),
    ("itertools", "batched"), ("random", "randbytes"),
})
# ``function(keyword=...)`` added after Python 3.8.
_POST_38_KEYWORDS = frozenset({
    ("zip", "strict"), ("dataclass", "kw_only"), ("dataclass", "slots"),
    ("get_type_hints", "include_extras"),
})


def _call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _python_floor_violations(tree):
    """``(lineno, message)`` for each construct newer than ``PYTHON_FLOOR``."""
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            for arg in (*arguments.posonlyargs, *arguments.args,
                        *arguments.kwonlyargs, arguments.vararg, arguments.kwarg):
                if arg is not None and arg.annotation is not None:
                    annotations.append(arg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif (isinstance(node, ast.Call) and _call_name(node.func) in ("isinstance", "issubclass")
              and len(node.args) == 2):
            annotations.append(node.args[1])
    for annotation in annotations:
        for node in ast.walk(annotation):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                yield node.lineno, "X | Y union (PEP 604, Python 3.10)"
    abc_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "collections.abc":
            abc_names.update(alias.asname or alias.name for alias in node.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            value = node.value
            if isinstance(value, ast.Name) and value.id in _BUILTIN_GENERICS:
                yield node.lineno, f"builtin generic {value.id}[...] (PEP 585, Python 3.9)"
            elif isinstance(value, ast.Name) and value.id in abc_names:
                yield node.lineno, f"collections.abc generic {value.id}[...] (Python 3.9)"
            elif (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Attribute)
                  and value.value.attr == "abc"):
                yield node.lineno, f"collections.abc generic {value.attr}[...] (Python 3.9)"
        elif isinstance(node, ast.ImportFrom) and node.module in ("typing", "functools"):
            for alias in node.names:
                if node.module == "typing" and alias.name in _POST_38_TYPING:
                    yield node.lineno, f"typing.{alias.name} (after Python 3.8)"
                if node.module == "functools" and alias.name == "cache":
                    yield node.lineno, "functools.cache (Python 3.9)"
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "typing" \
                    and node.attr in _POST_38_TYPING:
                yield node.lineno, f"typing.{node.attr} (after Python 3.8)"
            elif node.attr in _POST_38_METHODS:
                yield node.lineno, f".{node.attr}() (after Python 3.8)"
            elif isinstance(node.value, ast.Name) and (node.value.id, node.attr) in _POST_38_FUNCTIONS:
                yield node.lineno, f"{node.value.id}.{node.attr} (after Python 3.8)"
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (_call_name(node.func), keyword.arg) in _POST_38_KEYWORDS:
                    yield node.lineno, f"{_call_name(node.func)}({keyword.arg}=...) (after Python 3.8)"


def _public_annotated_objects():
    """``(label, object)`` for each public class, its public methods and each
    public function defined by the package, its plugin modules or the examples."""
    modules = (alm, alm_checkpoint, alm_continuation, alm_history, alm_policy,
               *EXAMPLE_MODULES)
    owners = {module.__name__ for module in EXAMPLE_MODULES}
    seen = set()
    for module in modules:
        for name in sorted(vars(module)):
            value = getattr(module, name)
            if name.startswith("_") or id(value) in seen:
                continue
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            owner = getattr(value, "__module__", "")
            if not (owner.startswith("simsopt.solve.alm") or owner in owners):
                continue
            seen.add(id(value))
            yield f"{value.__module__}.{name}", value
            if inspect.isclass(value):
                for method_name, method in vars(value).items():
                    if inspect.isfunction(method) and not method_name.startswith("_"):
                        yield f"{value.__module__}.{name}.{method_name}", method


class AlmPublicAnnotationTests(unittest.TestCase):
    def test_every_public_annotation_resolves(self):
        labels = []
        for label, value in _public_annotated_objects():
            with self.subTest(label=label):
                typing.get_type_hints(value)
            labels.append(label)
        self.assertIn("simsopt.solve.alm.control.minimize_alm", labels)
        self.assertIn("simsopt.solve.alm.checkpoint.ALMTransitionSnapshot", labels)
        self.assertIn("simsopt.solve.alm.history.ALMHistoryRecorder", labels)
        self.assertIn("boozerQA_alm.BoozerQAProblem", labels)


class AlmPythonFloorTests(unittest.TestCase):
    def test_sources_use_no_construct_newer_than_the_python_floor(self):
        self.assertEqual(len(PYTHON_FLOOR_SOURCES), len(set(PYTHON_FLOOR_SOURCES)))
        violations = []
        for path in PYTHON_FLOOR_SOURCES:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path), feature_version=PYTHON_FLOOR)
            violations.extend(
                f"{path.relative_to(REPO_ROOT)}:{lineno}: {message}"
                for lineno, message in _python_floor_violations(tree)
            )
        self.assertEqual(violations, [], "\n" + "\n".join(sorted(set(violations))))

    def test_the_evaluation_schema_annotations_are_eager(self):
        # On Python 3.8 a TypedDict subclass defined in another module resolves
        # inherited string annotations in that module's namespace, so the
        # ALMEvaluation schema must not use postponed annotations.
        path = REPO_ROOT / "src" / "simsopt" / "solve" / "alm" / "evaluation.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        self.assertFalse(any(
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
            for node in tree.body
        ))

    def test_the_floor_matches_requires_python(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.8"', pyproject)

    def test_the_rules_flag_each_construct(self):
        source = "\n".join((
            "from typing import TypeAlias",
            "from collections.abc import Mapping",
            "from functools import cache",
            "import ast, math",
            "Alias = tuple[int, ...]",
            "def f(x: int | None, y: Mapping[str, int]) -> list[str]: ...",
            "isinstance(1, int | str)",
            "'a'.removeprefix('b')",
            "ast.unparse(None)",
            "zip((), (), strict=True)",
        ))
        messages = [message for _, message in _python_floor_violations(ast.parse(source))]
        self.assertEqual(len(messages), 10, messages)


if __name__ == "__main__":
    unittest.main()
