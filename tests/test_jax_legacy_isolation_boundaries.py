from __future__ import annotations

import ast
import subprocess
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOURCE_ROOT = "src/simsopt"
FORBIDDEN_IMPORT_ROOTS = ("jax", "simsopt_jax", "simsopt_jax_adapters")
JAX_NATIVE_MARKER = "_simsopt_jax_native_field"
JAX_SOURCE_ROOTS = ("src/simsopt_jax", "src/simsopt_jax_adapters")

ALLOWED_JAX_SOURCE_LOCAL_IMPORTS_BY_REASON = {
    "jax_runtime_configuration_before_jax_import": (
        (
            "src/simsopt_jax/backend/runtime.py",
            "_detect_local_jax_device_count",
            "jax",
            1,
        ),
        (
            "src/simsopt_jax/backend/runtime.py",
            "_detect_global_jax_device_count",
            "jax",
            1,
        ),
        (
            "src/simsopt_jax/backend/runtime.py",
            "maybe_initialize_distributed_jax",
            "jax",
            1,
        ),
        ("src/simsopt_jax/backend/runtime.py", "apply_jax_runtime_config", "jax", 1),
        (
            "src/simsopt_jax/backend/runtime.py",
            "_CpuDeviceConstructionContext.__enter__",
            "jax",
            1,
        ),
    ),
}


ALLOWED_IMPORTS_BY_REASON = {
    "upstream_json_array_serialization": (("src/simsopt/_core/json.py", "jax", 1),),
    "direct_legacy_jax_math": (
        ("src/simsopt/field/coil.py", "jax.numpy", 1),
        ("src/simsopt/field/force.py", "jax", 1),
        ("src/simsopt/field/force.py", "jax.lax", 1),
        ("src/simsopt/field/force.py", "jax.numpy", 1),
        ("src/simsopt/field/force.py", "jax.scipy", 1),
        ("src/simsopt/field/selffield.py", "jax.numpy", 1),
        ("src/simsopt/geo/curve.py", "jax", 1),
        ("src/simsopt/geo/curve.py", "jax.numpy", 1),
        ("src/simsopt/geo/curvehelical.py", "jax.numpy", 1),
        ("src/simsopt/geo/curveobjectives.py", "jax", 1),
        ("src/simsopt/geo/curveobjectives.py", "jax.numpy", 1),
        ("src/simsopt/geo/curveplanarfourier.py", "jax.numpy", 1),
        ("src/simsopt/geo/curvexyzfourier.py", "jax.numpy", 1),
        ("src/simsopt/geo/curvexyzfouriersymmetries.py", "jax.numpy", 1),
        ("src/simsopt/geo/framedcurve.py", "jax", 1),
        ("src/simsopt/geo/framedcurve.py", "jax.numpy", 1),
        ("src/simsopt/geo/jit.py", "jax", 2),
        ("src/simsopt/geo/strain_optimization.py", "jax", 1),
        ("src/simsopt/geo/strain_optimization.py", "jax.numpy", 1),
        ("src/simsopt/geo/surfacerzfourier.py", "jax", 1),
        ("src/simsopt/geo/surfacerzfourier.py", "jax.numpy", 1),
    ),
}


ALLOWED_MARKERS_BY_REASON = {}


def _tracked_legacy_python_files():
    result = subprocess.run(
        ["git", "ls-files", LEGACY_SOURCE_ROOT],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        path
        for path in result.stdout.splitlines()
        if path.endswith(".py") and (REPO_ROOT / path).is_file()
    )


def _tracked_jax_source_python_files():
    result = subprocess.run(
        ["git", "ls-files", *JAX_SOURCE_ROOTS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        path
        for path in result.stdout.splitlines()
        if path.endswith(".py") and (REPO_ROOT / path).is_file()
    )


def _is_forbidden_import_module(module):
    return any(
        module == root or module.startswith(root + ".")
        for root in FORBIDDEN_IMPORT_ROOTS
    )


def _absolute_import_modules(node):
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return (node.module,)
    return ()


def _source_style_import_modules(node):
    if isinstance(node, ast.ImportFrom) and node.level > 0:
        prefix = "." * node.level
        if node.module:
            return (prefix + node.module,)
        return tuple(prefix + alias.name for alias in node.names)
    return _absolute_import_modules(node)


def _scan_tracked_legacy_sources(scanner):
    occurrences = Counter()
    for relative_path in _tracked_legacy_python_files():
        source_path = REPO_ROOT / relative_path
        occurrences.update(
            scanner(
                source_path.read_text(encoding="utf-8"),
                relative_path=relative_path,
            )
        )
    return occurrences


def _actual_forbidden_imports():
    return _scan_tracked_legacy_sources(_forbidden_imports_in_source)


def _forbidden_imports_in_source(source, *, relative_path):
    occurrences = Counter()
    tree = ast.parse(source, filename=relative_path)
    for node in ast.walk(tree):
        for module in _absolute_import_modules(node):
            if _is_forbidden_import_module(module):
                occurrences[(relative_path, module)] += 1
    return occurrences


def _actual_jax_native_markers():
    return _scan_tracked_legacy_sources(_jax_native_markers_in_source)


def _scan_tracked_jax_sources(scanner):
    occurrences = Counter()
    for relative_path in _tracked_jax_source_python_files():
        source_path = REPO_ROOT / relative_path
        occurrences.update(
            scanner(
                source_path.read_text(encoding="utf-8"),
                relative_path=relative_path,
            )
        )
    return occurrences


class _JaxSourceImportStyleVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path):
        self.relative_path = relative_path
        self.class_stack = []
        self.function_stack = []
        self.local_imports = Counter()
        self.dynamic_imports = Counter()

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _visit_function(self, node):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _current_function_name(self):
        if not self.function_stack:
            return None
        return ".".join((*self.class_stack, self.function_stack[-1]))

    def _record_local_import(self, module):
        function_name = self._current_function_name()
        if function_name is not None:
            self.local_imports[(self.relative_path, function_name, module)] += 1

    def visit_Import(self, node):
        for alias in node.names:
            self._record_local_import(alias.name)

    def visit_ImportFrom(self, node):
        for module in _source_style_import_modules(node):
            self._record_local_import(module)

    def visit_Call(self, node):
        func = node.func
        is_dynamic_import = (
            isinstance(func, ast.Name)
            and func.id == "__import__"
            or isinstance(func, ast.Attribute)
            and func.attr == "import_module"
            and isinstance(func.value, ast.Name)
            and func.value.id == "importlib"
        )
        if is_dynamic_import:
            function_name = self._current_function_name() or "<module>"
            self.dynamic_imports[(self.relative_path, function_name)] += 1
        self.generic_visit(node)


def _jax_source_import_style_in_source(source, *, relative_path):
    tree = ast.parse(source, filename=relative_path)
    visitor = _JaxSourceImportStyleVisitor(relative_path=relative_path)
    visitor.visit(tree)
    return visitor.local_imports, visitor.dynamic_imports


def _actual_jax_source_local_imports():
    return _scan_tracked_jax_sources(
        lambda source, *, relative_path: _jax_source_import_style_in_source(
            source,
            relative_path=relative_path,
        )[0]
    )


def _actual_jax_source_dynamic_imports():
    return _scan_tracked_jax_sources(
        lambda source, *, relative_path: _jax_source_import_style_in_source(
            source,
            relative_path=relative_path,
        )[1]
    )


def _jax_native_markers_in_source(source, *, relative_path):
    occurrences = Counter()
    for line in source.splitlines():
        if JAX_NATIVE_MARKER in line:
            occurrences[(relative_path, JAX_NATIVE_MARKER, line.strip())] += 1
    return occurrences


def _expected_imports():
    return _counter_from_grouped_allowlist(ALLOWED_IMPORTS_BY_REASON)


def _expected_markers():
    return _counter_from_grouped_allowlist(ALLOWED_MARKERS_BY_REASON)


def _expected_jax_source_local_imports():
    return _counter_from_grouped_allowlist(ALLOWED_JAX_SOURCE_LOCAL_IMPORTS_BY_REASON)


def _counter_from_grouped_allowlist(grouped_allowlist):
    occurrences = Counter()
    for entries in grouped_allowlist.values():
        for *key, count in entries:
            occurrences[tuple(key)] += count
    return occurrences


def _format_delta(title, delta):
    if not delta:
        return ""
    formatted = [title]
    for key, count in sorted(delta.items()):
        formatted.append(f"  {key!r} x{count}")
    return "\n".join(formatted)


def _assert_counter_matches_allowlist(actual, expected):
    unexpected = actual - expected
    stale = expected - actual
    message_parts = (
        _format_delta("Unexpected legacy JAX references:", unexpected),
        _format_delta("Stale allowlist entries:", stale),
    )
    message = "\n".join(part for part in message_parts if part)
    assert actual == expected, message


def test_legacy_package_jax_imports_match_ratcheting_allowlist():
    _assert_counter_matches_allowlist(
        actual=_actual_forbidden_imports(),
        expected=_expected_imports(),
    )


def test_legacy_package_jax_native_markers_match_ratcheting_allowlist():
    _assert_counter_matches_allowlist(
        actual=_actual_jax_native_markers(),
        expected=_expected_markers(),
    )


def test_import_scanner_tracks_only_absolute_jax_imports():
    source = """
import jax
import jax.numpy as jnp
from jax import jit
from simsopt_jax.core import specs
"""
    assert _forbidden_imports_in_source(
        source,
        relative_path="src/simsopt/configs/example.py",
    ) == Counter(
        {
            ("src/simsopt/configs/example.py", "jax"): 2,
            ("src/simsopt/configs/example.py", "jax.numpy"): 1,
            ("src/simsopt/configs/example.py", "simsopt_jax.core"): 1,
        }
    )


def test_import_scanner_ignores_relative_import_ast_nodes():
    assert (
        _absolute_import_modules(
            ast.ImportFrom(
                module="jax_core",
                names=[ast.alias(name="make_curve_helical_spec")],
                level=2,
            )
        )
        == ()
    )
    assert (
        _absolute_import_modules(
            ast.ImportFrom(
                module="geo.jit",
                names=[ast.alias(name="jit", asname="legacy_jit")],
                level=2,
            )
        )
        == ()
    )


def test_jax_source_import_style_scanner_tracks_function_local_relative_imports():
    source = """
def use_reference_lane():
    from . import reference

    return reference
"""
    local_imports, dynamic_imports = _jax_source_import_style_in_source(
        source,
        relative_path="src/simsopt_jax/geo/optimizers/optimizer.py",
    )

    assert local_imports == Counter(
        {
            (
                "src/simsopt_jax/geo/optimizers/optimizer.py",
                "use_reference_lane",
                ".reference",
            ): 1,
        }
    )
    assert dynamic_imports == Counter()


def test_import_scanner_ignores_text_mentions():
    source = '''
"""from simsopt_jax.core import specs"""
# import jax
value = "simsopt_jax_adapters.field.biotsavart_backend"
'''
    assert (
        _forbidden_imports_in_source(
            source,
            relative_path="src/simsopt/geo/comment_only.py",
        )
        == Counter()
    )


def test_marker_scanner_is_separate_from_import_scanner():
    source = """
field._simsopt_jax_native_field = True
from jax import numpy as jnp
"""
    relative_path = "src/simsopt/field/example.py"
    assert _forbidden_imports_in_source(
        source,
        relative_path=relative_path,
    ) == Counter({(relative_path, "jax"): 1})
    assert _jax_native_markers_in_source(
        source,
        relative_path=relative_path,
    ) == Counter(
        {
            (
                relative_path,
                JAX_NATIVE_MARKER,
                "field._simsopt_jax_native_field = True",
            ): 1,
        }
    )


def test_jax_source_packages_have_no_dynamic_imports():
    assert _actual_jax_source_dynamic_imports() == Counter()


def test_jax_source_package_local_imports_match_ratcheting_allowlist():
    _assert_counter_matches_allowlist(
        actual=_actual_jax_source_local_imports(),
        expected=_expected_jax_source_local_imports(),
    )
