"""
Import smoke tests for the JAX code path.

These tests verify that JAX modules can be imported through the real
``simsopt`` package entrypoints (not via ``importlib.util`` bypass).
They run in the no-simsoptpp environment to catch import-chain regressions.

Each test launches a fresh Python subprocess so that ``sys.modules`` is
guaranteed clean — other test modules in this repo inject package stubs
at import time, which would contaminate in-process imports.

This file also keeps a small number of process-isolated JAX runtime
regressions whose contract depends on a fresh subprocess. The historical
name stays for continuity, but larger functional subprocess programs
should live in real Python modules rather than inline ``python -c`` blobs.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import pytest

# Resolve the src/ directory relative to the repo root so subprocesses
# can import simsopt without a pip install.
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_OPTIMIZER_JAX_PATH = Path(_SRC_DIR) / "simsopt" / "geo" / "optimizer_jax.py"
_OPTIMIZER_PRIVATE_DIR = Path(_SRC_DIR) / "simsopt" / "geo" / "optimizer_jax_private"
_RUNTIME_BACKEND_PATH = Path(_SRC_DIR) / "simsopt" / "backend" / "runtime.py"
_CPU_RUN_CODE_BENCHMARK_PATH = (
    Path(_REPO_ROOT) / "benchmarks" / "cpu_run_code_benchmark.py"
)
_JAX_SUBPROCESS_CASES_PATH = (
    Path(_REPO_ROOT) / "tests" / "subprocess" / "jax_runtime_cases.py"
)
_ENTRYPOINT_RUNTIME_AUDIT_PATHS = (
    Path(_REPO_ROOT) / "benchmarks" / "biot_savart_kernel_scaling.py",
    Path(_REPO_ROOT) / "benchmarks" / "cpu_run_code_benchmark.py",
    Path(_REPO_ROOT) / "benchmarks" / "gpu_run_code_benchmark.py",
    Path(_REPO_ROOT) / "benchmarks" / "jax_derivative_benchmark.py",
    Path(_REPO_ROOT) / "benchmarks" / "jax_feasibility_spike.py",
    Path(_REPO_ROOT) / "benchmarks" / "optimistix_eval.py",
    (
        Path(_REPO_ROOT)
        / "examples"
        / "single_stage_optimization"
        / "SINGLE_STAGE"
        / "single_stage_banana_example.py"
    ),
    (
        Path(_REPO_ROOT)
        / "examples"
        / "single_stage_optimization"
        / "STAGE_2"
        / "banana_coil_solver.py"
    ),
)
_BACKEND_SELECTOR_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_JAX_DEBUG_NANS",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "SIMSOPT_JAX_COMPILATION_CACHE_DIR",
    "SIMSOPT_BACKEND",
    "STAGE2_BACKEND",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
)


def _build_clean_subprocess_env(
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for name in _BACKEND_SELECTOR_ENV_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env is not None:
        env.update(extra_env)
    return env


def _run_import_check(code, *, timeout=30, extra_env=None):
    """Run *code* in a clean subprocess and return (returncode, stderr)."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_REPO_ROOT,
        env=_build_clean_subprocess_env(extra_env),
    )
    return result.returncode, result.stderr.strip()


def _run_python_script(
    script_path: Path,
    *,
    args: Sequence[str] = (),
    timeout: int = 30,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a repo-local Python script in a clean subprocess."""
    result = subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_REPO_ROOT,
        env=_build_clean_subprocess_env(extra_env),
    )
    return result.returncode, result.stderr.strip()


def _assert_python_script_passes(
    script_path: Path,
    *,
    args: Sequence[str] = (),
    failure_message: str,
    timeout: int = 30,
    extra_env: dict[str, str] | None = None,
) -> None:
    rc, err = _run_python_script(
        script_path,
        args=args,
        timeout=timeout,
        extra_env=extra_env,
    )
    assert rc == 0, f"{failure_message}:\n{err}"


def _assert_import_check_passes(
    code,
    *,
    failure_message,
    timeout=30,
    extra_env=None,
):
    rc, err = _run_import_check(code, timeout=timeout, extra_env=extra_env)
    assert rc == 0, f"{failure_message}:\n{err}"


def _block_private_optimizer_imports():
    return """
        import importlib.abc
        import sys

        class _BlockPrivateOptimizer(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "simsopt.geo.optimizer_jax_private" or fullname.startswith(
                    "simsopt.geo.optimizer_jax_private."
                ):
                    raise ImportError("blocked private optimizer package for smoke test")
                return None

        sys.meta_path.insert(0, _BlockPrivateOptimizer())
    """


def _block_simsoptpp_imports():
    return """
        import importlib.abc
        import sys

        class _BlockSimsoptpp(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "simsoptpp" or fullname.startswith("simsoptpp."):
                    raise ImportError("blocked simsoptpp for smoke test")
                return None

        sys.meta_path.insert(0, _BlockSimsoptpp())
    """


def _strip_simsopt_editable_finders(*, include_import=True):
    import_block = "import sys\n\n" if include_import else ""
    return f"""
        {import_block}sys.meta_path = [
            finder
            for finder in sys.meta_path
            if type(finder).__module__ != "_simsopt_editable"
            and (
                not type(finder).__module__.startswith("__editable__")
                or "simsopt" not in type(finder).__module__.lower()
            )
        ]
    """


def _find_private_jax_src_usages(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    jax_names = {"jax"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax":
                    jax_names.add(alias.asname or alias.name)
    usages: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            usages.extend(
                f"{alias.name} @ L{node.lineno}"
                for alias in node.names
                if alias.name.startswith("jax._src")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("jax._src"):
                usages.append(f"{module} @ L{node.lineno}")
            elif module == "jax":
                usages.extend(
                    f"from jax import {alias.name} @ L{node.lineno}"
                    for alias in node.names
                    if alias.name == "_src"
                )
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "_src"
            and isinstance(node.value, ast.Name)
            and node.value.id in jax_names
        ):
            usages.append(f"{node.value.id}._src @ L{node.lineno}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "_src"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in jax_names
        ):
            usages.append(f'getattr({node.args[0].id}, "_src") @ L{node.lineno}')
    return usages


def _assert_no_private_jax_src_usage(path: Path, *, label: str) -> None:
    forbidden_usages = _find_private_jax_src_usages(path)
    assert forbidden_usages == [], f"{label} must not use jax._src: {forbidden_usages}"


def _find_import_line(path: Path, module_name: str) -> int | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    import_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module_name for alias in node.names):
                import_lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module == module_name:
                import_lines.append(node.lineno)
    return min(import_lines) if import_lines else None


def _find_named_call_lines(path: Path, function_name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    call_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == function_name:
                call_lines.append(node.lineno)
    return sorted(call_lines)


def test_find_private_jax_src_usages_detects_alias_attribute_access(tmp_path):
    path = tmp_path / "module.py"
    path.write_text(
        'import jax as jj\nvalue = jj._src\nshadow = getattr(jj, "_src")\n',
        encoding="utf-8",
    )

    usages = _find_private_jax_src_usages(path)

    assert "jj._src @ L2" in usages
    assert 'getattr(jj, "_src") @ L3' in usages


def test_import_package_root():
    """simsopt package imports without simsoptpp."""
    rc, err = _run_import_check("""
        import simsopt
        assert hasattr(simsopt, "__version__")
    """)
    assert rc == 0, f"import simsopt failed:\n{err}"


def test_import_package_root_without_generated_version_file():
    """Raw source imports should tolerate a missing generated _version.py."""
    init_path = Path(_SRC_DIR) / "simsopt" / "__init__.py"
    strip_editable_finders = textwrap.indent(
        textwrap.dedent(_strip_simsopt_editable_finders(include_import=False)).strip(),
        " " * 12,
    )
    _assert_import_check_passes(
        f"""
        import sys
        import tempfile
        from pathlib import Path

        src_init = Path({str(init_path)!r}).read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            src_root = Path(tmp) / "src"
            package_root = src_root / "simsopt"
            (package_root / "backend").mkdir(parents=True)
            (package_root / "_core").mkdir(parents=True)
            (package_root / "__init__.py").write_text(src_init, encoding="utf-8")
            (package_root / "backend" / "__init__.py").write_text(
                "def apply_jax_runtime_config():\\n    return None\\n\\n"
                "def should_eagerly_configure_jax():\\n    return False\\n",
                encoding="utf-8",
            )
            (package_root / "_core" / "__init__.py").write_text(
                "def make_optimizable(*args, **kwargs):\\n    return None\\n\\n"
                "def load(*args, **kwargs):\\n    return None\\n\\n"
                "def save(*args, **kwargs):\\n    return None\\n",
                encoding="utf-8",
            )
{strip_editable_finders}
            sys.path.insert(0, str(src_root))
            import simsopt

            assert Path(simsopt.__file__).resolve().is_relative_to(package_root.resolve())
            assert simsopt.__version__ == "0+unknown"
        """,
        failure_message="raw source import should not require generated _version.py",
    )


def test_repo_bootstrap_synthesizes_version_for_clean_source_tree():
    """repo_bootstrap should tolerate source trees without generated _version.py."""
    _assert_import_check_passes(
        """
        import tempfile
        from pathlib import Path

        from repo_bootstrap import bootstrap_local_simsopt

        with tempfile.TemporaryDirectory() as tmp:
            src_root = Path(tmp) / "src"
            package_root = src_root / "simsopt"
            package_root.mkdir(parents=True)
            (package_root / "__init__.py").write_text(
                "from ._version import version as __version__\\n",
                encoding="utf-8",
            )

            bootstrap_local_simsopt(src_root)

            import simsopt

            assert simsopt.__version__ == "0.0.dev0+source"
    """,
        failure_message="repo_bootstrap clean-source version smoke failed",
    )


def test_repo_bootstrap_is_idempotent_for_local_source_tree():
    """Repeated bootstrap calls must not churn class identity for local imports."""
    _assert_import_check_passes(
        """
        import importlib
        from pathlib import Path

        from repo_bootstrap import bootstrap_local_simsopt

        src_root = Path.cwd() / "src"
        bootstrap_local_simsopt(src_root)

        import simsopt
        from simsopt.configs.zoo import get_data
        from simsopt.geo import Curve

        curve_before = Curve
        package_before = simsopt

        bootstrap_local_simsopt(src_root)

        import simsopt as simsopt_after

        curve_after = importlib.import_module("simsopt.geo.curve").Curve
        base_curves, *_ = get_data("STAR_Lite-A_low")

        assert package_before is simsopt_after
        assert curve_before is curve_after
        assert isinstance(base_curves[0], curve_before)
    """,
        failure_message="repo_bootstrap should be idempotent for local source imports",
    )


def test_root_conftest_imports_without_jax_installed():
    """Root test fixtures must not fail collection in non-JAX environments."""
    _assert_import_check_passes(
        """
        import importlib.abc
        import runpy
        import sys
        from pathlib import Path

        class _BlockJax(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "jax" or fullname.startswith("jax."):
                    raise ModuleNotFoundError("blocked jax import for smoke test")
                return None

        sys.meta_path.insert(0, _BlockJax())

        conftest_path = Path.cwd() / "tests" / "conftest.py"
        module_globals = runpy.run_path(str(conftest_path), run_name="simsopt_tests_conftest")

        assert module_globals["jax"] is None
        parity_rng = module_globals["parity_rng"]
        assert parity_rng(3).randint(0, 1000) == parity_rng(3).randint(0, 1000)
        """,
        failure_message="root tests/conftest.py should import cleanly without JAX",
    )


def test_repo_bootstrap_purges_detached_local_submodules():
    """A second bootstrap must purge detached ``simsopt.*`` submodules."""
    _assert_import_check_passes(
        """
        import importlib
        import sys
        import types
        from pathlib import Path

        from repo_bootstrap import bootstrap_local_simsopt

        src_root = Path.cwd() / "src"
        bootstrap_local_simsopt(src_root)

        import simsopt

        real_module = importlib.import_module("simsopt.geo.optimizer_jax")
        fake_module = types.ModuleType("simsopt.geo.optimizer_jax")
        fake_module.marker = "fake"
        sys.modules["simsopt.geo.optimizer_jax"] = fake_module

        bootstrap_local_simsopt(src_root)

        assert "simsopt.geo.optimizer_jax" not in sys.modules

        reloaded_module = importlib.import_module("simsopt.geo.optimizer_jax")

        assert reloaded_module is not fake_module
        assert reloaded_module is not real_module
        assert Path(reloaded_module.__file__).resolve() == (
            src_root / "simsopt" / "geo" / "optimizer_jax.py"
        ).resolve()
        assert getattr(importlib.import_module("simsopt.geo"), "optimizer_jax") is reloaded_module
        assert simsopt is sys.modules["simsopt"]
    """,
        failure_message="repo_bootstrap should purge detached local submodules",
    )


def test_repo_bootstrap_strips_editable_meta_path_finders_on_fast_path():
    """Warm bootstraps must remove editable finders before later submodule imports."""
    _assert_import_check_passes(
        """
        import importlib
        import importlib.abc
        import importlib.util
        import sys
        from pathlib import Path

        from repo_bootstrap import bootstrap_local_simsopt

        src_root = Path.cwd() / "src"
        bootstrap_local_simsopt(src_root)

        import simsopt

        class _FakeEditableLoader(importlib.abc.Loader):
            def create_module(self, spec):
                del spec
                return None

            def exec_module(self, module):
                module.marker = "fake-editable"
                module.__file__ = "/tmp/fake_optimizer_jax.py"

        class _FakeEditableFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "simsopt.geo.optimizer_jax":
                    return importlib.util.spec_from_loader(
                        fullname,
                        _FakeEditableLoader(),
                    )
                return None

        _FakeEditableFinder.__module__ = "__editable___simsopt_demo"
        fake_finder = _FakeEditableFinder()
        sys.meta_path.insert(0, fake_finder)

        bootstrap_local_simsopt(src_root)

        assert fake_finder not in sys.meta_path
        assert not any(
            type(finder).__module__ == "_simsopt_editable"
            or (
                type(finder).__module__.startswith("__editable__")
                and "simsopt" in type(finder).__module__.lower()
            )
            for finder in sys.meta_path
        )

        reloaded_module = importlib.import_module("simsopt.geo.optimizer_jax")

        assert getattr(reloaded_module, "marker", None) is None
        assert Path(reloaded_module.__file__).resolve() == (
            src_root / "simsopt" / "geo" / "optimizer_jax.py"
        ).resolve()
        assert simsopt is sys.modules["simsopt"]
    """,
        failure_message="repo_bootstrap should strip editable meta_path finders",
    )


def test_repo_bootstrap_preserves_unrelated_editable_meta_path_finders():
    """Warm bootstraps must not remove editable finders for unrelated packages."""
    _assert_import_check_passes(
        """
        import importlib.abc
        import sys
        from pathlib import Path

        from repo_bootstrap import bootstrap_local_simsopt

        src_root = Path.cwd() / "src"

        class _UnrelatedEditableFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del fullname, path, target
                return None

        _UnrelatedEditableFinder.__module__ = "__editable___otherpkg"
        unrelated_finder = _UnrelatedEditableFinder()
        sys.meta_path.insert(0, unrelated_finder)

        bootstrap_local_simsopt(src_root)

        assert unrelated_finder in sys.meta_path
    """,
        failure_message="repo_bootstrap should preserve unrelated editable finders",
    )


def test_import_package_root_native_cpu_does_not_require_jax_runtime():
    """Importing package root without JAX selectors must not force a JAX import."""
    rc, err = _run_import_check("""
        import importlib.abc
        import os
        import sys

        class _BlockJax(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "jax" or fullname.startswith("jax."):
                    raise ImportError("blocked jax import for package-root smoke")
                return None

        sys.meta_path.insert(0, _BlockJax())

        import simsopt

        assert hasattr(simsopt, "__version__")
        assert os.environ["JAX_ENABLE_X64"] == "True"
    """)
    assert rc == 0, f"package root import unexpectedly required jax:\n{err}"


def test_entrypoint_runtime_helper_configures_cpu_before_import():
    _assert_import_check_passes(
        """
        import os

        from repo_bootstrap import configure_entrypoint_jax_runtime

        requested = configure_entrypoint_jax_runtime(
            ["--platform", "cpu"],
            default_platform="cuda",
            respect_existing_env=False,
        )

        import jax

        assert requested == "cpu"
        assert jax.default_backend() == "cpu"
        assert os.environ["JAX_PLATFORMS"] == "cpu"
        assert os.environ["SIMSOPT_JAX_PLATFORM"] == "cpu"
        assert os.environ["SIMSOPT_JAX_BACKEND"] == "cpu"
        assert os.environ["JAX_ENABLE_X64"] == "True"
    """,
        failure_message="entrypoint runtime helper should pin CPU before importing jax",
    )


def test_entrypoint_runtime_helper_auto_clears_stale_platform_env():
    _assert_import_check_passes(
        """
        import os

        from repo_bootstrap import configure_entrypoint_jax_runtime

        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["SIMSOPT_JAX_PLATFORM"] = "cpu"
        os.environ["SIMSOPT_JAX_BACKEND"] = "cpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        requested = configure_entrypoint_jax_runtime(
            ["--platform", "auto"],
            respect_existing_env=False,
        )

        assert requested is None
        assert "JAX_PLATFORMS" not in os.environ
        assert "SIMSOPT_JAX_PLATFORM" not in os.environ
        assert "SIMSOPT_JAX_BACKEND" not in os.environ
        assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
        assert os.environ["JAX_ENABLE_X64"] == "True"
    """,
        failure_message="entrypoint runtime helper should clear stale platform env when auto is requested",
    )


def test_run_code_benchmark_common_import_is_jax_cold():
    _assert_import_check_passes(
        f"""
        import importlib.abc

        class _BlockJax(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "jax" or fullname.startswith("jax."):
                    raise ImportError("blocked jax import for benchmark helper smoke")
                return None

        import sys

        sys.meta_path.insert(0, _BlockJax())
        sys.path.insert(0, {str(Path(_REPO_ROOT))!r})

        import benchmarks.run_code_benchmark_common as benchmark_common

        assert callable(benchmark_common.resolve_benchmark_backends)
    """,
        failure_message="run_code_benchmark_common import should not initialize jax",
    )


def test_cpu_run_code_benchmark_pins_cpu_before_import():
    _assert_import_check_passes(
        f"""
        import runpy
        import sys

        sys.argv = [{str(_CPU_RUN_CODE_BENCHMARK_PATH)!r}]
        runpy.run_path({str(_CPU_RUN_CODE_BENCHMARK_PATH)!r}, run_name="benchmark_smoke")

        import jax

        assert jax.default_backend() == "cpu"
    """,
        failure_message="cpu_run_code_benchmark should request CPU before importing jax",
    )


def test_audited_entrypoints_configure_runtime_before_importing_jax():
    for path in _ENTRYPOINT_RUNTIME_AUDIT_PATHS:
        configure_lines = _find_named_call_lines(
            path, "configure_entrypoint_jax_runtime"
        )
        first_jax_import = _find_import_line(path, "jax")

        assert configure_lines, (
            f"{path.name} must call configure_entrypoint_jax_runtime"
        )
        assert first_jax_import is not None, f"{path.name} must import jax explicitly"
        assert min(configure_lines) < first_jax_import, (
            f"{path.name} must configure the JAX runtime before importing jax"
        )


def test_programmatic_backend_selection_configures_jax_runtime():
    """The public config API should support the new mode-based backend contract."""
    rc, err = _run_import_check("""
        import simsopt.config as simsopt_config
        import simsopt.backend as backend

        cfg = simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            debug_nans=True,
            transfer_guard="log",
            compilation_cache_dir="/tmp/simsopt-jax-cache",
        )
        policy = simsopt_config.get_backend_policy()

        assert cfg.mode == "jax_cpu_parity"
        assert cfg.backend == "jax"
        assert cfg.jax_platform == "cpu"
        assert cfg.strict is True
        assert cfg.debug_nans is True
        assert cfg.transfer_guard == "log"
        assert cfg.compilation_cache_dir == "/tmp/simsopt-jax-cache"
        assert policy.mode == "jax_cpu_parity"
        assert policy.parity_mode is True
        assert policy.chunk_policy == "stable_default"
        assert policy.tolerance_tier == "parity"
        assert policy.compilation_cache_policy == "optional_persistent"
        assert policy.provenance_label == "jax_cpu_parity"
        assert policy.debug_nans is True
        assert policy.transfer_guard == "log"
        assert policy.compilation_cache_dir == "/tmp/simsopt-jax-cache"
        assert backend.get_backend_mode() == "jax_cpu_parity"
        assert backend.is_backend_strict() is True
        assert backend.get_point_chunk_size("jax_cpu_parity") == 256

        import jax

        assert jax.numpy.zeros(1).dtype == jax.numpy.float64
        assert jax.config.jax_debug_nans is True
        assert jax.config.jax_transfer_guard == "log"
        assert jax.config.jax_compilation_cache_dir == "/tmp/simsopt-jax-cache"
    """)
    assert rc == 0, f"programmatic backend config failed:\n{err}"


def test_parity_mode_defaults_transfer_guard_and_keeps_x64_enabled():
    """Parity modes should own x64 and transfer-guard defaults without extra flags."""
    _assert_import_check_passes(
        """
        import simsopt.config as simsopt_config
        import jax

        cfg = simsopt_config.set_backend("jax_cpu_parity")
        policy = simsopt_config.get_backend_policy()

        assert cfg.transfer_guard == "log"
        assert policy.transfer_guard == "log"
        assert policy.requires_x64 is True
        assert jax.config.jax_enable_x64 is True
        assert jax.config.jax_transfer_guard == "log"
        assert jax.numpy.zeros(1).dtype == jax.numpy.float64
    """,
        failure_message="parity mode guardrail contract failed",
    )


def test_env_selected_guardrails_eagerly_configure_jax_runtime():
    """Import-time eager config should honor parity x64/debug-nans/transfer-guard envs."""
    _assert_import_check_passes(
        """
        import os

        os.environ["SIMSOPT_BACKEND_MODE"] = "jax_cpu_parity"
        os.environ["SIMSOPT_JAX_DEBUG_NANS"] = "1"
        os.environ["SIMSOPT_JAX_TRANSFER_GUARD"] = "log"

        import simsopt
        import simsopt.config as simsopt_config
        import jax

        policy = simsopt_config.get_backend_policy()

        assert policy.mode == "jax_cpu_parity"
        assert policy.requires_x64 is True
        assert policy.debug_nans is True
        assert policy.transfer_guard == "log"
        assert jax.config.jax_enable_x64 is True
        assert jax.config.jax_debug_nans is True
        assert jax.config.jax_transfer_guard == "log"
        assert jax.numpy.zeros(1).dtype == jax.numpy.float64
    """,
        failure_message="eager guardrail config failed",
    )


def test_transfer_guard_disallow_rejects_implicit_host_to_device_jit_inputs():
    """Disallow mode should catch implicit NumPy->JAX transfers at a JIT boundary."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        import jax
        import jax.numpy as jnp

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        fn = jax.jit(lambda x: x + 1.0)

        try:
            fn(np.ones((2,), dtype=np.float64))
        except Exception as exc:
            message = str(exc)
            assert "host-to-device" in message
        else:
            raise AssertionError("expected transfer guard to reject implicit host input")
    """,
        failure_message="transfer-guard disallow smoke failed",
    )


def test_transfer_guard_disallow_allows_target_backend_x64_guard():
    """Target-lane x64 checks must not allocate JAX arrays under disallow mode."""
    _assert_import_check_passes(
        """
        import simsopt.config as simsopt_config
        from simsopt.geo.optimizer_jax import require_target_backend_x64

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        require_target_backend_x64("ondevice")
    """,
        failure_message="target-backend x64 guard should be transfer-clean",
    )


def test_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes():
    """Private ondevice L-BFGS lanes must stay transfer-clean under disallow."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo.optimizer_jax import (
            PRIVATE_OPTIMIZER_JAX_VERSION,
            jax_minimize,
            private_optimizer_runtime_is_supported,
        )

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )
        if not private_optimizer_runtime_is_supported(jax.__version__):
            raise SystemExit(0)

        half = jax.device_put(np.asarray(0.5, dtype=np.float64))

        def quad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return half * jnp.dot(x, x)

        def quad_value_and_grad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return half * jnp.dot(x, x), x

        x0 = jnp.asarray(np.array([1.0, -2.0], dtype=np.float64))
        result = jax_minimize(quad, x0, method="lbfgs-ondevice", maxiter=5)
        result_vg = jax_minimize(
            quad_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            maxiter=5,
            value_and_grad=True,
        )

        assert result.success is True
        assert result_vg.success is True
        assert float(result.fun) < float(quad(x0))
        assert float(result_vg.fun) < float(quad(x0))
    """,
        failure_message="lbfgs-ondevice transfer-guard smoke failed",
    )


def _assert_ondevice_optimizer_reuses_compiled_solver(method: str) -> None:
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("compile-count", method),
        failure_message=f"{method} compile-count smoke failed",
        extra_env={"JAX_ENABLE_COMPILATION_CACHE": "0"},
    )


def test_lbfgs_ondevice_reuses_compiled_solver_across_identical_calls():
    """Repeated identical lbfgs-ondevice calls must not recompile run_solver."""
    _assert_ondevice_optimizer_reuses_compiled_solver("lbfgs-ondevice")


def test_bfgs_ondevice_reuses_compiled_solver_across_identical_calls():
    """Repeated identical bfgs-ondevice calls must not recompile run_solver."""
    _assert_ondevice_optimizer_reuses_compiled_solver("bfgs-ondevice")


def test_ondevice_solver_cache_respects_mutable_objective_state():
    """Unmarked mutable callables must retrace so updated host state is observed."""
    _assert_python_script_passes(
        _JAX_SUBPROCESS_CASES_PATH,
        args=("mutable-objective-state",),
        failure_message="ondevice solver cache must not freeze mutable objective state",
    )


def test_transfer_guard_disallow_allows_adam_ondevice_quadratic_smokes():
    """Public ondevice Adam lane must stay transfer-clean under disallow."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo.optimizer_jax import (
            PRIVATE_OPTIMIZER_JAX_VERSION,
            jax_minimize,
            private_optimizer_runtime_is_supported,
        )

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )
        if not private_optimizer_runtime_is_supported(jax.__version__):
            raise SystemExit(0)

        half = jax.device_put(np.asarray(0.5, dtype=np.float64))
        target = jax.device_put(np.asarray([0.25, -0.75], dtype=np.float64))

        def quad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            diff = x - target
            return half * jnp.dot(diff, diff)

        x0 = jnp.asarray(np.array([1.5, -2.5], dtype=np.float64))
        result = jax_minimize(
            quad,
            x0,
            method="adam-ondevice",
            maxiter=200,
            tol=1e-5,
            options={"step_size": 0.05},
        )

        assert result.success is True
        assert float(result.fun) < float(quad(x0))
        assert np.allclose(np.asarray(result.x), np.asarray([0.25, -0.75]), atol=1e-4)
    """,
        failure_message="adam-ondevice transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_lm_ondevice_quadratic_smokes():
    """Ondevice LM least-squares must stay transfer-clean under disallow."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo.optimizer_jax import (
            PRIVATE_OPTIMIZER_JAX_VERSION,
            jax_least_squares,
            private_optimizer_runtime_is_supported,
        )

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )
        if not private_optimizer_runtime_is_supported(jax.__version__):
            raise SystemExit(0)

        x0 = jnp.asarray(np.array([1.5, -2.5], dtype=np.float64))
        target = jax.device_put(np.asarray([0.25, -0.75], dtype=np.float64))

        def residual(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return x - target

        result = jax_least_squares(residual, x0, method="lm-ondevice", maxiter=8)

        assert result.success is True
        assert float(result.fun) < 0.5 * float(jnp.dot(residual(x0), residual(x0)))
        assert np.allclose(np.asarray(result.x), np.asarray([0.25, -0.75]))
    """,
        failure_message="lm-ondevice transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_ondevice_loops_with_host_closure_constants():
    """Ondevice optimizer loops must compile even when objectives capture host arrays."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo.optimizer_jax import (
            PRIVATE_OPTIMIZER_JAX_VERSION,
            jax_minimize,
            private_optimizer_runtime_is_supported,
        )

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )
        if not private_optimizer_runtime_is_supported(jax.__version__):
            raise SystemExit(0)

        captured = np.arange(9, dtype=np.float64)
        half = jax.device_put(np.asarray(0.5, dtype=np.float64))
        x0 = jax.device_put(np.ones(9, dtype=np.float64))

        def closure_quad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            target = jax.device_put(captured)
            diff = x - target
            return half * jnp.dot(diff, diff)

        baseline = float(jax.device_get(closure_quad(x0)))
        bfgs = jax_minimize(closure_quad, x0, method="bfgs-ondevice", maxiter=5)
        lbfgs = jax_minimize(closure_quad, x0, method="lbfgs-ondevice", maxiter=5)

        assert float(bfgs.fun) < baseline
        assert float(lbfgs.fun) < baseline
        assert int(bfgs.nit) > 0
        assert int(lbfgs.nit) > 0
    """,
        failure_message="ondevice optimizer loop closure-constant smoke failed",
    )


def test_transfer_guard_disallow_allows_gpu_ondevice_loops_with_host_constants():
    """GPU ondevice optimizers must not capture device-backed compile constants."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo.optimizer_jax import (
            PRIVATE_OPTIMIZER_JAX_VERSION,
            jax_minimize,
            private_optimizer_runtime_is_supported,
        )

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None or not private_optimizer_runtime_is_supported(jax.__version__):
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )

        captured = np.arange(9, dtype=np.float64)
        x0 = jax.device_put(np.ones(9, dtype=np.float64), device=gpu)

        def closure_quad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            target = jax.device_put(captured, device=gpu)
            diff = x - target
            half = jax.device_put(np.asarray(0.5, dtype=np.float64), device=gpu)
            return half * jnp.dot(diff, diff)

        baseline = float(jax.device_get(closure_quad(x0)))
        bfgs = jax_minimize(closure_quad, x0, method="bfgs-ondevice", maxiter=5)
        lbfgs = jax_minimize(closure_quad, x0, method="lbfgs-ondevice", maxiter=5)

        assert float(bfgs.fun) < baseline
        assert float(lbfgs.fun) < baseline
        assert int(bfgs.nit) > 0
        assert int(lbfgs.nit) > 0
    """,
        failure_message="GPU ondevice optimizer transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_traceable_newton_with_host_closure_constants():
    """Traceable Newton helpers must not eagerly cross host/device boundaries."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo.optimizer_jax import newton_polish_traceable

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )

        captured = np.arange(9, dtype=np.float64)
        half = jax.device_put(np.asarray(0.5, dtype=np.float64))
        x0 = jax.device_put(np.ones(9, dtype=np.float64))

        def closure_quad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            target = jax.device_put(captured)
            diff = x - target
            return half * jnp.dot(diff, diff)

        result = newton_polish_traceable(closure_quad, x0, maxiter=3, tol=1e-9)

        assert result["x"].shape == x0.shape
        assert result["grad"].shape == x0.shape
        assert jnp.all(jnp.isfinite(result["x"]))
        assert jnp.all(jnp.isfinite(result["grad"]))
    """,
        failure_message="traceable Newton closure-constant smoke failed",
    )


def test_transfer_guard_disallow_allows_boozer_residual_host_scalars():
    """Boozer residual kernels must explicitly materialize legacy host scalars."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo.boozer_residual_jax import (
            boozer_residual_scalar,
            boozer_residual_vector,
        )

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )

        B = jax.device_put(np.ones((2, 3, 3), dtype=np.float64))
        xphi = jax.device_put(np.full((2, 3, 3), 2.0, dtype=np.float64))
        xtheta = jax.device_put(np.full((2, 3, 3), -0.5, dtype=np.float64))
        scalar_value = boozer_residual_scalar(1.25, -0.2, B, xphi, xtheta, True)
        vector_value = boozer_residual_vector(1.25, -0.2, B, xphi, xtheta, True)

        assert scalar_value.shape == ()
        assert vector_value.shape == (18,)
        assert jnp.all(jnp.isfinite(vector_value))
    """,
        failure_message="boozer residual host-scalar transfer smoke failed",
    )


def test_transfer_guard_disallow_allows_biot_savart_point_chunking():
    """Point-chunked Biot-Savart kernels must stay traceable under JAX loops."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.jax_core.biotsavart import biot_savart_B

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )

        points = jax.device_put(np.arange(257 * 3, dtype=np.float64).reshape(257, 3) * 1e-3)
        gammas = jax.device_put(np.linspace(0.0, 1.0, 2 * 8 * 3, dtype=np.float64).reshape(2, 8, 3))
        gammadashs = jax.device_put(np.full((2, 8, 3), 0.25, dtype=np.float64))
        currents = jax.device_put(np.array([1.0, -0.5], dtype=np.float64))

        B = biot_savart_B(points, gammas, gammadashs, currents)

        assert B.shape == (257, 3)
        assert np.all(np.isfinite(np.asarray(B)))
    """,
        failure_message="Biot-Savart point-chunking smoke failed",
    )


def test_transfer_guard_disallow_allows_grouped_biot_savart_gpu_spec_eval():
    """GPU grouped-field kernels must not close over device-backed selector constants."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.jax_core.field import (
            grouped_biot_savart_B_from_spec,
            grouped_coil_set_spec_from_lists,
        )

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None:
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )

        points = jax.device_put(
            np.linspace(0.0, 1.0, 4 * 3, dtype=np.float64).reshape(4, 3),
            device=gpu,
        )
        gamma = jax.device_put(
            np.linspace(0.2, 0.8, 8 * 3, dtype=np.float64).reshape(8, 3),
            device=gpu,
        )
        gammadash = jax.device_put(
            np.full((8, 3), 0.1, dtype=np.float64),
            device=gpu,
        )
        current = jax.device_put(np.asarray(1.25, dtype=np.float64), device=gpu)

        coil_spec = grouped_coil_set_spec_from_lists([gamma], [gammadash], [current])
        B = grouped_biot_savart_B_from_spec(points, coil_spec)

        assert B.shape == (4, 3)
        assert np.all(np.isfinite(np.asarray(jax.device_get(B))))
    """,
        failure_message="grouped Biot-Savart GPU spec transfer-guard smoke failed",
    )


def test_grouped_biot_savart_accepts_explicit_point_sharding():
    """Grouped-field kernels should accept explicitly sharded point clouds."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from simsopt.jax_core.field import (
            grouped_biot_savart_B_from_spec,
            grouped_coil_set_spec_from_lists,
        )

        mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("d",))
        points = jax.device_put(
            np.linspace(0.0, 1.0, 4 * 3, dtype=np.float64).reshape(4, 3),
            NamedSharding(mesh, P("d", None)),
        )
        gamma = jax.device_put(
            np.linspace(0.2, 0.8, 8 * 3, dtype=np.float64).reshape(8, 3),
        )
        gammadash = jax.device_put(
            np.full((8, 3), 0.1, dtype=np.float64),
        )
        current = jax.device_put(np.asarray(1.25, dtype=np.float64))

        coil_spec = grouped_coil_set_spec_from_lists([gamma], [gammadash], [current])
        B = grouped_biot_savart_B_from_spec(points, coil_spec)

        assert B.shape == (4, 3)
        assert isinstance(B.sharding, NamedSharding)
        assert jax.numpy.all(jax.numpy.isfinite(B))
    """,
        failure_message="grouped Biot-Savart explicit point sharding smoke failed",
    )


def test_pairwise_penalty_accepts_explicit_row_sharding():
    """Pairwise penalty kernels should accept explicitly sharded row-owned inputs."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from simsopt.geo.curveobjectives import pairwise_min_distance_pure

        mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("d",))
        points_a = jax.device_put(
            np.linspace(0.0, 0.9, 4 * 3, dtype=np.float64).reshape(4, 3),
            NamedSharding(mesh, P("d", None)),
        )
        points_b = np.linspace(0.1, 0.7, 3 * 3, dtype=np.float64).reshape(3, 3)

        value = pairwise_min_distance_pure(points_a, points_b, chunk_size=2)

        assert np.isfinite(float(value))
        assert float(value) >= 0.0
    """,
        failure_message="pairwise penalty explicit row sharding smoke failed",
    )


def test_transfer_guard_disallow_allows_grouped_biot_savart_gpu_current_arrays():
    """Grouped coil specs should accept staged current arrays without Python indexing."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.jax_core.field import (
            grouped_biot_savart_B_from_spec,
            grouped_coil_set_spec_from_lists,
        )

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None:
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )

        points = jax.device_put(
            np.linspace(0.0, 1.0, 4 * 3, dtype=np.float64).reshape(4, 3),
            device=gpu,
        )
        gamma0 = jax.device_put(
            np.linspace(0.2, 0.8, 8 * 3, dtype=np.float64).reshape(8, 3),
            device=gpu,
        )
        gamma1 = jax.device_put(
            np.linspace(0.3, 0.9, 8 * 3, dtype=np.float64).reshape(8, 3),
            device=gpu,
        )
        gammadash0 = jax.device_put(
            np.full((8, 3), 0.1, dtype=np.float64),
            device=gpu,
        )
        gammadash1 = jax.device_put(
            np.full((8, 3), 0.15, dtype=np.float64),
            device=gpu,
        )
        currents = jax.device_put(
            np.asarray([1.25, -0.75], dtype=np.float64),
            device=gpu,
        )

        coil_spec = grouped_coil_set_spec_from_lists(
            (gamma0, gamma1),
            (gammadash0, gammadash1),
            currents,
        )
        B = grouped_biot_savart_B_from_spec(points, coil_spec)

        assert B.shape == (4, 3)
        assert np.all(np.isfinite(np.asarray(jax.device_get(B))))
    """,
        failure_message="grouped Biot-Savart GPU current-array transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_grouped_biot_savart_host_spec_vjp():
    """Host-backed grouped coil specs must remain usable in eager VJP paths."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.jax_core.field import (
            grouped_biot_savart_B_from_spec,
            grouped_coil_set_spec_from_lists,
        )

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None:
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )

        points = jax.device_put(
            np.linspace(0.0, 1.0, 4 * 3, dtype=np.float64).reshape(4, 3),
            device=gpu,
        )
        gamma = np.linspace(0.2, 0.8, 8 * 3, dtype=np.float64).reshape(8, 3)
        gammadash = np.full((8, 3), 0.1, dtype=np.float64)
        current = np.asarray(1.25, dtype=np.float64)

        coil_spec = grouped_coil_set_spec_from_lists([gamma], [gammadash], [current])

        def objective(eval_points):
            return jnp.sum(grouped_biot_savart_B_from_spec(eval_points, coil_spec))

        value, pullback = jax.vjp(objective, points)
        grad = pullback(
            jax.device_put(np.asarray(1.0, dtype=np.float64), device=gpu)
        )[0]

        assert np.isfinite(float(jax.device_get(value)))
        assert grad.shape == points.shape
        assert bool(jax.device_get(jnp.all(jnp.isfinite(grad))))
    """,
        failure_message="grouped Biot-Savart host-spec VJP transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_preserves_shifted_grid_axis_sample():
    """Shifted quadrature grids must use the sampled surface point for axis-z."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier
        from simsopt.geo.boozersurface_jax import _surface_sample_z

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )

        rz = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.array([0.17]),
            quadpoints_theta=np.array([0.31]),
        )
        rz.set_zs(1, 0, 1.0)
        rz_gamma = np.asarray(rz.gamma(), dtype=np.float64)
        rz_sample = float(jax.device_get(_surface_sample_z(jax.device_put(rz_gamma))))
        assert np.isclose(rz_sample, float(rz_gamma[0, 0, 2]))

        xyz = SurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            stellsym=True,
            nfp=1,
            quadpoints_phi=np.array([0.23]),
            quadpoints_theta=np.array([0.37]),
        )
        xyz_dofs = xyz.get_dofs().copy()
        for i in range(min(6, xyz_dofs.size)):
            xyz_dofs[-(i + 1)] += 0.01 * (i + 1)
        xyz.set_dofs(xyz_dofs)
        xyz_gamma = np.asarray(xyz.gamma(), dtype=np.float64)
        xyz_sample = float(jax.device_get(_surface_sample_z(jax.device_put(xyz_gamma))))
        assert np.isclose(xyz_sample, float(xyz_gamma[0, 0, 2]))
    """,
        failure_message="shifted-grid axis sample smoke failed",
    )


def test_transfer_guard_disallow_allows_curvecwsfouriercpp_init():
    """CurveCWSFourierCPP should explicitly materialize quadpoints under disallow mode."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier
        from simsopt.geo.curvecwsfourier import CurveCWSFourierCPP

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
        surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(64) / 64,
            quadpoints_theta=np.arange(64) / 64,
        )
        curve = CurveCWSFourierCPP(quadpoints, 3, surf, G=0, H=0)
        assert curve.numquadpoints == 33
    """,
        failure_message="CurveCWSFourierCPP transfer-guard init smoke failed",
    )


def test_transfer_guard_disallow_allows_curvecwsfouriercpp_curve_length_gradient():
    """CurveCWSFourierCPP length gradient should use explicit host/device boundaries."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier
        from simsopt.geo.curveobjectives import CurveLength
        from simsopt.geo.curvecwsfourier import CurveCWSFourierCPP

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
        surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(64) / 64,
            quadpoints_theta=np.arange(64) / 64,
        )
        curve = CurveCWSFourierCPP(quadpoints, 3, surf, G=0, H=0)
        curve.set("thetas(1)", 0.1)
        curve.set("phic(1)", 0.05)
        value = CurveLength(curve).J()
        grad = CurveLength(curve).dJ(partials=True)(curve)
        assert np.isfinite(float(value))
        assert grad.shape == (curve.dof_size,)
        assert np.all(np.isfinite(grad))
    """,
        failure_message="CurveCWSFourierCPP CurveLength transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_curvecwsfouriercpp_curve_distance_gradient():
    """CurveCWSFourierCPP distance gradients should materialize JAX geometry before slicing."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.geo.curveobjectives import CurveCurveDistance
        from simsopt.geo.curvecwsfourier import CurveCWSFourierCPP

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
        surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(64) / 64,
            quadpoints_theta=np.arange(64) / 64,
        )
        banana_curve = CurveCWSFourierCPP(quadpoints, 3, surf, G=0, H=0)
        banana_curve.set("phic(0)", 0.05)
        banana_curve.set("thetas(1)", 0.1)
        tf_curves = create_equally_spaced_curves(
            2,
            5,
            stellsym=False,
            R0=1.0,
            R1=0.35,
            order=1,
            numquadpoints=33,
        )
        objective = CurveCurveDistance([banana_curve, *tf_curves], 0.05)
        value = objective.J()
        grad = objective.dJ(partials=True)(banana_curve)
        assert np.isfinite(float(value))
        assert grad.shape == (banana_curve.dof_size,)
        assert np.all(np.isfinite(grad))
    """,
        failure_message="CurveCWSFourierCPP CurveCurveDistance transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_stage2_target_objective_host_closure_constants():
    """Direct Stage 2 objective evaluation must tolerate strict transfer guard."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.field import Coil, Current, ScaledCurrent, coils_via_symmetries
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.geo.curvecwsfourier import CurveCWSFourierCPP
        from simsopt.objectives.stage2_target_objective_jax import (
            Stage2PenaltyConfig,
            build_stage2_target_objective,
        )

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None:
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )

        eval_surf = SurfaceRZFourier.from_nphi_ntheta(
            nfp=1,
            stellsym=True,
            mpol=1,
            ntor=0,
            nphi=16,
            ntheta=16,
        )
        eval_dofs = eval_surf.get_dofs()
        eval_dofs[0] = 1.0
        eval_dofs[1] = 0.15
        eval_surf.set_dofs(eval_dofs)

        coil_surf = SurfaceRZFourier.from_nphi_ntheta(
            nfp=1,
            stellsym=True,
            mpol=1,
            ntor=0,
            nphi=16,
            ntheta=16,
        )
        coil_dofs = coil_surf.get_dofs()
        coil_dofs[0] = 1.15
        coil_dofs[1] = 0.18
        coil_surf.set_dofs(coil_dofs)

        tf_curves = create_equally_spaced_curves(
            2,
            1,
            stellsym=False,
            R0=1.0,
            R1=0.25,
            order=1,
            numquadpoints=33,
        )
        tf_currents = [Current(1.0) * 1e5 for _ in tf_curves]
        for tf_curve in tf_curves:
            tf_curve.fix_all()
        for tf_current in tf_currents:
            tf_current.fix_all()
        tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

        quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
        banana_curve = CurveCWSFourierCPP(quadpoints, 2, coil_surf, G=0, H=0)
        banana_curve.set("phic(0)", 0.05)
        banana_curve.set("thetac(0)", 0.45)
        banana_curve.set("phic(1)", 0.03)
        banana_curve.set("thetas(1)", 0.08)
        banana_current = Current(1.0)
        banana_coils = coils_via_symmetries(
            [banana_curve],
            [ScaledCurrent(banana_current, 1e4)],
            coil_surf.nfp,
            coil_surf.stellsym,
        )

        bundle = build_stage2_target_objective(
            surface=eval_surf,
            tf_coils=tf_coils,
            banana_coils=banana_coils,
            banana_curve=banana_curve,
            penalty_config=Stage2PenaltyConfig(
                squared_flux_weight=1.0,
                length_weight=0.0005,
                length_target=1.75,
                cc_weight=100.0,
                cc_threshold=0.05,
                curvature_weight=0.0001,
                curvature_threshold=40.0,
                curvature_p_norm=4,
            ),
        )

        dofs = np.concatenate(
            (
                np.array([1.0], dtype=np.float64),
                np.asarray(banana_curve.get_dofs(), dtype=np.float64),
            )
        )
        dofs_jax = jax.device_put(dofs, device=gpu)
        value = bundle.objective(dofs_jax)

        assert np.isfinite(float(jax.device_get(value)))
    """,
        failure_message="Stage 2 direct objective transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_stage2_target_objective_ondevice_entry():
    """The real ondevice optimizer entry must tolerate strict transfer guard."""
    _assert_import_check_passes(
        """
        import numpy as np
        import jax
        import simsopt.config as simsopt_config
        from simsopt.geo import (
            SurfaceRZFourier,
            CurveCWSFourierCPP,
            create_equally_spaced_curves,
        )
        from simsopt.field import Current, Coil, coils_via_symmetries
        from simsopt.field.coil import ScaledCurrent
        from simsopt.geo.optimizer_jax import jax_minimize
        from simsopt.objectives.stage2_target_objective_jax import (
            Stage2PenaltyConfig,
            build_stage2_target_objective,
        )

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None:
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )

        eval_surf = SurfaceRZFourier.from_nphi_ntheta(
            nfp=1,
            stellsym=True,
            mpol=1,
            ntor=0,
            nphi=16,
            ntheta=16,
        )
        eval_dofs = eval_surf.get_dofs()
        eval_dofs[0] = 1.0
        eval_dofs[1] = 0.15
        eval_surf.set_dofs(eval_dofs)

        coil_surf = SurfaceRZFourier.from_nphi_ntheta(
            nfp=1,
            stellsym=True,
            mpol=1,
            ntor=0,
            nphi=16,
            ntheta=16,
        )
        coil_dofs = coil_surf.get_dofs()
        coil_dofs[0] = 1.15
        coil_dofs[1] = 0.18
        coil_surf.set_dofs(coil_dofs)

        tf_curves = create_equally_spaced_curves(
            2,
            1,
            stellsym=False,
            R0=1.0,
            R1=0.25,
            order=1,
            numquadpoints=33,
        )
        tf_currents = [Current(1.0) * 1e5 for _ in tf_curves]
        for tf_curve in tf_curves:
            tf_curve.fix_all()
        for tf_current in tf_currents:
            tf_current.fix_all()
        tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

        quadpoints = np.linspace(0.0, 1.0, 33, endpoint=False)
        banana_curve = CurveCWSFourierCPP(quadpoints, 2, coil_surf, G=0, H=0)
        banana_curve.set("phic(0)", 0.05)
        banana_curve.set("thetac(0)", 0.45)
        banana_curve.set("phic(1)", 0.03)
        banana_curve.set("thetas(1)", 0.08)
        banana_current = Current(1.0)
        banana_coils = coils_via_symmetries(
            [banana_curve],
            [ScaledCurrent(banana_current, 1e4)],
            coil_surf.nfp,
            coil_surf.stellsym,
        )

        bundle = build_stage2_target_objective(
            surface=eval_surf,
            tf_coils=tf_coils,
            banana_coils=banana_coils,
            banana_curve=banana_curve,
            penalty_config=Stage2PenaltyConfig(
                squared_flux_weight=1.0,
                length_weight=0.0005,
                length_target=1.75,
                cc_weight=100.0,
                cc_threshold=0.05,
                curvature_weight=0.0001,
                curvature_threshold=40.0,
                curvature_p_norm=4,
            ),
        )

        dofs = np.concatenate(
            (
                np.array([1.0], dtype=np.float64),
                np.asarray(banana_curve.get_dofs(), dtype=np.float64),
            )
        )
        dofs_jax = jax.device_put(dofs, device=gpu)
        _, vjp = jax.vjp(bundle.objective, dofs_jax)
        grad = vjp(jax.device_put(np.array(1.0, dtype=np.float64), device=gpu))[0]
        result = jax_minimize(bundle.objective, dofs_jax, method="lbfgs-ondevice", maxiter=1)

        assert np.isfinite(float(jax.device_get(bundle.objective(dofs_jax))))
        assert np.all(np.isfinite(np.asarray(jax.device_get(grad), dtype=np.float64)))
        assert hasattr(result, "success")
        """,
        failure_message="Stage 2 ondevice transfer-guard entry smoke failed",
        timeout=120,
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_transfer_guard_disallow_allows_gamma_2d_eager_host_constants():
    """Eager curve geometry helpers must keep host literals explicit under strict guard."""
    _assert_import_check_passes(
        """
        import numpy as np
        import jax
        import simsopt.config as simsopt_config
        from simsopt.geo.curve import gamma_2d

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None:
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )
        modes = np.zeros(10, dtype=np.float64)
        qpts = np.linspace(0.0, 1.0, 8, endpoint=False)
        phi, theta = gamma_2d(modes, qpts, 2, G=1, H=0)

        assert phi.shape == (8,)
        assert theta.shape == (8,)
        """,
        failure_message="gamma_2d strict transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_closed_curve_self_intersection_summary():
    """Strict GPU geometry probes must not materialize shape scalars on the host."""
    _assert_import_check_passes(
        """
        import numpy as np
        import jax
        import simsopt.config as simsopt_config
        from simsopt.jax_core.curve_geometry import closed_curve_self_intersection_summary

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None:
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )
        gamma = jax.device_put(
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                ),
                dtype=np.float64,
            ),
            device=gpu,
        )
        summary = closed_curve_self_intersection_summary(gamma, neighbor_skip=1)
        min_distance = jax.device_get(summary[0])
        penalty = jax.device_get(summary[2])
        violation = jax.device_get(summary[3])

        assert np.isfinite(float(min_distance))
        assert np.isfinite(float(penalty))
        assert bool(violation)
        """,
        failure_message="closed-curve self-intersection strict transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_surface_xyztensorfourier_gamma_from_dofs():
    """SurfaceXYZTensorFourier geometry should not close over device constants."""
    _assert_import_check_passes(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceXYZTensorFourier
        from simsopt.geo.surface_fourier_jax import (
            stellsym_scatter_indices,
            surface_gamma_from_dofs,
        )

        gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
        if gpu is None:
            raise SystemExit(0)

        simsopt_config.set_backend(
            "jax_gpu_fast",
            strict=True,
            transfer_guard="disallow",
        )
        surf = SurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            stellsym=True,
            nfp=1,
            quadpoints_phi=np.array([0.23, 0.41]),
            quadpoints_theta=np.array([0.37, 0.59]),
        )
        dofs = np.asarray(surf.get_dofs(), dtype=np.float64)
        scatter = stellsym_scatter_indices(surf.mpol, surf.ntor)
        gamma_fn = lambda d: surface_gamma_from_dofs(
            d,
            surf.quadpoints_phi,
            surf.quadpoints_theta,
            surf.mpol,
            surf.ntor,
            surf.nfp,
            surf.stellsym,
            scatter,
        )

        gamma = jax.jit(gamma_fn)(jax.device_put(dofs, device=gpu))

        assert gamma.shape == (2, 2, 3)
        assert bool(jax.device_get(jnp.all(jnp.isfinite(gamma))))
        """,
        failure_message="SurfaceXYZTensorFourier gamma strict transfer-guard smoke failed",
        timeout=120,
        extra_env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def test_transfer_guard_disallow_allows_coil_symmetry_spec_identity_default():
    """Coil symmetry defaults should build the identity rotation explicitly."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.jax_core.specs import make_coil_symmetry_spec

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        symmetry = make_coil_symmetry_spec(scale=2.5)
        assert symmetry.rotmat.shape == (3, 3)
        assert np.allclose(np.asarray(symmetry.rotmat), np.eye(3))
        assert symmetry.has_rotation is False
    """,
        failure_message="coil symmetry identity default should be transfer-clean",
    )


def _assert_transfer_guard_curve_objective_smoke(
    *,
    objective_imports: str,
    setup_code: str,
    objective_code: str,
    failure_message: str,
):
    objective_code = textwrap.indent(textwrap.dedent(objective_code).strip(), " " * 8)
    setup_code = textwrap.indent(textwrap.dedent(setup_code).strip(), " " * 8)
    _assert_import_check_passes(
        f"""
        import numpy as np
        import simsopt.config as simsopt_config
{textwrap.indent(textwrap.dedent(objective_imports).strip(), " " * 8)}

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
{setup_code}

{objective_code}
    """,
        failure_message=failure_message,
    )


_TRANSFER_GUARD_CURVE_VALUE_IMPORTS = """
from simsopt.geo import (
    FrameRotation,
    FramedCurveCentroid,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.geo.curveobjectives import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    FramedCurveTwist,
    LpCurveCurvature,
    LpCurveCurvatureBarrier,
    LpCurveTorsion,
)
"""


_TRANSFER_GUARD_CURVE_VALUE_SETUP = """
curves = create_equally_spaced_curves(
    2,
    1,
    stellsym=False,
    R0=1.0,
    R1=0.2,
    order=3,
    numquadpoints=33,
)
surface = SurfaceRZFourier(
    nfp=1,
    stellsym=False,
    mpol=1,
    ntor=0,
    quadpoints_phi=np.arange(16) / 16,
    quadpoints_theta=np.arange(16) / 16,
)
rotation = FrameRotation(curves[0].quadpoints, order=1)
rotation.x = np.array([0.1, -0.2, 0.05])
"""


_TRANSFER_GUARD_CURVE_GRADIENT_IMPORTS = """
from simsopt.geo import (
    FrameRotation,
    FramedCurveCentroid,
    create_equally_spaced_curves,
)
from simsopt.geo.curveobjectives import (
    FramedCurveTwist,
    LpCurveCurvatureBarrier,
    LpCurveTorsion,
)
"""


_TRANSFER_GUARD_CURVE_GRADIENT_SETUP = """
curves = create_equally_spaced_curves(
    2,
    1,
    stellsym=False,
    R0=1.0,
    R1=0.2,
    order=3,
    numquadpoints=33,
)
rotation = FrameRotation(curves[0].quadpoints, order=1)
rotation.x = np.array([0.1, -0.2, 0.05])
"""


@pytest.mark.parametrize(
    ("label", "code"),
    [
        (
            "CurveLength",
            """
            value = CurveLength(curves[0]).J()
            assert np.isfinite(float(value))
            """,
        ),
        (
            "LpCurveCurvature",
            """
            value = LpCurveCurvature(curves[0], p=4, threshold=10.0).J()
            assert np.isfinite(float(value))
            """,
        ),
        (
            "CurveCurveDistance",
            """
            value = CurveCurveDistance(curves, 0.05).J()
            assert np.isfinite(float(value))
            """,
        ),
        (
            "CurveSurfaceDistance",
            """
            value = CurveSurfaceDistance(curves, surface, 0.02).J()
            assert np.isfinite(float(value))
            """,
        ),
        (
            "LpCurveCurvatureBarrier",
            """
            value = LpCurveCurvatureBarrier(curves[0], threshold=10.0).J()
            assert np.isfinite(float(value))
            """,
        ),
        (
            "LpCurveTorsion",
            """
            value = LpCurveTorsion(curves[0], p=4, threshold=10.0).J()
            assert np.isfinite(float(value))
            """,
        ),
        (
            "FramedCurveTwist",
            """
            value = FramedCurveTwist(
                FramedCurveCentroid(curves[0], rotation),
                f="lp",
                p=2,
            ).J()
            assert np.isfinite(float(value))
            """,
        ),
    ],
)
def test_transfer_guard_disallow_allows_legacy_curve_objective_values(label, code):
    """Legacy curve objectives must use explicit host/device boundaries under disallow."""
    _assert_transfer_guard_curve_objective_smoke(
        objective_imports=_TRANSFER_GUARD_CURVE_VALUE_IMPORTS,
        setup_code=_TRANSFER_GUARD_CURVE_VALUE_SETUP,
        objective_code=code,
        failure_message=f"{label} transfer-guard value smoke failed",
    )


@pytest.mark.parametrize(
    ("label", "code"),
    [
        (
            "LpCurveCurvatureBarrier",
            """
            grad = np.asarray(
                LpCurveCurvatureBarrier(curves[0], threshold=10.0).dJ(),
                dtype=float,
            )
            assert grad.size > 0
            assert np.all(np.isfinite(grad))
            """,
        ),
        (
            "LpCurveTorsion",
            """
            grad = np.asarray(
                LpCurveTorsion(curves[0], p=4, threshold=10.0).dJ(),
                dtype=float,
            )
            assert grad.size > 0
            assert np.all(np.isfinite(grad))
            """,
        ),
        (
            "FramedCurveTwist",
            """
            grad = np.asarray(
                FramedCurveTwist(
                    FramedCurveCentroid(curves[0], rotation),
                    f="lp",
                    p=2,
                ).dJ(),
                dtype=float,
            )
            assert grad.size > 0
            assert np.all(np.isfinite(grad))
            """,
        ),
    ],
)
def test_transfer_guard_disallow_allows_legacy_curve_objective_gradients(label, code):
    """Legacy curve-objective gradients must keep host/device transfers explicit."""
    _assert_transfer_guard_curve_objective_smoke(
        objective_imports=_TRANSFER_GUARD_CURVE_GRADIENT_IMPORTS,
        setup_code=_TRANSFER_GUARD_CURVE_GRADIENT_SETUP,
        objective_code=code,
        failure_message=f"{label} transfer-guard gradient smoke failed",
    )


def test_transfer_guard_disallow_allows_surfacerzfourier_spec_defaults():
    """SurfaceRZFourier spec defaults should avoid zeros_like scalar materialization."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(16) / 16,
            quadpoints_theta=np.arange(16) / 16,
        )
        spec = surf.surface_spec()
        assert spec.rs.shape == spec.rc.shape
        assert spec.zc.shape == spec.rc.shape
    """,
        failure_message="SurfaceRZFourier transfer-guard spec smoke failed",
    )


def test_transfer_guard_disallow_allows_surface_rzfourier_gamma_from_spec():
    """Surface gamma evaluation should avoid implicit eager scalar transfers."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier
        from simsopt.jax_core.surface_rzfourier import surface_rz_fourier_gamma_from_spec

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(16) / 16,
            quadpoints_theta=np.arange(16) / 16,
        )
        gamma = surface_rz_fourier_gamma_from_spec(surf.surface_spec())
        assert gamma.shape == (16, 16, 3)
    """,
        failure_message="SurfaceRZFourier gamma transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_surface_rzfourier_normal_from_spec():
    """Surface normal evaluation should stay transfer-clean under disallow mode."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier
        from simsopt.jax_core.surface_rzfourier import surface_rz_fourier_normal_from_spec

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        surf = SurfaceRZFourier(
            nfp=5,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(16) / 16,
            quadpoints_theta=np.arange(16) / 16,
        )
        normal = surface_rz_fourier_normal_from_spec(surf.surface_spec())
        assert normal.shape == (16, 16, 3)
    """,
        failure_message="SurfaceRZFourier normal transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_squaredfluxjax_construction():
    """SquaredFluxJAX construction should not fail in fixed-surface setup."""
    _assert_import_check_passes(
        """
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier, CurveXYZFourier
        from simsopt.field import BiotSavartJAX, Coil, Current
        from simsopt.objectives import SquaredFluxJAX

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
        surf = SurfaceRZFourier(
            nfp=1,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(8) / 8,
            quadpoints_theta=np.arange(8) / 8,
        )
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
        bs_jax = BiotSavartJAX([Coil(curve, Current(1.0))])
        objective = SquaredFluxJAX(surf, bs_jax)
        assert objective._flux_spec.normal.shape == (8, 8, 3)
    """,
        failure_message="SquaredFluxJAX transfer-guard construction smoke failed",
    )


def test_transfer_guard_disallow_allows_squaredfluxjax_host_surface_fallback():
    """SquaredFluxJAX fallback surface geometry must use explicit JAX placement."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import CurveXYZFourier
        from simsopt.field import BiotSavartJAX, Coil, Current
        from simsopt.objectives import SquaredFluxJAX

        class HostSurface:
            def __init__(self):
                phi, theta = np.meshgrid(
                    np.arange(8, dtype=np.float64) / 8.0,
                    np.arange(8, dtype=np.float64) / 8.0,
                    indexing="ij",
                )
                gamma = np.zeros((8, 8, 3), dtype=np.float64)
                gamma[..., 0] = 1.0 + 0.1 * np.cos(2.0 * np.pi * theta)
                gamma[..., 1] = 0.1 * np.sin(2.0 * np.pi * theta)
                gamma[..., 2] = 0.05 * np.sin(2.0 * np.pi * phi)
                normal = np.zeros_like(gamma)
                normal[..., 2] = 1.0
                self._gamma = gamma
                self._normal = normal

            def gamma(self):
                return self._gamma

            def normal(self):
                return self._normal

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )

        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
        bs_jax = BiotSavartJAX([Coil(curve, Current(1.0))])
        objective = SquaredFluxJAX(HostSurface(), bs_jax)

        assert isinstance(objective._flux_spec.normal, jax.Array)
        assert isinstance(objective._flux_spec.target, jax.Array)
        assert objective._flux_spec.normal.shape == (8, 8, 3)
        assert objective._flux_spec.target.shape == (8, 8)
    """,
        failure_message="SquaredFluxJAX host-surface fallback transfer-guard smoke failed",
    )


def test_transfer_guard_disallow_allows_lpcurveforce_shared_state_packing():
    """LpCurveForce shared-state packing must explicitly place host geometry on JAX arrays."""
    _assert_import_check_passes(
        """
        import jax
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.field import Current, LpCurveForce, RegularizedCoil
        from simsopt.geo import CurveXYZFourier

        def make_curve(radius, z):
            curve = CurveXYZFourier(16, 1)
            curve.x = np.array(
                [0.0, 0.0, radius, 0.0, 1.0, 0.0, 0.0, z, 0.0],
                dtype=np.float64,
            )
            return curve

        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )

        target = RegularizedCoil(
            make_curve(1.0, 0.0),
            Current(1.0),
            np.float64(0.05**2 / np.sqrt(np.e)),
        )
        source = RegularizedCoil(
            make_curve(1.2, 0.1),
            Current(-0.6),
            np.float64(0.04**2 / np.sqrt(np.e)),
        )
        objective = LpCurveForce(target, [source], p=2.0, threshold=1.0e-3)
        value = objective.J()
        args = objective._J_args()

        assert np.isfinite(float(value))
        assert not isinstance(objective.p, jax.Array)
        assert not isinstance(objective.threshold, jax.Array)
        for array_arg in args[:14]:
            assert isinstance(array_arg, jax.Array)
    """,
        failure_message="LpCurveForce shared-state transfer-guard smoke failed",
    )


def test_native_cpu_backend_selection_does_not_require_jax_runtime():
    """native_cpu config must not force a JAX import when only CPU mode is selected."""
    rc, err = _run_import_check("""
        import importlib.abc
        import sys
        import types
        import types

        class _BlockJax(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "jax" or fullname.startswith("jax."):
                    raise ImportError("blocked jax import for native_cpu smoke")
                return None

        sys.meta_path.insert(0, _BlockJax())

        import simsopt.config as simsopt_config

        cfg = simsopt_config.set_backend(
            "native_cpu",
            debug_nans=True,
            transfer_guard="log",
            compilation_cache_dir="/tmp/ignored-native-cache",
        )
        assert cfg.mode == "native_cpu"
        assert cfg.backend == "cpu"
    """)
    assert rc == 0, f"native_cpu config unexpectedly required jax:\n{err}"


def test_native_cpu_policy_matches_import_time_x64_contract():
    """The default/native policy should match the package's import-time x64 state."""
    rc, err = _run_import_check("""
        import simsopt.config as simsopt_config
        import jax

        policy = simsopt_config.get_backend_policy()

        assert policy.mode == "native_cpu"
        assert policy.requires_x64 is True
        assert jax.config.jax_enable_x64 is True
        assert jax.numpy.zeros(1).dtype == jax.numpy.float64
    """)
    assert rc == 0, f"native_cpu x64 policy mismatch:\n{err}"


def test_import_biotsavart_jax():
    """BiotSavartJAX is importable through the real package entrypoint."""
    rc, err = _run_import_check("""
        from simsopt.field import BiotSavartJAX
        assert BiotSavartJAX is not None
    """)
    assert rc == 0, f"import BiotSavartJAX failed:\n{err}"


def test_import_jax_core_specs():
    """The pure JAX kernel-layer package imports through the real package tree."""
    rc, err = _run_import_check("""
        from simsopt.jax_core import (
            CoilSpec,
            CoilGroupSpec,
            CoilSymmetrySpec,
            CurveCWSFourierRZSpec,
            CurveFilamentSpec,
            CurveHelicalSpec,
            CurvePlanarFourierSpec,
            CurveSpec,
            CurveSpecKind,
            CurvePerturbedSpec,
            CurrentValueSpec,
            CurveRZFourierSpec,
            CurveXYZFourierSpec,
            FieldEvalSpec,
            FrameRotationSpec,
            GroupedCoilSetSpec,
            FixedSurfaceFluxSpec,
            OptimizableDofMapSpec,
            SurfaceRZFourierSpec,
            ZeroRotationSpec,
            curve_spec_kind,
        )

        assert CoilSpec is not None
        assert CoilGroupSpec is not None
        assert CoilSymmetrySpec is not None
        assert CurveCWSFourierRZSpec is not None
        assert CurveFilamentSpec is not None
        assert CurveHelicalSpec is not None
        assert CurvePlanarFourierSpec is not None
        assert CurveSpec is not None
        assert CurveSpecKind is not None
        assert CurvePerturbedSpec is not None
        assert CurrentValueSpec is not None
        assert CurveRZFourierSpec is not None
        assert CurveXYZFourierSpec is not None
        assert FieldEvalSpec is not None
        assert FrameRotationSpec is not None
        assert GroupedCoilSetSpec is not None
        assert FixedSurfaceFluxSpec is not None
        assert OptimizableDofMapSpec is not None
        assert SurfaceRZFourierSpec is not None
        assert ZeroRotationSpec is not None
        assert curve_spec_kind is not None
    """)
    assert rc == 0, f"import simsopt.jax_core failed:\n{err}"


def test_jax_core_specs_are_pytrees():
    """Immutable JAX specs must flatten and survive JIT as real pytrees."""
    rc, err = _run_import_check("""
        import jax
        import jax.numpy as jnp
        import numpy as np

        from simsopt.jax_core import (
            CoilSpec,
            CoilSymmetrySpec,
            CurveCWSFourierRZSpec,
            CurveFilamentSpec,
            CurveHelicalSpec,
            CurvePlanarFourierSpec,
            CurvePerturbedSpec,
            CurrentValueSpec,
            CurveRZFourierSpec,
            CurveXYZFourierSpec,
            FieldEvalSpec,
            FrameRotationSpec,
            FixedSurfaceFluxSpec,
            GroupedCoilSetSpec,
            OptimizableDofMapSpec,
            SurfaceRZFourierSpec,
            ZeroRotationSpec,
            curve_gamma_and_dash_from_dofs,
            curve_gamma_and_dash_from_spec,
            curve_geometry_from_dofs,
            curve_geometry_from_spec,
            fixed_surface_flux_integral_from_B,
            grouped_biot_savart_B_from_spec,
            grouped_coil_currents_from_spec,
            grouped_coil_index_lists_from_spec,
            grouped_coil_set_spec_from_coil_specs,
            grouped_coil_set_spec_from_source,
            grouped_field_data_from_spec,
            grouped_field_inputs_from_spec,
            invalidate_kernel_cache,
            make_coil_spec,
            make_coil_symmetry_spec,
            make_fixed_surface_flux_spec,
            make_current_value_spec,
            make_curve_cwsfourier_rz_spec,
            make_curve_filament_spec,
            make_curve_helical_spec,
            make_curve_planarfourier_spec,
            make_curve_perturbed_spec,
            make_curve_rzfourier_spec,
            make_curve_xyzfourier_spec,
            make_field_eval_spec,
            make_frame_rotation_spec,
            make_grouped_coil_set_spec,
            make_optimizable_dof_map_spec,
            make_surface_rzfourier_spec,
            make_zero_rotation_spec,
            surface_rz_fourier_dofs_from_spec,
            surface_rz_fourier_gamma_from_spec,
        )

        coil_spec = make_grouped_coil_set_spec([
            (
                jnp.asarray(
                    [[[1.0, 0.0, 0.0], [1.1, 0.2, 0.1]]],
                    dtype=jnp.float64,
                ),
                jnp.asarray(
                    [[[0.0, 0.8, 0.1], [0.0, 0.6, 0.1]]],
                    dtype=jnp.float64,
                ),
                jnp.asarray([1.0]),
                [0],
            )
        ])
        flux_spec = make_fixed_surface_flux_spec(
            points=jnp.zeros((4, 3)),
            normal=jnp.ones((2, 2, 3)),
            target=jnp.zeros((2, 2)),
            definition="quadratic flux",
        )
        curve_xyz_spec = make_curve_xyzfourier_spec(
            dofs=jnp.asarray([1.0, 0.0, 0.0]),
            quadpoints=jnp.asarray([0.0, 0.5]),
            order=0,
        )
        curve_rz_spec = make_curve_rzfourier_spec(
            dofs=jnp.asarray([1.0, 0.0]),
            quadpoints=jnp.asarray([0.0, 0.5]),
            order=0,
            nfp=1,
            stellsym=True,
        )
        curve_planar_spec = make_curve_planarfourier_spec(
            dofs=jnp.asarray([1.1, 0.2, -0.1, 0.05, -0.03, 1.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.05]),
            quadpoints=jnp.asarray([0.0, 0.5]),
            order=2,
        )
        curve_helical_spec = make_curve_helical_spec(
            dofs=jnp.asarray([0.1, -0.03, 0.02, 0.04, -0.01]),
            quadpoints=jnp.asarray([0.0, 0.5]),
            order=2,
            m=5,
            ell=2,
            R0=1.0,
            r=0.3,
        )
        identity_curve_map = make_optimizable_dof_map_spec(
            template_full_dofs=jnp.asarray([0.1, -0.03, 0.02, 0.04, -0.01]),
            owner_segments=((0, 5, 0, 5),),
            input_mode="local",
            input_start=0,
            input_end=5,
        )
        curve_perturbed_spec = make_curve_perturbed_spec(
            dofs=jnp.asarray([0.1, -0.03, 0.02, 0.04, -0.01]),
            quadpoints=jnp.asarray([0.0, 0.5]),
            base_curve=curve_helical_spec,
            base_curve_map=identity_curve_map,
            sample_gamma=jnp.asarray([[1.0e-3, 0.0, 0.0], [0.0, -2.0e-3, 0.0]]),
            sample_gammadash=jnp.asarray([[0.0, 3.0e-3, 0.0], [-4.0e-3, 0.0, 0.0]]),
            sample_gammadashdash=jnp.asarray([[0.0, 0.0, 5.0e-3], [0.0, -6.0e-3, 0.0]]),
            sample_gammadashdashdash=jnp.asarray([[7.0e-3, 0.0, 0.0], [0.0, 0.0, -8.0e-3]]),
        )
        frame_rotation_spec = make_frame_rotation_spec(
            dofs=jnp.asarray([0.07, -0.03, 0.02]),
            quadpoints=jnp.asarray([0.0, 0.5]),
            order=1,
            scale=1.0,
        )
        zero_rotation_spec = make_zero_rotation_spec(quadpoints=jnp.asarray([0.0, 0.5]))
        filament_owner_dofs = jnp.asarray([0.1, -0.03, 0.02, 0.04, -0.01, 0.07, -0.03, 0.02])
        filament_curve_map = make_optimizable_dof_map_spec(
            template_full_dofs=jnp.asarray([0.1, -0.03, 0.02, 0.04, -0.01]),
            owner_segments=((0, 5, 0, 5),),
            input_mode="local",
            input_start=0,
            input_end=5,
        )
        filament_rotation_map = make_optimizable_dof_map_spec(
            template_full_dofs=jnp.asarray([0.07, -0.03, 0.02]),
            owner_segments=((5, 8, 0, 3),),
            input_mode="local",
            input_start=0,
            input_end=3,
        )
        curve_filament_spec = make_curve_filament_spec(
            dofs=filament_owner_dofs,
            quadpoints=jnp.asarray([0.0, 0.5]),
            base_curve=curve_helical_spec,
            base_curve_map=filament_curve_map,
            rotation=frame_rotation_spec,
            rotation_map=filament_rotation_map,
            frame_kind="centroid",
            dn=0.01,
            db=-0.02,
        )
        current_spec = make_current_value_spec(2.0)
        field_eval_spec = make_field_eval_spec(jnp.zeros((4, 3)))
        coil_value_spec = make_coil_spec(
            curve=curve_xyz_spec,
            current=current_spec,
        )
        surface_spec = make_surface_rzfourier_spec(
            rc=jnp.asarray([[1.0], [0.25]]),
            zs=jnp.asarray([[0.0], [0.2]]),
            quadpoints_phi=jnp.asarray([0.0, 0.5]),
            quadpoints_theta=jnp.asarray([0.0, 0.5]),
            nfp=1,
            stellsym=True,
        )
        curve_cws_spec = make_curve_cwsfourier_rz_spec(
            dofs=jnp.asarray([0.1, 0.0, 0.2, 0.0, 0.0, 0.0]),
            quadpoints=jnp.asarray([0.0, 0.5]),
            surface=surface_spec,
            order=1,
        )
        surface_spec_nonstellsym = make_surface_rzfourier_spec(
            rc=jnp.asarray([[1.0], [0.25]]),
            zs=jnp.asarray([[0.0], [0.2]]),
            rs=jnp.asarray([[0.0], [0.15]]),
            zc=jnp.asarray([[0.05], [0.0]]),
            quadpoints_phi=jnp.asarray([0.0, 0.5]),
            quadpoints_theta=jnp.asarray([0.0, 0.5]),
            nfp=1,
            stellsym=False,
        )
        curve_cws_nonstellsym_spec = make_curve_cwsfourier_rz_spec(
            dofs=jnp.asarray([0.1, 0.0, 0.2, 0.0, 0.0, 0.0]),
            quadpoints=jnp.asarray([0.0, 0.5]),
            surface=surface_spec_nonstellsym,
            order=1,
        )
        coil_symmetry_spec = make_coil_symmetry_spec(scale=2.5)

        def assert_surface_dofs_derivable(curve_spec, expected_ndofs):
            derived = curve_spec.surface_dofs()
            assert derived.shape == (expected_ndofs,)
            assert np.all(np.isfinite(np.asarray(derived)))

        assert isinstance(coil_value_spec, CoilSpec)
        assert isinstance(coil_symmetry_spec, CoilSymmetrySpec)
        assert isinstance(curve_cws_spec, CurveCWSFourierRZSpec)
        assert isinstance(curve_filament_spec, CurveFilamentSpec)
        assert isinstance(curve_helical_spec, CurveHelicalSpec)
        assert isinstance(curve_planar_spec, CurvePlanarFourierSpec)
        assert isinstance(curve_perturbed_spec, CurvePerturbedSpec)
        assert isinstance(current_spec, CurrentValueSpec)
        assert isinstance(curve_rz_spec, CurveRZFourierSpec)
        assert isinstance(curve_xyz_spec, CurveXYZFourierSpec)
        assert isinstance(field_eval_spec, FieldEvalSpec)
        assert isinstance(frame_rotation_spec, FrameRotationSpec)
        assert isinstance(coil_spec, GroupedCoilSetSpec)
        assert isinstance(flux_spec, FixedSurfaceFluxSpec)
        assert isinstance(identity_curve_map, OptimizableDofMapSpec)
        assert isinstance(surface_spec, SurfaceRZFourierSpec)
        assert isinstance(zero_rotation_spec, ZeroRotationSpec)

        curve_xyz_leaves, _ = jax.tree_util.tree_flatten(curve_xyz_spec)
        curve_rz_leaves, _ = jax.tree_util.tree_flatten(curve_rz_spec)
        curve_planar_leaves, _ = jax.tree_util.tree_flatten(curve_planar_spec)
        curve_helical_leaves, _ = jax.tree_util.tree_flatten(curve_helical_spec)
        curve_cws_leaves, _ = jax.tree_util.tree_flatten(curve_cws_spec)
        curve_perturbed_leaves, _ = jax.tree_util.tree_flatten(curve_perturbed_spec)
        curve_filament_leaves, _ = jax.tree_util.tree_flatten(curve_filament_spec)
        coil_symmetry_leaves, _ = jax.tree_util.tree_flatten(coil_symmetry_spec)
        current_leaves, _ = jax.tree_util.tree_flatten(current_spec)
        field_eval_leaves, _ = jax.tree_util.tree_flatten(field_eval_spec)
        coil_value_leaves, _ = jax.tree_util.tree_flatten(coil_value_spec)
        coil_leaves, _ = jax.tree_util.tree_flatten(coil_spec)
        flux_leaves, _ = jax.tree_util.tree_flatten(flux_spec)
        frame_rotation_leaves, _ = jax.tree_util.tree_flatten(frame_rotation_spec)
        dof_map_leaves, _ = jax.tree_util.tree_flatten(identity_curve_map)
        surface_leaves, _ = jax.tree_util.tree_flatten(surface_spec)
        zero_rotation_leaves, _ = jax.tree_util.tree_flatten(zero_rotation_spec)

        def assert_round_trip(spec):
            leaves, treedef = jax.tree_util.tree_flatten(spec)
            rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
            rebuilt_leaves, rebuilt_treedef = jax.tree_util.tree_flatten(rebuilt)
            assert rebuilt_treedef == treedef
            assert len(rebuilt_leaves) == len(leaves)
            for expected, actual in zip(leaves, rebuilt_leaves):
                np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))

        assert len(curve_xyz_leaves) == 2
        assert len(curve_rz_leaves) == 2
        assert len(curve_planar_leaves) == 2
        assert len(curve_helical_leaves) == 2
        assert len(curve_cws_leaves) == 8
        assert len(curve_perturbed_leaves) == 9
        assert len(curve_filament_leaves) == 8
        assert len(coil_symmetry_leaves) == 1
        assert len(current_leaves) == 1
        assert len(field_eval_leaves) == 1
        assert len(frame_rotation_leaves) == 2
        assert len(coil_value_leaves) == 4
        assert len(coil_leaves) == 3
        assert len(dof_map_leaves) == 1
        assert len(flux_leaves) == 3
        assert len(surface_leaves) == 6
        assert len(zero_rotation_leaves) == 1
        assert len(grouped_field_inputs_from_spec(coil_spec)) == 1
        assert len(grouped_field_data_from_spec(coil_spec)) == 1
        assert grouped_coil_index_lists_from_spec(coil_spec) == ([0],)
        assert grouped_coil_currents_from_spec(coil_spec).shape == (1,)
        assert grouped_coil_set_spec_from_coil_specs((coil_value_spec,)).groups[0].coil_indices == (0,)
        assert grouped_coil_set_spec_from_source(coil_spec) is coil_spec
        assert callable(invalidate_kernel_cache)
        assert_surface_dofs_derivable(curve_cws_spec, 3)  # stellsym: 2 rc + 1 zs
        assert_surface_dofs_derivable(curve_cws_nonstellsym_spec, 6)  # 2rc+1rs+2zc+1zs
        assert_round_trip(curve_perturbed_spec)
        assert_round_trip(curve_filament_spec)
        assert_round_trip(coil_spec)

        curve_xyz_gamma, curve_xyz_gammadash = jax.jit(curve_gamma_and_dash_from_spec)(curve_xyz_spec)
        curve_rz_gamma, _ = jax.jit(curve_gamma_and_dash_from_spec)(curve_rz_spec)
        curve_cws_gamma, curve_cws_gammadash = jax.jit(curve_gamma_and_dash_from_spec)(curve_cws_spec)
        curve_perturbed_gamma, curve_perturbed_gammadash = jax.jit(curve_gamma_and_dash_from_spec)(curve_perturbed_spec)
        curve_filament_gamma, curve_filament_gammadash = jax.jit(curve_gamma_and_dash_from_spec)(curve_filament_spec)
        curve_cws_gamma_from_dofs, curve_cws_gammadash_from_dofs = jax.jit(curve_gamma_and_dash_from_dofs)(
            curve_cws_spec,
            curve_cws_spec.dofs,
        )
        _, _, curve_cws_gammadashdash = jax.jit(curve_geometry_from_spec)(curve_cws_spec)
        _, _, curve_cws_gammadashdash_from_dofs = jax.jit(curve_geometry_from_dofs)(
            curve_cws_spec,
            curve_cws_spec.dofs,
        )
        B = jax.jit(grouped_biot_savart_B_from_spec)(jnp.zeros((4, 3)), coil_spec)
        value = jax.jit(fixed_surface_flux_integral_from_B)(B, flux_spec)
        gamma = jax.jit(surface_rz_fourier_gamma_from_spec)(surface_spec)

        assert B.shape == (4, 3)
        assert curve_xyz_gamma.shape == (2, 3)
        assert curve_xyz_gammadash.shape == (2, 3)
        assert curve_rz_gamma.shape == (2, 3)
        assert curve_cws_gamma.shape == (2, 3)
        assert curve_cws_gamma_from_dofs.shape == (2, 3)
        assert curve_cws_gammadash.shape == (2, 3)
        assert curve_cws_gammadash_from_dofs.shape == (2, 3)
        assert curve_perturbed_gamma.shape == (2, 3)
        assert curve_perturbed_gammadash.shape == (2, 3)
        assert curve_filament_gamma.shape == (2, 3)
        assert curve_filament_gammadash.shape == (2, 3)
        assert curve_cws_gammadashdash.shape == (2, 3)
        assert curve_cws_gammadashdash_from_dofs.shape == (2, 3)
        assert gamma.shape == (2, 2, 3)
        assert np.isfinite(np.asarray(value))
    """)
    assert rc == 0, f"jax_core pytree contract failed:\n{err}"


def test_jax_core_grouped_field_chunking_matches_dense_sum():
    """Chunked grouped-field evaluation must preserve dense grouped parity."""
    rc, err = _run_import_check("""
        import jax
        import jax.numpy as jnp

        from simsopt import config as simsopt_config
        from simsopt.field.biotsavart_jax import (
            biot_savart_B,
            biot_savart_B_and_dB,
            biot_savart_dB_by_dX,
        )
        from simsopt.jax_core import (
            grouped_biot_savart_B_and_dB_from_spec,
            grouped_biot_savart_B_from_spec,
            grouped_biot_savart_dB_by_dX_from_spec,
            grouped_field_inputs_from_spec,
            make_grouped_coil_set_spec,
        )

        def _sum_group_kernel(groups, kernel):
            return sum(
                kernel(points, gammas, gammadashs, currents)
                for gammas, gammadashs, currents in groups
            )

        def _sum_group_combo(groups):
            combo = [
                biot_savart_B_and_dB(points, gammas, gammadashs, currents)
                for gammas, gammadashs, currents in groups
            ]
            return sum(Bi for Bi, _ in combo), sum(dBi for _, dBi in combo)

        simsopt_config.set_backend("jax_cpu_parity")

        points = jnp.stack(
            [
                jnp.linspace(-0.2, 0.2, 300),
                jnp.linspace(0.3, 0.7, 300),
                jnp.linspace(-0.1, 0.1, 300),
            ],
            axis=1,
        )
        coil_spec = make_grouped_coil_set_spec(
            [
                (
                    jnp.asarray(
                        [
                            [[1.0, 0.0, 0.0], [1.1, 0.2, 0.1]],
                            [[-1.0, 0.1, 0.2], [-1.1, 0.3, 0.4]],
                        ]
                    ),
                    jnp.asarray(
                        [
                            [[0.0, 1.0, 0.0], [0.0, 0.8, 0.1]],
                            [[0.0, -1.0, 0.0], [0.0, -0.8, -0.1]],
                        ]
                    ),
                    jnp.asarray([1.2, -0.7]),
                    [0, 1],
                ),
                (
                    jnp.asarray(
                        [
                            [[0.6, -0.4, 0.3], [0.7, -0.2, 0.4], [0.8, -0.1, 0.5]],
                        ]
                    ),
                    jnp.asarray(
                        [
                            [[0.2, 0.1, 0.0], [0.2, 0.1, 0.0], [0.2, 0.1, 0.0]],
                        ]
                    ),
                    jnp.asarray([0.9]),
                    [2],
                ),
            ]
        )

        groups = grouped_field_inputs_from_spec(coil_spec)
        B_ref = _sum_group_kernel(groups, biot_savart_B)
        dB_ref = _sum_group_kernel(groups, biot_savart_dB_by_dX)

        B = jax.jit(grouped_biot_savart_B_from_spec)(points, coil_spec)
        dB = jax.jit(grouped_biot_savart_dB_by_dX_from_spec)(points, coil_spec)
        B_combo, dB_combo = jax.jit(grouped_biot_savart_B_and_dB_from_spec)(points, coil_spec)

        B_combo_ref, dB_combo_ref = _sum_group_combo(groups)

        assert B.shape == (300, 3)
        assert dB.shape == (300, 3, 3)
        assert B_combo.shape == (300, 3)
        assert dB_combo.shape == (300, 3, 3)
        assert jnp.allclose(B, B_ref, rtol=1e-12, atol=1e-14)
        assert jnp.allclose(dB, dB_ref, rtol=1e-12, atol=1e-14)
        assert jnp.allclose(B_combo, B_combo_ref, rtol=1e-12, atol=1e-14)
        assert jnp.allclose(dB_combo, dB_combo_ref, rtol=1e-12, atol=1e-14)
    """)
    assert rc == 0, f"jax_core grouped chunking contract failed:\n{err}"


def test_import_squaredflux_jax():
    """SquaredFluxJAX is importable through the real package entrypoint."""
    rc, err = _run_import_check("""
        from simsopt.objectives import SquaredFluxJAX
        assert SquaredFluxJAX is not None
    """)
    assert rc == 0, f"import SquaredFluxJAX failed:\n{err}"


def test_import_boozersurface_jax():
    """BoozerSurfaceJAX is importable through the real package entrypoint."""
    rc, err = _run_import_check("""
        from simsopt.geo import BoozerSurfaceJAX
        assert BoozerSurfaceJAX is not None
    """)
    assert rc == 0, f"import BoozerSurfaceJAX failed:\n{err}"


def test_import_core_optimizable():
    """Optimizable base class imports without simsoptpp."""
    rc, err = _run_import_check("""
        from simsopt._core.optimizable import Optimizable
        assert Optimizable is not None
    """)
    assert rc == 0, f"import Optimizable failed:\n{err}"


def test_optimizer_jax_import_is_lazy():
    """Importing the public optimizer module must not eagerly load the private package."""
    rc, err = _run_import_check("""
        import sys

        from simsopt.geo import optimizer_jax

        assert optimizer_jax._private_pkg is None
        assert "simsopt.geo.optimizer_jax_private" not in sys.modules
    """)
    assert rc == 0, f"optimizer_jax lazy import check failed:\n{err}"


def test_optimizer_jax_public_reference_methods_work_without_private_package():
    """Public SciPy methods must work even when the private package cannot import."""
    rc, err = _run_import_check(
        _block_private_optimizer_imports()
        + """
        import sys

        from simsopt.geo import optimizer_jax
        import jax.numpy as jnp

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.asarray([1.0, -2.0])
        assert "simsopt.geo.optimizer_jax_private" not in sys.modules

        for method in ("bfgs", "lbfgs"):
            result = optimizer_jax.jax_minimize(quad, x0, method=method, maxiter=5)
            assert result.success
            assert float(result.fun) < float(quad(x0))
            assert "simsopt.geo.optimizer_jax_private" not in sys.modules
    """
    )
    assert rc == 0, f"public optimizer_jax reference methods failed:\n{err}"


def test_optimizer_jax_private_methods_require_private_package_when_blocked():
    """Private optimizer methods must raise ImportError when the private package is absent."""
    rc, err = _run_import_check(
        _block_private_optimizer_imports()
        + """
        from simsopt.geo import optimizer_jax
        import jax.numpy as jnp

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        try:
            optimizer_jax.jax_minimize(
                quad,
                jnp.asarray([1.0, -2.0]),
                method="bfgs-ondevice",
                maxiter=1,
            )
        except ImportError as exc:
            message = str(exc)
            assert "private optimizer package" in message
            assert "simsopt.geo.optimizer_jax_private" in message
        else:
            raise AssertionError("expected ImportError for blocked private optimizer package")
    """
    )
    assert rc == 0, f"private optimizer import guard failed:\n{err}"


def test_optimizer_jax_public_module_has_no_private_jax_src_usage():
    """Section 6 public optimizer module must remain free of jax._src usage."""
    _assert_no_private_jax_src_usage(
        _OPTIMIZER_JAX_PATH,
        label="optimizer_jax.py in the public lane",
    )


def test_optimizer_jax_private_package_has_no_private_jax_src_usage():
    """Private optimizer modules must also stay on public JAX APIs."""
    forbidden_usages = {}
    for path in sorted(_OPTIMIZER_PRIVATE_DIR.glob("*.py")):
        usages = _find_private_jax_src_usages(path)
        if usages:
            forbidden_usages[str(path.relative_to(_OPTIMIZER_PRIVATE_DIR.parent))] = (
                usages
            )

    assert forbidden_usages == {}, (
        f"optimizer_jax_private must not use jax._src: {forbidden_usages}"
    )


def test_backend_runtime_module_has_no_private_jax_src_usage():
    """Backend runtime helpers must stay on public JAX APIs."""
    _assert_no_private_jax_src_usage(
        _RUNTIME_BACKEND_PATH,
        label="runtime.py in backend helpers",
    )


def test_jax_classes_inherit_optimizable():
    """JAX adapter classes use the real Optimizable metaclass."""
    rc, err = _run_import_check("""
        from simsopt._core.optimizable import Optimizable
        from simsopt.field import BiotSavartJAX
        from simsopt.objectives import SquaredFluxJAX
        assert issubclass(BiotSavartJAX, Optimizable)
        assert issubclass(SquaredFluxJAX, Optimizable)
    """)
    assert rc == 0, f"inheritance check failed:\n{err}"


def test_import_pure_jax_modules():
    """Pure JAX compute modules (M1) import through the package."""
    rc, err = _run_import_check("""
        from simsopt.field.biotsavart_jax import biot_savart_B
        from simsopt.geo.surface_fourier_jax import stellsym_scatter_indices
        from simsopt.geo.boozer_residual_jax import boozer_residual_scalar
        from simsopt.objectives.integral_bdotn_jax import integral_BdotN
        assert callable(biot_savart_B)
        assert callable(stellsym_scatter_indices)
        assert callable(boozer_residual_scalar)
        assert callable(integral_BdotN)
    """)
    assert rc == 0, f"import pure JAX modules failed:\n{err}"


def test_m5_classes_require_simsoptpp():
    """M5 single-stage wrappers remain package-gated on simsoptpp availability."""
    rc, err = _run_import_check("""
        import simsopt.geo

        try:
            from simsoptpp import Curve
            has_simsoptpp = True
        except (ImportError, AttributeError):
            has_simsoptpp = False

        for name in ["BoozerResidualJAX", "IotasJAX", "NonQuasiSymmetricRatioJAX"]:
            available = hasattr(simsopt.geo, name)
            if has_simsoptpp:
                assert available, f"{name} should be available with simsoptpp"
            else:
                assert not available, f"{name} should NOT be available without simsoptpp"
    """)
    assert rc == 0, f"M5 availability check failed:\n{err}"


def test_direct_curve_modules_raise_clear_importerror_without_simsoptpp():
    """Direct geo-module imports should fail clearly at instantiation time."""
    rc, err = _run_import_check(f"""
        {_block_simsoptpp_imports()}

        from simsopt.geo.framedcurve import FramedCurve
        from simsopt.geo.curveplanarfourier import CurvePlanarFourier
        from simsopt.geo.curveperturbed import CurvePerturbed
        from simsopt.geo.curverzfourier import CurveRZFourier
        from simsopt.geo.curvexyzfourier import CurveXYZFourier

        def _assert_missing(factory, expected_name):
            try:
                factory()
            except ImportError as exc:
                message = str(exc)
                assert expected_name in message
                assert "simsoptpp is required to instantiate" in message
            else:
                raise AssertionError(f"{{expected_name}} should require simsoptpp")

        constructors = (
            (lambda: FramedCurve(object()), "FramedCurve"),
            (lambda: CurvePlanarFourier([0.0], 1), "CurvePlanarFourier"),
            (lambda: CurveRZFourier([0.0], 1, 1, True), "CurveRZFourier"),
            (lambda: CurveXYZFourier([0.0], 1), "CurveXYZFourier"),
            (lambda: CurvePerturbed(object(), object()), "CurvePerturbed"),
        )
        for factory, expected_name in constructors:
            _assert_missing(factory, expected_name)
    """)
    assert rc == 0, f"direct geo-module simsoptpp fallback smoke failed:\n{err}"


def test_direct_optional_geo_modules_import_without_simsoptpp():
    """Optional geo modules should remain directly importable without simsoptpp."""
    rc, err = _run_import_check(f"""
        {_block_simsoptpp_imports()}

        import simsopt.geo.curvecwsfourier as curvecwsfourier
        import simsopt.geo.curveobjectives as curveobjectives

        assert hasattr(curvecwsfourier, "CurveCWSFourierCPP")
        for name in (
            "CurveCurveDistance",
            "CurveSurfaceDistance",
            "LinkingNumber",
            "pairwise_min_distance_pure",
        ):
            assert hasattr(curveobjectives, name), name
    """)
    assert rc == 0, f"optional geo-module import smoke failed:\n{err}"


def test_curveobjectives_optional_cpp_helpers_raise_clear_importerror_without_simsoptpp():
    """Optional simsoptpp helpers in curveobjectives should fail clearly on use."""
    rc, err = _run_import_check(f"""
        {_block_simsoptpp_imports()}

        import numpy as np

        from simsopt._core.optimizable import Optimizable
        from simsopt.geo.curveobjectives import (
            CurveCurveDistance,
            CurveSurfaceDistance,
            LinkingNumber,
        )

        class _FakeCurve(Optimizable):
            def __init__(self, offset):
                self.quadpoints = np.linspace(0.0, 1.0, 4, endpoint=False)
                self._gamma = np.array(
                    [
                        [offset + 0.00, 0.0, 0.0],
                        [offset + 0.10, 0.0, 0.0],
                        [offset + 0.20, 0.0, 0.0],
                        [offset + 0.30, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                )
                self._gammadash = np.full((4, 3), [0.1, 0.0, 0.0], dtype=np.float64)
                super().__init__(x0=np.zeros(0))

            def recompute_bell(self, parent=None):
                del parent

            def gamma(self):
                return self._gamma

            def gammadash(self):
                return self._gammadash

        class _FakeSurface:
            def gamma(self):
                return np.array(
                    [
                        [[0.0, 0.2, 0.0], [0.1, 0.2, 0.0]],
                        [[0.0, 0.3, 0.0], [0.1, 0.3, 0.0]],
                    ],
                    dtype=np.float64,
                )

            def normal(self):
                return np.array(
                    [
                        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    ],
                    dtype=np.float64,
                )

        curves = [_FakeCurve(0.0), _FakeCurve(0.4)]
        surface = _FakeSurface()

        def _assert_missing(callable_obj, expected_name):
            try:
                callable_obj()
            except ImportError as exc:
                message = str(exc)
                assert expected_name in message
                assert "simsoptpp is required to use" in message
            else:
                raise AssertionError(f"{{expected_name}} should require simsoptpp")

        _assert_missing(
            lambda: CurveCurveDistance(curves, 0.05).compute_candidates(),
            "get_pointclouds_closer_than_threshold_within_collection",
        )
        _assert_missing(
            lambda: CurveSurfaceDistance(curves, surface, 0.05).compute_candidates(),
            "get_pointclouds_closer_than_threshold_between_two_collections",
        )
        _assert_missing(
            lambda: LinkingNumber(curves).J(),
            "compute_linking_number",
        )
    """)
    assert rc == 0, f"curveobjectives simsoptpp helper smoke failed:\n{err}"


def test_framedcurve_direct_module_import_smoke():
    """Direct import of simsopt.geo.framedcurve should not hit jax_core cycles."""
    rc, err = _run_import_check("""
        import simsopt.geo.framedcurve as framedcurve

        assert hasattr(framedcurve, "FramedCurve")
        assert hasattr(framedcurve, "FramedCurveCentroid")
        assert hasattr(framedcurve, "FramedCurveFrenet")
    """)
    assert rc == 0, f"direct framedcurve import smoke failed:\n{err}"


def test_biotsavart_jax_backend_does_not_import_coil_unwrap_helper():
    """The JAX backend must not depend on field/coil.py for graph unwrapping."""
    backend_path = Path(_SRC_DIR) / "simsopt" / "field" / "biotsavart_jax_backend.py"
    tree = ast.parse(backend_path.read_text(encoding="utf-8"))

    direct_coil_imports = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "simsopt.field.coil" and node.module != "coil":
            continue
        imported_names = {alias.name for alias in node.names}
        if "_unwrap_coil_curve_and_current_objects" in imported_names:
            direct_coil_imports.append(node.lineno)

    assert not direct_coil_imports, (
        "biotsavart_jax_backend.py must not import "
        "_unwrap_coil_curve_and_current_objects from field/coil.py"
    )


def test_surfaceobjectives_jax_has_no_tensor_surface_imports():
    """Single-stage JAX wrappers should not instantiate tensor surfaces internally."""
    objectives_path = Path(_SRC_DIR) / "simsopt" / "geo" / "surfaceobjectives_jax.py"
    tree = ast.parse(objectives_path.read_text(encoding="utf-8"))

    tensor_surface_import_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        in (
            "simsopt.geo.surfacexyztensorfourier",
            "surfacexyztensorfourier",
        )
        and any(alias.name == "SurfaceXYZTensorFourier" for alias in node.names)
    ]

    assert not tensor_surface_import_lines, (
        "surfaceobjectives_jax.py must not import SurfaceXYZTensorFourier "
        "for its JAX wrapper/runtime helpers"
    )


def test_import_cpu_package_entrypoints_with_simsoptpp():
    """CPU package entrypoints must import cleanly when simsoptpp is available."""
    try:
        from simsoptpp import Curve as _  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    rc, err = _run_import_check("""
        import simsopt.configs
        import simsopt.field
        import simsopt.geo
        import simsopt.objectives
        import simsopt.solve
        import simsopt.util

        assert hasattr(simsopt.field, "BiotSavart")
        assert hasattr(simsopt.geo, "BoozerSurface")
        assert hasattr(simsopt.objectives, "LeastSquaresProblem")
    """)
    assert rc == 0, f"CPU entrypoint import check failed:\n{err}"


def test_field_package_import_is_lazy_with_simsoptpp():
    """Bare package import must not eagerly load CPU field modules."""
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    rc, err = _run_import_check("""
        import sys
        import simsopt.field

        assert "simsopt.field.coil" not in sys.modules
        assert "simsopt.field.biotsavart" not in sys.modules

        from simsopt.field import BiotSavartJAX

        assert BiotSavartJAX is not None
        assert "simsopt.field.coil" not in sys.modules
    """)
    assert rc == 0, f"field package import was not lazy:\n{err}"


def test_geo_package_import_is_lazy_with_simsoptpp():
    """Bare package import must not eagerly load CPU geometry modules."""
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    rc, err = _run_import_check("""
        import sys
        import simsopt.geo

        assert "simsopt.geo.surfacexyztensorfourier" not in sys.modules
        assert "simsopt.geo.boozersurface" not in sys.modules
        assert simsopt.geo.parameters["jit"] in (True, False)
    """)
    assert rc == 0, f"geo package import was not lazy:\n{err}"


def test_import_cpu_geo_core_entrypoints_without_jax():
    """Core CPU geo entrypoints should import when simsoptpp is present but JAX is absent."""
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    strip_editable_finders = textwrap.indent(
        textwrap.dedent(_strip_simsopt_editable_finders(include_import=False)).strip(),
        " " * 8,
    )
    rc, err = _run_import_check(f"""
        import importlib.abc
        import sys

{strip_editable_finders}

        class _BlockJax(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "jax" or fullname.startswith("jax."):
                    raise ImportError("blocked jax import for geo CPU smoke")
                return None

        sys.meta_path.insert(0, _BlockJax())

        import simsopt.geo

        assert hasattr(simsopt.geo, "Curve")
        assert hasattr(simsopt.geo, "CurveRZFourier")
        assert hasattr(simsopt.geo, "CurveXYZFourier")
        assert hasattr(simsopt.geo, "CurvePlanarFourier")
        assert hasattr(simsopt.geo, "CurvePerturbed")
        assert hasattr(simsopt.geo, "BoozerSurface")
        assert not hasattr(simsopt.geo, "CurveCWSFourier")
    """)
    assert rc == 0, f"CPU geo import unexpectedly required jax:\n{err}"
