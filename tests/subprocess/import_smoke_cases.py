"""
Extracted import-smoke test case functions.

Each ``case_*`` function is a standalone callable that runs the exact same
logic previously embedded as inline string literals in
``tests/test_jax_import_smoke.py``.  The module intentionally avoids
non-stdlib imports at the top level so that subprocess isolation semantics
are preserved: each case function handles its own imports internally.
"""

from __future__ import annotations

import ast
import importlib.abc
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants (computed from this module's location)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
_CPU_RUN_CODE_BENCHMARK_PATH = _REPO_ROOT / "benchmarks" / "cpu_run_code_benchmark.py"
_LOCAL_SIMSOPT_IMPORT_PATHS = (_REPO_ROOT, _SRC_DIR)


class SkippedCase(RuntimeError):
    pass


def _skip_case(reason: str) -> None:
    raise SkippedCase(reason)


# ---------------------------------------------------------------------------
# Meta-path blocker helpers
# ---------------------------------------------------------------------------


def block_private_optimizer_imports() -> None:
    """Install a meta-path finder that blocks ``simsopt_jax.geo.optimizers.private``."""

    class _BlockPrivateOptimizer(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            del path, target
            if fullname == "simsopt_jax.geo.optimizers.private" or fullname.startswith(
                "simsopt_jax.geo.optimizers.private."
            ):
                exc = ModuleNotFoundError(
                    "blocked private optimizer package for smoke test"
                )
                exc.name = "simsopt_jax.geo.optimizers.private"
                raise exc
            return None

    sys.meta_path.insert(0, _BlockPrivateOptimizer())


def block_private_optimizer_submodule_import(submodule: str) -> None:
    """Install a meta-path finder that blocks one private optimizer submodule."""

    class _BlockPrivateOptimizerSubmodule(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            del path, target
            if fullname == submodule or fullname.startswith(f"{submodule}."):
                raise ModuleNotFoundError(
                    f"blocked private optimizer dependency {submodule} for smoke test"
                )
            return None

    sys.meta_path.insert(0, _BlockPrivateOptimizerSubmodule())


def block_simsoptpp_imports() -> None:
    """Install a meta-path finder that blocks ``simsoptpp``."""

    class _BlockSimsoptpp(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            del path, target
            if fullname == "simsoptpp" or fullname.startswith("simsoptpp."):
                raise ModuleNotFoundError("blocked simsoptpp for smoke test")
            return None

    sys.meta_path.insert(0, _BlockSimsoptpp())


def block_jax_imports(
    *,
    message: str = "blocked jax import for smoke test",
    error_cls: type[Exception] = ModuleNotFoundError,
) -> None:
    """Install a meta-path finder that blocks ``jax``."""

    class _BlockJax(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            del path, target
            if fullname == "jax" or fullname.startswith("jax."):
                raise error_cls(message)
            return None

    sys.meta_path.insert(0, _BlockJax())


def strip_simsopt_editable_finders() -> None:
    """Remove editable finders for simsopt from ``sys.meta_path``."""
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if type(finder).__module__ != "_simsopt_editable"
        and (
            not type(finder).__module__.startswith(("__editable__", "_editable_"))
            or "simsopt" not in type(finder).__module__.lower()
        )
    ]


def _prepend_sys_path(path: Path) -> None:
    """Move one path to the front of ``sys.path`` without duplicates."""
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def prefer_local_simsopt_source_tree() -> None:
    """Prefer this checkout over editable installs in sibling repositories."""
    strip_simsopt_editable_finders()
    for path in _LOCAL_SIMSOPT_IMPORT_PATHS:
        _prepend_sys_path(path)


def _record_host_arrays(points, *, dtype):
    """Build a callback that captures array payloads as NumPy arrays."""
    import numpy as np

    def callback(x):
        points.append(np.asarray(x, dtype=dtype))

    return callback


def _record_progress(points):
    """Build a callback that captures progress tuples in host-native types."""

    def callback(nit, fun, grad_norm):
        points.append((int(nit), float(fun), float(grad_norm)))

    return callback


# ---------------------------------------------------------------------------
# Case functions
# ---------------------------------------------------------------------------


def case_import_package_root() -> None:
    import simsopt

    assert hasattr(simsopt, "__version__")
    assert hasattr(simsopt, "__built_with_xsimd__")
    assert "simsopt._core" in sys.modules
    assert "simsoptpp" in sys.modules


def case_import_package_root_with_generated_version_file() -> None:
    import tempfile

    init_path = _SRC_DIR / "simsopt" / "__init__.py"
    src_init = init_path.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        src_root = Path(tmp) / "src"
        package_root = src_root / "simsopt"
        (package_root / "backend").mkdir(parents=True)
        (package_root / "_core").mkdir(parents=True)
        (package_root / "__init__.py").write_text(src_init, encoding="utf-8")
        (package_root / "backend" / "__init__.py").write_text(
            "def apply_jax_runtime_config():\n    return None\n\n"
            "def should_eagerly_configure_jax():\n    return False\n\n"
            "def validate_cuda_determinism_environment():\n    return None\n",
            encoding="utf-8",
        )
        (package_root / "_core" / "__init__.py").write_text(
            "def make_optimizable(*args, **kwargs):\n    return None\n\n"
            "def load(*args, **kwargs):\n    return None\n\n"
            "def save(*args, **kwargs):\n    return None\n",
            encoding="utf-8",
        )
        (package_root / "_version.py").write_text(
            'version = "0.0.dev0+fixture"\n',
            encoding="utf-8",
        )
        (src_root / "simsoptpp.py").write_text(
            "using_xsimd = False\n",
            encoding="utf-8",
        )
        strip_simsopt_editable_finders()
        sys.path.insert(0, str(src_root))
        import simsopt

        assert Path(simsopt.__file__).resolve().is_relative_to(package_root.resolve())
        assert simsopt.__version__ == "0.0.dev0+fixture"
        assert simsopt.__built_with_xsimd__ is False


def case_repo_bootstrap_synthesizes_version_for_clean_source_tree() -> None:
    import tempfile

    from repo_bootstrap import bootstrap_local_simsopt

    with tempfile.TemporaryDirectory() as tmp:
        src_root = Path(tmp) / "src"
        package_root = src_root / "simsopt"
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text(
            "from ._version import version as __version__\n",
            encoding="utf-8",
        )

        bootstrap_local_simsopt(src_root)

        import simsopt

        assert simsopt.__version__ == "0.0.dev0+source"


def case_repo_bootstrap_is_idempotent_for_local_source_tree() -> None:
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
    import simsopt.geo.curve as curve_module_after

    curve_after = curve_module_after.Curve
    base_curves, *_ = get_data("STAR_Lite-A_low")

    assert package_before is simsopt_after
    assert curve_before is curve_after
    assert isinstance(base_curves[0], curve_before)


def case_root_conftest_imports_without_jax_installed() -> None:
    import runpy

    block_jax_imports(
        message="blocked jax import for smoke test",
        error_cls=ModuleNotFoundError,
    )

    conftest_path = Path.cwd() / "tests" / "conftest.py"
    module_globals = runpy.run_path(
        str(conftest_path), run_name="simsopt_tests_conftest"
    )

    assert module_globals["jax"] is None
    parity_rng = module_globals["parity_rng"]
    assert parity_rng(3).randint(0, 1000) == parity_rng(3).randint(0, 1000)


def case_legacy_magneticfield_source_avoids_jax_import() -> None:
    magneticfield_path = _SRC_DIR / "simsopt" / "field" / "magneticfield.py"
    tree = ast.parse(
        magneticfield_path.read_text(encoding="utf-8"),
        filename=str(magneticfield_path),
    )
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(
                alias.name
                for alias in node.names
                if alias.name.partition(".")[0] == "jax"
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.partition(".")[0] == "jax":
                offenders.append(module)

    assert not offenders, f"magneticfield.py imports JAX modules: {offenders}"


def case_root_conftest_bootstraps_local_simsopt_over_foreign_resolution() -> None:
    import importlib
    import importlib.abc
    import importlib.util
    import runpy
    import tempfile

    conftest_path = _REPO_ROOT / "tests" / "conftest.py"
    local_init = (_SRC_DIR / "simsopt" / "__init__.py").resolve()

    with tempfile.TemporaryDirectory() as tmp:
        foreign_src = Path(tmp) / "foreign-src"
        foreign_package_root = foreign_src / "simsopt"
        foreign_init = foreign_package_root / "__init__.py"
        foreign_package_root.mkdir(parents=True)
        foreign_init.write_text("__version__ = 'foreign-package'\n", encoding="utf-8")

        class _ForeignLoader(importlib.abc.Loader):
            def create_module(self, spec):
                del spec
                return None

            def exec_module(self, module):
                module.__file__ = str(foreign_init)
                module.__path__ = [str(foreign_package_root)]
                module.__package__ = "simsopt"
                module.__version__ = "foreign-package"

        class _ForeignFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname != "simsopt":
                    return None
                spec = importlib.util.spec_from_loader(
                    fullname,
                    _ForeignLoader(),
                    is_package=True,
                )
                assert spec is not None
                spec.origin = str(foreign_init)
                spec.submodule_search_locations = [str(foreign_package_root)]
                return spec

        _ForeignFinder.__module__ = "__editable___simsopt_foreign"
        fake_finder = _ForeignFinder()
        sys.meta_path.insert(0, fake_finder)

        import simsopt as foreign_simsopt

        assert foreign_simsopt.__version__ == "foreign-package"
        assert Path(foreign_simsopt.__file__).resolve() == foreign_init.resolve()
        assert fake_finder in sys.meta_path

        runpy.run_path(str(conftest_path), run_name="simsopt_tests_conftest")

        import simsopt

        assert Path(simsopt.__file__).resolve() == local_init
        assert simsopt.__version__ != "foreign-package"
        assert fake_finder not in sys.meta_path


def case_repo_bootstrap_purges_detached_local_submodules() -> None:
    import types

    from repo_bootstrap import bootstrap_local_simsopt

    src_root = Path.cwd() / "src"
    bootstrap_local_simsopt(src_root)

    import simsopt

    import simsopt.configs.zoo as real_module

    fake_module = types.ModuleType("simsopt.configs.zoo")
    fake_module.marker = "fake"
    sys.modules["simsopt.configs.zoo"] = fake_module

    bootstrap_local_simsopt(src_root)

    assert "simsopt.configs.zoo" not in sys.modules

    import simsopt.configs.zoo as reloaded_module

    assert reloaded_module is not fake_module
    assert reloaded_module is not real_module
    assert (
        Path(reloaded_module.__file__).resolve()
        == (src_root / "simsopt" / "configs" / "zoo.py").resolve()
    )
    assert simsopt is sys.modules["simsopt"]


def case_repo_bootstrap_strips_editable_meta_path_finders_on_fast_path() -> None:
    import importlib
    import importlib.util

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
            module.__file__ = "/tmp/fake_zoo.py"

    class _FakeEditableFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            del path, target
            if fullname == "simsopt.configs.zoo":
                return importlib.util.spec_from_loader(
                    fullname,
                    _FakeEditableLoader(),
                )
            return None

    _FakeEditableFinder.__module__ = "__editable___simsopt_demo"
    fake_finder = _FakeEditableFinder()

    class _FakeScikitBuildEditableFinder(_FakeEditableFinder):
        pass

    _FakeScikitBuildEditableFinder.__module__ = "_editable_skbc_simsopt"
    scikit_build_finder = _FakeScikitBuildEditableFinder()
    sys.meta_path.insert(0, fake_finder)
    sys.meta_path.insert(0, scikit_build_finder)

    bootstrap_local_simsopt(src_root)

    assert fake_finder not in sys.meta_path
    assert scikit_build_finder not in sys.meta_path
    assert not any(
        type(finder).__module__ == "_simsopt_editable"
        or (
            type(finder).__module__.startswith(("__editable__", "_editable_"))
            and "simsopt" in type(finder).__module__.lower()
        )
        for finder in sys.meta_path
    )

    import simsopt.configs.zoo as reloaded_module

    assert getattr(reloaded_module, "marker", None) is None
    assert (
        Path(reloaded_module.__file__).resolve()
        == (src_root / "simsopt" / "configs" / "zoo.py").resolve()
    )
    assert simsopt is sys.modules["simsopt"]


def case_repo_bootstrap_preserves_unrelated_editable_meta_path_finders() -> None:
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


def case_repo_bootstrap_reloads_local_simsoptpp_over_foreign_module() -> None:
    import types
    from unittest import mock

    import repo_bootstrap
    from repo_bootstrap import bootstrap_local_simsopt

    src_root = Path.cwd() / "src"
    fake_extension = Path.cwd() / "build" / "fake" / "simsoptpp.test.so"
    foreign_module = types.ModuleType("simsoptpp")
    foreign_module.__file__ = "/tmp/foreign-simsoptpp.so"
    foreign_module.using_xsimd = True
    sys.modules["simsoptpp"] = foreign_module

    def _fake_find_extension(repo_root: Path) -> Path:
        del repo_root
        return fake_extension

    def _fake_load_extension(module_name: str, extension_path: Path):
        module = types.ModuleType(module_name)
        module.__file__ = str(extension_path)
        module.Curve = type("Curve", (), {})
        module.using_xsimd = False
        sys.modules[module_name] = module
        return module

    with mock.patch.object(
        repo_bootstrap,
        "_find_local_simsoptpp_extension",
        _fake_find_extension,
    ), mock.patch.object(
        repo_bootstrap,
        "_load_extension_module",
        _fake_load_extension,
    ):
        bootstrap_local_simsopt(src_root)

        import simsopt
        import simsoptpp

        assert (
            Path(simsopt.__file__)
            .resolve()
            .is_relative_to((src_root / "simsopt").resolve())
        )
        assert Path(simsoptpp.__file__) == fake_extension
        assert simsoptpp.using_xsimd is False


def case_import_package_root_native_cpu_does_not_require_jax_runtime() -> None:
    import os

    block_jax_imports(message="blocked jax import for package-root smoke")

    import simsopt

    assert hasattr(simsopt, "__version__")
    assert "JAX_ENABLE_X64" not in os.environ


def case_package_root_jax_selector_propagates_missing_jax() -> None:
    import os

    os.environ["SIMSOPT_BACKEND_MODE"] = "jax_cpu_parity"
    block_jax_imports(message="blocked jax import for explicit jax selector")

    try:
        import simsopt_jax.backend as backend

        backend.apply_jax_runtime_config()
    except ModuleNotFoundError as exc:
        assert "blocked jax import for explicit jax selector" in str(exc)
    else:
        raise AssertionError("explicit JAX selector must not hide missing jax")


def case_package_root_propagates_backend_import_error() -> None:
    class _BlockBackendRuntime(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            del path, target
            if fullname == "simsopt_jax.backend.runtime":
                raise ImportError("blocked backend runtime import")
            return None

    sys.meta_path.insert(0, _BlockBackendRuntime())

    try:
        import simsopt_jax.backend  # noqa: F401
    except ImportError as exc:
        assert "blocked backend runtime import" in str(exc)
    else:
        raise AssertionError(
            "strict backend root must not hide backend import failures"
        )


def case_entrypoint_runtime_helper_configures_cpu_before_import() -> None:
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


def case_entrypoint_runtime_helper_auto_clears_stale_platform_env() -> None:
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


def case_entrypoint_runtime_helper_adds_detected_cuda_toolchain_root() -> None:
    import os
    import tempfile

    import repo_bootstrap
    from repo_bootstrap import configure_entrypoint_jax_runtime

    with tempfile.TemporaryDirectory() as tmp:
        cuda_root = Path(tmp) / "cuda"
        (cuda_root / "bin").mkdir(parents=True)
        repo_bootstrap._DEFAULT_CUDA_TOOLCHAIN_ROOT = cuda_root
        repo_bootstrap.sys.prefix = str(Path(tmp) / "no-python-env-cuda")
        os.environ.pop("CONDA_PREFIX", None)
        os.environ.pop("VIRTUAL_ENV", None)
        os.environ["PATH"] = "/usr/bin"
        os.environ.pop("SIMSOPT_JAX_CUDA_LIBRARY_MODE", None)
        os.environ.pop("XLA_FLAGS", None)

        requested = configure_entrypoint_jax_runtime(
            ["--platform", "cuda"],
            respect_existing_env=False,
        )

        assert requested == "cuda"
        assert os.environ["JAX_PLATFORMS"] == "cuda"
        assert os.environ["SIMSOPT_JAX_PLATFORM"] == "cuda"
        assert os.environ["SIMSOPT_JAX_BACKEND"] == "cuda"
        assert os.environ["PATH"].split(os.pathsep)[0] == str(cuda_root / "bin")
        assert (
            os.environ["XLA_FLAGS"].split()[0] == f"--xla_gpu_cuda_data_dir={cuda_root}"
        )


def case_entrypoint_runtime_helper_detects_pip_cuda_nvcc_toolchain() -> None:
    import os
    import tempfile

    import repo_bootstrap
    from repo_bootstrap import configure_entrypoint_jax_runtime

    with tempfile.TemporaryDirectory() as tmp:
        active_root = Path(tmp) / "active-env"
        site_packages_root = (
            active_root
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        cuda_nvcc_root = site_packages_root / "nvidia" / "cuda_nvcc"
        cuda_nvcc_bin = cuda_nvcc_root / "bin"
        cuda_nvcc_bin.mkdir(parents=True)
        (cuda_nvcc_bin / "ptxas").touch()
        nvjitlink_lib = site_packages_root / "nvidia" / "nvjitlink" / "lib"
        nvjitlink_lib.mkdir(parents=True)
        (nvjitlink_lib / "libnvJitLink.so.12").touch()

        default_cuda = Path(tmp) / "system-cuda"
        (default_cuda / "bin").mkdir(parents=True)
        (default_cuda / "bin" / "nvlink").touch()
        (default_cuda / "bin" / "nvlink").chmod(0o755)
        (default_cuda / "bin" / "ptxas").touch()
        repo_bootstrap._DEFAULT_CUDA_TOOLCHAIN_ROOT = default_cuda
        repo_bootstrap.sys.prefix = str(active_root)
        os.environ.pop("CONDA_PREFIX", None)
        os.environ.pop("VIRTUAL_ENV", None)
        os.environ.pop("SIMSOPT_JAX_CUDA_LIBRARY_MODE", None)
        os.environ.pop("LD_LIBRARY_PATH", None)
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_libnvjitlink=true "
            "--xla_force_host_platform_device_count=1"
        )
        os.environ["PATH"] = os.pathsep.join([str(default_cuda / "bin"), "/usr/bin"])

        requested = configure_entrypoint_jax_runtime(
            ["--platform", "cuda"],
            respect_existing_env=False,
        )

        assert requested == "cuda"
        assert os.environ["JAX_PLATFORMS"] == "cuda"
        assert os.environ["PATH"].split(os.pathsep)[0] == str(cuda_nvcc_bin)
        assert str(default_cuda / "bin") not in os.environ["PATH"].split(os.pathsep)
        assert "LD_LIBRARY_PATH" not in os.environ
        assert (
            os.environ["XLA_FLAGS"].split()[0]
            == f"--xla_gpu_cuda_data_dir={cuda_nvcc_root}"
        )
        assert (
            "--xla_gpu_enable_libnvjitlink=true" not in os.environ["XLA_FLAGS"].split()
        )
        assert (
            "--xla_force_host_platform_device_count=1"
            in os.environ["XLA_FLAGS"].split()
        )


def case_entrypoint_runtime_helper_accepts_multi_platform_env_list() -> None:
    import os

    from repo_bootstrap import configure_entrypoint_jax_runtime

    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    os.environ.pop("SIMSOPT_JAX_PLATFORM", None)
    os.environ.pop("SIMSOPT_JAX_BACKEND", None)
    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)

    requested = configure_entrypoint_jax_runtime(
        [],
        respect_existing_env=True,
    )

    assert requested == "cuda"
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"
    assert os.environ["SIMSOPT_JAX_PLATFORM"] == "cuda"
    assert os.environ["SIMSOPT_JAX_BACKEND"] == "cuda"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def case_entrypoint_runtime_helper_promotes_cuda_to_cuda_cpu_for_callback_flags() -> (
    None
):
    import os

    from repo_bootstrap import configure_entrypoint_jax_runtime

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.pop("SIMSOPT_JAX_PLATFORM", None)
    os.environ.pop("SIMSOPT_JAX_BACKEND", None)
    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)

    requested = configure_entrypoint_jax_runtime(
        ["--diagnostic-callbacks"],
        respect_existing_env=True,
        require_cpu_platform_when_flags=("--diagnostic-callbacks",),
    )

    assert requested == "cuda"
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"
    assert os.environ["SIMSOPT_JAX_PLATFORM"] == "cuda"
    assert os.environ["SIMSOPT_JAX_BACKEND"] == "cuda"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def case_run_code_benchmark_common_import_is_jax_cold() -> None:
    block_jax_imports(message="blocked jax import for benchmark helper smoke")

    sys.path.insert(0, str(_REPO_ROOT))

    import benchmarks.run_code_benchmark_common as benchmark_common

    assert callable(benchmark_common.resolve_benchmark_backends)


def case_cpu_run_code_benchmark_pins_cpu_before_import() -> None:
    import runpy

    sys.argv = [str(_CPU_RUN_CODE_BENCHMARK_PATH)]
    runpy.run_path(str(_CPU_RUN_CODE_BENCHMARK_PATH), run_name="benchmark_smoke")

    import jax

    assert jax.default_backend() == "cpu"


def case_programmatic_backend_selection_configures_jax_runtime() -> None:
    import simsopt_jax.config as simsopt_config
    import simsopt_jax.backend as backend

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
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == -1


def case_programmatic_backend_persistent_cache_writes_small_kernel() -> None:
    import tempfile

    import simsopt_jax.config as simsopt_config

    with tempfile.TemporaryDirectory(prefix="simsopt-jax-cache-smoke-") as cache_dir:
        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            compilation_cache_dir=cache_dir,
        )

        import jax
        import jax.numpy as jnp

        @jax.jit
        def small_kernel(x):
            return jnp.sin(x).sum()

        result = small_kernel(jnp.arange(16.0, dtype=jnp.float64))
        jax.block_until_ready(result)
        jax.effects_barrier()

        cache_files = [path for path in Path(cache_dir).rglob("*") if path.is_file()]
        assert cache_files, "small JAX kernel did not write a persistent-cache entry"


def _run_shared_persistent_cache_small_kernel(cache_dir: str) -> None:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def small_kernel(x):
        return jnp.sin(x).sum() + jnp.cos(x).sum()

    result = small_kernel(jnp.arange(16.0, dtype=jnp.float64))
    jax.block_until_ready(result)
    jax.effects_barrier()

    cache_files = [path for path in Path(cache_dir).rglob("*") if path.is_file()]
    assert cache_files, "small JAX kernel did not use the shared persistent cache"


def case_programmatic_backend_persistent_cache_shared_small_kernel() -> None:
    import os

    import simsopt_jax.config as simsopt_config

    cache_dir = os.environ["SIMSOPT_TEST_PERSISTENT_CACHE_DIR"]
    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        compilation_cache_dir=cache_dir,
    )
    _run_shared_persistent_cache_small_kernel(cache_dir)


def case_env_backend_persistent_cache_shared_small_kernel() -> None:
    import os

    cache_dir = os.environ["SIMSOPT_TEST_PERSISTENT_CACHE_DIR"]
    os.environ["SIMSOPT_BACKEND_MODE"] = "jax_cpu_parity"
    os.environ["SIMSOPT_BACKEND_STRICT"] = "1"
    os.environ["SIMSOPT_JAX_COMPILATION_CACHE_DIR"] = cache_dir

    import simsopt_jax.config as simsopt_config

    assert simsopt_config.get_compilation_cache_dir() == cache_dir
    simsopt_config.apply_jax_runtime_config()
    _run_shared_persistent_cache_small_kernel(cache_dir)


def case_parity_mode_defaults_transfer_guard_and_keeps_x64_enabled() -> None:
    import simsopt_jax.config as simsopt_config
    import jax

    cfg = simsopt_config.set_backend("jax_cpu_parity")
    policy = simsopt_config.get_backend_policy()

    assert cfg.transfer_guard == "log"
    assert policy.transfer_guard == "log"
    assert policy.requires_x64 is True
    assert jax.config.jax_enable_x64 is True
    assert jax.config.jax_transfer_guard == "log"
    assert jax.numpy.zeros(1).dtype == jax.numpy.float64


def case_env_selected_guardrails_eagerly_configure_jax_runtime() -> None:
    import os

    os.environ["SIMSOPT_BACKEND_MODE"] = "jax_cpu_parity"
    os.environ["SIMSOPT_JAX_DEBUG_NANS"] = "1"
    os.environ["SIMSOPT_JAX_TRANSFER_GUARD"] = "log"

    import simsopt_jax.config as simsopt_config

    simsopt_config.apply_jax_runtime_config()
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


def case_transfer_guard_disallow_rejects_implicit_host_to_device_jit_inputs() -> None:
    import numpy as np
    import simsopt_jax.config as simsopt_config
    import jax
    import jax.numpy as jnp  # noqa: F401

    simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
    fn = jax.jit(lambda x: x + 1.0)

    try:
        fn(np.ones((2,), dtype=np.float64))
    except RuntimeError as exc:
        message = str(exc)
        normalized_message = message.lower()
        assert "transfer" in normalized_message and (
            "guard" in normalized_message or "disallow" in normalized_message
        )
    else:
        raise AssertionError("expected transfer guard to reject implicit host input")


def case_transfer_guard_disallow_allows_target_backend_x64_guard() -> None:
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import require_target_backend_x64

    simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
    require_target_backend_x64("ondevice")


def case_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import (
        PRIVATE_OPTIMIZER_JAX_VERSION,  # noqa: F401
        private_optimizer_runtime_is_supported,
    )
    from simsopt_jax.solve import Driver, SimsoptLBFGSBOptions
    from simsopt_jax.solve.dispatch import minimize

    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )
    if not private_optimizer_runtime_is_supported(jax.__version__):
        _skip_case(f"private optimizer runtime unsupported for JAX {jax.__version__}")

    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def quad(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        return half * jnp.dot(x, x)

    def quad_value_and_grad(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        return half * jnp.dot(x, x), x

    def run_lbfgs(value_and_grad_fun, x0_device):
        callback_points = []
        result = minimize(
            value_and_grad_fun,
            x0_device,
            driver=Driver.SIMSOPT_LBFGSB,
            options=SimsoptLBFGSBOptions(maxiter=5),
            callback=lambda event: callback_points.append(
                np.asarray(event.x, dtype=np.float64)
            ),
        )
        return result, callback_points

    x0_numpy = np.array([1.0, -2.0], dtype=np.float64)
    with jax.transfer_guard_host_to_device("allow"):
        x0 = jnp.asarray(x0_numpy)
    result, callback_points = run_lbfgs(jax.value_and_grad(quad), x0)
    with jax.transfer_guard_host_to_device("allow"):
        x0_from_numpy = jnp.asarray(x0_numpy)
    result_numpy, callback_points_numpy = run_lbfgs(
        jax.value_and_grad(quad),
        x0_from_numpy,
    )
    result_vg, callback_points_vg = run_lbfgs(quad_value_and_grad, x0)

    assert result.success is True
    assert result_numpy.success is True
    assert result_vg.success is True
    assert callback_points
    assert callback_points_numpy
    assert callback_points_vg
    assert float(result.fun) < float(quad(x0))
    assert float(result_numpy.fun) < float(quad(x0))
    assert float(result_vg.fun) < float(quad(x0))


def case_transfer_guard_disallow_allows_target_minimize_structured_pytree_entry() -> (
    None
):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import (
        private_optimizer_runtime_is_supported,
        target_minimize,
    )

    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )
    if not private_optimizer_runtime_is_supported(jax.__version__):
        _skip_case(f"private optimizer runtime unsupported for JAX {jax.__version__}")
    half = jax.device_put(np.asarray(0.5, dtype=np.float64))

    def quad(state):
        return half * (
            jnp.dot(state["surface"], state["surface"])
            + jnp.dot(state["current"], state["current"])
        )

    x0 = {
        "surface": jnp.asarray(np.array([1.0, -2.0], dtype=np.float64)),
        "current": jnp.asarray(np.array([0.5], dtype=np.float64)),
    }
    callback_states = []
    progress_points = []
    result = target_minimize(
        quad,
        x0,
        method="lbfgs-ondevice",
        maxiter=10,
        callback=callback_states.append,
        progress_callback=_record_progress(progress_points),
    )

    assert result.success is True
    assert callback_states
    assert progress_points
    assert isinstance(result.x, dict)
    assert isinstance(result.jac, dict)
    np.testing.assert_allclose(result.x["surface"], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(result.x["current"], np.zeros(1), atol=1e-12)


def case_transfer_guard_disallow_allows_adam_ondevice_quadratic_smokes() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import (
        PRIVATE_OPTIMIZER_JAX_VERSION,  # noqa: F401
        private_optimizer_runtime_is_supported,
    )
    from simsopt_jax.solve import Driver, SimsoptAdamOptions
    from simsopt_jax.solve.dispatch import minimize

    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )
    if not private_optimizer_runtime_is_supported(jax.__version__):
        _skip_case(f"private optimizer runtime unsupported for JAX {jax.__version__}")

    half = jax.device_put(np.asarray(0.5, dtype=np.float64))
    target = jax.device_put(np.asarray([0.25, -0.75], dtype=np.float64))

    def quad(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        diff = x - target
        return half * jnp.dot(diff, diff)

    def quad_value_and_grad(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        diff = x - target
        return half * jnp.dot(diff, diff), diff

    x0 = jnp.asarray(np.array([1.5, -2.5], dtype=np.float64))
    result = minimize(
        quad_value_and_grad,
        x0,
        driver=Driver.SIMSOPT_ADAM,
        options=SimsoptAdamOptions(maxiter=200, gtol=1e-5, learning_rate=0.05),
    )

    assert result.success is True
    assert float(result.fun) < float(quad(x0))
    assert np.allclose(np.asarray(result.x), np.asarray([0.25, -0.75]), atol=1e-4)


def case_transfer_guard_disallow_allows_lm_ondevice_quadratic_smokes() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import (
        PRIVATE_OPTIMIZER_JAX_VERSION,  # noqa: F401
        private_optimizer_runtime_is_supported,
    )
    from simsopt_jax.solve import Driver, SimsoptLMGMRESOptions
    from simsopt_jax.solve.dispatch import least_squares

    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )
    if not private_optimizer_runtime_is_supported(jax.__version__):
        _skip_case(f"private optimizer runtime unsupported for JAX {jax.__version__}")

    x0 = jnp.asarray(np.array([1.5, -2.5], dtype=np.float64))
    target = jax.device_put(np.asarray([0.25, -0.75], dtype=np.float64))

    def residual(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        return x - target

    result = least_squares(
        residual,
        x0,
        driver=Driver.SIMSOPT_LM_GMRES,
        options=SimsoptLMGMRESOptions(maxiter=8),
    )

    assert result.success is True
    assert float(result.fun) < 0.5 * float(jnp.dot(residual(x0), residual(x0)))
    assert np.allclose(np.asarray(result.x), np.asarray([0.25, -0.75]))


def case_transfer_guard_disallow_allows_target_least_squares_structured_entry() -> None:
    import jax  # noqa: F401
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import target_least_squares

    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )

    def residual_fn(state):
        return jnp.asarray(
            [
                state["surface"][0] - 2.0,
                state["surface"][1] + 1.0,
                state["iota"] - 0.25,
            ],
            dtype=jnp.float64,
        )

    x0 = {
        "surface": jnp.asarray(np.array([5.0, 3.0], dtype=np.float64)),
        "iota": jnp.asarray(np.array(0.0, dtype=np.float64)),
    }
    result = target_least_squares(
        residual_fn,
        x0,
        method="lm-ondevice",
        maxiter=25,
        tol=1e-12,
    )

    assert result.success is True
    assert isinstance(result.x, dict)
    assert isinstance(result.jac, dict)
    np.testing.assert_allclose(result.x["surface"], np.array([2.0, -1.0]))
    np.testing.assert_allclose(result.x["iota"], 0.25)


def case_transfer_guard_disallow_allows_ondevice_loops_with_host_closure_constants() -> (
    None
):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import (
        PRIVATE_OPTIMIZER_JAX_VERSION,  # noqa: F401
        private_optimizer_runtime_is_supported,
        target_minimize,
    )

    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )
    if not private_optimizer_runtime_is_supported(jax.__version__):
        _skip_case(f"private optimizer runtime unsupported for JAX {jax.__version__}")

    captured = np.arange(9, dtype=np.float64)
    half = jax.device_put(np.asarray(0.5, dtype=np.float64))
    x0 = jax.device_put(np.ones(9, dtype=np.float64))

    def closure_quad(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        target = jax.device_put(captured)
        diff = x - target
        return half * jnp.dot(diff, diff)

    baseline = float(jax.device_get(closure_quad(x0)))
    bfgs = target_minimize(closure_quad, x0, method="bfgs-ondevice", maxiter=5)
    lbfgs = target_minimize(closure_quad, x0, method="lbfgs-ondevice", maxiter=5)

    assert float(bfgs.fun) < baseline
    assert float(lbfgs.fun) < baseline
    assert int(bfgs.nit) > 0
    assert int(lbfgs.nit) > 0


def case_transfer_guard_disallow_allows_gpu_ondevice_loops_with_host_constants() -> (
    None
):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import (
        PRIVATE_OPTIMIZER_JAX_VERSION,  # noqa: F401
        private_optimizer_runtime_is_supported,
        target_minimize,
    )

    gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
    if gpu is None:
        _skip_case("GPU device is required")
    if not private_optimizer_runtime_is_supported(jax.__version__):
        _skip_case(f"private optimizer runtime unsupported for JAX {jax.__version__}")

    simsopt_config.set_backend(
        "jax_gpu_fast",
        strict=True,
        transfer_guard="disallow",
    )

    captured = np.asarray([1.0, -1.0], dtype=np.float64)
    x0 = jax.device_put(np.asarray([-1.2, 1.0], dtype=np.float64), device=gpu)

    def closure_quad(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        target = jax.device_put(captured, device=gpu)
        diff = x - target
        squared = diff * diff
        return jnp.dot(squared + diff, squared + diff)

    baseline = float(jax.device_get(closure_quad(x0)))
    bfgs = target_minimize(closure_quad, x0, method="bfgs-ondevice", maxiter=2)
    lbfgs = target_minimize(closure_quad, x0, method="lbfgs-ondevice", maxiter=2)

    assert float(bfgs.fun) < baseline
    assert float(lbfgs.fun) < baseline
    assert int(bfgs.nit) >= 2
    assert int(lbfgs.nit) >= 2


def case_transfer_guard_disallow_allows_traceable_newton_with_host_closure_constants() -> (
    None
):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.optimizers.optimizer import newton_polish_traceable

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


def case_transfer_guard_disallow_allows_boozer_residual_host_scalars() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.boozer_residual import (
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


def case_transfer_guard_disallow_allows_boozer_decision_vector_split() -> None:
    import jax
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax.geo.boozer_residual import _split_decision_vector

    simsopt_config.set_backend(
        "jax_cpu_parity",
        strict=True,
        transfer_guard="disallow",
    )

    x = jax.device_put(np.arange(6, dtype=np.float64))
    sdofs, iota, G = _split_decision_vector(x, optimize_G=True)

    assert sdofs.shape == (4,)
    assert iota.shape == ()
    assert G.shape == ()


class _SpecSurface:
    def __init__(self, surface_spec, *, local_full_x):
        self._surface_spec = surface_spec
        self.local_full_x = local_full_x

    def surface_spec(self):
        return self._surface_spec


def _make_rz_spec_surface(*, quadpoints):
    import numpy as np
    from simsopt_jax.core import make_surface_rzfourier_spec

    full_x = np.asarray([1.0, 0.1, 0.1], dtype=np.float64)
    return _SpecSurface(
        make_surface_rzfourier_spec(
            rc=np.asarray([[1.0], [0.1]], dtype=np.float64),
            zs=np.asarray([[0.0], [0.1]], dtype=np.float64),
            quadpoints_phi=quadpoints,
            quadpoints_theta=quadpoints,
            nfp=1,
            stellsym=True,
        ),
        local_full_x=full_x,
    )


def _make_spec_backed_biot_savart_jax(coil_definitions):
    import numpy as np
    from simsopt_jax.core import (
        make_biot_savart_spec,
        make_coil_dof_extraction_spec,
        make_coil_set_dof_extraction_spec,
        make_curve_xyzfourier_spec,
        make_optimizable_dof_map_spec,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import (
        SpecBackedBiotSavartJAX,
    )

    extraction_specs = []
    owner_parts = []
    owner_offset = 0
    for curve_dofs, current_value, nquad in coil_definitions:
        curve_dofs = np.asarray(curve_dofs, dtype=np.float64)
        current_dofs = np.asarray([current_value], dtype=np.float64)
        curve_size = int(curve_dofs.size)

        curve_spec = make_curve_xyzfourier_spec(
            dofs=curve_dofs,
            quadpoints=np.arange(nquad, dtype=np.float64) / nquad,
            order=1,
        )
        curve_map = make_optimizable_dof_map_spec(
            template_full_dofs=curve_dofs,
            owner_segments=((owner_offset, owner_offset + curve_size, 0, curve_size),),
            input_mode="full",
            input_start=0,
            input_end=curve_size,
        )
        current_offset = owner_offset + curve_size
        current_map = make_optimizable_dof_map_spec(
            template_full_dofs=current_dofs,
            owner_segments=((current_offset, current_offset + 1, 0, 1),),
            input_mode="full",
            input_start=0,
            input_end=1,
        )
        extraction_specs.append(
            make_coil_dof_extraction_spec(
                curve=curve_spec,
                curve_map=curve_map,
                current_map=current_map,
            )
        )
        owner_parts.extend((curve_dofs, current_dofs))
        owner_offset = current_offset + 1

    owner_dofs = np.concatenate(owner_parts)
    return SpecBackedBiotSavartJAX(
        make_biot_savart_spec(
            coil_dof_extraction=make_coil_set_dof_extraction_spec(extraction_specs),
            coil_dofs=owner_dofs,
        )
    )


def case_transfer_guard_disallow_allows_squaredfluxjax_construction() -> None:
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
    quadpoints = np.arange(8) / 8
    surface = _make_rz_spec_surface(quadpoints=quadpoints)
    objective = SquaredFluxJAX(
        surface,
        _make_spec_backed_biot_savart_jax(
            (
                (
                    np.asarray([2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
                    1.0,
                    16,
                ),
            )
        ),
    )
    assert objective._flux_spec.normal.shape == (8, 8, 3)

    mixed_objective = SquaredFluxJAX(
        surface,
        _make_spec_backed_biot_savart_jax(
            (
                (
                    np.asarray([2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
                    1.0,
                    16,
                ),
                (
                    np.asarray([2.2, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]),
                    0.5,
                    12,
                ),
            )
        ),
    )
    assert np.isfinite(mixed_objective.J())


def case_transfer_guard_disallow_allows_clamped_xyztensor_surface_spec() -> None:
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt.geo import CurveXYZFourier, SurfaceXYZTensorFourier
    from simsopt.field import Coil, Current
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
    quadpoints = np.arange(8) / 8
    surf = SurfaceXYZTensorFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        clamped_dims=[True, False, False],
        quadpoints_phi=quadpoints,
        quadpoints_theta=quadpoints,
    )
    # Place the coil strictly outside the surface to avoid coincident
    # quadrature points (see case_transfer_guard_disallow_allows_squaredfluxjax_construction).
    curve = CurveXYZFourier(16, 1)
    curve.x = np.array([2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    bs_jax = BiotSavartJAX([Coil(curve, Current(1.0))])
    objective = SquaredFluxJAX(surf, bs_jax)
    assert objective._flux_spec.normal.shape == (8, 8, 3)
    assert np.isfinite(objective.J())


def case_transfer_guard_disallow_rejects_squaredfluxjax_surface_without_spec() -> None:
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

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

    bs_jax = _make_spec_backed_biot_savart_jax(
        (
            (
                np.asarray([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
                1.0,
                16,
            ),
        )
    )
    try:
        SquaredFluxJAX(HostSurface(), bs_jax)
    except NotImplementedError as exc:
        assert "surface_spec" in str(exc)
    else:
        raise AssertionError(
            "expected SquaredFluxJAX to reject surfaces without surface_spec()"
        )


def case_transfer_guard_disallow_allows_lpcurveforce_shared_state_packing() -> None:
    import jax
    import numpy as np
    import simsopt_jax.config as simsopt_config
    from simsopt.field import Current, RegularizedCoil
    from simsopt.geo import CurveXYZFourier
    from simsopt_jax_adapters.field.force import LpCurveForce

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


def case_native_cpu_backend_selection_does_not_require_jax_runtime() -> None:
    block_jax_imports(message="blocked jax import for native_cpu smoke")

    import simsopt_jax.config as simsopt_config

    cfg = simsopt_config.set_backend(
        "native_cpu",
        debug_nans=True,
        transfer_guard="log",
        compilation_cache_dir="/tmp/ignored-native-cache",
    )
    assert cfg.mode == "native_cpu"
    assert cfg.backend == "cpu"


def case_native_cpu_policy_does_not_configure_jax_without_selector() -> None:
    import simsopt_jax.config as simsopt_config
    import jax

    policy = simsopt_config.get_backend_policy()

    assert policy.mode == "native_cpu"
    assert policy.requires_x64 is True
    assert jax.config.jax_enable_x64 is False
    assert jax.numpy.zeros(1).dtype == jax.numpy.float32


def case_import_biotsavart_jax() -> None:
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

    assert BiotSavartJAX is not None


def case_import_jax_core_specs() -> None:
    block_simsoptpp_imports()

    from simsopt_jax.core import (
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
        SurfaceSpec,
        SurfaceSpecKind,
        SurfaceXYZFourierSpec,
        SurfaceXYZTensorFourierSpec,
        ZeroRotationSpec,
        curve_spec_kind,
        surface_spec_kind,
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
    assert SurfaceSpec is not None
    assert SurfaceSpecKind is not None
    assert SurfaceXYZFourierSpec is not None
    assert SurfaceXYZTensorFourierSpec is not None
    assert ZeroRotationSpec is not None
    assert curve_spec_kind is not None
    assert surface_spec_kind is not None
    assert "simsopt._core" not in sys.modules
    assert "simsoptpp" not in sys.modules


_EXPECTED_JAX_CORE_PUBLIC_EXPORTS = tuple(
    """
BiotSavartSpec build_filament_gamma_and_dash build_filament_gammas centroid_frame
centroid_frame_dash compute_filament_offsets frenet_frame frenet_frame_dash
rotated_centroid_frame rotated_centroid_frame_dash rotated_frenet_frame rotated_frenet_frame_dash
rotation_alpha rotation_alphadash rotation_dcoeff rotationdash_dcoeff
CoilSpec CoilGroupSpec CoilDofExtractionSpec CoilSetDofExtractionSpec
CoilSymmetrySpec apply_coil_symmetry CurveCWSFourierRZSpec CurveFilamentSpec
CurveHelicalSpec OrientedCurveXYZFourierSpec make_oriented_curve_xyzfourier_spec CurvePlanarFourierSpec
CurveSpec CurveSpecKind CurvePerturbedSpec CurrentValueSpec
CurveRZFourierSpec CurveXYZFourierSpec CurveXYZFourierSymmetriesSpec FieldEvalSpec
FrameRotationSpec FixedSurfaceFluxSpec GroupedCoilSetSpec OptimizableDofMapSpec
RotationSpec SingleStageRuntimeSpec SingleStageSeedSpec SurfaceGarabedianSpec
SurfaceHennebergSpec SurfaceRZFourierSpec SurfaceSpec SurfaceSpecKind
SurfaceXYZFourierSpec SurfaceXYZTensorFourierSpec ZeroRotationSpec build_fourier_basis
closed_curve_self_intersection_min_distance closed_curve_self_intersection_penalty closed_curve_self_intersection_summary closed_curve_self_intersection_tolerance
curve_dincremental_arclength_by_dcoeff_from_dofs curve_dincremental_arclength_by_dcoeff_vjp_from_dofs curve_dkappa_by_dcoeff_from_dofs curve_dkappa_by_dcoeff_vjp_from_dofs
curve_dtorsion_by_dcoeff_from_dofs curve_dtorsion_by_dcoeff_vjp_from_dofs curve_gamma_and_dash_from_dofs curve_gamma_and_dash_from_spec
curve_gamma_vjp_from_dofs curve_geometry_from_dofs curve_geometry_from_spec curve_gammadash_vjp_from_dofs
curve_gammadashdash_vjp_from_dofs curve_gammadashdashdash_vjp_from_dofs curve_incremental_arclength_from_dofs curve_incremental_arclength_from_spec
curve_kappa_from_dofs curve_kappa_from_spec curve_pullback_from_dofs curve_pullback_from_spec
curve_spec_kind surface_spec_kind curve_spec_from_curve curve_spec_with_dofs
curve_spec_with_quadpoints curve_torsion_from_dofs curve_torsion_from_spec coil_set_spec_from_dof_extraction_spec
coil_specs_from_dof_extraction_spec CircularCoilSpec circular_coil_A circular_coil_B
circular_coil_dB biot_savart_A biot_savart_B biot_savart_B_and_dB
biot_savart_B_vjp biot_savart_dA_by_dX biot_savart_dB_by_dX group_coil_data
grouped_biot_savart_A grouped_biot_savart_B invalidate_kernel_cache FLUX_FUNCTION_SCALARS
InterpolatedBoozerFieldFrozenState SYMMETRY_EXPLOIT_SCALARS build_flux_function_interpolant build_symmetry_exploit_interpolant
evaluate_interpolated_boozer_scalar fold_points_for_symmetry freeze_interpolated_boozer_field_state fixed_surface_flux_specs_from_surface
fixed_surface_flux_integral fixed_surface_flux_integral_from_B VALID_REDUCTION_MODES compensated_sum_flat
pairwise_sum_axis pairwise_sum_flat scalar_square_sum validate_reduction_mode
boozer_quasisymmetry_mode_indices boozer_quasisymmetry_residuals iota_target_metric_j iota_weighted_j
well_weighted_j grouped_coil_set_spec_from_coil_specs grouped_biot_savart_A_from_inputs grouped_biot_savart_A_from_spec
GroupedBiotSavartBDispatchEvidence GroupedBiotSavartBDispatchResult GroupedBiotSavartBDispatchRole grouped_biot_savart_B_and_dB_from_spec grouped_biot_savart_B_from_inputs grouped_biot_savart_B_from_spec grouped_biot_savart_B_from_spec_with_evidence grouped_biot_savart_d2A_by_dXdX_from_inputs
grouped_biot_savart_d2A_by_dXdX_from_spec grouped_biot_savart_d2B_by_dXdX_from_inputs grouped_biot_savart_d2B_by_dXdX_from_spec grouped_biot_savart_dA_by_dX_from_inputs
grouped_biot_savart_dA_by_dX_from_spec grouped_biot_savart_dB_by_dX_from_inputs grouped_biot_savart_dB_by_dX_from_spec grouped_coil_currents_from_inputs
grouped_coil_currents_from_spec grouped_coil_index_lists_from_spec grouped_coil_set_spec_from_grouped_data grouped_coil_set_spec_from_inputs
grouped_coil_set_spec_from_lists grouped_coil_set_spec_from_source grouped_field_data_from_spec grouped_field_inputs_from_spec
make_biot_savart_spec make_coil_spec make_coil_dof_extraction_spec make_coil_symmetry_spec
make_coil_group_spec make_coil_set_dof_extraction_spec make_current_value_spec make_curve_cwsfourier_rz_spec
make_curve_filament_spec make_curve_helical_spec make_curve_planarfourier_spec make_curve_perturbed_spec
make_curve_rzfourier_spec make_curve_xyzfourier_spec make_curve_xyzfouriersymmetries_spec make_field_eval_spec
make_frame_rotation_spec make_fixed_surface_flux_spec make_grouped_coil_set_spec make_optimizable_dof_map_spec
make_single_stage_runtime_spec make_single_stage_seed_spec make_surface_garabedian_spec make_surface_henneberg_spec
make_surface_rzfourier_spec make_zero_rotation_spec garabedian_to_rzfourier_spec pair_linking_number_pure
segment_segment_distance_pure surface_rz_fourier_aspect_ratio_from_spec surface_rz_fourier_aspect_ratio_from_dofs surface_rz_fourier_area_from_spec
surface_rz_fourier_area_from_dofs surface_rz_fourier_d2area_from_dofs surface_rz_fourier_d2aspect_ratio_from_dofs surface_rz_fourier_d2major_radius_from_dofs
surface_rz_fourier_d2minor_radius_from_dofs surface_rz_fourier_d2volume_from_dofs surface_rz_fourier_darea_from_dofs surface_rz_fourier_daspect_ratio_from_dofs
surface_rz_fourier_dfirst_fund_form_from_dofs surface_rz_fourier_dofs_from_spec surface_rz_fourier_dmajor_radius_from_dofs surface_rz_fourier_dmean_cross_sectional_area_from_dofs
surface_rz_fourier_dminor_radius_from_dofs surface_rz_fourier_dsecond_fund_form_from_dofs surface_rz_fourier_dsurface_curvatures_from_dofs surface_rz_fourier_dvolume_from_dofs
surface_rz_fourier_first_fund_form_from_spec surface_rz_fourier_first_fund_form_from_dofs surface_rz_fourier_gamma_from_spec surface_rz_fourier_gamma_from_dofs
surface_rz_fourier_gamma_lin_from_spec surface_rz_fourier_gamma_lin_from_dofs
surface_rz_fourier_gammadash1dash1_from_spec surface_rz_fourier_gammadash1dash1_from_dofs surface_rz_fourier_gammadash1dash1dash1_lin_from_spec surface_rz_fourier_gammadash1dash1dash1_lin_from_dofs
surface_rz_fourier_gammadash1dash1_vjp_from_dofs surface_rz_fourier_gammadash1dash2_from_spec surface_rz_fourier_gammadash1dash2_from_dofs surface_rz_fourier_gammadash1dash1dash2_lin_from_spec
surface_rz_fourier_gammadash1dash1dash2_lin_from_dofs surface_rz_fourier_gammadash1dash2dash2_lin_from_spec surface_rz_fourier_gammadash1dash2dash2_lin_from_dofs surface_rz_fourier_gammadash1dash2_vjp_from_dofs
surface_rz_fourier_gammadash1_from_spec surface_rz_fourier_gammadash1_from_dofs surface_rz_fourier_gammadash2dash2_from_spec surface_rz_fourier_gammadash2dash2_from_dofs
surface_rz_fourier_gammadash2dash2dash2_lin_from_spec surface_rz_fourier_gammadash2dash2dash2_lin_from_dofs surface_rz_fourier_gammadash2dash2_vjp_from_dofs surface_rz_fourier_gammadash2_from_spec
surface_rz_fourier_gammadash2_from_dofs surface_rz_fourier_major_radius_from_spec surface_rz_fourier_major_radius_from_dofs surface_rz_fourier_mean_cross_sectional_area_from_spec
surface_rz_fourier_mean_cross_sectional_area_from_dofs surface_rz_fourier_minor_radius_from_spec surface_rz_fourier_minor_radius_from_dofs surface_rz_fourier_normal_from_spec
surface_rz_fourier_normal_from_dofs surface_rz_fourier_second_fund_form_from_spec surface_rz_fourier_second_fund_form_from_dofs surface_rz_fourier_dgammadash1dash1_from_dofs
surface_rz_fourier_dgammadash1dash2_from_dofs surface_rz_fourier_dgammadash2dash2_from_dofs surface_rz_fourier_dnormal_from_dofs surface_rz_fourier_spec_from_dofs
surface_rz_fourier_surface_curvatures_from_spec surface_rz_fourier_surface_curvatures_from_dofs surface_rz_fourier_unitnormal_from_spec surface_rz_fourier_unitnormal_from_dofs
surface_rz_fourier_dunitnormal_from_dofs surface_rz_fourier_volume_from_spec surface_rz_fourier_volume_from_dofs surface_xyz_fourier_area_from_spec
surface_xyz_fourier_dfirst_fund_form_from_dofs surface_xyz_fourier_dsecond_fund_form_from_dofs surface_xyz_fourier_dsurface_curvatures_from_dofs surface_xyz_fourier_first_fund_form_from_spec
surface_xyz_fourier_first_fund_form_from_dofs surface_xyz_fourier_gamma_from_spec surface_xyz_fourier_gammadash1_from_spec surface_xyz_fourier_gammadash1dash1_from_spec
surface_xyz_fourier_gammadash1dash1_lin_from_spec surface_xyz_fourier_gammadash1dash1_lin_from_dofs surface_xyz_fourier_gammadash1dash1dash1_lin_from_spec surface_xyz_fourier_gammadash1dash1dash1_lin_from_dofs
surface_xyz_fourier_gammadash1dash1dash2_lin_from_spec surface_xyz_fourier_gammadash1dash1dash2_lin_from_dofs surface_xyz_fourier_gammadash1dash2_from_spec surface_xyz_fourier_gammadash1dash2_lin_from_spec
surface_xyz_fourier_gammadash1dash2_lin_from_dofs surface_xyz_fourier_gammadash1dash2dash2_lin_from_spec surface_xyz_fourier_gammadash1dash2dash2_lin_from_dofs surface_xyz_fourier_gammadash2_from_spec
surface_xyz_fourier_gammadash2dash2_from_spec surface_xyz_fourier_gammadash2dash2_lin_from_spec surface_xyz_fourier_gammadash2dash2_lin_from_dofs surface_xyz_fourier_gammadash2dash2dash2_lin_from_spec
surface_xyz_fourier_gammadash2dash2dash2_lin_from_dofs surface_xyz_fourier_normal_from_spec surface_xyz_fourier_second_fund_form_from_spec surface_xyz_fourier_second_fund_form_from_dofs
surface_xyz_fourier_surface_curvatures_from_spec surface_xyz_fourier_surface_curvatures_from_dofs surface_xyz_fourier_unitnormal_from_spec surface_xyz_fourier_volume_from_spec
surface_henneberg_area_from_spec surface_henneberg_gamma_from_spec surface_henneberg_gammadash1_from_spec surface_henneberg_gammadash2_from_spec
surface_henneberg_normal_from_spec surface_henneberg_unitnormal_from_spec surface_henneberg_volume_from_spec surface_xyz_tensor_fourier_area_from_spec
surface_xyz_tensor_fourier_dfirst_fund_form_from_dofs surface_xyz_tensor_fourier_dsecond_fund_form_from_dofs surface_xyz_tensor_fourier_dsurface_curvatures_from_dofs surface_xyz_tensor_fourier_first_fund_form_from_spec
surface_xyz_tensor_fourier_first_fund_form_from_dofs surface_xyz_tensor_fourier_gamma_from_spec surface_xyz_tensor_fourier_gammadash1_from_spec surface_xyz_tensor_fourier_gammadash1dash1_from_spec
surface_xyz_tensor_fourier_gammadash1dash1_lin_from_spec surface_xyz_tensor_fourier_gammadash1dash1_lin_from_dofs surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_spec surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_dofs
surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_spec surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_dofs surface_xyz_tensor_fourier_gammadash1dash2_from_spec surface_xyz_tensor_fourier_gammadash1dash2_lin_from_spec
surface_xyz_tensor_fourier_gammadash1dash2_lin_from_dofs surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_spec surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_dofs surface_xyz_tensor_fourier_gammadash2_from_spec
surface_xyz_tensor_fourier_gammadash2dash2_from_spec surface_xyz_tensor_fourier_gammadash2dash2_lin_from_spec surface_xyz_tensor_fourier_gammadash2dash2_lin_from_dofs surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_spec
surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_dofs surface_xyz_tensor_fourier_normal_from_spec surface_xyz_tensor_fourier_second_fund_form_from_spec surface_xyz_tensor_fourier_second_fund_form_from_dofs
surface_xyz_tensor_fourier_surface_curvatures_from_spec surface_xyz_tensor_fourier_surface_curvatures_from_dofs surface_xyz_tensor_fourier_unitnormal_from_spec surface_xyz_tensor_fourier_volume_from_spec
make_surface_xyz_fourier_spec make_surface_xyz_tensor_fourier_spec
""".split()
)


def case_jax_core_public_contract() -> None:
    block_simsoptpp_imports()

    import simsopt_jax.core as jax_core

    for module_name in (
        "simsopt_jax.core.curve_geometry",
        "simsopt_jax.core.field",
        "simsopt_jax.core.surface_rzfourier",
    ):
        assert module_name in sys.modules

    required_exports = (
        "BiotSavartSpec",
        "build_filament_gamma_and_dash",
        "centroid_frame",
        "CurveXYZFourierSymmetriesSpec",
        "curve_spec_kind",
        "curve_gamma_and_dash_from_spec",
        "coil_set_spec_from_dof_extraction_spec",
        "CircularCoilSpec",
        "biot_savart_B",
        "evaluate_interpolated_boozer_scalar",
        "fixed_surface_flux_integral_from_B",
        "pairwise_sum_axis",
        "iota_weighted_j",
        "surface_rz_fourier_gamma_from_spec",
        "surface_xyz_tensor_fourier_gamma_from_spec",
        "surface_henneberg_gamma_from_spec",
    )
    assert tuple(jax_core.__all__) == _EXPECTED_JAX_CORE_PUBLIC_EXPORTS
    for export_name in required_exports:
        assert export_name in _EXPECTED_JAX_CORE_PUBLIC_EXPORTS
        assert getattr(jax_core, export_name) is not None

    namespace: dict[str, object] = {}
    exec("from simsopt_jax.core import *", namespace)
    exported_names = tuple(name for name in namespace if not name.startswith("__"))
    assert exported_names == _EXPECTED_JAX_CORE_PUBLIC_EXPORTS
    assert "simsopt._core" not in sys.modules
    assert "simsoptpp" not in sys.modules


def case_jax_core_specs_are_pytrees() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from simsopt_jax.core import (
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
        SurfaceXYZFourierSpec,
        SurfaceXYZTensorFourierSpec,
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
        make_surface_xyz_fourier_spec,
        make_surface_xyz_tensor_fourier_spec,
        make_zero_rotation_spec,
        surface_rz_fourier_dofs_from_spec,  # noqa: F401
        surface_rz_fourier_gamma_from_spec,
        surface_spec_kind,
        surface_xyz_fourier_gamma_from_spec,
        surface_xyz_tensor_fourier_gamma_from_spec,
    )

    coil_spec = make_grouped_coil_set_spec(
        [
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
        ]
    )
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
        dofs=jnp.asarray(
            [1.1, 0.2, -0.1, 0.05, -0.03, 1.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.05]
        ),
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
    filament_owner_dofs = jnp.asarray(
        [0.1, -0.03, 0.02, 0.04, -0.01, 0.07, -0.03, 0.02]
    )
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
    surface_xyz_spec = make_surface_xyz_fourier_spec(
        dofs=jnp.asarray([1.0, 0.1, 0.0, 0.1]),
        quadpoints_phi=jnp.asarray([0.0, 0.5]),
        quadpoints_theta=jnp.asarray([0.0, 0.5]),
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
    )
    surface_xyztensor_spec = make_surface_xyz_tensor_fourier_spec(
        dofs=jnp.asarray([1.0, 0.1, 0.0, 0.1]),
        quadpoints_phi=jnp.asarray([0.0, 0.5]),
        quadpoints_theta=jnp.asarray([0.0, 0.5]),
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
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

    def assert_surface_dofs_derivable(cs, expected_ndofs):
        derived = cs.surface_dofs()
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
    assert isinstance(surface_xyz_spec, SurfaceXYZFourierSpec)
    assert isinstance(surface_xyztensor_spec, SurfaceXYZTensorFourierSpec)
    assert isinstance(zero_rotation_spec, ZeroRotationSpec)
    assert surface_spec_kind(surface_spec) == "rz_fourier"
    assert surface_spec_kind(surface_xyz_spec) == "xyz_fourier"
    assert surface_spec_kind(surface_xyztensor_spec) == "xyz_tensor_fourier"

    curve_xyz_leaves, _ = jax.tree.flatten(curve_xyz_spec)
    curve_rz_leaves, _ = jax.tree.flatten(curve_rz_spec)
    curve_planar_leaves, _ = jax.tree.flatten(curve_planar_spec)
    curve_helical_leaves, _ = jax.tree.flatten(curve_helical_spec)
    curve_cws_leaves, _ = jax.tree.flatten(curve_cws_spec)
    curve_perturbed_leaves, _ = jax.tree.flatten(curve_perturbed_spec)
    curve_filament_leaves, _ = jax.tree.flatten(curve_filament_spec)
    coil_symmetry_leaves, _ = jax.tree.flatten(coil_symmetry_spec)
    current_leaves, _ = jax.tree.flatten(current_spec)
    field_eval_leaves, _ = jax.tree.flatten(field_eval_spec)
    coil_value_leaves, _ = jax.tree.flatten(coil_value_spec)
    coil_leaves, _ = jax.tree.flatten(coil_spec)
    flux_leaves, _ = jax.tree.flatten(flux_spec)
    frame_rotation_leaves, _ = jax.tree.flatten(frame_rotation_spec)
    dof_map_leaves, _ = jax.tree.flatten(identity_curve_map)
    surface_leaves, _ = jax.tree.flatten(surface_spec)
    surface_xyz_leaves, _ = jax.tree.flatten(surface_xyz_spec)
    surface_xyztensor_leaves, _ = jax.tree.flatten(surface_xyztensor_spec)
    zero_rotation_leaves, _ = jax.tree.flatten(zero_rotation_spec)

    def assert_round_trip(spec):
        leaves, treedef = jax.tree.flatten(spec)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        rebuilt_leaves, rebuilt_treedef = jax.tree.flatten(rebuilt)
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
    assert len(surface_xyz_leaves) == 5
    assert len(surface_xyztensor_leaves) == 4
    assert len(zero_rotation_leaves) == 1
    assert len(grouped_field_inputs_from_spec(coil_spec)) == 1
    assert len(grouped_field_data_from_spec(coil_spec)) == 1
    assert grouped_coil_index_lists_from_spec(coil_spec) == ([0],)
    assert grouped_coil_currents_from_spec(coil_spec).shape == (1,)
    assert grouped_coil_set_spec_from_coil_specs((coil_value_spec,)).groups[
        0
    ].coil_indices == (0,)
    assert grouped_coil_set_spec_from_source(coil_spec) is coil_spec
    assert callable(invalidate_kernel_cache)
    assert_surface_dofs_derivable(curve_cws_spec, 3)  # stellsym: 2 rc + 1 zs
    assert_surface_dofs_derivable(curve_cws_nonstellsym_spec, 6)  # 2rc+1rs+2zc+1zs
    assert_round_trip(curve_perturbed_spec)
    assert_round_trip(curve_filament_spec)
    assert_round_trip(surface_xyz_spec)
    assert_round_trip(surface_xyztensor_spec)
    assert_round_trip(coil_spec)

    curve_xyz_gamma, curve_xyz_gammadash = jax.jit(curve_gamma_and_dash_from_spec)(
        curve_xyz_spec
    )
    curve_rz_gamma, _ = jax.jit(curve_gamma_and_dash_from_spec)(curve_rz_spec)
    curve_cws_gamma, curve_cws_gammadash = jax.jit(curve_gamma_and_dash_from_spec)(
        curve_cws_spec
    )
    curve_perturbed_gamma, curve_perturbed_gammadash = jax.jit(
        curve_gamma_and_dash_from_spec
    )(curve_perturbed_spec)
    curve_filament_gamma, curve_filament_gammadash = jax.jit(
        curve_gamma_and_dash_from_spec
    )(curve_filament_spec)
    curve_cws_gamma_from_dofs, curve_cws_gammadash_from_dofs = jax.jit(
        curve_gamma_and_dash_from_dofs
    )(
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
    xyz_gamma = jax.jit(surface_xyz_fourier_gamma_from_spec)(surface_xyz_spec)
    xyztensor_gamma = jax.jit(surface_xyz_tensor_fourier_gamma_from_spec)(
        surface_xyztensor_spec
    )

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
    assert xyz_gamma.shape == (2, 2, 3)
    assert xyztensor_gamma.shape == (2, 2, 3)
    assert np.isfinite(np.asarray(value))


def case_jax_core_grouped_field_chunking_matches_dense_sum() -> None:
    import jax
    import jax.numpy as jnp

    from simsopt_jax import config as simsopt_config
    from simsopt_jax.field.biotsavart import (
        biot_savart_B,
        biot_savart_B_and_dB,
        biot_savart_dB_by_dX,
    )
    from simsopt_jax.core import (
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
    B_combo, dB_combo = jax.jit(grouped_biot_savart_B_and_dB_from_spec)(
        points, coil_spec
    )

    B_combo_ref, dB_combo_ref = _sum_group_combo(groups)

    assert B.shape == (300, 3)
    assert dB.shape == (300, 3, 3)
    assert B_combo.shape == (300, 3)
    assert dB_combo.shape == (300, 3, 3)
    assert jnp.allclose(B, B_ref, rtol=1e-12, atol=1e-14)
    assert jnp.allclose(dB, dB_ref, rtol=1e-12, atol=1e-14)
    assert jnp.allclose(B_combo, B_combo_ref, rtol=1e-12, atol=1e-14)
    assert jnp.allclose(dB_combo, dB_combo_ref, rtol=1e-12, atol=1e-14)


def case_import_squaredflux_jax() -> None:
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    assert SquaredFluxJAX is not None


def case_import_boozersurface_jax() -> None:
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

    assert BoozerSurfaceJAX is not None


def case_import_core_optimizable() -> None:
    from simsopt._core.optimizable import Optimizable

    assert Optimizable is not None


def case_jax_classes_inherit_optimizable() -> None:
    from simsopt._core.optimizable import Optimizable
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    assert issubclass(BiotSavartJAX, Optimizable)
    assert issubclass(SquaredFluxJAX, Optimizable)


def case_import_pure_jax_modules() -> None:
    from simsopt_jax.field.biotsavart import biot_savart_B
    from simsopt_jax.geo.surface_fourier import stellsym_scatter_indices
    from simsopt_jax.geo.boozer_residual import boozer_residual_scalar
    from simsopt_jax.objectives.integral_bdotn import integral_BdotN

    assert callable(biot_savart_B)
    assert callable(stellsym_scatter_indices)
    assert callable(boozer_residual_scalar)
    assert callable(integral_BdotN)


def case_public_jax_helpers_are_exposed_on_package_roots() -> None:
    """Pin the public export surface for the JAX solve/geo helpers.

    Backend runtime helpers must resolve through the public
    ``simsopt_jax.backend`` package facade, not the shadowed legacy module path.
    JAX solve/geo helpers resolve through the canonical ``simsopt_jax``
    package tree, not the CPU package roots.
    """
    import simsopt_jax.backend as backend
    from simsopt_jax.backend import (
        get_tolerance_tier,
        raise_if_target_lane_bypass,
        strict_target_lane_purity,
        target_lane_purity_active,
        target_lane_purity_requested,
    )
    import simsopt.geo as geo
    import simsopt.solve as solve
    import simsopt_jax.geo.permanent_magnet_grid as permanent_magnet_grid
    import simsopt_jax.solve.permanent_magnet as permanent_magnet_solve
    import simsopt_jax_adapters.solve.wireframe as wireframe_solve
    from simsopt_jax_adapters.geo.surface_objectives import (
        coil_dofs_gradient_to_derivative,
    )

    backend_path = Path(backend.__file__).resolve()
    assert backend_path.parent == _SRC_DIR / "simsopt_jax" / "backend"
    assert backend_path.name == "__init__.py"
    backend_helpers = (
        get_tolerance_tier,
        raise_if_target_lane_bypass,
        strict_target_lane_purity,
        target_lane_purity_active,
        target_lane_purity_requested,
    )
    assert all(callable(helper) for helper in backend_helpers)

    permanent_magnet_helpers = (
        "relax_and_split_jax",
        "GPMO_ArbVec_backtracking_jax",
        "setup_initial_condition_jax",
    )
    missing_pm_solve = [
        name
        for name in permanent_magnet_helpers
        if not hasattr(permanent_magnet_solve, name)
    ]
    assert not missing_pm_solve, (
        "simsopt_jax.solve.permanent_magnet is missing public JAX helpers: "
        f"{missing_pm_solve}"
    )

    wireframe_helpers = (
        "optimize_wireframe_jax",
        "greedy_stellarator_coil_optimization_jax",
        "regularized_constrained_least_squares_jax",
    )
    missing_wireframe_solve = [
        name for name in wireframe_helpers if not hasattr(wireframe_solve, name)
    ]
    assert not missing_wireframe_solve, (
        "simsopt_jax_adapters.solve.wireframe is missing public JAX helpers: "
        f"{missing_wireframe_solve}"
    )

    geo_helpers = ("PermanentMagnetGridJAX", "permanent_magnet_grid_to_jax")
    missing_geo = [
        name for name in geo_helpers if not hasattr(permanent_magnet_grid, name)
    ]
    assert not missing_geo, (
        "simsopt_jax.geo.permanent_magnet_grid is missing public helpers: "
        f"{missing_geo}"
    )
    assert callable(coil_dofs_gradient_to_derivative)

    old_root_jax_names = (
        *permanent_magnet_helpers,
        *wireframe_helpers,
        "PermanentMagnetGridJAX",
        "permanent_magnet_grid_to_jax",
        "coil_dofs_gradient_to_derivative",
    )
    leaked_solve = [name for name in old_root_jax_names if hasattr(solve, name)]
    leaked_geo = [name for name in old_root_jax_names if hasattr(geo, name)]
    assert not leaked_solve, f"simsopt.solve leaked JAX helpers: {leaked_solve}"
    assert not leaked_geo, f"simsopt.geo leaked JAX helpers: {leaked_geo}"


def case_m5_classes_require_simsoptpp() -> None:
    import simsopt.geo as geo
    import simsopt_jax_adapters.geo.surface_objectives as surface_objectives

    names = ("BoozerResidualJAX", "IotasJAX", "NonQuasiSymmetricRatioJAX")
    for name in names:
        assert hasattr(surface_objectives, name), (
            f"simsopt_jax_adapters.geo.surface_objectives is missing {name}"
        )
        assert not hasattr(geo, name), f"simsopt.geo leaked JAX wrapper {name}"


def case_direct_optional_geo_modules_import_without_simsoptpp() -> None:
    block_simsoptpp_imports()

    try:
        import simsopt.geo.curveobjectives  # noqa: F401
    except ModuleNotFoundError as exc:
        assert getattr(exc, "name", None) == "simsoptpp" or "simsoptpp" in str(exc)
    else:
        raise AssertionError("curveobjectives should require simsoptpp at import time")


def case_framedcurve_direct_module_import_smoke() -> None:
    import simsopt.geo.framedcurve as framedcurve

    assert hasattr(framedcurve, "FramedCurve")
    assert hasattr(framedcurve, "FramedCurveCentroid")
    assert hasattr(framedcurve, "FramedCurveFrenet")


def case_import_cpu_package_entrypoints_with_simsoptpp() -> None:
    import simsopt.configs  # noqa: F401
    import simsopt.field
    import simsopt.geo
    import simsopt.objectives
    import simsopt.solve  # noqa: F401
    import simsopt.util  # noqa: F401

    assert hasattr(simsopt.field, "BiotSavart")
    assert hasattr(simsopt.geo, "BoozerSurface")
    assert hasattr(simsopt.objectives, "LeastSquaresProblem")


def case_import_core_json_without_jax() -> None:
    strip_simsopt_editable_finders()

    block_jax_imports(message="blocked jax import for core json CPU smoke")

    import simsopt._core.json as simson_json

    assert hasattr(simson_json, "GSONEncoder")
    assert "jax" not in sys.modules


def case_import_jax_solve_driver_without_optimizer_runtime_libraries() -> None:
    strip_simsopt_editable_finders()

    import simsopt_jax.solve.driver as driver

    assert hasattr(driver, "Driver")
    unexpected = [
        name for name in ("optax", "optimistix", "lineax") if name in sys.modules
    ]
    assert not unexpected, (
        f"solve.driver loaded optimizer runtime libraries: {unexpected}"
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _case_invariant(case_name: str) -> str:
    if "transfer_guard" in case_name:
        return "transfer_guard"
    if "entrypoint_runtime" in case_name or "backend" in case_name:
        return "runtime_configuration"
    if "repo_bootstrap" in case_name:
        return "repo_bootstrap"
    if "simsoptpp" in case_name:
        return "optional_cpp_import"
    if "jax" in case_name:
        return "jax_import_contract"
    return "import_contract"


def _case_payload(
    case_name: str,
    *,
    checked: bool,
    skipped: bool,
    skip_reason: str | None = None,
) -> dict[str, object]:
    simsopt_module_count = sum(
        name == "simsopt" or name.startswith("simsopt.") for name in sys.modules
    )
    payload: dict[str, object] = {
        "case": case_name,
        "checked": checked,
        "invariant": _case_invariant(case_name),
        "loaded_module_count": len(sys.modules),
        "skipped": skipped,
        "simsopt_module_count": simsopt_module_count,
    }
    if skip_reason is not None:
        payload["skip_reason"] = skip_reason
    return payload


if __name__ == "__main__":
    prefer_local_simsopt_source_tree()
    case_name = sys.argv[1]
    fn = globals().get(case_name)
    if fn is None or not callable(fn):
        print(f"Unknown case: {case_name}", file=sys.stderr)
        sys.exit(2)
    try:
        fn()
    except SkippedCase as exc:
        payload = _case_payload(
            case_name,
            checked=False,
            skipped=True,
            skip_reason=str(exc),
        )
    else:
        payload = _case_payload(case_name, checked=True, skipped=False)
    print(json.dumps(payload, sort_keys=True))
