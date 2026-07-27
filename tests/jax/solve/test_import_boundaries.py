import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
RUNTIME_OPTIMIZER_MODULES = ("optax", "optimistix", "lineax")


def _benchmark_imports(path: Path) -> tuple[str, ...]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "benchmarks" or node.module.startswith("benchmarks."):
                imports.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "benchmarks" or alias.name.startswith("benchmarks."):
                    imports.append(f"line {node.lineno}: import {alias.name}")
    return tuple(imports)


def _run_python_import_probe(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    local_pythonpath = os.pathsep.join((str(REPO_ROOT), str(SRC_ROOT)))
    env["PYTHONPATH"] = (
        local_pythonpath
        if existing_pythonpath is None
        else os.pathsep.join((local_pythonpath, existing_pythonpath))
    )
    probe_source = (
        textwrap.dedent(
            """
            from pathlib import Path
            import sys

            sys.meta_path = [
                finder
                for finder in sys.meta_path
                if finder.__class__.__module__ != "_simsopt_editable"
            ]

            from repo_bootstrap import bootstrap_local_simsopt

            bootstrap_local_simsopt(Path.cwd() / "src")
            """
        )
        + "\n"
        + textwrap.dedent(source)
    )
    return subprocess.run(
        [sys.executable, "-c", probe_source],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_driver_ssot_import_does_not_load_optimizer_runtime_libraries():
    result = _run_python_import_probe(
        """
        import sys
        import simsopt_jax.solve.driver

        unexpected = [
            name
            for name in ("optax", "optimistix", "lineax")
            if name in sys.modules
        ]
        if unexpected:
            raise SystemExit(f"unexpected runtime imports: {unexpected}")
        """
    )

    assert result.returncode == 0, result.stderr


def test_solve_package_import_does_not_import_public_jax_runtime_api():
    result = _run_python_import_probe(
        """
        import sys
        import simsopt.solve

        if "simsopt_jax.solve" in sys.modules:
            raise SystemExit("simsopt.solve imported simsopt_jax.solve")
        """
    )

    assert result.returncode == 0, result.stderr


def test_public_jax_runtime_api_import_is_lightweight():
    result = _run_python_import_probe(
        """
        import sys
        import simsopt_jax.solve

        unexpected = [
            name
            for name in ("optax", "optimistix", "lineax")
            if name in sys.modules
        ]
        if unexpected:
            raise SystemExit(f"unexpected runtime imports: {unexpected}")
        """
    )

    assert result.returncode == 0, result.stderr


def test_public_jax_runtime_api_keeps_dispatch_out_of_package_root():
    result = _run_python_import_probe(
        """
        import sys
        import simsopt_jax.solve as solve

        if "least_squares" in solve.__all__ or "minimize" in solve.__all__:
            raise SystemExit("dispatch APIs should be imported from dispatch")
        unexpected = [
            name
            for name in (
                "simsopt_jax.solve.dispatch",
                "optax",
                "optimistix",
                "lineax",
            )
            if name in sys.modules
        ]
        if unexpected:
            raise SystemExit(f"unexpected runtime imports: {unexpected}")
        """
    )

    assert result.returncode == 0, result.stderr


def test_public_policy_packages_are_lightweight_and_explicit():
    result = _run_python_import_probe(
        """
        import sys
        modules_before_policy_import = set(sys.modules)
        from simsopt_jax.geo.optimizers import TraceableNewtonLinearSolver

        assert TraceableNewtonLinearSolver is not None
        unexpected = [
            name
            for name in ("jax", "optax", "optimistix", "lineax")
            if name in sys.modules and name not in modules_before_policy_import
        ]
        if unexpected:
            raise SystemExit(f"unexpected policy-package imports: {unexpected}")

        from simsopt_jax.backend import PrecisionSelection, ResolvedPrecision

        assert PrecisionSelection is not None
        assert ResolvedPrecision is not None
        """
    )

    assert result.returncode == 0, result.stderr


def test_jax_gpu_extra_declares_public_runtime_optimizer_dependencies():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    jax_gpu_deps = "\n".join(pyproject["project"]["optional-dependencies"]["JAX_GPU"])

    for dependency in RUNTIME_OPTIMIZER_MODULES:
        assert dependency in jax_gpu_deps
    assert "equinox" in jax_gpu_deps


def test_pyright_is_pinned_and_reachable_for_the_green_jax_slice():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    workflow = (REPO_ROOT / ".github/workflows/jax_smoke.yml").read_text()
    pyright = pyproject["tool"]["pyright"]
    pyright_requirement = "pyright==1.1.411"

    assert pyright_requirement in pyproject["project"]["optional-dependencies"]["dev"]
    assert pyright["typeCheckingMode"] == "standard"
    assert pyright["pythonVersion"] == "3.11"
    assert pyright["extraPaths"] == ["src"]
    assert pyright["include"] == [
        "src/simsopt_jax/objectives",
        "src/simsopt_jax/solve/dispatch.py",
        "src/simsopt_jax/solve/minimize_runtime.py",
        "src/simsopt_jax/solve/optax",
        "src/simsopt_jax/solve/optimistix",
        "src/simsopt_jax/solve/scipy",
        "src/simsopt_jax/solve/shared",
        "src/simsopt_jax/solve/simsopt",
        "tests/jax/solve/test_algorithm_change_gates.py",
        "tests/jax/solve/test_compat_shim_translation.py",
        "tests/jax/solve/test_deprecation_warnings.py",
        "tests/jax/solve/test_import_boundaries.py",
        "tests/jax/solve/test_optimizer_result_schema.py",
        "tests/jax/solve/test_options_typing.py",
    ]
    assert pyright_requirement in workflow
    assert "pyright --warnings" in workflow


def test_jax_gpu_extra_pins_cuda12_compiler_and_linker_components():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    jax_gpu_deps = pyproject["project"]["optional-dependencies"]["JAX_GPU"]

    assert "jax[cuda12]==0.10.0" in jax_gpu_deps
    assert "nvidia-cuda-nvcc-cu12==12.9.86" in jax_gpu_deps
    assert "nvidia-cuda-nvrtc-cu12==12.9.86" in jax_gpu_deps
    assert "nvidia-cuda-runtime-cu12==12.9.79" in jax_gpu_deps
    assert "nvidia-nvjitlink-cu12==12.9.86" in jax_gpu_deps


def test_deploy_extras_do_not_route_through_stale_optimistix_alias():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert optional_deps["deploy"] == [
        "simsopt[JAX,test,ALGS]",
        "shapely>=2.1,<3",
        "numba>=0.64,<0.66",
    ]
    assert optional_deps["deploy_gpu"] == [
        "simsopt[JAX_GPU,test,ALGS]",
        "shapely>=2.1,<3",
        "numba>=0.64,<0.66",
    ]


def test_production_and_examples_never_import_benchmark_modules():
    violations = {
        str(path.relative_to(REPO_ROOT)): imports
        for root in (REPO_ROOT / "src", REPO_ROOT / "examples")
        for path in sorted(root.rglob("*.py"))
        if (imports := _benchmark_imports(path))
    }

    assert violations == {}, (
        "Production and example modules must import runtime contracts from "
        f"their canonical owners, not benchmark orchestration: {violations!r}"
    )
