import os
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
RUNTIME_OPTIMIZER_MODULES = ("optax", "optimistix", "lineax")


def _run_python_import_probe(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_ROOT)
        if existing_pythonpath is None
        else os.pathsep.join((str(SRC_ROOT), existing_pythonpath))
    )
    probe_source = (
        textwrap.dedent(
            """
            import sys

            sys.meta_path = [
                finder
                for finder in sys.meta_path
                if finder.__class__.__module__ != "_simsopt_editable"
            ]
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
        import simsopt.solve._jax_driver

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

        if "simsopt.solve.jax" in sys.modules:
            raise SystemExit("simsopt.solve imported simsopt.solve.jax")
        """
    )

    assert result.returncode == 0, result.stderr


def test_public_jax_runtime_api_imports_static_optimizer_runtime_libraries():
    result = _run_python_import_probe(
        """
        import sys
        import simsopt.solve.jax

        missing = [
            name
            for name in ("optax", "optimistix", "lineax")
            if name not in sys.modules
        ]
        if missing:
            raise SystemExit(f"missing runtime imports: {missing}")
        """
    )

    assert result.returncode == 0, result.stderr


def test_jax_gpu_extra_declares_public_runtime_optimizer_dependencies():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    jax_gpu_deps = "\n".join(pyproject["project"]["optional-dependencies"]["JAX_GPU"])

    for dependency in RUNTIME_OPTIMIZER_MODULES:
        assert dependency in jax_gpu_deps
    assert "equinox" in jax_gpu_deps


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
