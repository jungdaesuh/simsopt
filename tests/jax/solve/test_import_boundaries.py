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


def test_jax_gpu_extra_declares_public_runtime_optimizer_dependencies():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    jax_gpu_deps = "\n".join(pyproject["project"]["optional-dependencies"]["JAX_GPU"])

    for dependency in RUNTIME_OPTIMIZER_MODULES:
        assert dependency in jax_gpu_deps
    assert "equinox" in jax_gpu_deps


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
