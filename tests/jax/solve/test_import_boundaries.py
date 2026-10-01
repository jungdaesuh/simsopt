import os
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
# lineax backs the CG adjoint selector (it declares its own equinox
# dependency); optax is no longer a dependency.
RUNTIME_OPTIMIZER_MODULES = ("lineax",)


def _run_python_import_probe(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    local_pythonpath = os.pathsep.join((str(REPO_ROOT), str(SRC_ROOT)))
    env["PYTHONPATH"] = (
        local_pythonpath
        if existing_pythonpath is None
        else os.pathsep.join((local_pythonpath, existing_pythonpath))
    )
    probe_source = textwrap.dedent(source)
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
    assert "optax" not in jax_gpu_deps
