"""
Import smoke tests for the JAX code path.

These tests verify that JAX modules can be imported through the real
``simsopt`` package entrypoints (not via ``importlib.util`` bypass).
They run in the no-simsoptpp environment to catch import-chain regressions.

Each test launches a fresh Python subprocess so that ``sys.modules`` is
guaranteed clean — other test modules in this repo inject package stubs
at import time, which would contaminate in-process imports.
"""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Resolve the src/ directory relative to the repo root so subprocesses
# can import simsopt without a pip install.
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
_OPTIMIZER_JAX_PATH = Path(_SRC_DIR) / "simsopt" / "geo" / "optimizer_jax.py"


def _run_import_check(code):
    """Run *code* in a clean subprocess and return (returncode, stderr)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    return result.returncode, result.stderr.strip()


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


def test_import_package_root():
    """simsopt package imports without simsoptpp."""
    rc, err = _run_import_check("""
        import simsopt
        assert hasattr(simsopt, "__version__")
    """)
    assert rc == 0, f"import simsopt failed:\n{err}"


def test_import_biotsavart_jax():
    """BiotSavartJAX is importable through the real package entrypoint."""
    rc, err = _run_import_check("""
        from simsopt.field import BiotSavartJAX
        assert BiotSavartJAX is not None
    """)
    assert rc == 0, f"import BiotSavartJAX failed:\n{err}"


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

        import jax.numpy as jnp
        from simsopt.geo import optimizer_jax

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
        import jax.numpy as jnp
        from simsopt.geo import optimizer_jax

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


def test_optimizer_jax_public_module_has_no_jax_src_imports():
    """Section 6 public optimizer module must remain free of jax._src imports."""
    tree = ast.parse(_OPTIMIZER_JAX_PATH.read_text(encoding="utf-8"))
    forbidden_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("jax._src")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("jax._src"):
                forbidden_imports.append(module)

    assert forbidden_imports == [], (
        "optimizer_jax.py must not import jax._src in the public lane: "
        f"{forbidden_imports}"
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
    """M5 single-stage wrappers need SurfaceXYZTensorFourier (CPU class).

    BoozerResidualJAX, IotasJAX, NonQuasiSymmetricRatioJAX use CPU surface
    objects at the boundary (M0 adapter pattern). Without simsoptpp they
    are not importable via the package entrypoint. This is expected.
    """
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


def test_import_cpu_package_entrypoints_with_simsoptpp():
    """CPU package entrypoints must import cleanly when simsoptpp is available."""
    try:
        from simsoptpp import Curve as _  # noqa: F401
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
