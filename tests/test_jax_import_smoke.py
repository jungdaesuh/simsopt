"""
Import smoke tests for the JAX code path.

These tests verify that JAX modules can be imported through the real
``simsopt`` package entrypoints (not via ``importlib.util`` bypass).
They run in the no-simsoptpp environment to catch import-chain regressions.

Each test launches a fresh Python subprocess so that ``sys.modules`` is
guaranteed clean — other test modules in this repo inject package stubs
at import time, which would contaminate in-process imports.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Resolve the src/ directory relative to the repo root so subprocesses
# can import simsopt without a pip install.
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")


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
