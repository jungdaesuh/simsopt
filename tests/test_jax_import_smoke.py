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
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_OPTIMIZER_JAX_PATH = Path(_SRC_DIR) / "simsopt" / "geo" / "optimizer_jax.py"
_OPTIMIZER_PRIVATE_DIR = Path(_SRC_DIR) / "simsopt" / "geo" / "optimizer_jax_private"
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


def _run_import_check(code, *, timeout=30, extra_env=None):
    """Run *code* in a clean subprocess and return (returncode, stderr)."""
    env = os.environ.copy()
    for name in _BACKEND_SELECTOR_ENV_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env is not None:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_REPO_ROOT,
        env=env,
    )
    return result.returncode, result.stderr.strip()


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


def _strip_simsopt_editable_finders(*, include_import=True):
    import_block = "import sys\n\n" if include_import else ""
    return f"""
        {import_block}sys.meta_path = [
            finder
            for finder in sys.meta_path
            if type(finder).__module__ != "_simsopt_editable"
            and not type(finder).__module__.startswith("__editable__")
        ]
    """


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
        assert jax.numpy.all(jax.numpy.isfinite(B))
    """,
        failure_message="grouped Biot-Savart GPU spec transfer-guard smoke failed",
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
        assert jax.numpy.all(jax.numpy.isfinite(B))
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
    ],
)
def test_transfer_guard_disallow_allows_legacy_curve_objective_values(label, code):
    """Legacy curve objectives must use explicit host/device boundaries under disallow."""
    objective_code = textwrap.indent(textwrap.dedent(code).strip(), " " * 8)
    _assert_import_check_passes(
        f"""
        import numpy as np
        import simsopt.config as simsopt_config
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.geo.curveobjectives import (
            CurveCurveDistance,
            CurveLength,
            CurveSurfaceDistance,
            LpCurveCurvature,
        )

        simsopt_config.set_backend("jax_cpu_parity", transfer_guard="disallow")
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

{objective_code}
    """,
        failure_message=f"{label} transfer-guard value smoke failed",
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


def test_native_cpu_backend_selection_does_not_require_jax_runtime():
    """native_cpu config must not force a JAX import when only CPU mode is selected."""
    rc, err = _run_import_check("""
        import importlib.abc
        import sys

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
                jnp.zeros((1, 2, 3)),
                jnp.ones((1, 2, 3)),
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
        assert jnp.isfinite(value)
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


def test_optimizer_jax_private_package_has_no_jax_src_imports():
    """Private optimizer modules must also stay on public JAX APIs."""
    forbidden_imports = {}
    for path in sorted(_OPTIMIZER_PRIVATE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("jax._src")
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("jax._src"):
                    imports.append(module)
        if imports:
            forbidden_imports[str(path.relative_to(_OPTIMIZER_PRIVATE_DIR.parent))] = (
                imports
            )

    assert forbidden_imports == {}, (
        f"optimizer_jax_private must not import jax._src: {forbidden_imports}"
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


def test_import_cpu_geo_core_entrypoints_without_jax():
    """Core CPU geo entrypoints should import when simsoptpp is present but JAX is absent."""
    try:
        from simsoptpp import Curve as _  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("compiled simsoptpp symbols are not available in this environment")

    rc, err = _run_import_check(f"""
        import importlib.abc

        {_strip_simsopt_editable_finders()}

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
