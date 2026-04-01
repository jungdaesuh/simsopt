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
)


def _run_import_check(code):
    """Run *code* in a clean subprocess and return (returncode, stderr)."""
    env = os.environ.copy()
    for name in _BACKEND_SELECTOR_ENV_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_REPO_ROOT,
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
            CurrentValueSpec,
            CurveRZFourierSpec,
            CurveXYZFourierSpec,
            FieldEvalSpec,
            GroupedCoilSetSpec,
            FixedSurfaceFluxSpec,
            SurfaceRZFourierSpec,
        )

        assert CoilSpec is not None
        assert CoilGroupSpec is not None
        assert CoilSymmetrySpec is not None
        assert CurveCWSFourierRZSpec is not None
        assert CurrentValueSpec is not None
        assert CurveRZFourierSpec is not None
        assert CurveXYZFourierSpec is not None
        assert FieldEvalSpec is not None
        assert GroupedCoilSetSpec is not None
        assert FixedSurfaceFluxSpec is not None
        assert SurfaceRZFourierSpec is not None
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
            CurrentValueSpec,
            CurveRZFourierSpec,
            CurveXYZFourierSpec,
            FieldEvalSpec,
            FixedSurfaceFluxSpec,
            GroupedCoilSetSpec,
            SurfaceRZFourierSpec,
            curve_gamma_and_dash_from_dofs,
            curve_gamma_and_dash_from_spec,
            curve_geometry_from_dofs,
            curve_geometry_from_spec,
            fixed_surface_flux_integral_from_B,
            grouped_biot_savart_B_from_spec,
            grouped_coil_currents_from_spec,
            grouped_coil_index_lists_from_spec,
            grouped_coil_set_spec_from_coil_specs,
            grouped_field_data_from_spec,
            grouped_field_inputs_from_spec,
            make_coil_spec,
            make_coil_symmetry_spec,
            make_fixed_surface_flux_spec,
            make_current_value_spec,
            make_curve_cwsfourier_rz_spec,
            make_curve_rzfourier_spec,
            make_curve_xyzfourier_spec,
            make_field_eval_spec,
            make_grouped_coil_set_spec,
            make_surface_rzfourier_spec,
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
        assert isinstance(current_spec, CurrentValueSpec)
        assert isinstance(curve_rz_spec, CurveRZFourierSpec)
        assert isinstance(curve_xyz_spec, CurveXYZFourierSpec)
        assert isinstance(field_eval_spec, FieldEvalSpec)
        assert isinstance(coil_spec, GroupedCoilSetSpec)
        assert isinstance(flux_spec, FixedSurfaceFluxSpec)
        assert isinstance(surface_spec, SurfaceRZFourierSpec)

        curve_xyz_leaves, _ = jax.tree_util.tree_flatten(curve_xyz_spec)
        curve_rz_leaves, _ = jax.tree_util.tree_flatten(curve_rz_spec)
        curve_cws_leaves, _ = jax.tree_util.tree_flatten(curve_cws_spec)
        coil_symmetry_leaves, _ = jax.tree_util.tree_flatten(coil_symmetry_spec)
        current_leaves, _ = jax.tree_util.tree_flatten(current_spec)
        field_eval_leaves, _ = jax.tree_util.tree_flatten(field_eval_spec)
        coil_value_leaves, _ = jax.tree_util.tree_flatten(coil_value_spec)
        coil_leaves, _ = jax.tree_util.tree_flatten(coil_spec)
        flux_leaves, _ = jax.tree_util.tree_flatten(flux_spec)
        surface_leaves, _ = jax.tree_util.tree_flatten(surface_spec)

        assert len(curve_xyz_leaves) == 2
        assert len(curve_rz_leaves) == 2
        assert len(curve_cws_leaves) == 8
        assert len(coil_symmetry_leaves) == 1
        assert len(current_leaves) == 1
        assert len(field_eval_leaves) == 1
        assert len(coil_value_leaves) == 4
        assert len(coil_leaves) == 3
        assert len(flux_leaves) == 3
        assert len(surface_leaves) == 6
        assert len(grouped_field_inputs_from_spec(coil_spec)) == 1
        assert len(grouped_field_data_from_spec(coil_spec)) == 1
        assert grouped_coil_index_lists_from_spec(coil_spec) == ([0],)
        assert grouped_coil_currents_from_spec(coil_spec).shape == (1,)
        assert grouped_coil_set_spec_from_coil_specs((coil_value_spec,)).groups[0].coil_indices == (0,)
        assert_surface_dofs_derivable(curve_cws_spec, 3)  # stellsym: 2 rc + 1 zs
        assert_surface_dofs_derivable(curve_cws_nonstellsym_spec, 6)  # 2rc+1rs+2zc+1zs

        curve_xyz_gamma, curve_xyz_gammadash = jax.jit(curve_gamma_and_dash_from_spec)(curve_xyz_spec)
        curve_rz_gamma, _ = jax.jit(curve_gamma_and_dash_from_spec)(curve_rz_spec)
        curve_cws_gamma, curve_cws_gammadash = jax.jit(curve_gamma_and_dash_from_spec)(curve_cws_spec)
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
