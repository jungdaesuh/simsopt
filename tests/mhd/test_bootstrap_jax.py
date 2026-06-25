import os
import subprocess
import sys
import textwrap
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt_jax.core.mhd_bootstrap import compute_trapped_fraction_jax
from simsopt_jax.core.profiles import profile_polynomial_dfds, profile_polynomial_value
from simsopt_jax.core.redl_current import j_dot_B_Redl_jax_from_arrays
from simsopt.mhd.bootstrap import compute_trapped_fraction, j_dot_B_Redl
import simsopt_jax_adapters.mhd as mhd_jax
from simsopt_jax_adapters.mhd.bootstrap import RedlBootstrapJAX, j_dot_B_Redl_jax
from simsopt.mhd.profiles import ProfilePolynomial
from simsopt_jax_adapters.mhd.profiles import ProfilePolynomialJAX


_JAX_RUNTIME_ENV_VARS = (
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "XLA_FLAGS",
)


def _without_parent_jax_runtime_env():
    env = os.environ.copy()
    for name in _JAX_RUNTIME_ENV_VARS:
        env.pop(name, None)
    return env


def _as_numpy(value):
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _redl_inputs(helicity_n=1):
    # Same profiles and geometry as BootstrapTests.test_Redl_second_pass.
    ne = ProfilePolynomial(1.0e20 * np.array([1.0, 0.0, -0.8]))
    Te = ProfilePolynomial(25.0e3 * np.array([1.0, -0.9]))
    Ti = ProfilePolynomial(20.0e3 * np.array([1.0, -0.9]))
    Zeff = ProfilePolynomial(np.array([1.5, 0.5]))
    s = np.linspace(0.0, 1.0, 20)[1:]
    nfp = 4
    G = 32.0 - s
    iota = 0.95 - 0.7 * s
    R = 6.0 - 0.1 * s
    epsilon = 0.3 * np.sqrt(s)
    f_t = 1.46 * np.sqrt(epsilon)
    psi_edge = 68.0 / (2.0 * np.pi)
    return ne, Te, Ti, Zeff, helicity_n, s, G, R, iota, epsilon, f_t, psi_edge, nfp


def _redl_arrays(inputs):
    ne, Te, Ti, Zeff, helicity_n, s, G, R, iota, epsilon, f_t, psi_edge, nfp = inputs
    del helicity_n
    return (
        (
            ne(s),
            Te(s),
            Ti(s),
            Zeff(s),
            ne.dfds(s),
            Te.dfds(s),
            Ti.dfds(s),
        ),
        {
            "s": s,
            "G": G,
            "R": R,
            "iota": iota,
            "epsilon": epsilon,
            "f_t": f_t,
            "psi_edge": psi_edge,
            "nfp": nfp,
        },
    )


class _FixedRedlGeom(Optimizable):
    def __init__(self, **fields):
        self._data = SimpleNamespace(**fields)
        super().__init__()

    def __call__(self):
        return self._data


def _redl_jax_profiles():
    return (
        ProfilePolynomialJAX(1.0e20 * np.array([1.0, 0.0, -0.8])),
        ProfilePolynomialJAX(25.0e3 * np.array([1.0, -0.9])),
        ProfilePolynomialJAX(20.0e3 * np.array([1.0, -0.9])),
        ProfilePolynomialJAX(np.array([1.5, 0.5])),
    )


def _fixed_redl_geom(inputs):
    _, kwargs = _redl_arrays(inputs)
    return _FixedRedlGeom(
        surfaces=kwargs["s"],
        G=kwargs["G"],
        R=kwargs["R"],
        iota=kwargs["iota"],
        epsilon=kwargs["epsilon"],
        f_t=kwargs["f_t"],
        psi_edge=kwargs["psi_edge"],
        nfp=kwargs["nfp"],
    )


def _analytic_trapped_fraction_inputs(dimension):
    ns = 2
    ntheta = 15
    nphi = 7
    nfp = 3
    if dimension == 2:
        theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        modB = np.zeros((ntheta, ns))
        sqrtg = np.zeros((ntheta, ns))
        sqrtg[:, 0] = 10.0
        sqrtg[:, 1] = -25.0
        modB[:, 0] = 13.0 + 2.6 * np.cos(theta)
        modB[:, 1] = 9.0 + 3.7 * np.sin(theta)
    else:
        theta1d = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0.0, 2.0 * np.pi / nfp, nphi, endpoint=False)
        phi, theta = np.meshgrid(phi1d, theta1d)
        modB = np.zeros((ntheta, nphi, ns))
        sqrtg = np.zeros((ntheta, nphi, ns))
        sqrtg[:, :, 0] = 10.0
        sqrtg[:, :, 1] = -25.0
        modB[:, :, 0] = 13.0 + 2.6 * np.cos(theta)
        modB[:, :, 1] = 9.0 + 3.7 * np.sin(theta - nfp * phi)
    return modB, sqrtg


def test_not_a_knot_coefficients_uses_compact_banded_solve():
    source = Path("src/simsopt_jax/core/mhd_bootstrap.py").read_text()
    function_source = source[
        source.index("def _not_a_knot_coefficients") : source.index("def _eval_cubic")
    ]

    assert "tridiagonal_solve" in function_source
    assert "jnp.linalg.solve" not in function_source
    assert "jnp.zeros((size, size)" not in function_source
    assert "matrix_rows" not in function_source


def test_compute_trapped_fraction_jax_matches_cpu_2d_and_3d():
    """Oracle: upstream CPU trapped-fraction routine on analytic fixtures.

    Lane: direct_kernel for extrema/FSA; fixed quadrature vs CPU quad measured
    at first-order trapped-fraction tolerance.
    """
    for dimension in (2, 3):
        modB, sqrtg = _analytic_trapped_fraction_inputs(dimension)
        expected = compute_trapped_fraction(modB, sqrtg)
        actual = compute_trapped_fraction_jax(
            modB,
            sqrtg,
            quadrature_node_count=1024,
            extrema_iterations=20,
        )
        for index, (expected_part, actual_part) in enumerate(zip(expected, actual)):
            tolerance = (
                {"rtol": 5e-9, "atol": 5e-11}
                if index == 5
                else {"rtol": 1e-9, "atol": 1e-11}
            )
            np.testing.assert_allclose(
                _as_numpy(actual_part),
                expected_part,
                err_msg=f"dimension={dimension}, return_index={index}",
                **tolerance,
            )


def test_compute_trapped_fraction_jax_matches_cpu_off_grid_3d_extrema():
    """Oracle: CPU RectBivariateSpline/minimize path on non-analytic 3-D data."""
    rng = np.random.default_rng(2)
    modB = 9.0 + 0.35 * rng.normal(size=(15, 7, 2))
    sqrtg = 3.0 + 0.05 * rng.normal(size=modB.shape)
    expected = compute_trapped_fraction(modB, sqrtg)
    actual = compute_trapped_fraction_jax(
        modB,
        sqrtg,
        quadrature_node_count=1024,
        extrema_iterations=32,
    )

    for index, (expected_part, actual_part) in enumerate(zip(expected, actual)):
        tolerance = (
            {"rtol": 1e-8, "atol": 5e-11}
            if index == 5
            else {"rtol": 1e-9, "atol": 1e-10}
        )
        np.testing.assert_allclose(
            _as_numpy(actual_part),
            expected_part,
            err_msg=f"return_index={index}",
            **tolerance,
        )


def test_compute_trapped_fraction_jax_matches_cpu_off_grid_2d_extrema():
    """Oracle: CPU interp1d/minimize path on non-analytic 2-D data."""
    rng = np.random.default_rng(0)
    modB = 9.0 + 0.35 * rng.normal(size=(15, 2))
    sqrtg = 3.0 + 0.05 * rng.normal(size=modB.shape)
    expected = compute_trapped_fraction(modB, sqrtg)
    actual = compute_trapped_fraction_jax(
        modB,
        sqrtg,
        quadrature_node_count=1024,
        extrema_iterations=32,
    )

    for index, (expected_part, actual_part) in enumerate(zip(expected, actual)):
        tolerance = (
            {"rtol": 1e-8, "atol": 5e-11}
            if index == 5
            else {"rtol": 1e-9, "atol": 1e-10}
        )
        np.testing.assert_allclose(
            _as_numpy(actual_part),
            expected_part,
            err_msg=f"return_index={index}",
            **tolerance,
        )


def test_compute_trapped_fraction_jax_jit_transfer_guard_clean():
    """Oracle: JIT-compiled trapped-fraction inner loop runs without transfers."""
    modB, sqrtg = _analytic_trapped_fraction_inputs(2)
    compiled = jax.jit(
        compute_trapped_fraction_jax,
        static_argnames=("quadrature_node_count", "extrema_iterations"),
    )
    modB_jax = jnp.asarray(modB, dtype=jnp.float64)
    sqrtg_jax = jnp.asarray(sqrtg, dtype=jnp.float64)
    expected = compiled(
        modB_jax,
        sqrtg_jax,
        quadrature_node_count=128,
        extrema_iterations=12,
    )
    jax.block_until_ready(expected)

    with jax.transfer_guard("disallow"):
        actual = compiled(
            modB_jax,
            sqrtg_jax,
            quadrature_node_count=128,
            extrema_iterations=12,
        )
        jax.block_until_ready(actual)

    for expected_part, actual_part in zip(expected, actual):
        np.testing.assert_allclose(
            _as_numpy(actual_part), _as_numpy(expected_part), rtol=0.0, atol=0.0
        )


def test_compute_trapped_fraction_jax_modb_jvp_matches_centered_fd():
    """Oracle: centered finite difference of trapped-fraction ``f_t`` wrt ``modB``.

    Lane: derivative_heavy, first_derivative_rtol=5e-5, first_derivative_atol=1e-8.
    """
    modB_np, sqrtg_np = _analytic_trapped_fraction_inputs(2)
    modB = jnp.asarray(modB_np, dtype=jnp.float64)
    sqrtg = jnp.asarray(sqrtg_np, dtype=jnp.float64)
    direction = jnp.asarray(
        0.2 * np.sin(np.arange(modB_np.size, dtype=np.float64)).reshape(modB_np.shape),
        dtype=jnp.float64,
    )
    weights = jnp.array([0.4, 1.1], dtype=jnp.float64)

    def objective(modB_arg):
        *_, f_t = compute_trapped_fraction_jax(
            modB_arg,
            sqrtg,
            quadrature_node_count=512,
            extrema_iterations=20,
        )
        return jnp.vdot(weights, f_t)

    _value, tangent = jax.jvp(objective, (modB,), (direction,))
    step = 1.0e-4
    finite_difference = (
        objective(modB + step * direction) - objective(modB - step * direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        _as_numpy(tangent),
        _as_numpy(finite_difference),
        rtol=5.0e-5,
        atol=1.0e-8,
    )


def test_compute_trapped_fraction_jax_flat_modb_jvp_is_finite():
    """AD regression: constant-|B| extrema must not leak dead-branch NaNs."""
    modB = jnp.ones((4, 2), dtype=jnp.float64)
    sqrtg = jnp.ones_like(modB)
    direction = jnp.zeros_like(modB).at[0, 0].set(0.1)

    def objective(modB_arg):
        *_, f_t = compute_trapped_fraction_jax(
            modB_arg,
            sqrtg,
            quadrature_node_count=8,
            extrema_iterations=2,
        )
        return jnp.sum(f_t)

    value, tangent = jax.jvp(objective, (modB,), (direction,))

    assert jnp.isfinite(value)
    assert jnp.isfinite(tangent)


def test_bootstrap_jax_submodule_imports_without_simsoptpp():
    """Oracle: import blocker proving the public JAX submodule avoids simsoptpp."""
    script = """
from pathlib import Path
import importlib.abc
import sys

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path.cwd() / "src")

class BlockSimsoptpp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "simsoptpp" or fullname.startswith("simsoptpp."):
            raise ImportError("blocked simsoptpp")
        return None

sys.meta_path.insert(0, BlockSimsoptpp())
from simsopt_jax.mhd import (
    RedlDetailsJAX,
    compute_trapped_fraction_jax,
    j_dot_B_Redl_jax_from_arrays,
)
import simsopt_jax.mhd as mhd_jax
assert j_dot_B_Redl_jax_from_arrays.__name__ == "j_dot_B_Redl_jax_from_arrays"
assert compute_trapped_fraction_jax.__name__ == "compute_trapped_fraction_jax"
assert RedlDetailsJAX.__name__ == "RedlDetailsJAX"
assert "RedlBootstrapJAX" not in mhd_jax.__all__
assert "j_dot_B_Redl_jax" not in mhd_jax.__all__
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=_without_parent_jax_runtime_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stderr == ""


def test_mhd_all_without_jax_keeps_cpu_bootstrap_exports_only():
    """Oracle: optional JAX exports stay out of MHD ``__all__`` without JAX."""
    script = """
from pathlib import Path
import importlib.abc
import sys

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path.cwd() / "src")

class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise ModuleNotFoundError("blocked jax for mhd star import smoke")
        return None

sys.meta_path.insert(0, BlockJax())
import simsopt.mhd as mhd
from simsopt.mhd import j_dot_B_Redl

assert "j_dot_B_Redl" in mhd.__all__
assert "j_dot_B_Redl_jax" not in mhd.__all__
assert "RedlBootstrapJAX" not in mhd.__all__
assert j_dot_B_Redl.__name__ == "j_dot_B_Redl"
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=_without_parent_jax_runtime_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stderr == ""


def test_mhd_import_propagates_broken_jax_probe():
    """Oracle: broken JAX imports fail loudly instead of hiding JAX exports."""
    script = """
from pathlib import Path
import importlib.abc
import importlib.machinery
import sys

class BrokenJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise AttributeError("broken jax import for mhd probe")
        return None

sys.path.insert(0, str(Path.cwd() / "src"))
sys.meta_path.insert(0, BrokenJax())
try:
    import simsopt_jax.mhd  # noqa: F401
except AttributeError as exc:
    assert "broken jax import for mhd probe" in str(exc)
else:
    raise AssertionError("broken JAX probe was silently treated as unavailable")
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=_without_parent_jax_runtime_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stderr == ""


def test_mhd_import_propagates_broken_installed_jax_importerror():
    """Oracle: canonical JAX MHD import propagates broken installed JAX."""
    script = """
from pathlib import Path
import importlib.abc
import importlib.machinery
import sys

class BrokenInstalledJax(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax":
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError("broken installed jax import for mhd probe")

sys.path.insert(0, str(Path.cwd() / "src"))
sys.meta_path.insert(0, BrokenInstalledJax())
try:
    import simsopt_jax.mhd  # noqa: F401
except ImportError as exc:
    assert "broken installed jax import for mhd probe" in str(exc)
else:
    raise AssertionError("broken installed JAX did not fail canonical JAX MHD import")
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=_without_parent_jax_runtime_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stderr == ""


def test_j_dot_B_Redl_jax_matches_cpu_details_for_helicity_cases():
    """Oracle: direct CPU Redl implementation for the pinned second-pass fixture.

    Lane: direct_kernel, rtol=1e-12, atol=1e-12.
    """
    for helicity_n in (0, 1, -1):
        inputs = _redl_inputs(helicity_n=helicity_n)
        ne, Te, Ti, Zeff, _, s, G, R, iota, epsilon, f_t, psi_edge, nfp = inputs
        expected_jdotB, expected_details = j_dot_B_Redl(
            ne,
            Te,
            Ti,
            Zeff,
            helicity_n,
            s,
            G,
            R,
            iota,
            epsilon,
            f_t,
            psi_edge,
            nfp,
        )

        args, kwargs = _redl_arrays(inputs)
        actual_jdotB, actual_details = j_dot_B_Redl_jax_from_arrays(
            *args, helicity_n=helicity_n, **kwargs
        )

        np.testing.assert_allclose(
            _as_numpy(actual_jdotB), expected_jdotB, rtol=1e-12, atol=1e-12
        )
        for detail_field in fields(actual_details):
            expected = getattr(expected_details, detail_field.name)
            actual = getattr(actual_details, detail_field.name)
            np.testing.assert_allclose(
                _as_numpy(actual),
                expected,
                rtol=1e-12,
                atol=1e-12,
                err_msg=detail_field.name,
            )


def test_public_j_dot_B_Redl_jax_wrapper_matches_cpu():
    """Oracle: CPU wrapper with the same public profile inputs."""
    inputs = _redl_inputs(helicity_n=1)
    ne, Te, Ti, Zeff, helicity_n, s, G, R, iota, epsilon, f_t, psi_edge, nfp = inputs
    expected, _ = j_dot_B_Redl(
        ne, Te, Ti, Zeff, helicity_n, s, G, R, iota, epsilon, f_t, psi_edge, nfp
    )

    actual, _ = j_dot_B_Redl_jax(
        ne, Te, Ti, Zeff, helicity_n, s, G, R, iota, epsilon, f_t, psi_edge, nfp
    )

    np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-12, atol=1e-12)


def test_redl_bootstrap_jax_adapter_matches_public_wrapper_and_adapter_export():
    """Oracle: CPU Redl implementation plus public MHD adapter call path."""
    inputs = _redl_inputs(helicity_n=1)
    ne, Te, Ti, Zeff, _, s, G, R, iota, epsilon, f_t, psi_edge, nfp = inputs
    geom = _fixed_redl_geom(inputs)
    adapter = RedlBootstrapJAX(geom, ne, Te, Ti, Zeff, helicity_n=1)

    expected, expected_details = j_dot_B_Redl(
        ne, Te, Ti, Zeff, 1, s, G, R, iota, epsilon, f_t, psi_edge, nfp
    )
    actual = adapter.J()
    actual_jdotB, actual_details = adapter.details()

    exported_adapter = mhd_jax.RedlBootstrapJAX(
        geom, ne, Te, Ti, Zeff, helicity_n=1
    )
    np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        _as_numpy(actual_jdotB), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _as_numpy(exported_adapter.J()), expected, rtol=1e-12, atol=1e-12
    )
    for detail_field in fields(actual_details):
        np.testing.assert_allclose(
            _as_numpy(getattr(actual_details, detail_field.name)),
            getattr(expected_details, detail_field.name),
            rtol=1e-12,
            atol=1e-12,
            err_msg=detail_field.name,
        )


def test_redl_bootstrap_jax_adapter_converts_cpu_profiles_for_accessors():
    """Oracle: CPU Redl wrapper for converted numeric-Zeff adapter state."""
    inputs = _redl_inputs(helicity_n=1)
    ne, Te, Ti, _, _, s, G, R, iota, epsilon, f_t, psi_edge, nfp = inputs
    geom = _fixed_redl_geom(inputs)
    adapter = RedlBootstrapJAX(geom, ne, Te, Ti, Zeff=1.7, helicity_n=1)

    expected, _ = j_dot_B_Redl(
        adapter.ne,
        adapter.Te,
        adapter.Ti,
        adapter.Zeff,
        1,
        s,
        G,
        R,
        iota,
        epsilon,
        f_t,
        psi_edge,
        nfp,
    )
    actual = adapter.J()
    ne_jacobian = adapter.dJ_by_dne_dofs()
    zeff_jacobian = adapter.dJ_by_dZeff_dofs()

    assert adapter.ne is ne
    assert isinstance(adapter.Zeff, ProfilePolynomial)
    np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-12, atol=1e-12)
    assert ne_jacobian.shape == (len(s), len(adapter.ne.local_full_x))
    assert zeff_jacobian.shape == (len(s), len(adapter.Zeff.local_full_x))


def test_redl_bootstrap_jax_adapter_reflects_cpu_profile_dof_updates():
    """Oracle: accepted CPU profile dependencies are not snapshotted."""
    inputs = _redl_inputs(helicity_n=1)
    ne, Te, Ti, Zeff, _, s, G, R, iota, epsilon, f_t, psi_edge, nfp = inputs
    geom = _fixed_redl_geom(inputs)
    adapter = RedlBootstrapJAX(geom, ne, Te, Ti, Zeff, helicity_n=1)
    ne.local_full_x = ne.local_full_x + np.array([0.0, 0.0, 0.05e20])

    expected, _ = j_dot_B_Redl_jax(
        ne, Te, Ti, Zeff, 1, s, G, R, iota, epsilon, f_t, psi_edge, nfp
    )
    actual = adapter.J()

    assert adapter.ne is ne
    np.testing.assert_allclose(
        _as_numpy(actual), _as_numpy(expected), rtol=0.0, atol=0.0
    )


def test_j_dot_B_Redl_jax_jit_and_transfer_guard_clean():
    """Oracle: CPU Redl output; hot array kernel is reusable under JIT."""
    inputs = _redl_inputs(helicity_n=-1)
    expected, _ = j_dot_B_Redl(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        inputs[9],
        inputs[10],
        inputs[11],
        inputs[12],
    )
    args, kwargs = _redl_arrays(inputs)
    kwarg_names = tuple(kwargs)
    with jax.transfer_guard_host_to_device("allow"):
        array_inputs = tuple(jnp.asarray(value) for value in args)
        array_kwargs = tuple(jnp.asarray(kwargs[name]) for name in kwarg_names)

    compiled = jax.jit(
        lambda *flat_args: j_dot_B_Redl_jax_from_arrays(
            *flat_args[: len(array_inputs)],
            helicity_n=-1,
            **{
                name: value
                for name, value in zip(kwarg_names, flat_args[len(array_inputs) :])
            },
        )[0]
    )

    with jax.transfer_guard("disallow"):
        actual = compiled(*(array_inputs + array_kwargs))
        actual.block_until_ready()

    np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-12, atol=1e-12)


def test_redl_bootstrap_jax_profile_dof_jacobians_match_finite_difference():
    """Oracle: centered FD of each public profile-coefficient Jacobian accessor."""
    inputs = _redl_inputs(helicity_n=1)
    geom = _fixed_redl_geom(inputs)
    ne, Te, Ti, Zeff = _redl_jax_profiles()
    adapter = RedlBootstrapJAX(geom, ne, Te, Ti, Zeff, helicity_n=1)
    profiles = {"ne": ne, "Te": Te, "Ti": Ti, "Zeff": Zeff}
    geom_data = geom()

    def weighted_current(profile_name, profile_coeffs):
        s = jnp.asarray(geom_data.surfaces, dtype=jnp.float64)
        coeffs = {
            name: jnp.asarray(profile.local_full_x, dtype=jnp.float64)
            for name, profile in profiles.items()
        }
        coeffs[profile_name] = profile_coeffs
        ne_s = ne.f_from_dofs(coeffs["ne"], s)
        d_ne_d_s = ne.dfds_from_dofs(coeffs["ne"], s)
        Te_s = Te.f_from_dofs(coeffs["Te"], s)
        d_Te_d_s = Te.dfds_from_dofs(coeffs["Te"], s)
        Ti_s = Ti.f_from_dofs(coeffs["Ti"], s)
        d_Ti_d_s = Ti.dfds_from_dofs(coeffs["Ti"], s)
        Zeff_s = Zeff.f_from_dofs(coeffs["Zeff"], s)
        jdotB, _ = j_dot_B_Redl_jax_from_arrays(
            ne_s,
            Te_s,
            Ti_s,
            Zeff_s,
            d_ne_d_s,
            d_Te_d_s,
            d_Ti_d_s,
            helicity_n=1,
            s=geom_data.surfaces,
            G=geom_data.G,
            R=geom_data.R,
            iota=geom_data.iota,
            epsilon=geom_data.epsilon,
            f_t=geom_data.f_t,
            psi_edge=geom_data.psi_edge,
            nfp=geom_data.nfp,
        )
        return jnp.sum(weights * jdotB)

    cases = (
        ("ne", adapter.dJ_by_dne_dofs(), [0.03e20, -0.02e20, 0.01e20], 1.0e-5),
        ("Te", adapter.dJ_by_dTe_dofs(), [1.0e3, -0.8e3], 1.0e-4),
        ("Ti", adapter.dJ_by_dTi_dofs(), [0.9e3, -0.7e3], 1.0e-4),
        ("Zeff", adapter.dJ_by_dZeff_dofs(), [0.2, -0.1], 1.0e-4),
    )
    for profile_name, jacobian, direction_values, step in cases:
        weights = jnp.linspace(0.2, 1.2, jacobian.shape[0], dtype=jnp.float64)
        direction = jnp.asarray(direction_values, dtype=jnp.float64)
        coeffs = jnp.asarray(profiles[profile_name].local_full_x, dtype=jnp.float64)
        directional_derivative = jnp.vdot(weights, jacobian @ direction)
        finite_difference = (
            weighted_current(profile_name, coeffs + step * direction)
            - weighted_current(profile_name, coeffs - step * direction)
        ) / (2.0 * step)

        np.testing.assert_allclose(
            _as_numpy(directional_derivative),
            _as_numpy(finite_difference),
            rtol=2.0e-6,
            atol=1.0e-6,
            err_msg=profile_name,
        )


def test_j_dot_B_Redl_jax_grad_matches_finite_difference_profile_coefficients():
    """Oracle: centered finite difference of normalized density coefficients.

    Lane: derivative_heavy, first_derivative_rtol=1e-8, first_derivative_atol=1e-10.
    """
    s = jnp.linspace(0.08, 0.92, 8, dtype=jnp.float64)
    Te_coeffs = jnp.array([1.0, -0.72, 0.03], dtype=jnp.float64)
    Ti_coeffs = jnp.array([1.0, -0.66, 0.04], dtype=jnp.float64)
    Zeff_coeffs = jnp.array([1.45, 0.2], dtype=jnp.float64)
    G = 32.0 - 0.4 * s
    R = 5.8 - 0.15 * s
    iota = 0.82 - 0.18 * s
    epsilon = 0.22 + 0.08 * jnp.sqrt(s)
    f_t = 0.35 + 0.2 * s
    weights = jnp.linspace(0.3, 1.1, s.shape[0], dtype=jnp.float64)
    psi_edge = 68.0 / (2.0 * jnp.pi)

    def weighted_current(ne_coeffs):
        ne_s = 1.0e20 * profile_polynomial_value(ne_coeffs, s)
        d_ne_d_s = 1.0e20 * profile_polynomial_dfds(ne_coeffs, s)
        Te_s = 25.0e3 * profile_polynomial_value(Te_coeffs, s)
        d_Te_d_s = 25.0e3 * profile_polynomial_dfds(Te_coeffs, s)
        Ti_s = 20.0e3 * profile_polynomial_value(Ti_coeffs, s)
        d_Ti_d_s = 20.0e3 * profile_polynomial_dfds(Ti_coeffs, s)
        Zeff_s = profile_polynomial_value(Zeff_coeffs, s)
        jdotB, _ = j_dot_B_Redl_jax_from_arrays(
            ne_s,
            Te_s,
            Ti_s,
            Zeff_s,
            d_ne_d_s,
            d_Te_d_s,
            d_Ti_d_s,
            helicity_n=1,
            s=s,
            G=G,
            R=R,
            iota=iota,
            epsilon=epsilon,
            f_t=f_t,
            psi_edge=psi_edge,
            nfp=4,
        )
        return jnp.sum(weights * jdotB)

    ne_coeffs = jnp.array([1.0, 0.05, -0.8], dtype=jnp.float64)
    direction = jnp.array([0.03, -0.02, 0.01], dtype=jnp.float64)
    gradient = jax.grad(weighted_current)(ne_coeffs)
    directional_derivative = jnp.vdot(gradient, direction)

    eps = 1e-5
    fd = (
        weighted_current(ne_coeffs + eps * direction)
        - weighted_current(ne_coeffs - eps * direction)
    ) / (2.0 * eps)

    np.testing.assert_allclose(
        _as_numpy(directional_derivative),
        _as_numpy(fd),
        rtol=1e-8,
        atol=1e-10,
    )
