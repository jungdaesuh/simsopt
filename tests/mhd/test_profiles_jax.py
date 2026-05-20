import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import splev, splrep

from simsopt.jax_core.profiles import (
    profile_polynomial_value,
    profile_pressure_value,
    profile_spline_dfds,
    profile_spline_value,
)
from simsopt.mhd.profiles import (
    ProfilePolynomial,
    ProfilePressure,
    ProfileScaled,
    ProfileSpline,
)
import simsopt.mhd.profiles_jax as profiles_jax
from simsopt.mhd.profiles_jax import (
    ProfilePolynomialJAX,
    ProfilePressureJAX,
    ProfileScaledJAX,
    ProfileSplineJAX,
)

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


def test_profiles_jax_submodule_imports_without_simsoptpp():
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
from simsopt.mhd.profiles_jax import ProfilePolynomialJAX
from simsopt.jax_core.profiles import profile_polynomial_value
assert ProfilePolynomialJAX.__name__ == "ProfilePolynomialJAX"
assert profile_polynomial_value.__name__ == "profile_polynomial_value"
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


def test_profile_polynomial_jax_matches_closed_form():
    """Oracle: closed-form ascending polynomial formula.

    Lane: direct_kernel, rtol=1e-12, atol=1e-12.
    """
    coeffs = np.array([3.0, -1.5, 0.25, -0.75])
    s = np.linspace(0.0, 1.0, 17)
    expected = coeffs[0] + coeffs[1] * s + coeffs[2] * s**2 + coeffs[3] * s**3
    expected_dfds = coeffs[1] + 2.0 * coeffs[2] * s + 3.0 * coeffs[3] * s**2

    profile = ProfilePolynomialJAX(coeffs)

    np.testing.assert_allclose(
        _as_numpy(profile.f(s)), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(_as_numpy(profile(s)), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        _as_numpy(profile.dfds(s)), expected_dfds, rtol=1e-12, atol=1e-12
    )


def test_profile_polynomial_jax_grad_vmap_and_jit_match_analytic_oracle():
    """Oracle: analytic coefficient derivative of sum_i w_i p(s_i).

    Lane: derivative_heavy, first_derivative_rtol=1e-8, first_derivative_atol=1e-10.
    """
    coeffs = jnp.array([2.0, -0.5, 0.125, 0.75], dtype=jnp.float64)
    s = jnp.linspace(0.0, 1.0, 11)
    weights = jnp.linspace(0.25, 1.25, 11)

    def weighted_sum(coeff_vector):
        return jnp.sum(profile_polynomial_value(coeff_vector, s) * weights)

    gradient = jax.grad(weighted_sum)(coeffs)
    powers = np.vstack([np.asarray(s) ** power for power in range(coeffs.shape[0])])
    expected = powers @ np.asarray(weights)
    np.testing.assert_allclose(_as_numpy(gradient), expected, rtol=1e-10, atol=1e-12)

    eps = 1e-6
    basis = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float64)
    fd = (weighted_sum(coeffs + eps * basis) - weighted_sum(coeffs - eps * basis)) / (
        2.0 * eps
    )
    np.testing.assert_allclose(
        _as_numpy(gradient[2]), _as_numpy(fd), rtol=1e-8, atol=1e-10
    )

    coeff_rows = jnp.stack((coeffs, coeffs + 0.5))
    vmapped = jax.vmap(lambda row: profile_polynomial_value(row, s))(coeff_rows)
    scalar_vmapped = jax.vmap(
        lambda s_scalar: profile_polynomial_value(coeffs, s_scalar)
    )(s)
    jitted = jax.jit(lambda row: profile_polynomial_value(row, s))(coeffs)
    np.testing.assert_allclose(
        _as_numpy(vmapped[0]), _as_numpy(jitted), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _as_numpy(scalar_vmapped), _as_numpy(jitted), rtol=1e-12, atol=1e-12
    )


def test_profile_polynomial_jax_explicit_dofs_jit_tracks_updates():
    """Oracle: closed-form formula after DOF update through explicit JIT args."""
    s = jnp.array([0.0, 1.0], dtype=jnp.float64)
    profile = ProfilePolynomialJAX([1.0, 0.5])
    compiled = jax.jit(lambda dofs, s_arg: profile.f_from_dofs(dofs, s_arg))
    np.testing.assert_allclose(
        _as_numpy(compiled(jnp.asarray(profile.local_full_x), s)),
        np.array([1.0, 1.5]),
        rtol=1e-12,
        atol=1e-12,
    )

    profile.local_unfix_all()
    profile.x = [10.0, 5.0]
    np.testing.assert_allclose(
        _as_numpy(compiled(jnp.asarray(profile.local_full_x), s)),
        np.array([10.0, 15.0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_profile_scaled_and_pressure_jax_match_closed_form():
    """Oracle: closed-form scaled polynomial and product-rule pressure formula.

    Lane: direct_kernel, rtol=1e-12, atol=1e-12.
    """
    s = np.linspace(0.0, 1.0, 13)
    ne = ProfilePolynomialJAX([1.0, 0.0, -0.25])
    te = ProfilePolynomialJAX([4.0, -1.0])
    nD = ProfileScaledJAX(ne, 0.55)
    td = ProfilePolynomialJAX([3.0, 0.25])
    pressure = ProfilePressureJAX(ne, te, nD, td)

    ne_expected = 1.0 - 0.25 * s**2
    te_expected = 4.0 - s
    nD_expected = 0.55 * ne_expected
    td_expected = 3.0 + 0.25 * s
    expected = ne_expected * te_expected + nD_expected * td_expected
    dne_expected = -0.5 * s
    dte_expected = -np.ones_like(s)
    dnD_expected = 0.55 * dne_expected
    dtd_expected = 0.25 * np.ones_like(s)
    expected_dfds = (
        dne_expected * te_expected
        + ne_expected * dte_expected
        + dnD_expected * td_expected
        + nD_expected * dtd_expected
    )
    np.testing.assert_allclose(_as_numpy(nD.f(s)), nD_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        _as_numpy(pressure.f(s)), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _as_numpy(pressure.dfds(s)), expected_dfds, rtol=1e-12, atol=1e-12
    )


def test_profile_scaled_pressure_and_spline_jax_match_cpu_wrappers_after_updates():
    """Oracle: upstream CPU profile wrappers, including parent updates and resample."""
    s = np.linspace(0.0, 1.0, 13)
    density_cpu = ProfilePolynomial([1.0, 0.15, -0.3])
    density_jax = ProfilePolynomialJAX([1.0, 0.15, -0.3])
    temperature_cpu = ProfilePolynomial([3.0, -0.4])
    temperature_jax = ProfilePolynomialJAX([3.0, -0.4])
    scaled_cpu = ProfileScaled(density_cpu, 0.55)
    scaled_jax = ProfileScaledJAX(density_jax, 0.55)
    ion_temperature_cpu = ProfilePolynomial([2.5, 0.2])
    ion_temperature_jax = ProfilePolynomialJAX([2.5, 0.2])
    pressure_cpu = ProfilePressure(
        density_cpu,
        temperature_cpu,
        scaled_cpu,
        ion_temperature_cpu,
    )
    pressure_jax = ProfilePressureJAX(
        density_jax,
        temperature_jax,
        scaled_jax,
        ion_temperature_jax,
    )

    np.testing.assert_allclose(
        _as_numpy(scaled_jax.f(s)), scaled_cpu.f(s), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _as_numpy(pressure_jax.dfds(s)), pressure_cpu.dfds(s), rtol=1e-12, atol=1e-12
    )

    density_cpu.local_unfix_all()
    density_jax.local_unfix_all()
    density_cpu.x = [1.2, -0.1, -0.2]
    density_jax.x = [1.2, -0.1, -0.2]
    scaled_cpu.local_full_x = [0.8]
    scaled_jax.local_full_x = [0.8]

    np.testing.assert_allclose(
        _as_numpy(scaled_jax.dfds(s)), scaled_cpu.dfds(s), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _as_numpy(pressure_jax.f(s)), pressure_cpu.f(s), rtol=1e-12, atol=1e-12
    )

    spline_s = np.linspace(0.0, 1.0, 7)
    spline_f = 1.0 + 0.3 * spline_s - 0.2 * spline_s**2
    spline_cpu = ProfileSpline(spline_s, spline_f, degree=3)
    spline_jax = ProfileSplineJAX(spline_s, spline_f, degree=3)
    new_s = np.linspace(0.0, 1.0, 9)
    resampled_cpu = spline_cpu.resample(new_s, degree=2)
    resampled_jax = spline_jax.resample(new_s, degree=2)
    s_eval = np.linspace(0.0, 1.0, 15)

    np.testing.assert_allclose(
        _as_numpy(resampled_jax.f(s_eval)),
        resampled_cpu.f(s_eval),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _as_numpy(resampled_jax.dfds(s_eval)),
        resampled_cpu.dfds(s_eval),
        rtol=1e-10,
        atol=1e-12,
    )


def test_profile_pressure_jax_grad_matches_analytic_and_fd_oracles():
    """Oracle: analytic pressure derivative and centered finite difference.

    Lane: derivative_heavy, first_derivative_rtol=1e-8, first_derivative_atol=1e-10.
    """
    s = jnp.linspace(0.0, 1.0, 9)
    right_values = 1.5 - 0.25 * s

    def objective(left_coeffs):
        left_values = profile_polynomial_value(left_coeffs, s)
        return jnp.sum(profile_pressure_value((left_values, right_values)))

    coeffs = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)
    gradient = jax.grad(objective)(coeffs)
    powers = np.vstack([np.asarray(s) ** power for power in range(coeffs.shape[0])])
    expected = powers @ np.asarray(right_values)
    np.testing.assert_allclose(_as_numpy(gradient), expected, rtol=1e-10, atol=1e-12)

    eps = 1e-6
    basis = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
    fd = (objective(coeffs + eps * basis) - objective(coeffs - eps * basis)) / (
        2.0 * eps
    )
    np.testing.assert_allclose(
        _as_numpy(gradient[1]), _as_numpy(fd), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_profile_spline_jax_matches_scipy_fitpack_tck(degree):
    """Oracle: SciPy FITPACK ``splrep``/``splev`` tck evaluation.

    Lane: direct_kernel, rtol=1e-10, atol=1e-12.
    """
    s_nodes = np.linspace(0.0, 1.0, 8)
    f_nodes = 1.2 + 0.7 * s_nodes - 0.4 * s_nodes**2 + 0.1 * s_nodes**3
    tck = splrep(s_nodes, f_nodes, k=degree, s=0)
    s_eval = np.linspace(-0.05, 1.05, 31)

    value = profile_spline_value(tck[0], tck[1], tck[2], s_eval)
    derivative = profile_spline_dfds(tck[0], tck[1], tck[2], s_eval)

    np.testing.assert_allclose(
        _as_numpy(value), splev(s_eval, tck), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        _as_numpy(derivative), splev(s_eval, tck, der=1), rtol=1e-10, atol=1e-12
    )


def test_profile_spline_jax_exported_kernels_are_direct_jit_safe():
    """Oracle: SciPy FITPACK ``splrep``/``splev`` under direct ``jax.jit``."""
    s_nodes = np.linspace(0.0, 1.0, 8)
    f_nodes = 1.2 + 0.7 * s_nodes - 0.4 * s_nodes**2 + 0.1 * s_nodes**3
    tck = splrep(s_nodes, f_nodes, k=3, s=0)
    s_eval = np.linspace(-0.05, 1.05, 31)

    value = jax.jit(profile_spline_value)(tck[0], tck[1], tck[2], s_eval)
    derivative = jax.jit(profile_spline_dfds)(tck[0], tck[1], tck[2], s_eval)

    np.testing.assert_allclose(
        _as_numpy(value), splev(s_eval, tck), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        _as_numpy(derivative), splev(s_eval, tck, der=1), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("degree", [0, 6])
def test_profile_spline_jax_exported_kernels_reject_unsupported_degree(degree):
    """Oracle: public spline-kernel degree contract for direct and jitted calls."""
    s_nodes = np.linspace(0.0, 1.0, 8)
    f_nodes = 1.2 + 0.7 * s_nodes - 0.4 * s_nodes**2 + 0.1 * s_nodes**3
    tck = splrep(s_nodes, f_nodes, k=3, s=0)
    s_eval = np.linspace(-0.05, 1.05, 31)

    with pytest.raises(ValueError, match="degree must be in \\[1, 5\\]"):
        profile_spline_value(tck[0], tck[1], degree, s_eval)
    with pytest.raises(ValueError, match="degree must be in \\[1, 5\\]"):
        profile_spline_dfds(tck[0], tck[1], degree, s_eval)
    with pytest.raises(
        jax.errors.JaxRuntimeError, match="degree must be in \\[1, 5\\]"
    ):
        jax.jit(profile_spline_value)(
            tck[0], tck[1], degree, s_eval
        ).block_until_ready()
    with pytest.raises(
        jax.errors.JaxRuntimeError, match="degree must be in \\[1, 5\\]"
    ):
        jax.jit(profile_spline_dfds)(tck[0], tck[1], degree, s_eval).block_until_ready()


def test_profile_spline_jax_wrapper_tracks_fitpack_after_dof_update():
    """Oracle: SciPy FITPACK ``splrep``/``splev`` tck after DOF update.

    Lane: direct_kernel, at-knot rtol=1e-12, off-knot rtol=1e-8.
    """
    s_nodes = np.linspace(0.0, 1.0, 6)
    f_nodes = 1.0 + 0.2 * s_nodes + 0.4 * s_nodes**2
    s_eval = np.linspace(0.0, 1.0, 25)

    profile = ProfileSplineJAX(s_nodes, f_nodes, degree=3)
    tck = splrep(s_nodes, f_nodes, k=3, s=0)
    np.testing.assert_allclose(
        _as_numpy(profile.f(s_nodes)), splev(s_nodes, tck), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _as_numpy(profile.f(s_eval)), splev(s_eval, tck), rtol=1e-8, atol=1e-12
    )

    f_updated = -2.0 + 0.4 * s_nodes + 0.7 * s_nodes**2
    profile.local_unfix_all()
    profile.x = f_updated
    updated_tck = splrep(s_nodes, f_updated, k=3, s=0)
    np.testing.assert_allclose(
        _as_numpy(profile.dfds(s_eval)),
        splev(s_eval, updated_tck, der=1),
        rtol=1e-8,
        atol=1e-12,
    )


def test_profile_spline_jax_fits_once_per_dof_state(monkeypatch):
    """Oracle: explicit call-count contract for the host FITPACK boundary."""
    call_count = 0
    original_splrep = profiles_jax.splrep

    def counted_splrep(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_splrep(*args, **kwargs)

    monkeypatch.setattr(profiles_jax, "splrep", counted_splrep)
    s_nodes = np.linspace(0.0, 1.0, 6)
    f_nodes = 1.0 + 0.2 * s_nodes + 0.4 * s_nodes**2
    s_eval = np.linspace(0.0, 1.0, 25)

    profile = ProfileSplineJAX(s_nodes, f_nodes, degree=3)
    assert call_count == 1
    profile.f(s_eval)
    profile.dfds(s_eval)
    profile.f(s_eval)
    assert call_count == 1

    profile.local_unfix_all()
    profile.x = -2.0 + 0.4 * s_nodes + 0.7 * s_nodes**2
    assert call_count == 2
    profile.f(s_eval)
    profile.dfds(s_eval)
    assert call_count == 2


def test_profile_spline_jax_explicit_state_jit_tracks_updates():
    """Oracle: SciPy FITPACK tck after DOF update through explicit JIT args."""
    s_nodes = np.linspace(0.0, 1.0, 6)
    f_nodes = 1.0 + 0.2 * s_nodes + 0.4 * s_nodes**2
    s_eval = jnp.linspace(0.0, 1.0, 7)
    profile = ProfileSplineJAX(s_nodes, f_nodes, degree=3)
    compiled = jax.jit(
        lambda knots, coeffs, s_arg: profile.f_from_state(knots, coeffs, s_arg)
    )
    knots, coeffs = profile.spline_state
    tck = splrep(s_nodes, f_nodes, k=3, s=0)
    np.testing.assert_allclose(
        _as_numpy(compiled(knots, coeffs, s_eval)),
        splev(np.asarray(s_eval), tck),
        rtol=1e-10,
        atol=1e-12,
    )

    f_updated = -2.0 + 0.4 * s_nodes + 0.7 * s_nodes**2
    profile.local_unfix_all()
    profile.x = f_updated
    updated_knots, updated_coeffs = profile.spline_state
    updated_tck = splrep(s_nodes, f_updated, k=3, s=0)
    np.testing.assert_allclose(
        _as_numpy(compiled(updated_knots, updated_coeffs, s_eval)),
        splev(np.asarray(s_eval), updated_tck),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("degree", [0, 6])
def test_profile_spline_jax_rejects_unsupported_degree(degree):
    """Oracle: public ProfileSplineJAX constructor contract for degree bounds."""
    with pytest.raises(ValueError, match="degree must be in \\[1, 5\\]"):
        ProfileSplineJAX([0.0, 0.5, 1.0], [1.0, 1.5, 2.0], degree=degree)
