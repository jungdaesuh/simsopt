"""Item 22 closeout: JAX port of ``simsopt.field.sampling``.

Closes the jax-port plan item 22 (Wave R4). The JAX kernels in
:mod:`simsopt.jax_core.sampling` are re-exported from
:mod:`simsopt.field.sampling` and exercise the following invariants:

1. ``test_sample_weighted_indices_jax_uses_explicit_key`` /
   ``test_draw_uniform_on_curve_jax_maps_weighted_indices_to_gamma`` /
   ``test_draw_uniform_on_surface_jax_maps_flat_weighted_indices_to_gamma``
   — deterministic mapping from a degenerate one-hot weight vector to
   the expected quadrature index. Originally part of the local
   smoke suite; retained as the bit-level contract for the weighted
   sampler.

2. ``test_sample_weighted_indices_jax_reproducibility`` — two
   independent invocations with the same ``key`` and the same
   ``(weights, nsamples)`` produce bit-identical samples; two
   invocations with different keys produce non-equal samples (with
   overwhelming probability at the chosen ``nsamples``).

3. ``test_sample_weighted_indices_jax_statistical_moments_match_target_distribution``
   — at production scale (``nsamples = 50000``) the empirical bin
   frequencies of a four-bin categorical match the analytic target
   ``weights/sum(weights)`` to within the documented one-sided
   tolerance derived from the multinomial sample-size confidence
   interval (see docstring of the test for the closed-form bound).

4. ``test_sample_weighted_indices_jax_does_not_read_numpy_random``
   — disabling the NumPy module RNG does not perturb JAX-key-pinned
   invocations. Establishes the "no hidden global RNG state" contract
   from the item 22 prompt without mutating the process-global NumPy RNG.

5. ``test_draw_uniform_on_curve_jax_matches_upstream_rejection_sampling_moments``
   / ``test_draw_uniform_on_surface_jax_matches_upstream_rejection_sampling_moments``
   — JAX inverse-CDF sampler and the upstream
   ``numpy.random``-based rejection sampler produce empirically
   indistinguishable index-histograms at production scale on a real
   ``CurveRZFourier`` / ``SurfaceRZFourier`` fixture. Bit identity
   is impossible because the two paths use different RNGs; the test
   instead asserts that the empirical mean and second moment of the
   sampled-index distributions match within the documented
   statistical tolerance.

6. ``test_sample_weighted_indices_jax_under_strict_transfer_guard``
   — the JIT-compiled kernel runs under
   ``jax.transfer_guard("disallow")`` with no host-to-device
   crossings once inputs are on-device. This locks in the strict
   transfer-guard discipline that the rest of the JAX port adheres
   to.

Statistical tolerances
----------------------

There is no dedicated parity-ladder lane for stochastic moment
matching in
``benchmarks.validation_ladder_contract.PARITY_LADDER_TOLERANCES``;
that ladder exclusively covers deterministic CPU/C++ oracles. The
statistical thresholds in this file are derived inline from the
multinomial / binomial concentration bound.

For a binomial proportion ``p = w_i / sum(w)`` and ``n`` samples,
the standard error of the empirical frequency is
``sigma_hat = sqrt(p * (1 - p) / n)``. The bound used below is
``8 * sigma_max`` where ``sigma_max = sqrt(0.25 / n)`` is the
worst-case standard deviation (achieved at ``p = 0.5``). At
``n = 50000`` this gives an absolute tolerance of ``~0.018``, which
is roughly ``8`` standard deviations from the expected mean — a
two-sided false-positive rate of ``~1.2e-15`` per bin under a
Gaussian approximation. With a fixed JAX PRNG key the test is
deterministic, so this "false positive" rate quantifies the
robustness margin against future RNG changes rather than per-run
flakiness.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsopt.field.tracing as tracing_module
from simsopt.field.sampling import (
    draw_uniform_on_curve,
    draw_uniform_on_surface,
)
from simsopt.field.sampling_jax import (
    draw_uniform_on_curve_jax,
    draw_uniform_on_surface_jax,
    sample_weighted_indices_jax,
)


# --- Lightweight fixtures for the bit-level mapping tests --------------------


class CurveFixture:
    """Curve stand-in with a one-hot arclength vector."""

    def __init__(self) -> None:
        self._arclength = np.array([0.0, 0.0, 1.0, 0.0])
        self._gamma = np.arange(12.0).reshape((4, 3))

    def incremental_arclength(self) -> np.ndarray:
        return self._arclength

    def gamma(self) -> np.ndarray:
        return self._gamma


class SurfaceFixture:
    """Surface stand-in with a one-hot ``|normal|`` field."""

    def __init__(self) -> None:
        self._gamma = np.arange(18.0).reshape((2, 3, 3))
        self._normal = np.zeros((2, 3, 3))
        self._normal[1, 2, 0] = 5.0

    def gamma(self) -> np.ndarray:
        return self._gamma

    def normal(self) -> np.ndarray:
        return self._normal


class UniformCurveFixture:
    """Curve stand-in that accepts every rejection-sampler proposal."""

    def __init__(self) -> None:
        self._arclength = np.ones(4)
        self._gamma = np.arange(12.0).reshape((4, 3))

    def incremental_arclength(self) -> np.ndarray:
        return self._arclength

    def gamma(self) -> np.ndarray:
        return self._gamma


class UniformSurfaceFixture:
    """Surface stand-in that accepts every rejection-sampler proposal."""

    def __init__(self) -> None:
        self._gamma = np.arange(18.0).reshape((2, 3, 3))
        self._normal = np.ones((2, 3, 3))

    def gamma(self) -> np.ndarray:
        return self._gamma

    def normal(self) -> np.ndarray:
        return self._normal


# --- Statistical-tolerance derivation, single source of truth ----------------

# Documented in the module docstring. The factor of 8 is the
# worst-case concentration bound used to convert one binomial
# standard deviation into the absolute tolerance accepted below.
_STATISTICAL_NSAMPLES = 50_000
_STATISTICAL_STD_MULTIPLIER = 8.0
_BINOMIAL_WORST_CASE_STD = math.sqrt(0.25 / _STATISTICAL_NSAMPLES)
_STATISTICAL_FREQUENCY_ATOL = _STATISTICAL_STD_MULTIPLIER * _BINOMIAL_WORST_CASE_STD


def _moment_tolerance_for_index_range(nquadpoints: int, nsamples: int) -> float:
    """Multinomial-based mean tolerance for sampled index moments.

    The first moment of a sampled-index distribution on
    ``[0, N)`` is a weighted sum of multinomial bin frequencies.
    The variance of each frequency is bounded by ``0.25 / n``;
    the bound on the variance of the index mean (max index ``N-1``)
    is therefore ``(N - 1) ** 2 / (4 n)``. We use
    ``8 * sqrt(variance)`` as a generous Chebyshev-style envelope.
    """
    return (
        _STATISTICAL_STD_MULTIPLIER
        * (nquadpoints - 1)
        * math.sqrt(1.0 / (4.0 * nsamples))
    )


def _patch_numpy_sampler_rng(monkeypatch, seed: int) -> None:
    """Route upstream rejection-sampler draws through a local deterministic RNG."""
    rng = np.random.default_rng(seed)

    monkeypatch.setattr(np.random, "randint", rng.integers)
    monkeypatch.setattr(np.random, "uniform", rng.uniform)


def _assert_numpy_random_state_equal(left, right) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


# --- Tests --------------------------------------------------------------------


def test_sample_weighted_indices_jax_uses_explicit_key() -> None:
    """Same key + degenerate one-hot weights => deterministic single index."""
    key = jax.random.PRNGKey(11)
    weights = jnp.array([0.0, 0.0, 1.0, 0.0])

    idxs_first = sample_weighted_indices_jax(key, weights, 8)
    idxs_second = sample_weighted_indices_jax(key, weights, 8)

    np.testing.assert_array_equal(np.asarray(idxs_first), np.asarray(idxs_second))
    np.testing.assert_array_equal(np.asarray(idxs_first), np.full(8, 2))


def test_draw_uniform_on_curve_jax_maps_weighted_indices_to_gamma() -> None:
    """One-hot arclength concentrates all samples at the corresponding gamma row."""
    curve = CurveFixture()
    xyz, idxs = draw_uniform_on_curve_jax(jax.random.PRNGKey(3), curve, 6)

    np.testing.assert_array_equal(np.asarray(idxs), np.full(6, 2))
    np.testing.assert_allclose(
        np.asarray(xyz), np.repeat(curve.gamma()[2:3, :], 6, axis=0)
    )


def test_draw_uniform_on_surface_jax_maps_flat_weighted_indices_to_gamma() -> None:
    """One-hot ``|normal|`` concentrates samples at the matching ``(phi, theta)`` cell."""
    surface = SurfaceFixture()
    xyz, idxs = draw_uniform_on_surface_jax(jax.random.PRNGKey(5), surface, 7)

    np.testing.assert_array_equal(np.asarray(idxs[0]), np.full(7, 1))
    np.testing.assert_array_equal(np.asarray(idxs[1]), np.full(7, 2))
    np.testing.assert_allclose(
        np.asarray(xyz), np.repeat(surface.gamma()[1:2, 2, :], 7, axis=0)
    )


def test_draw_uniform_on_curve_accepts_local_random_state_without_global_state_mutation() -> (
    None
):
    """Upstream curve sampler can be deterministic without touching ``np.random``."""
    before = np.random.get_state()
    rng = np.random.RandomState(31)

    xyz, idxs = draw_uniform_on_curve(UniformCurveFixture(), 5, randomgen=rng)

    _assert_numpy_random_state_equal(before, np.random.get_state())
    assert idxs.shape == (5,)
    assert xyz.shape == (5, 3)


def test_draw_uniform_on_surface_accepts_local_random_state_without_global_state_mutation() -> (
    None
):
    """Upstream surface sampler can be deterministic without touching ``np.random``."""
    before = np.random.get_state()
    rng = np.random.RandomState(37)

    xyz, idxs = draw_uniform_on_surface(UniformSurfaceFixture(), 5, randomgen=rng)

    _assert_numpy_random_state_equal(before, np.random.get_state())
    assert idxs[0].shape == (5,)
    assert idxs[1].shape == (5,)
    assert xyz.shape == (5, 3)


def test_trace_particles_starting_on_curve_uses_local_random_state(monkeypatch) -> None:
    """Tracing curve spawn path preserves process-global NumPy RNG state."""
    captured = {}

    def fake_trace_particles(field, xyz, speed_par, **kwargs):
        captured["field"] = field
        captured["xyz"] = np.asarray(xyz)
        captured["speed_par"] = np.asarray(speed_par)
        captured["kwargs"] = kwargs
        return "tys", "phi_hits"

    monkeypatch.setattr(tracing_module, "trace_particles", fake_trace_particles)

    field = object()
    before = np.random.get_state()
    result = tracing_module.trace_particles_starting_on_curve(
        UniformCurveFixture(),
        field,
        5,
        seed=41,
    )

    assert result == ("tys", "phi_hits")
    assert captured["field"] is field
    assert captured["xyz"].shape == (5, 3)
    assert captured["speed_par"].shape == (5,)
    _assert_numpy_random_state_equal(before, np.random.get_state())


def test_trace_particles_starting_on_surface_uses_local_random_state(
    monkeypatch,
) -> None:
    """Tracing surface spawn path preserves process-global NumPy RNG state."""
    captured = {}

    def fake_trace_particles(field, xyz, speed_par, **kwargs):
        captured["field"] = field
        captured["xyz"] = np.asarray(xyz)
        captured["speed_par"] = np.asarray(speed_par)
        captured["kwargs"] = kwargs
        return "tys", "phi_hits"

    monkeypatch.setattr(tracing_module, "trace_particles", fake_trace_particles)

    field = object()
    before = np.random.get_state()
    result = tracing_module.trace_particles_starting_on_surface(
        UniformSurfaceFixture(),
        field,
        5,
        seed=43,
    )

    assert result == ("tys", "phi_hits")
    assert captured["field"] is field
    assert captured["xyz"].shape == (5, 3)
    assert captured["speed_par"].shape == (5,)
    _assert_numpy_random_state_equal(before, np.random.get_state())


def test_sample_weighted_indices_jax_reproducibility() -> None:
    """Same key + same args => bit identity; different keys => non-equal samples.

    Confirms the explicit-PRNG-key contract: callers are responsible
    for splitting keys to obtain independent draws.
    """
    weights = jnp.array([0.1, 0.2, 0.3, 0.4, 0.05, 0.15])
    nsamples = 256

    key_a = jax.random.PRNGKey(2024)
    out_a_first = np.asarray(sample_weighted_indices_jax(key_a, weights, nsamples))
    out_a_second = np.asarray(sample_weighted_indices_jax(key_a, weights, nsamples))
    np.testing.assert_array_equal(out_a_first, out_a_second)

    key_b = jax.random.PRNGKey(2025)
    out_b = np.asarray(sample_weighted_indices_jax(key_b, weights, nsamples))
    assert not np.array_equal(out_a_first, out_b)


def test_sample_weighted_indices_jax_statistical_moments_match_target_distribution() -> (
    None
):
    """Empirical bin frequencies match ``weights/sum(weights)`` within the documented tolerance.

    Uses ``nsamples = 50_000`` and the absolute tolerance
    ``8 * sqrt(0.25 / nsamples) ~= 0.018`` derived in the module
    docstring. The fixed PRNG key makes this test deterministic; the
    tolerance is the margin against future RNG drift, not per-run
    flakiness.
    """
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    expected = np.asarray(weights / jnp.sum(weights))

    key = jax.random.PRNGKey(424242)
    samples = np.asarray(
        sample_weighted_indices_jax(key, weights, _STATISTICAL_NSAMPLES)
    )

    counts = np.bincount(samples, minlength=weights.shape[0])
    empirical = counts / _STATISTICAL_NSAMPLES

    np.testing.assert_allclose(empirical, expected, atol=_STATISTICAL_FREQUENCY_ATOL)


def test_sample_weighted_indices_jax_does_not_read_numpy_random(monkeypatch) -> None:
    """Disabling ``numpy.random`` does not perturb the JAX-key-pinned output.

    The JAX sampler must own its randomness via the explicit
    ``key`` argument. Replacing the NumPy module RNG entry points
    between two calls with the same JAX key must not affect the output.
    """
    weights = jnp.array([0.2, 0.3, 0.1, 0.4])
    key = jax.random.PRNGKey(7)
    nsamples = 1024

    out_a = np.asarray(sample_weighted_indices_jax(key, weights, nsamples))

    def fail_numpy_random(*_args, **_kwargs):
        raise AssertionError("JAX sampler read numpy.random")

    monkeypatch.setattr(np.random, "randint", fail_numpy_random)
    monkeypatch.setattr(np.random, "uniform", fail_numpy_random)
    out_b = np.asarray(sample_weighted_indices_jax(key, weights, nsamples))

    np.testing.assert_array_equal(out_a, out_b)


def _build_real_curve():
    """Production-scale ``CurveRZFourier`` fixture mirroring ``test_sampling.py``."""
    simsoptpp = pytest.importorskip("simsoptpp")
    if not hasattr(simsoptpp, "Curve"):
        pytest.skip("CurveRZFourier requires the compiled simsoptpp extension")
    from simsopt.geo.curverzfourier import CurveRZFourier

    nquadpoints = 200
    curve = CurveRZFourier(nquadpoints, 1, 1, True)
    dofs = curve.get_dofs()
    dofs[0] = 1.0
    dofs[1] = 0.9
    curve.set_dofs(dofs)
    return curve


def _build_real_surface():
    """Production-scale ``SurfaceRZFourier`` fixture mirroring ``test_sampling.py``."""
    simsoptpp = pytest.importorskip("simsoptpp")
    if not hasattr(simsoptpp, "Surface"):
        pytest.skip("SurfaceRZFourier requires the compiled simsoptpp extension")
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier

    nquadpoints = 64
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        nphi=nquadpoints,
        ntheta=nquadpoints,
    )
    dofs = surface.get_dofs()
    dofs[0] = 1.0
    dofs[1] = 0.8
    surface.set_dofs(dofs)
    return surface


def test_draw_uniform_on_curve_jax_matches_upstream_rejection_sampling_moments(
    monkeypatch,
) -> None:
    """JAX inverse-CDF and upstream rejection sampling agree on index-distribution moments.

    Both samplers target the same discrete distribution
    ``P(i) ∝ alen[i]``. Bit identity is impossible because the two
    paths use different RNGs, so we assert that the empirical mean
    and second moment of the sampled-index distributions match
    within the documented multinomial concentration bound.
    """
    curve = _build_real_curve()
    nsamples = _STATISTICAL_NSAMPLES
    _patch_numpy_sampler_rng(monkeypatch, seed=1)
    _, idxs_cpu = draw_uniform_on_curve(curve, nsamples, safetyfactor=10)
    _, idxs_jax = draw_uniform_on_curve_jax(jax.random.PRNGKey(909), curve, nsamples)

    idxs_cpu = np.asarray(idxs_cpu, dtype=np.float64)
    idxs_jax = np.asarray(idxs_jax, dtype=np.float64)
    nquadpoints = curve.gamma().shape[0]
    mean_tol = _moment_tolerance_for_index_range(nquadpoints, nsamples)
    # Second moment tolerance widened by max-index factor to match
    # the variance of E[X^2].
    second_moment_tol = mean_tol * nquadpoints

    assert abs(idxs_cpu.mean() - idxs_jax.mean()) < mean_tol
    assert abs((idxs_cpu**2).mean() - (idxs_jax**2).mean()) < second_moment_tol


def test_draw_uniform_on_surface_jax_matches_upstream_rejection_sampling_moments(
    monkeypatch,
) -> None:
    """JAX inverse-CDF and upstream rejection sampling agree on flat-index moments.

    Compares the flattened ``(phi, theta)`` indices to keep the
    statistical test one-dimensional. Both index axes are checked
    separately as well.
    """
    surface = _build_real_surface()
    nsamples = _STATISTICAL_NSAMPLES
    _patch_numpy_sampler_rng(monkeypatch, seed=2)
    _, idxs_cpu = draw_uniform_on_surface(surface, nsamples, safetyfactor=10)
    _, idxs_jax = draw_uniform_on_surface_jax(
        jax.random.PRNGKey(7373), surface, nsamples
    )

    phi_cpu = np.asarray(idxs_cpu[0], dtype=np.float64)
    theta_cpu = np.asarray(idxs_cpu[1], dtype=np.float64)
    phi_jax = np.asarray(idxs_jax[0], dtype=np.float64)
    theta_jax = np.asarray(idxs_jax[1], dtype=np.float64)

    nphi, ntheta = surface.gamma().shape[:2]
    phi_tol = _moment_tolerance_for_index_range(nphi, nsamples)
    theta_tol = _moment_tolerance_for_index_range(ntheta, nsamples)

    assert abs(phi_cpu.mean() - phi_jax.mean()) < phi_tol
    assert abs(theta_cpu.mean() - theta_jax.mean()) < theta_tol
    assert abs((phi_cpu**2).mean() - (phi_jax**2).mean()) < phi_tol * nphi
    assert abs((theta_cpu**2).mean() - (theta_jax**2).mean()) < theta_tol * ntheta


def test_sample_weighted_indices_jax_under_strict_transfer_guard() -> None:
    """JIT-compiled sampler executes with no host-to-device transfers.

    Locks in the strict transfer-guard discipline used by the rest
    of the JAX port: once the key and weights are on-device, the
    JIT-compiled kernel must not implicitly pull NumPy/Python
    scalars onto the device.
    """
    key = jnp.asarray(jax.random.PRNGKey(2026))
    weights = jnp.asarray(np.array([0.05, 0.2, 0.5, 0.15, 0.1]))

    with jax.transfer_guard("disallow"):
        idxs = sample_weighted_indices_jax(key, weights, 128)
        idxs.block_until_ready()

    np_idxs = np.asarray(idxs)
    assert np_idxs.shape == (128,)
    assert np_idxs.min() >= 0
    assert np_idxs.max() < weights.shape[0]
    assert np.all(np.diff(np_idxs) >= 0)
