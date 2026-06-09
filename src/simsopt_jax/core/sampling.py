"""Pure JAX kernels for weighted curve/surface point sampling.

This is the kernel SSOT for the JAX port of
``simsopt.field.sampling`` (jax-port plan item 22, Wave R4). It
implements weighted point sampling with an explicit
:func:`jax.random.PRNGKey` contract — every public sampling function
takes a ``key`` as its first required argument and never reads
NumPy/JAX global RNG state. This makes the JAX path reproducible,
parallelisable, and independent of the upstream
``numpy.random``-based rejection sampler in
``simsopt/field/sampling.py``.

Mathematical equivalence to upstream rejection sampling
-------------------------------------------------------

The upstream ``draw_uniform_on_curve`` / ``draw_uniform_on_surface``
draw candidate quadrature indices uniformly from
``[0, N)`` and accept index ``i`` with probability
``alen[i] / max(alen)`` (likewise for ``|normal|``). After acceptance,
the marginal probability of any index ``i`` is

.. math:: P(i) \\propto \\frac{1}{N} \\cdot \\frac{w_i}{\\max_j w_j}
                       \\propto w_i

which is exactly the discrete categorical distribution with weights
proportional to ``w``. The JAX kernels here sample directly from that
distribution via the inverse-CDF method (``jnp.searchsorted`` on
``jnp.cumsum(weights / weights.sum())``). This is mathematically
equivalent to upstream rejection sampling, JIT-compatible, fully
deterministic in ``(key, weights, nsamples)``, and avoids the
``assert len(accept) > nsamples`` brittleness of the rejection path.

The inverse-CDF kernel is JIT-compiled with ``nsamples`` static so
that the output shape is fixed across calls. The result is sorted in
ascending order to match the upstream contract (``np.sort(idxs)`` at
the end of the rejection loop).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("nsamples",))
def sample_weighted_indices_jax(
    key: jax.Array, weights: jax.Array, nsamples: int
) -> jax.Array:
    """Draw ``nsamples`` indices from a discrete distribution proportional to ``weights``.

    Parameters
    ----------
    key : jax.Array
        A ``jax.random.PRNGKey``. Two calls with the same ``key`` and
        the same ``(weights, nsamples)`` produce bit-identical output.
        Independent samples require splitting the key
        (``jax.random.split``) before calling.
    weights : jax.Array
        1D non-negative weights. Need not sum to one; the kernel
        normalises internally. ``weights.sum()`` must be strictly
        positive — passing the all-zero vector is a caller error.
    nsamples : int
        Number of samples to draw. Marked static so the compiled
        graph has a fixed output shape.

    Returns
    -------
    jax.Array
        1D ``int32`` array of length ``nsamples`` with indices in
        ``[0, weights.shape[0])`` sorted in ascending order.
    """
    cdf = jnp.cumsum(weights / jnp.sum(weights))
    uniforms = jax.random.uniform(key, shape=(nsamples,), dtype=cdf.dtype)
    raw = jnp.searchsorted(cdf, uniforms, side="right")
    capped = jnp.minimum(raw, weights.shape[0] - 1)
    return jnp.sort(capped)


def draw_uniform_on_curve_jax(
    key: jax.Array, curve, nsamples: int
) -> tuple[jax.Array, jax.Array]:
    """Sample curve quadrature points with probability proportional to local arclength.

    JAX counterpart of :func:`simsopt.field.sampling.draw_uniform_on_curve`.
    The PRNG key is an explicit required argument so callers always
    own the randomness; there is no hidden ``numpy.random`` or
    ``jax.random`` global state.

    Parameters
    ----------
    key : jax.Array
        A ``jax.random.PRNGKey``.
    curve
        Any object exposing ``incremental_arclength()`` (1D array of
        length ``nquadpoints``) and ``gamma()`` (``(nquadpoints, 3)``
        Cartesian quadrature points). The upstream
        ``simsopt.geo.curve.Curve`` API satisfies this.
    nsamples : int
        Number of points to draw.

    Returns
    -------
    xyz : jax.Array
        ``(nsamples, 3)`` array of Cartesian coordinates.
    idxs : jax.Array
        1D ``int32`` array of sorted quadrature indices.
    """
    idxs = sample_weighted_indices_jax(
        key,
        jnp.asarray(curve.incremental_arclength()),
        nsamples,
    )
    xyz = jnp.asarray(curve.gamma())[idxs, :]
    return xyz, idxs


def draw_uniform_on_surface_jax(
    key: jax.Array, surface, nsamples: int
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Sample surface quadrature points with probability proportional to local area.

    JAX counterpart of :func:`simsopt.field.sampling.draw_uniform_on_surface`.
    The PRNG key is an explicit required argument so callers always
    own the randomness; there is no hidden ``numpy.random`` or
    ``jax.random`` global state.

    Parameters
    ----------
    key : jax.Array
        A ``jax.random.PRNGKey``.
    surface
        Any object exposing ``gamma()`` (``(nphi, ntheta, 3)``
        Cartesian quadrature points) and ``normal()`` (``(nphi, ntheta, 3)``
        surface-normal vectors whose magnitude is the local area
        element). The upstream ``simsopt.geo.surface.Surface`` API
        satisfies this.
    nsamples : int
        Number of points to draw.

    Returns
    -------
    xyz : jax.Array
        ``(nsamples, 3)`` array of Cartesian coordinates.
    idxs : tuple of two jax.Array
        ``(phi_idxs, theta_idxs)``, each ``(nsamples,)`` int arrays
        matching the upstream
        :func:`simsopt.field.sampling.draw_uniform_on_surface` tuple
        layout. Memory order (C vs Fortran) is preserved from
        ``surface.gamma()``.
    """
    gamma_host = surface.gamma()
    order = "F" if _is_fortran_contiguous(gamma_host) else "C"
    nphi, ntheta = gamma_host.shape[:2]
    flat_gamma = jnp.reshape(jnp.asarray(gamma_host), (-1, 3), order=order)
    flat_normal = jnp.reshape(jnp.asarray(surface.normal()), (-1, 3), order=order)
    idxs = sample_weighted_indices_jax(
        key,
        jnp.linalg.norm(flat_normal, axis=1),
        nsamples,
    )
    xyz = flat_gamma[idxs, :]
    if order == "F":
        phi_idxs = idxs % nphi
        theta_idxs = idxs // nphi
    else:
        phi_idxs = idxs // ntheta
        theta_idxs = idxs % ntheta
    return xyz, (phi_idxs, theta_idxs)


def _is_fortran_contiguous(array) -> bool:
    """Detect Fortran-contiguous memory layout for both NumPy and JAX arrays.

    ``jax.Array`` does not expose ``flags['F_CONTIGUOUS']``; fall back
    to ``False`` for JAX arrays since JAX uses C order by default.
    """
    flags = getattr(array, "flags", None)
    if flags is None:
        return False
    fortran_flag = getattr(flags, "f_contiguous", None)
    if fortran_flag is None:
        return False
    return bool(fortran_flag)
