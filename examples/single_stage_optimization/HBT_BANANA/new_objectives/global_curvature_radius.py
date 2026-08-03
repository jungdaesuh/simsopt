import numpy as np
from jax import grad
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.jit import jit


# ─────────────────────────────────────────────────────────────────────────────
# Global radius of curvature objective (Gonzalez--Maddocks self-contact)
#
# Arclength-weighted, exponentially smoothed barrier on the global radius of
# curvature of a single curve. Fires on any near-self-contact pair, so it
# replaces an index-neighbourhood mask with a purely geometric criterion.
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel returned for quadpoint pairs where the self-contact function is
# undefined (the diagonal, or a chord exactly parallel to the tangent).
# Large enough that such pairs never win a min and never move the barrier.
_UNDEFINED_RADIUS = 1.0e12


@jit
def _global_radius_curvature_pure(gamma, gammadash,
                                  minimum_radius, exp_weight):
    r"""Gonzalez--Maddocks self-contact barrier in 3D Cartesian space.

    Parameters
    ----------
    gamma : (N, 3) array
        Curve points in Cartesian coordinates.
    gammadash : (N, 3) array
        Curve tangents d(gamma)/ds.
    minimum_radius : float
        Activation threshold :math:`R_{\min}` (meters).
    exp_weight : float
        Barrier softness :math:`\varepsilon` (meters).
    """
    dl = jnp.linalg.norm(gammadash, axis=-1)
    safe_dl = jnp.where(dl > 0.0, dl, 1.0)
    tau = gammadash / safe_dl[:, None]

    diff = gamma[None, :, :] - gamma[:, None, :]  # diff[i, j] = gamma_j - gamma_i
    dsq = jnp.sum(diff * diff, axis=-1)
    safe_dsq = jnp.where(dsq > 0.0, dsq, 1.0)
    dist = jnp.where(dsq > 0.0, jnp.sqrt(safe_dsq), 0.0)

    dot_pj_tj = jnp.sum(diff * tau[None, :, :], axis=-1)
    cos_safe = dot_pj_tj / jnp.where(dist > 0.0, dist, 1.0)
    n_ratio = 1.0 - cos_safe * cos_safe
    safe_nr = jnp.where(n_ratio > 0.0, n_ratio, 1.0)
    off_diagonal = ~jnp.eye(gamma.shape[0], dtype=bool)
    coincident = off_diagonal & (dist == 0.0)
    valid = off_diagonal & (n_ratio > 0.0) & (dist > 0.0)
    S_C = jnp.where(
        coincident,
        0.0,
        jnp.where(valid, dist / safe_nr,
                  jnp.full_like(dist, _UNDEFINED_RADIUS)),
    )

    barrier = jnp.exp(-(S_C - minimum_radius) / exp_weight)

    N = gamma.shape[0]
    weights = dl[:, None] * dl[None, :]
    return jnp.sum(barrier * weights) / (N * N)


def global_curvature_radii(gamma, gammadash):
    r"""Per-quadpoint global radius of curvature, in pure NumPy.

    Returns the length-``N`` array
    :math:`\rho_{G,i} = \min_{j \ne i} S_C(\gamma_i, \gamma_j, \hat\tau_j)`,
    the raw (unsmoothed) quantity that :class:`GlobalRadiusCurvature`
    penalises. The ``i == j`` diagonal is excluded; off-diagonal coincident
    points report radius zero, while pairs where :math:`S_C` is undefined
    report the sentinel ``1e12``. Pure NumPy, so it is cheap to call from a
    driver's diagnostics CSV and needs no curve object -- but it is
    non-smooth, so do not optimise against it.

    Parameters
    ----------
    gamma : (N, 3) array
        Curve points in Cartesian coordinates.
    gammadash : (N, 3) array
        Curve tangents d(gamma)/ds.
    """
    gamma = np.asarray(gamma)
    gammadash = np.asarray(gammadash)

    dl = np.linalg.norm(gammadash, axis=-1)
    tau = gammadash / dl[:, None]
    diff = gamma[None, :, :] - gamma[:, None, :]
    dsq = np.sum(diff * diff, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = np.sqrt(np.where(dsq > 0.0, dsq, 1.0))
        proj = np.sum(diff * tau[None, :, :], axis=-1) \
            / np.where(dist > 0.0, dist, 1.0)
        n_ratio = 1.0 - proj * proj
        off_diagonal = ~np.eye(gamma.shape[0], dtype=bool)
        coincident = off_diagonal & (dsq == 0.0)
        valid = off_diagonal & (n_ratio > 0.0) & (dsq > 0.0)
        S_C = np.where(
            coincident,
            0.0,
            np.where(valid,
                     dist / np.where(n_ratio > 0.0, n_ratio, 1.0),
                     _UNDEFINED_RADIUS),
        )
    np.fill_diagonal(S_C, _UNDEFINED_RADIUS)
    return np.min(S_C, axis=1)


class GlobalRadiusCurvature(Optimizable):
    r"""
    Smooth self-intersection penalty based on the global radius of curvature
    of a curve, computed directly from its 3D Cartesian coordinates.

    The objective is the arc-length-weighted, exponentially smoothed double
    integral of the Gonzalez--Maddocks self-contact function

    .. math::
        J = \frac{1}{N^{2}} \sum_{i \ne j}
            \exp\!\bigl(-(S_C(\gamma_i, \gamma_j, \hat\tau_j) - R_{\min})
                       / \varepsilon\bigr)\,
            \,\lVert \dot\gamma_i \rVert \, \lVert \dot\gamma_j \rVert,

    with

    .. math::
        S_C(p_1, p_2, \tau_2) =
            \frac{\lVert p_1 - p_2 \rVert}
                 {1 - \bigl((p_2 - p_1)\cdot \tau_2 /
                            \lVert p_1 - p_2 \rVert\bigr)^{2}}.

    :math:`J` is smooth away from exact coincidences and the undefined
    tangent-parallel branch, accumulates contributions from every
    near-self-contact pair, and acts as a finite exponential soft barrier on
    the constraint :math:`\rho_G \ge R_{\min}`. The
    :math:`1 - \cos^2(\angle)` factor makes the barrier vanish on
    adjacent-quadpoint pairs automatically, so no neighbour-skip mask is
    required.

    Diagnostic methods :meth:`global_curvature_radii` and
    :meth:`shortest_radius` expose the raw global radius of
    curvature for post-hoc inspection; the raw minimum is non-smooth and is
    not recommended as an optimisation target. Exact off-diagonal
    coincidences receive radius zero and a finite, maximally violating
    exponential penalty; only the diagonal is excluded from the barrier.

    Parameters
    ----------
    curve : simsopt.geo.curve.Curve
        Curve to penalise. Must expose ``gamma()``, ``gammadash()``,
        ``dgamma_by_dcoeff_vjp`` and ``dgammadash_by_dcoeff_vjp``.
    minimum_radius : float
        Activation threshold :math:`R_{\min}` (meters).
    exp_weight : float, optional
        Barrier softness :math:`\varepsilon` (meters). Default 0.01.
    """

    def __init__(self, curve, minimum_radius, exp_weight=0.01):
        self.curve = curve
        self.minimum_radius = minimum_radius
        self.exp_weight = exp_weight
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda g, gd: _global_radius_curvature_pure(
            g, gd, minimum_radius, exp_weight))
        self.dJ_dgamma = jit(lambda g, gd: grad(self.J_jax, argnums=0)(g, gd))
        self.dJ_dgammadash = jit(
            lambda g, gd: grad(self.J_jax, argnums=1)(g, gd))

    # ── Diagnostics ────────────────────────────────────────────────────
    def global_curvature_radii(self):
        """Per-quadpoint global radius of curvature of ``self.curve``.

        Binds the curve to the module-level :func:`global_curvature_radii`.
        """
        return global_curvature_radii(self.curve.gamma(),
                                      self.curve.gammadash())

    def shortest_radius(self):
        """Smallest global radius of curvature over all quadpoints."""
        return float(np.min(self.global_curvature_radii()))

    # ── Optimizable API ────────────────────────────────────────────────
    def J(self):
        return float(self.J_jax(self.curve.gamma(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        return (self.curve.dgamma_by_dcoeff_vjp(
                    np.asarray(self.dJ_dgamma(g, gd)))
                + self.curve.dgammadash_by_dcoeff_vjp(
                    np.asarray(self.dJ_dgammadash(g, gd))))

    return_fn_map = {'J': J, 'dJ': dJ}
