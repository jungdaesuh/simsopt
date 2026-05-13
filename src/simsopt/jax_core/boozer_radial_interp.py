"""JAX port of ``boozerradialinterpolant.cpp`` (Tier P5 item 32).

This module re-implements the six Fourier helper kernels defined in
``src/simsoptpp/boozerradialinterpolant.cpp`` (upstream sha
``1b0cc3a96063197cdbdd01559e04c25456fbe6ff``):

- :func:`compute_kmnc_kmns` — non-stellsym K Fourier projection on a
  ``(num_modes, num_surf)`` half-grid.
- :func:`compute_kmns` — stellsym K Fourier projection.
- :func:`fourier_transform_odd` — sin-mode coefficient projection with
  per-mode L2 normalisation.
- :func:`fourier_transform_even` — cos-mode coefficient projection with
  per-mode L2 normalisation.
- :func:`inverse_fourier_transform_odd` — sum ``kmns(im,...) * sin(...)``.
- :func:`inverse_fourier_transform_even` — sum ``kmns(im,...) * cos(...)``.

Scope notes (matches the C++ contract one-for-one):

- ``compute_kmnc_kmns`` and ``compute_kmns`` use the **trapezoidal** Fourier
  projection rule built into the calling Python wrapper (the wrapper rolls
  ``dtheta * dzeta * nfp / psi0`` in afterwards). The C++ kernel emits the
  ``1/(2*pi**2)`` (and ``1/(4*pi**2)`` for the ``im=0`` term in the
  stellsym-broken cos block) normalisation factors verbatim.
- ``fourier_transform_{odd,even}`` use the **point-sum** normalisation
  (``sum f * basis / sum basis**2``); they are used by upstream tests but
  are not exercised by the production ``BoozerRadialInterpolant`` path.
- ``inverse_fourier_transform_{odd,even}`` accept either a 1D
  ``(num_modes,)`` coefficient vector (broadcast across all evaluation
  points) or a 2D ``(num_modes, num_points)`` coefficient table (the
  diagonal-broadcast variant used by ``BoozerRadialInterpolant._K_impl``).

The radial spline construction itself lives in the Python wrapper using
``scipy.interpolate.InterpolatedUnivariateSpline`` and is **out of
scope** for this port. The downstream
``BoozerMagneticField``/``BoozerRadialInterpolant`` adapter wrapper is
tracked separately as item 33 (P5) and is **not** part of this item.

All kernels are pure JAX, ``jit``-compatible, and matmul-style: the C++
double-nested loops collapse to ``einsum``s over the
``(num_points, num_modes)`` angle table. Numerical results match the C++
kernel to ``direct_kernel`` tolerance (``rtol=1e-10``, ``atol=1e-12``).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _build_angle_basis(
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return ``cos(angle)``, ``sin(angle)`` tables of shape
    ``(num_points, num_modes)``.

    ``angle[ip, im] = xm[im] * thetas[ip] - xn[im] * zetas[ip]``.
    """
    # (num_points, num_modes)
    angle = thetas[:, None] * xm[None, :] - zetas[:, None] * xn[None, :]
    return jnp.cos(angle), jnp.sin(angle)


# ----------------------------------------------------------------------
# compute_kmns / compute_kmnc_kmns
# ----------------------------------------------------------------------


def _compute_K_per_point(
    *,
    cos_a: jax.Array,
    sin_a: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    rmnc: jax.Array,
    drmncds: jax.Array,
    zmns: jax.Array,
    dzmnsds: jax.Array,
    numns: jax.Array,
    dnumnsds: jax.Array,
    bmnc: jax.Array,
    zetas: jax.Array,
    iota_isurf: jax.Array,
    G_isurf: jax.Array,
    I_isurf: jax.Array,
    rmns: jax.Array | None = None,
    drmnsds: jax.Array | None = None,
    zmnc: jax.Array | None = None,
    dzmncds: jax.Array | None = None,
    numnc: jax.Array | None = None,
    dnumncds: jax.Array | None = None,
    bmns: jax.Array | None = None,
) -> jax.Array:
    """Evaluate ``K(theta, zeta)`` at a single surface for every angle
    sample using closed-form Boozer-coordinate geometry.

    Inputs are restricted to one surface column (``rmnc[:, isurf]`` etc.).
    ``cos_a``/``sin_a`` have shape ``(num_points, num_modes)``.

    Returns ``K`` of shape ``(num_points,)``.
    """
    # Fourier sums over modes; result shape (num_points,).
    have_asym = rmns is not None

    if have_asym:
        B = cos_a @ bmnc + sin_a @ bmns
        R = cos_a @ rmnc + sin_a @ rmns
        dRdtheta = -(sin_a * xm[None, :]) @ rmnc + (cos_a * xm[None, :]) @ rmns
        dRdzeta = (sin_a * xn[None, :]) @ rmnc - (cos_a * xn[None, :]) @ rmns
        dRds = cos_a @ drmncds + sin_a @ drmnsds
        dZdtheta = (cos_a * xm[None, :]) @ zmns - (sin_a * xm[None, :]) @ zmnc
        dZdzeta = -(cos_a * xn[None, :]) @ zmns + (sin_a * xn[None, :]) @ zmnc
        dZds = sin_a @ dzmnsds + cos_a @ dzmncds
        nu = sin_a @ numns + cos_a @ numnc
        dnuds = sin_a @ dnumnsds + cos_a @ dnumncds
        dnudtheta = (cos_a * xm[None, :]) @ numns - (sin_a * xm[None, :]) @ numnc
        dnudzeta = -(cos_a * xn[None, :]) @ numns + (sin_a * xn[None, :]) @ numnc
    else:
        B = cos_a @ bmnc
        R = cos_a @ rmnc
        dRdtheta = -(sin_a * xm[None, :]) @ rmnc
        dRdzeta = (sin_a * xn[None, :]) @ rmnc
        dRds = cos_a @ drmncds
        dZdtheta = (cos_a * xm[None, :]) @ zmns
        dZdzeta = -(cos_a * xn[None, :]) @ zmns
        dZds = sin_a @ dzmnsds
        nu = sin_a @ numns
        dnuds = sin_a @ dnumnsds
        dnudtheta = (cos_a * xm[None, :]) @ numns
        dnudzeta = -(cos_a * xn[None, :]) @ numns

    phi = zetas - nu
    dphids = -dnuds
    dphidtheta = -dnudtheta
    dphidzeta = 1.0 - dnudzeta

    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    dXdtheta = dRdtheta * cos_phi - R * sin_phi * dphidtheta
    dYdtheta = dRdtheta * sin_phi + R * cos_phi * dphidtheta
    dXds = dRds * cos_phi - R * sin_phi * dphids
    dYds = dRds * sin_phi + R * cos_phi * dphids
    dXdzeta = dRdzeta * cos_phi - R * sin_phi * dphidzeta
    dYdzeta = dRdzeta * sin_phi + R * cos_phi * dphidzeta

    gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds
    gszeta = dXdzeta * dXds + dYdzeta * dYds + dZdzeta * dZds
    sqrtg = (G_isurf + iota_isurf * I_isurf) / (B * B)
    return (gszeta + iota_isurf * gstheta) / sqrtg


@partial(jax.jit, static_argnames=())
def compute_kmns(
    rmnc: jax.Array,
    drmncds: jax.Array,
    zmns: jax.Array,
    dzmnsds: jax.Array,
    numns: jax.Array,
    dnumnsds: jax.Array,
    bmnc: jax.Array,
    iota: jax.Array,
    G: jax.Array,
    I: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """JAX port of ``simsoptpp.compute_kmns`` (stellsym).

    Returns ``kmns`` of shape ``(num_modes, num_surf)``.

    The ``im=0`` row is always zero (matches the C++ ``for (im=1; ...)`` loop).
    The accumulated stellsym K projection per surface is::

        kmns[im, isurf] = sum_ip K(ip, isurf) * sin(angle(ip, im)) / (2 * pi**2)

    where ``K`` is the half-grid metric quantity computed inline.
    """
    cos_a, sin_a = _build_angle_basis(xm, xn, thetas, zetas)
    num_surf = rmnc.shape[1]

    def per_surface(isurf):
        K = _compute_K_per_point(
            cos_a=cos_a,
            sin_a=sin_a,
            xm=xm,
            xn=xn,
            rmnc=rmnc[:, isurf],
            drmncds=drmncds[:, isurf],
            zmns=zmns[:, isurf],
            dzmnsds=dzmnsds[:, isurf],
            numns=numns[:, isurf],
            dnumnsds=dnumnsds[:, isurf],
            bmnc=bmnc[:, isurf],
            zetas=zetas,
            iota_isurf=iota[isurf],
            G_isurf=G[isurf],
            I_isurf=I[isurf],
        )
        # kmns[im] = sum_ip K[ip] * sin(angle[ip, im]) / (2 pi**2), im>=1
        # im=0 row is zeroed.
        sin_only = sin_a.at[:, 0].set(0.0)
        return (K[None, :] @ sin_only).ravel() / (2.0 * jnp.pi * jnp.pi)

    # Loop over surfaces, stack into (num_surf, num_modes), then transpose.
    kmns_T = jax.vmap(per_surface)(jnp.arange(num_surf))
    return kmns_T.T  # (num_modes, num_surf)


@partial(jax.jit, static_argnames=())
def compute_kmnc_kmns(
    rmnc: jax.Array,
    drmncds: jax.Array,
    zmns: jax.Array,
    dzmnsds: jax.Array,
    numns: jax.Array,
    dnumnsds: jax.Array,
    bmnc: jax.Array,
    rmns: jax.Array,
    drmnsds: jax.Array,
    zmnc: jax.Array,
    dzmncds: jax.Array,
    numnc: jax.Array,
    dnumncds: jax.Array,
    bmns: jax.Array,
    iota: jax.Array,
    G: jax.Array,
    I: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """JAX port of ``simsoptpp.compute_kmnc_kmns`` (non-stellsym).

    Returns ``kmnc_kmns`` of shape ``(2, num_modes, num_surf)`` where
    ``[0]`` is the cos coefficients and ``[1]`` is the sin coefficients.

    Normalisation matches the C++ kernel exactly:

    - ``kmnc[im=0]`` accumulates with ``1 / (4 * pi**2)``.
    - ``kmnc[im>=1]`` and ``kmns[im>=1]`` accumulate with ``1 / (2 * pi**2)``.
    - ``kmns[im=0]`` is identically zero (matches C++ ``if (im > 0)`` guard).
    """
    cos_a, sin_a = _build_angle_basis(xm, xn, thetas, zetas)
    num_surf = rmnc.shape[1]

    def per_surface(isurf):
        K = _compute_K_per_point(
            cos_a=cos_a,
            sin_a=sin_a,
            xm=xm,
            xn=xn,
            rmnc=rmnc[:, isurf],
            drmncds=drmncds[:, isurf],
            zmns=zmns[:, isurf],
            dzmnsds=dzmnsds[:, isurf],
            numns=numns[:, isurf],
            dnumnsds=dnumnsds[:, isurf],
            bmnc=bmnc[:, isurf],
            zetas=zetas,
            iota_isurf=iota[isurf],
            G_isurf=G[isurf],
            I_isurf=I[isurf],
            rmns=rmns[:, isurf],
            drmnsds=drmnsds[:, isurf],
            zmnc=zmnc[:, isurf],
            dzmncds=dzmncds[:, isurf],
            numnc=numnc[:, isurf],
            dnumncds=dnumncds[:, isurf],
            bmns=bmns[:, isurf],
        )
        # Cos coefficients: im=0 uses 1/(4 pi**2), im>=1 uses 1/(2 pi**2).
        cos_proj = (K[None, :] @ cos_a).ravel()  # (num_modes,)
        # Scale factors per mode index.
        # weight[0] = 1/(4 pi**2); weight[im>=1] = 1/(2 pi**2)
        pi2 = jnp.pi * jnp.pi
        scale = jnp.where(
            jnp.arange(cos_proj.shape[0]) == 0,
            1.0 / (4.0 * pi2),
            1.0 / (2.0 * pi2),
        )
        kmnc_isurf = cos_proj * scale

        # Sin coefficients: im=0 zeroed, im>=1 uses 1/(2 pi**2).
        sin_only = sin_a.at[:, 0].set(0.0)
        kmns_isurf = (K[None, :] @ sin_only).ravel() / (2.0 * pi2)

        return kmnc_isurf, kmns_isurf

    kmnc_T, kmns_T = jax.vmap(per_surface)(jnp.arange(num_surf))
    # Stack along axis 0 to get (2, num_modes, num_surf).
    return jnp.stack([kmnc_T.T, kmns_T.T], axis=0)


# ----------------------------------------------------------------------
# Per-mode point-sum projections (fourier_transform_{odd,even})
# ----------------------------------------------------------------------


@partial(jax.jit, static_argnames=())
def fourier_transform_odd(
    K: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """JAX port of ``simsoptpp.fourier_transform_odd``.

    Per-mode sin coefficient projection with point-sum normalisation::

        kmns[im] = sum_ip K[ip] * sin(angle[ip, im]) /
                   sum_ip sin(angle[ip, im])**2,   im >= 1

    ``kmns[0]`` is set to zero (matches the C++ ``for (im=1; ...)`` loop;
    upstream uses the returned array only for ``im >= 1`` consumers).
    """
    _, sin_a = _build_angle_basis(xm, xn, thetas, zetas)
    numer = sin_a.T @ K  # (num_modes,)
    denom = jnp.sum(sin_a * sin_a, axis=0)  # (num_modes,)
    # Guard im=0 (which the C++ skips entirely).
    safe_denom = jnp.where(jnp.arange(numer.shape[0]) == 0, 1.0, denom)
    result = numer / safe_denom
    return jnp.where(jnp.arange(numer.shape[0]) == 0, 0.0, result)


@partial(jax.jit, static_argnames=())
def fourier_transform_even(
    K: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """JAX port of ``simsoptpp.fourier_transform_even``.

    Per-mode cos coefficient projection with point-sum normalisation::

        kmnc[im] = sum_ip K[ip] * cos(angle[ip, im]) /
                   sum_ip cos(angle[ip, im])**2
    """
    cos_a, _ = _build_angle_basis(xm, xn, thetas, zetas)
    numer = cos_a.T @ K  # (num_modes,)
    denom = jnp.sum(cos_a * cos_a, axis=0)  # (num_modes,)
    return numer / denom


# ----------------------------------------------------------------------
# Inverse projection (inverse_fourier_transform_{odd,even})
# ----------------------------------------------------------------------


@partial(jax.jit, static_argnames=())
def inverse_fourier_transform_odd_1d(
    kmns: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """JAX port of the 1D branch of ``simsoptpp.inverse_fourier_transform_odd``.

    For a single-vector coefficient ``kmns`` of shape ``(num_modes,)``,
    returns ``K[ip] = sum_{im>=1} kmns[im] * sin(angle[ip, im])``.

    The ``im=0`` term is suppressed (matches the C++ ``for (im=1; ...)`` loop).
    """
    _, sin_a = _build_angle_basis(xm, xn, thetas, zetas)
    # Zero the im=0 contribution to match C++ semantics.
    kmns_no_dc = kmns.at[0].set(0.0)
    return sin_a @ kmns_no_dc


@partial(jax.jit, static_argnames=())
def inverse_fourier_transform_odd_2d(
    kmns: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """JAX port of the 2D branch of ``simsoptpp.inverse_fourier_transform_odd``.

    For a per-point coefficient table ``kmns`` of shape
    ``(num_modes, num_points)``, returns
    ``K[ip] = sum_{im>=1} kmns[im, ip] * sin(angle[ip, im])``.

    This is the "diagonal-broadcast" variant used by
    ``BoozerRadialInterpolant._K_impl`` where each evaluation point uses
    its own radial coefficient column.
    """
    _, sin_a = _build_angle_basis(xm, xn, thetas, zetas)
    # Zero im=0 row to match C++ semantics.
    kmns_no_dc = kmns.at[0, :].set(0.0)
    # Pointwise: K[ip] = sum_im kmns[im, ip] * sin_a[ip, im]
    return jnp.einsum("mp,pm->p", kmns_no_dc, sin_a)


def inverse_fourier_transform_odd(
    kmns: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """Dispatch to the 1D or 2D variant based on ``kmns.ndim``.

    Matches the polymorphic C++ ``inverse_fourier_transform_odd`` signature.
    The C++ kernel accumulates ``K += ...``; this Python entry point
    **returns** the contribution and leaves accumulation to the caller.
    """
    if kmns.ndim == 1:
        return inverse_fourier_transform_odd_1d(kmns, xm, xn, thetas, zetas)
    if kmns.ndim == 2:
        return inverse_fourier_transform_odd_2d(kmns, xm, xn, thetas, zetas)
    raise ValueError(
        f"kmns must have ndim 1 or 2, got ndim={kmns.ndim}; shape={tuple(kmns.shape)!r}"
    )


@partial(jax.jit, static_argnames=())
def inverse_fourier_transform_even_1d(
    kmnc: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """JAX port of the 1D branch of ``simsoptpp.inverse_fourier_transform_even``.

    For a single-vector coefficient ``kmnc`` of shape ``(num_modes,)``,
    returns ``K[ip] = sum_{im>=0} kmnc[im] * cos(angle[ip, im])``.
    """
    cos_a, _ = _build_angle_basis(xm, xn, thetas, zetas)
    return cos_a @ kmnc


@partial(jax.jit, static_argnames=())
def inverse_fourier_transform_even_2d(
    kmnc: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """JAX port of the 2D branch of ``simsoptpp.inverse_fourier_transform_even``.

    For a per-point coefficient table ``kmnc`` of shape
    ``(num_modes, num_points)``, returns
    ``K[ip] = sum_{im>=0} kmnc[im, ip] * cos(angle[ip, im])``.
    """
    cos_a, _ = _build_angle_basis(xm, xn, thetas, zetas)
    return jnp.einsum("mp,pm->p", kmnc, cos_a)


def inverse_fourier_transform_even(
    kmnc: jax.Array,
    xm: jax.Array,
    xn: jax.Array,
    thetas: jax.Array,
    zetas: jax.Array,
) -> jax.Array:
    """Dispatch to the 1D or 2D variant based on ``kmnc.ndim``.

    Matches the polymorphic C++ ``inverse_fourier_transform_even`` signature.
    """
    if kmnc.ndim == 1:
        return inverse_fourier_transform_even_1d(kmnc, xm, xn, thetas, zetas)
    if kmnc.ndim == 2:
        return inverse_fourier_transform_even_2d(kmnc, xm, xn, thetas, zetas)
    raise ValueError(
        f"kmnc must have ndim 1 or 2, got ndim={kmnc.ndim}; shape={tuple(kmnc.shape)!r}"
    )


__all__ = [
    "compute_kmnc_kmns",
    "compute_kmns",
    "fourier_transform_even",
    "fourier_transform_odd",
    "inverse_fourier_transform_even",
    "inverse_fourier_transform_even_1d",
    "inverse_fourier_transform_even_2d",
    "inverse_fourier_transform_odd",
    "inverse_fourier_transform_odd_1d",
    "inverse_fourier_transform_odd_2d",
]
