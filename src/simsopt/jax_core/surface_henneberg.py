"""Pure JAX kernels for SurfaceHenneberg geometry.

CPU oracle: ``simsopt.geo.surfacehenneberg.SurfaceHenneberg.gamma_impl``,
``gammadash1_impl``, and ``gammadash2_impl`` (see
``src/simsopt/geo/surfacehenneberg.py:588-740``). The math implements the
Henneberg-Helander-Drevlak parameterisation (*J. Plasma Phys.* 87,
905870503, 2021).

Position formulas
-----------------
With ``α = 0.5·nfp·alpha_fac`` and ``θ̄ = θ − α·φ``:

    R0H(φ) = Σ_n R0nH[n] cos(nfp·n·φ)        n = 0..nmax
    Z0H(φ) = Σ_n Z0nH[n] sin(nfp·n·φ)        n = 1..nmax
    b(φ)   = Σ_n  bn[n]  cos(nfp·n·φ)        n = 0..nmax
    ρ(θ,φ) = Σ_{m,n} ρ_{m,n} cos(m·θ + nfp·n·φ - α·φ)
                                              m = 0..mmax, n = -nmax..nmax
                                              ρ_{0, n<=0} ≡ 0
    ζ(θ,φ) = b(φ) · sin(θ̄)

    R(θ,φ) = R0H(φ) + ρ cos(α·φ) - ζ sin(α·φ)
    Z(θ,φ) = Z0H(φ) + ρ sin(α·φ) + ζ cos(α·φ)
    γ      = (R cos(φ), R sin(φ), Z)

All kernels return ``(nphi, ntheta, 3)`` arrays in ``γ`` axis order to
match the CPU oracle (and the SIMSOPT surface ``data[:, :, xyz]``
convention used throughout this codebase).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..geo.surface_fourier_jax import surface_area, surface_volume
from ._math_utils import as_jax_float64 as _as_float64_array
from .specs import SurfaceHennebergSpec


# ---------------------------------------------------------------------------
# Static index helpers
# ---------------------------------------------------------------------------


def _two_pi_like(reference: jax.Array) -> jax.Array:
    """Return ``2π`` as a device-resident float64 scalar of ``reference``'s dtype."""
    return jnp.asarray(2.0 * np.pi, dtype=reference.dtype)


def _alpha_scalar(spec: SurfaceHennebergSpec, reference: jax.Array) -> jax.Array:
    """Return ``α = 0.5·nfp·alpha_fac`` as a device-resident float64 scalar."""
    alpha = 0.5 * float(spec.nfp) * float(spec.alpha_fac)
    return jnp.asarray(alpha, dtype=reference.dtype)


def _n_indices_1d(spec: SurfaceHennebergSpec) -> jax.Array:
    """Integer ``n`` axis values 0..nmax (host-built; baked into trace)."""
    return _as_float64_array(np.arange(spec.nmax + 1, dtype=np.float64))


def _mn_indices_2d(
    spec: SurfaceHennebergSpec,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return broadcast-ready ``(M, N, valid_mask)`` for the ρ sum.

    ``M`` has shape ``(mmax+1, 2·nmax+1)`` with integer rows ``m = 0..mmax``.
    ``N`` has the same shape with integer columns ``n = -nmax..nmax``.
    ``valid_mask`` is the boolean mask of host-class allowed ``(m, n)``
    cells: ``True`` everywhere except ``(m=0, n<=0)`` which is held at
    zero by the host class convention (``surfacehenneberg.py:610-614``).
    The mask is materialised as a float64 ``{0.0, 1.0}`` array so the
    kernel can pre-multiply ρ_{m, n} without a host branch.
    """
    mmax = int(spec.mmax)
    nmax = int(spec.nmax)
    m_axis = np.arange(mmax + 1, dtype=np.float64).reshape(mmax + 1, 1)
    n_axis = np.arange(-nmax, nmax + 1, dtype=np.float64).reshape(1, 2 * nmax + 1)
    m_grid = np.broadcast_to(m_axis, (mmax + 1, 2 * nmax + 1))
    n_grid = np.broadcast_to(n_axis, (mmax + 1, 2 * nmax + 1))

    # Host-class convention: skip (m=0, n<=0) — see surfacehenneberg.py:610-614
    # (``nmin = 1 if m == 0 else -nmax``).
    valid = np.ones((mmax + 1, 2 * nmax + 1), dtype=np.float64)
    valid[0, : nmax + 1] = 0.0  # zero out n = -nmax..0 in the m=0 row

    return (
        _as_float64_array(m_grid),
        _as_float64_array(n_grid),
        _as_float64_array(valid),
    )


def _z0_n_mask(spec: SurfaceHennebergSpec) -> jax.Array:
    """Mask that zeroes the n=0 column of ``Z0nH`` (CPU loop starts at n=1)."""
    mask = np.ones(spec.nmax + 1, dtype=np.float64)
    mask[0] = 0.0
    return _as_float64_array(mask)


# ---------------------------------------------------------------------------
# Shared real-space tensors (phi axis, theta axis, then 2D grid)
# ---------------------------------------------------------------------------


def _phi_theta_radian_grid(
    spec: SurfaceHennebergSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``(phi_1d_rad, theta_1d_rad, phi_2d_rad, theta_2d_rad)``.

    The CPU oracle scales quadpoints by 2π so that the cos/sin arguments
    are evaluated in radian space (``surfacehenneberg.py:594-595``,
    ``648-649``, ``712-713``). The 2D grid is ``ij`` indexed so the
    output shape is ``(nphi, ntheta)`` directly without an explicit
    transpose.
    """
    two_pi = _two_pi_like(spec.quadpoints_phi)
    phi_1d = spec.quadpoints_phi * two_pi
    theta_1d = spec.quadpoints_theta * two_pi
    phi_2d, theta_2d = jnp.meshgrid(phi_1d, theta_1d, indexing="ij")
    return phi_1d, theta_1d, phi_2d, theta_2d


def _phi_axis_modes(
    spec: SurfaceHennebergSpec, phi_1d: jax.Array
) -> dict[str, jax.Array]:
    """Compute the 1D φ-axis modes ``R0H, Z0H, b, d_R0H/dφ, d_Z0H/dφ, d_b/dφ``.

    Each return value has shape ``(nphi,)``. The trig arguments are
    ``nfp·n·φ`` over integer ``n ∈ {0, .., nmax}``. The ``Z0nH`` entry
    at ``n=0`` is masked out (the CPU loop starts at ``n=1``).
    """
    n_axis = _n_indices_1d(spec)  # shape (nmax+1,)
    nfp = jnp.asarray(float(spec.nfp), dtype=phi_1d.dtype)
    n_nfp = n_axis * nfp  # shape (nmax+1,)
    # Outer product (nphi, nmax+1)
    angle = phi_1d[:, None] * n_nfp[None, :]
    cos_ang = jnp.cos(angle)
    sin_ang = jnp.sin(angle)

    z_mask = _z0_n_mask(spec)  # (nmax+1,) with z_mask[0] = 0
    z_modes = spec.Z0nH * z_mask

    R0H = cos_ang @ spec.R0nH
    Z0H = sin_ang @ z_modes
    b = cos_ang @ spec.bn

    # Derivatives: d/dφ cos(nfp·n·φ) = -nfp·n·sin(nfp·n·φ);
    #              d/dφ sin(nfp·n·φ) = +nfp·n·cos(nfp·n·φ).
    dR0H_dphi = (-sin_ang * n_nfp[None, :]) @ spec.R0nH
    dZ0H_dphi = (cos_ang * n_nfp[None, :]) @ z_modes
    db_dphi = (-sin_ang * n_nfp[None, :]) @ spec.bn

    return {
        "R0H": R0H,
        "Z0H": Z0H,
        "b": b,
        "dR0H_dphi": dR0H_dphi,
        "dZ0H_dphi": dZ0H_dphi,
        "db_dphi": db_dphi,
    }


def _rho_and_partials(
    spec: SurfaceHennebergSpec,
    phi_2d: jax.Array,
    theta_2d: jax.Array,
) -> dict[str, jax.Array]:
    """Compute ρ(θ,φ), ∂ρ/∂φ, ∂ρ/∂θ on the full (nphi, ntheta) grid.

    The mode sum is ``Σ_{m,n} ρ_{m,n} · cos(m·θ + (nfp·n - α)·φ)``,
    excluding the host-class-zero (m=0, n<=0) cells. Implemented as a
    contraction with the pre-broadcast (mmax+1, 2·nmax+1) mode grids;
    no Python loop over (m, n).
    """
    nfp = jnp.asarray(float(spec.nfp), dtype=phi_2d.dtype)
    alpha = _alpha_scalar(spec, phi_2d)

    m_grid, n_grid, valid_mask = _mn_indices_2d(spec)

    # Effective per-mode phi multiplier: ``nfp·n - α``  shape (mmax+1, 2·nmax+1).
    phi_mult = nfp * n_grid - alpha
    # Effective per-mode theta multiplier: ``m``        shape (mmax+1, 2·nmax+1).

    # Angle tensor: shape (nphi, ntheta, mmax+1, 2·nmax+1)
    angle = (
        m_grid[None, None, :, :] * theta_2d[:, :, None, None]
        + phi_mult[None, None, :, :] * phi_2d[:, :, None, None]
    )

    cos_ang = jnp.cos(angle)
    sin_ang = jnp.sin(angle)

    masked_rhomn = spec.rhomn * valid_mask  # (mmax+1, 2·nmax+1)

    rho = jnp.einsum("ptmn,mn->pt", cos_ang, masked_rhomn)
    # ∂ρ/∂φ = -ρ_{m,n} · (nfp·n - α) · sin(...)
    drho_dphi = -jnp.einsum("ptmn,mn->pt", sin_ang, masked_rhomn * phi_mult)
    # ∂ρ/∂θ = -ρ_{m,n} · m · sin(...)
    drho_dtheta = -jnp.einsum("ptmn,mn->pt", sin_ang, masked_rhomn * m_grid)
    return {
        "rho": rho,
        "drho_dphi": drho_dphi,
        "drho_dtheta": drho_dtheta,
    }


# ---------------------------------------------------------------------------
# Public spec-driven evaluators
# ---------------------------------------------------------------------------


def surface_henneberg_gamma_from_spec(spec: SurfaceHennebergSpec) -> jax.Array:
    """Return γ as shape ``(nphi, ntheta, 3)``.

    Mirrors ``SurfaceHenneberg.gamma_impl`` (see
    ``surfacehenneberg.py:626-640``).
    """
    phi_1d, theta_1d, phi_2d, theta_2d = _phi_theta_radian_grid(spec)
    phi_modes = _phi_axis_modes(spec, phi_1d)
    rho_data = _rho_and_partials(spec, phi_2d, theta_2d)

    alpha = _alpha_scalar(spec, phi_2d)
    sin_aphi = jnp.sin(alpha * phi_2d)
    cos_aphi = jnp.cos(alpha * phi_2d)

    # Broadcast R0H/Z0H/b (nphi,) -> (nphi, ntheta)
    R0H_2d = phi_modes["R0H"][:, None]
    Z0H_2d = phi_modes["Z0H"][:, None]
    b_2d = phi_modes["b"][:, None]

    zeta = b_2d * jnp.sin(theta_2d - alpha * phi_2d)

    rho = rho_data["rho"]
    R = R0H_2d + rho * cos_aphi - zeta * sin_aphi
    Z = Z0H_2d + rho * sin_aphi + zeta * cos_aphi

    cos_phi = jnp.cos(phi_2d)
    sin_phi = jnp.sin(phi_2d)

    return jnp.stack((R * cos_phi, R * sin_phi, Z), axis=-1)


def surface_henneberg_gammadash1_from_spec(spec: SurfaceHennebergSpec) -> jax.Array:
    """Return ∂γ/∂(quadpoint_phi) as shape ``(nphi, ntheta, 3)``.

    Mirrors ``SurfaceHenneberg.gammadash1_impl`` (see
    ``surfacehenneberg.py:642-704``). The trailing ``2π`` multiplier
    converts the radian-space derivative to the SIMSOPT
    quadpoint-space convention.
    """
    phi_1d, theta_1d, phi_2d, theta_2d = _phi_theta_radian_grid(spec)
    phi_modes = _phi_axis_modes(spec, phi_1d)
    rho_data = _rho_and_partials(spec, phi_2d, theta_2d)

    alpha = _alpha_scalar(spec, phi_2d)
    two_pi = _two_pi_like(phi_2d)

    sin_aphi = jnp.sin(alpha * phi_2d)
    cos_aphi = jnp.cos(alpha * phi_2d)

    R0H_2d = phi_modes["R0H"][:, None]
    b_2d = phi_modes["b"][:, None]
    dR0H_2d = phi_modes["dR0H_dphi"][:, None]
    dZ0H_2d = phi_modes["dZ0H_dphi"][:, None]
    db_2d = phi_modes["db_dphi"][:, None]

    sin_tbar = jnp.sin(theta_2d - alpha * phi_2d)
    cos_tbar = jnp.cos(theta_2d - alpha * phi_2d)

    zeta = b_2d * sin_tbar
    # d/dφ ζ = (db/dφ)·sin(θ̄) + b·cos(θ̄)·(-α)
    dzeta_dphi = db_2d * sin_tbar - b_2d * cos_tbar * alpha

    rho = rho_data["rho"]
    drho_dphi = rho_data["drho_dphi"]

    R = R0H_2d + rho * cos_aphi - zeta * sin_aphi
    # dR/dφ = dR0H/dφ + dρ/dφ·cos(αφ) + ρ·(-α·sin(αφ))
    #         - dζ/dφ·sin(αφ) - ζ·(α·cos(αφ))
    dR_dphi = (
        dR0H_2d
        + drho_dphi * cos_aphi
        + rho * (-alpha * sin_aphi)
        - dzeta_dphi * sin_aphi
        - zeta * (alpha * cos_aphi)
    )
    # dZ/dφ = dZ0H/dφ + dρ/dφ·sin(αφ) + ρ·(α·cos(αφ))
    #         + dζ/dφ·cos(αφ) + ζ·(-α·sin(αφ))
    dZ_dphi = (
        dZ0H_2d
        + drho_dphi * sin_aphi
        + rho * (alpha * cos_aphi)
        + dzeta_dphi * cos_aphi
        - zeta * alpha * sin_aphi
    )

    cos_phi = jnp.cos(phi_2d)
    sin_phi = jnp.sin(phi_2d)

    # d/dφ (R cos φ) = (dR/dφ) cos φ - R sin φ
    # d/dφ (R sin φ) = (dR/dφ) sin φ + R cos φ
    dx_dphi = dR_dphi * cos_phi - R * sin_phi
    dy_dphi = dR_dphi * sin_phi + R * cos_phi
    dz_dphi = dZ_dphi

    return two_pi * jnp.stack((dx_dphi, dy_dphi, dz_dphi), axis=-1)


def surface_henneberg_gammadash2_from_spec(spec: SurfaceHennebergSpec) -> jax.Array:
    """Return ∂γ/∂(quadpoint_theta) as shape ``(nphi, ntheta, 3)``.

    Mirrors ``SurfaceHenneberg.gammadash2_impl`` (see
    ``surfacehenneberg.py:706-739``). R0H and Z0H do not depend on θ;
    only ρ and ζ contribute.
    """
    phi_1d, theta_1d, phi_2d, theta_2d = _phi_theta_radian_grid(spec)
    phi_modes = _phi_axis_modes(spec, phi_1d)
    rho_data = _rho_and_partials(spec, phi_2d, theta_2d)

    alpha = _alpha_scalar(spec, phi_2d)
    two_pi = _two_pi_like(phi_2d)

    sin_aphi = jnp.sin(alpha * phi_2d)
    cos_aphi = jnp.cos(alpha * phi_2d)

    b_2d = phi_modes["b"][:, None]
    cos_tbar = jnp.cos(theta_2d - alpha * phi_2d)

    # d/dθ ζ = b · cos(θ - α·φ)
    dzeta_dtheta = b_2d * cos_tbar
    drho_dtheta = rho_data["drho_dtheta"]

    dR_dtheta = drho_dtheta * cos_aphi - dzeta_dtheta * sin_aphi
    dZ_dtheta = drho_dtheta * sin_aphi + dzeta_dtheta * cos_aphi

    cos_phi = jnp.cos(phi_2d)
    sin_phi = jnp.sin(phi_2d)

    # R depends on θ only through ρ and ζ; γ = (R cos φ, R sin φ, Z),
    # and φ is independent of θ, so:
    # d/dθ (R cos φ) = (dR/dθ) cos φ
    # d/dθ (R sin φ) = (dR/dθ) sin φ
    dx_dtheta = dR_dtheta * cos_phi
    dy_dtheta = dR_dtheta * sin_phi
    dz_dtheta = dZ_dtheta

    return two_pi * jnp.stack((dx_dtheta, dy_dtheta, dz_dtheta), axis=-1)


def surface_henneberg_normal_from_spec(spec: SurfaceHennebergSpec) -> jax.Array:
    """Return the unnormalised surface normal as shape ``(nphi, ntheta, 3)``.

    Defined as ``∂γ/∂φ × ∂γ/∂θ`` to match
    ``simsopt.geo.surface.Surface.normal``.
    """
    gd1 = surface_henneberg_gammadash1_from_spec(spec)
    gd2 = surface_henneberg_gammadash2_from_spec(spec)
    return jnp.cross(gd1, gd2)


def surface_henneberg_unitnormal_from_spec(spec: SurfaceHennebergSpec) -> jax.Array:
    """Return the unit surface normal as shape ``(nphi, ntheta, 3)``."""
    normal = surface_henneberg_normal_from_spec(spec)
    norm = jnp.sqrt(jnp.sum(normal * normal, axis=-1, keepdims=True))
    return normal / norm


def surface_henneberg_area_from_spec(spec: SurfaceHennebergSpec) -> jax.Array:
    """Return the surface area scalar.

    Uses the shared ``surface_area`` helper from
    ``simsopt.geo.surface_fourier_jax``.
    """
    return surface_area(surface_henneberg_normal_from_spec(spec))


def surface_henneberg_volume_from_spec(spec: SurfaceHennebergSpec) -> jax.Array:
    """Return the enclosed volume scalar.

    Uses the shared ``surface_volume`` helper from
    ``simsopt.geo.surface_fourier_jax``.
    """
    return surface_volume(
        surface_henneberg_gamma_from_spec(spec),
        surface_henneberg_normal_from_spec(spec),
    )
