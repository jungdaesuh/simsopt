"""REGCOIL / NESCOIL least-squares floor on a winding surface (offline diagnostic).

This is a self-contained numpy kernel — NO simsopt C++ winding-surface / REGCOIL
solver is used. It computes the theoretical minimum quadratic flux

    J = 0.5 * mean_{plasma grid} (B.n)^2 * |n|

achievable by ANY divergence-free surface current living on the winding surface,
i.e. the unrestricted REGCOIL/NESCOIL lower bound. Comparing it to the flux the
real (single-shared-current) banana coils achieve decomposes the residual into a
winding-surface-physics floor versus a coil-parameterization deficit.

Physics / convention contract (verified numerically against the live artifact;
see tests/geo/test_regcoil_floor.py and the run_regcoil_floor.py validation log):

* Surfaces are simsopt ``SurfaceRZFourier`` objects parametrized on
  (phi, theta) in [0, 1)^2. The geometric toroidal angle is 2*pi*phi and the
  poloidal angle is 2*pi*theta. ``gammadash1`` is d(Gamma)/d(phi_quad),
  ``gammadash2`` is d(Gamma)/d(theta_quad), and ``normal() == gammadash1 x
  gammadash2`` with ``|normal|`` the area Jacobian per unit (phi_quad, theta_quad)
  (surface area == mean(|normal|)). All three facts are exercised by tests.

* Current potential (units: Amperes):
      Phi(phi, theta) = Phi_single_valued(phi, theta) + G_amperes * phi + I_amperes * theta
  where ``G_amperes`` is the net poloidal current linking the torus
  (= num_tf_coils * tf_current_A, the secular TF current) and ``I_amperes`` is the
  net toroidal (secular) current (0 for the vacuum floor). phi, theta are the
  [0, 1) quadpoints, so G_amperes * phi == (G_amperes / 2pi) * (2pi*phi) is the
  standard NESCOIL secular term written against the geometric toroidal angle.

* The single-valued potential is expanded in a truncated stellarator-symmetric
  Fourier basis. With stellarator symmetry Phi is odd, so only the sine basis
  ``sin(2*pi*(m*theta - n*nfp*phi))`` is retained (m in [0, mpol], n in
  [-ntor, ntor], excluding the (m=0, n<=0) duplicates / the constant). Without
  stellarator symmetry both sin and cos are retained. The constant term carries
  no current and is always excluded.

* Surface current density (units: A/m):
      K = (1 / |N|) * ( dPhi/dtheta * e_phi  -  dPhi/dphi * e_theta )
  with e_phi = gammadash1, e_theta = gammadash2, N = normal(), |N| = |N|.
  In coordinate-free form this is the divergence-free contravariant current
  ``K = grad Phi x n_hat`` (NOT ``n_hat x grad Phi`` — that flips the sign). The
  least-squares floor is sign-invariant (it minimizes a quadratic in B.n), so the
  global orientation convention is immaterial to the reported floor; the order is
  documented here to match the verified code exactly.
  Verified: the secular term alone integrates to the prescribed net poloidal /
  toroidal currents, and its Biot-Savart field reproduces the analytic toroidal
  field mu0 * G_amperes / (2pi R) with the correct sign.

* Magnetic field of a surface current (SI, units: Tesla):
      B(x) = (mu0 / 4pi) * sum_grid (K(x') x (x - x')) / |x - x'|^3 * dA'
  with dA' = |N| / (nphi*ntheta) the quadrature area weight.

The least-squares floor solves min_c || A c - b ||_w^2 in the SquaredFlux
quadratic-flux metric, where column j of A is the plasma-grid (B.n) from unit
single-valued coefficient c_j and b is -(B_secular . n). The minimized residual,
reduced identically to ``simsopt.objectives.SquaredFlux`` with
``definition="quadratic flux"`` (0.5 * mean(Bn^2 * |n|), see
src/simsoptpp/integral_BdotN.cpp), is the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Vacuum permeability (SI). Single source of truth: banana_opt.current_contracts.
from banana_opt.current_contracts import MU0


@dataclass(frozen=True)
class SurfaceQuadrature:
    """Quadrature-grid geometry of a surface, in the simsopt (phi, theta) frame.

    gamma:            (nphi, ntheta, 3) Cartesian positions.
    gammadash1:       (nphi, ntheta, 3) d(gamma)/d(phi_quad) == e_phi.
    gammadash2:       (nphi, ntheta, 3) d(gamma)/d(theta_quad) == e_theta.
    normal:           (nphi, ntheta, 3) gammadash1 x gammadash2.
    abs_normal:       (nphi, ntheta) |normal|, the area Jacobian per unit (phi, theta).
    quadpoints_phi:   (nphi,) the surface's actual [0, 1) toroidal quadpoints.
    quadpoints_theta: (ntheta,) the surface's actual [0, 1) poloidal quadpoints.
    """

    gamma: NDArray[np.float64]
    gammadash1: NDArray[np.float64]
    gammadash2: NDArray[np.float64]
    normal: NDArray[np.float64]
    abs_normal: NDArray[np.float64]
    quadpoints_phi: NDArray[np.float64]
    quadpoints_theta: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.gamma.shape[0], self.gamma.shape[1])

    @property
    def n_quad(self) -> int:
        return int(self.gamma.shape[0] * self.gamma.shape[1])


def surface_quadrature(surface) -> SurfaceQuadrature:
    """Snapshot a SurfaceRZFourier's quadrature geometry into plain arrays.

    Pure read of the simsopt surface; no mutation. ``normal`` is asserted to equal
    ``gammadash1 x gammadash2`` so downstream code can rely on the contravariant
    current identity.
    """
    gamma = np.ascontiguousarray(surface.gamma(), dtype=float)
    g1 = np.ascontiguousarray(surface.gammadash1(), dtype=float)
    g2 = np.ascontiguousarray(surface.gammadash2(), dtype=float)
    normal = np.ascontiguousarray(surface.normal(), dtype=float)
    if not np.allclose(normal, np.cross(g1, g2), rtol=1e-9, atol=1e-12):
        raise ValueError(
            "Surface normal() is not gammadash1 x gammadash2; the contravariant "
            "current identity assumed by regcoil_floor does not hold for this surface."
        )
    abs_normal = np.linalg.norm(normal, axis=2)
    if not np.all(abs_normal > 0.0):
        raise ValueError("Surface has a degenerate (zero-area) quadrature point.")
    quadpoints_phi = np.ascontiguousarray(surface.quadpoints_phi, dtype=float)
    quadpoints_theta = np.ascontiguousarray(surface.quadpoints_theta, dtype=float)
    return SurfaceQuadrature(
        gamma=gamma,
        gammadash1=g1,
        gammadash2=g2,
        normal=normal,
        abs_normal=abs_normal,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )


@dataclass(frozen=True)
class CurrentPotentialBasis:
    """Single-valued current-potential Fourier basis on a winding surface.

    Each basis function f_j(phi, theta) is one trigonometric mode; the
    single-valued potential is Phi_sv = sum_j c_j f_j. ``dphi`` and ``dtheta`` are
    the per-mode (phi, theta) derivatives df_j/d(phi_quad), df_j/d(theta_quad),
    evaluated on the winding-surface grid.

    dphi:    (n_basis, nphi, ntheta) df_j/d(phi_quad).
    dtheta:  (n_basis, nphi, ntheta) df_j/d(theta_quad).
    modes:   (n_basis, 3) integer (m, n, parity) labels; parity 0 == sin, 1 == cos.
    """

    dphi: NDArray[np.float64]
    dtheta: NDArray[np.float64]
    modes: NDArray[np.int64]

    @property
    def n_basis(self) -> int:
        return int(self.dphi.shape[0])


def build_current_potential_basis(
    quad: SurfaceQuadrature,
    *,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
) -> CurrentPotentialBasis:
    """Build the truncated single-valued current-potential basis derivatives.

    Modes are ``f = sin/cos(2*pi*(m*theta - n*nfp*phi))`` for m in [0, mpol],
    n in [-ntor, ntor]. The (m=0, n<=0) sine modes and the (m=0, n=0) cosine mode
    are excluded (they are either duplicates of an n>0 mode or the current-free
    constant). With ``stellsym`` only the odd (sine) modes are kept. Returns the
    (phi, theta) derivatives of every retained mode on the winding-surface grid.
    """
    if mpol < 0 or ntor < 0:
        raise ValueError("mpol and ntor must be non-negative.")
    if nfp <= 0:
        raise ValueError("nfp must be positive.")
    nphi, ntheta = quad.shape
    # Use the surface's ACTUAL [0, 1) quadpoints rather than reconstructing a
    # uniform full-torus grid; this removes the latent assumption that the grid is
    # exactly arange(n)/n (uniform, full torus, offset-free).
    phi = quad.quadpoints_phi[:, None]
    theta = quad.quadpoints_theta[None, :]
    dphi_modes: list[NDArray[np.float64]] = []
    dtheta_modes: list[NDArray[np.float64]] = []
    labels: list[tuple[int, int, int]] = []
    for m in range(0, mpol + 1):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n <= 0:
                # n=0 -> constant (no current); n<0 duplicates the n>0 sine/cos pair.
                continue
            angle = 2.0 * np.pi * (m * theta - n * nfp * phi)
            d_angle_dtheta = 2.0 * np.pi * m
            d_angle_dphi = -2.0 * np.pi * n * nfp
            # sine mode: f = sin(angle)
            cos_angle = np.cos(angle)
            sin_basis_dphi = cos_angle * d_angle_dphi
            sin_basis_dtheta = cos_angle * d_angle_dtheta
            dphi_modes.append(np.broadcast_to(sin_basis_dphi, (nphi, ntheta)).copy())
            dtheta_modes.append(np.broadcast_to(sin_basis_dtheta, (nphi, ntheta)).copy())
            labels.append((m, n, 0))
            if not stellsym:
                # cosine mode: f = cos(angle)
                sin_angle = np.sin(angle)
                cos_basis_dphi = -sin_angle * d_angle_dphi
                cos_basis_dtheta = -sin_angle * d_angle_dtheta
                dphi_modes.append(
                    np.broadcast_to(cos_basis_dphi, (nphi, ntheta)).copy()
                )
                dtheta_modes.append(
                    np.broadcast_to(cos_basis_dtheta, (nphi, ntheta)).copy()
                )
                labels.append((m, n, 1))
    if not dphi_modes:
        raise ValueError(
            "Empty current-potential basis; increase --regcoil-mpol/--regcoil-ntor."
        )
    return CurrentPotentialBasis(
        dphi=np.stack(dphi_modes, axis=0),
        dtheta=np.stack(dtheta_modes, axis=0),
        modes=np.asarray(labels, dtype=np.int64),
    )


def surface_current_from_potential_derivatives(
    quad: SurfaceQuadrature,
    dphi: NDArray[np.float64],
    dtheta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Contravariant surface current K (A/m) from potential derivatives.

    K = (1 / |N|) * ( dPhi/dtheta * e_phi  -  dPhi/dphi * e_theta ).

    In coordinate-free form this is ``K = grad Phi x n_hat`` (the cross-product
    order is grad-Phi-first; ``n_hat x grad Phi`` would flip the sign). The floor
    is sign-invariant, so the overall orientation is immaterial to the result.

    ``dphi`` (dPhi/d(phi_quad)) and ``dtheta`` (dPhi/d(theta_quad)) are
    (..., nphi, ntheta) arrays; the returned K is (..., nphi, ntheta, 3). Leading
    axes (e.g. a basis index) broadcast.
    """
    inv_abs_normal = 1.0 / quad.abs_normal
    e_phi = quad.gammadash1
    e_theta = quad.gammadash2
    k = (
        dtheta[..., None] * e_phi - dphi[..., None] * e_theta
    ) * inv_abs_normal[..., None]
    return k


def biot_savart_normal_field(
    source_points: NDArray[np.float64],
    surface_current: NDArray[np.float64],
    area_weights: NDArray[np.float64],
    eval_points: NDArray[np.float64],
    eval_unit_normals: NDArray[np.float64],
    *,
    eval_chunk_size: int = 512,
) -> NDArray[np.float64]:
    """B.n (Tesla) at eval points from a surface current via Biot-Savart.

    B(x) = (mu0/4pi) * sum_src (K x (x - x')) / |x - x'|^3 * dA'.

    source_points:     (S, 3) winding-surface quadrature positions x'.
    surface_current:   (S, 3) single K, or (B, S, 3) a batch of B currents.
    area_weights:      (S,) dA' = |N|/(nphi*ntheta) at each source.
    eval_points:       (P, 3) plasma-grid positions x.
    eval_unit_normals: (P, 3) unit normals n_hat at each plasma point.
    eval_chunk_size:   plasma points processed per pass; bounds peak memory to
                       O(chunk * S) so production grids do not allocate a full
                       (P, S, 3) tensor.

    Returns (P,) for a single current or (B, P) for a batch of B.n_hat. The whole
    batch is contracted at once per plasma chunk via a single matmul against a
    batch-independent geometric kernel (no per-column loop).
    """
    single = surface_current.ndim == 2
    currents = surface_current[None] if single else surface_current  # (B, S, 3)
    n_batch, n_source = currents.shape[0], currents.shape[1]
    n_eval = eval_points.shape[0]
    prefactor = MU0 / (4.0 * np.pi)
    # Scalar-triple-product identity: cross(K, r).n_hat == K.(r x n_hat). Building
    # the batch-independent geometric kernel G_{p,s,:} = weighted_diff x n_hat once
    # per plasma chunk turns the field evaluation into a single (B, 3S) x (3S, c)
    # matmul, bounding peak memory to O(chunk * S) regardless of the batch size.
    currents_flat = currents.reshape(n_batch, n_source * 3)  # (B, 3S)
    out = np.empty((n_batch, n_eval), dtype=float)
    for start in range(0, n_eval, eval_chunk_size):
        stop = min(start + eval_chunk_size, n_eval)
        diff = eval_points[start:stop, None, :] - source_points[None, :, :]  # (c, S, 3)
        dist = np.linalg.norm(diff, axis=2)  # (c, S)
        if np.any(dist <= 0.0):
            raise ValueError(
                "Biot-Savart source and evaluation grids touch (zero separation); "
                "the winding surface must strictly enclose the plasma grid."
            )
        weighted_diff = diff * (area_weights[None, :] / dist**3)[:, :, None]  # (c, S, 3)
        geom_kernel = np.cross(
            weighted_diff, eval_unit_normals[start:stop, None, :]
        )  # (c, S, 3)
        geom_flat = geom_kernel.reshape(stop - start, n_source * 3)  # (c, 3S)
        out[:, start:stop] = prefactor * (currents_flat @ geom_flat.T)  # (B, c)
    return out[0] if single else out


def quadratic_flux_objective(
    bn: NDArray[np.float64], abs_normal: NDArray[np.float64]
) -> float:
    """SquaredFlux quadratic-flux reduction: 0.5 * mean(Bn^2 * |n|).

    Mirrors src/simsoptpp/integral_BdotN.cpp for definition="quadratic flux":
    numerator_sum = sum (B.n_hat)^2 * |n|, result = 0.5 * numerator_sum / (nphi*ntheta).
    ``bn`` and ``abs_normal`` are flattened-or-2d arrays of identical size; |n| is
    the area Jacobian (NOT the unit normal). This is mean over all grid points.
    """
    bn_flat = np.asarray(bn, dtype=float).reshape(-1)
    absn_flat = np.asarray(abs_normal, dtype=float).reshape(-1)
    if bn_flat.shape != absn_flat.shape:
        raise ValueError("bn and abs_normal must have the same number of points.")
    return 0.5 * float(np.mean(bn_flat**2 * absn_flat))


@dataclass(frozen=True)
class RegcoilFloorResult:
    """Result of a REGCOIL/NESCOIL least-squares floor computation.

    floor_field_objective:   the minimized 0.5*mean(Bn^2*|n|) (SquaredFlux units).
    secular_field_objective:  same reduction with all single-valued c == 0
                              (secular-only baseline; the c=0 consistency value).
    coefficients:             (n_basis,) optimal single-valued coefficients (A).
    n_basis:                  number of single-valued basis functions.
    inductance_cond:          condition number cond(A_w) of the weighted inductance
                              matrix (sigma_max / sigma_min); large => the
                              unrestricted floor is an ill-posed overfitting limit,
                              not a surface-quality discriminator.
    inductance_rank:          numerical rank of A_w (count of singular values above
                              the default lstsq tolerance); < n_basis flags a
                              rank-deficient (degenerate) inductance operator.
    """

    floor_field_objective: float
    secular_field_objective: float
    coefficients: NDArray[np.float64]
    n_basis: int
    inductance_cond: float
    inductance_rank: int


def compute_regcoil_floor(
    *,
    plasma_quad: SurfaceQuadrature,
    winding_quad: SurfaceQuadrature,
    basis: CurrentPotentialBasis,
    net_poloidal_current_amperes: float,
    net_toroidal_current_amperes: float,
    tikhonov_lambda: float = 0.0,
) -> RegcoilFloorResult:
    """Solve the unrestricted REGCOIL floor in the SquaredFlux quadratic-flux metric.

    Builds the secular B.n on the plasma grid from the prescribed net currents,
    each single-valued basis function's B.n response (the inductance matrix), and
    solves the weighted least-squares min_c ||A c - b||_w^2 + lambda ||c||^2 where
    the weight is the quadratic-flux area Jacobian so the minimized residual equals
    0.5*mean(Bn^2*|n|) exactly. lambda == 0 gives the true (unregularized) floor.

    The plasma and winding quadratures must already be on the SAME scaled surfaces
    the achieved comparison uses; this function does no scaling or I/O.
    """
    if tikhonov_lambda < 0.0:
        raise ValueError("tikhonov_lambda must be >= 0.")
    nphi_w, ntheta_w = winding_quad.shape
    source_points = winding_quad.gamma.reshape(-1, 3)
    area_weights = (winding_quad.abs_normal / (nphi_w * ntheta_w)).reshape(-1)

    eval_points = plasma_quad.gamma.reshape(-1, 3)
    unit_normals = (
        plasma_quad.normal / plasma_quad.abs_normal[..., None]
    ).reshape(-1, 3)
    plasma_abs_normal = plasma_quad.abs_normal.reshape(-1)

    # Secular current and its B.n on the plasma grid (the RHS to cancel).
    dphi_sec = np.full(winding_quad.shape, net_poloidal_current_amperes)
    dtheta_sec = np.full(winding_quad.shape, net_toroidal_current_amperes)
    k_sec = surface_current_from_potential_derivatives(
        winding_quad, dphi_sec, dtheta_sec
    ).reshape(-1, 3)
    bn_secular = biot_savart_normal_field(
        source_points, k_sec, area_weights, eval_points, unit_normals
    )  # (n_plasma,)

    # Single-valued basis B.n responses (the inductance matrix columns).
    k_basis = surface_current_from_potential_derivatives(
        winding_quad, basis.dphi, basis.dtheta
    ).reshape(basis.n_basis, -1, 3)
    bn_basis = biot_savart_normal_field(
        source_points, k_basis, area_weights, eval_points, unit_normals
    )  # (n_basis, n_plasma)

    # Weighted least squares so the minimized residual norm IS the SquaredFlux
    # reduction: J = 0.5 * mean(Bn^2 * |n|) = (0.5 / P) * sum_i (sqrt(|n_i|) * Bn_i)^2.
    # Embedding sqrt(|n_i|) into the rows of A and b makes lstsq minimize exactly
    # sum_i (sqrt(|n_i|) * (Bn_total)_i)^2, so J = (0.5 / P) * ||A_w c - b_w||^2.
    sqrt_w = np.sqrt(plasma_abs_normal)
    a_weighted = (bn_basis * sqrt_w[None, :]).T  # (n_plasma, n_basis)
    b_weighted = -bn_secular * sqrt_w  # cancel the secular field: A c ~ -Bn_secular

    # Conditioning of the (unregularized) inductance operator A_w. A large cond or a
    # rank < n_basis means the c=0 single-valued problem is rank-deficient / an
    # overfitting limit, so a near-zero unregularized floor is NOT a surface-quality
    # signal. Computed from the same A_w that lstsq factorizes, before augmentation.
    singular_values = np.linalg.svd(a_weighted, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
    inductance_cond = sigma_max / sigma_min if sigma_min > 0.0 else float("inf")
    # numpy lstsq default rcond=None uses max(M, N) * eps as the relative cutoff.
    rank_tol = max(a_weighted.shape) * np.finfo(float).eps * sigma_max
    inductance_rank = int(np.count_nonzero(singular_values > rank_tol))

    if tikhonov_lambda > 0.0:
        reg = np.sqrt(tikhonov_lambda) * np.eye(basis.n_basis)
        a_aug = np.vstack([a_weighted, reg])
        b_aug = np.concatenate([b_weighted, np.zeros(basis.n_basis)])
        coefficients, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
    else:
        coefficients, *_ = np.linalg.lstsq(a_weighted, b_weighted, rcond=None)

    bn_single_valued = coefficients @ bn_basis  # (n_plasma,)
    bn_total = bn_secular + bn_single_valued
    floor = quadratic_flux_objective(bn_total, plasma_abs_normal)
    secular_only = quadratic_flux_objective(bn_secular, plasma_abs_normal)
    return RegcoilFloorResult(
        floor_field_objective=floor,
        secular_field_objective=secular_only,
        coefficients=np.asarray(coefficients, dtype=float),
        n_basis=basis.n_basis,
        inductance_cond=inductance_cond,
        inductance_rank=inductance_rank,
    )


def net_poloidal_current_amperes(num_tf_coils: int, tf_current_amperes: float) -> float:
    """Net poloidal (secular TF) current linking the torus, in Amperes.

    G_amperes = num_tf_coils * tf_current_A. The companion poloidal-current
    constant G [T*m] is mu0 * G_amperes / (2*pi); this returns the raw Ampere
    linking current that drives the secular current potential Phi = G_amperes * phi.
    """
    return float(num_tf_coils) * float(tf_current_amperes)


def poloidal_current_constant_tesla_meter(
    num_tf_coils: int, tf_current_amperes: float
) -> float:
    """Poloidal-current constant G [T*m] = mu0 * (num_tf_coils * tf_current_A) / 2pi.

    This is the quantity reported as ``G_poloidal_Tm`` in the JSON sidecar; the
    secular current potential itself uses the raw Ampere linking current.
    """
    return MU0 * net_poloidal_current_amperes(num_tf_coils, tf_current_amperes) / (
        2.0 * np.pi
    )
