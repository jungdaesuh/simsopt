"""
Pure JAX Boozer residual functions.

This module provides two layers:

**M1 primitives** — ``boozer_residual_scalar``, ``boozer_residual_grad``,
``boozer_residual_hessian`` operate on pre-computed B/xphi/xtheta arrays
and differentiate only through (iota, G).  Surface DOF derivatives are
zero because field data is treated as constant.

**M3 composed pipeline** — ``boozer_penalty_composed``,
``boozer_penalty_grad_composed``, ``boozer_penalty_hvp_composed``,
``boozer_residual_jvp_composed``, ``boozer_residual_vjp_composed``,
``boozer_residual_jacobian_composed``, ``boozer_residual_coil_vjp`` trace
through the full
DOFs → surface geometry → Biot-Savart → residual chain via JAX autodiff,
replacing the C++ kernels ``sopp.boozer_residual_ds``,
``sopp.boozer_residual_ds2``, and ``sopp.boozer_dresidual_dc``.

The residual at each grid point is

.. math::

    \\tilde{r}_{ij} = w_{ij}\\bigl[G\\,\\mathbf B_{ij}
        - |\\mathbf B_{ij}|^2 (\\mathbf x_\\varphi + \\iota\\,\\mathbf x_\\theta)\\bigr]

with ``w = 1/|B|`` when *weight_inv_modB* is True, else ``w = 1``.
At exactly ``|B| = 0``, the weighted path intentionally exposes the
upstream singularity as non-finite output; no floor is inserted.

Defaults follow upstream SIMSOPT convention
(``simsopt/geo/surfaceobjectives.py`` and ``simsopt/geo/boozersurface.py``).
Low-level residual primitives (``boozer_residual_scalar``, ``_grad``,
``_hessian``, ``_vector``, ``_scalar_and_grad_cpu_ordered``) default to
``weight_inv_modB=False`` -- bare algebraic residual, matching upstream
``boozer_surface_residual`` / ``boozer_surface_residual_dB``. LS-context
wrappers (``boozer_penalty_composed``, ``boozer_residual_coil_vjp``) default
to ``True`` -- matching upstream ``boozer_penalty_constraints_vectorized`` and
``boozer_surface_dlsqgrad_dcoils_vjp``.

The scalar objective is

.. math::

    J = \\frac{1}{2 N}\\sum_{i,j} \\|\\tilde{r}_{ij}\\|^2

where ``N = 3 · nphi · ntheta``. This is the JAX scalar normalization
``1 / (3 · nphi · ntheta)``; the raw C++ symbol
``sopp.boozer_residual`` does not carry this normalization. The CPU
production path applies the same factor inline in ``boozersurface.py``.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from ..jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
    as_runtime_float64 as _as_runtime_float64,
    concat_jax_float64 as _concat_jax_float64,
    explicit_rsqrt as _explicit_rsqrt,
    require_float64_dtype as _require_float64_dtype,
)
from ..jax_core.surface_rzfourier import (
    surface_rz_fourier_geometry_from_spec,
    surface_rz_fourier_spec_from_dofs,
)
from ..jax_core.reductions import (
    pairwise_sum_axis,
    scalar_square_sum,
    validate_reduction_mode,
)
from ..field.biotsavart_jax import grouped_biot_savart_B
from .label_constraints_jax import compute_G_from_currents

_BOOZER_CPU_ORDERED_REDUCTION_MODE = "cpu_ordered"

__all__ = [
    "boozer_residual_scalar",
    "boozer_residual_scalar_and_grad_cpu_ordered",
    "boozer_residual_grad",
    "boozer_residual_hessian",
    "boozer_residual_vector",
    "boozer_penalty_composed",
    "boozer_penalty_grad_composed",
    "boozer_penalty_hvp_composed",
    "boozer_residual_jvp_composed",
    "boozer_residual_vjp_composed",
    "boozer_residual_jacobian_composed",
    "boozer_residual_coil_vjp",
]


def _split_decision_vector(x, *, optimize_G):
    x_jax = _as_jax_float64(x)
    tail_size = 2 if optimize_G else 1
    surface_size = int(x_jax.shape[0]) - tail_size
    if surface_size < 0:
        raise ValueError(
            f"decision vector length {int(x_jax.shape[0])} is too short for "
            f"optimize_G={optimize_G}; expected at least {tail_size} entries."
        )
    sdofs = jnp.take(x_jax, _as_jax_int32(np.arange(surface_size)), axis=0)
    iota = jnp.take(x_jax, _as_jax_int32(surface_size), axis=0)
    if optimize_G:
        G = jnp.take(x_jax, _as_jax_int32(surface_size + 1), axis=0)
        return sdofs, iota, G
    return sdofs, iota, None


def _inverse_modB(B2):
    """Return ``1 / |B|``; degenerate zero-field inputs surface as non-finite."""
    return _explicit_rsqrt(B2)


def _boozer_weighted_residual(G, iota, B, xphi, xtheta, weight_inv_modB):
    tang = xphi + iota * xtheta
    B2 = pairwise_sum_axis(B * B, axis=-1)
    residual = G * B - B2[..., None] * tang

    if weight_inv_modB:
        residual = _inverse_modB(B2)[..., None] * residual
    return residual


def _require_boozer_float64_inputs(B, xphi, xtheta):
    _require_float64_dtype("B", B)
    _require_float64_dtype("xphi", xphi)
    _require_float64_dtype("xtheta", xtheta)
    return jnp.asarray(B), jnp.asarray(xphi), jnp.asarray(xtheta)


def _cpu_ordered_boozer_square_sum(rtil):
    """Sum residual squares in the same point/component order as sopp."""
    nphi, ntheta = rtil.shape[:2]
    zero = jnp.sum(rtil, dtype=rtil.dtype) - jnp.sum(rtil, dtype=rtil.dtype)

    def phi_body(i, phi_acc):
        def theta_body(j, theta_acc):
            r0 = rtil[i, j, 0]
            r1 = rtil[i, j, 1]
            r2 = rtil[i, j, 2]
            return theta_acc + (r0 * r0 + r1 * r1 + r2 * r2)

        return lax.fori_loop(0, ntheta, theta_body, phi_acc)

    return lax.fori_loop(0, nphi, phi_body, zero)


def boozer_residual_scalar(
    G,
    iota,
    B,
    xphi,
    xtheta,
    weight_inv_modB=False,
    reduction_mode="default",
):
    """Boozer residual scalar objective (forward pass).

    Args:
        G:     scalar (Boozer G constant).
        iota:  scalar (rotational transform).
        B:     (nphi, ntheta, 3)  magnetic field on the surface.
        xphi:  (nphi, ntheta, 3)  surface tangent dγ/dφ.
        xtheta:(nphi, ntheta, 3)  surface tangent dγ/dθ.
        weight_inv_modB: if True, weight residual by 1/|B|.
        reduction_mode: ``"default"`` keeps the validated pairwise scalar
            accumulation, ``"strict_oracle"`` promotes the final scalar
            contraction to compensated summation for oracle investigations,
            and ``"cpu_ordered"`` mirrors the C++ point/component accumulation
            order for host-SciPy Boozer LS parity checks.

    Returns:
        J: scalar objective value.
    """
    B, xphi, xtheta = _require_boozer_float64_inputs(B, xphi, xtheta)
    G = _as_runtime_float64(G, reference=B)
    iota = _as_runtime_float64(iota, reference=B)
    nphi, ntheta, _ = B.shape
    num_res = _as_runtime_float64(3 * nphi * ntheta, reference=B)
    rtil = _boozer_weighted_residual(G, iota, B, xphi, xtheta, weight_inv_modB)
    if reduction_mode == _BOOZER_CPU_ORDERED_REDUCTION_MODE:
        square_sum = _cpu_ordered_boozer_square_sum(rtil)
    else:
        validate_reduction_mode(reduction_mode)
        square_sum = scalar_square_sum(
            rtil,
            reduction_mode=reduction_mode,
            default="pairwise",
        )
    return _as_runtime_float64(0.5, reference=rtil) * square_sum / num_res


# ---------------------------------------------------------------------------
# M1 gradient / Hessian wrappers (iota/G only, surface DOFs are constants).
# For the full composed pipeline through surface DOFs, use the M3 functions:
# boozer_penalty_grad_composed() and boozer_residual_jacobian_composed().
# ---------------------------------------------------------------------------


def _pack(surface_dofs, iota, G):
    """Pack (surface_dofs, iota, G) into a single vector for autodiff."""
    return _concat_jax_float64(surface_dofs, [iota, G])


def _unpack(x, nsurfdofs):
    """Unpack a single vector into (surface_dofs, iota, G)."""
    del nsurfdofs
    return _split_decision_vector(x, optimize_G=True)


def _boozer_objective_from_packed(
    x,
    nsurfdofs,
    B,
    xphi,
    xtheta,
    weight_inv_modB,
    reduction_mode,
):
    """Scalar objective as a function of the packed decision vector."""
    _, iota, G = _unpack(x, nsurfdofs)
    return boozer_residual_scalar(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
        reduction_mode=reduction_mode,
    )


def boozer_residual_grad(
    G,
    iota,
    B,
    xphi,
    xtheta,
    nsurfdofs,
    weight_inv_modB=False,
    reduction_mode="default",
):
    """Gradient of the Boozer residual w.r.t. [surface_dofs, iota, G].

    Surface DOF gradient entries are zero because B, xphi, xtheta are
    treated as constants (not differentiated through surface evaluation).
    Only iota and G entries are non-trivial.

    For the full composed pipeline (DOFs → geometry → field → residual),
    use :func:`boozer_penalty_grad_composed` instead.

    Returns:
        grad: (nsurfdofs + 2,) gradient vector.
    """
    zero_surface_dofs = _as_runtime_float64(
        np.zeros(nsurfdofs, dtype=np.float64),
        reference=B,
    )
    x0 = _pack(
        zero_surface_dofs,
        _as_runtime_float64(iota, reference=B),
        _as_runtime_float64(G, reference=B),
    )
    grad_fn = jax.grad(
        lambda x: _boozer_objective_from_packed(
            x,
            nsurfdofs,
            B,
            xphi,
            xtheta,
            weight_inv_modB,
            reduction_mode,
        )
    )
    return grad_fn(x0)


def boozer_residual_hessian(
    G,
    iota,
    B,
    xphi,
    xtheta,
    nsurfdofs,
    weight_inv_modB=False,
    reduction_mode="default",
):
    """Hessian of the Boozer residual w.r.t. [surface_dofs, iota, G].

    Surface DOF Hessian blocks are zero because B, xphi, xtheta are
    treated as constants.  For the full composed pipeline, use
    ``jax.hessian(boozer_penalty_composed)`` instead.

    Returns:
        H: (nsurfdofs + 2, nsurfdofs + 2) Hessian matrix.
    """
    zero_surface_dofs = _as_runtime_float64(
        np.zeros(nsurfdofs, dtype=np.float64),
        reference=B,
    )
    x0 = _pack(
        zero_surface_dofs,
        _as_runtime_float64(iota, reference=B),
        _as_runtime_float64(G, reference=B),
    )
    hess_fn = jax.hessian(
        lambda x: _boozer_objective_from_packed(
            x,
            nsurfdofs,
            B,
            xphi,
            xtheta,
            weight_inv_modB,
            reduction_mode,
        )
    )
    return hess_fn(x0)


# ---------------------------------------------------------------------------
# M3: Composed derivative path (DOFs → geometry → field → residual)
# ---------------------------------------------------------------------------


def boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB=False):
    """Boozer residual vector (not the scalar 0.5||r||²/N).

    Returns the residual at each grid point, flattened.

    Args:
        G, iota: Boozer parameters.
        B:      (nphi, ntheta, 3) magnetic field on the surface.
        xphi:   (nphi, ntheta, 3) toroidal tangent.
        xtheta: (nphi, ntheta, 3) poloidal tangent.
        weight_inv_modB: weight by 1/|B| if True.

    Returns:
        (nphi*ntheta*3,) flattened residual vector.
    """
    G = _as_runtime_float64(G, reference=B)
    iota = _as_runtime_float64(iota, reference=B)
    return _boozer_weighted_residual(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB,
    ).ravel()


def boozer_residual_scalar_and_grad_cpu_ordered(
    G,
    iota,
    B,
    dB_dX,
    xphi,
    xtheta,
    dx_ds,
    dxphi_ds,
    dxtheta_ds,
    *,
    optimize_G,
    weight_inv_modB=False,
):
    """CPU-ordered Boozer LS scalar and first derivative.

    Mirrors ``sopp.boozer_residual_ds`` for the scalarized first-derivative
    path: each ``(phi, theta)`` point contributes component triples in order,
    and the gradient is accumulated in the same point order before the final
    ``num_res`` normalization.

    ``dB_dX`` has shape ``(nphi, ntheta, 3, 3)`` with
    ``dB_dX[i, j, k, m] = ∂_k B_m(x[i, j])``. After flattening the Boozer grid,
    this is the same derivative-direction-first convention documented as
    ``dB_by_dX[p, j, l]`` in ``CLAUDE.md``.
    """
    G = _as_runtime_float64(G, reference=B)
    iota = _as_runtime_float64(iota, reference=B)
    nphi, ntheta = B.shape[:2]
    nsurfdofs = dx_ds.shape[-1]
    grad_size = nsurfdofs + (2 if optimize_G else 1)
    num_res = _as_runtime_float64(3 * nphi * ntheta, reference=B)
    zero = jnp.sum(B, dtype=B.dtype) - jnp.sum(B, dtype=B.dtype)
    grad0 = jnp.zeros((grad_size,), dtype=B.dtype)

    def point_body(flat_index, state):
        value, grad = state
        i = flat_index // ntheta
        j = flat_index - i * ntheta

        B0 = B[i, j, 0]
        B1 = B[i, j, 1]
        B2_component = B[i, j, 2]
        B2 = B0 * B0 + B1 * B1 + B2_component * B2_component
        rB2 = _as_runtime_float64(1.0, reference=B) / B2
        wij = (
            jnp.sqrt(rB2) if weight_inv_modB else _as_runtime_float64(1.0, reference=B)
        )

        tang0 = xphi[i, j, 0] + iota * xtheta[i, j, 0]
        tang1 = xphi[i, j, 1] + iota * xtheta[i, j, 1]
        tang2 = xphi[i, j, 2] + iota * xtheta[i, j, 2]

        res0 = G * B0 - B2 * tang0
        res1 = G * B1 - B2 * tang1
        res2 = G * B2_component - B2 * tang2

        rtil0 = res0 * wij
        rtil1 = res1 * wij
        rtil2 = res2 * wij
        value = value + _as_runtime_float64(0.5, reference=B) * (
            rtil0 * rtil0 + rtil1 * rtil1 + rtil2 * rtil2
        )

        dx0 = dx_ds[i, j, 0, :]
        dx1 = dx_ds[i, j, 1, :]
        dx2 = dx_ds[i, j, 2, :]
        dB0 = dB_dX[i, j, 0, 0] * dx0 + (
            dB_dX[i, j, 1, 0] * dx1 + dB_dX[i, j, 2, 0] * dx2
        )
        dB1 = (
            dB_dX[i, j, 0, 1] * dx0 + dB_dX[i, j, 1, 1] * dx1 + dB_dX[i, j, 2, 1] * dx2
        )
        dB2_component = (
            dB_dX[i, j, 0, 2] * dx0 + dB_dX[i, j, 1, 2] * dx1 + dB_dX[i, j, 2, 2] * dx2
        )
        dB2 = _as_runtime_float64(2.0, reference=B) * (
            B0 * dB0 + B1 * dB1 + B2_component * dB2_component
        )

        dtang_factors = jnp.asarray(
            [iota, _as_runtime_float64(1.0, reference=B)], dtype=B.dtype
        )
        dtang0 = jnp.tensordot(
            dtang_factors,
            jnp.stack((dxtheta_ds[i, j, 0, :], dxphi_ds[i, j, 0, :])),
            axes=((0,), (0,)),
        )
        dtang1 = jnp.tensordot(
            dtang_factors,
            jnp.stack((dxtheta_ds[i, j, 1, :], dxphi_ds[i, j, 1, :])),
            axes=((0,), (0,)),
        )
        dtang2 = jnp.tensordot(
            dtang_factors,
            jnp.stack((dxtheta_ds[i, j, 2, :], dxphi_ds[i, j, 2, :])),
            axes=((0,), (0,)),
        )

        dres0 = G * dB0 - (dB2 * tang0 + B2 * dtang0)
        dres1 = G * dB1 - (dB2 * tang1 + B2 * dtang1)
        dres2 = G * dB2_component - (dB2 * tang2 + B2 * dtang2)

        if weight_inv_modB:
            dmodB = _as_runtime_float64(0.5, reference=B) * dB2 * wij
            dw = -dmodB * rB2
        else:
            dw = jnp.zeros_like(dB2)
        drtil0 = dres0 * wij + dw * res0
        drtil1 = dres1 * wij + dw * res1
        drtil2 = dres2 * wij + dw * res2
        surface_grad = rtil0 * drtil0 + (rtil1 * drtil1 + rtil2 * drtil2)
        grad = grad.at[:nsurfdofs].add(surface_grad)

        dres0_iota = -B2 * xtheta[i, j, 0]
        dres1_iota = -B2 * xtheta[i, j, 1]
        dres2_iota = -B2 * xtheta[i, j, 2]
        iota_grad = (
            rtil0 * dres0_iota * wij
            + rtil1 * dres1_iota * wij
            + rtil2 * dres2_iota * wij
        )
        grad = grad.at[nsurfdofs].add(iota_grad)

        if optimize_G:
            G_grad = rtil0 * wij * B0 + rtil1 * wij * B1 + rtil2 * wij * B2_component
            grad = grad.at[nsurfdofs + 1].add(G_grad)

        return value, grad

    value, grad = lax.fori_loop(0, nphi * ntheta, point_body, (zero, grad0))
    return value / num_res, grad / num_res


def _get_surface_fns():
    """Lazily import generic surface geometry helpers."""
    from simsopt.geo.surface_fourier_jax import (
        surface_gamma_from_dofs,
        surface_gammadash1_from_dofs,
        surface_gammadash2_from_dofs,
    )

    return (
        surface_gamma_from_dofs,
        surface_gammadash1_from_dofs,
        surface_gammadash2_from_dofs,
    )


def _get_surface_xyzfourier_fns():
    """Lazily import ``SurfaceXYZFourier`` geometry helpers."""
    from simsopt.geo.surface_fourier_jax import (
        surface_xyzfourier_gamma_from_dofs,
        surface_xyzfourier_gammadash1_from_dofs,
        surface_xyzfourier_gammadash2_from_dofs,
    )

    return (
        surface_xyzfourier_gamma_from_dofs,
        surface_xyzfourier_gammadash1_from_dofs,
        surface_xyzfourier_gammadash2_from_dofs,
    )


def _surface_geometry_from_dofs(
    sdofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    surface_kind="generic",
):
    """Evaluate gamma, gammadash1, gammadash2 from surface DOFs.

    Pure function suitable for JAX tracing.  Used by both the composed
    M3 pipeline and the M4 solver (``boozersurface_jax.py``).
    """
    if surface_kind == "rzfourier":
        del scatter_indices
        surface_spec = surface_rz_fourier_spec_from_dofs(
            sdofs,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
        )
        return surface_rz_fourier_geometry_from_spec(surface_spec)

    if surface_kind == "xyzfourier":
        sgf, sg1f, sg2f = _get_surface_xyzfourier_fns()
        args = (
            sdofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
        )
        return sgf(*args), sg1f(*args), sg2f(*args)

    if surface_kind not in {"generic", "xyztensorfourier"}:
        raise ValueError(f"Unsupported Boozer JAX surface_kind {surface_kind!r}.")

    sgf, sg1f, sg2f = _get_surface_fns()
    args = (
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
    )
    return sgf(*args), sg1f(*args), sg2f(*args)


def _unpack_decision_vector(x, coil_arrays, optimize_G):
    """Unpack decision vector into (sdofs, iota, G).

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
    """
    sdofs, iota, G = _split_decision_vector(x, optimize_G=optimize_G)
    if optimize_G:
        return sdofs, iota, G
    all_currents = jnp.concatenate([c for _, _, c in coil_arrays])
    return sdofs, iota, compute_G_from_currents(all_currents)


def _composed_pipeline(
    x,
    *,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    optimize_G,
):
    """Shared pipeline: unpack x → surface geometry → Biot-Savart field.

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.

    Returns (sdofs, iota, G, gamma, xphi, xtheta, B).
    """
    sdofs, iota, G = _unpack_decision_vector(x, coil_arrays, optimize_G)

    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
    )

    B = grouped_biot_savart_B(gamma.reshape(-1, 3), coil_arrays)
    B = B.reshape(gamma.shape)

    return sdofs, iota, G, gamma, xphi, xtheta, B


def boozer_penalty_composed(
    x,
    *,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    optimize_G,
    weight_inv_modB=True,
    reduction_mode="default",
):
    """Composed scalar penalty objective: DOFs → geometry → field → residual → scalar.

    The decision vector is ``x = [surface_dofs, iota]`` (optimize_G=False)
    or ``x = [surface_dofs, iota, G]`` (optimize_G=True). In
    ``optimize_G=False`` mode this function derives ``G`` from the supplied
    coil currents with ``compute_G_from_currents``; pinning an unrelated
    user-supplied fixed ``G`` is not part of this API.

    Args:
        x: (n,) flat decision vector.
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
        quadpoints_phi, quadpoints_theta: quadrature grids.
        mpol, ntor, nfp: surface resolution.
        stellsym: stellarator symmetry flag.
        scatter_indices: stellsym DOF scatter indices (or None).
        optimize_G: whether G is in the decision vector.
        weight_inv_modB: weight residual by 1/|B|.
        reduction_mode: ``"default"`` keeps the validated pairwise scalar
            accumulation, ``"strict_oracle"`` enables the dedicated
            compensated scalar contraction for oracle investigations, and
            ``"cpu_ordered"`` mirrors the C++ point/component accumulation
            order for host-SciPy Boozer LS parity checks.

    Returns:
        Scalar objective value.
    """
    _, iota, G, _, xphi, xtheta, B = _composed_pipeline(
        x,
        coil_arrays=coil_arrays,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        optimize_G=optimize_G,
    )
    return boozer_residual_scalar(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
        reduction_mode=reduction_mode,
    )


def boozer_penalty_grad_composed(x, **kwargs):
    """VJP-based gradient of the composed penalty objective.

    Uses reverse-mode autodiff through the full pipeline:
    DOFs → surface geometry → Biot-Savart → residual → scalar.

    Args:
        x: (n,) flat decision vector.
        **kwargs: forwarded to :func:`boozer_penalty_composed`.

    Returns:
        (val, grad): scalar objective value and (n,) gradient vector.
    """
    return jax.value_and_grad(boozer_penalty_composed)(x, **kwargs)


def boozer_penalty_hvp_composed(x, v, **kwargs):
    """Hessian-vector product of the composed penalty objective.

    Computes ``H @ v`` for ``H = hessian(boozer_penalty_composed)(x)`` using
    forward-over-reverse autodiff. This keeps callers on the scalar composed
    objective and avoids materializing the dense Hessian when only a product is
    needed.

    Args:
        x: (n,) flat decision vector.
        v: (n,) tangent vector.
        **kwargs: forwarded to :func:`boozer_penalty_composed`.

    Returns:
        (n,) Hessian-vector product.
    """
    x_jax = jnp.asarray(x)
    v_jax = jnp.asarray(v)
    grad_fn = jax.grad(lambda y: boozer_penalty_composed(y, **kwargs))
    return jax.jvp(grad_fn, (x_jax,), (v_jax,))[1]


def _boozer_residual_vector_composed(
    x,
    *,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    optimize_G=True,
    weight_inv_modB=False,
):
    """Composed residual vector: DOFs → geometry → field → residual vector.

    The decision vector is ``x = [surface_dofs, iota, G]`` (optimize_G=True,
    default for BoozerExact) or ``x = [surface_dofs, iota]``
    (optimize_G=False). In ``optimize_G=False`` mode ``G`` is derived from
    coil currents; a separate user-fixed ``G`` is intentionally unsupported.

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.

    Returns:
        (nphi*ntheta*3,) flattened residual vector.
    """
    _, iota, G, _, xphi, xtheta, B = _composed_pipeline(
        x,
        coil_arrays=coil_arrays,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        optimize_G=optimize_G,
    )
    return boozer_residual_vector(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
    )


def boozer_residual_jvp_composed(x, v, **kwargs):
    """JVP of the composed residual vector without building a dense Jacobian.

    Computes ``J @ v`` where ``J = jacobian(residual)(x)`` and ``residual`` is
    :func:`_boozer_residual_vector_composed`.

    Args:
        x: (n,) flat decision vector.
        v: (n,) tangent vector.
        **kwargs: forwarded to :func:`_boozer_residual_vector_composed`.

    Returns:
        (r, Jv): residual vector and Jacobian-vector product.
    """
    x_jax = jnp.asarray(x)
    v_jax = jnp.asarray(v)
    residual_fn = lambda x_value: _boozer_residual_vector_composed(x_value, **kwargs)
    return jax.jvp(residual_fn, (x_jax,), (v_jax,))


def boozer_residual_vjp_composed(x, cotangent, **kwargs):
    """VJP of the composed residual vector without building a dense Jacobian.

    Computes ``J.T @ cotangent`` where
    ``J = jacobian(residual)(x)`` and ``residual`` is
    :func:`_boozer_residual_vector_composed`.

    Args:
        x: (n,) flat decision vector.
        cotangent: (n_res,) residual-space cotangent vector.
        **kwargs: forwarded to :func:`_boozer_residual_vector_composed`.

    Returns:
        (r, J.T @ cotangent): residual vector and vector-Jacobian product.
    """
    x_jax = jnp.asarray(x)
    cotangent_jax = jnp.asarray(cotangent)
    residual_fn = lambda x_value: _boozer_residual_vector_composed(x_value, **kwargs)
    r, vjp_fn = jax.vjp(residual_fn, x_jax)
    (product,) = vjp_fn(cotangent_jax)
    return r, product


def boozer_residual_jacobian_composed(
    x,
    **kwargs,
):
    """Explicit Jacobian of the composed residual vector.

    Computes the full Jacobian matrix ``J[i,k] = ∂r_i/∂x_k`` where ``r`` is
    the residual vector and ``x = [surface_dofs, iota, G]``
    (optimize_G=True) or ``x = [surface_dofs, iota]`` (optimize_G=False).
    The implementation uses reverse-mode ``jax.vjp`` when the residual vector
    is smaller than the decision vector, otherwise it uses forward-mode
    ``jax.linearize``; both paths are batched with ``jax.vmap``. When
    ``optimize_G=False``, the composed residual derives ``G`` from coil
    currents rather than accepting a pinned user-supplied ``G``.

    This replaces the hand-coded C++ chain:
    ``sopp.boozer_dresidual_dc`` + ``dgamma_by_dcoeff`` + ``dB_by_dX``.

    Args:
        x: (n,) flat decision vector.
        **kwargs: forwarded to :func:`_boozer_residual_vector_composed`.

    Returns:
        (r, J): residual vector (n_res,) and Jacobian (n_res, n).
    """
    x_jax = jnp.asarray(x)
    n_dofs = int(x_jax.shape[0])
    n_res = (
        3
        * int(np.shape(kwargs["quadpoints_phi"])[0])
        * int(np.shape(kwargs["quadpoints_theta"])[0])
    )
    residual_fn = lambda x_value: _boozer_residual_vector_composed(x_value, **kwargs)

    if n_res < n_dofs:
        r, vjp_fn = jax.vjp(residual_fn, x_jax)
        cotangent_basis = jnp.eye(n_res, dtype=r.dtype)
        (J,) = jax.vmap(vjp_fn)(cotangent_basis)
        return r, J

    r, jvp_fn = jax.linearize(residual_fn, x_jax)
    tangent_basis = jnp.eye(n_dofs, dtype=x_jax.dtype)
    return r, jnp.swapaxes(jax.vmap(jvp_fn)(tangent_basis), 0, 1)


def boozer_residual_coil_vjp(
    adjoint,
    *,
    gamma,
    xphi,
    xtheta,
    coil_arrays,
    iota,
    G,
    weight_inv_modB=True,
):
    """VJP of Boozer residual w.r.t. coil parameters (public derivative helper).

    Given an adjoint vector (from the outer optimization), computes
    sensitivities of ``adjoint^T @ r`` w.r.t. coil geometry and currents
    via reverse-mode autodiff through Biot-Savart.

    Production exact solves route through the operator-backed adjoint path;
    this helper remains as the public/test derivative surface for direct coil
    residual VJP checks.

    This replaces the CPU chain:
    ``boozer_surface_residual_dB()`` → ``B_vjp()`` →
    ``sopp.biot_savart_vjp_graph()``.

    The surface geometry (gamma, xphi, xtheta) is held fixed — this
    function computes how the residual changes when the magnetic field
    changes due to coil parameter variations.

    Args:
        adjoint: (nphi*ntheta*3,) adjoint vector from outer solve.
        gamma:   (nphi, ntheta, 3) fixed surface positions.
        xphi:    (nphi, ntheta, 3) fixed toroidal tangent.
        xtheta:  (nphi, ntheta, 3) fixed poloidal tangent.
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
        iota: rotational transform (scalar).
        G: Boozer G constant (scalar).
        weight_inv_modB: weight residual by 1/|B|.

    Returns:
        ``(d_coil_arrays,)`` — 1-tuple of grouped cotangent list matching
        the ``coil_arrays`` pytree structure.
    """
    nphi, ntheta = gamma.shape[:2]
    expected = nphi * ntheta * 3
    if adjoint.shape != (expected,):
        raise ValueError(
            f"adjoint shape {adjoint.shape} != expected ({expected},) "
            f"for nphi={nphi}, ntheta={ntheta}"
        )

    def residual_of_coils(ca):
        points = gamma.reshape(-1, 3)
        B = grouped_biot_savart_B(points, ca)
        B = B.reshape(nphi, ntheta, 3)
        return boozer_residual_vector(
            G,
            iota,
            B,
            xphi,
            xtheta,
            weight_inv_modB=weight_inv_modB,
        )

    _, vjp_fn = jax.vjp(residual_of_coils, coil_arrays)
    return vjp_fn(adjoint)
