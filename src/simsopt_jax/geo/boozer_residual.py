"""
Pure JAX Boozer residual functions.

This module provides two layers:

**M1 primitives** — ``boozer_residual_scalar``, ``boozer_residual_grad``,
``boozer_residual_hessian`` operate on pre-computed B/xphi/xtheta arrays
and differentiate only through (iota, G).  Surface DOF derivatives are
zero because field data is treated as constant.

**Composed derivative pipeline** — ``boozer_penalty_composed``,
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

from simsopt_jax.core._device_scalars import staged_like as _staged_like
from simsopt_jax.core._math_utils import (
    as_runtime_value as _as_runtime_value,
    explicit_rsqrt as _explicit_rsqrt,
    require_runtime_dtype as _require_runtime_dtype,
)
from simsopt_jax.core.surface_rzfourier import (
    surface_rz_fourier_geometry_from_spec,
    surface_rz_fourier_spec_from_dofs,
)
from simsopt_jax.core.specs import SurfaceXYZTensorFourierSpec
from simsopt_jax.core.surface_fourier import (
    surface_xyz_tensor_fourier_gamma_from_spec,
    surface_xyz_tensor_fourier_gammadash1_from_spec,
    surface_xyz_tensor_fourier_gammadash2_from_spec,
)
from simsopt_jax.core.reductions import (
    pairwise_sum_axis,
    scalar_square_sum,
    validate_reduction_mode,
)
from simsopt_jax.field.biotsavart import grouped_biot_savart_B
from simsopt_jax.geo.surface_fourier import (
    surface_gamma_from_dofs,
    surface_gammadash1_from_dofs,
    surface_gammadash2_from_dofs,
    surface_xyzfourier_gamma_from_dofs,
    surface_xyzfourier_gammadash1_from_dofs,
    surface_xyzfourier_gammadash2_from_dofs,
)
from .label_constraints import compute_G_from_currents

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
    # Follow the caller's decision-vector dtype. Certificate enforcement lives
    # at the adjoint interface pins (boozer_surface: the fp64 tangent pin and
    # the fp64 decision-vector construction), not at this split.
    x_jax = jnp.asarray(x)
    _validate_decision_vector_tail(x_jax, optimize_G=optimize_G)
    return _split_decision_vector_lax(x_jax, optimize_G=optimize_G)


def _split_decision_vector_jvp_safe(x, *, optimize_G):
    x_jax = jnp.asarray(x)
    _validate_decision_vector_tail(x_jax, optimize_G=optimize_G)
    return _split_decision_vector_lax(x_jax, optimize_G=optimize_G)


def _split_decision_vector_for_mode(x, *, optimize_G, decision_split_mode):
    if decision_split_mode == "reverse":
        return _split_decision_vector(x, optimize_G=optimize_G)
    if decision_split_mode == "jvp":
        return _split_decision_vector_jvp_safe(x, optimize_G=optimize_G)
    raise ValueError(
        f"unknown decision_split_mode {decision_split_mode!r}; "
        "expected 'reverse' or 'jvp'."
    )


def _validate_decision_vector_tail(x_jax, *, optimize_G):
    tail_size = 2 if optimize_G else 1
    if int(x_jax.shape[0]) < tail_size:
        raise ValueError(
            f"decision vector length {int(x_jax.shape[0])} is too short for "
            f"optimize_G={optimize_G}; expected at least {tail_size} entries."
        )


def _split_decision_vector_lax(x_jax, *, optimize_G):
    surface_size = int(x_jax.shape[0]) - (2 if optimize_G else 1)
    if optimize_G:
        sdofs, iota, G = jnp.split(x_jax, (surface_size, surface_size + 1))
        iota = jnp.reshape(iota, ())
        G = jnp.reshape(G, ())
        return sdofs, iota, G
    sdofs, iota = jnp.split(x_jax, (surface_size,))
    iota = jnp.reshape(iota, ())
    return sdofs, iota, None


@jax.custom_vjp
def _split_decision_vector_with_G_vjp_safe(x_jax):
    return _split_decision_vector_lax(x_jax, optimize_G=True)


def _split_decision_vector_with_G_vjp_safe_fwd(x_jax):
    return _split_decision_vector_lax(x_jax, optimize_G=True), None


def _split_decision_vector_with_G_vjp_safe_bwd(_residual, cotangents):
    sdofs_ct, iota_ct, G_ct = cotangents
    return (jnp.concatenate((sdofs_ct, jnp.ravel(iota_ct), jnp.ravel(G_ct))),)


_split_decision_vector_with_G_vjp_safe.defvjp(
    _split_decision_vector_with_G_vjp_safe_fwd,
    _split_decision_vector_with_G_vjp_safe_bwd,
)


@jax.custom_vjp
def _split_decision_vector_without_G_vjp_safe(x_jax):
    sdofs, iota, _G = _split_decision_vector_lax(x_jax, optimize_G=False)
    return sdofs, iota


def _split_decision_vector_without_G_vjp_safe_fwd(x_jax):
    sdofs, iota, _G = _split_decision_vector_lax(x_jax, optimize_G=False)
    return (sdofs, iota), None


def _split_decision_vector_without_G_vjp_safe_bwd(_residual, cotangents):
    sdofs_ct, iota_ct = cotangents
    return (jnp.concatenate((sdofs_ct, jnp.ravel(iota_ct))),)


_split_decision_vector_without_G_vjp_safe.defvjp(
    _split_decision_vector_without_G_vjp_safe_fwd,
    _split_decision_vector_without_G_vjp_safe_bwd,
)


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


def _require_boozer_runtime_inputs(B, xphi, xtheta):
    _require_runtime_dtype("B", B)
    _require_runtime_dtype("xphi", xphi)
    _require_runtime_dtype("xtheta", xtheta)
    return jnp.asarray(B), jnp.asarray(xphi), jnp.asarray(xtheta)


def _boozer_inputs(B, xphi, xtheta, *, dtype=None):
    if dtype is None:
        return _require_boozer_runtime_inputs(B, xphi, xtheta)
    resolved_dtype = np.dtype(dtype)
    return (
        jnp.asarray(B, dtype=resolved_dtype),
        jnp.asarray(xphi, dtype=resolved_dtype),
        jnp.asarray(xtheta, dtype=resolved_dtype),
    )


def _boozer_scalar(value, *, reference, dtype=None):
    if dtype is None:
        return _as_runtime_value(value, reference=reference)
    # Guard-safe staging for the dtype-threaded lane: an eager ``jnp.asarray``
    # host literal is an implicit H2D put that ``transfer_guard("disallow")``
    # rejects (the fp64 branch above already stages explicitly).
    return _staged_like(reference, value, dtype=np.dtype(dtype))


def _dtype_scalar_like(reference, value):
    return _staged_like(reference, value)


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
    dtype=None,
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
    B, xphi, xtheta = _boozer_inputs(B, xphi, xtheta, dtype=dtype)
    G = _boozer_scalar(G, reference=B, dtype=dtype)
    iota = _boozer_scalar(iota, reference=B, dtype=dtype)
    nphi, ntheta, _ = B.shape
    num_res = _boozer_scalar(3 * nphi * ntheta, reference=B, dtype=dtype)
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
    return _boozer_scalar(0.5, reference=rtil, dtype=dtype) * square_sum / num_res


# ---------------------------------------------------------------------------
# M1 gradient / Hessian wrappers (iota/G only, surface DOFs are constants).
# For the full composed pipeline through surface DOFs, use the composed functions:
# boozer_penalty_grad_composed() and boozer_residual_jacobian_composed().
# ---------------------------------------------------------------------------


def _pack(surface_dofs, iota, G):
    """Pack (surface_dofs, iota, G) into a single vector for autodiff."""
    return jnp.concatenate(
        (
            jnp.ravel(jnp.asarray(surface_dofs)),
            jnp.ravel(jnp.asarray(iota)),
            jnp.ravel(jnp.asarray(G)),
        )
    )


def _unpack(x, nsurfdofs):
    """Unpack a single vector into (surface_dofs, iota, G)."""
    x_jax = _as_runtime_value(x, reference=x)
    surface_size = int(nsurfdofs)
    sdofs, iota, G = jnp.split(x_jax, (surface_size, surface_size + 1))
    return sdofs, jnp.reshape(iota, ()), jnp.reshape(G, ())


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
        dtype=B.dtype,
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
    B, xphi, xtheta = _require_boozer_runtime_inputs(B, xphi, xtheta)
    zero_surface_dofs = _as_runtime_value(
        np.zeros(nsurfdofs, dtype=np.float64),
        reference=B,
    )
    x0 = _pack(
        zero_surface_dofs,
        _as_runtime_value(iota, reference=B),
        _as_runtime_value(G, reference=B),
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
    B, xphi, xtheta = _require_boozer_runtime_inputs(B, xphi, xtheta)
    zero_surface_dofs = _as_runtime_value(
        np.zeros(nsurfdofs, dtype=np.float64),
        reference=B,
    )
    x0 = _pack(
        zero_surface_dofs,
        _as_runtime_value(iota, reference=B),
        _as_runtime_value(G, reference=B),
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
# Composed derivative path (DOFs → geometry → field → residual)
# ---------------------------------------------------------------------------


def boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB=False, dtype=None):
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
    B, xphi, xtheta = _boozer_inputs(B, xphi, xtheta, dtype=dtype)
    G = _boozer_scalar(G, reference=B, dtype=dtype)
    iota = _boozer_scalar(iota, reference=B, dtype=dtype)
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
    B = jnp.asarray(B)
    G = _dtype_scalar_like(B, G)
    iota = _dtype_scalar_like(B, iota)
    nphi, ntheta = B.shape[:2]
    nsurfdofs = dx_ds.shape[-1]
    grad_size = nsurfdofs + (2 if optimize_G else 1)
    num_res = _dtype_scalar_like(B, 3 * nphi * ntheta)
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
        one = _dtype_scalar_like(B, 1.0)
        rB2 = one / B2
        wij = jnp.sqrt(rB2) if weight_inv_modB else one

        tang0 = xphi[i, j, 0] + iota * xtheta[i, j, 0]
        tang1 = xphi[i, j, 1] + iota * xtheta[i, j, 1]
        tang2 = xphi[i, j, 2] + iota * xtheta[i, j, 2]

        res0 = G * B0 - B2 * tang0
        res1 = G * B1 - B2 * tang1
        res2 = G * B2_component - B2 * tang2

        rtil0 = res0 * wij
        rtil1 = res1 * wij
        rtil2 = res2 * wij
        value = value + _dtype_scalar_like(B, 0.5) * (
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
        dB2 = _dtype_scalar_like(B, 2.0) * (
            B0 * dB0 + B1 * dB1 + B2_component * dB2_component
        )

        dtang_factors = jnp.asarray([iota, _dtype_scalar_like(B, 1.0)], dtype=B.dtype)
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
            dmodB = _dtype_scalar_like(B, 0.5) * dB2 * wij
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
    """Return generic surface geometry helpers."""
    return (
        surface_gamma_from_dofs,
        surface_gammadash1_from_dofs,
        surface_gammadash2_from_dofs,
    )


def _get_surface_xyzfourier_fns():
    """Return ``SurfaceXYZFourier`` geometry helpers."""
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
    clamped_dims=(False, False, False),
    use_compute_dtype=False,
):
    """Evaluate gamma, gammadash1, gammadash2 from surface DOFs.

    Pure function suitable for JAX tracing.  Used by both the composed
    composed derivative pipeline and the Boozer solver.
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
            use_compute_dtype=use_compute_dtype,
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
        if use_compute_dtype:
            args = (*args, None, None)
            return (
                sgf(*args, use_compute_dtype=True),
                sg1f(*args, use_compute_dtype=True),
                sg2f(*args, use_compute_dtype=True),
            )
        return sgf(*args), sg1f(*args), sg2f(*args)

    if surface_kind == "xyztensorfourier":
        # The immutable spec jointly owns the compact coefficient mapping and
        # Cartesian boundary clamps. Keeping both in this one payload prevents
        # Boozer callers from reconstructing either representation independently.
        surface_spec = SurfaceXYZTensorFourierSpec(
            dofs=jnp.asarray(sdofs),
            quadpoints_phi=jnp.asarray(quadpoints_phi),
            quadpoints_theta=jnp.asarray(quadpoints_theta),
            scatter_indices=(
                jnp.asarray(scatter_indices, dtype=jnp.int32)
                if stellsym
                else jnp.zeros((0,), dtype=jnp.int32)
            ),
            nfp=int(nfp),
            stellsym=bool(stellsym),
            mpol=int(mpol),
            ntor=int(ntor),
            clamped_dims=tuple(bool(flag) for flag in clamped_dims),
        )
        return (
            surface_xyz_tensor_fourier_gamma_from_spec(
                surface_spec,
                use_compute_dtype=use_compute_dtype,
            ),
            surface_xyz_tensor_fourier_gammadash1_from_spec(
                surface_spec,
                use_compute_dtype=use_compute_dtype,
            ),
            surface_xyz_tensor_fourier_gammadash2_from_spec(
                surface_spec,
                use_compute_dtype=use_compute_dtype,
            ),
        )

    if surface_kind != "generic":
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
    return (
        sgf(*args, use_compute_dtype=use_compute_dtype),
        sg1f(*args, use_compute_dtype=use_compute_dtype),
        sg2f(*args, use_compute_dtype=use_compute_dtype),
    )


def _unpack_decision_vector(
    x,
    coil_arrays,
    optimize_G,
    *,
    decision_split_mode="reverse",
):
    """Unpack decision vector into (sdofs, iota, G).

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
    """
    sdofs, iota, G = _split_decision_vector_for_mode(
        x,
        optimize_G=optimize_G,
        decision_split_mode=decision_split_mode,
    )
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
    decision_split_mode="reverse",
):
    """Shared pipeline: unpack x → surface geometry → Biot-Savart field.

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.

    Returns (sdofs, iota, G, gamma, xphi, xtheta, B).
    """
    sdofs, iota, G = _unpack_decision_vector(
        x,
        coil_arrays,
        optimize_G,
        decision_split_mode=decision_split_mode,
    )

    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        use_compute_dtype=True,
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
    decision_split_mode="jvp",
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
        decision_split_mode: ``"jvp"`` keeps the scalar objective compatible
            with forward-over-reverse transforms such as ``jax.hessian``;
            reverse-mode-only wrappers select ``"reverse"`` explicitly.

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
        decision_split_mode=decision_split_mode,
    )
    return boozer_residual_scalar(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
        reduction_mode=reduction_mode,
        dtype=B.dtype,
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
    grad_kwargs = dict(kwargs)
    grad_kwargs.setdefault("decision_split_mode", "reverse")
    return jax.value_and_grad(boozer_penalty_composed)(x, **grad_kwargs)


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
    hvp_kwargs = dict(kwargs)
    hvp_kwargs["decision_split_mode"] = "jvp"
    grad_fn = jax.grad(lambda y: boozer_penalty_composed(y, **hvp_kwargs))
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
    decision_split_mode="reverse",
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
        decision_split_mode=decision_split_mode,
    )
    return boozer_residual_vector(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
        dtype=B.dtype,
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
    jvp_kwargs = dict(kwargs)
    jvp_kwargs["decision_split_mode"] = "jvp"
    residual_fn = lambda x_value: _boozer_residual_vector_composed(
        x_value, **jvp_kwargs
    )
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
    if n_res < n_dofs:
        residual_fn = lambda x_value: _boozer_residual_vector_composed(
            x_value, **kwargs
        )
        r, vjp_fn = jax.vjp(residual_fn, x_jax)
        row_indices = jnp.arange(n_res, dtype=jnp.int32)

        def vjp_row(row_index):
            cotangent = jnp.zeros((n_res,), dtype=r.dtype).at[row_index].set(1.0)
            (row,) = vjp_fn(cotangent)
            return row

        J = jax.lax.map(vjp_row, row_indices)
        return r, J

    jvp_kwargs = dict(kwargs)
    jvp_kwargs["decision_split_mode"] = "jvp"
    residual_fn = lambda x_value: _boozer_residual_vector_composed(
        x_value, **jvp_kwargs
    )
    r, jvp_fn = jax.linearize(residual_fn, x_jax)
    column_indices = jnp.arange(n_dofs, dtype=jnp.int32)

    def jvp_column(column_index):
        tangent = jnp.zeros((n_dofs,), dtype=x_jax.dtype).at[column_index].set(1.0)
        return jvp_fn(tangent)

    return r, jnp.swapaxes(jax.lax.map(jvp_column, column_indices), 0, 1)


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
            dtype=B.dtype,
        )

    _, vjp_fn = jax.vjp(residual_of_coils, coil_arrays)
    return vjp_fn(adjoint)
