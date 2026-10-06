"""Derived rounding bound of the native-boozer FIRST-stage penalty objective, value and gradient.

PLAN.md amendment 5 part 2, B2 (registered 2026-09-29 17:02 EDT, before any formal run).  The
first stage of ``2_Intermediate/boozer.py`` minimizes, over ``x = (surface dofs, iota, G)``,

    F(x) = (1/N) sum_p 1/2 |r_p|^2 + 1/2 cw (A(x) - A0)^2 + 1/2 cw z00(x)^2,
    r_p = w_p (G B_p - |B_p|^2 (x_phi + iota x_theta)),  w_p = 1 / |B_p|,  N = 3 nphi ntheta,

evaluated by the native lane (``BoozerSurface.boozer_penalty_constraints_vectorized``: C++
``boozer_residual_ds``, ``BiotSavart``, ``SurfaceXYZTensorFourier``, ``Area``) and by the JAX
lane (the value and gradient of ``BoozerSurfaceJAX._make_penalty_objective_with``, the function
its L-BFGS-B minimizes).  :func:`first_stage_objective_bound` returns ``F`` at one state as a
:class:`forward_roundoff_bound.Bounded` whose bookkeeping bounds how far EITHER lane may round
away from the exact value and gradient.  No constant is fitted to a measured gap.

Traced op by op in upstream's form: Biot-Savart, the residual, the weight, the label and the
sum.  Injected with derived counts (units of ``u = 2**-53``, first order):

* Surface ``gamma``, ``x_phi``, ``x_theta`` and their dof Jacobians.  Every lane form is a
  rearrangement of the monomials ``kappa c v_n(phi) w_m(theta) R(phi)`` (``kappa`` an integer
  or ``2 pi``, ``R`` the rotation's cos/sin), so ``A = sum |monomials|`` is form-invariant and
  an element is within ``K A + T`` of exact.  ``K`` = products per monomial in the longer lane
  form (C++ ``surfacexyztensorfourier.h``; JAX ``surface_fourier_kernels.py`` matmuls) plus
  ``M - 1`` additions, ``M`` the element's monomials: gamma xy ``3``, z ``2``; x_phi xy ``5``,
  z ``4``; x_theta xy ``5``, z ``4``.  Jacobian entries: ``2, 1, 5, 3, 4, 3``.
* Coil ``gamma``/``gamma'`` of the base ``CurveXYZFourier`` curves, ``K = 1 + 2 order`` and
  ``2 + 2 order``, rotated by the shared ``rotmat`` (traced).
* Trig factors: ``T_f = r |arg| |f'(arg)| + 2 eps |f(arg)|`` with ``r = 2`` for ``k 2pi q``, ``1``
  for the rotation angle, ``0`` at an exact zero argument; ``eps`` per lane in :data:`TRIG_ULP`.
* Biot-Savart per point, traced with the three point coordinates as components and chained
  through the point's Jacobian (:func:`forward_roundoff_bound.compose`); ``1/r^3`` is the
  envelope of the native ``(1/sqrt r2)^3`` and the JAX ``rsqrt(r2) / r2``; the native closed-form
  ``dB/dX`` (``biot_savart_impl.h:74-96``) is charged its own traced error plus the fma into
  ``dB/dc``.
* ``w``: envelope of ``sqrt(1/B2)`` and ``rsqrt(B2)``; ``|n|``: envelope of the norm and the nested
  ``jnp.hypot`` (+4 roundings); the label term scale 50 with one extra rounding.

Library constants (established on the campaign builds simsoptpp 0149cc25, glibc 2.43, jax/jaxlib 0.10.0, CUDA 12.9;
the test re-establishes the trig constants for the build that runs, R4):
sin/cos glibc and XLA CPU 0.55 ulp (glibc ``s_sin.c`` "~0.55 ULP"); XLA GPU libdevice 2.5 ulp
(CUDA 12.9 Programming Guide Table 19: 2 ulp from the correctly rounded result); sqrt, divide,
+ - x correctly rounded; rsqrt charged 3 u (XLA CPU: vrsqrt14pd plus two Newton steps, derived
<= 2.43 u; XLA GPU: CUDA Table 19 lists 1 ulp but its section 17.1 says NOT guaranteed).  The CPU
pair is therefore a derived first-order bound; the GPU pair is an ENGINEERING ERROR ENVELOPE in
the rsqrt constant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import forward_roundoff_bound as fb
import numpy as np

#: The formula's shared ``2 pi`` constant: C++ ``2*M_PI``, JAX ``two_pi()`` (asserted equal).
TWO_PI = 2.0 * np.pi
#: ``mu0 / (4 pi)`` as both lanes hold it.
MU0_OVER_4PI = 1.0e-7
#: sin/cos error in ulps of the exact value, per JAX lane (the native lane's glibc is 0.55).
TRIG_ULP: Mapping[str, float] = {"cpu": 0.55, "gpu": 2.5}
#: rsqrt (XLA rewrites ``1/sqrt``) charged 3 u against the modelled ``1/sqrt``'s 2.
INVERSE_SQRT_EXTRA_ROUNDINGS = 1.0
#: Nested ``jnp.hypot`` |n| <= 6.5 u against ``norm``'s 2.5 u; its JVP partial <= 8.5 vs 10.5.
HYPOT_EXTRA_ROUNDINGS = 4.0
#: A divisor or root argument this close to its own rounding band is refused.
BAND_REFUSAL = 2.0**-26

#: Products per monomial (longer lane form), per element and Cartesian component.
_VALUE_PRODUCTS = {
    "gamma": (3, 3, 2),
    "gammadash1": (5, 5, 4),
    "gammadash2": (5, 5, 4),
}
#: Roundings of one dof-Jacobian entry, per element and Cartesian component.
_JACOBIAN_ROUNDINGS = {
    "gamma": (2, 2, 1),
    "gammadash1": (5, 5, 3),
    "gammadash2": (4, 4, 3),
}


def refuse_inside_band(quantity: fb.Bounded, name: str) -> None:
    """R1: a divisor or root argument must be normal and far outside its rounding band."""
    magnitude = np.abs(quantity.v)
    if not np.all(np.isfinite(quantity.v)) or np.any(
        magnitude < np.finfo(np.float64).tiny
    ):
        raise ValueError(f"{name}: a non-normal divisor or root argument")
    if np.any(quantity.value_bound() >= BAND_REFUSAL * magnitude):
        raise ValueError(f"{name}: within 2**-26 relative of its own rounding band")


# ---------------------------------------------------------------------------
# Trig factors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Factor:
    """A rounded factor's float value and its absolute error bound (units of u)."""

    value: np.ndarray
    error: np.ndarray


def _trig(arg: np.ndarray, cosine: bool, roundings: float, ulp: float) -> _Factor:
    """``cos(arg)`` or ``sin(arg)`` of an argument rounded ``roundings`` times."""
    value = np.cos(arg) if cosine else np.sin(arg)
    slope = np.abs(np.sin(arg) if cosine else np.cos(arg))
    argument = np.where(arg == 0.0, 0.0, roundings * np.abs(arg))
    return _Factor(value, argument * slope + 2.0 * ulp * np.abs(value))


def _scaled(factor: _Factor, integer: float) -> _Factor:
    return _Factor(integer * factor.value, abs(integer) * factor.error)


def _monomial(
    kappa: float, factors: tuple[_Factor, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``kappa * prod(factors)``: value, |value| and the propagated factor errors."""
    value = kappa * np.ones(())
    absolute = abs(kappa) * np.ones(())
    for factor in factors:
        value = value * factor.value
        absolute = absolute * np.abs(factor.value)
    error = np.zeros(())
    for index, factor in enumerate(factors):
        term = abs(kappa) * factor.error
        for other, second in enumerate(factors):
            if other != index:
                term = term * np.abs(second.value)
        error = error + term
    return value, absolute, error


def used_trig_arguments(
    grid: SurfaceGrid, coil_quadpoints: np.ndarray, order: int
) -> dict[str, np.ndarray]:
    """Every float argument a lane passes to sin/cos (both product associations)."""
    theta = TWO_PI * grid.quadpoints_theta
    phi = TWO_PI * grid.quadpoints_phi
    m = np.arange(grid.mpol + 1, dtype=np.float64)
    n = np.arange(grid.ntor + 1, dtype=np.float64) * grid.nfp
    j = np.arange(1, order + 1, dtype=np.float64)
    return {
        "theta": np.outer(theta, m),
        "theta_reassociated": np.outer(grid.quadpoints_theta, TWO_PI * m),
        "phi": np.outer(phi, n),
        "phi_reassociated": np.outer(grid.quadpoints_phi, TWO_PI * n),
        "rotation": phi,
        "coil_cpp": np.outer(coil_quadpoints, TWO_PI * j),
        "coil_jax": np.outer(TWO_PI * coil_quadpoints, j),
    }


# ---------------------------------------------------------------------------
# Surface elements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceGrid:
    """A stellarator-symmetric ``SurfaceXYZTensorFourier`` quadrature layout."""

    mpol: int
    ntor: int
    nfp: int
    quadpoints_phi: np.ndarray
    quadpoints_theta: np.ndarray

    @classmethod
    def from_surface(cls, surface) -> SurfaceGrid:
        if not surface.stellsym or any(surface.clamped_dims):
            raise ValueError("the bound models a stellsym, unclamped tensor surface")
        return cls(
            int(surface.mpol),
            int(surface.ntor),
            int(surface.nfp),
            np.asarray(surface.quadpoints_phi, dtype=np.float64),
            np.asarray(surface.quadpoints_theta, dtype=np.float64),
        )

    def dofs(self) -> tuple[tuple[int, int, int], ...]:
        """``(dim, m, n)`` per dof in the C++ ``set_dofs_impl`` order and ``skip`` rule."""

        def skip(dim: int, m: int, n: int) -> bool:
            if dim == 0:
                return (n <= self.ntor and m > self.mpol) or (
                    n > self.ntor and m <= self.mpol
                )
            return (n <= self.ntor and m <= self.mpol) or (
                n > self.ntor and m > self.mpol
            )

        return tuple(
            (dim, m, n)
            for dim in range(3)
            for m in range(2 * self.mpol + 1)
            for n in range(2 * self.ntor + 1)
            if not skip(dim, m, n)
        )


@dataclass(frozen=True)
class SurfaceElements:
    """Injected ``gamma``, ``x_phi``, ``x_theta`` (``[nphi, ntheta, 3]``) with their Jacobians."""

    gamma: fb.Bounded
    gammadash1: fb.Bounded
    gammadash2: fb.Bounded


def _surface_factors(grid: SurfaceGrid, ulp: float):
    """Per tensor index: basis and its angle derivative, on the phi and theta grids."""
    theta = TWO_PI * grid.quadpoints_theta
    phi = TWO_PI * grid.quadpoints_phi
    w, dw, v, dv = [], [], [], []
    for m in range(2 * grid.mpol + 1):
        cosine = m <= grid.mpol
        mode = float(m if cosine else m - grid.mpol)
        factor = _trig(mode * theta, cosine, 2.0, ulp)
        w.append(factor)
        derivative = _trig(mode * theta, not cosine, 2.0, ulp)
        dw.append(_scaled(derivative, -mode if cosine else mode))
    for n in range(2 * grid.ntor + 1):
        cosine = n <= grid.ntor
        mode = float(grid.nfp * (n if cosine else n - grid.ntor))
        factor = _trig(mode * phi, cosine, 2.0, ulp)
        v.append(factor)
        derivative = _trig(mode * phi, not cosine, 2.0, ulp)
        dv.append(_scaled(derivative, -mode if cosine else mode))
    rotation_cos = _trig(phi, True, 1.0, ulp)
    rotation_sin = _trig(phi, False, 1.0, ulp)
    return w, dw, v, dv, rotation_cos, rotation_sin


def _grid(factor: _Factor, axis: int) -> _Factor:
    index = (slice(None), None) if axis == 0 else (None, slice(None))
    return _Factor(factor.value[index], factor.error[index])


def surface_elements(
    grid: SurfaceGrid, surface_dofs: np.ndarray, components: int, ulp: float
) -> SurfaceElements:
    """The three geometry arrays as injected elements over ``components`` derivative columns."""
    layout = grid.dofs()
    if len(layout) != surface_dofs.size:
        raise ValueError("surface dofs do not match the tensor-Fourier layout")
    w, dw, v, dv, cos_r, sin_r = _surface_factors(grid, ulp)
    cos_phi, sin_phi = _grid(cos_r, 0), _grid(sin_r, 0)
    shape = (grid.quadpoints_phi.size, grid.quadpoints_theta.size, 3, surface_dofs.size)
    tables = {
        name: tuple(np.zeros(shape) for _ in range(3))
        for name in ("gamma", "gammadash1", "gammadash2")
    }
    counts = {name: np.zeros(3) for name in tables}

    def add(name: str, axis: int, index: int, kappa: float, factors) -> None:
        value, absolute, error = _monomial(kappa, factors)
        tables[name][0][:, :, axis, index] += value
        tables[name][1][:, :, axis, index] += absolute
        tables[name][2][:, :, axis, index] += error
        counts[name][axis] += 1

    for index, (dim, m, n) in enumerate(layout):
        basis = (_grid(v[n], 0), _grid(w[m], 1))
        basis_dphi = (_grid(dv[n], 0), _grid(w[m], 1))
        basis_dtheta = (_grid(v[n], 0), _grid(dw[m], 1))
        if dim == 2:
            add("gamma", 2, index, 1.0, basis)
            add("gammadash1", 2, index, TWO_PI, basis_dphi)
            add("gammadash2", 2, index, TWO_PI, basis_dtheta)
            continue
        # x-dofs rotate as (cos, sin), y-dofs as (-sin, cos).
        along, across = (cos_phi, sin_phi) if dim == 0 else (sin_phi, cos_phi)
        sign_x = 1.0 if dim == 0 else -1.0
        add("gamma", 0, index, sign_x, (*basis, along))
        add("gamma", 1, index, 1.0, (*basis, across))
        add("gammadash1", 0, index, sign_x * TWO_PI, (*basis_dphi, along))
        add("gammadash1", 0, index, -TWO_PI, (*basis, across))
        add("gammadash1", 1, index, TWO_PI, (*basis_dphi, across))
        add("gammadash1", 1, index, sign_x * TWO_PI, (*basis, along))
        add("gammadash2", 0, index, sign_x * TWO_PI, (*basis_dtheta, along))
        add("gammadash2", 1, index, TWO_PI, (*basis_dtheta, across))

    coefficients = np.asarray(surface_dofs, dtype=np.float64)
    elements = {}
    for name, (jacobian, jacobian_abs, jacobian_trig) in tables.items():
        count = np.asarray(_VALUE_PRODUCTS[name]) + counts[name] - 1.0
        value = jacobian @ coefficients
        absolute = jacobian_abs @ np.abs(coefficients)
        error = count * absolute + jacobian_trig @ np.abs(coefficients)
        roundings = np.asarray(_JACOBIAN_ROUNDINGS[name], dtype=np.float64)
        derivative_error = roundings[:, None] * jacobian_abs + jacobian_trig
        pad = ((0, 0), (0, 0), (0, 0), (0, components - surface_dofs.size))
        elements[name] = fb.element(
            value,
            error,
            np.pad(jacobian, pad),
            np.pad(jacobian_abs, pad),
            np.pad(derivative_error, pad),
        )
    return SurfaceElements(**elements)


# ---------------------------------------------------------------------------
# Coils
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoilSet:
    """Coil geometry as rounded constants: values and absolute errors (units of u)."""

    gamma: np.ndarray  # [coils, points, 3]
    gamma_error: np.ndarray
    gammadash: np.ndarray
    gammadash_error: np.ndarray
    currents: np.ndarray  # [coils], exact and shared


def unwrapped_coil_curve(curve) -> tuple[object, np.ndarray | None]:
    """Base curve and its shared ``rotmat`` (``None`` for an unrotated base curve)."""
    if not hasattr(curve, "rotmat"):
        return curve, None
    if hasattr(curve.curve, "rotmat"):
        raise ValueError("the bound models at most one RotatedCurve level per coil")
    return curve.curve, np.asarray(curve.rotmat, dtype=np.float64)


def _base_curve_geometry(curve, ulp: float) -> tuple[fb.Bounded, fb.Bounded]:
    """``gamma``, ``gamma'`` of one CurveXYZFourier as components-free constants."""
    order = int(curve.order)
    quadpoints = np.asarray(curve.quadpoints, dtype=np.float64)
    coefficients = np.asarray(curve.local_full_x, dtype=np.float64).reshape(
        3, 2 * order + 1
    )
    shape = (quadpoints.size, 3)
    gamma = np.broadcast_to(coefficients[:, 0], shape).copy()
    gamma_abs = np.broadcast_to(np.abs(coefficients[:, 0]), shape).copy()
    gamma_trig = np.zeros(shape)
    dash, dash_abs, dash_trig = (np.zeros(shape) for _ in range(3))
    for j in range(1, order + 1):
        arg = (TWO_PI * j) * quadpoints
        sin_j = _trig(arg, False, 2.0, ulp)
        cos_j = _trig(arg, True, 2.0, ulp)
        s = coefficients[:, 2 * j - 1][None, :]
        c = coefficients[:, 2 * j][None, :]
        gamma += s * sin_j.value[:, None] + c * cos_j.value[:, None]
        gamma_abs += (
            np.abs(s) * np.abs(sin_j.value)[:, None]
            + np.abs(c) * np.abs(cos_j.value)[:, None]
        )
        gamma_trig += (
            np.abs(s) * sin_j.error[:, None] + np.abs(c) * cos_j.error[:, None]
        )
        scale = TWO_PI * j
        dash += scale * (s * cos_j.value[:, None] - c * sin_j.value[:, None])
        dash_abs += scale * (
            np.abs(s) * np.abs(cos_j.value)[:, None]
            + np.abs(c) * np.abs(sin_j.value)[:, None]
        )
        dash_trig += scale * (
            np.abs(s) * cos_j.error[:, None] + np.abs(c) * sin_j.error[:, None]
        )
    gamma_count = 1.0 + 2 * order
    dash_count = 2.0 + 2 * order
    return (
        fb.constant(gamma, 0, gamma_count * gamma_abs + gamma_trig),
        fb.constant(dash, 0, dash_count * dash_abs + dash_trig),
    )


def coil_set(coils, ulp: float) -> CoilSet:
    """The Biot-Savart coils: base geometry, rotated by each rotated coil's shared ``rotmat``."""
    base_geometry: dict[int, tuple[fb.Bounded, fb.Bounded]] = {}
    gammas, gamma_errors, dashes, dash_errors, currents = [], [], [], [], []
    for coil in coils:
        base, rotmat = unwrapped_coil_curve(coil.curve)
        if id(base) not in base_geometry:
            base_geometry[id(base)] = _base_curve_geometry(base, ulp)
        gamma, dash = base_geometry[id(base)]
        rotated_gamma = gamma if rotmat is None else fb.matmul_constant(gamma, rotmat)
        rotated_dash = dash if rotmat is None else fb.matmul_constant(dash, rotmat)
        gammas.append(rotated_gamma.v)
        gamma_errors.append(rotated_gamma.e)
        dashes.append(rotated_dash.v)
        dash_errors.append(rotated_dash.e)
        currents.append(float(coil.current.get_value()))
    return CoilSet(
        np.stack(gammas),
        np.stack(gamma_errors),
        np.stack(dashes),
        np.stack(dash_errors),
        np.asarray(currents, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Biot-Savart
# ---------------------------------------------------------------------------


def _inverse_sqrt_forms(argument: fb.Bounded) -> tuple[fb.Bounded, fb.Bounded]:
    """``1/sqrt(x)`` as C++ forms it (``1./sqrt``) and as XLA does (``rsqrt``, +1 rounding)."""
    root = fb.sqrt(argument)
    refuse_inside_band(argument, "sqrt argument")
    refuse_inside_band(root, "divisor sqrt")
    one = fb.constant(np.ones_like(argument.v), argument.components)
    native = fb.div(one, root)
    return native, fb.extra_roundings(native, INVERSE_SQRT_EXTRA_ROUNDINGS)


def native_inverse_cube(distance_squared: fb.Bounded) -> fb.Bounded:
    """``1/r^3`` as ``(1/sqrt r2)^3`` (``biot_savart_impl.h:65-67``)."""
    inverse, _ = _inverse_sqrt_forms(distance_squared)
    return fb.mul(fb.mul(inverse, inverse), inverse)


def enveloped_inverse_cube(distance_squared: fb.Bounded) -> fb.Bounded:
    """``1/r^3``: envelope of the native form and JAX's ``rsqrt(r2) * (1/r2)``."""
    inverse, jax_inverse = _inverse_sqrt_forms(distance_squared)
    native = fb.mul(fb.mul(inverse, inverse), inverse)
    one = fb.constant(np.ones_like(distance_squared.v), distance_squared.components)
    jax_form = fb.mul(jax_inverse, fb.div(one, distance_squared))
    return fb.envelope(native, jax_form)


def biot_savart_field(
    points: fb.Bounded,
    coils: CoilSet,
    inverse_cube: Callable[[fb.Bounded], fb.Bounded] = enveloped_inverse_cube,
) -> fb.Bounded:
    """``B`` at ``points`` (``[P, 3]``) in upstream's form: ``1e-7/Q sum I (gamma' x d)/|d|^3``."""
    components = points.components
    point = fb.Bounded(
        points.v[:, None, None, :],
        points.e[:, None, None, :],
        points.d[:, None, None, :, :],
        points.D[:, None, None, :, :],
        points.Ed[:, None, None, :, :],
        points.paths[:, None, None, :, :],
    )
    gamma = fb.constant(coils.gamma[None], components, coils.gamma_error[None])
    dash = fb.constant(coils.gammadash[None], components, coils.gammadash_error[None])
    current = fb.constant(coils.currents[None, :, None, None], components)
    difference = fb.sub(point, gamma)
    inverse_cubed = inverse_cube(fb.dot(difference, difference))
    weighted = fb.mul(
        fb.mul(fb.cross(dash, difference), inverse_cubed[..., None]), current
    )
    quadrature = coils.gamma.shape[1]
    # 1e-7 / Q: each lane may round this constant twice.
    return fb.scale(
        fb.total(weighted, (1, 2)), MU0_OVER_4PI / quadrature, relative_error=2.0
    )


def _cross_unit(a: fb.Bounded, k: int) -> fb.Bounded:
    """``a x e_k`` (``vec3dsimd.h:190-197``): a signed permutation, exact."""
    zero = fb.constant(np.zeros_like(a.v[..., 0]), a.components)
    x, y, z = a[..., 0], a[..., 1], a[..., 2]
    columns = (
        (zero, z, fb.neg(y)),
        (fb.neg(z), zero, x),
        (y, fb.neg(x), zero),
    )[k]
    return fb.stack(list(columns), axis=-1)


def closed_form_gradient_error(
    points: np.ndarray, point_error: np.ndarray, coils: CoilSet
) -> np.ndarray:
    """Error of the native ``dB/dX`` closed form (``biot_savart_impl.h:74-96``), ``[P, a, k]``.

    Summed over coil points without their addition charge: those additions are the
    path merges the ``(paths - 1) D`` charge of the chained trace already bounds.
    """
    point = fb.constant(points[:, None, None, :], 0, point_error[:, None, None, :])
    gamma = fb.constant(coils.gamma[None], 0, coils.gamma_error[None])
    dash = fb.constant(coils.gammadash[None], 0, coils.gammadash_error[None])
    difference = fb.sub(point, gamma)
    distance_squared = fb.dot(difference, difference)
    inverse, _ = _inverse_sqrt_forms(distance_squared)
    inverse3 = fb.mul(fb.mul(inverse, inverse), inverse)
    inverse4 = fb.mul(inverse3, inverse)
    three_cross = fb.mul(fb.cross(dash, difference), fb.scale(inverse, 3.0)[..., None])
    dash_norm = fb.mul(dash, fb.mul(distance_squared, inverse)[..., None])
    rows = [
        fb.mul(
            fb.sub(
                _cross_unit(dash_norm, k),
                fb.mul(three_cross, difference[..., k][..., None]),
            ),
            inverse4[..., None],
        )
        for k in range(3)
    ]
    current = fb.constant(coils.currents[None, :, None, None, None], 0)
    scaled = fb.scale(
        fb.mul(fb.stack(rows, axis=-2), current),
        MU0_OVER_4PI / coils.gamma.shape[1],
        relative_error=2.0,
    )
    return np.swapaxes(np.sum(scaled.e, axis=(1, 2)), -1, -2)


def local_seed(values: np.ndarray, errors: np.ndarray) -> fb.Bounded:
    """Points ``[P, 3]`` as the three local components of each point's own trace."""
    eye = np.broadcast_to(np.eye(3), values.shape + (3,)).copy()
    return fb.Bounded(values, errors, eye, eye.copy(), np.zeros_like(eye), eye.copy())


def field_element(points: fb.Bounded, coils: CoilSet, chunk: int) -> fb.Bounded:
    """``B`` at ``points`` chained through their Jacobian; the native closed form enveloped."""
    parts = []
    for start in range(0, points.v.shape[0], chunk):
        rows = slice(start, start + chunk)
        local = biot_savart_field(local_seed(points.v[rows], points.e[rows]), coils)
        closed = closed_form_gradient_error(points.v[rows], points.e[rows], coils)
        # dB/dc = dB/dX dX/dc through one fma: one more rounding per path.
        local = fb.Bounded(
            local.v,
            local.e,
            local.d,
            local.D,
            np.maximum(local.Ed, closed + local.D),
            local.paths,
        )
        parts.append(fb.compose(local, points[rows]))
    return fb.Bounded(
        *(
            np.concatenate([getattr(part, name) for part in parts])
            for name in ("v", "e", "d", "D", "Ed", "paths")
        )
    )


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstStageBound:
    """``F`` at one state with its rounding bookkeeping, its terms and its elements."""

    total: fb.Bounded
    terms: Mapping[str, fb.Bounded]
    surface: SurfaceElements


def flat_points(quantity: fb.Bounded) -> fb.Bounded:
    """``[nphi, ntheta, 3]`` -> ``[nphi * ntheta, 3]`` in the native row-major order."""
    size = quantity.components
    return fb.Bounded(
        quantity.v.reshape(-1, 3),
        quantity.e.reshape(-1, 3),
        quantity.d.reshape(-1, 3, size),
        quantity.D.reshape(-1, 3, size),
        quantity.Ed.reshape(-1, 3, size),
        quantity.paths.reshape(-1, 3, size),
    )


def first_stage_objective_bound(
    grid: SurfaceGrid,
    coils: CoilSet,
    x: np.ndarray,
    target_label: float,
    constraint_weight: float,
    ulp: float,
) -> FirstStageBound:
    """Upstream's first-stage penalty ``F`` at ``x = (surface dofs, iota, G)`` as a Bounded."""
    x = np.asarray(x, dtype=np.float64)
    size = x.size
    surface_dof_count = size - 2
    seeds = fb.seed(x)
    iota, G = seeds[surface_dof_count], seeds[surface_dof_count + 1]
    elements = surface_elements(grid, x[:surface_dof_count], size, ulp)
    gamma = flat_points(elements.gamma)
    xphi = flat_points(elements.gammadash1)
    xtheta = flat_points(elements.gammadash2)
    point_count = gamma.v.shape[0]

    field = field_element(gamma, coils, chunk=grid.quadpoints_theta.size)
    tangent = fb.add(xphi, fb.mul(iota, xtheta))
    field_squared = fb.dot(field, field)
    _, jax_inverse_root = _inverse_sqrt_forms(field_squared)
    one = fb.constant(np.ones_like(field_squared.v), size)
    reciprocal = fb.div(one, field_squared)
    refuse_inside_band(reciprocal, "sqrt argument 1/B2")
    weight = fb.envelope(fb.sqrt(reciprocal), jax_inverse_root)
    residual = fb.sub(fb.mul(G, field), fb.mul(field_squared[..., None], tangent))
    weighted = fb.mul(residual, weight[..., None])
    count = 3 * point_count
    boozer = fb.scale(
        fb.total(fb.square(weighted), (0, 1)), 0.5 / count, relative_error=1.0
    )

    normal = fb.cross(xphi, xtheta)
    normal_squared = fb.dot(normal, normal)
    refuse_inside_band(normal_squared, "sqrt argument |n|^2")
    norm = fb.norm(normal)
    refuse_inside_band(norm, "divisor |n|")
    normal_norm = fb.envelope(norm, fb.extra_roundings(norm, HYPOT_EXTRA_ROUNDINGS))
    area = fb.mean(normal_norm, 0)
    excess = fb.sub(area, fb.constant(target_label, size))
    half_weight = 0.5 * constraint_weight
    # 3 roundings beyond the excess's own: JAX's (50 d) d takes 2.  Native 0.5 (10 d)**2 takes
    # 2 + 1.08 through glibc pow (<= 0.54 ulp, e_pow.c:29); its 0.08 |term| excess is inside
    # the total's addition charge below (3 summands charged 2 sum|x|, one addition taken, the
    # axis term an exact zero), so the value bound of F still holds for the native form.
    label = fb.scale(fb.square(excess), half_weight, relative_error=1.0)
    axis_z = elements.gamma[0, 0, 2]
    axis = fb.scale(fb.square(axis_z), half_weight, relative_error=1.0)

    terms = {"boozer": boozer, "label": label, "axis_z": axis}
    total = fb.total(fb.stack(list(terms.values())), 0)
    return FirstStageBound(total=total, terms=terms, surface=elements)
