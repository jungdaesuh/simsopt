from __future__ import annotations

import numpy as np
from jax import jacfwd, jit, jvp, vjp
import jax.numpy as jnp
import simsoptpp as sopp

from simsopt._core.derivative import Derivative
from simsopt._core.json import GSONDecoder
from simsopt.geo.curve import Curve
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.core.curve_kernels import (
    curve_cws_rz_gamma_from_dofs,
    kappa_pure as _kappa_pure,
    torsion_pure as _torsion_pure,
)
from simsopt_jax.core.specs import make_curve_cwsfourier_rz_spec
from simsopt_jax.core.surface_fourier_kernels import surface_gamma_lin_from_dofs
from simsopt_jax.core.surface_rzfourier import (
    _surface_rz_fourier_derivative_lin_from_spec,
    surface_rz_fourier_spec_from_dofs,
)
from simsopt_jax.runtime.host_boundary import host_float64 as _as_numpy_float64

__all__ = ["CurveCWSFourierCPP", "CurveCWSFourier"]


def gamma_2d(cdofs, qpts, order, G: int = 0, H: int = 0):
    """Given some dofs, return curve position in 2D cartesian coordinate

    Args:
     - cdofs: Input dofs. Array of size 2*(2*order+1)
     - qpts: quadrature points. Array of floats from 0 to 1, of size N.
     - order: Maximum Fourier series order.

    Returns:
     - phi: Array of size N x 1.
     - theta: Array of size N x 1.
    """
    # Unpack dofs
    phic = cdofs[: order + 1]
    phis = cdofs[order + 1 : 2 * order + 1]
    thetac = cdofs[2 * order + 1 : 3 * order + 2]
    thetas = cdofs[3 * order + 2 :]

    # Construct theta and phi arrays
    theta = jnp.zeros((qpts.size,))
    phi = jnp.zeros((qpts.size,))

    ll = qpts * 2.0 * jnp.pi
    for ii in range(order + 1):
        theta = theta + thetac[ii] * jnp.cos(ii * ll)
        phi = phi + phic[ii] * jnp.cos(ii * ll)

    for ii in range(order):
        theta = theta + thetas[ii] * jnp.sin((ii + 1) * ll)
        phi = phi + phis[ii] * jnp.sin((ii + 1) * ll)

    # Add secular terms
    theta = theta + G * qpts
    phi = phi + H * qpts

    # Prepare output
    out = jnp.zeros((qpts.size, 2))
    out = out.at[:, 0].set(phi)
    out = out.at[:, 1].set(theta)

    return out


def gamma_2d_numpy(cdofs, qpts, order, G: int = 0, H: int = 0):
    phic = cdofs[: order + 1]
    phis = cdofs[order + 1 : 2 * order + 1]
    thetac = cdofs[2 * order + 1 : 3 * order + 2]
    thetas = cdofs[3 * order + 2 :]

    qpts_arr = np.asarray(qpts, dtype=np.float64)
    ll = qpts_arr * (2.0 * np.pi)
    phi = np.zeros(qpts_arr.shape, dtype=np.float64)
    theta = np.zeros(qpts_arr.shape, dtype=np.float64)

    for ii in range(order + 1):
        angle = ii * ll
        theta += thetac[ii] * np.cos(angle)
        phi += phic[ii] * np.cos(angle)

    for ii in range(order):
        mode = ii + 1
        angle = mode * ll
        theta += thetas[ii] * np.sin(angle)
        phi += phis[ii] * np.sin(angle)

    theta += G * qpts_arr
    phi += H * qpts_arr
    return np.column_stack((phi, theta))


def gammadash_2d_numpy(cdofs, qpts, order, G: int = 0, H: int = 0):
    phic = cdofs[: order + 1]
    phis = cdofs[order + 1 : 2 * order + 1]
    thetac = cdofs[2 * order + 1 : 3 * order + 2]
    thetas = cdofs[3 * order + 2 :]

    qpts_arr = np.asarray(qpts, dtype=np.float64)
    ll = qpts_arr * (2.0 * np.pi)
    two_pi = 2.0 * np.pi
    phi = np.full(qpts_arr.shape, H, dtype=np.float64)
    theta = np.full(qpts_arr.shape, G, dtype=np.float64)

    for ii in range(order + 1):
        factor = ii * two_pi
        angle = ii * ll
        theta -= thetac[ii] * factor * np.sin(angle)
        phi -= phic[ii] * factor * np.sin(angle)

    for ii in range(order):
        mode = ii + 1
        factor = mode * two_pi
        angle = mode * ll
        theta += thetas[ii] * factor * np.cos(angle)
        phi += phis[ii] * factor * np.cos(angle)

    return np.column_stack((phi, theta))


def gammadashdash_2d_numpy(cdofs, qpts, order, G: int = 0, H: int = 0):
    phic = cdofs[: order + 1]
    phis = cdofs[order + 1 : 2 * order + 1]
    thetac = cdofs[2 * order + 1 : 3 * order + 2]
    thetas = cdofs[3 * order + 2 :]

    qpts_arr = np.asarray(qpts, dtype=np.float64)
    ll = qpts_arr * (2.0 * np.pi)
    two_pi = 2.0 * np.pi
    phi = np.zeros(qpts_arr.shape, dtype=np.float64)
    theta = np.zeros(qpts_arr.shape, dtype=np.float64)

    for ii in range(order + 1):
        factor_sq = (ii * two_pi) ** 2
        angle = ii * ll
        theta -= thetac[ii] * factor_sq * np.cos(angle)
        phi -= phic[ii] * factor_sq * np.cos(angle)

    for ii in range(order):
        mode = ii + 1
        factor_sq = (mode * two_pi) ** 2
        angle = mode * ll
        theta -= thetas[ii] * factor_sq * np.sin(angle)
        phi -= phis[ii] * factor_sq * np.sin(angle)

    return np.column_stack((phi, theta))


def gamma_curve_on_surface(
    curve_dofs,
    qpts,
    order,
    G,
    H,
    surf_dofs,
    surf_type,
    mpol,
    ntor,
    nfp,
    stellsym=True,
):
    if surf_type == "RZ_Fourier":
        return curve_cws_rz_gamma_from_dofs(
            curve_dofs,
            qpts,
            order,
            G,
            H,
            surf_dofs,
            mpol,
            ntor,
            nfp,
            stellsym,
        )

    gamma_2d_values = gamma_2d(curve_dofs, qpts, order, G, H)
    phi = gamma_2d_values[:, 0]
    theta = gamma_2d_values[:, 1]
    if surf_type == "XYZ_Tensor_Fourier":
        return surface_gamma_lin_from_dofs(
            surf_dofs,
            phi,
            theta,
            mpol,
            ntor,
            nfp,
            stellsym,
        )
    if surf_type is None:
        return phi, theta
    raise NotImplementedError(f"Unsupported CWS surface type {surf_type!r}.")


def vjp_contraction_1d(mat, v):
    # contract matrix of size ijk times vector of size jk into array of size i
    return np.einsum("ij,i->j", mat, v)


def vjp_contraction_2d(mat, v):
    # contract matrix of size ijk times vector of size jk into array of size i
    return np.einsum("ijk,ij->k", mat, v)


class CurveCWSFourierCPP(Curve, sopp.Curve):
    def __init__(self, quadpoints, order, surf, G=0, H=0, **kwargs):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0.0, 1.0, int(quadpoints), endpoint=False)

        # Curve order. Number of Fourier harmonics for phi and theta
        self.order = order
        self.G = G
        self.H = H

        # Modes are order as phic, phis, thetac, thetas
        self.modes = [
            np.zeros((order + 1,)),
            np.zeros((order,)),
            np.zeros((order + 1,)),
            np.zeros((order,)),
        ]

        # self.quadpoints = quadpoints
        self.surf = surf

        if isinstance(surf, SurfaceRZFourier):
            self.surf_type = "RZ_Fourier"
        elif isinstance(surf, SurfaceXYZTensorFourier):
            self.surf_type = "XYZ_Tensor_Fourier"
        else:
            raise NotImplementedError(
                "CurveCWSFourierCPP is only implemented for SurfaceRZFourier "
                "and SurfaceXYZTensorFourier classes."
            )

        # Initialize C++ class and Curve class
        sopp.Curve.__init__(self, quadpoints)
        Curve.__init__(
            self,
            x0=self.get_dofs(),
            depends_on=[],
            names=self._make_names(),
            external_dof_setter=CurveCWSFourierCPP.set_dofs_impl,
            **kwargs,
        )

        self.numquadpoints = self.quadpoints.size

        # useful functions
        quadpoints = np.asarray(self.quadpoints, dtype=np.float64)
        points = quadpoints
        ones = np.ones_like(quadpoints)
        current_curve_dofs = lambda: _as_jax_float64(self.get_dofs())
        current_surface_dofs = lambda: _as_jax_float64(self.surf.get_dofs())

        def gamma_on_surface(curve_dofs, surface_dofs, qpts):
            return gamma_curve_on_surface(
                curve_dofs,
                qpts,
                self.order,
                self.G,
                self.H,
                surface_dofs,
                self.surf_type,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.nfp,
                self.surf.stellsym,
            )

        def gammadash_on_surface(curve_dofs, surface_dofs, qpts):
            return jvp(
                lambda curve_qpts: gamma_on_surface(
                    curve_dofs, surface_dofs, curve_qpts
                ),
                (qpts,),
                (ones,),
            )[1]

        def _arg0_vjp_kernel(fun):
            return jit(
                lambda cdofs, sdofs, v: vjp(
                    lambda local_cdofs: fun(local_cdofs, sdofs),
                    cdofs,
                )[1](v)[0]
            )

        def _arg1_vjp_kernel(fun):
            return jit(
                lambda cdofs, sdofs, v: vjp(
                    lambda local_sdofs: fun(cdofs, local_sdofs),
                    sdofs,
                )[1](v)[0]
            )

        def _bind_live_surface(fun):
            def bound(cdofs):
                return fun(cdofs, current_surface_dofs())

            return bound

        def _bind_live_surface_vjp(fun):
            def bound(cdofs, v):
                return fun(cdofs, current_surface_dofs(), v)

            return bound

        def _bind_live_curve_vjp(fun):
            def bound(sdofs, v):
                return fun(current_curve_dofs(), sdofs, v)

            return bound

        self.gamma_pure = jit(gamma_on_surface)

        def gamma_at_points(cdofs, sdofs):
            return self.gamma_pure(cdofs, sdofs, points)

        self.gamma_jax = jit(gamma_at_points)
        dgamma_by_dcoeff_vjp_kernel = _arg0_vjp_kernel(gamma_at_points)
        dgamma_by_dsurf_vjp_kernel = _arg1_vjp_kernel(gamma_at_points)
        self.dgamma_by_dcoeff_vjp_jax = _bind_live_surface_vjp(
            dgamma_by_dcoeff_vjp_kernel
        )
        self.dgamma_by_dsurf_vjp_jax = _bind_live_curve_vjp(dgamma_by_dsurf_vjp_kernel)

        self.gammadash_pure = jit(gammadash_on_surface)

        def gammadash_at_points(cdofs, sdofs):
            return self.gammadash_pure(cdofs, sdofs, points)

        self.gammadash_jax = jit(gammadash_at_points)
        dgammadash_by_dcoeff_vjp_kernel = _arg0_vjp_kernel(gammadash_at_points)
        dgammadash_by_dsurf_vjp_kernel = _arg1_vjp_kernel(gammadash_at_points)
        self.dgammadash_by_dcoeff_vjp_jax = _bind_live_surface_vjp(
            dgammadash_by_dcoeff_vjp_kernel
        )
        self.dgammadash_by_dsurf_vjp_jax = _bind_live_curve_vjp(
            dgammadash_by_dsurf_vjp_kernel
        )

        self.gammadashdash_pure = jit(
            lambda cdofs, sdofs, qpts: jvp(
                lambda curve_qpts: self.gammadash_pure(cdofs, sdofs, curve_qpts),
                (qpts,),
                (ones,),
            )[1]
        )

        def gammadashdash_at_points(cdofs, sdofs):
            return self.gammadashdash_pure(cdofs, sdofs, points)

        self.gammadashdash_jax = jit(gammadashdash_at_points)
        dgammadashdash_by_dcoeff_vjp_kernel = _arg0_vjp_kernel(gammadashdash_at_points)
        dgammadashdash_by_dsurf_vjp_kernel = _arg1_vjp_kernel(gammadashdash_at_points)
        self.dgammadashdash_by_dcoeff_vjp_jax = _bind_live_surface_vjp(
            dgammadashdash_by_dcoeff_vjp_kernel
        )
        self.dgammadashdash_by_dsurf_vjp_jax = _bind_live_curve_vjp(
            dgammadashdash_by_dsurf_vjp_kernel
        )

        # The third derivative is implemented with the same live-surface JAX
        # contract as the lower derivatives so composed CPU/JAX paths share
        # one geometry source.
        self.gammadashdashdash_pure = jit(
            lambda cdofs, sdofs, qpts: jvp(
                lambda curve_qpts: self.gammadashdash_pure(cdofs, sdofs, curve_qpts),
                (qpts,),
                (ones,),
            )[1]
        )

        def gammadashdashdash_at_points(cdofs, sdofs):
            return self.gammadashdashdash_pure(cdofs, sdofs, points)

        self.gammadashdashdash_jax = jit(gammadashdashdash_at_points)
        self.gammacdashdashdash_jax = _bind_live_surface(self.gammadashdashdash_jax)
        dgammadashdashdash_by_dcoeff_kernel = jit(
            jacfwd(gammadashdashdash_at_points, argnums=0)
        )
        dgammadashdashdash_by_dcoeff_vjp_kernel = _arg0_vjp_kernel(
            gammadashdashdash_at_points
        )
        dgammadashdashdash_by_dsurf_vjp_kernel = _arg1_vjp_kernel(
            gammadashdashdash_at_points
        )
        self.dgammadashdashdash_by_dcoeff_jax = _bind_live_surface(
            dgammadashdashdash_by_dcoeff_kernel
        )
        self.dgammadashdashdash_by_dcoeff_vjp_jax = _bind_live_surface_vjp(
            dgammadashdashdash_by_dcoeff_vjp_kernel
        )
        self.dgammadashdashdash_by_dsurf_vjp_jax = _bind_live_curve_vjp(
            dgammadashdashdash_by_dsurf_vjp_kernel
        )

        ## gamma
        self.gamma_2d_pure = jit(
            lambda cdofs, qpts: gamma_2d(cdofs, qpts, self.order, self.G, self.H)
        )
        self.gamma_2d_jax = jit(lambda cdofs: self.gamma_2d_pure(cdofs, points))
        self.dgamma_2d_by_dcoeff_jax = jit(
            lambda cdofs: jacfwd(self.gamma_2d_jax)(cdofs)
        )

        ## gammadash
        self.gammadash_2d_pure = jit(
            lambda cdofs, q: jvp(
                lambda qpts: self.gamma_2d_pure(cdofs, qpts), (q,), (ones,)
            )[1]
        )
        self.gammadash_2d_jax = jit(lambda cdofs: self.gammadash_2d_pure(cdofs, points))
        self.dgammadash_2d_by_dcoeff_jax = jit(
            lambda cdofs: jacfwd(self.gammadash_2d_jax)(cdofs)
        )

        ## gammadashdash
        self.gammadashdash_2d_pure = jit(
            lambda cdofs, q: jvp(
                lambda qpts: self.gammadash_2d_pure(cdofs, qpts), (q,), (ones,)
            )[1]
        )
        self.gammadashdash_2d_jax = jit(
            lambda cdofs: self.gammadashdash_2d_pure(cdofs, points)
        )
        self.dgammadashdash_2d_by_dcoeff_jax = jit(
            lambda cdofs: jacfwd(self.gammadashdash_2d_jax)(cdofs)
        )

        # determine sign for normal
        nr = self.unit_normal_impl(np.array([0]), np.array([0]))  # theta=phi=0
        if nr[0, 0] > 0:
            self.sgn_r = 1
            nz = self.unit_normal_impl(
                np.array([0]), np.array([0.25])
            )  # this is on top of the device
            if nz[0, 2] > 0:
                self.sgn_z = 1
            else:
                self.sgn_z = -1
        else:
            self.sgn_r = -1
            nz = self.unit_normal_impl(
                np.array([0]), np.array([-0.25])
            )  # this is on top of the device
            if nz[0, 2] > 0:
                self.sgn_z = 1
            else:
                self.sgn_z = -1

    def set_dofs(self, dofs):
        self.local_x = dofs
        sopp.Curve.set_dofs(self, dofs)

    def num_dofs(self):
        return 2 * (self.order + 1) + 2 * self.order

    @staticmethod
    def _surface_lin_inputs(phi, theta):
        phi_arr = np.ascontiguousarray(_as_numpy_float64(phi))
        theta_arr = np.ascontiguousarray(_as_numpy_float64(theta))
        return phi_arr, theta_arr

    def _surface_lin_inputs_from_gamma2d(self, g2):
        g2_arr = _as_numpy_float64(g2)
        return self._surface_lin_inputs(g2_arr[:, 0], g2_arr[:, 1])

    def _surface_rz_spec_at(self, phi, theta):
        return surface_rz_fourier_spec_from_dofs(
            _as_jax_float64(self.surf.get_dofs()),
            quadpoints_phi=_as_jax_float64(phi),
            quadpoints_theta=_as_jax_float64(theta),
            mpol=self.surf.mpol,
            ntor=self.surf.ntor,
            nfp=self.surf.nfp,
            stellsym=self.surf.stellsym,
        )

    def _surface_derivative_lin(
        self,
        out,
        phi,
        theta,
        phi_order,
        theta_order,
        legacy_method_name,
    ):
        if self.surf_type == "RZ_Fourier":
            spec = self._surface_rz_spec_at(phi, theta)
            out[:, :] = _as_numpy_float64(
                _surface_rz_fourier_derivative_lin_from_spec(
                    spec,
                    _as_jax_float64(phi),
                    _as_jax_float64(theta),
                    phi_order,
                    theta_order,
                )
            )
            return
        getattr(self.surf, legacy_method_name)(out, phi, theta)

    def _surface_gammadash1_lin(self, out, phi, theta):
        self._surface_derivative_lin(out, phi, theta, 1, 0, "gammadash1_lin")

    def _surface_gammadash2_lin(self, out, phi, theta):
        self._surface_derivative_lin(out, phi, theta, 0, 1, "gammadash2_lin")

    def _surface_gammadash1dash1_lin(self, out, phi, theta):
        self._surface_derivative_lin(out, phi, theta, 2, 0, "gammadash1dash1_lin")

    def _surface_gammadash1dash2_lin(self, out, phi, theta):
        self._surface_derivative_lin(out, phi, theta, 1, 1, "gammadash1dash2_lin")

    def _surface_gammadash2dash2_lin(self, out, phi, theta):
        self._surface_derivative_lin(out, phi, theta, 0, 2, "gammadash2dash2_lin")

    def _surface_gammadash1dash1dash1_lin(self, out, phi, theta):
        self._surface_derivative_lin(
            out,
            phi,
            theta,
            3,
            0,
            "gammadash1dash1dash1_lin",
        )

    def _surface_gammadash1dash1dash2_lin(self, out, phi, theta):
        self._surface_derivative_lin(
            out,
            phi,
            theta,
            2,
            1,
            "gammadash1dash1dash2_lin",
        )

    def _surface_gammadash1dash2dash2_lin(self, out, phi, theta):
        self._surface_derivative_lin(
            out,
            phi,
            theta,
            1,
            2,
            "gammadash1dash2dash2_lin",
        )

    def _surface_gammadash2dash2dash2_lin(self, out, phi, theta):
        self._surface_derivative_lin(
            out,
            phi,
            theta,
            0,
            3,
            "gammadash2dash2dash2_lin",
        )

    def get_dofs(self):
        return np.concatenate(self.modes)

    def set_dofs_impl(self, dofs):
        self.modes[0] = dofs[0 : self.order + 1]
        self.modes[1] = dofs[self.order + 1 : 2 * self.order + 1]
        self.modes[2] = dofs[2 * self.order + 1 : 3 * self.order + 2]
        self.modes[3] = dofs[3 * self.order + 2 : 4 * self.order + 2]

    def to_spec(self):
        if self.surf_type != "RZ_Fourier":
            raise NotImplementedError(
                "Immutable CWS curve specs are implemented for SurfaceRZFourier."
            )
        curve_dofs = self.get_dofs()
        gamma_2d_values = gamma_2d_numpy(
            curve_dofs,
            self.quadpoints,
            self.order,
            self.G,
            self.H,
        )
        surface_spec = surface_rz_fourier_spec_from_dofs(
            _as_jax_float64(self.surf.get_dofs()),
            quadpoints_phi=_as_jax_float64(gamma_2d_values[:, 0]),
            quadpoints_theta=_as_jax_float64(gamma_2d_values[:, 1]),
            mpol=self.surf.mpol,
            ntor=self.surf.ntor,
            nfp=self.surf.nfp,
            stellsym=self.surf.stellsym,
        )
        return make_curve_cwsfourier_rz_spec(
            dofs=curve_dofs,
            quadpoints=self.quadpoints,
            surface=surface_spec,
            order=self.order,
            G=self.G,
            H=self.H,
        )

    def _derivative_from_curve_surface_vjps(self, curve_vjp, surface_vjp, v):
        cdofs = _as_jax_float64(self.get_dofs())
        sdofs = _as_jax_float64(self.surf.get_dofs())
        v_jax = _as_jax_float64(v)
        return Derivative(
            {
                self: _as_numpy_float64(curve_vjp(cdofs, v_jax)),
                self.surf: _as_numpy_float64(surface_vjp(sdofs, v_jax)),
            }
        )

    def _geometry_derivatives_jax(self, cdofs, sdofs):
        return (
            self.gammadash_jax(cdofs, sdofs),
            self.gammadashdash_jax(cdofs, sdofs),
            self.gammadashdashdash_jax(cdofs, sdofs),
        )

    def _make_names(self):
        dofs_name = []
        for mode in ["phic", "phis", "thetac", "thetas"]:
            for ii in range(self.order + 1):
                if mode == "phis" and ii == 0:
                    continue

                if mode == "thetas" and ii == 0:
                    continue

                dofs_name.append(f"{mode}({ii})")

        return dofs_name

    # =========================================================================
    # GAMMA
    # -----
    def gamma_2d(self):
        cdofs = _as_jax_float64(self.get_dofs())
        return self.gamma_2d_jax(cdofs)

    def gamma_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:, :] = gamma_2d_numpy(cdofs, quadpoints, self.order, self.G, self.H)

    def gamma(self):
        g2 = gamma_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        phi, theta = self._surface_lin_inputs_from_gamma2d(g2)
        out = np.zeros((self.numquadpoints, 3))
        self.surf.gamma_lin(out, phi, theta)
        return out

    def gamma_impl(self, gamma, quadpoints):
        g2 = np.zeros((quadpoints.size, 2))
        self.gamma_2d_impl(g2, quadpoints)
        phi, theta = self._surface_lin_inputs_from_gamma2d(g2)
        self.surf.gamma_lin(gamma, phi, theta)

    def dgamma_2d_by_dcoeff(self):
        cdofs = _as_jax_float64(self.get_dofs())
        return self.dgamma_2d_by_dcoeff_jax(cdofs)

    def dgamma_by_dcoeff(self):
        g2 = self.gamma_2d()
        phi, theta = self._surface_lin_inputs_from_gamma2d(g2)
        dsurf_dphi = np.zeros((self.numquadpoints, 3))  # shape nqpts x 3
        dsurf_dtheta = np.zeros((self.numquadpoints, 3))  # shape nqpts x 3
        self._surface_gammadash1_lin(dsurf_dphi, phi, theta)
        self._surface_gammadash2_lin(dsurf_dtheta, phi, theta)

        dg2_by_dcoeff = _as_numpy_float64(
            self.dgamma_2d_by_dcoeff()
        )  # shape nqpts x 2 x ndofs
        dphi_by_dcoeff = dg2_by_dcoeff[:, 0, :]  # shape nqpts x ndofs
        dtheta_by_dcoeff = dg2_by_dcoeff[:, 1, :]  # shape nqpts x ndofs

        # Evaluate dgamma_by_dcoeff, size nqpts x 3 x ndofs
        return np.einsum("ij,ik->ijk", dsurf_dphi, dphi_by_dcoeff) + np.einsum(
            "ij,ik->ijk", dsurf_dtheta, dtheta_by_dcoeff
        )

    def dgamma_by_dcoeff_impl(self, v):
        v[:, :, :] = self.dgamma_by_dcoeff()

    def dgamma_by_dcoeff_vjp(self, v):
        return self._derivative_from_curve_surface_vjps(
            self.dgamma_by_dcoeff_vjp_jax,
            self.dgamma_by_dsurf_vjp_jax,
            v,
        )

    def dgamma_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgamma_by_dcoeff(), v)

    # =========================================================================
    # GAMMADASH
    # ---------
    def gammadash_2d(self):
        cdofs = _as_jax_float64(self.get_dofs())
        return self.gammadash_2d_jax(cdofs)

    def dgammadash_2d_by_dcoeff(self):
        cdofs = _as_jax_float64(self.get_dofs())
        return self.dgammadash_2d_by_dcoeff_jax(cdofs)

    def gammadash(self):
        g2 = gamma_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        phi, theta = self._surface_lin_inputs_from_gamma2d(g2)
        dsurf_dphi = np.zeros((self.numquadpoints, 3))  # shape nqpts x 3
        dsurf_dtheta = np.zeros((self.numquadpoints, 3))  # shape nqpts x 3
        self._surface_gammadash1_lin(dsurf_dphi, phi, theta)
        self._surface_gammadash2_lin(dsurf_dtheta, phi, theta)

        g2dash = gammadash_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        phidash = g2dash[:, 0]  # shape nqpts
        thetadash = g2dash[:, 1]  # shape nqpts

        # Evaluate dgamma_by_dcoeff, size nqpts x 3
        return np.einsum("ij,i->ij", dsurf_dphi, phidash) + np.einsum(
            "ij,i->ij", dsurf_dtheta, thetadash
        )

    def gammadash_impl(self, gammadash):
        gammadash[:, :] = self.gammadash()

    def dgammadash_by_dcoeff(self):  # dgammadash by dcoeff
        g2 = self.gamma_2d()
        phi, theta = self._surface_lin_inputs_from_gamma2d(g2)
        dsurf_dphi = np.zeros((self.numquadpoints, 3))
        dsurf_dtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dphidphi = np.zeros((self.numquadpoints, 3))
        dsurf_dphidtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dthetadtheta = np.zeros((self.numquadpoints, 3))
        self._surface_gammadash1_lin(dsurf_dphi, phi, theta)
        self._surface_gammadash2_lin(dsurf_dtheta, phi, theta)
        self._surface_gammadash1dash1_lin(dsurf_dphidphi, phi, theta)
        self._surface_gammadash1dash2_lin(dsurf_dphidtheta, phi, theta)
        self._surface_gammadash2dash2_lin(dsurf_dthetadtheta, phi, theta)

        g2dash = _as_numpy_float64(self.gammadash_2d())
        phidash = g2dash[:, 0]
        thetadash = g2dash[:, 1]

        dg2_by_dcoef = _as_numpy_float64(self.dgamma_2d_by_dcoeff())
        dphi_by_dcoef = dg2_by_dcoef[:, 0, :]
        dtheta_by_dcoef = dg2_by_dcoef[:, 1, :]

        dg2dash_by_dcoeff = _as_numpy_float64(self.dgammadash_2d_by_dcoeff())
        dphidash_by_dcoeff = dg2dash_by_dcoeff[:, 0, :]  # shape nqpts x ndofs
        dthetadash_by_dcoeff = dg2dash_by_dcoeff[:, 1, :]  # shape nqpts x ndofs

        # Evaluate dgamma_by_dcoeff, size nqpts x 3 x ndofs
        return (
            np.einsum("ij,ik->ijk", dsurf_dphi, dphidash_by_dcoeff)
            + np.einsum("ij,ik->ijk", dsurf_dtheta, dthetadash_by_dcoeff)
            + np.einsum("ij,i,ik->ijk", dsurf_dphidphi, phidash, dphi_by_dcoef)
            + np.einsum("ij,i,ik->ijk", dsurf_dphidtheta, phidash, dtheta_by_dcoef)
            + np.einsum("ij,i,ik->ijk", dsurf_dphidtheta, thetadash, dphi_by_dcoef)
            + np.einsum("ij,i,ik->ijk", dsurf_dthetadtheta, thetadash, dtheta_by_dcoef)
        )

    def dgammadash_by_dcoeff_impl(self, v):
        v[:, :, :] = self.dgammadash_by_dcoeff()

    def dgammadash_by_dcoeff_vjp(self, v):
        return self._derivative_from_curve_surface_vjps(
            self.dgammadash_by_dcoeff_vjp_jax,
            self.dgammadash_by_dsurf_vjp_jax,
            v,
        )

    def dgammadash_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgammadash_by_dcoeff(), v)

    # =========================================================================
    # GAMMADASHDASH
    # -------------
    def gammadashdash(self):
        g2 = gamma_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        phi, theta = self._surface_lin_inputs_from_gamma2d(g2)
        dsurf_dphi = np.zeros((self.numquadpoints, 3))
        dsurf_dtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dphidphi = np.zeros((self.numquadpoints, 3))
        dsurf_dphidtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dthetadtheta = np.zeros((self.numquadpoints, 3))
        self._surface_gammadash1_lin(dsurf_dphi, phi, theta)
        self._surface_gammadash2_lin(dsurf_dtheta, phi, theta)
        self._surface_gammadash1dash1_lin(dsurf_dphidphi, phi, theta)
        self._surface_gammadash1dash2_lin(dsurf_dphidtheta, phi, theta)
        self._surface_gammadash2dash2_lin(dsurf_dthetadtheta, phi, theta)

        g2dash = gammadash_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        phidash = g2dash[:, 0]  # self.numquadpoints
        thetadash = g2dash[:, 1]  # self.numquadpoints

        g2dashdash = gammadashdash_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        phidashdash = g2dashdash[:, 0]  # self.numquadpoints
        thetadashdash = g2dashdash[:, 1]  # self.numquadpoints

        return (
            np.einsum("ij,i->ij", dsurf_dphidphi, phidash**2)
            + np.einsum("ij,i->ij", dsurf_dthetadtheta, thetadash**2)
            + 2 * np.einsum("ij,i,i->ij", dsurf_dphidtheta, phidash, thetadash)
            + np.einsum("ij,i->ij", dsurf_dphi, phidashdash)
            + np.einsum("ij,i->ij", dsurf_dtheta, thetadashdash)
        )

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:, :] = self.gammadashdash()

    def dgammadashdash_by_dcoeff(self):
        # This is ugly, but I don't know how to make it better!
        g2 = self.gamma_2d()
        phi, theta = self._surface_lin_inputs_from_gamma2d(g2)

        ## First order derivative
        dsurf_dphi = np.zeros((self.numquadpoints, 3))
        dsurf_dtheta = np.zeros((self.numquadpoints, 3))
        self._surface_gammadash1_lin(dsurf_dphi, phi, theta)
        self._surface_gammadash2_lin(dsurf_dtheta, phi, theta)

        ## Second order derivative
        dsurf_dphidphi = np.zeros((self.numquadpoints, 3))
        dsurf_dphidtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dthetadtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dphidphidphi = np.zeros((self.numquadpoints, 3))
        self._surface_gammadash1dash1_lin(dsurf_dphidphi, phi, theta)
        self._surface_gammadash1dash2_lin(dsurf_dphidtheta, phi, theta)
        self._surface_gammadash2dash2_lin(dsurf_dthetadtheta, phi, theta)

        ## Third order derivative
        dsurf_dphidphidtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dphidthetadtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dthetadthetadtheta = np.zeros((self.numquadpoints, 3))
        self._surface_gammadash1dash1dash1_lin(dsurf_dphidphidphi, phi, theta)
        self._surface_gammadash1dash1dash2_lin(dsurf_dphidphidtheta, phi, theta)
        self._surface_gammadash1dash2dash2_lin(dsurf_dphidthetadtheta, phi, theta)
        self._surface_gammadash2dash2dash2_lin(dsurf_dthetadthetadtheta, phi, theta)

        cdofs = _as_jax_float64(self.get_dofs())
        dg2_by_dcoef = _as_numpy_float64(self.dgamma_2d_by_dcoeff_jax(cdofs))
        dphi_by_dcoef = dg2_by_dcoef[:, 0, :]
        dtheta_by_dcoef = dg2_by_dcoef[:, 1, :]

        g2dash = _as_numpy_float64(self.gammadash_2d_jax(cdofs))
        phidash = g2dash[:, 0]  # self.numquadpoints
        thetadash = g2dash[:, 1]  # self.numquadpoints

        g2dashdash = _as_numpy_float64(self.gammadashdash_2d_jax(cdofs))
        phidashdash = g2dashdash[:, 0]  # self.numquadpoints
        thetadashdash = g2dashdash[:, 1]  # self.numquadpoints

        dg2dash_by_dcoeff = _as_numpy_float64(self.dgammadash_2d_by_dcoeff_jax(cdofs))
        dphidash_by_dcoeff = dg2dash_by_dcoeff[:, 0]  # self.numquadpoints
        dthetadash_by_dcoeff = dg2dash_by_dcoeff[:, 1]  # self.numquadpoints

        dg2dashdash_by_dcoeff = _as_numpy_float64(
            self.dgammadashdash_2d_by_dcoeff_jax(cdofs)
        )
        dphidashdash_by_dcoeff = dg2dashdash_by_dcoeff[:, 0]  # self.numquadpoints
        dthetadashdash_by_dcoeff = dg2dashdash_by_dcoeff[:, 1]  # self.numquadpoints

        # l1-l6 denotes lines in my hand-written notes...
        l1 = (
            np.einsum(
                "ij,ik,i->ijk", dsurf_dthetadthetadtheta, dtheta_by_dcoef, thetadash**2
            )
            + np.einsum(
                "ij,ik,i,i->ijk",
                dsurf_dphidthetadtheta,
                dtheta_by_dcoef,
                thetadash,
                phidash,
            )
            + np.einsum(
                "ij,ik,i->ijk", dsurf_dthetadtheta, dthetadash_by_dcoeff, thetadash
            )
            + np.einsum(
                "ij,ik,i->ijk", dsurf_dthetadtheta, dtheta_by_dcoef, thetadashdash
            )
        )

        l2 = (
            np.einsum(
                "ij,ik,i,i->ijk",
                dsurf_dphidthetadtheta,
                dtheta_by_dcoef,
                thetadash,
                phidash,
            )
            + np.einsum(
                "ij,ik,i->ijk", dsurf_dphidphidtheta, dtheta_by_dcoef, phidash**2
            )
            + np.einsum("ij,ik,i->ijk", dsurf_dphidtheta, dthetadash_by_dcoeff, phidash)
            + np.einsum("ij,ik,i->ijk", dsurf_dphidtheta, dtheta_by_dcoef, phidashdash)
        )

        l3 = (
            np.einsum(
                "ij,ik,i->ijk", dsurf_dthetadtheta, dthetadash_by_dcoeff, thetadash
            )
            + np.einsum("ij,ik,i->ijk", dsurf_dphidtheta, dthetadash_by_dcoeff, phidash)
            + np.einsum("ij,ik->ijk", dsurf_dtheta, dthetadashdash_by_dcoeff)
        )

        l4 = (
            np.einsum(
                "ij,ik,i,i->ijk",
                dsurf_dphidphidtheta,
                dphi_by_dcoef,
                thetadash,
                phidash,
            )
            + np.einsum("ij,ik,i->ijk", dsurf_dphidphidphi, dphi_by_dcoef, phidash**2)
            + np.einsum("ij,ik,i->ijk", dsurf_dphidphi, dphidash_by_dcoeff, phidash)
            + np.einsum("ij,ik,i->ijk", dsurf_dphidphi, dphi_by_dcoef, phidashdash)
        )

        l5 = (
            np.einsum(
                "ij,ik,i->ijk", dsurf_dphidthetadtheta, dphi_by_dcoef, thetadash**2
            )
            + np.einsum(
                "ij,ik,i,i->ijk",
                dsurf_dphidphidtheta,
                dphi_by_dcoef,
                phidash,
                thetadash,
            )
            + np.einsum("ij,ik,i->ijk", dsurf_dphidtheta, dphidash_by_dcoeff, thetadash)
            + np.einsum("ij,ik,i->ijk", dsurf_dphidtheta, dphi_by_dcoef, thetadashdash)
        )

        l6 = (
            np.einsum("ij,ik,i->ijk", dsurf_dphidtheta, dphidash_by_dcoeff, thetadash)
            + np.einsum("ij,ik,i->ijk", dsurf_dphidphi, dphidash_by_dcoeff, phidash)
            + np.einsum("ij,ik->ijk", dsurf_dphi, dphidashdash_by_dcoeff)
        )

        return l1 + l2 + l3 + l4 + l5 + l6

    def dgammadashdash_by_dcoeff_impl(self, v):
        v[:, :, :] = self.dgammadashdash_by_dcoeff()

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self._derivative_from_curve_surface_vjps(
            self.dgammadashdash_by_dcoeff_vjp_jax,
            self.dgammadashdash_by_dsurf_vjp_jax,
            v,
        )

    def dgammadashdash_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgammadashdash_by_dcoeff(), v)

    # =========================================================================
    # GAMMADASHDASHDASH
    # -----------------
    def gammadashdashdash(self):
        cdofs = _as_jax_float64(self.get_dofs())
        return _as_numpy_float64(self.gammacdashdashdash_jax(cdofs))

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:, :] = self.gammadashdashdash()

    def dgammadashdashdash_by_dcoeff(self):
        cdofs = _as_jax_float64(self.get_dofs())
        return _as_numpy_float64(self.dgammadashdashdash_by_dcoeff_jax(cdofs))

    def dgammadashdashdash_by_dcoeff_impl(self, v):
        v[:, :, :] = self.dgammadashdashdash_by_dcoeff()

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self._derivative_from_curve_surface_vjps(
            self.dgammadashdashdash_by_dcoeff_vjp_jax,
            self.dgammadashdashdash_by_dsurf_vjp_jax,
            v,
        )

    def dgammadashdashdash_by_dcoeff_vjp_impl(self, v):
        cdofs = _as_jax_float64(self.get_dofs())
        v_jax = _as_jax_float64(v)
        return _as_numpy_float64(
            self.dgammadashdashdash_by_dcoeff_vjp_jax(cdofs, v_jax)
        )

    def dkappa_by_dcoeff_vjp(self, v):
        cdofs = _as_jax_float64(self.get_dofs())
        sdofs = _as_jax_float64(self.surf.get_dofs())
        v_jax = _as_jax_float64(v)

        def kappa_from_dofs(curve_dofs, surface_dofs):
            d1, d2, _d3 = self._geometry_derivatives_jax(curve_dofs, surface_dofs)
            return _kappa_pure(d1, d2)

        _kappa, pullback = vjp(kappa_from_dofs, cdofs, sdofs)
        curve_cotangent, surface_cotangent = pullback(v_jax)
        return Derivative(
            {
                self: _as_numpy_float64(curve_cotangent),
                self.surf: _as_numpy_float64(surface_cotangent),
            }
        )

    def dtorsion_by_dcoeff_vjp(self, v):
        cdofs = _as_jax_float64(self.get_dofs())
        sdofs = _as_jax_float64(self.surf.get_dofs())
        v_jax = _as_jax_float64(v)

        def torsion_from_dofs(curve_dofs, surface_dofs):
            return _torsion_pure(
                *self._geometry_derivatives_jax(curve_dofs, surface_dofs)
            )

        _torsion, pullback = vjp(torsion_from_dofs, cdofs, sdofs)
        curve_cotangent, surface_cotangent = pullback(v_jax)
        return Derivative(
            {
                self: _as_numpy_float64(curve_cotangent),
                self.surf: _as_numpy_float64(surface_cotangent),
            }
        )

    # =========================================================================
    # NORMAL
    # ------
    def unit_normal(self):
        g2 = gamma_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        return self.unit_normal_impl(g2[:, 0], g2[:, 1])

    def unit_normal_impl(self, phi, theta):
        phi, theta = self._surface_lin_inputs(phi, theta)
        npts = phi.size
        dxdtheta = np.zeros((npts, 3))
        dxdphi = np.zeros((npts, 3))
        self._surface_gammadash1_lin(dxdphi, phi, theta)
        self._surface_gammadash2_lin(dxdtheta, phi, theta)

        normal = np.cross(dxdphi, dxdtheta)
        unit_normal = normal / np.linalg.norm(normal, axis=1)[:, None]
        return unit_normal

    def dunit_normal_by_dcoeff(self):
        g2 = gamma_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        phi, theta = self._surface_lin_inputs_from_gamma2d(g2)
        dxdtheta = np.zeros((self.numquadpoints, 3))
        dxdphi = np.zeros((self.numquadpoints, 3))
        self._surface_gammadash1_lin(dxdphi, phi, theta)
        self._surface_gammadash2_lin(dxdtheta, phi, theta)

        normal = np.cross(dxdphi, dxdtheta)
        normal_norm = np.linalg.norm(normal, axis=1)

        dg2_by_dcoeff = _as_numpy_float64(self.dgamma_2d_by_dcoeff())
        dxdthetadtheta = np.zeros((self.numquadpoints, 3))
        dxdphidtheta = np.zeros((self.numquadpoints, 3))
        dxdphidphi = np.zeros((self.numquadpoints, 3))
        self._surface_gammadash1dash1_lin(dxdphidphi, phi, theta)
        self._surface_gammadash1dash2_lin(dxdphidtheta, phi, theta)
        self._surface_gammadash2dash2_lin(dxdthetadtheta, phi, theta)

        p0 = np.cross(dxdphi, dxdphidtheta) + np.cross(dxdphidphi, dxdtheta)
        p1 = np.cross(dxdphi, dxdthetadtheta) + np.cross(dxdphidtheta, dxdtheta)
        dnormal_by_dcoeff = np.einsum(
            "ik,ij->ijk", dg2_by_dcoeff[:, 0, :], p0
        ) + np.einsum("ik,ij->ijk", dg2_by_dcoeff[:, 1, :], p1)

        t1 = np.einsum(
            "ijk,i->ijk", dnormal_by_dcoeff, 1.0 / normal_norm
        )  # this has shape (nqpts,3,ndofs)
        prod = np.einsum("ij,ijk->ik", normal, dnormal_by_dcoeff)
        t2 = np.einsum("ik,ij,i->ijk", prod, normal, normal_norm ** (-3))
        dunit_normal_by_dcoef = t1 - t2
        return dunit_normal_by_dcoef

    def zfactor(self):
        return self.sgn_z * self.unit_normal()[:, 2]

    def dzfactor_by_dcoeff(self):
        return self.sgn_z * self.dunit_normal_by_dcoeff()[:, 2, :]

    def dzfactor_by_dcoeff_vjp(self, v):
        return Derivative({self: vjp_contraction_1d(self.dzfactor_by_dcoeff(), v)})

    def rfactor(self):
        g2 = gamma_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        unit_normal = (
            self.unit_normal()
        )  # negative sign to point outside the surface...

        # Now project in the radial direction...
        return self.sgn_r * (
            unit_normal[:, 0] * np.cos(g2[:, 0]) + unit_normal[:, 1] * np.sin(g2[:, 0])
        )

    def drfactor_by_dcoeff(self):
        g2 = gamma_2d_numpy(
            self.get_dofs(), self.quadpoints, self.order, self.G, self.H
        )
        dg2_by_dcoef = _as_numpy_float64(self.dgamma_2d_by_dcoeff())
        unit_normal = self.unit_normal()
        dunit_normal_by_dcoef = self.dunit_normal_by_dcoeff()

        # Now project in the radial direction...
        return self.sgn_r * (
            dunit_normal_by_dcoef[:, 0, :] * np.cos(g2[:, 0, None])
            + dunit_normal_by_dcoef[:, 1, :] * np.sin(g2[:, 0, None])
            + dg2_by_dcoef[:, 0, :]
            * (
                -unit_normal[:, 0, None] * np.sin(g2[:, 0, None])
                + unit_normal[:, 1, None] * np.cos(g2[:, 0, None])
            )
        )

    def drfactor_by_dcoeff_vjp(self, v):
        return Derivative({self: vjp_contraction_1d(self.drfactor_by_dcoeff(), v)})


_LEGACY_ARTIFACT_KEYS = ("idofs", "mpol", "nfp", "ntor", "stellsym")


def legacy_cws_dofs_to_modes(legacy_dofs, legacy_free, order):
    """Convert a pre-2024 ``CurveCWSFourier`` dof vector to this class' encoding.

    The upstream class that wrote those artifacts (simsopt ``20265c3fa``,
    ``src/simsoptpp/curvecwsfourier.h``) carried ``2 * (2 * order + 1) + 2`` dofs
    laid out as ``[theta_l, theta_c[0..order], theta_s[1..order], phi_l,
    phi_c[0..order], phi_s[1..order]]`` and evaluated, with angles in radians,
    ``theta = sum_m theta_c[m] cos(2 pi m t) + ... + theta_l * (2 pi t)``.
    This class carries ``2 * (2 * order + 1)`` dofs laid out as
    ``[phi_c, phi_s, theta_c, theta_s]``, holds the secular coefficients as the
    constructor arguments ``G`` (theta) and ``H`` (phi), and works in turns --
    the winding surface multiplies ``(phi, theta)`` by ``2 pi``
    (``simsopt_jax.core.curve_kernels._surface_rz_fourier_gamma_pointwise``).
    The two encodings are therefore the same curve under a reordering, a
    ``1 / (2 pi)`` rescaling of the harmonics, and the exact identifications
    ``G = theta_l`` and ``H = phi_l``.

    The free/fixed mask travels with the values under the same reordering, with
    one exception that this function rejects rather than silently drops: the
    secular pair becomes ``G``/``H``, which are constructor arguments and not
    dofs, so a legacy artifact that left ``theta_l`` or ``phi_l`` free is asking
    for an optimization this class cannot perform. Non-integral secular values
    are rejected for the same reason: ``G`` and ``H`` are whole winding numbers
    (``gamma_2d`` adds ``G * t`` with ``G`` taken as an integer), so a fractional
    ``theta_l`` would be silently truncated.

    Args:
        legacy_dofs: The legacy dof vector, of size ``2 * (2 * order + 1) + 2``.
        legacy_free: The legacy free/fixed mask, of the same size.
        order: Maximum Fourier order of the curve.

    Returns:
        ``(modes, free, G, H)`` for :class:`CurveCWSFourierCPP`, where ``modes``
        and ``free`` are the ``2 * (2 * order + 1)`` harmonic values and their
        free flags in this class' order.
    """
    theta_block, phi_block = np.asarray(legacy_dofs, dtype=np.float64).reshape(
        2, 2 * order + 2
    )
    theta_free, phi_free = np.asarray(legacy_free, dtype=bool).reshape(2, 2 * order + 2)
    secular_theta = theta_block[0]
    secular_phi = phi_block[0]
    if not np.isfinite(secular_theta) or not np.isfinite(secular_phi):
        raise ValueError(
            "Legacy CurveCWSFourier secular coefficients (theta_l="
            f"{secular_theta}, phi_l={secular_phi}) are not finite; "
            "CurveCWSFourier represents them as the integer winding numbers "
            "G and H."
        )
    if secular_theta != int(secular_theta) or secular_phi != int(secular_phi):
        raise ValueError(
            "Legacy CurveCWSFourier secular coefficients (theta_l="
            f"{secular_theta}, phi_l={secular_phi}) are not whole turns; "
            "CurveCWSFourier represents them as the integer winding numbers "
            "G and H."
        )
    if theta_free[0] or phi_free[0]:
        raise ValueError(
            "Legacy CurveCWSFourier secular coefficients (theta_l free="
            f"{bool(theta_free[0])}, phi_l free={bool(phi_free[0])}) are free "
            "dofs; CurveCWSFourier carries them as the constructor arguments G "
            "and H, which are not dofs and cannot be optimized."
        )
    modes = np.concatenate((phi_block[1:], theta_block[1:])) / (2.0 * np.pi)
    free = np.concatenate((phi_free[1:], theta_free[1:]))
    return modes, free, int(secular_theta), int(secular_phi)


class CurveCWSFourier(CurveCWSFourierCPP):
    """A curve on a winding surface that reads both serialized shapes.

    Modern artifacts are written by the inherited ``GSONable.as_dict`` (via
    :class:`~simsopt._core.optimizable.Optimizable`), which emits this class'
    constructor arguments -- ``quadpoints``, ``order``, ``surf``, ``G``, ``H`` --
    plus the ``dofs`` block; :meth:`from_dict` hands that dict straight back to
    the inherited ``GSONable.from_dict``, so the ``surf`` entry round-trips by
    reference and a loaded curve gets the saved winding surface itself (its
    quadpoints, DOFs and identity with any object that shares it).

    Legacy artifacts (upstream simsopt ``20265c3fa`` and earlier, e.g.
    ``examples/3_Advanced/optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus``)
    instead carry ``idofs``, ``mpol``, ``nfp``, ``ntor`` and ``stellsym``, the
    arguments of the constructor that class had. That shape is recognised by the
    presence of those keys and rebuilt by :meth:`_from_legacy_dict`:
    the winding surface from ``nfp``/``stellsym``/``mpol``/``ntor``/``idofs``,
    the curve modes and the ``G``/``H`` winding numbers from the legacy dof
    vector (see :func:`legacy_cws_dofs_to_modes`), and the saved ``quadpoints``
    array verbatim -- the legacy constructor kept only its length.

    The legacy free/fixed mask is carried over with the values, reordered the
    same way; a legacy artifact whose secular dofs are free, or whose secular
    values are non-integral or non-finite, is rejected with a ``ValueError``
    rather than silently reinterpreted (see :func:`legacy_cws_dofs_to_modes`).

    Three things a legacy dict cannot express, and which are therefore
    reconstructed at their defaults: the winding surface's ``quadpoints_phi`` /
    ``quadpoints_theta`` (legacy artifacts store no surface grid) and its
    identity with other objects (they store surface dofs inline, not a
    reference); a non-``SurfaceRZFourier`` winding surface (the legacy class
    only implemented that one); and the per-dof names and bounds (legacy names
    are positional ``x0..xN`` over a different layout, so this class' own
    ``phic``/``phis``/``thetac``/``thetas`` names and default bounds are used).
    """

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        if all(key in d for key in _LEGACY_ARTIFACT_KEYS):
            return cls._from_legacy_dict(d, serial_objs_dict, recon_objs)
        return super().from_dict(d, serial_objs_dict, recon_objs)

    @classmethod
    def _from_legacy_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        quadpoints = np.asarray(
            decoder.process_decoded(d["quadpoints"], serial_objs_dict, recon_objs),
            dtype=np.float64,
        )
        dofs = decoder.process_decoded(d["dofs"], serial_objs_dict, recon_objs)
        order = int(d["order"])
        surf = SurfaceRZFourier(
            nfp=int(d["nfp"]),
            stellsym=bool(d["stellsym"]),
            mpol=int(d["mpol"]),
            ntor=int(d["ntor"]),
        )
        surf.set_dofs(np.asarray(d["idofs"], dtype=np.float64))
        modes, free, secular_theta, secular_phi = legacy_cws_dofs_to_modes(
            dofs.full_x, dofs.free_status, order
        )
        curve = cls(
            quadpoints=quadpoints,
            order=order,
            surf=surf,
            G=secular_theta,
            H=secular_phi,
        )
        curve.local_full_x = modes
        for index in np.flatnonzero(~free):
            curve.fix(int(index))
        return curve
