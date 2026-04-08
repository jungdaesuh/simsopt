import numpy as np
import jax
from jax import device_put, jacfwd, jvp, vjp
import jax.numpy as jnp

from .._core.derivative import Derivative
from ._simsoptpp import sopp_namespace
from .curve import (
    Curve,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    gamma_curve_on_surface,
)

from .jit import jit

sopp = sopp_namespace("Curve")

__all__ = ["CurveCWSFourierCPP"]


def _as_jax_float64(value):
    if hasattr(value, "devices"):
        return jnp.asarray(value, dtype=jnp.float64)
    return device_put(np.asarray(value, dtype=np.float64))


def _as_numpy_float64(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if hasattr(value, "devices"):
        return np.asarray(jax.device_get(value), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


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


def vjp_contraction_1d(mat, v):
    # contract matrix of size ijk times vector of size jk into array of size i
    return np.einsum("ij,i->j", mat, v)


def vjp_contraction_2d(mat, v):
    # contract matrix of size ijk times vector of size jk into array of size i
    return np.einsum("ijk,ij->k", mat, v)


class CurveCWSFourierCPP(Curve, sopp.Curve):
    def __init__(self, quadpoints, order, surf, G=0, H=0, **kwargs):
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

        def _bind_live_curve(fun):
            def bound(sdofs):
                return fun(current_curve_dofs(), sdofs)

            return bound

        def _bind_live_curve_vjp(fun):
            def bound(sdofs, v):
                return fun(current_curve_dofs(), sdofs, v)

            return bound

        self.gamma_pure = jit(gamma_on_surface)

        def gamma_at_points(cdofs, sdofs):
            return self.gamma_pure(cdofs, sdofs, points)

        self.gamma_jax = jit(gamma_at_points)
        self.gamma_impl_jax = jit(
            lambda cdofs, sdofs, qpts: self.gamma_pure(cdofs, sdofs, qpts)
        )
        # Keep the bound one-arg adapters outside jit so they re-read live DOFs.
        self.gammac_jax = _bind_live_surface(self.gamma_jax)
        self.gammas_jax = _bind_live_curve(self.gamma_jax)
        dgamma_by_dcoeff_kernel = jit(jacfwd(gamma_at_points, argnums=0))
        dgamma_by_dcoeff_vjp_kernel = _arg0_vjp_kernel(gamma_at_points)
        dgamma_by_dsurf_kernel = jit(jacfwd(gamma_at_points, argnums=1))
        dgamma_by_dsurf_vjp_kernel = _arg1_vjp_kernel(gamma_at_points)
        self.dgamma_by_dcoeff_jax = _bind_live_surface(dgamma_by_dcoeff_kernel)
        self.dgamma_by_dcoeff_vjp_jax = _bind_live_surface_vjp(
            dgamma_by_dcoeff_vjp_kernel
        )
        self.dgamma_by_dsurf_jax = _bind_live_curve(dgamma_by_dsurf_kernel)
        self.dgamma_by_dsurf_vjp_jax = _bind_live_curve_vjp(dgamma_by_dsurf_vjp_kernel)

        self.gammadash_pure = jit(gammadash_on_surface)

        def gammadash_at_points(cdofs, sdofs):
            return self.gammadash_pure(cdofs, sdofs, points)

        self.gammadash_jax = jit(gammadash_at_points)
        self.gammacdash_jax = _bind_live_surface(self.gammadash_jax)
        self.gammasdash_jax = _bind_live_curve(self.gammadash_jax)
        dgammadash_by_dcoeff_kernel = jit(jacfwd(gammadash_at_points, argnums=0))
        dgammadash_by_dcoeff_vjp_kernel = _arg0_vjp_kernel(gammadash_at_points)
        dgammadash_by_dsurf_kernel = jit(jacfwd(gammadash_at_points, argnums=1))
        dgammadash_by_dsurf_vjp_kernel = _arg1_vjp_kernel(gammadash_at_points)
        self.dgammadash_by_dcoeff_jax = _bind_live_surface(dgammadash_by_dcoeff_kernel)
        self.dgammadash_by_dcoeff_vjp_jax = _bind_live_surface_vjp(
            dgammadash_by_dcoeff_vjp_kernel
        )
        self.dgammadash_by_dsurf_jax = _bind_live_curve(dgammadash_by_dsurf_kernel)
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
        self.gammacdashdash_jax = _bind_live_surface(self.gammadashdash_jax)
        self.gammasdashdash_jax = _bind_live_curve(self.gammadashdash_jax)
        dgammadashdash_by_dcoeff_kernel = jit(
            jacfwd(gammadashdash_at_points, argnums=0)
        )
        dgammadashdash_by_dcoeff_vjp_kernel = _arg0_vjp_kernel(gammadashdash_at_points)
        dgammadashdash_by_dsurf_kernel = jit(jacfwd(gammadashdash_at_points, argnums=1))
        dgammadashdash_by_dsurf_vjp_kernel = _arg1_vjp_kernel(gammadashdash_at_points)
        self.dgammadashdash_by_dcoeff_jax = _bind_live_surface(
            dgammadashdash_by_dcoeff_kernel
        )
        self.dgammadashdash_by_dcoeff_vjp_jax = _bind_live_surface_vjp(
            dgammadashdash_by_dcoeff_vjp_kernel
        )
        self.dgammadashdash_by_dsurf_jax = _bind_live_curve(
            dgammadashdash_by_dsurf_kernel
        )
        self.dgammadashdash_by_dsurf_vjp_jax = _bind_live_curve_vjp(
            dgammadashdash_by_dsurf_vjp_kernel
        )

        # The CPU contract for CurveCWSFourierCPP stops at gammadashdash():
        # this Python-defined curve uses the generic sopp.Curve hook, and that
        # hook has no raw gammadashdashdash_impl for this class. Keep the JAX
        # third derivative only as an internal support primitive for composed
        # JAX paths such as finite-build wrappers and BiotSavartJAX pullbacks.
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
        self.gammasdashdashdash_jax = _bind_live_curve(self.gammadashdashdash_jax)
        dgammadashdashdash_by_dcoeff_kernel = jit(
            jacfwd(gammadashdashdash_at_points, argnums=0)
        )
        dgammadashdashdash_by_dcoeff_vjp_kernel = _arg0_vjp_kernel(
            gammadashdashdash_at_points
        )
        dgammadashdashdash_by_dsurf_kernel = jit(
            jacfwd(gammadashdashdash_at_points, argnums=1)
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
        self.dgammadashdashdash_by_dsurf_jax = _bind_live_curve(
            dgammadashdashdash_by_dsurf_kernel
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
        self.dgamma_2d_by_dcoeff_vjp = jit(
            lambda cdofs, v: vjp(self.gamma_2d_jax, cdofs)[1](v)[0]
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
        self.dgammadash_2d_by_dcoeff_vjp = jit(
            lambda cdofs, v: vjp(self.gammadash_2d_jax, cdofs)[1](v)[0]
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
        self.dgammadashdash_2d_by_dcoeff_vjp = jit(
            lambda cdofs, v: vjp(self.gammadashdash_2d_jax, cdofs)[1](v)[0]
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

    def get_dofs(self):
        return np.concatenate(self.modes)

    def set_dofs_impl(self, dofs):
        self.modes[0] = dofs[0 : self.order + 1]
        self.modes[1] = dofs[self.order + 1 : 2 * self.order + 1]
        self.modes[2] = dofs[2 * self.order + 1 : 3 * self.order + 2]
        self.modes[3] = dofs[3 * self.order + 2 : 4 * self.order + 2]

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
        self.surf.gammadash1_lin(dsurf_dphi, phi, theta)
        self.surf.gammadash2_lin(dsurf_dtheta, phi, theta)

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
        return Derivative({self: self.dgamma_by_dcoeff_vjp_impl(v)})

    def dgamma_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgamma_by_dcoeff(), v)

    # =========================================================================
    # GAMMADASH
    # ---------
    def gammadash_2d(self):
        cdofs = _as_jax_float64(self.get_dofs())
        return self.gammadash_2d_jax(cdofs)

    def gammadash_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:, :] = gammadash_2d_numpy(cdofs, quadpoints, self.order, self.G, self.H)

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
        self.surf.gammadash1_lin(dsurf_dphi, phi, theta)
        self.surf.gammadash2_lin(dsurf_dtheta, phi, theta)

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
        self.surf.gammadash1_lin(dsurf_dphi, phi, theta)
        self.surf.gammadash2_lin(dsurf_dtheta, phi, theta)
        self.surf.gammadash1dash1_lin(dsurf_dphidphi, phi, theta)
        self.surf.gammadash1dash2_lin(dsurf_dphidtheta, phi, theta)
        self.surf.gammadash2dash2_lin(dsurf_dthetadtheta, phi, theta)

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
        return Derivative({self: self.dgammadash_by_dcoeff_vjp_impl(v)})

    def dgammadash_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgammadash_by_dcoeff(), v)

    # =========================================================================
    # GAMMADASHDASH
    # -------------
    def gammadashdash_2d(self):
        cdofs = _as_jax_float64(self.get_dofs())
        return self.gammadashdash_2d_jax(cdofs)

    def gammadashdash_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:, :] = gammadashdash_2d_numpy(cdofs, quadpoints, self.order, self.G, self.H)

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
        self.surf.gammadash1_lin(dsurf_dphi, phi, theta)
        self.surf.gammadash2_lin(dsurf_dtheta, phi, theta)
        self.surf.gammadash1dash1_lin(dsurf_dphidphi, phi, theta)
        self.surf.gammadash1dash2_lin(dsurf_dphidtheta, phi, theta)
        self.surf.gammadash2dash2_lin(dsurf_dthetadtheta, phi, theta)

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
        self.surf.gammadash1_lin(dsurf_dphi, phi, theta)
        self.surf.gammadash2_lin(dsurf_dtheta, phi, theta)

        ## Second order derivative
        dsurf_dphidphi = np.zeros((self.numquadpoints, 3))
        dsurf_dphidtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dthetadtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dphidphidphi = np.zeros((self.numquadpoints, 3))
        self.surf.gammadash1dash1_lin(dsurf_dphidphi, phi, theta)
        self.surf.gammadash1dash2_lin(dsurf_dphidtheta, phi, theta)
        self.surf.gammadash2dash2_lin(dsurf_dthetadtheta, phi, theta)

        ## Third order derivative
        dsurf_dphidphidtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dphidthetadtheta = np.zeros((self.numquadpoints, 3))
        dsurf_dthetadthetadtheta = np.zeros((self.numquadpoints, 3))
        self.surf.gammadash1dash1dash1_lin(dsurf_dphidphidphi, phi, theta)
        self.surf.gammadash1dash1dash2_lin(dsurf_dphidphidtheta, phi, theta)
        self.surf.gammadash1dash2dash2_lin(dsurf_dphidthetadtheta, phi, theta)
        self.surf.gammadash2dash2dash2_lin(dsurf_dthetadthetadtheta, phi, theta)

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
        return Derivative({self: self.dgammadashdash_by_dcoeff_vjp_impl(v)})

    def dgammadashdash_by_dcoeff_vjp_impl(self, v):
        return vjp_contraction_2d(self.dgammadashdash_by_dcoeff(), v)

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
        self.surf.gammadash1_lin(dxdphi, phi, theta)
        self.surf.gammadash2_lin(dxdtheta, phi, theta)

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
        self.surf.gammadash1_lin(dxdphi, phi, theta)
        self.surf.gammadash2_lin(dxdtheta, phi, theta)

        normal = np.cross(dxdphi, dxdtheta)
        normal_norm = np.linalg.norm(normal, axis=1)

        dg2_by_dcoeff = _as_numpy_float64(self.dgamma_2d_by_dcoeff())
        dxdthetadtheta = np.zeros((self.numquadpoints, 3))
        dxdphidtheta = np.zeros((self.numquadpoints, 3))
        dxdphidphi = np.zeros((self.numquadpoints, 3))
        self.surf.gammadash1dash1_lin(dxdphidphi, phi, theta)
        self.surf.gammadash1dash2_lin(dxdphidtheta, phi, theta)
        self.surf.gammadash2dash2_lin(dxdthetadtheta, phi, theta)

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
