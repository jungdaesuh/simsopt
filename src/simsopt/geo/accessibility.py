from functools import partial

import numpy as np
from jax import grad, hessian, jacfwd, jacrev, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from .curve import Curve, CurveCWSFourier
from .curveobjectives import ArclengthVariation

from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative

from scipy.linalg import lu
from scipy.optimize import minimize

__all__ = [
    "PortSize",
    "ProjectedEnclosedArea",
    "ProjectedCurveCurveDistance",
    "ProjectedCurveConvexity",
    "DirectedFacingPort",
    "CurveInPortPenalty",
]


class PortSize(Optimizable):
    def __init__(
        self,
        port,
        curves,
        curve_port_distance_threshold,
        direction,
        solver="jax",
        port_coil_distance_weight=1e2,
        port_arc_penalty_weight=1e-2,
        port_forward_facing_weight=1e2,
    ):
        if not isinstance(port, CurveCWSFourier):
            raise ValueError("Port should be a CurveCWSFourier instance")
        if not isinstance(curves, list):
            raise ValueError("Curves should be a list")
        if len(curves) < 1:
            raise ValueError("Should have at least one curve")
        if any([not isinstance(c, Curve) for c in curves]):
            raise ValueError("Curves elements should be instances of curves")
        if not isinstance(curve_port_distance_threshold, (int, float)):
            raise ValueError(
                "Threshold value for curve to port distance should be an integer or a float"
            )
        if not ((direction == "radial") or (direction == "vertical")):
            raise ValueError("Only radial and vertical directions are implemented")
        if not ((solver == "jax") or (solver == "explicit")):
            raise ValueError("Solver should be either 'jax' or 'explicit'")

        self.port = port
        self.curves = curves
        self.solver = solver
        self.curve_port_distance_threshold = curve_port_distance_threshold
        self.direction = direction

        self.port_arc_penalty_weight = port_arc_penalty_weight
        self.port_coil_distance_weight = port_coil_distance_weight
        self.port_forward_facing_weight = port_forward_facing_weight

        # Define objective
        if self.direction == "radial":
            projection = "zphi"
        elif self.direction == "vertical":
            projection = "xy"

        self.Jxyarea = ProjectedEnclosedArea(self.port, projection=projection)
        self.Jccxydist = ProjectedCurveCurveDistance(
            self.curves,
            self.port,
            self.curve_port_distance_threshold,
            projection=projection,
        )
        self.Jarc = ArclengthVariation(self.port)
        self.Jufp = DirectedFacingPort(self.port, projection=projection)

        # self.Jport = -1*self.Jxyarea \
        #     + self.port_coil_distance_weight * self.Jccxydist \
        #     + self.port_forward_facing_weight * self.Jufp
        # + self.port_arc_penalty_weight * self.Jarc \

        # Initialize parent
        # Does NOT depend on port! Port is specified by this inner
        # optimization.
        super().__init__(depends_on=self.curves)

        # Some cache boolean
        self.need_to_run_code = True

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def objective(self, dofs, hessian_bool):
        self.port.x = dofs  # We only optimize the port dofs

        if hessian_bool:  # Don't include the arclength penalty here
            J = (
                -1 * self.Jxyarea.J()
                + self.port_coil_distance_weight * self.Jccxydist.J()
                + self.port_forward_facing_weight * self.Jufp.J()
            )

            dJ = (
                -1 * self.Jxyarea.dJ(partials=True).data[self.port]
                + self.port_coil_distance_weight
                * self.Jccxydist.dJ(partials=True).data[self.port]
                + self.port_forward_facing_weight
                * self.Jufp.dJ(partials=True).data[self.port]
            )

            hess = (
                -1 * self.Jxyarea.ddJ_ddport()
                + self.port_coil_distance_weight * self.Jccxydist.ddJ_ddport()
                + self.port_forward_facing_weight * self.Jufp.ddJ_ddport()
            )

            return J, dJ, hess

        else:
            J = (
                -1 * self.Jxyarea.J()
                + self.port_coil_distance_weight * self.Jccxydist.J()
                + self.port_forward_facing_weight * self.Jufp.J()
            )
            # + self.port_arc_penalty_weight * self.Jarc.J() \

            dJ = (
                -1 * self.Jxyarea.dJ(partials=True).data[self.port]
                + self.port_coil_distance_weight
                * self.Jccxydist.dJ(partials=True).data[self.port]
                + self.port_forward_facing_weight
                * self.Jufp.dJ(partials=True).data[self.port]
            )
            # + self.port_arc_penalty_weight * self.Jarc.dJ(partials=True).data[self.port] \

            return J, dJ

    def explicit_solve(self, verbose=True):
        if not self.need_to_run_code:
            return

        # Pre-conditioning. Move port s.t. constraints port-coil distance is satisfied.
        # This is necessary, otherwise results of optimization below will not satisfy constraints
        counter = 0
        while self.Jccxydist.J() > 0:
            print("Port intersect coils... reducing toroidal extend of port")
            for oo in range(1, self.port.order + 1):
                for par in ["c", "s"]:
                    key = f"phi{par}({oo})"
                    self.port.set(key, self.port.get(key) * (0.9**oo))
                    key = f"theta{par}({oo})"
                    self.port.set(key, self.port.get(key) * (0.9**oo))

            if counter == 20:
                raise ValueError("Initial port does not satisfy constraints")

        # First, L-BFGS solve
        fun = lambda x: self.objective(x, hessian_bool=False)
        dofs = self.port.x
        maxiter = 1000
        res = minimize(
            fun,
            dofs,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "gtol": 1e-08},
        )

        res_bfgs = {
            "fun": res.fun,
            "gradient": res.jac,
            "iter": res.nit,
            "info": res,
            "success": res.success,
        }

        if verbose:
            print(
                f"L-BFGS-B solve - {res_bfgs['success']}  iter={res_bfgs['iter']}, ||grad||_inf = {np.linalg.norm(res_bfgs['gradient'], ord=np.inf):.3e}",
                flush=True,
            )

        # Then, Newton step
        fun = lambda x: self.objective(x, hessian_bool=True)
        x = res.x
        maxiter = 10
        stab = 0.0
        tol = 1e-12
        J, dJ, d2J = fun(x)
        norm = np.linalg.norm(d2J)
        i = 0
        while i < maxiter and norm > tol:
            d2J += stab * np.identity(d2J.shape[0])
            dx = np.linalg.solve(d2J, dJ)
            if norm < 1e-9:
                dx += np.linalg.solve(d2J, dJ - d2J @ dx)
            x = x - dx
            J, dJ, d2J = fun(x)
            norm = np.linalg.norm(dJ)
            i = i + 1

        P, L, U = lu(d2J)
        res_newton = {
            "residual": J,
            "jacobian": dJ,
            "hessian": d2J,
            "iter": i,
            "success": norm <= tol,
            "G": None,
            "PLU": (P, L, U),
            "vjp": None,
        }

        if verbose:
            print(
                f"NEWTON solve - {res_newton['success']}  iter={res_newton['iter']}, ||grad||_inf = {np.linalg.norm(res_newton['jacobian'], ord=np.inf):.3e}",
                flush=True,
            )

        self.need_to_run_code = False

        return res_bfgs, res_newton

    def J(self):
        self.explicit_solve()
        return self.Jxyarea.J()

    @derivative_dec
    def dJ(self):
        self.explicit_solve()
        U1 = self.Jxyarea.ddJ_ddport()
        U2 = self.port_coil_distance_weight * self.Jccxydist.ddJ_ddport()
        U3 = self.port_forward_facing_weight * self.Jufp.ddJ_ddport()
        U = -U1 + U2 + U3

        x = np.linalg.solve(U, self.Jxyarea.dJ())

        d = Derivative()
        for c in self.curves:
            V1 = self.Jxyarea.ddJ_dportdcoil(c)
            V2 = self.port_coil_distance_weight * self.Jccxydist.ddJ_dportdcoil(c)
            V3 = self.port_forward_facing_weight * self.Jufp.ddJ_dportdcoil(c)
            V = -V1 + V2 + V3

            dAdxc = np.matmul(V, x)
            d += Derivative({c: dAdxc})

        return d

    def plot(self):
        _, ax = plt.subplots()

        # Plot surface
        if self.direction == "radial":
            x0 = np.mean(self.port.gamma(), axis=0)
            gproj = project(self.port.surf.gamma().reshape((-1, 3)), x0)
            gport = project(self.port.gamma(), x0)
        elif self.direction == "vertical":
            # Rerorganizing indices to be compatible with the projection in the case of radial access. Not very elegant, but I don't won't to reimplement the project function to order dimensions differently
            gproj = self.port.surf.gamma()[:, :, [2, 0, 1]]
            gport = self.port.gamma()[:, [2, 0, 1]]
        ind = np.argsort(gproj[:, 0])
        ax.scatter(gproj[ind, 1], gproj[ind, 2], c=gproj[ind, 0], s=10)

        # Plot port
        ax.fill(gport[:, 1], gport[:, 2], color="r", alpha=0.8)

        # Plot coils
        for c in self.curves:
            if self.direction == "radial":
                g = project(c.gamma(), x0)
            elif self.direction == "vertical":
                g = c.gamma()[:, [2, 0, 1]]
            zcurves = g[:, 0]

            ind = np.where(zcurves > 0)[0]
            g = g[ind, :]

            ax.scatter(g[:, 1], g[:, 2], color="k", marker="o", s=15)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")


def PolyArea(x, y):
    # Showlace formula, using https://stackoverflow.com/a/30408825
    shifty = jnp.roll(y, 1)
    shiftx = jnp.roll(x, 1)
    return 0.5 * jnp.abs(jnp.dot(x, shifty) - jnp.dot(y, shiftx))


def xy_enclosed_area(gamma):
    """
    Returns enclosed area in XY plane
    """

    x = gamma[:, 0]
    y = gamma[:, 1]
    return PolyArea(x, y)


def project(x, gamma):
    centroid = jnp.mean(gamma.reshape((-1, 3)), axis=0)
    phic = jnp.arctan2(centroid[1], centroid[0])
    unit_normal = jnp.array([jnp.cos(phic), jnp.sin(phic), jnp.zeros(phic.shape)])  # er
    unit_tangent = jnp.array(
        [-jnp.sin(phic), jnp.cos(phic), jnp.zeros(phic.shape)]
    )  # ephi
    unit_z = jnp.array(
        [jnp.zeros(phic.shape), jnp.zeros(phic.shape), jnp.ones(phic.shape)]
    )  # ez

    #                 r           phi         z
    M = jnp.array([unit_normal, unit_tangent, unit_z]).transpose()
    invM = jnp.linalg.inv(M)

    return jnp.einsum("ij,...j->...i", invM, x - centroid)  # return phi, r, z coords


def _project_xy_frame(x):
    return x[:, [2, 0, 1]]


def _project_accessibility_points(x, projection, gamma):
    if projection == "xy":
        return _project_xy_frame(x)
    if projection == "zphi":
        return project(x, gamma)
    raise ValueError(f"Unknown projection '{projection}'")


def zphi_enclosed_area(gamma):
    """
    Returns enclosed area in Z-phi plane
    """
    gcyl = _project_accessibility_points(gamma, "zphi", gamma)

    x = gcyl[:, 1]
    y = gcyl[:, 2]

    return PolyArea(x, y)


@jit
def _projected_enclosed_area_xy_grad(gamma):
    return grad(xy_enclosed_area)(gamma)


@jit
def _projected_enclosed_area_zphi_grad(gamma):
    return grad(zphi_enclosed_area)(gamma)


@jit
def _projected_enclosed_area_xy_hessian(gamma):
    return hessian(xy_enclosed_area)(gamma)


@jit
def _projected_enclosed_area_zphi_hessian(gamma):
    return hessian(zphi_enclosed_area)(gamma)


def _projected_enclosed_area_value(gamma, projection):
    if projection == "xy":
        return xy_enclosed_area(gamma)
    if projection == "zphi":
        return zphi_enclosed_area(gamma)
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_enclosed_area_grad(gamma, projection):
    if projection == "xy":
        return _projected_enclosed_area_xy_grad(gamma)
    if projection == "zphi":
        return _projected_enclosed_area_zphi_grad(gamma)
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_enclosed_area_hessian(gamma, projection):
    if projection == "xy":
        return _projected_enclosed_area_xy_hessian(gamma)
    if projection == "zphi":
        return _projected_enclosed_area_zphi_hessian(gamma)
    raise ValueError(f"Unknown projection '{projection}'")


class ProjectedEnclosedArea(Optimizable):
    def __init__(self, curve, projection="xy"):
        self.curve = curve
        self.projection = projection

        super().__init__(depends_on=[curve])

    def J(self):
        gamma = self.curve.gamma()
        return _projected_enclosed_area_value(gamma, self.projection)

    @derivative_dec
    def dJ(self):
        gamma = self.curve.gamma()
        grad0 = _projected_enclosed_area_grad(gamma, self.projection)

        dcurve = self.curve.dgamma_by_dcoeff_vjp(grad0)
        return dcurve

    def ddJ_ddport(self):
        gamma = self.curve.gamma()
        hess = _projected_enclosed_area_hessian(
            gamma, self.projection
        )  # this is d^2J/dgamma_i dgamma_j. Size npts x 3 x npts x 3
        dgdx = self.curve.dgamma_by_dcoeff()  # this is dgamma/dx, size npts x 3 x ndofs

        grad0 = _projected_enclosed_area_grad(
            gamma, self.projection
        )  # this is dJ/dgamma, size npts x 3
        curve_hessian = (
            self.curve.gamma_hessian()
        )  # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs

        return np.einsum("ijkl,ijm,kln->mn", hess, dgdx, dgdx) + np.einsum(
            "ij,ijkl->kl", grad0, curve_hessian
        )  # this should be size ndofs x ndofs

    def ddJ_dportdcoil(self, curve=None):
        return 0  # Does not depend on coils


def upward_facing_pure(nznorm):
    return jnp.sum(jnp.maximum(-nznorm, 0) ** 2)


@jit
def _upward_facing_grad(nznorm):
    return grad(upward_facing_pure)(nznorm)


@jit
def _upward_facing_hessian(nznorm):
    return hessian(upward_facing_pure)(nznorm)


def _directed_facing_inputs(curve, projection):
    if projection == "xy":
        return curve.zfactor(), curve.dzfactor_by_dcoeff_vjp
    if projection == "zphi":
        return curve.rfactor(), curve.drfactor_by_dcoeff_vjp
    raise ValueError(f"Unknown projection '{projection}'")


def _directed_facing_hessian_inputs(curve, projection):
    if projection == "xy":
        return curve.zfactor(), curve.dzfactor_by_dcoeff(), curve.zfactor_hessian()
    if projection == "zphi":
        return curve.rfactor(), curve.drfactor_by_dcoeff(), curve.rfactor_hessian()
    raise ValueError(f"Unknown projection '{projection}'")


class DirectedFacingPort(Optimizable):
    def __init__(self, curve, projection="xy"):
        self.curve = curve
        self.projection = projection

        super().__init__(depends_on=[curve])

    def J(self):
        n, _ = _directed_facing_inputs(self.curve, self.projection)
        return upward_facing_pure(n)

    @derivative_dec
    def dJ(self):
        n, f = _directed_facing_inputs(self.curve, self.projection)

        return f(_upward_facing_grad(n))

    def ddJ_ddport(self):
        n, dgdx, curve_hessian = _directed_facing_hessian_inputs(
            self.curve, self.projection
        )

        hess = _upward_facing_hessian(
            n
        )  # this is d^2J/dgamma_i dgamma_j. Size npts x npts
        grad0 = _upward_facing_grad(n)  # this is dJ/dgamma, size npts

        return np.einsum("ij,il,jm->lm", hess, dgdx, dgdx) + np.einsum(
            "i,ilm->lm", grad0, curve_hessian
        )  # this should be size ndofs x ndofs

    def ddJ_dportdcoil(self, curve=None):
        return 0  # does not depend on coils


def winding_number_2d_pure(g2, g2dash, pts):
    x = pts[:, None, 0] - g2[None, :, 0]
    y = pts[:, None, 1] - g2[None, :, 1]
    rsquared = x**2 + y**2

    # icurve, iport
    integrand = x / rsquared * g2dash[None, :, 1] - y / rsquared * g2dash[None, :, 0]
    return np.mean(integrand, axis=1) / (2 * np.pi)


def winding_number_2d_squared(g2, g2dash, pts):
    x = pts[:, None, 0] - g2[None, :, 0]
    y = pts[:, None, 1] - g2[None, :, 1]
    rsquared = x**2 + y**2

    # icurve, iport
    integrand = x / rsquared * g2dash[None, :, 1] - y / rsquared * g2dash[None, :, 0]
    return (jnp.mean(integrand, axis=1) / (2 * jnp.pi)) ** 2


def curve_inside_port_penalty_pure(
    gamma_port, gammadash_port, projection, gamma_curve, gammadash_curve, threshold=0.1
):
    gport_2d = _project_accessibility_points(gamma_port, projection, gamma_port)
    gportdash_2d = _project_accessibility_points(gammadash_port, projection, gamma_port)
    gamma_curve_2d = _project_accessibility_points(gamma_curve, projection, gamma_port)
    gammadash_curve_2d = _project_accessibility_points(
        gammadash_curve, projection, gamma_port
    )
    gammadash_curve_2d = gammadash_curve_2d.at[:, 0].set(1e-14)
    dlcurve = jnp.linalg.norm(gammadash_curve_2d, axis=1)

    # For each point of the curve, evaluate the winding number weighted by wether they are behind or in front of the port
    weights = jnp.maximum(gamma_curve_2d[:, 0], 0) ** 2
    winding_numbers = winding_number_2d_squared(
        gport_2d[:, 1:3], gportdash_2d[:, 1:3], gamma_curve_2d[:, 1:3]
    )

    # Return integral along curve of winding numbers, with some threshold
    return jnp.sum(dlcurve * weights * jnp.maximum(winding_numbers - threshold, 0) ** 2)


@jit
def _curve_in_port_penalty_xy_value(
    gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
):
    return curve_inside_port_penalty_pure(
        gamma_port, gammadash_port, "xy", gamma_curve, gammadash_curve, threshold
    )


@jit
def _curve_in_port_penalty_zphi_value(
    gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
):
    return curve_inside_port_penalty_pure(
        gamma_port, gammadash_port, "zphi", gamma_curve, gammadash_curve, threshold
    )


@jit
def _curve_in_port_penalty_xy_grad(
    gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
):
    return grad(_curve_in_port_penalty_xy_value, argnums=(0, 1, 2, 3))(
        gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
    )


@jit
def _curve_in_port_penalty_zphi_grad(
    gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
):
    return grad(_curve_in_port_penalty_zphi_value, argnums=(0, 1, 2, 3))(
        gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
    )


@jit
def _curve_in_port_penalty_xy_values(
    gamma_port, gammadash_port, gamma_curves, gammadash_curves, threshold
):
    return vmap(
        _curve_in_port_penalty_xy_value,
        in_axes=(None, None, 0, 0, None),
    )(
        gamma_port,
        gammadash_port,
        gamma_curves,
        gammadash_curves,
        threshold,
    )


@jit
def _curve_in_port_penalty_zphi_values(
    gamma_port, gammadash_port, gamma_curves, gammadash_curves, threshold
):
    return vmap(
        _curve_in_port_penalty_zphi_value,
        in_axes=(None, None, 0, 0, None),
    )(
        gamma_port,
        gammadash_port,
        gamma_curves,
        gammadash_curves,
        threshold,
    )


@jit
def _curve_in_port_penalty_xy_grads(
    gamma_port, gammadash_port, gamma_curves, gammadash_curves, threshold
):
    return vmap(
        _curve_in_port_penalty_xy_grad,
        in_axes=(None, None, 0, 0, None),
    )(
        gamma_port,
        gammadash_port,
        gamma_curves,
        gammadash_curves,
        threshold,
    )


@jit
def _curve_in_port_penalty_zphi_grads(
    gamma_port, gammadash_port, gamma_curves, gammadash_curves, threshold
):
    return vmap(
        _curve_in_port_penalty_zphi_grad,
        in_axes=(None, None, 0, 0, None),
    )(
        gamma_port,
        gammadash_port,
        gamma_curves,
        gammadash_curves,
        threshold,
    )


@partial(jit, static_argnums=(5, 6))
def _curve_in_port_penalty_xy_hessian(
    gamma_port,
    gammadash_port,
    gamma_curve,
    gammadash_curve,
    threshold,
    left_argnum,
    right_argnum,
):
    return jacfwd(
        jacrev(_curve_in_port_penalty_xy_value, argnums=left_argnum),
        argnums=right_argnum,
    )(gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold)


@partial(jit, static_argnums=(5, 6))
def _curve_in_port_penalty_zphi_hessian(
    gamma_port,
    gammadash_port,
    gamma_curve,
    gammadash_curve,
    threshold,
    left_argnum,
    right_argnum,
):
    return jacfwd(
        jacrev(_curve_in_port_penalty_zphi_value, argnums=left_argnum),
        argnums=right_argnum,
    )(gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold)


def _curve_in_port_penalty_value(
    gamma_port, gammadash_port, gamma_curve, gammadash_curve, projection, threshold
):
    if projection == "xy":
        return _curve_in_port_penalty_xy_value(
            gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
        )
    if projection == "zphi":
        return _curve_in_port_penalty_zphi_value(
            gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _curve_in_port_penalty_grad(
    gamma_port, gammadash_port, gamma_curve, gammadash_curve, projection, threshold
):
    if projection == "xy":
        return _curve_in_port_penalty_xy_grad(
            gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
        )
    if projection == "zphi":
        return _curve_in_port_penalty_zphi_grad(
            gamma_port, gammadash_port, gamma_curve, gammadash_curve, threshold
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _curve_in_port_penalty_values(
    gamma_port, gammadash_port, gamma_curves, gammadash_curves, projection, threshold
):
    if projection == "xy":
        return _curve_in_port_penalty_xy_values(
            gamma_port,
            gammadash_port,
            gamma_curves,
            gammadash_curves,
            threshold,
        )
    if projection == "zphi":
        return _curve_in_port_penalty_zphi_values(
            gamma_port,
            gammadash_port,
            gamma_curves,
            gammadash_curves,
            threshold,
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _curve_in_port_penalty_grads(
    gamma_port, gammadash_port, gamma_curves, gammadash_curves, projection, threshold
):
    if projection == "xy":
        return _curve_in_port_penalty_xy_grads(
            gamma_port,
            gammadash_port,
            gamma_curves,
            gammadash_curves,
            threshold,
        )
    if projection == "zphi":
        return _curve_in_port_penalty_zphi_grads(
            gamma_port,
            gammadash_port,
            gamma_curves,
            gammadash_curves,
            threshold,
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _curve_in_port_penalty_hessian(
    gamma_port,
    gammadash_port,
    gamma_curve,
    gammadash_curve,
    projection,
    threshold,
    left_argnum,
    right_argnum,
):
    if projection == "xy":
        return _curve_in_port_penalty_xy_hessian(
            gamma_port,
            gammadash_port,
            gamma_curve,
            gammadash_curve,
            threshold,
            left_argnum,
            right_argnum,
        )
    if projection == "zphi":
        return _curve_in_port_penalty_zphi_hessian(
            gamma_port,
            gammadash_port,
            gamma_curve,
            gammadash_curve,
            threshold,
            left_argnum,
            right_argnum,
        )
    raise ValueError(f"Unknown projection '{projection}'")


def count_inside_points(
    gamma_port, gammadash_port, projection, gamma_curve, threshold=0.1
):
    gport_2d = _project_accessibility_points(gamma_port, projection, gamma_port)
    gportdash_2d = _project_accessibility_points(gammadash_port, projection, gamma_port)
    gamma_curve_2d = _project_accessibility_points(gamma_curve, projection, gamma_port)

    # For each point of the curve, evaluate the winding number weighted by wether they are behind or in front of the port
    winding_numbers = winding_number_2d_pure(
        gport_2d[:, 1:3], gportdash_2d[:, 1:3], gamma_curve_2d[:, 1:3]
    )

    # Find closest point, evaluate if curve is in front or behind port
    dists = np.linalg.norm(gport_2d[:, None, :] - gamma_curve_2d[None, :, :], axis=2)
    min_dists_index = np.argmin(dists, axis=0)
    test = gamma_curve_2d[:, 0] > gport_2d[min_dists_index, 0]

    # Return integral along curve of winding numbers, with some threshold
    return np.sum(
        [((np.abs(w) > threshold) and (t)) for w, t in zip(winding_numbers, test)]
    )


def _curve_sample_batches(curves):
    grouped_samples = {}
    for idx, curve in enumerate(curves):
        gamma = curve.gamma()
        gammadash = curve.gammadash()
        key = (tuple(gamma.shape), tuple(gammadash.shape))
        if key not in grouped_samples:
            grouped_samples[key] = ([], [], [])
        indices, gammas, gammadashes = grouped_samples[key]
        indices.append(idx)
        gammas.append(gamma)
        gammadashes.append(gammadash)
    return [
        (indices, jnp.stack(gammas), jnp.stack(gammadashes))
        for indices, gammas, gammadashes in grouped_samples.values()
    ]


def _contract_projected_cc_distance_port_hessian_terms(
    hess00,
    hess01,
    hess11,
    grad0,
    grad1,
    dg1dx,
    dl1dx,
    gamma_hessian,
    gammadash_hessian,
):
    return (
        np.einsum("ijkl,ijm,kln->mn", hess00, dg1dx, dg1dx, optimize=True)
        + np.einsum("ijkl,ijm,kln->mn", hess11, dl1dx, dl1dx, optimize=True)
        + np.einsum("ij,ijkl->kl", grad0, gamma_hessian, optimize=True)
        + np.einsum("ij,ijkl->kl", grad1, gammadash_hessian, optimize=True)
        + np.einsum("kilj,kim,ljn->mn", hess01, dg1dx, dl1dx, optimize=True)
        + np.einsum("kilj,kin,ljm->mn", hess01, dg1dx, dl1dx, optimize=True)
    )


class CurveInPortPenalty(Optimizable):
    def __init__(self, port, curves, threshold, projection="zphi"):
        self.port = port
        self.curves = curves
        self.threshold = threshold
        self.projection = projection

        super().__init__(depends_on=curves + [port])

    def count_inside_points(self):
        gp = self.port.gamma()
        l1 = self.port.gammadash()

        N = 0
        for c in self.curves:
            gc = c.gamma()
            N += count_inside_points(
                gp, l1, self.projection, gc, threshold=self.threshold
            )

        return N

    def J(self):
        gp = self.port.gamma()
        l1 = self.port.gammadash()

        out = 0.0
        for indices, gammas, gammadashes in _curve_sample_batches(self.curves):
            if len(indices) == 1:
                out += _curve_in_port_penalty_value(
                    gp,
                    l1,
                    gammas[0],
                    gammadashes[0],
                    self.projection,
                    self.threshold,
                )
                continue
            out += jnp.sum(
                _curve_in_port_penalty_values(
                    gp,
                    l1,
                    gammas,
                    gammadashes,
                    self.projection,
                    self.threshold,
                )
            )

        return out

    @derivative_dec
    def dJ(self):
        cc = self.curves + [self.port]

        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in cc]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in cc]

        gp = self.port.gamma()
        l1 = self.port.gammadash()
        for indices, gammas, gammadashes in _curve_sample_batches(self.curves):
            if len(indices) == 1:
                grad0, grad1, grad2, grad3 = _curve_in_port_penalty_grad(
                    gp,
                    l1,
                    gammas[0],
                    gammadashes[0],
                    self.projection,
                    self.threshold,
                )
                grad0 = np.asarray(grad0)[None, ...]
                grad1 = np.asarray(grad1)[None, ...]
                grad2 = np.asarray(grad2)[None, ...]
                grad3 = np.asarray(grad3)[None, ...]
            else:
                grad0, grad1, grad2, grad3 = _curve_in_port_penalty_grads(
                    gp,
                    l1,
                    gammas,
                    gammadashes,
                    self.projection,
                    self.threshold,
                )
                grad0 = np.asarray(grad0)
                grad1 = np.asarray(grad1)
                grad2 = np.asarray(grad2)
                grad3 = np.asarray(grad3)

            # derivatives w.r.t port
            dgamma_by_dcoeff_vjp_vecs[-1] += np.sum(grad0, axis=0)
            dgammadash_by_dcoeff_vjp_vecs[-1] += np.sum(grad1, axis=0)

            # derivatives w.r.t curves
            for idx, curve_grad0, curve_grad1 in zip(indices, grad2, grad3):
                dgamma_by_dcoeff_vjp_vecs[idx] += curve_grad0
                dgammadash_by_dcoeff_vjp_vecs[idx] += curve_grad1

        res = [
            self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i])
            + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i])
            for i in range(len(self.curves))
        ]
        res.append(
            self.port.dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[-1])
            + self.port.dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[-1])
        )
        return sum(res)


def min_xy_distance(gamma1, gamma2):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula in xy plane.
    """
    g1cyl = gamma1[:, [2, 0, 1]]
    g2cyl = gamma2[:, [2, 0, 1]]

    g1 = gamma1[:, :2]
    g2 = gamma2[:, :2]

    dists = np.linalg.norm(g1[:, None, :] - g2[None, :, :], axis=2)
    f = np.maximum(g2cyl[None, :, 0] - g1cyl[:, None, 0], 0) / (
        g2cyl[None, :, 0] - g1cyl[:, None, 0]
    )

    if (f == 0).all():
        return np.nan
    else:
        return -np.min(-dists * f)


def cc_xy_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula in xy plane.
    """
    g1 = gamma1[:, :2]
    g2 = gamma2[:, :2]

    # Set gammadah in z-direction to almost 0. We cannot set it to zero, otherwise derivatives would be NaN when gammadash is purely along z
    l1 = l1.at[:, 2].set(1e-14)
    l2 = l2.at[:, 2].set(1e-14)

    # This is 0 for all points below (i.e z2<z1) their comparison point on the curve
    f = jnp.maximum(gamma2[None, :, 2] - gamma1[:, None, 2], 0) ** 2

    dists = jnp.sqrt(jnp.sum((g1[:, None, :] - g2[None, :, :]) ** 2, axis=2))

    l1norm = jnp.linalg.norm(l1, axis=1)
    l2norm = jnp.linalg.norm(l2, axis=1)

    alen = l1norm[:, None] * l2norm[None, :]
    return jnp.sum(alen * f * jnp.maximum(minimum_distance - dists, 0) ** 2)


def min_zphi_distance(gamma1, gamma2):
    g1cyl = _project_accessibility_points(gamma1, "zphi", gamma1)
    g2cyl = _project_accessibility_points(gamma2, "zphi", gamma1)
    g1 = g1cyl[:, 1:]
    g2 = g2cyl[:, 1:]

    dists = np.linalg.norm(g1[:, None, :] - g2[None, :, :], axis=2)
    mask = g2cyl[None, :, 0] > g1cyl[:, None, 0]

    if mask.any():
        return -np.min(-dists * mask)
    else:
        return np.nan


def cc_zphi_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    g1cyl = _project_accessibility_points(gamma1, "zphi", gamma1)
    g2cyl = _project_accessibility_points(gamma2, "zphi", gamma1)

    l1cyl = _project_accessibility_points(l1, "zphi", gamma1)
    l2cyl = _project_accessibility_points(l2, "zphi", gamma1)

    g1 = g1cyl[:, 1:]
    g2 = g2cyl[:, 1:]

    # Set gammadah in z-direction to almost 0. We cannot set it to zero, otherwise derivatives would be NaN when gammadash is purely along z
    l1cyl = l1cyl.at[:, 0].set(1e-14)
    l2cyl = l2cyl.at[:, 0].set(1e-14)

    # This is 0 for all points below (i.e z2<z1) their comparison point on the curve
    f = jnp.maximum(g2cyl[None, :, 0] - g1cyl[:, None, 0], 0) ** 2

    dists = jnp.sqrt(jnp.sum((g1[:, None, :] - g2[None, :, :]) ** 2, axis=2))

    l1norm = jnp.linalg.norm(l1cyl, axis=1)
    l2norm = jnp.linalg.norm(l2cyl, axis=1)

    alen = l1norm[:, None] * l2norm[None, :]
    return (
        jnp.sum(alen * f * jnp.maximum(minimum_distance - dists, 0) ** 2)
        / (g1.shape[0])
    )


@jit
def _projected_cc_distance_xy_value(gamma1, l1, gamma2, l2, minimum_distance):
    return cc_xy_distance_pure(gamma1, l1, gamma2, l2, minimum_distance)


@jit
def _projected_cc_distance_zphi_value(gamma1, l1, gamma2, l2, minimum_distance):
    return cc_zphi_distance_pure(gamma1, l1, gamma2, l2, minimum_distance)


@jit
def _projected_cc_distance_xy_grad(gamma1, l1, gamma2, l2, minimum_distance):
    return grad(_projected_cc_distance_xy_value, argnums=(0, 1, 2, 3))(
        gamma1, l1, gamma2, l2, minimum_distance
    )


@jit
def _projected_cc_distance_zphi_grad(gamma1, l1, gamma2, l2, minimum_distance):
    return grad(_projected_cc_distance_zphi_value, argnums=(0, 1, 2, 3))(
        gamma1, l1, gamma2, l2, minimum_distance
    )


@jit
def _projected_cc_distance_xy_values(gamma1, l1, gamma2s, l2s, minimum_distance):
    return vmap(
        _projected_cc_distance_xy_value,
        in_axes=(None, None, 0, 0, None),
    )(gamma1, l1, gamma2s, l2s, minimum_distance)


@jit
def _projected_cc_distance_zphi_values(gamma1, l1, gamma2s, l2s, minimum_distance):
    return vmap(
        _projected_cc_distance_zphi_value,
        in_axes=(None, None, 0, 0, None),
    )(gamma1, l1, gamma2s, l2s, minimum_distance)


@jit
def _projected_cc_distance_xy_grads(gamma1, l1, gamma2s, l2s, minimum_distance):
    return vmap(
        _projected_cc_distance_xy_grad,
        in_axes=(None, None, 0, 0, None),
    )(gamma1, l1, gamma2s, l2s, minimum_distance)


@jit
def _projected_cc_distance_zphi_grads(gamma1, l1, gamma2s, l2s, minimum_distance):
    return vmap(
        _projected_cc_distance_zphi_grad,
        in_axes=(None, None, 0, 0, None),
    )(gamma1, l1, gamma2s, l2s, minimum_distance)


@partial(jit, static_argnums=(5, 6))
def _projected_cc_distance_xy_hessian(
    gamma1, l1, gamma2, l2, minimum_distance, left_argnum, right_argnum
):
    return jacfwd(
        jacrev(_projected_cc_distance_xy_value, argnums=left_argnum),
        argnums=right_argnum,
    )(gamma1, l1, gamma2, l2, minimum_distance)


@partial(jit, static_argnums=(5, 6))
def _projected_cc_distance_zphi_hessian(
    gamma1, l1, gamma2, l2, minimum_distance, left_argnum, right_argnum
):
    return jacfwd(
        jacrev(_projected_cc_distance_zphi_value, argnums=left_argnum),
        argnums=right_argnum,
    )(gamma1, l1, gamma2, l2, minimum_distance)


@partial(jit, static_argnums=(5, 6))
def _projected_cc_distance_xy_hessians(
    gamma1, l1, gamma2s, l2s, minimum_distance, left_argnum, right_argnum
):
    return vmap(
        _projected_cc_distance_xy_hessian,
        in_axes=(None, None, 0, 0, None, None, None),
    )(gamma1, l1, gamma2s, l2s, minimum_distance, left_argnum, right_argnum)


@partial(jit, static_argnums=(5, 6))
def _projected_cc_distance_zphi_hessians(
    gamma1, l1, gamma2s, l2s, minimum_distance, left_argnum, right_argnum
):
    return vmap(
        _projected_cc_distance_zphi_hessian,
        in_axes=(None, None, 0, 0, None, None, None),
    )(gamma1, l1, gamma2s, l2s, minimum_distance, left_argnum, right_argnum)


def _projected_cc_distance_value(gamma1, l1, gamma2, l2, projection, minimum_distance):
    if projection == "xy":
        return _projected_cc_distance_xy_value(gamma1, l1, gamma2, l2, minimum_distance)
    if projection == "zphi":
        return _projected_cc_distance_zphi_value(
            gamma1, l1, gamma2, l2, minimum_distance
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_cc_distance_grad(gamma1, l1, gamma2, l2, projection, minimum_distance):
    if projection == "xy":
        return _projected_cc_distance_xy_grad(gamma1, l1, gamma2, l2, minimum_distance)
    if projection == "zphi":
        return _projected_cc_distance_zphi_grad(
            gamma1, l1, gamma2, l2, minimum_distance
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_cc_distance_values(
    gamma1, l1, gamma2s, l2s, projection, minimum_distance
):
    if projection == "xy":
        return _projected_cc_distance_xy_values(
            gamma1,
            l1,
            gamma2s,
            l2s,
            minimum_distance,
        )
    if projection == "zphi":
        return _projected_cc_distance_zphi_values(
            gamma1,
            l1,
            gamma2s,
            l2s,
            minimum_distance,
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_cc_distance_grads(
    gamma1, l1, gamma2s, l2s, projection, minimum_distance
):
    if projection == "xy":
        return _projected_cc_distance_xy_grads(
            gamma1,
            l1,
            gamma2s,
            l2s,
            minimum_distance,
        )
    if projection == "zphi":
        return _projected_cc_distance_zphi_grads(
            gamma1,
            l1,
            gamma2s,
            l2s,
            minimum_distance,
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_cc_distance_hessian(
    gamma1,
    l1,
    gamma2,
    l2,
    projection,
    minimum_distance,
    left_argnum,
    right_argnum,
):
    if projection == "xy":
        return _projected_cc_distance_xy_hessian(
            gamma1, l1, gamma2, l2, minimum_distance, left_argnum, right_argnum
        )
    if projection == "zphi":
        return _projected_cc_distance_zphi_hessian(
            gamma1, l1, gamma2, l2, minimum_distance, left_argnum, right_argnum
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_cc_distance_hessians(
    gamma1,
    l1,
    gamma2s,
    l2s,
    projection,
    minimum_distance,
    left_argnum,
    right_argnum,
):
    if projection == "xy":
        return _projected_cc_distance_xy_hessians(
            gamma1,
            l1,
            gamma2s,
            l2s,
            minimum_distance,
            left_argnum,
            right_argnum,
        )
    if projection == "zphi":
        return _projected_cc_distance_zphi_hessians(
            gamma1,
            l1,
            gamma2s,
            l2s,
            minimum_distance,
            left_argnum,
            right_argnum,
        )
    raise ValueError(f"Unknown projection '{projection}'")


class ProjectedCurveCurveDistance(Optimizable):
    def __init__(self, base_curves, curve, minimum_distance=0, projection="xy"):
        self.base_curves = base_curves
        self.curve = curve
        self.minimum_distance = minimum_distance

        self.projection = projection

        self.num_basecurves = len(base_curves)
        super().__init__(depends_on=base_curves + [curve])

    def shortest_distance(self):
        res = np.zeros((len(self.base_curves),))
        gamma = self.curve.gamma()
        for ii, c in enumerate(self.base_curves):
            gamma2 = c.gamma()
            if self.projection == "xy":
                res[ii] = min_xy_distance(gamma, gamma2)
            elif self.projection == "zphi":
                res[ii] = min_zphi_distance(gamma, gamma2)

        if all([np.isnan(k) for k in res]):
            raise RuntimeError("No curves in front of port")
        else:
            return np.min(res[~np.isnan(res)])

    def J(self):
        res = 0.0

        gamma = self.curve.gamma()
        l = self.curve.gammadash()
        for indices, gammas, gammadashes in _curve_sample_batches(self.base_curves):
            if len(indices) == 1:
                res += _projected_cc_distance_value(
                    gamma,
                    l,
                    gammas[0],
                    gammadashes[0],
                    self.projection,
                    self.minimum_distance,
                )
                continue
            res += jnp.sum(
                _projected_cc_distance_values(
                    gamma,
                    l,
                    gammas,
                    gammadashes,
                    self.projection,
                    self.minimum_distance,
                )
            )

        return res

    @derivative_dec
    def dJ(self):
        cc = self.base_curves + [self.curve]

        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in cc]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in cc]

        gamma1 = self.curve.gamma()
        l1 = self.curve.gammadash()
        for indices, gammas, gammadashes in _curve_sample_batches(self.base_curves):
            if len(indices) == 1:
                grad0, grad1, grad2, grad3 = _projected_cc_distance_grad(
                    gamma1,
                    l1,
                    gammas[0],
                    gammadashes[0],
                    self.projection,
                    self.minimum_distance,
                )
                grad0 = np.asarray(grad0)[None, ...]
                grad1 = np.asarray(grad1)[None, ...]
                grad2 = np.asarray(grad2)[None, ...]
                grad3 = np.asarray(grad3)[None, ...]
            else:
                grad0, grad1, grad2, grad3 = _projected_cc_distance_grads(
                    gamma1,
                    l1,
                    gammas,
                    gammadashes,
                    self.projection,
                    self.minimum_distance,
                )
                grad0 = np.asarray(grad0)
                grad1 = np.asarray(grad1)
                grad2 = np.asarray(grad2)
                grad3 = np.asarray(grad3)
            dgamma_by_dcoeff_vjp_vecs[-1] += np.sum(grad0, axis=0)
            dgammadash_by_dcoeff_vjp_vecs[-1] += np.sum(grad1, axis=0)
            for idx, curve_grad0, curve_grad1 in zip(indices, grad2, grad3):
                dgamma_by_dcoeff_vjp_vecs[idx] += curve_grad0
                dgammadash_by_dcoeff_vjp_vecs[idx] += curve_grad1

        res = [
            self.base_curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i])
            + self.base_curves[i].dgammadash_by_dcoeff_vjp(
                dgammadash_by_dcoeff_vjp_vecs[i]
            )
            for i in range(len(self.base_curves))
        ]
        res.append(
            self.curve.dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[-1])
            + self.curve.dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[-1])
        )
        return sum(res)

    def ddJ_ddport(self):
        g1 = self.curve.gamma()
        l1 = self.curve.gammadash()
        dg1dx = (
            self.curve.dgamma_by_dcoeff()
        )  # this is dgamma/dx, size npts x 3 x ndofs
        dl1dx = (
            self.curve.dgammadash_by_dcoeff()
        )  # this is dgamma/dx, size npts x 3 x ndofs
        gamma_hessian = (
            self.curve.gamma_hessian()
        )  # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs
        gammadash_hessian = (
            self.curve.gammadash_hessian()
        )  # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs

        ndofs_port = self.curve.num_dofs()
        res = np.zeros((ndofs_port, ndofs_port))
        for indices, gammas, gammadashes in _curve_sample_batches(self.base_curves):
            if len(indices) == 1:
                hess00 = _projected_cc_distance_hessian(
                    g1,
                    l1,
                    gammas[0],
                    gammadashes[0],
                    self.projection,
                    self.minimum_distance,
                    0,
                    0,
                )
                hess01 = _projected_cc_distance_hessian(
                    g1,
                    l1,
                    gammas[0],
                    gammadashes[0],
                    self.projection,
                    self.minimum_distance,
                    0,
                    1,
                )
                hess11 = _projected_cc_distance_hessian(
                    g1,
                    l1,
                    gammas[0],
                    gammadashes[0],
                    self.projection,
                    self.minimum_distance,
                    1,
                    1,
                )

                grad0, grad1, _, _ = _projected_cc_distance_grad(
                    g1,
                    l1,
                    gammas[0],
                    gammadashes[0],
                    self.projection,
                    self.minimum_distance,
                )

                res += _contract_projected_cc_distance_port_hessian_terms(
                    hess00,
                    hess01,
                    hess11,
                    np.asarray(grad0),
                    np.asarray(grad1),
                    dg1dx,
                    dl1dx,
                    gamma_hessian,
                    gammadash_hessian,
                )
                continue

            hess00 = np.asarray(
                _projected_cc_distance_hessians(
                    g1,
                    l1,
                    gammas,
                    gammadashes,
                    self.projection,
                    self.minimum_distance,
                    0,
                    0,
                )
            )
            hess01 = np.asarray(
                _projected_cc_distance_hessians(
                    g1,
                    l1,
                    gammas,
                    gammadashes,
                    self.projection,
                    self.minimum_distance,
                    0,
                    1,
                )
            )
            hess11 = np.asarray(
                _projected_cc_distance_hessians(
                    g1,
                    l1,
                    gammas,
                    gammadashes,
                    self.projection,
                    self.minimum_distance,
                    1,
                    1,
                )
            )
            grad0, grad1, _, _ = _projected_cc_distance_grads(
                g1,
                l1,
                gammas,
                gammadashes,
                self.projection,
                self.minimum_distance,
            )
            res += _contract_projected_cc_distance_port_hessian_terms(
                np.sum(hess00, axis=0),
                np.sum(hess01, axis=0),
                np.sum(hess11, axis=0),
                np.sum(np.asarray(grad0), axis=0),
                np.sum(np.asarray(grad1), axis=0),
                dg1dx,
                dl1dx,
                gamma_hessian,
                gammadash_hessian,
            )

        return res

    def ddJ_dportdcoil(self, curve=None):
        g1 = self.curve.gamma()
        l1 = self.curve.gammadash()
        dg1dx = (
            self.curve.dgamma_by_dcoeff()
        )  # this is dgamma/dx, size npts x 3 x ndofs
        dl1dx = (
            self.curve.dgammadash_by_dcoeff()
        )  # this is dgamma/dx, size npts x 3 x ndofs
        ndofs_c1 = self.curve.num_dofs()
        ndofs_bc = [c.num_dofs() for c in self.base_curves]
        res = np.zeros((sum(ndofs_bc), ndofs_c1))
        hess = []
        for k in self.unique_dof_lineage:
            if not isinstance(k, Curve):  # only consider coils
                continue
            if k is self.curve:  # skip dep on port
                continue

            if curve is not None:
                if k is not curve:
                    continue

            if np.any(k.dofs_free_status):
                shape = (k.local_dof_size, self.curve.num_dofs())
                res = np.zeros(shape)

                if k.local_dof_size > 0:
                    for opt in k.dofs.dep_opts():
                        g2 = opt.gamma()
                        l2 = opt.gammadash()
                        dg2dx = opt.dgamma_by_dcoeff()
                        dl2dx = opt.dgammadash_by_dcoeff()

                        hg1g2 = _projected_cc_distance_hessian(
                            g1, l1, g2, l2, self.projection, self.minimum_distance, 0, 2
                        )
                        hl1g2 = _projected_cc_distance_hessian(
                            g1, l1, g2, l2, self.projection, self.minimum_distance, 1, 2
                        )
                        hg1l2 = _projected_cc_distance_hessian(
                            g1, l1, g2, l2, self.projection, self.minimum_distance, 0, 3
                        )
                        hl1l2 = _projected_cc_distance_hessian(
                            g1, l1, g2, l2, self.projection, self.minimum_distance, 1, 3
                        )

                        a = np.einsum("ijkl,ijm,kln->nm", hg1g2, dg1dx, dg2dx)
                        b = np.einsum("ijkl,ijm,kln->nm", hl1g2, dl1dx, dg2dx)
                        c = np.einsum("ijkl,ijm,kln->nm", hg1l2, dg1dx, dl2dx)
                        d = np.einsum("ijkl,ijm,kln->nm", hl1l2, dl1dx, dl2dx)

                        res[:, :] += a + b + c + d

                hess.append(res)

        return np.concatenate(hess)


def xy_convexity(pts, g, gd, gdd):
    gd = gd.at[:, 2].set(1e-12)  # not zero to avoid singularities
    gdd = gdd.at[:, 2].set(1e-12)  # not zero to avoid singularities

    # First we evaluate the 2D curvature
    kappa = (gd[:, 0] * gdd[:, 1] - gd[:, 1] * gdd[:, 0]) / (
        gd[:, 0] ** 2 + gd[:, 1] ** 2
    ) ** (3.0 / 2.0)

    integral_of_kappa = jnp.trapezoid(jnp.abs(kappa) * jnp.linalg.norm(gd, axis=1), pts)

    # Allow 5% margin of error for numerical integration error
    return jnp.max(jnp.array([integral_of_kappa - 1.05 * 2.0 * jnp.pi]), 0) ** 2


def zphi_convexity(pts, g, gd, gdd):
    gd_projected = _project_accessibility_points(gd, "zphi", g)
    gdd_projected = _project_accessibility_points(gdd, "zphi", g)

    gd_projected = gd_projected.at[:, 0].set(1e-12)  # not zero to avoid singularities
    gdd_projected = gdd_projected.at[:, 0].set(1e-12)  # not zero to avoid singularities

    # First we evaluate the 2D curvature
    kappa = (
        gd_projected[:, 1] * gdd_projected[:, 2]
        - gd_projected[:, 2] * gdd_projected[:, 1]
    ) / (gd_projected[:, 1] ** 2 + gd_projected[:, 2] ** 2) ** (3.0 / 2.0)

    integral_of_kappa = jnp.trapezoid(
        jnp.abs(kappa) * jnp.linalg.norm(gd_projected, axis=1), pts
    )

    # Allow 5% margin of error for numerical integration error
    return jnp.max(jnp.array([integral_of_kappa - 1.05 * 2.0 * jnp.pi]), 0) ** 2


@jit
def _projected_curve_convexity_xy_value(quadpoints, gamma, gammadash, gammadashdash):
    return xy_convexity(quadpoints, gamma, gammadash, gammadashdash)


@jit
def _projected_curve_convexity_zphi_value(quadpoints, gamma, gammadash, gammadashdash):
    return zphi_convexity(quadpoints, gamma, gammadash, gammadashdash)


@jit
def _projected_curve_convexity_xy_grad(quadpoints, gamma, gammadash, gammadashdash):
    return grad(_projected_curve_convexity_xy_value, argnums=(1, 2, 3))(
        quadpoints, gamma, gammadash, gammadashdash
    )


@jit
def _projected_curve_convexity_zphi_grad(quadpoints, gamma, gammadash, gammadashdash):
    return grad(_projected_curve_convexity_zphi_value, argnums=(1, 2, 3))(
        quadpoints, gamma, gammadash, gammadashdash
    )


@partial(jit, static_argnums=(4,))
def _projected_curve_convexity_xy_hessian(
    quadpoints, gamma, gammadash, gammadashdash, argnum
):
    return hessian(_projected_curve_convexity_xy_value, argnums=argnum)(
        quadpoints, gamma, gammadash, gammadashdash
    )


@partial(jit, static_argnums=(4,))
def _projected_curve_convexity_zphi_hessian(
    quadpoints, gamma, gammadash, gammadashdash, argnum
):
    return hessian(_projected_curve_convexity_zphi_value, argnums=argnum)(
        quadpoints, gamma, gammadash, gammadashdash
    )


def _projected_curve_convexity_value(
    quadpoints, gamma, gammadash, gammadashdash, projection
):
    if projection == "xy":
        return _projected_curve_convexity_xy_value(
            quadpoints, gamma, gammadash, gammadashdash
        )
    if projection == "zphi":
        return _projected_curve_convexity_zphi_value(
            quadpoints, gamma, gammadash, gammadashdash
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_curve_convexity_grad(
    quadpoints, gamma, gammadash, gammadashdash, projection
):
    if projection == "xy":
        return _projected_curve_convexity_xy_grad(
            quadpoints, gamma, gammadash, gammadashdash
        )
    if projection == "zphi":
        return _projected_curve_convexity_zphi_grad(
            quadpoints, gamma, gammadash, gammadashdash
        )
    raise ValueError(f"Unknown projection '{projection}'")


def _projected_curve_convexity_hessian(
    quadpoints, gamma, gammadash, gammadashdash, projection, argnum
):
    if projection == "xy":
        return _projected_curve_convexity_xy_hessian(
            quadpoints, gamma, gammadash, gammadashdash, argnum
        )
    if projection == "zphi":
        return _projected_curve_convexity_zphi_hessian(
            quadpoints, gamma, gammadash, gammadashdash, argnum
        )
    raise ValueError(f"Unknown projection '{projection}'")


class ProjectedCurveConvexity(Optimizable):
    def __init__(self, curve, projection="xy"):
        self.curve = curve
        self.projection = projection

        super().__init__(depends_on=[curve])

    def J(self):
        gamma = self.curve.gamma()
        gammadash = self.curve.gammadash()
        gammadashdash = self.curve.gammadashdash()
        return _projected_curve_convexity_value(
            self.curve.quadpoints,
            gamma,
            gammadash,
            gammadashdash,
            self.projection,
        )

    @derivative_dec
    def dJ(self):
        gamma = self.curve.gamma()
        gammadash = self.curve.gammadash()
        gammadashdash = self.curve.gammadashdash()

        grad0, grad1, grad2 = _projected_curve_convexity_grad(
            self.curve.quadpoints,
            gamma,
            gammadash,
            gammadashdash,
            self.projection,
        )

        return (
            self.curve.dgamma_by_dcoeff_vjp(grad0)
            + self.curve.dgammadash_by_dcoeff_vjp(grad1)
            + self.curve.dgammadashdash_by_dcoeff_vjp(grad2)
        )

    def ddJ_ddport(self):
        gamma = self.curve.gamma()
        gammadash = self.curve.gammadash()
        gammadashdash = self.curve.gammadashdash()
        hess0 = _projected_curve_convexity_hessian(
            self.curve.quadpoints, gamma, gammadash, gammadashdash, self.projection, 1
        )  # this is d^2J/dgamma_i dgamma_j. Size npts x 3 x npts x 3
        hess1 = _projected_curve_convexity_hessian(
            self.curve.quadpoints, gamma, gammadash, gammadashdash, self.projection, 2
        )  # this is d^2J/dgamma_i dgamma_j. Size npts x 3 x npts x 3
        hess2 = _projected_curve_convexity_hessian(
            self.curve.quadpoints, gamma, gammadash, gammadashdash, self.projection, 3
        )  # this is d^2J/dgamma_i dgamma_j. Size npts x 3 x npts x 3
        dgdx = self.curve.dgamma_by_dcoeff()  # this is dgamma/dx, size npts x 3 x ndofs
        dgdxdash = (
            self.curve.dgammadash_by_dcoeff()
        )  # this is dgamma/dx, size npts x 3 x ndofs
        dgdxdashdash = (
            self.curve.dgammadashdash_by_dcoeff()
        )  # this is dgamma/dx, size npts x 3 x ndofs

        grad0, grad1, grad2 = _projected_curve_convexity_grad(
            self.curve.quadpoints,
            gamma,
            gammadash,
            gammadashdash,
            self.projection,
        )  # this is dJ/dgamma, dJ/dgammadash, dJ/dgammadashdash
        gamma_hessian = (
            self.curve.gamma_hessian()
        )  # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs
        gammadash_hessian = (
            self.curve.gammadash_hessian()
        )  # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs
        gammadashdash_hessian = (
            self.curve.gammadashdash_hessian()
        )  # this is d^2 gamma / dx_i dx_2, size 128 x 3 x ndofs x ndofs

        # Contribution for derivatives w.r.t gamma
        ddJ_dpport_1 = np.einsum("ijkl,ijm,kln->mn", hess0, dgdx, dgdx) + np.einsum(
            "ij,ijkl->kl", grad0, gamma_hessian
        )  # this should be size ndofs x ndofs
        ddJ_dpport_2 = np.einsum("ijkl,ijm,kln->mn", hess1, dgdxdash, dgdx) + np.einsum(
            "ij,ijkl->kl", grad1, gammadash_hessian
        )  # this should be size ndofs x ndofs
        ddJ_dpport_3 = np.einsum(
            "ijkl,ijm,kln->mn", hess2, dgdxdashdash, dgdx
        ) + np.einsum(
            "ij,ijkl->kl", grad2, gammadashdash_hessian
        )  # this should be size ndofs x ndofs

        raise NotImplementedError("Missing mixed terms")

        return ddJ_dpport_1 + ddJ_dpport_2 + ddJ_dpport_3

    def ddJ_dportdcoil(self, curve=None):
        return 0  # Does not depend on coils
